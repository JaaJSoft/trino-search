/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.jaaj.trino.search.vector.projection;

import dev.jaaj.trino.search.vector.VectorReader;
import io.trino.spi.TrinoException;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.function.AggregationFunction;
import io.trino.spi.function.AggregationState;
import io.trino.spi.function.CombineFunction;
import io.trino.spi.function.Description;
import io.trino.spi.function.InputFunction;
import io.trino.spi.function.OutputFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;

/**
 * Principal component analysis as an aggregation: the directions along which a group of vectors
 * varies most, which is what makes projection columns prune.
 * <p>
 * The state is the group's mean and covariance, which merge exactly across splits; the
 * eigendecomposition happens once, at output. Its cost is quadratic in the dimension per row and
 * cubic once per group, which is why this is a fit for a sample, not a per-row function.
 */
public final class VectorPcaAggregation
{
    private VectorPcaAggregation() {}

    @AggregationFunction("vector_pca_agg")
    @Description("Returns the k leading principal directions of the vectors in each group, with the share of the variance each carries")
    public static final class OfDoubleVectors
    {
        private OfDoubleVectors() {}

        @InputFunction
        public static void input(
                @AggregationState PcaState state,
                @SqlType("array(double)") Block vector,
                @SqlType(StandardTypes.BIGINT) long k)
        {
            accumulate(state, vector, k, DOUBLE_READER);
        }

        @CombineFunction
        public static void combine(@AggregationState PcaState state, @AggregationState PcaState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction("row(directions array(array(double)), explained_variance_ratio array(double))")
        public static void output(@AggregationState PcaState state, BlockBuilder out)
        {
            writeResult(state, out, false);
        }
    }

    @AggregationFunction("vector_pca_agg")
    @Description("Returns the k leading principal directions of the vectors in each group, with the share of the variance each carries")
    public static final class OfRealVectors
    {
        private OfRealVectors() {}

        @InputFunction
        public static void input(
                @AggregationState PcaState state,
                @SqlType("array(real)") Block vector,
                @SqlType(StandardTypes.BIGINT) long k)
        {
            accumulate(state, vector, k, REAL_READER);
        }

        @CombineFunction
        public static void combine(@AggregationState PcaState state, @AggregationState PcaState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction("row(directions array(array(real)), explained_variance_ratio array(double))")
        public static void output(@AggregationState PcaState state, BlockBuilder out)
        {
            writeResult(state, out, true);
        }
    }

    /**
     * A vector with a null component is skipped whole, as {@code vector_avg_agg} skips it: fitting
     * its other components would leave each dimension's variance computed over a different set of
     * rows, which is no longer the covariance of anything.
     */
    private static void accumulate(PcaState state, Block vector, long k, VectorReader reader)
    {
        if (k <= 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "k must be greater than zero, got " + k);
        }
        long seen = state.getComponents();
        if (seen == 0) {
            state.setComponents(k);
        }
        else {
            checkConstantWithinGroup(seen, k);
        }
        if (vector.hasNull()) {
            return;
        }

        Covariance covariance = state.getCovariance();
        if (covariance == null) {
            covariance = new Covariance(vector.getPositionCount());
            state.setCovariance(covariance);
        }
        covariance.add(vector, reader);
    }

    private static void mergeStates(PcaState state, PcaState otherState)
    {
        Covariance other = otherState.getCovariance();
        if (other == null) {
            return;
        }
        long seen = state.getComponents();
        if (seen == 0) {
            state.setComponents(otherState.getComponents());
        }
        else {
            checkConstantWithinGroup(seen, otherState.getComponents());
        }

        Covariance covariance = state.getCovariance();
        if (covariance == null) {
            state.setCovariance(other);
        }
        else {
            covariance.merge(other);
        }
    }

    /**
     * The number of directions is read from whichever row reaches a group first, so a k varying
     * within a group would otherwise be resolved silently by "first one wins".
     */
    private static void checkConstantWithinGroup(long k, long otherK)
    {
        if (k != otherK) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "k must be constant within a group of vector_pca_agg, found %s and %s".formatted(k, otherK));
        }
    }

    private static void writeResult(PcaState state, BlockBuilder out, boolean realDirections)
    {
        Covariance covariance = state.getCovariance();
        if (covariance == null) {
            out.appendNull();
            return;
        }
        PrincipalComponents fit = PrincipalComponents.fit(covariance, state.getComponents());
        ((RowBlockBuilder) out).buildEntry(fieldBuilders -> {
            ((ArrayBlockBuilder) fieldBuilders.get(0)).buildEntry(directionsBuilder -> {
                for (double[] direction : fit.directions()) {
                    ((ArrayBlockBuilder) directionsBuilder).buildEntry(elementBuilder -> {
                        for (double component : direction) {
                            if (realDirections) {
                                REAL.writeFloat(elementBuilder, (float) component);
                            }
                            else {
                                DOUBLE.writeDouble(elementBuilder, component);
                            }
                        }
                    });
                }
            });
            ((ArrayBlockBuilder) fieldBuilders.get(1)).buildEntry(elementBuilder -> {
                for (double ratio : fit.explainedVarianceRatio()) {
                    DOUBLE.writeDouble(elementBuilder, ratio);
                }
            });
        });
    }
}
