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
package dev.jaaj.trino.search.vector.quantize;

import dev.jaaj.trino.search.vector.VectorReader;
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

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;

/**
 * Fits per-dimension quantisation parameters over a corpus in one pass.
 * <p>
 * The state keeps the minima and maxima, because that is what merges associatively across splits.
 * The offset and scale the kernels want are derived once, at output.
 */
public final class VectorBoundsAggregation
{
    private VectorBoundsAggregation() {}

    @AggregationFunction("vector_bounds_agg")
    @Description("Fits per-dimension quantisation bounds over a corpus of vectors")
    public static final class OfDoubleVectors
    {
        private OfDoubleVectors() {}

        @InputFunction
        public static void input(@AggregationState BoundsState state, @SqlType("array(double)") Block vector)
        {
            accumulate(state, vector, DOUBLE_READER);
        }

        @CombineFunction
        public static void combine(@AggregationState BoundsState state, @AggregationState BoundsState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction(QuantizationBounds.BOUNDS_TYPE_SIGNATURE)
        public static void output(@AggregationState BoundsState state, BlockBuilder out)
        {
            writeResult(state, out);
        }
    }

    @AggregationFunction("vector_bounds_agg")
    @Description("Fits per-dimension quantisation bounds over a corpus of vectors")
    public static final class OfRealVectors
    {
        private OfRealVectors() {}

        @InputFunction
        public static void input(@AggregationState BoundsState state, @SqlType("array(real)") Block vector)
        {
            accumulate(state, vector, REAL_READER);
        }

        @CombineFunction
        public static void combine(@AggregationState BoundsState state, @AggregationState BoundsState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction(QuantizationBounds.BOUNDS_TYPE_SIGNATURE)
        public static void output(@AggregationState BoundsState state, BlockBuilder out)
        {
            writeResult(state, out);
        }
    }

    /**
     * A vector with a null component says nothing usable about any dimension's range, so the whole
     * row is skipped rather than fitted dimension by dimension.
     */
    private static void accumulate(BoundsState state, Block vector, VectorReader reader)
    {
        if (vector.hasNull()) {
            return;
        }
        double[] values = new double[vector.getPositionCount()];
        for (int i = 0; i < values.length; i++) {
            values[i] = reader.read(vector, i);
        }
        state.accumulate(values);
    }

    private static void mergeStates(BoundsState state, BoundsState otherState)
    {
        double[] otherMinimums = otherState.getMinimums();
        if (otherMinimums != null) {
            state.merge(otherMinimums, otherState.getMaximums());
        }
    }

    private static void writeResult(BoundsState state, BlockBuilder out)
    {
        double[] minimums = state.getMinimums();
        if (minimums == null) {
            out.appendNull();
            return;
        }
        double[] maximums = state.getMaximums();
        ((RowBlockBuilder) out).buildEntry(fieldBuilders -> {
            ((ArrayBlockBuilder) fieldBuilders.get(0)).buildEntry(elementBuilder -> {
                for (int i = 0; i < minimums.length; i++) {
                    DOUBLE.writeDouble(elementBuilder, (minimums[i] + maximums[i]) / 2);
                }
            });
            ((ArrayBlockBuilder) fieldBuilders.get(1)).buildEntry(elementBuilder -> {
                for (int i = 0; i < minimums.length; i++) {
                    DOUBLE.writeDouble(elementBuilder, (maximums[i] - minimums[i]) / QuantizationBounds.CODE_LEVELS);
                }
            });
        });
    }
}
