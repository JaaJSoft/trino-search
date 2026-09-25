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
package dev.jaaj.trino.search.vector.centroid;

import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
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
import static io.trino.spi.type.RealType.REAL;

/**
 * The element-wise mean of a group of vectors: the centroid update of Lloyd's algorithm, and the
 * usual way to pool chunk embeddings into a document embedding.
 * <p>
 * Both overloads accumulate in {@code double}. A {@code float} accumulator stops registering small
 * components once the running sum is large enough, and at a million rows that is the common case,
 * not an edge.
 */
public final class VectorAvgAggregation
{
    private VectorAvgAggregation() {}

    @AggregationFunction("vector_avg_agg")
    @Description("Returns the element-wise mean of the vectors in each group")
    public static final class OfDoubleVectors
    {
        private OfDoubleVectors() {}

        @InputFunction
        public static void input(@AggregationState VectorSumState state, @SqlType("array(double)") Block vector)
        {
            if (!vector.hasNull()) {
                state.accumulate(vector, DOUBLE_READER);
            }
        }

        @CombineFunction
        public static void combine(@AggregationState VectorSumState state, @AggregationState VectorSumState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction("array(double)")
        public static void output(@AggregationState VectorSumState state, BlockBuilder out)
        {
            double[] sums = state.getSums();
            if (sums == null) {
                out.appendNull();
                return;
            }
            double count = state.getCount();
            ((ArrayBlockBuilder) out).buildEntry(elementBuilder -> {
                for (double sum : sums) {
                    DOUBLE.writeDouble(elementBuilder, sum / count);
                }
            });
        }
    }

    @AggregationFunction("vector_avg_agg")
    @Description("Returns the element-wise mean of the vectors in each group")
    public static final class OfRealVectors
    {
        private OfRealVectors() {}

        @InputFunction
        public static void input(@AggregationState VectorSumState state, @SqlType("array(real)") Block vector)
        {
            if (!vector.hasNull()) {
                state.accumulate(vector, REAL_READER);
            }
        }

        @CombineFunction
        public static void combine(@AggregationState VectorSumState state, @AggregationState VectorSumState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction("array(real)")
        public static void output(@AggregationState VectorSumState state, BlockBuilder out)
        {
            double[] sums = state.getSums();
            if (sums == null) {
                out.appendNull();
                return;
            }
            double count = state.getCount();
            ((ArrayBlockBuilder) out).buildEntry(elementBuilder -> {
                for (double sum : sums) {
                    REAL.writeFloat(elementBuilder, (float) (sum / count));
                }
            });
        }
    }

    private static void mergeStates(VectorSumState state, VectorSumState otherState)
    {
        double[] otherSums = otherState.getSums();
        if (otherSums != null) {
            state.merge(otherSums, otherState.getCount());
        }
    }
}
