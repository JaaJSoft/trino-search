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
package dev.jaaj.trino.search.vector.hnsw;

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.quantize.BinaryCodes;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.AggregationFunction;
import io.trino.spi.function.AggregationState;
import io.trino.spi.function.CombineFunction;
import io.trino.spi.function.Description;
import io.trino.spi.function.InputFunction;
import io.trino.spi.function.OutputFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.VarbinaryType.VARBINARY;

/**
 * Builds one HNSW graph over the vectors of each group and returns it as a varbinary, to be stored
 * in a table and searched with {@code hnsw_search}.
 * <p>
 * One nested class per vector representation, for the reason {@code KnnAggregation} gives.
 */
public final class HnswBuildAggregation
{
    static final int MIN_M = 2;
    static final int MAX_M = 256;
    static final int MAX_EF = 10_000;

    private HnswBuildAggregation() {}

    @AggregationFunction("hnsw_build_agg")
    @Description("Builds an HNSW graph over the vectors of each group and returns it serialized")
    public static final class OfDoubleVectors
    {
        private OfDoubleVectors() {}

        @InputFunction
        public static void input(
                @AggregationState HnswBuildState state,
                @SqlType(StandardTypes.BIGINT) long key,
                @SqlType("array(double)") Block vector,
                @SqlType(StandardTypes.BIGINT) long m,
                @SqlType(StandardTypes.BIGINT) long efConstruction,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            addRow(state, key, vector, m, efConstruction, metricName, ElementType.DOUBLE);
        }

        @CombineFunction
        public static void combine(@AggregationState HnswBuildState state, @AggregationState HnswBuildState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction(StandardTypes.VARBINARY)
        public static void output(@AggregationState HnswBuildState state, BlockBuilder out)
        {
            writeGraph(state, out);
        }
    }

    @AggregationFunction("hnsw_build_agg")
    @Description("Builds an HNSW graph over the vectors of each group and returns it serialized")
    public static final class OfRealVectors
    {
        private OfRealVectors() {}

        @InputFunction
        public static void input(
                @AggregationState HnswBuildState state,
                @SqlType(StandardTypes.BIGINT) long key,
                @SqlType("array(real)") Block vector,
                @SqlType(StandardTypes.BIGINT) long m,
                @SqlType(StandardTypes.BIGINT) long efConstruction,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            addRow(state, key, vector, m, efConstruction, metricName, ElementType.REAL);
        }

        @CombineFunction
        public static void combine(@AggregationState HnswBuildState state, @AggregationState HnswBuildState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction(StandardTypes.VARBINARY)
        public static void output(@AggregationState HnswBuildState state, BlockBuilder out)
        {
            writeGraph(state, out);
        }
    }

    /**
     * Codes are only comparable through the bounds they were fitted against, so the bounds are a
     * mandatory argument, stored once in the graph, and required to be the same on every row.
     */
    @AggregationFunction("hnsw_build_agg")
    @Description("Builds an HNSW graph over the vectors of each group and returns it serialized")
    public static final class OfQuantizedVectors
    {
        private OfQuantizedVectors() {}

        @InputFunction
        public static void input(
                @AggregationState HnswBuildState state,
                @SqlType(StandardTypes.BIGINT) long key,
                @SqlType("array(tinyint)") Block vector,
                @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow,
                @SqlType(StandardTypes.BIGINT) long m,
                @SqlType(StandardTypes.BIGINT) long efConstruction,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            checkParameters(m, efConstruction);
            QuantizationBounds bounds = QuantizationBounds.of(boundsRow);
            bounds.checkDimension(vector.getPositionCount());
            if (vector.hasNull()) {
                return;
            }
            GraphInput input = state.getInput();
            if (input == null) {
                state.setInput(new GraphInput(ElementType.INT8, Metric.fromName(metricName), (int) m, (int) efConstruction, vector.getPositionCount(), bounds));
            }
            else {
                input.checkSameParameters(m, efConstruction, metricName);
                input.checkSameBounds(bounds);
            }
            state.add(key, vector);
        }

        @CombineFunction
        public static void combine(@AggregationState HnswBuildState state, @AggregationState HnswBuildState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction(StandardTypes.VARBINARY)
        public static void output(@AggregationState HnswBuildState state, BlockBuilder out)
        {
            writeGraph(state, out);
        }
    }

    @AggregationFunction("hnsw_build_agg")
    @Description("Builds an HNSW graph over the vectors of each group and returns it serialized")
    public static final class OfBinaryVectors
    {
        private OfBinaryVectors() {}

        @InputFunction
        public static void input(
                @AggregationState HnswBuildState state,
                @SqlType(StandardTypes.BIGINT) long key,
                @SqlType(StandardTypes.VARBINARY) Slice vector,
                @SqlType(StandardTypes.BIGINT) long m,
                @SqlType(StandardTypes.BIGINT) long efConstruction,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            checkParameters(m, efConstruction);
            GraphInput input = state.getInput();
            if (input == null) {
                state.setInput(new GraphInput(ElementType.BINARY, Metric.fromName(metricName), (int) m, (int) efConstruction, BinaryCodes.dimension(vector), null));
            }
            else {
                input.checkSameParameters(m, efConstruction, metricName);
            }
            state.add(key, vector);
        }

        @CombineFunction
        public static void combine(@AggregationState HnswBuildState state, @AggregationState HnswBuildState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction(StandardTypes.VARBINARY)
        public static void output(@AggregationState HnswBuildState state, BlockBuilder out)
        {
            writeGraph(state, out);
        }
    }

    private static void addRow(HnswBuildState state, long key, Block vector, long m, long efConstruction, Slice metricName, ElementType elementType)
    {
        checkParameters(m, efConstruction);
        // A row that cannot be placed in the graph is skipped rather than failing the build, as in
        // knn_agg, and before it can fix the group's dimension.
        if (vector.hasNull()) {
            return;
        }

        GraphInput input = state.getInput();
        if (input == null) {
            Metric metric = Metric.fromName(metricName);
            state.setInput(new GraphInput(elementType, metric, (int) m, (int) efConstruction, vector.getPositionCount(), null));
        }
        else {
            input.checkSameParameters(m, efConstruction, metricName);
        }
        state.add(key, vector);
    }

    private static void checkParameters(long m, long efConstruction)
    {
        if (m < MIN_M || m > MAX_M) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "m must be between %s and %s, got %s".formatted(MIN_M, MAX_M, m));
        }
        if (efConstruction < m || efConstruction > MAX_EF) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "ef_construction must be between m (%s) and %s, got %s".formatted(m, MAX_EF, efConstruction));
        }
    }

    private static void mergeStates(HnswBuildState state, HnswBuildState otherState)
    {
        GraphInput other = otherState.getInput();
        if (other == null) {
            return;
        }
        if (state.getInput() == null) {
            state.setInput(other);
            return;
        }
        state.addAll(other);
    }

    private static void writeGraph(HnswBuildState state, BlockBuilder out)
    {
        GraphInput input = state.getInput();
        if (input == null) {
            out.appendNull();
            return;
        }
        VARBINARY.writeSlice(out, HnswGraphBuilder.build(input));
    }
}
