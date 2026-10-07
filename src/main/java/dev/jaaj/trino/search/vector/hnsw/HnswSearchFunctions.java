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

import dev.jaaj.trino.search.vector.VectorReader;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.RowType;
import io.trino.spi.type.StandardTypes;

import java.util.List;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static dev.jaaj.trino.search.vector.hnsw.HnswBuildAggregation.MAX_EF;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.DoubleType.DOUBLE;

public final class HnswSearchFunctions
{
    /**
     * The cap {@code knn_agg} puts on k, for the same reason: the result is built in memory before
     * anything is returned.
     */
    static final int MAX_K = 10_000;

    private static final String RESULT_TYPE_SIGNATURE = "array(row(key bigint, distance double))";
    private static final RowType NEIGHBOUR_TYPE = RowType.from(List.of(
            RowType.field("key", BIGINT),
            RowType.field("distance", DOUBLE)));

    private HnswSearchFunctions() {}

    @Description("Returns the approximate k nearest neighbours of a query vector in a graph built by hnsw_build_agg")
    @ScalarFunction("hnsw_search")
    @SqlType(RESULT_TYPE_SIGNATURE)
    @SqlNullable
    public static Block search(
            @SqlType(StandardTypes.VARBINARY) Slice graph,
            @SqlType("array(double)") Block query,
            @SqlType(StandardTypes.BIGINT) long k,
            @SqlType(StandardTypes.BIGINT) long efSearch)
    {
        return searchGraph(graph, query, DOUBLE_READER, k, efSearch);
    }

    @Description("Returns the approximate k nearest neighbours of a query vector in a graph built by hnsw_build_agg")
    @ScalarFunction("hnsw_search")
    @SqlType(RESULT_TYPE_SIGNATURE)
    @SqlNullable
    public static Block searchReal(
            @SqlType(StandardTypes.VARBINARY) Slice graph,
            @SqlType("array(real)") Block query,
            @SqlType(StandardTypes.BIGINT) long k,
            @SqlType(StandardTypes.BIGINT) long efSearch)
    {
        return searchGraph(graph, query, REAL_READER, k, efSearch);
    }

    private static Block searchGraph(Slice graph, Block query, VectorReader queryReader, long k, long efSearch)
    {
        if (k <= 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "k must be greater than zero, got " + k);
        }
        if (k > MAX_K) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "k of hnsw_search must be less than or equal to %s; found %s".formatted(MAX_K, k));
        }
        if (efSearch < k || efSearch > MAX_EF) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "ef_search must be between k (%s) and %s, got %s".formatted(k, MAX_EF, efSearch));
        }
        HnswGraph hnswGraph = HnswGraph.read(graph);
        if (query.hasNull()) {
            return null;
        }

        List<HnswGraph.Neighbour> neighbours = hnswGraph.search(query, queryReader, (int) k, (int) efSearch);
        BlockBuilder builder = NEIGHBOUR_TYPE.createBlockBuilder(null, neighbours.size());
        for (HnswGraph.Neighbour neighbour : neighbours) {
            ((RowBlockBuilder) builder).buildEntry(fieldBuilders -> {
                BIGINT.writeLong(fieldBuilders.get(0), neighbour.key());
                DOUBLE.writeDouble(fieldBuilders.get(1), neighbour.distance());
            });
        }
        return builder.build();
    }
}
