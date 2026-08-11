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
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.Session;
import io.trino.plugin.tpch.TpchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedResult;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static io.trino.testing.QueryAssertions.assertEqualsIgnoreOrder;
import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * The {@code OfQuantizedVectors} and {@code OfBinaryVectors} counterpart to
 * {@link TestKnnAggregationDistributed}: {@code UNNEST(sequence(...))} only forces local pipeline
 * parallelism, not a real partial/final split, so it cannot catch a broken combine either. A TPCH
 * table does.
 */
public class TestKnnAggregationQuantizedDistributed
        extends AbstractTestQueryFramework
{
    private static final String UNIT_BOUNDS_1D =
            "CAST(ROW(ARRAY[CAST(0.0 AS DOUBLE)], ARRAY[CAST(1.0 AS DOUBLE)]) "
                    + "AS row(offsets array(double), scales array(double)))";

    private static final String UNIT_BOUNDS_3D =
            "CAST(ROW(repeat(CAST(0.0 AS DOUBLE), 3), repeat(CAST(1.0 AS DOUBLE), 3)) "
                    + "AS row(offsets array(double), scales array(double)))";

    /**
     * orderkey reduced modulo 100 and centred on zero, so every quantised code round-trips exactly
     * under {@link #UNIT_BOUNDS_1D} (offset 0, scale 1) and stays well inside the signed byte
     * range. Using a TPCH table spreads the rows across splits, which is what forces Trino to run
     * the partial and final aggregation steps and therefore to serialize the state.
     */
    private static final String QUANTIZED_VECTORS =
            """
            (SELECT orderkey AS id, orderstatus,
                    quantize_vector_tinyint(ARRAY[CAST(orderkey %% 100 - 50 AS DOUBLE)], %s) AS v
             FROM tpch.tiny.orders)
            """.formatted(UNIT_BOUNDS_1D);

    private static final String QUANTIZED_QUERY_VECTOR = "ARRAY[CAST(0 AS TINYINT)]";

    /**
     * The low three bits of orderkey, read out one at a time so each becomes its own quantised
     * dimension: a value at or above the zero offset packs to a set bit, below it to a clear one.
     */
    private static final String BINARY_VECTORS =
            """
            (SELECT orderkey AS id, orderstatus,
                    quantize_vector_varbinary(
                        ARRAY[CAST(bitwise_and(orderkey, 1) AS DOUBLE) - 0.5,
                              CAST(bitwise_and(bitwise_right_shift(orderkey, 1), 1) AS DOUBLE) - 0.5,
                              CAST(bitwise_and(bitwise_right_shift(orderkey, 2), 1) AS DOUBLE) - 0.5],
                        %s) AS v
             FROM tpch.tiny.orders)
            """.formatted(UNIT_BOUNDS_3D);

    private static final String BINARY_QUERY_VECTOR =
            "quantize_vector_varbinary(repeat(CAST(-0.5 AS DOUBLE), 3), %s)".formatted(UNIT_BOUNDS_3D);

    @Override
    protected QueryRunner createQueryRunner()
    {
        Session session = testSessionBuilder()
                .setCatalog("tpch")
                .setSchema("tiny")
                .build();
        QueryRunner queryRunner = new StandaloneQueryRunner(session);
        queryRunner.installPlugin(new TpchPlugin());
        queryRunner.createCatalog("tpch", "tpch", Map.of("tpch.splits-per-node", "4"));
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    /**
     * Confirms the quantised query actually runs a partial aggregation on each split plus a final
     * aggregation that merges them, rather than collapsing to a single-stage aggregation. Only
     * that shape exercises {@code KnnStateSerializer} and {@code OfQuantizedVectors.combine}.
     */
    @Test
    public void testQuantizedPlanHasPartialAndFinalAggregation()
    {
        MaterializedResult plan = computeActual(
                "EXPLAIN (TYPE DISTRIBUTED) SELECT knn_agg(id, v, %s, %s, 10, 'euclidean') FROM %s"
                        .formatted(QUANTIZED_QUERY_VECTOR, UNIT_BOUNDS_1D, QUANTIZED_VECTORS));
        String text = (String) plan.getOnlyValue();
        assertThat(text).contains("PARTIAL");
        assertThat(text).contains("FINAL");
    }

    /**
     * The reduction modulo 100 gives every quantised code many rows, so several candidates tie on
     * distance and which of them the heap keeps is not defined. Comparing the multiset of
     * distances rather than the keys sidesteps that tie-break and still requires the combine to
     * have kept the true 10 smallest, across every split, to pass.
     */
    @Test
    public void testQuantizedDistancesMatchTheNativeFunctionAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                "SELECT transform(knn_agg(id, v, %s, %s, 10, 'euclidean'), x -> x[2]) FROM %s"
                        .formatted(QUANTIZED_QUERY_VECTOR, UNIT_BOUNDS_1D, QUANTIZED_VECTORS));
        MaterializedResult expected = computeActual(
                """
                SELECT array_agg(d ORDER BY d) FROM (
                    SELECT euclidean_distance(v, %s, %s) AS d FROM %s ORDER BY d LIMIT 10)
                """.formatted(QUANTIZED_QUERY_VECTOR, UNIT_BOUNDS_1D, QUANTIZED_VECTORS));
        assertEqualsIgnoreOrder(actual, expected);
    }

    /**
     * The binary counterpart of {@link #testQuantizedPlanHasPartialAndFinalAggregation}: only a
     * plan with a partial and a final stage exercises {@code OfBinaryVectors.combine}.
     */
    @Test
    public void testBinaryPlanHasPartialAndFinalAggregation()
    {
        MaterializedResult plan = computeActual(
                "EXPLAIN (TYPE DISTRIBUTED) SELECT knn_agg(id, v, %s, 10, 'euclidean') FROM %s"
                        .formatted(BINARY_QUERY_VECTOR, BINARY_VECTORS));
        String text = (String) plan.getOnlyValue();
        assertThat(text).contains("PARTIAL");
        assertThat(text).contains("FINAL");
    }

    /**
     * The three-bit code only has eight distinct values, so most candidates tie on distance; as in
     * {@link #testQuantizedDistancesMatchTheNativeFunctionAcrossSplits} the multiset of distances
     * is what the combine must get right, not any particular key.
     */
    @Test
    public void testBinaryDistancesMatchTheNativeFunctionAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                "SELECT transform(knn_agg(id, v, %s, 10, 'euclidean'), x -> x[2]) FROM %s"
                        .formatted(BINARY_QUERY_VECTOR, BINARY_VECTORS));
        MaterializedResult expected = computeActual(
                """
                SELECT array_agg(d ORDER BY d) FROM (
                    SELECT euclidean_distance(v, %s) AS d FROM %s ORDER BY d LIMIT 10)
                """.formatted(BINARY_QUERY_VECTOR, BINARY_VECTORS));
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testQuantizedGroupedAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT orderstatus, cardinality(knn_agg(id, v, %s, %s, 3, 'euclidean'))
                FROM %s
                GROUP BY orderstatus
                ORDER BY orderstatus
                """.formatted(QUANTIZED_QUERY_VECTOR, UNIT_BOUNDS_1D, QUANTIZED_VECTORS));
        MaterializedResult expected = computeActual(
                "SELECT orderstatus, CAST(3 AS bigint) FROM tpch.tiny.orders GROUP BY orderstatus ORDER BY orderstatus");
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testBinaryGroupedAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT orderstatus, cardinality(knn_agg(id, v, %s, 3, 'euclidean'))
                FROM %s
                GROUP BY orderstatus
                ORDER BY orderstatus
                """.formatted(BINARY_QUERY_VECTOR, BINARY_VECTORS));
        MaterializedResult expected = computeActual(
                "SELECT orderstatus, CAST(3 AS bigint) FROM tpch.tiny.orders GROUP BY orderstatus ORDER BY orderstatus");
        assertEqualsIgnoreOrder(actual, expected);
    }
}
