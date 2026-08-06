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

public class TestKnnAggregationDistributed
        extends AbstractTestQueryFramework
{
    /**
     * A vector built from orderkey so that the nearest neighbours of the origin are the smallest
     * keys. Using a TPCH table spreads the rows across splits, which is what forces Trino to run
     * the partial and final aggregation steps and therefore to serialize the state.
     */
    private static final String VECTORS =
            """
            (SELECT orderkey AS id, ARRAY[CAST(orderkey AS double), 0.0] AS v FROM tpch.tiny.orders)
            """;

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
     * Confirms the queries in this class actually run a partial aggregation on each split plus a
     * final aggregation that merges them, rather than collapsing to a single-stage aggregation.
     * Only that shape exercises {@code KnnStateSerializer} and {@code @CombineFunction}.
     */
    @Test
    public void testPlanHasPartialAndFinalAggregation()
    {
        MaterializedResult plan = computeActual(
                "EXPLAIN (TYPE DISTRIBUTED) SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 10, 'euclidean') FROM " + VECTORS);
        String text = (String) plan.getOnlyValue();
        assertThat(text).contains("PARTIAL");
        assertThat(text).contains("FINAL");
    }

    /**
     * The query vector's second element is CAST(orderkey AS double), i.e. already a double, so
     * ARRAY[CAST(orderkey AS double), 0.0] resolves to array(double) (one typed element is enough
     * to rule out the array(real) overload, which array(decimal) would otherwise coerce to
     * preferentially). Distinguishing the overload is not needed here the way it is in
     * TestKnnAggregation: the point of this class is the serialize/combine cycle, which is shared
     * verbatim between OfDoubleVectors and OfRealVectors.
     * <p>
     * These tests compare two {@code computeActual} results instead of using {@code assertQuery}:
     * that helper runs its expected side through the framework's H2 comparison database rather
     * than the query runner under test, and H2 has neither the {@code tpch} schema nor this
     * project's scalar distance functions. Running both sides through {@code computeActual} keeps
     * them on the same Trino instance, which has both installed.
     */
    @Test
    public void testMatchesOrderByLimitAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 10, 'euclidean'), x -> x[1]) FROM " + VECTORS);
        MaterializedResult expected = computeActual(
                "SELECT array_agg(id ORDER BY id) FROM (SELECT orderkey AS id FROM tpch.tiny.orders ORDER BY orderkey LIMIT 10)");
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testDistancesMatchTheNativeFunctionAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 5, 'euclidean'), x -> x[2]) FROM " + VECTORS);
        MaterializedResult expected = computeActual(
                """
                SELECT array_agg(d ORDER BY d) FROM (
                    SELECT euclidean_distance(ARRAY[CAST(orderkey AS double), 0.0], ARRAY[0.0, 0.0]) AS d
                    FROM tpch.tiny.orders ORDER BY d LIMIT 5)
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testGroupedAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT orderstatus, cardinality(knn_agg(orderkey, ARRAY[CAST(orderkey AS double), 0.0], ARRAY[0.0, 0.0], 3, 'euclidean'))
                FROM tpch.tiny.orders
                GROUP BY orderstatus
                ORDER BY orderstatus
                """);
        MaterializedResult expected = computeActual(
                "SELECT orderstatus, CAST(3 AS bigint) FROM tpch.tiny.orders GROUP BY orderstatus ORDER BY orderstatus");
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testCosineMetricMatchesTheNativeFunctionAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT transform(knn_agg(orderkey, ARRAY[CAST(orderkey AS double), 1.0], ARRAY[1.0, 1.0], 5, 'cosine'), x -> x[1])
                FROM tpch.tiny.orders
                """);
        MaterializedResult expected = computeActual(
                """
                SELECT array_agg(orderkey ORDER BY d, orderkey) FROM (
                    SELECT orderkey, cosine_distance(ARRAY[CAST(orderkey AS double), 1.0], ARRAY[1.0, 1.0]) AS d
                    FROM tpch.tiny.orders ORDER BY d, orderkey LIMIT 5)
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }

    /**
     * {@code dot_product} is the only metric with {@code higherIsCloser = true}: every other
     * metric tested in this class ranks smaller as closer. An inverted direction across the
     * serialize/combine boundary (for example a heap that keeps sifting toward the smallest dot
     * product instead of the largest) would return the farthest neighbours while looking like a
     * plausible answer, which single-partition coverage cannot catch.
     */
    @Test
    public void testDotProductMetricMatchesTheNativeFunctionAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT transform(knn_agg(orderkey, ARRAY[CAST(orderkey AS double), 1.0], ARRAY[1.0, 1.0], 5, 'dot_product'), x -> x[1])
                FROM tpch.tiny.orders
                """);
        MaterializedResult expected = computeActual(
                """
                SELECT array_agg(orderkey ORDER BY d DESC, orderkey) FROM (
                    SELECT orderkey, dot_product(ARRAY[CAST(orderkey AS double), 1.0], ARRAY[1.0, 1.0]) AS d
                    FROM tpch.tiny.orders ORDER BY d DESC, orderkey LIMIT 5)
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }
}
