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
 * On a single partition Trino never serializes an aggregation state, so a partial state that
 * dropped rows or a deserializer that leaked rows from one position into the next would pass every
 * other test. A TPCH table read as four splits forces the partial and final steps.
 */
public class TestHnswBuildAggregationDistributed
        extends AbstractTestQueryFramework
{
    private static final String BUILD =
            "hnsw_build_agg(orderkey, ARRAY[CAST(custkey AS double), CAST(orderkey AS double) / 100], 8, 32, 'euclidean')";

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

    @Test
    public void testPlanHasPartialAndFinalAggregation()
    {
        String plan = (String) computeActual("EXPLAIN (TYPE DISTRIBUTED) SELECT " + BUILD + " FROM orders").getOnlyValue();
        assertThat(plan).contains("PARTIAL");
        assertThat(plan).contains("FINAL");
    }

    /**
     * The graph is a function of the set of rows, so the one assembled from four serialized partial
     * states has to be byte for byte the one built from the rows in a single step.
     */
    @Test
    public void testGraphAcrossSplitsIsTheGraphOfASingleStep()
    {
        Session singleStep = Session.builder(getSession())
                .setSystemProperty("prefer_partial_aggregation", "false")
                .build();
        String query = "SELECT to_hex(sha256(" + BUILD + ")) FROM orders";
        assertThat((String) computeActual(singleStep, "EXPLAIN (TYPE DISTRIBUTED) " + query).getOnlyValue())
                .doesNotContain("PARTIAL");

        assertThat(computeScalar(query)).isEqualTo(computeActual(singleStep, query).getOnlyValue());
    }

    /**
     * The int8 partial state carries the bounds as well as the codes, and the binary one carries
     * whole codes rather than components, so each has its own serialized form to get wrong.
     */
    @Test
    public void testQuantisedGraphsAcrossSplitsAreTheGraphsOfASingleStep()
    {
        Session singleStep = Session.builder(getSession())
                .setSystemProperty("prefer_partial_aggregation", "false")
                .build();
        for (String codes : new String[] {"quantize_vector_tinyint(v, bounds), bounds", "quantize_vector_varbinary(v, bounds)"}) {
            String query =
                    """
                    WITH vectors AS (SELECT orderkey, ARRAY[CAST(custkey AS double), CAST(orderkey AS double) / 100] AS v FROM orders),
                    params AS (SELECT vector_bounds_agg(v) AS bounds FROM vectors)
                    SELECT to_hex(sha256(hnsw_build_agg(orderkey, %s, 8, 32, 'euclidean')))
                    FROM vectors CROSS JOIN params
                    """.formatted(codes);
            assertThat(computeScalar(query)).as(codes).isEqualTo(computeActual(singleStep, query).getOnlyValue());
        }
    }

    @Test
    public void testGraphHoldsEveryRowAcrossSplits()
    {
        assertThat(computeScalar(
                """
                SELECT cardinality(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 10000, 10000))
                FROM (SELECT %s AS graph FROM orders WHERE orderkey <= 20000)
                """.formatted(BUILD)))
                .isEqualTo(computeScalar("SELECT count(*) FROM orders WHERE orderkey <= 20000"));
    }

    @Test
    public void testSearchMatchesTheExactNeighboursAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT n.key
                FROM (SELECT %s AS graph FROM orders)
                CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[DOUBLE '700', DOUBLE '300'], 10, 200)) AS n(key, distance)
                """.formatted(BUILD));
        MaterializedResult expected = computeActual(
                """
                SELECT n.key
                FROM (SELECT knn_agg(orderkey, ARRAY[CAST(custkey AS double), CAST(orderkey AS double) / 100], ARRAY[DOUBLE '700', DOUBLE '300'], 10, 'euclidean') AS ns FROM orders)
                CROSS JOIN UNNEST(ns) AS n(key, distance)
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testGroupedAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT orderstatus, cardinality(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 3, 3))
                FROM (SELECT orderstatus, %s AS graph FROM orders GROUP BY orderstatus)
                """.formatted(BUILD));
        MaterializedResult expected = computeActual(
                "SELECT orderstatus, CAST(3 AS bigint) FROM orders GROUP BY orderstatus");
        assertEqualsIgnoreOrder(actual, expected);
    }
}
