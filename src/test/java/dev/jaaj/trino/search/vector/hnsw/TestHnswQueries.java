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
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedResult;
import io.trino.testing.MaterializedRow;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import java.util.List;

import static io.trino.testing.QueryAssertions.assertEqualsIgnoreOrder;
import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

public class TestHnswQueries
        extends AbstractTestQueryFramework
{
    private static final String DOCUMENTS =
            """
            (VALUES
                (1, ARRAY[DOUBLE '0', DOUBLE '0'], DATE '2026-08-01'),
                (2, ARRAY[DOUBLE '1', DOUBLE '0'], DATE '2026-08-01'),
                (3, ARRAY[DOUBLE '3', DOUBLE '4'], DATE '2026-08-01'),
                (4, ARRAY[DOUBLE '0', DOUBLE '-2'], DATE '2026-08-02'),
                (5, ARRAY[DOUBLE '6', DOUBLE '8'], DATE '2026-08-02'),
                (6, ARRAY[DOUBLE '-1', DOUBLE '1'], DATE '2026-08-03'))
            AS documents(id, embedding, day)
            """;

    private static final String GRAPH = "(SELECT hnsw_build_agg(id, embedding, 4, 8, 'euclidean') AS graph FROM " + DOCUMENTS + ")";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testReturnsTheNearestKeysAndDistancesNearestFirst()
    {
        assertThat(computeActual(
                "SELECT n.key, n.distance FROM %s CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 3, 6)) WITH ORDINALITY AS n(key, distance, rank) ORDER BY n.rank"
                        .formatted(GRAPH))
                .getMaterializedRows())
                .extracting(MaterializedRow::getFields)
                .containsExactly(
                        List.of(1L, 0.0),
                        List.of(2L, 1.0),
                        List.of(6L, Math.sqrt(2)));
    }

    @Test
    public void testBuildReturnsAVarbinary()
    {
        assertThat(computeScalar("SELECT typeof(graph) FROM " + GRAPH)).isEqualTo("varbinary");
    }

    @Test
    public void testSearchReturnsNamedRows()
    {
        assertThat(computeScalar("SELECT typeof(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 1, 1)) FROM " + GRAPH))
                .isEqualTo("array(row(\"key\" bigint, \"distance\" double))");
        assertThat(computeScalar("SELECT hnsw_search(graph, ARRAY[DOUBLE '3', DOUBLE '4'], 1, 1)[1].key FROM " + GRAPH))
                .isEqualTo(3L);
    }

    @Test
    public void testRealVectors()
    {
        assertQuery(
                """
                SELECT n.key, n.distance
                FROM (SELECT hnsw_build_agg(id, CAST(embedding AS array(real)), 4, 8, 'manhattan') AS graph FROM %s)
                CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[REAL '3', REAL '3'], 1, 4)) AS n(key, distance)
                """.formatted(DOCUMENTS),
                "VALUES (3, 1.0)");
    }

    /**
     * An untyped decimal literal is the way a query vector is usually written by hand, and it has
     * to resolve to one overload rather than fail as ambiguous.
     */
    @Test
    public void testDecimalLiteralQueryVector()
    {
        assertQuery(
                "SELECT hnsw_search(graph, ARRAY[0.9, 0.1], 1, 1)[1].key FROM " + GRAPH,
                "VALUES 2");
    }

    @Test
    public void testDotProductReturnsTheHighestSimilarityFirst()
    {
        assertQuery(
                """
                SELECT n.key, n.distance
                FROM (SELECT hnsw_build_agg(id, embedding, 4, 8, 'dot_product') AS graph FROM %s)
                CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[DOUBLE '1', DOUBLE '0'], 1, 6)) AS n(key, distance)
                """.formatted(DOCUMENTS),
                "VALUES (5, 6.0)");
    }

    @Test
    public void testMetricNameIsCaseInsensitive()
    {
        assertQuery(
                """
                SELECT hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '-1'], 1, 6)[1].key
                FROM (SELECT hnsw_build_agg(id, embedding, 4, 8, 'Euclidean') AS graph FROM %s)
                """.formatted(DOCUMENTS),
                "VALUES 4");
    }

    /**
     * The shape the plugin is for: one graph per partition, stored next to the partition key,
     * pruned by the same predicate as the base table and merged with an ordinary ORDER BY. With a
     * candidate list as long as each graph the merge must match the exact answer.
     */
    @Test
    public void testOneGraphPerPartitionMergedWithOrderBy()
    {
        MaterializedResult actual = computeActual(
                """
                WITH graphs AS (
                    SELECT day, hnsw_build_agg(id, embedding, 4, 8, 'euclidean') AS graph
                    FROM %s
                    GROUP BY day)
                SELECT n.key
                FROM graphs
                CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 2, 6)) AS n(key, distance)
                WHERE day BETWEEN DATE '2026-08-02' AND DATE '2026-08-03'
                ORDER BY n.distance
                LIMIT 2
                """.formatted(DOCUMENTS));
        MaterializedResult expected = computeActual(
                """
                SELECT CAST(id AS bigint)
                FROM %s
                WHERE day BETWEEN DATE '2026-08-02' AND DATE '2026-08-03'
                ORDER BY euclidean_distance(embedding, ARRAY[DOUBLE '0', DOUBLE '0'])
                LIMIT 2
                """.formatted(DOCUMENTS));
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testEmptyGroupBuildsNoGraph()
    {
        assertQuery(
                "SELECT hnsw_build_agg(id, embedding, 4, 8, 'euclidean') IS NULL FROM %s WHERE id < 0".formatted(DOCUMENTS),
                "VALUES true");
    }

    @Test
    public void testRowsThatCannotBePlacedAreIgnored()
    {
        assertQuery(
                """
                SELECT cardinality(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 10, 10))
                FROM (
                    SELECT hnsw_build_agg(id, embedding, 4, 8, 'euclidean') AS graph
                    FROM (VALUES
                        (BIGINT '1', ARRAY[DOUBLE '0', DOUBLE '0']),
                        (NULL, ARRAY[DOUBLE '1', DOUBLE '1']),
                        (BIGINT '3', NULL),
                        (BIGINT '4', ARRAY[DOUBLE '2', NULL]),
                        (BIGINT '5', ARRAY[DOUBLE '3', DOUBLE '3'])) AS t(id, embedding))
                """,
                "VALUES 2");
    }

    /**
     * A row with a null component is skipped before it can set the group's dimension, as in
     * {@code vector_avg_agg}.
     */
    @Test
    public void testRowWithANullComponentDoesNotSetTheDimension()
    {
        assertQuery(
                """
                SELECT cardinality(hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 10, 10))
                FROM (
                    SELECT hnsw_build_agg(id, embedding, 4, 8, 'euclidean') AS graph
                    FROM (VALUES
                        (BIGINT '1', ARRAY[DOUBLE '0', DOUBLE '0', NULL]),
                        (BIGINT '2', ARRAY[DOUBLE '3', DOUBLE '3'])) AS t(id, embedding))
                """,
                "VALUES 1");
    }

    @Test
    public void testQueryWithANullComponentReturnsNull()
    {
        assertQuery(
                "SELECT hnsw_search(graph, ARRAY[DOUBLE '0', NULL], 1, 1) IS NULL FROM " + GRAPH,
                "VALUES true");
    }

    @Test
    public void testNullGraphReturnsNull()
    {
        assertQuery(
                "SELECT hnsw_search(CAST(NULL AS varbinary), ARRAY[DOUBLE '0'], 1, 1) IS NULL",
                "VALUES true");
    }

    @Test
    public void testMOutOfRangeIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_build_agg(id, embedding, 1, 8, 'euclidean') FROM " + DOCUMENTS,
                "m must be between 2 and 256, got 1");
        assertQueryFails(
                "SELECT hnsw_build_agg(id, embedding, 257, 300, 'euclidean') FROM " + DOCUMENTS,
                "m must be between 2 and 256, got 257");
    }

    @Test
    public void testEfConstructionBelowMIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_build_agg(id, embedding, 16, 8, 'euclidean') FROM " + DOCUMENTS,
                "ef_construction must be between m \\(16\\) and 10000, got 8");
    }

    @Test
    public void testUnknownMetricIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_build_agg(id, embedding, 4, 8, 'hamming') FROM " + DOCUMENTS,
                "Unknown metric 'hamming'.*");
    }

    @Test
    public void testParametersMustBeConstantWithinAGroup()
    {
        assertQueryFails(
                "SELECT hnsw_build_agg(id, embedding, CAST(4 + id % 2 AS bigint), 8, 'euclidean') FROM " + DOCUMENTS,
                "m must be constant within a group of hnsw_build_agg, found .* and .*");
        assertQueryFails(
                "SELECT hnsw_build_agg(id, embedding, 4, 8, IF(id = 3, 'cosine', 'euclidean')) FROM " + DOCUMENTS,
                "metric must be constant within a group of hnsw_build_agg, found '.*' and '.*'");
    }

    @Test
    public void testMixedDimensionsAreRejected()
    {
        assertQueryFails(
                "SELECT hnsw_build_agg(id, IF(id = 3, ARRAY[DOUBLE '1'], embedding), 4, 8, 'euclidean') FROM " + DOCUMENTS,
                "The vectors of hnsw_build_agg must have the same length, found .* and .*");
    }

    @Test
    public void testKOutOfRangeIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 0, 1) FROM " + GRAPH,
                "k must be greater than zero, got 0");
        assertQueryFails(
                "SELECT hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 10001, 10001) FROM " + GRAPH,
                "k of hnsw_search must be less than or equal to 10000; found 10001");
    }

    @Test
    public void testEfSearchBelowKIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 5, 4) FROM " + GRAPH,
                "ef_search must be between k \\(5\\) and 10000, got 4");
        assertQueryFails(
                "SELECT hnsw_search(graph, ARRAY[DOUBLE '0', DOUBLE '0'], 5, 10001) FROM " + GRAPH,
                "ef_search must be between k \\(5\\) and 10000, got 10001");
    }

    @Test
    public void testQueryOfAnotherDimensionIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_search(graph, ARRAY[DOUBLE '0'], 1, 1) FROM " + GRAPH,
                "The query vector must have the dimension of the graph, found 1 and 2");
    }

    @Test
    public void testValueThatIsNotAGraphIsRejected()
    {
        assertQueryFails(
                "SELECT hnsw_search(from_hex('0011223344'), ARRAY[DOUBLE '0'], 1, 1)",
                "The value is not a graph built by hnsw_build_agg");
    }
}
