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

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedResult;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

public class TestNearestVectorQueries
        extends AbstractTestQueryFramework
{
    private static final String CANDIDATES =
            "ARRAY[ARRAY[DOUBLE '3', DOUBLE '4'], ARRAY[DOUBLE '1', DOUBLE '0'], ARRAY[DOUBLE '-2', DOUBLE '0']]";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testReturnsThePositionAndDistanceOfTheNearestCandidate()
    {
        assertQuery(
                "SELECT n.position, n.distance FROM (SELECT nearest_vector(ARRAY[DOUBLE '0', DOUBLE '0'], %s, 'euclidean') AS n)"
                        .formatted(CANDIDATES),
                "SELECT 2, 1.0");
    }

    @Test
    public void testResultIsANamedRow()
    {
        assertThat(computeScalar("SELECT typeof(nearest_vector(ARRAY[DOUBLE '0', DOUBLE '0'], %s, 'euclidean'))"
                .formatted(CANDIDATES)))
                .isEqualTo("row(\"position\" integer, \"distance\" double)");
    }

    @Test
    public void testRealVectorsAreAccepted()
    {
        assertQuery(
                """
                SELECT n.position, n.distance
                FROM (SELECT nearest_vector(ARRAY[REAL '0', REAL '0'], ARRAY[ARRAY[REAL '0', REAL '5'], ARRAY[REAL '0', REAL '-2']], 'manhattan') AS n)
                """,
                "SELECT 2, 2.0");
    }

    @Test
    public void testMetricNameIsCaseInsensitive()
    {
        assertQuery(
                "SELECT nearest_vector(ARRAY[DOUBLE '0', DOUBLE '0'], %s, 'Euclidean').position".formatted(CANDIDATES),
                "SELECT 2");
    }

    @Test
    public void testHigherIsCloserForDotProduct()
    {
        assertQuery(
                "SELECT nearest_vector(ARRAY[DOUBLE '-1', DOUBLE '0'], %s, 'dot_product').position".formatted(CANDIDATES),
                "SELECT 3");
    }

    @Test
    public void testUnknownMetricIsRejected()
    {
        assertQueryFails(
                "SELECT nearest_vector(ARRAY[DOUBLE '0', DOUBLE '0'], %s, 'hamming')".formatted(CANDIDATES),
                "Unknown metric 'hamming'.*");
    }

    @Test
    public void testCandidateOfAnotherDimensionIsRejected()
    {
        assertQueryFails(
                "SELECT nearest_vector(ARRAY[DOUBLE '0'], %s, 'euclidean')".formatted(CANDIDATES),
                "The arguments must have the same length");
    }

    @Test
    public void testVectorWithANullElementReturnsNull()
    {
        assertQuery(
                "SELECT nearest_vector(ARRAY[DOUBLE '0', NULL], %s, 'euclidean') IS NULL".formatted(CANDIDATES),
                "SELECT true");
    }

    @Test
    public void testNullCandidatesReturnNull()
    {
        assertQuery(
                "SELECT nearest_vector(ARRAY[DOUBLE '0'], CAST(NULL AS array(array(double))), 'euclidean') IS NULL",
                "SELECT true");
    }

    @Test
    public void testNoCandidatesReturnsNull()
    {
        assertQuery(
                "SELECT nearest_vector(ARRAY[DOUBLE '0'], CAST(ARRAY[] AS array(array(double))), 'euclidean') IS NULL",
                "SELECT true");
    }

    @Test
    public void testNullCandidateKeepsTheOthersPositions()
    {
        assertQuery(
                """
                SELECT nearest_vector(ARRAY[DOUBLE '0'], ARRAY[NULL, ARRAY[DOUBLE '9'], ARRAY[DOUBLE '1']], 'euclidean').position
                """,
                "SELECT 3");
    }

    /**
     * The assignment a table of centroids is meant to replace: a cross join that materialises one
     * row per centroid per input row and keeps the closest. The two must agree on every row,
     * including the equidistant row 4, which both resolve to the lowest cluster id.
     */
    @Test
    public void testAgreesWithTheCrossJoinAssignment()
    {
        String centroids = "(VALUES (1, ARRAY[DOUBLE '0', DOUBLE '0']), (2, ARRAY[DOUBLE '10', DOUBLE '0']), (3, ARRAY[DOUBLE '0', DOUBLE '10'])) AS c(cluster_id, centroid)";
        String rows = "(VALUES (1, ARRAY[DOUBLE '1', DOUBLE '1']), (2, ARRAY[DOUBLE '9', DOUBLE '-2']), (3, ARRAY[DOUBLE '-1', DOUBLE '12']), (4, ARRAY[DOUBLE '5', DOUBLE '0'])) AS r(id, embedding)";

        MaterializedResult assigned = computeActual(
                """
                SELECT id, nearest_vector(embedding, (SELECT array_agg(centroid ORDER BY cluster_id) FROM %s), 'euclidean').position
                FROM %s
                """.formatted(centroids, rows));
        MaterializedResult crossJoined = computeActual(
                """
                SELECT id, CAST(min_by(cluster_id, ROW(euclidean_distance(embedding, centroid), cluster_id)) AS integer)
                FROM %s CROSS JOIN %s
                GROUP BY id
                """.formatted(rows, centroids));

        assertThat(assigned.getMaterializedRows()).containsExactlyInAnyOrderElementsOf(crossJoined.getMaterializedRows());
    }

    /**
     * The size the issue that introduced this function worried about: 1024 centroids of dimension
     * 768, about 3 MB as {@code array(array(real))}, handed to every row through an uncorrelated
     * scalar subquery. Each row is one of the centroids, so it must land on its own position at
     * distance zero.
     */
    @Test
    public void testAssignsAgainstAFullSizeCodebook()
    {
        MaterializedResult result = computeActual(
                """
                WITH centroids AS (
                    SELECT c AS cluster_id, transform(sequence(1, 768), d -> CAST(sin(c * d) AS real)) AS centroid
                    FROM UNNEST(sequence(1, 1024)) AS t(c)
                ),
                codebook AS (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids)
                SELECT count(*), count_if(nearest_vector(c.centroid, b.centroids, 'euclidean').position = c.cluster_id
                                          AND nearest_vector(c.centroid, b.centroids, 'euclidean').distance = 0)
                FROM centroids c CROSS JOIN codebook b
                WHERE c.cluster_id % 97 = 0
                """);

        assertThat(result.getMaterializedRows().getFirst().getFields()).containsExactly(10L, 10L);
    }

    /**
     * The subquery reaches the function through a join against a single row, not as a literal
     * serialised into the plan, so the plan stays small however large the codebook is.
     */
    @Test
    public void testCodebookIsNotInlinedIntoThePlan()
    {
        String plan = (String) computeActual(
                """
                EXPLAIN
                WITH centroids AS (
                    SELECT c AS cluster_id, transform(sequence(1, 768), d -> CAST(sin(c * d) AS real)) AS centroid
                    FROM UNNEST(sequence(1, 1024)) AS t(c)
                )
                SELECT nearest_vector(centroid, (SELECT array_agg(centroid ORDER BY cluster_id) FROM centroids), 'euclidean')
                FROM centroids
                """).getOnlyValue();

        assertThat(plan.length()).isLessThan(100_000);
    }
}
