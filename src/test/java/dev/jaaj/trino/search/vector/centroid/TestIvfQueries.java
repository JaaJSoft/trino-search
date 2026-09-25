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
import io.trino.Session;
import io.trino.plugin.memory.MemoryPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedRow;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/**
 * The whole IVF workflow the documentation describes, run as SQL against a real engine: fit
 * centroids with Lloyd's algorithm, assign every row to its nearest centroid, then answer a query
 * by probing the nearest clusters only. These are properties of the recipe, not of a fixture, so
 * none of them depends on how good the fitted centroids happen to be.
 */
@TestInstance(PER_CLASS)
public class TestIvfQueries
        extends AbstractTestQueryFramework
{
    private static final int ROWS = 2000;
    private static final int DIMENSION = 16;
    private static final int CLUSTERS = 16;
    private static final int ITERATIONS = 6;
    private static final int K = 10;
    private static final List<Integer> QUERY_IDS = List.of(3, 250, 777, 1024, 1999);

    private final List<Double> inertiaByIteration = new ArrayList<>();

    @Override
    protected QueryRunner createQueryRunner()
    {
        Session session = testSessionBuilder()
                .setCatalog("memory")
                .setSchema("default")
                .build();
        QueryRunner queryRunner = new StandaloneQueryRunner(session);
        queryRunner.installPlugin(new MemoryPlugin());
        queryRunner.createCatalog("memory", "memory", Map.of());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    /**
     * Eight blobs, each row its blob's centre plus a deterministic perturbation, so the corpus has
     * genuine cluster structure and every run sees the same rows. Sixteen centroids are fitted
     * over eight blobs, so a blob is typically split across several clusters and probing one
     * cluster is not the same as probing a whole blob.
     */
    @BeforeAll
    public void fitAndAssign()
    {
        assertUpdate(
                """
                CREATE TABLE sample AS
                SELECT id, transform(sequence(1, %s), d -> 10 * sin((id %% 8) * 7 + d * 3) + 3 * sin(id * 12.9898 + d * 78.233)) AS embedding
                FROM UNNEST(sequence(1, %s)) AS t(id)
                """.formatted(DIMENSION, ROWS),
                ROWS);

        // Random initialisation: any distinct rows will do, since the centroids only have to
        // partition the space. The ids must be dense from 1, because nearest_vector returns a
        // position in an array built in cluster_id order.
        assertUpdate(
                "CREATE TABLE centroids_0 AS SELECT CAST(id AS integer) AS cluster_id, embedding AS centroid FROM sample WHERE id <= " + CLUSTERS,
                CLUSTERS);

        for (int i = 0; i < ITERATIONS; i++) {
            inertiaByIteration.add(inertia("centroids_" + i));
            assertUpdate(
                    """
                    CREATE TABLE centroids_%2$s AS
                    WITH codebook AS (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids_%1$s),
                    assigned AS (
                        SELECT nearest_vector(s.embedding, b.centroids, 'euclidean').position AS cluster_id, s.embedding
                        FROM sample s CROSS JOIN codebook b
                    ),
                    means AS (SELECT cluster_id, vector_avg_agg(embedding) AS centroid FROM assigned GROUP BY cluster_id)
                    SELECT c.cluster_id, coalesce(m.centroid, c.centroid) AS centroid
                    FROM centroids_%1$s c
                    LEFT JOIN means m ON m.cluster_id = c.cluster_id
                    """.formatted(i, i + 1),
                    CLUSTERS);
        }
        inertiaByIteration.add(inertia("centroids_" + ITERATIONS));

        assertUpdate(
                """
                CREATE TABLE embeddings AS
                SELECT s.id, s.embedding, nearest_vector(s.embedding, b.centroids, 'euclidean').position AS cluster_id
                FROM sample s
                CROSS JOIN (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids_%s) b
                """.formatted(ITERATIONS),
                ROWS);
    }

    /**
     * The within-cluster sum of squares, with every row assigned to its nearest centroid.
     */
    private double inertia(String centroids)
    {
        return (double) computeScalar(
                """
                SELECT sum(nearest_vector(s.embedding, b.centroids, 'euclidean_squared').distance)
                FROM sample s
                CROSS JOIN (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM %s) b
                """.formatted(centroids));
    }

    /**
     * Lloyd's algorithm never increases the within-cluster sum of squares: assignment picks each
     * row's nearest centroid and the mean minimises the squared distance to a fixed set of rows.
     * An empty cluster that keeps its old centroid, as the {@code LEFT JOIN} and {@code coalesce}
     * arrange, does not break that. A wrong mean, a misrouted assignment or a renumbered cluster
     * would.
     */
    @Test
    public void testEveryIterationLowersOrKeepsTheInertia()
    {
        for (int i = 1; i < inertiaByIteration.size(); i++) {
            assertThat(inertiaByIteration.get(i))
                    .as("iteration %s", i)
                    .isLessThanOrEqualTo(inertiaByIteration.get(i - 1) * (1 + 1e-12));
        }
        assertThat(inertiaByIteration.getLast()).isLessThan(inertiaByIteration.getFirst());
    }

    /**
     * The last iteration recomputed with built-in functions only: every row assigned under the
     * previous centroids, then {@code avg} per cluster and dimension. The inertia test alone cannot
     * see a mean that is slightly off, since Lloyd still converges on slightly wrong means.
     * <p>
     * The tolerance only absorbs the summation order, which differs between the two and moves the
     * last bits of a sum of non-integer components.
     */
    @Test
    public void testFittedCentroidsAreTheMeansOfTheirAssignedRows()
    {
        List<MaterializedRow> expected = computeActual(
                """
                WITH codebook AS (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids_%1$s),
                assigned AS (
                    SELECT nearest_vector(s.embedding, b.centroids, 'euclidean').position AS cluster_id, s.embedding
                    FROM sample s CROSS JOIN codebook b
                )
                SELECT a.cluster_id, u.d, avg(u.e)
                FROM assigned a CROSS JOIN UNNEST(a.embedding) WITH ORDINALITY AS u(e, d)
                GROUP BY a.cluster_id, u.d
                ORDER BY a.cluster_id, u.d
                """.formatted(ITERATIONS - 1)).getMaterializedRows();
        List<MaterializedRow> actual = computeActual(
                """
                SELECT c.cluster_id, u.d, u.e
                FROM centroids_%s c CROSS JOIN UNNEST(c.centroid) WITH ORDINALITY AS u(e, d)
                WHERE c.cluster_id IN (SELECT DISTINCT nearest_vector(s.embedding, b.centroids, 'euclidean').position
                                       FROM sample s CROSS JOIN (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids_%s) b)
                ORDER BY c.cluster_id, u.d
                """.formatted(ITERATIONS, ITERATIONS - 1)).getMaterializedRows();

        assertThat(actual).hasSameSizeAs(expected);
        for (int i = 0; i < expected.size(); i++) {
            assertThat(actual.get(i).getField(0)).isEqualTo(expected.get(i).getField(0));
            assertThat(actual.get(i).getField(1)).isEqualTo(expected.get(i).getField(1));
            assertThat((double) actual.get(i).getField(2))
                    .as("cluster %s, dimension %s", expected.get(i).getField(0), expected.get(i).getField(1))
                    .isCloseTo((double) expected.get(i).getField(2), within(1e-9));
        }
    }

    /**
     * The initialisation {@code docs/ivf.md} gives. Numbering the rows before sampling them would
     * leave gaps in the ids, and a gap shifts every position {@code nearest_vector} returns after
     * it.
     */
    @Test
    public void testDocumentedInitialisationNumbersTheClustersDensely()
    {
        assertUpdate(
                """
                CREATE TABLE random_centroids AS
                SELECT CAST(row_number() OVER () AS integer) AS cluster_id, embedding AS centroid
                FROM (SELECT embedding FROM sample ORDER BY rand() LIMIT %s)
                """.formatted(CLUSTERS),
                CLUSTERS);

        assertQuery(
                "SELECT count(DISTINCT cluster_id), min(cluster_id), max(cluster_id) FROM random_centroids",
                "SELECT %s, 1, %s".formatted(CLUSTERS, CLUSTERS));
    }

    @Test
    public void testEveryClusterIdIsKept()
    {
        assertQuery(
                "SELECT count(*), min(cluster_id), max(cluster_id) FROM centroids_" + ITERATIONS,
                "SELECT %s, 1, %s".formatted(CLUSTERS, CLUSTERS));
    }

    /**
     * Probing every cluster reads every row, so it has to return exactly what the unrestricted
     * search returns.
     */
    @Test
    public void testProbingEveryClusterIsExact()
    {
        for (int queryId : QUERY_IDS) {
            assertThat(probe(queryId, CLUSTERS)).as("query %s", queryId).isEqualTo(exact(queryId));
        }
    }

    /**
     * Each probe reads a superset of the clusters the previous one read, and a true neighbour
     * that is read always makes the top k of what was read, so recall can only go up with the
     * number of clusters probed.
     */
    @Test
    public void testRecallNeverDropsAsMoreClustersAreProbed()
    {
        for (int queryId : QUERY_IDS) {
            Set<Long> exact = exact(queryId);
            int previousHits = -1;
            for (int probes = 1; probes <= CLUSTERS; probes *= 2) {
                Set<Long> found = new HashSet<>(probe(queryId, probes));
                found.retainAll(exact);
                assertThat(found.size()).as("query %s, %s probes", queryId, probes).isGreaterThanOrEqualTo(previousHits);
                previousHits = found.size();
            }
        }
    }

    private static String queryVector(int queryId)
    {
        return "(SELECT transform(embedding, x -> x + 0.5) FROM sample WHERE id = %s)".formatted(queryId);
    }

    private Set<Long> exact(int queryId)
    {
        return keys(
                """
                SELECT n.id
                FROM (SELECT knn_agg(id, embedding, %s, %s, 'euclidean') AS ns FROM embeddings)
                CROSS JOIN UNNEST(ns) AS n(id, distance)
                """.formatted(queryVector(queryId), K));
    }

    private Set<Long> probe(int queryId, int probes)
    {
        return keys(
                """
                WITH probed AS (
                    SELECT p.cluster_id
                    FROM (SELECT knn_agg(cluster_id, centroid, %1$s, %2$s, 'euclidean') AS ps FROM centroids_%3$s)
                    CROSS JOIN UNNEST(ps) AS p(cluster_id, distance)
                )
                SELECT n.id
                FROM (
                    SELECT knn_agg(id, embedding, %1$s, %4$s, 'euclidean') AS ns
                    FROM embeddings
                    WHERE cluster_id IN (SELECT cluster_id FROM probed)
                )
                CROSS JOIN UNNEST(ns) AS n(id, distance)
                """.formatted(queryVector(queryId), probes, ITERATIONS, K));
    }

    private Set<Long> keys(String sql)
    {
        Set<Long> keys = new HashSet<>();
        for (MaterializedRow row : computeActual(sql).getMaterializedRows()) {
            keys.add((Long) row.getField(0));
        }
        return keys;
    }
}
