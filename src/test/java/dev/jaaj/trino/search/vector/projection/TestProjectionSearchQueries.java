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
package dev.jaaj.trino.search.vector.projection;

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

import java.util.List;
import java.util.Map;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/**
 * The exact search the documentation describes, run as SQL against a real engine: estimate a
 * radius from the projection columns alone, run {@code knn_agg} under a box predicate on those
 * columns, and accept the result only when its k-th distance is within the radius. What is pinned
 * here is that an accepted result is always the true top k, whatever the radius guess was worth,
 * and that a result the check rejects really can be wrong, so the check is doing work.
 */
@TestInstance(PER_CLASS)
public class TestProjectionSearchQueries
        extends AbstractTestQueryFramework
{
    private static final int ROWS = 2000;
    private static final int DIMENSION = 16;
    private static final int K = 10;
    private static final List<Integer> QUERY_SEEDS = List.of(2001, 2500, 3333, 4096, 7777);

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
     * Four orthonormal directions, rotated away from the axes so that no projection is a raw
     * component, and a corpus written through the documented insert.
     */
    @BeforeAll
    public void writeTable()
    {
        assertUpdate(
                """
                CREATE TABLE projections AS
                SELECT idx, CAST(direction AS array(real)) AS direction
                FROM (VALUES
                    (1, ARRAY[0.6, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    (2, ARRAY[-0.8, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    (3, ARRAY[0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0.8]),
                    (4, ARRAY[0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, -0.6])) AS t(idx, direction)
                """,
                4);
        assertUpdate(
                """
                CREATE TABLE staging AS
                SELECT id, %s AS embedding
                FROM UNNEST(sequence(1, %s)) AS t(id)
                """.formatted(embedding("id"), ROWS),
                ROWS);
        assertUpdate(
                """
                CREATE TABLE embeddings AS
                SELECT id, embedding, p[1] AS pc_1, p[2] AS pc_2, p[3] AS pc_3, p[4] AS pc_4
                FROM (SELECT id, embedding,
                             vector_projections(embedding,
                                 (SELECT array_agg(direction ORDER BY idx) FROM projections)) AS p
                      FROM staging)
                """,
                ROWS);
    }

    /**
     * The check is exact in both directions: a radius at or beyond the true k-th distance puts the
     * whole sphere of true neighbours inside the box, so the result is the true top k and passes,
     * while a radius short of it leaves a k-th distance the radius cannot cover, so it fails.
     */
    @Test
    public void testAcceptedExactlyWhenTheRadiusReachesTheKthDistance()
    {
        for (int seed : QUERY_SEEDS) {
            String query = embedding(String.valueOf(seed));
            List<Double> p = projections(query);
            List<MaterializedRow> truth = unrestricted(query);
            double kth = (double) truth.getLast().getField(1);

            for (double factor : new double[] {0.5, 0.9, 0.999, 1.001, 1.5}) {
                double rho = kth * factor;
                List<MaterializedRow> found = boxed(query, p, rho);
                assertThat(accepted(found, rho))
                        .describedAs("seed %s, radius %s times the k-th distance", seed, factor)
                        .isEqualTo(factor > 1);
                if (factor > 1) {
                    assertThat(found).isEqualTo(truth);
                }
            }
        }
    }

    @Test
    public void testWideningFromTheRadiusGuessEndsOnTheTrueTopK()
    {
        for (int seed : QUERY_SEEDS) {
            String query = embedding(String.valueOf(seed));
            List<Double> p = projections(query);
            double rho = radiusGuess(p);

            List<MaterializedRow> found = boxed(query, p, rho);
            while (!accepted(found, rho)) {
                rho *= 2;
                found = boxed(query, p, rho);
            }
            assertThat(found).isEqualTo(unrestricted(query));
        }
    }

    /**
     * The corpus is noise with no structure for the projections to capture, which is the worst
     * case for the radius guess: the box holds well over k rows, so k come back, and none of the
     * query's true neighbours need be among them. Counting rows cannot tell this apart from a
     * correct result; only the distance check can.
     */
    @Test
    public void testABoxTooNarrowReturnsKWrongRowsThatTheCheckRejects()
    {
        for (int seed : QUERY_SEEDS) {
            String query = embedding(String.valueOf(seed));
            List<Double> p = projections(query);
            double rho = radiusGuess(p);

            List<MaterializedRow> found = boxed(query, p, rho);
            assertThat(found).hasSize(K);
            assertThat(found).isNotEqualTo(unrestricted(query));
            assertThat(accepted(found, rho)).isFalse();
        }
    }

    /**
     * A vector of the table's dimension whose components are a deterministic hash of the seed and
     * the dimension, uniform in [-0.5, 0.5).
     */
    private static String embedding(String seed)
    {
        return "transform(sequence(1, %s), d -> CAST(43758.5453 * sin(%2$s * 12.9898 + d * 78.233) - floor(43758.5453 * sin(%2$s * 12.9898 + d * 78.233)) - 0.5 AS real))"
                .formatted(DIMENSION, seed);
    }

    private List<Double> projections(String query)
    {
        @SuppressWarnings("unchecked")
        List<Double> p = (List<Double>) computeScalar(
                "SELECT vector_projections(%s, (SELECT array_agg(direction ORDER BY idx) FROM projections))".formatted(query));
        return p;
    }

    /**
     * Step 1: the (10 k)-th smallest distance between projections, which reads the four
     * scalar columns and no vector.
     */
    private double radiusGuess(List<Double> p)
    {
        return (double) computeScalar(
                """
                SELECT sqrt(pow(pc_1 - %s, 2) + pow(pc_2 - %s, 2) + pow(pc_3 - %s, 2) + pow(pc_4 - %s, 2)) AS rho
                FROM embeddings
                ORDER BY rho
                OFFSET %s LIMIT 1
                """.formatted(p.get(0), p.get(1), p.get(2), p.get(3), 10 * K - 1));
    }

    /**
     * Step 2: the top k among the rows inside the box, nearest first.
     */
    private List<MaterializedRow> boxed(String query, List<Double> p, double rho)
    {
        return computeActual(
                """
                SELECT n.id, n.distance
                FROM (SELECT knn_agg(id, embedding, %s, %s, 'euclidean') AS top
                      FROM embeddings
                      WHERE pc_1 BETWEEN %3$s - %7$s AND %3$s + %7$s
                        AND pc_2 BETWEEN %4$s - %7$s AND %4$s + %7$s
                        AND pc_3 BETWEEN %5$s - %7$s AND %5$s + %7$s
                        AND pc_4 BETWEEN %6$s - %7$s AND %6$s + %7$s) t
                CROSS JOIN UNNEST(t.top) WITH ORDINALITY AS n(id, distance, rank)
                ORDER BY n.rank
                """.formatted(query, K, p.get(0), p.get(1), p.get(2), p.get(3), rho))
                .getMaterializedRows();
    }

    /**
     * Step 3: a result is the true top k only when it holds k rows and the k-th is within the
     * radius. Fewer than k rows means the box held fewer than k candidates, which proves nothing.
     */
    private static boolean accepted(List<MaterializedRow> found, double rho)
    {
        return found.size() == K && (double) found.getLast().getField(1) <= rho;
    }

    private List<MaterializedRow> unrestricted(String query)
    {
        return computeActual(
                """
                SELECT n.id, n.distance
                FROM (SELECT knn_agg(id, embedding, %s, %s, 'euclidean') AS top FROM embeddings) t
                CROSS JOIN UNNEST(t.top) WITH ORDINALITY AS n(id, distance, rank)
                ORDER BY n.rank
                """.formatted(query, K))
                .getMaterializedRows();
    }
}
