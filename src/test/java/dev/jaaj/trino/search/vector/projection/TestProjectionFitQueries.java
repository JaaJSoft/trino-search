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

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/**
 * The principal component fit the documentation describes, run as SQL against a real engine:
 * subspace iteration, where each step multiplies the directions by the sample covariance through
 * {@code vector_projections} and {@code vector_avg_agg}, then orthonormalises them. The sample is
 * built around four known orthogonal axes with variances in the ratio 64 : 16 : 4 : 1, offset from
 * the origin, so the right answer is known without an eigensolver.
 */
@TestInstance(PER_CLASS)
public class TestProjectionFitQueries
        extends AbstractTestQueryFramework
{
    private static final int ROWS = 4000;
    private static final int DIMENSION = 16;
    private static final int ITERATIONS = 12;
    private static final double OFFSET = 3;
    private static final double NOISE = 0.2;
    private static final double[] SCALES = {8, 4, 2, 1};
    private static final double[][] AXES = {
            axis(Map.of(1, 0.6, 2, 0.8)),
            axis(Map.of(1, -0.8, 2, 0.6)),
            axis(Map.of(9, 0.6, 16, 0.8)),
            axis(Map.of(9, 0.8, 16, -0.6)),
    };

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

    @BeforeAll
    public void fit()
    {
        StringBuilder components = new StringBuilder("%s".formatted(OFFSET));
        for (int j = 0; j < AXES.length; j++) {
            components.append(" + %s * %s * %s[d]".formatted(SCALES[j], hash("id", String.valueOf(j + 1)), arrayLiteral(AXES[j])));
        }
        components.append(" + %s * %s".formatted(NOISE, hash("id", "100 + d")));
        assertUpdate(
                """
                CREATE TABLE sample AS
                SELECT id, transform(sequence(1, %s), d -> CAST(%s AS real)) AS embedding
                FROM UNNEST(sequence(1, %s)) AS t(id)
                """.formatted(DIMENSION, components, ROWS),
                ROWS);

        assertUpdate("CREATE TABLE pca_mean AS SELECT vector_avg_agg(CAST(embedding AS array(double))) AS mu FROM sample", 1);
        // Leaving the rows uncentred is the same recipe against a mean of zero.
        assertUpdate("CREATE TABLE zero_mean AS SELECT repeat(DOUBLE '0', %s) AS mu".formatted(DIMENSION), 1);

        // Any rows will do as a start, as long as they are not all in a plane the axes are orthogonal to.
        assertUpdate(
                """
                CREATE TABLE pca_directions_0 AS
                SELECT CAST(row_number() OVER (ORDER BY id) AS integer) AS idx, CAST(embedding AS array(double)) AS direction
                FROM (SELECT id, embedding FROM sample ORDER BY id LIMIT 4)
                """,
                4);
        assertUpdate("CREATE TABLE uncentred_0 AS SELECT * FROM pca_directions_0", 4);

        for (int i = 0; i < ITERATIONS; i++) {
            iterate("pca_mean", "pca_directions_" + i, "pca_directions_" + (i + 1));
            iterate("zero_mean", "uncentred_" + i, "uncentred_" + (i + 1));
        }
    }

    @Test
    public void testFittedDirectionsAreOrthonormal()
    {
        List<double[]> directions = directions("pca_directions_" + ITERATIONS);
        for (int i = 0; i < directions.size(); i++) {
            for (int j = 0; j < directions.size(); j++) {
                assertThat(dot(directions.get(i), directions.get(j))).isCloseTo(i == j ? 1 : 0, within(1e-9));
            }
        }
    }

    /**
     * Each direction converges onto its own axis, in decreasing order of variance. The sign of an
     * eigenvector is arbitrary, so only the alignment is checked.
     */
    @Test
    public void testFittedDirectionsAreThePrincipalAxesInOrder()
    {
        List<double[]> directions = directions("pca_directions_" + ITERATIONS);
        for (int j = 0; j < AXES.length; j++) {
            assertThat(Math.abs(dot(directions.get(j), AXES[j])))
                    .describedAs("direction %s", j + 1)
                    .isGreaterThan(0.99);
        }
    }

    @Test
    public void testExplainedVarianceRatioMatchesTheConstruction()
    {
        List<MaterializedRow> rows = computeActual(
                """
                WITH basis AS (SELECT array_agg(direction ORDER BY idx) AS directions FROM pca_directions_%s),
                centred AS (
                    SELECT zip_with(CAST(s.embedding AS array(double)), m.mu, (x, mu) -> x - mu) AS y
                    FROM sample s CROSS JOIN pca_mean m
                )
                SELECT j, avg(p[j] * p[j]) / avg(dot_product(y, y)) AS explained_variance_ratio
                FROM (SELECT c.y, vector_projections(c.y, b.directions) AS p FROM centred c CROSS JOIN basis b)
                CROSS JOIN UNNEST(sequence(1, cardinality(p))) AS t(j)
                GROUP BY j
                ORDER BY j
                """.formatted(ITERATIONS))
                .getMaterializedRows();

        // A uniform variable on [-1/2, 1/2) has variance 1/12, which cancels in the ratio.
        double total = DIMENSION * NOISE * NOISE;
        for (double scale : SCALES) {
            total += scale * scale;
        }
        assertThat(rows).hasSize(SCALES.length);
        for (int j = 0; j < SCALES.length; j++) {
            double expected = SCALES[j] * SCALES[j] / total;
            assertThat((double) rows.get(j).getField(1))
                    .describedAs("direction %s", j + 1)
                    .isCloseTo(expected, within(expected * 0.05));
        }
    }

    /**
     * Without centring, the second moment the iteration multiplies by is dominated by the mean,
     * and the first direction converges onto the mean rather than onto the axis of largest
     * variance.
     */
    @Test
    public void testUncentredFitPointsTheFirstDirectionAlongTheMean()
    {
        double[] meanDirection = new double[DIMENSION];
        Arrays.fill(meanDirection, 1 / Math.sqrt(DIMENSION));

        double[] first = directions("uncentred_" + ITERATIONS).getFirst();
        assertThat(Math.abs(dot(first, meanDirection))).isGreaterThan(0.95);
        assertThat(Math.abs(dot(first, AXES[0]))).isLessThan(0.5);
    }

    /**
     * One step of subspace iteration, as the documentation gives it.
     */
    private void iterate(String mean, String from, String to)
    {
        assertUpdate(
                """
                CREATE TABLE %3$s AS
                WITH basis AS (SELECT array_agg(direction ORDER BY idx) AS directions FROM %2$s),
                centred AS (
                    SELECT zip_with(CAST(s.embedding AS array(double)), m.mu, (x, mu) -> x - mu) AS y
                    FROM sample s CROSS JOIN %1$s m
                ),
                projected AS (
                    SELECT c.y, vector_projections(c.y, b.directions) AS p
                    FROM centred c CROSS JOIN basis b
                ),
                products AS (
                    SELECT array_agg(w ORDER BY j) AS ws
                    FROM (SELECT j, vector_avg_agg(transform(y, x -> x * p[j])) AS w
                          FROM projected CROSS JOIN UNNEST(sequence(1, cardinality(p))) AS t(j)
                          GROUP BY j)
                )
                SELECT CAST(idx AS integer) AS idx, direction
                FROM products
                CROSS JOIN UNNEST(reduce(
                    ws,
                    CAST(ARRAY[] AS array(array(double))),
                    (done, w) -> done || normalize_vector(
                        reduce(done, w, (r, v) -> zip_with(r, v, (x, e) -> x - dot_product(r, v) * e), r -> r)),
                    done -> done)) WITH ORDINALITY AS u(direction, idx)
                """.formatted(mean, from, to),
                4);
    }

    private List<double[]> directions(String table)
    {
        return computeActual("SELECT direction FROM %s ORDER BY idx".formatted(table)).getMaterializedRows().stream()
                .map(row -> {
                    @SuppressWarnings("unchecked")
                    List<Double> components = (List<Double>) row.getField(0);
                    return components.stream().mapToDouble(Double::doubleValue).toArray();
                })
                .toList();
    }

    /**
     * Uniform on [-0.5, 0.5), a deterministic function of the two SQL expressions.
     */
    private static String hash(String first, String second)
    {
        String scaled = "43758.5453 * sin(%s * 12.9898 + (%s) * 78.233)".formatted(first, second);
        return "(%1$s - floor(%1$s) - 0.5)".formatted(scaled);
    }

    private static double[] axis(Map<Integer, Double> components)
    {
        double[] axis = new double[DIMENSION];
        components.forEach((position, value) -> axis[position - 1] = value);
        return axis;
    }

    private static String arrayLiteral(double[] vector)
    {
        StringBuilder literal = new StringBuilder("ARRAY[");
        for (int i = 0; i < vector.length; i++) {
            literal.append(i == 0 ? "" : ", ").append("DOUBLE '").append(vector[i]).append("'");
        }
        return literal.append("]").toString();
    }

    private static double dot(double[] first, double[] second)
    {
        double sum = 0;
        for (int i = 0; i < first.length; i++) {
            sum += first[i] * second[i];
        }
        return sum;
    }
}
