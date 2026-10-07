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
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;
import java.util.Map;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/**
 * Fitting the directions as the documentation describes, against a real engine: one
 * {@code vector_pca_agg} over a sample, kept as a table, then unpacked into the
 * {@code projections} table the write path reads. The sample is built around four known
 * orthogonal axes with variances in the ratio 64 : 16 : 4 : 1, offset from the origin so that a
 * fit which forgot to centre would point its first direction at the mean instead.
 */
@TestInstance(PER_CLASS)
public class TestProjectionFitQueries
        extends AbstractTestQueryFramework
{
    private static final int ROWS = 4000;
    private static final int DIMENSION = 16;
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

        assertUpdate("CREATE TABLE pca AS SELECT vector_pca_agg(embedding, 4) AS fit FROM sample", 1);
        assertUpdate(
                """
                CREATE TABLE projections AS
                SELECT CAST(idx AS integer) AS idx, direction
                FROM pca CROSS JOIN UNNEST(fit.directions) WITH ORDINALITY AS t(direction, idx)
                """,
                4);
    }

    @Test
    public void testFittedDirectionsAreOrthonormal()
    {
        List<double[]> directions = directions("projections");
        for (int i = 0; i < directions.size(); i++) {
            for (int j = 0; j < directions.size(); j++) {
                assertThat(dot(directions.get(i), directions.get(j))).isCloseTo(i == j ? 1 : 0, within(1e-6));
            }
        }
    }

    /**
     * Each direction lands on its own axis, in decreasing order of variance. Only the alignment is
     * checked: the axes were written down without regard to the sign convention of the fit.
     */
    @Test
    public void testFittedDirectionsAreThePrincipalAxesInOrder()
    {
        List<double[]> directions = directions("projections");
        for (int j = 0; j < AXES.length; j++) {
            assertThat(Math.abs(dot(directions.get(j), AXES[j])))
                    .describedAs("direction %s", j + 1)
                    .isGreaterThan(0.99);
        }
    }

    @Test
    public void testExplainedVarianceRatioMatchesTheConstruction()
    {
        @SuppressWarnings("unchecked")
        List<Double> ratios = (List<Double>) computeScalar("SELECT fit.explained_variance_ratio FROM pca");

        // A uniform variable on [-1/2, 1/2) has variance 1/12, which cancels in the ratio.
        double total = DIMENSION * NOISE * NOISE;
        for (double scale : SCALES) {
            total += scale * scale;
        }
        assertThat(ratios).hasSize(SCALES.length);
        for (int j = 0; j < SCALES.length; j++) {
            double expected = SCALES[j] * SCALES[j] / total;
            assertThat(ratios.get(j))
                    .describedAs("direction %s", j + 1)
                    .isCloseTo(expected, within(expected * 0.05));
        }
    }

    /**
     * The table the write path gathers with {@code array_agg(direction ORDER BY idx)}, with the
     * element type of the sample's vectors so that {@code vector_projections} binds without a cast.
     */
    @Test
    public void testProjectionsTableHasTheWritePathTypes()
    {
        assertThat(computeScalar("SELECT typeof(array_agg(direction ORDER BY idx)) FROM projections"))
                .isEqualTo("array(array(real))");
        assertThat(computeScalar("SELECT typeof(idx) FROM projections LIMIT 1")).isEqualTo("integer");
    }

    private List<double[]> directions(String table)
    {
        return computeActual("SELECT direction FROM %s ORDER BY idx".formatted(table)).getMaterializedRows().stream()
                .map(row -> {
                    @SuppressWarnings("unchecked")
                    List<Float> components = (List<Float>) row.getField(0);
                    return components.stream().mapToDouble(Float::doubleValue).toArray();
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
