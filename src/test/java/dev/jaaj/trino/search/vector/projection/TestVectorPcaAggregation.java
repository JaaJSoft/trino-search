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
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedRow;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import java.util.List;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestVectorPcaAggregation
        extends AbstractTestQueryFramework
{
    /**
     * Five points spread mostly along (1, 1, 0), a little along (1, -1, 0), and not at all along
     * the third axis.
     */
    private static final String SPREAD =
            """
            (VALUES
                (ARRAY[DOUBLE '4', DOUBLE '4', DOUBLE '7']),
                (ARRAY[DOUBLE '-4', DOUBLE '-4', DOUBLE '7']),
                (ARRAY[DOUBLE '1', DOUBLE '-1', DOUBLE '7']),
                (ARRAY[DOUBLE '-1', DOUBLE '1', DOUBLE '7']),
                (ARRAY[DOUBLE '0', DOUBLE '0', DOUBLE '7'])) AS t(v)
            """;

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    /**
     * The deviations from the mean (0, 0, 7) project onto (1, 1, 0) / sqrt(2) as two of
     * +/- 8 / sqrt(2), onto (1, -1, 0) / sqrt(2) as two of +/- 2 / sqrt(2), and onto z not at all:
     * squares summing to 64, 4 and 0.
     */
    @Test
    public void testFindsTheDirectionsOfLargestVariance()
    {
        Fit fit = fit("SELECT vector_pca_agg(v, 3) FROM " + SPREAD);

        double half = Math.sqrt(0.5);
        assertThat(fit.directions().get(0)).containsExactly(new double[] {half, half, 0}, within(1e-12));
        assertThat(Math.abs(fit.directions().get(1)[0])).isCloseTo(half, within(1e-12));
        assertThat(fit.directions().get(1)[0]).isCloseTo(-fit.directions().get(1)[1], within(1e-12));
        assertThat(Math.abs(fit.directions().get(2)[2])).isCloseTo(1, within(1e-12));
        assertThat(components(fit.ratios())).containsExactly(new double[] {64.0 / 68, 4.0 / 68, 0}, within(1e-12));
    }

    @Test
    public void testReturnsTheFirstKDirections()
    {
        Fit fit = fit("SELECT vector_pca_agg(v, 1) FROM " + SPREAD);

        assertThat(fit.directions()).hasSize(1);
        assertThat(components(fit.ratios())).containsExactly(new double[] {64.0 / 68}, within(1e-12));
    }

    /**
     * There are only as many principal directions as dimensions, as {@code knn_agg} returns only
     * as many neighbours as a group has rows.
     */
    @Test
    public void testKLargerThanTheDimensionReturnsEveryDirection()
    {
        Fit fit = fit("SELECT vector_pca_agg(v, 10) FROM " + SPREAD);

        assertThat(fit.directions()).hasSize(3);
        assertThat(fit.ratios()).hasSize(3);
    }

    /**
     * An eigenvector is only defined up to its sign. Fixing it, by making the component of
     * largest magnitude positive, keeps a refit on the same data from flipping a projection
     * column.
     */
    @Test
    public void testSignMakesTheLargestComponentPositive()
    {
        Fit fit = fit(
                """
                SELECT vector_pca_agg(v, 1)
                FROM (VALUES (ARRAY[DOUBLE '1', DOUBLE '-3']), (ARRAY[DOUBLE '-1', DOUBLE '3']), (ARRAY[DOUBLE '2', DOUBLE '-6'])) AS t(v)
                """);

        double norm = Math.sqrt(10);
        assertThat(fit.directions().getFirst()).containsExactly(new double[] {-1 / norm, 3 / norm}, within(1e-12));
    }

    @Test
    public void testDirectionsAreOrthonormal()
    {
        Fit fit = fit(
                """
                SELECT vector_pca_agg(v, 5)
                FROM (SELECT transform(sequence(1, 5), d -> sin(id * d) + d * cos(id)) AS v FROM UNNEST(sequence(1, 50)) AS t(id))
                """);

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                assertThat(dot(fit.directions().get(i), fit.directions().get(j))).isCloseTo(i == j ? 1 : 0, within(1e-12));
            }
        }
        assertThat(fit.ratios().stream().mapToDouble(Double::doubleValue).sum()).isCloseTo(1, within(1e-12));
    }

    @Test
    public void testDoubleInputReturnsDoubleDirections()
    {
        assertThat(computeScalar("SELECT typeof(vector_pca_agg(v, 1)) FROM " + SPREAD))
                .isEqualTo("row(\"directions\" array(array(double)), \"explained_variance_ratio\" array(double))");
    }

    @Test
    public void testRealInputReturnsRealDirections()
    {
        assertThat(computeScalar("SELECT typeof(vector_pca_agg(v, 1)) FROM (VALUES (ARRAY[REAL '1', REAL '2'])) AS t(v)"))
                .isEqualTo("row(\"directions\" array(array(real)), \"explained_variance_ratio\" array(double))");
    }

    /**
     * The directions of an {@code array(real)} fit have the type {@code vector_projections}
     * takes alongside the same vectors, with no cast in between.
     */
    @Test
    public void testRealDirectionsFeedVectorProjections()
    {
        assertQuery(
                """
                WITH sample(v) AS (VALUES (ARRAY[REAL '3', REAL '0']), (ARRAY[REAL '-3', REAL '0']), (ARRAY[REAL '0', REAL '1']), (ARRAY[REAL '0', REAL '-1']))
                SELECT abs(p[1] - 2) < 1e-6 AND abs(p[2] - 5) < 1e-6
                FROM (SELECT vector_projections(ARRAY[REAL '2', REAL '5'], (SELECT vector_pca_agg(v, 2).directions FROM sample)) AS p)
                """,
                "SELECT true");
    }

    @Test
    public void testFitsPerGroup()
    {
        List<MaterializedRow> rows = computeActual(
                """
                SELECT g, vector_pca_agg(v, 1).directions[1]
                FROM (VALUES
                    ('x', ARRAY[DOUBLE '1', DOUBLE '0']), ('x', ARRAY[DOUBLE '-1', DOUBLE '0']),
                    ('y', ARRAY[DOUBLE '0', DOUBLE '2']), ('y', ARRAY[DOUBLE '0', DOUBLE '-2'])) AS t(g, v)
                GROUP BY g
                ORDER BY g
                """).getMaterializedRows();

        assertThat(components(rows.get(0).getField(1))).containsExactly(new double[] {1, 0}, within(1e-15));
        assertThat(components(rows.get(1).getField(1))).containsExactly(new double[] {0, 1}, within(1e-15));
    }

    /**
     * A row is skipped whole when its vector is null or holds a null, as {@code vector_avg_agg}
     * skips it, so the fit is the fit of the other rows.
     */
    @Test
    public void testNullVectorsAndNullElementsAreIgnored()
    {
        Fit withNulls = fit(
                """
                SELECT vector_pca_agg(v, 3)
                FROM (SELECT v FROM %s
                      UNION ALL VALUES (CAST(NULL AS array(double))), (ARRAY[DOUBLE '100', NULL, DOUBLE '-100']))
                """.formatted(SPREAD));
        Fit without = fit("SELECT vector_pca_agg(v, 3) FROM " + SPREAD);

        for (int j = 0; j < 3; j++) {
            assertThat(withNulls.directions().get(j)).containsExactly(without.directions().get(j), within(1e-12));
        }
        assertThat(components(withNulls.ratios())).containsExactly(components(without.ratios()), within(1e-12));
    }

    /**
     * Compared with {@code IS NULL} rather than a literal null row, since H2 runs the expected
     * side of {@code assertQuery}.
     */
    @Test
    public void testEmptyInputReturnsNull()
    {
        assertQuery(
                "SELECT vector_pca_agg(v, 2) IS NULL FROM (SELECT CAST(NULL AS array(double)) AS v WHERE false)",
                "SELECT true");
    }

    @Test
    public void testGroupWhereEveryRowIsSkippedReturnsNull()
    {
        assertQuery(
                "SELECT vector_pca_agg(v, 2) IS NULL FROM (VALUES (ARRAY[CAST(NULL AS DOUBLE)]), (CAST(NULL AS array(double)))) AS t(v)",
                "SELECT true");
    }

    /**
     * A single row has no variance: any orthonormal basis is as good as any other, and the share
     * of a total of zero is undefined.
     */
    @Test
    public void testSingleRowHasUndefinedRatios()
    {
        Fit fit = fit("SELECT vector_pca_agg(ARRAY[DOUBLE '3', DOUBLE '4'], 2)");

        assertThat(dot(fit.directions().get(0), fit.directions().get(1))).isCloseTo(0, within(1e-15));
        assertThat(dot(fit.directions().get(0), fit.directions().get(0))).isCloseTo(1, within(1e-15));
        assertThat(fit.ratios()).allMatch(ratio -> Double.isNaN(ratio));
    }

    @Test
    public void testKMustBeGreaterThanZero()
    {
        assertQueryFails("SELECT vector_pca_agg(v, 0) FROM " + SPREAD, "k must be greater than zero, got 0");
    }

    @Test
    public void testKMustBeConstantWithinAGroup()
    {
        assertQueryFails(
                "SELECT vector_pca_agg(v, k) FROM (VALUES (ARRAY[DOUBLE '1'], 1), (ARRAY[DOUBLE '2'], 2)) AS t(v, k)",
                "k must be constant within a group of vector_pca_agg, found 1 and 2");
    }

    @Test
    public void testMixedDimensionsAreRejected()
    {
        assertQueryFails(
                "SELECT vector_pca_agg(v, 1) FROM (VALUES (ARRAY[DOUBLE '1']), (ARRAY[DOUBLE '1', DOUBLE '2'])) AS t(v)",
                "The vectors of vector_pca_agg must have the same length, found 1 and 2");
    }

    @Test
    public void testNonFiniteComponentsAreRejected()
    {
        assertQueryFails(
                "SELECT vector_pca_agg(v, 1) FROM (VALUES (ARRAY[DOUBLE '1', infinity()]), (ARRAY[DOUBLE '2', DOUBLE '0'])) AS t(v)",
                "The covariance of the vectors in vector_pca_agg is not finite");
    }

    private Fit fit(String aggregation)
    {
        MaterializedRow row = computeActual(
                "SELECT p.directions, p.explained_variance_ratio FROM (%s) AS t(p)".formatted(aggregation))
                .getMaterializedRows().getFirst();
        @SuppressWarnings("unchecked")
        List<Object> directions = (List<Object>) row.getField(0);
        @SuppressWarnings("unchecked")
        List<Double> ratios = (List<Double>) row.getField(1);
        return new Fit(directions.stream().map(TestVectorPcaAggregation::components).toList(), ratios);
    }

    private static double dot(double[] first, double[] second)
    {
        double sum = 0;
        for (int i = 0; i < first.length; i++) {
            sum += first[i] * second[i];
        }
        return sum;
    }

    private static double[] components(Object array)
    {
        @SuppressWarnings("unchecked")
        List<Double> components = (List<Double>) array;
        return components.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private record Fit(List<double[]> directions, List<Double> ratios) {}
}
