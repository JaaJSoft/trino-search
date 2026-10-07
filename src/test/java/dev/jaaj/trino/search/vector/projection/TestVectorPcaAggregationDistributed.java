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
import io.trino.plugin.tpch.TpchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedRow;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestVectorPcaAggregationDistributed
        extends AbstractTestQueryFramework
{
    /**
     * Two dimensions of comparable spread, correlated weakly enough that neither axis is the
     * answer, so a combine that drops a split, swaps a dimension or forgets the correction between
     * two means moves the result.
     */
    private static final String VECTORS =
            "(SELECT orderstatus AS g, totalprice / 100 AS x, CAST(custkey AS double) AS y, ARRAY[totalprice / 100, CAST(custkey AS double)] AS v FROM tpch.tiny.orders)";

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
     * Only a partial aggregation per split followed by a final one exercises
     * {@code PcaStateSerializer} and {@code @CombineFunction}.
     */
    @Test
    public void testPlanHasPartialAndFinalAggregation()
    {
        String plan = (String) computeActual("EXPLAIN (TYPE DISTRIBUTED) SELECT vector_pca_agg(v, 2) FROM " + VECTORS).getOnlyValue();

        assertThat(plan).contains("PARTIAL");
        assertThat(plan).contains("FINAL");
    }

    /**
     * In two dimensions the answer has a closed form in the variances and the covariance, which
     * the engine's own {@code var_pop} and {@code covar_pop} compute independently of anything
     * here.
     */
    @Test
    public void testAgreesWithTheClosedFormAcrossSplits()
    {
        MaterializedRow fit = computeActual(
                "SELECT p.directions, p.explained_variance_ratio FROM (SELECT vector_pca_agg(v, 2) AS p FROM %s)".formatted(VECTORS))
                .getMaterializedRows().getFirst();
        MaterializedRow moments = computeActual(
                "SELECT var_pop(x), var_pop(y), covar_pop(x, y) FROM " + VECTORS)
                .getMaterializedRows().getFirst();

        assertMatchesClosedForm(fit.getField(0), fit.getField(1), moments);
    }

    @Test
    public void testAgreesWithTheClosedFormPerGroupAcrossSplits()
    {
        List<MaterializedRow> fits = computeActual(
                "SELECT g, p.directions, p.explained_variance_ratio FROM (SELECT g, vector_pca_agg(v, 2) AS p FROM %s GROUP BY g) ORDER BY g".formatted(VECTORS))
                .getMaterializedRows();
        List<MaterializedRow> moments = computeActual(
                "SELECT g, var_pop(x), var_pop(y), covar_pop(x, y) FROM %s GROUP BY g ORDER BY g".formatted(VECTORS))
                .getMaterializedRows();

        assertThat(fits).hasSize(3);
        for (int i = 0; i < fits.size(); i++) {
            assertThat(fits.get(i).getField(0)).isEqualTo(moments.get(i).getField(0));
            assertMatchesClosedForm(
                    fits.get(i).getField(1),
                    fits.get(i).getField(2),
                    new MaterializedRow(List.of(moments.get(i).getField(1), moments.get(i).getField(2), moments.get(i).getField(3))));
        }
    }

    /**
     * The same rows gathered into a single split first, through an array one row carries. The
     * split layout changes the order of the floating point additions and so the last bits, but
     * nothing more.
     */
    @Test
    public void testOneSplitAndManySplitsAgree()
    {
        MaterializedRow many = computeActual(
                "SELECT p.directions, p.explained_variance_ratio FROM (SELECT vector_pca_agg(v, 2) AS p FROM %s)".formatted(VECTORS))
                .getMaterializedRows().getFirst();
        MaterializedRow one = computeActual(
                """
                SELECT p.directions, p.explained_variance_ratio
                FROM (SELECT vector_pca_agg(v, 2) AS p FROM (SELECT array_agg(v) AS vs FROM %s) CROSS JOIN UNNEST(vs) AS t(v))
                """.formatted(VECTORS))
                .getMaterializedRows().getFirst();

        List<double[]> manyDirections = directions(many.getField(0));
        List<double[]> oneDirections = directions(one.getField(0));
        for (int j = 0; j < 2; j++) {
            assertThat(manyDirections.get(j)).containsExactly(oneDirections.get(j), within(1e-12));
        }
        assertThat(components(many.getField(1))).containsExactly(components(one.getField(1)), within(1e-12));
    }

    private static void assertMatchesClosedForm(Object directionsField, Object ratiosField, MaterializedRow moments)
    {
        double a = (double) moments.getField(0);
        double b = (double) moments.getField(1);
        double c = (double) moments.getField(2);
        double largest = (a + b) / 2 + Math.hypot((a - b) / 2, c);
        double[] expected = {c, largest - a};
        double norm = Math.hypot(expected[0], expected[1]);

        double[] first = directions(directionsField).getFirst();
        double alignment = Math.abs(first[0] * expected[0] + first[1] * expected[1]) / norm;
        assertThat(alignment).isCloseTo(1, within(1e-9));
        assertThat(components(ratiosField)).containsExactly(new double[] {largest / (a + b), 1 - largest / (a + b)}, within(1e-9));
    }

    private static List<double[]> directions(Object field)
    {
        @SuppressWarnings("unchecked")
        List<Object> directions = (List<Object>) field;
        return directions.stream().map(TestVectorPcaAggregationDistributed::components).toList();
    }

    private static double[] components(Object array)
    {
        @SuppressWarnings("unchecked")
        List<Double> components = (List<Double>) array;
        return components.stream().mapToDouble(Double::doubleValue).toArray();
    }
}
