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
package dev.jaaj.trino.search.vector.quantize;

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

public class TestVectorBoundsAggregationDistributed
        extends AbstractTestQueryFramework
{
    /**
     * Two components, each derived from orderkey differently so the two dimensions have genuinely
     * different minima and maxima, and the fitted offsets and scale can be checked against
     * MIN/MAX(orderkey) computed independently. Using a TPCH table spreads the rows across splits,
     * which is what forces Trino to run the partial and final aggregation steps and therefore to
     * serialize the state, the same way TestKnnAggregationDistributed forces it for knn_agg.
     */
    private static final String VECTORS =
            "(SELECT ARRAY[CAST(orderkey AS double), CAST(2 * orderkey AS double)] AS v FROM tpch.tiny.orders)";

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
     * Confirms this query actually runs a partial aggregation on each split plus a final
     * aggregation that merges them, rather than collapsing to a single-stage aggregation. Only
     * that shape exercises {@code BoundsStateSerializer} and {@code @CombineFunction}.
     */
    @Test
    public void testPlanHasPartialAndFinalAggregation()
    {
        MaterializedResult plan = computeActual(
                "EXPLAIN (TYPE DISTRIBUTED) SELECT vector_bounds_agg(v) FROM " + VECTORS);
        String text = (String) plan.getOnlyValue();
        assertThat(text).contains("PARTIAL");
        assertThat(text).contains("FINAL");
    }

    /**
     * A single partition never calls the combine function, so this is the only shape that would
     * catch a serializer dropping a dimension or a combine that keeps only one side's extremes.
     * Asserting every offset and the scale, rather than only the first dimension, is what makes a
     * dropped second dimension or a one-sided combine actually fail here.
     */
    @Test
    public void testFitsAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                """
                SELECT b.offsets[1], b.offsets[2], b.scale
                FROM (SELECT vector_bounds_agg(v) AS b FROM %s)
                """.formatted(VECTORS));
        MaterializedResult expected = computeActual(
                """
                SELECT (CAST(MIN(orderkey) AS double) + CAST(MAX(orderkey) AS double)) / 2,
                       (CAST(MIN(2 * orderkey) AS double) + CAST(MAX(2 * orderkey) AS double)) / 2,
                       GREATEST(CAST(MAX(orderkey) AS double) - CAST(MIN(orderkey) AS double),
                                CAST(MAX(2 * orderkey) AS double) - CAST(MIN(2 * orderkey) AS double)) / 255.0
                FROM tpch.tiny.orders
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }
}
