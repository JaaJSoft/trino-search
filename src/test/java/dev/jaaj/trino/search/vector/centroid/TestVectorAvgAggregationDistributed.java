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

public class TestVectorAvgAggregationDistributed
        extends AbstractTestQueryFramework
{
    /**
     * Every component is an integer, and every partial sum of them stays far below 2^53, so each
     * addition is exact in {@code double} whatever order the splits are combined in. That is what
     * lets the tests below demand equality rather than a tolerance: with arbitrary fractions,
     * floating-point addition is not associative and a different split layout legitimately moves
     * the last bits of the mean.
     * <p>
     * The two dimensions are derived from orderkey differently so that a serializer or a combine
     * that drops or swaps a dimension is caught.
     */
    private static final String VECTORS =
            "(SELECT orderstatus AS g, ARRAY[CAST(orderkey AS double), CAST(-3 * custkey AS double)] AS v FROM tpch.tiny.orders)";

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
     * {@code VectorSumStateSerializer} and {@code @CombineFunction}.
     */
    @Test
    public void testPlanHasPartialAndFinalAggregation()
    {
        MaterializedResult plan = computeActual(
                "EXPLAIN (TYPE DISTRIBUTED) SELECT vector_avg_agg(v) FROM " + VECTORS);
        String text = (String) plan.getOnlyValue();
        assertThat(text).contains("PARTIAL");
        assertThat(text).contains("FINAL");
    }

    /**
     * Checked against {@code sum} and {@code count} computed independently, not against
     * {@code avg}: the built-in divides the same exact sum by the same count, but saying so
     * directly leaves nothing for the reader to trust.
     */
    @Test
    public void testAveragesAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                "SELECT m[1], m[2] FROM (SELECT vector_avg_agg(v) AS m FROM %s)".formatted(VECTORS));
        MaterializedResult expected = computeActual(
                """
                SELECT CAST(sum(orderkey) AS double) / count(*), CAST(sum(-3 * custkey) AS double) / count(*)
                FROM tpch.tiny.orders
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }

    @Test
    public void testAveragesPerGroupAcrossSplits()
    {
        MaterializedResult actual = computeActual(
                "SELECT g, m[1], m[2] FROM (SELECT g, vector_avg_agg(v) AS m FROM %s GROUP BY g)".formatted(VECTORS));
        MaterializedResult expected = computeActual(
                """
                SELECT orderstatus, CAST(sum(orderkey) AS double) / count(*), CAST(sum(-3 * custkey) AS double) / count(*)
                FROM tpch.tiny.orders
                GROUP BY orderstatus
                """);
        assertEqualsIgnoreOrder(actual, expected);
    }

    /**
     * The same rows gathered into a single split first, through an array that one row carries,
     * must give the identical mean: with exact partial sums, the split layout is the only thing
     * that differs, and it must not change a single bit.
     */
    @Test
    public void testOneSplitAndManySplitsAgreeExactly()
    {
        MaterializedResult manySplits = computeActual(
                "SELECT vector_avg_agg(v) FROM " + VECTORS);
        MaterializedResult oneSplit = computeActual(
                """
                SELECT vector_avg_agg(v)
                FROM (SELECT array_agg(v) AS vs FROM %s)
                CROSS JOIN UNNEST(vs) AS t(v)
                """.formatted(VECTORS));
        assertThat(manySplits.getOnlyValue()).isEqualTo(oneSplit.getOnlyValue());
    }
}
