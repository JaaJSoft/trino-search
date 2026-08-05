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
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestKnnAggregation
        extends AbstractTestQueryFramework
{
    private static final String POINTS =
            """
            (VALUES
                ('a', ARRAY[0.0, 0.0]),
                ('b', ARRAY[1.0, 0.0]),
                ('c', ARRAY[2.0, 0.0]),
                ('d', ARRAY[3.0, 0.0])) AS t(id, v)
            """;

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testNearestNeighboursInOrder()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[1]) FROM " + POINTS,
                "SELECT ARRAY['a', 'b']");
    }

    @Test
    public void testDistancesAreTheRawMetricValues()
    {
        // The expected side must CAST each element to DOUBLE: MaterializedResult.toTestTypes()
        // only normalizes top-level row fields, not values nested inside an array, so an
        // uncast decimal array literal like ARRAY[0.0, 1.0] keeps its elements as SqlDecimal
        // and never compares equal to the actual DOUBLE values, even though both print "0.0"
        // and "1.0" (see the same workaround in TestVectorFunctionQueries.testNormalizeVectorValues).
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[2]) FROM " + POINTS,
                "SELECT ARRAY[CAST(0.0 AS DOUBLE), CAST(1.0 AS DOUBLE)]");
    }

    @Test
    public void testDotProductRanksHigherAsCloser()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[1.0, 0.0], 2, 'dot_product'), x -> x[1]) FROM " + POINTS,
                "SELECT ARRAY['d', 'c']");
    }

    @Test
    public void testGroupsAreIndependent()
    {
        assertQuery(
                """
                SELECT g, transform(knn_agg(id, v, ARRAY[0.0, 0.0], 1, 'euclidean'), x -> x[1])
                FROM (VALUES
                    (1, 'a', ARRAY[5.0, 0.0]),
                    (1, 'b', ARRAY[1.0, 0.0]),
                    (2, 'c', ARRAY[9.0, 0.0]),
                    (2, 'd', ARRAY[7.0, 0.0])) AS t(g, id, v)
                GROUP BY g ORDER BY g
                """,
                "VALUES (1, ARRAY['b']), (2, ARRAY['d'])");
    }

    @Test
    public void testKLargerThanTheGroupReturnsEverything()
    {
        assertQuery(
                "SELECT cardinality(knn_agg(id, v, ARRAY[0.0, 0.0], 100, 'euclidean')) FROM " + POINTS,
                "SELECT 4");
    }

    @Test
    public void testZeroKIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 0, 'euclidean') FROM " + POINTS,
                ".*k must be greater than zero.*");
    }

    @Test
    public void testUnknownMetricIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'hamming') FROM " + POINTS,
                ".*Unknown metric 'hamming'.*");
    }

    @Test
    public void testEmptyGroupGivesNull()
    {
        assertQuery(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean') IS NULL FROM " + POINTS + " WHERE id = 'zzz'",
                "SELECT true");
    }

    @Test
    public void testNullVectorRowsAreIgnored()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[1])
                FROM (VALUES
                    ('a', ARRAY[9.0, 0.0]),
                    ('b', CAST(NULL AS array(double))),
                    ('c', ARRAY[1.0, 0.0])) AS t(id, v)
                """,
                "SELECT ARRAY['c', 'a']");
    }

    @Test
    public void testNullKeysAreKept()
    {
        assertQuery(
                """
                SELECT cardinality(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'))
                FROM (VALUES
                    (CAST(NULL AS varchar), ARRAY[1.0, 0.0]),
                    ('b', ARRAY[2.0, 0.0])) AS t(id, v)
                """,
                "SELECT 2");
    }

    @Test
    public void testBigintKeys()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 1, 'euclidean'), x -> x[1])
                FROM (VALUES
                    (10, ARRAY[5.0, 0.0]),
                    (20, ARRAY[1.0, 0.0])) AS t(id, v)
                """,
                "SELECT ARRAY[20]");
    }

    @Test
    public void testRealOverload()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[REAL '0.0', REAL '0.0'], 1, 'euclidean'), x -> x[1])
                FROM (VALUES
                    ('a', ARRAY[REAL '5.0', REAL '0.0']),
                    ('b', ARRAY[REAL '1.0', REAL '0.0'])) AS t(id, v)
                """,
                "SELECT ARRAY['b']");
    }
}
