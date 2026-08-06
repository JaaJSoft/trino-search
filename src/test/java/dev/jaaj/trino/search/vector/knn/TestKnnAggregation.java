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

import java.util.List;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

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

    // 0.1, 0.2 and 0.3 are not exactly representable in float32 (0.1f widens to
    // 0.10000000149011612 as a double), so a query against this table discriminates which
    // overload (array(double) vs array(real)) actually read the vector: a reader swap changes
    // the computed distances measurably, not just their printed form.
    private static final String FRACTIONAL_POINTS =
            """
            (VALUES
                ('a', ARRAY[0.3]),
                ('b', ARRAY[0.2]),
                ('c', ARRAY[0.1])) AS t(id, v)
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
    public void testKAboveTheCapIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 10001, 'euclidean') FROM " + POINTS,
                ".*k of knn_agg must be less than or equal to 10000; found 10001.*");
    }

    @Test
    public void testKAtTheCapIsAccepted()
    {
        assertQuery(
                "SELECT cardinality(knn_agg(id, v, ARRAY[0.0, 0.0], 10000, 'euclidean')) FROM " + POINTS,
                "SELECT 4");
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
        // The brief's original values (5.0, 1.0, 0.0) are exactly representable in float32, so
        // this test used to pass unchanged even if OfRealVectors silently read its vector through
        // DOUBLE_READER: nothing about the result would have differed. 0.1, 0.2 and 0.3 are not
        // exact in float32, so CAST(... AS array(real)) forces exact-match resolution to
        // OfRealVectors AND the resulting distances only match float32 precision, not double
        // precision - this fails loudly if the real overload's reader is swapped for the double
        // one (see the report for the deliberate-break run that proves it).
        assertQuery(
                """
                SELECT transform(
                        knn_agg(id, CAST(v AS array(real)), CAST(ARRAY[0.0] AS array(real)), 3, 'euclidean'),
                        x -> x[1])
                FROM """ + FRACTIONAL_POINTS,
                "SELECT ARRAY['c', 'b', 'a']");

        List<?> distances = (List<?>) computeActual(
                """
                SELECT transform(
                        knn_agg(id, CAST(v AS array(real)), CAST(ARRAY[0.0] AS array(real)), 3, 'euclidean'),
                        x -> x[2])
                FROM """ + FRACTIONAL_POINTS)
                .getOnlyValue();
        assertThat((Double) distances.get(0)).isCloseTo((double) 0.1f, within(1e-10));
        assertThat((Double) distances.get(1)).isCloseTo((double) 0.2f, within(1e-10));
        assertThat((Double) distances.get(2)).isCloseTo((double) 0.3f, within(1e-10));
    }

    @Test
    public void testDoubleOverload()
    {
        // knn_agg is registered as two overloads, array(double) and array(real), but every other
        // test in this class binds to array(real): an untyped decimal array literal such as
        // ARRAY[0.1] coerces to array(real) preferentially (see README and
        // TestVectorRealFunctionQueries), so nothing exercised OfDoubleVectors at all before this
        // test. CAST(... AS array(double)) forces exact-match resolution to OfDoubleVectors, and
        // 0.1/0.2/0.3 are not exact in float32, so comparing the returned distances against the
        // double-precision values (not the float32-widened ones used in testRealOverload) pins
        // that this overload reads through DOUBLE_READER: it fails if OfDoubleVectors used
        // REAL_READER instead (see the report for the deliberate-break run that proves it).
        assertQuery(
                """
                SELECT transform(
                        knn_agg(id, CAST(v AS array(double)), CAST(ARRAY[0.0] AS array(double)), 3, 'euclidean'),
                        x -> x[1])
                FROM """ + FRACTIONAL_POINTS,
                "SELECT ARRAY['c', 'b', 'a']");

        List<?> distances = (List<?>) computeActual(
                """
                SELECT transform(
                        knn_agg(id, CAST(v AS array(double)), CAST(ARRAY[0.0] AS array(double)), 3, 'euclidean'),
                        x -> x[2])
                FROM """ + FRACTIONAL_POINTS)
                .getOnlyValue();
        assertThat((Double) distances.get(0)).isCloseTo(0.1, within(1e-12));
        assertThat((Double) distances.get(1)).isCloseTo(0.2, within(1e-12));
        assertThat((Double) distances.get(2)).isCloseTo(0.3, within(1e-12));
    }

    @Test
    public void testRowWithNullVectorElementIsIgnored()
    {
        // Distinct from testNullVectorRowsAreIgnored, whose CAST(NULL AS array(double)) is a
        // NULL array argument: the engine's non-nullable-argument rule skips the input call
        // entirely, so it never reaches vector.hasNull(). Here the array value itself is not
        // NULL, only one of its elements is, so the input call happens and hasNull() is what
        // causes the row to be skipped.
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[1])
                FROM (VALUES
                    ('a', ARRAY[9.0, 0.0]),
                    ('b', ARRAY[1.0, CAST(NULL AS double)]),
                    ('c', ARRAY[1.0, 0.0])) AS t(id, v)
                """,
                "SELECT ARRAY['c', 'a']");
    }

    @Test
    public void testNullElementInQueryVectorNullsTheWholeGroup()
    {
        // The query vector is the same value for every row in the group, so a NULL element in it
        // makes queryVector.hasNull() true on every call to addCandidate: the heap is created
        // (on the first row) but addCandidate always returns before heap.add(), so it stays
        // empty. writeResult treats an empty heap the same as an absent one and outputs SQL
        // NULL. This differs from a NULL element in a single row's own vector, which only drops
        // that one row (see testRowWithNullVectorElementIsIgnored) - pinned here as the current,
        // deliberate contract for a NULL in the query vector.
        assertQuery(
                "SELECT knn_agg(id, v, ARRAY[0.0, NULL], 2, 'euclidean') IS NULL FROM " + POINTS,
                "SELECT true");
    }

    @Test
    public void testMismatchedVectorLengthsAreRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0], 2, 'euclidean') FROM " + POINTS,
                ".*The arguments must have the same length.*");
    }

    @Test
    public void testVaryingKWithinAGroupIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], CASE WHEN id = 'a' THEN 1 ELSE 2 END, 'euclidean') FROM " + POINTS,
                ".*k must be constant within a group of knn_agg.*");
    }

    @Test
    public void testVaryingMetricWithinAGroupIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 2, CASE WHEN id = 'a' THEN 'dot_product' ELSE 'euclidean' END) FROM " + POINTS,
                ".*metric must be constant within a group of knn_agg.*");
    }
}
