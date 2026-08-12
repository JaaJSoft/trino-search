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

public class TestKnnAggregationQuantized
        extends AbstractTestQueryFramework
{
    private static final String UNIT_BOUNDS =
            "CAST(ROW(ARRAY[CAST(0.0 AS DOUBLE)], CAST(1.0 AS DOUBLE)) "
                    + "AS row(offsets array(double), scale double))";
    private static final String TWO_DIMENSION_BOUNDS =
            "CAST(ROW(ARRAY[CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE)], CAST(1.0 AS DOUBLE)) "
                    + "AS row(offsets array(double), scale double))";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testQuantizedKnnReturnsTheNearestKeysNearestFirst()
    {
        assertQuery(
                """
                SELECT n[1][1], n[2][1]
                FROM (
                    SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT)], %s, 2, 'euclidean') AS n
                    FROM (VALUES
                        (1, ARRAY[CAST(10 AS TINYINT)]),
                        (2, ARRAY[CAST(1 AS TINYINT)]),
                        (3, ARRAY[CAST(5 AS TINYINT)])) AS t(id, v)
                )
                """.formatted(UNIT_BOUNDS),
                "SELECT 2, 3");
    }

    /**
     * A single partition never calls the combine function, so only a plan with several splits
     * would catch a combine that drops candidates.
     */
    @Test
    public void testQuantizedKnnAcrossMultipleSplits()
    {
        assertQuery(
                """
                SELECT cardinality(n), n[1][1]
                FROM (
                    SELECT knn_agg(n, CAST(ARRAY[CAST(n - 128 AS TINYINT)] AS array(tinyint)),
                                   ARRAY[CAST(-127 AS TINYINT)], %s, 3, 'euclidean') AS n
                    FROM UNNEST(sequence(1, 200)) AS t(n)
                )
                """.formatted(UNIT_BOUNDS),
                "SELECT 3, 1");
    }

    /**
     * Bounds fitted on fewer dimensions than the vectors carry: without the check the fast path is
     * refused, the general kernel runs, and a block accessor throws an internal error on the first
     * position past the end of the bounds.
     */
    @Test
    public void testBoundsShorterThanTheVectorsAreRejected()
    {
        assertQueryFails(
                """
                SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT), CAST(0 AS TINYINT)], %s, 2, 'euclidean')
                FROM (VALUES (1, ARRAY[CAST(3 AS TINYINT), CAST(4 AS TINYINT)])) AS t(id, v)
                """.formatted(UNIT_BOUNDS),
                ".*2 components but the quantisation bounds were fitted on 1.*");
    }

    /**
     * The other direction, and the one that fails silently: without the check the aggregation ranks
     * on the dimensions the vectors do have and returns plausible distances, while the identical
     * scalar expression throws.
     */
    @Test
    public void testBoundsLongerThanTheVectorsAreRejected()
    {
        assertQueryFails(
                """
                SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT)], %s, 2, 'euclidean')
                FROM (VALUES (1, ARRAY[CAST(3 AS TINYINT)])) AS t(id, v)
                """.formatted(TWO_DIMENSION_BOUNDS),
                ".*1 components but the quantisation bounds were fitted on 2.*");
    }

    @Test
    public void testMismatchedVectorLengthsAreRejected()
    {
        assertQueryFails(
                """
                SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT)], %s, 2, 'euclidean')
                FROM (VALUES (1, ARRAY[CAST(3 AS TINYINT), CAST(4 AS TINYINT)])) AS t(id, v)
                """.formatted(UNIT_BOUNDS),
                ".*The arguments must have the same length.*");
    }

    /**
     * Compared with {@code IS NULL} rather than against a null literal: H2 runs the expected side
     * of {@code assertQuery} and does not parse Trino's type constructors, so the comparison is
     * made inside Trino and only a boolean crosses over. {@code TestKnnAggregation} settles the
     * same problem the same way.
     */
    @Test
    public void testEmptyGroupGivesNull()
    {
        assertQuery(
                """
                SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT)], %s, 2, 'euclidean') IS NULL
                FROM (VALUES (1, ARRAY[CAST(1 AS TINYINT)])) AS t(id, v)
                WHERE id = 999
                """.formatted(UNIT_BOUNDS),
                "SELECT true");
    }

    @Test
    public void testNullVectorRowsAreIgnored()
    {
        assertQuery(
                """
                SELECT cardinality(n), n[1][1]
                FROM (
                    SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT)], %s, 2, 'euclidean') AS n
                    FROM (VALUES
                        (1, ARRAY[CAST(9 AS TINYINT)]),
                        (2, CAST(NULL AS array(tinyint))),
                        (3, ARRAY[CAST(1 AS TINYINT)])) AS t(id, v)
                )
                """.formatted(UNIT_BOUNDS),
                "SELECT 2, 3");
    }

    @Test
    public void testNullKeysAreKept()
    {
        assertQuery(
                """
                SELECT cardinality(n)
                FROM (
                    SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT)], %s, 2, 'euclidean') AS n
                    FROM (VALUES
                        (CAST(NULL AS varchar), ARRAY[CAST(1 AS TINYINT)]),
                        ('b', ARRAY[CAST(2 AS TINYINT)])) AS t(id, v)
                )
                """.formatted(UNIT_BOUNDS),
                "SELECT 2");
    }

    /**
     * Distinct from {@link #testNullVectorRowsAreIgnored}, whose {@code CAST(NULL AS
     * array(tinyint))} is a null argument the engine skips before the input function runs. Here the
     * array itself is not null, only one of its codes is, so the input function does run and its
     * own null check is what drops the row.
     */
    @Test
    public void testRowWithNullVectorElementIsIgnored()
    {
        assertQuery(
                """
                SELECT cardinality(n), n[1][1]
                FROM (
                    SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT), CAST(0 AS TINYINT)], %s, 2, 'euclidean') AS n
                    FROM (VALUES
                        (1, ARRAY[CAST(9 AS TINYINT), CAST(0 AS TINYINT)]),
                        (2, ARRAY[CAST(1 AS TINYINT), CAST(NULL AS TINYINT)]),
                        (3, ARRAY[CAST(1 AS TINYINT), CAST(0 AS TINYINT)])) AS t(id, v)
                )
                """.formatted(TWO_DIMENSION_BOUNDS),
                "SELECT 2, 3");
    }

    /**
     * The query vector is the same value on every row, so a null code in it drops every candidate:
     * the heap is created and stays empty, and an empty heap is written as SQL null. The same
     * contract as the float overloads.
     */
    @Test
    public void testNullElementInQueryVectorNullsTheWholeGroup()
    {
        assertQuery(
                """
                SELECT knn_agg(id, v, ARRAY[CAST(0 AS TINYINT), CAST(NULL AS TINYINT)], %s, 2, 'euclidean') IS NULL
                FROM (VALUES
                    (1, ARRAY[CAST(1 AS TINYINT), CAST(0 AS TINYINT)]),
                    (2, ARRAY[CAST(2 AS TINYINT), CAST(0 AS TINYINT)])) AS t(id, v)
                """.formatted(TWO_DIMENSION_BOUNDS),
                "SELECT true");
    }

    @Test
    public void testBinaryKnnReturnsTheNearestKeysNearestFirst()
    {
        assertQuery(
                """
                SELECT n[1][1], n[2][1]
                FROM (
                    SELECT knn_agg(id, v, from_hex('00000004FF'), 2, 'euclidean') AS n
                    FROM (VALUES
                        (1, from_hex('0000000400')),
                        (2, from_hex('000000040F')),
                        (3, from_hex('0000000403'))) AS t(id, v)
                )
                """,
                "SELECT 2, 3");
    }

    /**
     * Every binary metric is monotone in one Hamming count, so all five must return the same keys
     * in the same order. This needs no reference implementation to be a real check.
     */
    @Test
    public void testEveryBinaryMetricRanksIdentically()
    {
        for (String metric : new String[] {"euclidean", "euclidean_squared", "cosine", "dot_product", "manhattan"}) {
            assertQuery(
                    """
                    SELECT n[1][1], n[2][1], n[3][1]
                    FROM (
                        SELECT knn_agg(id, v, from_hex('00000008FF'), 3, '%s') AS n
                        FROM (VALUES
                            (1, from_hex('0000000800')),
                            (2, from_hex('00000008FF')),
                            (3, from_hex('000000080F')),
                            (4, from_hex('0000000801'))) AS t(id, v)
                    )
                    """.formatted(metric),
                    "SELECT 2, 3, 4");
        }
    }

    @Test
    public void testBinaryEmptyGroupGivesNull()
    {
        assertQuery(
                """
                SELECT knn_agg(id, v, from_hex('00000004FF'), 2, 'euclidean') IS NULL
                FROM (VALUES (1, from_hex('0000000400'))) AS t(id, v)
                WHERE id = 999
                """,
                "SELECT true");
    }

    /**
     * The binary overload takes {@code varbinary} operands rather than arrays, so a null vector is
     * a null argument and there is no such thing as a null element inside one: the engine skips a
     * row whose non-nullable argument is null before the input function runs, which is why this
     * class has no binary equivalent of {@code testRowWithNullVectorElementIsIgnored} or of
     * {@code testNullElementInQueryVectorNullsTheWholeGroup}. A null query vector, by the same
     * rule, skips every row and leaves the group empty.
     */
    @Test
    public void testBinaryNullVectorRowsAreIgnored()
    {
        assertQuery(
                """
                SELECT cardinality(n), n[1][1]
                FROM (
                    SELECT knn_agg(id, v, from_hex('00000004FF'), 2, 'euclidean') AS n
                    FROM (VALUES
                        (1, from_hex('0000000400')),
                        (2, CAST(NULL AS varbinary)),
                        (3, from_hex('000000040F'))) AS t(id, v)
                )
                """,
                "SELECT 2, 3");
    }

    @Test
    public void testBinaryNullKeysAreKept()
    {
        assertQuery(
                """
                SELECT cardinality(n)
                FROM (
                    SELECT knn_agg(id, v, from_hex('00000004FF'), 2, 'euclidean') AS n
                    FROM (VALUES
                        (CAST(NULL AS varchar), from_hex('000000040F')),
                        ('b', from_hex('0000000403'))) AS t(id, v)
                )
                """,
                "SELECT 2");
    }

    @Test
    public void testBinaryKnnAcrossMultipleSplits()
    {
        assertQuery(
                """
                SELECT cardinality(n)
                FROM (
                    SELECT knn_agg(n, to_big_endian_32(4) || from_hex('0F'), to_big_endian_32(4) || from_hex('0F'),
                                   3, 'euclidean') AS n
                    FROM UNNEST(sequence(1, 1000)) AS t(n)
                )
                """,
                "SELECT 3");
    }
}
