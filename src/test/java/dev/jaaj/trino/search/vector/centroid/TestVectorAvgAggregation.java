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
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

public class TestVectorAvgAggregation
        extends AbstractTestQueryFramework
{
    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testAveragesEachDimensionIndependently()
    {
        assertQuery(
                """
                SELECT m[1], m[2]
                FROM (
                    SELECT vector_avg_agg(v) AS m
                    FROM (VALUES
                        (ARRAY[CAST(1.0 AS DOUBLE), CAST(10.0 AS DOUBLE)]),
                        (ARRAY[CAST(2.0 AS DOUBLE), CAST(20.0 AS DOUBLE)]),
                        (ARRAY[CAST(6.0 AS DOUBLE), CAST(60.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 3.0, 30.0");
    }

    @Test
    public void testAveragesPerGroup()
    {
        assertQuery(
                """
                SELECT g, vector_avg_agg(v)[1]
                FROM (VALUES
                    ('a', ARRAY[CAST(1.0 AS DOUBLE)]),
                    ('a', ARRAY[CAST(3.0 AS DOUBLE)]),
                    ('b', ARRAY[CAST(10.0 AS DOUBLE)])) AS t(g, v)
                GROUP BY g
                """,
                "VALUES ('a', 2.0), ('b', 10.0)");
    }

    @Test
    public void testDoubleInputReturnsDouble()
    {
        assertThat(computeScalar("SELECT typeof(vector_avg_agg(v)) FROM (VALUES (ARRAY[CAST(1.0 AS DOUBLE)])) AS t(v)"))
                .isEqualTo("array(double)");
    }

    @Test
    public void testRealInputReturnsReal()
    {
        assertThat(computeScalar("SELECT typeof(vector_avg_agg(v)) FROM (VALUES (ARRAY[REAL '1.0'])) AS t(v)"))
                .isEqualTo("array(real)");
    }

    @Test
    public void testRealVectorsAreAveraged()
    {
        assertQuery(
                """
                SELECT CAST(m[1] AS double), CAST(m[2] AS double)
                FROM (
                    SELECT vector_avg_agg(v) AS m
                    FROM (VALUES (ARRAY[REAL '1.0', REAL '-4.0']), (ARRAY[REAL '2.0', REAL '4.0'])) AS t(v)
                )
                """,
                "SELECT 1.5, 0.0");
    }

    /**
     * In {@code float}, 2^24 + 1 rounds back to 2^24, so a float accumulator loses both ones and
     * returns 2^24 / 3, which the nearest float puts at 5592405.5. In {@code double} the sum is
     * 2^24 + 2 and the mean is exactly 5592406, which a float represents.
     */
    @Test
    public void testRealVectorsAccumulateInDouble()
    {
        assertQuery(
                """
                SELECT CAST(vector_avg_agg(v)[1] AS double)
                FROM (VALUES (ARRAY[REAL '16777216.0']), (ARRAY[REAL '1.0']), (ARRAY[REAL '1.0'])) AS t(v)
                """,
                "SELECT 5592406.0");
    }

    @Test
    public void testNullVectorsAreIgnored()
    {
        assertQuery(
                """
                SELECT vector_avg_agg(v)[1]
                FROM (VALUES (ARRAY[CAST(1.0 AS DOUBLE)]), (CAST(NULL AS array(double))), (ARRAY[CAST(3.0 AS DOUBLE)])) AS t(v)
                """,
                "SELECT 2.0");
    }

    /**
     * A vector with a null component is skipped whole, as {@code knn_agg} and
     * {@code vector_bounds_agg} skip it. Averaging its other components would divide each
     * dimension by a different count, so the result would no longer be the mean of any set of
     * vectors.
     */
    @Test
    public void testVectorsWithANullElementAreIgnored()
    {
        assertQuery(
                """
                SELECT m[1], m[2]
                FROM (
                    SELECT vector_avg_agg(v) AS m
                    FROM (VALUES
                        (ARRAY[CAST(1.0 AS DOUBLE), CAST(1.0 AS DOUBLE)]),
                        (ARRAY[CAST(100.0 AS DOUBLE), NULL]),
                        (ARRAY[CAST(3.0 AS DOUBLE), CAST(3.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 2.0, 2.0");
    }

    /**
     * Compared with {@code IS NULL} rather than a literal null array, for the reason
     * {@code TestVectorBoundsAggregation} gives: H2 runs the expected side of {@code assertQuery}.
     */
    @Test
    public void testEmptyInputReturnsNull()
    {
        assertQuery(
                "SELECT vector_avg_agg(v) IS NULL FROM (SELECT CAST(NULL AS array(double)) AS v WHERE false)",
                "SELECT true");
    }

    @Test
    public void testGroupWhereEveryRowIsSkippedReturnsNull()
    {
        assertQuery(
                "SELECT vector_avg_agg(v) IS NULL FROM (VALUES (ARRAY[CAST(NULL AS DOUBLE)]), (CAST(NULL AS array(double)))) AS t(v)",
                "SELECT true");
    }

    @Test
    public void testMixedDimensionsAreRejected()
    {
        assertQueryFails(
                """
                SELECT vector_avg_agg(v)
                FROM (VALUES
                    (ARRAY[CAST(1.0 AS DOUBLE)]),
                    (ARRAY[CAST(1.0 AS DOUBLE), CAST(2.0 AS DOUBLE)])) AS t(v)
                """,
                "The vectors of vector_avg_agg must have the same length, found 1 and 2");
    }

    /**
     * A zero-length vector is a vector of dimension zero, not a missing one, so its mean is an
     * empty array rather than null.
     */
    @Test
    public void testEmptyVectorsAverageToAnEmptyVector()
    {
        assertQuery(
                "SELECT cardinality(vector_avg_agg(v)) FROM (VALUES (CAST(ARRAY[] AS array(double)))) AS t(v)",
                "SELECT 0");
    }

    // The multiple-splits case (VectorSumStateSerializer, @CombineFunction) is covered in
    // TestVectorAvgAggregationDistributed, which uses a splittable connector: this class has no
    // catalog able to force more than one split.
}
