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
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestQuantizeFunctionQueries
        extends AbstractTestQueryFramework
{
    /**
     * Offsets 0 and scales 1, so a code is the value itself and the expected numbers can be read
     * without doing the arithmetic in one's head.
     */
    private static final String UNIT_BOUNDS =
            "CAST(ROW(ARRAY[CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE)], ARRAY[CAST(1.0 AS DOUBLE), CAST(1.0 AS DOUBLE)]) "
                    + "AS row(offsets array(double), scales array(double)))";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    /**
     * Compared with an in-Trino array equality rather than a literal {@code array(tinyint)} on the
     * expected side: H2, which runs the expected side of {@code assertQuery}, does not agree with
     * Trino on an {@code array(tinyint)} literal even when the printed rows look identical, the same
     * problem {@code TestVectorBoundsAggregation.testEmptyInputReturnsNull} works around for
     * {@code vector_bounds_agg}.
     */
    @Test
    public void testQuantizeToTinyint()
    {
        assertQuery(
                "SELECT quantize_vector_tinyint(CAST(ARRAY[3.0, -4.0] AS array(double)), " + UNIT_BOUNDS + ") "
                        + "= ARRAY[CAST(3 AS TINYINT), CAST(-4 AS TINYINT)]",
                "SELECT true");
    }

    @Test
    public void testInt8AliasResolvesToTheSameFunction()
    {
        assertQuery(
                "SELECT quantize_vector_int8(CAST(ARRAY[3.0, -4.0] AS array(double)), " + UNIT_BOUNDS + ") "
                        + "= ARRAY[CAST(3 AS TINYINT), CAST(-4 AS TINYINT)]",
                "SELECT true");
    }

    @Test
    public void testRealVectorsAreAccepted()
    {
        assertQuery(
                "SELECT quantize_vector_tinyint(ARRAY[REAL '3.0', REAL '-4.0'], " + UNIT_BOUNDS + ") "
                        + "= ARRAY[CAST(3 AS TINYINT), CAST(-4 AS TINYINT)]",
                "SELECT true");
    }

    @Test
    public void testValuesOutsideTheBoundsClamp()
    {
        assertQuery(
                "SELECT quantize_vector_tinyint(CAST(ARRAY[1000.0, -1000.0] AS array(double)), " + UNIT_BOUNDS + ") "
                        + "= ARRAY[CAST(127 AS TINYINT), CAST(-128 AS TINYINT)]",
                "SELECT true");
    }

    @Test
    public void testNullElementYieldsNull()
    {
        assertQuery(
                "SELECT quantize_vector_tinyint(CAST(ARRAY[1.0, NULL] AS array(double)), " + UNIT_BOUNDS + ") IS NULL",
                "SELECT true");
    }

    @Test
    public void testDimensionMismatchWithTheBoundsFails()
    {
        assertQueryFails(
                "SELECT quantize_vector_tinyint(CAST(ARRAY[1.0, 2.0, 3.0] AS array(double)), " + UNIT_BOUNDS + ")",
                ".*3 components but the quantisation bounds were fitted on 2.*");
    }

    /**
     * The whole pipeline in one statement: fit, encode, and check that the midpoint of the fitted
     * range lands on code zero.
     */
    @Test
    public void testFittedBoundsPutTheMidpointOnZero()
    {
        assertQuery(
                """
                SELECT quantize_vector_tinyint(CAST(ARRAY[1.0] AS array(double)), b) = ARRAY[CAST(0 AS TINYINT)]
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES (ARRAY[CAST(-1.0 AS DOUBLE)]), (ARRAY[CAST(3.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT true");
    }
}
