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

public class TestQuantizedDistanceQueries
        extends AbstractTestQueryFramework
{
    private static final String UNIT_BOUNDS =
            "CAST(ROW(ARRAY[CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE)], CAST(1.0 AS DOUBLE)) "
                    + "AS row(offsets array(double), scale double))";
    private static final String ORIGIN = "ARRAY[CAST(0 AS TINYINT), CAST(0 AS TINYINT)]";
    private static final String THREE_FOUR = "ARRAY[CAST(3 AS TINYINT), CAST(4 AS TINYINT)]";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testEuclideanDistance()
    {
        assertQuery(
                "SELECT euclidean_distance(" + ORIGIN + ", " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT 5.0");
    }

    @Test
    public void testEuclideanSquaredDistance()
    {
        assertQuery(
                "SELECT euclidean_squared_distance(" + ORIGIN + ", " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT 25.0");
    }

    @Test
    public void testManhattanDistance()
    {
        assertQuery(
                "SELECT manhattan_distance(" + ORIGIN + ", " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT 7.0");
    }

    @Test
    public void testDotProduct()
    {
        assertQuery(
                "SELECT dot_product(ARRAY[CAST(1 AS TINYINT), CAST(2 AS TINYINT)], " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT 11.0");
    }

    @Test
    public void testCosineSimilarityOfIdenticalVectors()
    {
        assertQuery(
                "SELECT cosine_similarity(" + THREE_FOUR + ", " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT 1.0");
    }

    @Test
    public void testCosineDistanceOfIdenticalVectors()
    {
        assertQuery(
                "SELECT cosine_distance(" + THREE_FOUR + ", " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT 0.0");
    }

    @Test
    public void testNullElementYieldsNull()
    {
        assertQuery(
                "SELECT euclidean_distance(CAST(ARRAY[CAST(1 AS TINYINT), NULL] AS array(tinyint)), "
                        + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                "SELECT CAST(NULL AS DOUBLE)");
    }

    @Test
    public void testDimensionMismatchFails()
    {
        assertQueryFails(
                "SELECT euclidean_distance(ARRAY[CAST(1 AS TINYINT)], " + THREE_FOUR + ", " + UNIT_BOUNDS + ")",
                ".*same length.*");
    }

    /**
     * The bounds argument is not optional on code arrays, and there is nothing in the type system
     * that can enforce it: array(tinyint) coerces implicitly to the float vector types, so a
     * two-argument call compiles, binds to an exact-vector overload and computes on the raw codes
     * with no scale at all. This pins that the trap exists and is silent, which is what makes it a
     * trap; which of the two float overloads it lands on is not observable from here, since a code
     * widens to the same value through either reader and both kernels accumulate in double.
     */
    @Test
    public void testTwoArgumentCallOnCodeArraysComputesOnTheRawCodesWithNoScale()
    {
        assertQuery(
                "SELECT euclidean_squared_distance(" + ORIGIN + ", " + THREE_FOUR + ")",
                "SELECT 25.0");
    }
}
