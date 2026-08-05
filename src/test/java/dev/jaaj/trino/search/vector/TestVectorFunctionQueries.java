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
package dev.jaaj.trino.search.vector;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestVectorFunctionQueries
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
    public void testEuclideanSquaredDistance()
    {
        assertQuery("SELECT euclidean_squared_distance(ARRAY[0.0, 0.0], ARRAY[3.0, 4.0])", "SELECT 25.0");
    }

    @Test
    public void testManhattanDistance()
    {
        assertQuery("SELECT manhattan_distance(ARRAY[1.0, -2.0], ARRAY[4.0, 2.0])", "SELECT 7.0");
    }

    @Test
    public void testL2Norm()
    {
        assertQuery("SELECT l2_norm(ARRAY[3.0, 4.0])", "SELECT 5.0");
    }

    @Test
    public void testNormalizeVectorHasUnitNorm()
    {
        assertQuery("SELECT l2_norm(normalize_vector(ARRAY[3.0, 4.0]))", "SELECT 1.0");
    }

    @Test
    public void testNormalizeVectorValues()
    {
        // CAST to array(double) forces exact-match overload resolution. Since Task 5 added an
        // array(real) overload of normalize_vector with the same SQL name, an untyped decimal
        // array literal like ARRAY[3.0, 4.0] now resolves to the array(real) overload instead
        // (Trino's function binder prefers the narrower real coercion over the double one when
        // both are applicable through coercion). A real array(double) column is unaffected: it
        // binds to the double overload by exact match, with no coercion or ambiguity involved.
        assertQuery("SELECT normalize_vector(CAST(ARRAY[3.0, 4.0] AS array(double)))", "SELECT ARRAY[CAST(0.6 AS DOUBLE), CAST(0.8 AS DOUBLE)]");
    }

    @Test
    public void testLengthMismatchIsRejected()
    {
        assertQueryFails(
                "SELECT manhattan_distance(ARRAY[1.0], ARRAY[1.0, 2.0])",
                ".*The arguments must have the same length.*");
    }

    @Test
    public void testNullElementGivesNull()
    {
        assertQuery("SELECT manhattan_distance(ARRAY[1.0, NULL], ARRAY[1.0, 2.0]) IS NULL", "SELECT true");
        assertQuery("SELECT l2_norm(ARRAY[1.0, NULL]) IS NULL", "SELECT true");
    }

    @Test
    public void testNullArgumentGivesNull()
    {
        assertQuery("SELECT manhattan_distance(NULL, ARRAY[1.0, 2.0]) IS NULL", "SELECT true");
    }

    @Test
    public void testNormalizeZeroVectorIsRejected()
    {
        assertQueryFails(
                "SELECT normalize_vector(ARRAY[0.0, 0.0])",
                ".*Vector magnitude cannot be zero.*");
    }

    @Test
    public void testEmptyVectors()
    {
        assertQuery("SELECT euclidean_squared_distance(ARRAY[], ARRAY[])", "SELECT 0.0");
        assertQuery("SELECT l2_norm(CAST(ARRAY[] AS array(double)))", "SELECT 0.0");
    }
}
