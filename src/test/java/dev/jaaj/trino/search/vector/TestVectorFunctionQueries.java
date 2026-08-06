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
import io.trino.spi.type.ArrayType;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedResult;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import java.util.List;

import static io.trino.spi.type.RealType.REAL;
import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

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
    public void testNormalizeVectorUncastFormNowBindsToRealOverload()
    {
        // Coverage for the natural, uncast call form that testNormalizeVectorValues stopped
        // exercising once it needed an explicit CAST (see the comment there). Since Task 5, an
        // untyped decimal array literal such as ARRAY[3.0, 4.0] resolves to the plugin's
        // array(real) overload of normalize_vector, not the engine's array(double) one - this is
        // an accepted consequence of reusing the native SQL name (see README.md). This test pins
        // that: the uncast form now returns an array(real), with the values still correct within
        // float32 precision (3.0 and 4.0 are exactly representable in float32, so 0.6 and 0.8
        // still need a tolerance because the division is not exact in either precision).
        MaterializedResult result = computeActual("SELECT normalize_vector(ARRAY[3.0, 4.0])");
        assertThat(result.getTypes()).containsExactly(new ArrayType(REAL));

        List<Object> row = result.getMaterializedRows().get(0).getFields();
        List<?> values = (List<?>) row.get(0);
        assertThat(values).hasSize(2);
        assertThat((Float) values.get(0)).isCloseTo(0.6f, within(1e-6f));
        assertThat((Float) values.get(1)).isCloseTo(0.8f, within(1e-6f));
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

    @Test
    public void testEuclideanSquaredDistanceOnDoubleOverload()
    {
        // An untyped decimal array literal binds to the array(real) overload (see
        // testNormalizeVectorUncastFormNowBindsToRealOverload), so every other test in this class
        // that uses ARRAY[0.0, 0.0]-style literals for euclidean_squared_distance, manhattan_distance
        // and l2_norm never exercises their array(double) path: 0.0, 3.0, 4.0, 1.0, -2.0 and 2.0 are
        // all exactly representable in float32, so those tests pass identically whichever reader
        // computes them. The explicit CAST here forces exact-match resolution to the array(double)
        // overload, and 0.1/0.2/0.3/0.4 are not exact in float32, so DOUBLE_READER and REAL_READER
        // disagree measurably: this pins that the array(double) overload actually reads through
        // DOUBLE_READER. assertQuery is not used because it rounds to 5 significant digits, which
        // would hide the gap between the double-precision and float32-widened results.
        assertThat(actualDouble(
                "SELECT euclidean_squared_distance(CAST(ARRAY[0.1, 0.2] AS array(double)), CAST(ARRAY[0.3, 0.4] AS array(double)))"))
                .isCloseTo(0.08, within(1e-9));
    }

    @Test
    public void testManhattanDistanceOnDoubleOverload()
    {
        // See testEuclideanSquaredDistanceOnDoubleOverload for why the CAST and the tolerance-based
        // assertion are needed to actually exercise DOUBLE_READER.
        assertThat(actualDouble(
                "SELECT manhattan_distance(CAST(ARRAY[0.1, 0.2] AS array(double)), CAST(ARRAY[0.3, 0.4] AS array(double)))"))
                .isCloseTo(0.4, within(1e-9));
    }

    @Test
    public void testL2NormOnDoubleOverload()
    {
        // See testEuclideanSquaredDistanceOnDoubleOverload for why the CAST and the tolerance-based
        // assertion are needed to actually exercise DOUBLE_READER.
        assertThat(actualDouble("SELECT l2_norm(CAST(ARRAY[0.1, 0.2] AS array(double)))"))
                .isCloseTo(0.223606797749979, within(1e-9));
    }

    private double actualDouble(String sql)
    {
        return (double) (Double) computeActual(sql).getOnlyValue();
    }
}
