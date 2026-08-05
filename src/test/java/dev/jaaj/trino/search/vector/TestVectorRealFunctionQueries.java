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

import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;
import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestVectorRealFunctionQueries
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
    public void testEuclideanDistanceOnReal()
    {
        assertQuery("SELECT euclidean_distance(ARRAY[REAL '0.0', REAL '0.0'], ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 5.0");
    }

    @Test
    public void testDotProductOnReal()
    {
        assertQuery("SELECT dot_product(ARRAY[REAL '1.0', REAL '2.0'], ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 11.0");
    }

    @Test
    public void testCosineSimilarityOnReal()
    {
        assertQuery("SELECT cosine_similarity(ARRAY[REAL '1.0', REAL '0.0'], ARRAY[REAL '0.0', REAL '1.0'])", "SELECT 0.0");
    }

    @Test
    public void testCosineDistanceOnReal()
    {
        assertQuery("SELECT cosine_distance(ARRAY[REAL '1.0', REAL '2.0'], ARRAY[REAL '1.0', REAL '2.0'])", "SELECT 0.0");
    }

    @Test
    public void testNewMetricsOnReal()
    {
        assertQuery("SELECT euclidean_squared_distance(ARRAY[REAL '0.0', REAL '0.0'], ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 25.0");
        assertQuery("SELECT manhattan_distance(ARRAY[REAL '1.0', REAL '-2.0'], ARRAY[REAL '4.0', REAL '2.0'])", "SELECT 7.0");
        assertQuery("SELECT l2_norm(ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 5.0");
    }

    @Test
    public void testNormalizeVectorKeepsRealType()
    {
        // AbstractTestQueryFramework.assertQuery runs the expected clause through H2, which does
        // not accept the "REAL 'x'" literal syntax and would otherwise fail to parse it. 0.6 and
        // 0.8 are also not exactly representable in float32, so the assertion is restructured to
        // check the returned type directly on the Trino result and compare the values with a
        // tolerance, instead of routing an expected clause through H2.
        MaterializedResult result = computeActual("SELECT normalize_vector(ARRAY[REAL '3.0', REAL '4.0'])");
        assertThat(result.getTypes()).containsExactly(new ArrayType(REAL));

        List<Object> row = result.getMaterializedRows().get(0).getFields();
        List<?> values = (List<?>) row.get(0);
        assertThat(values).hasSize(2);
        assertThat((Float) values.get(0)).isCloseTo(0.6f, within(1e-6f));
        assertThat((Float) values.get(1)).isCloseTo(0.8f, within(1e-6f));
    }

    @Test
    public void testDoubleOverloadStillResolves()
    {
        // adding real overloads must not shadow the double ones
        assertQuery("SELECT euclidean_distance(ARRAY[0.0, 0.0], ARRAY[3.0, 4.0])", "SELECT 5.0");
        assertQuery("SELECT manhattan_distance(ARRAY[1.0, -2.0], ARRAY[4.0, 2.0])", "SELECT 7.0");
    }

    @Test
    public void testDoubleTypedArrayStillResolvesToDoubleNormalizeVector()
    {
        // Not in the brief. normalize_vector is the only function in this class whose return
        // type differs between overloads (array(double) vs array(real)), so it is the only one
        // where an overload-resolution mistake is visible in the result type rather than just in
        // float32-vs-float64 precision. An explicitly array(double)-typed argument must still
        // resolve to the array(double) overload by exact match: this is what a real
        // array(double) table column experiences, so it is unaffected by the coercion-based
        // preference for array(real) that applies to untyped decimal literals (see the report for
        // that finding).
        MaterializedResult result = computeActual("SELECT normalize_vector(CAST(ARRAY[3.0, 4.0] AS array(double)))");
        assertThat(result.getTypes()).containsExactly(new ArrayType(DOUBLE));
    }

    @Test
    public void testRealLengthMismatchIsRejected()
    {
        assertQueryFails(
                "SELECT euclidean_distance(ARRAY[REAL '1.0'], ARRAY[REAL '1.0', REAL '2.0'])",
                ".*The arguments must have the same length.*");
    }
}
