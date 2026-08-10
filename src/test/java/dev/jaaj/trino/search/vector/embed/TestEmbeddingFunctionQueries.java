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
package dev.jaaj.trino.search.vector.embed;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.spi.type.ArrayType;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;
import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestEmbeddingFunctionQueries
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
    public void testReturnTypes()
    {
        assertThat(computeActual("SELECT to_vector_real('hello world', 8, 'word')").getTypes())
                .containsExactly(new ArrayType(REAL));
        assertThat(computeActual("SELECT to_vector_double('hello world', 8, 'word')").getTypes())
                .containsExactly(new ArrayType(DOUBLE));
    }

    @Test
    public void testDimensionIsRespected()
    {
        assertQuery("SELECT cardinality(to_vector_real('hello world', 768, 'word'))", "SELECT CAST(768 AS BIGINT)");
        assertQuery("SELECT cardinality(to_vector_double('hello world', 768, 'word'))", "SELECT CAST(768 AS BIGINT)");
    }

    @Test
    public void testResultHasUnitNorm()
    {
        assertThat((Double) computeScalar("SELECT l2_norm(to_vector_real('hello world', 256, 'word'))"))
                .isCloseTo(1.0, within(1e-6));
        assertThat((Double) computeScalar("SELECT l2_norm(to_vector_double('hello world', 256, 'word'))"))
                .isCloseTo(1.0, within(1e-12));
    }

    /**
     * The aliases are the ML-vocabulary spelling of the same functions. Only an end-to-end query
     * proves they were actually registered, since nothing in the Java code references them.
     */
    @Test
    public void testAliasesResolveToTheSameFunction()
    {
        assertQuery(
                "SELECT to_vector_fp32('hello world', 16, 'word') = to_vector_real('hello world', 16, 'word')",
                "SELECT true");
        assertQuery(
                "SELECT to_vector_fp64('hello world', 16, 'word') = to_vector_double('hello world', 16, 'word')",
                "SELECT true");
    }

    @Test
    public void testEveryAlgorithmNameResolves()
    {
        for (String algorithm : new String[] {"word", "char_3gram", "char_4gram", "char_5gram"}) {
            assertQuery(
                    "SELECT cardinality(to_vector_real('hello world', 32, '%s'))".formatted(algorithm),
                    "SELECT CAST(32 AS BIGINT)");
        }
    }

    @Test
    public void testNullTextGivesNull()
    {
        assertQuery("SELECT to_vector_real(CAST(NULL AS varchar), 8, 'word') IS NULL", "SELECT true");
        assertQuery("SELECT to_vector_double(CAST(NULL AS varchar), 8, 'word') IS NULL", "SELECT true");
    }

    /**
     * A text column with an empty row must not fail the scan, which is why this returns the zero
     * vector where normalize_vector raises on a zero magnitude.
     */
    @Test
    public void testTextWithoutTokenGivesTheZeroVector()
    {
        assertQuery(
                "SELECT to_vector_double('!!!', 4, 'word')",
                "SELECT ARRAY[CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE)]");
    }

    @Test
    public void testInvalidDimensionIsRejected()
    {
        assertQueryFails("SELECT to_vector_real('a', 0, 'word')", ".*dimension must be between 1 and 65536.*");
        assertQueryFails("SELECT to_vector_real('a', 65537, 'word')", ".*dimension must be between 1 and 65536.*");
    }

    @Test
    public void testUnknownAlgorithmIsRejected()
    {
        assertQueryFails(
                "SELECT to_vector_real('a', 8, 'bag_of_words')",
                ".*Unknown embedding algorithm 'bag_of_words'.*");
    }

    /**
     * The whole justification for returning array(real) is that the result chains into the
     * plugin's array(real) metric overloads with no CAST. A CAST would double the memory read and
     * defeat the choice of return type, so it is pinned rather than assumed.
     */
    @Test
    public void testResultChainsIntoTheRealMetricsWithoutCast()
    {
        assertThat((Double) computeScalar(
                "SELECT euclidean_distance("
                        + "to_vector_real('hello world', 64, 'word'), "
                        + "to_vector_real('hello world', 64, 'word'))"))
                .isCloseTo(0.0, within(1e-6));
        assertThat((Double) computeScalar(
                "SELECT cosine_similarity("
                        + "to_vector_real('hello world', 64, 'word'), "
                        + "to_vector_real('hello world', 64, 'word'))"))
                .isCloseTo(1.0, within(1e-6));
    }

    @Test
    public void testDifferentTextsAreFurtherApartThanIdenticalOnes()
    {
        double same = (Double) computeScalar(
                "SELECT cosine_distance("
                        + "to_vector_double('the quick brown fox', 256, 'word'), "
                        + "to_vector_double('the quick brown fox', 256, 'word'))");
        double different = (Double) computeScalar(
                "SELECT cosine_distance("
                        + "to_vector_double('the quick brown fox', 256, 'word'), "
                        + "to_vector_double('entirely unrelated wording here', 256, 'word'))");
        assertThat(same).isLessThan(different);
    }
}
