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

public class TestBinaryDistanceQueries
        extends AbstractTestQueryFramework
{
    /**
     * Dimension 4, all four bits set.
     */
    private static final String ALL_ONES = "from_hex('000000040F')";
    /**
     * Dimension 4, no bits set.
     */
    private static final String ALL_ZEROS = "from_hex('0000000400')";
    /**
     * Dimension 8, all bits set: a different dimension, to check the header is enforced.
     */
    private static final String WIDER = "from_hex('00000008FF')";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testHammingDistance()
    {
        assertQuery("SELECT hamming_distance(" + ALL_ONES + ", " + ALL_ZEROS + ")", "SELECT CAST(4 AS BIGINT)");
        assertQuery("SELECT hamming_distance(" + ALL_ONES + ", " + ALL_ONES + ")", "SELECT CAST(0 AS BIGINT)");
    }

    @Test
    public void testEuclideanSquaredDistanceIsFourTimesHamming()
    {
        assertQuery("SELECT euclidean_squared_distance(" + ALL_ONES + ", " + ALL_ZEROS + ")", "SELECT 16.0");
    }

    @Test
    public void testEuclideanDistance()
    {
        assertQuery("SELECT euclidean_distance(" + ALL_ONES + ", " + ALL_ZEROS + ")", "SELECT 4.0");
    }

    @Test
    public void testManhattanDistanceIsTwiceHamming()
    {
        assertQuery("SELECT manhattan_distance(" + ALL_ONES + ", " + ALL_ZEROS + ")", "SELECT 8.0");
    }

    @Test
    public void testDotProductOfOppositeVectors()
    {
        assertQuery("SELECT dot_product(" + ALL_ONES + ", " + ALL_ZEROS + ")", "SELECT -4.0");
    }

    @Test
    public void testCosineSimilarityAndDistance()
    {
        assertQuery("SELECT cosine_similarity(" + ALL_ONES + ", " + ALL_ZEROS + ")", "SELECT -1.0");
        assertQuery("SELECT cosine_distance(" + ALL_ONES + ", " + ALL_ONES + ")", "SELECT 0.0");
    }

    @Test
    public void testMismatchedDimensionsFail()
    {
        assertQueryFails(
                "SELECT euclidean_distance(" + ALL_ONES + ", " + WIDER + ")",
                ".*same length.*");
    }

    /**
     * Cosine is the one metric that reads a dimension of its own before comparing the two, so it is
     * the one that can report a mismatch as something else.
     */
    @Test
    public void testMismatchedDimensionsFailForCosineToo()
    {
        assertQueryFails(
                "SELECT cosine_similarity(" + ALL_ONES + ", " + WIDER + ")",
                ".*same length.*");
    }

    @Test
    public void testMalformedHeaderFails()
    {
        assertQueryFails(
                "SELECT hamming_distance(from_hex('0000'), from_hex('0000'))",
                ".*not a binary vector.*");
    }

    /**
     * The round trip a user actually performs: quantise two vectors against the same fitted bounds
     * and compare the codes.
     */
    @Test
    public void testQuantiseThenCompare()
    {
        assertQuery(
                """
                SELECT hamming_distance(
                    quantize_vector_varbinary(CAST(ARRAY[1.0, 1.0] AS array(double)), b),
                    quantize_vector_varbinary(CAST(ARRAY[1.0, -1.0] AS array(double)), b))
                FROM (
                    SELECT CAST(ROW(ARRAY[CAST(0.0 AS DOUBLE), CAST(0.0 AS DOUBLE)],
                                    ARRAY[CAST(1.0 AS DOUBLE), CAST(1.0 AS DOUBLE)])
                        AS row(offsets array(double), scales array(double))) AS b
                )
                """,
                "SELECT CAST(1 AS BIGINT)");
    }
}
