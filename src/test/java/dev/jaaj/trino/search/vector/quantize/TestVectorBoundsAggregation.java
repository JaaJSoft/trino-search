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

public class TestVectorBoundsAggregation
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
    public void testOffsetIsTheMidpointAndScaleSpansTheCodeRange()
    {
        assertQuery(
                """
                SELECT b.offsets[1], b.scale
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES (ARRAY[CAST(-1.0 AS DOUBLE)]), (ARRAY[CAST(3.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 1.0, 4.0 / 255.0");
    }

    @Test
    public void testFitsEachDimensionIndependently()
    {
        assertQuery(
                """
                SELECT b.offsets[1], b.offsets[2]
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES
                        (ARRAY[CAST(0.0 AS DOUBLE), CAST(10.0 AS DOUBLE)]),
                        (ARRAY[CAST(2.0 AS DOUBLE), CAST(20.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 1.0, 15.0");
    }

    /**
     * A corpus whose minimum equals its maximum across every dimension has no range at all. The
     * scale is zero rather than an error, and QuantizationBounds treats that as "every value here
     * is the offset".
     */
    @Test
    public void testConstantCorpusYieldsZeroScale()
    {
        assertQuery(
                """
                SELECT b.scale
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES (ARRAY[CAST(7.0 AS DOUBLE)]), (ARRAY[CAST(7.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT CAST(0.0 AS DOUBLE)");
    }

    /**
     * A dimension that never varies on its own still inherits the scale fitted across the whole
     * corpus, rather than getting a scale of its own: the scale is a single value, not one per
     * dimension.
     */
    @Test
    public void testConstantDimensionInAVaryingCorpusGetsTheGlobalScale()
    {
        assertQuery(
                """
                SELECT b.scale
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES
                        (ARRAY[CAST(0.0 AS DOUBLE), CAST(7.0 AS DOUBLE)]),
                        (ARRAY[CAST(4.0 AS DOUBLE), CAST(7.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 4.0 / 255.0");
    }

    /**
     * The scale has to fit the widest single-dimension range, not the span from the smallest
     * minimum to the largest maximum across dimensions. Two dimensions with the same range but
     * sitting at very different absolute levels must still yield the same scale as either one
     * alone.
     */
    @Test
    public void testScaleReflectsWidestDimensionRangeNotGlobalSpan()
    {
        assertQuery(
                """
                SELECT b.scale
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES
                        (ARRAY[CAST(0.0 AS DOUBLE), CAST(100.0 AS DOUBLE)]),
                        (ARRAY[CAST(1.0 AS DOUBLE), CAST(101.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 1.0 / 255.0");
    }

    @Test
    public void testRealVectorsAreAccepted()
    {
        assertQuery(
                """
                SELECT b.offsets[1]
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES (ARRAY[REAL '-1.0']), (ARRAY[REAL '3.0'])) AS t(v)
                )
                """,
                "SELECT 1.0");
    }

    /**
     * Bounds fitted across vectors of different length describe nothing, so this fails rather than
     * quietly fitting the shorter prefix.
     */
    @Test
    public void testMixedDimensionsAreRejected()
    {
        assertQueryFails(
                """
                SELECT vector_bounds_agg(v)
                FROM (VALUES
                    (ARRAY[CAST(1.0 AS DOUBLE)]),
                    (ARRAY[CAST(1.0 AS DOUBLE), CAST(2.0 AS DOUBLE)])) AS t(v)
                """,
                ".*same length.*");
    }

    @Test
    public void testNullVectorsAreIgnored()
    {
        assertQuery(
                """
                SELECT b.offsets[1]
                FROM (
                    SELECT vector_bounds_agg(v) AS b
                    FROM (VALUES (ARRAY[CAST(-1.0 AS DOUBLE)]), (CAST(NULL AS array(double))), (ARRAY[CAST(3.0 AS DOUBLE)])) AS t(v)
                )
                """,
                "SELECT 1.0");
    }

    /**
     * Compared with {@code IS NULL} rather than a literal null row: H2, which runs the expected
     * side of {@code assertQuery}, does not parse a {@code CAST(NULL AS row(...))} the way Trino
     * does, and {@code TestKnnAggregation} settles the same problem for {@code knn_agg} the same
     * way.
     */
    @Test
    public void testEmptyInputReturnsNull()
    {
        assertQuery(
                "SELECT vector_bounds_agg(v) IS NULL FROM (SELECT CAST(NULL AS array(double)) AS v WHERE false)",
                "SELECT true");
    }

    // The multiple-splits case (BoundsStateSerializer, @CombineFunction) is covered in
    // TestVectorBoundsAggregationDistributed, which uses a splittable connector: this class has no
    // catalog able to force more than one split.
}
