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

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestVectorMath
{
    private static Block doubles(Double... values)
    {
        BlockBuilder builder = DOUBLE.createBlockBuilder(null, values.length);
        for (Double value : values) {
            if (value == null) {
                builder.appendNull();
            }
            else {
                DOUBLE.writeDouble(builder, value);
            }
        }
        return builder.build();
    }

    @Test
    public void testEuclidean()
    {
        // 3-4-5 triangle
        assertThat(VectorMath.euclidean(doubles(0.0, 0.0), doubles(3.0, 4.0), DOUBLE_READER))
                .isCloseTo(5.0, within(1e-12));
    }

    @Test
    public void testEuclideanSquaredAvoidsSqrt()
    {
        assertThat(VectorMath.euclideanSquared(doubles(0.0, 0.0), doubles(3.0, 4.0), DOUBLE_READER))
                .isCloseTo(25.0, within(1e-12));
    }

    @Test
    public void testManhattan()
    {
        assertThat(VectorMath.manhattan(doubles(1.0, -2.0), doubles(4.0, 2.0), DOUBLE_READER))
                .isCloseTo(7.0, within(1e-12));
    }

    @Test
    public void testDotProduct()
    {
        assertThat(VectorMath.dotProduct(doubles(1.0, 2.0), doubles(3.0, 4.0), DOUBLE_READER))
                .isCloseTo(11.0, within(1e-12));
    }

    @Test
    public void testNorm()
    {
        assertThat(VectorMath.norm(doubles(3.0, 4.0), DOUBLE_READER)).isCloseTo(5.0, within(1e-12));
    }

    @Test
    public void testCosineSimilarityOfIdenticalVectorsIsOne()
    {
        assertThat(VectorMath.cosineSimilarity(doubles(1.0, 2.0), doubles(1.0, 2.0), DOUBLE_READER))
                .isCloseTo(1.0, within(1e-12));
    }

    @Test
    public void testCosineSimilarityOfOrthogonalVectorsIsZero()
    {
        assertThat(VectorMath.cosineSimilarity(doubles(1.0, 0.0), doubles(0.0, 1.0), DOUBLE_READER))
                .isCloseTo(0.0, within(1e-12));
    }

    @Test
    public void testDistancesAreSymmetric()
    {
        Block first = doubles(1.0, 5.0, -3.0);
        Block second = doubles(-2.0, 0.5, 4.0);
        assertThat(VectorMath.euclidean(first, second, DOUBLE_READER))
                .isCloseTo(VectorMath.euclidean(second, first, DOUBLE_READER), within(1e-12));
        assertThat(VectorMath.manhattan(first, second, DOUBLE_READER))
                .isCloseTo(VectorMath.manhattan(second, first, DOUBLE_READER), within(1e-12));
    }

    @Test
    public void testDistanceToSelfIsZero()
    {
        Block vector = doubles(1.0, 5.0, -3.0);
        assertThat(VectorMath.euclidean(vector, vector, DOUBLE_READER)).isCloseTo(0.0, within(1e-12));
        assertThat(VectorMath.manhattan(vector, vector, DOUBLE_READER)).isCloseTo(0.0, within(1e-12));
    }

    @Test
    public void testTriangleInequality()
    {
        Block a = doubles(0.0, 0.0);
        Block b = doubles(3.0, 4.0);
        Block c = doubles(1.0, 7.0);
        assertThat(VectorMath.euclidean(a, c, DOUBLE_READER))
                .isLessThanOrEqualTo(VectorMath.euclidean(a, b, DOUBLE_READER) + VectorMath.euclidean(b, c, DOUBLE_READER) + 1e-12);
    }

    @Test
    public void testEmptyVectorsGiveZero()
    {
        assertThat(VectorMath.euclideanSquared(doubles(), doubles(), DOUBLE_READER)).isEqualTo(0.0);
        assertThat(VectorMath.manhattan(doubles(), doubles(), DOUBLE_READER)).isEqualTo(0.0);
        assertThat(VectorMath.norm(doubles(), DOUBLE_READER)).isEqualTo(0.0);
    }

    @Test
    public void testLengthMismatchIsRejected()
    {
        assertThatThrownBy(() -> VectorMath.checkSameLength(doubles(1.0), doubles(1.0, 2.0)))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("The arguments must have the same length");
    }

    @Test
    public void testNullDetection()
    {
        assertThat(VectorMath.hasNulls(doubles(1.0, null), doubles(1.0, 2.0))).isTrue();
        assertThat(VectorMath.hasNulls(doubles(1.0, 2.0), doubles(1.0, 2.0))).isFalse();
    }

    @Test
    public void testCosineSimilarityWhenTheMagnitudeProductOverflows()
    {
        // each squared magnitude is finite at 1e308, but their product is not
        Block huge = doubles(1e154);
        assertThat(VectorMath.cosineSimilarity(huge, huge, DOUBLE_READER)).isCloseTo(1.0, within(1e-12));
    }

    @Test
    public void testCosineSimilarityWhenTheMagnitudesThemselvesOverflow()
    {
        Block huge = doubles(1e200, 1e200);
        assertThat(VectorMath.cosineSimilarity(huge, huge, DOUBLE_READER)).isCloseTo(1.0, within(1e-12));
        assertThat(VectorMath.cosineSimilarity(doubles(1e200, 0.0), doubles(0.0, 1e200), DOUBLE_READER))
                .isCloseTo(0.0, within(1e-12));
    }

    @Test
    public void testCosineSimilarityOfTinyVectorsIsNotMistakenForZeroMagnitude()
    {
        Block tiny = doubles(1e-200, 1e-200);
        assertThat(VectorMath.cosineSimilarity(tiny, tiny, DOUBLE_READER)).isCloseTo(1.0, within(1e-12));
    }

    @Test
    public void testNormWhenTheSumOfSquaresOverflows()
    {
        assertThat(VectorMath.norm(doubles(1e200, 1e200), DOUBLE_READER))
                .isCloseTo(Math.sqrt(2) * 1e200, within(1e188));
    }
}
