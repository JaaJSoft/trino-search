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

import io.trino.spi.block.Block;
import io.trino.spi.block.ByteArrayBlock;
import io.trino.spi.block.DictionaryBlock;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Optional;
import java.util.SplittableRandom;

import static io.trino.spi.type.TinyintType.TINYINT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestQuantizedVectorMath
{
    private static Block codes(int... values)
    {
        byte[] bytes = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            bytes[i] = (byte) values[i];
        }
        return new ByteArrayBlock(bytes.length, Optional.empty(), bytes);
    }

    /**
     * What the kernel must agree with: dequantise both sides, then compute in double. Anything the
     * kernel does beyond this, including dropping the offsets, is an optimisation that has to
     * leave this number unchanged.
     */
    private static double referenceEuclideanSquared(Block first, Block second, QuantizationBounds bounds)
    {
        double sum = 0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            double difference = bounds.decode(i, TINYINT.getByte(first, i)) - bounds.decode(i, TINYINT.getByte(second, i));
            sum += difference * difference;
        }
        return sum;
    }

    @Test
    public void testEuclideanSquaredAgainstAHandComputedValue()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.euclideanSquared(codes(0, 0), codes(3, 4), bounds)).isEqualTo(25.0);
    }

    /**
     * The offsets cancel in a difference, so the kernel is free not to read them. A non-zero offset
     * that changed the answer would mean it reads them wrongly rather than not at all.
     */
    @Test
    public void testEuclideanSquaredIgnoresTheOffsets()
    {
        QuantizationBounds atZero = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        QuantizationBounds shifted = QuantizationBounds.forTesting(new double[] {17.5, -3.25}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.euclideanSquared(codes(0, 0), codes(3, 4), shifted))
                .isEqualTo(QuantizedVectorMath.euclideanSquared(codes(0, 0), codes(3, 4), atZero));
    }

    @Test
    public void testEuclideanSquaredAppliesThePerDimensionScales()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {2, 0.5});
        // (2 * 3)^2 + (0.5 * 4)^2 = 36 + 4
        assertThat(QuantizedVectorMath.euclideanSquared(codes(0, 0), codes(3, 4), bounds)).isEqualTo(40.0);
    }

    @Test
    public void testManhattanAppliesThePerDimensionScales()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {2, 0.5});
        assertThat(QuantizedVectorMath.manhattan(codes(0, 0), codes(3, -4), bounds)).isEqualTo(6.0 + 2.0);
    }

    @Test
    public void testEuclideanIsTheRootOfTheSquare()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.euclidean(codes(0, 0), codes(3, 4), bounds)).isEqualTo(5.0);
    }

    @Test
    public void testMatchesTheDequantiseThenComputeReferenceOverRandomVectors()
    {
        SplittableRandom random = new SplittableRandom(42);
        int dimension = 200;
        double[] offsets = new double[dimension];
        double[] scales = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            offsets[i] = random.nextDouble(-5, 5);
            scales[i] = random.nextDouble(0.001, 0.5);
        }
        QuantizationBounds bounds = QuantizationBounds.forTesting(offsets, scales);

        for (int trial = 0; trial < 20; trial++) {
            int[] left = new int[dimension];
            int[] right = new int[dimension];
            for (int i = 0; i < dimension; i++) {
                left[i] = random.nextInt(-128, 128);
                right[i] = random.nextInt(-128, 128);
            }
            Block first = codes(left);
            Block second = codes(right);
            assertThat(QuantizedVectorMath.euclideanSquared(first, second, bounds))
                    .isCloseTo(referenceEuclideanSquared(first, second, bounds), within(1e-9));
        }
    }

    /**
     * A bounded call may return anything above the limit once the partial sum passes it, so the
     * only contract to check is that it does not claim a candidate is under the limit when it is
     * not, and that it is exact when the true value is under.
     */
    @Test
    public void testBoundedFormAgreesBelowTheLimitAndGivesUpAbove()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.euclideanSquaredBounded(codes(0, 0), codes(3, 4), bounds, 100.0))
                .isEqualTo(25.0);
        assertThat(QuantizedVectorMath.euclideanSquaredBounded(codes(0, 0), codes(3, 4), bounds, 1.0))
                .isGreaterThan(1.0);
    }

    private static double referenceDotProduct(Block first, Block second, QuantizationBounds bounds)
    {
        double sum = 0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            sum += bounds.decode(i, TINYINT.getByte(first, i)) * bounds.decode(i, TINYINT.getByte(second, i));
        }
        return sum;
    }

    @Test
    public void testDotProductAgainstAHandComputedValue()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.dotProduct(codes(1, 2), codes(3, 4), bounds)).isEqualTo(11.0);
    }

    /**
     * A product leaves cross terms in the offset, so unlike euclidean this one must read them. A
     * kernel that dropped them would pass every zero-offset test and be wrong on real bounds.
     */
    @Test
    public void testDotProductReadsTheOffsets()
    {
        QuantizationBounds shifted = QuantizationBounds.forTesting(new double[] {10, 10}, new double[] {1, 1});
        // (10+1)*(10+3) + (10+2)*(10+4) = 143 + 168
        assertThat(QuantizedVectorMath.dotProduct(codes(1, 2), codes(3, 4), shifted)).isEqualTo(311.0);
    }

    @Test
    public void testCosineSimilarityOfIdenticalVectorsIsOne()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.cosineSimilarity(codes(3, 4), codes(3, 4), bounds)).isEqualTo(1.0);
    }

    @Test
    public void testCosineSimilarityOfOrthogonalVectorsIsZero()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(QuantizedVectorMath.cosineSimilarity(codes(5, 0), codes(0, 7), bounds)).isEqualTo(0.0);
    }

    @Test
    public void testDotProductMatchesTheDequantiseThenComputeReference()
    {
        SplittableRandom random = new SplittableRandom(7);
        int dimension = 128;
        double[] offsets = new double[dimension];
        double[] scales = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            offsets[i] = random.nextDouble(-2, 2);
            scales[i] = random.nextDouble(0.001, 0.1);
        }
        QuantizationBounds bounds = QuantizationBounds.forTesting(offsets, scales);

        int[] left = new int[dimension];
        int[] right = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            left[i] = random.nextInt(-128, 128);
            right[i] = random.nextInt(-128, 128);
        }
        assertThat(QuantizedVectorMath.dotProduct(codes(left), codes(right), bounds))
                .isCloseTo(referenceDotProduct(codes(left), codes(right), bounds), within(1e-9));
    }

    /**
     * The fast path reads the block's backing array directly, so it may only run on a block that
     * stores its components contiguously and in order. A dictionary block maps position i somewhere
     * else in a shared array, so indexing its raw values would compute a distance between vectors
     * nobody asked for.
     */
    @Test
    public void testDictionaryBlocksTakeTheGeneralPath()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        Block underlying = codes(0, 0, 3, 4);
        Block first = DictionaryBlock.create(2, underlying, new int[] {0, 1});
        Block second = DictionaryBlock.create(2, underlying, new int[] {2, 3});
        assertThat(QuantizedVectorMath.euclideanSquared(first, second, bounds)).isEqualTo(25.0);
    }

    /**
     * Dimensions either side of a vector register's width are where a loop bound that forgets its
     * tail, or one that processes it twice, shows up.
     */
    @Test
    public void testVectorisedAndTailLengthsAgreeWithTheReference()
    {
        SplittableRandom random = new SplittableRandom(101);
        for (int dimension : new int[] {1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 768}) {
            double[] offsets = new double[dimension];
            double[] scales = new double[dimension];
            Arrays.fill(scales, 0.25);
            QuantizationBounds bounds = QuantizationBounds.forTesting(offsets, scales);

            int[] left = new int[dimension];
            int[] right = new int[dimension];
            for (int i = 0; i < dimension; i++) {
                left[i] = random.nextInt(-128, 128);
                right[i] = random.nextInt(-128, 128);
            }
            Block first = codes(left);
            Block second = codes(right);
            assertThat(QuantizedVectorMath.euclideanSquared(first, second, bounds))
                    .as("dimension %s", dimension)
                    .isCloseTo(referenceEuclideanSquared(first, second, bounds), within(1e-9));
        }
    }

    /**
     * The scales vary per dimension, so a kernel that hoisted one scale out of the loop would pass
     * every uniform-scale test above and be wrong here.
     */
    @Test
    public void testVectorisedPathHonoursPerDimensionScales()
    {
        int dimension = 128;
        double[] offsets = new double[dimension];
        double[] scales = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            scales[i] = 1.0 / (i + 1);
        }
        QuantizationBounds bounds = QuantizationBounds.forTesting(offsets, scales);

        int[] left = new int[dimension];
        int[] right = new int[dimension];
        Arrays.fill(left, 10);
        Arrays.fill(right, 4);
        Block first = codes(left);
        Block second = codes(right);
        assertThat(QuantizedVectorMath.euclideanSquared(first, second, bounds))
                .isCloseTo(referenceEuclideanSquared(first, second, bounds), within(1e-9));
    }
}
