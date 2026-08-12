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

import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestBinaryVectorMath
{
    /**
     * The codes stand for a vector of -1 and +1 components. This is the direct computation every
     * closed form below has to agree with.
     */
    private static double[] asPlusMinusOne(boolean[] bits)
    {
        double[] vector = new double[bits.length];
        for (int i = 0; i < bits.length; i++) {
            vector[i] = bits[i] ? 1.0 : -1.0;
        }
        return vector;
    }

    private static Slice pack(boolean[] bits)
    {
        return BinaryCodes.pack(bits.length, i -> bits[i]);
    }

    private static boolean[] randomBits(SplittableRandom random, int dimension)
    {
        boolean[] bits = new boolean[dimension];
        for (int i = 0; i < dimension; i++) {
            bits[i] = random.nextBoolean();
        }
        return bits;
    }

    @Test
    public void testIdenticalVectorsHaveZeroDistanceAndUnitCosine()
    {
        Slice codes = pack(new boolean[] {true, false, true, true});
        assertThat(BinaryVectorMath.euclideanSquared(codes, codes)).isZero();
        assertThat(BinaryVectorMath.euclidean(codes, codes)).isZero();
        assertThat(BinaryVectorMath.manhattan(codes, codes)).isZero();
        assertThat(BinaryVectorMath.cosineSimilarity(codes, codes)).isEqualTo(1.0);
    }

    @Test
    public void testOppositeVectorsAreAntipodal()
    {
        Slice first = pack(new boolean[] {true, true, true, true});
        Slice second = pack(new boolean[] {false, false, false, false});
        // every component differs by 2, so the squared distance is 4 * 4
        assertThat(BinaryVectorMath.euclideanSquared(first, second)).isEqualTo(16.0);
        assertThat(BinaryVectorMath.dotProduct(first, second)).isEqualTo(-4.0);
        assertThat(BinaryVectorMath.cosineSimilarity(first, second)).isEqualTo(-1.0);
    }

    @Test
    public void testEveryClosedFormMatchesTheDirectPlusMinusOneComputation()
    {
        SplittableRandom random = new SplittableRandom(3);
        for (int dimension : new int[] {1, 7, 8, 63, 64, 65, 200, 768}) {
            boolean[] leftBits = randomBits(random, dimension);
            boolean[] rightBits = randomBits(random, dimension);
            Slice left = pack(leftBits);
            Slice right = pack(rightBits);
            double[] leftVector = asPlusMinusOne(leftBits);
            double[] rightVector = asPlusMinusOne(rightBits);

            double squared = 0;
            double absolute = 0;
            double dot = 0;
            for (int i = 0; i < dimension; i++) {
                double difference = leftVector[i] - rightVector[i];
                squared += difference * difference;
                absolute += Math.abs(difference);
                dot += leftVector[i] * rightVector[i];
            }

            assertThat(BinaryVectorMath.euclideanSquared(left, right)).as("d=%s squared", dimension).isEqualTo(squared);
            assertThat(BinaryVectorMath.euclidean(left, right)).as("d=%s euclidean", dimension)
                    .isCloseTo(Math.sqrt(squared), within(1e-12));
            assertThat(BinaryVectorMath.manhattan(left, right)).as("d=%s manhattan", dimension).isEqualTo(absolute);
            assertThat(BinaryVectorMath.dotProduct(left, right)).as("d=%s dot", dimension).isEqualTo(dot);
            assertThat(BinaryVectorMath.cosineSimilarity(left, right)).as("d=%s cosine", dimension)
                    .isCloseTo(dot / dimension, within(1e-12));
        }
    }

    /**
     * Cosine reads its own dimension to divide by it, which it must not do before the two operands
     * have been compared: a mismatch is a mismatch, not a vector without a magnitude, and every
     * other binary metric reports it as one.
     */
    @Test
    public void testCosineReportsALengthMismatchRatherThanAZeroMagnitude()
    {
        Slice first = pack(new boolean[] {true, false, true, true});
        Slice second = pack(new boolean[] {true, false});
        assertThatThrownBy(() -> BinaryVectorMath.cosineSimilarity(first, second))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("same length");
    }

    @Test
    public void testAZeroDimensionalVectorHasNoMagnitude()
    {
        Slice empty = pack(new boolean[0]);
        assertThatThrownBy(() -> BinaryVectorMath.cosineSimilarity(empty, empty))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("magnitude");
    }

    /**
     * Every metric here is a monotone function of one Hamming count, so they all induce the same
     * ranking. A ranking that differs between two of them is a bug that needs no reference
     * implementation to detect.
     */
    @Test
    public void testAllMetricsInduceTheSameRanking()
    {
        SplittableRandom random = new SplittableRandom(9);
        int dimension = 128;
        Slice query = pack(randomBits(random, dimension));
        Slice[] candidates = new Slice[50];
        for (int i = 0; i < candidates.length; i++) {
            candidates[i] = pack(randomBits(random, dimension));
        }

        Integer[] byEuclidean = order(candidates, candidate -> BinaryVectorMath.euclidean(query, candidate));
        assertThat(order(candidates, candidate -> BinaryVectorMath.euclideanSquared(query, candidate)))
                .isEqualTo(byEuclidean);
        assertThat(order(candidates, candidate -> BinaryVectorMath.manhattan(query, candidate)))
                .isEqualTo(byEuclidean);
        assertThat(order(candidates, candidate -> -BinaryVectorMath.dotProduct(query, candidate)))
                .isEqualTo(byEuclidean);
        assertThat(order(candidates, candidate -> -BinaryVectorMath.cosineSimilarity(query, candidate)))
                .isEqualTo(byEuclidean);
    }

    private static Integer[] order(Slice[] candidates, ToDoubleFunction<Slice> score)
    {
        Integer[] indexes = new Integer[candidates.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, Comparator
                .comparingDouble((Integer i) -> score.applyAsDouble(candidates[i]))
                .thenComparingInt(i -> i));
        return indexes;
    }
}
