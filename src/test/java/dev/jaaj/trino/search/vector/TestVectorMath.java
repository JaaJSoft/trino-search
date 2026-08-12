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
import io.trino.spi.block.DictionaryBlock;
import io.trino.spi.block.RunLengthEncodedBlock;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestVectorMath
{
    /**
     * Two whole vectors of the step the kernels actually take on the running machine, plus one
     * component that cannot fill a third. A hardcoded length instead ties the test to an assumption
     * about lane count that it never states: on a machine whose preferred species is wider, the
     * loop bound falls to zero, the whole vector is read by the scalar remainder, and the test goes
     * on passing without ever reaching the lanes it exists to pin.
     * <p>
     * The double kernels step by the long species and the {@code array(real)} ones by the int
     * species, which is half as many components per step, so the two lengths differ.
     */
    private static final int DOUBLE_LANE_SPANNING_LENGTH = 2 * LongVector.SPECIES_PREFERRED.length() + 1;
    private static final int REAL_LANE_SPANNING_LENGTH = 2 * IntVector.SPECIES_PREFERRED.length() + 1;

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

    private static Block reals(Float... values)
    {
        BlockBuilder builder = REAL.createBlockBuilder(null, values.length);
        for (Float value : values) {
            if (value == null) {
                builder.appendNull();
            }
            else {
                REAL.writeFloat(builder, value);
            }
        }
        return builder.build();
    }

    /**
     * Components are read as floats and the arithmetic happens in double, so a fast path that
     * accumulated in float instead would still look right on values that are exact in float32.
     * 0.1 and 0.3 are not, and their widened difference is not the float difference, so this pins
     * where the widening happens.
     */
    @Test
    public void testRealVectorsWidenBeforeTheArithmetic()
    {
        double left = 0.1f;
        double right = 0.3f;
        double expected = (left - right) * (left - right);

        assertThat(VectorMath.euclideanSquared(reals(0.1f), reals(0.3f), REAL_READER))
                .isEqualTo(expected);
    }

    /**
     * The counterpart of {@link #testRealVectorsWidenBeforeTheArithmetic} for the vectorised body.
     * That one passes a single component, so the whole of it runs in the scalar remainder and the
     * widened lanes are never reached. This vector spans several whole steps of the wide loop with
     * components left over for the remainder, so both are covered.
     * <p>
     * Operands within a factor of two of each other subtract exactly in float, by Sterbenz, which
     * hides the mistake this pins: the long region test above uses whole numbers and would pass
     * against lanes that subtracted as floats and widened afterwards. A constant ratio of three
     * keeps every pair outside that range.
     * <p>
     * The tolerance separates the two orders of magnitude that matter and nothing finer. Lanes
     * reduced four at a time do not sum in the same order as the reference loop below, which costs
     * a couple of last bits; subtracting in float instead of double costs seven decimal digits. A
     * bound tight enough to reject the ordering would be asserting a bit-exact result that
     * {@link VectorMath#euclideanSquared} does not promise, and would break on the next machine
     * with a different lane count.
     */
    @Test
    public void testTheVectorisedRealBodyWidensBeforeSubtracting()
    {
        int length = REAL_LANE_SPANNING_LENGTH;
        Float[] left = new Float[length];
        Float[] right = new Float[length];
        for (int i = 0; i < length; i++) {
            left[i] = 0.1f * (i + 1);
            right[i] = 0.3f * (i + 1);
        }

        double expected = 0;
        for (int i = 0; i < length; i++) {
            double difference = (double) (float) (0.1f * (i + 1)) - (double) (float) (0.3f * (i + 1));
            expected += difference * difference;
        }

        assertThat(VectorMath.euclideanSquared(reals(left), reals(right), REAL_READER))
                .isCloseTo(expected, within(1e-9));
    }

    @Test
    public void testADictionaryEncodedRealVectorIsReadThroughItsPositions()
    {
        Block dictionary = reals(1.0f, 2.0f, 3.0f, 4.0f);
        Block vector = DictionaryBlock.create(2, dictionary, new int[] {3, 2});

        assertThat(VectorMath.euclideanSquared(vector, reals(0.0f, 0.0f), REAL_READER))
                .isCloseTo(25.0, within(1e-9));
    }

    @Test
    public void testARunLengthEncodedRealVectorIsReadThroughItsPositions()
    {
        Block vector = RunLengthEncodedBlock.create(reals(2.0f), 3);

        assertThat(VectorMath.euclideanSquared(vector, reals(0.0f, 0.0f, 0.0f), REAL_READER))
                .isCloseTo(12.0, within(1e-9));
    }

    /**
     * A length that is not a whole number of vector lanes, read from a block whose components do
     * not start at the beginning of its backing array: the two mistakes a raw-array fast path makes
     * are dropping the tail and ignoring the offset, and this fails on either.
     */
    @Test
    public void testALongRealRegionIsReadFromItsOwnOffsetThroughout()
    {
        int length = REAL_LANE_SPANNING_LENGTH;
        Float[] backing = new Float[length + 5];
        for (int i = 0; i < backing.length; i++) {
            backing[i] = (float) i;
        }
        Block region = reals(backing).getRegion(5, length);

        Float[] expected = new Float[length];
        for (int i = 0; i < length; i++) {
            expected[i] = (float) (i + 5);
        }

        assertThat(VectorMath.euclideanSquared(region, reals(expected), REAL_READER))
                .isEqualTo(0.0);

        Float[] zeroes = new Float[length];
        java.util.Arrays.fill(zeroes, 0.0f);
        double sumOfSquares = 0;
        for (int i = 0; i < length; i++) {
            sumOfSquares += (double) (i + 5) * (i + 5);
        }
        assertThat(VectorMath.euclideanSquared(region, reals(zeroes), REAL_READER))
                .isCloseTo(sumOfSquares, within(1e-9));
    }

    @Test
    public void testBoundedEuclideanSquaredIsExactWhenTheCandidateBeatsTheLimit()
    {
        assertThat(VectorMath.euclideanSquaredBounded(doubles(0.0, 0.0), doubles(3.0, 4.0), DOUBLE_READER, 26.0))
                .isCloseTo(25.0, within(1e-12));
    }

    /**
     * The limit is passed once the excess is already spent, several checks before the end of the
     * vector: what this pins is that giving up early still reports a value the caller cannot
     * mistake for a winning one, and that the components never read do not have to be zero for
     * that to hold.
     */
    @Test
    public void testBoundedEuclideanSquaredGivesUpOnALongVectorWithItsExcessUpFront()
    {
        int length = 512;
        Double[] far = new Double[length];
        Double[] origin = new Double[length];
        java.util.Arrays.fill(far, 1.0);
        java.util.Arrays.fill(origin, 0.0);

        assertThat(VectorMath.euclideanSquaredBounded(doubles(origin), doubles(far), DOUBLE_READER, 10.0))
                .isGreaterThan(10.0);
    }

    @Test
    public void testBoundedEuclideanSquaredIsExactAcrossManyChecks()
    {
        int length = 512;
        Double[] ones = new Double[length];
        Double[] origin = new Double[length];
        java.util.Arrays.fill(ones, 1.0);
        java.util.Arrays.fill(origin, 0.0);

        assertThat(VectorMath.euclideanSquaredBounded(doubles(origin), doubles(ones), DOUBLE_READER, 1e9))
                .isCloseTo(length, within(1e-9));
    }

    @Test
    public void testBoundedEuclideanSquaredOnRealsIsExactWhenItBeatsTheLimit()
    {
        assertThat(VectorMath.euclideanSquaredBounded(reals(0.0f, 0.0f), reals(3.0f, 4.0f), REAL_READER, 26.0))
                .isCloseTo(25.0, within(1e-9));
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

    /**
     * The counterpart of {@link #testTheVectorisedRealBodyWidensBeforeSubtracting} for the dot
     * product: components are read as floats and multiplied in double, so a body that multiplied
     * the float lanes and widened the product afterwards would answer a different question.
     * <p>
     * Whole numbers would hide it, since their products are exact in float. A tenth and three
     * tenths are not, and neither is their product, so every term here separates the two orders.
     */
    @Test
    public void testTheVectorisedRealDotProductWidensBeforeMultiplying()
    {
        int length = REAL_LANE_SPANNING_LENGTH;
        Float[] left = new Float[length];
        Float[] right = new Float[length];
        for (int i = 0; i < length; i++) {
            left[i] = 0.1f * (i + 1);
            right[i] = 0.3f * (i + 1);
        }

        double expected = 0;
        for (int i = 0; i < length; i++) {
            expected += (double) (float) (0.1f * (i + 1)) * (double) (float) (0.3f * (i + 1));
        }

        assertThat(VectorMath.dotProduct(reals(left), reals(right), REAL_READER))
                .isCloseTo(expected, within(1e-9));
    }

    /**
     * A vector spanning several whole steps of the wide loop with components left over for the
     * remainder, cut as a region so that it starts partway into a shared backing array. Dropping
     * the tail and reading from index zero are the two mistakes a raw-array fast path makes, and
     * each changes the answer here.
     */
    @Test
    public void testALongDotProductRegionIsReadFromItsOwnOffsetThroughout()
    {
        int length = DOUBLE_LANE_SPANNING_LENGTH;
        Double[] backing = new Double[length + 5];
        for (int i = 0; i < backing.length; i++) {
            backing[i] = (double) i;
        }
        Block region = doubles(backing).getRegion(5, length);

        Double[] ones = new Double[length];
        java.util.Arrays.fill(ones, 1.0);
        double sum = 0;
        double sumOfSquares = 0;
        Double[] same = new Double[length];
        for (int i = 0; i < length; i++) {
            same[i] = (double) (i + 5);
            sum += i + 5;
            sumOfSquares += (double) (i + 5) * (i + 5);
        }

        assertThat(VectorMath.dotProduct(region, doubles(ones), DOUBLE_READER))
                .isCloseTo(sum, within(1e-9));
        assertThat(VectorMath.dotProduct(region, doubles(same), DOUBLE_READER))
                .isCloseTo(sumOfSquares, within(1e-9));
    }

    @Test
    public void testALongRealDotProductRegionIsReadFromItsOwnOffsetThroughout()
    {
        int length = REAL_LANE_SPANNING_LENGTH;
        Float[] backing = new Float[length + 5];
        for (int i = 0; i < backing.length; i++) {
            backing[i] = (float) i;
        }
        Block region = reals(backing).getRegion(5, length);

        Float[] ones = new Float[length];
        java.util.Arrays.fill(ones, 1.0f);
        double sum = 0;
        double sumOfSquares = 0;
        Float[] same = new Float[length];
        for (int i = 0; i < length; i++) {
            same[i] = (float) (i + 5);
            sum += i + 5;
            sumOfSquares += (double) (i + 5) * (i + 5);
        }

        assertThat(VectorMath.dotProduct(region, reals(ones), REAL_READER))
                .isCloseTo(sum, within(1e-9));
        assertThat(VectorMath.dotProduct(region, reals(same), REAL_READER))
                .isCloseTo(sumOfSquares, within(1e-9));
    }

    @Test
    public void testADictionaryEncodedRealVectorDotProductIsReadThroughItsPositions()
    {
        Block dictionary = reals(9.0f, 3.0f, 4.0f, 7.0f);
        Block vector = DictionaryBlock.create(2, dictionary, new int[] {1, 2});

        assertThat(VectorMath.dotProduct(vector, reals(1.0f, 1.0f), REAL_READER))
                .isCloseTo(7.0, within(1e-9));
    }

    @Test
    public void testARunLengthEncodedRealVectorDotProductIsReadThroughItsPositions()
    {
        Block vector = RunLengthEncodedBlock.create(reals(2.0f), 3);

        assertThat(VectorMath.dotProduct(vector, reals(1.0f, 1.0f, 1.0f), REAL_READER))
                .isCloseTo(6.0, within(1e-9));
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
    public void testCosineSimilarityAgainstAZeroVectorWhoseCounterpartOverflows()
    {
        // an infinite magnitude times a zero one is NaN, which is neither zero nor infinite
        assertThatThrownBy(() -> VectorMath.cosineSimilarity(doubles(1e200), doubles(0.0), DOUBLE_READER))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("Vector magnitude cannot be zero");
        assertThatThrownBy(() -> VectorMath.cosineSimilarity(doubles(0.0), doubles(1e200), DOUBLE_READER))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("Vector magnitude cannot be zero");
    }

    @Test
    public void testNormWhenTheSumOfSquaresOverflows()
    {
        assertThat(VectorMath.norm(doubles(1e200, 1e200), DOUBLE_READER))
                .isCloseTo(Math.sqrt(2) * 1e200, within(1e188));
    }

    /**
     * A vector does not always arrive as a flat block of its own components. The engine hands over
     * whatever shape the page happened to have, and a dictionary block maps position i somewhere
     * else entirely in a shared backing array, so the components of one vector are neither
     * contiguous nor in order there.
     */
    @Test
    public void testADictionaryEncodedVectorIsReadThroughItsPositions()
    {
        Block dictionary = doubles(9.0, 3.0, 4.0, 7.0);
        // Selects 3.0 then 4.0, in an order and at offsets that do not match the backing array.
        Block vector = DictionaryBlock.create(2, dictionary, new int[] {1, 2});

        assertThat(VectorMath.euclideanSquared(vector, doubles(0.0, 0.0), DOUBLE_READER))
                .isCloseTo(25.0, within(1e-12));
        assertThat(VectorMath.dotProduct(vector, doubles(1.0, 1.0), DOUBLE_READER))
                .isCloseTo(7.0, within(1e-12));
    }

    /**
     * A run-length block carries one value and a count, so its backing array holds a single
     * component no matter how long the vector is.
     */
    @Test
    public void testARunLengthEncodedVectorIsReadThroughItsPositions()
    {
        Block vector = RunLengthEncodedBlock.create(doubles(2.0), 3);

        assertThat(VectorMath.euclideanSquared(vector, doubles(0.0, 0.0, 0.0), DOUBLE_READER))
                .isCloseTo(12.0, within(1e-12));
        assertThat(VectorMath.manhattan(vector, doubles(1.0, 1.0, 1.0), DOUBLE_READER))
                .isCloseTo(3.0, within(1e-12));
    }

    /**
     * A region shares the backing array of the block it was cut from and starts partway into it,
     * so reading that array from index zero would silently return a neighbouring vector.
     */
    @Test
    public void testAVectorTakenAsARegionStartsAtItsOwnOffset()
    {
        Block vector = doubles(100.0, 200.0, 3.0, 4.0).getRegion(2, 2);

        assertThat(VectorMath.euclideanSquared(vector, doubles(0.0, 0.0), DOUBLE_READER))
                .isCloseTo(25.0, within(1e-12));
        assertThat(VectorMath.dotProduct(vector, doubles(1.0, 1.0), DOUBLE_READER))
                .isCloseTo(7.0, within(1e-12));
    }

    /**
     * The region above is too short to reach a wide loop, which would leave the offset applied
     * only on the one-component-at-a-time tail. This one spans several whole steps of it, read in
     * whatever the widest step happens to be on the running machine, and every component differs
     * from its neighbour so that starting even one position early changes the answer.
     */
    @Test
    public void testALongRegionIsReadFromItsOwnOffsetThroughout()
    {
        int length = DOUBLE_LANE_SPANNING_LENGTH;
        Double[] backing = new Double[length + 5];
        for (int i = 0; i < backing.length; i++) {
            backing[i] = (double) i;
        }
        Block region = doubles(backing).getRegion(5, length);

        Double[] expected = new Double[length];
        for (int i = 0; i < length; i++) {
            expected[i] = (double) (i + 5);
        }

        assertThat(VectorMath.euclideanSquared(region, doubles(expected), DOUBLE_READER))
                .isEqualTo(0.0);

        Double[] zeroes = new Double[length];
        java.util.Arrays.fill(zeroes, 0.0);
        double sumOfSquares = 0;
        for (int i = 0; i < length; i++) {
            sumOfSquares += (double) (i + 5) * (i + 5);
        }
        assertThat(VectorMath.euclideanSquared(region, doubles(zeroes), DOUBLE_READER))
                .isCloseTo(sumOfSquares, within(1e-9));
    }
}
