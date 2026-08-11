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

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.ByteArrayBlock;
import io.trino.spi.block.LongArrayBlock;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.TinyintType.TINYINT;

/**
 * Distance kernels over two vectors of one-byte codes fitted against the same bounds.
 * <p>
 * Both operands carry the same per-dimension offset, so it cancels wherever the components appear
 * as a difference. Euclidean and manhattan therefore never read the offsets at all, and the code
 * difference is exact integer arithmetic before it is widened. Dot product and cosine do read
 * them, because a product of two dequantised components leaves cross terms in the offset that do
 * not cancel.
 */
public final class QuantizedVectorMath
{
    private static final int UNROLL = 4;

    private static final int CHECK_STRIDE = 64;

    private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;

    /**
     * How many double vectors one byte vector's worth of components widens into. Both species are
     * the preferred ones, so they share a shape and this is the ratio of their element sizes.
     */
    private static final int WIDENING_PARTS = BYTE_SPECIES.length() / DOUBLE_SPECIES.length();

    private QuantizedVectorMath() {}

    public static void checkSameLength(Block first, Block second, QuantizationBounds bounds)
    {
        if (first.getPositionCount() != second.getPositionCount()) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }
        bounds.checkDimension(first.getPositionCount());
    }

    public static double euclideanSquared(Block first, Block second, QuantizationBounds bounds)
    {
        return euclideanSquaredBounded(first, second, bounds, Double.POSITIVE_INFINITY);
    }

    public static double euclidean(Block first, Block second, QuantizationBounds bounds)
    {
        return Math.sqrt(euclideanSquared(first, second, bounds));
    }

    /**
     * The squared distance, or some value above {@code limit} once the components read so far
     * already put it there. Every term is a square and so non-negative, which is what makes giving
     * up sound.
     */
    public static double euclideanSquaredBounded(Block first, Block second, QuantizationBounds bounds, double limit)
    {
        // Only a plain ByteArrayBlock stores a vector's components contiguously and in order. A
        // dictionary block maps position i somewhere else in a shared array and a run-length block
        // holds a single component, so indexing their backing arrays would compute a distance
        // between vectors nobody asked for. The scales are read the same way, as the raw double
        // bits behind an array(double) element block.
        //
        // Nothing upstream checks the bounds against the vectors' length, only the two vectors
        // against each other: a short bounds array is caught by the scalar path's block accessor
        // throwing on an out-of-range position, so the fast path has to fail the same way rather
        // than reading past the end of the raw long[] it indexes directly.
        if (first instanceof ByteArrayBlock left
                && second instanceof ByteArrayBlock right
                && bounds.scales() instanceof LongArrayBlock scales
                && !left.mayHaveNull()
                && !right.mayHaveNull()
                && !scales.mayHaveNull()
                && scales.getPositionCount() >= first.getPositionCount()) {
            return euclideanSquaredVectorized(left, right, scales, limit);
        }
        return euclideanSquaredUnrolled(first, second, bounds, limit);
    }

    /**
     * One byte vector holds {@link #WIDENING_PARTS} double vectors' worth of components, so each
     * load feeds that many halves through the same accumulator. The offsets are not read at all:
     * both operands carry the same one and it cancels in the difference.
     * <p>
     * The codes are widened to double lanes before they are subtracted. Widening first and
     * subtracting after is what keeps the arithmetic exact: a difference of two values in
     * {@code [-128, 127]} does not fit back into a byte, so computing it in byte lanes would wrap
     * silently instead of producing a value like 255.
     */
    private static double euclideanSquaredVectorized(
            ByteArrayBlock first,
            ByteArrayBlock second,
            LongArrayBlock scaleBits,
            double limit)
    {
        byte[] leftCodes = first.getRawValues();
        byte[] rightCodes = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        long[] scales = scaleBits.getRawValues();
        int scaleBase = scaleBits.getRawValuesOffset();
        int length = first.getPositionCount();

        DoubleVector sum = DoubleVector.zero(DOUBLE_SPECIES);
        int lanes = BYTE_SPECIES.length();
        int vectorized = BYTE_SPECIES.loopBound(length);
        int stride = checkStride(lanes, vectorized, limit);
        int i = 0;
        while (i < vectorized) {
            int checkpoint = Math.min(i + stride, vectorized);
            for (; i < checkpoint; i += lanes) {
                ByteVector left = ByteVector.fromArray(BYTE_SPECIES, leftCodes, leftBase + i);
                ByteVector right = ByteVector.fromArray(BYTE_SPECIES, rightCodes, rightBase + i);
                // Part p of a widened byte vector covers DOUBLE_SPECIES.length() consecutive
                // components starting at offset p * DOUBLE_SPECIES.length() within the byte vector's
                // own lanes, exactly as VectorMath's array(real) kernel widens an int vector with
                // F2D. The matching scales sit at that same offset from the current position i, since
                // the scale array and the code arrays advance together one component at a time. Off
                // by one part here pairs every difference with the wrong dimension's scale while
                // still producing a plausible-looking number, which is why the per-dimension-scale
                // test exists.
                for (int part = 0; part < WIDENING_PARTS; part++) {
                    DoubleVector difference =
                            ((DoubleVector) left.convertShape(VectorOperators.B2D, DOUBLE_SPECIES, part))
                                    .sub((DoubleVector) right.convertShape(VectorOperators.B2D, DOUBLE_SPECIES, part));
                    DoubleVector scale = LongVector
                            .fromArray(LONG_SPECIES, scales, scaleBase + i + part * DOUBLE_SPECIES.length())
                            .reinterpretAsDoubles();
                    DoubleVector scaled = difference.mul(scale);
                    sum = scaled.fma(scaled, sum);
                }
            }
            if (sum.reduceLanes(VectorOperators.ADD) > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            double difference = (leftCodes[leftBase + i] - rightCodes[rightBase + i])
                    * Double.longBitsToDouble(scales[scaleBase + i]);
            total += difference * difference;
        }
        return total;
    }

    private static double euclideanSquaredUnrolled(Block first, Block second, QuantizationBounds bounds, double limit)
    {
        int length = first.getPositionCount();
        double sum0 = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;

        int unrolled = length - (length % UNROLL);
        int stride = checkStride(UNROLL, unrolled, limit);
        int i = 0;
        while (i < unrolled) {
            int checkpoint = Math.min(i + stride, unrolled);
            for (; i < checkpoint; i += UNROLL) {
                double difference0 = scaledDifference(first, second, bounds, i);
                double difference1 = scaledDifference(first, second, bounds, i + 1);
                double difference2 = scaledDifference(first, second, bounds, i + 2);
                double difference3 = scaledDifference(first, second, bounds, i + 3);
                sum0 += difference0 * difference0;
                sum1 += difference1 * difference1;
                sum2 += difference2 * difference2;
                sum3 += difference3 * difference3;
            }
            if ((sum0 + sum1) + (sum2 + sum3) > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        double sum = (sum0 + sum1) + (sum2 + sum3);
        for (; i < length; i++) {
            double difference = scaledDifference(first, second, bounds, i);
            sum += difference * difference;
        }
        return sum;
    }

    public static double manhattan(Block first, Block second, QuantizationBounds bounds)
    {
        double sum = 0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            sum += Math.abs(scaledDifference(first, second, bounds, i));
        }
        return sum;
    }

    /**
     * The code difference is computed in int, where it is exact, and only then scaled. Subtracting
     * two dequantised doubles instead would add a rounding of the offset to every component for no
     * gain, since the offset cancels.
     */
    private static double scaledDifference(Block first, Block second, QuantizationBounds bounds, int i)
    {
        return (TINYINT.getByte(first, i) - TINYINT.getByte(second, i)) * bounds.scale(i);
    }

    /**
     * A caller with no limit still goes through the same loop, so it is given a stride covering the
     * whole vector and pays one comparison rather than one per {@link #CHECK_STRIDE} components.
     */
    private static int checkStride(int step, int end, double limit)
    {
        if (limit == Double.POSITIVE_INFINITY) {
            return Math.max(step, end);
        }
        return Math.max(step, CHECK_STRIDE - (CHECK_STRIDE % step));
    }

    public static double dotProduct(Block first, Block second, QuantizationBounds bounds)
    {
        double sum = 0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            sum += bounds.decode(i, TINYINT.getByte(first, i)) * bounds.decode(i, TINYINT.getByte(second, i));
        }
        return sum;
    }

    /**
     * Both magnitudes and the dot product in one pass over the codes. The vectors here are bounded
     * by construction, since a code cannot exceed 127 and a scale is finite, so none of the
     * overflow rescaling the float kernels carry is reachable: the only degenerate case left is a
     * vector that dequantises to all zeros.
     */
    public static double cosineSimilarity(Block first, Block second, QuantizationBounds bounds)
    {
        double firstMagnitude = 0;
        double secondMagnitude = 0;
        double dotProduct = 0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            double firstValue = bounds.decode(i, TINYINT.getByte(first, i));
            double secondValue = bounds.decode(i, TINYINT.getByte(second, i));
            firstMagnitude += firstValue * firstValue;
            secondMagnitude += secondValue * secondValue;
            dotProduct += firstValue * secondValue;
        }
        if (firstMagnitude == 0 || secondMagnitude == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }
        return dotProduct / Math.sqrt(firstMagnitude * secondMagnitude);
    }
}
