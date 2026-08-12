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
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.TinyintType.TINYINT;

/**
 * Distance kernels over two vectors of one-byte codes fitted against the same bounds.
 * <p>
 * Both operands carry the same per-dimension offset, so it cancels wherever the components appear
 * as a difference: euclidean_squared = scale^2 * sum(diff^2) and manhattan = scale * sum(|diff|).
 * Both inner sums are pure integer arithmetic over the raw codes, with the scale applied once at
 * the end rather than loaded on every component, and euclidean and manhattan therefore never read
 * the offsets at all. Dot product and cosine do read them, because a product of two dequantised
 * components leaves cross terms in the offset that do not cancel.
 */
public final class QuantizedVectorMath
{
    private static final int UNROLL = 4;

    private static final int CHECK_STRIDE = 64;

    private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;

    /**
     * How many int vectors one byte vector's worth of components widens into. Both species are the
     * preferred ones, so they share a shape and this is the ratio of their element sizes.
     */
    private static final int WIDENING_PARTS = BYTE_SPECIES.length() / INT_SPECIES.length();

    /**
     * How many components' worth of terms an {@code IntVector} accumulator may absorb before it is
     * folded into the {@code long} running total. {@link VectorOperators#ADD} reduceLanes sums every
     * lane into one scalar {@code int}, so what has to stay under {@code Integer.MAX_VALUE} is the
     * total across all lanes together, not any one lane on its own: at most this many components,
     * each contributing a term of at most {@code 255 * 255 = 65025} for a squared difference. This
     * is chosen well under {@code Integer.MAX_VALUE / 65025}, which is roughly 33025, to leave
     * headroom for the handful of extra components one more widened byte vector can add past the
     * checkpoint before the loop notices.
     */
    private static final int FLUSH_COMPONENTS = 16384;

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
        double scale = bounds.scale();
        double limitInCodeUnits = codeUnitLimit(limit, scale);
        // Only a plain ByteArrayBlock stores a vector's components contiguously and in order. A
        // dictionary block maps position i somewhere else in a shared array and a run-length block
        // holds a single component, so indexing their backing arrays would compute a distance
        // between vectors nobody asked for.
        if (first instanceof ByteArrayBlock left
                && second instanceof ByteArrayBlock right
                && !left.mayHaveNull()
                && !right.mayHaveNull()) {
            return scale * scale * euclideanSquaredCodeUnitsVectorized(left, right, limitInCodeUnits);
        }
        return scale * scale * euclideanSquaredCodeUnitsUnrolled(first, second, limitInCodeUnits);
    }

    /**
     * Converts a limit expressed in the caller's real distance-squared units into the pure integer
     * code-difference units the loop below accumulates, so the running total and the limit are
     * compared in one consistent space: {@code euclidean_squared = scale^2 * sum}, so
     * {@code sum = limit / scale^2}. A zero scale is legal: a corpus with one distinct value fits
     * every code to zero, so the loop below always finds a total of zero and multiplying it by a
     * scale of zero always lands on the correct answer of zero anyway, without ever needing to
     * abandon early to get there. Converting would divide by that zero, so this disables the early
     * return instead of doing so.
     */
    private static double codeUnitLimit(double limit, double scale)
    {
        if (scale == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return limit / (scale * scale);
    }

    /**
     * One byte vector holds {@link #WIDENING_PARTS} int vectors' worth of components, so each load
     * feeds that many parts through the same accumulator. The offsets are not read at all: both
     * operands carry the same one and it cancels in the difference, and the scale is left for the
     * caller to apply once rather than loaded and multiplied on every component.
     * <p>
     * The codes are widened to int lanes before they are subtracted. Widening first and subtracting
     * after is what keeps the arithmetic exact: a difference of two values in {@code [-128, 127]}
     * does not fit back into a byte, so computing it in byte lanes would wrap silently instead of
     * producing a value like 255.
     * <p>
     * The {@code IntVector} accumulator is periodically folded into a {@code long} running total,
     * both to answer the caller's limit and to stay within {@link #FLUSH_COMPONENTS} of its own
     * 32-bit lanes: a squared difference is at most {@code 255 * 255 = 65025}, small individually,
     * but a dimension with no cap on it would otherwise wrap an {@code int} lane silently and could
     * turn a distant vector into the nearest neighbour.
     */
    private static double euclideanSquaredCodeUnitsVectorized(ByteArrayBlock first, ByteArrayBlock second, double limit)
    {
        byte[] leftCodes = first.getRawValues();
        byte[] rightCodes = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        IntVector sum = IntVector.zero(INT_SPECIES);
        long total = 0;
        int lanes = BYTE_SPECIES.length();
        int vectorized = BYTE_SPECIES.loopBound(length);
        int stride = Math.min(checkStride(lanes, vectorized, limit), FLUSH_COMPONENTS);
        int i = 0;
        while (i < vectorized) {
            int checkpoint = Math.min(i + stride, vectorized);
            for (; i < checkpoint; i += lanes) {
                ByteVector left = ByteVector.fromArray(BYTE_SPECIES, leftCodes, leftBase + i);
                ByteVector right = ByteVector.fromArray(BYTE_SPECIES, rightCodes, rightBase + i);
                // Part p of a widened byte vector covers INT_SPECIES.length() consecutive
                // components starting at offset p * INT_SPECIES.length() within the byte vector's
                // own lanes, exactly as VectorMath's array(real) kernel widens an int vector with
                // F2D.
                for (int part = 0; part < WIDENING_PARTS; part++) {
                    IntVector difference = ((IntVector) left.convertShape(VectorOperators.B2I, INT_SPECIES, part))
                            .sub((IntVector) right.convertShape(VectorOperators.B2I, INT_SPECIES, part));
                    sum = difference.mul(difference).add(sum);
                }
            }
            total += sum.reduceLanes(VectorOperators.ADD);
            sum = IntVector.zero(INT_SPECIES);
            if (total > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        for (; i < length; i++) {
            int difference = leftCodes[leftBase + i] - rightCodes[rightBase + i];
            total += (long) difference * difference;
        }
        return total;
    }

    private static double euclideanSquaredCodeUnitsUnrolled(Block first, Block second, double limit)
    {
        int length = first.getPositionCount();
        long sum0 = 0;
        long sum1 = 0;
        long sum2 = 0;
        long sum3 = 0;

        int unrolled = length - (length % UNROLL);
        int stride = checkStride(UNROLL, unrolled, limit);
        int i = 0;
        while (i < unrolled) {
            int checkpoint = Math.min(i + stride, unrolled);
            for (; i < checkpoint; i += UNROLL) {
                long difference0 = codeDifference(first, second, i);
                long difference1 = codeDifference(first, second, i + 1);
                long difference2 = codeDifference(first, second, i + 2);
                long difference3 = codeDifference(first, second, i + 3);
                sum0 += difference0 * difference0;
                sum1 += difference1 * difference1;
                sum2 += difference2 * difference2;
                sum3 += difference3 * difference3;
            }
            if ((sum0 + sum1) + (sum2 + sum3) > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        long sum = (sum0 + sum1) + (sum2 + sum3);
        for (; i < length; i++) {
            long difference = codeDifference(first, second, i);
            sum += difference * difference;
        }
        return sum;
    }

    /**
     * The fitted scale is never negative, since it comes from {@code (max - min) / CODE_LEVELS}
     * over a corpus, but bounds may equally arrive as a user-written row that nothing here
     * validates. Multiplying by its absolute value rather than the scale itself is what keeps a
     * negative one from turning a distance into a negative number: {@code decode(i, code) = offset
     * + code * scale} still expands to the same real components either way, so the reference this
     * kernel has to agree with, dequantise-then-compute, sums {@code |scale * diff|}, not
     * {@code scale * |diff|}.
     */
    public static double manhattan(Block first, Block second, QuantizationBounds bounds)
    {
        double scale = Math.abs(bounds.scale());
        if (first instanceof ByteArrayBlock left
                && second instanceof ByteArrayBlock right
                && !left.mayHaveNull()
                && !right.mayHaveNull()) {
            return scale * manhattanCodeUnitsVectorized(left, right);
        }
        return scale * manhattanCodeUnitsUnrolled(first, second);
    }

    /**
     * Same shape as {@link #euclideanSquaredCodeUnitsVectorized}, but the term is an absolute
     * difference rather than a squared one: smaller individually, at most 255, so the flush
     * interval sized against euclidean's larger term is more than safe here too.
     */
    private static long manhattanCodeUnitsVectorized(ByteArrayBlock first, ByteArrayBlock second)
    {
        byte[] leftCodes = first.getRawValues();
        byte[] rightCodes = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        IntVector sum = IntVector.zero(INT_SPECIES);
        long total = 0;
        int lanes = BYTE_SPECIES.length();
        int vectorized = BYTE_SPECIES.loopBound(length);
        int stride = FLUSH_COMPONENTS;
        int i = 0;
        while (i < vectorized) {
            int checkpoint = Math.min(i + stride, vectorized);
            for (; i < checkpoint; i += lanes) {
                ByteVector left = ByteVector.fromArray(BYTE_SPECIES, leftCodes, leftBase + i);
                ByteVector right = ByteVector.fromArray(BYTE_SPECIES, rightCodes, rightBase + i);
                for (int part = 0; part < WIDENING_PARTS; part++) {
                    IntVector difference = ((IntVector) left.convertShape(VectorOperators.B2I, INT_SPECIES, part))
                            .sub((IntVector) right.convertShape(VectorOperators.B2I, INT_SPECIES, part));
                    sum = difference.abs().add(sum);
                }
            }
            total += sum.reduceLanes(VectorOperators.ADD);
            sum = IntVector.zero(INT_SPECIES);
        }

        for (; i < length; i++) {
            total += Math.abs(leftCodes[leftBase + i] - rightCodes[rightBase + i]);
        }
        return total;
    }

    private static long manhattanCodeUnitsUnrolled(Block first, Block second)
    {
        long total = 0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            total += Math.abs(codeDifference(first, second, i));
        }
        return total;
    }

    /**
     * The code difference, computed in int, where it is exact. Reading two dequantised doubles and
     * subtracting them instead would add a rounding of the offset to every component for no gain,
     * since the offset cancels.
     */
    private static int codeDifference(Block first, Block second, int i)
    {
        return TINYINT.getByte(first, i) - TINYINT.getByte(second, i);
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
     * Both magnitudes and the dot product in one pass over the codes. None of the overflow
     * rescaling the float kernels carry is reproduced here: a code cannot exceed 127 in magnitude,
     * so a dequantised component is only as large as the bounds it was fitted against, and bounds
     * large enough to overflow a sum of squares describe a corpus the float kernels could not have
     * held either. Nothing validates the bounds, which come from a user-written row or from
     * {@code vector_bounds_agg} over a corpus of the caller's choosing, so bounds carrying an
     * infinite or NaN scale propagate into the result rather than being rescaled away. The only
     * case rejected outright is a vector that dequantises to all zeros.
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
