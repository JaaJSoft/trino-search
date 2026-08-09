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
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

final class VectorMath
{
    /**
     * Enough independent accumulators to cover the latency of a floating point add, which is three
     * to four cycles on current cores against one cycle of throughput.
     */
    private static final int UNROLL = 4;

    /**
     * {@code jdk.incubator.vector} is an incubator module, absent unless the JVM was started with
     * {@code --add-modules}. Nothing here guards against that, because a Trino server cannot start
     * without it either: {@code io.trino.simd.BlockEncodingSimdSupport} names the same classes in
     * its own class body, so a JVM able to run the engine is a JVM able to run this.
     */
    private static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;

    /**
     * How many double vectors one int vector's worth of components widens into. Both species are
     * the preferred ones, so they share a shape and this is the ratio of their element sizes.
     */
    private static final int WIDENING_PARTS = INT_SPECIES.length() / DOUBLE_SPECIES.length();

    /**
     * How many components are accumulated between two checks against a caller's limit. Small
     * enough that a hopeless candidate is abandoned early in a long vector, large enough that the
     * lane reduction each check costs is lost among the multiply-adds it guards.
     */
    private static final int CHECK_STRIDE = 64;

    private VectorMath() {}

    /**
     * How many components to accumulate between checks. A caller with no limit still goes through
     * the same loop, so it is given a stride covering the whole vector: it then pays one comparison
     * for the vector instead of one per {@link #CHECK_STRIDE} components. Measured at dimension
     * 768, checking throughout cost the unbounded kernels about fifteen percent, which is more than
     * a branch nobody needs is worth.
     */
    private static int checkStride(int step, int end, double limit)
    {
        if (limit == Double.POSITIVE_INFINITY) {
            return Math.max(step, end);
        }
        return Math.max(step, CHECK_STRIDE - (CHECK_STRIDE % step));
    }

    static void checkSameLength(Block first, Block second)
    {
        if (first.getPositionCount() != second.getPositionCount()) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }
    }

    static boolean hasNulls(Block first, Block second)
    {
        return first.hasNull() || second.hasNull();
    }

    /**
     * Four partial sums rather than one. A single accumulator makes every iteration wait for the
     * previous addition to retire, which pins the loop to the latency of one floating point add
     * however much of the processor sits idle; independent sums let those additions overlap.
     * <p>
     * The trade is that the components are added in a different order, so a result can differ from
     * the textbook left-to-right sum in its last bits. That is a change of rounding, not of
     * accuracy: the pairwise shape below is if anything the better-conditioned of the two.
     */
    static double euclideanSquared(Block first, Block second, VectorReader reader)
    {
        return euclideanSquaredBounded(first, second, reader, Double.POSITIVE_INFINITY);
    }

    /**
     * The squared distance, or some value above {@code limit} once the components read so far
     * already put it there. Every term is a square and so non-negative, which is what makes giving
     * up sound: a partial sum past the limit can never come back under it, so the components left
     * unread cannot change the answer to the only question a caller passing a limit is asking.
     * <p>
     * An infinite limit is how the unbounded form is expressed, rather than by a second set of
     * kernels: the comparison it leaves in the loop runs once per {@link #CHECK_STRIDE} components
     * and is never true.
     */
    static double euclideanSquaredBounded(Block first, Block second, VectorReader reader, double limit)
    {
        // Only a plain LongArrayBlock stores a vector's components contiguously and in order. A
        // dictionary block maps position i somewhere else in a shared array and a run-length block
        // holds a single component, so indexing their backing arrays would compute a distance
        // between vectors nobody asked for. Comparing the reader by identity is what ties the
        // longs below to double bits rather than to some other fixed-width type.
        if (reader == VectorReader.DOUBLE_READER
                && first instanceof LongArrayBlock left
                && second instanceof LongArrayBlock right
                && !left.mayHaveNull()
                && !right.mayHaveNull()) {
            return euclideanSquaredVectorized(left, right, limit);
        }
        // Same reasoning for array(real), whose components are float bits in an int array. The
        // arithmetic still happens in double, so the lanes are widened rather than kept as floats:
        // accumulating in float would answer a different question than every other path here.
        if (reader == VectorReader.REAL_READER
                && first instanceof IntArrayBlock left
                && second instanceof IntArrayBlock right
                && !left.mayHaveNull()
                && !right.mayHaveNull()) {
            return euclideanSquaredVectorized(left, right, limit);
        }
        return euclideanSquaredUnrolled(first, second, reader, limit);
    }

    /**
     * The raw longs already hold {@code Double.doubleToLongBits}, so reinterpreting the lanes
     * costs nothing: the components move from the block's array into a vector register without a
     * copy and without a conversion instruction.
     */
    private static double euclideanSquaredVectorized(LongArrayBlock first, LongArrayBlock second, double limit)
    {
        long[] leftBits = first.getRawValues();
        long[] rightBits = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        DoubleVector sum = DoubleVector.zero(DoubleVector.SPECIES_PREFERRED);
        int lanes = LONG_SPECIES.length();
        int vectorized = LONG_SPECIES.loopBound(length);
        int stride = checkStride(lanes, vectorized, limit);
        int i = 0;
        while (i < vectorized) {
            int checkpoint = Math.min(i + stride, vectorized);
            for (; i < checkpoint; i += lanes) {
                DoubleVector left = LongVector.fromArray(LONG_SPECIES, leftBits, leftBase + i).reinterpretAsDoubles();
                DoubleVector right = LongVector.fromArray(LONG_SPECIES, rightBits, rightBase + i).reinterpretAsDoubles();
                DoubleVector difference = left.sub(right);
                sum = difference.fma(difference, sum);
            }
            if (sum.reduceLanes(VectorOperators.ADD) > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            double difference = Double.longBitsToDouble(leftBits[leftBase + i])
                    - Double.longBitsToDouble(rightBits[rightBase + i]);
            total += difference * difference;
        }
        return total;
    }

    /**
     * One int vector holds twice the components a double vector does, so each load feeds
     * {@link #WIDENING_PARTS} halves through the same accumulator. Half the memory traffic of the
     * double path for the same amount of arithmetic.
     */
    private static double euclideanSquaredVectorized(IntArrayBlock first, IntArrayBlock second, double limit)
    {
        int[] leftBits = first.getRawValues();
        int[] rightBits = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        DoubleVector sum = DoubleVector.zero(DOUBLE_SPECIES);
        int lanes = INT_SPECIES.length();
        int vectorized = INT_SPECIES.loopBound(length);
        int stride = checkStride(lanes, vectorized, limit);
        int i = 0;
        while (i < vectorized) {
            int checkpoint = Math.min(i + stride, vectorized);
            for (; i < checkpoint; i += lanes) {
                FloatVector left = IntVector.fromArray(INT_SPECIES, leftBits, leftBase + i).reinterpretAsFloats();
                FloatVector right = IntVector.fromArray(INT_SPECIES, rightBits, rightBase + i).reinterpretAsFloats();
                for (int part = 0; part < WIDENING_PARTS; part++) {
                    DoubleVector difference = ((DoubleVector) left.convertShape(VectorOperators.F2D, DOUBLE_SPECIES, part))
                            .sub((DoubleVector) right.convertShape(VectorOperators.F2D, DOUBLE_SPECIES, part));
                    sum = difference.fma(difference, sum);
                }
            }
            if (sum.reduceLanes(VectorOperators.ADD) > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            // Both casts are load-bearing: without them this subtracts in float, which is neither
            // what the lanes above do nor what VectorReader.read gives every other path.
            double difference = (double) Float.intBitsToFloat(leftBits[leftBase + i])
                    - (double) Float.intBitsToFloat(rightBits[rightBase + i]);
            total += difference * difference;
        }
        return total;
    }

    private static double euclideanSquaredUnrolled(Block first, Block second, VectorReader reader, double limit)
    {
        int length = first.getPositionCount();
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;

        int unrolled = length - (length % UNROLL);
        int stride = checkStride(UNROLL, unrolled, limit);
        int i = 0;
        while (i < unrolled) {
            int checkpoint = Math.min(i + stride, unrolled);
            for (; i < checkpoint; i += UNROLL) {
                double difference0 = reader.read(first, i) - reader.read(second, i);
                double difference1 = reader.read(first, i + 1) - reader.read(second, i + 1);
                double difference2 = reader.read(first, i + 2) - reader.read(second, i + 2);
                double difference3 = reader.read(first, i + 3) - reader.read(second, i + 3);
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
            double difference = reader.read(first, i) - reader.read(second, i);
            sum += difference * difference;
        }
        return sum;
    }

    static double euclidean(Block first, Block second, VectorReader reader)
    {
        return Math.sqrt(euclideanSquared(first, second, reader));
    }

    static double manhattan(Block first, Block second, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            sum += Math.abs(reader.read(first, i) - reader.read(second, i));
        }
        return sum;
    }

    /**
     * There is no bounded form. Every term is a product of two signed components, so a partial sum
     * says nothing about the total and no candidate can be abandoned before its last component.
     * {@link Metric#DOT_PRODUCT} keeps the default {@code computeBounded} for that reason, which is
     * why nothing below carries a limit or pays for a check.
     * <p>
     * The two fast paths are guarded exactly as in {@link #euclideanSquaredBounded}, and for the
     * reasons given there: only a plain array block stores a vector's components contiguously and
     * in order, and the reader identity is what ties the raw longs or ints to a floating point type
     * rather than to some other value of the same width.
     */
    static double dotProduct(Block first, Block second, VectorReader reader)
    {
        if (reader == VectorReader.DOUBLE_READER
                && first instanceof LongArrayBlock left
                && second instanceof LongArrayBlock right
                && !left.mayHaveNull()
                && !right.mayHaveNull()) {
            return dotProductVectorized(left, right);
        }
        if (reader == VectorReader.REAL_READER
                && first instanceof IntArrayBlock left
                && second instanceof IntArrayBlock right
                && !left.mayHaveNull()
                && !right.mayHaveNull()) {
            return dotProductVectorized(left, right);
        }
        return dotProductUnrolled(first, second, reader);
    }

    private static double dotProductVectorized(LongArrayBlock first, LongArrayBlock second)
    {
        long[] leftBits = first.getRawValues();
        long[] rightBits = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        DoubleVector sum = DoubleVector.zero(DOUBLE_SPECIES);
        int lanes = LONG_SPECIES.length();
        int vectorized = LONG_SPECIES.loopBound(length);
        int i = 0;
        for (; i < vectorized; i += lanes) {
            DoubleVector left = LongVector.fromArray(LONG_SPECIES, leftBits, leftBase + i).reinterpretAsDoubles();
            DoubleVector right = LongVector.fromArray(LONG_SPECIES, rightBits, rightBase + i).reinterpretAsDoubles();
            sum = left.fma(right, sum);
        }

        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            total += Double.longBitsToDouble(leftBits[leftBase + i])
                    * Double.longBitsToDouble(rightBits[rightBase + i]);
        }
        return total;
    }

    private static double dotProductVectorized(IntArrayBlock first, IntArrayBlock second)
    {
        int[] leftBits = first.getRawValues();
        int[] rightBits = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        DoubleVector sum = DoubleVector.zero(DOUBLE_SPECIES);
        int lanes = INT_SPECIES.length();
        int vectorized = INT_SPECIES.loopBound(length);
        int i = 0;
        for (; i < vectorized; i += lanes) {
            FloatVector left = IntVector.fromArray(INT_SPECIES, leftBits, leftBase + i).reinterpretAsFloats();
            FloatVector right = IntVector.fromArray(INT_SPECIES, rightBits, rightBase + i).reinterpretAsFloats();
            for (int part = 0; part < WIDENING_PARTS; part++) {
                sum = ((DoubleVector) left.convertShape(VectorOperators.F2D, DOUBLE_SPECIES, part))
                        .fma((DoubleVector) right.convertShape(VectorOperators.F2D, DOUBLE_SPECIES, part), sum);
            }
        }

        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            // Both casts are load-bearing: without them this multiplies in float, which is neither
            // what the lanes above do nor what VectorReader.read gives every other path.
            total += (double) Float.intBitsToFloat(leftBits[leftBase + i])
                    * (double) Float.intBitsToFloat(rightBits[rightBase + i]);
        }
        return total;
    }

    private static double dotProductUnrolled(Block first, Block second, VectorReader reader)
    {
        int length = first.getPositionCount();
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;

        int unrolled = length - (length % UNROLL);
        int i = 0;
        for (; i < unrolled; i += UNROLL) {
            sum0 += reader.read(first, i) * reader.read(second, i);
            sum1 += reader.read(first, i + 1) * reader.read(second, i + 1);
            sum2 += reader.read(first, i + 2) * reader.read(second, i + 2);
            sum3 += reader.read(first, i + 3) * reader.read(second, i + 3);
        }

        double sum = (sum0 + sum1) + (sum2 + sum3);
        for (; i < length; i++) {
            sum += reader.read(first, i) * reader.read(second, i);
        }
        return sum;
    }

    static double norm(Block vector, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < vector.getPositionCount(); i++) {
            double value = reader.read(vector, i);
            sum += value * value;
        }
        if (Double.isInfinite(sum) || sum == 0) {
            return scaledNorm(vector, reader);
        }
        return Math.sqrt(sum);
    }

    static double cosineSimilarity(Block first, Block second, VectorReader reader)
    {
        double firstMagnitude = 0.0;
        double secondMagnitude = 0.0;
        double dotProduct = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            double firstValue = reader.read(first, i);
            double secondValue = reader.read(second, i);
            firstMagnitude += firstValue * firstValue;
            secondMagnitude += secondValue * secondValue;
            dotProduct += firstValue * secondValue;
        }

        // A product that overflowed to infinity or underflowed to zero says nothing about the
        // vectors themselves, only about the accumulation. Rescale before concluding anything.
        // Splitting into sqrt(a) * sqrt(b) would avoid the overflow without a second pass, but
        // it loses the exactness that makes two identical vectors return exactly 1.
        // NaN covers an infinite magnitude meeting a zero one, which is neither of the above and
        // must still be reported as a zero magnitude rather than returned as NaN.
        double magnitudeProduct = firstMagnitude * secondMagnitude;
        if (magnitudeProduct == 0 || Double.isInfinite(magnitudeProduct) || Double.isNaN(magnitudeProduct)
                || Double.isInfinite(dotProduct)) {
            return scaledCosineSimilarity(first, second, reader);
        }
        return dotProduct / Math.sqrt(magnitudeProduct);
    }

    /**
     * Recomputes the norm after dividing out the largest component, for the rare vector whose
     * sum of squares overflows although the norm itself is representable.
     */
    private static double scaledNorm(Block vector, VectorReader reader)
    {
        double scale = maxAbsoluteValue(vector, reader);
        if (scale == 0 || Double.isInfinite(scale)) {
            return scale;
        }
        double sum = 0.0;
        for (int i = 0; i < vector.getPositionCount(); i++) {
            double value = reader.read(vector, i) / scale;
            sum += value * value;
        }
        return scale * Math.sqrt(sum);
    }

    /**
     * Cosine is invariant to scaling either vector, so each is divided by its own largest
     * component before the magnitudes are accumulated. This is also where a genuinely zero
     * vector is detected, since a zero magnitude alone cannot distinguish one from a vector
     * whose squares underflowed.
     */
    private static double scaledCosineSimilarity(Block first, Block second, VectorReader reader)
    {
        double firstScale = maxAbsoluteValue(first, reader);
        double secondScale = maxAbsoluteValue(second, reader);
        if (firstScale == 0 || secondScale == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }
        if (Double.isInfinite(firstScale) || Double.isInfinite(secondScale)) {
            return Double.NaN;
        }

        double firstMagnitude = 0.0;
        double secondMagnitude = 0.0;
        double dotProduct = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            double firstValue = reader.read(first, i) / firstScale;
            double secondValue = reader.read(second, i) / secondScale;
            firstMagnitude += firstValue * firstValue;
            secondMagnitude += secondValue * secondValue;
            dotProduct += firstValue * secondValue;
        }
        return dotProduct / (Math.sqrt(firstMagnitude) * Math.sqrt(secondMagnitude));
    }

    private static double maxAbsoluteValue(Block vector, VectorReader reader)
    {
        double max = 0.0;
        for (int i = 0; i < vector.getPositionCount(); i++) {
            max = Math.max(max, Math.abs(reader.read(vector, i)));
        }
        return max;
    }
}
