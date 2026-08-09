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
import io.trino.spi.block.LongArrayBlock;
import jdk.incubator.vector.DoubleVector;
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

    private VectorMath() {}

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
            return euclideanSquaredVectorized(left, right);
        }
        return euclideanSquaredUnrolled(first, second, reader);
    }

    /**
     * The raw longs already hold {@code Double.doubleToLongBits}, so reinterpreting the lanes
     * costs nothing: the components move from the block's array into a vector register without a
     * copy and without a conversion instruction.
     */
    private static double euclideanSquaredVectorized(LongArrayBlock first, LongArrayBlock second)
    {
        long[] leftBits = first.getRawValues();
        long[] rightBits = second.getRawValues();
        int leftBase = first.getRawValuesOffset();
        int rightBase = second.getRawValuesOffset();
        int length = first.getPositionCount();

        DoubleVector sum = DoubleVector.zero(DoubleVector.SPECIES_PREFERRED);
        int vectorized = LONG_SPECIES.loopBound(length);
        int i = 0;
        for (; i < vectorized; i += LONG_SPECIES.length()) {
            DoubleVector left = LongVector.fromArray(LONG_SPECIES, leftBits, leftBase + i).reinterpretAsDoubles();
            DoubleVector right = LongVector.fromArray(LONG_SPECIES, rightBits, rightBase + i).reinterpretAsDoubles();
            DoubleVector difference = left.sub(right);
            sum = difference.fma(difference, sum);
        }

        double total = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            double difference = Double.longBitsToDouble(leftBits[leftBase + i])
                    - Double.longBitsToDouble(rightBits[rightBase + i]);
            total += difference * difference;
        }
        return total;
    }

    private static double euclideanSquaredUnrolled(Block first, Block second, VectorReader reader)
    {
        int length = first.getPositionCount();
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;

        int unrolled = length - (length % UNROLL);
        int i = 0;
        for (; i < unrolled; i += UNROLL) {
            double difference0 = reader.read(first, i) - reader.read(second, i);
            double difference1 = reader.read(first, i + 1) - reader.read(second, i + 1);
            double difference2 = reader.read(first, i + 2) - reader.read(second, i + 2);
            double difference3 = reader.read(first, i + 3) - reader.read(second, i + 3);
            sum0 += difference0 * difference0;
            sum1 += difference1 * difference1;
            sum2 += difference2 * difference2;
            sum3 += difference3 * difference3;
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

    static double dotProduct(Block first, Block second, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
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
