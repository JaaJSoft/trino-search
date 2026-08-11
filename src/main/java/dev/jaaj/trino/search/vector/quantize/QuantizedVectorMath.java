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
}
