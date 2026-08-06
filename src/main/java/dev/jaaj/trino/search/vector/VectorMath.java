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

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

final class VectorMath
{
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

    static double euclideanSquared(Block first, Block second, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
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
        double magnitudeProduct = firstMagnitude * secondMagnitude;
        if (magnitudeProduct == 0 || Double.isInfinite(magnitudeProduct) || Double.isInfinite(dotProduct)) {
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
