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

        if (firstMagnitude == 0 || secondMagnitude == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }
        return dotProduct / Math.sqrt(firstMagnitude * secondMagnitude);
    }
}
