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

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * Distances between two one-bit-per-component vectors.
 * <p>
 * The codes stand for a vector whose components are -1 and +1, so two components either agree or
 * differ by exactly 2. Every metric is therefore a closed form in the Hamming distance {@code h}
 * and there is one kernel rather than six: at dimension 768 that is twelve exclusive ors and
 * twelve population counts.
 */
public final class BinaryVectorMath
{
    private BinaryVectorMath() {}

    public static double euclideanSquared(Slice first, Slice second)
    {
        return 4.0 * BinaryCodes.hamming(first, second);
    }

    public static double euclidean(Slice first, Slice second)
    {
        return 2.0 * Math.sqrt(BinaryCodes.hamming(first, second));
    }

    public static double manhattan(Slice first, Slice second)
    {
        return 2.0 * BinaryCodes.hamming(first, second);
    }

    public static double dotProduct(Slice first, Slice second)
    {
        return BinaryCodes.dimension(first) - 2.0 * BinaryCodes.hamming(first, second);
    }

    /**
     * Every binary vector has the same magnitude, the square root of the dimension, since each of
     * its components is -1 or +1. The cosine is therefore the dot product over the dimension, and
     * the zero-magnitude case the float kernels guard against cannot arise: only a zero-dimensional
     * vector has no magnitude.
     */
    public static double cosineSimilarity(Slice first, Slice second)
    {
        int dimension = BinaryCodes.dimension(first);
        if (dimension == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }
        return dotProduct(first, second) / dimension;
    }
}
