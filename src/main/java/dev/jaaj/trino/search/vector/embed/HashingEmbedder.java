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
package dev.jaaj.trino.search.vector.embed;

import io.airlift.slice.Slice;
import io.airlift.slice.SliceUtf8;
import io.airlift.slice.XxHash64;

/**
 * The hashing trick: every token is hashed to an index and a sign, and contributes one unit
 * there. No vocabulary, therefore nothing to distribute across workers and nothing that makes
 * one row's vector depend on any other row.
 */
final class HashingEmbedder
{
    /**
     * A ceiling rather than an absence of one, so that a mistyped dimension fails with a clear
     * message instead of asking for an array the JVM cannot allocate.
     */
    static final int MAX_DIMENSION = 65536;

    private HashingEmbedder() {}

    static double[] accumulate(Slice text, EmbeddingAlgorithm algorithm, int dimension)
    {
        double[] accumulator = new double[dimension];
        Slice lowercased = SliceUtf8.toLowerCase(text);
        algorithm.forEachToken(lowercased, (source, offset, length) -> {
            long hash = XxHash64.hash(source, offset, length);
            accumulator[indexOf(hash, dimension)] += signOf(hash);
        });
        return accumulator;
    }

    /**
     * Maps the top 32 bits of the hash onto {@code [0, dimension)} by a multiply and a shift,
     * rather than by a modulo. One multiplication replaces a 64-bit division per token.
     * <p>
     * Requiring a power-of-two dimension so a mask could be used was rejected: 768, the most
     * common embedding width in practice, is not one.
     */
    static int indexOf(long hash, int dimension)
    {
        return (int) (((hash >>> 32) * dimension) >>> 32);
    }

    /**
     * The sign is taken from bit 31, which {@link #indexOf} discards, so index and sign are
     * independent. Two distinct tokens landing on the same index then cancel in expectation
     * instead of reinforcing each other, which is what keeps a collision from inventing
     * similarity that is not there.
     */
    static double signOf(long hash)
    {
        return ((hash >>> 31) & 1) == 0 ? 1.0 : -1.0;
    }
}
