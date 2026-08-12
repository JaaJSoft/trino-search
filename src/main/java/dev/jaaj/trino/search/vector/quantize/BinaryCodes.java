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
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;

import java.util.function.IntPredicate;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * The layout of a one-bit-per-component vector: a four-byte big-endian dimension followed by
 * {@code ceil(dimension / 8)} bytes of packed bits, least significant bit first within a byte, and
 * zero padding in the last byte.
 * <p>
 * The header is what lets any dimension be stored rather than only multiples of eight, and it
 * turns a comparison between codes of different dimension into an error instead of a ranking
 * quietly computed over the wrong components.
 * <p>
 * Big-endian is not what {@link Slice} reads natively, hence the byte reversal on each end. It is
 * one instruction against a value read once per vector, and it keeps the format readable by
 * anything that is not this class.
 */
public final class BinaryCodes
{
    public static final int HEADER_BYTES = Integer.BYTES;

    private BinaryCodes() {}

    public static Slice pack(int dimension, IntPredicate bitSet)
    {
        Slice codes = Slices.allocate(HEADER_BYTES + payloadBytes(dimension));
        codes.setInt(0, Integer.reverseBytes(dimension));
        for (int i = 0; i < dimension; i++) {
            if (bitSet.test(i)) {
                int index = HEADER_BYTES + (i >>> 3);
                codes.setByte(index, codes.getByte(index) | (1 << (i & 7)));
            }
        }
        return codes;
    }

    public static int dimension(Slice codes)
    {
        if (codes.length() < HEADER_BYTES) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The value is not a binary vector: it is shorter than the header");
        }
        int dimension = Integer.reverseBytes(codes.getInt(0));
        if (dimension < 0 || codes.length() < HEADER_BYTES + payloadBytes(dimension)) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The value is not a binary vector: its header claims %s components, which do not fit in %s bytes"
                            .formatted(dimension, codes.length()));
        }
        return dimension;
    }

    /**
     * The number of components that differ.
     * <p>
     * Every payload byte, including the last, is counted unmasked by the word or byte loop; the
     * last byte's padding bits are then subtracted exactly once. Masking during the main loops
     * instead would need a special case for a payload that is a whole number of words, since the
     * word loop cannot mask a single byte within a word: counting unmasked everywhere and
     * correcting once afterwards avoids that split.
     */
    public static int hamming(Slice first, Slice second)
    {
        int dimension = dimension(first);
        if (dimension != dimension(second)) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }

        int bytes = payloadBytes(dimension);
        int count = 0;
        int i = 0;
        for (; i + Long.BYTES <= bytes; i += Long.BYTES) {
            count += Long.bitCount(first.getLong(HEADER_BYTES + i) ^ second.getLong(HEADER_BYTES + i));
        }
        for (; i < bytes; i++) {
            count += Integer.bitCount((first.getByte(HEADER_BYTES + i) ^ second.getByte(HEADER_BYTES + i)) & 0xFF);
        }
        if (bytes > 0) {
            int lastByte = bytes - 1;
            int mask = tailMask(dimension, lastByte);
            int differenceInLastByte = (first.getByte(HEADER_BYTES + lastByte) ^ second.getByte(HEADER_BYTES + lastByte)) & 0xFF;
            count -= Integer.bitCount(differenceInLastByte & ~mask & 0xFF);
        }
        return count;
    }

    /**
     * Which bits of byte {@code index} belong to a vector of {@code dimension} components. Every
     * byte but the last is fully occupied.
     */
    private static int tailMask(int dimension, int index)
    {
        int remaining = dimension - (index << 3);
        return remaining >= 8 ? 0xFF : (1 << remaining) - 1;
    }

    private static int payloadBytes(int dimension)
    {
        return (dimension + 7) >>> 3;
    }
}
