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

import io.trino.spi.block.Block;
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.util.Optional;

/**
 * Turns a raw accumulator into the unit-norm vector a caller receives.
 * <p>
 * The block is built from a primitive bit array rather than through a {@code BlockBuilder}.
 * {@code RealType} stores float bits in an {@code int[]} and {@code DoubleType} stores double
 * bits in a {@code long[]}, so this is their native representation and costs no copy. It is also
 * what leaves the loop vectorizable: a {@code BlockBuilder} only exposes per-element writes.
 */
final class NormalizedVectorBlocks
{
    private NormalizedVectorBlocks() {}

    static Block real(double[] accumulator)
    {
        double scale = reciprocalNorm(accumulator);
        int[] bits = new int[accumulator.length];
        for (int i = 0; i < accumulator.length; i++) {
            bits[i] = Float.floatToRawIntBits((float) (accumulator[i] * scale));
        }
        return new IntArrayBlock(accumulator.length, Optional.empty(), bits);
    }

    static Block doubles(double[] accumulator)
    {
        double scale = reciprocalNorm(accumulator);
        long[] bits = new long[accumulator.length];
        for (int i = 0; i < accumulator.length; i++) {
            bits[i] = Double.doubleToRawLongBits(accumulator[i] * scale);
        }
        return new LongArrayBlock(accumulator.length, Optional.empty(), bits);
    }

    /**
     * The reciprocal of the euclidean norm, so the scaling loop multiplies rather than divides.
     * <p>
     * A zero accumulator yields a zero factor rather than an infinite one, which turns the
     * scaling of the zero vector into zeros instead of NaN and removes the need for a branch
     * inside either loop. No overflow guard is needed here, unlike in {@code VectorMath.norm}:
     * the components are token counts, so reaching a magnitude that squares to infinity would
     * take more tokens than a text can hold.
     */
    static double reciprocalNorm(double[] values)
    {
        double sum = 0.0;
        for (double value : values) {
            sum += value * value;
        }
        return sum == 0 ? 0.0 : 1.0 / Math.sqrt(sum);
    }
}
