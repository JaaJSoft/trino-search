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
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

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
    /**
     * The same preferred species {@code VectorMath} uses, and available under the same condition:
     * {@code jdk.incubator.vector} is an incubator module, but a JVM unable to load it is a JVM
     * unable to start the engine this plugin runs inside.
     */
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;

    /**
     * How many double vectors narrow into one int vector's worth of components. Every species here
     * is the preferred one, so they share a shape and this is the ratio of their element sizes.
     */
    private static final int NARROWING_PARTS = INT_SPECIES.length() / DOUBLE_SPECIES.length();

    private NormalizedVectorBlocks() {}

    static Block real(double[] accumulator)
    {
        double scale = reciprocalNorm(accumulator);
        int[] bits = new int[accumulator.length];

        int lanes = INT_SPECIES.length();
        int vectorized = INT_SPECIES.loopBound(accumulator.length);
        int i = 0;
        for (; i < vectorized; i += lanes) {
            IntVector narrowed = IntVector.zero(INT_SPECIES);
            for (int part = 0; part < NARROWING_PARTS; part++) {
                DoubleVector scaled = DoubleVector.fromArray(DOUBLE_SPECIES, accumulator, i + part * DOUBLE_SPECIES.length())
                        .mul(scale);
                // A contracting conversion fills one block of the output and zeroes the rest, and
                // the part number that steers a block to lane offset part*DOUBLE_SPECIES.length()
                // is its negation. The blocks are then reassembled bitwise on the int view rather
                // than with FIRST_NONZERO on the floats, which would read a negative zero as empty
                // and hand back a positive one.
                narrowed = narrowed.or(((FloatVector) scaled.convertShape(VectorOperators.D2F, FLOAT_SPECIES, -part)).reinterpretAsInts());
            }
            narrowed.intoArray(bits, i);
        }

        for (; i < accumulator.length; i++) {
            bits[i] = Float.floatToRawIntBits((float) (accumulator[i] * scale));
        }
        return new IntArrayBlock(accumulator.length, Optional.empty(), bits);
    }

    static Block doubles(double[] accumulator)
    {
        double scale = reciprocalNorm(accumulator);
        long[] bits = new long[accumulator.length];

        int lanes = DOUBLE_SPECIES.length();
        int vectorized = DOUBLE_SPECIES.loopBound(accumulator.length);
        int i = 0;
        for (; i < vectorized; i += lanes) {
            DoubleVector.fromArray(DOUBLE_SPECIES, accumulator, i)
                    .mul(scale)
                    .reinterpretAsLongs()
                    .intoArray(bits, i);
        }

        for (; i < accumulator.length; i++) {
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
        // One partial sum per lane adds the squares in a different order than a left-to-right sum,
        // so the total can differ from it in its last bits. That is a change of rounding, not of
        // accuracy, and it is the same trade VectorMath.euclideanSquared already takes.
        DoubleVector squares = DoubleVector.zero(DOUBLE_SPECIES);
        int lanes = DOUBLE_SPECIES.length();
        int vectorized = DOUBLE_SPECIES.loopBound(values.length);
        int i = 0;
        for (; i < vectorized; i += lanes) {
            DoubleVector chunk = DoubleVector.fromArray(DOUBLE_SPECIES, values, i);
            squares = chunk.fma(chunk, squares);
        }

        double sum = squares.reduceLanes(VectorOperators.ADD);
        for (; i < values.length; i++) {
            sum += values[i] * values[i];
        }
        return sum == 0 ? 0.0 : 1.0 / Math.sqrt(sum);
    }
}
