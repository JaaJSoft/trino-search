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
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestNormalizedVectorBlocks
{
    private static double normOfDoubleBlock(Block block)
    {
        double sum = 0.0;
        for (int i = 0; i < block.getPositionCount(); i++) {
            double value = Double.longBitsToDouble(((LongArrayBlock) block).getLong(i));
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    private static double normOfRealBlock(Block block)
    {
        double sum = 0.0;
        for (int i = 0; i < block.getPositionCount(); i++) {
            double value = Float.intBitsToFloat(((IntArrayBlock) block).getInt(i));
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    @Test
    public void testDoubleBlockIsLongArrayBackedAndUnitNorm()
    {
        Block block = NormalizedVectorBlocks.doubles(new double[] {3.0, 4.0});
        assertThat(block).isInstanceOf(LongArrayBlock.class);
        assertThat(block.getPositionCount()).isEqualTo(2);
        assertThat(Double.longBitsToDouble(((LongArrayBlock) block).getLong(0))).isCloseTo(0.6, within(1e-12));
        assertThat(Double.longBitsToDouble(((LongArrayBlock) block).getLong(1))).isCloseTo(0.8, within(1e-12));
        assertThat(normOfDoubleBlock(block)).isCloseTo(1.0, within(1e-12));
    }

    @Test
    public void testRealBlockIsIntArrayBackedAndUnitNorm()
    {
        Block block = NormalizedVectorBlocks.real(new double[] {3.0, 4.0});
        assertThat(block).isInstanceOf(IntArrayBlock.class);
        assertThat(block.getPositionCount()).isEqualTo(2);
        assertThat(Float.intBitsToFloat(((IntArrayBlock) block).getInt(0))).isCloseTo(0.6f, within(1e-6f));
        assertThat(normOfRealBlock(block)).isCloseTo(1.0, within(1e-6));
    }

    @Test
    public void testNegativeComponentsSurviveNormalization()
    {
        Block block = NormalizedVectorBlocks.doubles(new double[] {-3.0, 4.0});
        assertThat(Double.longBitsToDouble(((LongArrayBlock) block).getLong(0))).isCloseTo(-0.6, within(1e-12));
    }

    /**
     * The zero accumulator is a text with no token. It must produce the zero vector, not NaN and
     * not an exception: the input is a user text column, and one empty string among a billion
     * rows cannot be allowed to fail the scan.
     */
    @Test
    public void testZeroAccumulatorGivesTheZeroVector()
    {
        Block block = NormalizedVectorBlocks.doubles(new double[] {0.0, 0.0, 0.0});
        for (int i = 0; i < 3; i++) {
            assertThat(Double.longBitsToDouble(((LongArrayBlock) block).getLong(i))).isEqualTo(0.0);
        }
        assertThat(NormalizedVectorBlocks.reciprocalNorm(new double[] {0.0, 0.0})).isEqualTo(0.0);
    }

    @Test
    public void testBlocksReportNoNulls()
    {
        assertThat(NormalizedVectorBlocks.doubles(new double[] {1.0, 2.0}).mayHaveNull()).isFalse();
        assertThat(NormalizedVectorBlocks.real(new double[] {1.0, 2.0}).mayHaveNull()).isFalse();
    }

    private static double[] accumulatorOfLength(int length)
    {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            // Signed, uneven, and not a round float, so a component written in the wrong order or
            // converted at the wrong width shows up as a difference rather than cancelling out.
            values[i] = ((i % 7) - 3) * (1.0 + (i % 13) / 17.0);
        }
        return values;
    }

    /**
     * Every component is the accumulator scaled by the same factor, at lengths that fall on both
     * sides of a vector register: below one, exactly one, and one past a whole number of them.
     * <p>
     * A vector loop over these arrays handles whole registers and leaves the remainder to a
     * scalar tail, and the boundary between the two is where an off-by-one lives. Comparing every
     * component exactly, at lengths straddling every plausible register width, is what keeps a
     * component from being dropped, doubled or written to the wrong half.
     * <p>
     * The per-component comparison takes its scale from {@code reciprocalNorm} and so says nothing
     * about the norm itself: a norm that skipped components would still scale every component by
     * the same wrong factor and pass. The unit-norm assertion is what closes that, because it
     * recomputes the norm from the block. It needs an accumulator that is non-zero near its end to
     * be worth anything, which is why it lives here on dense data rather than beside a query test,
     * where a hashed text leaves almost every component zero and a dropped tail invisible.
     */
    @Test
    public void testEveryComponentIsTheScaledAccumulatorAtEveryLength()
    {
        for (int length : new int[] {0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 127, 768, 769}) {
            double[] accumulator = accumulatorOfLength(length);
            double scale = NormalizedVectorBlocks.reciprocalNorm(accumulator);
            Block doubleBlock = NormalizedVectorBlocks.doubles(accumulator);
            Block realBlock = NormalizedVectorBlocks.real(accumulator);

            assertThat(doubleBlock.getPositionCount()).as("length %d", length).isEqualTo(length);
            assertThat(realBlock.getPositionCount()).as("length %d", length).isEqualTo(length);
            for (int i = 0; i < length; i++) {
                assertThat(Double.longBitsToDouble(((LongArrayBlock) doubleBlock).getLong(i)))
                        .as("double component %d of %d", i, length)
                        .isEqualTo(accumulator[i] * scale);
                assertThat(Float.intBitsToFloat(((IntArrayBlock) realBlock).getInt(i)))
                        .as("real component %d of %d", i, length)
                        .isEqualTo((float) (accumulator[i] * scale));
            }

            // The empty accumulator is excluded rather than special-cased: its norm is legitimately
            // zero, since there is no component to scale.
            if (length > 0) {
                assertThat(normOfDoubleBlock(doubleBlock))
                        .as("double norm at %d", length)
                        .isCloseTo(1.0, within(1e-12));
                assertThat(normOfRealBlock(realBlock))
                        .as("real norm at %d", length)
                        .isCloseTo(1.0, within(1e-6));
            }
        }
    }
}
