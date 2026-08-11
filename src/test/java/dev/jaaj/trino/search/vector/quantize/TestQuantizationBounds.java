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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class TestQuantizationBounds
{
    @Test
    public void testMidpointEncodesToZero()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0.0}, new double[] {1.0});
        assertThat(bounds.encode(0, 0.0)).isEqualTo((byte) 0);
    }

    /**
     * The bounds are fitted on a sample, so a later row can fall outside them. Without the clamp
     * the cast wraps and turns a far-away vector into a near neighbour, which is a wrong answer
     * rather than an imprecise one.
     */
    @Test
    public void testValuesOutsideTheBoundsClampInsteadOfWrapping()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0.0}, new double[] {1.0});
        assertThat(bounds.encode(0, 10_000.0)).isEqualTo((byte) 127);
        assertThat(bounds.encode(0, -10_000.0)).isEqualTo((byte) -128);
    }

    /**
     * A dimension whose corpus minimum equals its maximum has no range to spread over the codes.
     * Every value there is the offset exactly, so this is a division guard rather than an error.
     */
    @Test
    public void testConstantDimensionEncodesToZeroAndDecodesToTheOffset()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {4.5}, new double[] {0.0});
        assertThat(bounds.encode(0, 4.5)).isEqualTo((byte) 0);
        assertThat(bounds.encode(0, 99.0)).isEqualTo((byte) 0);
        assertThat(bounds.decode(0, (byte) 0)).isEqualTo(4.5);
    }

    @Test
    public void testRoundTripIsWithinHalfAStep()
    {
        double offset = 0.5;
        double scale = 2.0 / 255.0;
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {offset}, new double[] {scale});
        for (double value = -0.5; value <= 1.5; value += 0.01) {
            double roundTripped = bounds.decode(0, bounds.encode(0, value));
            assertThat(Math.abs(roundTripped - value)).isLessThanOrEqualTo(scale / 2 + 1e-12);
        }
    }

    @Test
    public void testNanEncodesToZeroRatherThanToAnExtreme()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0.0}, new double[] {1.0});
        assertThat(bounds.encode(0, Double.NaN)).isEqualTo((byte) 0);
    }
}
