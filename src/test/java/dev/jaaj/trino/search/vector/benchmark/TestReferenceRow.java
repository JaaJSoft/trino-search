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
package dev.jaaj.trino.search.vector.benchmark;

import org.junit.jupiter.api.Test;

import java.util.Locale;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestReferenceRow
{
    private static ReferenceRow row()
    {
        return new ReferenceRow(
                "2026-08-09",
                "#11",
                // Values already at one decimal on purpose: a score such as 102.95 sits close
                // enough to a rounding boundary that %.1f can land either side of it depending on
                // the binary representation, which would make this expectation flaky.
                new ReferenceRow.Measurement("128 double", 91.8, 1.0, 219.0, 2.0),
                new ReferenceRow.Measurement("128 real", 103.0, 1.0, 187.8, 2.0),
                new ReferenceRow.Measurement("768 double", 523.3, 5.0, 859.3, 8.0),
                new ReferenceRow.Measurement("768 real", 564.2, 5.0, 829.3, 8.0),
                "desktop-5950x",
                "AMD Ryzen 9 5950X 16-Core Processor",
                32,
                "25.0.1");
    }

    @Test
    public void testRatioIsThePerRowCostOverTheKernelCost()
    {
        assertThat(row().smallDouble().ratio()).isCloseTo(219.0 / 91.8, within(1e-12));
    }

    @Test
    public void testMarkdownRowRendersEveryColumn()
    {
        assertThat(row().toMarkdownRow()).isEqualTo(
                "| 2026-08-09 | #11 "
                        + "| 91.8 / 219.0 / 2.39 "
                        + "| 103.0 / 187.8 / 1.82 "
                        + "| 523.3 / 859.3 / 1.64 "
                        + "| 564.2 / 829.3 / 1.47 "
                        + "| desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |");
    }

    /**
     * The default locale on this project's development machine is French, where %.1f renders a
     * comma. A comma inside a cell that already reads "kernel / per-row / ratio" makes the cell
     * ambiguous and breaks anything that splits on it, so the formatting must pin Locale.ROOT.
     */
    @Test
    public void testDecimalSeparatorIsADotWhateverTheDefaultLocale()
    {
        Locale original = Locale.getDefault();
        try {
            Locale.setDefault(Locale.FRANCE);
            assertThat(row().toMarkdownRow()).contains("91.8 / 219.0 / 2.39").doesNotContain(",");
        }
        finally {
            Locale.setDefault(original);
        }
    }

    @Test
    public void testAQuietRunRecordsNothingAsTooNoisy()
    {
        assertThat(row().tooNoisyToRecord()).isEmpty();
    }

    /**
     * A relative error above the threshold on either half of a measurement is enough to push the
     * propagated ratio error above the threshold too: a ratio is only as trustworthy as its
     * noisier operand.
     */
    @Test
    public void testANoisyMeasurementIsNamed()
    {
        ReferenceRow noisy = new ReferenceRow(
                "2026-08-09",
                "#11",
                new ReferenceRow.Measurement("128 double", 91.8, 30.0, 219.0, 2.0),
                new ReferenceRow.Measurement("128 real", 103.0, 1.0, 187.8, 60.0),
                new ReferenceRow.Measurement("768 double", 523.3, 5.0, 859.3, 8.0),
                new ReferenceRow.Measurement("768 real", 564.2, 5.0, 829.3, 8.0),
                "desktop-5950x",
                "AMD Ryzen 9 5950X 16-Core Processor",
                32,
                "25.0.1");
        assertThat(noisy.tooNoisyToRecord()).containsExactly("128 double", "128 real");
    }

    /**
     * Neither half exceeds the old per-half threshold of 15 percent on its own, yet propagating
     * both through the ratio pushes the combined uncertainty to just under 20 percent, above the
     * gate. A gate that compared each half separately, as the previous implementation did, would
     * let this measurement through and publish a ratio with more uncertainty than the file tells
     * the reader to expect.
     */
    @Test
    public void testTwoHalvesEachQuietCombineIntoANoisyRatio()
    {
        ReferenceRow quietHalves = new ReferenceRow(
                "2026-08-09",
                "#11",
                new ReferenceRow.Measurement("128 double", 100.0, 14.0, 200.0, 28.0),
                new ReferenceRow.Measurement("128 real", 103.0, 1.0, 187.8, 2.0),
                new ReferenceRow.Measurement("768 double", 523.3, 5.0, 859.3, 8.0),
                new ReferenceRow.Measurement("768 real", 564.2, 5.0, 829.3, 8.0),
                "desktop-5950x",
                "AMD Ryzen 9 5950X 16-Core Processor",
                32,
                "25.0.1");
        assertThat(quietHalves.tooNoisyToRecord()).containsExactly("128 double");
    }

    @Test
    public void testRatioRelativeErrorPropagatesBothHalves()
    {
        ReferenceRow.Measurement measurement = new ReferenceRow.Measurement("x", 100.0, 3.0, 200.0, 4.0);
        assertThat(measurement.ratioRelativeError()).isCloseTo(Math.sqrt(0.03 * 0.03 + 0.02 * 0.02), within(1e-12));
    }

    @Test
    public void testANonFiniteRelativeErrorIsTreatedAsNoisy()
    {
        ReferenceRow.Measurement measurement = new ReferenceRow.Measurement("x", 100.0, Double.NaN, 200.0, 4.0);
        assertThat(Double.isFinite(measurement.ratioRelativeError())).isFalse();
        assertThat(measurement.ratioRelativeError()).isGreaterThan(ReferenceRow.RATIO_NOISE_THRESHOLD);
    }
}
