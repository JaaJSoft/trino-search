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
                "50e0043",
                // Values already at one decimal on purpose: a score such as 102.95 sits close
                // enough to a rounding boundary that %.1f can land either side of it depending on
                // the binary representation, which would make this expectation flaky.
                new ReferenceRow.Measurement("128 double", 91.8, 1.0, 219.0, 2.0),
                new ReferenceRow.Measurement("128 real", 103.0, 1.0, 187.8, 2.0),
                new ReferenceRow.Measurement("768 double", 523.3, 5.0, 859.3, 8.0),
                new ReferenceRow.Measurement("768 real", 564.2, 5.0, 829.3, 8.0),
                "laptop",
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
                "| 2026-08-09 | #11 | 50e0043 "
                        + "| 91.8 / 219.0 / 2.39 "
                        + "| 103.0 / 187.8 / 1.82 "
                        + "| 523.3 / 859.3 / 1.64 "
                        + "| 564.2 / 829.3 / 1.47 "
                        + "| laptop | 32 | 25.0.1 |");
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
     * A relative error above the threshold on either half of a measurement is enough: a ratio is
     * only as trustworthy as its noisier operand.
     */
    @Test
    public void testANoisyMeasurementIsNamed()
    {
        ReferenceRow noisy = new ReferenceRow(
                "2026-08-09",
                "#11",
                "50e0043",
                new ReferenceRow.Measurement("128 double", 91.8, 30.0, 219.0, 2.0),
                new ReferenceRow.Measurement("128 real", 103.0, 1.0, 187.8, 60.0),
                new ReferenceRow.Measurement("768 double", 523.3, 5.0, 859.3, 8.0),
                new ReferenceRow.Measurement("768 real", 564.2, 5.0, 829.3, 8.0),
                "laptop",
                32,
                "25.0.1");
        assertThat(noisy.tooNoisyToRecord()).containsExactly("128 double", "128 real");
    }

    @Test
    public void testMaxRelativeErrorTakesTheWorseOfTheTwoScores()
    {
        ReferenceRow.Measurement measurement = new ReferenceRow.Measurement("x", 100.0, 5.0, 200.0, 40.0);
        assertThat(measurement.maxRelativeError()).isCloseTo(0.2, within(1e-12));
    }
}
