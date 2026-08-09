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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * One row of {@code BENCHMARKS.md}: the four reference measurements of a single commit, plus the
 * provenance that makes them readable later.
 * <p>
 * Absolute nanoseconds from two different machines are not comparable, so every row carries the
 * ratio of the per-row cost to the kernel cost as well. Both halves of a ratio were measured on
 * the same CPU at the same clock under the same JIT, which is what lets a ratio survive being
 * recorded somewhere else.
 */
public record ReferenceRow(
        String date,
        String pullRequest,
        String commit,
        Measurement smallDouble,
        Measurement smallReal,
        Measurement largeDouble,
        Measurement largeReal,
        String machine,
        int cores,
        String jdk)
{
    /**
     * A JMH score whose confidence interval is this wide relative to the score cannot show a
     * regression smaller than itself, which covers most of what this file exists to reveal.
     */
    public static final double NOISE_THRESHOLD = 0.15;

    /**
     * @param kernelError half-width of the JMH confidence interval on {@code kernelNanos}, in the
     *         same unit as the score
     */
    public record Measurement(String label, double kernelNanos, double kernelError, double rowNanos, double rowError)
    {
        public double ratio()
        {
            return rowNanos / kernelNanos;
        }

        public double maxRelativeError()
        {
            return Math.max(kernelError / kernelNanos, rowError / rowNanos);
        }
    }

    public String toMarkdownRow()
    {
        return String.format(
                Locale.ROOT,
                "| %s | %s | %s | %s | %s | %s | %s | %s | %d | %s |",
                date,
                pullRequest,
                commit,
                cell(smallDouble),
                cell(smallReal),
                cell(largeDouble),
                cell(largeReal),
                machine,
                cores,
                jdk);
    }

    /**
     * Labels of the measurements whose confidence interval is too wide to be worth recording.
     */
    public List<String> tooNoisyToRecord()
    {
        List<String> noisy = new ArrayList<>();
        for (Measurement measurement : List.of(smallDouble, smallReal, largeDouble, largeReal)) {
            if (measurement.maxRelativeError() > NOISE_THRESHOLD) {
                noisy.add(measurement.label());
            }
        }
        return noisy;
    }

    /**
     * Locale.ROOT rather than the default: a machine whose locale renders a decimal comma would
     * produce a cell that cannot be read, since the cell already separates its three numbers.
     */
    private static String cell(Measurement measurement)
    {
        return String.format(
                Locale.ROOT,
                "%.1f / %.1f / %.2f",
                measurement.kernelNanos(),
                measurement.rowNanos(),
                measurement.ratio());
    }
}
