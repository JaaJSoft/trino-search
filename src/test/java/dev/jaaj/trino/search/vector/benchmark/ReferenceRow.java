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
 * the same CPU at the same clock under the same JIT, which cancels clock speed and thermal
 * throttling and is why the ratio survives being recorded somewhere else better than either raw
 * figure alone. It does not cancel cache size or memory bandwidth: {@link BenchmarkKnnAccumulator}
 * streams a much larger working set than {@link BenchmarkVectorDistances} keeps cache-resident, so
 * a machine with more cache speeds up one half more than the other and moves the ratio in a
 * machine-dependent direction. A same-machine ratio comparison is trustworthy; a cross-machine one
 * is a weaker signal, and the ratio is an upper bound on per-row bookkeeping cost, not a
 * measurement of it, for the same reason.
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
     * A ratio whose propagated relative error is this wide cannot show a regression smaller than
     * itself, which covers most of what this file exists to reveal. {@code BENCHMARKS.md} quotes
     * this same number as the ratio noise band, so the two cannot silently drift apart.
     */
    public static final double RATIO_NOISE_THRESHOLD = 0.15;

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

        /**
         * The relative error of a ratio built from two independently measured operands, by the
         * standard error-propagation formula for a quotient. A relative error on either half
         * under the noise band does not imply the ratio is: two halves each just under the band
         * can combine into a ratio error above it, which is exactly why the gate is computed here
         * rather than by comparing each half to the band separately.
         * <p>
         * JMH reports {@code NaN} for a score error when a run has too few data points to build a
         * confidence interval; propagating that through unchanged would make {@code NaN >
         * threshold} silently false and let the noisiest possible run pass, so a non-finite result
         * is reported as infinitely noisy instead.
         */
        public double ratioRelativeError()
        {
            double kernelRelative = kernelError / kernelNanos;
            double rowRelative = rowError / rowNanos;
            double propagated = Math.sqrt(kernelRelative * kernelRelative + rowRelative * rowRelative);
            return Double.isFinite(propagated) ? propagated : Double.POSITIVE_INFINITY;
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
     * Labels of the measurements whose ratio's propagated relative error is too wide to be worth
     * recording.
     */
    public List<String> tooNoisyToRecord()
    {
        List<String> noisy = new ArrayList<>();
        for (Measurement measurement : List.of(smallDouble, smallReal, largeDouble, largeReal)) {
            if (measurement.ratioRelativeError() > RATIO_NOISE_THRESHOLD) {
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
