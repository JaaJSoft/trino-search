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

import org.openjdk.jmh.results.Result;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Measures the four reference points and prints the {@code BENCHMARKS.md} row for them.
 * <p>
 * Eight JMH configurations instead of the full grid's fifty-four is what brings a recording down
 * from a quarter of an hour to a few minutes, which is the difference between a habit that
 * survives and one abandoned by the third pull request. Each benchmark keeps its own fork, warmup
 * and iteration counts, so a recorded row is produced under the same conditions as a full run.
 * <p>
 * It prints and does not write: re-measuring must not append a second row, and pasting by hand
 * forces whoever records it to look at what they are adding.
 */
public final class ReferenceRowRunner
{
    private static final String METRIC = "euclidean";
    private static final String K = "10";
    private static final int SMALL_DIMENSION = 128;
    private static final int LARGE_DIMENSION = 768;
    private static final Pattern PULL_REQUEST_NUMBER = Pattern.compile("#?\\d+");
    private static final String UNKNOWN_CPU = "unknown";

    private ReferenceRowRunner() {}

    public static void main(String[] args)
            throws RunnerException
    {
        if (args.length == 0) {
            System.err.println("usage: ReferenceRowRunner <machine-label> [pull-request-number]");
            System.err.println("  the machine label must name one specific machine, e.g. 'desktop-5950x', and must");
            System.err.println("  never be reused for a different machine; it must not be a hostname, because");
            System.err.println("  BENCHMARKS.md is committed to a public repository");
            System.exit(2);
            return;
        }

        BenchmarkRunner.repairForkClasspath();

        ReferenceRow row = new ReferenceRow(
                LocalDate.now().toString(),
                pullRequestLabel(args),
                currentCommit(),
                measure("128 double", SMALL_DIMENSION, "doubleVectors", "doubleRows"),
                measure("128 real", SMALL_DIMENSION, "realVectors", "realRows"),
                measure("768 double", LARGE_DIMENSION, "doubleVectors", "doubleRows"),
                measure("768 real", LARGE_DIMENSION, "realVectors", "realRows"),
                machineLabel(args),
                currentCpu(),
                Runtime.getRuntime().availableProcessors(),
                System.getProperty("java.version"));

        List<String> noisy = row.tooNoisyToRecord();
        if (!noisy.isEmpty()) {
            System.err.printf(
                    "warning: %s exceeded %.0f%% ratio relative error; this run is too noisy to record, "
                            + "measure again on a quieter machine%n",
                    noisy,
                    ReferenceRow.RATIO_NOISE_THRESHOLD * 100);
        }

        System.out.println(row.toMarkdownRow());

        if (!noisy.isEmpty()) {
            System.exit(1);
        }
    }

    static String machineLabel(String[] args)
    {
        return args[0];
    }

    /**
     * The processor model, which is what actually decides whether two rows can be compared: cache
     * size and memory bandwidth are exactly what the ratio does not cancel. There is no portable
     * way to ask the JVM for it, so this reads the two places that do know and degrades to
     * {@value #UNKNOWN_CPU} rather than failing, since a row without it is still worth recording.
     */
    static String currentCpu()
    {
        String fromLinux = cpuFromProcInfo();
        if (fromLinux != null) {
            return fromLinux;
        }
        String fromWindows = cpuFromWindowsRegistry();
        return fromWindows != null ? fromWindows : UNKNOWN_CPU;
    }

    private static String cpuFromProcInfo()
    {
        Path cpuInfo = Path.of("/proc/cpuinfo");
        if (!Files.isReadable(cpuInfo)) {
            return null;
        }
        try {
            for (String line : Files.readAllLines(cpuInfo, StandardCharsets.UTF_8)) {
                if (line.startsWith("model name")) {
                    return normaliseCpu(line.substring(line.indexOf(':') + 1));
                }
            }
        }
        catch (IOException e) {
            return null;
        }
        return null;
    }

    private static String cpuFromWindowsRegistry()
    {
        if (!System.getProperty("os.name", "").toLowerCase(Locale.ROOT).startsWith("windows")) {
            return null;
        }
        try {
            String output = runCommand(
                    "reg",
                    "query",
                    "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                    "/v",
                    "ProcessorNameString");
            int marker = output.indexOf("REG_SZ");
            return marker < 0 ? null : normaliseCpu(output.substring(marker + "REG_SZ".length()));
        }
        catch (IllegalStateException e) {
            return null;
        }
    }

    /**
     * Both sources pad the model with runs of spaces, which a markdown cell would keep.
     */
    private static String normaliseCpu(String raw)
    {
        String collapsed = raw.trim().replaceAll("\\s+", " ");
        return collapsed.isEmpty() ? null : collapsed;
    }

    /**
     * Accepts "11" and "#11" alike, because both are natural to type and the column is only
     * sortable and greppable if they land in the file identically. Anything else is refused here,
     * at the command line where it can still be retyped: stripping every hash and prefixing one
     * would turn "1#1" into "#11" and "draft" into "#draft", which look like real identifiers once
     * pasted into a row nobody revisits.
     */
    static String pullRequestLabel(String[] args)
    {
        if (args.length < 2) {
            return "#TBD";
        }
        if (!PULL_REQUEST_NUMBER.matcher(args[1]).matches()) {
            throw new IllegalArgumentException(
                    "the pull request must be a number, optionally prefixed with a hash; got '%s'"
                            .formatted(args[1]));
        }
        return "#" + args[1].replace("#", "");
    }

    private static ReferenceRow.Measurement measure(
            String label,
            int dimension,
            String kernelMethod,
            String accumulatorMethod)
            throws RunnerException
    {
        Result<?> kernel = runOne(
                BenchmarkVectorDistances.class,
                kernelMethod,
                Map.of("dimension", String.valueOf(dimension), "metricName", METRIC));
        Result<?> accumulator = runOne(
                BenchmarkKnnAccumulator.class,
                accumulatorMethod,
                Map.of("dimension", String.valueOf(dimension), "metricName", METRIC, "k", K));
        return new ReferenceRow.Measurement(
                label,
                kernel.getScore(),
                kernel.getScoreError(),
                accumulator.getScore(),
                accumulator.getScoreError());
    }

    private static Result<?> runOne(Class<?> benchmarkClass, String method, Map<String, String> params)
            throws RunnerException
    {
        ChainedOptionsBuilder builder = new OptionsBuilder()
                .include(benchmarkClass.getName() + "\\." + method + "$")
                .timeUnit(TimeUnit.NANOSECONDS)
                .shouldFailOnError(true);
        params.forEach(builder::param);

        Collection<RunResult> results = new Runner(builder.build()).run();
        if (results.size() != 1) {
            throw new IllegalStateException(
                    "expected exactly one result for %s.%s with %s, got %s"
                            .formatted(benchmarkClass.getSimpleName(), method, params, results.size()));
        }
        return results.iterator().next().getPrimaryResult();
    }

    /**
     * A row that cannot be traced back to the code it measured is worth less than no row, so a
     * missing or failing git is fatal rather than a placeholder. The commit alone is not enough:
     * measuring after changing a kernel but before committing stamps the new numbers with the old,
     * innocent-looking sha, so a dirty working tree is appended to the sha rather than hidden.
     */
    private static String currentCommit()
    {
        String commit = runGit("rev-parse", "--short", "HEAD");
        boolean dirty = !runGit("status", "--porcelain").isEmpty();
        return dirty ? commit + "-dirty" : commit;
    }

    private static String runGit(String... args)
    {
        String[] command = new String[args.length + 1];
        command[0] = "git";
        System.arraycopy(args, 0, command, 1, args.length);
        return runCommand(command);
    }

    private static String runCommand(String... command)
    {
        String description = String.join(" ", command);
        try {
            Process process = new ProcessBuilder(command)
                    .redirectErrorStream(true)
                    .start();
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line = reader.readLine();
                while (line != null) {
                    if (output.length() > 0) {
                        output.append('\n');
                    }
                    output.append(line);
                    line = reader.readLine();
                }
            }
            if (process.waitFor() != 0) {
                throw new IllegalStateException(description + " failed: " + output);
            }
            return output.toString().trim();
        }
        catch (IOException e) {
            throw new IllegalStateException("could not run " + description, e);
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while running " + description, e);
        }
    }
}
