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
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;
import org.openjdk.jmh.runner.options.VerboseMode;

import java.util.Collection;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Runs every benchmark in a deliberately degraded configuration so that a refactor cannot leave
 * them silently broken.
 * <p>
 * It asserts nothing about the numbers. A shared CI runner cannot produce a trustworthy
 * measurement, and a regression threshold on top of that noise would either fire at random or be
 * loose enough to detect nothing. Real numbers are collected by hand through
 * {@link BenchmarkRunner}.
 */
public class TestBenchmarksSmoke
{
    static void smokeRun(Class<?> benchmarkClass, Map<String, String> params)
            throws RunnerException
    {
        ChainedOptionsBuilder builder = new OptionsBuilder()
                .include(benchmarkClass.getName() + "\\..*")
                // forks(0) runs in the Surefire JVM, which already has the incubator vector
                // module through air.test.jvm.additional-arguments.
                .forks(0)
                .warmupIterations(0)
                .measurementIterations(1)
                .measurementTime(TimeValue.seconds(1))
                .shouldFailOnError(true)
                .verbosity(VerboseMode.SILENT);
        params.forEach(builder::param);

        Collection<RunResult> results = new Runner(builder.build()).run();
        assertThat(results).as("no benchmark matched %s", benchmarkClass.getName()).isNotEmpty();
    }

    @Test
    public void testVectorDistances()
            throws RunnerException
    {
        smokeRun(BenchmarkVectorDistances.class, Map.of("dimension", "8", "metricName", "euclidean"));
    }

    @Test
    public void testKnnHeap()
            throws RunnerException
    {
        smokeRun(BenchmarkKnnHeap.class, Map.of("k", "10", "arrivalOrder", "RANDOM"));
    }

    @Test
    public void testKnnAccumulator()
            throws RunnerException
    {
        smokeRun(BenchmarkKnnAccumulator.class, Map.of("dimension", "8", "metricName", "euclidean", "k", "10"));
    }

    @Test
    public void testKnnAggQuery()
            throws RunnerException
    {
        smokeRun(BenchmarkKnnAggQuery.class, Map.of("shape", "1000x8", "metricName", "euclidean"));
    }

    /**
     * The benchmark is only worth reading if the plan actually splits into a partial aggregation
     * per split plus a final one that merges them: that is the only shape exercising
     * KnnStateSerializer and the combine function. MemoryConfig defaults splits-per-node to the
     * processor count, so this holds without configuration, but it is asserted rather than
     * assumed.
     */
    @Test
    public void testKnnAggQueryPlanHasPartialAndFinalAggregation()
    {
        BenchmarkKnnAggQuery benchmark = new BenchmarkKnnAggQuery();
        benchmark.shape = "1000x8";
        benchmark.metricName = "euclidean";
        benchmark.setUp();
        try {
            String plan = benchmark.explainKnnAgg();
            assertThat(plan).contains("PARTIAL");
            assertThat(plan).contains("FINAL");
        }
        finally {
            benchmark.tearDown();
        }
    }
}
