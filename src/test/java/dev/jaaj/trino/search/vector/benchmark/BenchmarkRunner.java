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

import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;

/**
 * Entry point for a real measurement run. Warmup, iteration counts and JVM arguments come from
 * each benchmark's own annotations, so this only selects what to run.
 */
public final class BenchmarkRunner
{
    private static final String ALL_BENCHMARKS = "dev\\.jaaj\\.trino\\.search\\.vector\\.benchmark\\..*";

    private BenchmarkRunner() {}

    public static void main(String[] args)
            throws RunnerException
    {
        repairForkClasspath();
        Options options = new OptionsBuilder()
                .include(args.length > 0 ? args[0] : ALL_BENCHMARKS)
                .shouldFailOnError(true)
                .build();
        new Runner(options).run();
    }

    /**
     * A forked benchmark JVM is launched with the {@code java.class.path} system property, but
     * {@code exec:java} runs main() through its own {@link URLClassLoader} instead of the system
     * class loader and never updates that property, so JMH forks a child JVM that cannot find its
     * own {@code ForkedMain} class. Rebuilding the property from the context class loader's URLs
     * is what {@code exec:java} itself does not do.
     * <p>
     * That classpath easily exceeds the Windows process-creation command-line limit once
     * {@code trino-testing}'s transitive dependencies are on it, so
     * {@code jmh.separateClasspathJAR} is forced on: JMH then writes the classpath into a
     * manifest-only jar and launches the fork with {@code -cp} pointing at that jar instead of
     * the raw, and much longer, class path string.
     * <p>
     * Package private rather than private: {@link ReferenceRowRunner} also drives JMH through
     * {@code exec:java} and needs the identical repair.
     */
    static void repairForkClasspath()
            throws RunnerException
    {
        if (!(Thread.currentThread().getContextClassLoader() instanceof URLClassLoader classLoader)) {
            return;
        }
        StringBuilder classpath = new StringBuilder();
        try {
            for (URL url : classLoader.getURLs()) {
                if (classpath.length() > 0) {
                    classpath.append(File.pathSeparatorChar);
                }
                classpath.append(Path.of(url.toURI()).toAbsolutePath());
            }
        }
        catch (URISyntaxException e) {
            throw new RunnerException("Failed to rebuild the fork classpath", e);
        }
        System.setProperty("java.class.path", classpath.toString());
        System.setProperty("jmh.separateClasspathJAR", "true");
    }
}
