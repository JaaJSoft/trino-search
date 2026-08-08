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

import dev.jaaj.trino.search.vector.Metric;
import io.trino.spi.block.Block;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;

/**
 * The metric kernel on its own, called through {@link Metric#compute} because {@code VectorMath}
 * is package private and {@code Metric} is the entry point the aggregation actually uses.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class BenchmarkVectorDistances
{
    /**
     * The measured call cycles through a pool of pairs rather than reusing one. A single pair
     * stays resident in L1 and lets the JIT specialise on it, which produces a flattering number
     * unrelated to streaming a column. A power of two keeps the wrap-around a mask.
     */
    private static final int PAIR_POOL_SIZE = 256;

    @Param({"128", "768", "1536"})
    public int dimension;

    @Param({"euclidean", "euclidean_squared", "cosine", "dot_product", "manhattan"})
    public String metricName;

    private Metric metric;
    private Block[] doubleLeft;
    private Block[] doubleRight;
    private Block[] realLeft;
    private Block[] realRight;
    private int index;

    @Setup(Level.Trial)
    public void setUp()
    {
        metric = Metric.fromName(metricName);
        VectorDataset dataset = VectorDataset.generate(
                VectorDataset.Regime.CLUSTERED, PAIR_POOL_SIZE, PAIR_POOL_SIZE, dimension, 1L);
        doubleLeft = VectorBlocks.doubleVectors(dataset.base());
        doubleRight = VectorBlocks.doubleVectors(dataset.queries());
        realLeft = VectorBlocks.realVectors(dataset.base());
        realRight = VectorBlocks.realVectors(dataset.queries());
    }

    @Benchmark
    public double doubleVectors()
    {
        index = (index + 1) & (PAIR_POOL_SIZE - 1);
        return metric.compute(doubleLeft[index], doubleRight[index], DOUBLE_READER);
    }

    @Benchmark
    public double realVectors()
    {
        index = (index + 1) & (PAIR_POOL_SIZE - 1);
        return metric.compute(realLeft[index], realRight[index], REAL_READER);
    }
}
