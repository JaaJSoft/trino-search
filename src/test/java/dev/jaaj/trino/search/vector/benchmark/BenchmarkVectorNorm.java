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

import dev.jaaj.trino.search.vector.VectorFunctions;
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

/**
 * {@code l2_norm} on its own, called through {@link VectorFunctions} because the norm kernel is
 * package private and no {@code Metric} reaches it. The function adds only a null check around
 * the kernel, so this measures the kernel.
 * <p>
 * The pool is cache-resident for the same reason as {@link BenchmarkVectorDistances}'s, and is
 * not representative of streaming a column either.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class BenchmarkVectorNorm
{
    private static final int VECTOR_POOL_SIZE = 256;

    @Param({"128", "768", "1536"})
    public int dimension;

    private Block[] doubleVectors;
    private Block[] realVectors;
    private int index;

    @Setup(Level.Trial)
    public void setUp()
    {
        VectorDataset dataset = VectorDataset.generate(
                VectorDataset.Regime.CLUSTERED, VECTOR_POOL_SIZE, 1, dimension, 1L);
        doubleVectors = VectorBlocks.doubleVectors(dataset.base());
        realVectors = VectorBlocks.realVectors(dataset.base());
    }

    @Benchmark
    public Double doubleVectors()
    {
        index = (index + 1) & (VECTOR_POOL_SIZE - 1);
        return VectorFunctions.l2Norm(doubleVectors[index]);
    }

    @Benchmark
    public Double realVectors()
    {
        index = (index + 1) & (VECTOR_POOL_SIZE - 1);
        return VectorFunctions.l2NormReal(realVectors[index]);
    }
}
