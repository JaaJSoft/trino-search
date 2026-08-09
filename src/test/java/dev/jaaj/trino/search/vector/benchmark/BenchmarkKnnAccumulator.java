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

import dev.jaaj.trino.search.vector.knn.KnnAggregation;
import dev.jaaj.trino.search.vector.knn.KnnState;
import dev.jaaj.trino.search.vector.knn.KnnStateFactory;
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.block.Block;
import io.trino.spi.block.LongArrayBlock;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

import static io.trino.spi.type.BigintType.BIGINT;

/**
 * The real aggregation input path with no engine around it: block read, metric, heap and the
 * per-row bookkeeping in {@code addCandidate}, all together.
 * <p>
 * This is the figure that says how much of the per-row cost the kernel and the heap actually
 * account for. {@code addCandidate} also converts the metric name from a {@code Slice} to a
 * {@code String} and looks the enum up linearly on every single row; whether that matters next to
 * a dimension-768 distance is exactly what this measures.
 * <p>
 * Reading this benchmark's number against {@link BenchmarkVectorDistances} to isolate heap
 * maintenance and per-row bookkeeping is tempting but not sound: the two run on different working
 * sets. This one streams {@link #ROWS} base vectors against a single query vector that stays
 * resident, while the kernel benchmark cycles a small pool with both operands rotating and
 * deliberately cache-resident. The difference between the two carries a memory-locality term in an
 * indeterminate direction, so it is an upper bound on the bookkeeping cost, not a measurement of
 * it.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class BenchmarkKnnAccumulator
{
    private static final int ROWS = 4096;

    @Param({"128", "768"})
    public int dimension;

    @Param({"euclidean", "cosine", "dot_product"})
    public String metricName;

    @Param({"10", "100"})
    public int k;

    private LongArrayBlock keys;
    private Block[] doubleVectors;
    private Block[] realVectors;
    private Block doubleQuery;
    private Block realQuery;
    private Slice metricSlice;

    @Setup(Level.Trial)
    public void setUp()
    {
        VectorDataset dataset = VectorDataset.generate(VectorDataset.Regime.CLUSTERED, ROWS, 1, dimension, 2L);
        doubleVectors = VectorBlocks.doubleVectors(dataset.base());
        realVectors = VectorBlocks.realVectors(dataset.base());
        doubleQuery = VectorBlocks.doubleVector(dataset.queries()[0]);
        realQuery = VectorBlocks.realVector(dataset.queries()[0]);
        metricSlice = Slices.utf8Slice(metricName);
        keys = VectorBlocks.sequentialKeys(ROWS);
    }

    @Benchmark
    @OperationsPerInvocation(ROWS)
    public int doubleRows()
    {
        KnnState state = new KnnStateFactory(BIGINT).createSingleState();
        for (int i = 0; i < ROWS; i++) {
            KnnAggregation.OfDoubleVectors.input(BIGINT, state, keys, i, doubleVectors[i], doubleQuery, k, metricSlice);
        }
        return state.getHeap().size();
    }

    @Benchmark
    @OperationsPerInvocation(ROWS)
    public int realRows()
    {
        KnnState state = new KnnStateFactory(BIGINT).createSingleState();
        for (int i = 0; i < ROWS; i++) {
            KnnAggregation.OfRealVectors.input(BIGINT, state, keys, i, realVectors[i], realQuery, k, metricSlice);
        }
        return state.getHeap().size();
    }
}
