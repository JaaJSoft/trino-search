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
import dev.jaaj.trino.search.vector.knn.KnnHeap;
import dev.jaaj.trino.search.vector.knn.KnnState;
import dev.jaaj.trino.search.vector.knn.KnnStateFactory;
import dev.jaaj.trino.search.vector.knn.KnnStateSerializer;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
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

import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;

import static io.trino.spi.type.BigintType.BIGINT;

/**
 * A partial state's trip between stages: {@code serialize} on the partial side, {@code deserialize}
 * on the final side, once per group per split.
 * <p>
 * One operation handles a whole batch of full states, so {@code @OperationsPerInvocation} reports
 * the per-state cost. The block {@code deserialize} reads is written by the serializer under test,
 * because the order the neighbours are written in decides how much the restoring heap has to move.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 2, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class BenchmarkKnnStateSerializer
{
    private static final int GROUPS = 256;

    @Param({"10", "100", "1000"})
    public int k;

    @Param({"EUCLIDEAN", "DOT_PRODUCT"})
    public Metric metric;

    private final KnnStateSerializer serializer = new KnnStateSerializer(BIGINT);
    private final KnnStateFactory factory = new KnnStateFactory(BIGINT);
    private KnnState[] states;
    private Block serialized;

    @Setup(Level.Trial)
    public void setUp()
    {
        // More candidates than slots, so every heap is full and its array holds a real heap
        // layout rather than the arrival order.
        int candidates = 4 * k;
        LongArrayBlock keys = VectorBlocks.sequentialKeys(candidates);
        SplittableRandom random = new SplittableRandom(7);
        states = new KnnState[GROUPS];
        for (int group = 0; group < GROUPS; group++) {
            KnnState state = factory.createSingleState();
            state.setK(k);
            state.setMetric(metric);
            state.setHeap(new KnnHeap(k, metric.higherIsCloser()));
            for (int i = 0; i < candidates; i++) {
                state.addToHeap(keys, i, random.nextDouble() * 1000);
            }
            states[group] = state;
        }
        serialized = serializeAll();
    }

    private Block serializeAll()
    {
        BlockBuilder builder = serializer.getSerializedType().createBlockBuilder(null, GROUPS);
        for (KnnState state : states) {
            serializer.serialize(state, builder);
        }
        return builder.build();
    }

    @Benchmark
    @OperationsPerInvocation(GROUPS)
    public int serialize()
    {
        return serializeAll().getPositionCount();
    }

    @Benchmark
    @OperationsPerInvocation(GROUPS)
    public int deserialize()
    {
        KnnState scratch = factory.createSingleState();
        int total = 0;
        for (int position = 0; position < GROUPS; position++) {
            serializer.deserialize(serialized, position, scratch);
            total += scratch.getHeap().size();
        }
        return total;
    }
}
