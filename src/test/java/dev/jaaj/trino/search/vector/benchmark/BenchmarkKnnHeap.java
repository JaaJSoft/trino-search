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

import dev.jaaj.trino.search.vector.knn.KnnHeap;
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

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;

/**
 * The bounded heap on its own, with no vectors involved.
 * <p>
 * One operation fills a fresh heap with a whole batch of precomputed distances rather than adding
 * a single candidate: a per-invocation setup that reset the heap would cost as much as the
 * measured work itself. {@code @OperationsPerInvocation} makes JMH report the per-candidate cost.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class BenchmarkKnnHeap
{
    private static final int BATCH = 10_000;

    public enum ArrivalOrder
    {
        /**
         * The typical case: most candidates lose to the root and are rejected on one comparison.
         */
        RANDOM,
        /**
         * The worst case: every candidate beats the root, so every add triggers a siftDown.
         */
        ASCENDING,
        /**
         * The best case: nothing is accepted once the heap is full.
         */
        DESCENDING,
    }

    @Param({"1", "10", "100", "1000"})
    public int k;

    @Param({"RANDOM", "ASCENDING", "DESCENDING"})
    public String arrivalOrder;

    private double[] distances;
    private Object[] keys;
    private KnnHeap other;

    @Setup(Level.Trial)
    public void setUp()
    {
        SplittableRandom random = new SplittableRandom(5L);
        distances = new double[BATCH];
        keys = new Object[BATCH];
        for (int i = 0; i < BATCH; i++) {
            distances[i] = random.nextDouble();
            keys[i] = (long) i;
        }

        switch (ArrivalOrder.valueOf(arrivalOrder)) {
            case ASCENDING -> Arrays.sort(distances);
            case DESCENDING -> {
                Arrays.sort(distances);
                for (int i = 0, j = BATCH - 1; i < j; i++, j--) {
                    double swap = distances[i];
                    distances[i] = distances[j];
                    distances[j] = swap;
                }
            }
            case RANDOM -> {
                // Already in random order.
            }
        }

        other = fullHeap();
    }

    /**
     * The main number: the per-candidate cost of the common path.
     */
    @Benchmark
    @OperationsPerInvocation(BATCH)
    public int fillHeap()
    {
        KnnHeap heap = new KnnHeap(k, false);
        for (int i = 0; i < BATCH; i++) {
            heap.add(keys[i], distances[i]);
        }
        return heap.size();
    }

    /**
     * Baseline for {@link #buildAndMerge}: the same k adds, without the merge.
     */
    @Benchmark
    public int buildFullHeap()
    {
        return fullHeap().size();
    }

    /**
     * The cost paid per split during the final aggregation. {@code mergeFrom} mutates its target,
     * so the target has to be rebuilt every invocation; read this against
     * {@link #buildFullHeap} at the same k, the difference is the merge itself.
     */
    @Benchmark
    public int buildAndMerge()
    {
        KnnHeap heap = fullHeap();
        heap.mergeFrom(other);
        return heap.size();
    }

    private KnnHeap fullHeap()
    {
        KnnHeap heap = new KnnHeap(k, false);
        for (int i = 0; i < k; i++) {
            heap.add(keys[i], distances[i]);
        }
        return heap;
    }
}
