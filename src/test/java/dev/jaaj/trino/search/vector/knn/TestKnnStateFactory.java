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
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.vector.knn.KnnStateFactory.GroupedKnnState;
import dev.jaaj.trino.search.vector.knn.KnnStateFactory.SingleKnnState;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

public class TestKnnStateFactory
{
    @Test
    public void testSingleStateCountsItsHeap()
    {
        SingleKnnState state = new SingleKnnState();
        long empty = state.getEstimatedSize();

        KnnHeap heap = new KnnHeap(64, false);
        state.setHeap(heap);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + heap.estimatedSizeInBytes());
    }

    @Test
    public void testGroupedStateCountsEveryAttachedHeap()
    {
        GroupedKnnState state = new GroupedKnnState();
        state.ensureCapacity(4);
        long empty = state.getEstimatedSize();

        KnnHeap first = new KnnHeap(8, false);
        state.setGroupId(0);
        state.setHeap(first);
        assertThat(state.getEstimatedSize()).isEqualTo(empty + first.estimatedSizeInBytes());

        KnnHeap second = new KnnHeap(32, false);
        state.setGroupId(3);
        state.setHeap(second);
        assertThat(state.getEstimatedSize())
                .isEqualTo(empty + first.estimatedSizeInBytes() + second.estimatedSizeInBytes());
    }

    @Test
    public void testGroupedStateGrowsWithCapacity()
    {
        GroupedKnnState state = new GroupedKnnState();
        state.ensureCapacity(16);
        long small = state.getEstimatedSize();

        state.ensureCapacity(4096);

        assertThat(state.getEstimatedSize()).isGreaterThan(small);
    }

    /**
     * {@code deserialize} attaches a brand new heap to a group that may already hold one, so the
     * replaced heap must stop being counted.
     */
    @Test
    public void testGroupedStateStopsCountingAReplacedHeap()
    {
        GroupedKnnState state = new GroupedKnnState();
        state.ensureCapacity(1);
        long empty = state.getEstimatedSize();

        state.setGroupId(0);
        state.setHeap(new KnnHeap(1024, false));
        KnnHeap replacement = new KnnHeap(2, false);
        state.setHeap(replacement);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + replacement.estimatedSizeInBytes());
    }

    /**
     * {@code HashAggregationOperator.addInput} calls {@code updateMemory()} once per input page,
     * which walks {@code getEstimatedSize()} of every grouped aggregator, so a size computation
     * that is linear in the group count makes the whole aggregation quadratic in the input.
     * <p>
     * The budget below is deliberately crude: a constant-time implementation needs microseconds
     * for these calls, while walking {@code GROUP_COUNT} entries {@code CALLS} times is over
     * 4e10 reads spanning tens of megabytes, which no machine finishes within a second.
     */
    @Test
    public void testGroupedStateSizeDoesNotWalkTheGroups()
    {
        int groupCount = 1 << 23;
        int calls = 5_000;

        GroupedKnnState state = new GroupedKnnState();
        state.ensureCapacity(groupCount);
        state.setGroupId(groupCount - 1);
        state.setHeap(new KnnHeap(16, false));

        long start = System.nanoTime();
        long total = 0;
        for (int i = 0; i < calls; i++) {
            total += state.getEstimatedSize();
        }
        Duration elapsed = Duration.ofNanos(System.nanoTime() - start);

        assertThat(total).isEqualTo(calls * state.getEstimatedSize());
        assertThat(elapsed).isLessThan(Duration.ofSeconds(1));
    }
}
