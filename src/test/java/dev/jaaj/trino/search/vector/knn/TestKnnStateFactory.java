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
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.ValueBlock;
import io.trino.spi.type.ArrayType;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.VarcharType.VARCHAR;
import static org.assertj.core.api.Assertions.assertThat;

public class TestKnnStateFactory
{
    /**
     * Wide enough that a total which stops at what an empty heap reports is off by orders of
     * magnitude rather than by a rounding difference.
     */
    private static final String WIDE_KEY = "x".repeat(100_000);
    private static final Slice EUCLIDEAN = Slices.utf8Slice("euclidean");
    private static final ArrayType VECTOR_TYPE = new ArrayType(DOUBLE);

    private static ValueBlock varcharKey(String value)
    {
        BlockBuilder builder = VARCHAR.createBlockBuilder(null, 1);
        VARCHAR.writeString(builder, value);
        return builder.buildValueBlock();
    }

    private static Block vector(double... values)
    {
        ArrayBlockBuilder builder = (ArrayBlockBuilder) VECTOR_TYPE.createBlockBuilder(null, 1);
        builder.buildEntry(elements -> {
            for (double value : values) {
                DOUBLE.writeDouble(elements, value);
            }
        });
        return VECTOR_TYPE.getObject(builder.build(), 0);
    }

    private static void input(KnnState state, String key, double coordinate, int k)
    {
        KnnAggregation.OfDoubleVectors.input(
                state, varcharKey(key), 0, vector(coordinate), vector(0.0), k, EUCLIDEAN);
    }

    private static GroupedKnnState singleGroupState()
    {
        GroupedKnnState state = new GroupedKnnState();
        state.ensureCapacity(1);
        state.setGroupId(0);
        return state;
    }

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

    /**
     * The heap of a group is attached on its first row and filled on that row and every later
     * one, so a total that only follows {@code setHeap} freezes the group's contribution at what
     * an empty heap reports and the keys it retains stay invisible to the memory tracker.
     */
    @Test
    public void testGroupedStateCountsKeysAddedAfterTheHeapIsAttached()
    {
        GroupedKnnState state = singleGroupState();
        long empty = state.getEstimatedSize();

        input(state, WIDE_KEY, 1.0, 4);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + state.getHeap().estimatedSizeInBytes());
        assertThat(state.getEstimatedSize()).isGreaterThan(empty + WIDE_KEY.length());
    }

    /**
     * Combining into a group that already holds a heap mutates that heap in place, which is the
     * second way a group grows after its heap was attached.
     */
    @Test
    public void testGroupedStateCountsKeysMergedIntoAnAttachedHeap()
    {
        GroupedKnnState state = singleGroupState();
        long empty = state.getEstimatedSize();
        input(state, "narrow", 1.0, 4);

        SingleKnnState other = new SingleKnnState();
        input(other, WIDE_KEY, 2.0, 4);
        KnnAggregation.OfDoubleVectors.combine(state, other);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + state.getHeap().estimatedSizeInBytes());
        assertThat(state.getEstimatedSize()).isGreaterThan(empty + WIDE_KEY.length());
    }

    /**
     * The third way: {@code deserialize} attaches an empty heap and then fills it from the
     * serialized neighbours, so the whole intermediate state of a group arrives uncounted.
     */
    @Test
    public void testGroupedStateCountsKeysRestoredByDeserialize()
    {
        SingleKnnState source = new SingleKnnState();
        input(source, WIDE_KEY, 1.0, 4);

        KnnStateSerializer serializer = new KnnStateSerializer(VARCHAR);
        BlockBuilder serialized = serializer.getSerializedType().createBlockBuilder(null, 1);
        serializer.serialize(source, serialized);

        GroupedKnnState state = singleGroupState();
        long empty = state.getEstimatedSize();
        serializer.deserialize(serialized.build(), 0, state);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + state.getHeap().estimatedSizeInBytes());
        assertThat(state.getEstimatedSize()).isGreaterThan(empty + WIDE_KEY.length());
    }

    /**
     * The heap is bounded, so a candidate that displaces the incumbent releases its key. A total
     * that only ever adds would drift upwards for the rest of the query.
     */
    @Test
    public void testGroupedStateFollowsTheHeapDownWhenAKeyIsEvicted()
    {
        GroupedKnnState state = singleGroupState();
        long empty = state.getEstimatedSize();
        input(state, WIDE_KEY, 2.0, 1);
        long wide = state.getEstimatedSize();

        input(state, "narrow", 1.0, 1);

        assertThat(state.getEstimatedSize())
                .isLessThan(wide)
                .isEqualTo(empty + state.getHeap().estimatedSizeInBytes());
    }
}
