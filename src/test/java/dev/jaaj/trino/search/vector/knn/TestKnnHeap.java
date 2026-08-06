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

import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.block.ValueBlock;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.VarcharType.VARCHAR;
import static org.assertj.core.api.Assertions.assertThat;

public class TestKnnHeap
{
    /**
     * Wide enough that a key still viewing its source page reports a retained size an order of
     * magnitude above its own.
     */
    private static final int PAGE_POSITIONS = 1024;
    private static final String PADDING = "x".repeat(512);

    private static ValueBlock varcharKey(String value)
    {
        BlockBuilder builder = VARCHAR.createBlockBuilder(null, 1);
        if (value == null) {
            builder.appendNull();
        }
        else {
            VARCHAR.writeString(builder, value);
        }
        return builder.buildValueBlock();
    }

    private static void add(KnnHeap heap, String key, double distance)
    {
        heap.add(varcharKey(key), 0, distance);
    }

    private static void add(KnnHeap heap, long key, double distance)
    {
        BlockBuilder builder = BIGINT.createBlockBuilder(null, 1);
        BIGINT.writeLong(builder, key);
        heap.add(builder.buildValueBlock(), 0, distance);
    }

    private static List<String> keysOf(KnnHeap heap)
    {
        List<String> keys = new ArrayList<>();
        for (KnnHeap.Neighbour neighbour : heap.drainSorted()) {
            keys.add(neighbour.key().isNull(0) ? null : VARCHAR.getSlice(neighbour.key(), 0).toStringUtf8());
        }
        return keys;
    }

    private static List<Long> longKeysOf(KnnHeap heap)
    {
        return heap.drainSorted().stream().map(neighbour -> BIGINT.getLong(neighbour.key(), 0)).toList();
    }

    private static ValueBlock varcharPage()
    {
        BlockBuilder builder = VARCHAR.createBlockBuilder(null, PAGE_POSITIONS);
        for (int i = 0; i < PAGE_POSITIONS; i++) {
            VARCHAR.writeString(builder, i + PADDING);
        }
        return builder.buildValueBlock();
    }

    private static ValueBlock arrayPage()
    {
        ArrayBlockBuilder builder = (ArrayBlockBuilder) new ArrayType(BIGINT).createBlockBuilder(null, PAGE_POSITIONS);
        for (int i = 0; i < PAGE_POSITIONS; i++) {
            long first = i;
            builder.buildEntry(elements -> {
                for (int j = 0; j < 64; j++) {
                    BIGINT.writeLong(elements, first + j);
                }
            });
        }
        return builder.buildValueBlock();
    }

    private static ValueBlock rowPage()
    {
        RowType rowType = RowType.anonymous(List.of(BIGINT, VARCHAR));
        RowBlockBuilder builder = (RowBlockBuilder) rowType.createBlockBuilder(null, PAGE_POSITIONS);
        for (int i = 0; i < PAGE_POSITIONS; i++) {
            long first = i;
            builder.buildEntry(fields -> {
                BIGINT.writeLong(fields.get(0), first);
                VARCHAR.writeString(fields.get(1), first + PADDING);
            });
        }
        return builder.buildValueBlock();
    }

    /**
     * The failure this pins down is invisible in the results: a key kept by reference still reads
     * back correctly, it just drags every other position of its page along and reports none of
     * that weight to the memory tracker, so the query outlives its limit and the worker dies
     * instead.
     */
    private static void assertKeyIsDetachedFromItsPage(ValueBlock page)
    {
        KnnHeap heap = new KnnHeap(1, false);
        long empty = heap.estimatedSizeInBytes();
        heap.add(page, 0, 1.0);

        ValueBlock stored = heap.drainSorted().getFirst().key();

        assertThat(stored.getPositionCount()).isEqualTo(1);
        assertThat(stored.getRetainedSizeInBytes()).isLessThan(page.getRetainedSizeInBytes() / 8);
        assertThat(heap.estimatedSizeInBytes()).isGreaterThanOrEqualTo(empty + stored.getRetainedSizeInBytes());
    }

    @Test
    public void testKeepsTheKSmallestDistances()
    {
        KnnHeap heap = new KnnHeap(3, false);
        add(heap, "e", 5.0);
        add(heap, "a", 1.0);
        add(heap, "d", 4.0);
        add(heap, "b", 2.0);
        add(heap, "c", 3.0);

        assertThat(keysOf(heap)).containsExactly("a", "b", "c");
    }

    @Test
    public void testKeepsTheKLargestWhenHigherIsCloser()
    {
        KnnHeap heap = new KnnHeap(2, true);
        add(heap, "a", 1.0);
        add(heap, "e", 5.0);
        add(heap, "c", 3.0);

        assertThat(keysOf(heap)).containsExactly("e", "c");
    }

    @Test
    public void testInsertionOrderDoesNotMatter()
    {
        KnnHeap ascending = new KnnHeap(3, false);
        KnnHeap descending = new KnnHeap(3, false);
        for (int i = 0; i < 20; i++) {
            add(ascending, i, i);
            add(descending, 19 - i, 19 - i);
        }

        assertThat(longKeysOf(ascending)).isEqualTo(longKeysOf(descending));
    }

    @Test
    public void testFewerElementsThanK()
    {
        KnnHeap heap = new KnnHeap(10, false);
        add(heap, "a", 1.0);
        add(heap, "b", 2.0);

        assertThat(heap.size()).isEqualTo(2);
        assertThat(keysOf(heap)).containsExactly("a", "b");
    }

    @Test
    public void testEmptyHeap()
    {
        KnnHeap heap = new KnnHeap(5, false);

        assertThat(heap.size()).isZero();
        assertThat(heap.drainSorted()).isEmpty();
    }

    @Test
    public void testNullKeysAreKept()
    {
        KnnHeap heap = new KnnHeap(2, false);
        add(heap, (String) null, 1.0);
        add(heap, "b", 2.0);

        assertThat(keysOf(heap)).containsExactly(null, "b");
    }

    @Test
    public void testMergeKeepsTheBestAcrossBothHeaps()
    {
        KnnHeap left = new KnnHeap(3, false);
        add(left, "a", 1.0);
        add(left, "d", 4.0);
        add(left, "f", 6.0);

        KnnHeap right = new KnnHeap(3, false);
        add(right, "b", 2.0);
        add(right, "c", 3.0);
        add(right, "e", 5.0);

        left.mergeFrom(right);

        assertThat(keysOf(left)).containsExactly("a", "b", "c");
    }

    @Test
    public void testMergeWithAnEmptyHeapChangesNothing()
    {
        KnnHeap left = new KnnHeap(2, false);
        add(left, "a", 1.0);
        left.mergeFrom(new KnnHeap(2, false));

        assertThat(keysOf(left)).containsExactly("a");
    }

    @Test
    public void testMergeCountsTheKeysItTakesOver()
    {
        KnnHeap right = new KnnHeap(1, false);
        add(right, PADDING, 1.0);

        KnnHeap left = new KnnHeap(1, false);
        long empty = left.estimatedSizeInBytes();
        left.mergeFrom(right);

        assertThat(left.estimatedSizeInBytes()).isGreaterThanOrEqualTo(empty + PADDING.length());
    }

    @Test
    public void testDrainDoesNotMutate()
    {
        KnnHeap heap = new KnnHeap(2, false);
        add(heap, "a", 1.0);
        add(heap, "b", 2.0);

        assertThat(keysOf(heap)).containsExactly("a", "b");
        assertThat(keysOf(heap)).containsExactly("a", "b");
    }

    @Test
    public void testVarcharKeysDoNotRetainTheirSourcePage()
    {
        assertKeyIsDetachedFromItsPage(varcharPage());
    }

    @Test
    public void testArrayKeysDoNotRetainTheirSourcePage()
    {
        assertKeyIsDetachedFromItsPage(arrayPage());
    }

    @Test
    public void testRowKeysDoNotRetainTheirSourcePage()
    {
        assertKeyIsDetachedFromItsPage(rowPage());
    }

    @Test
    public void testDetachedKeysKeepTheirValue()
    {
        KnnHeap heap = new KnnHeap(1, false);
        heap.add(rowPage(), 7, 1.0);

        ValueBlock stored = heap.drainSorted().getFirst().key();
        List<Object> fields = List.copyOf((List<?>) RowType.anonymous(List.of(BIGINT, VARCHAR)).getObjectValue(stored, 0));

        assertThat(fields).containsExactly(7L, 7 + PADDING);
    }

    @Test
    public void testEstimatedSizeCountsRetainedVarcharKeys()
    {
        KnnHeap heap = new KnnHeap(4, false);
        long empty = heap.estimatedSizeInBytes();
        add(heap, "x".repeat(1024), 1.0);

        assertThat(heap.estimatedSizeInBytes()).isGreaterThanOrEqualTo(empty + 1024);
    }

    @Test
    public void testEstimatedSizeCountsOnlyTheKeysStillInTheHeap()
    {
        KnnHeap heap = new KnnHeap(1, false);
        long empty = heap.estimatedSizeInBytes();
        for (int i = 0; i < 100; i++) {
            add(heap, "x".repeat(1024), 100.0 - i);
        }

        assertThat(heap.estimatedSizeInBytes())
                .isGreaterThanOrEqualTo(empty + 1024)
                .isLessThan(empty + 2048);
    }

    @Test
    public void testEstimatedSizeStaysBoundedForObjectKeys()
    {
        ValueBlock page = arrayPage();
        KnnHeap heap = new KnnHeap(4, false);
        for (int i = 0; i < PAGE_POSITIONS; i++) {
            heap.add(page, i, PAGE_POSITIONS - i);
        }

        assertThat(heap.estimatedSizeInBytes()).isLessThan(page.getRetainedSizeInBytes());
    }

    @Test
    public void testMatchesABruteForceSortOnRandomData()
    {
        int k = 7;
        KnnHeap heap = new KnnHeap(k, false);
        Random random = new Random(42);
        List<Double> all = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            double distance = random.nextDouble() * 1000;
            all.add(distance);
            add(heap, i, distance);
        }

        List<Double> expected = all.stream().sorted().limit(k).toList();
        List<Double> actual = heap.drainSorted().stream().map(KnnHeap.Neighbour::distance).toList();

        assertThat(actual).isEqualTo(expected);
    }
}
