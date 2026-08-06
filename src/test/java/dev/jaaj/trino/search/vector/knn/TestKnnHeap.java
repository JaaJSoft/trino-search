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

import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class TestKnnHeap
{
    private static final int SOURCE_PAGE_BYTES = 64 * 1024;

    private static List<Object> keysOf(KnnHeap heap)
    {
        return heap.drainSorted().stream().map(KnnHeap.Neighbour::key).toList();
    }

    /**
     * What {@code VariableWidthBlock.getSlice} hands to the aggregation: a window over the whole
     * page the key was read from, not a standalone buffer.
     */
    private static Slice keyViewingASourcePage(String value)
    {
        Slice page = Slices.allocate(SOURCE_PAGE_BYTES);
        Slice key = Slices.utf8Slice(value);
        page.setBytes(0, key);
        return page.slice(0, key.length());
    }

    @Test
    public void testKeepsTheKSmallestDistances()
    {
        KnnHeap heap = new KnnHeap(3, false);
        heap.add("e", 5.0);
        heap.add("a", 1.0);
        heap.add("d", 4.0);
        heap.add("b", 2.0);
        heap.add("c", 3.0);

        assertThat(keysOf(heap)).containsExactly("a", "b", "c");
    }

    @Test
    public void testKeepsTheKLargestWhenHigherIsCloser()
    {
        KnnHeap heap = new KnnHeap(2, true);
        heap.add("a", 1.0);
        heap.add("e", 5.0);
        heap.add("c", 3.0);

        assertThat(keysOf(heap)).containsExactly("e", "c");
    }

    @Test
    public void testInsertionOrderDoesNotMatter()
    {
        KnnHeap ascending = new KnnHeap(3, false);
        KnnHeap descending = new KnnHeap(3, false);
        for (int i = 0; i < 20; i++) {
            ascending.add(i, i);
            descending.add(19 - i, 19 - i);
        }

        assertThat(keysOf(ascending)).isEqualTo(keysOf(descending));
    }

    @Test
    public void testFewerElementsThanK()
    {
        KnnHeap heap = new KnnHeap(10, false);
        heap.add("a", 1.0);
        heap.add("b", 2.0);

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
        heap.add(null, 1.0);
        heap.add("b", 2.0);

        assertThat(keysOf(heap)).containsExactly(null, "b");
    }

    @Test
    public void testMergeKeepsTheBestAcrossBothHeaps()
    {
        KnnHeap left = new KnnHeap(3, false);
        left.add("a", 1.0);
        left.add("d", 4.0);
        left.add("f", 6.0);

        KnnHeap right = new KnnHeap(3, false);
        right.add("b", 2.0);
        right.add("c", 3.0);
        right.add("e", 5.0);

        left.mergeFrom(right);

        assertThat(keysOf(left)).containsExactly("a", "b", "c");
    }

    @Test
    public void testMergeWithAnEmptyHeapChangesNothing()
    {
        KnnHeap left = new KnnHeap(2, false);
        left.add("a", 1.0);
        left.mergeFrom(new KnnHeap(2, false));

        assertThat(keysOf(left)).containsExactly("a");
    }

    @Test
    public void testDrainDoesNotMutate()
    {
        KnnHeap heap = new KnnHeap(2, false);
        heap.add("a", 1.0);
        heap.add("b", 2.0);

        assertThat(keysOf(heap)).containsExactly("a", "b");
        assertThat(keysOf(heap)).containsExactly("a", "b");
    }

    @Test
    public void testSliceKeysDoNotRetainTheirSourcePage()
    {
        KnnHeap heap = new KnnHeap(1, false);
        heap.add(keyViewingASourcePage("a"), 1.0);

        Slice stored = (Slice) heap.drainSorted().getFirst().key();

        assertThat(stored).isEqualTo(Slices.utf8Slice("a"));
        assertThat(stored.getRetainedSize()).isLessThan(SOURCE_PAGE_BYTES);
    }

    @Test
    public void testEstimatedSizeCountsRetainedSliceKeys()
    {
        KnnHeap heap = new KnnHeap(4, false);
        long empty = heap.estimatedSizeInBytes();
        heap.add(Slices.utf8Slice("x".repeat(1024)), 1.0);

        assertThat(heap.estimatedSizeInBytes()).isGreaterThanOrEqualTo(empty + 1024);
    }

    @Test
    public void testEstimatedSizeCountsOnlyTheKeysStillInTheHeap()
    {
        KnnHeap heap = new KnnHeap(1, false);
        long empty = heap.estimatedSizeInBytes();
        for (int i = 0; i < 100; i++) {
            heap.add(Slices.utf8Slice("x".repeat(1024)), 100.0 - i);
        }

        assertThat(heap.estimatedSizeInBytes())
                .isGreaterThanOrEqualTo(empty + 1024)
                .isLessThan(empty + 2048);
    }

    @Test
    public void testMatchesABruteForceSortOnRandomData()
    {
        int k = 7;
        KnnHeap heap = new KnnHeap(k, false);
        java.util.Random random = new java.util.Random(42);
        java.util.List<Double> all = new java.util.ArrayList<>();
        for (int i = 0; i < 500; i++) {
            double distance = random.nextDouble() * 1000;
            all.add(distance);
            heap.add(i, distance);
        }

        List<Double> expected = all.stream().sorted().limit(k).toList();
        List<Double> actual = heap.drainSorted().stream().map(KnnHeap.Neighbour::distance).toList();

        assertThat(actual).isEqualTo(expected);
    }
}
