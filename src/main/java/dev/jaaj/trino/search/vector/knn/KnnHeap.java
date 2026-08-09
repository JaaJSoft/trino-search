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

import io.trino.spi.block.ValueBlock;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;

/**
 * Bounded heap keeping the k nearest neighbours seen so far.
 * <p>
 * The root holds the <em>worst</em> retained candidate, so deciding whether a new candidate
 * belongs in the result is a single comparison against position 0.
 * <p>
 * A key is held as a single-position {@link ValueBlock} of its own rather than as the native
 * value {@code TypeUtils.readNativeValue} would return. Those native values are windows over the
 * page they were read from - a {@code Slice} for {@code varchar}, a {@code Block} for
 * {@code array}, a {@code SqlRow} for {@code row}, a {@code SqlMap} for {@code map} - so keeping
 * one alive pins that whole page until the group is flushed, and the accounting below would only
 * see a reference. {@code getSingleValueBlock} is the SPI's copy for every one of those shapes at
 * once, which bounds retention to the key itself and makes {@link #estimatedSizeInBytes()} report
 * what the heap really holds. That number is what Trino kills a query on.
 */
public final class KnnHeap
{
    private static final long INSTANCE_SIZE = instanceSize(KnnHeap.class);

    public record Neighbour(ValueBlock key, double distance) {}

    private final int k;
    private final boolean higherIsCloser;
    private final double[] distances;
    private final ValueBlock[] keys;
    private int size;
    private long retainedKeyBytes;

    public KnnHeap(int k, boolean higherIsCloser)
    {
        this.k = k;
        this.higherIsCloser = higherIsCloser;
        this.distances = new double[k];
        this.keys = new ValueBlock[k];
    }

    public int size()
    {
        return size;
    }

    public void add(ValueBlock keyBlock, int position, double distance)
    {
        // Copying the key is the expensive half of an insertion, and once k neighbours are in,
        // nearly every candidate is dropped. Ranking first leaves those candidates costing a
        // comparison instead of a copy the heap throws away.
        if (!wouldAccept(distance)) {
            return;
        }
        addOwnedKey(keyBlock.getSingleValueBlock(position), distance);
    }

    /**
     * Whether {@link #add} would retain a candidate at this distance, which is the rule
     * {@link #addOwnedKey} applies once the key has been copied.
     */
    boolean wouldAccept(double distance)
    {
        return size < k || isCloser(distance, distances[0]);
    }

    /**
     * The other heap's keys are already single-position blocks it owns, so they move over as they
     * are. Both heaps count them for as long as both are alive, which over-reports rather than
     * under-reports and settles as soon as the merged-from state is dropped.
     */
    public void mergeFrom(KnnHeap other)
    {
        for (int i = 0; i < other.size; i++) {
            addOwnedKey(other.keys[i], other.distances[i]);
        }
    }

    private void addOwnedKey(ValueBlock key, double distance)
    {
        if (size < k) {
            distances[size] = distance;
            keys[size] = key;
            retainedKeyBytes += key.getRetainedSizeInBytes();
            siftUp(size);
            size++;
            return;
        }
        if (isCloser(distance, distances[0])) {
            retainedKeyBytes -= keys[0].getRetainedSizeInBytes();
            retainedKeyBytes += key.getRetainedSizeInBytes();
            distances[0] = distance;
            keys[0] = key;
            siftDown(0);
        }
    }

    public List<Neighbour> drainSorted()
    {
        List<Neighbour> neighbours = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            neighbours.add(new Neighbour(keys[i], distances[i]));
        }
        Comparator<Neighbour> byDistance = Comparator.comparingDouble(Neighbour::distance);
        neighbours.sort(higherIsCloser ? byDistance.reversed() : byDistance);
        return neighbours;
    }

    public long estimatedSizeInBytes()
    {
        return INSTANCE_SIZE + sizeOf(distances) + sizeOf(keys) + retainedKeyBytes;
    }

    private boolean isCloser(double candidate, double incumbent)
    {
        return higherIsCloser ? candidate > incumbent : candidate < incumbent;
    }

    private void siftUp(int start)
    {
        int index = start;
        while (index > 0) {
            int parent = (index - 1) >>> 1;
            if (!isCloser(distances[parent], distances[index])) {
                break;
            }
            swap(index, parent);
            index = parent;
        }
    }

    private void siftDown(int start)
    {
        int index = start;
        while (true) {
            int left = 2 * index + 1;
            int right = left + 1;
            int worst = index;
            if (left < size && !isCloser(distances[left], distances[worst])) {
                worst = left;
            }
            if (right < size && !isCloser(distances[right], distances[worst])) {
                worst = right;
            }
            if (worst == index) {
                return;
            }
            swap(index, worst);
            index = worst;
        }
    }

    private void swap(int first, int second)
    {
        double distance = distances[first];
        distances[first] = distances[second];
        distances[second] = distance;
        ValueBlock key = keys[first];
        keys[first] = keys[second];
        keys[second] = key;
    }
}
