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
 */
public final class KnnHeap
{
    private static final long INSTANCE_SIZE = instanceSize(KnnHeap.class);

    public record Neighbour(Object key, double distance) {}

    private final int k;
    private final boolean higherIsCloser;
    private final double[] distances;
    private final Object[] keys;
    private int size;

    public KnnHeap(int k, boolean higherIsCloser)
    {
        this.k = k;
        this.higherIsCloser = higherIsCloser;
        this.distances = new double[k];
        this.keys = new Object[k];
    }

    public int size()
    {
        return size;
    }

    public void add(Object key, double distance)
    {
        if (size < k) {
            distances[size] = distance;
            keys[size] = key;
            siftUp(size);
            size++;
            return;
        }
        if (isCloser(distance, distances[0])) {
            distances[0] = distance;
            keys[0] = key;
            siftDown(0);
        }
    }

    public void mergeFrom(KnnHeap other)
    {
        for (int i = 0; i < other.size; i++) {
            add(other.keys[i], other.distances[i]);
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
        return INSTANCE_SIZE + sizeOf(distances) + sizeOf(keys);
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
        Object key = keys[first];
        keys[first] = keys[second];
        keys[second] = key;
    }
}
