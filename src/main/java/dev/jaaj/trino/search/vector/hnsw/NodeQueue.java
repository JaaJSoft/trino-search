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
package dev.jaaj.trino.search.vector.hnsw;

import java.util.Arrays;

/**
 * A binary heap of graph nodes keyed on their rank, the internal distance where lower is closer
 * whatever the metric (see {@link LayerSearch#toRank}).
 * <p>
 * Ranks are ordered by {@link Double#compare}, never by {@code <}: an infinite component turns a
 * distance into NaN, and {@code <} is false both ways against NaN, which would leave the heap with
 * no consistent order to maintain. {@code Double.compare} sorts NaN after everything, so such a node
 * is simply the farthest one.
 */
final class NodeQueue
{
    private final boolean furthestFirst;
    private int[] nodes;
    private double[] ranks;
    private int size;

    private NodeQueue(boolean furthestFirst, int capacity)
    {
        this.furthestFirst = furthestFirst;
        this.nodes = new int[Math.max(capacity, 1)];
        this.ranks = new double[nodes.length];
    }

    static NodeQueue nearestFirst(int capacity)
    {
        return new NodeQueue(false, capacity);
    }

    static NodeQueue furthestFirst(int capacity)
    {
        return new NodeQueue(true, capacity);
    }

    int size()
    {
        return size;
    }

    void clear()
    {
        size = 0;
    }

    void push(int node, double rank)
    {
        if (size == nodes.length) {
            nodes = Arrays.copyOf(nodes, size * 2);
            ranks = Arrays.copyOf(ranks, size * 2);
        }
        nodes[size] = node;
        ranks[size] = rank;
        siftUp(size);
        size++;
    }

    int peekNode()
    {
        return nodes[0];
    }

    double peekRank()
    {
        return ranks[0];
    }

    int pop()
    {
        int top = nodes[0];
        size--;
        nodes[0] = nodes[size];
        ranks[0] = ranks[size];
        siftDown(0);
        return top;
    }

    /**
     * Empties the queue into a list ordered nearest first, whichever end of the order this queue
     * pops from.
     */
    ScoredNodes drainNearestFirst()
    {
        int count = size;
        int[] sortedNodes = new int[count];
        double[] sortedRanks = new double[count];
        for (int i = 0; i < count; i++) {
            int slot = furthestFirst ? count - 1 - i : i;
            sortedRanks[slot] = peekRank();
            sortedNodes[slot] = pop();
        }
        return new ScoredNodes(sortedNodes, sortedRanks);
    }

    private boolean above(double rank, double other)
    {
        int comparison = Double.compare(rank, other);
        return furthestFirst ? comparison > 0 : comparison < 0;
    }

    private void siftUp(int start)
    {
        int index = start;
        while (index > 0) {
            int parent = (index - 1) >>> 1;
            if (!above(ranks[index], ranks[parent])) {
                return;
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
            int top = index;
            if (left < size && above(ranks[left], ranks[top])) {
                top = left;
            }
            if (right < size && above(ranks[right], ranks[top])) {
                top = right;
            }
            if (top == index) {
                return;
            }
            swap(index, top);
            index = top;
        }
    }

    private void swap(int first, int second)
    {
        int node = nodes[first];
        nodes[first] = nodes[second];
        nodes[second] = node;
        double rank = ranks[first];
        ranks[first] = ranks[second];
        ranks[second] = rank;
    }

    /**
     * Nodes with their ranks, nearest first.
     */
    static final class ScoredNodes
    {
        private final int[] nodes;
        private final double[] ranks;

        private ScoredNodes(int[] nodes, double[] ranks)
        {
            this.nodes = nodes;
            this.ranks = ranks;
        }

        static ScoredNodes single(int node, double rank)
        {
            return new ScoredNodes(new int[] {node}, new double[] {rank});
        }

        /**
         * Takes ownership of both arrays, which the caller has already sorted nearest first.
         */
        static ScoredNodes ofSorted(int[] nodes, double[] ranks)
        {
            return new ScoredNodes(nodes, ranks);
        }

        int count()
        {
            return nodes.length;
        }

        int node(int index)
        {
            return nodes[index];
        }

        double rank(int index)
        {
            return ranks[index];
        }
    }
}
