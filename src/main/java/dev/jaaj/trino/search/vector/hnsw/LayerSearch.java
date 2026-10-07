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

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.VectorReader;
import dev.jaaj.trino.search.vector.hnsw.NodeQueue.ScoredNodes;
import io.trino.spi.block.Block;

/**
 * The greedy best-first search over one layer of an HNSW graph (algorithm 2 of Malkov and
 * Yashunin), shared by insertion while a graph is built and by a query against a finished one.
 * <p>
 * Every distance is handled as a rank, lower meaning closer. For the distance metrics the rank is
 * the metric value itself; for dot product, where a higher value is closer, it is the negated value.
 * Negation is exact, so the value handed back to the user is recovered bit for bit.
 */
final class LayerSearch
{
    private final GraphLayers graph;
    private final Metric metric;
    private final VectorReader reader;
    private final VisitedNodes visited;
    private final int[] neighbours;
    private final NodeQueue candidates = NodeQueue.nearestFirst(64);

    LayerSearch(GraphLayers graph, Metric metric, VectorReader reader, int maxNeighbours)
    {
        this.graph = graph;
        this.metric = metric;
        this.reader = reader;
        this.visited = new VisitedNodes(graph.nodeCount());
        this.neighbours = new int[maxNeighbours];
    }

    static double toRank(Metric metric, double value)
    {
        return metric.higherIsCloser() ? -value : value;
    }

    static double toValue(Metric metric, double rank)
    {
        return metric.higherIsCloser() ? -rank : rank;
    }

    /**
     * The rank of {@code node} against {@code query}, or, once it is known not to beat
     * {@code limit}, some rank that does not beat it either.
     */
    double rank(int node, Block query, double limit)
    {
        return toRank(metric, metric.computeBounded(graph.vector(node), query, reader, toValue(metric, limit)));
    }

    /**
     * The {@code ef} nodes nearest to {@code query} on {@code level} that the search reaches from
     * {@code entryPoints}, nearest first.
     */
    ScoredNodes search(Block query, ScoredNodes entryPoints, int ef, int level)
    {
        visited.clear();
        candidates.clear();
        NodeQueue found = NodeQueue.furthestFirst(ef + 1);
        for (int i = 0; i < entryPoints.count(); i++) {
            int node = entryPoints.node(i);
            visited.visit(node);
            candidates.push(node, entryPoints.rank(i));
            found.push(node, entryPoints.rank(i));
            if (found.size() > ef) {
                found.pop();
            }
        }

        while (candidates.size() > 0) {
            // Every remaining candidate is at least as far as this one, so once it cannot improve a
            // full result set, none of them can.
            if (found.size() >= ef && Double.compare(candidates.peekRank(), found.peekRank()) > 0) {
                break;
            }
            int current = candidates.pop();
            int count = graph.neighbours(current, level, neighbours);
            for (int i = 0; i < count; i++) {
                int neighbour = neighbours[i];
                if (!visited.visit(neighbour)) {
                    continue;
                }
                boolean full = found.size() >= ef;
                double limit = full ? found.peekRank() : Double.POSITIVE_INFINITY;
                double rank = rank(neighbour, query, limit);
                // Strict, because a candidate abandoned part way through its distance comes back
                // with a rank that cannot beat the limit but may equal it.
                if (!full || Double.compare(rank, limit) < 0) {
                    candidates.push(neighbour, rank);
                    found.push(neighbour, rank);
                    if (found.size() > ef) {
                        found.pop();
                    }
                }
            }
        }
        return found.drainNearestFirst();
    }
}
