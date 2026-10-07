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
 * The nodes one layer search has already scored.
 * <p>
 * A build runs one search per node and level, so clearing a set sized to the graph before each of
 * them would cost more than the searches themselves. Each node is marked with the generation that
 * visited it instead, and clearing is moving to the next generation.
 */
final class VisitedNodes
{
    private final int[] generations;
    private int generation;

    VisitedNodes(int nodeCount)
    {
        this.generations = new int[nodeCount];
    }

    void clear()
    {
        generation++;
        if (generation == 0) {
            Arrays.fill(generations, 0);
            generation = 1;
        }
    }

    /**
     * Marks {@code node} as visited and returns whether it was not already.
     */
    boolean visit(int node)
    {
        if (generations[node] == generation) {
            return false;
        }
        generations[node] = generation;
        return true;
    }
}
