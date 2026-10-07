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

/**
 * The link structure {@link LayerSearch} walks, whether the graph is still being built on the heap
 * or read back from its serialized form.
 */
interface GraphLayers
{
    int nodeCount();

    /**
     * Copies the neighbours of {@code node} on {@code level} into {@code into}, which holds at least
     * {@code 2 * m} entries, and returns how many there are.
     */
    int neighbours(int node, int level, int[] into);
}
