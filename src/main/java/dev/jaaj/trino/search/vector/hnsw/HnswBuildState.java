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

import io.trino.spi.block.Block;
import io.trino.spi.function.AccumulatorState;
import io.trino.spi.function.AccumulatorStateMetadata;
import io.trino.spi.type.StandardTypes;

@AccumulatorStateMetadata(
        stateFactoryClass = HnswBuildStateFactory.class,
        stateSerializerClass = HnswBuildStateSerializer.class,
        serializedType = StandardTypes.VARBINARY)
public interface HnswBuildState
        extends AccumulatorState
{
    /**
     * The rows of this group, or null before its first one. Read only: every mutation goes through
     * {@link #add} or {@link #addAll} so that the reported size follows the rows.
     */
    GraphInput getInput();

    void setInput(GraphInput input);

    void add(long key, Block vector);

    void addAll(GraphInput other);
}
