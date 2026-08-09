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

import dev.jaaj.trino.search.vector.Metric;
import io.trino.spi.block.ValueBlock;
import io.trino.spi.function.AccumulatorState;
import io.trino.spi.function.AccumulatorStateMetadata;

@AccumulatorStateMetadata(
        stateFactoryClass = KnnStateFactory.class,
        stateSerializerClass = KnnStateSerializer.class,
        typeParameters = "K",
        serializedType = "ROW(BIGINT, VARCHAR, ARRAY(ROW(K, DOUBLE)))")
public interface KnnState
        extends AccumulatorState
{
    /**
     * The attached heap, to read from or to hand to another state's {@link #mergeIntoHeap}. Never
     * to add to: a heap retains the keys it keeps, so its size moves on every candidate, and the
     * grouped implementation cannot afford to recompute that by walking the groups. Growing a
     * heap through this reference leaves the size it reports stale, which is the number Trino
     * kills a query on. Go through {@link #addToHeap} and {@link #mergeIntoHeap} instead.
     */
    KnnHeap getHeap();

    void setHeap(KnnHeap heap);

    void addToHeap(ValueBlock keyBlock, int position, double distance);

    void mergeIntoHeap(KnnHeap other);

    int getK();

    void setK(int k);

    Metric getMetric();

    void setMetric(Metric metric);
}
