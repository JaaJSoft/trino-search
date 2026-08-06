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
    KnnHeap getHeap();

    void setHeap(KnnHeap heap);

    int getK();

    void setK(int k);

    String getMetricName();

    void setMetricName(String metricName);
}
