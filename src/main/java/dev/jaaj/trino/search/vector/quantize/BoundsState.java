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
package dev.jaaj.trino.search.vector.quantize;

import io.trino.spi.function.AccumulatorState;
import io.trino.spi.function.AccumulatorStateMetadata;

@AccumulatorStateMetadata(
        stateFactoryClass = BoundsStateFactory.class,
        stateSerializerClass = BoundsStateSerializer.class,
        serializedType = "ROW(ARRAY(DOUBLE), ARRAY(DOUBLE))")
public interface BoundsState
        extends AccumulatorState
{
    /**
     * The running per-dimension minima, or null before the first vector of this group. Read only:
     * every mutation goes through {@link #accumulate} or {@link #merge} so that the reported size
     * stays in step with the arrays.
     */
    double[] getMinimums();

    double[] getMaximums();

    /**
     * Widens this group's extremes to cover {@code vector}, allocating the arrays on the first
     * call.
     */
    void accumulate(double[] vector);

    void merge(double[] otherMinimums, double[] otherMaximums);
}
