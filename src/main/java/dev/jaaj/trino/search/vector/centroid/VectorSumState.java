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
package dev.jaaj.trino.search.vector.centroid;

import dev.jaaj.trino.search.vector.VectorReader;
import io.trino.spi.block.Block;
import io.trino.spi.function.AccumulatorState;
import io.trino.spi.function.AccumulatorStateMetadata;

/**
 * A running element-wise sum and the number of vectors in it. The sum rather than the mean is
 * kept because that is what merges associatively across splits; the division happens once, at
 * output.
 */
@AccumulatorStateMetadata(
        stateFactoryClass = VectorSumStateFactory.class,
        stateSerializerClass = VectorSumStateSerializer.class,
        serializedType = "ROW(ARRAY(DOUBLE), BIGINT)")
public interface VectorSumState
        extends AccumulatorState
{
    /**
     * The per-dimension sums, or null before the first vector of this group. Read only: every
     * mutation goes through {@link #accumulate} or {@link #merge} so that the reported size stays
     * in step with the arrays.
     */
    double[] getSums();

    long getCount();

    /**
     * Adds {@code vector} into this group's sums, allocating them on the first call. The caller
     * has already rejected a vector with a null component.
     */
    void accumulate(Block vector, VectorReader reader);

    void merge(double[] otherSums, long otherCount);

    /**
     * Discards whatever this group held and takes {@code sums} as its own array. Deserialization
     * needs this rather than {@link #merge}, because Trino deserializes every intermediate
     * position into one reused scratch state before combining it into the target group.
     */
    void replace(double[] sums, long count);
}
