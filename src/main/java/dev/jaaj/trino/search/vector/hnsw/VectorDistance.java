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
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.trino.spi.block.Block;

/**
 * The raw metric value between two stored vectors of one representation, {@code V} being how that
 * representation is handed to its kernels: a {@link Block} for the array ones, a {@link Slice} for
 * binary codes.
 */
@FunctionalInterface
interface VectorDistance<V>
{
    /**
     * The metric value, or, once it is known not to beat {@code limit}, some value that does not
     * beat it either, with the meaning {@link Metric#computeBounded} gives a limit.
     * <p>
     * The second operand is the one that stays the same across consecutive calls, which is the
     * operand cosine over floats remembers the magnitude of.
     */
    double compute(V first, V second, double limit);

    static VectorDistance<Block> floats(Metric metric, VectorReader reader)
    {
        return (first, second, limit) -> metric.computeBounded(first, second, reader, limit);
    }

    static VectorDistance<Block> int8(Metric metric, QuantizationBounds bounds)
    {
        return (first, second, limit) -> metric.computeQuantizedBounded(first, second, bounds, limit);
    }

    /**
     * A binary distance is a single Hamming count, so there is nothing to abandon part way and the
     * limit is ignored.
     */
    static VectorDistance<Slice> binary(Metric metric)
    {
        return (first, second, _) -> metric.computeBinary(first, second);
    }
}
