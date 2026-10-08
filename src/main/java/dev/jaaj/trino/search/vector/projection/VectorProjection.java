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
package dev.jaaj.trino.search.vector.projection;

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.VectorReader;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.type.ArrayType;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;

/**
 * The dot product of one vector with each of several directions, as the elements of an
 * {@code array(double)}.
 */
public final class VectorProjection
{
    private VectorProjection() {}

    /**
     * Element j is the projection onto direction j, or null when that direction is null or holds a
     * null component: the position is how a projection maps back to its column, so a missing one
     * must not shift the others. Directions are used as given, without normalising them.
     * <p>
     * Each direction goes through the same kernel as {@code dot_product}, so the vector is read
     * once per direction. Only the first of those reads reaches memory: a vector of dimension 768
     * is a few kilobytes and is still in the first-level cache for the directions after it.
     */
    public static Block project(Block vector, Block directions, ArrayType directionType, VectorReader reader)
    {
        int count = directions.getPositionCount();
        BlockBuilder output = DOUBLE.createFixedSizeBlockBuilder(count);
        for (int i = 0; i < count; i++) {
            if (directions.isNull(i)) {
                output.appendNull();
                continue;
            }
            Block direction = directionType.getObject(directions, i);
            if (direction.getPositionCount() != vector.getPositionCount()) {
                throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
            }
            if (direction.hasNull()) {
                output.appendNull();
                continue;
            }
            DOUBLE.writeDouble(output, Metric.DOT_PRODUCT.compute(vector, direction, reader));
        }
        return output.build();
    }
}
