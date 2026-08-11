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

import io.trino.spi.block.Block;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

/**
 * Distances between two vectors of one-byte codes.
 * <p>
 * The bounds are the third argument rather than folded into the codes because ranking on the raw
 * codes would be a reweighted metric, with every dimension implicitly scaled by the reciprocal of
 * its own scale. For euclidean that only matters when the scales differ between dimensions; for
 * cosine it is fatal, since cosine is not translation-invariant and a non-zero offset moves the
 * angle.
 * <p>
 * Both operands must have been fitted against the bounds passed. Nothing here can check that.
 */
public final class QuantizedDistanceFunctions
{
    private QuantizedDistanceFunctions() {}

    @Description("Calculates the euclidean distance between two quantised vectors")
    @ScalarFunction("euclidean_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanDistance(
            @SqlType("array(tinyint)") Block first,
            @SqlType("array(tinyint)") Block second,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow)
    {
        QuantizationBounds bounds = prepare(first, second, boundsRow);
        return bounds == null ? null : QuantizedVectorMath.euclidean(first, second, bounds);
    }

    @Description("Calculates the squared euclidean distance between two quantised vectors")
    @ScalarFunction("euclidean_squared_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanSquaredDistance(
            @SqlType("array(tinyint)") Block first,
            @SqlType("array(tinyint)") Block second,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow)
    {
        QuantizationBounds bounds = prepare(first, second, boundsRow);
        return bounds == null ? null : QuantizedVectorMath.euclideanSquared(first, second, bounds);
    }

    @Description("Calculates the manhattan distance between two quantised vectors")
    @ScalarFunction("manhattan_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double manhattanDistance(
            @SqlType("array(tinyint)") Block first,
            @SqlType("array(tinyint)") Block second,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow)
    {
        QuantizationBounds bounds = prepare(first, second, boundsRow);
        return bounds == null ? null : QuantizedVectorMath.manhattan(first, second, bounds);
    }

    @Description("Calculates the dot product between two quantised vectors")
    @ScalarFunction("dot_product")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double dotProduct(
            @SqlType("array(tinyint)") Block first,
            @SqlType("array(tinyint)") Block second,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow)
    {
        QuantizationBounds bounds = prepare(first, second, boundsRow);
        return bounds == null ? null : QuantizedVectorMath.dotProduct(first, second, bounds);
    }

    @Description("Calculates the cosine similarity between two quantised vectors")
    @ScalarFunction("cosine_similarity")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double cosineSimilarity(
            @SqlType("array(tinyint)") Block first,
            @SqlType("array(tinyint)") Block second,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow)
    {
        QuantizationBounds bounds = prepare(first, second, boundsRow);
        return bounds == null ? null : QuantizedVectorMath.cosineSimilarity(first, second, bounds);
    }

    @Description("Calculates the cosine distance between two quantised vectors")
    @ScalarFunction("cosine_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double cosineDistance(
            @SqlType("array(tinyint)") Block first,
            @SqlType("array(tinyint)") Block second,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow boundsRow)
    {
        Double similarity = cosineSimilarity(first, second, boundsRow);
        return similarity == null ? null : 1.0 - similarity;
    }

    /**
     * @return the bounds, or null when a null component makes the whole distance null
     */
    private static QuantizationBounds prepare(Block first, Block second, SqlRow boundsRow)
    {
        QuantizationBounds bounds = QuantizationBounds.of(boundsRow);
        QuantizedVectorMath.checkSameLength(first, second, bounds);
        if (first.hasNull() || second.hasNull()) {
            return null;
        }
        return bounds;
    }
}
