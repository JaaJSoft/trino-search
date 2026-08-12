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

import io.airlift.slice.Slice;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

/**
 * Distances between two one-bit-per-component vectors.
 * <p>
 * There is no bounds argument: a binary code needs nothing beyond itself to be interpreted, since
 * the threshold that produced it is already spent and every component is -1 or +1.
 */
public final class BinaryDistanceFunctions
{
    private BinaryDistanceFunctions() {}

    @Description("Counts the components that differ between two binary vectors")
    @ScalarFunction("hamming_distance")
    @SqlType(StandardTypes.BIGINT)
    public static long hammingDistance(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return BinaryCodes.hamming(first, second);
    }

    @Description("Calculates the euclidean distance between two binary vectors")
    @ScalarFunction("euclidean_distance")
    @SqlType(StandardTypes.DOUBLE)
    public static double euclideanDistance(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return BinaryVectorMath.euclidean(first, second);
    }

    @Description("Calculates the squared euclidean distance between two binary vectors")
    @ScalarFunction("euclidean_squared_distance")
    @SqlType(StandardTypes.DOUBLE)
    public static double euclideanSquaredDistance(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return BinaryVectorMath.euclideanSquared(first, second);
    }

    @Description("Calculates the manhattan distance between two binary vectors")
    @ScalarFunction("manhattan_distance")
    @SqlType(StandardTypes.DOUBLE)
    public static double manhattanDistance(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return BinaryVectorMath.manhattan(first, second);
    }

    @Description("Calculates the dot product between two binary vectors")
    @ScalarFunction("dot_product")
    @SqlType(StandardTypes.DOUBLE)
    public static double dotProduct(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return BinaryVectorMath.dotProduct(first, second);
    }

    @Description("Calculates the cosine similarity between two binary vectors")
    @ScalarFunction("cosine_similarity")
    @SqlType(StandardTypes.DOUBLE)
    public static double cosineSimilarity(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return BinaryVectorMath.cosineSimilarity(first, second);
    }

    @Description("Calculates the cosine distance between two binary vectors")
    @ScalarFunction("cosine_distance")
    @SqlType(StandardTypes.DOUBLE)
    public static double cosineDistance(
            @SqlType(StandardTypes.VARBINARY) Slice first,
            @SqlType(StandardTypes.VARBINARY) Slice second)
    {
        return 1.0 - BinaryVectorMath.cosineSimilarity(first, second);
    }
}
