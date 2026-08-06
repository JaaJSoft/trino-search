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
package dev.jaaj.trino.search.vector;

import io.trino.spi.block.Block;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;

public final class VectorDistanceFunctions
{
    private VectorDistanceFunctions() {}

    @Description("Calculates the squared euclidean distance between two vectors")
    @ScalarFunction("euclidean_squared_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanSquaredDistance(@SqlType("array(double)") Block first, @SqlType("array(double)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.euclideanSquared(first, second, DOUBLE_READER);
    }

    @Description("Calculates the manhattan distance between two vectors")
    @ScalarFunction("manhattan_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double manhattanDistance(@SqlType("array(double)") Block first, @SqlType("array(double)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.manhattan(first, second, DOUBLE_READER);
    }

    @Description("Calculates the squared euclidean distance between two vectors")
    @ScalarFunction("euclidean_squared_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanSquaredDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.euclideanSquared(first, second, REAL_READER);
    }

    @Description("Calculates the manhattan distance between two vectors")
    @ScalarFunction("manhattan_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double manhattanDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.manhattan(first, second, REAL_READER);
    }

    @Description("Calculates the euclidean distance between two vectors")
    @ScalarFunction("euclidean_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.euclidean(first, second, REAL_READER);
    }

    @Description("Calculates the dot product between two vectors")
    @ScalarFunction("dot_product")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double dotProductReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.dotProduct(first, second, REAL_READER);
    }

    @Description("Calculates the cosine similarity between two vectors")
    @ScalarFunction("cosine_similarity")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double cosineSimilarityReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.cosineSimilarity(first, second, REAL_READER);
    }

    @Description("Calculates the cosine distance between two vectors")
    @ScalarFunction("cosine_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double cosineDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        Double similarity = cosineSimilarityReal(first, second);
        if (similarity == null) {
            return null;
        }
        return 1.0 - similarity;
    }
}
