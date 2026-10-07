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

import dev.jaaj.trino.search.vector.VectorReader;
import io.trino.spi.block.Block;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.ArrayType;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;

public final class VectorProjectionFunctions
{
    private static final ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);
    private static final ArrayType REAL_ARRAY = new ArrayType(REAL);

    private VectorProjectionFunctions() {}

    @Description("Returns the dot product of a vector with each of several directions")
    @ScalarFunction("vector_projections")
    @SqlType("array(double)")
    @SqlNullable
    public static Block vectorProjections(
            @SqlType("array(double)") Block vector,
            @SqlType("array(array(double))") Block directions)
    {
        return project(vector, directions, DOUBLE_ARRAY, DOUBLE_READER);
    }

    @Description("Returns the dot product of a vector with each of several directions")
    @ScalarFunction("vector_projections")
    @SqlType("array(double)")
    @SqlNullable
    public static Block vectorProjectionsReal(
            @SqlType("array(real)") Block vector,
            @SqlType("array(array(real))") Block directions)
    {
        return project(vector, directions, REAL_ARRAY, REAL_READER);
    }

    private static Block project(Block vector, Block directions, ArrayType directionType, VectorReader reader)
    {
        if (vector.hasNull()) {
            return null;
        }
        return VectorProjection.project(vector, directions, directionType, reader);
    }
}
