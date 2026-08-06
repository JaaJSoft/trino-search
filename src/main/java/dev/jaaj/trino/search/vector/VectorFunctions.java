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

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;

public final class VectorFunctions
{
    private VectorFunctions() {}

    @Description("Calculates the euclidean norm of a vector")
    @ScalarFunction("l2_norm")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double l2Norm(@SqlType("array(double)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        return VectorMath.norm(vector, DOUBLE_READER);
    }

    @Description("Scales a vector to unit norm")
    @ScalarFunction("normalize_vector")
    @SqlType("array(double)")
    @SqlNullable
    public static Block normalizeVector(@SqlType("array(double)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        double norm = VectorMath.norm(vector, DOUBLE_READER);
        if (norm == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }

        BlockBuilder output = DOUBLE.createFixedSizeBlockBuilder(vector.getPositionCount());
        for (int i = 0; i < vector.getPositionCount(); i++) {
            DOUBLE.writeDouble(output, DOUBLE_READER.read(vector, i) / norm);
        }
        return output.build();
    }

    @Description("Calculates the euclidean norm of a vector")
    @ScalarFunction("l2_norm")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double l2NormReal(@SqlType("array(real)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        return VectorMath.norm(vector, REAL_READER);
    }

    @Description("Scales a vector to unit norm")
    @ScalarFunction("normalize_vector")
    @SqlType("array(real)")
    @SqlNullable
    public static Block normalizeVectorReal(@SqlType("array(real)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        double norm = VectorMath.norm(vector, REAL_READER);
        if (norm == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }

        BlockBuilder output = REAL.createFixedSizeBlockBuilder(vector.getPositionCount());
        for (int i = 0; i < vector.getPositionCount(); i++) {
            REAL.writeFloat(output, (float) (REAL_READER.read(vector, i) / norm));
        }
        return output.build();
    }
}
