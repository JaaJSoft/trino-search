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

import dev.jaaj.trino.search.vector.VectorReader;
import io.airlift.slice.Slice;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.TinyintType.TINYINT;

/**
 * Encodes a vector against fitted bounds.
 * <p>
 * There is one function name per output type rather than one overloaded name, for the reason
 * {@code EmbeddingFunctions} gives for {@code to_vector_real} and {@code to_vector_double}: Trino
 * resolves overloads on argument types, and these take identical arguments. The canonical name is
 * the Trino type and the alias is the machine representation.
 */
public final class QuantizeFunctions
{
    private QuantizeFunctions() {}

    @Description("Quantises a vector to one signed byte per component against fitted bounds")
    @ScalarFunction(value = "quantize_vector_tinyint", alias = "quantize_vector_int8")
    @SqlType("array(tinyint)")
    @SqlNullable
    public static Block quantizeToTinyint(
            @SqlType("array(double)") Block vector,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow bounds)
    {
        return quantizeToTinyint(vector, bounds, DOUBLE_READER);
    }

    @Description("Quantises a vector to one signed byte per component against fitted bounds")
    @ScalarFunction(value = "quantize_vector_tinyint", alias = "quantize_vector_int8")
    @SqlType("array(tinyint)")
    @SqlNullable
    public static Block quantizeRealToTinyint(
            @SqlType("array(real)") Block vector,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow bounds)
    {
        return quantizeToTinyint(vector, bounds, REAL_READER);
    }

    private static Block quantizeToTinyint(Block vector, SqlRow boundsRow, VectorReader reader)
    {
        QuantizationBounds bounds = QuantizationBounds.of(boundsRow);
        int length = vector.getPositionCount();
        bounds.checkDimension(length);
        if (vector.hasNull()) {
            return null;
        }

        BlockBuilder output = TINYINT.createFixedSizeBlockBuilder(length);
        for (int i = 0; i < length; i++) {
            TINYINT.writeByte(output, bounds.encode(i, reader.read(vector, i)));
        }
        return output.build();
    }

    @Description("Quantises a vector to one bit per component against fitted bounds")
    @ScalarFunction(
            value = "quantize_vector_varbinary",
            alias = {"quantize_vector_binary", "quantize_vector_int1"})
    @SqlType(StandardTypes.VARBINARY)
    @SqlNullable
    public static Slice quantizeToVarbinary(
            @SqlType("array(double)") Block vector,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow bounds)
    {
        return quantizeToVarbinary(vector, bounds, DOUBLE_READER);
    }

    @Description("Quantises a vector to one bit per component against fitted bounds")
    @ScalarFunction(
            value = "quantize_vector_varbinary",
            alias = {"quantize_vector_binary", "quantize_vector_int1"})
    @SqlType(StandardTypes.VARBINARY)
    @SqlNullable
    public static Slice quantizeRealToVarbinary(
            @SqlType("array(real)") Block vector,
            @SqlType(QuantizationBounds.BOUNDS_TYPE_SIGNATURE) SqlRow bounds)
    {
        return quantizeToVarbinary(vector, bounds, REAL_READER);
    }

    /**
     * Sign quantisation about each dimension's own midpoint, which is what keeps the codes centred
     * instead of dominated by whichever side of zero the embedding happens to sit on.
     */
    private static Slice quantizeToVarbinary(Block vector, SqlRow boundsRow, VectorReader reader)
    {
        QuantizationBounds bounds = QuantizationBounds.of(boundsRow);
        int length = vector.getPositionCount();
        bounds.checkDimension(length);
        if (vector.hasNull()) {
            return null;
        }
        return BinaryCodes.pack(length, i -> reader.read(vector, i) >= bounds.offset(i));
    }
}
