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
package dev.jaaj.trino.search.vector.embed;

import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * Text to vector by feature hashing. Deterministic and self-contained: no model, no vocabulary
 * and no network, so the result of a row depends on that row alone.
 * <p>
 * The two entry points differ only in the width of the output. There is one name per return type
 * because Trino resolves overloads on argument types, and both take the same arguments.
 */
public final class EmbeddingFunctions
{
    private EmbeddingFunctions() {}

    @Description("Embeds text into a unit-norm vector by feature hashing")
    @ScalarFunction(value = "to_vector_real", alias = "to_vector_fp32")
    @SqlType("array(real)")
    public static Block toVectorReal(
            @SqlType(StandardTypes.VARCHAR) Slice text,
            @SqlType(StandardTypes.BIGINT) long dimension,
            @SqlType(StandardTypes.VARCHAR) Slice algorithm)
    {
        return NormalizedVectorBlocks.real(
                HashingEmbedder.accumulate(text, EmbeddingAlgorithm.fromName(algorithm), checkDimension(dimension)));
    }

    @Description("Embeds text into a unit-norm vector by feature hashing")
    @ScalarFunction(value = "to_vector_double", alias = "to_vector_fp64")
    @SqlType("array(double)")
    public static Block toVectorDouble(
            @SqlType(StandardTypes.VARCHAR) Slice text,
            @SqlType(StandardTypes.BIGINT) long dimension,
            @SqlType(StandardTypes.VARCHAR) Slice algorithm)
    {
        return NormalizedVectorBlocks.doubles(
                HashingEmbedder.accumulate(text, EmbeddingAlgorithm.fromName(algorithm), checkDimension(dimension)));
    }

    private static int checkDimension(long dimension)
    {
        if (dimension < 1 || dimension > HashingEmbedder.MAX_DIMENSION) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT,
                    "Vector dimension must be between 1 and %d, got %d".formatted(
                            HashingEmbedder.MAX_DIMENSION, dimension));
        }
        return (int) dimension;
    }
}
