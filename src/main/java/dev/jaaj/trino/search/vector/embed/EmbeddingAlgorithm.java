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
import io.airlift.slice.SliceUtf8;
import io.trino.spi.TrinoException;

import java.util.Locale;
import java.util.stream.Stream;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * How a text is cut into the tokens that get hashed. Every constant receives text that has
 * already been lowercased, so the cost is paid once per row rather than once per branch.
 */
enum EmbeddingAlgorithm
{
    WORD("word") {
        @Override
        void forEachToken(Slice text, TokenSink sink)
        {
            int length = text.length();
            int position = 0;
            while (position < length) {
                if (!Character.isLetterOrDigit(SliceUtf8.getCodePointAt(text, position))) {
                    position += SliceUtf8.lengthOfCodePoint(text, position);
                    continue;
                }
                int start = position;
                while (position < length && Character.isLetterOrDigit(SliceUtf8.getCodePointAt(text, position))) {
                    position += SliceUtf8.lengthOfCodePoint(text, position);
                }
                sink.accept(text, start, position - start);
            }
        }
    };

    private final String sqlName;

    EmbeddingAlgorithm(String sqlName)
    {
        this.sqlName = sqlName;
    }

    /**
     * @param text lowercased UTF-8. The walk assumes well-formed input, which holds because the
     *         only caller passes the output of {@link SliceUtf8#toLowerCase}, itself always
     *         well-formed.
     */
    abstract void forEachToken(Slice text, TokenSink sink);

    String sqlName()
    {
        return sqlName;
    }

    static EmbeddingAlgorithm fromName(Slice name)
    {
        return fromName(name.toStringUtf8());
    }

    static EmbeddingAlgorithm fromName(String name)
    {
        String normalized = name.toLowerCase(Locale.ROOT);
        for (EmbeddingAlgorithm algorithm : values()) {
            if (algorithm.sqlName.equals(normalized)) {
                return algorithm;
            }
        }
        throw new TrinoException(INVALID_FUNCTION_ARGUMENT,
                "Unknown embedding algorithm '%s', expected one of: %s".formatted(
                        name,
                        Stream.of(values()).map(algorithm -> algorithm.sqlName).toList()));
    }
}
