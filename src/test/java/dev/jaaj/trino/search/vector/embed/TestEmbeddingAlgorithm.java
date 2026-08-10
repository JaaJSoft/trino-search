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
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class TestEmbeddingAlgorithm
{
    /**
     * Tokens as strings, for readability. The production path never materialises them: it hashes
     * the byte range in place, which is what {@link TokenSink} exists for.
     */
    private static List<String> tokens(EmbeddingAlgorithm algorithm, String lowercasedText)
    {
        List<String> collected = new ArrayList<>();
        Slice text = Slices.utf8Slice(lowercasedText);
        algorithm.forEachToken(text, (source, offset, length) ->
                collected.add(source.slice(offset, length).toStringUtf8()));
        return collected;
    }

    @Test
    public void testWordSplitsOnPunctuationAndSpace()
    {
        assertThat(tokens(EmbeddingAlgorithm.WORD, "hello, world!")).containsExactly("hello", "world");
    }

    @Test
    public void testWordKeepsDigitsAndSplitsOnUnderscore()
    {
        assertThat(tokens(EmbeddingAlgorithm.WORD, "user_42 id-7")).containsExactly("user", "42", "id", "7");
    }

    @Test
    public void testWordCollapsesRunsOfSeparators()
    {
        assertThat(tokens(EmbeddingAlgorithm.WORD, "  a   ...  b  ")).containsExactly("a", "b");
    }

    @Test
    public void testWordKeepsNonAsciiLetters()
    {
        assertThat(tokens(EmbeddingAlgorithm.WORD, "café, naïve")).containsExactly("café", "naïve");
    }

    /**
     * An astral code point is two UTF-16 chars but one code point. Walking by char rather than by
     * code point would split it into two surrogates, both of which fail isLetterOrDigit, and the
     * emoji would silently become a separator instead of a token.
     */
    @Test
    public void testWordTreatsAstralCodePointAsOneUnit()
    {
        assertThat(tokens(EmbeddingAlgorithm.WORD, "a\uD83D\uDE00b")).containsExactly("a", "b");
    }

    @Test
    public void testWordOnTextWithoutLetterOrDigitYieldsNoToken()
    {
        assertThat(tokens(EmbeddingAlgorithm.WORD, "!!! ...")).isEmpty();
        assertThat(tokens(EmbeddingAlgorithm.WORD, "")).isEmpty();
    }

    @Test
    public void testFromNameIsCaseInsensitive()
    {
        assertThat(EmbeddingAlgorithm.fromName("WORD")).isEqualTo(EmbeddingAlgorithm.WORD);
        assertThat(EmbeddingAlgorithm.fromName(Slices.utf8Slice("word"))).isEqualTo(EmbeddingAlgorithm.WORD);
    }

    @Test
    public void testFromNameRejectsUnknownNameAndListsTheValidOnes()
    {
        assertThatThrownBy(() -> EmbeddingAlgorithm.fromName("bag_of_words"))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("Unknown embedding algorithm 'bag_of_words'")
                .hasMessageContaining("word");
    }

    @Test
    public void testCharNgramSlidesOverEveryCodePointIncludingSpaces()
    {
        assertThat(tokens(EmbeddingAlgorithm.CHAR_3GRAM, "ab cd"))
                .containsExactly("ab ", "b c", " cd");
    }

    @Test
    public void testCharNgramIsSensitiveToOrder()
    {
        assertThat(tokens(EmbeddingAlgorithm.CHAR_3GRAM, "abc"))
                .isNotEqualTo(tokens(EmbeddingAlgorithm.CHAR_3GRAM, "cba"));
    }

    @Test
    public void testCharNgramSizesDiffer()
    {
        assertThat(tokens(EmbeddingAlgorithm.CHAR_4GRAM, "abcde")).containsExactly("abcd", "bcde");
        assertThat(tokens(EmbeddingAlgorithm.CHAR_5GRAM, "abcde")).containsExactly("abcde");
    }

    /**
     * The window is a window over code points, not over bytes. A three-code-point text of
     * multi-byte characters must yield exactly one trigram, spanning all of its bytes.
     */
    @Test
    public void testCharNgramWindowsCodePointsNotBytes()
    {
        assertThat(tokens(EmbeddingAlgorithm.CHAR_3GRAM, "éàü")).containsExactly("éàü");
    }

    @Test
    public void testCharNgramOnTextShorterThanTheWindowYieldsNoToken()
    {
        assertThat(tokens(EmbeddingAlgorithm.CHAR_5GRAM, "abcd")).isEmpty();
        assertThat(tokens(EmbeddingAlgorithm.CHAR_3GRAM, "")).isEmpty();
    }

    @Test
    public void testEveryAlgorithmResolvesByItsSqlName()
    {
        for (EmbeddingAlgorithm algorithm : EmbeddingAlgorithm.values()) {
            assertThat(EmbeddingAlgorithm.fromName(algorithm.sqlName())).isEqualTo(algorithm);
        }
    }
}
