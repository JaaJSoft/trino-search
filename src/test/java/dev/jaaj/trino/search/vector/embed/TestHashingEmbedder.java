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

import io.airlift.slice.Slices;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class TestHashingEmbedder
{
    private static double[] embed(String text, EmbeddingAlgorithm algorithm, int dimension)
    {
        return HashingEmbedder.accumulate(Slices.utf8Slice(text), algorithm, dimension);
    }

    private static double sumOfAbsoluteValues(double[] values)
    {
        double total = 0.0;
        for (double value : values) {
            total += Math.abs(value);
        }
        return total;
    }

    /**
     * The index comes from the top 32 bits scaled onto the dimension, so it is a monotone map of
     * those bits onto [0, dimension) with no division. These four cases pin the ends and the
     * middle of that map, which is what a modulo would get wrong in a way property tests would
     * not notice.
     */
    @Test
    public void testIndexSpansTheWholeRange()
    {
        assertThat(HashingEmbedder.indexOf(0L, 256)).isEqualTo(0);
        assertThat(HashingEmbedder.indexOf(0xFFFFFFFF00000000L, 256)).isEqualTo(255);
        assertThat(HashingEmbedder.indexOf(0x8000000000000000L, 256)).isEqualTo(128);
        assertThat(HashingEmbedder.indexOf(0x4000000000000000L, 256)).isEqualTo(64);
    }

    @Test
    public void testIndexIgnoresTheLowBitsThatCarryTheSign()
    {
        assertThat(HashingEmbedder.indexOf(0x00000000FFFFFFFFL, 256)).isEqualTo(0);
    }

    /**
     * A dimension that is not a power of two is the common case, 768 being the usual embedding
     * width, so the reduction must stay in range there too rather than only for masks.
     */
    @Test
    public void testIndexStaysInRangeForEveryDimension()
    {
        for (int dimension : new int[] {1, 3, 768, 1000, HashingEmbedder.MAX_DIMENSION}) {
            for (long hash : new long[] {0L, -1L, 0x8000000000000000L, 0x7FFFFFFFFFFFFFFFL, 123456789L}) {
                assertThat(HashingEmbedder.indexOf(hash, dimension)).isBetween(0, dimension - 1);
            }
        }
    }

    @Test
    public void testSignComesFromBitThirtyOne()
    {
        assertThat(HashingEmbedder.signOf(0L)).isEqualTo(1.0);
        assertThat(HashingEmbedder.signOf(0x0000000080000000L)).isEqualTo(-1.0);
    }

    @Test
    public void testAccumulatorHasTheRequestedLength()
    {
        assertThat(embed("hello world", EmbeddingAlgorithm.WORD, 768)).hasSize(768);
    }

    @Test
    public void testIsDeterministic()
    {
        assertThat(embed("hello world", EmbeddingAlgorithm.WORD, 64))
                .isEqualTo(embed("hello world", EmbeddingAlgorithm.WORD, 64));
    }

    @Test
    public void testIsCaseAndPunctuationInsensitiveForWords()
    {
        assertThat(embed("Hello, World!", EmbeddingAlgorithm.WORD, 64))
                .isEqualTo(embed("hello world", EmbeddingAlgorithm.WORD, 64));
    }

    @Test
    public void testDifferentTextsGiveDifferentAccumulators()
    {
        assertThat(embed("hello world", EmbeddingAlgorithm.WORD, 256))
                .isNotEqualTo(embed("goodbye moon", EmbeddingAlgorithm.WORD, 256));
    }

    @Test
    public void testTextWithoutTokenGivesTheZeroAccumulator()
    {
        assertThat(embed("!!! ...", EmbeddingAlgorithm.WORD, 32)).containsOnly(0.0);
        assertThat(embed("", EmbeddingAlgorithm.WORD, 32)).containsOnly(0.0);
        assertThat(embed("abcd", EmbeddingAlgorithm.CHAR_5GRAM, 32)).containsOnly(0.0);
    }

    /**
     * Each token contributes exactly one unit of magnitude, so with no collision the absolute
     * values sum to the token count. Dimension 1 forces every token onto the same index, where
     * the hashed sign makes contributions cancel: the sum of absolute values then drops below the
     * token count. An unsigned scatter would keep it at exactly the token count, which is the
     * bias the sign exists to remove.
     */
    @Test
    public void testHashedSignMakesCollidingTokensCancel()
    {
        String text = "alpha beta gamma delta epsilon zeta eta theta";
        int tokenCount = 8;
        assertThat(sumOfAbsoluteValues(embed(text, EmbeddingAlgorithm.WORD, 4096)))
                .isEqualTo(tokenCount);
        assertThat(sumOfAbsoluteValues(embed(text, EmbeddingAlgorithm.WORD, 1)))
                .isLessThan(tokenCount);
    }
}
