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
import io.trino.spi.block.Block;
import io.trino.spi.block.LongArrayBlock;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * A varchar column read from a connector can carry bytes that are not valid UTF-8, and no SQL
 * literal can express such a value, so these cases hand the raw bytes to the entry points
 * directly. The contract is the same as for token-free text: embed what can be read rather than
 * raise, so that one bad row cannot fail a scan.
 */
public class TestEmbeddingFunctions
{
    private static final int DIMENSION = 16;
    private static final Slice ALGORITHM_WORD = Slices.utf8Slice("word");
    private static final Slice ALGORITHM_CHAR_3GRAM = Slices.utf8Slice("char_3gram");

    /**
     * A two-byte sequence whose continuation byte was cut off, which is what a substring taken on
     * a byte boundary leaves behind.
     */
    private static Slice truncatedSequence()
    {
        return Slices.wrappedBuffer(new byte[] {'a', 'b', (byte) 0xC3});
    }

    /**
     * A continuation byte with no lead byte before it.
     */
    private static Slice bareContinuationByte()
    {
        return Slices.wrappedBuffer(new byte[] {(byte) 0x80});
    }

    private static Slice continuationByteBetweenLetters()
    {
        return Slices.wrappedBuffer(new byte[] {'a', (byte) 0x80, 'b'});
    }

    @Test
    public void testMalformedUtf8EmbedsAsWords()
    {
        for (Slice text : new Slice[] {truncatedSequence(), bareContinuationByte(), continuationByteBetweenLetters()}) {
            assertThat(EmbeddingFunctions.toVectorReal(text, DIMENSION, ALGORITHM_WORD).getPositionCount())
                    .isEqualTo(DIMENSION);
            assertThat(EmbeddingFunctions.toVectorDouble(text, DIMENSION, ALGORITHM_WORD).getPositionCount())
                    .isEqualTo(DIMENSION);
        }
    }

    @Test
    public void testMalformedUtf8EmbedsAsCharNgrams()
    {
        for (Slice text : new Slice[] {truncatedSequence(), bareContinuationByte(), continuationByteBetweenLetters()}) {
            assertThat(EmbeddingFunctions.toVectorReal(text, DIMENSION, ALGORITHM_CHAR_3GRAM).getPositionCount())
                    .isEqualTo(DIMENSION);
            assertThat(EmbeddingFunctions.toVectorDouble(text, DIMENSION, ALGORITHM_CHAR_3GRAM).getPositionCount())
                    .isEqualTo(DIMENSION);
        }
    }

    /**
     * The valid part of a malformed text must still be read, otherwise sanitising would silently
     * degrade to returning the zero vector for the whole row.
     */
    @Test
    public void testMalformedUtf8StillEmbedsTheReadableTokens()
    {
        Block trailingGarbage = EmbeddingFunctions.toVectorDouble(
                Slices.wrappedBuffer(new byte[] {'a', 'b', (byte) 0xC3}), DIMENSION, ALGORITHM_WORD);
        assertThat(isZeroVector(trailingGarbage)).isFalse();
    }

    private static boolean isZeroVector(Block block)
    {
        for (int position = 0; position < block.getPositionCount(); position++) {
            if (Double.longBitsToDouble(((LongArrayBlock) block).getLong(position)) != 0.0) {
                return false;
            }
        }
        return true;
    }
}
