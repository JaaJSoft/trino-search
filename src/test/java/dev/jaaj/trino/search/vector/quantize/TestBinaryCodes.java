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
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class TestBinaryCodes
{
    @Test
    public void testHeaderCarriesTheDimension()
    {
        Slice codes = BinaryCodes.pack(11, _ -> false);
        assertThat(BinaryCodes.dimension(codes)).isEqualTo(11);
    }

    /**
     * Four header bytes plus one byte per eight components, rounded up. A dimension that is not a
     * multiple of eight is the case the header exists for.
     */
    @Test
    public void testLengthIsHeaderPlusCeilingOfDimensionOverEight()
    {
        assertThat(BinaryCodes.pack(8, _ -> false).length()).isEqualTo(BinaryCodes.HEADER_BYTES + 1);
        assertThat(BinaryCodes.pack(9, _ -> false).length()).isEqualTo(BinaryCodes.HEADER_BYTES + 2);
        assertThat(BinaryCodes.pack(768, _ -> false).length()).isEqualTo(BinaryCodes.HEADER_BYTES + 96);
    }

    @Test
    public void testHammingCountsDifferingBits()
    {
        Slice all = BinaryCodes.pack(8, _ -> true);
        Slice none = BinaryCodes.pack(8, _ -> false);
        Slice alternating = BinaryCodes.pack(8, i -> i % 2 == 0);

        assertThat(BinaryCodes.hamming(all, all)).isZero();
        assertThat(BinaryCodes.hamming(all, none)).isEqualTo(8);
        assertThat(BinaryCodes.hamming(all, alternating)).isEqualTo(4);
    }

    @Test
    public void testHammingIsSymmetric()
    {
        Slice first = BinaryCodes.pack(70, i -> i % 3 == 0);
        Slice second = BinaryCodes.pack(70, i -> i % 5 == 0);
        assertThat(BinaryCodes.hamming(first, second)).isEqualTo(BinaryCodes.hamming(second, first));
    }

    /**
     * A dimension past a multiple of 64 is what exercises the tail path after the long-at-a-time
     * loop, which is where an off-by-one silently counts nothing or counts twice.
     */
    @Test
    public void testHammingCrossesTheLongBoundary()
    {
        Slice first = BinaryCodes.pack(200, _ -> false);
        Slice second = BinaryCodes.pack(200, i -> i >= 190);
        assertThat(BinaryCodes.hamming(first, second)).isEqualTo(10);
    }

    /**
     * The packer zeroes its padding, so masking cannot change its output. A hand-built varbinary
     * need not, and junk in the padding must not be counted as a differing component.
     */
    @Test
    public void testPaddingBitsAreNotCounted()
    {
        Slice clean = BinaryCodes.pack(3, _ -> false);
        Slice dirty = clean.copy();
        dirty.setByte(BinaryCodes.HEADER_BYTES, 0b1111_1000);
        assertThat(BinaryCodes.hamming(clean, dirty)).isZero();
    }

    @Test
    public void testMismatchedDimensionsFail()
    {
        assertThatThrownBy(() -> BinaryCodes.hamming(BinaryCodes.pack(8, _ -> false), BinaryCodes.pack(16, _ -> false)))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("same length");
    }

    @Test
    public void testTruncatedValueIsRejected()
    {
        assertThatThrownBy(() -> BinaryCodes.dimension(Slices.wrappedBuffer(new byte[] {0, 0})))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("not a binary vector");
    }

    /**
     * A header claiming more components than the payload can hold would make the Hamming loop read
     * past the end of the slice.
     */
    @Test
    public void testHeaderLongerThanThePayloadIsRejected()
    {
        Slice codes = Slices.allocate(BinaryCodes.HEADER_BYTES + 1);
        codes.setInt(0, Integer.reverseBytes(64));
        assertThatThrownBy(() -> BinaryCodes.dimension(codes))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("not a binary vector");
    }
}
