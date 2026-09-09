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
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.DictionaryBlock;
import io.trino.spi.block.RunLengthEncodedBlock;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.SquaredMagnitudeCache.MISS;
import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;
import static org.assertj.core.api.Assertions.assertThat;

public class TestSquaredMagnitudeCache
{
    private static Block doubles(Double... values)
    {
        BlockBuilder builder = DOUBLE.createBlockBuilder(null, values.length);
        for (Double value : values) {
            if (value == null) {
                builder.appendNull();
            }
            else {
                DOUBLE.writeDouble(builder, value);
            }
        }
        return builder.build();
    }

    private static Block reals(Float... values)
    {
        BlockBuilder builder = REAL.createBlockBuilder(null, values.length);
        for (Float value : values) {
            REAL.writeFloat(builder, value);
        }
        return builder.build();
    }

    @Test
    public void testAFreshCacheRemembersNothing()
    {
        assertThat(new SquaredMagnitudeCache().lookup(doubles(3.0, 4.0), DOUBLE_READER, 2)).isEqualTo(MISS);
    }

    @Test
    public void testAStoredVectorIsFoundAgain()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block vector = doubles(3.0, 4.0);
        cache.store(vector, DOUBLE_READER, 2, 25.0);

        assertThat(cache.lookup(vector, DOUBLE_READER, 2)).isEqualTo(25.0);
    }

    @Test
    public void testARealVectorIsFoundAgain()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block vector = reals(3.0f, 4.0f);
        cache.store(vector, REAL_READER, 2, 25.0);

        assertThat(cache.lookup(vector, REAL_READER, 2)).isEqualTo(25.0);
    }

    /**
     * The case the cache exists for. Every row of an aggregation reads the same query vector
     * through a fresh region object over the same components, so an entry keyed on the block
     * itself would be missed on every row after the one that stored it.
     */
    @Test
    public void testAnotherBlockOverTheSameComponentsIsFoundAgain()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block page = doubles(3.0, 4.0, 5.0, 12.0);
        Block stored = page.getRegion(0, 2);
        Block sameComponents = page.getRegion(0, 2);
        assertThat(stored).isNotSameAs(sameComponents);

        cache.store(stored, DOUBLE_READER, 2, 25.0);

        assertThat(cache.lookup(sameComponents, DOUBLE_READER, 2)).isEqualTo(25.0);
    }

    /**
     * Two vectors of the same page share one backing array and differ only in where they start,
     * so an entry keyed without the offset would hand the first vector's magnitude to the second.
     */
    @Test
    public void testAVectorAtAnotherOffsetInTheSameArrayIsNotFound()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block page = doubles(3.0, 4.0, 5.0, 12.0);
        cache.store(page.getRegion(0, 2), DOUBLE_READER, 2, 25.0);

        assertThat(cache.lookup(page.getRegion(2, 2), DOUBLE_READER, 2)).isEqualTo(MISS);
    }

    @Test
    public void testAVectorOfAnotherLengthIsNotFound()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block page = doubles(3.0, 4.0, 5.0, 12.0);
        cache.store(page.getRegion(0, 2), DOUBLE_READER, 2, 25.0);

        assertThat(cache.lookup(page.getRegion(0, 3), DOUBLE_READER, 3)).isEqualTo(MISS);
    }

    /**
     * A dictionary block maps position i somewhere else entirely in a shared array and a
     * run-length block holds a single component however long the vector is, so neither one's
     * components are the stretch of array the key names.
     */
    @Test
    public void testAVectorThatIsNotLaidOutPlainlyIsNotRemembered()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block dictionary = DictionaryBlock.create(2, doubles(9.0, 3.0, 4.0, 7.0), new int[] {1, 2});
        cache.store(dictionary, DOUBLE_READER, 2, 25.0);
        assertThat(cache.lookup(dictionary, DOUBLE_READER, 2)).isEqualTo(MISS);

        Block runLength = RunLengthEncodedBlock.create(doubles(2.0), 3);
        cache.store(runLength, DOUBLE_READER, 3, 12.0);
        assertThat(cache.lookup(runLength, DOUBLE_READER, 3)).isEqualTo(MISS);
    }

    /**
     * A null mask makes the array alone an incomplete description of the vector: two blocks over
     * the same components with different masks would otherwise share a key.
     */
    @Test
    public void testAVectorCarryingANullMaskIsNotRemembered()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block vector = doubles(3.0, null);
        cache.store(vector, DOUBLE_READER, 2, 25.0);

        assertThat(cache.lookup(vector, DOUBLE_READER, 2)).isEqualTo(MISS);
    }

    /**
     * The reader is what says whether the raw values are double bits or float bits, so an entry
     * stored for one is not an entry for the other.
     */
    @Test
    public void testAVectorReadThroughTheOtherReaderIsNotFound()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block vector = reals(3.0f, 4.0f);
        cache.store(vector, REAL_READER, 2, 25.0);

        assertThat(cache.lookup(vector, DOUBLE_READER, 2)).isEqualTo(MISS);
    }

    @Test
    public void testStoringAnotherVectorForgetsThePreviousOne()
    {
        SquaredMagnitudeCache cache = new SquaredMagnitudeCache();
        Block first = doubles(3.0, 4.0);
        Block second = doubles(5.0, 12.0);
        cache.store(first, DOUBLE_READER, 2, 25.0);
        cache.store(second, DOUBLE_READER, 2, 169.0);

        assertThat(cache.lookup(second, DOUBLE_READER, 2)).isEqualTo(169.0);
        assertThat(cache.lookup(first, DOUBLE_READER, 2)).isEqualTo(MISS);
    }
}
