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
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.lang.ref.WeakReference;

/**
 * One remembered sum of squares, for the vector that does not change from row to row.
 * <p>
 * The quantity is the accumulation before the square root, which is what
 * {@link VectorMath#cosineSimilarity} works with: it compares the product of the two operands'
 * sums of squares against zero and infinity before taking any root.
 * <p>
 * The entry is keyed on the identity of the block's backing array, the offset the vector starts
 * at and the number of components covered, never on the identity of the block:
 * {@code ArrayType.getObject} hands back a fresh region object per row over the same components,
 * so a key on the block would miss every time. A miss only costs a recomputation, but a false hit
 * would rank candidates against another vector's magnitude and return the wrong neighbours, which
 * is why the key is exact rather than merely cheap. Anything whose components are not laid out
 * plainly in that array, a dictionary or a run-length block, is not keyed at all, and neither is
 * a block carrying a null mask, whose components are not the array's alone.
 * <p>
 * The array is held weakly. A driver thread is pooled and outlives the query that handed it a
 * page, so a strong reference from a cache that lives as long as the thread would keep that page's
 * components alive for as long as the thread, outside anything the engine accounts for.
 */
final class SquaredMagnitudeCache
{
    /**
     * Returned by {@link #lookup} when nothing is remembered for a vector. A sum of squares is
     * either non-negative or {@code NaN}, so no accumulated value can be mistaken for it.
     */
    static final double MISS = -1.0;

    /**
     * The backing array and the offset the vector starts at, which are the halves of the key that
     * come from the block rather than from the caller.
     */
    private record Key(Object values, int offset) {}

    private WeakReference<Object> values = new WeakReference<>(null);
    private int offset;
    private int length;
    private double squaredMagnitude;

    /**
     * The remembered sum of squares of the first {@code length} components of {@code vector}, or
     * {@link #MISS} if this vector is not the remembered one or cannot be keyed exactly.
     */
    double lookup(Block vector, VectorReader reader, int length)
    {
        Key key = keyOf(vector, reader);
        if (key == null
                || key.values() != values.get()
                || key.offset() != offset
                || length != this.length) {
            return MISS;
        }
        return squaredMagnitude;
    }

    /**
     * Remembers {@code squaredMagnitude} as the sum of squares of the first {@code length}
     * components of {@code vector}, replacing whatever was remembered before. A vector that cannot
     * be keyed exactly is not remembered at all.
     */
    void store(Block vector, VectorReader reader, int length, double squaredMagnitude)
    {
        Key key = keyOf(vector, reader);
        if (key == null) {
            return;
        }
        this.values = new WeakReference<>(key.values());
        this.offset = key.offset();
        this.length = length;
        this.squaredMagnitude = squaredMagnitude;
    }

    /**
     * The halves of the key a block can supply, or null for a block whose components the key
     * cannot name. The guards are the ones the vectorized kernels in {@link VectorMath} use, and
     * hold for the same reasons: only a plain array block stores a vector's components
     * contiguously and in order, and comparing the reader by identity is what ties the raw longs
     * or ints to a floating point type rather than to some other value of the same width.
     */
    private static Key keyOf(Block vector, VectorReader reader)
    {
        if (reader == VectorReader.DOUBLE_READER
                && vector instanceof LongArrayBlock block
                && !block.mayHaveNull()) {
            return new Key(block.getRawValues(), block.getRawValuesOffset());
        }
        if (reader == VectorReader.REAL_READER
                && vector instanceof IntArrayBlock block
                && !block.mayHaveNull()) {
            return new Key(block.getRawValues(), block.getRawValuesOffset());
        }
        return null;
    }
}
