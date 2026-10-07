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
package dev.jaaj.trino.search.vector.hnsw;

import dev.jaaj.trino.search.vector.VectorReader;
import io.trino.spi.block.Block;
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.util.Optional;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;

/**
 * The component type a graph stores its vectors in, which is the element type of the arrays it was
 * built from. Components are kept as the raw bits the matching Trino type stores, doubles as long
 * bits and reals as int bits, because those are the layouts the vectorised kernels read without a
 * conversion.
 */
public enum ElementType
{
    DOUBLE((byte) 0, Double.BYTES, DOUBLE_READER) {
        @Override
        Block plainCopy(Block vector, VectorReader sourceReader)
        {
            long[] bits = new long[vector.getPositionCount()];
            for (int i = 0; i < bits.length; i++) {
                bits[i] = Double.doubleToRawLongBits(sourceReader.read(vector, i));
            }
            return new LongArrayBlock(bits.length, Optional.empty(), bits);
        }
    },
    REAL((byte) 1, Float.BYTES, REAL_READER) {
        @Override
        Block plainCopy(Block vector, VectorReader sourceReader)
        {
            int[] bits = new int[vector.getPositionCount()];
            for (int i = 0; i < bits.length; i++) {
                bits[i] = Float.floatToRawIntBits((float) sourceReader.read(vector, i));
            }
            return new IntArrayBlock(bits.length, Optional.empty(), bits);
        }
    };

    /**
     * Persisted in every graph, so it is spelled out rather than taken from the ordinal: reordering
     * the constants must not change what an existing graph decodes to.
     */
    private final byte code;
    private final int byteSize;
    private final VectorReader reader;

    ElementType(byte code, int byteSize, VectorReader reader)
    {
        this.code = code;
        this.byteSize = byteSize;
        this.reader = reader;
    }

    /**
     * {@code vector}, read through {@code sourceReader}, as a block of this type with no null mask
     * and its components starting at offset zero of an array of its own, which is the shape every
     * vectorised kernel recognises. A double read into a real is rounded to the nearest float,
     * exactly as a {@code CAST} would.
     */
    abstract Block plainCopy(Block vector, VectorReader sourceReader);

    byte code()
    {
        return code;
    }

    int byteSize()
    {
        return byteSize;
    }

    VectorReader reader()
    {
        return reader;
    }

    static ElementType fromCode(byte code)
    {
        for (ElementType type : values()) {
            if (type.code == code) {
                return type;
            }
        }
        return null;
    }
}
