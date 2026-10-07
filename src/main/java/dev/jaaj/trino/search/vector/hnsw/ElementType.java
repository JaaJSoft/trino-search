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

import dev.jaaj.trino.search.vector.quantize.BinaryCodes;

/**
 * The representation a graph stores its vectors in, which is the one they were built from.
 * <p>
 * Each representation is kept in the layout its distance kernels read without a conversion:
 * doubles as long bits, reals as int bits, int8 codes as bytes, and binary codes as the complete
 * {@link BinaryCodes} value of each vector, header included, so that a stored code is a valid
 * binary vector on its own.
 */
public enum ElementType
{
    DOUBLE((byte) 0, "array(double)"),
    REAL((byte) 1, "array(real)"),
    INT8((byte) 2, "array(tinyint)"),
    BINARY((byte) 3, "varbinary");

    /**
     * Persisted in every graph, so it is spelled out rather than taken from the ordinal: reordering
     * the constants must not change what an existing graph decodes to.
     */
    private final byte code;
    private final String sqlType;

    ElementType(byte code, String sqlType)
    {
        this.code = code;
        this.sqlType = sqlType;
    }

    byte code()
    {
        return code;
    }

    String sqlType()
    {
        return sqlType;
    }

    /**
     * The two float representations can be queried with either float type, since converting a
     * query between them is exactly a {@code CAST}. A code cannot be produced from a float without
     * the bounds it was fitted against, so a quantised graph is only queried with codes.
     */
    boolean isFloat()
    {
        return this == DOUBLE || this == REAL;
    }

    /**
     * How many elements of its backing array one vector of {@code dimension} components takes:
     * one long, int or byte per component, or the whole binary code for a binary vector.
     */
    int unitsPerVector(int dimension)
    {
        if (this == BINARY) {
            return BinaryCodes.HEADER_BYTES + (dimension + 7) / 8;
        }
        return dimension;
    }

    long bytesPerVector(int dimension)
    {
        long units = unitsPerVector(dimension);
        return switch (this) {
            case DOUBLE -> units * Long.BYTES;
            case REAL -> units * Integer.BYTES;
            case INT8, BINARY -> units;
        };
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
