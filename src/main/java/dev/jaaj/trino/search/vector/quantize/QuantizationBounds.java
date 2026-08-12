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

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.LongArrayBlock;
import io.trino.spi.block.SqlRow;
import io.trino.spi.type.ArrayType;

import java.util.Optional;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;

/**
 * The fitted quantisation parameters, as a read-only view onto the offsets array and the single
 * scale a {@code vector_bounds_agg} row carries.
 * <p>
 * The offsets block is wrapped rather than copied. A scalar receives this row once per input row,
 * so copying a {@code double[dimension]} array out of it would cost more per row at dimension 768
 * than the distance the caller is trying to compute.
 */
public final class QuantizationBounds
{
    public static final String BOUNDS_TYPE_SIGNATURE = "row(offsets array(double), scale double)";

    private static final ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);

    /**
     * A signed byte holds 256 codes, and the fitted range is spread over 255 steps about its own
     * midpoint: the midpoint encodes to 0, the fitted minimum to -127 and the fitted maximum to
     * +127. The codes are therefore centred, and the one code left over, -128, is only reached by a
     * value below the fitted range.
     */
    public static final int CODE_LEVELS = 255;

    private final Block offsets;
    private final double scale;

    private QuantizationBounds(Block offsets, double scale)
    {
        this.offsets = offsets;
        this.scale = scale;
    }

    public static QuantizationBounds of(SqlRow row)
    {
        int index = row.getRawIndex();
        Block offsets = DOUBLE_ARRAY.getObject(row.getRawFieldBlock(0), index);
        double scale = DOUBLE.getDouble(row.getRawFieldBlock(1), index);
        return new QuantizationBounds(offsets, scale);
    }

    public static QuantizationBounds forTesting(double[] offsets, double scale)
    {
        return new QuantizationBounds(doubleBlock(offsets), scale);
    }

    private static Block doubleBlock(double[] values)
    {
        long[] bits = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = Double.doubleToLongBits(values[i]);
        }
        return new LongArrayBlock(values.length, Optional.empty(), bits);
    }

    public int dimension()
    {
        return offsets.getPositionCount();
    }

    public Block offsets()
    {
        return offsets;
    }

    public double scale()
    {
        return scale;
    }

    public double offset(int i)
    {
        return DOUBLE.getDouble(offsets, i);
    }

    /**
     * Clamping is what keeps a value fitted outside the sampled range imprecise rather than wrong:
     * an unclamped narrowing cast wraps, which sends a far-away component to the opposite end of
     * the scale and turns a distant vector into a near neighbour.
     * <p>
     * {@code Math.round} sends NaN to zero, which lands on the offset. That is the only defined
     * answer available and it keeps a NaN component from dragging a code to an extreme.
     */
    public byte encode(int i, double value)
    {
        if (scale == 0) {
            return 0;
        }
        long rounded = Math.round((value - offset(i)) / scale);
        return (byte) Math.max(Byte.MIN_VALUE, Math.min(Byte.MAX_VALUE, rounded));
    }

    public double decode(int i, byte code)
    {
        return offset(i) + code * scale;
    }

    public void checkDimension(int length)
    {
        if (length != dimension()) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The vector has %s components but the quantisation bounds were fitted on %s"
                            .formatted(length, dimension()));
        }
    }
}
