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

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.AccumulatorStateSerializer;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import io.trino.spi.type.Type;

import java.util.List;

import static io.trino.spi.type.DoubleType.DOUBLE;

public final class BoundsStateSerializer
        implements AccumulatorStateSerializer<BoundsState>
{
    private static final ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);
    private static final RowType SERIALIZED_TYPE = RowType.anonymous(List.of(DOUBLE_ARRAY, DOUBLE_ARRAY));

    @Override
    @SuppressFBWarnings(
            value = "EI_EXPOSE_REP",
            justification = "AccumulatorStateSerializer.getSerializedType() must return the exact Type "
                    + "instance the engine will use; Trino's Type values are effectively immutable "
                    + "singletons, not defensively-copyable data.")
    public Type getSerializedType()
    {
        return SERIALIZED_TYPE;
    }

    @Override
    public void serialize(BoundsState state, BlockBuilder out)
    {
        double[] minimums = state.getMinimums();
        if (minimums == null) {
            out.appendNull();
            return;
        }
        double[] maximums = state.getMaximums();
        ((RowBlockBuilder) out).buildEntry(fieldBuilders -> {
            writeDoubles(fieldBuilders.get(0), minimums);
            writeDoubles(fieldBuilders.get(1), maximums);
        });
    }

    @Override
    public void deserialize(Block block, int index, BoundsState state)
    {
        SqlRow row = SERIALIZED_TYPE.getObject(block, index);
        int offset = row.getRawIndex();
        state.merge(
                readDoubles(DOUBLE_ARRAY.getObject(row.getRawFieldBlock(0), offset)),
                readDoubles(DOUBLE_ARRAY.getObject(row.getRawFieldBlock(1), offset)));
    }

    private static void writeDoubles(BlockBuilder builder, double[] values)
    {
        ((ArrayBlockBuilder) builder).buildEntry(elementBuilder -> {
            for (double value : values) {
                DOUBLE.writeDouble(elementBuilder, value);
            }
        });
    }

    private static double[] readDoubles(Block block)
    {
        double[] values = new double[block.getPositionCount()];
        for (int i = 0; i < values.length; i++) {
            values[i] = DOUBLE.getDouble(block, i);
        }
        return values;
    }
}
