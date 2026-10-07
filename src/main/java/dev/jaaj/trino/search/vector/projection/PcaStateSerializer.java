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
package dev.jaaj.trino.search.vector.projection;

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

import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.DoubleType.DOUBLE;

/**
 * A group as its number of directions, its row count, its mean and its packed co-moments.
 */
public final class PcaStateSerializer
        implements AccumulatorStateSerializer<PcaState>
{
    private static final ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);
    private static final RowType SERIALIZED_TYPE = RowType.anonymous(List.of(BIGINT, BIGINT, DOUBLE_ARRAY, DOUBLE_ARRAY));

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
    public void serialize(PcaState state, BlockBuilder out)
    {
        Covariance covariance = state.getCovariance();
        if (covariance == null) {
            out.appendNull();
            return;
        }
        long components = state.getComponents();
        ((RowBlockBuilder) out).buildEntry(fieldBuilders -> {
            BIGINT.writeLong(fieldBuilders.get(0), components);
            BIGINT.writeLong(fieldBuilders.get(1), covariance.count());
            writeArray(fieldBuilders.get(2), covariance.mean());
            writeArray(fieldBuilders.get(3), covariance.comoments());
        });
    }

    /**
     * Always attaches a new covariance: a combine may have kept the one the state held before, so
     * filling that one in place would rewrite a group that has already taken it.
     */
    @Override
    public void deserialize(Block block, int index, PcaState state)
    {
        SqlRow row = SERIALIZED_TYPE.getObject(block, index);
        int offset = row.getRawIndex();
        state.setComponents(BIGINT.getLong(row.getRawFieldBlock(0), offset));
        state.setCovariance(Covariance.of(
                BIGINT.getLong(row.getRawFieldBlock(1), offset),
                readArray(row.getRawFieldBlock(2), offset),
                readArray(row.getRawFieldBlock(3), offset)));
    }

    private static void writeArray(BlockBuilder out, double[] values)
    {
        ((ArrayBlockBuilder) out).buildEntry(elementBuilder -> {
            for (double value : values) {
                DOUBLE.writeDouble(elementBuilder, value);
            }
        });
    }

    private static double[] readArray(Block arrays, int offset)
    {
        Block array = DOUBLE_ARRAY.getObject(arrays, offset);
        double[] values = new double[array.getPositionCount()];
        for (int i = 0; i < values.length; i++) {
            values[i] = DOUBLE.getDouble(array, i);
        }
        return values;
    }
}
