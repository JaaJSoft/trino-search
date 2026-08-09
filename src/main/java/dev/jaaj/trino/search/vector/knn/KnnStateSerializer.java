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
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.vector.Metric;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.AccumulatorStateSerializer;
import io.trino.spi.function.TypeParameter;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import io.trino.spi.type.Type;

import java.util.List;

import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.VarcharType.VARCHAR;
import static java.util.Objects.requireNonNull;

public final class KnnStateSerializer
        implements AccumulatorStateSerializer<KnnState>
{
    private final RowType neighbourType;
    private final ArrayType neighbourArrayType;
    private final RowType serializedType;

    public KnnStateSerializer(@TypeParameter("K") Type keyType)
    {
        requireNonNull(keyType, "keyType is null");
        this.neighbourType = RowType.anonymous(List.of(keyType, DOUBLE));
        this.neighbourArrayType = new ArrayType(neighbourType);
        this.serializedType = RowType.anonymous(List.of(BIGINT, VARCHAR, neighbourArrayType));
    }

    @Override
    @SuppressFBWarnings(
            value = "EI_EXPOSE_REP",
            justification = "AccumulatorStateSerializer.getSerializedType() must return the exact Type "
                    + "instance the engine will use to read/write this state; Trino's Type values are "
                    + "effectively immutable singletons, not defensively-copyable data.")
    public Type getSerializedType()
    {
        return serializedType;
    }

    @Override
    public void serialize(KnnState state, BlockBuilder out)
    {
        KnnHeap heap = state.getHeap();
        if (heap == null) {
            out.appendNull();
            return;
        }

        ((RowBlockBuilder) out).buildEntry(fieldBuilders -> {
            BIGINT.writeLong(fieldBuilders.get(0), state.getK());
            VARCHAR.writeString(fieldBuilders.get(1), state.getMetric().sqlName());
            ((ArrayBlockBuilder) fieldBuilders.get(2)).buildEntry(elementBuilder -> {
                for (KnnHeap.Neighbour neighbour : heap.drainSorted()) {
                    ((RowBlockBuilder) elementBuilder).buildEntry(neighbourFields -> {
                        neighbourFields.get(0).append(neighbour.key(), 0);
                        DOUBLE.writeDouble(neighbourFields.get(1), neighbour.distance());
                    });
                }
            });
        });
    }

    @Override
    public void deserialize(Block block, int index, KnnState state)
    {
        SqlRow row = serializedType.getObject(block, index);
        int offset = row.getRawIndex();

        int k = (int) BIGINT.getLong(row.getRawFieldBlock(0), offset);
        Metric metric = Metric.fromName(VARCHAR.getSlice(row.getRawFieldBlock(1), offset));
        state.setK(k);
        state.setMetric(metric);

        // Trino's generated addIntermediateAsCombine reuses one scratch state object for
        // every non-null position in an intermediate block, calling deserialize(...) then
        // combine(...) once per position. A heap kept from a previous call here would leak
        // that earlier position's neighbours into this one, so every call must start from a
        // brand new heap, never the one already attached to state (if any).
        state.setHeap(new KnnHeap(k, metric.higherIsCloser()));

        Block neighbours = neighbourArrayType.getObject(row.getRawFieldBlock(2), offset);
        for (int i = 0; i < neighbours.getPositionCount(); i++) {
            SqlRow neighbour = neighbourType.getObject(neighbours, i);
            double distance = DOUBLE.getDouble(neighbour.getRawFieldBlock(1), neighbour.getRawIndex());
            state.addToHeap(neighbour.getUnderlyingFieldBlock(0), neighbour.getUnderlyingFieldPosition(0), distance);
        }
    }
}
