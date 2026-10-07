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

import dev.jaaj.trino.search.vector.projection.PcaStateFactory.SinglePcaState;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;

public class TestPcaStateSerializer
{
    private static final PcaStateSerializer SERIALIZER = new PcaStateSerializer();

    private static Block vector(double... values)
    {
        BlockBuilder builder = DOUBLE.createFixedSizeBlockBuilder(values.length);
        for (double value : values) {
            DOUBLE.writeDouble(builder, value);
        }
        return builder.build();
    }

    private static SinglePcaState state(long components, double[]... rows)
    {
        SinglePcaState state = new SinglePcaState();
        state.setComponents(components);
        Covariance covariance = new Covariance(rows[0].length);
        for (double[] row : rows) {
            covariance.add(vector(row), DOUBLE_READER);
        }
        state.setCovariance(covariance);
        return state;
    }

    private static Block serialized(SinglePcaState... states)
    {
        BlockBuilder builder = SERIALIZER.getSerializedType().createBlockBuilder(null, states.length);
        for (SinglePcaState state : states) {
            SERIALIZER.serialize(state, builder);
        }
        return builder.build();
    }

    @Test
    public void testRoundTrip()
    {
        SinglePcaState original = state(2, new double[] {1, 2}, new double[] {3, 6}, new double[] {5, 4});
        SinglePcaState restored = new SinglePcaState();

        SERIALIZER.deserialize(serialized(original), 0, restored);

        assertThat(restored.getComponents()).isEqualTo(2);
        assertThat(restored.getCovariance().count()).isEqualTo(3);
        assertThat(restored.getCovariance().mean()).containsExactly(original.getCovariance().mean());
        assertThat(restored.getCovariance().comoments()).containsExactly(original.getCovariance().comoments());
    }

    /**
     * The engine deserializes every position of an intermediate block into one scratch state and
     * combines it into that position's group, so a deserialize must replace what the scratch held
     * rather than merge into it.
     */
    @Test
    public void testDeserializeReplacesWhatTheStateHeld()
    {
        Block block = serialized(state(1, new double[] {0, 0}, new double[] {2, 2}), state(3, new double[] {10, 20}));
        SinglePcaState scratch = new SinglePcaState();

        SERIALIZER.deserialize(block, 0, scratch);
        SERIALIZER.deserialize(block, 1, scratch);

        assertThat(scratch.getComponents()).isEqualTo(3);
        assertThat(scratch.getCovariance().count()).isEqualTo(1);
        assertThat(scratch.getCovariance().mean()).containsExactly(10.0, 20.0);
        assertThat(scratch.getCovariance().comoments()).containsOnly(0.0);
    }

    /**
     * A combine may keep the covariance of the state it was handed, so a deserialize must give
     * the scratch state a new one rather than overwrite the one a group now owns.
     */
    @Test
    public void testDeserializeDoesNotReuseTheCovarianceAGroupMayHaveTaken()
    {
        Block block = serialized(state(1, new double[] {1, 1}), state(1, new double[] {7, 7}));
        SinglePcaState scratch = new SinglePcaState();

        SERIALIZER.deserialize(block, 0, scratch);
        Covariance taken = scratch.getCovariance();
        SERIALIZER.deserialize(block, 1, scratch);

        assertThat(taken.mean()).containsExactly(1.0, 1.0);
    }

    @Test
    public void testEmptyStateSerializesAsNull()
    {
        BlockBuilder builder = SERIALIZER.getSerializedType().createBlockBuilder(null, 1);
        SERIALIZER.serialize(new SinglePcaState(), builder);

        assertThat(builder.build().isNull(0)).isTrue();
    }
}
