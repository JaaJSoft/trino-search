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
package dev.jaaj.trino.search.vector.centroid;

import dev.jaaj.trino.search.vector.centroid.VectorSumStateFactory.SingleVectorSumState;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class TestVectorSumStateSerializer
{
    private static final VectorSumStateSerializer SERIALIZER = new VectorSumStateSerializer();

    private static Block serialized(double[]... groups)
    {
        BlockBuilder builder = SERIALIZER.getSerializedType().createBlockBuilder(null, groups.length);
        for (double[] sums : groups) {
            SingleVectorSumState state = new SingleVectorSumState();
            state.merge(sums, 1);
            SERIALIZER.serialize(state, builder);
        }
        return builder.build();
    }

    @Test
    public void testRoundTrip()
    {
        SingleVectorSumState state = new SingleVectorSumState();
        SERIALIZER.deserialize(serialized(new double[] {1.5, -2.0}), 0, state);

        assertThat(state.getSums()).containsExactly(1.5, -2.0);
        assertThat(state.getCount()).isEqualTo(1);
    }

    /**
     * The engine deserializes every position of an intermediate block into one scratch state and
     * combines it into that position's group. A deserialize that merged instead of replacing would
     * carry every earlier group's sums into the next one.
     */
    @Test
    public void testDeserializeReplacesWhatTheStateHeld()
    {
        Block block = serialized(new double[] {1.0, 1.0}, new double[] {10.0, 20.0});
        SingleVectorSumState scratch = new SingleVectorSumState();

        SERIALIZER.deserialize(block, 0, scratch);
        SERIALIZER.deserialize(block, 1, scratch);

        assertThat(scratch.getSums()).containsExactly(10.0, 20.0);
        assertThat(scratch.getCount()).isEqualTo(1);
    }

    @Test
    public void testEmptyStateSerializesAsNull()
    {
        BlockBuilder builder = SERIALIZER.getSerializedType().createBlockBuilder(null, 1);
        SERIALIZER.serialize(new SingleVectorSumState(), builder);

        assertThat(builder.build().isNull(0)).isTrue();
    }
}
