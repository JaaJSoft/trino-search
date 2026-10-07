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

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.benchmark.VectorBlocks;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.function.GroupedAccumulatorState;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class TestHnswBuildStateFactory
{
    private static final int DIMENSION = 64;

    /**
     * The rows of a group are the whole memory cost of a build, so the reported size has to grow
     * with them, through either way of adding rows; a size fixed at the first row would let a
     * partition of any size through.
     */
    @Test
    public void testGroupedSizeFollowsTheRows()
    {
        HnswBuildState state = new HnswBuildStateFactory().createGroupedState();
        GroupedAccumulatorState grouped = (GroupedAccumulatorState) state;
        grouped.ensureCapacity(2);

        grouped.setGroupId(0);
        state.setInput(input());
        long sizeBefore = state.getEstimatedSize();
        long rowsBefore = state.getInput().getRetainedSizeInBytes();
        for (int i = 0; i < 1000; i++) {
            state.add(i, vector());
        }
        // 1000 doubles of dimension 64 alone are 512 kB.
        assertThat(state.getEstimatedSize() - sizeBefore)
                .isGreaterThanOrEqualTo(1000L * DIMENSION * Double.BYTES)
                .isEqualTo(state.getInput().getRetainedSizeInBytes() - rowsBefore);

        grouped.setGroupId(1);
        state.setInput(input());
        sizeBefore = state.getEstimatedSize();
        rowsBefore = state.getInput().getRetainedSizeInBytes();
        GraphInput other = input();
        for (int i = 0; i < 100; i++) {
            other.add(i, vector());
        }
        state.addAll(other);
        assertThat(state.getEstimatedSize() - sizeBefore)
                .isPositive()
                .isEqualTo(state.getInput().getRetainedSizeInBytes() - rowsBefore);
    }

    /**
     * Deserializing attaches fresh rows to a group that may already hold some, so replacing them
     * must give the old ones' size back.
     */
    @Test
    public void testReplacingAGroupsRowsReleasesTheirSize()
    {
        HnswBuildState state = new HnswBuildStateFactory().createGroupedState();
        GroupedAccumulatorState grouped = (GroupedAccumulatorState) state;
        grouped.ensureCapacity(1);
        grouped.setGroupId(0);
        long empty = state.getEstimatedSize();

        state.setInput(input());
        for (int i = 0; i < 100; i++) {
            state.add(i, vector());
        }
        state.setInput(null);

        assertThat(state.getEstimatedSize()).isEqualTo(empty);
    }

    /**
     * Trino deserializes every intermediate position into one reused scratch state, so a position
     * must come back with its own rows only.
     */
    @Test
    public void testDeserializingIntoAReusedStateDoesNotCarryRowsOver()
    {
        HnswBuildStateSerializer serializer = new HnswBuildStateSerializer();
        BlockBuilder builder = serializer.getSerializedType().createBlockBuilder(null, 2);
        HnswBuildState first = new HnswBuildStateFactory().createSingleState();
        first.setInput(input());
        first.add(1, vector());
        first.add(2, vector());
        serializer.serialize(first, builder);
        HnswBuildState second = new HnswBuildStateFactory().createSingleState();
        second.setInput(input());
        second.add(3, vector());
        serializer.serialize(second, builder);
        Block serialized = builder.build();

        HnswBuildState scratch = new HnswBuildStateFactory().createSingleState();
        serializer.deserialize(serialized, 0, scratch);
        assertThat(scratch.getInput().size()).isEqualTo(2);
        serializer.deserialize(serialized, 1, scratch);
        assertThat(scratch.getInput().size()).isEqualTo(1);
        assertThat(scratch.getInput().key(0)).isEqualTo(3);
    }

    private static GraphInput input()
    {
        return new GraphInput(ElementType.DOUBLE, Metric.EUCLIDEAN, 8, 32, DIMENSION);
    }

    private static Block vector()
    {
        return VectorBlocks.doubleVector(new double[DIMENSION]);
    }
}
