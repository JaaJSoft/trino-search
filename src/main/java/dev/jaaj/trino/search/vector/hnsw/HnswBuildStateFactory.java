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

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import io.airlift.slice.Slice;
import io.trino.spi.block.Block;
import io.trino.spi.function.AccumulatorStateFactory;
import io.trino.spi.function.GroupedAccumulatorState;

import java.util.Arrays;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;

/**
 * The state of a group is every vector it has received, which at a few hundred thousand rows of
 * dimension 768 is hundreds of megabytes. That is the number Trino decides to kill a query on, so
 * it is reported in full and kept current on every row.
 */
public final class HnswBuildStateFactory
        implements AccumulatorStateFactory<HnswBuildState>
{
    @Override
    public HnswBuildState createSingleState()
    {
        return new SingleHnswBuildState();
    }

    @Override
    public HnswBuildState createGroupedState()
    {
        return new GroupedHnswBuildState();
    }

    public static class SingleHnswBuildState
            implements HnswBuildState
    {
        private static final long INSTANCE_SIZE = instanceSize(SingleHnswBuildState.class);

        private GraphInput input;

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP",
                justification = "The combine function hands these rows to the surviving state, which keeps "
                        + "appending to them; copying a whole partition per combine is what this avoids.")
        public GraphInput getInput()
        {
            return input;
        }

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP2",
                justification = "HnswBuildState.setInput is required to store the caller's live rows: "
                        + "the aggregation keeps appending to them through add and addAll.")
        public void setInput(GraphInput input)
        {
            this.input = input;
        }

        @Override
        public void add(long key, Block vector)
        {
            input.add(key, vector);
        }

        @Override
        public void add(long key, Slice codes)
        {
            input.add(key, codes);
        }

        @Override
        public void addAll(GraphInput other)
        {
            input.addAll(other);
        }

        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + (input == null ? 0 : input.getRetainedSizeInBytes());
        }
    }

    public static class GroupedHnswBuildState
            implements GroupedAccumulatorState, HnswBuildState
    {
        private static final long INSTANCE_SIZE = instanceSize(GroupedHnswBuildState.class);

        private GraphInput[] inputs = new GraphInput[0];
        private int groupId;
        private long inputsSizeInBytes;

        @Override
        public void setGroupId(int groupId)
        {
            this.groupId = groupId;
        }

        @Override
        public void ensureCapacity(int size)
        {
            if (size > inputs.length) {
                inputs = Arrays.copyOf(inputs, size);
            }
        }

        @Override
        public GraphInput getInput()
        {
            return inputs[groupId];
        }

        /**
         * {@code deserialize} attaches fresh rows to a group that may already hold some, hence the
         * subtraction.
         */
        @Override
        public void setInput(GraphInput input)
        {
            GraphInput previous = inputs[groupId];
            if (previous != null) {
                inputsSizeInBytes -= previous.getRetainedSizeInBytes();
            }
            if (input != null) {
                inputsSizeInBytes += input.getRetainedSizeInBytes();
            }
            inputs[groupId] = input;
        }

        /**
         * Brackets the mutation so the running total follows the group's own size in constant
         * time, without walking the groups.
         */
        @Override
        public void add(long key, Block vector)
        {
            GraphInput input = inputs[groupId];
            inputsSizeInBytes -= input.getRetainedSizeInBytes();
            input.add(key, vector);
            inputsSizeInBytes += input.getRetainedSizeInBytes();
        }

        @Override
        public void add(long key, Slice codes)
        {
            GraphInput input = inputs[groupId];
            inputsSizeInBytes -= input.getRetainedSizeInBytes();
            input.add(key, codes);
            inputsSizeInBytes += input.getRetainedSizeInBytes();
        }

        @Override
        public void addAll(GraphInput other)
        {
            GraphInput input = inputs[groupId];
            inputsSizeInBytes -= input.getRetainedSizeInBytes();
            input.addAll(other);
            inputsSizeInBytes += input.getRetainedSizeInBytes();
        }

        /**
         * Called once per input page by {@code HashAggregationOperator.updateMemory}, so it must
         * stay constant time.
         */
        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(inputs) + inputsSizeInBytes;
        }
    }
}
