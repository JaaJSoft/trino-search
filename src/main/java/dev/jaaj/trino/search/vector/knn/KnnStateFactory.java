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

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import io.trino.spi.function.AccumulatorStateFactory;
import io.trino.spi.function.GroupedAccumulatorState;
import io.trino.spi.function.TypeParameter;
import io.trino.spi.type.Type;

import java.util.Arrays;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;
import static java.util.Objects.requireNonNull;

public final class KnnStateFactory
        implements AccumulatorStateFactory<KnnState>
{
    public KnnStateFactory(@TypeParameter("K") Type keyType)
    {
        requireNonNull(keyType, "keyType is null");
    }

    @Override
    public KnnState createSingleState()
    {
        return new SingleKnnState();
    }

    @Override
    public KnnState createGroupedState()
    {
        return new GroupedKnnState();
    }

    public static class SingleKnnState
            implements KnnState
    {
        private static final long INSTANCE_SIZE = instanceSize(SingleKnnState.class);

        private KnnHeap heap;
        private int k;
        private String metricName;

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP",
                justification = "The accumulator input/combine functions mutate the returned heap in place; "
                        + "that is the whole point of this state, not an accidental leak.")
        public KnnHeap getHeap()
        {
            return heap;
        }

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP2",
                justification = "KnnState.setHeap is required to store the caller's live heap: the "
                        + "aggregation keeps mutating it through further getHeap() calls.")
        public void setHeap(KnnHeap heap)
        {
            this.heap = heap;
        }

        @Override
        public int getK()
        {
            return k;
        }

        @Override
        public void setK(int k)
        {
            this.k = k;
        }

        @Override
        public String getMetricName()
        {
            return metricName;
        }

        @Override
        public void setMetricName(String metricName)
        {
            this.metricName = metricName;
        }

        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + (heap == null ? 0 : heap.estimatedSizeInBytes());
        }
    }

    public static class GroupedKnnState
            implements GroupedAccumulatorState, KnnState
    {
        private static final long INSTANCE_SIZE = instanceSize(GroupedKnnState.class);

        private KnnHeap[] heaps = new KnnHeap[0];
        private int[] ks = new int[0];
        private String[] metricNames = new String[0];
        private int groupId;
        private long heapsSizeInBytes;

        @Override
        public void setGroupId(int groupId)
        {
            this.groupId = groupId;
        }

        @Override
        public void ensureCapacity(int size)
        {
            if (size > heaps.length) {
                heaps = Arrays.copyOf(heaps, size);
                ks = Arrays.copyOf(ks, size);
                metricNames = Arrays.copyOf(metricNames, size);
            }
        }

        @Override
        public KnnHeap getHeap()
        {
            return heaps[groupId];
        }

        /**
         * A heap never grows after construction, so the running total only has to follow which
         * heap is attached to which group. {@code deserialize} attaches a fresh heap to a group
         * that may already hold one, hence the subtraction.
         */
        @Override
        public void setHeap(KnnHeap heap)
        {
            KnnHeap previous = heaps[groupId];
            if (previous != null) {
                heapsSizeInBytes -= previous.estimatedSizeInBytes();
            }
            if (heap != null) {
                heapsSizeInBytes += heap.estimatedSizeInBytes();
            }
            heaps[groupId] = heap;
        }

        @Override
        public int getK()
        {
            return ks[groupId];
        }

        @Override
        public void setK(int k)
        {
            ks[groupId] = k;
        }

        @Override
        public String getMetricName()
        {
            return metricNames[groupId];
        }

        @Override
        public void setMetricName(String metricName)
        {
            metricNames[groupId] = metricName;
        }

        /**
         * Called once per input page by {@code HashAggregationOperator.updateMemory}, so it must
         * stay constant time: walking the groups here makes the whole aggregation quadratic.
         */
        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(heaps) + sizeOf(ks) + sizeOf(metricNames) + heapsSizeInBytes;
        }
    }
}
