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

import dev.jaaj.trino.search.vector.VectorReader;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.function.AccumulatorStateFactory;
import io.trino.spi.function.GroupedAccumulatorState;

import java.util.Arrays;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * Unlike {@code KnnHeap}, whose arrays can start small because a group may never fill them, the
 * sums array is sized to the dimension on a group's first vector and never changes length: the
 * first vector already touches every dimension, so there is nothing to defer.
 */
public final class VectorSumStateFactory
        implements AccumulatorStateFactory<VectorSumState>
{
    @Override
    public VectorSumState createSingleState()
    {
        return new SingleVectorSumState();
    }

    @Override
    public VectorSumState createGroupedState()
    {
        return new GroupedVectorSumState();
    }

    static void checkSameDimension(int seen, int incoming)
    {
        if (seen != incoming) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The vectors of vector_avg_agg must have the same length, found %s and %s"
                            .formatted(seen, incoming));
        }
    }

    private static void addInto(double[] sums, Block vector, VectorReader reader)
    {
        for (int i = 0; i < sums.length; i++) {
            sums[i] += reader.read(vector, i);
        }
    }

    private static void addInto(double[] sums, double[] otherSums)
    {
        for (int i = 0; i < sums.length; i++) {
            sums[i] += otherSums[i];
        }
    }

    public static class SingleVectorSumState
            implements VectorSumState
    {
        private static final long INSTANCE_SIZE = instanceSize(SingleVectorSumState.class);

        private double[] sums;
        private long count;

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP",
                justification = "The serializer and the output read this array directly; copying a "
                        + "double[dimension] per call is the cost this state exists to avoid.")
        public double[] getSums()
        {
            return sums;
        }

        @Override
        public long getCount()
        {
            return count;
        }

        @Override
        public void accumulate(Block vector, VectorReader reader)
        {
            if (sums == null) {
                sums = new double[vector.getPositionCount()];
            }
            else {
                checkSameDimension(sums.length, vector.getPositionCount());
            }
            addInto(sums, vector, reader);
            count++;
        }

        @Override
        public void merge(double[] otherSums, long otherCount)
        {
            if (sums == null) {
                sums = otherSums.clone();
            }
            else {
                checkSameDimension(sums.length, otherSums.length);
                addInto(sums, otherSums);
            }
            count += otherCount;
        }

        @Override
        public void replace(double[] sums, long count)
        {
            this.sums = sums;
            this.count = count;
        }

        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(sums);
        }
    }

    public static class GroupedVectorSumState
            implements GroupedAccumulatorState, VectorSumState
    {
        private static final long INSTANCE_SIZE = instanceSize(GroupedVectorSumState.class);

        private double[][] sums = new double[0][];
        private long[] counts = new long[0];
        private int groupId;
        private long sumsSizeInBytes;

        @Override
        public void setGroupId(int groupId)
        {
            this.groupId = groupId;
        }

        @Override
        public void ensureCapacity(int size)
        {
            if (size > sums.length) {
                sums = Arrays.copyOf(sums, size);
                counts = Arrays.copyOf(counts, size);
            }
        }

        @Override
        public double[] getSums()
        {
            return sums[groupId];
        }

        @Override
        public long getCount()
        {
            return counts[groupId];
        }

        @Override
        public void accumulate(Block vector, VectorReader reader)
        {
            double[] current = sums[groupId];
            if (current == null) {
                current = new double[vector.getPositionCount()];
                sums[groupId] = current;
                sumsSizeInBytes += sizeOf(current);
            }
            else {
                checkSameDimension(current.length, vector.getPositionCount());
            }
            addInto(current, vector, reader);
            counts[groupId]++;
        }

        @Override
        public void merge(double[] otherSums, long otherCount)
        {
            double[] current = sums[groupId];
            if (current == null) {
                sums[groupId] = otherSums.clone();
                sumsSizeInBytes += sizeOf(otherSums);
            }
            else {
                checkSameDimension(current.length, otherSums.length);
                addInto(current, otherSums);
            }
            counts[groupId] += otherCount;
        }

        @Override
        public void replace(double[] sums, long count)
        {
            sumsSizeInBytes += sizeOf(sums) - sizeOf(this.sums[groupId]);
            this.sums[groupId] = sums;
            counts[groupId] = count;
        }

        /**
         * Called once per input page by {@code HashAggregationOperator.updateMemory}, so it must
         * stay constant time. The per-group arrays never change length once allocated, so a
         * running total kept at allocation is exact without walking the groups.
         */
        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(sums) + sizeOf(counts) + sumsSizeInBytes;
        }
    }
}
