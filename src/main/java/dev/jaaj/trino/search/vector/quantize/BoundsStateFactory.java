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
import io.trino.spi.TrinoException;
import io.trino.spi.function.AccumulatorStateFactory;
import io.trino.spi.function.GroupedAccumulatorState;

import java.util.Arrays;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

public final class BoundsStateFactory
        implements AccumulatorStateFactory<BoundsState>
{
    @Override
    public BoundsState createSingleState()
    {
        return new SingleBoundsState();
    }

    @Override
    public BoundsState createGroupedState()
    {
        return new GroupedBoundsState();
    }

    static void checkSameDimension(int seen, int incoming)
    {
        if (seen != incoming) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The vectors of vector_bounds_agg must have the same length, found %s and %s"
                            .formatted(seen, incoming));
        }
    }

    public static class SingleBoundsState
            implements BoundsState
    {
        private static final long INSTANCE_SIZE = instanceSize(SingleBoundsState.class);

        private double[] minimums;
        private double[] maximums;

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP",
                justification = "The serializer reads these arrays directly; copying a double[dimension] "
                        + "per serialize call is the cost this state exists to avoid.")
        public double[] getMinimums()
        {
            return minimums;
        }

        @Override
        @SuppressFBWarnings(value = "EI_EXPOSE_REP", justification = "See getMinimums.")
        public double[] getMaximums()
        {
            return maximums;
        }

        @Override
        public void accumulate(double[] vector)
        {
            if (minimums == null) {
                minimums = vector.clone();
                maximums = vector.clone();
                return;
            }
            checkSameDimension(minimums.length, vector.length);
            for (int i = 0; i < vector.length; i++) {
                minimums[i] = Math.min(minimums[i], vector[i]);
                maximums[i] = Math.max(maximums[i], vector[i]);
            }
        }

        @Override
        public void merge(double[] otherMinimums, double[] otherMaximums)
        {
            if (minimums == null) {
                minimums = otherMinimums.clone();
                maximums = otherMaximums.clone();
                return;
            }
            checkSameDimension(minimums.length, otherMinimums.length);
            for (int i = 0; i < minimums.length; i++) {
                minimums[i] = Math.min(minimums[i], otherMinimums[i]);
                maximums[i] = Math.max(maximums[i], otherMaximums[i]);
            }
        }

        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(minimums) + sizeOf(maximums);
        }
    }

    public static class GroupedBoundsState
            implements GroupedAccumulatorState, BoundsState
    {
        private static final long INSTANCE_SIZE = instanceSize(GroupedBoundsState.class);

        private double[][] minimums = new double[0][];
        private double[][] maximums = new double[0][];
        private int groupId;
        private long arraysSizeInBytes;

        @Override
        public void setGroupId(int groupId)
        {
            this.groupId = groupId;
        }

        @Override
        public void ensureCapacity(int size)
        {
            if (size > minimums.length) {
                minimums = Arrays.copyOf(minimums, size);
                maximums = Arrays.copyOf(maximums, size);
            }
        }

        @Override
        public double[] getMinimums()
        {
            return minimums[groupId];
        }

        @Override
        public double[] getMaximums()
        {
            return maximums[groupId];
        }

        @Override
        public void accumulate(double[] vector)
        {
            double[] currentMinimums = minimums[groupId];
            if (currentMinimums == null) {
                minimums[groupId] = vector.clone();
                maximums[groupId] = vector.clone();
                arraysSizeInBytes += 2L * sizeOf(vector);
                return;
            }
            checkSameDimension(currentMinimums.length, vector.length);
            double[] currentMaximums = maximums[groupId];
            for (int i = 0; i < vector.length; i++) {
                currentMinimums[i] = Math.min(currentMinimums[i], vector[i]);
                currentMaximums[i] = Math.max(currentMaximums[i], vector[i]);
            }
        }

        @Override
        public void merge(double[] otherMinimums, double[] otherMaximums)
        {
            double[] currentMinimums = minimums[groupId];
            if (currentMinimums == null) {
                minimums[groupId] = otherMinimums.clone();
                maximums[groupId] = otherMaximums.clone();
                arraysSizeInBytes += 2L * sizeOf(otherMinimums);
                return;
            }
            checkSameDimension(currentMinimums.length, otherMinimums.length);
            double[] currentMaximums = maximums[groupId];
            for (int i = 0; i < currentMinimums.length; i++) {
                currentMinimums[i] = Math.min(currentMinimums[i], otherMinimums[i]);
                currentMaximums[i] = Math.max(currentMaximums[i], otherMaximums[i]);
            }
        }

        /**
         * Called once per input page by {@code HashAggregationOperator.updateMemory}, so it must
         * stay constant time. The per-group arrays never change length once allocated, so a
         * running total kept at allocation is exact without walking the groups.
         */
        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(minimums) + sizeOf(maximums) + arraysSizeInBytes;
        }
    }
}
