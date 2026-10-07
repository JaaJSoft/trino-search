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
import io.trino.spi.function.AccumulatorStateFactory;
import io.trino.spi.function.GroupedAccumulatorState;

import java.util.Arrays;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;

public final class PcaStateFactory
        implements AccumulatorStateFactory<PcaState>
{
    @Override
    public PcaState createSingleState()
    {
        return new SinglePcaState();
    }

    @Override
    public PcaState createGroupedState()
    {
        return new GroupedPcaState();
    }

    public static class SinglePcaState
            implements PcaState
    {
        private static final long INSTANCE_SIZE = instanceSize(SinglePcaState.class);

        private Covariance covariance;
        private long components;

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP",
                justification = "The input function accumulates into this covariance and the combine "
                        + "function hands it to the surviving state; both are the point of this state.")
        public Covariance getCovariance()
        {
            return covariance;
        }

        @Override
        @SuppressFBWarnings(
                value = "EI_EXPOSE_REP2",
                justification = "The state keeps the caller's live covariance, which the aggregation keeps adding to.")
        public void setCovariance(Covariance covariance)
        {
            this.covariance = covariance;
        }

        @Override
        public long getComponents()
        {
            return components;
        }

        @Override
        public void setComponents(long components)
        {
            this.components = components;
        }

        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + (covariance == null ? 0 : covariance.getRetainedSizeInBytes());
        }
    }

    public static class GroupedPcaState
            implements GroupedAccumulatorState, PcaState
    {
        private static final long INSTANCE_SIZE = instanceSize(GroupedPcaState.class);

        private Covariance[] covariances = new Covariance[0];
        private long[] components = new long[0];
        private int groupId;
        private long covariancesSizeInBytes;

        @Override
        public void setGroupId(int groupId)
        {
            this.groupId = groupId;
        }

        @Override
        public void ensureCapacity(int size)
        {
            if (size > covariances.length) {
                covariances = Arrays.copyOf(covariances, size);
                components = Arrays.copyOf(components, size);
            }
        }

        @Override
        public Covariance getCovariance()
        {
            return covariances[groupId];
        }

        /**
         * Deserialization gives a group a new covariance where it may already hold one, hence the
         * subtraction.
         */
        @Override
        public void setCovariance(Covariance covariance)
        {
            Covariance previous = covariances[groupId];
            if (previous != null) {
                covariancesSizeInBytes -= previous.getRetainedSizeInBytes();
            }
            if (covariance != null) {
                covariancesSizeInBytes += covariance.getRetainedSizeInBytes();
            }
            covariances[groupId] = covariance;
        }

        @Override
        public long getComponents()
        {
            return components[groupId];
        }

        @Override
        public void setComponents(long components)
        {
            this.components[groupId] = components;
        }

        /**
         * Called once per input page by {@code HashAggregationOperator.updateMemory}, so it must
         * stay constant time: a covariance never changes size once created, so a running total
         * kept as they are attached is exact without walking the groups.
         */
        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + sizeOf(covariances) + sizeOf(components) + covariancesSizeInBytes;
        }
    }
}
