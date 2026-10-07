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

import dev.jaaj.trino.search.vector.projection.PcaStateFactory.GroupedPcaState;
import dev.jaaj.trino.search.vector.projection.PcaStateFactory.SinglePcaState;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

public class TestPcaStateFactory
{
    private static final int DIMENSION = 768;

    @Test
    public void testSingleStateCountsItsCovariance()
    {
        SinglePcaState state = new SinglePcaState();
        long empty = state.getEstimatedSize();
        Covariance covariance = new Covariance(DIMENSION);

        state.setCovariance(covariance);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + covariance.getRetainedSizeInBytes());
    }

    @Test
    public void testGroupedStateCountsTheCovarianceOfEveryGroup()
    {
        GroupedPcaState state = new GroupedPcaState();
        state.ensureCapacity(4);
        long empty = state.getEstimatedSize();
        Covariance first = new Covariance(DIMENSION);
        Covariance second = new Covariance(DIMENSION);

        state.setGroupId(0);
        state.setCovariance(first);
        state.setGroupId(3);
        state.setCovariance(second);

        assertThat(state.getEstimatedSize())
                .isEqualTo(empty + first.getRetainedSizeInBytes() + second.getRetainedSizeInBytes());
    }

    /**
     * Deserialization gives a group a new covariance, and the replaced one must stop being counted.
     */
    @Test
    public void testGroupedStateStopsCountingAReplacedCovariance()
    {
        GroupedPcaState state = new GroupedPcaState();
        state.ensureCapacity(1);
        long empty = state.getEstimatedSize();
        Covariance small = new Covariance(4);

        state.setGroupId(0);
        state.setCovariance(new Covariance(DIMENSION));
        state.setCovariance(small);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + small.getRetainedSizeInBytes());
    }

    @Test
    public void testGroupedStateKeepsComponentsPerGroup()
    {
        GroupedPcaState state = new GroupedPcaState();
        state.ensureCapacity(2);

        state.setGroupId(0);
        state.setComponents(4);
        state.setGroupId(1);
        state.setComponents(8);
        state.setGroupId(0);

        assertThat(state.getComponents()).isEqualTo(4);
    }

    /**
     * {@code HashAggregationOperator} reads {@code getEstimatedSize()} once per input page, so it
     * has to stay constant time however many groups there are.
     */
    @Test
    public void testGroupedStateSizeDoesNotWalkTheGroups()
    {
        int groupCount = 1 << 23;
        int calls = 5_000;

        GroupedPcaState state = new GroupedPcaState();
        state.ensureCapacity(groupCount);
        state.setGroupId(groupCount - 1);
        state.setCovariance(new Covariance(4));

        long start = System.nanoTime();
        long total = 0;
        for (int i = 0; i < calls; i++) {
            total += state.getEstimatedSize();
        }
        Duration elapsed = Duration.ofNanos(System.nanoTime() - start);

        assertThat(total).isEqualTo(calls * state.getEstimatedSize());
        assertThat(elapsed).isLessThan(Duration.ofSeconds(1));
    }
}
