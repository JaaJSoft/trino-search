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

import dev.jaaj.trino.search.vector.centroid.VectorSumStateFactory.GroupedVectorSumState;
import dev.jaaj.trino.search.vector.centroid.VectorSumStateFactory.SingleVectorSumState;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.type.ArrayType;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.airlift.slice.SizeOf.sizeOf;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;

public class TestVectorSumStateFactory
{
    private static final ArrayType VECTOR_TYPE = new ArrayType(DOUBLE);

    /**
     * The dimension of a typical embedding, and wide enough that a size which forgets the sums
     * array is off by kilobytes per group rather than by a few header bytes.
     */
    private static final int DIMENSION = 768;

    private static Block vector(int dimension, double value)
    {
        ArrayBlockBuilder builder = (ArrayBlockBuilder) VECTOR_TYPE.createBlockBuilder(null, 1);
        builder.buildEntry(elements -> {
            for (int i = 0; i < dimension; i++) {
                DOUBLE.writeDouble(elements, value);
            }
        });
        return VECTOR_TYPE.getObject(builder.build(), 0);
    }

    @Test
    public void testSingleStateCountsItsSums()
    {
        SingleVectorSumState state = new SingleVectorSumState();
        long empty = state.getEstimatedSize();

        state.accumulate(vector(DIMENSION, 1.0), DOUBLE_READER);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + sizeOf(new double[DIMENSION]));
    }

    @Test
    public void testGroupedStateCountsTheSumsOfEveryGroup()
    {
        GroupedVectorSumState state = new GroupedVectorSumState();
        state.ensureCapacity(4);
        long empty = state.getEstimatedSize();

        state.setGroupId(0);
        state.accumulate(vector(DIMENSION, 1.0), DOUBLE_READER);
        state.setGroupId(3);
        state.accumulate(vector(DIMENSION, 1.0), DOUBLE_READER);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + 2 * sizeOf(new double[DIMENSION]));
    }

    /**
     * The array is allocated once, on a group's first vector, and summed into in place afterwards,
     * so later rows and merged partial states must not be counted again.
     */
    @Test
    public void testGroupedStateCountsAGroupOnceWhateverItReceives()
    {
        GroupedVectorSumState state = new GroupedVectorSumState();
        state.ensureCapacity(1);
        state.setGroupId(0);
        state.accumulate(vector(DIMENSION, 1.0), DOUBLE_READER);
        long afterFirstRow = state.getEstimatedSize();

        state.accumulate(vector(DIMENSION, 2.0), DOUBLE_READER);
        state.merge(new double[DIMENSION], 5);

        assertThat(state.getEstimatedSize()).isEqualTo(afterFirstRow);
    }

    /**
     * A group created by a merge rather than by an input row, which is how every group of a final
     * aggregation comes into being, holds the same array and must be counted the same way.
     */
    @Test
    public void testGroupedStateCountsAGroupCreatedByMerge()
    {
        GroupedVectorSumState state = new GroupedVectorSumState();
        state.ensureCapacity(1);
        long empty = state.getEstimatedSize();

        state.setGroupId(0);
        state.merge(new double[DIMENSION], 5);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + sizeOf(new double[DIMENSION]));
    }

    /**
     * Deserialization replaces a group's array, and the replaced one must stop being counted.
     */
    @Test
    public void testGroupedStateStopsCountingAReplacedArray()
    {
        GroupedVectorSumState state = new GroupedVectorSumState();
        state.ensureCapacity(1);
        long empty = state.getEstimatedSize();

        state.setGroupId(0);
        state.accumulate(vector(DIMENSION, 1.0), DOUBLE_READER);
        state.replace(new double[4], 3);

        assertThat(state.getEstimatedSize()).isEqualTo(empty + sizeOf(new double[4]));
    }

    @Test
    public void testGroupedStateGrowsWithCapacity()
    {
        GroupedVectorSumState state = new GroupedVectorSumState();
        state.ensureCapacity(16);
        long small = state.getEstimatedSize();

        state.ensureCapacity(4096);

        assertThat(state.getEstimatedSize()).isGreaterThan(small);
    }

    /**
     * {@code HashAggregationOperator} reads {@code getEstimatedSize()} once per input page, so a
     * size computed by walking the groups makes the whole aggregation quadratic in its input. The
     * budget is as crude as the one in {@code TestKnnStateFactory}, for the same reason.
     */
    @Test
    public void testGroupedStateSizeDoesNotWalkTheGroups()
    {
        int groupCount = 1 << 23;
        int calls = 5_000;

        GroupedVectorSumState state = new GroupedVectorSumState();
        state.ensureCapacity(groupCount);
        state.setGroupId(groupCount - 1);
        state.accumulate(vector(4, 1.0), DOUBLE_READER);

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
