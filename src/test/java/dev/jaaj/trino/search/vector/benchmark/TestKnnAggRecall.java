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
package dev.jaaj.trino.search.vector.benchmark;

import dev.jaaj.trino.search.vector.knn.KnnAggregation;
import dev.jaaj.trino.search.vector.knn.KnnHeap;
import dev.jaaj.trino.search.vector.knn.KnnState;
import dev.jaaj.trino.search.vector.knn.KnnStateFactory;
import io.airlift.slice.Slices;
import io.trino.spi.block.Block;
import io.trino.spi.block.LongArrayBlock;
import org.junit.jupiter.api.Test;

import java.util.List;

import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.CLUSTERED;
import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.UNIFORM;
import static io.trino.spi.type.BigintType.BIGINT;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * The exact aggregation returns the true nearest neighbours by construction, so a recall below
 * one here means the harness is wrong, not the aggregation. This is what makes the recall of a
 * future approximate implementation worth reading.
 */
public class TestKnnAggRecall
{
    private static final int BASE_SIZE = 500;
    private static final int QUERY_COUNT = 5;
    private static final int DIMENSION = 32;
    private static final int K = 10;

    @Test
    public void testExactAggregationHasPerfectRecallOnDoubleVectors()
    {
        for (VectorDataset.Regime regime : List.of(CLUSTERED, UNIFORM)) {
            VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 11L);
            for (BruteForce.Distance distance : BruteForce.Distance.values()) {
                assertThat(meanRecall(dataset, distance, false, false))
                        .as("%s / %s", regime, distance.sqlName())
                        .isEqualTo(1.0);
            }
        }
    }

    @Test
    public void testExactAggregationHasPerfectRecallOnRealVectors()
    {
        for (VectorDataset.Regime regime : List.of(CLUSTERED, UNIFORM)) {
            VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 12L);
            for (BruteForce.Distance distance : BruteForce.Distance.values()) {
                assertThat(meanRecall(dataset, distance, true, false))
                        .as("%s / %s", regime, distance.sqlName())
                        .isEqualTo(1.0);
            }
        }
    }

    /**
     * A single partition never calls {@code @CombineFunction}: the input rows all land in one
     * {@code KnnState} and there is nothing to merge. Splitting the base vectors across two states
     * and merging them with {@code combine} before draining is what would catch a combine that
     * drops candidates or a serialized state that loses part of the heap.
     */
    @Test
    public void testExactAggregationHasPerfectRecallOnDoubleVectorsAcrossCombine()
    {
        for (VectorDataset.Regime regime : List.of(CLUSTERED, UNIFORM)) {
            VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 11L);
            for (BruteForce.Distance distance : BruteForce.Distance.values()) {
                assertThat(meanRecall(dataset, distance, false, true))
                        .as("%s / %s", regime, distance.sqlName())
                        .isEqualTo(1.0);
            }
        }
    }

    /**
     * Real-vector counterpart of {@link #testExactAggregationHasPerfectRecallOnDoubleVectorsAcrossCombine}.
     */
    @Test
    public void testExactAggregationHasPerfectRecallOnRealVectorsAcrossCombine()
    {
        for (VectorDataset.Regime regime : List.of(CLUSTERED, UNIFORM)) {
            VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 12L);
            for (BruteForce.Distance distance : BruteForce.Distance.values()) {
                assertThat(meanRecall(dataset, distance, true, true))
                        .as("%s / %s", regime, distance.sqlName())
                        .isEqualTo(1.0);
            }
        }
    }

    private static double meanRecall(VectorDataset dataset, BruteForce.Distance distance, boolean realVectors, boolean acrossCombine)
    {
        // The array(real) path computes from float-rounded components, so the oracle has to see
        // the same values or it would rank near-ties differently.
        double[][] base = realVectors ? VectorBlocks.roundedToFloat(dataset.base()) : dataset.base();
        double[][] queries = realVectors ? VectorBlocks.roundedToFloat(dataset.queries()) : dataset.queries();
        Block[] baseBlocks = realVectors
                ? VectorBlocks.realVectors(dataset.base())
                : VectorBlocks.doubleVectors(dataset.base());

        LongArrayBlock keys = VectorBlocks.sequentialKeys(base.length);

        double total = 0;
        for (int q = 0; q < queries.length; q++) {
            Block queryBlock = realVectors
                    ? VectorBlocks.realVector(dataset.queries()[q])
                    : VectorBlocks.doubleVector(dataset.queries()[q]);
            double[] returned = acrossCombine
                    ? runAggregationAcrossCombine(keys, baseBlocks, queryBlock, distance, realVectors)
                    : runAggregation(keys, baseBlocks, queryBlock, distance, realVectors);
            total += Recall.at(
                    K,
                    returned,
                    BruteForce.sortedDistances(queries[q], base, distance),
                    distance.higherIsCloser());
        }
        return total / queries.length;
    }

    private static double[] runAggregation(
            LongArrayBlock keys,
            Block[] baseBlocks,
            Block queryBlock,
            BruteForce.Distance distance,
            boolean realVectors)
    {
        KnnState state = new KnnStateFactory(BIGINT).createSingleState();
        feedInput(state, keys, baseBlocks, 0, baseBlocks.length, queryBlock, distance, realVectors);
        return drainDistances(state);
    }

    /**
     * Splits the base vectors across two states fed independently, then merges them with the
     * aggregation's {@code @CombineFunction} before draining: the shape a partial-per-split plus
     * final aggregation plan produces, which {@link #runAggregation} never exercises.
     */
    private static double[] runAggregationAcrossCombine(
            LongArrayBlock keys,
            Block[] baseBlocks,
            Block queryBlock,
            BruteForce.Distance distance,
            boolean realVectors)
    {
        int split = baseBlocks.length / 2;
        KnnState state = new KnnStateFactory(BIGINT).createSingleState();
        feedInput(state, keys, baseBlocks, 0, split, queryBlock, distance, realVectors);
        KnnState otherState = new KnnStateFactory(BIGINT).createSingleState();
        feedInput(otherState, keys, baseBlocks, split, baseBlocks.length, queryBlock, distance, realVectors);

        if (realVectors) {
            KnnAggregation.OfRealVectors.combine(state, otherState);
        }
        else {
            KnnAggregation.OfDoubleVectors.combine(state, otherState);
        }
        return drainDistances(state);
    }

    private static void feedInput(
            KnnState state,
            LongArrayBlock keys,
            Block[] baseBlocks,
            int fromInclusive,
            int toExclusive,
            Block queryBlock,
            BruteForce.Distance distance,
            boolean realVectors)
    {
        for (int i = fromInclusive; i < toExclusive; i++) {
            if (realVectors) {
                KnnAggregation.OfRealVectors.input(
                        BIGINT, state, keys, i, baseBlocks[i], queryBlock, K, Slices.utf8Slice(distance.sqlName()));
            }
            else {
                KnnAggregation.OfDoubleVectors.input(
                        BIGINT, state, keys, i, baseBlocks[i], queryBlock, K, Slices.utf8Slice(distance.sqlName()));
            }
        }
    }

    private static double[] drainDistances(KnnState state)
    {
        List<KnnHeap.Neighbour> neighbours = state.getHeap().drainSorted();
        double[] distances = new double[neighbours.size()];
        for (int i = 0; i < distances.length; i++) {
            distances[i] = neighbours.get(i).distance();
        }
        return distances;
    }
}
