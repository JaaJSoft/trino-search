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
import java.util.Optional;

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
                assertThat(meanRecall(dataset, distance, false))
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
                assertThat(meanRecall(dataset, distance, true))
                        .as("%s / %s", regime, distance.sqlName())
                        .isEqualTo(1.0);
            }
        }
    }

    private static double meanRecall(VectorDataset dataset, BruteForce.Distance distance, boolean realVectors)
    {
        // The array(real) path computes from float-rounded components, so the oracle has to see
        // the same values or it would rank near-ties differently.
        double[][] base = realVectors ? VectorBlocks.roundedToFloat(dataset.base()) : dataset.base();
        double[][] queries = realVectors ? VectorBlocks.roundedToFloat(dataset.queries()) : dataset.queries();
        Block[] baseBlocks = realVectors
                ? VectorBlocks.realVectors(dataset.base())
                : VectorBlocks.doubleVectors(dataset.base());

        long[] ids = new long[base.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i;
        }
        LongArrayBlock keys = new LongArrayBlock(ids.length, Optional.empty(), ids);

        double total = 0;
        for (int q = 0; q < queries.length; q++) {
            Block queryBlock = realVectors
                    ? VectorBlocks.realVector(dataset.queries()[q])
                    : VectorBlocks.doubleVector(dataset.queries()[q]);
            double[] returned = runAggregation(keys, baseBlocks, queryBlock, distance, realVectors);
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
        for (int i = 0; i < baseBlocks.length; i++) {
            if (realVectors) {
                KnnAggregation.OfRealVectors.input(
                        BIGINT, state, keys, i, baseBlocks[i], queryBlock, K, Slices.utf8Slice(distance.sqlName()));
            }
            else {
                KnnAggregation.OfDoubleVectors.input(
                        BIGINT, state, keys, i, baseBlocks[i], queryBlock, K, Slices.utf8Slice(distance.sqlName()));
            }
        }

        List<KnnHeap.Neighbour> neighbours = state.getHeap().drainSorted();
        double[] distances = new double[neighbours.size()];
        for (int i = 0; i < distances.length; i++) {
            distances[i] = neighbours.get(i).distance();
        }
        return distances;
    }
}
