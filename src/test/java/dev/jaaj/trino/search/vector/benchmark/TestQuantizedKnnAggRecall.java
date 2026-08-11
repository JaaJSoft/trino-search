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
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.block.Block;
import io.trino.spi.block.LongArrayBlock;
import io.trino.spi.block.SqlRow;
import org.junit.jupiter.api.Test;

import java.util.List;

import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.CLUSTERED;
import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.UNIFORM;
import static io.trino.spi.type.BigintType.BIGINT;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * Recall floors for the quantised representations, against the exact neighbours of the original
 * float vectors.
 * <p>
 * The exact aggregation scores 1.0 by construction, which is what {@link TestKnnAggRecall} pins.
 * Here the number is genuinely below one and the floors are the measurement, so they were read off
 * a first run rather than chosen in advance. A floor that has to be lowered is a fact about the
 * representation and belongs in the commit message; a floor raised after an improvement is the
 * point of having them.
 * <p>
 * With {@code QUERY_COUNT * K = 50} recall slots, every mean recall this class computes is an
 * exact multiple of {@code 1/50 = 0.02}, never an in-between value. Each floor below sits one such
 * step under the measurement it pins rather than flush against it, so a single flipped near-tie
 * (for instance from a kernel change that sums the same terms in a different order, which a
 * vectorised implementation is free to do) does not fail this test on its own; a genuine loss of
 * recall large enough to drop two steps still does.
 */
public class TestQuantizedKnnAggRecall
{
    private static final int BASE_SIZE = 500;
    private static final int QUERY_COUNT = 5;
    private static final int DIMENSION = 32;
    private static final int K = 10;

    @Test
    public void testInt8RecallAtNoOversampling()
    {
        assertRecallAtLeast(CLUSTERED, K, 0.96, false);
        assertRecallAtLeast(UNIFORM, K, 0.98, false);
    }

    /**
     * One bit per component keeps only the sign about each dimension's midpoint, so a shortlist has
     * to be oversampled before it can be re-ranked usefully. This regime (isotropic clusters, the
     * default) leaves the code plenty to exploit: recall reaches the measured ceiling once the
     * shortlist is widened to 10x k.
     */
    @Test
    public void testBinaryRecallImprovesWithOversampling()
    {
        double atOne = meanRecall(CLUSTERED, K, true);
        double atTen = meanRecall(CLUSTERED, K * 10, true);
        assertThat(atTen).as("oversampling must not lose neighbours").isGreaterThanOrEqualTo(atOne);
        assertRecallAtLeast(CLUSTERED, K * 10, 0.98, true);
    }

    /**
     * The adversarial regime named in {@link VectorDataset.Regime#UNIFORM}'s own javadoc: in high
     * dimension every pairwise distance concentrates around the same value, leaving a one-bit-per
     * -component code little to exploit. Measured recall here is well below the CLUSTERED ceiling
     * even at 10x oversampling, which is the expected shape of the representation under this
     * regime rather than a defect: the floor records that shape instead of hiding it.
     */
    @Test
    public void testBinaryRecallUnderUniform()
    {
        double atOne = meanRecall(UNIFORM, K, true);
        double atTen = meanRecall(UNIFORM, K * 10, true);
        assertThat(atTen).as("oversampling must not lose neighbours").isGreaterThanOrEqualTo(atOne);
        assertRecallAtLeast(UNIFORM, K * 10, 0.86, true);
    }

    private static void assertRecallAtLeast(VectorDataset.Regime regime, int shortlist, double floor, boolean binary)
    {
        assertThat(meanRecall(regime, shortlist, binary))
                .as("%s / %s / shortlist %s", regime, binary ? "binary" : "int8", shortlist)
                .isGreaterThanOrEqualTo(floor);
    }

    /**
     * Recall at K over a shortlist of {@code shortlist} keys: how many of the true nearest K the
     * approximate ranking put anywhere in its shortlist. That is what the SQL re-ranking pattern
     * recovers, since the join back rescues any shortlisted key.
     */
    private static double meanRecall(VectorDataset.Regime regime, int shortlist, boolean binary)
    {
        VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 13L);
        QuantizationBounds bounds = VectorBlocks.fitBounds(dataset.base());
        SqlRow boundsRow = VectorBlocks.boundsRow(bounds);
        LongArrayBlock keys = VectorBlocks.sequentialKeys(dataset.base().length);

        Block[] int8Base = binary ? null : VectorBlocks.int8Vectors(dataset.base(), bounds);
        Slice[] binaryBase = binary ? VectorBlocks.binaryVectors(dataset.base(), bounds) : null;

        double total = 0;
        for (int q = 0; q < dataset.queries().length; q++) {
            double[] query = dataset.queries()[q];
            KnnState state = new KnnStateFactory(BIGINT).createSingleState();
            if (binary) {
                Slice queryCodes = VectorBlocks.binaryVector(query, bounds);
                for (int i = 0; i < binaryBase.length; i++) {
                    KnnAggregation.OfBinaryVectors.input(
                            state, keys, i, binaryBase[i], queryCodes, shortlist, Slices.utf8Slice("euclidean"));
                }
            }
            else {
                Block queryCodes = VectorBlocks.int8Vector(query, bounds);
                for (int i = 0; i < int8Base.length; i++) {
                    KnnAggregation.OfQuantizedVectors.input(
                            state,
                            keys,
                            i,
                            int8Base[i],
                            queryCodes,
                            boundsRow,
                            shortlist,
                            Slices.utf8Slice("euclidean"));
                }
            }
            total += recallAtK(state, query, dataset.base());
        }
        return total / dataset.queries().length;
    }

    /**
     * The returned keys are compared against the true nearest K by key, not by distance: a
     * quantised distance is not the exact one, so scoring on the values would measure the
     * quantisation error rather than the recall.
     */
    private static double recallAtK(KnnState state, double[] query, double[][] base)
    {
        List<KnnHeap.Neighbour> returned = state.getHeap().drainSorted();
        boolean[] shortlisted = new boolean[base.length];
        for (KnnHeap.Neighbour neighbour : returned) {
            shortlisted[(int) BIGINT.getLong(neighbour.key(), 0)] = true;
        }

        int[] trueNearest = BruteForce.sortedKeys(query, base, BruteForce.Distance.EUCLIDEAN, K);
        int found = 0;
        for (int key : trueNearest) {
            if (shortlisted[key]) {
                found++;
            }
        }
        return (double) found / K;
    }
}
