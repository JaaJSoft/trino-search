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

import dev.jaaj.trino.search.vector.hnsw.HnswBuildAggregation;
import dev.jaaj.trino.search.vector.hnsw.HnswBuildState;
import dev.jaaj.trino.search.vector.hnsw.HnswBuildStateFactory;
import dev.jaaj.trino.search.vector.hnsw.HnswSearchFunctions;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.type.RowType;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.CLUSTERED;
import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.UNIFORM;
import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.VarbinaryType.VARBINARY;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * Recall floors for {@code hnsw_search} over graphs built by {@code hnsw_build_agg}, against the
 * exact neighbours, through the same entry points the engine calls.
 * <p>
 * As in {@link TestQuantizedKnnAggRecall}, the floors are measurements read off a first run, not
 * targets chosen in advance. With {@code QUERY_COUNT * K = 200} recall slots every mean recall here
 * is a multiple of 0.005, and each floor sits two such steps under the value it pins, so that one
 * flipped near-tie does not fail the build while a real loss of recall does. The graph is
 * deterministic, so these numbers do not move between runs on their own.
 * <p>
 * Dot product is left out on purpose: over vectors that are not unit-norm it is not a distance,
 * which {@code TestHnswGraph} pins, and over unit-norm ones it ranks exactly as cosine does.
 */
public class TestHnswRecall
{
    private static final int BASE_SIZE = 2000;
    private static final int QUERY_COUNT = 20;
    private static final int DIMENSION = 32;
    private static final int K = 10;
    private static final int M = 16;
    private static final int EF_CONSTRUCTION = 100;

    private static final RowType NEIGHBOUR_TYPE = RowType.from(List.of(
            RowType.field("key", BIGINT),
            RowType.field("distance", DOUBLE)));

    /**
     * {@code ef_search = k} is the cheapest search there is. Uniform data in 32 dimensions has no
     * structure for the graph to exploit, which is why its floor sits so far under the clustered
     * one.
     */
    @Test
    public void testRecallAtTheSmallestCandidateList()
    {
        assertRecallAtLeast(CLUSTERED, BruteForce.Distance.EUCLIDEAN, false, K, 0.95);
        assertRecallAtLeast(UNIFORM, BruteForce.Distance.EUCLIDEAN, false, K, 0.78);
        assertRecallAtLeast(CLUSTERED, BruteForce.Distance.COSINE, false, K, 0.955);
        assertRecallAtLeast(UNIFORM, BruteForce.Distance.COSINE, false, K, 0.75);
    }

    @Test
    public void testRecallAtEf64()
    {
        assertRecallAtLeast(CLUSTERED, BruteForce.Distance.EUCLIDEAN, false, 64, 0.99);
        assertRecallAtLeast(UNIFORM, BruteForce.Distance.EUCLIDEAN, false, 64, 0.99);
        assertRecallAtLeast(CLUSTERED, BruteForce.Distance.COSINE, false, 64, 0.99);
        assertRecallAtLeast(UNIFORM, BruteForce.Distance.COSINE, false, 64, 0.98);
        assertRecallAtLeast(CLUSTERED, BruteForce.Distance.MANHATTAN, false, 64, 0.99);
    }

    @Test
    public void testRecallOnRealVectors()
    {
        assertRecallAtLeast(CLUSTERED, BruteForce.Distance.EUCLIDEAN, true, 64, 0.99);
        assertRecallAtLeast(UNIFORM, BruteForce.Distance.COSINE, true, 64, 0.98);
    }

    /**
     * A longer candidate list keeps exploring past the point where a shorter one stops, so recall
     * must not drop as it grows. A drop would mean the search terminates on something other than
     * the candidate list.
     */
    @Test
    public void testRecallDoesNotDecreaseWithEf()
    {
        for (VectorDataset.Regime regime : List.of(CLUSTERED, UNIFORM)) {
            double previous = 0;
            for (int ef : new int[] {K, 32, 64, 128, 256}) {
                double recall = meanRecall(regime, BruteForce.Distance.EUCLIDEAN, false, ef);
                assertThat(recall).as("%s at ef %s", regime, ef).isGreaterThanOrEqualTo(previous);
                previous = recall;
            }
        }
    }

    /**
     * A graph over int8 codes, scored by key against the exact neighbours of the float vectors, as
     * {@link TestQuantizedKnnAggRecall} scores the exact scan over the same codes: the loss here
     * is the quantisation's and the graph's together.
     */
    @Test
    public void testInt8GraphRecall()
    {
        assertShortlistRecallAtLeast(CLUSTERED, false, K, 64, 0.945);
        assertShortlistRecallAtLeast(UNIFORM, false, K, 64, 0.985);
        assertShortlistRecallAtLeast(UNIFORM, false, 2 * K, 64, 0.99);
    }

    /**
     * Binary codes need an oversampled shortlist before re-ranking is useful, graph or not, and
     * under the uniform regime a wide one: one bit per component has little to exploit when every
     * pairwise distance is about the same.
     */
    @Test
    public void testBinaryGraphRecallWithOversampling()
    {
        assertShortlistRecallAtLeast(CLUSTERED, true, 10 * K, 200, 0.99);
        assertShortlistRecallAtLeast(UNIFORM, true, 10 * K, 200, 0.71);
        assertShortlistRecallAtLeast(UNIFORM, true, 20 * K, 400, 0.885);
    }

    /**
     * What the graph costs on top of the quantisation: a candidate list as long as the corpus reads
     * every code, so it measures the codes alone. The graph at a usual candidate list is pinned to
     * within one flipped near-tie of it.
     */
    @Test
    public void testGraphOverCodesLosesNothingOverAnExhaustiveSearchOfTheSameCodes()
    {
        for (VectorDataset.Regime regime : List.of(CLUSTERED, UNIFORM)) {
            double exhaustive = shortlistRecall(regime, false, K, BASE_SIZE);
            assertThat(shortlistRecall(regime, false, K, 64)).as("int8 / %s", regime).isGreaterThanOrEqualTo(exhaustive - 0.005);
        }
        double exhaustive = shortlistRecall(CLUSTERED, true, 10 * K, BASE_SIZE);
        assertThat(shortlistRecall(CLUSTERED, true, 10 * K, 200)).as("binary").isGreaterThanOrEqualTo(exhaustive - 0.005);
    }

    private static void assertShortlistRecallAtLeast(VectorDataset.Regime regime, boolean binary, int shortlist, int ef, double floor)
    {
        assertThat(shortlistRecall(regime, binary, shortlist, ef))
                .as("%s / %s / shortlist %s at ef %s", regime, binary ? "binary" : "int8", shortlist, ef)
                .isGreaterThanOrEqualTo(floor);
    }

    /**
     * How many of the true nearest K the search put anywhere in its shortlist, which is what a
     * re-ranking join against the exact vectors recovers.
     */
    private static double shortlistRecall(VectorDataset.Regime regime, boolean binary, int shortlist, int ef)
    {
        VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 31L);
        QuantizationBounds bounds = VectorBlocks.fitBounds(dataset.base());
        Slice metric = Slices.utf8Slice("euclidean");

        HnswBuildState state = new HnswBuildStateFactory().createSingleState();
        for (int i = 0; i < dataset.base().length; i++) {
            if (binary) {
                HnswBuildAggregation.OfBinaryVectors.input(state, i, VectorBlocks.binaryVector(dataset.base()[i], bounds), M, EF_CONSTRUCTION, metric);
            }
            else {
                HnswBuildAggregation.OfQuantizedVectors.input(state, i, VectorBlocks.int8Vector(dataset.base()[i], bounds), VectorBlocks.boundsRow(bounds), M, EF_CONSTRUCTION, metric);
            }
        }
        BlockBuilder out = VARBINARY.createBlockBuilder(null, 1);
        if (binary) {
            HnswBuildAggregation.OfBinaryVectors.output(state, out);
        }
        else {
            HnswBuildAggregation.OfQuantizedVectors.output(state, out);
        }
        Slice graph = VARBINARY.getSlice(out.build(), 0);

        double total = 0;
        for (double[] query : dataset.queries()) {
            Block result = binary
                    ? HnswSearchFunctions.searchBinary(graph, VectorBlocks.binaryVector(query, bounds), shortlist, Math.max(ef, shortlist))
                    : HnswSearchFunctions.searchQuantized(graph, VectorBlocks.int8Vector(query, bounds), shortlist, Math.max(ef, shortlist));
            Set<Long> shortlisted = new HashSet<>();
            for (int i = 0; i < result.getPositionCount(); i++) {
                SqlRow row = NEIGHBOUR_TYPE.getObject(result, i);
                shortlisted.add(BIGINT.getLong(row.getRawFieldBlock(0), row.getRawIndex()));
            }
            int found = 0;
            for (int key : BruteForce.sortedKeys(query, dataset.base(), BruteForce.Distance.EUCLIDEAN, K)) {
                if (shortlisted.contains((long) key)) {
                    found++;
                }
            }
            total += (double) found / K;
        }
        return total / dataset.queries().length;
    }

    private static void assertRecallAtLeast(VectorDataset.Regime regime, BruteForce.Distance distance, boolean realVectors, int ef, double floor)
    {
        double recall = meanRecall(regime, distance, realVectors, ef);
        assertThat(recall)
                .as("%s / %s / %s at ef %s", regime, distance.sqlName(), realVectors ? "real" : "double", ef)
                .isGreaterThanOrEqualTo(floor);
    }

    private static double meanRecall(VectorDataset.Regime regime, BruteForce.Distance distance, boolean realVectors, int ef)
    {
        VectorDataset dataset = VectorDataset.generate(regime, BASE_SIZE, QUERY_COUNT, DIMENSION, 31L);
        // The array(real) path computes from float-rounded components, so the oracle has to see
        // the same values or it would rank near-ties differently.
        double[][] base = realVectors ? VectorBlocks.roundedToFloat(dataset.base()) : dataset.base();
        double[][] queries = realVectors ? VectorBlocks.roundedToFloat(dataset.queries()) : dataset.queries();

        Slice graph = build(dataset.base(), distance, realVectors);
        double total = 0;
        for (int q = 0; q < queries.length; q++) {
            Block result = realVectors
                    ? HnswSearchFunctions.searchReal(graph, VectorBlocks.realVector(dataset.queries()[q]), K, ef)
                    : HnswSearchFunctions.search(graph, VectorBlocks.doubleVector(dataset.queries()[q]), K, ef);
            total += Recall.at(
                    K,
                    distances(result),
                    BruteForce.sortedDistances(queries[q], base, distance),
                    distance.higherIsCloser());
        }
        return total / queries.length;
    }

    private static Slice build(double[][] base, BruteForce.Distance distance, boolean realVectors)
    {
        HnswBuildState state = new HnswBuildStateFactory().createSingleState();
        Slice metric = Slices.utf8Slice(distance.sqlName());
        for (int i = 0; i < base.length; i++) {
            if (realVectors) {
                HnswBuildAggregation.OfRealVectors.input(state, i, VectorBlocks.realVector(base[i]), M, EF_CONSTRUCTION, metric);
            }
            else {
                HnswBuildAggregation.OfDoubleVectors.input(state, i, VectorBlocks.doubleVector(base[i]), M, EF_CONSTRUCTION, metric);
            }
        }
        BlockBuilder out = VARBINARY.createBlockBuilder(null, 1);
        if (realVectors) {
            HnswBuildAggregation.OfRealVectors.output(state, out);
        }
        else {
            HnswBuildAggregation.OfDoubleVectors.output(state, out);
        }
        return VARBINARY.getSlice(out.build(), 0);
    }

    private static double[] distances(Block neighbours)
    {
        double[] distances = new double[neighbours.getPositionCount()];
        for (int i = 0; i < distances.length; i++) {
            SqlRow row = NEIGHBOUR_TYPE.getObject(neighbours, i);
            distances[i] = DOUBLE.getDouble(row.getRawFieldBlock(1), row.getRawIndex());
        }
        return distances;
    }
}
