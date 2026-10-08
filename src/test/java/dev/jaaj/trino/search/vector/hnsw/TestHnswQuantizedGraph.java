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
package dev.jaaj.trino.search.vector.hnsw;

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.benchmark.VectorBlocks;
import dev.jaaj.trino.search.vector.benchmark.VectorDataset;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.CLUSTERED;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Graphs over quantised codes. The reference here is the quantised kernels themselves, not the
 * float vectors the codes came from: what is being tested is that the graph finds the nearest
 * codes, while how close the codes rank to the original vectors is the quantisation's own business
 * and is measured by the recall tests.
 */
public class TestHnswQuantizedGraph
{
    private static final int M = 8;
    private static final int EF_CONSTRUCTION = 32;
    private static final int K = 10;

    private static final VectorDataset DATASET = VectorDataset.generate(CLUSTERED, 300, 5, 16, 41L);
    private static final QuantizationBounds BOUNDS = VectorBlocks.fitBounds(DATASET.base());

    /**
     * Distances rather than keys are compared: binary codes put many vectors at exactly the same
     * Hamming distance, and any of them is a correct answer.
     */
    @Test
    public void testInt8CandidateListAsLongAsTheGraphFindsTheNearestCodes()
    {
        for (Metric metric : List.of(Metric.EUCLIDEAN, Metric.EUCLIDEAN_SQUARED, Metric.MANHATTAN, Metric.COSINE)) {
            HnswGraph graph = HnswGraph.read(HnswGraphBuilder.build(int8Input(metric)));
            for (double[] query : DATASET.queries()) {
                Block queryCodes = VectorBlocks.int8Vector(query, BOUNDS);
                double[] expected = Arrays.stream(DATASET.base())
                        .mapToDouble(vector -> metric.computeQuantized(VectorBlocks.int8Vector(vector, BOUNDS), queryCodes, BOUNDS))
                        .sorted()
                        .limit(K)
                        .toArray();

                List<HnswGraph.Neighbour> neighbours = graph.search(queryCodes, ElementType.INT8, K, DATASET.base().length);

                assertThat(distances(neighbours)).as(metric.sqlName()).containsExactly(expected);
            }
        }
    }

    @Test
    public void testBinaryCandidateListAsLongAsTheGraphFindsTheNearestCodes()
    {
        for (Metric metric : List.of(Metric.EUCLIDEAN, Metric.MANHATTAN, Metric.COSINE)) {
            HnswGraph graph = HnswGraph.read(HnswGraphBuilder.build(binaryInput(metric)));
            for (double[] query : DATASET.queries()) {
                Slice queryCodes = VectorBlocks.binaryVector(query, BOUNDS);
                double[] expected = Arrays.stream(DATASET.base())
                        .mapToDouble(vector -> metric.computeBinary(VectorBlocks.binaryVector(vector, BOUNDS), queryCodes))
                        .sorted()
                        .limit(K)
                        .toArray();

                List<HnswGraph.Neighbour> neighbours = graph.searchBinary(queryCodes, K, DATASET.base().length);

                assertThat(distances(neighbours)).as(metric.sqlName()).containsExactly(expected);
            }
        }
    }

    /**
     * The bounds travel inside the partial state and inside the graph; a merge that lost them, or a
     * graph that stored them wrongly, would rank with the wrong scale and still return codes.
     */
    @Test
    public void testInt8GraphDoesNotDependOnHowTheRowsWereSplit()
    {
        GraphInput merged = new GraphInput(ElementType.INT8, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, BOUNDS.dimension(), BOUNDS);
        GraphInput other = new GraphInput(ElementType.INT8, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, BOUNDS.dimension(), BOUNDS);
        for (int i = 0; i < DATASET.base().length; i++) {
            (i % 3 == 0 ? other : merged).add(i, VectorBlocks.int8Vector(DATASET.base()[i], BOUNDS));
        }
        merged.addAll(GraphInput.deserialize(other.serialize()));

        assertThat(HnswGraphBuilder.build(merged)).isEqualTo(HnswGraphBuilder.build(int8Input(Metric.EUCLIDEAN)));
    }

    @Test
    public void testBinaryGraphDoesNotDependOnHowTheRowsWereSplit()
    {
        GraphInput merged = new GraphInput(ElementType.BINARY, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, BOUNDS.dimension(), null);
        GraphInput other = new GraphInput(ElementType.BINARY, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, BOUNDS.dimension(), null);
        for (int i = DATASET.base().length - 1; i >= 0; i--) {
            (i % 2 == 0 ? other : merged).add(i, VectorBlocks.binaryVector(DATASET.base()[i], BOUNDS));
        }
        merged.addAll(GraphInput.deserialize(other.serialize()));

        assertThat(HnswGraphBuilder.build(merged)).isEqualTo(HnswGraphBuilder.build(binaryInput(Metric.EUCLIDEAN)));
    }

    /**
     * The distance a search returns is the one the quantised scalar functions return for the same
     * two codes and bounds, which is what makes it comparable with them.
     */
    @Test
    public void testInt8DistanceIsTheQuantisedMetricValue()
    {
        HnswGraph graph = HnswGraph.read(HnswGraphBuilder.build(int8Input(Metric.EUCLIDEAN)));
        Block queryCodes = VectorBlocks.int8Vector(DATASET.queries()[0], BOUNDS);

        HnswGraph.Neighbour nearest = graph.search(queryCodes, ElementType.INT8, 1, 64).getFirst();

        Block nearestCodes = VectorBlocks.int8Vector(DATASET.base()[(int) nearest.key()], BOUNDS);
        assertThat(nearest.distance()).isEqualTo(Metric.EUCLIDEAN.computeQuantized(nearestCodes, queryCodes, BOUNDS));
    }

    @Test
    public void testBoundsMustBeConstantWithinAGroup()
    {
        GraphInput input = int8Input(Metric.EUCLIDEAN);
        double[] shifted = new double[BOUNDS.dimension()];
        for (int i = 0; i < shifted.length; i++) {
            shifted[i] = BOUNDS.offset(i) + 1;
        }

        assertThatThrownBy(() -> input.checkSameBounds(QuantizationBounds.of(shifted, BOUNDS.scale())))
                .isInstanceOf(TrinoException.class)
                .hasMessage("bounds must be constant within a group of hnsw_build_agg");
    }

    /**
     * A code means nothing without the bounds it was fitted against, and a float cannot be turned
     * into one without them, so a graph is only searched with a query of its own kind; the two float
     * types are the one exception, since converting between them is a cast.
     */
    @Test
    public void testQueryOfAnotherRepresentationIsRejected()
    {
        HnswGraph int8Graph = HnswGraph.read(HnswGraphBuilder.build(int8Input(Metric.EUCLIDEAN)));
        assertThatThrownBy(() -> int8Graph.search(VectorBlocks.realVector(DATASET.queries()[0]), ElementType.REAL, 1, 1))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The graph was built from array(tinyint) vectors and cannot be searched with a array(real) query");
        assertThatThrownBy(() -> int8Graph.searchBinary(VectorBlocks.binaryVector(DATASET.queries()[0], BOUNDS), 1, 1))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The graph was built from array(tinyint) vectors and cannot be searched with a varbinary query");

        HnswGraph binaryGraph = HnswGraph.read(HnswGraphBuilder.build(binaryInput(Metric.EUCLIDEAN)));
        assertThatThrownBy(() -> binaryGraph.search(VectorBlocks.int8Vector(DATASET.queries()[0], BOUNDS), ElementType.INT8, 1, 1))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The graph was built from varbinary vectors and cannot be searched with a array(tinyint) query");
    }

    @Test
    public void testBinaryQueryOfAnotherDimensionIsRejected()
    {
        HnswGraph graph = HnswGraph.read(HnswGraphBuilder.build(binaryInput(Metric.EUCLIDEAN)));
        Slice shorter = VectorBlocks.binaryVector(new double[] {1, 2, 3}, QuantizationBounds.of(new double[3], 1));

        assertThatThrownBy(() -> graph.searchBinary(shorter, 1, 1))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The query vector must have the dimension of the graph, found 3 and 16");
    }

    /**
     * The point of storing codes. The dimension is a realistic one on purpose: at a small dimension
     * the links outweigh the vectors, and binary codes, whose many tied distances leave the
     * neighbour selection less to prune, can even end up with the larger graph.
     */
    @Test
    public void testCodesMakeTheGraphSmaller()
    {
        double[][] base = VectorDataset.generate(CLUSTERED, 200, 1, 256, 43L).base();
        QuantizationBounds bounds = VectorBlocks.fitBounds(base);
        GraphInput doubles = new GraphInput(ElementType.DOUBLE, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, 256, null);
        GraphInput int8 = new GraphInput(ElementType.INT8, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, 256, bounds);
        GraphInput binary = new GraphInput(ElementType.BINARY, Metric.EUCLIDEAN, M, EF_CONSTRUCTION, 256, null);
        for (int i = 0; i < base.length; i++) {
            doubles.add(i, VectorBlocks.doubleVector(base[i]));
            int8.add(i, VectorBlocks.int8Vector(base[i], bounds));
            binary.add(i, VectorBlocks.binaryVector(base[i], bounds));
        }
        int doubleBytes = HnswGraphBuilder.build(doubles).length();
        int int8Bytes = HnswGraphBuilder.build(int8).length();
        int binaryBytes = HnswGraphBuilder.build(binary).length();

        assertThat(int8Bytes).isLessThan(doubleBytes / 4);
        assertThat(binaryBytes).isLessThan(int8Bytes / 2);
    }

    private static GraphInput int8Input(Metric metric)
    {
        GraphInput input = new GraphInput(ElementType.INT8, metric, M, EF_CONSTRUCTION, BOUNDS.dimension(), BOUNDS);
        for (int i = 0; i < DATASET.base().length; i++) {
            input.add(i, VectorBlocks.int8Vector(DATASET.base()[i], BOUNDS));
        }
        return input;
    }

    private static GraphInput binaryInput(Metric metric)
    {
        GraphInput input = new GraphInput(ElementType.BINARY, metric, M, EF_CONSTRUCTION, BOUNDS.dimension(), null);
        for (int i = 0; i < DATASET.base().length; i++) {
            input.add(i, VectorBlocks.binaryVector(DATASET.base()[i], BOUNDS));
        }
        return input;
    }

    private static double[] distances(List<HnswGraph.Neighbour> neighbours)
    {
        return neighbours.stream().mapToDouble(HnswGraph.Neighbour::distance).toArray();
    }
}
