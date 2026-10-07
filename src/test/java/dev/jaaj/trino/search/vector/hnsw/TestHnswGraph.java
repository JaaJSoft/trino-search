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
import dev.jaaj.trino.search.vector.benchmark.BruteForce;
import dev.jaaj.trino.search.vector.benchmark.VectorBlocks;
import dev.jaaj.trino.search.vector.benchmark.VectorDataset;
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.CLUSTERED;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestHnswGraph
{
    private static final int M = 8;
    private static final int EF_CONSTRUCTION = 32;

    /**
     * With a candidate list as long as the graph, the layer-0 search keeps every node it reaches,
     * so on a connected graph it reads all of them and the answer is exact. A miss here is a broken
     * link structure, not an approximation.
     * <p>
     * Dot product is measured on unit-norm vectors, the only case in which it is a distance (it is
     * then cosine similarity); see {@link #testDotProductOverUnnormalisedVectorsCanLeaveNodesUnreachable}.
     */
    @Test
    public void testCandidateListAsLongAsTheGraphFindsTheExactNeighbours()
    {
        VectorDataset dataset = VectorDataset.generate(CLUSTERED, 300, 5, 8, 21L);
        for (BruteForce.Distance distance : BruteForce.Distance.values()) {
            double[][] base = distance == BruteForce.Distance.DOT_PRODUCT ? normalised(dataset.base()) : dataset.base();
            Metric metric = Metric.fromName(distance.sqlName());
            HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, metric, base));
            for (double[] query : dataset.queries()) {
                List<HnswGraph.Neighbour> neighbours = graph.search(VectorBlocks.doubleVector(query), ElementType.DOUBLE, 10, base.length);
                int[] expected = BruteForce.sortedKeys(query, base, distance, 10);
                assertThat(neighbours.stream().mapToLong(HnswGraph.Neighbour::key).toArray())
                        .as(distance.sqlName())
                        .containsExactly(Arrays.stream(expected).asLongStream().toArray());
            }
        }
    }

    /**
     * Pins a property of HNSW rather than of this implementation, so that the documentation stays
     * honest about it. Dot product is not a distance once norms differ: a long vector has a larger
     * product with almost everything than a short one does, so the neighbour selection prunes
     * short vectors out of every list and nothing links to them any more. An exhaustive search
     * then cannot reach them, which is why the documentation asks for unit-norm vectors.
     */
    @Test
    public void testDotProductOverUnnormalisedVectorsCanLeaveNodesUnreachable()
    {
        double[][] base = VectorDataset.generate(CLUSTERED, 300, 1, 8, 21L).base();
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.DOT_PRODUCT, base));

        List<HnswGraph.Neighbour> reached = graph.search(VectorBlocks.doubleVector(base[0]), ElementType.DOUBLE, base.length, base.length);

        assertThat(reached).hasSizeLessThan(base.length);
    }

    @Test
    public void testDistancesAreTheRawMetricValuesNearestFirst()
    {
        double[][] base = {{0, 0}, {3, 4}, {1, 0}, {0, -2}};
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.EUCLIDEAN, base));

        List<HnswGraph.Neighbour> neighbours = graph.search(VectorBlocks.doubleVector(new double[] {0, 0}), ElementType.DOUBLE, 4, 4);

        assertThat(neighbours).containsExactly(
                new HnswGraph.Neighbour(0, 0.0),
                new HnswGraph.Neighbour(2, 1.0),
                new HnswGraph.Neighbour(3, 2.0),
                new HnswGraph.Neighbour(1, 5.0));
    }

    @Test
    public void testDotProductReturnsTheHighestSimilarityFirst()
    {
        double[][] base = {{1, 0}, {3, 0}, {-2, 0}, {2, 0}};
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.DOT_PRODUCT, base));

        List<HnswGraph.Neighbour> neighbours = graph.search(VectorBlocks.doubleVector(new double[] {1, 0}), ElementType.DOUBLE, 2, 4);

        assertThat(neighbours).containsExactly(
                new HnswGraph.Neighbour(1, 3.0),
                new HnswGraph.Neighbour(3, 2.0));
    }

    @Test
    public void testKLargerThanTheGraphReturnsEveryNode()
    {
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.EUCLIDEAN, new double[][] {{0}, {1}, {2}}));

        assertThat(graph.search(VectorBlocks.doubleVector(new double[] {0}), ElementType.DOUBLE, 10, 10)).hasSize(3);
    }

    @Test
    public void testSingleVectorGraph()
    {
        HnswGraph graph = HnswGraph.read(build(ElementType.REAL, Metric.COSINE, new double[][] {{1, 1}}));

        assertThat(graph.search(VectorBlocks.realVector(new double[] {1, 1}), ElementType.REAL, 3, 3))
                .containsExactly(new HnswGraph.Neighbour(0, 0.0));
    }

    /**
     * Copies of one vector are all at distance zero from each other, so the neighbour selection
     * cannot tell them apart and a full list keeps the copies it already had. Up to {@code 2 * m}
     * copies every one of them still has room to be linked; past that the later ones can lose all
     * their incoming links, which the documentation states. This pins the side that holds.
     */
    @Test
    public void testDuplicateVectorsAreAllReturnedUpToTheLevelZeroListSize()
    {
        double[][] base = new double[2 * M][];
        for (int i = 0; i < base.length; i++) {
            base[i] = new double[] {1, 2, 3};
        }
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.EUCLIDEAN, base));

        List<HnswGraph.Neighbour> neighbours = graph.search(VectorBlocks.doubleVector(new double[] {1, 2, 3}), ElementType.DOUBLE, base.length, base.length);

        assertThat(neighbours).hasSize(base.length);
        assertThat(neighbours).allSatisfy(neighbour -> assertThat(neighbour.distance()).isEqualTo(0.0));
    }

    /**
     * A graph of reals answers in its own representation: the query is rounded to float first, as a
     * {@code CAST} would, and the distance is the one the {@code array(real)} kernels give.
     */
    @Test
    public void testDoubleQueryAgainstARealGraphIsRoundedToFloat()
    {
        double[][] base = {{0.1, 0.2}, {5, 5}};
        HnswGraph graph = HnswGraph.read(build(ElementType.REAL, Metric.EUCLIDEAN_SQUARED, base));

        double[] query = {0.3, 0.7};
        List<HnswGraph.Neighbour> neighbours = graph.search(VectorBlocks.doubleVector(query), ElementType.DOUBLE, 1, 2);

        double dx = (double) (float) 0.1 - (double) (float) 0.3;
        double dy = (double) (float) 0.2 - (double) (float) 0.7;
        assertThat(neighbours).hasSize(1);
        assertThat(neighbours.getFirst().key()).isEqualTo(0);
        assertThat(neighbours.getFirst().distance()).isEqualTo(dx * dx + dy * dy);
    }

    @Test
    public void testRealQueryAgainstADoubleGraphIsWidened()
    {
        double[][] base = {{0.5, 0.25}, {4, 4}};
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.MANHATTAN, base));

        List<HnswGraph.Neighbour> neighbours = graph.search(VectorBlocks.realVector(new double[] {1, 1}), ElementType.REAL, 1, 2);

        assertThat(neighbours).containsExactly(new HnswGraph.Neighbour(0, 1.25));
    }

    /**
     * The graph is a function of the set of rows: the engine is free to hand them over in any
     * order and split across any number of partial states, and a rebuild has to reproduce the same
     * bytes for a stored graph to be comparable with, or replaceable by, the next one.
     */
    @Test
    public void testGraphDoesNotDependOnArrivalOrderOrOnHowTheRowsWereSplit()
    {
        double[][] base = VectorDataset.generate(CLUSTERED, 200, 1, 8, 5L).base();

        GraphInput forward = input(ElementType.REAL, Metric.COSINE, 8);
        for (int i = 0; i < base.length; i++) {
            forward.add(i, VectorBlocks.realVector(base[i]));
        }
        GraphInput backward = input(ElementType.REAL, Metric.COSINE, 8);
        for (int i = base.length - 1; i >= 0; i--) {
            backward.add(i, VectorBlocks.realVector(base[i]));
        }
        GraphInput merged = input(ElementType.REAL, Metric.COSINE, 8);
        GraphInput other = input(ElementType.REAL, Metric.COSINE, 8);
        for (int i = 0; i < base.length; i++) {
            (i % 3 == 0 ? other : merged).add(i, VectorBlocks.realVector(base[i]));
        }
        merged.addAll(GraphInput.deserialize(other.serialize()));

        Slice expected = HnswGraphBuilder.build(forward);
        assertThat(HnswGraphBuilder.build(backward)).isEqualTo(expected);
        assertThat(HnswGraphBuilder.build(merged)).isEqualTo(expected);
    }

    @Test
    public void testIntermediateStateRoundTrips()
    {
        GraphInput input = input(ElementType.DOUBLE, Metric.MANHATTAN, 3);
        input.add(7, VectorBlocks.doubleVector(new double[] {1, 2, 3}));
        input.add(-4, VectorBlocks.doubleVector(new double[] {-1, 0.5, Double.MAX_VALUE}));

        GraphInput copy = GraphInput.deserialize(input.serialize());

        assertThat(copy.size()).isEqualTo(2);
        assertThat(copy.metric()).isEqualTo(Metric.MANHATTAN);
        assertThat(copy.m()).isEqualTo(M);
        assertThat(copy.efConstruction()).isEqualTo(EF_CONSTRUCTION);
        assertThat(copy.dimension()).isEqualTo(3);
        assertThat(copy.serialize()).isEqualTo(input.serialize());
    }

    @Test
    public void testMixedDimensionsAreRejected()
    {
        GraphInput input = input(ElementType.DOUBLE, Metric.EUCLIDEAN, 2);
        input.add(1, VectorBlocks.doubleVector(new double[] {1, 2}));

        assertThatThrownBy(() -> input.add(2, VectorBlocks.doubleVector(new double[] {1, 2, 3})))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The vectors of hnsw_build_agg must have the same length, found 2 and 3");
    }

    @Test
    public void testParametersMustBeConstantAcrossMergedStates()
    {
        GraphInput input = input(ElementType.DOUBLE, Metric.EUCLIDEAN, 1);
        GraphInput other = new GraphInput(ElementType.DOUBLE, Metric.COSINE, M, EF_CONSTRUCTION, 1, null);

        assertThatThrownBy(() -> input.addAll(other))
                .isInstanceOf(TrinoException.class)
                .hasMessage("metric must be constant within a group of hnsw_build_agg, found 'euclidean' and 'cosine'");
    }

    @Test
    public void testQueryOfAnotherDimensionIsRejected()
    {
        HnswGraph graph = HnswGraph.read(build(ElementType.DOUBLE, Metric.EUCLIDEAN, new double[][] {{0, 0}}));

        assertThatThrownBy(() -> graph.search(VectorBlocks.doubleVector(new double[] {0, 0, 0}), ElementType.DOUBLE, 1, 1))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The query vector must have the dimension of the graph, found 3 and 2");
    }

    @Test
    public void testArbitraryBytesAreRejected()
    {
        assertNotAGraph(Slices.EMPTY_SLICE);
        assertNotAGraph(Slices.utf8Slice("definitely not a graph, but long enough to hold a header"));
    }

    @Test
    public void testTruncatedOrExtendedGraphIsRejected()
    {
        Slice graph = build(ElementType.DOUBLE, Metric.EUCLIDEAN, new double[][] {{0, 0}, {1, 1}, {2, 2}});

        assertNotAGraph(graph.slice(0, graph.length() - 1));
        Slice extended = Slices.allocate(graph.length() + 1);
        extended.setBytes(0, graph);
        assertNotAGraph(extended);
    }

    /**
     * A neighbour id is read from the value and then used as an index, so one pointing past the
     * last node has to be caught when it is read, not when it is dereferenced.
     */
    @Test
    public void testNeighbourOutOfRangeIsRejected()
    {
        Slice graph = build(ElementType.DOUBLE, Metric.EUCLIDEAN, new double[][] {{0, 0}, {1, 1}});
        Slice corrupted = graph.copy();
        // The last int of the value is the last neighbour id written.
        corrupted.setInt(corrupted.length() - Integer.BYTES, 99);

        assertThatThrownBy(() -> HnswGraph.read(corrupted).search(VectorBlocks.doubleVector(new double[] {0, 0}), ElementType.DOUBLE, 2, 2))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The value is not a graph built by hnsw_build_agg");
    }

    @Test
    public void testUnknownFormatVersionIsRejected()
    {
        Slice graph = build(ElementType.DOUBLE, Metric.EUCLIDEAN, new double[][] {{0, 0}}).copy();
        graph.setByte(Integer.BYTES, 9);

        assertThatThrownBy(() -> HnswGraph.read(graph))
                .isInstanceOf(TrinoException.class)
                .hasMessage("Unsupported HNSW graph format version 9, expected 1; rebuild the graph with hnsw_build_agg");
    }

    @Test
    public void testLevelsAreGeometric()
    {
        double multiplier = 1.0 / Math.log(16);
        int[] counts = new int[4];
        int total = 100_000;
        for (long key = 0; key < total; key++) {
            int level = HnswGraphBuilder.levelOf(key, multiplier);
            if (level < counts.length) {
                counts[level]++;
            }
        }
        // P(level >= l) = 16^-l, so each level holds about 15/16 of the ones that reach it.
        assertThat(counts[0] / (double) total).isCloseTo(15.0 / 16, within(0.005));
        assertThat(counts[1] / (double) total).isCloseTo(15.0 / 256, within(0.005));
    }

    private static double[][] normalised(double[][] vectors)
    {
        double[][] result = new double[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
            double norm = Math.sqrt(Arrays.stream(vectors[i]).map(component -> component * component).sum());
            result[i] = Arrays.stream(vectors[i]).map(component -> component / norm).toArray();
        }
        return result;
    }

    private static void assertNotAGraph(Slice value)
    {
        assertThatThrownBy(() -> HnswGraph.read(value))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The value is not a graph built by hnsw_build_agg");
    }

    private static Slice build(ElementType elementType, Metric metric, double[][] base)
    {
        GraphInput input = input(elementType, metric, base[0].length);
        for (int i = 0; i < base.length; i++) {
            input.add(i, elementType == ElementType.DOUBLE ? VectorBlocks.doubleVector(base[i]) : VectorBlocks.realVector(base[i]));
        }
        return HnswGraphBuilder.build(input);
    }

    private static GraphInput input(ElementType elementType, Metric metric, int dimension)
    {
        return new GraphInput(elementType, metric, M, EF_CONSTRUCTION, dimension, null);
    }
}
