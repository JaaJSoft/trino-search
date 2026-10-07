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
import dev.jaaj.trino.search.vector.VectorReader;
import dev.jaaj.trino.search.vector.hnsw.NodeQueue.ScoredNodes;
import io.airlift.slice.Slice;
import io.airlift.slice.SliceOutput;
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;

import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.stream.IntStream;

import static dev.jaaj.trino.search.vector.hnsw.LayerSearch.toRank;
import static dev.jaaj.trino.search.vector.hnsw.LayerSearch.toValue;
import static io.trino.spi.StandardErrorCode.EXCEEDED_FUNCTION_MEMORY_LIMIT;

/**
 * Builds an HNSW graph (Malkov and Yashunin, 2016) over the rows of a {@link GraphInput} and
 * writes it in the format {@link HnswGraph} reads.
 * <p>
 * The graph is a pure function of the set of rows. Rows are inserted in an order derived from a
 * hash of their key, and each node's level is drawn from a second hash of the same key, so neither
 * depends on the order the rows reached the aggregation. Two builds over the same rows with
 * distinct keys therefore produce identical bytes however the engine split them, and the hashed
 * order is also what keeps an input sorted by key, or arriving cluster by cluster, from being
 * inserted in an order that leaves the graph poorly connected.
 */
public final class HnswGraphBuilder
        implements GraphLayers
{
    private static final long LEVEL_SALT = 0x9E3779B97F4A7C15L;

    private final GraphInput input;
    private final Metric metric;
    private final VectorReader reader;
    private final int m;
    private final int dimension;
    private final int nodeCount;
    private final Block components;
    private final LayerSearch search;

    /**
     * The input row of each node. Node ids are insertion ranks, which is what makes them
     * independent of arrival order.
     */
    private final int[] rows;
    private final byte[] levels;

    /**
     * For each node and each of its levels, the neighbour count followed by the neighbours, sized
     * for the most that level allows.
     */
    private final int[][][] links;

    private final int[] selectedNodes;
    private final double[] selectedRanks;

    private int entryPoint = -1;
    private int maxLevel = -1;

    private HnswGraphBuilder(GraphInput input)
    {
        this.input = input;
        this.metric = input.metric();
        this.reader = input.elementType().reader();
        this.m = input.m();
        this.dimension = input.dimension();
        this.nodeCount = input.size();
        this.components = input.components();
        this.rows = IntStream.range(0, nodeCount)
                .boxed()
                .sorted(Comparator.comparingLong((Integer row) -> mix(input.key(row))))
                .mapToInt(Integer::intValue)
                .toArray();
        double levelMultiplier = 1.0 / Math.log(m);
        this.levels = new byte[nodeCount];
        for (int node = 0; node < nodeCount; node++) {
            levels[node] = (byte) levelOf(input.key(rows[node]), levelMultiplier);
        }
        this.links = new int[nodeCount][][];
        this.search = new LayerSearch(this, metric, reader, maxNeighbours(0));
        this.selectedNodes = new int[maxNeighbours(0)];
        this.selectedRanks = new double[maxNeighbours(0)];
    }

    /**
     * The serialized graph over every row of {@code input}, which must hold at least one.
     */
    public static Slice build(GraphInput input)
    {
        HnswGraphBuilder builder = new HnswGraphBuilder(input);
        for (int node = 0; node < builder.nodeCount; node++) {
            builder.insert(node);
        }
        return builder.write();
    }

    @Override
    public int nodeCount()
    {
        return nodeCount;
    }

    @Override
    public int neighbours(int node, int level, int[] into)
    {
        int[] list = links[node][level];
        System.arraycopy(list, 1, into, 0, list[0]);
        return list[0];
    }

    /**
     * A region of the input's own array, distinct per node and never overwritten while the graph
     * is built, so it is safe as either operand of a metric, more than the contract promises.
     */
    @Override
    public Block vector(int node)
    {
        return components.getRegion(rows[node] * dimension, dimension);
    }

    private int maxNeighbours(int level)
    {
        return level == 0 ? 2 * m : m;
    }

    /**
     * Algorithm 1 of the paper, with the neighbour selection heuristic of algorithm 4.
     */
    private void insert(int node)
    {
        int level = levels[node];
        links[node] = new int[level + 1][];
        for (int i = 0; i <= level; i++) {
            links[node][i] = new int[maxNeighbours(i) + 1];
        }
        if (entryPoint < 0) {
            entryPoint = node;
            maxLevel = level;
            return;
        }

        Block query = vector(node);
        ScoredNodes entryPoints = ScoredNodes.single(entryPoint, search.rank(entryPoint, query, Double.POSITIVE_INFINITY));
        for (int i = maxLevel; i > level; i--) {
            entryPoints = search.search(query, entryPoints, 1, i);
        }
        for (int i = Math.min(level, maxLevel); i >= 0; i--) {
            ScoredNodes found = search.search(query, entryPoints, input.efConstruction(), i);
            int selected = selectNeighbours(found, m);
            int[] own = links[node][i];
            own[0] = selected;
            System.arraycopy(selectedNodes, 0, own, 1, selected);
            // Copied out first: connecting a neighbour reuses the selection buffers.
            int[] neighbours = new int[selected];
            double[] ranks = new double[selected];
            System.arraycopy(selectedNodes, 0, neighbours, 0, selected);
            System.arraycopy(selectedRanks, 0, ranks, 0, selected);
            for (int j = 0; j < selected; j++) {
                connect(neighbours[j], node, ranks[j], i);
            }
            entryPoints = found;
        }
        if (level > maxLevel) {
            entryPoint = node;
            maxLevel = level;
        }
    }

    /**
     * Adds {@code node} to the neighbours of {@code neighbour} on {@code level}. A full list is
     * re-selected from scratch among its current members and the newcomer, so the newcomer is kept
     * only if it earns its place.
     */
    private void connect(int neighbour, int node, double rank, int level)
    {
        int[] list = links[neighbour][level];
        int count = list[0];
        if (count < list.length - 1) {
            list[count + 1] = node;
            list[0] = count + 1;
            return;
        }

        Block base = vector(neighbour);
        int[] nodes = new int[count + 1];
        double[] ranks = new double[count + 1];
        for (int i = 0; i < count; i++) {
            nodes[i] = list[i + 1];
            ranks[i] = toRank(metric, metric.compute(vector(nodes[i]), base, reader));
        }
        nodes[count] = node;
        ranks[count] = rank;
        sortNearestFirst(nodes, ranks);

        int kept = selectNeighbours(ScoredNodes.ofSorted(nodes, ranks), count);
        list[0] = kept;
        System.arraycopy(selectedNodes, 0, list, 1, kept);
    }

    /**
     * The neighbour selection heuristic: walking the candidates nearest first, one is kept only if
     * it is closer to the base than to every neighbour already kept. Keeping the plain nearest ones
     * instead would spend the whole list on a single tight cluster and leave no edge leading out of
     * it, which is what makes clustered data hard for a graph search.
     * <p>
     * Writes the kept candidates into {@link #selectedNodes} and {@link #selectedRanks} and returns
     * how many there are.
     */
    private int selectNeighbours(ScoredNodes candidates, int limit)
    {
        int selected = 0;
        for (int i = 0; i < candidates.count() && selected < limit; i++) {
            int candidate = candidates.node(i);
            double rank = candidates.rank(i);
            Block candidateVector = vector(candidate);
            boolean diverse = true;
            for (int j = 0; j < selected && diverse; j++) {
                // The candidate is the second operand because it is the one that stays the same
                // through this loop, which is the operand cosine remembers the magnitude of.
                double between = toRank(metric, metric.computeBounded(vector(selectedNodes[j]), candidateVector, reader, toValue(metric, rank)));
                diverse = Double.compare(between, rank) >= 0;
            }
            if (diverse) {
                selectedNodes[selected] = candidate;
                selectedRanks[selected] = rank;
                selected++;
            }
        }
        return selected;
    }

    private Slice write()
    {
        byte[] metricName = metric.sqlName().getBytes(StandardCharsets.UTF_8);
        long linksLength = 0;
        for (int[][] nodeLinks : links) {
            for (int[] list : nodeLinks) {
                linksLength += 1 + list[0];
            }
        }
        long totalBytes = HnswGraph.headerBytes(metricName.length)
                + (long) nodeCount * (Long.BYTES + (long) dimension * input.elementType().byteSize() + Byte.BYTES + Integer.BYTES)
                + linksLength * Integer.BYTES;
        if (totalBytes > GraphInput.MAX_SERIALIZED_BYTES) {
            throw new TrinoException(
                    EXCEEDED_FUNCTION_MEMORY_LIMIT,
                    "The HNSW graph over %s vectors of dimension %s would take %s bytes, more than a single varbinary value can hold; partition the input more finely"
                            .formatted(nodeCount, dimension, totalBytes));
        }

        Slice slice = Slices.allocate((int) totalBytes);
        SliceOutput out = slice.getOutput();
        out.writeInt(HnswGraph.MAGIC);
        out.writeByte(HnswGraph.FORMAT_VERSION);
        out.writeByte(input.elementType().code());
        out.writeByte(metricName.length);
        out.writeBytes(metricName);
        out.writeInt(dimension);
        out.writeInt(m);
        out.writeInt(nodeCount);
        out.writeInt(entryPoint);
        out.writeInt(maxLevel);
        out.writeInt((int) linksLength);
        for (int node = 0; node < nodeCount; node++) {
            out.writeLong(input.key(rows[node]));
        }
        for (int node = 0; node < nodeCount; node++) {
            input.writeComponents(out, rows[node]);
        }
        out.writeBytes(levels);
        int offset = 0;
        for (int[][] nodeLinks : links) {
            out.writeInt(offset);
            for (int[] list : nodeLinks) {
                offset += 1 + list[0];
            }
        }
        for (int[][] nodeLinks : links) {
            for (int[] list : nodeLinks) {
                out.writeInts(list, 0, 1 + list[0]);
            }
        }
        return slice;
    }

    /**
     * Insertion sort, since a list never holds more than {@code 2 * m + 1} entries.
     */
    private static void sortNearestFirst(int[] nodes, double[] ranks)
    {
        for (int i = 1; i < nodes.length; i++) {
            int node = nodes[i];
            double rank = ranks[i];
            int j = i - 1;
            while (j >= 0 && Double.compare(ranks[j], rank) > 0) {
                nodes[j + 1] = nodes[j];
                ranks[j + 1] = ranks[j];
                j--;
            }
            nodes[j + 1] = node;
            ranks[j + 1] = rank;
        }
    }

    /**
     * The level a key's node lives up to: geometric, with the level multiplier {@code 1 / ln(m)}
     * the paper recommends, drawn from a hash of the key rather than from a random generator so
     * that a rebuild over the same rows reproduces it.
     */
    static int levelOf(long key, double levelMultiplier)
    {
        // The top 53 bits as a double in (0, 1], so the logarithm is always finite.
        double uniform = ((mix(key + LEVEL_SALT) >>> 11) + 1) * 0x1.0p-53;
        return (int) (-Math.log(uniform) * levelMultiplier);
    }

    /**
     * The splitmix64 finaliser. A bijection on 64-bit values, so distinct keys never share an
     * insertion rank.
     */
    static long mix(long value)
    {
        long mixed = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
        mixed = (mixed ^ (mixed >>> 27)) * 0x94D049BB133111EBL;
        return mixed ^ (mixed >>> 31);
    }
}
