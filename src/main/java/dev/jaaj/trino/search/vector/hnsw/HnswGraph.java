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
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static dev.jaaj.trino.search.vector.hnsw.LayerSearch.toValue;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * A serialized HNSW graph, searched in place.
 * <p>
 * Nothing is decoded up front. A search touches a few thousand nodes whatever the size of the
 * graph, so copying every vector out of the value first would turn it back into a pass over the
 * whole partition. Each visited node's components are copied into one reused buffer instead, and
 * its neighbours are read straight from the value.
 * <p>
 * The layout, every number little-endian:
 * <pre>
 * int     magic, "HNSW"
 * byte    format version
 * byte    element type code
 * byte    length of the metric name, then its UTF-8 bytes
 * int     dimension, m, node count, entry point, top level, length of the link section in ints
 * long    key of each node
 * double  or float, the components of each node, dimension per node
 * byte    top level of each node
 * int     offset of each node's links in the link section, in ints
 * int     the link section: for each node and each of its levels from 0 up, the neighbour count
 *         followed by the neighbours
 * </pre>
 * The value is user input like any other varbinary, so every offset and node id read from it is
 * checked before it is used, and a value that does not hold together is rejected rather than read
 * out of bounds.
 */
public final class HnswGraph
        implements GraphLayers
{
    static final int MAGIC = 'H' | 'N' << 8 | 'S' << 16 | 'W' << 24;
    static final byte FORMAT_VERSION = 1;

    public record Neighbour(long key, double distance) {}

    private final Slice slice;
    private final ElementType elementType;
    private final Metric metric;
    private final int dimension;
    private final int m;
    private final int nodeCount;
    private final int entryPoint;
    private final int maxLevel;
    private final int linksLength;

    private final int keysOffset;
    private final int componentsOffset;
    private final int levelsOffset;
    private final int linkOffsetsOffset;
    private final int linksOffset;

    // One of these two backs the block vector() hands out, depending on the element type.
    private final long[] doubleBuffer;
    private final int[] realBuffer;
    private final Block vectorBuffer;

    private HnswGraph(Slice slice, ElementType elementType, Metric metric, int headerBytes, int dimension, int m, int nodeCount, int entryPoint, int maxLevel, int linksLength)
    {
        this.slice = slice;
        this.elementType = elementType;
        this.metric = metric;
        this.dimension = dimension;
        this.m = m;
        this.nodeCount = nodeCount;
        this.entryPoint = entryPoint;
        this.maxLevel = maxLevel;
        this.linksLength = linksLength;

        this.keysOffset = headerBytes;
        this.componentsOffset = keysOffset + nodeCount * Long.BYTES;
        this.levelsOffset = componentsOffset + nodeCount * dimension * elementType.byteSize();
        this.linkOffsetsOffset = levelsOffset + nodeCount;
        this.linksOffset = linkOffsetsOffset + nodeCount * Integer.BYTES;

        if (elementType == ElementType.DOUBLE) {
            this.doubleBuffer = new long[dimension];
            this.realBuffer = null;
            this.vectorBuffer = new LongArrayBlock(dimension, Optional.empty(), doubleBuffer);
        }
        else {
            this.doubleBuffer = null;
            this.realBuffer = new int[dimension];
            this.vectorBuffer = new IntArrayBlock(dimension, Optional.empty(), realBuffer);
        }
    }

    static int headerBytes(int metricNameLength)
    {
        return Integer.BYTES + 3 * Byte.BYTES + metricNameLength + 6 * Integer.BYTES;
    }

    public static HnswGraph read(Slice slice)
    {
        if (slice.length() < headerBytes(0) || slice.getInt(0) != MAGIC) {
            throw corrupt();
        }
        byte version = slice.getByte(Integer.BYTES);
        if (version != FORMAT_VERSION) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "Unsupported HNSW graph format version %s, expected %s; rebuild the graph with hnsw_build_agg".formatted(version, FORMAT_VERSION));
        }
        ElementType elementType = ElementType.fromCode(slice.getByte(Integer.BYTES + 1));
        int metricNameLength = slice.getUnsignedByte(Integer.BYTES + 2);
        int headerBytes = headerBytes(metricNameLength);
        if (elementType == null || slice.length() < headerBytes) {
            throw corrupt();
        }
        Metric metric = metricNamed(slice.toString(Integer.BYTES + 3, metricNameLength, StandardCharsets.UTF_8));
        int position = Integer.BYTES + 3 + metricNameLength;
        int dimension = slice.getInt(position);
        int m = slice.getInt(position + 4);
        int nodeCount = slice.getInt(position + 8);
        int entryPoint = slice.getInt(position + 12);
        int maxLevel = slice.getInt(position + 16);
        int linksLength = slice.getInt(position + 20);
        if (metric == null
                || dimension < 0
                || m < 2
                || nodeCount < 1
                || entryPoint < 0
                || entryPoint >= nodeCount
                || maxLevel < 0
                || linksLength < 0) {
            throw corrupt();
        }
        long expectedLength = headerBytes
                + (long) nodeCount * (Long.BYTES + (long) dimension * elementType.byteSize() + Byte.BYTES + Integer.BYTES)
                + (long) linksLength * Integer.BYTES;
        if (expectedLength != slice.length()) {
            throw corrupt();
        }
        return new HnswGraph(slice, elementType, metric, headerBytes, dimension, m, nodeCount, entryPoint, maxLevel, linksLength);
    }

    /**
     * The {@code k} nodes nearest to {@code query} that a search with a candidate list of
     * {@code ef} finds, nearest first. {@code query} is read through {@code queryReader} and then
     * converted to the graph's element type, so the distances returned are the ones the graph's own
     * representation gives.
     */
    public List<Neighbour> search(Block query, VectorReader queryReader, int k, int ef)
    {
        if (query.getPositionCount() != dimension) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The query vector must have the dimension of the graph, found %s and %s".formatted(query.getPositionCount(), dimension));
        }
        Block plainQuery = elementType.plainCopy(query, queryReader);
        LayerSearch search = new LayerSearch(this, metric, elementType.reader(), 2 * m);

        ScoredNodes entryPoints = ScoredNodes.single(entryPoint, search.rank(entryPoint, plainQuery, Double.POSITIVE_INFINITY));
        for (int level = maxLevel; level > 0; level--) {
            entryPoints = search.search(plainQuery, entryPoints, 1, level);
        }
        ScoredNodes found = search.search(plainQuery, entryPoints, ef, 0);

        int count = Math.min(k, found.count());
        List<Neighbour> neighbours = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int node = found.node(i);
            neighbours.add(new Neighbour(slice.getLong(keysOffset + node * Long.BYTES), toValue(metric, found.rank(i))));
        }
        return neighbours;
    }

    @Override
    public int nodeCount()
    {
        return nodeCount;
    }

    @Override
    public int neighbours(int node, int level, int[] into)
    {
        if (level > slice.getUnsignedByte(levelsOffset + node)) {
            throw corrupt();
        }
        int position = slice.getInt(linkOffsetsOffset + node * Integer.BYTES);
        for (int i = 0; i < level; i++) {
            position += 1 + link(position);
        }
        int count = link(position);
        if (count < 0 || count > (level == 0 ? 2 * m : m) || position + count >= linksLength) {
            throw corrupt();
        }
        for (int i = 0; i < count; i++) {
            int neighbour = slice.getInt(linksOffset + (position + 1 + i) * Integer.BYTES);
            if (neighbour < 0 || neighbour >= nodeCount) {
                throw corrupt();
            }
            into[i] = neighbour;
        }
        return count;
    }

    @Override
    public Block vector(int node)
    {
        int offset = componentsOffset + node * dimension * elementType.byteSize();
        if (elementType == ElementType.DOUBLE) {
            slice.getLongs(offset, doubleBuffer, 0, dimension);
        }
        else {
            slice.getInts(offset, realBuffer, 0, dimension);
        }
        return vectorBuffer;
    }

    private int link(int position)
    {
        if (position < 0 || position >= linksLength) {
            throw corrupt();
        }
        return slice.getInt(linksOffset + position * Integer.BYTES);
    }

    private static Metric metricNamed(String name)
    {
        for (Metric metric : Metric.values()) {
            if (metric.sqlName().equals(name)) {
                return metric;
            }
        }
        return null;
    }

    private static TrinoException corrupt()
    {
        return new TrinoException(INVALID_FUNCTION_ARGUMENT, "The value is not a graph built by hnsw_build_agg");
    }
}
