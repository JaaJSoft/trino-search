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
import dev.jaaj.trino.search.vector.quantize.BinaryCodes;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.ByteArrayBlock;
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.IntFunction;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static dev.jaaj.trino.search.vector.hnsw.LayerSearch.toValue;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.TinyintType.TINYINT;

/**
 * A serialized HNSW graph, searched in place.
 * <p>
 * Nothing is decoded up front. A search touches a few thousand nodes whatever the size of the
 * graph, so copying every vector out of the value first would turn it back into a pass over the
 * whole partition. Each visited node's components are copied into one reused buffer instead, or,
 * for binary codes, read through a view onto the value, and its neighbours are read straight from
 * the value.
 * <p>
 * The layout, every number little-endian:
 * <pre>
 * int     magic, "HNSW"
 * byte    format version
 * byte    representation code
 * byte    length of the metric name, then its UTF-8 bytes
 * int     dimension, m, node count, entry point, top level, length of the link section in ints
 * double  for int8 codes only: the scale, then one offset per dimension
 * long    key of each node
 * ...     the vector of each node, in the layout {@link ElementType} describes
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
    private final QuantizationBounds bounds;

    private final int keysOffset;
    private final int vectorsOffset;
    private final int unitsPerVector;
    private final int bytesPerVector;
    private final int levelsOffset;
    private final int linkOffsetsOffset;
    private final int linksOffset;

    private HnswGraph(
            Slice slice,
            ElementType elementType,
            Metric metric,
            int headerBytes,
            int dimension,
            int m,
            int nodeCount,
            int entryPoint,
            int maxLevel,
            int linksLength,
            QuantizationBounds bounds)
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
        this.bounds = bounds;

        this.unitsPerVector = elementType.unitsPerVector(dimension);
        this.bytesPerVector = (int) elementType.bytesPerVector(dimension);
        this.keysOffset = headerBytes + (int) GraphInput.boundsBytes(elementType, dimension);
        this.vectorsOffset = keysOffset + nodeCount * Long.BYTES;
        this.levelsOffset = vectorsOffset + nodeCount * bytesPerVector;
        this.linkOffsetsOffset = levelsOffset + nodeCount;
        this.linksOffset = linkOffsetsOffset + nodeCount * Integer.BYTES;
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
                + GraphInput.boundsBytes(elementType, dimension)
                + (long) nodeCount * (Long.BYTES + elementType.bytesPerVector(dimension) + Byte.BYTES + Integer.BYTES)
                + (long) linksLength * Integer.BYTES;
        if (expectedLength != slice.length()) {
            throw corrupt();
        }
        QuantizationBounds bounds = elementType == ElementType.INT8
                ? GraphInput.readBounds(slice.slice(headerBytes, slice.length() - headerBytes).getInput(), dimension)
                : null;
        return new HnswGraph(slice, elementType, metric, headerBytes, dimension, m, nodeCount, entryPoint, maxLevel, linksLength, bounds);
    }

    /**
     * The {@code k} nodes nearest to {@code query} that a search with a candidate list of
     * {@code ef} finds, nearest first, for a query of an array representation.
     * <p>
     * A float query against a float graph is converted to the graph's representation first, so the
     * distances returned are the ones the graph's own representation gives. An int8 query is used
     * as it is, and has to have been quantised with the bounds the graph was built with.
     */
    public List<Neighbour> search(Block query, ElementType queryType, int k, int ef)
    {
        checkQueryType(queryType);
        checkDimension(query.getPositionCount());
        Block plainQuery = plainCopy(query, queryType);
        return switch (elementType) {
            case DOUBLE -> {
                long[] buffer = new long[dimension];
                Block vector = new LongArrayBlock(dimension, Optional.empty(), buffer);
                yield search(plainQuery, node -> {
                    slice.getLongs(vectorOffset(node), buffer, 0, dimension);
                    return vector;
                }, VectorDistance.floats(metric, DOUBLE_READER), k, ef);
            }
            case REAL -> {
                int[] buffer = new int[dimension];
                Block vector = new IntArrayBlock(dimension, Optional.empty(), buffer);
                yield search(plainQuery, node -> {
                    slice.getInts(vectorOffset(node), buffer, 0, dimension);
                    return vector;
                }, VectorDistance.floats(metric, REAL_READER), k, ef);
            }
            case INT8 -> {
                byte[] buffer = new byte[dimension];
                Block vector = new ByteArrayBlock(dimension, Optional.empty(), buffer);
                yield search(plainQuery, node -> {
                    slice.getBytes(vectorOffset(node), buffer, 0, dimension);
                    return vector;
                }, VectorDistance.int8(metric, bounds), k, ef);
            }
            case BINARY -> throw new IllegalStateException("checked above");
        };
    }

    /**
     * The binary counterpart of {@link #search(Block, ElementType, int, int)}. A stored code is
     * read through a view onto the value, without a copy.
     */
    public List<Neighbour> searchBinary(Slice query, int k, int ef)
    {
        checkQueryType(ElementType.BINARY);
        checkDimension(BinaryCodes.dimension(query));
        return search(query, node -> slice.slice(vectorOffset(node), unitsPerVector), VectorDistance.binary(metric), k, ef);
    }

    private <V> List<Neighbour> search(V query, IntFunction<V> vectors, VectorDistance<V> distance, int k, int ef)
    {
        LayerSearch<V> search = new LayerSearch<>(this, vectors, distance, metric, 2 * m);
        ScoredNodes entryPoints = ScoredNodes.single(entryPoint, search.rank(entryPoint, query, Double.POSITIVE_INFINITY));
        for (int level = maxLevel; level > 0; level--) {
            entryPoints = search.search(query, entryPoints, 1, level);
        }
        ScoredNodes found = search.search(query, entryPoints, ef, 0);

        int count = Math.min(k, found.count());
        List<Neighbour> neighbours = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int node = found.node(i);
            neighbours.add(new Neighbour(slice.getLong(keysOffset + node * Long.BYTES), toValue(metric, found.rank(i))));
        }
        return neighbours;
    }

    private void checkQueryType(ElementType queryType)
    {
        if (queryType != elementType && !(queryType.isFloat() && elementType.isFloat())) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The graph was built from %s vectors and cannot be searched with a %s query".formatted(elementType.sqlType(), queryType.sqlType()));
        }
    }

    private void checkDimension(int queryDimension)
    {
        if (queryDimension != dimension) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The query vector must have the dimension of the graph, found %s and %s".formatted(queryDimension, dimension));
        }
    }

    /**
     * {@code query} as a block of the graph's representation with no null mask and its components
     * starting at offset zero of an array of its own, which is the shape every vectorised kernel
     * recognises. A double read into a real is rounded to the nearest float, exactly as a
     * {@code CAST} would.
     */
    private Block plainCopy(Block query, ElementType queryType)
    {
        int length = query.getPositionCount();
        return switch (elementType) {
            case DOUBLE -> {
                VectorReader reader = readerOf(queryType);
                long[] bits = new long[length];
                for (int i = 0; i < length; i++) {
                    bits[i] = Double.doubleToRawLongBits(reader.read(query, i));
                }
                yield new LongArrayBlock(length, Optional.empty(), bits);
            }
            case REAL -> {
                VectorReader reader = readerOf(queryType);
                int[] bits = new int[length];
                for (int i = 0; i < length; i++) {
                    bits[i] = Float.floatToRawIntBits((float) reader.read(query, i));
                }
                yield new IntArrayBlock(length, Optional.empty(), bits);
            }
            case INT8 -> {
                byte[] codes = new byte[length];
                for (int i = 0; i < length; i++) {
                    codes[i] = TINYINT.getByte(query, i);
                }
                yield new ByteArrayBlock(length, Optional.empty(), codes);
            }
            case BINARY -> throw new IllegalStateException("binary codes are not an array");
        };
    }

    private static VectorReader readerOf(ElementType floatType)
    {
        return floatType == ElementType.DOUBLE ? DOUBLE_READER : REAL_READER;
    }

    private int vectorOffset(int node)
    {
        return vectorsOffset + node * bytesPerVector;
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
