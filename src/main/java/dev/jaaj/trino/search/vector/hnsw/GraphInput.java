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
import io.airlift.slice.Slice;
import io.airlift.slice.SliceInput;
import io.airlift.slice.SliceOutput;
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.util.Arrays;
import java.util.Optional;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;
import static io.trino.spi.StandardErrorCode.EXCEEDED_FUNCTION_MEMORY_LIMIT;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;
import static java.lang.Math.toIntExact;

/**
 * The rows of one group of {@code hnsw_build_agg}: the parameters the graph will be built with,
 * and every key and vector seen so far, in arrival order.
 * <p>
 * No graph is built until the aggregation's output. Merging two partial graphs would mean
 * re-inserting one into the other anyway, so the partial states carry the rows alone, and a combine
 * is an append. It also leaves the final graph a function of the set of rows rather than of how the
 * engine happened to split them, which is what lets {@link HnswGraphBuilder} make it deterministic.
 */
public final class GraphInput
{
    private static final long INSTANCE_SIZE = instanceSize(GraphInput.class);

    /**
     * The largest {@code byte[]} a JVM allocates, and therefore the largest varbinary value, which
     * bounds both the serialized partial state and the graph built from it.
     */
    static final int MAX_SERIALIZED_BYTES = Integer.MAX_VALUE - 8;

    private static final int SERIALIZED_HEADER_BYTES = 2 * Byte.BYTES + 4 * Integer.BYTES;
    private static final int INITIAL_CAPACITY = 16;

    private final ElementType elementType;
    private final Metric metric;
    private final int m;
    private final int efConstruction;
    private final int dimension;
    private final int maxRows;

    private long[] keys;
    // Exactly one of these is non-null, the one matching elementType.
    private long[] doubleComponents;
    private int[] realComponents;
    private int size;

    public GraphInput(ElementType elementType, Metric metric, int m, int efConstruction, int dimension)
    {
        this(elementType, metric, m, efConstruction, dimension, INITIAL_CAPACITY);
    }

    private GraphInput(ElementType elementType, Metric metric, int m, int efConstruction, int dimension, int capacity)
    {
        this.elementType = elementType;
        this.metric = metric;
        this.m = m;
        this.efConstruction = efConstruction;
        this.dimension = dimension;
        this.maxRows = (int) Math.min(Integer.MAX_VALUE, (MAX_SERIALIZED_BYTES - SERIALIZED_HEADER_BYTES) / bytesPerRow(elementType, dimension));
        allocate(Math.min(capacity, maxRows));
    }

    public ElementType elementType()
    {
        return elementType;
    }

    public Metric metric()
    {
        return metric;
    }

    public int m()
    {
        return m;
    }

    public int efConstruction()
    {
        return efConstruction;
    }

    public int dimension()
    {
        return dimension;
    }

    public int size()
    {
        return size;
    }

    long key(int row)
    {
        return keys[row];
    }

    /**
     * Every component of every row, as one block of {@link #elementType}: row {@code i} is the
     * region starting at {@code i * dimension}. Only valid until the next {@link #add} or
     * {@link #addAll}, which may move the components to a larger array.
     */
    Block components()
    {
        int count = size * dimension;
        if (elementType == ElementType.DOUBLE) {
            return new LongArrayBlock(count, Optional.empty(), doubleComponents);
        }
        return new IntArrayBlock(count, Optional.empty(), realComponents);
    }

    void writeComponents(SliceOutput out, int row)
    {
        if (elementType == ElementType.DOUBLE) {
            out.writeLongs(doubleComponents, row * dimension, dimension);
        }
        else {
            out.writeInts(realComponents, row * dimension, dimension);
        }
    }

    /**
     * Fails unless a row's arguments are the ones this group started with. The heap of
     * {@code knn_agg} has the same rule for the same reason: a graph is built once, with one set of
     * parameters, so a varying argument would otherwise be settled silently by whichever row
     * arrived first.
     */
    public void checkSameParameters(long otherM, long otherEfConstruction, Slice otherMetricName)
    {
        if (otherM != m) {
            throw notConstant("m", m, otherM);
        }
        if (otherEfConstruction != efConstruction) {
            throw notConstant("ef_construction", efConstruction, otherEfConstruction);
        }
        // The canonical spelling is checked on the raw bytes first, since this runs once per row
        // for an argument that is constant in every real query.
        if (!metric.hasCanonicalName(otherMetricName)) {
            Metric otherMetric = Metric.fromName(otherMetricName);
            if (otherMetric != metric) {
                throw notConstant("metric", "'" + metric.sqlName() + "'", "'" + otherMetric.sqlName() + "'");
            }
        }
    }

    /**
     * Appends a row. The caller has already skipped a vector with a null component.
     */
    public void add(long key, Block vector)
    {
        checkSameDimension(vector.getPositionCount());
        ensureCapacity(size + 1);
        keys[size] = key;
        int base = size * dimension;
        if (elementType == ElementType.DOUBLE) {
            for (int i = 0; i < dimension; i++) {
                doubleComponents[base + i] = Double.doubleToRawLongBits(DOUBLE.getDouble(vector, i));
            }
        }
        else {
            for (int i = 0; i < dimension; i++) {
                realComponents[base + i] = Float.floatToRawIntBits(REAL.getFloat(vector, i));
            }
        }
        size++;
    }

    public void addAll(GraphInput other)
    {
        checkSameParameters(other.m, other.efConstruction, Slices.utf8Slice(other.metric.sqlName()));
        checkSameDimension(other.dimension);
        ensureCapacity(size + other.size);
        System.arraycopy(other.keys, 0, keys, size, other.size);
        if (elementType == ElementType.DOUBLE) {
            System.arraycopy(other.doubleComponents, 0, doubleComponents, size * dimension, other.size * dimension);
        }
        else {
            System.arraycopy(other.realComponents, 0, realComponents, size * dimension, other.size * dimension);
        }
        size += other.size;
    }

    public long getRetainedSizeInBytes()
    {
        return INSTANCE_SIZE + sizeOf(keys) + sizeOf(doubleComponents) + sizeOf(realComponents);
    }

    /**
     * The partial state as it travels between stages. Never persisted, so unlike a graph it needs
     * neither a version nor a stable metric encoding: both ends run the same plugin build.
     */
    public Slice serialize()
    {
        Slice slice = Slices.allocate(SERIALIZED_HEADER_BYTES + toIntExact((long) size * bytesPerRow(elementType, dimension)));
        SliceOutput out = slice.getOutput();
        out.writeByte(elementType.code());
        out.writeByte(metric.ordinal());
        out.writeInt(m);
        out.writeInt(efConstruction);
        out.writeInt(dimension);
        out.writeInt(size);
        out.writeLongs(keys, 0, size);
        if (elementType == ElementType.DOUBLE) {
            out.writeLongs(doubleComponents, 0, size * dimension);
        }
        else {
            out.writeInts(realComponents, 0, size * dimension);
        }
        return slice;
    }

    /**
     * Always a new instance. Trino deserializes every intermediate position into one reused
     * scratch state before combining it, so anything carried over from the previous position would
     * leak its rows into this one.
     */
    public static GraphInput deserialize(Slice slice)
    {
        SliceInput in = slice.getInput();
        ElementType elementType = ElementType.fromCode(in.readByte());
        Metric metric = Metric.values()[in.readByte()];
        int m = in.readInt();
        int efConstruction = in.readInt();
        int dimension = in.readInt();
        int size = in.readInt();

        GraphInput input = new GraphInput(elementType, metric, m, efConstruction, dimension, size);
        in.readLongs(input.keys, 0, size);
        if (elementType == ElementType.DOUBLE) {
            in.readLongs(input.doubleComponents, 0, size * dimension);
        }
        else {
            in.readInts(input.realComponents, 0, size * dimension);
        }
        input.size = size;
        return input;
    }

    private void checkSameDimension(int otherDimension)
    {
        if (otherDimension != dimension) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The vectors of hnsw_build_agg must have the same length, found %s and %s".formatted(dimension, otherDimension));
        }
    }

    private void ensureCapacity(int required)
    {
        if (required <= keys.length) {
            return;
        }
        if (required > maxRows) {
            throw new TrinoException(
                    EXCEEDED_FUNCTION_MEMORY_LIMIT,
                    "hnsw_build_agg cannot hold more than %s vectors of dimension %s in one group, since the graph has to fit in a single varbinary value; partition the input more finely"
                            .formatted(maxRows, dimension));
        }
        int capacity = (int) Math.min(maxRows, Math.max(required, keys.length + (long) (keys.length >> 1)));
        keys = Arrays.copyOf(keys, capacity);
        if (elementType == ElementType.DOUBLE) {
            doubleComponents = Arrays.copyOf(doubleComponents, capacity * dimension);
        }
        else {
            realComponents = Arrays.copyOf(realComponents, capacity * dimension);
        }
    }

    private void allocate(int capacity)
    {
        keys = new long[capacity];
        if (elementType == ElementType.DOUBLE) {
            doubleComponents = new long[capacity * dimension];
        }
        else {
            realComponents = new int[capacity * dimension];
        }
    }

    private static long bytesPerRow(ElementType elementType, int dimension)
    {
        return Long.BYTES + (long) dimension * elementType.byteSize();
    }

    private static TrinoException notConstant(String argument, Object value, Object otherValue)
    {
        return new TrinoException(
                INVALID_FUNCTION_ARGUMENT,
                "%s must be constant within a group of hnsw_build_agg, found %s and %s".formatted(argument, value, otherValue));
    }
}
