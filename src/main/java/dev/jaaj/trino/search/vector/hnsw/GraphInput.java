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
import dev.jaaj.trino.search.vector.quantize.BinaryCodes;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.airlift.slice.SliceInput;
import io.airlift.slice.SliceOutput;
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.ByteArrayBlock;
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
import static io.trino.spi.type.TinyintType.TINYINT;
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
    // Only for INT8 codes, which mean nothing without them. A private copy, never a view onto the
    // page the first row came from.
    private final QuantizationBounds bounds;
    private final int unitsPerVector;
    private final int maxRows;

    private long[] keys;
    // Exactly one of these is non-null: longs for DOUBLE, ints for REAL, bytes for INT8 and BINARY.
    private long[] longs;
    private int[] ints;
    private byte[] bytes;
    private int size;

    public GraphInput(ElementType elementType, Metric metric, int m, int efConstruction, int dimension, QuantizationBounds bounds)
    {
        this(elementType, metric, m, efConstruction, dimension, bounds == null ? null : copyOf(bounds), INITIAL_CAPACITY);
    }

    private GraphInput(ElementType elementType, Metric metric, int m, int efConstruction, int dimension, QuantizationBounds bounds, int capacity)
    {
        if ((elementType == ElementType.INT8) != (bounds != null)) {
            throw new IllegalArgumentException("bounds are required for INT8 codes and only for them");
        }
        this.elementType = elementType;
        this.metric = metric;
        this.m = m;
        this.efConstruction = efConstruction;
        this.dimension = dimension;
        this.bounds = bounds;
        this.unitsPerVector = elementType.unitsPerVector(dimension);
        long fixedBytes = SERIALIZED_HEADER_BYTES + boundsBytes(elementType, dimension);
        this.maxRows = (int) Math.min(Integer.MAX_VALUE, (MAX_SERIALIZED_BYTES - fixedBytes) / (Long.BYTES + elementType.bytesPerVector(dimension)));
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

    QuantizationBounds bounds()
    {
        return bounds;
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
     * Every component of every row of an array representation, as one block: row {@code i} is the
     * region starting at {@code i * dimension}. Only valid until the next {@link #add} or
     * {@link #addAll}, which may move the components to a larger array.
     */
    Block components()
    {
        int count = size * unitsPerVector;
        return switch (elementType) {
            case DOUBLE -> new LongArrayBlock(count, Optional.empty(), longs);
            case REAL -> new IntArrayBlock(count, Optional.empty(), ints);
            case INT8 -> new ByteArrayBlock(count, Optional.empty(), bytes);
            case BINARY -> throw new IllegalStateException("binary codes are not an array");
        };
    }

    /**
     * Every binary code, back to back: row {@code i} is the slice starting at
     * {@code i * unitsPerVector}. Same validity as {@link #components}.
     */
    Slice binaryCodes()
    {
        return Slices.wrappedBuffer(bytes, 0, size * unitsPerVector);
    }

    void writeVector(SliceOutput out, int row)
    {
        int offset = row * unitsPerVector;
        switch (elementType) {
            case DOUBLE -> out.writeLongs(longs, offset, unitsPerVector);
            case REAL -> out.writeInts(ints, offset, unitsPerVector);
            case INT8, BINARY -> out.writeBytes(bytes, offset, unitsPerVector);
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
     * The graph stores one set of bounds and computes every distance with it, so codes fitted
     * against different bounds in one group would be ranked as if they were comparable.
     */
    public void checkSameBounds(QuantizationBounds otherBounds)
    {
        if (!bounds.sameValuesAs(otherBounds)) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "bounds must be constant within a group of hnsw_build_agg");
        }
    }

    /**
     * Appends a row of an array representation. The caller has already skipped a vector with a
     * null component.
     */
    public void add(long key, Block vector)
    {
        checkSameDimension(vector.getPositionCount());
        ensureCapacity(size + 1);
        keys[size] = key;
        int base = size * unitsPerVector;
        switch (elementType) {
            case DOUBLE -> {
                for (int i = 0; i < dimension; i++) {
                    longs[base + i] = Double.doubleToRawLongBits(DOUBLE.getDouble(vector, i));
                }
            }
            case REAL -> {
                for (int i = 0; i < dimension; i++) {
                    ints[base + i] = Float.floatToRawIntBits(REAL.getFloat(vector, i));
                }
            }
            case INT8 -> {
                for (int i = 0; i < dimension; i++) {
                    bytes[base + i] = TINYINT.getByte(vector, i);
                }
            }
            case BINARY -> throw new IllegalStateException("binary codes are not an array");
        }
        size++;
    }

    /**
     * Appends a row of binary codes, keeping only the bytes the header says belong to the vector.
     */
    public void add(long key, Slice codes)
    {
        checkSameDimension(BinaryCodes.dimension(codes));
        ensureCapacity(size + 1);
        keys[size] = key;
        codes.getBytes(0, bytes, size * unitsPerVector, unitsPerVector);
        size++;
    }

    public void addAll(GraphInput other)
    {
        checkSameParameters(other.m, other.efConstruction, Slices.utf8Slice(other.metric.sqlName()));
        checkSameDimension(other.dimension);
        if (bounds != null) {
            checkSameBounds(other.bounds);
        }
        ensureCapacity(size + other.size);
        System.arraycopy(other.keys, 0, keys, size, other.size);
        int offset = size * unitsPerVector;
        int count = other.size * unitsPerVector;
        switch (elementType) {
            case DOUBLE -> System.arraycopy(other.longs, 0, longs, offset, count);
            case REAL -> System.arraycopy(other.ints, 0, ints, offset, count);
            case INT8, BINARY -> System.arraycopy(other.bytes, 0, bytes, offset, count);
        }
        size += other.size;
    }

    public long getRetainedSizeInBytes()
    {
        return INSTANCE_SIZE + sizeOf(keys) + sizeOf(longs) + sizeOf(ints) + sizeOf(bytes) + boundsBytes(elementType, dimension);
    }

    /**
     * The partial state as it travels between stages. Never persisted, so unlike a graph it needs
     * neither a version nor a stable metric encoding: both ends run the same plugin build.
     */
    public Slice serialize()
    {
        long length = SERIALIZED_HEADER_BYTES
                + boundsBytes(elementType, dimension)
                + (long) size * (Long.BYTES + elementType.bytesPerVector(dimension));
        Slice slice = Slices.allocate(toIntExact(length));
        SliceOutput out = slice.getOutput();
        out.writeByte(elementType.code());
        out.writeByte(metric.ordinal());
        out.writeInt(m);
        out.writeInt(efConstruction);
        out.writeInt(dimension);
        out.writeInt(size);
        if (bounds != null) {
            writeBounds(out, bounds);
        }
        out.writeLongs(keys, 0, size);
        int count = size * unitsPerVector;
        switch (elementType) {
            case DOUBLE -> out.writeLongs(longs, 0, count);
            case REAL -> out.writeInts(ints, 0, count);
            case INT8, BINARY -> out.writeBytes(bytes, 0, count);
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
        QuantizationBounds bounds = elementType == ElementType.INT8 ? readBounds(in, dimension) : null;

        GraphInput input = new GraphInput(elementType, metric, m, efConstruction, dimension, bounds, size);
        in.readLongs(input.keys, 0, size);
        int count = size * input.unitsPerVector;
        switch (elementType) {
            case DOUBLE -> in.readLongs(input.longs, 0, count);
            case REAL -> in.readInts(input.ints, 0, count);
            case INT8, BINARY -> in.readBytes(input.bytes, 0, count);
        }
        input.size = size;
        return input;
    }

    /**
     * The scale first, then one offset per dimension: the layout graphs store their bounds in too.
     */
    static void writeBounds(SliceOutput out, QuantizationBounds bounds)
    {
        out.writeDouble(bounds.scale());
        for (int i = 0; i < bounds.dimension(); i++) {
            out.writeDouble(bounds.offset(i));
        }
    }

    static QuantizationBounds readBounds(SliceInput in, int dimension)
    {
        double scale = in.readDouble();
        double[] offsets = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            offsets[i] = in.readDouble();
        }
        return QuantizationBounds.of(offsets, scale);
    }

    static long boundsBytes(ElementType elementType, int dimension)
    {
        return elementType == ElementType.INT8 ? Double.BYTES * (1L + dimension) : 0;
    }

    private static QuantizationBounds copyOf(QuantizationBounds bounds)
    {
        double[] offsets = new double[bounds.dimension()];
        for (int i = 0; i < offsets.length; i++) {
            offsets[i] = bounds.offset(i);
        }
        return QuantizationBounds.of(offsets, bounds.scale());
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
        switch (elementType) {
            case DOUBLE -> longs = Arrays.copyOf(longs, capacity * unitsPerVector);
            case REAL -> ints = Arrays.copyOf(ints, capacity * unitsPerVector);
            case INT8, BINARY -> bytes = Arrays.copyOf(bytes, capacity * unitsPerVector);
        }
    }

    private void allocate(int capacity)
    {
        keys = new long[capacity];
        switch (elementType) {
            case DOUBLE -> longs = new long[capacity * unitsPerVector];
            case REAL -> ints = new int[capacity * unitsPerVector];
            case INT8, BINARY -> bytes = new byte[capacity * unitsPerVector];
        }
    }

    private static TrinoException notConstant(String argument, Object value, Object otherValue)
    {
        return new TrinoException(
                INVALID_FUNCTION_ARGUMENT,
                "%s must be constant within a group of hnsw_build_agg, found %s and %s".formatted(argument, value, otherValue));
    }
}
