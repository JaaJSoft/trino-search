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
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.VectorReader;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.block.ValueBlock;
import io.trino.spi.function.AggregationFunction;
import io.trino.spi.function.AggregationState;
import io.trino.spi.function.BlockIndex;
import io.trino.spi.function.BlockPosition;
import io.trino.spi.function.CombineFunction;
import io.trino.spi.function.Description;
import io.trino.spi.function.InputFunction;
import io.trino.spi.function.OutputFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.function.TypeParameter;
import io.trino.spi.type.StandardTypes;
import io.trino.spi.type.Type;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.TypeUtils.readNativeValue;
import static io.trino.spi.type.TypeUtils.writeNativeValue;

/**
 * Returns the k nearest neighbours of a query vector within each group.
 * <p>
 * The engine requires every {@code @InputFunction} that shares an open type variable (here
 * {@code K}, the neighbour key) to resolve to an identical signature
 * ({@code ParametricImplementationsGroup.determineGenericSignature}, via
 * {@code FunctionsParserHelper.validateSignaturesCompatibility}), so a single generic aggregation
 * class cannot host both the {@code array(double)} and {@code array(real)} vector overloads: the
 * concrete {@code array(double)}/{@code array(real)} argument types collide with that "one
 * signature per open type variable" rule even though {@code K} itself stays open. The fix is an
 * outer, non-registered class holding one {@code @AggregationFunction}-annotated nested class per
 * concrete overload, each with its own input/combine/output methods, delegating to shared private
 * helpers on the outer class so the two overloads do not duplicate logic.
 */
public final class KnnAggregation
{
    /**
     * Mirrors the cap Trino's own {@code min_n}/{@code max_n} family enforces in
     * {@code io.trino.operator.aggregation.minmaxn.MinNStateFactory}: an unbounded k would let
     * {@link KnnHeap}'s constructor eagerly allocate {@code k}-sized arrays before any input row
     * is seen, which a large enough k turns into an out-of-memory kill of the worker process.
     */
    static final int MAX_K = 10_000;

    private KnnAggregation() {}

    @AggregationFunction("knn_agg")
    @Description("Returns the k nearest neighbours of a query vector within each group")
    public static final class OfDoubleVectors
    {
        private OfDoubleVectors() {}

        @InputFunction
        @TypeParameter("K")
        public static void input(
                @TypeParameter("K") Type keyType,
                @AggregationState("K") KnnState state,
                @SqlNullable @BlockPosition @SqlType("K") ValueBlock key,
                @BlockIndex int position,
                @SqlType("array(double)") Block vector,
                @SqlType("array(double)") Block queryVector,
                @SqlType(StandardTypes.BIGINT) long k,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            addCandidate(state, keyType, key, position, vector, queryVector, k, metricName, DOUBLE_READER);
        }

        @CombineFunction
        public static void combine(
                @AggregationState("K") KnnState state,
                @AggregationState("K") KnnState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction("array(row(K, double))")
        public static void output(
                @TypeParameter("K") Type keyType,
                @AggregationState("K") KnnState state,
                BlockBuilder out)
        {
            writeResult(keyType, state, out);
        }
    }

    @AggregationFunction("knn_agg")
    @Description("Returns the k nearest neighbours of a query vector within each group")
    public static final class OfRealVectors
    {
        private OfRealVectors() {}

        @InputFunction
        @TypeParameter("K")
        public static void input(
                @TypeParameter("K") Type keyType,
                @AggregationState("K") KnnState state,
                @SqlNullable @BlockPosition @SqlType("K") ValueBlock key,
                @BlockIndex int position,
                @SqlType("array(real)") Block vector,
                @SqlType("array(real)") Block queryVector,
                @SqlType(StandardTypes.BIGINT) long k,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            addCandidate(state, keyType, key, position, vector, queryVector, k, metricName, REAL_READER);
        }

        @CombineFunction
        public static void combine(
                @AggregationState("K") KnnState state,
                @AggregationState("K") KnnState otherState)
        {
            mergeStates(state, otherState);
        }

        @SqlNullable
        @OutputFunction("array(row(K, double))")
        public static void output(
                @TypeParameter("K") Type keyType,
                @AggregationState("K") KnnState state,
                BlockBuilder out)
        {
            writeResult(keyType, state, out);
        }
    }

    private static void addCandidate(
            KnnState state,
            Type keyType,
            ValueBlock key,
            int position,
            Block vector,
            Block queryVector,
            long k,
            Slice metricName,
            VectorReader reader)
    {
        if (k <= 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "k must be greater than zero, got " + k);
        }
        if (k > MAX_K) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "k of knn_agg must be less than or equal to %s; found %s".formatted(MAX_K, k));
        }

        Metric metric = Metric.fromName(metricName.toStringUtf8());
        KnnHeap heap = state.getHeap();
        if (heap == null) {
            heap = new KnnHeap((int) k, metric.higherIsCloser());
            state.setHeap(heap);
            state.setK((int) k);
            state.setMetricName(metricName.toStringUtf8());
        }
        else {
            checkConstantWithinGroup(state.getK(), (int) k, state.getMetricName(), metricName.toStringUtf8());
        }

        if (vector.getPositionCount() != queryVector.getPositionCount()) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }
        if (vector.hasNull() || queryVector.hasNull()) {
            return;
        }

        heap.add(readNativeValue(keyType, key, position), metric.compute(vector, queryVector, reader));
    }

    private static void mergeStates(KnnState state, KnnState otherState)
    {
        KnnHeap other = otherState.getHeap();
        if (other == null) {
            return;
        }
        KnnHeap heap = state.getHeap();
        if (heap == null) {
            state.setHeap(other);
            state.setK(otherState.getK());
            state.setMetricName(otherState.getMetricName());
            return;
        }
        checkConstantWithinGroup(state.getK(), otherState.getK(), state.getMetricName(), otherState.getMetricName());
        heap.mergeFrom(other);
    }

    /**
     * The heap is created from whichever row or partial state reaches this code first, so a
     * varying k or metric within the same group would otherwise be resolved silently by "first
     * one wins": the heap keeps ranking every later candidate by the first row's direction
     * (closer-is-smaller vs. closer-is-larger), which for a mismatched metric returns the
     * farthest neighbours under a plausible-looking label instead of failing.
     */
    private static void checkConstantWithinGroup(int k, int otherK, String metricName, String otherMetricName)
    {
        if (k != otherK) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "k must be constant within a group of knn_agg, found %s and %s".formatted(k, otherK));
        }
        if (Metric.fromName(metricName) != Metric.fromName(otherMetricName)) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "metric must be constant within a group of knn_agg, found '%s' and '%s'".formatted(metricName, otherMetricName));
        }
    }

    private static void writeResult(Type keyType, KnnState state, BlockBuilder out)
    {
        KnnHeap heap = state.getHeap();
        if (heap == null || heap.size() == 0) {
            out.appendNull();
            return;
        }

        ((ArrayBlockBuilder) out).buildEntry(elementBuilder -> {
            for (KnnHeap.Neighbour neighbour : heap.drainSorted()) {
                ((RowBlockBuilder) elementBuilder).buildEntry(fieldBuilders -> {
                    writeNativeValue(keyType, fieldBuilders.get(0), neighbour.key());
                    DOUBLE.writeDouble(fieldBuilders.get(1), neighbour.distance());
                });
            }
        });
    }
}
