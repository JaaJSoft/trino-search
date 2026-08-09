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

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;

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
                @AggregationState("K") KnnState state,
                @SqlNullable @BlockPosition @SqlType("K") ValueBlock key,
                @BlockIndex int position,
                @SqlType("array(double)") Block vector,
                @SqlType("array(double)") Block queryVector,
                @SqlType(StandardTypes.BIGINT) long k,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            addCandidate(state, key, position, vector, queryVector, k, metricName, DOUBLE_READER);
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
                @AggregationState("K") KnnState state,
                BlockBuilder out)
        {
            writeResult(state, out);
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
                @AggregationState("K") KnnState state,
                @SqlNullable @BlockPosition @SqlType("K") ValueBlock key,
                @BlockIndex int position,
                @SqlType("array(real)") Block vector,
                @SqlType("array(real)") Block queryVector,
                @SqlType(StandardTypes.BIGINT) long k,
                @SqlType(StandardTypes.VARCHAR) Slice metricName)
        {
            addCandidate(state, key, position, vector, queryVector, k, metricName, REAL_READER);
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
                @AggregationState("K") KnnState state,
                BlockBuilder out)
        {
            writeResult(state, out);
        }
    }

    private static void addCandidate(
            KnnState state,
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

        Metric metric;
        if (state.getHeap() == null) {
            metric = Metric.fromName(metricName);
            state.setHeap(new KnnHeap((int) k, metric.higherIsCloser()));
            state.setK((int) k);
            state.setMetric(metric);
        }
        else {
            metric = state.getMetric();
            // Resolving the name costs two String allocations and a scan of the enum, on a path
            // the engine walks once per row for an argument that is constant in every real query.
            // Comparing the raw bytes against the canonical spelling answers the same question
            // for free. Anything else, including another spelling of the same metric, falls
            // through to the resolving check.
            if (state.getK() != (int) k || !metric.hasCanonicalName(metricName)) {
                checkConstantWithinGroup(state.getK(), (int) k, metric, Metric.fromName(metricName));
            }
        }

        if (vector.getPositionCount() != queryVector.getPositionCount()) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }
        if (vector.hasNull() || queryVector.hasNull()) {
            return;
        }

        // Handing the heap's current cut-off to the metric lets a candidate that cannot make the
        // result be abandoned part way through its distance, which once k neighbours are in is
        // most of them.
        double distance = metric.computeBounded(vector, queryVector, reader, state.getHeap().retentionLimit());
        state.addToHeap(key, position, distance);
    }

    private static void mergeStates(KnnState state, KnnState otherState)
    {
        KnnHeap other = otherState.getHeap();
        if (other == null) {
            return;
        }
        if (state.getHeap() == null) {
            state.setHeap(other);
            state.setK(otherState.getK());
            state.setMetric(otherState.getMetric());
            return;
        }
        checkConstantWithinGroup(state.getK(), otherState.getK(), state.getMetric(), otherState.getMetric());
        state.mergeIntoHeap(other);
    }

    /**
     * The heap is created from whichever row or partial state reaches this code first, so a
     * varying k or metric within the same group would otherwise be resolved silently by "first
     * one wins": the heap keeps ranking every later candidate by the first row's direction
     * (closer-is-smaller vs. closer-is-larger), which for a mismatched metric returns the
     * farthest neighbours under a plausible-looking label instead of failing.
     */
    private static void checkConstantWithinGroup(int k, int otherK, Metric metric, Metric otherMetric)
    {
        if (k != otherK) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "k must be constant within a group of knn_agg, found %s and %s".formatted(k, otherK));
        }
        if (metric != otherMetric) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "metric must be constant within a group of knn_agg, found '%s' and '%s'".formatted(metric.sqlName(), otherMetric.sqlName()));
        }
    }

    private static void writeResult(KnnState state, BlockBuilder out)
    {
        KnnHeap heap = state.getHeap();
        if (heap == null || heap.size() == 0) {
            out.appendNull();
            return;
        }

        ((ArrayBlockBuilder) out).buildEntry(elementBuilder -> {
            for (KnnHeap.Neighbour neighbour : heap.drainSorted()) {
                ((RowBlockBuilder) elementBuilder).buildEntry(fieldBuilders -> {
                    fieldBuilders.get(0).append(neighbour.key(), 0);
                    DOUBLE.writeDouble(fieldBuilders.get(1), neighbour.distance());
                });
            }
        });
    }
}
