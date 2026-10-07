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
import dev.jaaj.trino.search.vector.knn.KnnStateFactory.SingleKnnState;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.block.ValueBlock;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.VarcharType.VARCHAR;
import static org.assertj.core.api.Assertions.assertThat;

public class TestKnnStateSerializer
{
    private static final KnnStateSerializer SERIALIZER = new KnnStateSerializer(VARCHAR);

    private static ValueBlock varcharKey(String value)
    {
        BlockBuilder builder = VARCHAR.createBlockBuilder(null, 1);
        if (value == null) {
            builder.appendNull();
        }
        else {
            VARCHAR.writeString(builder, value);
        }
        return builder.buildValueBlock();
    }

    /**
     * Alternating key and distance, so {@code state(EUCLIDEAN, 2, "a", 1.0, "b", 2.0)} holds two
     * neighbours.
     */
    private static SingleKnnState state(Metric metric, int k, Object... neighbours)
    {
        SingleKnnState state = new SingleKnnState();
        state.setHeap(new KnnHeap(k, metric.higherIsCloser()));
        state.setK(k);
        state.setMetric(metric);
        for (int i = 0; i < neighbours.length; i += 2) {
            state.addToHeap(varcharKey((String) neighbours[i]), 0, (double) neighbours[i + 1]);
        }
        return state;
    }

    private static Block serialized(KnnState... states)
    {
        BlockBuilder builder = SERIALIZER.getSerializedType().createBlockBuilder(null, states.length);
        for (KnnState state : states) {
            SERIALIZER.serialize(state, builder);
        }
        return builder.build();
    }

    private static SingleKnnState deserialized(Block block, int position)
    {
        SingleKnnState state = new SingleKnnState();
        SERIALIZER.deserialize(block, position, state);
        return state;
    }

    private static List<String> keysOf(KnnState state)
    {
        List<String> keys = new ArrayList<>();
        for (KnnHeap.Neighbour neighbour : state.getHeap().drainSorted()) {
            keys.add(neighbour.key().isNull(0) ? null : VARCHAR.getSlice(neighbour.key(), 0).toStringUtf8());
        }
        return keys;
    }

    private static List<Double> distancesOf(KnnState state)
    {
        return state.getHeap().drainSorted().stream()
                .map(KnnHeap.Neighbour::distance)
                .toList();
    }

    /**
     * Two flat arrays rather than one array of rows: a row per neighbour costs an entry to build
     * on the partial side and a row to unwrap on the final side, for the widest value the
     * aggregation ships between stages.
     */
    @Test
    public void testSerializedTypeCarriesKeysAndDistancesAsParallelArrays()
    {
        assertThat(SERIALIZER.getSerializedType()).isEqualTo(RowType.anonymous(List.of(
                BIGINT, VARCHAR, new ArrayType(VARCHAR), new ArrayType(DOUBLE))));
    }

    @Test
    public void testRoundTripKeepsEveryNeighbourWithItsDistance()
    {
        Block block = serialized(state(Metric.EUCLIDEAN, 3, "c", 3.0, "a", 1.0, "b", 2.0));

        SingleKnnState state = deserialized(block, 0);

        assertThat(state.getK()).isEqualTo(3);
        assertThat(state.getMetric()).isEqualTo(Metric.EUCLIDEAN);
        assertThat(keysOf(state)).containsExactly("a", "b", "c");
        assertThat(distancesOf(state)).containsExactly(1.0, 2.0, 3.0);
    }

    @Test
    public void testRoundTripOfAFullHeapReturnsTheSameNeighbours()
    {
        for (Metric metric : List.of(Metric.EUCLIDEAN, Metric.DOT_PRODUCT)) {
            SingleKnnState original = randomState(metric, 16, 200);

            SingleKnnState restored = deserialized(serialized(original), 0);

            assertThat(keysOf(restored)).containsExactlyElementsOf(keysOf(original));
            assertThat(distancesOf(restored)).containsExactlyElementsOf(distancesOf(original));
        }
    }

    /**
     * Only the final output is ordered by distance, so sorting on the way out of a partial state
     * is work nobody observes. Writing the heap's own order also hands the restoring side a
     * sequence that is already a valid heap, which it rebuilds without moving anything.
     */
    @Test
    public void testSerializeWritesNeighboursInHeapOrder()
    {
        SingleKnnState state = randomState(Metric.EUCLIDEAN, 16, 200);

        Block block = serialized(state);
        SqlRow row = serializedType().getObject(block, 0);
        Block distances = new ArrayType(DOUBLE).getObject(row.getRawFieldBlock(3), row.getRawIndex());

        List<Double> written = new ArrayList<>();
        for (int i = 0; i < distances.getPositionCount(); i++) {
            written.add(DOUBLE.getDouble(distances, i));
        }
        assertThat(written).containsExactlyElementsOf(state.getHeap().drainUnsorted().stream()
                .map(KnnHeap.Neighbour::distance)
                .toList());
    }

    private static RowType serializedType()
    {
        return (RowType) SERIALIZER.getSerializedType();
    }

    private static SingleKnnState randomState(Metric metric, int k, int candidates)
    {
        Random random = new Random(42);
        Object[] neighbours = new Object[2 * candidates];
        for (int i = 0; i < candidates; i++) {
            neighbours[2 * i] = "key" + i;
            neighbours[2 * i + 1] = random.nextDouble() * 1000;
        }
        return state(metric, k, neighbours);
    }

    /**
     * A partial state holds fewer than k neighbours whenever its split saw fewer than k rows of
     * the group; the restored heap must keep its capacity so the final stage can still fill it.
     */
    @Test
    public void testRoundTripOfAPartialHeapKeepsItsCapacity()
    {
        Block block = serialized(state(Metric.EUCLIDEAN, 3, "a", 1.0));

        SingleKnnState state = deserialized(block, 0);
        state.addToHeap(varcharKey("b"), 0, 2.0);
        state.addToHeap(varcharKey("c"), 0, 3.0);

        assertThat(keysOf(state)).containsExactly("a", "b", "c");
    }

    /**
     * The key is nullable, so a null key is a neighbour like any other and has to come back
     * paired with its own distance.
     */
    @Test
    public void testRoundTripKeepsNullKeys()
    {
        Block block = serialized(state(Metric.EUCLIDEAN, 3, "a", 2.0, null, 1.0, "b", 3.0));

        SingleKnnState state = deserialized(block, 0);

        assertThat(keysOf(state)).containsExactly(null, "a", "b");
        assertThat(distancesOf(state)).containsExactly(1.0, 2.0, 3.0);
    }

    @Test
    public void testRoundTripKeepsTheDirectionOfAHigherIsCloserMetric()
    {
        Block block = serialized(state(Metric.DOT_PRODUCT, 2, "a", 1.0, "b", 3.0, "c", 2.0));

        SingleKnnState state = deserialized(block, 0);
        state.addToHeap(varcharKey("d"), 0, 2.5);

        assertThat(state.getMetric()).isEqualTo(Metric.DOT_PRODUCT);
        assertThat(keysOf(state)).containsExactly("b", "d");
    }

    /**
     * An intermediate block holds one state per group, so each state's neighbours sit at an
     * offset inside the key and distance arrays. Reading from the start of those arrays instead
     * of from the requested position returns the first group's neighbours for every group.
     */
    @Test
    public void testDeserializeReadsOnlyTheRequestedPosition()
    {
        Block block = serialized(
                state(Metric.EUCLIDEAN, 2, "a", 1.0, "b", 2.0),
                state(Metric.MANHATTAN, 3, "x", 10.0, "y", 20.0, "z", 30.0));

        SingleKnnState state = deserialized(block, 1);

        assertThat(state.getK()).isEqualTo(3);
        assertThat(state.getMetric()).isEqualTo(Metric.MANHATTAN);
        assertThat(keysOf(state)).containsExactly("x", "y", "z");
        assertThat(distancesOf(state)).containsExactly(10.0, 20.0, 30.0);
    }

    /**
     * The engine deserializes every position of an intermediate block into one scratch state.
     * Restoring into the heap already attached would carry the previous position's neighbours
     * into the next group.
     */
    @Test
    public void testDeserializeReplacesWhatTheStateHeld()
    {
        Block block = serialized(
                state(Metric.EUCLIDEAN, 2, "a", 1.0, "b", 2.0),
                state(Metric.EUCLIDEAN, 2, "x", 10.0, "y", 20.0));
        SingleKnnState scratch = new SingleKnnState();

        SERIALIZER.deserialize(block, 0, scratch);
        SERIALIZER.deserialize(block, 1, scratch);

        assertThat(keysOf(scratch)).containsExactly("x", "y");
    }

    @Test
    public void testStateWithoutHeapSerializesAsNull()
    {
        Block block = serialized(new SingleKnnState());

        assertThat(block.isNull(0)).isTrue();
    }
}
