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
package dev.jaaj.trino.search.vector.benchmark;

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.quantize.BinaryCodes;
import dev.jaaj.trino.search.vector.quantize.BoundsState;
import dev.jaaj.trino.search.vector.quantize.BoundsStateFactory;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import dev.jaaj.trino.search.vector.quantize.VectorBoundsAggregation;
import io.airlift.slice.Slice;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.TinyintType.TINYINT;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestVectorBlocks
{
    @Test
    public void testDoubleVectorReadsBackThroughTheProductionReader()
    {
        double[] values = {1.5, -2.25, 0.0, 1e-300};
        Block block = VectorBlocks.doubleVector(values);
        assertThat(block.getPositionCount()).isEqualTo(4);
        for (int i = 0; i < values.length; i++) {
            assertThat(DOUBLE_READER.read(block, i)).isEqualTo(values[i]);
        }
    }

    @Test
    public void testRealVectorReadsBackThroughTheProductionReader()
    {
        double[] values = {1.5, -2.25, 0.0, 0.1};
        Block block = VectorBlocks.realVector(values);
        assertThat(block.getPositionCount()).isEqualTo(4);
        for (int i = 0; i < values.length; i++) {
            assertThat(REAL_READER.read(block, i)).isEqualTo((float) values[i]);
        }
    }

    /**
     * The encoding is only correct if the production metric code agrees with a hand computation.
     * Reading a value back through the same reader would still pass if both sides shared a wrong
     * assumption about which block class DOUBLE uses.
     */
    @Test
    public void testMetricAgreesWithHandComputation()
    {
        Block first = VectorBlocks.doubleVector(new double[] {3.0, 4.0});
        Block second = VectorBlocks.doubleVector(new double[] {0.0, 0.0});
        assertThat(Metric.EUCLIDEAN.compute(first, second, DOUBLE_READER)).isCloseTo(5.0, within(1e-12));
        assertThat(Metric.MANHATTAN.compute(first, second, DOUBLE_READER)).isCloseTo(7.0, within(1e-12));
    }

    @Test
    public void testRoundedToFloatMatchesTheRealBlockValues()
    {
        double[] values = {0.1, 0.2, 0.3};
        double[] rounded = VectorBlocks.roundedToFloat(values);
        Block block = VectorBlocks.realVector(values);
        for (int i = 0; i < values.length; i++) {
            assertThat(rounded[i]).isEqualTo(REAL_READER.read(block, i));
        }
    }

    @Test
    public void testFittedBoundsSpanTheDataAndCentreTheCodes()
    {
        double[][] vectors = {{-1.0, 5.0}, {3.0, 5.0}};
        QuantizationBounds bounds = VectorBlocks.fitBounds(vectors);
        assertThat(bounds.offset(0)).isEqualTo(1.0);
        assertThat(bounds.offset(1)).isEqualTo(5.0);
        // The global range spans both dimensions: -1 to 5, since dimension 1 never varies and
        // dimension 0 alone would understate it.
        assertThat(bounds.scale()).isEqualTo(6.0 / 255.0);
    }

    /**
     * The harness derives the offsets and the scale itself, so the hand-computed values above only
     * say that its own arithmetic is what it was written to be. Recall measured against parameters
     * no user would ever be given is not recall, so the derivation is pinned against the one
     * {@code vector_bounds_agg} actually produces.
     */
    @Test
    public void testFitBoundsAgreesWithTheAggregation()
    {
        double[][] vectors = {{-1.0, 5.0, 0.25}, {3.0, 5.0, -7.5}, {0.5, 5.0, 2.0}};
        QuantizationBounds harness = VectorBlocks.fitBounds(vectors);
        QuantizationBounds aggregated = throughVectorBoundsAgg(vectors);

        assertThat(harness.dimension()).isEqualTo(aggregated.dimension());
        assertThat(harness.scale()).isEqualTo(aggregated.scale());
        for (int i = 0; i < aggregated.dimension(); i++) {
            assertThat(harness.offset(i)).as("offset %s", i).isEqualTo(aggregated.offset(i));
        }
    }

    private static QuantizationBounds throughVectorBoundsAgg(double[][] vectors)
    {
        BoundsState state = new BoundsStateFactory().createSingleState();
        for (double[] vector : vectors) {
            VectorBoundsAggregation.OfDoubleVectors.input(state, VectorBlocks.doubleVector(vector));
        }
        ArrayType doubleArray = new ArrayType(DOUBLE);
        RowType rowType = RowType.anonymous(List.of(doubleArray, DOUBLE));
        BlockBuilder out = rowType.createBlockBuilder(null, 1);
        VectorBoundsAggregation.OfDoubleVectors.output(state, out);
        return QuantizationBounds.of(rowType.getObject(out.build(), 0));
    }

    @Test
    public void testInt8VectorEncodesAgainstTheBounds()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0.0}, 1.0);
        Block codes = VectorBlocks.int8Vector(new double[] {7.0}, bounds);
        assertThat(codes.getPositionCount()).isEqualTo(1);
        assertThat(TINYINT.getByte(codes, 0)).isEqualTo((byte) 7);
    }

    @Test
    public void testBinaryVectorSetsABitPerComponentAboveTheMidpoint()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0.0, 0.0}, 1.0);
        Slice codes = VectorBlocks.binaryVector(new double[] {1.0, -1.0}, bounds);
        assertThat(BinaryCodes.dimension(codes)).isEqualTo(2);
        assertThat(BinaryCodes.hamming(codes, VectorBlocks.binaryVector(new double[] {1.0, 1.0}, bounds)))
                .isEqualTo(1);
    }
}
