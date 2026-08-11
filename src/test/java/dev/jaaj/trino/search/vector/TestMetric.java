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
package dev.jaaj.trino.search.vector;

import dev.jaaj.trino.search.vector.quantize.BinaryCodes;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.ByteArrayBlock;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestMetric
{
    private static Block doubles(double... values)
    {
        BlockBuilder builder = DOUBLE.createBlockBuilder(null, values.length);
        for (double value : values) {
            DOUBLE.writeDouble(builder, value);
        }
        return builder.build();
    }

    @Test
    public void testNamesAreResolved()
    {
        assertThat(Metric.fromName("euclidean")).isEqualTo(Metric.EUCLIDEAN);
        assertThat(Metric.fromName("euclidean_squared")).isEqualTo(Metric.EUCLIDEAN_SQUARED);
        assertThat(Metric.fromName("cosine")).isEqualTo(Metric.COSINE);
        assertThat(Metric.fromName("dot_product")).isEqualTo(Metric.DOT_PRODUCT);
        assertThat(Metric.fromName("manhattan")).isEqualTo(Metric.MANHATTAN);
    }

    @Test
    public void testNameResolutionIsCaseInsensitive()
    {
        assertThat(Metric.fromName("EUCLIDEAN")).isEqualTo(Metric.EUCLIDEAN);
        assertThat(Metric.fromName("Dot_Product")).isEqualTo(Metric.DOT_PRODUCT);
    }

    @Test
    public void testUnknownNameListsTheValidOnes()
    {
        assertThatThrownBy(() -> Metric.fromName("hamming"))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("hamming")
                .hasMessageContaining("euclidean")
                .hasMessageContaining("manhattan");
    }

    @Test
    public void testOnlyDotProductRanksHigherAsCloser()
    {
        assertThat(Metric.DOT_PRODUCT.higherIsCloser()).isTrue();
        assertThat(Metric.EUCLIDEAN.higherIsCloser()).isFalse();
        assertThat(Metric.EUCLIDEAN_SQUARED.higherIsCloser()).isFalse();
        assertThat(Metric.COSINE.higherIsCloser()).isFalse();
        assertThat(Metric.MANHATTAN.higherIsCloser()).isFalse();
    }

    @Test
    public void testComputeMatchesTheCore()
    {
        Block first = doubles(0.0, 0.0);
        Block second = doubles(3.0, 4.0);
        assertThat(Metric.EUCLIDEAN.compute(first, second, DOUBLE_READER)).isCloseTo(5.0, within(1e-12));
        assertThat(Metric.EUCLIDEAN_SQUARED.compute(first, second, DOUBLE_READER)).isCloseTo(25.0, within(1e-12));
        assertThat(Metric.MANHATTAN.compute(first, second, DOUBLE_READER)).isCloseTo(7.0, within(1e-12));
        assertThat(Metric.DOT_PRODUCT.compute(doubles(1.0, 2.0), doubles(3.0, 4.0), DOUBLE_READER)).isCloseTo(11.0, within(1e-12));
    }

    /**
     * Long enough to reach a checkpoint: a bounded computation only consults its limit after a
     * run of components, so a two-component vector never gives up whatever the limit says and
     * cannot tell a correct implementation from a wrong one.
     */
    private static Block spike(double first, int length)
    {
        double[] values = new double[length];
        values[0] = first;
        return doubles(values);
    }

    /**
     * The accumulation is squared while the limit is a distance, so the two live in different
     * spaces and neither can be compared against the other as given. A candidate at distance 3
     * beats a limit of 4; its squared accumulation reaches 9, which does not. An implementation
     * comparing the partial sum against the raw limit abandons this candidate and loses a
     * neighbour that belongs in the result.
     */
    @Test
    public void testEuclideanGivesUpInSquaredSpace()
    {
        assertThat(Metric.EUCLIDEAN.computeBounded(spike(0.0, 512), spike(3.0, 512), DOUBLE_READER, 4.0))
                .isCloseTo(3.0, within(1e-12));
    }

    @Test
    public void testEuclideanGivesUpOnACandidateThatCannotWin()
    {
        assertThat(Metric.EUCLIDEAN.computeBounded(spike(0.0, 512), spike(30.0, 512), DOUBLE_READER, 4.0))
                .isGreaterThan(4.0);
    }

    /**
     * Dot product ranks higher as closer, so its terms are signed and a partial sum says nothing
     * about the total. Its bounded form has to compute the whole thing.
     */
    @Test
    public void testASignedMetricIgnoresTheLimit()
    {
        Block first = doubles(-100.0, 1.0);
        Block second = doubles(1.0, 1.0);

        assertThat(Metric.DOT_PRODUCT.computeBounded(first, second, DOUBLE_READER, 0.0))
                .isEqualTo(Metric.DOT_PRODUCT.compute(first, second, DOUBLE_READER));
    }

    @Test
    public void testCosineMetricIsADistance()
    {
        // identical vectors are at distance 0, not similarity 1
        assertThat(Metric.COSINE.compute(doubles(1.0, 2.0), doubles(1.0, 2.0), DOUBLE_READER))
                .isCloseTo(0.0, within(1e-12));
    }

    private static Block quantizedCodes(int... values)
    {
        byte[] bytes = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            bytes[i] = (byte) values[i];
        }
        return new ByteArrayBlock(bytes.length, Optional.empty(), bytes);
    }

    @Test
    public void testEveryMetricComputesOnQuantisedCodes()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        Block origin = quantizedCodes(0, 0);
        Block threeFour = quantizedCodes(3, 4);

        assertThat(Metric.EUCLIDEAN.computeQuantizedBounded(origin, threeFour, bounds, Double.POSITIVE_INFINITY))
                .isEqualTo(5.0);
        assertThat(Metric.EUCLIDEAN_SQUARED.computeQuantizedBounded(origin, threeFour, bounds, Double.POSITIVE_INFINITY))
                .isEqualTo(25.0);
        assertThat(Metric.MANHATTAN.computeQuantizedBounded(origin, threeFour, bounds, Double.POSITIVE_INFINITY))
                .isEqualTo(7.0);
        assertThat(Metric.DOT_PRODUCT.computeQuantizedBounded(quantizedCodes(1, 2), threeFour, bounds, Double.POSITIVE_INFINITY))
                .isEqualTo(11.0);
        assertThat(Metric.COSINE.computeQuantizedBounded(threeFour, threeFour, bounds, Double.POSITIVE_INFINITY))
                .isEqualTo(0.0);
    }

    /**
     * The bounded form must never claim a candidate is under the limit when it is not. Metrics
     * whose terms are signed cannot settle that early and are expected to ignore the limit.
     */
    @Test
    public void testQuantisedEuclideanRespectsTheLimit()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, new double[] {1, 1});
        assertThat(Metric.EUCLIDEAN_SQUARED.computeQuantizedBounded(
                quantizedCodes(0, 0), quantizedCodes(3, 4), bounds, 1.0))
                .isGreaterThan(1.0);
    }

    @Test
    public void testEveryMetricComputesOnBinaryCodes()
    {
        Slice all = BinaryCodes.pack(4, _ -> true);
        Slice none = BinaryCodes.pack(4, _ -> false);

        assertThat(Metric.EUCLIDEAN.computeBinary(all, none)).isEqualTo(4.0);
        assertThat(Metric.EUCLIDEAN_SQUARED.computeBinary(all, none)).isEqualTo(16.0);
        assertThat(Metric.MANHATTAN.computeBinary(all, none)).isEqualTo(8.0);
        assertThat(Metric.DOT_PRODUCT.computeBinary(all, none)).isEqualTo(-4.0);
        assertThat(Metric.COSINE.computeBinary(all, none)).isEqualTo(2.0);
    }
}
