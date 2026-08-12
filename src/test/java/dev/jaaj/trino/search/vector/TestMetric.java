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
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, 1.0);
        Block origin = quantizedCodes(0, 0);
        Block threeFour = quantizedCodes(3, 4);

        assertThat(Metric.EUCLIDEAN.computeQuantized(origin, threeFour, bounds)).isEqualTo(5.0);
        assertThat(Metric.EUCLIDEAN_SQUARED.computeQuantized(origin, threeFour, bounds)).isEqualTo(25.0);
        assertThat(Metric.MANHATTAN.computeQuantized(origin, threeFour, bounds)).isEqualTo(7.0);
        assertThat(Metric.DOT_PRODUCT.computeQuantized(quantizedCodes(1, 2), threeFour, bounds)).isEqualTo(11.0);
        assertThat(Metric.COSINE.computeQuantized(threeFour, threeFour, bounds)).isEqualTo(0.0);
    }

    /**
     * A metric whose terms are signed cannot give up part way through, so its bounded form must
     * return the same value as its unbounded one however tight the limit is.
     */
    @Test
    public void testASignedMetricIgnoresTheLimitOnQuantisedCodes()
    {
        QuantizationBounds bounds = QuantizationBounds.forTesting(new double[] {0, 0}, 1.0);
        Block first = quantizedCodes(-100, 1);
        Block second = quantizedCodes(1, 1);

        assertThat(Metric.DOT_PRODUCT.computeQuantizedBounded(first, second, bounds, 0.0))
                .isEqualTo(Metric.DOT_PRODUCT.computeQuantized(first, second, bounds));
    }

    /**
     * The number of components a quantised vector needs before the bounded kernel's own loop runs
     * at all. The kernel processes one vector register at a time and consults the limit only
     * between registers, so anything shorter than the widest preferred species falls straight
     * through to the scalar tail, which never reads the limit: at 8 components the check that a
     * candidate is abandoned passes whether the early return exists or not.
     */
    private static final int VECTORISED_LENGTH = 512;

    private static QuantizationBounds unitBounds(int length)
    {
        return QuantizationBounds.forTesting(new double[length], 1.0);
    }

    /**
     * The whole distance in the first two components, so the first checkpoint already settles
     * whether the candidate can win and abandonment happens part way rather than at the last
     * component.
     */
    private static Block codeSpike(int first, int second, int length)
    {
        byte[] codes = new byte[length];
        codes[0] = (byte) first;
        codes[1] = (byte) second;
        return new ByteArrayBlock(length, Optional.empty(), codes);
    }

    /**
     * Infinity rather than "some value above the limit": the exact answer, 25, is above a limit of
     * 1 as well, so a comparison against the limit alone cannot tell a candidate that was abandoned
     * from one that was computed in full.
     */
    @Test
    public void testQuantisedEuclideanSquaredAbandonsPastATightLimitButComputesExactlyUnderAGenerousOne()
    {
        QuantizationBounds bounds = unitBounds(VECTORISED_LENGTH);
        Block origin = codeSpike(0, 0, VECTORISED_LENGTH);
        Block threeFour = codeSpike(3, 4, VECTORISED_LENGTH);

        assertThat(Metric.EUCLIDEAN_SQUARED.computeQuantizedBounded(origin, threeFour, bounds, 1.0))
                .isInfinite();
        assertThat(Metric.EUCLIDEAN_SQUARED.computeQuantizedBounded(origin, threeFour, bounds, 1000.0))
                .isEqualTo(25.0);
    }

    /**
     * The limit passed in is 10, strictly between the true distance of 5 and its square of 25:
     * only squaring it before comparing keeps this candidate. A caller that forgot to square would
     * compare the squared sum of 25 against the raw 10, abandon early, and return infinity instead
     * of 5.0.
     */
    @Test
    public void testQuantisedEuclideanSquaresTheLimitBeforeComparingToTheSquaredSum()
    {
        QuantizationBounds bounds = unitBounds(VECTORISED_LENGTH);
        Block origin = codeSpike(0, 0, VECTORISED_LENGTH);
        Block threeFour = codeSpike(3, 4, VECTORISED_LENGTH);

        assertThat(Metric.EUCLIDEAN.computeQuantizedBounded(origin, threeFour, bounds, 10.0))
                .isEqualTo(5.0);
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
