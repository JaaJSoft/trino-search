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

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

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

    @Test
    public void testCosineMetricIsADistance()
    {
        // identical vectors are at distance 0, not similarity 1
        assertThat(Metric.COSINE.compute(doubles(1.0, 2.0), doubles(1.0, 2.0), DOUBLE_READER))
                .isCloseTo(0.0, within(1e-12));
    }
}
