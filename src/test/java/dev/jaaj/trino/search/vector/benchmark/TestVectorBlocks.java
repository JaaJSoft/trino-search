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
import io.trino.spi.block.Block;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
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
}
