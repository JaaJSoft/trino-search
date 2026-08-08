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
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestBruteForce
{
    /**
     * The oracle deliberately re-implements the metrics instead of calling Metric, so the only
     * thing keeping the two definitions aligned is this test. Cosine in particular is a distance
     * on the production side (1 - similarity) while dot product is a similarity.
     */
    @Test
    public void testEveryDistanceMatchesTheProductionMetric()
    {
        double[] first = {0.3, -1.2, 4.0, 0.5};
        double[] second = {1.1, 0.7, -2.0, 3.5};
        for (BruteForce.Distance distance : BruteForce.Distance.values()) {
            assertThat(distance.between(first, second))
                    .as(distance.sqlName())
                    .isCloseTo(
                            Metric.fromName(distance.sqlName()).compute(
                                    VectorBlocks.doubleVector(first),
                                    VectorBlocks.doubleVector(second),
                                    DOUBLE_READER),
                            within(1e-12));
        }
    }

    @Test
    public void testHigherIsCloserMatchesTheProductionMetric()
    {
        for (BruteForce.Distance distance : BruteForce.Distance.values()) {
            assertThat(distance.higherIsCloser())
                    .as(distance.sqlName())
                    .isEqualTo(Metric.fromName(distance.sqlName()).higherIsCloser());
        }
    }

    @Test
    public void testSortedDistancesAreOrderedBestFirst()
    {
        double[][] base = {{10.0}, {1.0}, {5.0}};
        double[] distances = BruteForce.sortedDistances(new double[] {0.0}, base, BruteForce.Distance.EUCLIDEAN);
        assertThat(distances).containsExactly(1.0, 5.0, 10.0);
    }

    @Test
    public void testSortedDistancesPutTheLargestFirstForDotProduct()
    {
        double[][] base = {{1.0}, {3.0}, {2.0}};
        double[] distances = BruteForce.sortedDistances(new double[] {1.0}, base, BruteForce.Distance.DOT_PRODUCT);
        assertThat(distances).containsExactly(3.0, 2.0, 1.0);
    }
}
