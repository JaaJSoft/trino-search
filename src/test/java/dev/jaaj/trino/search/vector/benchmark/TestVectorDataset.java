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

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.CLUSTERED;
import static dev.jaaj.trino.search.vector.benchmark.VectorDataset.Regime.UNIFORM;
import static org.assertj.core.api.Assertions.assertThat;

public class TestVectorDataset
{
    @Test
    public void testShape()
    {
        VectorDataset dataset = VectorDataset.generate(CLUSTERED, 200, 10, 16, 42L);
        assertThat(dataset.dimension()).isEqualTo(16);
        assertThat(dataset.base()).hasNumberOfRows(200);
        assertThat(dataset.queries()).hasNumberOfRows(10);
        assertThat(dataset.base()[0]).hasSize(16);
        assertThat(dataset.queries()[0]).hasSize(16);
    }

    @Test
    public void testSameSeedProducesIdenticalVectors()
    {
        VectorDataset first = VectorDataset.generate(CLUSTERED, 50, 5, 8, 7L);
        VectorDataset second = VectorDataset.generate(CLUSTERED, 50, 5, 8, 7L);
        assertThat(first.base()).isDeepEqualTo(second.base());
        assertThat(first.queries()).isDeepEqualTo(second.queries());
    }

    @Test
    public void testDifferentSeedProducesDifferentVectors()
    {
        VectorDataset first = VectorDataset.generate(CLUSTERED, 50, 5, 8, 7L);
        VectorDataset second = VectorDataset.generate(CLUSTERED, 50, 5, 8, 8L);
        assertThat(Arrays.deepEquals(first.base(), second.base())).isFalse();
    }

    /**
     * Clustering is asserted without exposing cluster labels: in a clustered dataset the nearest
     * other vector is much closer than the average vector, while in a uniform one the two are
     * comparable. The ratio is what separates the regimes, not either quantity alone.
     */
    @Test
    public void testClusteredRegimeActuallyClusters()
    {
        assertThat(nearestNeighbourRatio(VectorDataset.generate(CLUSTERED, 200, 1, 16, 3L)))
                .isLessThan(nearestNeighbourRatio(VectorDataset.generate(UNIFORM, 200, 1, 16, 3L)) / 2);
    }

    private static double nearestNeighbourRatio(VectorDataset dataset)
    {
        double[][] vectors = dataset.base();
        double nearestSum = 0;
        double pairSum = 0;
        long pairCount = 0;
        for (int i = 0; i < vectors.length; i++) {
            double nearest = Double.MAX_VALUE;
            for (int j = 0; j < vectors.length; j++) {
                if (i == j) {
                    continue;
                }
                double distance = euclidean(vectors[i], vectors[j]);
                nearest = Math.min(nearest, distance);
                pairSum += distance;
                pairCount++;
            }
            nearestSum += nearest;
        }
        return (nearestSum / vectors.length) / (pairSum / pairCount);
    }

    private static double euclidean(double[] first, double[] second)
    {
        double sum = 0;
        for (int i = 0; i < first.length; i++) {
            double difference = first[i] - second[i];
            sum += difference * difference;
        }
        return Math.sqrt(sum);
    }
}
