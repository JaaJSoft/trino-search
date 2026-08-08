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

import java.util.SplittableRandom;

/**
 * Deterministic synthetic vectors for the benchmarks and the recall harness.
 * <p>
 * Everything is derived from an explicit seed through {@link SplittableRandom}, so a given set of
 * parameters yields bit-identical vectors on any machine and nothing has to be committed or
 * downloaded.
 */
public record VectorDataset(double[][] base, double[][] queries, int dimension)
{
    public static final int DEFAULT_CLUSTER_COUNT = 32;
    public static final double DEFAULT_CLUSTER_SPREAD = 0.05;

    public enum Regime
    {
        /**
         * A mixture of isotropic Gaussians. Real embeddings occupy a manifold of far lower
         * intrinsic dimension than their nominal one, which is exactly what an approximate index
         * exploits, so this is the default regime.
         */
        CLUSTERED,
        /**
         * Independent uniform components. In high dimension every pairwise distance concentrates
         * around the same value, leaving an approximate index nothing to exploit. Kept as the
         * adversarial baseline.
         */
        UNIFORM,
    }

    public static VectorDataset generate(Regime regime, int baseSize, int queryCount, int dimension, long seed)
    {
        return generate(regime, baseSize, queryCount, dimension, DEFAULT_CLUSTER_COUNT, DEFAULT_CLUSTER_SPREAD, seed);
    }

    public static VectorDataset generate(
            Regime regime,
            int baseSize,
            int queryCount,
            int dimension,
            int clusterCount,
            double clusterSpread,
            long seed)
    {
        SplittableRandom random = new SplittableRandom(seed);
        double[][] centers = regime == Regime.CLUSTERED ? uniformVectors(random, clusterCount, dimension) : null;
        return new VectorDataset(
                vectors(random, baseSize, dimension, centers, clusterSpread),
                vectors(random, queryCount, dimension, centers, clusterSpread),
                dimension);
    }

    private static double[][] vectors(
            SplittableRandom random,
            int count,
            int dimension,
            double[][] centers,
            double clusterSpread)
    {
        if (centers == null) {
            return uniformVectors(random, count, dimension);
        }
        double[][] vectors = new double[count][dimension];
        for (int i = 0; i < count; i++) {
            double[] center = centers[random.nextInt(centers.length)];
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = center[j] + clusterSpread * random.nextGaussian();
            }
        }
        return vectors;
    }

    private static double[][] uniformVectors(SplittableRandom random, int count, int dimension)
    {
        double[][] vectors = new double[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextDouble();
            }
        }
        return vectors;
    }
}
