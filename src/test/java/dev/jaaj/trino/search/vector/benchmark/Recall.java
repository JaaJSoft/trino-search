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

/**
 * Recall at k, the fraction of the true k nearest neighbours an implementation actually returned.
 */
public final class Recall
{
    /**
     * The oracle accumulates its sums in a different order than the production kernel, so two
     * genuinely identical neighbours can differ in the last bits. Anything looser than this would
     * start counting real misses as hits.
     */
    private static final double RELATIVE_TOLERANCE = 1e-9;

    private Recall() {}

    /**
     * Correctness is decided on distances rather than on keys. When several base vectors sit at
     * exactly the k-th best distance, any of them is a correct k-th neighbour, and requiring the
     * returned keys to equal the oracle's would penalise a perfectly good answer. Ties are rare in
     * continuous synthetic data and common in quantized or duplicated data, and the resulting
     * under-reporting is silent.
     *
     * @param sortedTrueDistances every distance from the query to the base set, sorted best first
     * @param higherIsCloser true for a similarity such as dot product, false for a distance
     */
    public static double at(int k, double[] returnedDistances, double[] sortedTrueDistances, boolean higherIsCloser)
    {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be greater than zero, got " + k);
        }
        if (sortedTrueDistances.length < k) {
            throw new IllegalArgumentException(
                    "the oracle holds %s distances, which is fewer than k = %s"
                            .formatted(sortedTrueDistances.length, k));
        }

        double threshold = sortedTrueDistances[k - 1];
        double tolerance = RELATIVE_TOLERANCE * Math.max(1.0, Math.abs(threshold));
        int hits = 0;
        for (double distance : returnedDistances) {
            boolean atLeastAsGood = higherIsCloser
                    ? distance >= threshold - tolerance
                    : distance <= threshold + tolerance;
            if (atLeastAsGood) {
                hits++;
            }
        }
        return Math.min(hits, k) / (double) k;
    }
}
