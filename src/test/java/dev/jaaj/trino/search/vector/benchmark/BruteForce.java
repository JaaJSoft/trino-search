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

import java.util.Arrays;
import java.util.Locale;

/**
 * The oracle the recall harness measures against.
 * <p>
 * It re-implements the metrics on plain arrays rather than calling {@code Metric} or
 * {@code VectorMath} on purpose: an oracle that runs the code under test cannot detect that code
 * being wrong. {@code TestBruteForce} is what keeps the two definitions in agreement.
 */
public final class BruteForce
{
    private BruteForce() {}

    public enum Distance
    {
        EUCLIDEAN(false) {
            @Override
            public double between(double[] first, double[] second)
            {
                return Math.sqrt(EUCLIDEAN_SQUARED.between(first, second));
            }
        },
        EUCLIDEAN_SQUARED(false) {
            @Override
            public double between(double[] first, double[] second)
            {
                double sum = 0;
                for (int i = 0; i < first.length; i++) {
                    double difference = first[i] - second[i];
                    sum += difference * difference;
                }
                return sum;
            }
        },
        COSINE(false) {
            @Override
            public double between(double[] first, double[] second)
            {
                double dot = 0;
                double firstMagnitude = 0;
                double secondMagnitude = 0;
                for (int i = 0; i < first.length; i++) {
                    dot += first[i] * second[i];
                    firstMagnitude += first[i] * first[i];
                    secondMagnitude += second[i] * second[i];
                }
                return 1.0 - dot / Math.sqrt(firstMagnitude * secondMagnitude);
            }
        },
        DOT_PRODUCT(true) {
            @Override
            public double between(double[] first, double[] second)
            {
                double dot = 0;
                for (int i = 0; i < first.length; i++) {
                    dot += first[i] * second[i];
                }
                return dot;
            }
        },
        MANHATTAN(false) {
            @Override
            public double between(double[] first, double[] second)
            {
                double sum = 0;
                for (int i = 0; i < first.length; i++) {
                    sum += Math.abs(first[i] - second[i]);
                }
                return sum;
            }
        };

        private final boolean higherIsCloser;

        Distance(boolean higherIsCloser)
        {
            this.higherIsCloser = higherIsCloser;
        }

        public abstract double between(double[] first, double[] second);

        public boolean higherIsCloser()
        {
            return higherIsCloser;
        }

        public String sqlName()
        {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /**
     * Every distance from the query to the base set, sorted best first: ascending for a distance,
     * descending for a similarity.
     */
    public static double[] sortedDistances(double[] query, double[][] base, Distance distance)
    {
        double[] distances = new double[base.length];
        for (int i = 0; i < base.length; i++) {
            distances[i] = distance.between(query, base[i]);
        }
        Arrays.sort(distances);
        if (distance.higherIsCloser()) {
            for (int i = 0, j = distances.length - 1; i < j; i++, j--) {
                double swap = distances[i];
                distances[i] = distances[j];
                distances[j] = swap;
            }
        }
        return distances;
    }
}
