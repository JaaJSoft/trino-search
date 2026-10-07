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
package dev.jaaj.trino.search.vector.projection;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class TestSymmetricEigen
{
    @Test
    public void testDiagonalMatrixYieldsTheAxesInDecreasingOrder()
    {
        SymmetricEigen eigen = SymmetricEigen.decompose(new double[][] {
                {1, 0, 0},
                {0, 3, 0},
                {0, 0, 2},
        });

        assertThat(eigen.values()).containsExactly(new double[] {3, 2, 1}, within(1e-15));
        assertThat(Math.abs(eigen.vectors()[0][1])).isCloseTo(1, within(1e-15));
        assertThat(Math.abs(eigen.vectors()[1][2])).isCloseTo(1, within(1e-15));
        assertThat(Math.abs(eigen.vectors()[2][0])).isCloseTo(1, within(1e-15));
    }

    @Test
    public void testTwoByTwo()
    {
        SymmetricEigen eigen = SymmetricEigen.decompose(new double[][] {
                {2, 1},
                {1, 2},
        });

        assertThat(eigen.values()).containsExactly(new double[] {3, 1}, within(1e-14));
        double half = Math.sqrt(0.5);
        assertThat(Math.abs(eigen.vectors()[0][0])).isCloseTo(half, within(1e-14));
        assertThat(eigen.vectors()[0][0] * eigen.vectors()[0][1]).isCloseTo(0.5, within(1e-14));
        assertThat(eigen.vectors()[1][0] * eigen.vectors()[1][1]).isCloseTo(-0.5, within(1e-14));
    }

    /**
     * The defining properties, on matrices nobody chose: every pair satisfies A v = lambda v, the
     * vectors are orthonormal, and the values add up to the trace. Dimension 97 is odd and large
     * enough for every branch of the reduction to run many times.
     */
    @Test
    public void testRandomSymmetricMatrices()
    {
        for (long seed = 1; seed <= 5; seed++) {
            double[][] matrix = randomSymmetric(new Random(seed), 97);
            assertDecomposes(matrix);
        }
    }

    /**
     * A covariance matrix is positive semi-definite and often close to singular, which is the shape
     * the aggregation actually hands over.
     */
    @Test
    public void testLowRankCovarianceShape()
    {
        Random random = new Random(3);
        int dimension = 40;
        double[][] matrix = new double[dimension][dimension];
        for (int sample = 0; sample < 5; sample++) {
            double[] x = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                x[i] = random.nextGaussian();
            }
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    matrix[i][j] += x[i] * x[j];
                }
            }
        }

        SymmetricEigen eigen = assertDecomposes(matrix);
        for (int j = 5; j < dimension; j++) {
            assertThat(eigen.values()[j]).isCloseTo(0, within(1e-12));
        }
    }

    /**
     * Repeated eigenvalues leave the eigenvectors of that eigenvalue free to be any orthonormal
     * basis of its eigenspace, so only the defining properties can be checked.
     */
    @Test
    public void testRepeatedEigenvalues()
    {
        double[][] rotation = orthonormalBasis(new Random(9), 6);
        double[] spectrum = {5, 5, 5, 2, 2, 0};
        double[][] matrix = new double[6][6];
        for (int k = 0; k < 6; k++) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    matrix[i][j] += spectrum[k] * rotation[k][i] * rotation[k][j];
                }
            }
        }

        SymmetricEigen eigen = assertDecomposes(matrix);
        assertThat(eigen.values()).containsExactly(spectrum, within(1e-12));
    }

    @Test
    public void testZeroMatrixYieldsAnOrthonormalBasis()
    {
        SymmetricEigen eigen = assertDecomposes(new double[4][4]);
        assertThat(eigen.values()).containsOnly(0.0);
    }

    @Test
    public void testOneByOne()
    {
        SymmetricEigen eigen = SymmetricEigen.decompose(new double[][] {{-2.5}});

        assertThat(eigen.values()).containsExactly(-2.5);
        assertThat(eigen.vectors()[0]).containsExactly(1.0);
    }

    @Test
    public void testEmpty()
    {
        SymmetricEigen eigen = SymmetricEigen.decompose(new double[0][0]);

        assertThat(eigen.values()).isEmpty();
        assertThat(eigen.vectors()).isEmpty();
    }

    /**
     * Returns the decomposition after checking A v = lambda v for every pair, orthonormality, the
     * trace, and the order of the values.
     */
    private static SymmetricEigen assertDecomposes(double[][] matrix)
    {
        int n = matrix.length;
        double[][] original = new double[n][];
        double scale = 0;
        double trace = 0;
        for (int i = 0; i < n; i++) {
            original[i] = matrix[i].clone();
            trace += matrix[i][i];
            for (int j = 0; j < n; j++) {
                scale = Math.max(scale, Math.abs(matrix[i][j]));
            }
        }
        double tolerance = 1e-12 * Math.max(1, scale) * n;

        SymmetricEigen eigen = SymmetricEigen.decompose(matrix);
        double[] values = eigen.values();
        double[][] vectors = eigen.vectors();
        assertThat(values).hasSize(n);
        assertThat(vectors).hasNumberOfRows(n);

        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += values[k];
            if (k > 0) {
                assertThat(values[k]).isLessThanOrEqualTo(values[k - 1]);
            }
            for (int i = 0; i < n; i++) {
                double product = 0;
                for (int j = 0; j < n; j++) {
                    product += original[i][j] * vectors[k][j];
                }
                assertThat(product).isCloseTo(values[k] * vectors[k][i], within(tolerance));
            }
            for (int other = 0; other < n; other++) {
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += vectors[k][i] * vectors[other][i];
                }
                assertThat(dot).isCloseTo(k == other ? 1 : 0, within(1e-12));
            }
        }
        assertThat(sum).isCloseTo(trace, within(tolerance));
        return eigen;
    }

    private static double[][] randomSymmetric(Random random, int n)
    {
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double value = random.nextGaussian();
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
        return matrix;
    }

    private static double[][] orthonormalBasis(Random random, int n)
    {
        double[][] basis = new double[n][n];
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                basis[k][i] = random.nextGaussian();
            }
            for (int previous = 0; previous < k; previous++) {
                double overlap = 0;
                for (int i = 0; i < n; i++) {
                    overlap += basis[k][i] * basis[previous][i];
                }
                for (int i = 0; i < n; i++) {
                    basis[k][i] -= overlap * basis[previous][i];
                }
            }
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += basis[k][i] * basis[k][i];
            }
            norm = Math.sqrt(norm);
            for (int i = 0; i < n; i++) {
                basis[k][i] /= norm;
            }
        }
        return basis;
    }
}
