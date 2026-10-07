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

import io.trino.spi.TrinoException;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

import static io.trino.spi.StandardErrorCode.GENERIC_INTERNAL_ERROR;

/**
 * The eigenvalues and eigenvectors of a real symmetric matrix, largest eigenvalue first, each
 * eigenvector a row of {@link #vectors()}.
 * <p>
 * Householder reduction to tridiagonal form followed by the implicit QL algorithm: the EISPACK
 * routines {@code tred2} and {@code tql2}, in the form the public domain JAMA library gave them.
 * Both are O(n^3) and need no convergence tuning, which is why they are used here rather than an
 * iteration aimed at the leading vectors only: the matrix is a covariance decomposed once per fit,
 * and a solver whose accuracy depends on the gaps between eigenvalues would make the fit quietly
 * worse on exactly the flat spectra embeddings tend to have.
 */
final class SymmetricEigen
{
    /**
     * EISPACK allows 30 QL iterations per eigenvalue. Convergence is cubic, so a matrix that needs
     * more holds something other than finite numbers.
     */
    private static final int MAX_ITERATIONS = 30;
    private static final double EPSILON = Math.ulp(1.0);

    private final double[] values;
    private final double[][] vectors;

    private SymmetricEigen(double[] values, double[][] vectors)
    {
        this.values = values;
        this.vectors = vectors;
    }

    /**
     * Decomposes {@code matrix}, which must be square and symmetric and is overwritten.
     */
    static SymmetricEigen decompose(double[][] matrix)
    {
        int n = matrix.length;
        if (n == 0) {
            return new SymmetricEigen(new double[0], new double[0][]);
        }
        double[] diagonal = new double[n];
        double[] offDiagonal = new double[n];
        tridiagonalize(matrix, diagonal, offDiagonal);
        diagonalize(diagonal, offDiagonal, matrix);

        Integer[] order = IntStream.range(0, n).boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> diagonal[i]).reversed())
                .toArray(Integer[]::new);
        double[] values = new double[n];
        double[][] vectors = new double[n][];
        for (int k = 0; k < n; k++) {
            values[k] = diagonal[order[k]];
            vectors[k] = matrix[order[k]];
        }
        return new SymmetricEigen(values, vectors);
    }

    double[] values()
    {
        return values;
    }

    double[][] vectors()
    {
        return vectors;
    }

    /**
     * {@code tred2}, with every {@code V[a][b]} of the original written {@code w[b][a]}: the
     * routine's inner loops walk down columns of {@code V}, which on an n by n row-major matrix
     * misses the cache on every step once n reaches the hundreds. A symmetric input is its own
     * transpose, so running on {@code w = V^T} needs no copy and turns those walks into
     * contiguous ones. On return row k of {@code w} is the k-th basis vector in which the matrix
     * is tridiagonal, with that diagonal in {@code d} and the subdiagonal in {@code e[1..n-1]}.
     */
    private static void tridiagonalize(double[][] w, double[] d, double[] e)
    {
        int n = d.length;
        for (int j = 0; j < n; j++) {
            d[j] = w[j][n - 1];
        }

        for (int i = n - 1; i > 0; i--) {
            // Scale to avoid under/overflow.
            double scale = 0.0;
            double h = 0.0;
            for (int k = 0; k < i; k++) {
                scale += Math.abs(d[k]);
            }
            if (scale == 0.0) {
                e[i] = d[i - 1];
                for (int j = 0; j < i; j++) {
                    d[j] = w[j][i - 1];
                    w[j][i] = 0.0;
                    w[i][j] = 0.0;
                }
            }
            else {
                // Generate the Householder vector.
                for (int k = 0; k < i; k++) {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                double f = d[i - 1];
                double g = Math.sqrt(h);
                if (f > 0) {
                    g = -g;
                }
                e[i] = scale * g;
                h -= f * g;
                d[i - 1] = f - g;
                for (int j = 0; j < i; j++) {
                    e[j] = 0.0;
                }

                // Apply the similarity transformation to the remaining columns.
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    w[i][j] = f;
                    g = e[j] + w[j][j] * f;
                    for (int k = j + 1; k <= i - 1; k++) {
                        g += w[j][k] * d[k];
                        e[k] += w[j][k] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for (int j = 0; j < i; j++) {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                double hh = f / (h + h);
                for (int j = 0; j < i; j++) {
                    e[j] -= hh * d[j];
                }
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    g = e[j];
                    for (int k = j; k <= i - 1; k++) {
                        w[j][k] -= f * e[k] + g * d[k];
                    }
                    d[j] = w[j][i - 1];
                    w[j][i] = 0.0;
                }
            }
            d[i] = h;
        }

        // Accumulate the transformations.
        for (int i = 0; i < n - 1; i++) {
            w[i][n - 1] = w[i][i];
            w[i][i] = 1.0;
            double h = d[i + 1];
            if (h != 0.0) {
                for (int k = 0; k <= i; k++) {
                    d[k] = w[i + 1][k] / h;
                }
                for (int j = 0; j <= i; j++) {
                    double g = 0.0;
                    for (int k = 0; k <= i; k++) {
                        g += w[i + 1][k] * w[j][k];
                    }
                    for (int k = 0; k <= i; k++) {
                        w[j][k] -= g * d[k];
                    }
                }
            }
            for (int k = 0; k <= i; k++) {
                w[i + 1][k] = 0.0;
            }
        }
        for (int j = 0; j < n; j++) {
            d[j] = w[j][n - 1];
            w[j][n - 1] = 0.0;
        }
        w[n - 1][n - 1] = 1.0;
        e[0] = 0.0;
    }

    /**
     * {@code tql2}, on the rows {@link #tridiagonalize} left: each eigenvector stays in a row of
     * its own, so a rotation walks two contiguous arrays rather than two columns. On return
     * {@code d} holds the eigenvalues, unsorted, and row k of {@code rows} the eigenvector of
     * {@code d[k]}.
     */
    private static void diagonalize(double[] d, double[] e, double[][] rows)
    {
        int n = d.length;
        for (int i = 1; i < n; i++) {
            e[i - 1] = e[i];
        }
        e[n - 1] = 0.0;

        double f = 0.0;
        double tst1 = 0.0;
        for (int l = 0; l < n; l++) {
            // Find a small subdiagonal element.
            tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
            int m = l;
            while (m < n) {
                if (Math.abs(e[m]) <= EPSILON * tst1) {
                    break;
                }
                m++;
            }

            // If m == l, d[l] is already an eigenvalue; otherwise iterate.
            if (m > l) {
                int iterations = 0;
                do {
                    iterations++;
                    if (iterations > MAX_ITERATIONS) {
                        throw new TrinoException(GENERIC_INTERNAL_ERROR, "The eigendecomposition did not converge");
                    }

                    // Compute the implicit shift.
                    double g = d[l];
                    double p = (d[l + 1] - g) / (2.0 * e[l]);
                    double r = Math.hypot(p, 1.0);
                    if (p < 0) {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    double dl1 = d[l + 1];
                    double h = g - d[l];
                    for (int i = l + 2; i < n; i++) {
                        d[i] -= h;
                    }
                    f += h;

                    // Implicit QL transformation.
                    p = d[m];
                    double c = 1.0;
                    double c2 = c;
                    double c3 = c;
                    double el1 = e[l + 1];
                    double s = 0.0;
                    double s2 = 0.0;
                    for (int i = m - 1; i >= l; i--) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = Math.hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        // Accumulate the transformation.
                        double[] lower = rows[i];
                        double[] upper = rows[i + 1];
                        for (int k = 0; k < n; k++) {
                            h = upper[k];
                            upper[k] = s * lower[k] + c * h;
                            lower[k] = c * lower[k] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                }
                while (Math.abs(e[l]) > EPSILON * tst1);
            }
            d[l] += f;
            e[l] = 0.0;
        }
    }

    @Override
    public String toString()
    {
        return "SymmetricEigen" + Arrays.toString(values);
    }
}
