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

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * The leading principal directions of a set of vectors, unit length and mutually orthogonal, with
 * the share of the total variance each one carries.
 */
record PrincipalComponents(double[][] directions, double[] explainedVarianceRatio)
{
    /**
     * Fits {@code components} directions, or as many as there are dimensions if that is fewer.
     * <p>
     * Each direction's sign is fixed by making its component of largest magnitude positive: an
     * eigenvector is only defined up to its sign, and a refit on the same data must not flip
     * every value of a projection column.
     * <p>
     * The ratios divide by the trace, the total variance. With no variance at all, as for a
     * single row, every direction is as good as any other and every ratio is 0 / 0, NaN.
     */
    static PrincipalComponents fit(Covariance covariance, long components)
    {
        double[][] matrix = covariance.matrix();
        double trace = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (double value : matrix[i]) {
                if (!Double.isFinite(value)) {
                    throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The covariance of the vectors in vector_pca_agg is not finite");
                }
            }
            trace += matrix[i][i];
        }

        SymmetricEigen eigen = SymmetricEigen.decompose(matrix);
        int count = (int) Math.min(components, matrix.length);
        double[][] directions = new double[count][];
        double[] ratios = new double[count];
        for (int k = 0; k < count; k++) {
            directions[k] = withLargestComponentPositive(eigen.vectors()[k]);
            // A covariance has no negative eigenvalue; rounding can still leave one a hair below
            // zero, which is no share of anything.
            ratios[k] = Math.max(0, eigen.values()[k]) / trace;
        }
        return new PrincipalComponents(directions, ratios);
    }

    private static double[] withLargestComponentPositive(double[] direction)
    {
        int largest = 0;
        for (int i = 1; i < direction.length; i++) {
            if (Math.abs(direction[i]) > Math.abs(direction[largest])) {
                largest = i;
            }
        }
        if (direction[largest] < 0) {
            for (int i = 0; i < direction.length; i++) {
                direction[i] = -direction[i];
            }
        }
        return direction;
    }
}
