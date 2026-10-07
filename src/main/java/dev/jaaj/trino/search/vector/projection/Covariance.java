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

import dev.jaaj.trino.search.vector.VectorReader;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * The running mean and co-moments of a set of vectors, from which their covariance follows.
 * <p>
 * The co-moments are the sums of products of deviations from the mean, updated row by row from
 * the running mean (Welford) and combined across splits with a correction for the distance
 * between the two means (Chan et al). The textbook alternative, a sum of squares from which the
 * squared mean is subtracted at the end, cancels away every significant digit once the data sits
 * far from the origin relative to its spread.
 * <p>
 * The matrix is symmetric, so only its upper triangle is kept, row after row: entry (i, j) for
 * {@code j >= i} sits at {@code rowStart(i) + j - i}. At dimension 768 that is 295,296 doubles,
 * about 2.4 MB.
 */
public final class Covariance
{
    /**
     * The state grows with the square of the dimension and the decomposition at output with its
     * cube: 4096 is about 67 MB of state and a few minutes of decomposition, past which the fit
     * belongs outside the engine.
     */
    static final int MAX_DIMENSION = 4096;

    private static final long INSTANCE_SIZE = instanceSize(Covariance.class);

    private final int dimension;
    private final double[] mean;
    private final double[] comoments;
    private final double[] deviation;
    private long count;

    public Covariance(int dimension)
    {
        this(dimension, 0, new double[dimension], new double[packedLength(dimension)]);
    }

    private Covariance(int dimension, long count, double[] mean, double[] comoments)
    {
        this.dimension = dimension;
        this.count = count;
        this.mean = mean;
        this.comoments = comoments;
        this.deviation = new double[dimension];
    }

    /**
     * Takes the arrays as they are, without copying them: this is how a deserialized state is
     * rebuilt, from arrays allocated for it alone.
     */
    public static Covariance of(long count, double[] mean, double[] comoments)
    {
        if (comoments.length != packedLength(mean.length)) {
            throw new IllegalArgumentException("%s co-moments do not match dimension %s".formatted(comoments.length, mean.length));
        }
        return new Covariance(mean.length, count, mean, comoments);
    }

    private static int packedLength(int dimension)
    {
        if (dimension > MAX_DIMENSION) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "vector_pca_agg supports vectors of up to %s dimensions, found %s".formatted(MAX_DIMENSION, dimension));
        }
        return dimension * (dimension + 1) / 2;
    }

    public int dimension()
    {
        return dimension;
    }

    public long count()
    {
        return count;
    }

    @SuppressFBWarnings(value = "EI_EXPOSE_REP", justification = "Read by the serializer, which only copies it out.")
    public double[] mean()
    {
        return mean;
    }

    /**
     * The upper triangle of the co-moment matrix, row after row.
     */
    @SuppressFBWarnings(value = "EI_EXPOSE_REP", justification = "Read by the serializer, which only copies it out.")
    public double[] comoments()
    {
        return comoments;
    }

    /**
     * Adds one vector. The caller has already rejected a vector with a null component.
     */
    public void add(Block vector, VectorReader reader)
    {
        checkSameDimension(vector.getPositionCount());
        count++;
        for (int i = 0; i < dimension; i++) {
            double value = reader.read(vector, i);
            deviation[i] = value - mean[i];
            mean[i] += deviation[i] / count;
        }
        // (x - old mean)(x - new mean)^T, written as a multiple of (x - old mean)(x - old mean)^T
        // so that each update is symmetric and the upper triangle alone carries it.
        addOuterProduct(deviation, (count - 1) / (double) count);
    }

    public void merge(Covariance other)
    {
        checkSameDimension(other.dimension);
        if (other.count == 0) {
            return;
        }
        if (count == 0) {
            System.arraycopy(other.mean, 0, mean, 0, dimension);
            System.arraycopy(other.comoments, 0, comoments, 0, comoments.length);
            count = other.count;
            return;
        }
        long total = count + other.count;
        for (int i = 0; i < dimension; i++) {
            deviation[i] = other.mean[i] - mean[i];
            mean[i] += deviation[i] * other.count / total;
        }
        for (int p = 0; p < comoments.length; p++) {
            comoments[p] += other.comoments[p];
        }
        addOuterProduct(deviation, (double) count * other.count / total);
        count = total;
    }

    /**
     * The population covariance as a full symmetric matrix, freshly allocated.
     */
    double[][] matrix()
    {
        double[][] matrix = new double[dimension][dimension];
        int p = 0;
        for (int i = 0; i < dimension; i++) {
            for (int j = i; j < dimension; j++) {
                double value = comoments[p++] / count;
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
        return matrix;
    }

    public long getRetainedSizeInBytes()
    {
        return INSTANCE_SIZE + sizeOf(mean) + sizeOf(comoments) + sizeOf(deviation);
    }

    /**
     * The inner loop runs over a contiguous stretch of both arrays, which is what lets the JIT
     * vectorise it: this is the d^2 / 2 part of every row.
     */
    private void addOuterProduct(double[] vector, double weight)
    {
        int rowStart = 0;
        for (int i = 0; i < dimension; i++) {
            double scaled = weight * vector[i];
            int base = rowStart - i;
            for (int j = i; j < dimension; j++) {
                comoments[base + j] += scaled * vector[j];
            }
            rowStart += dimension - i;
        }
    }

    private void checkSameDimension(int incoming)
    {
        if (incoming != dimension) {
            throw new TrinoException(
                    INVALID_FUNCTION_ARGUMENT,
                    "The vectors of vector_pca_agg must have the same length, found %s and %s".formatted(dimension, incoming));
        }
    }
}
