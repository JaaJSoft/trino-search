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
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.airlift.slice.SizeOf.sizeOf;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestCovariance
{
    private static Block vector(double... values)
    {
        BlockBuilder builder = DOUBLE.createFixedSizeBlockBuilder(values.length);
        for (double value : values) {
            DOUBLE.writeDouble(builder, value);
        }
        return builder.build();
    }

    private static Covariance of(double[]... rows)
    {
        Covariance covariance = new Covariance(rows[0].length);
        for (double[] row : rows) {
            covariance.add(vector(row), DOUBLE_READER);
        }
        return covariance;
    }

    /**
     * Deviations from the mean (3, 4) are (-2, -2), (0, 2) and (2, 0), so the co-moments are
     * 4 + 0 + 4, 4 + 0 + 0 and 4 + 4 + 0.
     */
    @Test
    public void testHandComputedCoMoments()
    {
        Covariance covariance = of(new double[] {1, 2}, new double[] {3, 6}, new double[] {5, 4});

        assertThat(covariance.count()).isEqualTo(3);
        assertThat(covariance.mean()).containsExactly(new double[] {3, 4}, within(1e-15));
        assertThat(covariance.comoments()).containsExactly(new double[] {8, 4, 8}, within(1e-14));
        assertThat(covariance.matrix()).isDeepEqualTo(new double[][] {{8.0 / 3, 4.0 / 3}, {4.0 / 3, 8.0 / 3}});
    }

    @Test
    public void testPackedLayoutIsTheUpperTriangleByRows()
    {
        Covariance covariance = of(new double[] {0, 0, 0}, new double[] {1, 2, 3});

        // The deviations are plus and minus (0.5, 1, 1.5), so entry (i, j) is x_i * x_j / 2 for
        // x = (1, 2, 3), listed (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2).
        assertThat(covariance.comoments()).containsExactly(
                new double[] {0.5, 1, 1.5, 2, 3, 4.5},
                within(1e-15));
    }

    /**
     * Merging is what the engine does with the states of two splits, and it has to give what one
     * pass over every row would, whichever way the rows were divided.
     */
    @Test
    public void testMergeAgreesWithASinglePass()
    {
        Random random = new Random(5);
        double[][] rows = new double[200][7];
        for (double[] row : rows) {
            for (int i = 0; i < row.length; i++) {
                row[i] = random.nextGaussian() * (i + 1) + i;
            }
        }
        Covariance single = of(rows);

        for (int split : new int[] {1, 67, 199}) {
            Covariance left = new Covariance(7);
            Covariance right = new Covariance(7);
            for (int r = 0; r < rows.length; r++) {
                (r < split ? left : right).add(vector(rows[r]), DOUBLE_READER);
            }
            left.merge(right);

            assertThat(left.count()).isEqualTo(single.count());
            assertThat(left.mean()).containsExactly(single.mean(), within(1e-12));
            assertThat(left.comoments()).containsExactly(single.comoments(), within(1e-9));
        }
    }

    @Test
    public void testMergeIntoAnEmptyCovarianceTakesTheOther()
    {
        Covariance empty = new Covariance(2);
        Covariance other = of(new double[] {1, 2}, new double[] {3, 6}, new double[] {5, 4});

        empty.merge(other);

        assertThat(empty.count()).isEqualTo(3);
        assertThat(empty.mean()).containsExactly(other.mean());
        assertThat(empty.comoments()).containsExactly(other.comoments());
    }

    @Test
    public void testMergingAnEmptyCovarianceChangesNothing()
    {
        Covariance covariance = of(new double[] {1, 2}, new double[] {3, 6});
        double[] comoments = covariance.comoments().clone();

        covariance.merge(new Covariance(2));

        assertThat(covariance.count()).isEqualTo(2);
        assertThat(covariance.comoments()).containsExactly(comoments);
    }

    /**
     * Far from the origin, the textbook sum of squares minus the squared mean cancels away every
     * significant digit: at 1e9 the squares are 1e18 and a double resolves them to 128. Updating
     * the deviations from a running mean is what keeps the variance of 1, 2, 3, 4, which is 1.25.
     */
    @Test
    public void testAccurateFarFromTheOrigin()
    {
        double offset = 1e9;
        Covariance single = of(new double[] {offset + 1}, new double[] {offset + 2}, new double[] {offset + 3}, new double[] {offset + 4});
        assertThat(single.matrix()[0][0]).isCloseTo(1.25, within(1e-6));

        Covariance left = of(new double[] {offset + 1}, new double[] {offset + 2});
        left.merge(of(new double[] {offset + 3}, new double[] {offset + 4}));
        assertThat(left.matrix()[0][0]).isCloseTo(1.25, within(1e-6));
    }

    @Test
    public void testVectorOfAnotherDimensionIsRejected()
    {
        Covariance covariance = of(new double[] {1, 2});

        assertThatThrownBy(() -> covariance.add(vector(1, 2, 3), DOUBLE_READER))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The vectors of vector_pca_agg must have the same length, found 2 and 3");
        assertThatThrownBy(() -> covariance.merge(of(new double[] {1})))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The vectors of vector_pca_agg must have the same length, found 2 and 1");
    }

    /**
     * The state grows with the square of the dimension, and so does the decomposition with its
     * cube, so a dimension past the limit fails before allocating anything rather than after.
     */
    @Test
    public void testDimensionAboveTheLimitIsRejected()
    {
        assertThatThrownBy(() -> new Covariance(Covariance.MAX_DIMENSION + 1))
                .isInstanceOf(TrinoException.class)
                .hasMessage("vector_pca_agg supports vectors of up to 4096 dimensions, found 4097");
    }

    @Test
    public void testRetainedSizeCountsItsArrays()
    {
        Covariance covariance = new Covariance(768);

        assertThat(covariance.getRetainedSizeInBytes())
                .isGreaterThan(sizeOf(new double[768 * 769 / 2]) + 2 * sizeOf(new double[768]));
    }
}
