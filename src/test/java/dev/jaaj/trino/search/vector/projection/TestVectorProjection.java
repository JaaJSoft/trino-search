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
import io.trino.spi.TrinoException;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.Type;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestVectorProjection
{
    private static final ArrayType DOUBLE_VECTOR = new ArrayType(DOUBLE);
    private static final ArrayType REAL_VECTOR = new ArrayType(REAL);

    private static Block vector(Double... values)
    {
        return vector(DOUBLE_VECTOR, values);
    }

    private static Block vector(ArrayType type, Double... values)
    {
        ArrayBlockBuilder builder = (ArrayBlockBuilder) type.createBlockBuilder(null, 1);
        builder.buildEntry(elements -> writeComponents(type.getElementType(), elements, values));
        return type.getObject(builder.build(), 0);
    }

    /**
     * A null entry stands for a null direction.
     */
    private static Block directions(Double[]... vectors)
    {
        return directions(DOUBLE_VECTOR, vectors);
    }

    private static Block directions(ArrayType type, Double[]... vectors)
    {
        ArrayType directionsType = new ArrayType(type);
        ArrayBlockBuilder builder = (ArrayBlockBuilder) directionsType.createBlockBuilder(null, 1);
        builder.buildEntry(directionBuilder -> {
            for (Double[] values : vectors) {
                if (values == null) {
                    directionBuilder.appendNull();
                }
                else {
                    ((ArrayBlockBuilder) directionBuilder).buildEntry(elements -> writeComponents(type.getElementType(), elements, values));
                }
            }
        });
        return directionsType.getObject(builder.build(), 0);
    }

    private static void writeComponents(Type elementType, BlockBuilder elements, Double[] values)
    {
        for (Double value : values) {
            if (value == null) {
                elements.appendNull();
            }
            else if (elementType == REAL) {
                REAL.writeFloat(elements, value.floatValue());
            }
            else {
                DOUBLE.writeDouble(elements, value);
            }
        }
    }

    private static Double[] v(Double... values)
    {
        return values;
    }

    private static List<Double> project(Block vector, Block directions)
    {
        return project(vector, directions, DOUBLE_VECTOR, DOUBLE_READER);
    }

    private static List<Double> project(Block vector, Block directions, ArrayType directionType, VectorReader reader)
    {
        Block projections = VectorProjection.project(vector, directions, directionType, reader);
        List<Double> values = new ArrayList<>();
        for (int i = 0; i < projections.getPositionCount(); i++) {
            values.add(projections.isNull(i) ? null : DOUBLE.getDouble(projections, i));
        }
        return values;
    }

    @Test
    public void testProjectsOntoEachDirectionInOrder()
    {
        assertThat(project(
                vector(1.0, 2.0, 3.0),
                directions(v(0.0, 0.0, 1.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0))))
                .containsExactly(3.0, 1.0, 2.0);
    }

    @Test
    public void testEachProjectionIsTheDotProduct()
    {
        assertThat(project(
                vector(3.0, 4.0),
                directions(v(0.6, 0.8), v(-0.8, 0.6))))
                .containsExactly(3.0 * 0.6 + 4.0 * 0.8, -3.0 * 0.8 + 4.0 * 0.6);
    }

    /**
     * The bound that makes a projection column prunable holds for unit directions only, but
     * normalising here would silently answer a question other than the one asked.
     */
    @Test
    public void testDirectionsAreUsedAsGiven()
    {
        assertThat(project(vector(1.0, 1.0), directions(v(2.0, 0.0))))
                .containsExactly(2.0);
    }

    /**
     * Element j of the result is the projection onto direction j, which is how it maps back to a
     * column, so a direction without a projection must not shift the ones after it.
     */
    @Test
    public void testNullDirectionYieldsANullInItsPlace()
    {
        assertThat(project(vector(1.0, 2.0), directions(v(1.0, 0.0), null, v(0.0, 1.0))))
                .containsExactly(1.0, null, 2.0);
    }

    @Test
    public void testDirectionWithANullElementYieldsANullInItsPlace()
    {
        assertThat(project(vector(1.0, 2.0), directions(v(0.0, null), v(0.0, 1.0))))
                .containsExactly(null, 2.0);
    }

    @Test
    public void testNoDirectionsYieldsAnEmptyArray()
    {
        assertThat(project(vector(1.0, 2.0), directions())).isEmpty();
    }

    @Test
    public void testDirectionOfAnotherDimensionIsRejected()
    {
        assertThatThrownBy(() -> project(vector(1.0, 2.0), directions(v(1.0, 0.0), v(1.0))))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The arguments must have the same length");
    }

    /**
     * The dimension check comes before the null check, so a malformed direction table fails
     * whether or not the offending direction happens to hold a null.
     */
    @Test
    public void testDirectionOfAnotherDimensionIsRejectedEvenWithANullElement()
    {
        assertThatThrownBy(() -> project(vector(1.0, 2.0), directions(v((Double) null))))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The arguments must have the same length");
    }

    /**
     * Long enough to go through the vectorised loop and its scalar tail, against a sum written out
     * by hand in the order of the components.
     */
    @Test
    public void testRealVectorsMatchAPlainSum()
    {
        int dimension = 771;
        Random random = new Random(32);
        Double[] x = randomComponents(random, dimension);
        Double[] first = randomComponents(random, dimension);
        Double[] second = randomComponents(random, dimension);

        List<Double> projections = project(
                vector(REAL_VECTOR, x),
                directions(REAL_VECTOR, first, second),
                REAL_VECTOR,
                REAL_READER);

        assertThat(projections).hasSize(2);
        assertThat(projections.get(0)).isCloseTo(plainDotProduct(x, first), within(1e-9));
        assertThat(projections.get(1)).isCloseTo(plainDotProduct(x, second), within(1e-9));
    }

    /**
     * What makes a projection column prunable: for a unit direction v, |x.v - q.v| never exceeds
     * ||x - q||, so a neighbour within R of q has its projection within R of q's.
     */
    @Test
    public void testProjectionDifferenceIsBoundedByTheDistance()
    {
        int dimension = 64;
        Random random = new Random(7);
        Double[][] unitDirections = orthonormal(random, 8, dimension);
        for (int trial = 0; trial < 200; trial++) {
            Double[] x = randomComponents(random, dimension);
            Double[] q = randomComponents(random, dimension);
            List<Double> px = project(vector(x), directions(unitDirections));
            List<Double> pq = project(vector(q), directions(unitDirections));
            double distance = euclidean(x, q);
            for (int j = 0; j < unitDirections.length; j++) {
                assertThat(Math.abs(px.get(j) - pq.get(j))).isLessThanOrEqualTo(distance + 1e-12);
            }
        }
    }

    /**
     * Bessel's inequality: on orthonormal directions the distance between projections never
     * exceeds the distance between the vectors, which is what lets a radius be estimated from the
     * projection columns alone without overshooting any true distance.
     */
    @Test
    public void testProjectedDistanceIsALowerBoundOnOrthonormalDirections()
    {
        int dimension = 64;
        Random random = new Random(11);
        Double[][] unitDirections = orthonormal(random, 8, dimension);
        for (int trial = 0; trial < 200; trial++) {
            Double[] x = randomComponents(random, dimension);
            Double[] q = randomComponents(random, dimension);
            List<Double> px = project(vector(x), directions(unitDirections));
            List<Double> pq = project(vector(q), directions(unitDirections));
            double projected = 0;
            for (int j = 0; j < unitDirections.length; j++) {
                double difference = px.get(j) - pq.get(j);
                projected += difference * difference;
            }
            assertThat(Math.sqrt(projected)).isLessThanOrEqualTo(euclidean(x, q) + 1e-12);
        }
    }

    private static Double[] randomComponents(Random random, int dimension)
    {
        Double[] components = new Double[dimension];
        for (int i = 0; i < dimension; i++) {
            // Exactly representable as float, so the real path and the plain sum see the same inputs.
            components[i] = (double) (float) random.nextGaussian();
        }
        return components;
    }

    private static Double[][] orthonormal(Random random, int count, int dimension)
    {
        double[][] basis = new double[count][];
        for (int j = 0; j < count; j++) {
            double[] candidate = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                candidate[i] = random.nextGaussian();
            }
            for (int previous = 0; previous < j; previous++) {
                double overlap = 0;
                for (int i = 0; i < dimension; i++) {
                    overlap += candidate[i] * basis[previous][i];
                }
                for (int i = 0; i < dimension; i++) {
                    candidate[i] -= overlap * basis[previous][i];
                }
            }
            double norm = Math.sqrt(Arrays.stream(candidate).map(c -> c * c).sum());
            for (int i = 0; i < dimension; i++) {
                candidate[i] /= norm;
            }
            basis[j] = candidate;
        }
        return Arrays.stream(basis)
                .map(direction -> Arrays.stream(direction).boxed().toArray(Double[]::new))
                .toArray(Double[][]::new);
    }

    private static double plainDotProduct(Double[] first, Double[] second)
    {
        double sum = 0;
        for (int i = 0; i < first.length; i++) {
            sum += first[i] * second[i];
        }
        return sum;
    }

    private static double euclidean(Double[] first, Double[] second)
    {
        double sum = 0;
        for (int i = 0; i < first.length; i++) {
            double difference = first[i] - second[i];
            sum += difference * difference;
        }
        return Math.sqrt(sum);
    }
}
