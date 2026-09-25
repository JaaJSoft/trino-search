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
package dev.jaaj.trino.search.vector.centroid;

import dev.jaaj.trino.search.vector.Metric;
import io.trino.spi.TrinoException;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.type.ArrayType;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class TestNearestVector
{
    private static final ArrayType VECTOR_TYPE = new ArrayType(DOUBLE);
    private static final ArrayType CANDIDATES_TYPE = new ArrayType(VECTOR_TYPE);

    private static Block vector(Double... values)
    {
        ArrayBlockBuilder builder = (ArrayBlockBuilder) VECTOR_TYPE.createBlockBuilder(null, 1);
        builder.buildEntry(elements -> writeComponents(elements, values));
        return VECTOR_TYPE.getObject(builder.build(), 0);
    }

    /**
     * A null entry stands for a null candidate.
     */
    private static Block candidates(Double[]... vectors)
    {
        ArrayBlockBuilder builder = (ArrayBlockBuilder) CANDIDATES_TYPE.createBlockBuilder(null, 1);
        builder.buildEntry(candidateBuilder -> {
            for (Double[] values : vectors) {
                if (values == null) {
                    candidateBuilder.appendNull();
                }
                else {
                    ((ArrayBlockBuilder) candidateBuilder).buildEntry(elements -> writeComponents(elements, values));
                }
            }
        });
        return CANDIDATES_TYPE.getObject(builder.build(), 0);
    }

    private static void writeComponents(BlockBuilder elements, Double[] values)
    {
        for (Double value : values) {
            if (value == null) {
                elements.appendNull();
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

    private static NearestVector find(Block vector, Block candidates, Metric metric)
    {
        return NearestVector.find(vector, candidates, VECTOR_TYPE, DOUBLE_READER, metric);
    }

    @Test
    public void testFindsTheClosestCandidate()
    {
        NearestVector nearest = find(
                vector(0.0, 0.0),
                candidates(v(3.0, 4.0), v(1.0, 1.0), v(-2.0, 0.0)),
                Metric.EUCLIDEAN);

        assertThat(nearest).isEqualTo(new NearestVector(2, Math.sqrt(2)));
    }

    /**
     * The first candidate sets the limit the others are bounded against, and the winner comes
     * after it. Its distance must be the exact value, not whatever an abandoned accumulation
     * returned.
     */
    @Test
    public void testDistanceOfALaterWinnerIsExact()
    {
        NearestVector nearest = find(
                vector(0.0, 0.0, 0.0),
                candidates(v(10.0, 10.0, 10.0), v(1.0, 2.0, 2.0), v(100.0, 0.0, 0.0)),
                Metric.EUCLIDEAN);

        assertThat(nearest).isEqualTo(new NearestVector(2, 3.0));
    }

    /**
     * A candidate abandoned against the running best returns some value that cannot beat it, not
     * its distance, so it must never be taken for a winner.
     */
    @Test
    public void testAnAbandonedCandidateNeverWins()
    {
        NearestVector nearest = find(
                vector(0.0, 0.0),
                candidates(v(1.0, 0.0), v(1.0, 5.0)),
                Metric.MANHATTAN);

        assertThat(nearest).isEqualTo(new NearestVector(1, 1.0));
    }

    @Test
    public void testDotProductRanksTheHighestAsClosest()
    {
        NearestVector nearest = find(
                vector(1.0, 0.0),
                candidates(v(1.0, 0.0), v(5.0, 0.0), v(-9.0, 0.0)),
                Metric.DOT_PRODUCT);

        assertThat(nearest).isEqualTo(new NearestVector(2, 5.0));
    }

    @Test
    public void testCosineIgnoresMagnitude()
    {
        NearestVector nearest = find(
                vector(1.0, 0.0),
                candidates(v(1.0, 1.0), v(50.0, 0.0)),
                Metric.COSINE);

        assertThat(nearest).isEqualTo(new NearestVector(2, 0.0));
    }

    /**
     * An assignment has to be reproducible from one run to the next, so equidistant candidates
     * resolve to the lowest position rather than to whichever the loop happened to see last.
     */
    @Test
    public void testTiesGoToTheLowestPosition()
    {
        NearestVector nearest = find(
                vector(0.0, 0.0),
                candidates(v(0.0, 1.0), v(1.0, 0.0), v(0.0, -1.0)),
                Metric.EUCLIDEAN);

        assertThat(nearest).isEqualTo(new NearestVector(1, 1.0));
    }

    @Test
    public void testTiesGoToTheLowestPositionWhenHigherIsCloser()
    {
        NearestVector nearest = find(
                vector(1.0, 1.0),
                candidates(v(0.0, 2.0), v(2.0, 0.0)),
                Metric.DOT_PRODUCT);

        assertThat(nearest).isEqualTo(new NearestVector(1, 2.0));
    }

    /**
     * The position is an index into the array the caller passed, which is how it maps back to a
     * cluster id, so skipping a candidate must not renumber the ones after it.
     */
    @Test
    public void testNullCandidateIsSkippedWithoutRenumbering()
    {
        NearestVector nearest = find(
                vector(0.0, 0.0),
                candidates(null, v(5.0, 0.0), v(1.0, 0.0)),
                Metric.EUCLIDEAN);

        assertThat(nearest).isEqualTo(new NearestVector(3, 1.0));
    }

    @Test
    public void testCandidateWithANullElementIsSkipped()
    {
        NearestVector nearest = find(
                vector(0.0, 0.0),
                candidates(v(0.0, null), v(5.0, 0.0)),
                Metric.EUCLIDEAN);

        assertThat(nearest).isEqualTo(new NearestVector(2, 5.0));
    }

    @Test
    public void testNoUsableCandidateReturnsNull()
    {
        assertThat(find(vector(0.0), candidates(null, v((Double) null)), Metric.EUCLIDEAN)).isNull();
    }

    @Test
    public void testNoCandidatesReturnsNull()
    {
        assertThat(find(vector(0.0), candidates(), Metric.EUCLIDEAN)).isNull();
    }

    /**
     * A distance that overflows to infinity is still a distance, and on its own it is the nearest
     * one there is.
     */
    @Test
    public void testAnInfiniteDistanceCanStillWin()
    {
        NearestVector nearest = find(
                vector(0.0),
                candidates(v(Double.MAX_VALUE)),
                Metric.EUCLIDEAN_SQUARED);

        assertThat(nearest).isEqualTo(new NearestVector(1, Double.POSITIVE_INFINITY));
    }

    @Test
    public void testCandidateOfAnotherDimensionIsRejected()
    {
        assertThatThrownBy(() -> find(vector(0.0, 0.0), candidates(v(1.0, 0.0), v(1.0)), Metric.EUCLIDEAN))
                .isInstanceOf(TrinoException.class)
                .hasMessage("The arguments must have the same length");
    }
}
