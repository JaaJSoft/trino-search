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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestRecall
{
    private static final double[] TRUTH = {1.0, 2.0, 3.0, 4.0, 5.0};

    @Test
    public void testPerfectResultScoresOne()
    {
        assertThat(Recall.at(3, new double[] {1.0, 2.0, 3.0}, TRUTH, false)).isEqualTo(1.0);
    }

    @Test
    public void testEmptyResultScoresZero()
    {
        assertThat(Recall.at(3, new double[] {}, TRUTH, false)).isEqualTo(0.0);
    }

    @Test
    public void testHalfTheNeighboursScoresAHalf()
    {
        assertThat(Recall.at(4, new double[] {1.0, 2.0, 9.0, 9.0}, TRUTH, false)).isEqualTo(0.5);
    }

    /**
     * A neighbour tied in distance with the k-th true neighbour is a correct answer: with k = 3
     * the threshold is 3.0, and a result returning a different vector also at distance 3.0 must
     * not be penalised. Comparing key sets instead of distances would silently under-report here.
     */
    @Test
    public void testNeighbourTiedWithTheThresholdCounts()
    {
        assertThat(Recall.at(3, new double[] {1.0, 2.0, 3.0000000000001}, TRUTH, false)).isEqualTo(1.0);
    }

    @Test
    public void testHigherIsCloserInvertsTheComparison()
    {
        double[] truth = {5.0, 4.0, 3.0, 2.0, 1.0};
        assertThat(Recall.at(3, new double[] {5.0, 4.0, 3.0}, truth, true)).isEqualTo(1.0);
        assertThat(Recall.at(3, new double[] {5.0, 4.0, 1.0}, truth, true)).isCloseTo(2.0 / 3.0, within(1e-12));
    }

    /**
     * Three results all at or below the k = 2 threshold would score 3/2 without the clamp.
     */
    @Test
    public void testMoreResultsThanKCannotExceedOne()
    {
        assertThat(Recall.at(2, new double[] {1.0, 2.0, 2.0}, TRUTH, false)).isEqualTo(1.0);
    }

    @Test
    public void testInvalidArgumentsAreRejected()
    {
        assertThatThrownBy(() -> Recall.at(0, new double[] {1.0}, TRUTH, false))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> Recall.at(9, new double[] {1.0}, TRUTH, false))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
