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
import dev.jaaj.trino.search.vector.VectorReader;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.type.ArrayType;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

/**
 * The candidate closest to a vector, as its 1-based position in the candidates array and the raw
 * metric value between the two.
 */
public record NearestVector(int position, double distance)
{
    /**
     * Returns the nearest usable candidate, or null when there is none. A null candidate, or one
     * with a null component, is skipped without renumbering the others: the position is an index
     * into the array the caller built, which is how it maps back to a cluster id.
     * <p>
     * Equidistant candidates resolve to the lowest position, so an assignment is reproducible.
     * That is also why the comparison is strict: {@link Metric#computeBounded} returns, for a
     * candidate it abandoned, a value that cannot beat the limit but may equal it.
     */
    public static NearestVector find(Block vector, Block candidates, ArrayType candidateType, VectorReader reader, Metric metric)
    {
        int bestPosition = -1;
        double bestDistance = Double.NaN;
        for (int i = 0; i < candidates.getPositionCount(); i++) {
            if (candidates.isNull(i)) {
                continue;
            }
            Block candidate = candidateType.getObject(candidates, i);
            if (candidate.getPositionCount() != vector.getPositionCount()) {
                throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
            }
            if (candidate.hasNull()) {
                continue;
            }
            // The vector is the second operand because it is the one that does not change from
            // candidate to candidate, which is the operand cosine remembers the magnitude of.
            double distance;
            boolean closer;
            if (bestPosition < 0) {
                distance = metric.compute(candidate, vector, reader);
                closer = !Double.isNaN(distance);
            }
            else if (metric.higherIsCloser()) {
                distance = metric.compute(candidate, vector, reader);
                closer = distance > bestDistance;
            }
            else {
                distance = metric.computeBounded(candidate, vector, reader, bestDistance);
                closer = distance < bestDistance;
            }
            if (closer) {
                bestPosition = i;
                bestDistance = distance;
            }
        }
        if (bestPosition < 0) {
            return null;
        }
        return new NearestVector(bestPosition + 1, bestDistance);
    }
}
