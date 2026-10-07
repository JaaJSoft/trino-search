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

import io.trino.spi.function.AccumulatorState;
import io.trino.spi.function.AccumulatorStateMetadata;

@AccumulatorStateMetadata(
        stateFactoryClass = PcaStateFactory.class,
        stateSerializerClass = PcaStateSerializer.class,
        serializedType = "ROW(BIGINT, BIGINT, ARRAY(DOUBLE), ARRAY(DOUBLE))")
public interface PcaState
        extends AccumulatorState
{
    /**
     * The group's covariance, or null before its first usable vector. Mutating it in place is
     * safe for the size this state reports: a covariance allocates every array it will ever hold
     * when it is created.
     */
    Covariance getCovariance();

    void setCovariance(Covariance covariance);

    /**
     * The number of directions asked for, or zero before the group's first row.
     */
    long getComponents();

    void setComponents(long components);
}
