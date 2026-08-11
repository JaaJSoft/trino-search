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
package dev.jaaj.trino.search;

import dev.jaaj.trino.search.vector.VectorDistanceFunctions;
import dev.jaaj.trino.search.vector.VectorFunctions;
import dev.jaaj.trino.search.vector.knn.KnnAggregation;
import dev.jaaj.trino.search.vector.quantize.VectorBoundsAggregation;
import io.trino.spi.Plugin;

import java.util.Set;

public class SearchPlugin
        implements Plugin
{
    @Override
    public Set<Class<?>> getFunctions()
    {
        return Set.of(
                VectorDistanceFunctions.class,
                VectorFunctions.class,
                KnnAggregation.OfDoubleVectors.class,
                KnnAggregation.OfRealVectors.class,
                VectorBoundsAggregation.OfDoubleVectors.class,
                VectorBoundsAggregation.OfRealVectors.class);
    }
}
