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
import io.airlift.slice.Slice;
import io.trino.spi.block.Block;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import io.trino.spi.type.StandardTypes;

import java.util.List;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.block.RowValueBuilder.buildRowValue;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.IntegerType.INTEGER;
import static io.trino.spi.type.RealType.REAL;

public final class NearestVectorFunctions
{
    private static final String RESULT_TYPE_SIGNATURE = "row(position integer, distance double)";
    private static final RowType RESULT_TYPE = RowType.from(List.of(
            RowType.field("position", INTEGER),
            RowType.field("distance", DOUBLE)));
    private static final ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);
    private static final ArrayType REAL_ARRAY = new ArrayType(REAL);

    private NearestVectorFunctions() {}

    @Description("Returns the 1-based position of the candidate nearest to a vector, and its distance")
    @ScalarFunction("nearest_vector")
    @SqlType(RESULT_TYPE_SIGNATURE)
    @SqlNullable
    public static SqlRow nearestVector(
            @SqlType("array(double)") Block vector,
            @SqlType("array(array(double))") Block candidates,
            @SqlType(StandardTypes.VARCHAR) Slice metricName)
    {
        return nearest(vector, candidates, DOUBLE_ARRAY, DOUBLE_READER, metricName);
    }

    @Description("Returns the 1-based position of the candidate nearest to a vector, and its distance")
    @ScalarFunction("nearest_vector")
    @SqlType(RESULT_TYPE_SIGNATURE)
    @SqlNullable
    public static SqlRow nearestVectorReal(
            @SqlType("array(real)") Block vector,
            @SqlType("array(array(real))") Block candidates,
            @SqlType(StandardTypes.VARCHAR) Slice metricName)
    {
        return nearest(vector, candidates, REAL_ARRAY, REAL_READER, metricName);
    }

    private static SqlRow nearest(Block vector, Block candidates, ArrayType candidateType, VectorReader reader, Slice metricName)
    {
        Metric metric = Metric.fromName(metricName);
        if (vector.hasNull()) {
            return null;
        }
        NearestVector nearest = NearestVector.find(vector, candidates, candidateType, reader, metric);
        if (nearest == null) {
            return null;
        }
        return buildRowValue(RESULT_TYPE, fieldBuilders -> {
            INTEGER.writeLong(fieldBuilders.get(0), nearest.position());
            DOUBLE.writeDouble(fieldBuilders.get(1), nearest.distance());
        });
    }
}
