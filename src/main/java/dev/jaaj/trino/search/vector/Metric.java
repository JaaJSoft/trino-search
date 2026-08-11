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
package dev.jaaj.trino.search.vector;

import dev.jaaj.trino.search.vector.quantize.BinaryVectorMath;
import dev.jaaj.trino.search.vector.quantize.QuantizationBounds;
import dev.jaaj.trino.search.vector.quantize.QuantizedVectorMath;
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;

import java.util.Locale;
import java.util.stream.Stream;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

public enum Metric
{
    EUCLIDEAN("euclidean", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.euclidean(first, second, reader);
        }

        /**
         * The accumulation is squared while the limit is a distance, so the limit is squared to
         * meet it. Comparing a partial sum of squares against the limit as given would abandon
         * candidates that beat it, silently returning the wrong neighbours.
         */
        @Override
        public double computeBounded(Block first, Block second, VectorReader reader, double limit)
        {
            return Math.sqrt(VectorMath.euclideanSquaredBounded(first, second, reader, limit * limit));
        }

        /**
         * The accumulation is squared while the limit is a distance, so the limit is squared to
         * meet it, exactly as {@link #computeBounded} does for the float representation.
         */
        @Override
        public double computeQuantizedBounded(Block first, Block second, QuantizationBounds bounds, double limit)
        {
            return Math.sqrt(QuantizedVectorMath.euclideanSquaredBounded(first, second, bounds, limit * limit));
        }

        @Override
        public double computeBinary(Slice first, Slice second)
        {
            return BinaryVectorMath.euclidean(first, second);
        }
    },
    EUCLIDEAN_SQUARED("euclidean_squared", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.euclideanSquared(first, second, reader);
        }

        @Override
        public double computeBounded(Block first, Block second, VectorReader reader, double limit)
        {
            return VectorMath.euclideanSquaredBounded(first, second, reader, limit);
        }

        @Override
        public double computeQuantizedBounded(Block first, Block second, QuantizationBounds bounds, double limit)
        {
            return QuantizedVectorMath.euclideanSquaredBounded(first, second, bounds, limit);
        }

        @Override
        public double computeBinary(Slice first, Slice second)
        {
            return BinaryVectorMath.euclideanSquared(first, second);
        }
    },
    COSINE("cosine", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return 1.0 - VectorMath.cosineSimilarity(first, second, reader);
        }

        @Override
        public double computeQuantizedBounded(Block first, Block second, QuantizationBounds bounds, double limit)
        {
            return 1.0 - QuantizedVectorMath.cosineSimilarity(first, second, bounds);
        }

        @Override
        public double computeBinary(Slice first, Slice second)
        {
            return 1.0 - BinaryVectorMath.cosineSimilarity(first, second);
        }
    },
    DOT_PRODUCT("dot_product", true) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.dotProduct(first, second, reader);
        }

        @Override
        public double computeQuantizedBounded(Block first, Block second, QuantizationBounds bounds, double limit)
        {
            return QuantizedVectorMath.dotProduct(first, second, bounds);
        }

        @Override
        public double computeBinary(Slice first, Slice second)
        {
            return BinaryVectorMath.dotProduct(first, second);
        }
    },
    MANHATTAN("manhattan", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.manhattan(first, second, reader);
        }

        @Override
        public double computeQuantizedBounded(Block first, Block second, QuantizationBounds bounds, double limit)
        {
            return QuantizedVectorMath.manhattan(first, second, bounds);
        }

        @Override
        public double computeBinary(Slice first, Slice second)
        {
            return BinaryVectorMath.manhattan(first, second);
        }
    };

    private final String sqlName;
    private final Slice sqlNameUtf8;
    private final boolean higherIsCloser;

    Metric(String sqlName, boolean higherIsCloser)
    {
        this.sqlName = sqlName;
        this.sqlNameUtf8 = Slices.utf8Slice(sqlName);
        this.higherIsCloser = higherIsCloser;
    }

    public abstract double compute(Block first, Block second, VectorReader reader);

    /**
     * The metric over two vectors of one-byte codes fitted against the same bounds. The limit has
     * the meaning {@link #computeBounded} gives it: a metric accumulated from non-negative terms
     * may return any value above it once the components read so far already put it there.
     */
    public abstract double computeQuantizedBounded(Block first, Block second, QuantizationBounds bounds, double limit);

    /**
     * The metric over two one-bit-per-component vectors. There is no bounded form: every one of
     * these is a closed form in a single Hamming count that is already computed in one pass, so
     * there is nothing to abandon part way through.
     */
    public abstract double computeBinary(Slice first, Slice second);

    /**
     * The metric value, or, once the components read so far already show the candidate cannot beat
     * {@code limit}, some value that cannot beat it either. A caller ranking candidates against a
     * running best is asking whether this one wins, not what its exact value is, and for a metric
     * accumulated from non-negative terms that question is often settled long before the last
     * component.
     * <p>
     * Metrics whose terms are signed, which here means the ones that rank higher as closer, cannot
     * settle it early: a partial sum says nothing about the total. They compute the whole thing and
     * ignore the limit, which is what this default does.
     */
    public double computeBounded(Block first, Block second, VectorReader reader, double limit)
    {
        return compute(first, second, reader);
    }

    public String sqlName()
    {
        return sqlName;
    }

    /**
     * Whether a raw UTF-8 name is the canonical spelling of this metric, answered without decoding
     * it, for callers that ask once per row. A false result only means the name is not the
     * canonical spelling: {@code 'EUCLIDEAN'} still resolves to {@link #EUCLIDEAN} through
     * {@link #fromName}.
     */
    public boolean hasCanonicalName(Slice name)
    {
        return sqlNameUtf8.equals(name);
    }

    /**
     * Dot product is a similarity, every other metric is a distance. The heap needs the
     * direction; the value handed back to the user is always the raw metric value.
     */
    public boolean higherIsCloser()
    {
        return higherIsCloser;
    }

    public static Metric fromName(Slice name)
    {
        return fromName(name.toStringUtf8());
    }

    public static Metric fromName(String name)
    {
        String normalized = name.toLowerCase(Locale.ROOT);
        for (Metric metric : values()) {
            if (metric.sqlName.equals(normalized)) {
                return metric;
            }
        }
        throw new TrinoException(INVALID_FUNCTION_ARGUMENT,
                "Unknown metric '%s', expected one of: %s".formatted(
                        name,
                        Stream.of(values()).map(metric -> metric.sqlName).toList()));
    }
}
