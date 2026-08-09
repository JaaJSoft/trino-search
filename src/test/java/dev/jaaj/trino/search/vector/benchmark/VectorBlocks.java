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

import io.trino.spi.block.Block;
import io.trino.spi.block.IntArrayBlock;
import io.trino.spi.block.LongArrayBlock;

import java.util.Optional;

/**
 * Turns plain {@code double[]} vectors into the blocks the aggregation actually reads.
 * <p>
 * There is no {@code DoubleArrayBlock} in the Trino SPI: {@code DoubleType.getDouble} casts the
 * underlying value block to {@link LongArrayBlock} and reinterprets the bits, so an
 * {@code array(double)} vector is a block of {@code Double.doubleToLongBits} values. An
 * {@code array(real)} vector is an {@link IntArrayBlock} of {@code Float.floatToRawIntBits}
 * values, which is what {@code RealType.getFloat} reads.
 */
public final class VectorBlocks
{
    private VectorBlocks() {}

    public static Block doubleVector(double[] vector)
    {
        long[] values = new long[vector.length];
        for (int i = 0; i < vector.length; i++) {
            values[i] = Double.doubleToLongBits(vector[i]);
        }
        return new LongArrayBlock(vector.length, Optional.empty(), values);
    }

    public static Block realVector(double[] vector)
    {
        int[] values = new int[vector.length];
        for (int i = 0; i < vector.length; i++) {
            values[i] = Float.floatToRawIntBits((float) vector[i]);
        }
        return new IntArrayBlock(vector.length, Optional.empty(), values);
    }

    public static Block[] doubleVectors(double[][] vectors)
    {
        Block[] blocks = new Block[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            blocks[i] = doubleVector(vectors[i]);
        }
        return blocks;
    }

    public static Block[] realVectors(double[][] vectors)
    {
        Block[] blocks = new Block[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            blocks[i] = realVector(vectors[i]);
        }
        return blocks;
    }

    /**
     * Sequential {@code 0..count-1} keys, the key column {@code KnnAggregation.input} expects
     * when the row order itself is the identity to recover from the heap.
     */
    public static LongArrayBlock sequentialKeys(int count)
    {
        long[] ids = new long[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i;
        }
        return new LongArrayBlock(count, Optional.empty(), ids);
    }

    /**
     * The {@code array(real)} path computes distances from float-rounded components. An oracle
     * fed the original doubles would rank near-ties differently and report a recall below one for
     * a perfectly correct result, so it must be given these values instead.
     */
    public static double[] roundedToFloat(double[] vector)
    {
        double[] rounded = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            rounded[i] = (float) vector[i];
        }
        return rounded;
    }

    public static double[][] roundedToFloat(double[][] vectors)
    {
        double[][] rounded = new double[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
            rounded[i] = roundedToFloat(vectors[i]);
        }
        return rounded;
    }
}
