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

import dev.jaaj.trino.search.vector.embed.EmbeddingFunctions;
import io.airlift.slice.Slice;
import io.airlift.slice.Slices;
import io.trino.spi.block.Block;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * The cost of embedding one row, through the SQL entry points because everything below them is
 * package private.
 * <p>
 * The two axes are the ones that trade against each other: the scatter is linear in the number of
 * tokens and the normalization tail is linear in the dimension, so a short text in a wide vector
 * is dominated by the tail and a long text in a narrow one by the scatter. Reading one shape
 * alone would support whichever conclusion the reader already held.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
public class BenchmarkToVector
{
    /**
     * A pool rather than one text, so the measurement is not of a single string resident in L1
     * with a JIT specialised on its length. A power of two keeps the wrap-around a mask.
     */
    private static final int TEXT_POOL_SIZE = 256;

    private static final String ALPHABET = "abcdefghijklmnopqrstuvwxyz";

    @Param({"128", "768", "1536"})
    public int dimension;

    @Param({"8", "64", "512"})
    public int wordCount;

    @Param({"word", "char_3gram"})
    public String algorithmName;

    private Slice[] texts;
    private Slice algorithm;
    private int index;

    @Setup(Level.Trial)
    public void setUp()
    {
        Random random = new Random(1L);
        texts = new Slice[TEXT_POOL_SIZE];
        for (int i = 0; i < TEXT_POOL_SIZE; i++) {
            StringBuilder builder = new StringBuilder();
            for (int word = 0; word < wordCount; word++) {
                if (word > 0) {
                    builder.append(' ');
                }
                int letters = 3 + random.nextInt(6);
                for (int letter = 0; letter < letters; letter++) {
                    builder.append(ALPHABET.charAt(random.nextInt(ALPHABET.length())));
                }
            }
            texts[i] = Slices.utf8Slice(builder.toString());
        }
        algorithm = Slices.utf8Slice(algorithmName);
    }

    @Benchmark
    public Block realVectors()
    {
        index = (index + 1) & (TEXT_POOL_SIZE - 1);
        return EmbeddingFunctions.toVectorReal(texts[index], dimension, algorithm);
    }

    @Benchmark
    public Block doubleVectors()
    {
        index = (index + 1) & (TEXT_POOL_SIZE - 1);
        return EmbeddingFunctions.toVectorDouble(texts[index], dimension, algorithm);
    }
}
