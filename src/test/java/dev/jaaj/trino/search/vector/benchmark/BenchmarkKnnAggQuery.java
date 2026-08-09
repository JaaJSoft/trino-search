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

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.Session;
import io.trino.plugin.memory.MemoryPlugin;
import io.trino.plugin.tpch.TpchPlugin;
import io.trino.testing.MaterializedResult;
import io.trino.testing.StandaloneQueryRunner;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import static io.trino.testing.TestingSession.testSessionBuilder;

/**
 * A knn_agg query through a real engine, next to the ORDER BY ... LIMIT k it replaces.
 * <p>
 * The dataset is materialized once into the memory connector at trial setup, so building the
 * vectors is never inside a measurement. It deliberately does not share data with
 * {@link VectorDataset}: loading Java-generated vectors into Trino would need gigabyte-scale
 * INSERT statements, and generating them inside the query would dominate the measurement. This
 * benchmark reports timing only, and the cost of a distance does not depend on the values.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "-Xmx4g"})
@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 5, time = 5)
public class BenchmarkKnnAggQuery
{
    private static final int K = 10;

    /**
     * Rows and dimension travel together as one parameter because JMH crosses {@code @Param}
     * axes: separate parameters would mechanically generate 1000000x768, which is 6.1 GB of
     * doubles.
     */
    @Param({"100000x128", "1000000x128", "100000x768"})
    public String shape;

    @Param({"euclidean", "cosine"})
    public String metricName;

    private StandaloneQueryRunner queryRunner;
    private String knnAggQuery;
    private String orderByLimitQuery;

    @Setup(Level.Trial)
    public void setUp()
    {
        int rows = Integer.parseInt(shape.split("x")[0]);
        int dimension = Integer.parseInt(shape.split("x")[1]);

        Session session = testSessionBuilder().setCatalog("memory").setSchema("default").build();
        queryRunner = new StandaloneQueryRunner(session);
        queryRunner.installPlugin(new TpchPlugin());
        queryRunner.createCatalog("tpch", "tpch", Map.of());
        queryRunner.installPlugin(new MemoryPlugin());
        // The default max-data-per-node is 128 MB, which the largest shape exceeds by an order of
        // magnitude and which would make CREATE TABLE fail rather than any measurement.
        //
        // splits-per-node defaults to the processor count, which a CPU-constrained container
        // reports as 1: that would collapse the plan to a single aggregation stage, the one shape
        // that never exercises KnnStateSerializer or the combine function. Pinning it also fixes
        // the scan parallelism, so the measured numbers do not silently change with the core count
        // of whatever machine runs them.
        queryRunner.createCatalog(
                "memory",
                "memory",
                Map.of("memory.max-data-per-node", "2GB", "memory.splits-per-node", "4"));
        queryRunner.installPlugin(new SearchPlugin());

        queryRunner.execute("CREATE TABLE memory.default.vectors AS " + generatingQuery(rows, dimension));

        String queryVector = queryVector(dimension);
        knnAggQuery = "SELECT knn_agg(id, v, %s, %s, '%s') FROM memory.default.vectors"
                .formatted(queryVector, K, metricName);
        orderByLimitQuery = "SELECT id FROM memory.default.vectors ORDER BY %s(v, %s) ASC LIMIT %s"
                .formatted(scalarFunctionFor(metricName), queryVector, K);
    }

    /**
     * The two measured queries are only comparable while the scalar function ranks rows the same
     * way knn_agg's metric does. A metric added to the parameter list without a counterpart here
     * has to fail loudly rather than fall back to a distance that quietly answers a different
     * question and makes the comparison meaningless.
     */
    private static String scalarFunctionFor(String metricName)
    {
        return switch (metricName) {
            case "euclidean" -> "euclidean_distance";
            case "cosine" -> "cosine_distance";
            default -> throw new IllegalArgumentException(
                    "no scalar counterpart is known for metric '%s'".formatted(metricName));
        };
    }

    @TearDown(Level.Trial)
    public void tearDown()
    {
        if (queryRunner != null) {
            queryRunner.close();
            queryRunner = null;
        }
    }

    /**
     * Deterministic vectors derived from orderkey. The cost is paid once, in the CREATE TABLE at
     * setup, never inside a measured query.
     */
    public static String generatingQuery(int rows, int dimension)
    {
        return """
               SELECT orderkey AS id,
                      transform(sequence(1, %s), i -> sin(orderkey * 0.37 + i * 1.13)) AS v
               FROM tpch.sf1.orders
               LIMIT %s
               """.formatted(dimension, rows);
    }

    private static String queryVector(int dimension)
    {
        return "transform(sequence(1, %s), i -> sin(i * 0.91))".formatted(dimension);
    }

    String explainKnnAgg()
    {
        MaterializedResult result = queryRunner.execute("EXPLAIN (TYPE DISTRIBUTED) " + knnAggQuery);
        return (String) result.getOnlyValue();
    }

    @Benchmark
    public long knnAgg()
    {
        return queryRunner.execute(knnAggQuery).getRowCount();
    }

    @Benchmark
    public long orderByLimit()
    {
        return queryRunner.execute(orderByLimitQuery).getRowCount();
    }
}
