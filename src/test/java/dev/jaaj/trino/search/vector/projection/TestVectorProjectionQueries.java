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

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.MaterializedResult;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

public class TestVectorProjectionQueries
        extends AbstractTestQueryFramework
{
    private static final String DIRECTIONS =
            "ARRAY[ARRAY[DOUBLE '1', DOUBLE '0'], ARRAY[DOUBLE '0.5', DOUBLE '-0.25']]";

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testProjectsOntoEachDirection()
    {
        assertThat(computeScalar("SELECT vector_projections(ARRAY[DOUBLE '3', DOUBLE '4'], %s)".formatted(DIRECTIONS)))
                .isEqualTo(List.of(3.0, 0.5));
    }

    @Test
    public void testRealVectorsAreAccepted()
    {
        assertThat(computeScalar("SELECT vector_projections(ARRAY[REAL '3', REAL '4'], ARRAY[ARRAY[REAL '1', REAL '0'], ARRAY[REAL '0', REAL '1']])"))
                .isEqualTo(List.of(3.0, 4.0));
    }

    /**
     * Projections land in double columns whatever the vector's representation, so both overloads
     * return {@code array(double)}.
     */
    @Test
    public void testResultIsDoubleForEitherRepresentation()
    {
        assertThat(computeScalar("SELECT typeof(vector_projections(ARRAY[REAL '1'], ARRAY[ARRAY[REAL '1']]))"))
                .isEqualTo("array(double)");
        assertThat(computeScalar("SELECT typeof(vector_projections(ARRAY[DOUBLE '1'], ARRAY[ARRAY[DOUBLE '1']]))"))
                .isEqualTo("array(double)");
    }

    @Test
    public void testRealVectorAgainstDoubleDirectionsIsCoerced()
    {
        assertThat(computeScalar("SELECT vector_projections(ARRAY[REAL '3', REAL '4'], %s)".formatted(DIRECTIONS)))
                .isEqualTo(List.of(3.0, 0.5));
    }

    @Test
    public void testVectorWithANullElementReturnsNull()
    {
        assertQuery(
                "SELECT vector_projections(ARRAY[DOUBLE '3', NULL], %s) IS NULL".formatted(DIRECTIONS),
                "SELECT true");
    }

    @Test
    public void testNullDirectionsReturnNull()
    {
        assertQuery(
                "SELECT vector_projections(ARRAY[DOUBLE '3'], CAST(NULL AS array(array(double)))) IS NULL",
                "SELECT true");
    }

    @Test
    public void testNullDirectionYieldsANullInItsPlace()
    {
        assertThat(computeScalar("SELECT vector_projections(ARRAY[DOUBLE '3', DOUBLE '4'], ARRAY[NULL, ARRAY[DOUBLE '0', DOUBLE '1']])"))
                .isEqualTo(Arrays.asList(null, 4.0));
    }

    @Test
    public void testNoDirectionsYieldsAnEmptyArray()
    {
        assertQuery(
                "SELECT cardinality(vector_projections(ARRAY[DOUBLE '3'], CAST(ARRAY[] AS array(array(double)))))",
                "SELECT 0");
    }

    @Test
    public void testDirectionOfAnotherDimensionIsRejected()
    {
        assertQueryFails(
                "SELECT vector_projections(ARRAY[DOUBLE '3'], %s)".formatted(DIRECTIONS),
                "The arguments must have the same length");
    }

    /**
     * The function exists so that N directions cost one call rather than N, not to compute
     * anything {@code dot_product} does not: element j must be the dot product with direction j,
     * on rows long enough to reach the vectorised kernels.
     * <p>
     * The comparison allows a tolerance because exact equality is not reproducible even between
     * two calls of the same kernel: {@code reduceLanes(ADD)} does not fix the order it sums the
     * lanes in, and the Java fallback and the C2 intrinsic sum them differently, so a result moves
     * by a few ulps the moment the kernel gets compiled. The bound sits far above that drift and
     * far below what a wrong direction or a dropped tail would cost.
     */
    @Test
    public void testAgreesWithOneDotProductPerDirection()
    {
        MaterializedResult result = computeActual(
                """
                WITH directions AS (
                    SELECT array_agg(transform(sequence(1, 768), d -> CAST(cos(j * d) AS real)) ORDER BY j) AS ds
                    FROM UNNEST(sequence(1, 4)) AS t(j)
                ),
                rows AS (
                    SELECT id, transform(sequence(1, 768), d -> CAST(sin(id * d) AS real)) AS embedding
                    FROM UNNEST(sequence(1, 50)) AS t(id)
                )
                SELECT count_if(all_match(
                        zip_with(
                                vector_projections(r.embedding, b.ds),
                                transform(b.ds, v -> dot_product(r.embedding, v)),
                                (projection, dot) -> abs(projection - dot) <= 1e-9),
                        matches -> matches))
                FROM rows r CROSS JOIN directions b
                """);

        assertThat(result.getOnlyValue()).isEqualTo(50L);
    }

    /**
     * The write path the documentation describes: directions kept as rows of a table, gathered
     * into one array in a fixed order, and each projection unpacked into its own column.
     */
    @Test
    public void testUnpacksIntoColumnsFromADirectionTable()
    {
        assertQuery(
                """
                WITH projections(idx, direction) AS (
                    VALUES (2, ARRAY[REAL '0', REAL '1']), (1, ARRAY[REAL '1', REAL '0'])
                ),
                staging(id, embedding) AS (
                    VALUES (1, ARRAY[REAL '3', REAL '4']), (2, ARRAY[REAL '-1', REAL '2'])
                )
                SELECT id, p[1], p[2]
                FROM (SELECT id, vector_projections(embedding, (SELECT array_agg(direction ORDER BY idx) FROM projections)) AS p
                      FROM staging)
                """,
                "VALUES (1, 3.0, 4.0), (2, -1.0, 2.0)");
    }
}
