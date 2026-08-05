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

import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

public class TestSearchPlugin
{
    @Test
    public void testPluginLoads()
    {
        try (QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build())) {
            queryRunner.installPlugin(new SearchPlugin());
            assertThat(queryRunner.execute("SELECT 1").getOnlyValue()).isEqualTo(1);
        }
    }
}
