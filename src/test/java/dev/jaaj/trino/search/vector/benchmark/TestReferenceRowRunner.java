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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class TestReferenceRowRunner
{
    @Test
    public void testMachineLabelIsTheFirstArgument()
    {
        assertThat(ReferenceRowRunner.machineLabel(new String[] {"laptop"})).isEqualTo("laptop");
        assertThat(ReferenceRowRunner.machineLabel(new String[] {"laptop", "11"})).isEqualTo("laptop");
    }

    /**
     * The pull request does not exist yet while the measurement runs, so a row can legitimately
     * be recorded without one and completed by hand when the pull request is opened.
     */
    @Test
    public void testAMissingPullRequestBecomesTbd()
    {
        assertThat(ReferenceRowRunner.pullRequestLabel(new String[] {"laptop"})).isEqualTo("#TBD");
    }

    /**
     * Both "11" and "#11" are natural things to type, and the column is only sortable and
     * greppable if they land in the file identically.
     */
    @Test
    public void testAPullRequestIsNormalisedToASingleLeadingHash()
    {
        assertThat(ReferenceRowRunner.pullRequestLabel(new String[] {"laptop", "11"})).isEqualTo("#11");
        assertThat(ReferenceRowRunner.pullRequestLabel(new String[] {"laptop", "#11"})).isEqualTo("#11");
    }

    /**
     * Stripping every hash and prefixing one turned "1#1" into "#11" and "draft" into "#draft",
     * both of which look like a legitimate identifier once pasted. A row is only worth keeping if
     * its pull request column points at a real pull request, so anything that is not a number
     * fails at the command line where it can still be retyped.
     */
    /**
     * The detected model goes straight into a markdown cell, so what matters is not which string
     * comes back but that it cannot break the row: both sources pad the name with runs of spaces,
     * and a pipe would end the cell early and shift every column after it.
     */
    @Test
    public void testTheDetectedCpuIsFitForAMarkdownCell()
    {
        String cpu = ReferenceRowRunner.currentCpu();
        assertThat(cpu)
                .isNotBlank()
                .isEqualTo(cpu.strip())
                .doesNotContain("  ")
                .doesNotContain("|")
                .doesNotContain("\n");
    }

    @Test
    public void testAPullRequestThatIsNotANumberIsRejected()
    {
        for (String rejected : new String[] {"1#1", "draft", "", "#", "11a", "-11", "#1#1"}) {
            assertThatThrownBy(() -> ReferenceRowRunner.pullRequestLabel(new String[] {"laptop", rejected}))
                    .as(rejected)
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }
}
