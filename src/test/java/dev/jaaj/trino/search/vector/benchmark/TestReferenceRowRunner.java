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
}
