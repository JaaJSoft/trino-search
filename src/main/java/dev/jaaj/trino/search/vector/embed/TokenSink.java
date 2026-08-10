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
package dev.jaaj.trino.search.vector.embed;

import io.airlift.slice.Slice;

/**
 * Receives each token as a byte range of the text it came from. Handing over a range rather than
 * a {@code Slice} keeps a long text from allocating one object per token, which on a column scan
 * is the difference between a few allocations per row and a few thousand.
 */
@FunctionalInterface
interface TokenSink
{
    void accept(Slice text, int offset, int length);
}
