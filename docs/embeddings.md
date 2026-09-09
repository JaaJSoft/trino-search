# Text embeddings

```sql
to_vector_real(text, dimension, algorithm)   -> array(real)
to_vector_double(text, dimension, algorithm) -> array(double)
```

`to_vector_fp32` and `to_vector_fp64` are aliases of the `real` and `double` forms.

Embeds text by feature hashing: every token is hashed to an index and a sign, and contributes one
unit there. No model, no vocabulary and no external call, so the vector of a row depends on that
row alone and is stable across servers and restarts.

That stability holds within a plugin version and is not guaranteed across upgrades: a vector
stored in a table should be recomputed when the plugin is upgraded, or compared only against
vectors produced by the same version.

## Arguments

`algorithm` is one of:

| Value | Tokens |
| --- | --- |
| `'word'` | runs of alphanumeric characters, splitting on everything else |
| `'char_3gram'` | every sliding window of 3 characters |
| `'char_4gram'` | every sliding window of 4 characters |
| `'char_5gram'` | every sliding window of 5 characters |

The n-gram variants tolerate typos and handle languages that do not separate words with spaces.

`dimension` must be between 1 and 65536.

## The zero vector

The result has unit norm, so euclidean and cosine distance rank identically, with one exception:
text containing no token returns the zero vector rather than raising, so a single empty row cannot
fail a scan by itself. Text shorter than the n-gram window contains no token either, since no
window fits in it: `to_vector_double('hi', 256, 'char_5gram')` is the zero vector.

That zero vector still works with euclidean distance, but a zero vector has no direction for
cosine to compare, so passing it to `cosine_similarity`, `cosine_distance` or `knn_agg` with the
`'cosine'` metric raises `Vector magnitude cannot be zero`. Filter out the empty and too-short
rows, or use euclidean distance, if the input can contain them.

Text that is not valid UTF-8 is likewise not fatal: invalid byte sequences are replaced with the
Unicode replacement character before tokenizing, so a row with corrupted encoding still embeds
instead of failing the query.

## What feature hashing is good for

Feature hashing captures token overlap, not meaning: two texts sharing no word are far apart even
if they say the same thing. It suits deduplication, tag and identifier matching, and
near-duplicate detection, and it is not a substitute for a learned embedding model.

```sql
-- three nearest titles per category, embedded on the fly
SELECT category, knn_agg(
           id,
           to_vector_double(title, 256, 'word'),
           to_vector_double('trino query engine', 256, 'word'),
           3,
           'euclidean') AS neighbours
FROM documents
GROUP BY category;
```
