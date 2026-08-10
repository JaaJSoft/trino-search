# Trino-search

[Trino](https://trino.io) plugin providing search functions. The first family covers vectors:
distance metrics, normalization and exact k-nearest-neighbour search.

The plugin exposes functions only, no catalog and no connector: dropping the JAR into the
plugin directory makes the functions available globally.

## Functions

Vectors are `array(double)` or `array(real)`. No custom type, no `CAST`.

### Metrics

| Function | Description |
| --- | --- |
| `euclidean_squared_distance(x, y)` | squared euclidean distance, without the `sqrt` |
| `manhattan_distance(x, y)` | L1 distance |
| `l2_norm(x)` | euclidean norm |
| `normalize_vector(x)` | unit-norm vector, of the same type as the input |

Trino already ships `euclidean_distance`, `dot_product`, `cosine_similarity` and
`cosine_distance`, but only on `array(double)`. This plugin adds the `array(real)` overloads
under the same names, so an `array(real)` column does not need a `CAST` that would double the
memory read per row.

> One consequence: an untyped decimal literal such as `ARRAY[0.1, 0.2]` binds to the
> `array(real)` overload, with the matching precision and `NULL` handling. A genuinely typed
> `array(double)` column, an explicit `CAST` or `DOUBLE 'x'` literals all keep the engine's
> native implementation.

### Text embeddings

```sql
to_vector_real(text, dimension, algorithm)   -> array(real)
to_vector_double(text, dimension, algorithm) -> array(double)
```

Embeds text by feature hashing: every token is hashed to an index and a sign, and contributes one
unit there. No model, no vocabulary and no external call, so the vector of a row depends on that
row alone and is stable across servers and restarts.

`algorithm` is one of `'word'`, `'char_3gram'`, `'char_4gram'` or `'char_5gram'`. `'word'` splits
on non-alphanumeric characters; the n-gram variants slide a window of that many characters, which
tolerates typos and handles languages that do not separate words with spaces. `dimension` must be
between 1 and 65536.

The result is always of unit norm, so cosine distance and euclidean distance rank identically.
Text containing no token returns the zero vector rather than raising, so a single empty row cannot
fail a scan. `to_vector_fp32` and `to_vector_fp64` are aliases of the `real` and `double` forms.

Feature hashing captures token overlap, not meaning: two texts sharing no word are far apart even
if they say the same thing. It suits deduplication, tag and identifier matching, and near-duplicate
detection, and it is not a substitute for a learned embedding model.

```sql
-- three nearest titles per category, embedded on the fly
SELECT category, knn_agg(id, to_vector_double(title, 256, 'word'), to_vector_double('trino query engine', 256, 'word'), 3, 'cosine')
FROM documents
GROUP BY category;
```

### Aggregation

```sql
knn_agg(key, vector, query_vector, k, metric) -> array(row(key, distance))
```

Returns the `k` nearest neighbours of `query_vector` **per group**, nearest first. `metric` is
one of `'euclidean'`, `'euclidean_squared'`, `'cosine'`, `'dot_product'` or `'manhattan'`. `k`
is capped at 10000, and both `k` and `metric` must be constant within a group.

## Examples

```sql
-- global top 10 over an array(real) column
SELECT id, euclidean_distance(embedding, ARRAY[REAL '0.1', REAL '0.2', REAL '0.3']) AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- top 3 per category
SELECT category, knn_agg(id, embedding, ARRAY[0.1, 0.2, 0.3], 3, 'cosine') AS neighbors
FROM documents
GROUP BY category;
```

## Compatibility

Trino 483, Java 25.

## Installation

```bash
./mvnw clean package
```

Copy the contents of `target/trino-search-<version>/` into `<trino>/plugin/search/`, then
restart the server.

## Status

v1 implements exact KNN. Approximate search (ANN) is planned.

## Benchmarks

Vector search performance is tracked per pull request in [`BENCHMARKS.md`](BENCHMARKS.md).

## License

Apache License 2.0
