# Trino-search

[Trino](https://trino.io) plugin providing search functions. The first family covers vectors:
distance metrics, normalization and exact k-nearest-neighbour search.

The plugin exposes functions only, no catalog and no connector: dropping the JAR into the
plugin directory makes the functions available globally.

## Functions

Vectors are `array(double)`, `array(real)`, `array(tinyint)` for one byte per component, or
`varbinary` for one bit. No custom type, no `CAST`.

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

### Quantisation

A vector can be stored quantised instead of as `double` or `real` components: one signed byte
per component (`array(tinyint)`), or one bit (`varbinary`). At dimension 768 an `array(double)`
row is 6144 bytes, an `array(real)` row is 3072, an `array(tinyint)` row is 768, and a binary row
is 100 (a four-byte header plus `ceil(768 / 8)` packed bytes). That is four times fewer bytes than
`array(real)` at the int8 end of the range, and about sixty-one times fewer than `array(double)`
at the binary end.

That byte reduction is the point: it is what a scan reads and what a corpus costs to store, not a
faster distance computation. `BENCHMARKS.md` measures int8's kernel as more expensive per
component than the float kernels, because both of its benchmarks are arithmetic-bound and int8
does more per-component work: each code widens into a double lane and is multiplied by a scale
before the kernel can proceed. Binary's kernel, an XOR and a population count, genuinely is
several times cheaper. Read the "How to read this" section of `BENCHMARKS.md` before drawing a
performance conclusion from any one row; int8's payoff is storage and scan volume at corpus
scale, which no benchmark in this repository is large enough to reach.

| Function | Description |
| --- | --- |
| `vector_bounds_agg(x)` | fits per-dimension offsets and a single global scale over a corpus |
| `quantize_vector_tinyint(x, bounds)` | one signed byte per component; alias `quantize_vector_int8` |
| `quantize_vector_varbinary(x, bounds)` | one bit per component; aliases `quantize_vector_binary` and `quantize_vector_int1` |
| `hamming_distance(x, y)` | components that differ between two binary vectors |

Fit the bounds once over a corpus, then encode:

```sql
CREATE TABLE quantisation AS SELECT vector_bounds_agg(embedding) AS p FROM documents;

ALTER TABLE documents ADD COLUMN embedding_int8 array(tinyint);
UPDATE documents SET embedding_int8 = quantize_vector_tinyint(embedding, (SELECT p FROM quantisation));
```

Every metric gains an overload per representation. On `array(tinyint)` both vectors are codes and
the fitted bounds are a **mandatory third argument**:

```sql
euclidean_distance(codes_a, codes_b, bounds)
```

> Dropping that argument does not fail. `array(tinyint)` coerces implicitly to the float vector
> types, so a two-argument call compiles, binds to an exact-vector overload, and computes on the raw
> codes with no scale applied at all: a plausible-looking number that is not the true distance. The
> bounds are what turn the raw codes back into the real metric, and for `cosine_similarity` they are
> what make it meaningful at all, since cosine is not translation-invariant.

Both operands must have been fitted against the bounds passed. Nothing checks that.

On `varbinary` the metrics take two arguments and no bounds, since a binary code needs nothing
beyond itself to be read. The codes stand for a vector of `-1` and `+1` components, so every
metric is a closed form in the Hamming distance and all of them rank identically. A value is a
four-byte big-endian dimension header followed by `ceil(dimension / 8)` bytes, least significant
bit first.

The `quantize_vector_int1` spelling is there for the vocabulary that counts bits. The codes decode
to `-1` and `+1`, not to the `-1` and `0` a one-bit two's complement integer would hold.

### Approximate search by quantisation

Ranking on codes is approximate. The suite's measured recall floors (`TestQuantizedKnnAggRecall`)
show why oversampling matters, particularly for binary codes:

| representation | regime | oversampling | measured recall |
| --- | --- | --- | --- |
| int8 | clustered | 1x | 0.98 |
| int8 | uniform | 1x | 1.00 |
| binary | clustered | 10x | 1.00 |
| binary | uniform | 1x | 0.32 |
| binary | uniform | 10x | 0.88 |

A one-bit code recovering only about a third of the true neighbours at 1x under the uniform regime
is not a defect: in high dimension every pairwise distance concentrates, leaving a one-bit code
little to exploit. Oversampling the shortlist is how binary search becomes usable.

To recover the exact order, oversample the shortlist and join back to the exact column:

```sql
WITH params AS (SELECT p FROM quantisation),
     query AS (SELECT quantize_vector_tinyint(:embedding, p) AS codes, p FROM params),
     shortlist AS (
         SELECT knn_agg(d.id, d.embedding_int8, q.codes, q.p, 100, 'euclidean') AS candidates
         FROM documents d CROSS JOIN query q
     )
SELECT d.id, euclidean_distance(d.embedding, :embedding) AS distance
FROM shortlist, UNNEST(candidates) AS c(id, approximate)
JOIN documents d ON d.id = c.id
ORDER BY distance
LIMIT 10;
```

The scan reads only the codes; the exact vectors are read for the shortlist alone.

### Aggregation

```sql
knn_agg(key, vector, query_vector, k, metric) -> array(row(key, distance))
```

Returns the `k` nearest neighbours of `query_vector` **per group**, nearest first. `metric` is
one of `'euclidean'`, `'euclidean_squared'`, `'cosine'`, `'dot_product'` or `'manhattan'`. `k`
is capped at 10000, and both `k` and `metric` must be constant within a group.

The quantised overloads follow the same argument order as the scalar distance functions above:

```sql
knn_agg(key, array(tinyint) vector, array(tinyint) query, bounds, k, metric)
knn_agg(key, varbinary vector, varbinary query, k, metric)
```

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

v1 implements exact KNN. Approximate search is available through quantisation: rank on int8 or
binary codes, oversample, and re-rank against the exact vectors in SQL. Index-based approximate
search is planned.

## Benchmarks

Vector search performance is tracked per pull request in [`BENCHMARKS.md`](BENCHMARKS.md).

## License

Apache License 2.0
