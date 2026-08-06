# trino-search

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

## License

Apache License 2.0
