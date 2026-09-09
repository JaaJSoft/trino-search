# k-nearest-neighbour search

```sql
knn_agg(key, vector, query_vector, k, metric) -> array(row(key, distance))
```

Returns the `k` nearest neighbours of `query_vector` **within each group**, nearest first. `key`
is whatever identifies the row: any Trino type, including `array`, `row` and `map`. Each element
of the result is a two-field row holding that key and the raw metric value.

```sql
-- three nearest documents per category
SELECT category, knn_agg(id, embedding, ARRAY[0.1, 0.2, 0.3], 3, 'cosine') AS neighbours
FROM documents
GROUP BY category;

-- flattened back to rows
SELECT category, n.id, n.distance
FROM (
    SELECT category, knn_agg(id, embedding, ARRAY[0.1, 0.2, 0.3], 3, 'cosine') AS neighbours
    FROM documents
    GROUP BY category
) t
CROSS JOIN UNNEST(neighbours) AS n(id, distance);
```

Without a `GROUP BY` the aggregation returns the global top `k`, which is the same answer as
`ORDER BY distance LIMIT k` on a bounded amount of state: the aggregation keeps a heap of `k`
candidates per group rather than sorting the whole input.

## Metrics

`metric` is one of `'euclidean'`, `'euclidean_squared'`, `'cosine'`, `'dot_product'` or
`'manhattan'`. The name is case-insensitive. `'dot_product'` and `'cosine'` rank higher
similarity as closer; the other three rank a smaller value as closer.

An unknown name raises `Unknown metric '...'`.

## Overloads

The vector and the query vector must share a representation. The quantised overloads follow the
same argument order as the scalar distance functions:

```sql
knn_agg(key, array(double) vector,  array(double) query,  k, metric)
knn_agg(key, array(real) vector,    array(real) query,    k, metric)
knn_agg(key, array(tinyint) vector, array(tinyint) query, bounds, k, metric)
knn_agg(key, varbinary vector,      varbinary query,      k, metric)
```

The `array(tinyint)` form takes the fitted bounds as a mandatory argument; see
[quantization.md](quantization.md), which also covers oversampling and re-ranking a quantised
shortlist against the exact vectors.

## Constraints

- `k` must be greater than zero and at most 10000.
- `k` and `metric` must be constant within a group. Two different values in one group raise
  `k must be constant within a group of knn_agg` or the equivalent message for the metric. The
  same metric spelled with different capitalisation is not a change.
- The stored vectors and the query vector must have the same dimension.

## Null and edge-case behaviour

A single unusable row must not fail a scan of a corpus, so `knn_agg` skips rather than raises
where a scalar function would return `NULL`:

| Situation | Behaviour |
| --- | --- |
| the group is empty | the aggregation returns `NULL` |
| a row whose vector is `NULL` | the row is ignored |
| a row whose vector contains a `NULL` element | the row is ignored |
| the query vector contains a `NULL` element | every row is ignored, so the group returns `NULL` |
| a row whose key is `NULL` | the row is kept, with a `NULL` key in the result |
| `k` is larger than the group | every row in the group is returned |
| a vector whose dimension differs from the query vector | error |

A `NULL` element in the query vector nulling the whole group, where the same element in a single
row's vector only drops that row, is deliberate: the query vector is the same value for every row,
so nothing in the group can be ranked against it.
