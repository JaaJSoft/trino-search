# Vectors and distance metrics

Vectors are ordinary Trino values. There is no custom type and no `CAST` to a vector type: a
vector is an `array(double)`, an `array(real)`, an `array(tinyint)` for one signed byte per
component, or a `varbinary` for one bit per component.

| Representation | Component | Payload at dimension 768 | Documented in |
| --- | --- | --- | --- |
| `array(double)` | 64-bit float | 6144 bytes | this page |
| `array(real)` | 32-bit float | 3072 bytes | this page |
| `array(tinyint)` | signed byte code | 768 bytes | [quantization.md](quantization.md) |
| `varbinary` | one bit | 100 bytes | [quantization.md](quantization.md) |

The binary payload is a four-byte header plus `ceil(768 / 8)` packed bytes.

## Metrics

| Function | `array(double)` | `array(real)` | Description |
| --- | --- | --- | --- |
| `euclidean_distance(x, y)` | Trino | plugin | L2 distance |
| `euclidean_squared_distance(x, y)` | plugin | plugin | L2 distance without the `sqrt` |
| `manhattan_distance(x, y)` | plugin | plugin | L1 distance |
| `dot_product(x, y)` | Trino | plugin | inner product |
| `cosine_similarity(x, y)` | Trino | plugin | cosine of the angle |
| `cosine_distance(x, y)` | Trino | plugin | `1 - cosine_similarity` |
| `l2_norm(x)` | plugin | plugin | euclidean norm |
| `normalize_vector(x)` | plugin | plugin | unit-norm vector, same type as the input |

All of them return `double`. The quantised representations carry their own overloads of the same
names, described in [quantization.md](quantization.md).

## Why the `array(real)` overloads exist

Trino already ships `euclidean_distance`, `dot_product`, `cosine_similarity` and
`cosine_distance`, but only on `array(double)`. This plugin adds the `array(real)` overloads under
the same names, so an `array(real)` column does not need a `CAST` that would double the memory
read per row.

One consequence is worth knowing before it surprises you: an untyped decimal literal such as
`ARRAY[0.1, 0.2]` binds to the `array(real)` overload, with the matching precision and `NULL`
handling. A genuinely typed `array(double)` column, an explicit `CAST` or `DOUBLE 'x'` literals
all keep the engine's native implementation.

```sql
SELECT euclidean_distance(ARRAY[0.1, 0.2], ARRAY[0.3, 0.4]);                        -- real overload
SELECT euclidean_distance(CAST(ARRAY[0.1, 0.2] AS array(double)), ARRAY[0.3, 0.4]); -- native double
```

## Normalised vectors

A unit-norm vector has magnitude 1, so `cosine_similarity(x, y)` is `dot_product(x, y)` and
`cosine_distance(x, y)` is `1 - dot_product(x, y)`. Normalising once at write time with
`normalize_vector` and ranking on `'dot_product'` afterwards therefore returns the same neighbours
as `'cosine'`, on a metric that needs no magnitudes at all:

```sql
CREATE TABLE documents AS SELECT id, category, normalize_vector(embedding) AS embedding FROM raw;

SELECT category, knn_agg(id, embedding, normalize_vector(ARRAY[0.1, 0.2, 0.3]), 10, 'dot_product')
FROM documents
GROUP BY category;
```

Normalising the stored vectors is what makes the ranking identical. Normalising the query vector
too is what makes the value that comes back the cosine similarity itself rather than a fixed
multiple of it, and it is what lets `'euclidean'` rank identically as well, since the squared
distance between two unit-norm vectors is `2 - 2 * dot_product`.

`normalize_vector` raises `Vector magnitude cannot be zero` on the zero vector, exactly where
cosine would: a row with nothing to normalise has to be filtered out either way. The text
embedding functions already return unit-norm vectors, with the zero-vector exception described in
[embeddings.md](embeddings.md).

## Null and dimension handling

| Input | Result |
| --- | --- |
| a `NULL` argument | `NULL`, without evaluating the function |
| an array containing a `NULL` element | `NULL` |
| two vectors of different lengths | error, `The arguments must have the same length` |

`knn_agg` treats nulls differently, because a null row must not fail a scan; see
[knn.md](knn.md).
