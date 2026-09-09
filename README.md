# Trino-search

[Trino](https://trino.io) plugin providing search functions as SQL functions. The first family
covers vectors: distance metrics, normalisation, text embeddings, quantisation and exact
k-nearest-neighbour search.

The plugin exposes functions only, no catalog and no connector: dropping the JAR into the plugin
directory makes the functions available globally. Vectors are ordinary Trino values, so there is
no custom type and no `CAST` to learn.

## Installation

Requires Trino 483 and Java 25.

```bash
./mvnw clean package
```

Copy the contents of `target/trino-search-<version>/` into `<trino>/plugin/search/`, then restart
the server.

## Quick start

```sql
-- global top 10 over an array(real) column
SELECT id, euclidean_distance(embedding, ARRAY[REAL '0.1', REAL '0.2', REAL '0.3']) AS distance
FROM documents
ORDER BY distance
LIMIT 10;

-- top 3 per category, in one pass
SELECT category, knn_agg(id, embedding, ARRAY[0.1, 0.2, 0.3], 3, 'cosine') AS neighbours
FROM documents
GROUP BY category;
```

## Documentation

| Page | Covers |
| --- | --- |
| [Vectors and distance metrics](docs/vectors.md) | representations, the metric functions, normalisation, null and dimension handling |
| [k-nearest-neighbour search](docs/knn.md) | `knn_agg`, its overloads, constraints and edge cases |
| [Quantisation and approximate search](docs/quantization.md) | int8 and binary codes, fitting bounds, recall, oversample and re-rank |
| [Text embeddings](docs/embeddings.md) | `to_vector_*`, feature hashing and its limits |
| [Benchmarks](BENCHMARKS.md) | recorded measurements and how to read them |

## Function index

| Function | Page |
| --- | --- |
| `euclidean_distance`, `euclidean_squared_distance`, `manhattan_distance` | [vectors](docs/vectors.md), [quantisation](docs/quantization.md) |
| `dot_product`, `cosine_similarity`, `cosine_distance` | [vectors](docs/vectors.md), [quantisation](docs/quantization.md) |
| `l2_norm`, `normalize_vector` | [vectors](docs/vectors.md) |
| `knn_agg` | [knn](docs/knn.md) |
| `vector_bounds_agg`, `quantize_vector_tinyint`, `quantize_vector_varbinary`, `hamming_distance` | [quantisation](docs/quantization.md) |
| `to_vector_real`, `to_vector_double` | [embeddings](docs/embeddings.md) |

## Status

v1 implements exact KNN. Approximate search is available through quantisation: rank on int8 or
binary codes, oversample, and re-rank against the exact vectors in SQL. Index-based approximate
search is planned.

## License

Apache License 2.0
