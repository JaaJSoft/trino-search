# Graph search (HNSW)

An HNSW graph (Malkov and Yashunin, 2016) answers a nearest-neighbour query by walking a few
thousand vectors instead of comparing the query against all of them. Building one is expensive,
so it has to be built once and kept, and the plugin keeps nothing between queries. The graph is
therefore an ordinary `varbinary` value: an aggregation builds it, a table you own stores it, and
a scalar function searches it. The engine's own partition pruning picks which graphs a query
reads.

## Functions

| Function | Description |
| --- | --- |
| `hnsw_build_agg(key, vector, [bounds,] m, ef_construction, metric)` | builds one graph over the vectors of each group |
| `hnsw_search(graph, query_vector, k, ef_search)` | the approximate `k` nearest neighbours of `query_vector` in one graph |

## hnsw_build_agg

```sql
hnsw_build_agg(key bigint, vector array(double),  m bigint, ef_construction bigint, metric varchar) -> varbinary
hnsw_build_agg(key bigint, vector array(real),    m bigint, ef_construction bigint, metric varchar) -> varbinary
hnsw_build_agg(key bigint, vector array(tinyint), bounds, m bigint, ef_construction bigint, metric varchar) -> varbinary
hnsw_build_agg(key bigint, vector varbinary,      m bigint, ef_construction bigint, metric varchar) -> varbinary
```

The last two build over quantised codes, see [Graphs over quantised codes](#graphs-over-quantised-codes).

- `key` identifies the row and is what a search returns. It is a `bigint` because the graph is a
  plain value with no type attached: join the result back to the base table for anything else.
- `m` is the number of neighbours each node is linked to, `2 * m` on the bottom layer. It is the
  main recall and size knob; 16 is the usual default, 32 or more for high-dimensional data that
  needs very high recall.
- `ef_construction` is the length of the candidate list while inserting. Higher builds a better
  graph, slower; 100 to 200 is usual.
- `metric` takes the same names as [`knn_agg`](knn.md#metrics) and is stored in the graph, so a
  search ranks by the metric the graph was built for.

The vectors are stored in the graph in the representation they arrived in: eight bytes per
component for `array(double)`, four for `array(real)`, one for `array(tinyint)` codes, and one bit
for binary codes.

| Constraint | Message |
| --- | --- |
| `m` between 2 and 256 | `m must be between 2 and 256` |
| `ef_construction` between `m` and 10000 | `ef_construction must be between m (...) and 10000` |
| `m`, `ef_construction`, `metric` and `bounds` constant within a group | `... must be constant within a group of hnsw_build_agg` |
| every vector of a group has the same dimension | `The vectors of hnsw_build_agg must have the same length` |

| Situation | Behaviour |
| --- | --- |
| the group is empty | `NULL` |
| a row whose key or vector is `NULL` | the row is ignored |
| a row whose vector contains a `NULL` element | the row is ignored |
| every row of the group was ignored | `NULL` |

The graph is a function of the rows alone. Rows are inserted in an order derived from a hash of
their key, and each node's layer comes from another hash of the same key, so two builds over the
same rows with distinct keys produce identical bytes however the engine split and ordered them.

## hnsw_search

```sql
hnsw_search(graph varbinary, query_vector array(double),  k bigint, ef_search bigint) -> array(row(key bigint, distance double))
hnsw_search(graph varbinary, query_vector array(real),    k bigint, ef_search bigint) -> array(row(key bigint, distance double))
hnsw_search(graph varbinary, query_vector array(tinyint), k bigint, ef_search bigint) -> array(row(key bigint, distance double))
hnsw_search(graph varbinary, query_vector varbinary,      k bigint, ef_search bigint) -> array(row(key bigint, distance double))
```

Returns up to `k` neighbours, nearest first, with the raw metric value as `knn_agg` reports it:
for `'dot_product'` the highest similarity comes first. `ef_search` is the length of the candidate
list and the recall knob of a search: higher finds more of the true neighbours, slower. It must
be at least `k`; 64 to 128 for `k = 10` is a usual range.

The query has to be in the representation of the graph, with one exception: the two float types
are interchangeable, since converting between them is a `CAST`. Against a graph of `array(real)`
vectors an `array(double)` query is rounded to `real` exactly as a `CAST` would, and the
distances are the ones the `array(real)` distance functions return.

| Situation | Behaviour |
| --- | --- |
| `graph` is `NULL` | `NULL` |
| `query_vector` contains a `NULL` element | `NULL` |
| `k` larger than the graph | every node of the graph is returned |
| `k` not between 1 and 10000 | error |
| `ef_search` not between `k` and 10000 | error |
| `query_vector` of another dimension than the graph | error |
| `query_vector` of another representation than the graph, floats aside | `The graph was built from ... vectors and cannot be searched with a ... query` |
| `graph` is not a value built by `hnsw_build_agg` | `The value is not a graph built by hnsw_build_agg` |

## One graph per partition

The base table is partitioned, and a side table holds one graph per partition. Day is the
granularity used here; see [Sizing](#sizing) for why production may need finer.

```sql
CREATE TABLE document_graphs (
    day date,
    graph varbinary
)
WITH (partitioning = ARRAY['day']);

INSERT INTO document_graphs
SELECT day, hnsw_build_agg(id, embedding, 16, 200, 'euclidean')
FROM documents
WHERE day = DATE '2026-08-09'
GROUP BY day;
```

A search reads the graphs of the partitions it is restricted to, and merges their answers with an
ordinary `ORDER BY`:

```sql
SELECT n.key AS id, n.distance
FROM document_graphs
CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[...], 10, 64)) AS n(key, distance)
WHERE day BETWEEN DATE '2026-08-01' AND DATE '2026-08-09'
ORDER BY n.distance
LIMIT 10;
```

`hnsw_search` runs once per selected row of `document_graphs`, so once per partition, never once
per document. With `'dot_product'` the merge is `ORDER BY n.distance DESC`.

A partition that is still receiving rows is not worth a graph yet: rows written after the build
are not in it. Build a day's graph once the day is closed, and search the open day exactly:

```sql
WITH candidates AS (
    SELECT n.key AS id, n.distance
    FROM document_graphs
    CROSS JOIN UNNEST(hnsw_search(graph, ARRAY[...], 10, 64)) AS n(key, distance)
    WHERE day BETWEEN DATE '2026-08-01' AND DATE '2026-08-09'
    UNION ALL
    SELECT n.id, n.distance
    FROM (SELECT knn_agg(id, embedding, ARRAY[...], 10, 'euclidean') AS ns FROM documents WHERE day = DATE '2026-08-10')
    CROSS JOIN UNNEST(ns) AS n(id, distance)
)
SELECT id, distance FROM candidates ORDER BY distance LIMIT 10;
```

A closed partition never changes, so neither does its graph: a new day appends one row to
`document_graphs` and touches nothing that already exists. Rebuilding a partition's graph, after
a backfill for instance, is the same `INSERT` after deleting its row.

## Graphs over quantised codes

A graph holds every vector of its partition, because a search computes distances against the
nodes it visits and a function cannot fetch rows from another table in the middle of a search.
Building over [quantised codes](quantization.md) instead of floats is what makes the graph much
smaller than the column it indexes, at the price of ranking on codes: an oversampled shortlist is
then re-ranked on the exact vectors.

```sql
CREATE TABLE quantisation AS SELECT vector_bounds_agg(embedding) AS bounds FROM documents;

INSERT INTO document_graphs
SELECT d.day, hnsw_build_agg(d.id, quantize_vector_tinyint(d.embedding, q.bounds), q.bounds, 16, 200, 'euclidean')
FROM documents d
CROSS JOIN quantisation q
WHERE d.day = DATE '2026-08-09'
GROUP BY d.day;
```

The bounds are stored in the graph, once, so `hnsw_search` does not take them; the query has to
be quantised against the same bounds, which nothing can check. A binary graph is built the same
way from `quantize_vector_varbinary(embedding, bounds)`, and needs no bounds at all.

```sql
WITH shortlist AS (
    SELECT n.key AS id
    FROM document_graphs
    CROSS JOIN quantisation q
    CROSS JOIN UNNEST(hnsw_search(graph, quantize_vector_tinyint(ARRAY[...], q.bounds), 40, 64)) AS n(key, distance)
    WHERE day BETWEEN DATE '2026-08-01' AND DATE '2026-08-09'
)
SELECT d.id, euclidean_distance(d.embedding, ARRAY[...]) AS distance
FROM shortlist s
JOIN documents d ON d.id = s.id
WHERE d.day BETWEEN DATE '2026-08-01' AND DATE '2026-08-09'
ORDER BY distance
LIMIT 10;
```

The distances `hnsw_search` returns are the quantised ones, the values the quantised distance
functions give for the same codes and bounds; only the re-ranking produces exact distances.
The re-ranking join reads the exact vectors of the shortlisted rows, and how much of the vector
column dynamic filtering lets it skip depends on the layout of the base table: sorted by key
within a partition, Parquet statistics can skip most row groups.

As with the scalar distance functions, the bounds argument of the `array(tinyint)` build is not
optional in practice: without it, `array(tinyint)` coerces to a float array and the call binds to
a float overload that builds a graph over the raw codes, with no scale applied.

## Sizing

**Build memory.** `hnsw_build_agg` holds every vector of its group in the aggregation state and
reports that to Trino, so a partition too large for the memory limits fails the query with a
memory error rather than exhausting the worker. The graph is built when the group is output,
which needs about the same amount again for the links and the serialised value, briefly and
outside what the state reports.

**Value size.** The graph holds every vector of the partition plus its links and bookkeeping,
up to about 150 bytes per vector at `m = 16`, and it is a single `varbinary`, which cannot exceed
2 GB. The build fails with a message saying so beyond it. At dimension 768 and `m = 16`:

| Representation | Vector | Graph, per vector | Vectors per graph, at most |
| --- | --- | --- | --- |
| `array(real)` | 3072 bytes | about 3.2 kB | about 650,000 |
| `array(tinyint)` | 768 bytes | about 0.9 kB | about 2,300,000 |
| binary | 100 bytes | about 0.25 kB | about 8,000,000 |

Below the int8 row the links outweigh the vectors, so a binary graph is far from 32 times smaller
than a float one, and its many tied distances leave the neighbour selection less to prune.

**Search cost.** A search computes a few thousand distances whatever the size of the graph,
instead of one per row. Over float vectors it does not reduce the bytes read: the graph contains
every vector of its partition, so reading it costs at least what scanning that partition's vector
column would, and what HNSW saves is computation and the engine's per-row work. Over codes it
saves I/O too: an int8 graph is under a third of an `array(real)` column, a binary one under a
tenth, plus the exact vectors of the shortlist read by the re-ranking.

**Granularity.** Search effort grows roughly with the logarithm of a graph's size, so splitting a
partition into `N` smaller graphs turns one search of `log(n)` into `N` searches of `log(n / N)`:
more total work, in exchange for smaller builds and smaller values. Partition no finer than build
memory and the value size require. A query over a long range of small partitions searches every
one of them; compacting old partitions into coarser graphs, a month built from its days for
instance, bounds that fan-out without ever rebuilding everything.

## Recall

Recall is measured by `TestHnswRecall` against brute force, on 2000 vectors of dimension 32, with
`m = 16` and `ef_construction = 100`:

| Data | Metric | `ef_search = k` | `ef_search = 64` |
| --- | --- | --- | --- |
| clustered | euclidean | 0.96 | 1.00 |
| clustered | cosine | 0.965 | 1.00 |
| uniform | euclidean | 0.79 | 1.00 |
| uniform | cosine | 0.76 | 0.99 |

Uniform data in 32 dimensions is the hard case, with no structure for a graph to exploit. The
numbers are a check that the implementation behaves, not a prediction for a real corpus: measure
recall on your own data by comparing `hnsw_search` against `knn_agg` over a sample of queries.

Over codes, recall is the share of the true `k = 10` nearest float vectors that the shortlist
contains, which is what re-ranking recovers, with euclidean distance:

| Codes | Data | Shortlist | `ef_search` | Recall |
| --- | --- | --- | --- | --- |
| int8 | clustered | 10 | 64 | 0.955 |
| int8 | uniform | 10 | 64 | 0.995 |
| int8 | uniform | 20 | 64 | 1.00 |
| binary | clustered | 100 | 200 | 1.00 |
| binary | uniform | 100 | 200 | 0.72 |
| binary | uniform | 200 | 400 | 0.895 |

In these measurements the graph costs nothing on top of the quantisation: a search whose
candidate list covers the whole graph, and so reads every code, finds the same shortlist. The
loss is the codes', as in the exact scan over them that [quantization.md](quantization.md#recall)
measures.

## Limitations

- **`'dot_product'` needs unit-norm vectors.** Over vectors of different norms a dot product is
  not a distance: a long vector scores higher against almost everything than a short one, so the
  graph prunes the short ones out of every neighbour list and no search can reach them. Normalise
  the vectors with `normalize_vector`, as [vectors.md](vectors.md#normalised-vectors) recommends,
  or use `'cosine'`.
- **Many exact duplicates.** Copies of one vector cannot be told apart by distance, and once the
  neighbour lists around them are full a later copy may be left with no incoming link. A vector
  repeated a few times is harmless; one repeated more than `2 * m` times can lose copies, so
  deduplicate before building.
- **Zero vectors under `'cosine'`** raise `Vector magnitude cannot be zero` as soon as a distance
  to one is computed, during the build or a search, as they do everywhere else.
