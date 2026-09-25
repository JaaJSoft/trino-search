# Clustering and IVF search

An inverted-file (IVF) index splits a corpus into clusters around fitted centroids, stores each
row's cluster id next to it, and answers a query by reading only the few clusters whose centroids
are nearest to the query. With the table partitioned by cluster id, partition pruning is what
skips the rest. The plugin keeps nothing between queries: the centroids are an ordinary table and
the cluster id an ordinary column.

At a billion rows of dimension 768 stored as `array(real)`, a full scan reads about 3 TB. Probing
16 clusters out of 1024 reads about 48 GB.

## Functions

| Function | Description |
| --- | --- |
| `nearest_vector(x, candidates, metric)` | the position of the candidate nearest to `x`, and its distance |
| `vector_avg_agg(x)` | the element-wise mean of the vectors in each group |

Both accept `array(double)` or `array(real)`. Neither is specific to IVF: `nearest_vector` is any
lookup of a vector against a small codebook, and `vector_avg_agg` is how chunk embeddings are
pooled into a document embedding, or a user's history into a user embedding.

## nearest_vector

```sql
nearest_vector(vector, candidates, metric) -> row(position integer, distance double)
```

`candidates` is an array of vectors of the same representation as `vector`. `position` is 1-based,
so it is the index `element_at` takes. `distance` is the raw metric value, as `knn_agg` reports it.
`metric` takes the same names as [`knn_agg`](knn.md#metrics), and `'dot_product'` and `'cosine'`
rank a higher similarity as closer.

```sql
SELECT nearest_vector(embedding, ARRAY[ARRAY[REAL '0', REAL '1'], ARRAY[REAL '1', REAL '0']], 'cosine').position
FROM documents;
```

Equidistant candidates resolve to the lowest position, so an assignment is reproducible from one
run to the next.

| Situation | Behaviour |
| --- | --- |
| `vector` contains a `NULL` element | `NULL` |
| a candidate is `NULL` or contains a `NULL` element | the candidate is skipped; the others keep their positions |
| no usable candidate, including an empty array | `NULL` |
| a candidate whose dimension differs from `vector` | error |
| an unknown metric name | error, as for `knn_agg` |

## vector_avg_agg

```sql
vector_avg_agg(array(double)) -> array(double)
vector_avg_agg(array(real))   -> array(real)
```

The result has the input's element type. Accumulation is in `double` for both, since a `float`
running sum stops registering small components long before a million rows.

| Situation | Behaviour |
| --- | --- |
| the group is empty | `NULL` |
| a row whose vector is `NULL` | the row is ignored |
| a row whose vector contains a `NULL` element | the row is ignored |
| every row of the group was ignored | `NULL` |
| vectors of different dimensions in one group | error |

Skipping the whole row rather than only its `NULL` component keeps the result the mean of one set
of vectors. Averaging each dimension over the rows that happen to have it would not be.

The state of a group is one `double` per dimension plus a count, about 6 kB at dimension 768. That
is negligible for the few hundred groups of a centroid fit, and it is the whole memory cost of a
`GROUP BY` over millions of groups.

## Building an index

The walkthrough below uses Iceberg, whose partitioning does the pruning, and euclidean distance.
Any metric works, as long as the same one is used to fit, to assign and to probe: a row assigned
under one metric and probed under another lands in a cluster the probe never reads.

### Fit the centroids

Lloyd's algorithm alternates two steps, assigning every row to its nearest centroid and moving
every centroid to the mean of its rows. Each step is one statement. Fit on a sample: a million
rows is plenty for 1024 centroids.

Initialise with random rows. k-means++ needs one pass per centroid, too many at this size, and the
centroids only have to partition the space, not be optimal.

```sql
CREATE TABLE centroids AS
SELECT CAST(row_number() OVER () AS integer) AS cluster_id, embedding AS centroid
FROM (SELECT embedding FROM sample ORDER BY rand() LIMIT 1024);
```

The ids are numbered after sampling, not before: numbering the whole sample and keeping 1024 rows
of it would leave gaps.

Then run one iteration, about fifteen times. Each writes the next centroids into a new table and
swaps it in:

```sql
CREATE TABLE centroids_next AS
WITH codebook AS (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids),
assigned AS (
    SELECT nearest_vector(s.embedding, b.centroids, 'euclidean').position AS cluster_id, s.embedding
    FROM sample s CROSS JOIN codebook b
),
means AS (SELECT cluster_id, vector_avg_agg(embedding) AS centroid FROM assigned GROUP BY cluster_id)
SELECT c.cluster_id, coalesce(m.centroid, c.centroid) AS centroid
FROM centroids c
LEFT JOIN means m ON m.cluster_id = c.cluster_id;

DROP TABLE centroids;
ALTER TABLE centroids_next RENAME TO centroids;
```

Three details in that statement are load-bearing:

- **`cluster_id` must be dense from 1**, and `array_agg(centroid ORDER BY cluster_id)` must be the
  only way the codebook is ever built. `nearest_vector` returns a position in that array, and the
  position is the cluster id only because of both.
- **The `LEFT JOIN` from `centroids` keeps a cluster that received no row.** Grouping the assigned
  rows alone would drop it, the next codebook would be one entry shorter, and every position after
  it would silently point at the wrong cluster. With the join, an empty cluster keeps its old
  centroid.
- **The codebook reaches the rows through a join against a single row**, not as a literal in the
  plan, so 1024 centroids of dimension 768 cost one 3 MB value per worker rather than a 3 MB plan.

Each iteration reads the sample once and computes one distance per row and centroid, so a fit is
cheap next to a single full scan of the corpus.

### Assign every row

```sql
CREATE TABLE embeddings
WITH (partitioning = ARRAY['cluster_id'])
AS
SELECT d.id, d.embedding, nearest_vector(d.embedding, b.centroids, 'euclidean').position AS cluster_id
FROM documents d
CROSS JOIN (SELECT array_agg(centroid ORDER BY cluster_id) AS centroids FROM centroids) b;
```

New rows are assigned the same way when they are inserted. This replaces a
`CROSS JOIN centroids`, which would materialise 1024 rows per inserted row to keep one.

### Probe

Probing is `knn_agg` over the centroid table, then `knn_agg` over the rows of the clusters it
returned:

```sql
WITH probed AS (
    SELECT p.cluster_id
    FROM (SELECT knn_agg(cluster_id, centroid, ARRAY[...], 16, 'euclidean') AS ps FROM centroids)
    CROSS JOIN UNNEST(ps) AS p(cluster_id, distance)
)
SELECT n.id, n.distance
FROM (
    SELECT knn_agg(id, embedding, ARRAY[...], 10, 'euclidean') AS ns
    FROM embeddings
    WHERE cluster_id IN (SELECT cluster_id FROM probed)
)
CROSS JOIN UNNEST(ns) AS n(id, distance);
```

The number of clusters probed is the recall knob. Probing more can only find more of the true
neighbours: every cluster read by a smaller probe is read by a larger one, and probing every
cluster returns exactly the unrestricted `knn_agg`. `EXPLAIN ANALYZE` shows whether the scan of
`embeddings` read only the probed partitions.

## Maintenance

Centroids drift from the data as it grows, but an IVF table rarely needs a global refit. The two
situations that do arise are both local:

- **An oversized cluster.** Fit a few centroids on that cluster's rows alone, with the same Lloyd
  statement restricted by `WHERE cluster_id = ...`. Replace the cluster's centroid with one of them
  and append the others with ids after the current maximum, then reassign that cluster's rows only.
  One partition is rewritten.
- **A new region of the space.** Append centroids with fresh ids. Every existing row keeps its
  assignment and nothing is rewritten.

Rows in other clusters are not reassigned, even when a new centroid is now nearer to them. That
costs some recall at cluster edges and never correctness: every distance returned is still exact.
Appending keeps the ids dense, which is what the codebook needs.

Oversized clusters show up in the table's metadata, without reading any data:

```sql
SELECT partition.cluster_id, record_count
FROM "embeddings$partitions"
ORDER BY record_count DESC;
```
