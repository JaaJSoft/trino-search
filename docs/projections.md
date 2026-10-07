# Projection columns and data skipping

Table formats such as Iceberg skip files on per-column `min`/`max` statistics, and an
`array(real)` column has none they can use. Projecting each vector onto a few fixed directions
and storing the results as ordinary `double` columns gives the table something to skip on, and
the search stays exact.

This needs no index, no centroids and no state in the plugin: the directions are an ordinary
table, the projections ordinary columns and the pruning an ordinary range predicate.

## vector_projections

```sql
vector_projections(array(double), array(array(double))) -> array(double)
vector_projections(array(real),   array(array(real)))   -> array(double)
```

Element `j` of the result is `dot_product(vector, directions[j])`. Projecting onto N directions
this way is one call rather than N `dot_product` calls, which matters on the write path where it
runs once per row. Directions are used as given: the function does not normalise them, and the
bound below needs unit directions.

| Situation | Behaviour |
| --- | --- |
| a `NULL` argument | `NULL` |
| `vector` contains a `NULL` element | `NULL` |
| a direction is `NULL` or contains a `NULL` element | `NULL` at that position; the others keep theirs |
| no directions | an empty array |
| a direction whose dimension differs from `vector` | error |

## Why a range predicate is exact

For a unit vector `v`, Cauchy-Schwarz gives `|x.v - q.v| <= ||x - q||`. If every neighbour wanted
is within `R` of the query `q`, every one of them has `x.v` in `[q.v - R, q.v + R]`, which is a
range predicate on a scalar column. That holds for any unit direction: the choice of directions
only decides how much the range prunes, never whether the result is right.

On orthonormal directions, Bessel's inequality adds a second bound: the euclidean distance between
two vectors' projections never exceeds the distance between the vectors. That is what lets a
radius be estimated from the projection columns alone, below.

## Choosing the directions

The directions decide everything about how much is pruned, and the obvious choices prune nothing.

- **Raw components.** Storing a few dimensions as columns costs nothing to compute and skips
  nothing. For normalised embeddings at dimension 768 a component has a standard deviation of
  about 0.036, so the difference between a row's and the query's is about 0.051, while the radius
  of a neighbour at cosine similarity 0.9 is 0.45. The range is nine standard deviations wide.
  `R` bounds the whole difference and a component only sees `1/d` of it, so this gets worse with
  the dimension.
- **Random directions.** No better than a raw component, for the same reason.
- **Leading principal components.** A leading principal component carries 5 to 15 percent of the
  variance instead of `1/768` of it. With four components carrying 15, 8, 5 and 4 percent, the
  four ranges keep about 59, 74, 84 and 89 percent of the rows, roughly a factor of 3 combined.

These are orders of magnitude for typical embedding data, not measurements: the explained
variance of the actual corpus is the number that decides whether projection columns are worth
adding. Four components carrying 30 percent of the variance give the factor of 3 above; four
carrying 3 percent give nothing.

### Fitting

Fit the directions once, on a sample. The eigendecomposition is not something SQL does, and for
something computed once there is no reason to try:

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.asarray(sample, dtype=np.float32)           # (1e6, 768)
pca = PCA(n_components=4, svd_solver="randomized").fit(X)

V = pca.components_                                 # (4, 768), orthonormal
print(pca.explained_variance_ratio_)                # the number that decides
```

`PCA` centres the data to find the directions, and that matters: without centring the first
component points along the mean, which carries magnitude but little variance. The stored
projections need no centring, since the bound is unaffected by a translation applied to both
sides.

The directions go in a table, one row each, rather than in a literal of 3072 floats:

```sql
CREATE TABLE projections (idx integer, direction array(real));
```

The directions never need refitting as the table grows: a stale direction prunes less, it never
returns a wrong result.

## Writing the projection columns

```sql
CREATE TABLE embeddings (
    id bigint,
    embedding array(real),
    pc_1 double, pc_2 double, pc_3 double, pc_4 double)
WITH (sorted_by = ARRAY['pc_1', 'pc_2']);

INSERT INTO embeddings
SELECT id, embedding, p[1], p[2], p[3], p[4]
FROM (SELECT id, embedding,
             vector_projections(embedding,
                 (SELECT array_agg(direction ORDER BY idx) FROM projections)) AS p
      FROM staging);
```

`array_agg(direction ORDER BY idx)` is how the directions must be gathered everywhere, on write
and on query alike: column `pc_j` holds the projection onto the `j`-th direction in that order.
The subquery reaches each row through a join against a single row, so the directions are not
serialised into the plan.

## Exact search

A range predicate needs a radius before it can prune. A guessed radius proves nothing on its own:
a box that is too narrow can still hold more than `k` rows while having excluded a true neighbour,
so counting rows cannot tell a wrong result from a right one. What makes the search exact is
checking the result against the radius it was computed with.

Take `:p1` to `:p4` as the query's projections, from
`vector_projections(:q, (SELECT array_agg(direction ORDER BY idx) FROM projections))`.

```sql
-- 1. A radius guess, from the four scalar columns alone. On orthonormal directions the
--    projected distance is a lower bound on the true one, so it ranks candidates honestly.
--    No vector is read: at a billion rows that is 32 GB rather than 3 TB.
SELECT sqrt(pow(pc_1 - :p1, 2) + pow(pc_2 - :p2, 2)
          + pow(pc_3 - :p3, 2) + pow(pc_4 - :p4, 2)) AS rho
FROM embeddings
ORDER BY rho
OFFSET 9999 LIMIT 1;

-- 2. One scan under a box predicate, which is what the file statistics can prune on.
SELECT n.id, n.distance
FROM (SELECT knn_agg(id, embedding, :q, 10, 'euclidean') AS top
      FROM embeddings
      WHERE pc_1 BETWEEN :p1 - :rho AND :p1 + :rho
        AND pc_2 BETWEEN :p2 - :rho AND :p2 + :rho
        AND pc_3 BETWEEN :p3 - :rho AND :p3 + :rho
        AND pc_4 BETWEEN :p4 - :rho AND :p4 + :rho) t
CROSS JOIN UNNEST(t.top) AS n(id, distance);

-- 3. If 10 rows came back and the 10th distance is at most rho, they are the true top 10.
--    Otherwise widen rho, doubling it for instance, and run step 2 again.
```

Why step 3 is a proof: a row `y` outside the box has `|y.v_j - q.v_j| > rho` for some direction
`j`, hence `||y - q|| > rho`. If the k-th distance found is at most `rho`, no row outside the box
can be closer than it, so nothing was missed. The converse holds too: once `rho` reaches the true
k-th distance, the box contains every true neighbour and the check passes. A poor first guess
costs a retry, never a wrong answer.

Two details are load-bearing:

- **Fewer than `k` rows fails the check.** It means the box held fewer than `k` rows, which says
  nothing about the rows outside it.
- **The bound is on euclidean distance.** On normalised vectors cosine distance is
  `||x - q||^2 / 2`, so the same search answers `'cosine'` queries with the check in step 3 read
  as `distance <= rho * rho / 2`. `'dot_product'` on vectors of varying magnitude has no such
  bound.

The box is a superset of the sphere step 1 ranks on. That is deliberate: file statistics are
per-column ranges, so a sphere cannot be pushed down to them and the box is what survives into the
file filter.

`OFFSET 9999` takes the thousand-k-th smallest projected distance. A larger offset reads more rows
in step 2 and retries less often, a smaller one the reverse; how good a first guess it is depends
on how much of the corpus' variance the directions capture. Step 2's result is a usable
approximate top `k` whether or not it passes, and step 3 is what turns it into a proven one.

Any `k` rows of the table give a radius that cannot be too small: their k-th distance to the
query is at least the true one, so a box of that radius always passes the check. Where an
[IVF index](ivf.md) exists, the k-th distance found by probing a few clusters is such a radius,
and a tight one, but nothing here depends on it.

## Iceberg details

All three of these fail silently: the query stays correct and reads everything.

- **Metrics are only collected for the first hundred columns.** Iceberg infers column bounds by
  default for the first hundred columns of a table (`write.metadata.metrics.max-inferred-column-defaults`).
  Projection columns beyond that get no bounds and prune nothing. Declare them early in the schema,
  or set `write.metadata.metrics.column.pc_1 = 'full'` and the like for each of them.
- **Keep the columns flat.** A `struct` of projections keeps its statistics, since every leaf has
  its own field id and bounds, but pushing a predicate on a subfield all the way down to file
  pruning is not something to assume. Flat columns are safe. `EXPLAIN ANALYZE` of step 2 shows how
  many rows and files were actually read, and is the check either way.
- **Appends erode the ordering.** File pruning works when files cover narrow, mostly disjoint
  ranges of `pc_1`. `sorted_by` orders the rows within each file Trino writes, which helps row
  groups inside a file, but every append writes files whose ranges overlap the existing ones, and
  once the ranges overlap their bounds prune nothing. The table needs a periodic rewrite that sorts
  across files, such as Spark's `rewrite_data_files` with the `sort` strategy. This is the one real
  maintenance cost here: the directions never need refitting, but the file layout does, and unlike
  an IVF table's `cluster_id`, which partitioning routes naturally, there is no layout that does not
  degrade under appends.

## What to expect

A factor of about 3 on a linear scan with good directions, against roughly 60 for the
[IVF](ivf.md) configuration described there. What projection columns add is exactness with a proof, through the check in step
3, and independence: the three steps need the four columns and nothing else.
