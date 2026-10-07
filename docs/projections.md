# Projection columns and data skipping

A nearest-neighbour query over an Iceberg table reads every vector. Iceberg skips files using the
`min`/`max` it keeps for each column of each file, and it keeps nothing usable for an
`array(real)` column, so there is nothing to skip on.

Projection columns fix that. Each row stores a few extra `double` columns, its coordinates along a
few well-chosen directions. A filter on those columns can skip files, and the search stays exact.

## The idea

Picture the vectors in two dimensions, and store each one's `x` coordinate in a column `pc_1`.

Every point within distance `r` of a query `q` has its `x` coordinate within `r` of `q`'s. So the
filter

```sql
WHERE pc_1 BETWEEN q_x - r AND q_x + r
```

cannot drop a single neighbour within `r`. It is an ordinary range filter on a `double` column,
which is exactly what Iceberg can skip files on: any file whose `pc_1` range lies outside the
interval is never read.

The same holds in any dimension, for any unit direction `v` in place of the `x` axis:
`|x.v - q.v| <= ||x - q||`, which is the Cauchy-Schwarz inequality. `vector_projections` computes
those coordinates, `x.v`, for several directions at once.

## In short

Once:

1. **Fit the directions.** Usually four principal components of a sample, stored as rows of a
   table. This can be done [in SQL](#fitting-the-directions-in-sql).
2. **Fill the projection columns.** Every row gets `pc_1` to `pc_4`, computed with
   `vector_projections` [when it is written](#writing-the-projection-columns).

On every query:

1. **Guess a radius `rho`** from the four `double` columns alone, without reading any vector.
2. **Run `knn_agg` under a box filter** of half-width `rho` on the four columns. Iceberg skips the
   files outside the box.
3. **Check.** If `k` rows came back and the k-th is within `rho`, they are the true top k.
   Otherwise double `rho` and run step 2 again.

The check is what makes the search exact: a wrong guess costs a retry, never a wrong answer.

## vector_projections

```sql
vector_projections(array(double), array(array(double))) -> array(double)
vector_projections(array(real),   array(array(real)))   -> array(double)
```

Element `j` of the result is `dot_product(vector, directions[j])`. Projecting onto N directions
this way is one call rather than N `dot_product` calls, which matters on the write path where it
runs once per row. Directions are used as given: the function does not normalise them, and the
filter above is only safe for unit directions.

| Situation | Behaviour |
| --- | --- |
| a `NULL` argument | `NULL` |
| `vector` contains a `NULL` element | `NULL` |
| a direction is `NULL` or contains a `NULL` element | `NULL` at that position; the others keep theirs |
| no directions | an empty array |
| a direction whose dimension differs from `vector` | error |

## Choosing the directions

Any unit direction keeps the search exact. The directions decide how much is skipped, and the
obvious choices skip nothing.

- **Raw components.** Storing a few of the vector's own dimensions as columns costs nothing to
  compute and skips nothing. For normalised embeddings at dimension 768 a component has a standard
  deviation of about 0.036, so it differs between a row and the query by about 0.051, while the
  radius of a neighbour at cosine similarity 0.9 is 0.45. The interval is nine standard deviations
  wide and keeps nearly every row. The higher the dimension, the worse this gets.
- **Random directions.** No better than a raw component, for the same reason.
- **Leading principal components.** The directions along which the data varies most. A leading
  principal component carries 5 to 15 percent of the variance instead of `1/768` of it. With four
  components carrying 15, 8, 5 and 4 percent, the four intervals keep about 59, 74, 84 and 89
  percent of the rows, roughly a factor of 3 combined.

These are orders of magnitude for typical embedding data, not measurements. The explained variance
of the actual corpus decides whether projection columns are worth adding: four components carrying
30 percent of the variance give the factor of 3 above, four carrying 3 percent give nothing.

## Fitting the directions in SQL

Principal components are the leading eigenvectors of the covariance matrix, but they can be found
without an eigensolver. Subspace iteration repeats two steps: multiply the current directions by
the covariance, then make them orthonormal again. They converge onto the principal components in
order of variance. Multiplying a direction `v` by the covariance is an average over the rows,
`avg(y * (y.v))` where `y` is a row minus the mean, so a step is `vector_projections` and
`vector_avg_agg` in one statement.

Fit on a sample: a million rows is plenty. The walkthrough fits four directions from a table
`sample(id, embedding array(real))`.

### The mean

```sql
CREATE TABLE pca_mean AS
SELECT vector_avg_agg(CAST(embedding AS array(double))) AS mu
FROM sample;
```

Centring matters here, and only here. Without it the first direction converges onto the mean,
which carries magnitude but little variance. The projection columns themselves need no centring:
the filter compares a row's projection with the query's, and a shift applied to both cancels out.

### The starting directions

Any four rows will do:

```sql
CREATE TABLE pca_directions AS
SELECT CAST(row_number() OVER () AS integer) AS idx, CAST(embedding AS array(double)) AS direction
FROM (SELECT embedding FROM sample ORDER BY rand() LIMIT 4);
```

### One iteration

Repeat this statement, swapping the new table in each time:

```sql
CREATE TABLE pca_directions_next AS
WITH basis AS (SELECT array_agg(direction ORDER BY idx) AS directions FROM pca_directions),
centred AS (
    SELECT zip_with(CAST(s.embedding AS array(double)), m.mu, (x, mu) -> x - mu) AS y
    FROM sample s CROSS JOIN pca_mean m
),
projected AS (
    SELECT c.y, vector_projections(c.y, b.directions) AS p
    FROM centred c CROSS JOIN basis b
),
products AS (
    SELECT array_agg(w ORDER BY j) AS ws
    FROM (SELECT j, vector_avg_agg(transform(y, x -> x * p[j])) AS w
          FROM projected CROSS JOIN UNNEST(sequence(1, cardinality(p))) AS t(j)
          GROUP BY j)
)
SELECT CAST(idx AS integer) AS idx, direction
FROM products
CROSS JOIN UNNEST(reduce(
    ws,
    CAST(ARRAY[] AS array(array(double))),
    (done, w) -> done || normalize_vector(
        reduce(done, w, (r, v) -> zip_with(r, v, (x, e) -> x - dot_product(r, v) * e), r -> r)),
    done -> done)) WITH ORDINALITY AS u(direction, idx);

DROP TABLE pca_directions;
ALTER TABLE pca_directions_next RENAME TO pca_directions;
```

`products` is the multiplication by the covariance: one `vector_avg_agg` per direction, all in a
single scan of the sample. The `reduce` at the end is Gram-Schmidt: each new direction has its
components along the earlier ones subtracted, then is scaled to unit length. That keeps the
directions orthonormal, which the radius guess in step 1 of the search relies on, and keeps them
from all converging onto the first component.

Each iteration reads the sample once. How many are needed depends on how quickly the variance
decays past the fourth component, so measure rather than guess: stop when the explained variance
no longer grows from one iteration to the next.

### The explained variance

The fraction of the sample's variance each direction captures, which is the number that decides
whether the columns are worth adding:

```sql
WITH basis AS (SELECT array_agg(direction ORDER BY idx) AS directions FROM pca_directions),
centred AS (
    SELECT zip_with(CAST(s.embedding AS array(double)), m.mu, (x, mu) -> x - mu) AS y
    FROM sample s CROSS JOIN pca_mean m
)
SELECT j, avg(p[j] * p[j]) / avg(dot_product(y, y)) AS explained_variance_ratio
FROM (SELECT c.y, vector_projections(c.y, b.directions) AS p FROM centred c CROSS JOIN basis b)
CROSS JOIN UNNEST(sequence(1, cardinality(p))) AS t(j)
GROUP BY j
ORDER BY j;
```

### Keeping the result

```sql
CREATE TABLE projections AS
SELECT idx, CAST(direction AS array(real)) AS direction
FROM pca_directions;
```

The directions never need refitting as the table grows: stale directions skip less, they never
return a wrong result.

The fit can also be done outside Trino. scikit-learn's `PCA(n_components=4).fit(sample)` gives the
same directions in `components_` and the same ratios in `explained_variance_ratio_`, and the four
rows go into the same `projections` table.

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

A box filter needs a radius, and the right radius is the distance to the k-th neighbour, which is
what the search is looking for. So the radius is guessed, and the result is checked against it.

Compute the query's own projections first, with the same directions in the same order:

```sql
SELECT vector_projections(:q, (SELECT array_agg(direction ORDER BY idx) FROM projections));
-- gives :p1, :p2, :p3, :p4
```

### Step 1: guess a radius

```sql
SELECT sqrt(pow(pc_1 - :p1, 2) + pow(pc_2 - :p2, 2)
          + pow(pc_3 - :p3, 2) + pow(pc_4 - :p4, 2)) AS rho
FROM embeddings
ORDER BY rho
OFFSET 9999 LIMIT 1;
```

This reads the four `double` columns and no vector: at a billion rows, 32 GB rather than 3 TB. On
orthonormal directions the distance between projections is never larger than the true distance
(Bessel's inequality), so ranking rows by it is a fair first estimate.

`OFFSET 9999` takes the 1000k-th smallest value. A larger offset makes a wider box, which reads
more in step 2 and needs fewer retries; a smaller one the reverse.

### Step 2: search inside the box

```sql
SELECT n.id, n.distance
FROM (SELECT knn_agg(id, embedding, :q, 10, 'euclidean') AS top
      FROM embeddings
      WHERE pc_1 BETWEEN :p1 - :rho AND :p1 + :rho
        AND pc_2 BETWEEN :p2 - :rho AND :p2 + :rho
        AND pc_3 BETWEEN :p3 - :rho AND :p3 + :rho
        AND pc_4 BETWEEN :p4 - :rho AND :p4 + :rho) t
CROSS JOIN UNNEST(t.top) AS n(id, distance);
```

This is the scan Iceberg can prune. Its result is already a usable approximate top 10.

### Step 3: check

- **10 rows came back and the 10th distance is at most `rho`:** they are the true top 10.
- **Otherwise:** double `rho` and run step 2 again.

Why this is a proof: a row outside the box is more than `rho` away from the query along at least
one direction, so it is more than `rho` away from the query. If the 10th row found is within
`rho`, no row outside the box can beat it, and nothing was missed. And once `rho` reaches the
true 10th distance, the box holds every true neighbour, so the loop always ends.

Counting rows is not enough. A box that is too narrow can still hold more than 10 rows while
leaving out true neighbours; only the distance check tells the two apart. Fewer than 10 rows fails
the check for the same reason: it says nothing about the rows outside the box.

The bound is on euclidean distance. On normalised vectors cosine distance is `||x - q||^2 / 2`,
so a `'cosine'` search works the same way with the check read as `distance <= rho * rho / 2`.
`'dot_product'` on vectors of varying magnitude has no such bound.

Any `k` rows of the table give a radius that cannot be too small: their k-th distance to the query
is at least the true one, so a box of that radius always passes the check. Where an
[IVF index](ivf.md) exists, the k-th distance found by probing a few clusters is such a radius, and
a tight one, but nothing here depends on it.

## Iceberg details

All three of these fail silently: the query stays correct and reads everything.

- **Metrics are only collected for the first hundred columns.** Iceberg infers column bounds by
  default for the first hundred columns of a table (`write.metadata.metrics.max-inferred-column-defaults`).
  Projection columns beyond that get no bounds and prune nothing. Declare them early in the schema,
  or set `write.metadata.metrics.column.pc_1 = 'full'` and the like for each of them.
- **Keep the columns flat.** A `struct` of projections keeps its statistics, since every leaf has
  its own field id and bounds, but pushing a filter on a subfield all the way down to file pruning
  is not something to assume. Flat columns are safe. `EXPLAIN ANALYZE` of step 2 shows how many
  rows and files were actually read, and is the check either way.
- **Appends erode the ordering.** Files are skipped when each covers a narrow range of `pc_1` that
  the others do not. `sorted_by` orders the rows within each file Trino writes, which helps the row
  groups inside a file, but every append writes files whose ranges overlap the existing ones, and
  overlapping ranges skip nothing. The table needs a periodic rewrite that sorts across files, such
  as Spark's `rewrite_data_files` with the `sort` strategy. This is the one real maintenance cost
  here: the directions never need refitting, but the file layout does.

## What to expect

A factor of about 3 on a linear scan with good directions, against roughly 60 for the
[IVF](ivf.md) configuration described there. What projection columns add is exactness with a
proof, through the check in step 3, and independence: the search needs the four columns and
nothing else.
