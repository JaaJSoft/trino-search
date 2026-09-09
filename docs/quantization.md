# Quantisation and approximate search

A vector can be stored quantised instead of as `double` or `real` components: one signed byte per
component (`array(tinyint)`), or one bit (`varbinary`). At dimension 768 an `array(double)`
vector's uncompressed payload is 6144 bytes, an `array(real)` payload is 3072, an `array(tinyint)`
payload is 768, and a binary payload is 100 (a four-byte header plus `ceil(768 / 8)` packed
bytes). That is four times fewer bytes than `array(real)` at the int8 end of the range, and about
sixty-one times fewer than `array(double)` at the binary end.

That byte reduction is the main payoff, and it is not the only one: on the machine
[`BENCHMARKS.md`](../BENCHMARKS.md) records, int8's euclidean kernel is also the fastest of the
three at dimension 768, because its inner sum is pure integer arithmetic over the raw codes with
the scale applied once at the end rather than on every component. Binary's kernel, an XOR and a
population count, is cheaper again. Read the "How to read this" section of `BENCHMARKS.md` before
drawing a performance conclusion from any one row; it explains what the ratio column does and does
not tell you.

## Functions

| Function | Description |
| --- | --- |
| `vector_bounds_agg(x)` | fits per-dimension offsets and a single global scale over a corpus |
| `quantize_vector_tinyint(x, bounds)` | one signed byte per component; alias `quantize_vector_int8` |
| `quantize_vector_varbinary(x, bounds)` | one bit per component; aliases `quantize_vector_binary` and `quantize_vector_int1` |
| `hamming_distance(x, y)` | components that differ between two binary vectors |

Both quantise functions accept an `array(double)` or an `array(real)` input.

## Fitting the bounds

`vector_bounds_agg` returns `row(offsets array(double), scale double)`: the offset of a dimension
is the midpoint of the range observed over the corpus, and the single scale is the widest range
across all dimensions divided by 255. A signed byte holds 256 codes, so the fitted range is spread
over 255 steps about its own midpoint, with the midpoint encoding to 0, the fitted minimum to
-127 and the fitted maximum to +127. A value below the fitted range is the only one that reaches
the leftover code, -128.

Values outside the fitted range clamp rather than wrap, which keeps a vector fitted outside the
sampled range imprecise rather than wrong. A row containing a `NULL` component says nothing usable
about any dimension, so it is skipped by the fit; the same row quantises to `NULL`.

Fit once over a corpus, then encode:

```sql
CREATE TABLE quantisation AS SELECT vector_bounds_agg(embedding) AS p FROM documents;

ALTER TABLE documents ADD COLUMN embedding_int8 array(tinyint);
UPDATE documents SET embedding_int8 = quantize_vector_tinyint(embedding, (SELECT p FROM quantisation));
```

Binary codes come from the same bounds: a bit is set for every component above its dimension's
midpoint. Quantising about the midpoint rather than about zero is what keeps the codes centred
instead of dominated by whichever side of zero the embedding happens to sit on.

## Distances on codes

Every metric gains an overload per representation. On `array(tinyint)` both vectors are codes and
the fitted bounds are a **mandatory third argument**:

```sql
euclidean_distance(codes_a, codes_b, bounds)
euclidean_squared_distance(codes_a, codes_b, bounds)
manhattan_distance(codes_a, codes_b, bounds)
dot_product(codes_a, codes_b, bounds)
cosine_similarity(codes_a, codes_b, bounds)
cosine_distance(codes_a, codes_b, bounds)
```

> Dropping that argument does not fail. `array(tinyint)` coerces implicitly to the float vector
> types, so a two-argument call compiles, binds to an exact-vector overload, and computes on the
> raw codes with no scale applied at all: a plausible-looking number that is not the true distance.
> The bounds are what turn the raw codes back into the real metric, and for `cosine_similarity`
> they are what make it meaningful at all, since cosine is not translation-invariant.

Both operands must have been fitted against the bounds passed. Nothing checks that. A vector whose
dimension differs from the one the bounds were fitted on does raise.

On `varbinary` the metrics take two arguments and no bounds, since a binary code needs nothing
beyond itself to be read. The codes stand for a vector of `-1` and `+1` components, so every
metric is a closed form in the Hamming distance and all of them rank identically. A value is a
four-byte big-endian dimension header followed by `ceil(dimension / 8)` bytes, least significant
bit first; the header is what allows any dimension rather than only multiples of eight, and turns
a comparison between codes of different dimension into an error instead of a ranking quietly
computed over the wrong components.

The `quantize_vector_int1` spelling is there for the vocabulary that counts bits. The codes decode
to `-1` and `+1`, not to the `-1` and `0` a one-bit two's complement integer would hold.

## Recall

Ranking on codes is approximate. `TestQuantizedKnnAggRecall` pins the following floors, measured
at `k = 10` on a synthetic corpus:

| Representation | Regime | Shortlist | Recall floor |
| --- | --- | --- | --- |
| int8 | clustered | 1x k | 0.96 |
| int8 | uniform | 1x k | 0.98 |
| binary | clustered | 10x k | 0.98 |
| binary | uniform | 10x k | 0.86 |

Binary at 1x under the uniform regime is far below all of these, and the suite pins only that
widening the shortlist never loses neighbours. That is not a defect: under a uniform regime every
pairwise distance concentrates around the same value, leaving a one-bit-per-component code little
to exploit. Oversampling the shortlist is how binary search becomes usable.

## Oversample and re-rank

To recover the exact order, oversample the shortlist on the codes and join back to the exact
column:

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
