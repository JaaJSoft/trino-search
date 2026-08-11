# Benchmarks

One row per pull request, for the vector search functions. Measured with the metric `euclidean`,
`k = 10` and the `CLUSTERED` data regime; a row measured on anything else is not comparable with
these.

Each cell reads `kernel / per-row / ratio`: nanoseconds per distance computation, nanoseconds per
row of the aggregation input path, and the second divided by the first.

How to read this:

- The ratio is the signal: both of its halves were measured on the same machine at the same clock,
  which cancels clock speed and thermal throttling and is why the ratio survives being recorded
  somewhere else better than either raw figure alone. It does not cancel cache size or memory
  bandwidth: the kernel benchmark keeps a small pool cache-resident while the per-row benchmark
  streams a much larger one (see `BenchmarkKnnAccumulator`'s class Javadoc), so a machine with more
  cache speeds up one half more than the other and moves the ratio in a machine-dependent
  direction. A same-machine ratio comparison is trustworthy; a cross-machine one is a weaker
  signal. The ratio is also an upper bound on the per-row bookkeeping cost, not a measurement of
  it, for the same reason.
- Absolute nanoseconds are comparable only between rows sharing the same `Machine` label.
- A drift of up to 20 percent in absolutes, or up to 15 percent in a ratio, is noise rather than a
  change.
- Ratios are comparable down a column, not across columns. A quantised kernel is several times
  cheaper than a float one while the per-row bookkeeping is unchanged, so an int8 or int1 ratio is
  far larger than a double one without anything having regressed: the denominator shrank. Read a
  quantised column's ratio as how completely the per-row path now dominates, which is the number
  that says when shrinking the representation further has stopped buying anything.

To record a row, from a build with Java 25:

```bash
./mvnw test-compile exec:java \
  -Dexec.classpathScope=test \
  -Dexec.mainClass=dev.jaaj.trino.search.vector.benchmark.ReferenceRowRunner \
  -Dexec.args="desktop-5950x"
```

The argument must name one specific machine and must never be reused for a different one:
absolute nanoseconds are only comparable between rows sharing the same label, so reusing it across
two different machines would certify numbers as comparable that are not. It is free-form but must
never be a hostname, since this file is public. The optional second argument is the pull request
number, which does not exist yet at measurement time; without it the row carries `#TBD`. The
command prints the row and does not write this file, so paste the output at the bottom of the
table yourself. It warns on standard error and exits with a non-zero status when a measurement is
too imprecise to be worth recording, judged on how precisely each mean is known rather than on
how widely single iterations happened to scatter; the row is still printed, since the warning can
scroll past unnoticed, but the failing exit status cannot. It also warns when the working tree has
uncommitted changes, because a row names a pull request and so cannot otherwise show that it
measured code that never reached one.

| Date | PR | 128 double | 128 real | 128 int8 | 128 int1 | 768 double | 768 real | 768 int8 | 768 int1 | Machine | CPU | Cores | JDK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-09 | #11 | 78.0 / 162.2 / 2.08 | 86.1 / 161.5 / 1.88 | - | - | 492.9 / 685.1 / 1.39 | 506.0 / 629.2 / 1.24 | - | - | desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |
| 2026-08-09 | #12 | 77.3 / 100.1 / 1.29 | 84.5 / 100.4 / 1.19 | - | - | 491.7 / 527.0 / 1.07 | 505.2 / 512.8 / 1.01 | - | - | desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |
| 2026-08-09 | #13 | 40.3 / 49.9 / 1.24 | 79.6 / 95.0 / 1.19 | - | - | 185.4 / 300.6 / 1.62 | 365.5 / 371.3 / 1.02 | - | - | desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |
| 2026-08-09 | #15 | 35.4 / 54.8 / 1.55 | 47.5 / 62.4 / 1.31 | - | - | 177.2 / 309.2 / 1.74 | 218.8 / 307.7 / 1.41 | - | - | desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |
| 2026-08-09 | #17 | 37.4 / 42.1 / 1.13 | 54.2 / 44.8 / 0.83 | - | - | 184.5 / 55.1 / 0.30 | 249.1 / 63.3 / 0.25 | - | - | desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |
| 2026-08-12 | #TBD | 48.0 / 49.6 / 1.03 | 56.6 / 49.6 / 0.88 | 100.6 / 83.6 / 0.83 | 13.8 / 28.7 / 2.07 | 196.4 / 64.6 / 0.33 | 266.0 / 72.5 / 0.27 | 521.2 / 133.7 / 0.26 | 27.9 / 39.6 / 1.42 | desktop-5950x | AMD Ryzen 9 5950X 16-Core Processor | 32 | 25.0.1 |
