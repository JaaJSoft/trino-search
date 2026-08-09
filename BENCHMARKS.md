# Benchmarks

One row per pull request, for the vector search functions. Measured with the metric `euclidean`,
`k = 10` and the `CLUSTERED` data regime; a row measured on anything else is not comparable with
these.

Each cell reads `kernel / per-row / ratio`: nanoseconds per distance computation, nanoseconds per
row of the aggregation input path, and the second divided by the first.

How to read this:

- The ratio is the signal. Both of its halves were measured on the same machine at the same clock,
  so it survives being recorded somewhere else.
- Absolute nanoseconds are comparable only between rows sharing the same `Machine` label.
- A 10 to 20 percent drift in absolutes, or under 10 percent in a ratio, is noise rather than a
  change.

To record a row, from a build with Java 25:

```bash
./mvnw test-compile exec:java \
  -Dexec.classpathScope=test \
  -Dexec.mainClass=dev.jaaj.trino.search.vector.benchmark.ReferenceRowRunner \
  -Dexec.args="laptop"
```

The argument is a short free-form machine label, never a hostname: this file is public. An
optional second argument is the pull request number, which does not exist yet at measurement time;
without it the row carries `#TBD`. The command prints the row and does not write this file, so
paste the output at the bottom of the table yourself. It warns on standard error when a
measurement is too noisy to be worth recording.

| Date | PR | Commit | 128 double | 128 real | 768 double | 768 real | Machine | Cores | JDK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-09 | #11 | 0f73caf | 78.0 / 162.2 / 2.08 | 86.1 / 161.5 / 1.88 | 492.9 / 685.1 / 1.39 | 506.0 / 629.2 / 1.24 | laptop | 32 | 25.0.1 |
