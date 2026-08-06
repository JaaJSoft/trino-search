# trino-search - Claude guidance

Trino plugin providing search functions as SQL functions. The first family covers vectors:
distance metrics, normalization and exact KNN search. Standalone repository
(<https://github.com/JaaJSoft/trino-search>), not part of the Trino tree.

The name is broader than the current content on purpose: further families of search functions
are expected, so each family lives in its own subpackage under `dev.jaaj.trino.search` and
`SearchPlugin` is the only class at the root.

The tests are the specification. Edge-case behavior (null elements, zero norms, dimension
mismatches, empty groups, a `k` larger than the group) is pinned by name in the test classes,
and `README.md` documents the user-facing contract. Read both before changing behavior; if a
change makes a test fail, the test is the thing to argue with, not to edit.

## Workflow artifacts are never committed

Implementation plans, task briefs, progress notes, assistant scratch directories: none of it
goes in git. It is bookkeeping, it dates the moment the code lands, and it leaks local machine
paths and tooling names into a public repository. Only source, tests, build files, `README.md`
and this file belong here.

## Build

Maven is not installed system-wide; use the committed wrapper. `JAVA_HOME` points at JDK 8 on
this machine, so **every** build command must select JDK 25 explicitly:

```bash
JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot" ./mvnw clean verify
```

Trino 483 requires Java 25. A build that silently picks up JDK 8 fails on `air.java.version`.

Run a single test class:

```bash
JAVA_HOME="..." ./mvnw test -Dtest=TestVectorDistances
```

## Java conventions

The project inherits `io.airlift:airbase`, the same parent Trino uses, so checkstyle and
modernizer enforce most of the Trino code style on build. Rules the tooling does not catch:

- No wildcard imports.
- Braces around single-statement `if` / `for` / `while` bodies.
- No `@author` in JavaDoc - the commit history is the record.
- Apache license header on every source file.
- Root package is `dev.jaaj.trino.search`. Never `io.trino.*`: that groupId belongs to the
  Trino project, and a split package would break the isolated plugin classloader.
- Everything is written in English: code, comments, commit messages, PR descriptions, issues
  and `README.md`.

## Testing

Development is test-driven: the test comes first and must fail for the right reason before the
implementation exists.

Three levels, all of which matter:

1. **Unit tests on the computation core**, with hand-computed reference values and mathematical
   properties (symmetry, triangle inequality).
2. **End-to-end SQL** through `StandaloneQueryRunner` and `AbstractTestQueryFramework`, which
   boot a real Trino engine in-process. This is the only level that exercises function
   resolution, overload selection and error messages.
3. **Differential tests** comparing `knn_agg` against the equivalent `ORDER BY ... LIMIT k`.

**Aggregations must be tested across multiple splits.** On a single partition Trino bypasses
the serialize/combine cycle entirely, so a broken `@CombineFunction` or an incomplete
serialized state passes every naive test and produces wrong results in production.

## PR Descriptions

PR descriptions must follow the structure of `.github/PULL_REQUEST_TEMPLATE.md` (Summary /
Changes / Screenshots / Testing / Notes). GitHub only pre-fills that template when no body is
provided, and `gh pr create --body` bypasses it, so when writing a body, reproduce the
structure manually. Rules:

- Write for a human reviewer and for whoever reads the PR in a year. Explain intent and impact;
  never paraphrase the diff line by line - the diff is right below.
- `Summary` is mandatory: 2-5 sentences, outcome first, approach only if non-obvious.
- Every other section is optional - **delete** sections that don't apply, never leave empty
  headings or "N/A". Small PRs often need only `Summary` and `Testing`.
- `Changes` lists only what a reviewer needs to navigate the diff. Omit mechanical fallout
  (renames, import updates, generated files, "updated tests accordingly") and process narration
  ("verified X still works", "explored Y before choosing Z").
- `Testing` is factual: suites run with results, what new regression tests pin down, manual
  checks performed. No checklists, no ✅ theater.
- `Notes` only when there is real content: breaking changes, known limitations, deliberate
  follow-ups, review focus areas.
- The PR is the project's public face: no internal tooling details and no self-congratulation.

### Voice and attribution

Commit messages and PR bodies are written in the project's voice: plain declarative English,
explaining intent rather than narrating the authoring process. Tooling trailers such as
`Co-Authored-By` are fine; prose that reads as a machine describing its own work is not.

## Backward Compatibility

By default, do not preserve backward compatibility. In doubt, ask.

Rationale: SQL function signatures, serialized aggregation state and behavior are not
versioned - no deployment outside this project depends on stable contracts yet. Shipping a
breaking change (renamed function, changed return shape, stricter validation) is the right call
if it simplifies the code or the model: update all call sites and remove the old path, with no
legacy aliases, shims, deprecated parameters or dual code paths kept around "just in case".
Only preserve compatibility when:

- The change would require a migration strategy (persisted data, user-facing schema changes).
- A production system or external integration actively depends on the old interface.
- You're explicitly unsure whether a caller exists - ask before breaking it.

When in doubt between "the old way is dead code" and "someone might use this", ask rather than
guessing.

## Refactoring and Optimization

Before any refactor or optimization, verify that at least one test covers the code being
touched. If no test exists, **write the test first** (it must pass against the current code),
then start the refactor. The test acts as a safety net guaranteeing the behavior is preserved.

This matters more than usual here: the distance kernels are the kind of code that invites
micro-optimization, and a subtly wrong loop produces plausible numbers rather than an obvious
failure.

## Bug Fixes

Every bug fix must ship with a regression test. Write the test alongside the fix and **verify
it fails against the buggy code** (stash the fix, run the test, re-apply), so you have evidence
the test actually pins the bug down rather than passing for unrelated reasons. Without this
proof the test is decorative: a future regression of the same bug would slip through CI. The
test belongs in the same package as the code being fixed.

## Code Comments

Comment only when it helps someone understand the code when re-reading it cold in 6 months,
never to explain the change to the PR reviewer. "Now uses X", "moved from Y", "replaces the old
Z", justifications of why the change is correct: that context belongs in the commit message and
PR description, and becomes noise the moment the PR merges.

Always prefer making the code self-explanatory through variable and function names over adding
a comment; a comment that paraphrases what the code already says is noise to delete. The
comments worth writing state what the code cannot: an invariant, a non-obvious constraint, a
gotcha, why the seemingly simpler approach doesn't work.

## Typography

Never use em-dashes or en-dashes in any text: code comments, JavaDoc, commit messages, PR
descriptions, markdown. Use a plain hyphen.
