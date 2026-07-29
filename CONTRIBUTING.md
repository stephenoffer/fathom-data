# Contributing

Thanks for looking. This document is the short version of what the test suite already
enforces, so that you find out from a paragraph rather than from a red build.

## Setup

```bash
uv venv && uv pip install -e '.[dev]'
pytest -q
ruff check src tests && ruff format --check src tests && mypy
```

Python 3.11 or newer. The `dev` extra pulls in DuckDB, PyIceberg, and s3fs, because
the suite creates real Iceberg tables and real Parquet files rather than mocking the
formats it claims to read.

## The one rule that is not negotiable

**The planner may over-invalidate. It must never under-invalidate.**

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Any change to `core/partitions.py`, `core/grains.py`, or `graph/model.py` is a change
to this. Widening is always acceptable and costs compute; narrowing can serve stale
data, and no amount of measured speedup buys it. If a change makes a plan smaller,
the pull request has to say why the smaller answer is provably still complete.

Erasure carries the mirrored rule: it may under-delete and refuse, and must never
over-delete.

The property tests in `tests/core/test_partitions.py` check this across generated
grain and window combinations. They are not optional — they caught a real composition
bug where `hour → day → hour` lost the truncation anchor.

## What the test suite enforces about structure

These fail the build, so they are worth knowing before you start:

- **Layers import downward only** (`tests/test_layering.py`). `core` knows nothing;
  `graph` knows `core`; `ai` may know all of them. An upward import always arrives as
  one innocuous convenience, which is why a test rejects it rather than a reviewer.
- **No directory grows past twelve modules or ten subpackages.** Depth is free;
  breadth is not. If you are adding the thirteenth, the package needs splitting.
- **Every module and package has a docstring**, and a package docstring says what
  belongs in it and what does not — that is how the next module lands in the right
  place.
- **The test tree mirrors the source tree.** A new package needs a test directory.
- **Every docstring example runs** (`tests/test_doctests.py`). An `Example:` block
  that raises is worse than no example, because it costs the reader their confidence
  in everything else you wrote.
- **Every documented CLI command, adapter, and config key exists**
  (`tests/test_docs.py`), and every relative link in the docs resolves.
- **Every example in `examples/` executes** (`tests/test_examples.py`).

## Style, as it is actually applied here

- Line length 100. `ruff format` decides the rest; do not argue with it in review.
- Type annotations everywhere. `mypy --strict` passes and should keep passing.
- **Comments explain why, never what.** The code says what. A comment earns its place
  by recording a decision, a constraint, or a bug that a future reader would
  otherwise reintroduce.
- **Error messages name the thing and the next action.** A message the reader cannot
  act on has cost them the same as no message. If you add a `raise`, the message
  should quote the offending value and say what to write instead. `did_you_mean` and
  `options` in `core/util/text.py` are there for this.
- **Docstrings on public functions carry Args, Returns, Raises, and an Example**
  where the call is not obvious from the signature. The example is executed.

## Adding things

### A new module

Put it in the package matching the lifecycle stage it belongs to — that is what the
package docstrings are for. If none fits, that is worth discussing in the issue
before the pull request.

### A new adapter

Adapters declare capabilities rather than implement everything, so start by writing
down what yours can honestly do (`Capabilities`), and let the planner degrade around
the rest. A `DeclaredCatalog` that reports `LIST_DIFF` and no pushdown is a legitimate
adapter and a fine place to start.

The conformance suite in `tests/adapters/conformance/` is the contract. Run it against
your adapter; anything it cannot pass should be reflected in the declared capabilities
rather than worked around.

### A new CLI command

Give it a section in `_Sections.SECTIONS` in `cli/main.py` — a test fails if a command
ends up unsectioned — and an example in its help text, which is also tested. If it
introduces a term, add the term to `cli/explain.py`.

### A new concept

If the output prints a word that means something specific here, it needs an entry in
`cli/explain.py` with a "What to do:" line, and probably one in
[the glossary](docs/guide/glossary.md).

## Documentation

`docs/problems.md` is the catalog of problems this exists to solve, each with an
honest status: Solved, Partial, or Gap. If your change closes or narrows one, update
its entry. A `Gap` is a commitment to say so out loud rather than something to be
embarrassed about, so please do not quietly upgrade one.

Documentation that is not executed rots, which is why the examples run in CI and the
docstring examples are doctests. Prefer adding to a runnable example over adding
untested prose.

## Pull requests

- One concern per pull request.
- Say what problem it solves, in the terms `docs/problems.md` uses if one applies.
- If it touches partition mapping, grain arithmetic, or invalidation, say explicitly
  which direction the change errs in.
- New behaviour comes with a test that fails without it.

## Reporting a bug

The most useful report includes the graph (`fathom lineage`), the seeds, and what you
expected versus what you got. If it is a planning bug, `fathom plan --explain
THE_DATASET` output is worth more than a description of the plan.

A **missed partition in shadow mode is a soundness failure** and the highest-priority
class of bug here. Please report it as such — the planner called something clean that
a full rebuild proved dirty, and that is the failure mode this whole design exists to
prevent.

## Licence

Apache-2.0. By contributing you agree your contribution is licensed under it.
