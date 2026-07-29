# 7. Packages follow the lifecycle, and the layering is enforced

Status: accepted

## Context

The library grew from six modules to seventy. The flat layout that was obvious at six
stopped being navigable somewhere around twenty-five: `src/fathom/` held twenty-nine
modules with no signal about which depended on which, and three pairs had quietly
started importing each other.

Two of those cycles were real design errors rather than accidents:

- `graph.diff` imported `govern.policy`, because diffing labels felt like diffing.
  Structural graph diffing has no business knowing what a label is.
- `observe.shadow` imported `store.Store` to persist a result, and `store` imported
  `observe.profile` to hydrate one. Neither import was wrong on its own.

Both would have been caught by anything that looked. Nothing looked.

## Decision

**Packages follow the lifecycle of the data, not the taxonomy of the code.** A module
belongs where its stage belongs — how the graph is learned (`ingest`), what the graph
is (`graph`), what the data looks like (`observe`), what you may do with it
(`govern`), what AI does to all of it (`ai`), and how it comes back out (`report`).

**The layers form a total order, and imports only go downward.**

```
core → graph → observe → govern → ai → adapters → ingest → store → report → cli
```

`core` is the vocabulary — identity, grains, the partition lattice — and imports
nothing from the rest of the library, because a vocabulary that depends on its
speakers is not a vocabulary. Each subsequent layer may import anything below it.

**No more than twelve modules or ten subpackages at any level.** Depth is free and
breadth is not: a directory of thirty files is a directory nobody reads, and a new
contributor picking a home for a module is choosing from thirty equally plausible
neighbours. When a package reaches twelve, it has earned a subpackage.

**All three rules are tests.** `tests/test_layering.py` parses every module's imports
and fails on an upward one, naming both ends and the fix. It also fails on an
oversized directory and on a package `__init__` that does not explain what belongs in
it. The test directory mirrors the package tree for the same reason.

## Consequences

Fixing the layering forced three modules to move, and each move was an improvement
that the flat layout had been hiding:

- `diff` split three ways — structural graph diffs stay in `graph`, schema diffs go to
  `observe.schema`, label diffs to `govern.diff`. Each now sits in the layer that owns
  the thing being compared.
- `compliance` moved from `govern` to `report`, because it reads the AI graph. It was
  never a governance primitive; it is a document generated from several.
- `ShadowObservation` moved from `store` to `observe.shadow`, and `shadow` now depends
  on a one-method `ShadowLedger` protocol rather than on SQLite. The record belongs
  with the thing that produces it, not with the thing that files it.

The same pass collapsed four duplicated helpers into `core/util`: content addressing,
Markdown assembly, UTC normalization, and token estimation. Each had between three and
six copies that had already begun to disagree — two different characters-per-token
constants, four private `_aware()` functions, six ways to hash a payload.

The cost is import churn for anyone who was reaching past the top-level namespace.
The mitigation is that the flat names still work: `from fathom import query, cost, ai`
resolves regardless of how deep the defining module sits.

The rule this trades away is "put it wherever it is used". A helper needed by two
layers must move down to their common ancestor rather than being imported sideways.
That is more work in the moment and is the only version that stays true.
