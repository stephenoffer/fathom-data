# 2. Soundness invariants, and which direction each one errs

Status: accepted

## Context

Every module here can be wrong in two directions, and the two are not symmetric. A
planner that rebuilds too much wastes money. A planner that rebuilds too little
serves stale data silently, for weeks, until someone notices a number is wrong. An
erasure tool that deletes too little leaves a compliance gap. One that deletes too
much destroys data that cannot be recovered.

Getting these backwards in even one code path undermines the whole tool, because
users cannot verify our claims cheaply — that is why they wanted the tool.

## Decision

Three invariants, each pointing a different way.

**The planner may over-invalidate, never under-invalidate.**

```
apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }
```

Anything unprovable widens to `UNBOUNDED`: an unparseable statement, an opaque UDF,
a partition spec mismatch, a `MERGE`, a cycle revisited too many times, a cross
product past the enumeration cap. Each of these is a place where a plausible guess
would have been possible and is deliberately refused.

**Erasure may under-delete and refuse, never over-delete.** Dry run by default, a
signed proof artifact listing every file touched, and an explicit refusal on
Object Lock and WORM buckets rather than a delete that silently does nothing.

**Drift detection may under-report on small samples, never manufacture confidence.**
A threshold without a sample-size guard is a noise generator; below the floor,
findings downgrade to `INFO` rather than paging someone about forty rows.

## Consequences

- Plans are wider than a perfect planner would produce. That is the price, and it is
  measurable: shadow mode reports both savings and misses, so users can see it.
- Refusing to guess makes some results look unhelpfully coarse. Widening honestly is
  better than a narrow answer that is wrong 2% of the time, because 2% wrong is
  indistinguishable from working until it isn't.
- The property tests in `tests/test_partitions.py` are the enforcement mechanism for
  the first invariant. They are not optional, and they already caught one real bug in
  composition through a truncation boundary.
