---
name: Soundness failure
about: The planner called a partition clean that a full rebuild proved dirty
labels: bug, soundness
---

## This is the highest-priority class of bug here

The planner may over-invalidate. It must never under-invalidate. If shadow mode
reported a non-zero `missed`, or you found stale data downstream of a partition the
plan skipped, this is the right template.

**Do not enable apply mode until this is resolved.**

## What was missed

<!-- Which dataset, which partitions. `fathom shadow` output if you have it. -->

## The graph

<!-- `fathom lineage` for the path from the seeded dataset to the one that was missed.
     The partition mappings on those edges are the thing to look at. -->

```
```

## The seeds

<!-- What you passed to --dirty, or what --detect reported. -->

## Partition specs

<!-- The `partition:` blocks for every dataset on that path, from fathom.yml.
     A spec that does not match how the data is actually laid out is the most
     common cause, and it is a real bug on our side if we accepted it. -->

## Environment

- fathom version:
- Platform / warehouse:
