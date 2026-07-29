---
name: Bug report
about: Something behaved differently from what it documents
labels: bug
---

## What happened

<!-- What you ran, and what came back. Paste the output rather than describing it. -->

## What you expected

## Reproducing it

<!--
The three things that make a planning bug diagnosable, in order of usefulness:

  fathom lineage                     the graph, or the relevant part of it
  fathom plan ... --explain DATASET  why the planner reached its answer
  the seeds you passed

For anything reading data, the partition spec from fathom.yml matters too — most
surprising plans are a spec that does not match the data.
-->

```
```

## Environment

- fathom version: <!-- fathom --version -->
- Python version:
- Platform / warehouse:
- Installed extras: <!-- e.g. [iceberg], [cloud] -->
