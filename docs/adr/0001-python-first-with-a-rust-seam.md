# 1. Python first, with a seam for a Rust core

Status: accepted

## Context

The design calls for two things with very different performance profiles. Adapter
breadth, dialect handling, and integrations with dbt and Airflow are ecosystem work,
and the ecosystem is Python. Object listing, Parquet footer parsing, and Arrow-native
profiling are hot paths where a Rust core using `object_store` and DataFusion would
eventually be faster.

Deciding this late means a rewrite. Deciding it early in favour of Rust means months
before anything is usable, and a contribution barrier that shrinks the adapter matrix
we depend on.

## Decision

Build M0 in Python. The heavy lifting already runs in compiled code: pyarrow parses
Parquet footers, sqlglot parses SQL, DuckDB executes. We are orchestrating native
libraries, not implementing them.

Keep the seam explicit. `profile.py` and the storage adapters are the only modules
that touch bytes, and they exchange plain dataclasses with everything else. A Rust
core can replace them behind the same interfaces without the graph, the lattice, or
the adapters noticing.

## Consequences

- Fast iteration on the part that is actually novel — the partition algebra — instead
  of on FFI plumbing.
- Adapter contributions stay cheap, which is the whole strategy for reaching 25
  providers without hiring for it.
- We will hit a wall on very large object listings and full-scan profiling. That is
  the trigger to build the Rust core, not the assumption we start from.
- Profile a real petabyte-scale bucket before deciding. The footer-only path may make
  the wall much further away than expected.
