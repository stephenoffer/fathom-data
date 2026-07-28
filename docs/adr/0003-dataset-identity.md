# 3. Dataset identity follows OpenLineage naming

Status: accepted

## Context

A Spark job writes `s3a://lake/raw/events`. A Trino query reads the same bytes as
`hive.raw.events`. A notebook reads `/dbfs/mnt/lake/raw/events`. A Snowflake external
table calls it `RAW.EVENTS`.

Unless all four collapse to one node, the dependency graph is four disconnected
fragments and every downstream feature is worthless — the planner finds no path, drift
attribution finds no upstream, erasure misses derived copies.

This is the single highest-leverage correctness problem in the project, and it is
entirely unglamorous.

## Decision

Adopt the OpenLineage dataset naming convention: a `namespace` locating the system
and a `name` locating the dataset within it. Do not invent a fourth spelling.

Normalize mechanically where the rules are unambiguous — protocol aliases (`s3a`,
`s3n`, `gcs`, `wasbs`), Azure's dual hostnames, duplicate separators, bucket case,
and per-system identifier folding (Snowflake up, Databricks down, BigQuery neither).

Where the connection is not discoverable from either reference — an external Hive
table pointing at an S3 prefix, a DBFS mount — require a declaration through
`AliasRegistry` or a mount table. Do not infer it.

## Consequences

- What we emit interoperates with the OpenLineage ecosystem for free, so `fathom`
  complements Marquez and friends rather than competing on format.
- Identifier folding rules are per-system and will be wrong for systems we have not
  studied. `_IDENTIFIER_CASE` defaults to case-sensitive, which fragments rather than
  wrongly merges — the safer failure.
- Requiring declarations for mounts and external tables is friction. The alternative
  is silently merging two unrelated datasets, which produces confidently wrong plans.
