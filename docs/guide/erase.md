# erase — where a subject's data is, and how to destroy it

Deleting one subject from a lakehouse is a rewrite-the-world operation until you
know which files in which derived tables actually hold their rows. That is the same
question invalidation answers, so erasure reuses the planner.

## Planning

```bash
fathom erase \
  --subject u1 \
  --key-column user_id \
  --origin raw.events \
  --partition dt=2026-03-14 \
  --reference DSR-77 \
  --proof proof.json
```

```
erasure plan for subject bfb16fdb365e… (user_id) across 3 dataset(s)
  duckdb/raw.events: rewrite  dt=2026-03-14T00:00:00
  duckdb/silver.events: rewrite  dt=2026-03-14T00:00:00/region=ANY
  duckdb/gold.monthly: storage refuses deletion (Object Lock, WORM, or an adapter
    without erasure support); crypto-shredding is the only remaining option
  REFUSED: storage refuses deletion ...

This plan is INCOMPLETE. Some copies cannot be destroyed by this tool;
do not report the request as fulfilled.
```

Exits 1 when incomplete.

Targets are listed in **topological order** — sources before anything derived from
them. That is load-bearing, and the next section explains why.

## Two different operations

The obvious model of "delete this subject everywhere" is wrong, and it fails
silently.

A **source** table holds the subject's rows directly, so a `DELETE` works.

A **derived** table often holds aggregates with no subject column at all —
`gold.monthly` has a revenue sum, not a `user_id`. There is nothing to delete. The
subject's contribution is baked into a number, and the only correct action is to
**re-derive** the affected partitions from already-erased upstream.

Which makes ordering matter. During development this project shipped exactly that
bug: targets were ordered alphabetically, so `gold.monthly` was rebuilt *before*
`raw.events` was erased. The rebuild read data that still contained the subject,
produced identical output, reported success, and the proof said complete. See
[ADR 6](../adr/0006-erasure-is-not-deletion.md).

## Executing

The CLI never executes. Executing needs a live engine binding:

```python
from fathom.erasure import ErasureRequest, apply_erasure

request = ErasureRequest(
    subject="u1",
    key_column="user_id",
    origin=raw,
    partitions=frozenset({KeyPredicate.of(dt=datetime(2026, 3, 14))}),
    reference="DSR-77",
)

plan = project.locate(request)
if not plan.is_complete:
    raise SystemExit(plan.summary())      # do not partially erase and call it done

proof = apply_erasure(plan, {raw: engine, silver: engine, gold: engine},
                      dry_run=False, salt=os.environ["FATHOM_SALT"])
Path("proof.json").write_text(proof.to_json())
```

`dry_run` defaults to `True` and must be turned off explicitly.

## The invariant runs backwards

The planner may over-invalidate, because a wasted rebuild costs money. Erasure may
**under**-delete and refuse, because an over-broad delete destroys data that cannot
be recovered.

Consequences:

- Dry run by default.
- An explicit refusal on Object Lock and WORM buckets, rather than a delete that
  silently does nothing.
- A source with neither the key column nor a model raises instead of deleting by
  some other column and hoping.
- Any blocked target forces `complete: false` in the proof.

Over-approximating the *scope* is still safe: it means scanning a few extra
partitions, not deleting extra rows, because the delete is keyed on the subject.

## Why unconfigured datasets block

Erasure only acts on datasets whose adapter was chosen deliberately — an explicit
`adapter:` in `fathom.yml`, or a path dataset whose format was sniffed. A bare table
name falling back to the project's default `system` is a guess, and an erasure plan
acting on a guess is how a request gets reported fulfilled while the data survives
somewhere nobody configured.

Unconfigured datasets appear as blocked with `no adapter configured; cannot verify
the data can be destroyed`.

## Proof artifacts

```json
{
  "subject_digest": "bfb16fdb365e0cecfb1dd12c...",
  "reference": "DSR-77",
  "generated": "2026-03-14T09:12:33+00:00",
  "executed": true,
  "complete": true,
  "entries": [
    {"dataset": "duckdb/raw.events", "mode": "rewrite", "status": "erased",
     "rows_deleted": 2, "partitions": ["dt=2026-03-14T00:00:00"], "files": [...]},
    ...
  ],
  "digest": "f81823657df8..."
}
```

The subject identifier is **hashed with an operator-supplied salt**, never written
in plaintext. Proofs are retained for years and read by people who should not learn
who the subject was.

```bash
export FATHOM_SALT='...'    # per-organization, secret, and stable
```

Changing the salt changes every digest, so store it as carefully as any other key.
The `digest` field is a SHA-256 over the body, so tampering is detectable.

## What is out of scope

Stated here because a compliance feature that implies coverage it does not have is
worse than one that admits its limits:

- **Backups and snapshots.** Not visible to any adapter, not covered by a proof.
- **Replicas in other regions.** Cross-region replication does not always propagate
  delete markers the way people assume.
- **Object versioning.** Deleting an object leaves prior versions. True erasure
  requires deleting every version, and lifecycle rules may be retaining them.
  Crypto-shredding is the only reliable answer, and the adapter reports
  `CRYPTO_SHRED` where that applies.
- **Downstream systems outside the graph.** A dashboard cache or an exported CSV is
  a copy the tool cannot see.
