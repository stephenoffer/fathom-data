# 6. Erasing a subject is two different operations, and the order matters

Status: accepted

## Context

The obvious model of "delete this subject everywhere" is wrong, and it fails
silently, which is the worst combination for a compliance feature.

A source table holds the subject's rows directly, so a `DELETE` works. A derived
table often holds *aggregates* of those rows and has no subject column at all —
`gold.monthly` has a revenue sum, not a `user_id`. There is nothing to delete there.
The subject's contribution is baked into a number.

Worse, an implementation that treats every dataset the same way tends to get the
ordering wrong. During development this project shipped exactly that bug: erasure
targets were ordered alphabetically, so `gold.monthly` was re-derived *before*
`raw.events` had been erased. The rebuild read data that still contained the subject,
produced identical output, reported success, and left the subject's contribution
intact in the aggregate. The proof artifact said complete.

## Decision

Two operations, chosen per dataset:

- **Delete** where the subject's rows exist and the key column is present.
- **Re-derive** where the dataset is a model. Rebuilding the affected partitions
  from already-erased upstream removes the contribution without needing a subject
  column at all.

Order targets **topologically**, never alphabetically, reusing the invalidation
planner's rebuild order. Sources are erased before anything downstream is re-derived.

Refuse rather than guess. A source without the key column and without a model raises
instead of deleting by some other column and hoping.

Reuse the planner for scoping. Seeding invalidation with the partitions holding the
subject yields, for every downstream dataset, the partitions that could hold derived
copies. Over-approximating is safe here: it means scanning a few extra partitions,
not deleting extra rows, because the delete is still keyed on the subject.

## Consequences

- Erasure inherits the planner's partition scoping for free, which is what turns a
  rewrite-the-world operation into a targeted one.
- The invariant runs opposite to the planner's: erasure may under-delete and refuse,
  never over-delete, because a wasted rebuild costs money and an over-broad delete
  costs data. Hence dry-run by default and an explicit refusal on WORM storage.
- Proof artifacts record per-dataset status, and any blocked target forces
  `complete: false`. A partial erasure must never read as a finished one.
- The subject identifier is hashed with an operator-supplied salt. Proofs are
  retained for years and read by people who should not learn who the subject was.
