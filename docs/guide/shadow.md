# Shadow mode — earning the right to skip work

`plan` asks you to trust a new tool with a decision whose failure mode is silent. A
planner that skips a partition it should have rebuilt does not crash, does not log
an error, and does not appear on a dashboard. It serves stale numbers until someone
notices a report is wrong, weeks later, and nobody connects it to the tool.

No amount of documentation earns that trust. Shadow mode is how you check.

## What it does

1. Fingerprint every target partition.
2. Run a **full** rebuild.
3. Fingerprint again.
4. Compare what actually changed against what the plan predicted.

Two numbers come out:

- **savings** — partitions the plan would have skipped
- **missed** — partitions the plan called clean that the rebuild proved dirty

`missed` must be zero. It is the direct empirical test of the
[soundness invariant](../adr/0002-soundness-invariants.md).

## Running it

Order matters. The target tables must still hold the **pre-change** build, so the
sequence is: source data lands, plan, shadow, then apply. Applying first destroys
the state there is nothing left to compare against.

```python
from fathom import shadow

plan = graph.invalidate({raw: [changed_partition]})
report = shadow.run(engine, plan, [silver, gold], store=store)

print(report.summary())
assert report.is_sound
```

```
shadow: SOUND across 2 dataset(s), 78% of partitions skipped
duckdb/silver.events: SOUND  planned=1 actual=1 total=6 savings=83% precision=100%
duckdb/gold.monthly: SOUND  planned=1 actual=1 total=4 savings=75% precision=100%
```

`precision` is the other direction: how many planned partitions genuinely needed
rebuilding. Low precision costs money. Low soundness costs correctness. They are
not equally important.

## Accumulating evidence

Passing a `store` records every run:

```bash
fathom shadow
```

```
runs        14
partitions  38 planned of 420 total
savings     91%
missed      0

no missed partitions across every run recorded here
```

If `missed` is ever non-zero, the command exits 1 and says so bluntly:

```
MISSED PARTITIONS ARE A SOUNDNESS FAILURE. The planner called them clean and a
full rebuild proved otherwise. Do not enable apply mode.
```

Wire that into CI. It is the one check that matters.

## In a nightly job

```python
with Project.load() as project:
    changes = project.detect()
    plan = project.plan({ds: c.partitions for ds, c in changes.items() if c.partitions})

    report = shadow.run(engine, plan, models, store=project.store)
    if not report.is_sound:
        alert(report.summary())        # a real bug; stop here
    # the full rebuild has already run, so the data is correct either way
```

The valuable property of this shape is that it is **risk-free**. The full rebuild
runs regardless, so the data is right whether or not the planner is. You are only
collecting evidence.

Run it for as long as it takes to convince you — weeks is normal — then switch the
full rebuild for `engine.apply(...)` and keep `fathom shadow` in CI as a periodic
audit.

## How fingerprints work

Per partition, an order-independent content hash:

```sql
SELECT dt, region, MD5(STRING_AGG(row_text, CHR(30) ORDER BY row_text))
FROM gold.monthly GROUP BY dt, region
```

Sorting inside the aggregate is what makes it order-independent. Without it, a
rebuild that produced identical rows in a different order would report every
partition as changed and the metric would be meaningless.

Rows are joined with unit separators (`CHR(31)`) rather than commas, so a value
containing a delimiter cannot forge a column boundary and make two different rows
hash the same.

## Proving the checker can fail

A soundness checker that cannot report a failure is decoration. The test suite
feeds shadow mode a deliberately wrong plan and asserts it notices:

```python
plan.dirty[silver] = frozenset({wrong_partition})   # point at something unchanged
report = shadow.run(engine, plan, [silver, gold])
assert not report.is_sound
assert report.missed_total > 0
```

If you extend shadow mode, keep that test.

## Cost

Shadow mode runs a full rebuild every time, so it costs what you were already
spending, plus two fingerprint passes. That is the right trade while trust is being
established and the wrong one afterwards — it is a rollout tool, not a permanent
fixture. Once you switch to apply mode, run it weekly rather than nightly.
