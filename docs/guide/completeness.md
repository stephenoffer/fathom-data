# completeness — the partition that never arrived

Every other check reads data that showed up and asks whether it looks right. This one
asks the question those cannot.

A partition that was never written has no profile to drift, no rows to fail an
expectation, and no signal at all. From downstream it is indistinguishable from a
partition that legitimately holds nothing: both contribute nothing to a join, both
make a `SUM` smaller, and neither raises. The gap gets found weeks later by somebody
noticing a dip in a chart.

```bash
fathom completeness --dataset raw.events --since 2026-03-01 --until 2026-03-31
```

```
incomplete: 5 of 62 partitions missing (92% present), in 2 run(s)
    dt=2026-03-04T00:00:00..2026-03-06T00:00:00 (3 days) [region=eu]
    dt=2026-03-04T00:00:00..2026-03-05T00:00:00 (2 days) [region=us]
    domains inferred from observed data: region (2 value(s)) — a value that never
    appeared cannot be reported missing
```

Exits non-zero when anything is missing.

## Runs, not alerts

Seven consecutive missing days is one incident with a start and an end. Reporting it
as seven findings buries the shape of the thing, which is the part that tells you
whether a job was down for a weekend or a source stopped delivering entirely.

Runs are computed **within each value slice separately**. `region=eu` missing Monday
through Wednesday and `region=us` missing only Tuesday are two incidents with
different causes, and merging them misreports both.

## Where "expected" comes from

Time fields enumerate from the spec's grain over the range you ask for. That part is
exact.

Value fields cannot be enumerated from a spec — nothing in `{field: region}` says
which regions exist. So either you declare the domain, or it is inferred from the
values actually observed across the range. The inference has one blind spot and the
report always states it: **a value that has never once appeared cannot be reported
missing.** A region switched off on day one is invisible until you declare it.

```python
from fathom.observe import completeness

result = completeness.report(
    dataset, spec, present,
    start=..., end=...,
    domains={"region": ["eu", "us", "apac"]},   # closes the blind spot
)
```

An oversized range raises rather than truncating. A shortened expected set makes an
incomplete dataset look complete, which is the exact failure this module exists to
prevent.

## What counts as present

`fathom completeness` reads the **arrival log**, not a directory listing.

```python
store.record_arrival(Arrival(dataset, key, observed, digest="sha256:...", row_count=1_000))
```

A listing tells you what exists now. An arrival log tells you what ever landed, which
still answers after a partition has been deleted — and "it was there in March and is
gone now" is a different incident from "it never arrived", worth being able to tell
apart.

## Arrivals: late, replayed, restated

Knowing a partition exists says nothing about whether it arrived on time, arrived
twice, or arrived twice with different contents.

```python
completeness.late_arrivals(arrivals, field_name="dt", grain=Grain.DAY,
                           tolerance=timedelta(hours=3))
completeness.replays(arrivals)        # same digest — an idempotent rewrite
completeness.restatements(arrivals)   # contents changed, or cannot be proven not to
```

Lag is measured from where the bucket **closes**, so a daily partition written at
02:00 the next morning is two hours late rather than twenty-six.

The split between a replay and a restatement is the one that matters, because a
restatement is what silently double-counts revenue. A repeat arrival whose digest is
unknown is classified as a **restatement**, never a replay: treating "cannot tell" as
"unchanged" is precisely what lets one through unnoticed.

## Related

- [check](check.md) — drift in the data that did arrive
- [plan](plan.md) — rebuilding the partitions a change affects
