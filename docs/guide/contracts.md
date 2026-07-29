# contracts and risk — promises, and what the columns jointly reveal

Two checks that read the same profiles everything else does, and answer questions
per-column analysis structurally cannot.

## A promise, with somebody's name on it

Everything needed to enforce a data contract already existed and none of it was bound
to a promise. `quality` checks a suite without saying whose. `schema` finds a breaking
change without saying who it breaks. `diff` gates a narrowing without knowing which
consumer it hurts. So the contract lives in a wiki, the producer never reads it, and
it is discovered to have been violated by the consumer, in production, on a Sunday.

```yaml
contracts:
  - dataset: gold.orders
    producer: platform
    consumers: [finance, ml]
    columns: [order_id, amount, currency]
    max_staleness: 6h
```

```bash
fathom contracts
```

```
duckdb/gold.orders by platform to finance, ml: 2 breach(es)
    [error] duckdb/gold.orders: promised column 'currency' is absent (owed by platform to finance, ml)
    [error] duckdb/gold.orders: 9.0h old, past the promised 6.0h (owed by platform to finance, ml)
```

Exits non-zero on any error.

**`verify` adds no checking machinery.** It dispatches to the modules that already
existed and collects what they say. What it adds is *attribution* — a breach names who
promised what to whom, which is the difference between an alert and an escalation.

**Severity follows the blast radius.** The same removed column is a warning against a
dataset nobody consumes and an error against one three teams read. That is the single
judgement this module makes on its own, and it is the one that makes the output
sortable by how much it matters.

**A promise with no evidence is `unchecked`, not passed.** A report that looks met
because the caller forgot to supply a profile is the failure mode being avoided.
`fathom doctor` will tell you when a contract is quietly in that state — a contract on
a never-profiled dataset reports "unchecked" forever, which reads as passing.

Durations take a number and a unit (`30m`, `6h`, `2d`, `1w`). A bare number is
rejected: every config format that accepts one ends up with two readers silently
disagreeing about whether it meant seconds or hours.

## Columns that identify nobody alone

`label` finds direct identifiers — a column named `email` holding things that look
like addresses. That is the easy half.

The hard half is that a birth date identifies nobody, a postcode identifies nobody,
and together they identify most of a population. A dataset can pass every
direct-identifier check and still be personal data, and no per-column label can see it,
because the property is not a property of any column.

```bash
fathom risk --min-k 5
```

```
duckdb/export.customers: 2 re-identification risk(s) proven at k=5
    [error] duckdb/export.customers: dob, zip together give an average group of at
            most 2.50 rows, below k=5
    [warn]  duckdb/export.customers: order_ref has 9,910 distinct values across
            10,000 rows, so it singles a row out whatever it is named
    quasi-identifiers present: dob, zip
    a clear result means no risk was proven, not that the data is safe — the minimum
    group size needs a scan this does not do
```

### Which direction it errs, and why

It over-reports risk and never under-reports it — the mirror of the planner, for the
same reason. A false alarm costs a review; a missed one is a disclosure that cannot be
recalled.

That commitment is what makes the arithmetic honest. From per-column profiles we know
each quasi-identifier's distinct count `dᵢ` and the row count `N`, but not the number
of distinct *combinations* `C`, which would need a scan. What we do know is
`C ≥ max(dᵢ)`, so the average group size is at most `N / max(dᵢ)`.

So this can **prove a dataset is risky** and can never prove one is safe. If the bound
comes back below the threshold, the average group is genuinely too small. If it comes
back above, the minimum group may still be one person hiding under a comfortable
average. `is_clear` is named for the absence of proven risk, and the summary says so
in its own text rather than reading as a clean bill of health.

Columns with no distinct count are reported as **not measurable** rather than skipped
silently, because "we did not check this" and "this was fine" are different facts.

### Across two datasets

Two exports that are each defensible become identifying when a shared ancestor means
their rows can be joined, and neither export's own review can see the other.

```python
from fathom.govern import reidentification as reid

reid.linkage_risks(graph, profiles, labels)
```

Compared by label *kind* rather than column name, so a `dob` joined to a `birth_date`
correctly adds nothing while a `dob` joined to a postcode does.

## Related

- [label](label.md) — inferring and propagating the labels both of these read
- [configuration](configuration.md) — the `contracts` block
