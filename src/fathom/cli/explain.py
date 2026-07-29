"""Concept explanations, for `fathom explain`.

Every warning this tool prints uses at least one word that means something specific
here: *widened*, *unbounded*, *seed*, *sound*, *sink*. A user who has just been told
their plan widened has one question, and the answer is a paragraph, not a link.

So the paragraph lives here, next to the code that prints the word, and a test
asserts that every term the CLI emits has an entry. Documentation that sits in a
different repository from the message that provokes the question does not get read.

Each entry is: what it is, why it works that way, and what to do about it. The last
part is what makes this different from a glossary — a definition that ends without a
next action leaves the reader exactly where they started.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TOPICS", "Topic", "lookup", "titles"]


@dataclass(frozen=True)
class Topic:
    """One concept: a one-line gloss, the full explanation, and where to read on."""

    name: str
    summary: str
    body: str
    see_also: tuple[str, ...] = ()
    doc: str = ""

    @property
    def gloss(self) -> str:
        """The first sentence of the summary, for the one-line topic listing.

        Keeps the terminator it already has — several summaries end in a question
        mark, and appending a full stop to one reads as a typo.
        """
        for index, char in enumerate(self.summary):
            if char in ".?!" and index > 30:
                return self.summary[: index + 1]
        return self.summary

    def render(self) -> str:
        """The full entry as printed text."""
        lines = [self.name, "=" * len(self.name), "", self.summary, "", self.body.strip()]
        if self.see_also:
            lines += ["", "Related: " + ", ".join(self.see_also)]
        if self.doc:
            lines += ["", f"More: {self.doc}"]
        return "\n".join(lines)


TOPICS: dict[str, Topic] = {
    "partition-mapping": Topic(
        name="partition mapping",
        summary="The rule on a graph edge answering: if this input partition is "
        "dirty, which output partitions are dirty?",
        body="""
Every edge in the dependency graph carries one. It is the whole reason a plan can be
smaller than "everything downstream" — without it, one changed day of source data
means rebuilding every table that reads it, in full.

Three forms cover almost everything:

  TimeWindow    the output bucket is a time bucket offset from the input's.
                Identity, a daily-to-monthly rollup, and a 7-day trailing window
                are all TimeWindows.
  Passthrough   the output value is the input value, unchanged. What a region or
                tenant column does.
  Unbounded     no provable relationship. See `fathom explain unbounded`.

Mappings compose along a path and join where two paths reconverge, so the reach of
a change three hops away is one mapping rather than three guesses.

What to do: declare partition specs for both sides of an edge in fathom.yml, then
check `fathom doctor` for edges still carrying Unbounded. Each one you can replace
with a real mapping is rebuild you stop paying for.
""",
        see_also=("unbounded", "widening", "grain"),
        doc="docs/guide/concepts.md",
    ),
    "unbounded": Topic(
        name="unbounded",
        summary="The mapping used wherever a relationship could not be proven. It "
        "means any change to the input rebuilds the whole output.",
        body="""
Unbounded is the honest answer, not a failure. It appears when the SQL could not be
parsed, the statement was a MERGE, a UDF is opaque, the two sides declare
incompatible partition specs, or nobody declared a spec at all.

It costs compute and never costs correctness. An edge that widens to unbounded
rebuilds more than strictly necessary; an edge that claims a narrower reach than it
has serves stale data, which is the one failure this library is built to avoid.

What to do: `fathom doctor` names the edges that are unbounded and why.
`fathom.metrics.coverage(graph)` gives the fraction of the graph that is precise
enough to plan on — that number is very close to the ceiling on how much of a
rebuild any plan can skip, so it is the number to move.
""",
        see_also=("partition-mapping", "widening", "coverage", "soundness"),
        doc="docs/guide/troubleshooting.md",
    ),
    "widening": Topic(
        name="widening",
        summary="Losing partition precision on the way through the graph, so a "
        "dataset is rebuilt whole rather than partition by partition.",
        body="""
A plan reports a dataset as widened when it could not scope the rebuild to
particular partitions. Three things cause it:

  1. An unbounded mapping somewhere on the path to that dataset.
  2. A dataset with no partition spec — there are no partitions to name.
  3. A cycle. A self-referencing incremental model can enlarge its own dirty set
     forever, so after a budget of passes the planner takes the whole dataset
     rather than looping.

Widening is contagious downstream: everything fed by a widened dataset inherits the
imprecision.

What to do: `plan.explain(dataset)` (or the `widened:` lines `fathom plan` prints)
names the reason for each one. Fix the earliest widening on the path — the ones
below it usually disappear with it.
""",
        see_also=("unbounded", "partition-mapping", "soundness"),
        doc="docs/guide/plan.md",
    ),
    "soundness": Topic(
        name="soundness",
        summary="The planner may over-invalidate but must never under-invalidate, "
        "and erasure carries the mirrored rule.",
        body="""
The invariant every operation preserves:

    apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }

Composing two edges never claims fewer dirty partitions than walking them one at a
time would. Precision is an optimization; soundness is not. Anything unprovable
widens, which costs compute and never costs correctness.

Erasure is the mirror image: it may under-delete and refuse, but must never
over-delete. Hence dry-run by default, and an explicit refusal on WORM storage
rather than a report of success.

What to do: do not take the invariant on trust. `fathom shadow` runs the planner
alongside your existing full rebuild and grades it, reporting a miss count that must
be zero. See `fathom explain shadow-mode`.
""",
        see_also=("widening", "shadow-mode", "unbounded"),
        doc="docs/adr/0002-soundness-invariants.md",
    ),
    "shadow-mode": Topic(
        name="shadow mode",
        summary="Running the planner beside your existing full rebuild and grading "
        "it, at zero risk, until you trust it.",
        body="""
Nobody should trust a new tool to decide what *not* to rebuild on the strength of a
README. Shadow mode runs the planner in parallel with the rebuild you already do,
and grades every decision it made against what the full rebuild proved.

Two numbers per dataset:

  savings   partitions the planner skipped, which is what adopting it would save.
  missed    partitions the planner called clean that the full rebuild proved dirty.
            This must be zero. `fathom shadow` exits non-zero the moment it is not.

The full rebuild happens either way, so there is no risk in running it, and there is
no substitute for weeks of it before anything writes.

What to do: run `fathom shadow` after each pipeline run, and accumulate. A single
clean day is not evidence; a month of them is.
""",
        see_also=("soundness", "plan"),
        doc="docs/guide/shadow.md",
    ),
    "grain": Topic(
        name="grain",
        summary="How wide one time partition bucket is: hour, day, month, or year.",
        body="""
A time partition field carries a grain, and it is not optional, because a plan
cannot tell a daily partition from a monthly one without it. Declare it in
fathom.yml as `{field: dt, grain: day}`.

Grains are ordered fine to coarse, and conversions only ever go fine to coarse. A
daily source feeding a monthly rollup is something we can reason about precisely. A
monthly source feeding a daily table is a refinement whose honest answer is "some
large part of the month", so it widens to unbounded instead.

Conversion always rounds outward. A 7-day window converted to months becomes 3
months, not 1, because seven days from an arbitrary start can straddle a boundary
and the input day can sit anywhere inside its own month.

What to do: if a plan looks larger than the arithmetic suggests, this is usually
why, and it is the invariant working rather than a defect.
""",
        see_also=("partition-mapping", "soundness"),
        doc="docs/guide/concepts.md",
    ),
    "seed": Topic(
        name="seed",
        summary="What you tell the planner changed at the source. Everything else "
        "in a plan is derived from it.",
        body="""
A plan starts from seeds: dataset plus dirty partitions. Give it
`raw.events@dt=2026-03-14,region=eu` and it propagates through partition mappings to
every downstream partition that reads it.

Two ways to produce them:

  --dirty   you state what changed, which is right after a vendor redelivery or a
            bug fix with a known affected range.
  --detect  the configured adapters are asked what changed since the last run.

A dataset absent from the seeds is treated as unchanged. This is the input, not a
filter — seeding nothing plans nothing.

What to do: seed the true range rather than a padded one. Padding a backfill window
for safety inflates every downstream dataset's plan too, which is the cost the
planner exists to remove.
""",
        see_also=("plan", "partition-mapping"),
        doc="docs/guide/plan.md",
    ),
    "plan": Topic(
        name="plan",
        summary="Given what changed at the source, the exact set of partitions that "
        "must be rebuilt, in build order.",
        body="""
The first of the four verbs. It reads the dependency graph, propagates the seeds you
give it through each edge's partition mapping, and returns every partition of every
downstream dataset that went stale.

Reading the output:

  the dataset list        is in build order, dependencies before dependents.
  (widened to whole)      means precision was lost reaching that dataset.
  cycles detected in      means the planner stopped enlarging rather than loop.

`fathom plan --json` emits the same thing for an orchestrator to act on. Screen
scraping the human summary is how a pipeline ends up rebuilding the wrong thing
after a wording change.

What to do: nothing is executed. `plan` prints. Applying needs an engine binding you
supply deliberately, and it should not be the first thing you do — see
`fathom explain shadow-mode`.
""",
        see_also=("seed", "widening", "shadow-mode"),
        doc="docs/guide/plan.md",
    ),
    "coverage": Topic(
        name="coverage",
        summary="How much of the graph is precise enough to plan on. The number that "
        "predicts what you will save.",
        body="""
`fathom.metrics.coverage(graph)` reports four fractions: datasets with a partition
spec, edges with a bounded mapping, edges with column-level detail, and field
mappings that are provable.

The last one, `field_ratio`, is very close to the ceiling on how much of a rebuild
any plan can skip. It is the number to publish and the number to move.

An edge with an unbounded mapping is in the graph and contributes nothing to a plan.
A lineage catalog full of such edges looks complete and saves nothing, which is the
difference between a lineage tool and an inventory of one.

What to do: raise it by declaring partition specs, then by giving the SQL parser
statements it can read.
""",
        see_also=("unbounded", "partition-mapping"),
        doc="docs/guide/concepts.md",
    ),
    "evidence": Topic(
        name="evidence",
        summary="How an edge was learned — native lineage, a parsed query, a dbt "
        "manifest, or your own declaration.",
        body="""
Every edge carries the source that claimed it, so a surprising plan can be traced
back to whatever asserted the dependency. `fathom lineage` prints it in brackets.

The sources, most trustworthy first: `native` (the platform maintains a lineage
table), `listener` (an execution-plan hook), `query_log` (historical SQL, parsed),
`declared` (you wrote it in fathom.yml).

A declared edge is exactly as right as what you wrote, and it does not notice when
the pipeline changes underneath it. That is a reasonable trade for a system with no
lineage of its own, and worth revisiting when one becomes available.

What to do: `fathom adapters` reports which of these each configured platform can
give you.
""",
        see_also=("capabilities", "partition-mapping"),
        doc="docs/guide/adapters.md",
    ),
    "capabilities": Topic(
        name="capabilities",
        summary="What a given adapter can actually do. The planner degrades to fit "
        "rather than failing.",
        body="""
Adapters declare capabilities rather than implement everything. One reporting
`list_diff` change detection and no pushdown still works — it is slower and coarser,
and the planner adapts.

Most surprises about a plan are an adapter's declared limits showing through: no
column lineage means drift attributes to a table rather than a column; no partition
awareness means change is reported for the whole dataset.

What to do: `fathom adapters --verbose` prints every capability with what it means
for your plans. Read it before concluding the planner is being conservative for no
reason.
""",
        see_also=("evidence", "widening"),
        doc="docs/guide/adapters.md",
    ),
    "drift": Topic(
        name="drift",
        summary="A column's distribution moving between profiles — and, with the "
        "graph, what upstream caused it.",
        body="""
The `check` verb compares each dataset against its last profile. On its own that is
an alert: "revenue moved 8%". With lineage it is a diagnosis: "revenue moved because
fx_rates changed three hops upstream".

That is the whole argument for computing lineage and profiles from one metadata
plane. Drift detection without lineage tells you something is wrong; it does not
tell you where to look.

What to do: run `fathom profile` to establish a baseline before `fathom check` can
say anything. A first check with no prior profile is not a clean bill of health.
""",
        see_also=("profile", "seasonality"),
        doc="docs/guide/check.md",
    ),
    "profile": Topic(
        name="profile",
        summary="Distributions, ranges, and cardinalities per partition, read from "
        "file footers rather than from the data.",
        body="""
Profiling reads Parquet footers, not data pages, so it costs a metadata read rather
than a scan. Where a warehouse can compute statistics for us — see
`fathom explain capabilities` — it does that instead.

This is what makes continuous profiling affordable, and it is only affordable
because the graph says which partitions changed. Scanning whole tables nightly costs
real credits; profiling the partitions the graph flagged costs almost nothing.

What to do: profile before checking, and profile after each plan runs so the next
check has a recent baseline.
""",
        see_also=("drift", "completeness"),
        doc="docs/guide/check.md",
    ),
    "completeness": Topic(
        name="completeness",
        summary="The partitions that should exist and do not. The only check that "
        "can see a partition which never arrived.",
        body="""
Everything else in `observe` reads data that showed up and asks whether it looks
right. A partition that was never written has no profile to drift, no rows to fail
an expectation, and no signal at all — downstream it is indistinguishable from a
partition that legitimately holds nothing.

The gap gets found weeks later by somebody noticing a dip in a chart.

Absences are collapsed into contiguous runs within each value slice separately:
region=eu missing three days and region=us missing one are two incidents, and
merging them misreports both.

What to do: `fathom completeness --dataset X --since ... --until ...`. It reads
recorded arrivals rather than a listing, so it still answers after a partition has
been deleted.
""",
        see_also=("profile", "drift"),
        doc="docs/guide/completeness.md",
    ),
    "sink": Topic(
        name="sink",
        summary="The last hop out of the warehouse: a dashboard, a report, a "
        "regulatory filing. Where a restatement is felt.",
        body="""
Conventional lineage stops at the tables, which means it cannot answer the question
asked during an incident: what have we already published from this number?

Sinks are terminal nodes for dashboards, reports, and filings. `fathom impact
--dataset X` names every published artefact downstream of a dataset, and exits
non-zero when one of them is a regulatory filing — because that is a different
conversation from a stale dashboard.

What to do: declare sinks in fathom.yml for the artefacts that matter. Nothing can
infer that a particular dashboard feeds a filing.
""",
        see_also=("plan", "contract"),
        doc="docs/guide/value.md",
    ),
    "erasure": Topic(
        name="erasure",
        summary="Locating a subject's data in every derived table, and what would "
        "actually destroy it. Dry run by default.",
        body="""
Deleting one subject from a lakehouse is a rewrite-the-world operation until you
know which files in which derived tables hold their rows. The graph makes it a
bounded question.

Erasure carries the mirror of the planner's invariant: it may under-delete and
refuse, but must never over-delete. So it dry-runs by default, and refuses outright
on WORM or Object Lock storage rather than reporting success it cannot deliver.

A proof needs a secret salt. Identifiers are low-entropy, so an unsalted digest
identifies the subject as well as the raw value does — and the proof is the artifact
handed to people who must not learn who they were.

What to do: `fathom erase --subject ... --key-column ... --origin ...` prints the
plan. It never writes. Note that erasure does not remove a subject from a trained
model; it names every model that retains them and what would discharge that.
""",
        see_also=("soundness", "sink"),
        doc="docs/guide/erase.md",
    ),
    "selector": Topic(
        name="selector",
        summary="dbt's selection syntax, resolved against the graph: `+model+`, "
        "`2+model`, `@model`, `tag:pii`.",
        body="""
Anywhere a subset of the graph is wanted, it is named the way your team already
names one:

  model         just that dataset
  +model        the dataset and everything upstream of it
  model+        the dataset and everything downstream
  +model+       both directions
  2+model       two hops upstream, and no further
  @model        the dataset, its ancestors, and everything those ancestors feed
  tag:pii       every dataset carrying that tag
  ns:snowflake  every dataset in that namespace

Combine with spaces for a union.

What to do: pass `--select` to `fathom lineage`. Start wide and narrow — an empty
result usually means the name did not resolve rather than that nothing matched.
""",
        see_also=("plan",),
        doc="docs/guide/concepts.md",
    ),
    "contract": Topic(
        name="contract",
        summary="What one team promised another about a dataset, checked against "
        "what is currently true.",
        body="""
A contract names the producer, the consumers, the columns that must be present, and
the staleness ceiling. `fathom contracts` verifies each one against the latest
profile and exits non-zero on a breach.

The part conventional testing misses is the consumer list. A failing test says a
column vanished; a breached contract says who was promised it and is therefore owed
a conversation. Those are different artifacts, and only one of them prevents the
incident from being discovered by the person downstream.

What to do: declare a `contracts:` block in fathom.yml for the datasets other teams
depend on.
""",
        see_also=("sink", "drift"),
        doc="docs/guide/contracts.md",
    ),
    "seasonality": Topic(
        name="seasonality",
        summary="A baseline bucketed by a cycle, for data where 1,000 rows is normal "
        "on Monday and an anomaly on Saturday.",
        body="""
A single flat band learned across a weekly cycle is wide enough to admit Tuesday's
floor and Sunday's ceiling — which is to say wide enough to catch nothing. Narrow it
to Tuesday and it fires every weekend. Teams resolve this by muting the check, and
the real failure is not a wrong alert but an absent one.

`fathom seasonal` learns a band per bucket, and reports how much of the variation
the cycle actually explains, so reaching for it stays a decision. A bucket with too
few observations is left unmodelled and not checked: two Sundays is not a baseline
for Sunday, and an invented band carries the same authority as a real one.

What to do: check the reported strength first. Below about 20%, the flat bound from
`fathom check` is the better tool, with less machinery behind it.
""",
        see_also=("drift", "profile"),
        doc="docs/guide/check.md",
    ),
    "store": Topic(
        name="store",
        summary="Where the graph, the profiles, and the history are kept. SQLite by "
        "default, behind a protocol.",
        body="""
Two durable artifacts need somewhere to live: the dependency graph and the profile
history. The default is a SQLite file at `.fathom/fathom.db`, set by the `store:`
key in fathom.yml or overridden with `--store`.

SQLite is a default rather than a premise — persistence sits behind a protocol, so a
team that needs the artifacts shared across CI runners can back it with something
else without touching anything above it.

What to do: commit fathom.yml; do not commit the store. It is a cache of things that
can be rebuilt by `fathom ingest` and `fathom profile`, and it will conflict.
""",
        see_also=("plan", "profile"),
        doc="docs/guide/configuration.md",
    ),
}

# Spellings people actually type, mapped to the canonical topic.
_ALIASES: dict[str, str] = {
    "mapping": "partition-mapping",
    "partition_mapping": "partition-mapping",
    "partition": "partition-mapping",
    "mappings": "partition-mapping",
    "widened": "widening",
    "widen": "widening",
    "sound": "soundness",
    "invariant": "soundness",
    "shadow": "shadow-mode",
    "shadow_mode": "shadow-mode",
    "grains": "grain",
    "granularity": "grain",
    "seeds": "seed",
    "dirty": "seed",
    "planning": "plan",
    "capability": "capabilities",
    "adapter": "capabilities",
    "adapters": "capabilities",
    "sinks": "sink",
    "erase": "erasure",
    "selectors": "selector",
    "select": "selector",
    "contracts": "contract",
    "seasonal": "seasonality",
    "cycle": "seasonality",
    "profiles": "profile",
    "profiling": "profile",
    "check": "drift",
    "coverage-ratio": "coverage",
    "unbound": "unbounded",
}


def titles() -> list[str]:
    """Every topic name, sorted — what `fathom explain` lists with no argument."""
    return sorted(TOPICS)


def lookup(name: str) -> Topic | None:
    """One topic by name or common alias, or None when nothing matches.

    Example:
        >>> lookup("widened").name
        'widening'
        >>> lookup("nonsense") is None
        True
    """
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    if key in TOPICS:
        return TOPICS[key]
    aliased = _ALIASES.get(key) or _ALIASES.get(key.replace("-", "_"))
    return TOPICS.get(aliased) if aliased else None
