"""Renderers for the newer artifacts, and the caveats they must not drop.

Each of these reports has a qualifying sentence its module refuses to be read
without. A Markdown table that drops it turns "no reads observed in 90 days" into
"unused", which is exactly the reading the underlying module declines to support.
These tests exist mostly to keep those sentences in the output.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping, TimeWindow
from fathom.core.types import ColumnRef, DatasetId, KeyPredicate, PartitionField, PartitionSpec
from fathom.govern import contracts
from fathom.govern import reidentification as reid
from fathom.govern.policy import Label
from fathom.graph import Edge, Graph, history, sinks
from fathom.graph.plan import lifetime
from fathom.graph.plan.cost import CostModel
from fathom.observe import completeness, usage
from fathom.observe.profile import ColumnProfile, Profile
from fathom.report import render

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.monthly")
DEAD = DatasetId("duckdb", "gold.dead")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MARCH = datetime(2026, 3, 1, tzinfo=UTC)


def days(*ns: int) -> list[KeyPredicate]:
    return [KeyPredicate.of(dt=datetime(2026, 3, n)) for n in ns]


# -- completeness --------------------------------------------------------------


def test_a_complete_dataset_renders_in_one_line():
    result = completeness.report(
        RAW, DAY, days(1, 2), start=datetime(2026, 3, 1), end=datetime(2026, 3, 2)
    )
    assert "complete" in render.completeness_to_markdown(result)


def test_missing_runs_render_as_a_table():
    result = completeness.report(
        RAW, DAY, days(1, 5), start=datetime(2026, 3, 1), end=datetime(2026, 3, 5)
    )
    text = render.completeness_to_markdown(result)
    assert "| Run |" in text
    assert "3 of 5 partition(s) missing" in text


def test_the_inferred_domain_caveat_survives_rendering():
    spec = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))
    present = [KeyPredicate.of(dt=datetime(2026, 3, 1), region="eu")]
    result = completeness.report(
        RAW, spec, present, start=datetime(2026, 3, 1), end=datetime(2026, 3, 2)
    )
    assert "never appeared cannot be reported missing" in render.completeness_to_markdown(result)


# -- usage ---------------------------------------------------------------------


def test_usage_renders_busiest_first():
    stats = usage.summarize(
        [
            usage.ReadEvent(GOLD, "ana", MARCH),
            usage.ReadEvent(GOLD, "ben", MARCH),
            usage.ReadEvent(RAW, "ana", MARCH),
        ]
    )
    text = render.usage_to_markdown(stats)
    assert text.index("gold.monthly") < text.index("raw.events")


def test_usage_separates_people_from_jobs():
    stats = usage.summarize(
        [usage.ReadEvent(GOLD, "airflow_worker", MARCH), usage.ReadEvent(GOLD, "ana", MARCH)]
    )
    assert "| People |" in render.usage_to_markdown(stats)


def test_retirement_keeps_the_review_list_caveat():
    """The single most important sentence in this module's output."""
    candidate = usage.RetirementCandidate(DEAD, "no reads observed", timedelta(days=90), 0)
    text = render.retirement_to_markdown([candidate])
    assert "not the same as no reads" in text
    assert "review list, not a delete list" in text


def test_no_retirement_candidates_renders_cleanly():
    assert render.retirement_to_markdown([]) == "**Retirement candidates** — none."


# -- re-identification ---------------------------------------------------------


def test_risk_never_renders_as_a_clean_bill():
    made = Profile(dataset=RAW, row_count=10, columns=(ColumnProfile("colour", "string"),))
    text = render.risk_to_markdown(reid.assess(made, {}))
    assert "no risk proven" in text
    assert "not that the data is safe" in text


def test_a_proven_risk_renders_its_columns():
    made = Profile(
        dataset=RAW,
        row_count=1000,
        columns=(
            ColumnProfile("dob", "string", distinct_estimate=800),
            ColumnProfile("zip", "string", distinct_estimate=400),
        ),
    )
    labels = {
        ColumnRef(RAW, "dob"): {Label("date_of_birth", 0.8)},
        ColumnRef(RAW, "zip"): {Label("postal_address", 0.8)},
    }
    text = render.risk_to_markdown(reid.assess(made, labels))
    assert "quasi_identifier_set" in text
    assert "dob, zip" in text


def test_unmeasurable_columns_are_named_in_the_render():
    made = Profile(
        dataset=RAW,
        row_count=1000,
        columns=(ColumnProfile("dob", "string"), ColumnProfile("zip", "string")),
    )
    labels = {
        ColumnRef(RAW, "dob"): {Label("date_of_birth", 0.8)},
        ColumnRef(RAW, "zip"): {Label("postal_address", 0.8)},
    }
    assert "Not measurable" in render.risk_to_markdown(reid.assess(made, labels))


# -- contracts -----------------------------------------------------------------


def test_a_breach_renders_who_it_is_owed_to():
    contract = contracts.Contract(GOLD, "platform", consumers=("finance",), columns=("amount",))
    result = contracts.verify(contract, profile=Profile(dataset=GOLD))
    text = render.contract_report_to_markdown(result)
    assert "finance" in text
    assert "missing_column" in text


def test_a_met_contract_renders_as_met():
    contract = contracts.Contract(GOLD, "platform", consumers=("finance",))
    assert "met" in render.contract_report_to_markdown(contracts.verify(contract))


def test_unchecked_promises_are_rendered():
    contract = contracts.Contract(GOLD, "platform", columns=("amount",))
    assert "Not checked" in render.contract_report_to_markdown(contracts.verify(contract))


# -- cost and value ------------------------------------------------------------


def test_lifetime_renders_most_expensive_first():
    totals = lifetime.accumulate(
        [
            lifetime.RunRecord(GOLD, MARCH, partitions=100),
            lifetime.RunRecord(RAW, MARCH, partitions=1),
        ],
        CostModel(price_per_partition=1.0),
    )
    text = render.lifetime_to_markdown(totals)
    assert text.index("gold.monthly") < text.index("raw.events")


def test_value_keeps_the_measured_versus_observed_caveat():
    totals = lifetime.accumulate(
        [lifetime.RunRecord(GOLD, MARCH, partitions=100)], CostModel(price_per_partition=1.0)
    )
    text = render.value_to_markdown(lifetime.value(totals, {}, threshold=10.0))
    assert "Cost is measured; usage is observed" in text
    assert "1 dataset(s) unread and above the threshold" in text


# -- history -------------------------------------------------------------------


def test_history_marks_the_unsafe_revision():
    def build(hi: int) -> Graph:
        g = Graph()
        g.add_dataset(RAW, DAY)
        g.add_dataset(GOLD, DAY)
        g.add_edge(
            Edge(RAW, GOLD, PartitionMapping.of(dt=TimeWindow("dt", 0, hi, Grain.DAY, Grain.DAY)))
        )
        return g

    log = history.History()
    wide = build(6)
    history.record(log, wide, author="ana", at=MARCH)
    history.record(log, build(1), author="ben", note="perf", at=MARCH, previous=wide)
    text = render.history_to_markdown(log)
    assert "**unsafe**" in text
    assert "ben" in text


def test_an_empty_history_renders_a_zero_count():
    assert "0 revision(s)" in render.history_to_markdown(history.History())


# -- restatement ---------------------------------------------------------------


def test_restatement_marks_the_regulatory_artefacts():
    g = Graph()
    g.add_dataset(GOLD, DAY)
    tenk = sinks.filing("10-K/2026", regulator="sec")
    sinks.record_publication(g, tenk, [GOLD])
    sinks.record_publication(g, sinks.dashboard("exec", tool="looker"), [GOLD])
    text = render.restatement_to_markdown(sinks.restatement_impact(g, GOLD))
    assert "filing 10-K/2026 on sec" in text
    assert "| Regulatory |" in text


def test_restatement_refuses_to_judge_materiality():
    g = Graph()
    g.add_dataset(GOLD, DAY)
    sinks.record_publication(g, sinks.dashboard("exec"), [GOLD])
    text = render.restatement_to_markdown(sinks.restatement_impact(g, GOLD))
    assert "not the same as material" in text


def test_nothing_published_renders_in_one_line():
    g = Graph()
    g.add_dataset(GOLD, DAY)
    text = render.restatement_to_markdown(sinks.restatement_impact(g, GOLD))
    assert "nothing published downstream" in text


# -- emit ----------------------------------------------------------------------


def test_a_sink_carries_a_published_artefact_facet():
    """A catalog that receives a filing and a staging table as identical nodes has
    lost the one property that made the filing worth tracking."""
    from fathom.report import emit

    facet = emit.sink_facet(sinks.filing("10-K/2026", regulator="sec"))
    assert facet["fathom_publishedArtefact"]["kind"] == "filing"
    assert facet["fathom_publishedArtefact"]["regulatory"] is True
    assert facet["fathom_publishedArtefact"]["terminal"] is True


def test_a_dashboard_is_not_marked_regulatory():
    from fathom.report import emit

    facet = emit.sink_facet(sinks.dashboard("exec", tool="looker"))
    assert facet["fathom_publishedArtefact"]["regulatory"] is False


def test_a_table_gets_no_artefact_facet():
    from fathom.report import emit

    assert emit.sink_facet(RAW) == {}


def test_dataset_facets_include_the_sink_marker():
    from fathom.report import emit

    g = Graph()
    g.add_dataset(GOLD, DAY)
    tenk = sinks.filing("10-K/2026", regulator="sec")
    sinks.record_publication(g, tenk, [GOLD])
    assert "fathom_publishedArtefact" in emit.dataset_facets(g, tenk)


def test_a_partitioned_table_keeps_its_spec_facet_and_gains_no_other():
    from fathom.report import emit

    g = Graph()
    g.add_dataset(GOLD, DAY)
    facets = emit.dataset_facets(g, GOLD)
    assert "fathom_partitionSpec" in facets
    assert "fathom_publishedArtefact" not in facets
