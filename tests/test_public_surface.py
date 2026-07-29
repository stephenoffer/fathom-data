"""Exercises public API that nothing else reaches.

A function no test ever calls is a function whose first real execution happens in
somebody's pipeline. These are not deep behavioural tests — those live beside the
module they cover — but every one of them asserts something true rather than merely
importing the name.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import (
    ColumnRef,
    DatasetId,
    KeyPredicate,
    PartitionField,
    PartitionSpec,
)
from fathom.graph import Edge, Graph

RAW = DatasetId("duckdb", "raw.events")
SILVER = DatasetId("duckdb", "silver.events")
GOLD = DatasetId("duckdb", "gold.monthly")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))


@pytest.fixture
def graph() -> Graph:
    g = Graph()
    g.add_dataset(RAW, DAY)
    g.add_dataset(SILVER, DAY)
    g.add_dataset(GOLD, MONTH)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping.identity(DAY), columns=(("amount", "amount"),)))
    g.add_edge(
        Edge(SILVER, GOLD, PartitionMapping.rollup(DAY, MONTH), columns=(("amount", "revenue"),))
    )
    return g


@pytest.fixture
def plan(graph: Graph):
    return graph.invalidate({RAW: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})


# -- core utilities ------------------------------------------------------------


def test_markdown_heading_levels():
    from fathom.core.util import markdown as md

    assert md.heading("Title", 1).startswith("# ")
    assert md.heading("Sub", 3).startswith("### ")


def test_chunked_splits_without_losing_items():
    from fathom.adapters.sql_runner import chunked

    batches = list(chunked(list(range(10)), 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert [item for b in batches for item in b] == list(range(10))


# -- graph plan surface --------------------------------------------------------


def test_schedule_renders_waves_to_mermaid_and_lists_their_partitions(graph, plan):
    from fathom.graph.plan import schedule

    built = schedule.schedule(graph, plan)
    assert schedule.to_mermaid(built).startswith("flowchart")
    first = schedule.partitions_in_wave(built, 0)
    assert first, "the first wave must contain the seeded dataset"


def test_cost_helpers_rank_and_compare(graph, plan):
    from fathom.graph.plan.cost import CostModel, compare_models, most_expensive

    cheap = CostModel(price_per_partition=1.0)
    dear = CostModel(price_per_partition=10.0)

    ranked = most_expensive(plan, dear, limit=2)
    assert ranked and ranked[0][1] > 0

    compared = compare_models(plan, [("cheap", cheap), ("dear", dear)])
    assert compared["dear"] > compared["cheap"]


def test_partition_counts_from_profiles():
    from fathom.graph.plan.cost import partition_counts_from

    counts = partition_counts_from(
        {
            RAW: [
                KeyPredicate.of(dt=datetime(2026, 3, 14)),
                KeyPredicate.of(dt=datetime(2026, 3, 15)),
            ],
            GOLD: [KeyPredicate.of(dt=datetime(2026, 3, 1))],
        }
    )
    assert counts[RAW] == 2
    assert counts[GOLD] == 1


# -- observe -------------------------------------------------------------------


def test_schema_diff_names_added_removed_and_retyped_columns():
    from fathom.observe import schema
    from fathom.observe.profile import ColumnProfile, Profile

    before = Profile(
        dataset=RAW,
        row_count=100,
        columns=(ColumnProfile("id", "string", 100), ColumnProfile("amount", "int64", 100)),
    )
    after = Profile(
        dataset=RAW,
        row_count=100,
        columns=(
            ColumnProfile("id", "string", 100),
            ColumnProfile("amount", "double", 100),
            ColumnProfile("region", "string", 100),
        ),
    )

    change, rendered = schema.diff_profiles(before, after)
    assert list(change.added) == ["region"]
    assert list(change.removed) == []
    assert list(change.retyped) == ["amount"]
    assert not change.is_empty
    assert "amount" in change.summary()
    assert any("amount" in line for line in rendered)


def test_breaking_schema_changes_are_the_removals_and_retypes():
    from fathom.observe import schema
    from fathom.observe.profile import ColumnProfile, Profile

    before = Profile(dataset=RAW, row_count=10, columns=(ColumnProfile("gone", "string", 10),))
    after = Profile(dataset=RAW, row_count=10, columns=())
    findings = schema.breaking_schema_changes(before, after)
    assert findings
    assert schema.worst_severity(findings) is not None


def test_freshness_helpers_over_a_chain(graph):
    from fathom.observe import freshness

    now = datetime(2026, 3, 20, tzinfo=UTC)
    built = {RAW: datetime(2026, 3, 1, tzinfo=UTC), SILVER: now, GOLD: now}

    ages = freshness.effective_freshness(graph, built, now=now)
    # gold is only as fresh as raw, which last landed on the 1st.
    assert ages[GOLD] == timedelta(days=19)

    stale = freshness.stale_closure(graph, GOLD, built, max_age=timedelta(days=5), now=now)
    assert RAW in stale


def test_sla_violations_and_expected_next_build(graph):
    from fathom.observe import freshness

    now = datetime(2026, 3, 20, tzinfo=UTC)
    sla = freshness.SLA(
        dataset=GOLD, max_age=timedelta(days=1), expected_interval=timedelta(days=1)
    )
    built = {RAW: datetime(2026, 3, 1, tzinfo=UTC), SILVER: now, GOLD: now}

    assert freshness.sla_violations(graph, [sla], built, now=now)
    assert freshness.expected_next_build(sla, built) == now + timedelta(days=1)


def test_quality_expectation_constructors_build_what_they_say():
    from fathom.observe import quality

    assert quality.null_rate_below("amount", 0.1).kind == "null_rate_below"
    assert quality.min_above("amount", 0.0).kind == "min_above"
    assert quality.column_count_between(1, 5).kind == "column_count_between"
    assert quality.name_matches("id", r"^\d+$").kind == "name_matches"
    assert quality.not_empty().kind == "row_count_between"


# -- govern --------------------------------------------------------------------


def test_license_reporting_helpers(graph):
    from fathom.govern import licenses

    assert "MIT" in licenses.known_licenses()

    declared = {RAW: licenses.parse_license("cc-by"), SILVER: licenses.parse_license("mit")}
    assert licenses.attribution_required(graph, GOLD, declared)
    breakdown = licenses.license_breakdown(declared)
    assert breakdown["CC-BY-4.0"] == 1

    report = licenses.report(graph, GOLD, declared)
    assert report.summary()


def test_label_diff_reports_newly_appearing_pii():
    from fathom.govern import diff as gdiff
    from fathom.govern.policy import Label

    before = {ColumnRef(RAW, "note"): {Label("monetary_amount", 0.6, "inferred")}}
    after = {
        ColumnRef(RAW, "note"): {Label("monetary_amount", 0.6, "inferred")},
        ColumnRef(RAW, "email"): {Label("pii", 0.9, "inferred")},
    }
    d = gdiff.diff_labels(before, after)
    assert not d.is_empty
    assert d.new_pii
    assert "pii" in d.summary()


def test_consent_breakdown_and_permitted_datasets(graph):
    from fathom.govern import consent

    scopes = {
        RAW: consent.ConsentScope(dataset=RAW, purposes=frozenset({consent.Purpose.TRAINING})),
        SILVER: consent.ConsentScope(
            dataset=SILVER, purposes=frozenset({consent.Purpose.TRAINING})
        ),
    }
    breakdown = consent.purposes_breakdown(scopes)
    assert breakdown[consent.Purpose.TRAINING] == 2
    assert SILVER in consent.training_permitted_datasets(graph, scopes)


# -- emit and compliance -------------------------------------------------------


def test_emit_helpers(graph):
    from fathom.report import emit

    namespaces = emit.to_marquez_namespaces(graph)
    assert {"name": "duckdb", "ownerName": "fathom"} in namespaces

    payload = emit.partition_payload([KeyPredicate.of(dt=datetime(2026, 3, 14))])
    assert payload


def test_erasure_attestation_names_the_origin(graph):
    from fathom.report import compliance

    text = compliance.erasure_attestation(graph, RAW, subject_digest="deadbeefcafe")
    assert "deadbeef" in text


# -- selectors set algebra -----------------------------------------------------


def test_selection_set_algebra(graph):
    from fathom.graph import selectors

    both = selectors.parse("raw.events silver.events")
    assert not both.is_empty

    resolved = set(selectors.resolve(graph, "raw.events silver.events"))
    assert resolved == {RAW, SILVER}

    intersected = set(selectors.resolve(graph, "+gold.monthly,name:silver.*"))
    assert intersected == {SILVER}


def test_select_columns_filters_by_label(graph):
    from fathom.govern.policy import Label, tag_index
    from fathom.graph import selectors

    labels = {ColumnRef(SILVER, "amount"): {Label("pii", 0.9, "inferred")}}
    # `tag:` selection acts on datasets, so the label set is flattened first.
    chosen = selectors.select_columns(graph, "tag:pii", labels=tag_index(labels))
    assert chosen == [ColumnRef(SILVER, "amount")]


# -- graph diff / schedule / cost ----------------------------------------------


def test_graph_diff_reports_changed_columns(graph):
    from fathom.graph import diff

    rewired = Graph()
    for ds in graph.datasets:
        rewired.add_dataset(ds, graph.spec(ds))
    rewired.add_edge(
        Edge(RAW, SILVER, PartitionMapping.identity(DAY), columns=(("amt", "amount"),))
    )
    rewired.add_edge(
        Edge(SILVER, GOLD, PartitionMapping.rollup(DAY, MONTH), columns=(("amount", "revenue"),))
    )

    d = diff.diff_graphs(graph, rewired)
    assert any(change.columns_changed for change in d.changed_edges)


def test_schedule_wave_lists_its_datasets(graph, plan):
    from fathom.graph.plan import schedule

    built = schedule.schedule(graph, plan)
    assert built.waves
    assert built.waves[0].datasets


def test_attributed_cost_splits_a_consumers_bill(graph, plan):
    from fathom.graph.plan.cost import CostModel, attributed_cost

    attributed = attributed_cost(graph, GOLD, CostModel(price_per_partition=2.0))
    assert attributed >= 0.0


# -- ingest --------------------------------------------------------------------


def test_graph_from_lineage_builds_edges_from_native_events():
    from fathom.adapters.base import LineageEvent
    from fathom.ingest.events import graph_from_lineage

    events = [LineageEvent(src=RAW, dst=GOLD, columns=(("amount", "revenue"),), evidence="native")]
    result = graph_from_lineage(events, specs={RAW: DAY, GOLD: MONTH})
    assert result.edges == 1
    edge = result.graph.edges[0]
    assert edge.src == RAW and edge.dst == GOLD
    # Two real specs on both ends means the mapping can be a rollup, not unbounded.
    assert not edge.mapping.is_unbounded


# -- store ---------------------------------------------------------------------


def test_store_profile_inventory_helpers():
    from fathom.observe.profile import Profile
    from fathom.store import Store

    store = Store(":memory:")
    key = KeyPredicate.of(dt=datetime(2026, 3, 14))
    store.save_profile(Profile(dataset=RAW, partition=key, row_count=5))

    assert store.profiled_partitions(RAW) == [key]
    assert store.last_profiled(RAW) is not None

    store.set_label(RAW, "email", "pii", confidence=0.9, origin="inferred")
    every = store.all_labels()
    assert every[RAW]["email"][0][0] == "pii"


# -- govern / ai summaries -----------------------------------------------------


def test_erasure_plan_and_proof_render_to_markdown(graph):
    from fathom.core.types import Capabilities, ChangeSource, ErasureMode, LineageSource
    from fathom.govern.erasure import ErasureRequest, apply_erasure, plan_erasure
    from fathom.report import render

    caps = {
        ds: Capabilities(
            lineage=LineageSource.DECLARED,
            change=ChangeSource.WATERMARK,
            erasure=ErasureMode.REWRITE,
        )
        for ds in graph.datasets
    }
    request = ErasureRequest(subject="u1", key_column="user_id", origin=RAW, reference="DSR-9")
    plan = plan_erasure(graph, request, capabilities=caps)

    assert plan.actionable
    markdown = render.erasure_plan_to_markdown(plan)
    assert "DSR-9" in markdown
    # Never a digest it could only compute unsalted.
    assert "subject `" not in markdown

    proof = apply_erasure(plan, {}, salt="org-secret")
    assert "subject digest" in render.proof_to_markdown(proof).lower()


def test_license_violations_names_a_non_commercial_source(graph):
    from fathom.govern import licenses

    declared = {RAW: licenses.parse_license("cc-by-nc")}
    found = licenses.violations(graph, declared, commercial_datasets=[GOLD])
    assert found


# -- adapter registry and declared catalog -------------------------------------


def test_adapter_registry_round_trip_and_duplicate_refusal():
    from fathom.adapters.base import get_adapter, register, registered

    @register("surface-probe")
    class Probe:
        name = "surface-probe"

    assert get_adapter("surface-probe") is Probe
    assert "surface-probe" in registered()

    with pytest.raises(ValueError):

        @register("surface-probe")
        class Other:
            name = "surface-probe"

    with pytest.raises(KeyError):
        get_adapter("no-such-adapter")


def test_declared_catalog_answers_only_what_it_was_told():
    from fathom.adapters.base import DeclaredCatalog

    catalog = DeclaredCatalog()
    catalog.declare(RAW, DAY)

    assert catalog.describe_partitioning(RAW) == DAY
    assert not catalog.describe_partitioning(GOLD).fields  # never declared

    changes = catalog.changed(RAW, "token-1")
    assert changes.is_empty, "a hand-declared catalog cannot detect change"
    assert changes.token == "token-1", "the cursor must not appear to advance"


def test_object_storage_erase_files_removes_only_what_it_is_given(tmp_path):
    from fathom.adapters.storage.objects import ObjectStorage
    from fathom.core.ids import normalize

    keep = tmp_path / "keep.parquet"
    drop = tmp_path / "drop.parquet"
    keep.write_bytes(b"x")
    drop.write_bytes(b"y")

    storage = ObjectStorage()
    removed = storage.erase_files(normalize(str(tmp_path)), [str(drop)])
    assert removed == 1
    assert keep.exists() and not drop.exists()


def test_local_filesystem_info_and_delete(tmp_path):
    from fathom.adapters.fs import filesystem_for

    target = tmp_path / "obj.parquet"
    target.write_bytes(b"abcd")
    fs = filesystem_for(str(target))

    info = fs.info(str(target))
    assert info.size == 4

    fs.delete(str(target))
    assert not target.exists()


def test_iceberg_declare_records_a_spec_without_a_catalog():
    pytest.importorskip("pyiceberg")
    from fathom.adapters.catalogs.iceberg import IcebergCatalog

    catalog = IcebergCatalog()
    catalog.declare(RAW, DAY)
    assert catalog.describe_partitioning(RAW) == DAY


# -- duckdb --------------------------------------------------------------------


def test_duckdb_reports_partitions_present_and_change(tmp_path):
    pytest.importorskip("duckdb")
    from fathom.adapters.engines.duckdb import DuckDBEngine

    engine = DuckDBEngine(database=str(tmp_path / "w.db"))
    dataset = DatasetId("duckdb", "main.events")
    engine.register_model(dataset, "SELECT 1", DAY)
    engine.connect().execute(
        "CREATE TABLE main.events AS SELECT TIMESTAMP '2026-03-14' AS dt, 1 AS amount"
    )

    present = engine.partitions_present(dataset)
    assert present, "a table with one day must report one partition"

    changes = engine.changed(dataset, None)
    assert changes is not None
    engine.close()


# -- project -------------------------------------------------------------------


def test_project_loads_from_a_config_and_accepts_a_runner(tmp_path):
    from fathom.cli.project import Project

    (tmp_path / "fathom.yml").write_text(
        "version: 1\nstore: .fathom/store.db\nsystem: duckdb\n"
        "datasets:\n  - name: raw.events\n    partition:\n      - {field: dt, grain: day}\n"
    )
    project = Project.load(tmp_path / "fathom.yml")
    try:
        assert project.config.system == "duckdb"
        assert project.config.sql_for(project.config.datasets[0]) is None
        sentinel = object()
        project.register_runner("duckdb", sentinel)
        assert project.runners["duckdb"] is sentinel
    finally:
        project.close()


# -- remaining summaries -------------------------------------------------------


def test_ai_summaries_state_their_own_gaps(graph):
    from fathom.ai import agents, assets, attribution, evals, features, prompts, training, vectors

    model = assets.model("scorer", registry="internal")
    bom = training.data_bill_of_materials(graph, model)
    assert isinstance(bom.is_complete, bool)
    assert bom.summary()
    assert isinstance(bom.direct, list)
    assert isinstance(bom.transitive, list)

    run = agents.AgentRun(agent=assets.agent("a"), run_id="r")
    assert run.summary()

    view = features.FeatureView(dataset=assets.feature_view("v"), entity="user")
    assert view.name
    skew = features.SkewReport(view=view.dataset)
    assert isinstance(skew.is_clean, bool)
    assert skew.summary()

    result = evals.EvalResult(model, assets.eval_set("holdout"), {"accuracy": 0.9})
    assert result.metric("accuracy") == 0.9
    assert result.metric("absent") is None

    template = prompts.PromptTemplate(dataset=assets.prompt("p"))
    template.commit("text")
    assert template.summary()

    plan = vectors.reindex_plan(assets.vector_index("i", store="pg"), indexed={}, current=[])
    assert plan.skipped == 0

    diagnosis = attribution.Diagnosis(target=RAW, target_column="amount")
    assert diagnosis.summary()
    assert diagnosis.best is None, "no causes recorded means nothing to blame"


def test_agent_unreviewed_writes_names_output_others_consume():
    from fathom.ai import agents, assets

    agent = assets.agent("triager")
    written = DatasetId("duckdb", "gold.agent_out")
    consumer = DatasetId("duckdb", "gold.dashboard")
    g = Graph()
    g.add_edge(Edge(agent, written, PartitionMapping()))
    g.add_edge(Edge(written, consumer, PartitionMapping()))
    assert agents.unreviewed_writes(g, agent) == [written]


def test_consent_report_summary(graph):
    from fathom.govern import consent

    scopes = {
        RAW: consent.ConsentScope(dataset=RAW, purposes=frozenset({consent.Purpose.ANALYTICS}))
    }
    report = consent.report(graph, GOLD, scopes, intended=[consent.Purpose.TRAINING])
    assert report.summary()
    # analytics-only consent upstream cannot authorise training downstream.
    assert not report.is_clear


def test_selection_union_and_intersect(graph):
    from fathom.graph import selectors

    a = selectors.parse("raw.events")
    b = selectors.parse("silver.events")
    assert set(selectors.resolve(graph, "raw.events silver.events")) == {RAW, SILVER}
    assert not a.is_empty and not b.is_empty


def test_the_last_of_the_public_surface(graph):
    """Small accessors that nothing else reaches."""
    from fathom.ai import assets, attribution, evals, training
    from fathom.graph import selectors
    from fathom.observe.profile import Finding, Severity

    # TrainingRun aggregates over its pins.
    run = training.TrainingRun(model=assets.model("m", registry="internal"))
    run.add_input(RAW, partitions=[KeyPredicate.of(dt=datetime(2026, 3, 14))], row_count=10)
    run.add_input(GOLD, row_count=5)
    assert run.datasets == [GOLD, RAW]
    assert run.total_rows == 15

    # An asset reference knows its own kind.
    ref = assets.parse_ref("model://internal/fraud.scorer@v3")
    assert ref.kind is assets.AssetKind.MODEL
    assert ref.version == "v3"

    # A cause reports the worst severity among its findings.
    cause = attribution.Cause(
        dataset=RAW,
        findings=[
            Finding("amount", "null_rate", Severity.WARN, "moved"),
            Finding("amount", "type_change", Severity.ERROR, "int -> string"),
        ],
    )
    assert cause.worst_severity is Severity.ERROR

    # A clean contamination report says so rather than staying silent.
    report = evals.ContaminationReport(
        model=assets.model("m", registry="internal"), eval_set=assets.eval_set("holdout")
    )
    assert "CLEAN" in report.summary()

    # Set algebra over selections.
    assert set(selectors.union({RAW}, {GOLD})) == {RAW, GOLD}
    assert set(selectors.intersect({RAW, GOLD}, {GOLD})) == {GOLD}
