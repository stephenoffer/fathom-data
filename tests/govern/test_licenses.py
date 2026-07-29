"""Licences combining most-restrictive-first along lineage."""

from __future__ import annotations

import pytest

from fathom.ai import assets, training
from fathom.core.grains import Grain
from fathom.core.partitions import PartitionMapping
from fathom.core.types import DatasetId, PartitionField, PartitionSpec
from fathom.govern import licenses
from fathom.graph import Edge, Graph

SCRAPED = DatasetId("s3://eu-west-1-lake", "corpus.scraped")
INTERNAL = DatasetId("s3://eu-west-1-lake", "raw.users")
JOINED = DatasetId("s3://us-east-1-lake", "gold.combined")
MODEL = assets.model("assistant", registry="internal")

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))


@pytest.fixture
def graph() -> Graph:
    """Two sources joining into one derived table that then trains a model."""
    g = Graph()
    g.add_dataset(SCRAPED, DAY)
    g.add_dataset(INTERNAL, DAY)
    g.add_dataset(JOINED, DAY)
    g.add_edge(Edge(SCRAPED, JOINED, PartitionMapping.identity(DAY), evidence="sql:1"))
    g.add_edge(Edge(INTERNAL, JOINED, PartitionMapping.identity(DAY), evidence="sql:1"))
    run = training.TrainingRun(model=MODEL, code_version="abc")
    run.add_input(JOINED, snapshot="v1")
    training.record_training_run(g, run)
    return g


# -- licences ------------------------------------------------------------------


def test_parse_license_never_guesses_permissive():
    assert licenses.parse_license("MIT").commercial is True
    assert licenses.parse_license("CC-BY-NC").commercial is False
    unknown = licenses.parse_license("SomethingCustom-1.0")
    assert unknown.is_unknown
    assert unknown.commercial is None


def test_combine_takes_the_most_restrictive_term():
    mit = licenses.parse_license("MIT")
    nc = licenses.parse_license("CC-BY-NC")
    combined = licenses.combine([mit, nc])
    assert combined.commercial is False
    assert combined.derivatives is True
    assert combined.attribution is True


def test_one_unknown_source_makes_the_combination_unknown():
    combined = licenses.combine([licenses.parse_license("MIT"), licenses.parse_license("custom")])
    assert combined.is_unknown


def test_share_alike_propagates_and_blocks_a_permissive_target():
    sa = licenses.parse_license("CC-BY-SA")
    mit = licenses.parse_license("MIT")
    assert licenses.combine([sa, mit]).share_alike
    assert not licenses.is_compatible(sa, mit)
    assert licenses.is_compatible(mit, sa)


def test_effective_license_walks_upstream(graph):
    declared = {
        SCRAPED: licenses.parse_license("CC-BY-NC"),
        INTERNAL: licenses.parse_license("internal"),
    }
    effective = licenses.effective_license(graph, MODEL, declared)
    assert effective.commercial is False
    assert licenses.commercial_use_allowed(graph, MODEL, declared) is False
    assert licenses.restrictive_sources(graph, MODEL, declared) == [SCRAPED]


def test_license_report_names_the_blocker(graph):
    declared = {SCRAPED: licenses.parse_license("CC-BY-NC")}
    report = licenses.report(graph, MODEL, declared, intended_commercial=True)
    assert not report.is_clear
    assert any("commercial use is forbidden" in note for note in report.blockers)
    assert str(SCRAPED) in report.blockers[0]


def test_no_derivatives_blocks_training(graph):
    declared = {SCRAPED: licenses.parse_license("CC-BY-ND")}
    report = licenses.report(graph, MODEL, declared)
    assert any("derivative works are forbidden" in note for note in report.blockers)
    assert licenses.training_permitted(graph, MODEL, declared) is False


def test_propagate_resolves_every_dataset(graph):
    declared = {SCRAPED: licenses.parse_license("CC-BY-NC")}
    resolved = licenses.propagate(graph, declared)
    assert resolved[JOINED].commercial is False
    assert resolved[MODEL].commercial is False


def test_attribution_manifest_is_generated_from_lineage(graph):
    declared = {SCRAPED: licenses.parse_license("CC-BY")}
    text = licenses.attribution_manifest(graph, MODEL, declared)
    assert str(SCRAPED) in text
    assert "require" in text

    quiet = licenses.attribution_manifest(graph, MODEL, {SCRAPED: licenses.parse_license("CC0")})
    assert "No upstream source requires attribution" in quiet


def test_unlicensed_lists_what_nobody_checked(graph):
    assert licenses.unlicensed(graph, {}) == sorted(graph.datasets, key=str)
    declared = {SCRAPED: licenses.parse_license("MIT"), INTERNAL: licenses.parse_license("MIT")}
    assert licenses.unlicensed(graph, declared) == []


def test_an_unrecorded_target_licence_is_not_permission():
    """Not declaring your terms must not beat declaring them honestly.

    Requiring the target to be explicitly commercial before rejecting a
    non-commercial source rewarded the team that never filled the field in.
    """
    non_commercial = licenses.parse_license("cc-by-nc")
    undeclared = licenses.License(name="our-model")  # commercial is None

    assert licenses.is_compatible(non_commercial, undeclared) is False
    assert licenses.is_compatible(non_commercial, licenses.parse_license("mit")) is False
    # An explicitly non-commercial target is still a legitimate home for it.
    assert licenses.is_compatible(non_commercial, non_commercial) is True
