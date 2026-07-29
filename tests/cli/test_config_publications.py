"""Publications declared in `fathom.yml`, and why they are declared rather than found."""

from __future__ import annotations

from pathlib import Path

import pytest

from fathom.cli.config import parse_config
from fathom.cli.project import Project
from fathom.core.errors import ConfigError
from fathom.core.types import DatasetId
from fathom.graph import sinks
from fathom.store.sqlite import Store

GOLD = DatasetId("duckdb", "gold.monthly")
SILVER = DatasetId("duckdb", "silver.revenue")


def config(body: str, root: Path):
    import yaml

    return parse_config(yaml.safe_load(body), root=root)


BASE = """\
version: 1
system: duckdb
datasets:
  - name: gold.monthly
  - name: silver.revenue
"""


# -- parsing -------------------------------------------------------------------


def test_a_publication_parses(tmp_path):
    parsed = config(
        BASE
        + """
publications:
  - name: revenue/exec
    kind: dashboard
    instance: looker
    inputs: [gold.monthly]
""",
        tmp_path,
    )
    (publication,) = parsed.publications
    assert publication.name == "revenue/exec"
    assert publication.kind == "dashboard"
    assert publication.inputs == (GOLD,)


def test_the_kind_defaults_to_dashboard(tmp_path):
    parsed = config(BASE + "\npublications:\n  - name: x\n    inputs: [gold.monthly]\n", tmp_path)
    assert parsed.publications[0].kind == "dashboard"


def test_a_single_input_may_be_a_bare_string(tmp_path):
    parsed = config(BASE + "\npublications:\n  - name: x\n    inputs: gold.monthly\n", tmp_path)
    assert parsed.publications[0].inputs == (GOLD,)


def test_every_kind_is_accepted(tmp_path):
    for kind in ("dashboard", "report", "filing", "export", "endpoint", "notebook"):
        parsed = config(
            BASE + f"\npublications:\n  - name: x\n    kind: {kind}\n    inputs: [gold.monthly]\n",
            tmp_path,
        )
        assert parsed.publications[0].kind == kind


def test_an_unknown_kind_is_rejected_with_the_valid_ones(tmp_path):
    with pytest.raises(ConfigError, match="unknown kind"):
        config(
            BASE + "\npublications:\n  - name: x\n    kind: poster\n    inputs: [gold.monthly]\n",
            tmp_path,
        )


def test_a_publication_with_no_inputs_is_rejected(tmp_path):
    """It would record nothing, which is worse than not being there."""
    with pytest.raises(ConfigError, match="records nothing"):
        config(BASE + "\npublications:\n  - name: x\n    kind: filing\n", tmp_path)


def test_a_publication_with_no_name_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="missing `name`"):
        config(BASE + "\npublications:\n  - kind: filing\n    inputs: [gold.monthly]\n", tmp_path)


def test_an_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        config(
            BASE + "\npublications:\n  - name: x\n    inputs: [gold.monthly]\n    owner: ana\n",
            tmp_path,
        )


def test_a_non_mapping_publication_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="must be a mapping"):
        config(BASE + "\npublications:\n  - just-a-string\n", tmp_path)


def test_inputs_resolve_against_the_configured_system(tmp_path):
    parsed = config(
        BASE + "\npublications:\n  - name: x\n    inputs: [gold.monthly, silver.revenue]\n",
        tmp_path,
    )
    assert parsed.publications[0].inputs == (GOLD, SILVER)


# -- reaching the graph --------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    (tmp_path / "fathom.yml").write_text(
        BASE
        + """
publications:
  - name: revenue/exec
    kind: dashboard
    instance: looker
    inputs: [gold.monthly]
  - name: 10-K/2026
    kind: filing
    instance: sec
    inputs: [silver.revenue]
"""
    )
    import yaml

    parsed = parse_config(yaml.safe_load((tmp_path / "fathom.yml").read_text()), root=tmp_path)
    with Store(":memory:") as store:
        yield Project(config=parsed, store=store)


def test_declared_publications_appear_in_the_graph(project):
    graph = project.graph()
    assert set(sinks.sinks_in(graph)) == {
        sinks.dashboard("revenue/exec", tool="looker"),
        sinks.filing("10-K/2026", regulator="sec"),
    }


def test_a_declared_publication_carries_its_evidence(project):
    graph = project.graph()
    edge = graph.in_edges(sinks.dashboard("revenue/exec", tool="looker"))[0]
    assert edge.evidence == "declared:config"


def test_restatement_impact_answers_from_config_alone(project):
    """No Python needed: declare the artefact, ask what a restatement touches."""
    impact = sinks.restatement_impact(project.graph(), SILVER)
    assert sinks.filing("10-K/2026", regulator="sec") in impact.regulatory


def test_loading_the_graph_twice_does_not_duplicate_edges(project):
    first = project.graph()
    second = project.graph()
    sink = sinks.dashboard("revenue/exec", tool="looker")
    assert len(first.in_edges(sink)) == len(second.in_edges(sink)) == 1


def test_removing_a_publication_from_config_removes_it_from_the_graph(tmp_path):
    """A declared artefact that outlived its declaration would still appear in notices."""
    import yaml

    with_it = parse_config(
        yaml.safe_load(
            BASE + "\npublications:\n  - name: x\n    kind: filing\n    inputs: [gold.monthly]\n"
        ),
        root=tmp_path,
    )
    without = parse_config(yaml.safe_load(BASE), root=tmp_path)

    with Store(":memory:") as store:
        assert sinks.sinks_in(Project(config=with_it, store=store).graph())
        assert sinks.sinks_in(Project(config=without, store=store).graph()) == []


def test_no_publications_block_is_simply_empty(tmp_path):
    parsed = config(BASE, tmp_path)
    assert parsed.publications == []
