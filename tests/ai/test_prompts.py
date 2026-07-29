"""Prompts as versioned datasets whose variables carry data in."""

from __future__ import annotations

from datetime import UTC, datetime

from fathom.ai import (
    assets,
    prompts,
)
from fathom.core.grains import Grain
from fathom.core.types import ColumnRef, DatasetId, PartitionField, PartitionSpec
from fathom.govern.policy import Label
from fathom.graph import Graph

RAW = DatasetId("duckdb", "raw.events")
GOLD = DatasetId("duckdb", "gold.training_set")
DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))

MODEL = assets.model("fraud.scorer", registry="internal")
BASE = assets.model("base.llm", registry="internal")
INDEX = assets.vector_index("docs", store="pgvector")
SPACE = assets.embedding_space("text-embed-3", provider="openai")
CORPUS = assets.corpus("handbook", store="s3")
EVAL = assets.eval_set("fraud.holdout", suite="internal")


# -- prompts -------------------------------------------------------------------


def test_prompt_versions_are_content_addressed():
    template = prompts.PromptTemplate(dataset=assets.prompt("triage.system"))
    first = template.commit("Answer using {{context}}.")
    again = template.commit("Answer   using   {{context}}.")  # only whitespace differs
    assert first.digest == again.digest
    assert len(template.versions) == 1

    changed = template.commit("Answer using {{context}} and be brief.")
    assert changed.digest != first.digest
    assert len(template.versions) == 2


def test_variables_and_rendering():
    text = "Hi {{name}}, about {topic}."
    assert prompts.variables_in(text) == ["name", "topic"]
    assert prompts.rendered(text, {"name": "Sam"}) == "Hi Sam, about {topic}."
    filled = prompts.render_digest(text, {"name": "Sam", "topic": "x"})
    assert filled != prompts.template_digest(text)


def test_prompt_bindings_become_graph_edges():
    template = prompts.PromptTemplate(dataset=assets.prompt("triage.system"))
    template.commit("Hello {{user_name}}")
    template.bind("user_name", RAW)
    graph = Graph()
    record = prompts.record_prompt(graph, template)
    assert record.in_edges(template.dataset)[0].src == RAW
    assert prompts.variable_sources(template) == {"user_name": RAW}
    assert prompts.unbound_variables(template) == []


def test_unbound_variables_are_a_hole_in_provenance():
    template = prompts.PromptTemplate(dataset=assets.prompt("p"))
    template.commit("{{a}} and {{b}}")
    template.bind("a", RAW)
    assert prompts.unbound_variables(template) == ["b"]


def test_prompt_labels_travel_through_bindings():
    template = prompts.PromptTemplate(dataset=assets.prompt("p"))
    template.commit("{{email}}")
    template.bind("email", RAW)
    graph = Graph()
    prompts.record_prompt(graph, template)
    labels = {ColumnRef(RAW, "email"): {Label("pii", confidence=0.9)}}
    assert prompts.labels_reaching(graph, template, labels)["pii"] == [ColumnRef(RAW, "email")]


def test_prompt_rollback_appends_rather_than_truncating():
    template = prompts.PromptTemplate(dataset=assets.prompt("p"))
    first = template.commit("v1 text")
    template.commit("v2 text")
    prompts.rollback(template, first.digest)
    assert len(template.versions) == 3
    assert template.current is not None
    assert template.current.digest == first.digest


def test_changed_variables_flags_a_removal():
    result = prompts.changed_variables("{{a}} {{b}}", "{{a}} {{c}}")
    assert result == {"added": ["c"], "removed": ["b"]}


def test_version_at_a_moment():
    template = prompts.PromptTemplate(dataset=assets.prompt("p"))
    template.versions.append(
        prompts.PromptVersion("d1", "old", created=datetime(2026, 3, 1, tzinfo=UTC))
    )
    template.versions.append(
        prompts.PromptVersion("d2", "new", created=datetime(2026, 3, 10, tzinfo=UTC))
    )
    found = prompts.version_at(template, datetime(2026, 3, 5, tzinfo=UTC))
    assert found is not None and found.digest == "d1"
    assert prompts.drifted(template, "d1")
