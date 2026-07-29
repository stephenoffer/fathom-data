"""`fathom explain`, and the help a new user reads before anything else.

The value of these is not that the text exists — it is that the text stays wired to
the code. A warning that says "widened" and a topic list that has dropped `widening`
is worse than no explain command, because the user follows the instruction and gets
an error.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from fathom.cli import explain
from fathom.cli.main import main


@pytest.fixture
def run() -> CliRunner:
    return CliRunner()


# -- the topics themselves -----------------------------------------------------


def test_every_topic_says_what_to_do_about_it():
    """A definition that ends without a next action leaves the reader where it found
    them. That is the difference between this and a glossary."""
    for name, topic in explain.TOPICS.items():
        assert "What to do:" in topic.body, f"{name} explains itself but suggests nothing"


def test_every_topic_has_a_summary_that_is_a_sentence():
    for name, topic in explain.TOPICS.items():
        assert topic.summary.endswith((".", "?")), f"{name}'s summary is not a sentence"
        assert len(topic.summary) > 40, f"{name}'s summary is too short to be useful"


def test_every_cross_reference_resolves():
    """A `Related:` line pointing at a topic that does not exist is a dead end."""
    for name, topic in explain.TOPICS.items():
        for related in topic.see_also:
            assert explain.lookup(related), f"{name} points at unknown topic {related!r}"


def test_every_referenced_doc_exists():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name, topic in explain.TOPICS.items():
        if topic.doc:
            assert (root / topic.doc).exists(), f"{name} points at missing doc {topic.doc}"


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("widened", "widening"),
        ("WIDENING", "widening"),
        ("partition mapping", "partition-mapping"),
        ("partition_mapping", "partition-mapping"),
        ("shadow", "shadow-mode"),
        ("adapters", "capabilities"),
        ("dirty", "seed"),
    ],
)
def test_the_spellings_people_type_resolve(typed: str, expected: str):
    found = explain.lookup(typed)
    assert found is not None and found is explain.TOPICS[expected]


def test_an_unknown_topic_is_absent_rather_than_an_error():
    assert explain.lookup("quantum entanglement") is None


def test_a_gloss_keeps_its_own_terminator():
    """Several summaries end in a question mark; appending a stop reads as a typo."""
    assert explain.TOPICS["partition-mapping"].gloss.endswith("?")


# -- the command ---------------------------------------------------------------


def test_explain_with_no_topic_lists_them_all(run: CliRunner):
    result = run.invoke(main, ["explain"])
    assert result.exit_code == 0
    for name in explain.titles():
        assert name in result.output


def test_explain_prints_the_body_and_where_to_read_on(run: CliRunner):
    result = run.invoke(main, ["explain", "widening"])
    assert result.exit_code == 0
    assert "What to do:" in result.output
    assert "Related:" in result.output
    assert "docs/guide/plan.md" in result.output


def test_an_unknown_topic_suggests_the_nearest_one(run: CliRunner):
    result = run.invoke(main, ["explain", "widenning"])
    assert result.exit_code != 0
    assert "Did you mean 'widening'?" in result.output


def test_an_unrecognisable_topic_still_says_how_to_list_them(run: CliRunner):
    result = run.invoke(main, ["explain", "zzzzz"])
    assert result.exit_code != 0
    assert "with no argument to list them" in result.output


# -- the help a new user reads first -------------------------------------------


def test_the_top_level_help_groups_commands_by_what_they_are_for(run: CliRunner):
    result = run.invoke(main, ["--help"])
    assert result.exit_code == 0
    for section in ("Start here:", "Build the graph:", "Plan a rebuild:", "Govern it:"):
        assert section in result.output


def test_the_top_level_help_says_nothing_writes(run: CliRunner):
    """The first question anyone asks of a tool aimed at their warehouse."""
    result = run.invoke(main, ["--help"])
    assert "Nothing here writes to your data" in result.output


def test_the_top_level_help_documents_the_exit_codes(run: CliRunner):
    result = run.invoke(main, ["--help"])
    assert "Exit codes:" in result.output


def test_every_command_appears_in_some_section(run: CliRunner):
    """A command missing from the sections falls through to `Other`, which is a
    reminder rather than a failure — but it should not be silent."""
    result = run.invoke(main, ["--help"])
    assert "Other:" not in result.output, (
        "a command is not in any section of `_Sections.SECTIONS`; add it to the one "
        "matching the stage it belongs to"
    )


def test_a_mistyped_command_suggests_the_real_one(run: CliRunner):
    result = run.invoke(main, ["plna"])
    assert result.exit_code != 0
    assert "Did you mean 'plan'?" in result.output


def test_a_command_with_no_near_match_falls_back_to_click(run: CliRunner):
    result = run.invoke(main, ["zzzzzzz"])
    assert result.exit_code != 0
    assert "No such command" in result.output


@pytest.mark.parametrize(
    "command",
    [
        "init",
        "doctor",
        "explain",
        "adapters",
        "ingest",
        "lineage",
        "detect",
        "plan",
        "profile",
        "check",
        "label",
        "erase",
        "shadow",
        "completeness",
        "usage",
        "value",
        "impact",
        "risk",
        "contracts",
        "history",
        "dag",
        "seasonal",
    ],
)
def test_every_command_shows_a_runnable_example(run: CliRunner, command: str):
    """Help that describes a command without showing one being used makes the reader
    assemble the invocation from the option list."""
    result = run.invoke(main, [command, "--help"])
    assert result.exit_code == 0
    assert f"fathom {command}" in result.output, f"`{command} --help` shows no example"


def test_planning_with_no_seeds_shows_the_invocation_it_wanted(run: CliRunner):
    result = run.invoke(main, ["plan"])
    assert result.exit_code != 0
    assert "--dirty 'raw.events@dt=2026-03-14'" in result.output
