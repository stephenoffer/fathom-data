"""Documentation integrity.

Docs rot in two ways: links break, and code samples stop working. The examples are
executed by `test_examples.py`; this covers links, and checks that every CLI command
and adapter the docs mention actually exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCS = sorted(ROOT.glob("docs/**/*.md")) + [ROOT / "README.md", ROOT / "examples/README.md"]

# Markdown links, ignoring images and anchors-only references.
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def relative_links(path: Path) -> list[str]:
    body = path.read_text()
    found = []
    for target in _LINK.findall(body):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        found.append(target)
    return found


def test_docs_exist():
    assert len(DOCS) > 10, "documentation is missing or the glob moved"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc: Path):
    broken = []
    for link in relative_links(doc):
        target = link.split("#", 1)[0]
        if not target:
            continue
        resolved = (doc.parent / target).resolve()
        if not resolved.exists():
            broken.append(link)
    assert not broken, f"{doc.relative_to(ROOT)} has broken link(s): {broken}"


def test_every_cli_command_in_the_docs_exists():
    """A documented command that does not exist is worse than an undocumented one."""
    from fathom.cli import main

    real = set(main.commands)
    mentioned = set()
    for doc in DOCS:
        # Only backtick-quoted references; "fathom documentation" is prose.
        mentioned.update(re.findall(r"`fathom ([a-z-]+)[^`]*`", doc.read_text()))

    # `fathom apply` is named only to say it deliberately does not exist.
    unknown = mentioned - real - {"apply"}
    assert not unknown, f"docs mention non-existent commands: {sorted(unknown)}"


def test_every_documented_adapter_is_registered():
    from fathom.adapters import registered

    matrix = (ROOT / "docs/guide/adapters.md").read_text()
    documented = set(re.findall(r"^\| `([a-z]+)` \|", matrix, re.MULTILINE))
    assert documented, "the adapter matrix did not parse; check its formatting"
    assert documented <= set(registered()), (
        f"documented but unregistered: {sorted(documented - set(registered()))}"
    )


def test_every_registered_adapter_is_documented():
    from fathom.adapters import registered

    matrix = (ROOT / "docs/guide/adapters.md").read_text()
    for name in registered():
        assert f"`{name}`" in matrix, f"{name} is registered but missing from the matrix"


def test_config_reference_covers_every_top_level_key():
    from fathom.config import _TOP_LEVEL

    reference = (ROOT / "docs/guide/configuration.md").read_text()
    for key in _TOP_LEVEL:
        assert f"`{key}`" in reference, f"config key {key} is undocumented"


def test_adrs_are_linked_from_the_readme():
    readme = (ROOT / "README.md").read_text()
    for adr in sorted((ROOT / "docs/adr").glob("*.md")):
        assert adr.name in readme, f"{adr.name} is not linked from the README"
