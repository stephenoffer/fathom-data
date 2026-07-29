"""Prompts as versioned datasets.

A prompt decides what the output is made of. That makes it a transformation, exactly
like the SQL in a model, and it belongs in lineage for the same reason: when results
move, somebody has to establish whether the data changed or the instructions did.

Prompts are worse than SQL in three specific ways, which is why they get a module:

- **They change without a schema change.** Editing one word alters every downstream
  result and breaks nothing that a test would notice.
- **They are assembled at runtime.** A template plus variables plus retrieved
  context is what the model actually saw, and only the template is in version
  control. `render_digest` hashes what was really sent.
- **They interpolate data.** A variable filled from a table carries that table's
  labels into the prompt, which is how personal data reaches a third-party endpoint
  without anyone writing a line of code that sends it. `variable_sources` makes that
  edge explicit.

Versioning here is content-addressed rather than sequential. Two teams editing the
same prompt produce different digests, and a rollback restores a digest rather than
a number that may since have been reused.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.types import ColumnRef, DatasetId
from ..core.util import digest as _digest
from ..core.util import markdown as md
from ..core.util import text as _text
from ..core.util.clock import as_utc
from ..govern.policy import LabelSet, labels_over
from ..graph.model import Graph, link
from ..graph.query import closure
from .assets import AssetKind, spec_for

__all__ = [
    "PromptTemplate",
    "PromptVersion",
    "changed_variables",
    "diff_prompts",
    "drifted",
    "history",
    "labels_reaching",
    "prompt_cards",
    "rollback",
    "token_estimate",
    "unbound_variables",
    "version_at",
    "outputs_using",
    "record_prompt",
    "render_digest",
    "rendered",
    "template_digest",
    "variable_sources",
    "variables_in",
]

# `{name}` and `{{name}}` both appear in the wild; the doubled form is what most
# templating layers use and the single form is what f-string-shaped prompts use.
_VARIABLE = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}|\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}"
)


def template_digest(text: str) -> str:
    """Content address for a template. Whitespace-normalized so reformatting is not a change.

    Reflowing a paragraph should not read as a new prompt version; changing a word
    should. Normalizing runs of whitespace gets that distinction right cheaply.
    """
    return _digest.short(_digest.of_text(text))


def variables_in(text: str) -> list[str]:
    """Variable names a template interpolates, in sorted order."""
    found = set()
    for doubled, single in _VARIABLE.findall(text):
        found.add(doubled or single)
    return sorted(found)


@dataclass(frozen=True)
class PromptVersion:
    """One content-addressed version of a prompt."""

    digest: str
    text: str
    author: str = ""
    created: datetime = field(default_factory=lambda: datetime.now(UTC))
    note: str = ""

    @property
    def variables(self) -> list[str]:
        """Variable names this version interpolates."""
        return variables_in(self.text)

    def __str__(self) -> str:
        return f"{self.digest} ({len(self.text)} chars, {len(self.variables)} variable(s))"


@dataclass
class PromptTemplate:
    """A named prompt and its version history.

    `sources` maps a template variable to the dataset it is filled from. That mapping
    is what turns a prompt into a real graph node rather than a string in a config
    file, because it is the edge along which labels and obligations travel.
    """

    dataset: DatasetId
    versions: list[PromptVersion] = field(default_factory=list)
    sources: dict[str, DatasetId] = field(default_factory=dict)

    @property
    def current(self) -> PromptVersion | None:
        """The newest version, or None when nothing has been recorded."""
        return self.versions[-1] if self.versions else None

    @property
    def digests(self) -> list[str]:
        """Every recorded version digest, newest first."""
        return [version.digest for version in self.versions]

    def commit(self, text: str, *, author: str = "", note: str = "") -> PromptVersion:
        """Record a new version, or return the existing one when nothing changed.

        Idempotent by content, so a pipeline that registers its prompt on every run
        does not accumulate identical versions.
        """
        digest = template_digest(text)
        for version in self.versions:
            if version.digest == digest:
                return version
        version = PromptVersion(digest=digest, text=text, author=author, note=note)
        self.versions.append(version)
        return version

    def get(self, digest: str) -> PromptVersion | None:
        """One version by digest, or None."""
        return next((v for v in self.versions if v.digest == digest), None)

    def bind(self, variable: str, dataset: DatasetId) -> None:
        """Declare where a template variable's value comes from."""
        self.sources[variable] = dataset

    def summary(self) -> str:
        """The template as text: current digest, variables, and sources."""
        current = self.current
        return (
            f"{self.dataset}: {len(self.versions)} version(s), "
            f"current {current.digest if current else 'none'}, "
            f"{len(self.sources)} bound variable(s)"
        )


def rendered(text: str, values: Mapping[str, object]) -> str:
    """Fill a template. Unbound variables are left in place rather than blanked.

    Leaving them visible means a missing binding shows up in the prompt that was
    actually sent, where somebody will notice, instead of silently becoming an empty
    string the model reads as an instruction to invent something.
    """

    def replace(match: re.Match[str]) -> str:
        """A copy of this template with fields replaced."""
        name = match.group(1) or match.group(2)
        return str(values[name]) if name in values else match.group(0)

    return _VARIABLE.sub(replace, text)


def render_digest(text: str, values: Mapping[str, object]) -> str:
    """Content address for what was actually sent, variables included.

    The template digest identifies the instructions; this identifies the instance.
    Two calls with the same render digest saw identical prompts, which is what a
    reproducibility question needs and what a template digest alone cannot answer.
    """
    return template_digest(rendered(text, values))


def record_prompt(graph: Graph, template: PromptTemplate) -> Graph:
    """Wire a prompt's variable sources into the graph.

    Each bound variable becomes an edge from its source dataset to the prompt. That
    is what makes `labels_in_context` and the erasure walk find data that reached a
    model through interpolation rather than through retrieval.
    """
    prompt_spec = spec_for(AssetKind.PROMPT)
    graph.add_dataset(template.dataset, prompt_spec)

    for variable, source in sorted(template.sources.items()):
        link(
            graph,
            source,
            template.dataset,
            evidence="prompt:variable",
            columns=((variable, variable),),
            dst_spec=prompt_spec,
        )
    return graph


def variable_sources(template: PromptTemplate) -> dict[str, DatasetId]:
    """Bound variables and where they are filled from."""
    return dict(sorted(template.sources.items()))


def unbound_variables(template: PromptTemplate) -> list[str]:
    """Variables the template uses with no declared source.

    Each one is data entering a prompt from somewhere the graph cannot see, which is
    a hole in every downstream claim about what reached the model.
    """
    current = template.current
    if current is None:
        return []
    return [name for name in current.variables if name not in template.sources]


def labels_reaching(
    graph: Graph, template: PromptTemplate, labels: LabelSet
) -> dict[str, list[ColumnRef]]:
    """Labels carried by anything that fills a variable in this prompt.

    Walks the upstream closure of each bound source, so a variable filled from a view
    over a table with email addresses carries the email label even when the view does
    not.
    """
    reachable: set[DatasetId] = set()
    for source in template.sources.values():
        reachable.update(closure(graph, source))
    return labels_over(labels, reachable)


def diff_prompts(before: str, after: str) -> list[str]:
    """A line-level diff of two prompt versions.

    Deliberately plain. A prompt change is reviewed by reading it, and the value here
    is having the two versions to compare at all rather than in the diff algorithm.
    """
    from difflib import unified_diff

    return [
        line.rstrip()
        for line in unified_diff(
            before.splitlines(), after.splitlines(), fromfile="before", tofile="after", lineterm=""
        )
    ]


def changed_variables(before: str, after: str) -> dict[str, list[str]]:
    """Variables added or removed between two versions.

    A removed variable is the dangerous direction: the binding usually stays in the
    pipeline config and the data quietly stops reaching the model.
    """
    b, a = set(variables_in(before)), set(variables_in(after))
    return {"added": sorted(a - b), "removed": sorted(b - a)}


def outputs_using(graph: Graph, prompt: DatasetId) -> list[DatasetId]:
    """Everything downstream of a prompt — what a prompt edit changes."""
    from ..graph.query import descendants

    return descendants(graph, prompt)


def version_at(template: PromptTemplate, moment: datetime) -> PromptVersion | None:
    """The version in effect at a point in time.

    What an incident review needs: results from Tuesday were produced by whichever
    prompt was current on Tuesday, not by the one in the repository today.
    """
    eligible = [v for v in template.versions if as_utc(v.created) <= moment]
    return eligible[-1] if eligible else None


def history(template: PromptTemplate) -> list[tuple[str, datetime, str]]:
    """Digest, timestamp, and author for every version, oldest first."""
    return [(v.digest, v.created, v.author) for v in template.versions]


def rollback(template: PromptTemplate, digest: str) -> PromptVersion:
    """Re-commit an earlier version as the current one.

    Appends rather than truncating, so the rollback is itself in the history. A
    history that can be rewritten cannot answer what was in effect last Tuesday.
    """
    target = template.get(digest)
    if target is None:
        raise ValueError(f"no version {digest!r} in {template.dataset}")
    # Appended directly rather than through `commit`, which is idempotent by content
    # and would return the old entry without making it current.
    restored = PromptVersion(
        digest=target.digest, text=target.text, author="rollback", note=f"restored {digest}"
    )
    template.versions.append(restored)
    return restored


def prompt_cards(templates: Sequence[PromptTemplate]) -> str:
    """A Markdown inventory of prompts, their versions, and their data bindings."""
    lines = ["# Prompt inventory", ""]
    for template in templates:
        current = template.current
        lines.extend(
            [
                f"## `{template.dataset}`",
                "",
                f"- versions: {len(template.versions)}",
                f"- current digest: `{current.digest if current else '—'}`",
                f"- variables: {', '.join(current.variables) if current else '—'}",
                "",
            ]
        )
        if template.sources:
            lines.extend(
                [
                    "Bound to:",
                    "",
                    md.bullets(
                        f"{md.code(variable)} ← {md.code(source)}"
                        for variable, source in sorted(template.sources.items())
                    ),
                    "",
                ]
            )
        unbound = unbound_variables(template)
        if unbound:
            lines.append(f"Unbound (source unknown): {', '.join(f'`{v}`' for v in unbound)}")
            lines.append("")
    return "\n".join(lines)


def drifted(template: PromptTemplate, deployed_digest: str) -> bool:
    """True when what is deployed is not the current version in the repository.

    The prompt equivalent of untracked infrastructure changes, and about as common.
    """
    current = template.current
    return current is not None and current.digest != deployed_digest


def token_estimate(text: str, *, chars_per_token: float = _text.CHARS_PER_TOKEN) -> int:
    """A rough token count for a rendered prompt, for budgeting a context window."""
    return _text.token_estimate(text, chars_per_token=chars_per_token)
