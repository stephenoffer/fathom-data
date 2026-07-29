"""Licences that travel with the data.

A model trained on a non-commercial corpus is a non-commercial model. A dataset
built from a share-alike source is share-alike. Nobody records this, so it is
discovered at the point where somebody wants to sell something — by which time the
corpus is four hops upstream and the person who added it has left.

Licences propagate along exactly the edges dirtiness does, with one difference in
direction: dirtiness takes the union, licences take the *most restrictive*
combination. A dataset built from MIT and CC-BY-NC sources is bound by both, which
means it is bound by the stricter.

The terms modelled here are the four that decide whether a use is permitted, not the
full text of any licence:

- **commercial use** — the one that stops a product
- **derivatives** — whether a trained model is even permitted
- **share-alike** — whether the output inherits the licence
- **attribution** — cheap to satisfy, embarrassing to discover you have not

`attribution_manifest` generates the NOTICE file from the graph, which is the only
version of that file that stays correct.

This is not legal advice and the module says so at every exit: `LicenseReport`
carries a `requires_review` flag that is set whenever anything is unknown, and
unknown is the default rather than permissive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..core.types import DatasetId
from ..core.util import markdown as md
from ..graph.model import Graph
from ..graph.query import ancestors, closure, fold_downstream

__all__ = [
    "License",
    "LicenseReport",
    "attribution_manifest",
    "attribution_required",
    "combine",
    "license_breakdown",
    "report",
    "restrictive_sources",
    "commercial_use_allowed",
    "effective_license",
    "is_compatible",
    "known_licenses",
    "parse_license",
    "propagate",
    "training_permitted",
    "unlicensed",
    "violations",
]


@dataclass(frozen=True)
class License:
    """The four terms that decide whether a use is permitted.

    `unknown` is a distinct state from "permissive". A source with no licence
    recorded blocks nothing by itself but taints everything downstream with a review
    requirement, which is the correct outcome — you cannot ship on data whose terms
    nobody checked.
    """

    name: str = "unknown"
    commercial: bool | None = None
    derivatives: bool | None = None
    share_alike: bool = False
    attribution: bool = False
    text_and_data_mining: bool | None = None

    @property
    def is_unknown(self) -> bool:
        """True when a term this decision needs was never recorded."""
        return self.commercial is None or self.derivatives is None

    def __str__(self) -> str:
        return self.name


# The licences that actually appear on data and model artefacts. `None` means the
# licence does not settle the question and a human has to.
_KNOWN: dict[str, License] = {
    "cc0": License("CC0-1.0", True, True, False, False, True),
    "cc0-1.0": License("CC0-1.0", True, True, False, False, True),
    "mit": License("MIT", True, True, False, True, True),
    "apache-2.0": License("Apache-2.0", True, True, False, True, True),
    "bsd-3-clause": License("BSD-3-Clause", True, True, False, True, True),
    "cc-by": License("CC-BY-4.0", True, True, False, True, True),
    "cc-by-4.0": License("CC-BY-4.0", True, True, False, True, True),
    "cc-by-sa": License("CC-BY-SA-4.0", True, True, True, True, True),
    "cc-by-sa-4.0": License("CC-BY-SA-4.0", True, True, True, True, True),
    "cc-by-nc": License("CC-BY-NC-4.0", False, True, False, True, True),
    "cc-by-nc-4.0": License("CC-BY-NC-4.0", False, True, False, True, True),
    "cc-by-nd": License("CC-BY-ND-4.0", True, False, False, True, False),
    "gpl-3.0": License("GPL-3.0", True, True, True, True, True),
    "agpl-3.0": License("AGPL-3.0", True, True, True, True, True),
    "odbl": License("ODbL-1.0", True, True, True, True, True),
    "proprietary": License("proprietary", False, False, False, False, False),
    "internal": License("internal", True, True, False, False, True),
    "unknown": License("unknown"),
}


def known_licenses() -> list[str]:
    """Every licence identifier this module recognizes."""
    return sorted({lic.name for lic in _KNOWN.values()})


def parse_license(text: str) -> License:
    """Resolve a licence identifier. Anything unrecognized becomes `unknown`.

    Unrecognized never becomes permissive. A typo in a licence field must not be the
    reason a non-commercial corpus ends up in a commercial model.
    """
    key = text.strip().lower().replace(" ", "-")
    return _KNOWN.get(key, License(name=text.strip() or "unknown"))


def combine(licenses: Iterable[License]) -> License:
    """The most restrictive combination of several licences.

    `None` beats `True` on every term: one unknown source makes the combination
    unknown, because it might be the one that forbids what you are about to do.
    """
    items = [lic for lic in licenses]
    if not items:
        return _KNOWN["unknown"]
    if len(items) == 1:
        return items[0]

    def strictest(values: list[bool | None]) -> bool | None:
        if any(v is False for v in values):
            return False
        if any(v is None for v in values):
            return None
        return True

    names = sorted({lic.name for lic in items})
    return License(
        name=" + ".join(names),
        commercial=strictest([lic.commercial for lic in items]),
        derivatives=strictest([lic.derivatives for lic in items]),
        share_alike=any(lic.share_alike for lic in items),
        attribution=any(lic.attribution for lic in items),
        text_and_data_mining=strictest([lic.text_and_data_mining for lic in items]),
    )


def is_compatible(source: License, target: License) -> bool:
    """True when data under `source` may be combined into something under `target`.

    Share-alike is the asymmetric case: CC-BY-SA content can go into a CC-BY-SA work
    and not into an MIT one, while the reverse is fine.

    An unrecorded term on the *target* is not permission. Requiring the target to be
    explicitly commercial before rejecting a non-commercial source meant a team that
    simply never declared their model's licence got a green light that a team who
    honestly wrote down "commercial" did not — rewarding the missing declaration.
    Share-alike already treated unknown as unproven; commercial now matches it.
    """
    if source.derivatives is False:
        return False
    if source.commercial is False and target.commercial is not False:
        return False
    return not (source.share_alike and not target.share_alike)


def propagate(graph: Graph, declared: Mapping[DatasetId, License]) -> dict[DatasetId, License]:
    """Flow licences downstream, combining at every join.

    A dataset nobody declared and nothing licensed feeds resolves to `unknown` rather
    than to permissive, which is the whole reason `unknown` is a state.
    """
    return fold_downstream(
        graph,
        dict(declared),
        combine=lambda a, b: combine([a, b]),
        default=_KNOWN["unknown"],
    )


def effective_license(
    graph: Graph, ds: DatasetId, declared: Mapping[DatasetId, License]
) -> License:
    """The licence a dataset is actually bound by, its whole upstream included."""
    return combine(declared[node] for node in closure(graph, ds) if node in declared)


def commercial_use_allowed(
    graph: Graph, ds: DatasetId, declared: Mapping[DatasetId, License]
) -> bool | None:
    """Whether this dataset may be used commercially. None means nobody can say."""
    return effective_license(graph, ds, declared).commercial


def training_permitted(
    graph: Graph, ds: DatasetId, declared: Mapping[DatasetId, License]
) -> bool | None:
    """Whether training on this data is permitted under its combined terms.

    Reads the text-and-data-mining term rather than the derivatives term, because
    several jurisdictions treat training as a distinct permission and several
    licences now say so explicitly.
    """
    return effective_license(graph, ds, declared).text_and_data_mining


def attribution_required(
    graph: Graph, ds: DatasetId, declared: Mapping[DatasetId, License]
) -> bool:
    """Whether anything upstream requires attribution."""
    return effective_license(graph, ds, declared).attribution


def unlicensed(graph: Graph, declared: Mapping[DatasetId, License]) -> list[DatasetId]:
    """Datasets with no licence recorded and no licensed ancestor.

    The list to work through before making a commercial claim about anything
    downstream of them.
    """
    return sorted(
        (
            ds
            for ds in graph.datasets
            if ds not in declared and not any(node in declared for node in ancestors(graph, ds))
        ),
        key=str,
    )


@dataclass
class LicenseReport:
    """Whether one dataset can be used as intended, and what has to happen first."""

    dataset: DatasetId
    effective: License = field(default_factory=lambda: _KNOWN["unknown"])
    sources: list[tuple[DatasetId, License]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    requires_review: bool = False

    @property
    def is_clear(self) -> bool:
        """True when no licence conflict was proven."""
        return not self.blockers and not self.requires_review

    def summary(self) -> str:
        """The report as text."""
        lines = [f"{self.dataset}: effective licence `{self.effective}`"]
        for blocker in self.blockers:
            lines.append(f"  BLOCKED: {blocker}")
        if self.requires_review:
            lines.append("  review needed: at least one upstream licence is unknown")
        if self.is_clear:
            lines.append("  no blockers found")
        return "\n".join(lines)


def report(
    graph: Graph,
    ds: DatasetId,
    declared: Mapping[DatasetId, License],
    *,
    intended_commercial: bool = True,
    intended_share_alike: bool = False,
) -> LicenseReport:
    """Check a dataset's combined licence against how it is meant to be used."""
    effective = effective_license(graph, ds, declared)
    sources = [(node, declared[node]) for node in ancestors(graph, ds) if node in declared]
    out = LicenseReport(
        dataset=ds, effective=effective, sources=sorted(sources, key=lambda p: str(p[0]))
    )

    if intended_commercial and effective.commercial is False:
        offenders = [str(node) for node, lic in sources if lic.commercial is False]
        out.blockers.append(
            "commercial use is forbidden by " + (", ".join(offenders) or "an upstream licence")
        )
    if effective.derivatives is False:
        out.blockers.append("derivative works are forbidden upstream; a model is a derivative")
    if effective.share_alike and not intended_share_alike:
        out.blockers.append("a share-alike source requires the output to carry the same licence")
    if effective.is_unknown or any(lic.is_unknown for _, lic in sources):
        out.requires_review = True
    return out


def violations(
    graph: Graph,
    declared: Mapping[DatasetId, License],
    *,
    commercial_datasets: Iterable[DatasetId] = (),
) -> list[str]:
    """Datasets used commercially whose upstream licences forbid it."""
    out: list[str] = []
    for ds in sorted(set(commercial_datasets), key=str):
        result = report(graph, ds, declared, intended_commercial=True)
        for blocker in result.blockers:
            out.append(f"{ds}: {blocker}")
    return out


def attribution_manifest(graph: Graph, ds: DatasetId, declared: Mapping[DatasetId, License]) -> str:
    """The NOTICE file, generated from lineage.

    Every upstream source whose licence requires attribution, named. Hand-maintained
    NOTICE files go stale the first time somebody adds a dependency without reading
    the contributing guide; this one cannot.
    """
    entries = [
        (node, declared[node])
        for node in closure(graph, ds)
        if node in declared and declared[node].attribution
    ]
    if not entries:
        return f"# Attributions for {ds.name}\n\nNo upstream source requires attribution."

    lines = [
        f"# Attributions for {ds.name}",
        "",
        "This work incorporates material from the following sources, which require "
        "attribution under their licences.",
        "",
    ]
    lines.append(
        md.bullets(
            f"{md.code(node)} — {lic.name}"
            for node, lic in sorted(entries, key=lambda pair: str(pair[0]))
        )
    )
    return "\n".join(lines)


def license_breakdown(declared: Mapping[DatasetId, License]) -> dict[str, int]:
    """Dataset counts per licence. The inventory line of a compliance review."""
    counts: dict[str, int] = {}
    for lic in declared.values():
        counts[lic.name] = counts.get(lic.name, 0) + 1
    return dict(sorted(counts.items()))


def restrictive_sources(
    graph: Graph, ds: DatasetId, declared: Mapping[DatasetId, License]
) -> list[DatasetId]:
    """The specific upstream datasets that constrain what `ds` may be used for.

    Removing these is usually the cheapest path to an unencumbered downstream asset,
    and naming them is what makes that a decision rather than a rewrite.
    """
    return sorted(
        (
            node
            for node in ancestors(graph, ds)
            if node in declared
            and (
                declared[node].commercial is not True
                or declared[node].derivatives is not True
                or declared[node].share_alike
            )
        ),
        key=str,
    )
