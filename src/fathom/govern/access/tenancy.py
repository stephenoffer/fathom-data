"""Isolation between tenants sharing one deployment.

A platform team running fathom for twenty product teams has, today, no boundary at
all: one store, one graph, everybody's profiles. That is fine until the first time
someone asks whether team A can see team B's column statistics, and the answer has
to be a shrug.

The hard part is not scoping reads. It is that **lineage crosses tenants and that is
usually correct** — a shared `raw.events` genuinely feeds both teams' marts. So the
model here is not a partition of the graph; it is ownership plus an explicit sharing
grant, with `leaks` reporting edges that cross a boundary without one.

A crossing is not automatically a violation. `classify_crossings` separates the
three cases, because they have three different remedies:

- **shared** — a grant exists; nothing to do
- **undeclared** — a real dependency nobody wrote down; declare it
- **violation** — a tenant reading data from a tenant that has not shared with it

Reporting all three as violations is how a boundary check gets switched off.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ...core.types import DatasetId

__all__ = [
    "Crossing",
    "CrossingKind",
    "IsolationReport",
    "Tenant",
    "TenantMap",
    "classify_crossings",
    "crossings",
    "datasets_of",
    "is_shared_with",
    "leaks",
    "owner_of",
    "scope_graph",
    "share",
    "shared_with",
    "tenant_summary",
    "unowned",
]


@dataclass(frozen=True)
class Tenant:
    """One isolated party sharing the deployment."""

    identifier: str
    name: str = ""
    contact: str = ""


class CrossingKind(StrEnum):
    """What a cross-tenant edge means. Three, not two."""

    SHARED = "shared"  # a grant exists
    UNDECLARED = "undeclared"  # real dependency, nobody wrote it down
    VIOLATION = "violation"  # reading from a tenant that has not shared


@dataclass(frozen=True)
class Crossing:
    """An edge whose endpoints belong to different tenants."""

    source: DatasetId
    target: DatasetId
    source_tenant: str
    target_tenant: str
    kind: CrossingKind

    def summary(self) -> str:
        return (
            f"{self.source_tenant} -> {self.target_tenant}: {self.source} feeds "
            f"{self.target} [{self.kind.value}]"
        )


@dataclass
class TenantMap:
    """Ownership, and who has shared what with whom."""

    tenants: dict[str, Tenant] = field(default_factory=dict)
    ownership: dict[DatasetId, str] = field(default_factory=dict)
    # sharing[owner] = {dataset: {recipients}}
    sharing: dict[str, dict[DatasetId, set[str]]] = field(default_factory=dict)

    def add(self, tenant: Tenant) -> None:
        self.tenants[tenant.identifier] = tenant

    def own(self, dataset: DatasetId, tenant: str) -> None:
        self.ownership[dataset] = tenant

    def owner_of(self, dataset: DatasetId) -> str:
        return self.ownership.get(dataset, "")

    def datasets_of(self, tenant: str) -> list[DatasetId]:
        return sorted((d for d, t in self.ownership.items() if t == tenant), key=str)


def owner_of(tenants: TenantMap, dataset: DatasetId) -> str:
    return tenants.owner_of(dataset)


def datasets_of(tenants: TenantMap, tenant: str) -> list[DatasetId]:
    return tenants.datasets_of(tenant)


def unowned(tenants: TenantMap, datasets: Iterable[DatasetId]) -> list[DatasetId]:
    """Datasets nobody claims.

    Worth surfacing on its own: an unowned dataset has no one to ask about a breach
    and no one to bill for its storage.
    """
    return sorted((d for d in datasets if not tenants.owner_of(d)), key=str)


def share(tenants: TenantMap, dataset: DatasetId, *, with_tenants: Iterable[str]) -> None:
    """Record that a dataset's owner has shared it.

    Sharing is recorded against the owner, so a tenant cannot share data it does not
    own by asserting a grant it has no standing to make.
    """
    owner = tenants.owner_of(dataset)
    if not owner:
        raise ValueError(f"{dataset} has no owner, so nobody can share it")
    tenants.sharing.setdefault(owner, {}).setdefault(dataset, set()).update(with_tenants)


def shared_with(tenants: TenantMap, dataset: DatasetId) -> set[str]:
    owner = tenants.owner_of(dataset)
    return set(tenants.sharing.get(owner, {}).get(dataset, set()))


def is_shared_with(tenants: TenantMap, dataset: DatasetId, tenant: str) -> bool:
    return tenant in shared_with(tenants, dataset)


def crossings(tenants: TenantMap, edges: Iterable[tuple[DatasetId, DatasetId]]) -> list[Crossing]:
    """Every edge whose endpoints belong to different tenants."""
    found: list[Crossing] = []
    for source, target in edges:
        source_tenant = tenants.owner_of(source)
        target_tenant = tenants.owner_of(target)
        if not source_tenant or not target_tenant or source_tenant == target_tenant:
            continue
        kind = (
            CrossingKind.SHARED
            if is_shared_with(tenants, source, target_tenant)
            else CrossingKind.VIOLATION
        )
        found.append(
            Crossing(
                source=source,
                target=target,
                source_tenant=source_tenant,
                target_tenant=target_tenant,
                kind=kind,
            )
        )
    return found


def classify_crossings(
    tenants: TenantMap,
    edges: Iterable[tuple[DatasetId, DatasetId]],
    *,
    declared: Iterable[tuple[DatasetId, DatasetId]] = (),
) -> list[Crossing]:
    """Crossings, with declared-but-ungranted ones separated from real violations.

    The separation exists so a boundary check stays usable. A tool that reports every
    legitimate shared dependency as a violation gets muted, and then the real one is
    invisible too.
    """
    known = {(str(s), str(t)) for s, t in declared}
    out: list[Crossing] = []
    for crossing in crossings(tenants, edges):
        kind = crossing.kind
        if (
            kind is CrossingKind.VIOLATION
            and (
                str(crossing.source),
                str(crossing.target),
            )
            in known
        ):
            kind = CrossingKind.UNDECLARED
        out.append(
            Crossing(
                source=crossing.source,
                target=crossing.target,
                source_tenant=crossing.source_tenant,
                target_tenant=crossing.target_tenant,
                kind=kind,
            )
        )
    return out


def leaks(tenants: TenantMap, edges: Iterable[tuple[DatasetId, DatasetId]]) -> list[Crossing]:
    """Only the genuine violations."""
    return [c for c in crossings(tenants, edges) if c.kind is CrossingKind.VIOLATION]


def scope_graph(tenants: TenantMap, datasets: Iterable[DatasetId], tenant: str) -> list[DatasetId]:
    """What one tenant may see: its own datasets, plus what has been shared with it."""
    visible = []
    for dataset in datasets:
        owner = tenants.owner_of(dataset)
        if owner == tenant or is_shared_with(tenants, dataset, tenant):
            visible.append(dataset)
    return sorted(visible, key=str)


@dataclass(frozen=True)
class IsolationReport:
    """Whether the boundaries hold."""

    tenants: int
    datasets: int
    unowned: tuple[DatasetId, ...]
    violations: tuple[Crossing, ...]
    undeclared: tuple[Crossing, ...]
    shared: tuple[Crossing, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        lines = [
            f"{self.tenants} tenant(s), {self.datasets} dataset(s), {len(self.unowned)} unowned"
        ]
        if self.shared:
            lines.append(f"  {len(self.shared)} shared crossing(s), all granted")
        for crossing in self.undeclared:
            lines.append(f"  UNDECLARED {crossing.summary()}")
        for crossing in self.violations:
            lines.append(f"  VIOLATION {crossing.summary()}")
        if self.ok and not self.undeclared:
            lines.append("  boundaries hold")
        return "\n".join(lines)


def tenant_summary(
    tenants: TenantMap,
    datasets: Sequence[DatasetId],
    edges: Iterable[tuple[DatasetId, DatasetId]],
    *,
    declared: Iterable[tuple[DatasetId, DatasetId]] = (),
) -> IsolationReport:
    classified = classify_crossings(tenants, edges, declared=declared)
    return IsolationReport(
        tenants=len(tenants.tenants),
        datasets=len(datasets),
        unowned=tuple(unowned(tenants, datasets)),
        violations=tuple(c for c in classified if c.kind is CrossingKind.VIOLATION),
        undeclared=tuple(c for c in classified if c.kind is CrossingKind.UNDECLARED),
        shared=tuple(c for c in classified if c.kind is CrossingKind.SHARED),
    )
