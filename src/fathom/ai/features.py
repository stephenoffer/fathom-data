"""Feature views, and the two ways they quietly break a model.

Feature stores exist because computing a feature twice — once for training, once for
serving — produces two different numbers often enough to matter. Lineage catches
both of the failures that follow, and neither shows up as an error anywhere:

- **Target leakage.** A feature computed from a column that is derived from the
  label. The model scores beautifully in training and collapses in production,
  because at serving time that column does not exist yet. This is a graph question:
  does a path exist from the label to the feature. `leaky_features` asks it.
- **Training/serving skew.** The offline feature and the online feature come from
  different code paths, so their distributions diverge. This is a profile question,
  and it is the same drift check the `check` verb already runs — pointed at two
  materializations of the same feature rather than at two days of one.

Freshness is the third thing here and the least subtle: a feature with a one-hour
TTL served from a table last built yesterday is wrong, and nothing in the serving
path knows it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.partitions import PartitionMapping
from ..core.types import ColumnRef, DatasetId, KeyPredicate, PartitionSpec
from ..core.util.clock import age as _age
from ..graph.model import Graph, InvalidationPlan, link
from ..graph.query import ancestors, column_ancestors, descendants, has_path, shortest_path
from ..observe.profile import Profile, drift
from .assets import AssetKind, is_model, spec_for

__all__ = [
    "FeatureView",
    "SkewReport",
    "backfill_plan",
    "feature_dependencies",
    "features_for_model",
    "freshness_age",
    "is_stale",
    "label_reaches_features",
    "leakage_path",
    "leaky_features",
    "models_using",
    "record_feature_view",
    "serving_risks",
    "skew",
    "stale_views",
    "training_serving_skew",
    "point_in_time_violations",
    "views_needing_backfill",
]


@dataclass
class FeatureView:
    """A named group of features, their entity key, and where they come from."""

    dataset: DatasetId
    entity: str = ""
    features: tuple[str, ...] = ()
    sources: list[DatasetId] = field(default_factory=list)
    ttl: timedelta | None = None
    online_dataset: DatasetId | None = None
    last_materialized: datetime | None = None

    @property
    def name(self) -> str:
        """Fully qualified feature name, view included."""
        return self.dataset.name

    def summary(self) -> str:
        """The comparison as text."""
        ttl = f", ttl {self.ttl}" if self.ttl else ""
        return (
            f"{self.dataset}: {len(self.features)} feature(s) on `{self.entity or '?'}` "
            f"from {len(self.sources)} source(s){ttl}"
        )


def record_feature_view(
    graph: Graph,
    view: FeatureView,
    *,
    source_specs: Mapping[DatasetId, PartitionSpec] | None = None,
) -> Graph:
    """Wire a feature view's sources into the graph.

    Source edges carry a real mapping derived from the two specs, because a feature
    view genuinely is a partition-wise transformation of its inputs — one changed day
    upstream is one changed day of features. The online copy gets an unbounded edge,
    since materialization to a key-value store is not partition-preserving in any way
    this can prove.
    """
    declared = dict(source_specs or {})
    view_spec = graph.spec(view.dataset)
    if not view_spec.fields:
        view_spec = spec_for(AssetKind.FEATURE_VIEW)
    graph.add_dataset(view.dataset, view_spec)

    for source in view.sources:
        source_spec = declared.get(source, graph.spec(source))
        graph.add_dataset(source, source_spec)
        mapping = (
            PartitionMapping.rollup(source_spec, view_spec)
            if source_spec.fields and view_spec.fields
            else PartitionMapping.unknown(view_spec)
        )
        link(
            graph,
            source,
            view.dataset,
            evidence="feature:offline",
            mapping=mapping,
            columns=((feature, feature) for feature in view.features),
        )

    if view.online_dataset is not None:
        link(
            graph,
            view.dataset,
            view.online_dataset,
            evidence="feature:online",
            columns=((feature, feature) for feature in view.features),
        )
    return graph


def feature_dependencies(graph: Graph, view: DatasetId) -> list[DatasetId]:
    """Everything a feature view is computed from, transitively."""
    return ancestors(graph, view)


def features_for_model(graph: Graph, model: DatasetId) -> list[DatasetId]:
    """Feature views feeding a model."""
    from .assets import kind_of

    return sorted(
        {ds for ds in ancestors(graph, model) if kind_of(ds) is AssetKind.FEATURE_VIEW}, key=str
    )


def models_using(graph: Graph, view: DatasetId) -> list[DatasetId]:
    """Models that depend on a feature view. The blast radius of changing it."""
    return sorted({ds for ds in descendants(graph, view) if is_model(ds)}, key=str)


# -- leakage -------------------------------------------------------------------


def leaky_features(
    graph: Graph, view: DatasetId, label: ColumnRef, *, max_depth: int = 8
) -> list[ColumnRef]:
    """Features derived, directly or transitively, from the label column.

    Target leakage is the single most common reason a model that validated at 0.98
    performs at chance in production, and it is invisible in the training data — the
    column is right there, computed correctly, from information that will not exist
    at prediction time.

    Requires column-level lineage on the path. Without it this returns nothing, which
    is a false negative and stated as such rather than approximated with a guess.
    """
    from ..graph.query import columns_of

    found: list[ColumnRef] = []
    for name in columns_of(graph, view):
        ref = ColumnRef(view, name)
        if label in column_ancestors(graph, ref, max_depth=max_depth):
            found.append(ref)
    return sorted(found, key=str)


def label_reaches_features(graph: Graph, label_dataset: DatasetId, view: DatasetId) -> bool:
    """Dataset-level leakage check, usable when column lineage is missing.

    Coarser than `leaky_features` and correspondingly noisier: a shared source table
    is normal. Worth investigating, not worth failing a build on.
    """
    return has_path(graph, label_dataset, view)


def leakage_path(graph: Graph, label_dataset: DatasetId, view: DatasetId) -> list[DatasetId]:
    """The shortest route by which a label reaches a feature view, for the write-up."""
    return shortest_path(graph, label_dataset, view) or []


# -- freshness -----------------------------------------------------------------


def freshness_age(view: FeatureView, *, now: datetime | None = None) -> timedelta | None:
    """How long since this view was last materialized, or None when unrecorded."""
    if view.last_materialized is None:
        return None
    return _age(view.last_materialized, reference=now)


def is_stale(view: FeatureView, *, now: datetime | None = None) -> bool:
    """True when the view is older than its own TTL.

    A view with no TTL is never stale, and a view never materialized always is —
    absence of a build is not freshness.
    """
    if view.ttl is None:
        return False
    age = freshness_age(view, now=now)
    return True if age is None else age > view.ttl


def stale_views(views: Iterable[FeatureView], *, now: datetime | None = None) -> list[FeatureView]:
    """Every view past its TTL, worst first."""
    late = [view for view in views if is_stale(view, now=now)]
    return sorted(late, key=lambda v: -(freshness_age(v, now=now) or timedelta(0)).total_seconds())


def serving_risks(
    graph: Graph, views: Sequence[FeatureView], *, now: datetime | None = None
) -> list[str]:
    """Everything about the current feature state that could produce a wrong prediction.

    Ordered by how directly it reaches a served model, because a stale feature nobody
    serves from is a hygiene issue and a stale feature behind a live model is an
    incident.
    """
    findings: list[str] = []
    for view in views:
        consumers = models_using(graph, view.dataset)
        if is_stale(view, now=now):
            age = freshness_age(view, now=now)
            detail = f"{age}" if age is not None else "never materialized"
            severity = "error" if consumers else "warn"
            findings.append(
                f"[{severity}] {view.dataset}: stale ({detail}, ttl {view.ttl}), "
                f"{len(consumers)} model(s) serving from it"
            )
        if view.online_dataset is None and consumers:
            findings.append(
                f"[warn] {view.dataset}: no online materialization recorded, so "
                "training/serving skew cannot be measured"
            )
        if not view.sources:
            findings.append(
                f"[warn] {view.dataset}: no sources recorded; its provenance is unknown"
            )
    return findings


# -- skew ----------------------------------------------------------------------


@dataclass
class SkewReport:
    """Distribution differences between offline and online copies of a feature view."""

    view: DatasetId
    findings: list[str] = field(default_factory=list)
    missing_online: list[str] = field(default_factory=list)
    missing_offline: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when no training-serving skew was found."""
        return not (self.findings or self.missing_online or self.missing_offline)

    def summary(self) -> str:
        """The comparison as text."""
        if self.is_clean:
            return f"{self.view}: offline and online agree"
        lines = [f"{self.view}: {len(self.findings)} skew finding(s)"]
        for name in self.missing_online:
            lines.append(f"  missing online: {name}")
        for name in self.missing_offline:
            lines.append(f"  missing offline: {name}")
        lines.extend(f"  {finding}" for finding in self.findings)
        return "\n".join(lines)


def skew(offline: Profile, online: Profile, *, tolerance: float = 0.05) -> SkewReport:
    """Compare two materializations of the same feature view.

    Reuses the drift comparison rather than inventing a second one, because "the same
    feature computed two ways disagrees" and "the same feature disagrees with
    yesterday" are the same question with different inputs.
    """
    report = SkewReport(view=offline.dataset)
    report.missing_online = sorted(set(offline.column_names) - set(online.column_names))
    report.missing_offline = sorted(set(online.column_names) - set(offline.column_names))
    report.findings = [
        str(finding)
        for finding in drift(offline, online, null_rate_tolerance=tolerance)
        if finding.kind != "row_count_shift"  # online holds current state, offline holds history
    ]
    return report


def training_serving_skew(offline: Profile, online: Profile, *, tolerance: float = 0.05) -> bool:
    """True when offline and online copies disagree beyond tolerance."""
    return not skew(offline, online, tolerance=tolerance).is_clean


# -- planning ------------------------------------------------------------------


def backfill_plan(
    graph: Graph, view: DatasetId, dirty: Mapping[DatasetId, Iterable[KeyPredicate]]
) -> InvalidationPlan:
    """Which feature partitions a source change obliges you to recompute.

    Feature backfills are the expensive half of most feature-store bills, and they
    are almost always run over a whole date range because nothing knows which slices
    moved.
    """
    return graph.invalidate(dirty)


def views_needing_backfill(
    graph: Graph, dirty: Mapping[DatasetId, Iterable[KeyPredicate]]
) -> list[DatasetId]:
    """Feature views affected by a source change."""
    from .assets import kind_of

    plan = graph.invalidate(dirty)
    return sorted(
        (ds for ds in plan.dirty if kind_of(ds) is AssetKind.FEATURE_VIEW),
        key=str,
    )


def point_in_time_violations(
    view: FeatureView, *, label_timestamp: str = "label_ts", feature_timestamp: str = "feature_ts"
) -> list[str]:
    """Structural reasons a view cannot support point-in-time-correct joins.

    Not a data check — it does not read rows. It checks that the view records the two
    timestamps such a join needs, because a view missing them cannot be joined
    correctly no matter how careful the query is.
    """
    findings: list[str] = []
    names = set(view.features)
    if label_timestamp not in names and feature_timestamp not in names:
        findings.append(
            f"{view.dataset}: neither `{feature_timestamp}` nor `{label_timestamp}` is "
            "recorded, so a point-in-time join cannot be verified as correct"
        )
    if not view.entity:
        findings.append(f"{view.dataset}: no entity key recorded; joins cannot be keyed")
    return findings
