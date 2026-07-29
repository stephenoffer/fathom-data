"""Labels, propagation, and enforcement — the `label` verb.

Nobody hand-annotates forty thousand columns, so inference proposes and a human
confirms. Confirmation is sticky: re-running inference never overwrites a decision
somebody made.

Two honest limits, stated up front rather than discovered later:

- **Footer-only inference is name-driven.** Parquet footers give min/max, null
  counts, and types — not value vocabularies. So a column called `email` is
  proposed as an email by its name and type, at a confidence that says so. Raising
  that confidence requires sampling real values, which costs money and is opt-in.
- **Propagation is only as good as column lineage.** Where an edge carries no
  column detail we propagate to the target as an unattributed label rather than
  guessing which column inherited it. Enforcement reads that as "may contain",
  which is the safe direction for a policy check.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..core.types import ColumnRef, DatasetId
from ..graph.model import Graph
from ..observe.profile import ColumnProfile, Profile

__all__ = [
    "Label",
    "LabelSet",
    "PolicyReport",
    "SinkPolicy",
    "UNATTRIBUTED",
    "Violation",
    "enforce",
    "infer",
    "labels_over",
    "propagate",
    "tag_index",
]

# Marks a label we know reached a dataset but cannot pin to a column.
UNATTRIBUTED = "*"

# Labels implying the data identifies a person. Kept separate from the detectors so
# a new detector only has to say what it finds, not what that means downstream.
_PII_LABELS = frozenset(
    {
        "email",
        "phone",
        "national_id",
        "date_of_birth",
        "person_name",
        "postal_address",
        "ip_address",
        "device_id",
        # A customer id singles a person out, which is what makes it personal data
        # under GDPR Art. 4(1) even though it is not a name. It is also the column
        # `erase` keys on: a sink policy forbidding `pii` that waves through a table
        # of user_ids is failing at exactly the thing it exists to catch.
        "user_identifier",
    }
)


@dataclass(frozen=True, order=True)
class Label:
    """One claim about what a column means or what policy attaches to it."""

    name: str
    confidence: float = 1.0
    origin: str = "declared"
    confirmed: bool = False

    def damped(self, factor: float, origin: str) -> Label:
        """A propagated copy. Confidence decays with distance from the evidence."""
        return Label(
            name=self.name,
            confidence=round(self.confidence * factor, 4),
            origin=origin,
            confirmed=False,
        )


LabelSet = dict[ColumnRef, set[Label]]


@dataclass(frozen=True)
class _Detector:
    label: str
    pattern: re.Pattern[str]
    types: tuple[str, ...] = ()  # substring match against the column's dtype
    confidence: float = 0.6


# Name-and-type heuristics. Confidence tops out well below 1.0 on purpose: these are
# proposals for a human to confirm, not conclusions.
_DETECTORS: tuple[_Detector, ...] = (
    _Detector("email", re.compile(r"(^|_)e?_?mail(_|$)"), ("string", "utf8"), 0.75),
    _Detector("phone", re.compile(r"(^|_)(phone|mobile|msisdn|tel)(_|$)"), ("string", "utf8"), 0.7),
    _Detector(
        "national_id",
        re.compile(r"(^|_)(ssn|nino|national_id|tax_id|passport)(_|$)"),
        ("string", "utf8"),
        0.8,
    ),
    _Detector("date_of_birth", re.compile(r"(^|_)(dob|birth_?date|date_of_birth)(_|$)"), (), 0.8),
    _Detector(
        "person_name",
        re.compile(r"(^|_)(first_name|last_name|full_name|surname|given_name)(_|$)"),
        ("string", "utf8"),
        0.7,
    ),
    _Detector(
        "postal_address",
        # Anchored to postal-specific tokens: a bare `address` counts, but
        # `email_address` and `ip_address` must not.
        re.compile(
            r"^address$|(^|_)(street|postcode|post_code|zip_?code"
            r"|billing_address|shipping_address|postal_address)(_|$)"
        ),
        ("string", "utf8"),
        0.65,
    ),
    _Detector(
        "ip_address", re.compile(r"(^|_)(ip|ip_address|client_ip)(_|$)"), ("string", "utf8"), 0.7
    ),
    _Detector("device_id", re.compile(r"(^|_)(device_id|idfa|advertising_id)(_|$)"), (), 0.7),
    _Detector(
        "currency_code", re.compile(r"(^|_)(currency|ccy)(_code)?(_|$)"), ("string", "utf8"), 0.7
    ),
    _Detector(
        "monetary_amount",
        re.compile(r"(^|_)(amount|price|revenue|cost|total|balance|fee)(_|$)"),
        ("double", "float", "decimal", "int"),
        0.6,
    ),
    _Detector("minor_units", re.compile(r"_(cents|pence|minor)$"), ("int",), 0.85),
    _Detector("latitude", re.compile(r"(^|_)(lat|latitude)(_|$)"), ("double", "float"), 0.6),
    _Detector("longitude", re.compile(r"(^|_)(lon|lng|longitude)(_|$)"), ("double", "float"), 0.6),
    _Detector(
        "user_identifier",
        re.compile(r"(^|_)(user_id|customer_id|account_id|subject_id)(_|$)"),
        (),
        0.7,
    ),
)


def _type_matches(column: ColumnProfile, detector: _Detector) -> bool:
    if not detector.types:
        return True
    dtype = column.dtype.lower()
    return any(t in dtype for t in detector.types)


def _range_supports(column: ColumnProfile, label: str) -> bool | None:
    """Use footer min/max to corroborate or refute a name-based guess.

    Returns None when the statistics cannot speak to it.
    """
    if column.min is None or column.max is None:
        return None
    try:
        low, high = float(column.min), float(column.max)
    except (TypeError, ValueError):
        return None
    if label == "latitude":
        return low >= -90.0 and high <= 90.0
    if label == "longitude":
        return low >= -180.0 and high <= 180.0
    if label == "minor_units":
        # Minor units are integral; a fractional value refutes the name.
        return float(low).is_integer() and float(high).is_integer()
    return None


def infer(profile: Profile) -> LabelSet:
    """Propose labels for one dataset's columns from a profile.

    Where footer statistics can corroborate a name-based guess they raise its
    confidence; where they contradict it, the guess is dropped entirely. A column
    called `latitude` holding values up to 4000 is not a latitude.
    """
    found: LabelSet = {}
    for column in profile.columns:
        name = column.name.lower()
        labels: set[Label] = set()
        for detector in _DETECTORS:
            if not detector.pattern.search(name) or not _type_matches(column, detector):
                continue
            support = _range_supports(column, detector.label)
            if support is False:
                continue  # statistics refute the name
            confidence = min(0.95, detector.confidence + 0.15) if support else detector.confidence
            labels.add(
                Label(
                    name=detector.label,
                    confidence=confidence,
                    origin="inferred:name" if support is None else "inferred:name+stats",
                )
            )
        if any(label.name in _PII_LABELS for label in labels):
            best = max(label.confidence for label in labels if label.name in _PII_LABELS)
            labels.add(Label(name="pii", confidence=best, origin="implied"))
        if labels:
            found[ColumnRef(profile.dataset, column.name)] = labels
    return found


def _merge(target: LabelSet, ref: ColumnRef, label: Label) -> bool:
    """Add a label, keeping the strongest claim per name. True when something changed."""
    existing = target.setdefault(ref, set())
    prior = next((x for x in existing if x.name == label.name), None)
    if prior is None:
        existing.add(label)
        return True
    if prior.confirmed:
        return False  # a human decision outranks any inference
    if label.confirmed or label.confidence > prior.confidence:
        existing.discard(prior)
        existing.add(label)
        return True
    return False


def propagate(
    graph: Graph,
    seeds: LabelSet,
    *,
    damping: float = 0.95,
    max_rounds: int = 32,
) -> LabelSet:
    """Flow labels downstream along column edges to a fixpoint.

    Labels travel the same edges dirtiness does — a column derived from an email
    column is still email-derived. Confidence decays per hop so a label six
    transformations away does not read as strongly as one at the source.

    Edges with no column detail deposit the label on the target as `UNATTRIBUTED`
    rather than being dropped. Losing track of PII because a Spark job used the
    DataFrame API is worse than an imprecise warning.

    An `UNATTRIBUTED` label keeps travelling across edges that *do* carry column
    detail. Column detail says which columns map where; it does not say the
    unattributed label is absent. Stopping there dropped PII entirely at the first
    SQL model below a DataFrame job, which is a very ordinary shape.
    """
    labels: LabelSet = {ref: set(values) for ref, values in seeds.items()}

    # Index by dataset. Rescanning every label for every edge is O(labels x edges)
    # per round, and the README's own example is 40,000 columns.
    by_dataset: dict[DatasetId, dict[ColumnRef, set[Label]]] = defaultdict(dict)
    for ref, values in labels.items():
        by_dataset[ref.dataset][ref] = values

    def merge(ref: ColumnRef, label: Label) -> bool:
        """Combine two label sets, keeping the strongest claim per label name."""
        was_present = ref in labels
        if _merge(labels, ref, label):
            if not was_present:
                by_dataset[ref.dataset][ref] = labels[ref]
            return True
        # `_merge` creates the entry even when it changes nothing, so keep the index
        # consistent with `labels` either way.
        if not was_present and ref in labels:
            by_dataset[ref.dataset][ref] = labels[ref]
        return False

    for _round in range(max_rounds):
        changed = False
        for edge in graph.edges:
            source_labels = by_dataset.get(edge.src)
            if not source_labels:
                continue

            for label in list(source_labels.get(ColumnRef(edge.src, UNATTRIBUTED), ())):
                moved = label.damped(damping, f"propagated-unattributed:{edge.src}")
                changed |= merge(ColumnRef(edge.dst, UNATTRIBUTED), moved)

            if edge.columns:
                for src_col, dst_col in edge.columns:
                    for label in list(source_labels.get(ColumnRef(edge.src, src_col), ())):
                        moved = label.damped(damping, f"propagated:{edge.src}")
                        changed |= merge(ColumnRef(edge.dst, dst_col), moved)
            else:
                # No column-level lineage on this edge: we know the label reached the
                # target dataset but not which column carries it.
                for ref, values in list(source_labels.items()):
                    if ref.column == UNATTRIBUTED:
                        continue  # already carried across above
                    for label in list(values):
                        moved = label.damped(damping, f"propagated-unattributed:{edge.src}")
                        changed |= merge(ColumnRef(edge.dst, UNATTRIBUTED), moved)
        if not changed:
            break

    return labels


def labels_over(labels: LabelSet, datasets: Iterable[DatasetId]) -> dict[str, list[ColumnRef]]:
    """Labels carried by any of `datasets`, grouped by label name.

    Three modules ask the same question of three different reaches — what a context
    window pulled in, what an agent run could see, what a prompt interpolates — and
    the only thing that differs between them is which datasets they pass. Grouping by
    label name rather than by column is what those callers want: the question is
    "did personal data get here", and the columns are the evidence for the answer.
    """
    reachable = set(datasets)
    found: dict[str, list[ColumnRef]] = {}
    for ref, values in labels.items():
        if ref.dataset not in reachable:
            continue
        for label in values:
            found.setdefault(label.name, []).append(ref)
    return {name: sorted(refs, key=str) for name, refs in sorted(found.items())}


@dataclass(frozen=True)
class SinkPolicy:
    """What may and must reach a given dataset."""

    dataset: DatasetId
    forbid: frozenset[str] = frozenset()
    require: frozenset[str] = frozenset()
    reason: str = ""

    @classmethod
    def no_pii(
        cls, dataset: DatasetId, *, reason: str = "sink is not cleared for personal data"
    ) -> SinkPolicy:
        """A policy forbidding anything implying personal data."""
        return cls(dataset=dataset, forbid=frozenset({"pii"}), reason=reason)


@dataclass(frozen=True)
class Violation:
    """A policy breach, with enough detail to act on."""

    dataset: DatasetId
    column: str
    label: str
    rule: str
    confidence: float
    reason: str = ""

    @property
    def is_unattributed(self) -> bool:
        """True when the label reached the dataset but not a known column."""
        return self.column == UNATTRIBUTED

    def __str__(self) -> str:
        where = "(column unknown)" if self.is_unattributed else self.column
        detail = f" — {self.reason}" if self.reason else ""
        return (
            f"{self.dataset} {where}: {self.rule} {self.label!r} "
            f"(confidence {self.confidence:.0%}){detail}"
        )


@dataclass
class PolicyReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no policy was violated."""
        return not self.violations

    def summary(self) -> str:
        """The report as text, one line per violation."""
        if self.ok:
            return "policy: no violations"
        lines = [f"policy: {len(self.violations)} violation(s)"]
        lines.extend(f"  {v}" for v in self.violations)
        return "\n".join(lines)


def enforce(
    labels: LabelSet,
    policies: Iterable[SinkPolicy],
    *,
    min_confidence: float = 0.5,
) -> PolicyReport:
    """Check labels against sink policies.

    `min_confidence` keeps weak inferences from blocking a pipeline. Confirmed
    labels always count regardless of confidence, because a human already decided.
    """
    # Two policies naming the same sink are both meant. Keeping only the last one
    # silently drops half the rules a user wrote, so they are unioned instead.
    merged: dict[DatasetId, SinkPolicy] = {}
    for rule in policies:
        prior = merged.get(rule.dataset)
        if prior is None:
            merged[rule.dataset] = rule
            continue
        merged[rule.dataset] = SinkPolicy(
            dataset=rule.dataset,
            forbid=prior.forbid | rule.forbid,
            require=prior.require | rule.require,
            reason="; ".join(r for r in (prior.reason, rule.reason) if r),
        )
    by_dataset: Mapping[DatasetId, SinkPolicy] = merged
    report = PolicyReport()

    def counts(label: Label) -> bool:
        """A label strong enough to act on — the same bar in both directions."""
        return label.confirmed or label.confidence >= min_confidence

    for ref, values in sorted(labels.items(), key=lambda kv: (str(kv[0].dataset), kv[0].column)):
        policy = by_dataset.get(ref.dataset)
        if policy is None:
            continue
        for label in sorted(values):
            if label.name not in policy.forbid:
                continue
            if not counts(label):
                continue
            report.violations.append(
                Violation(
                    dataset=ref.dataset,
                    column=ref.column,
                    label=label.name,
                    rule="forbidden label",
                    confidence=label.confidence,
                    reason=policy.reason,
                )
            )

    for policy in by_dataset.values():
        if not policy.require:
            continue
        # Same confidence bar as `forbid`. Without it a label damped to 0.19 over six
        # propagation hops satisfies a requirement that a 0.49 direct inference would
        # not, so `require` silently passes on evidence `forbid` would have ignored.
        present = {
            label.name
            for ref, values in labels.items()
            if ref.dataset == policy.dataset
            for label in values
            if counts(label)
        }
        for needed in sorted(policy.require - present):
            report.violations.append(
                Violation(
                    dataset=policy.dataset,
                    column=UNATTRIBUTED,
                    label=needed,
                    rule="required label missing",
                    confidence=1.0,
                    reason=policy.reason,
                )
            )

    return report


def tag_index(labels: LabelSet) -> dict[DatasetId, set[str]]:
    """Dataset to the label names it carries — the shape `tag:` selection wants.

    Selection has no business knowing about confidence or provenance, so this flattens
    a `LabelSet` to the only part a selector can act on.
    """
    out: dict[DatasetId, set[str]] = {}
    for ref, values in labels.items():
        out.setdefault(ref.dataset, set()).update(label.name for label in values)
    return out
