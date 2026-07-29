"""The partition mapping lattice.

This is the core of the planner. For every edge in the dependency graph we hold a
mapping that answers: if this input partition is dirty, which output partitions are
dirty? Mappings compose along paths and join where paths reconverge, and the whole
structure is a lattice whose top element is "everything".

The single invariant, which every operation here preserves:

    apply(compose(m1, m2), k)  ⊇  ⋃ { apply(m2, j) for j in apply(m1, k) }

That is, the composed mapping never claims fewer dirty partitions than walking the
two edges one at a time would. Precision is an optimization. Soundness is not.
Anything we cannot prove widens to `UNBOUNDED`, which costs compute and never
costs correctness.

Only three field-level forms are needed in practice:

    TimeWindow   the output bucket is a time bucket offset from the input's
    Passthrough  the output value is the input value, unchanged
    Unbounded    we could not prove a relationship

Time windows only ever coarsen. A daily source feeding a monthly table is a rollup
we can reason about precisely; a monthly source feeding an hourly table is a
refinement whose honest answer is "some large part of the month", so it widens to
`UNBOUNDED` rather than pretending to a precision we do not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeAlias

from .grains import Grain, convert_window, span, step, truncate
from .types import ANY, KeyPredicate, PartitionSpec

__all__ = [
    "UNBOUNDED",
    "FieldMapping",
    "PartitionMapping",
    "Passthrough",
    "TimeWindow",
    "Unbounded",
    "apply",
    "compose",
    "join",
    "leq",
]

# Ceiling on how many concrete partitions a single `apply` may enumerate. Past this
# we collapse dimensions to ANY, which widens the result. Without a cap, an hourly
# table fed by a yearly one would try to enumerate 8,784 keys per dirty partition.
MAX_ENUMERATED_KEYS = 4096


@dataclass(frozen=True)
class TimeWindow:
    """Output buckets `[lo, hi]` (inclusive, in `out_grain` units) around the input.

    `TimeWindow(src, 0, 0, DAY, DAY)` is identity. `(src, 0, 0, DAY, MONTH)` is a
    daily source rolled up to a monthly table. `(src, 0, 6, DAY, DAY)` is a seven-day
    trailing aggregate: one dirty day taints the six days that follow it.

    Offsets are counted in `out_grain` units from the input key's own `out_grain`
    bucket. `out_grain` may not be finer than `in_grain`; use `UNBOUNDED` for that.

    The three named constructors cover almost every real edge, and read better than
    the positional form at a call site:

    Example:
        >>> TimeWindow.identity("dt", Grain.DAY).explain()
        'a dirty dt taints the same day'
        >>> TimeWindow.rollup("dt", Grain.DAY, Grain.MONTH).explain()
        'a dirty dt day taints the month containing it'
        >>> TimeWindow.trailing("dt", 7, Grain.DAY).explain()
        'a dirty dt taints that day and the 6 days after it'
    """

    source: str
    lo: int
    hi: int
    in_grain: Grain
    out_grain: Grain

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(
                f"empty window [{self.lo}, {self.hi}] on {self.source!r}: `lo` must not "
                f"exceed `hi`. Offsets run from earliest to latest, so a 7-day trailing "
                f"window is (0, 6), not (6, 0) — or use "
                f"`TimeWindow.trailing({self.source!r}, 7, Grain.DAY)`"
            )
        if self.out_grain < self.in_grain:
            raise ValueError(
                f"{self.source!r}: output grain {self.out_grain.label} is finer than input "
                f"grain {self.in_grain.label}; refinement must be expressed as UNBOUNDED. "
                f"One dirty {self.in_grain.label} could touch any "
                f"{self.out_grain.label} inside it, and claiming a narrower reach is the "
                f"one error that serves stale data"
            )

    @classmethod
    def identity(cls, field: str, grain: Grain | str) -> TimeWindow:
        """The common case: one dirty input bucket dirties the same output bucket.

        Example:
            >>> TimeWindow.identity("dt", "day")
            TimeWindow('dt', 0, 0, day, day)
        """
        g = Grain.parse(grain)
        return cls(field, 0, 0, g, g)

    @classmethod
    def rollup(cls, field: str, frm: Grain | str, to: Grain | str) -> TimeWindow:
        """A finer source feeding a coarser table: daily rows into a monthly total.

        Example:
            >>> TimeWindow.rollup("dt", "day", "month")
            TimeWindow('dt', 0, 0, day, month)
        """
        return cls(field, 0, 0, Grain.parse(frm), Grain.parse(to))

    @classmethod
    def trailing(cls, field: str, length: int, grain: Grain | str) -> TimeWindow:
        """A rolling aggregate `length` buckets wide.

        A 7-day trailing average reads each day into that day's window and the six
        after it, so one restated day dirties seven outputs. Stated as a length
        rather than as offsets, because off-by-one here silently under-invalidates.

        Args:
            field: The time field both sides are keyed on.
            length: How many buckets each output aggregates over. Must be positive.
            grain: The bucket size, on both sides.

        Example:
            >>> TimeWindow.trailing("dt", 7, "day")
            TimeWindow('dt', 0, 6, day, day)
        """
        if length < 1:
            raise ValueError(
                f"a trailing window over {field!r} must cover at least one bucket, "
                f"not {length}. A 7-day rolling aggregate is `length=7`"
            )
        g = Grain.parse(grain)
        return cls(field, 0, length - 1, g, g)

    def explain(self) -> str:
        """This mapping as a sentence, for people rather than for the planner.

        Example:
            >>> TimeWindow("dt", -1, 1, Grain.DAY, Grain.DAY).explain()
            'a dirty dt taints the day before it, that day, and the day after it'
        """
        unit = self.out_grain.label
        if self.in_grain is not self.out_grain:
            if (self.lo, self.hi) == (0, 0):
                return (
                    f"a dirty {self.source} {self.in_grain.label} taints the {unit} containing it"
                )
            return (
                f"a dirty {self.source} {self.in_grain.label} taints {self.hi - self.lo + 1} "
                f"{unit}(s) around the {unit} containing it"
            )
        if (self.lo, self.hi) == (0, 0):
            return f"a dirty {self.source} taints the same {unit}"
        if self.lo == 0:
            return f"a dirty {self.source} taints that {unit} and the {self.hi} {unit}s after it"
        if self.hi == 0:
            return f"a dirty {self.source} taints that {unit} and the {-self.lo} {unit}s before it"
        if (self.lo, self.hi) == (-1, 1):
            return (
                f"a dirty {self.source} taints the {unit} before it, that {unit}, "
                f"and the {unit} after it"
            )
        return (
            f"a dirty {self.source} taints {self.hi - self.lo + 1} {unit}s, from "
            f"{self.lo:+d} to {self.hi:+d} around it"
        )

    def __repr__(self) -> str:
        return (
            f"TimeWindow({self.source!r}, {self.lo}, {self.hi}, {self.in_grain}, {self.out_grain})"
        )

    def __str__(self) -> str:
        window = "" if (self.lo, self.hi) == (0, 0) else f"[{self.lo:+d},{self.hi:+d}]"
        grains = (
            self.in_grain.label
            if self.in_grain is self.out_grain
            else f"{self.in_grain.label}->{self.out_grain.label}"
        )
        return f"{self.source}{window}@{grains}"


@dataclass(frozen=True)
class Passthrough:
    """The output field carries the input field's value unchanged.

    What a non-time dimension almost always does: rows for ``region=eu`` produce
    rows for ``region=eu``, so a dirty EU partition never touches the US one.

    Example:
        >>> Passthrough("region").explain()
        'a dirty region taints the output rows with that same region'
    """

    source: str

    def __str__(self) -> str:
        return f"{self.source}="

    def __repr__(self) -> str:
        return f"Passthrough({self.source!r})"

    def explain(self) -> str:
        """This mapping as a sentence, for people rather than for the planner."""
        return f"a dirty {self.source} taints the output rows with that same {self.source}"


@dataclass(frozen=True)
class Unbounded:
    """No provable relationship. Every value of this field is potentially affected.

    The top of the lattice, and the honest answer wherever proof ran out: an opaque
    UDF, an unparseable dialect, a `MERGE`, a spec mismatch, a cycle. It costs
    compute — the whole dataset is rebuilt — and never costs correctness.

    A plan that rebuilds more than you expected usually has these in it.
    `metrics.coverage(graph)` counts them, and `fathom doctor` names the edges.

    Example:
        >>> UNBOUNDED.explain()
        'any dirty input taints every value of this field, because no relationship was provable'
    """

    def __str__(self) -> str:
        return "*"

    def __repr__(self) -> str:
        return "UNBOUNDED"

    def explain(self) -> str:
        """This mapping as a sentence, for people rather than for the planner."""
        return (
            "any dirty input taints every value of this field, because no relationship was provable"
        )


FieldMapping: TypeAlias = "TimeWindow | Passthrough | Unbounded"

UNBOUNDED = Unbounded()


@dataclass(frozen=True)
class PartitionMapping:
    """A per-output-field mapping describing one graph edge.

    One of these hangs on every edge in the graph, and it is the whole reason a plan
    can be narrower than "rebuild everything downstream". A field with no entry is
    `UNBOUNDED`, so an incomplete mapping is coarse rather than wrong.

    Example:
        >>> m = PartitionMapping.of(
        ...     dt=TimeWindow.rollup("dt", "day", "month"),
        ...     region=Passthrough("region"),
        ... )
        >>> print(m)
        {dt: dt@day->month, region: region=}
        >>> print(m.explain())
        dt: a dirty dt day taints the month containing it
        region: a dirty region taints the output rows with that same region
    """

    fields: tuple[tuple[str, FieldMapping], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(sorted(self.fields, key=lambda kv: kv[0])))

    def get(self, name: str) -> FieldMapping:
        """The mapping for one output field, `UNBOUNDED` when unmapped."""
        for k, v in self.fields:
            if k == name:
                return v
        return UNBOUNDED

    @property
    def is_unbounded(self) -> bool:
        """True when no field carries a provable relationship."""
        return all(isinstance(v, Unbounded) for _, v in self.fields)

    @classmethod
    def of(cls, **fields: FieldMapping) -> PartitionMapping:
        """Build a mapping from keyword field names.

        Keywords are *output* field names; each value says where that output field's
        dirtiness comes from. Fields you leave out are `UNBOUNDED`.

        Example:
            >>> PartitionMapping.of(region=Passthrough("region")).get("region")
            Passthrough('region')
            >>> PartitionMapping.of(region=Passthrough("region")).get("tenant")
            UNBOUNDED
        """
        return cls(fields=tuple(fields.items()))

    @classmethod
    def identity(cls, spec: PartitionSpec) -> PartitionMapping:
        """Each output field takes the same-named input field, unchanged.

        The right mapping for a filter, a rename, or any 1:1 transform — the shape
        most edges in a warehouse actually have.

        Example:
            >>> print(PartitionMapping.identity(PartitionSpec.parse("dt:day, region")))
            {dt: dt@day, region: region=}
        """
        out: list[tuple[str, FieldMapping]] = []
        for f in spec.fields:
            if f.kind == "time":
                assert f.grain is not None
                out.append((f.name, TimeWindow(f.name, 0, 0, f.grain, f.grain)))
            else:
                out.append((f.name, Passthrough(f.name)))
        return cls(fields=tuple(out))

    @classmethod
    def unknown(cls, spec: PartitionSpec) -> PartitionMapping:
        """The top of the lattice: any input partition dirties the whole output.

        The correct mapping whenever the relationship could not be proven. Coarse,
        never wrong, and what every unparseable edge falls back to.

        Example:
            >>> PartitionMapping.unknown(PartitionSpec.parse("dt:day")).is_unbounded
            True
        """
        return cls(fields=tuple((f.name, UNBOUNDED) for f in spec.fields))

    @classmethod
    def rollup(cls, src: PartitionSpec, dst: PartitionSpec) -> PartitionMapping:
        """Derive a mapping from two specs that share field names.

        Same-named time fields become a grain change, same-named value fields become
        passthrough, and anything present in `dst` but absent from `src` is unbounded.

        Start here when both sides are declared and the transform does not shift
        time — it infers the whole mapping from the two specs, which is right far
        more often than it is worth writing out by hand.

        Example:
            >>> daily = PartitionSpec.parse("dt:day, region")
            >>> monthly = PartitionSpec.parse("dt:month, region")
            >>> print(PartitionMapping.rollup(daily, monthly))
            {dt: dt@day->month, region: region=}
        """
        out: list[tuple[str, FieldMapping]] = []
        for f in dst.fields:
            g = src.field(f.name)
            if g is None or g.kind != f.kind:
                out.append((f.name, UNBOUNDED))
            elif f.kind == "time":
                assert g.grain is not None and f.grain is not None
                if f.grain < g.grain:  # coarse source feeding a finer table
                    out.append((f.name, UNBOUNDED))
                else:
                    out.append((f.name, TimeWindow(f.name, 0, 0, g.grain, f.grain)))
            else:
                out.append((f.name, Passthrough(f.name)))
        return cls(fields=tuple(out))

    def __str__(self) -> str:
        if not self.fields:
            return "{}"
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.fields) + "}"

    def __repr__(self) -> str:
        return f"PartitionMapping({str(self)})"

    def explain(self) -> str:
        """Every field's mapping as a sentence, one per line.

        For `fathom lineage --explain`, for a notebook, and for the review where
        somebody has to agree that this edge is described correctly. A mapping is
        the one thing here nobody can verify by reading the code that produced it.
        """
        if not self.fields:
            return "nothing is mapped, so any dirty input rebuilds the whole output"
        return "\n".join(f"{name}: {fm.explain()}" for name, fm in self.fields)


def _compose_field(first: PartitionMapping, second: FieldMapping) -> FieldMapping:
    """Compose one output field of `second` back through `first`."""
    if isinstance(second, Unbounded):
        return UNBOUNDED

    inner = first.get(second.source)
    if isinstance(inner, Unbounded):
        return UNBOUNDED

    if isinstance(second, Passthrough):
        # The middle field is copied straight through, so inherit whatever produced it.
        return inner

    # `second` is a TimeWindow over the middle dataset's field.
    if isinstance(inner, Passthrough):
        # The middle field is a verbatim copy, so the window applies to the origin field.
        return TimeWindow(inner.source, second.lo, second.hi, second.in_grain, second.out_grain)

    if inner.out_grain is not second.in_grain:
        # The two edges disagree about the middle dataset's grain. Widen rather than
        # guess: this usually means one side's partition spec was declared wrong.
        return UNBOUNDED

    converted = convert_window(inner.lo, inner.hi, inner.out_grain, second.out_grain)
    if converted is None:  # refinement; no useful bound
        return UNBOUNDED
    lo, hi = converted
    return TimeWindow(
        inner.source, lo + second.lo, hi + second.hi, inner.in_grain, second.out_grain
    )


def compose(first: PartitionMapping, second: PartitionMapping) -> PartitionMapping:
    """Collapse a two-edge path A→B→C into a single A→C mapping.

    Composition is what makes a plan transitive: seed the raw table, and the reach
    into a table three hops away is one mapping rather than three walks. The result
    covers at least everything walking the edges one at a time would — never less.

    Args:
        first: The A→B mapping.
        second: The B→C mapping, whose field sources name B's fields.

    Returns:
        An A→C mapping over C's fields.

    Example:
        >>> daily = PartitionSpec.parse("dt:day")
        >>> monthly = PartitionSpec.parse("dt:month")
        >>> a_to_b = PartitionMapping.identity(daily)
        >>> b_to_c = PartitionMapping.rollup(daily, monthly)
        >>> print(compose(a_to_b, b_to_c))
        {dt: dt[+0,+1]@day->month}

    The composed window is `[0, +1]` rather than `[0, 0]`, and that is the
    invariant, not a defect: a day-to-month conversion has to allow for the input
    day sitting anywhere inside its month, so the reach covers the following month
    too. Precision is an optimization; soundness is not.
    """
    return PartitionMapping(
        fields=tuple((name, _compose_field(first, fm)) for name, fm in second.fields)
    )


def _join_field(a: FieldMapping, b: FieldMapping) -> FieldMapping:
    """Least upper bound: the narrowest mapping that covers both."""
    if isinstance(a, Unbounded) or isinstance(b, Unbounded):
        return UNBOUNDED
    if isinstance(a, Passthrough) and isinstance(b, Passthrough):
        return a if a.source == b.source else UNBOUNDED
    if isinstance(a, TimeWindow) and isinstance(b, TimeWindow):
        if (a.source, a.in_grain, a.out_grain) != (b.source, b.in_grain, b.out_grain):
            return UNBOUNDED
        return TimeWindow(a.source, min(a.lo, b.lo), max(a.hi, b.hi), a.in_grain, a.out_grain)
    # A Passthrough and a TimeWindow carry incompatible information about grain.
    return UNBOUNDED


def join(a: PartitionMapping, b: PartitionMapping) -> PartitionMapping:
    """Combine two mappings that reach the same dataset by different paths.

    A diamond in the graph — two branches reconverging on one table — needs a single
    mapping covering both routes. This takes the narrowest one that does. Two windows
    over the same field widen to span both; anything less comparable widens to
    `UNBOUNDED`, which is the whole point: the union of two reaches is never smaller
    than either.

    Example:
        >>> same = PartitionSpec.parse("dt:day")
        >>> today = PartitionMapping.identity(same)
        >>> week = PartitionMapping.of(dt=TimeWindow.trailing("dt", 7, "day"))
        >>> print(join(today, week))
        {dt: dt[+0,+6]@day}
    """
    names = {k for k, _ in a.fields} | {k for k, _ in b.fields}
    return PartitionMapping(
        fields=tuple((n, _join_field(a.get(n), b.get(n))) for n in sorted(names))
    )


def leq(a: PartitionMapping, b: PartitionMapping) -> bool:
    """True when `b` covers at least everything `a` does. Used to detect a fixpoint.

    Also what the merge gate is built on: `b` covering `a` means the edit from `a`
    to `b` widened, which costs compute. The reverse narrowed, which can serve stale
    data — see `graph.diff.diff_graphs`.

    Example:
        >>> one = PartitionMapping.of(dt=TimeWindow.identity("dt", "day"))
        >>> week = PartitionMapping.of(dt=TimeWindow.trailing("dt", 7, "day"))
        >>> leq(one, week)     # the wider window covers the narrow one
        True
        >>> leq(week, one)
        False
    """
    names = {k for k, _ in a.fields} | {k for k, _ in b.fields}
    for n in names:
        x, y = a.get(n), b.get(n)
        if isinstance(y, Unbounded):
            continue
        if isinstance(x, Unbounded):
            return False
        if isinstance(x, Passthrough) and isinstance(y, Passthrough):
            if x.source != y.source:
                return False
        elif isinstance(x, TimeWindow) and isinstance(y, TimeWindow):
            if (x.source, x.in_grain, x.out_grain) != (y.source, y.in_grain, y.out_grain):
                return False
            if x.lo < y.lo or x.hi > y.hi:
                return False
        else:
            return False
    return True


def _field_values(fm: FieldMapping, key: KeyPredicate) -> list[Any] | None:
    """Concrete output values for one field, or None meaning unconstrained."""
    if isinstance(fm, Unbounded):
        return None

    src = key.get(fm.source)
    if src is ANY:
        return None

    if isinstance(fm, Passthrough):
        return [src]

    if not isinstance(src, datetime):
        # A time window over a non-datetime value: the spec and the data disagree.
        return None

    anchor = truncate(src, fm.out_grain)
    lo_dt = step(anchor, fm.lo, fm.out_grain)
    hi_dt = step(anchor, fm.hi, fm.out_grain)
    return list(span(lo_dt, hi_dt, fm.out_grain))


def apply(
    mapping: PartitionMapping,
    key: KeyPredicate,
    out_spec: PartitionSpec,
    *,
    max_keys: int = MAX_ENUMERATED_KEYS,
) -> frozenset[KeyPredicate]:
    """Dirty output partitions implied by one dirty input partition.

    The single call the whole planner is built out of: given one dirty input
    partition and the mapping on an edge, which partitions of the output went stale?

    Returns a set of predicates rather than concrete keys so an unconstrained
    dimension stays expressible. If enumeration would exceed `max_keys`, the widest
    dimensions collapse to ANY — a deliberate loss of precision that preserves the
    over-approximation invariant.

    Args:
        mapping: The edge's mapping, keyed by output field name.
        key: One dirty input partition.
        out_spec: The output dataset's partition spec.
        max_keys: Ceiling on enumerated partitions before dimensions widen to ANY.

    Returns:
        The output partitions this input dirties. Never empty: an input that maps to
        nothing provable yields the whole-dataset predicate.

    Example:
        >>> from datetime import datetime
        >>> daily = PartitionSpec.parse("dt:day")
        >>> monthly = PartitionSpec.parse("dt:month")
        >>> dirty_day = KeyPredicate.of(dt=datetime(2026, 3, 14))
        >>> sorted(str(k) for k in apply(PartitionMapping.rollup(daily, monthly),
        ...                              dirty_day, monthly))
        ['dt=2026-03-01T00:00:00']

    One dirty day resolves to exactly one dirty month. A seven-day trailing window
    over the same day resolves to seven days:

        >>> plan = apply(PartitionMapping.of(dt=TimeWindow.trailing("dt", 7, "day")),
        ...              dirty_day, daily)
        >>> len(plan)
        7
    """
    per_field: list[tuple[str, list[Any] | None]] = []
    for f in out_spec.fields:
        per_field.append((f.name, _field_values(mapping.get(f.name), key)))

    # Collapse the largest dimensions first until the cross product fits.
    def product(items: list[tuple[str, list[Any] | None]]) -> int:
        n = 1
        for _, vs in items:
            n *= 1 if vs is None else max(len(vs), 1)
        return n

    while product(per_field) > max_keys:
        widest = -1
        widest_size = 0
        for index, (_, values) in enumerate(per_field):
            if values is not None and len(values) > widest_size:
                widest, widest_size = index, len(values)
        if widest < 0:
            break  # every dimension is already unconstrained
        per_field[widest] = (per_field[widest][0], None)

    combos: list[list[tuple[str, Any]]] = [[]]
    for name, values in per_field:
        choices: list[Any] = [ANY] if values is None else values
        combos = [prefix + [(name, v)] for prefix in combos for v in choices]

    return frozenset(KeyPredicate(bindings=tuple(c)) for c in combos)
