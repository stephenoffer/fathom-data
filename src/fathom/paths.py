"""Turning object paths into partition keys.

Hive-style layouts (``dt=2026-01-15/region=eu/``) are self-describing and parse
directly. Everything else — ``/2026/01/15/``, ``/year=2026/month=01/`` — needs a
declared template, because guessing which path segment is a month is exactly the
kind of inference that silently corrupts a rebuild plan.

Without a template a non-Hive path yields no bindings, which the planner reads as
"the whole dataset". Coarse, but never wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .grains import Grain, truncate
from .types import ANY, KeyPredicate, PartitionSpec

__all__ = ["PathTemplate", "key_from_path", "parse_hive_partitions"]

_HIVE_SEGMENT = re.compile(r"^([^=/]+)=([^/]*)$")

# Template placeholders and the regex each expands to.
_TOKENS: dict[str, str] = {
    "yyyy": r"(?P<yyyy>\d{4})",
    "MM": r"(?P<MM>\d{2})",
    "dd": r"(?P<dd>\d{2})",
    "HH": r"(?P<HH>\d{2})",
}

# Common spellings of a date written into a single path segment.
_DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y")


def parse_hive_partitions(path: str) -> dict[str, str]:
    """Extract ``key=value`` bindings from a path, ignoring the filename."""
    out: dict[str, str] = {}
    segments = path.strip("/").split("/")
    for segment in segments:
        m = _HIVE_SEGMENT.match(segment)
        if m:
            out[m.group(1)] = m.group(2)
    return out


@dataclass(frozen=True)
class PathTemplate:
    """A declared layout for paths that are not self-describing.

    ``events/{yyyy}/{MM}/{dd}`` binds a `dt` time field; ``{region}`` binds a value
    field of that name. Anchored at the end so the filename is ignored.
    """

    template: str
    time_field: str = "dt"

    def _pattern(self) -> re.Pattern[str]:
        parts: list[str] = []
        for literal, token in re.findall(r"([^{]*)(?:\{([^}]+)\})?", self.template):
            parts.append(re.escape(literal))
            if token:
                parts.append(_TOKENS.get(token, rf"(?P<{token}>[^/]+)"))
        return re.compile("".join(parts))

    def extract(self, path: str) -> dict[str, str]:
        m = self._pattern().search(path)
        return {k: v for k, v in m.groupdict().items() if v is not None} if m else {}


def _coerce_time(raw: str, grain: Grain) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return truncate(datetime.strptime(raw, fmt), grain)
        except ValueError:
            continue
    return None


def _assemble_time(bits: dict[str, str], grain: Grain) -> datetime | None:
    """Build a datetime from separate {yyyy}/{MM}/{dd}/{HH} captures."""
    if "yyyy" not in bits:
        return None
    try:
        return truncate(
            datetime(
                year=int(bits["yyyy"]),
                month=int(bits.get("MM", 1)),
                day=int(bits.get("dd", 1)),
                hour=int(bits.get("HH", 0)),
            ),
            grain,
        )
    except ValueError:
        return None


def key_from_path(
    path: str,
    spec: PartitionSpec,
    *,
    template: PathTemplate | None = None,
) -> KeyPredicate:
    """Derive a partition key from an object path.

    Any field we cannot resolve binds to ANY, widening rather than guessing.
    """
    if not spec.fields:
        return KeyPredicate()

    raw = dict(parse_hive_partitions(path))
    if template is not None:
        raw.update(template.extract(path))

    bindings: list[tuple[str, object]] = []
    for f in spec.fields:
        if f.kind == "value":
            bindings.append((f.name, raw.get(f.name, ANY)))
            continue

        assert f.grain is not None
        value: datetime | None = None
        if f.name in raw:
            value = _coerce_time(raw[f.name], f.grain)
        if value is None and template is not None and f.name == template.time_field:
            value = _assemble_time(raw, f.grain)
        bindings.append((f.name, value if value is not None else ANY))

    return KeyPredicate(bindings=tuple(bindings))
