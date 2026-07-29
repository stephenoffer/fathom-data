"""JSON encoding for the IR.

Partition values are not all strings. A `dt` binding is a datetime, a bucket number
is an int, and `ANY` is neither. Round-tripping through untagged JSON would turn
`datetime(2026, 3, 14)` into `"2026-03-14T00:00:00"` and quietly break every
comparison against a freshly computed key, so values carry their type.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .grains import Grain
from .partitions import UNBOUNDED, FieldMapping, PartitionMapping, Passthrough, TimeWindow
from .types import ANY, DatasetId, KeyPredicate, PartitionField, PartitionSpec

__all__ = [
    "dataset_from_json",
    "dataset_to_json",
    "key_from_json",
    "key_to_json",
    "mapping_from_json",
    "mapping_to_json",
    "spec_from_json",
    "spec_to_json",
    "stat_from_json",
    "stat_to_json",
]


def _value_to_json(value: Any) -> dict[str, Any]:
    if value is ANY:
        return {"t": "any"}
    # `datetime` before `date`, since every datetime is also a date.
    if isinstance(value, datetime):
        return {"t": "datetime", "v": value.isoformat()}
    if isinstance(value, date):
        return {"t": "date", "v": value.isoformat()}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "int", "v": value}
    if isinstance(value, float):
        return {"t": "float", "v": value}
    # A warehouse NUMERIC arrives as Decimal, and stringifying it here is how a
    # comparison against a freshly read value silently starts returning False.
    if isinstance(value, Decimal):
        return {"t": "decimal", "v": str(value)}
    if isinstance(value, bytes):
        return {"t": "bytes", "v": value.hex()}
    if value is None:
        return {"t": "null"}
    return {"t": "str", "v": str(value)}


def _value_from_json(blob: dict[str, Any]) -> Any:
    kind = blob["t"]
    if kind == "any":
        return ANY
    if kind == "null":
        return None
    if kind == "datetime":
        return datetime.fromisoformat(blob["v"])
    if kind == "date":
        return date.fromisoformat(blob["v"])
    if kind == "int":
        return int(blob["v"])
    if kind == "float":
        return float(blob["v"])
    if kind == "bool":
        return bool(blob["v"])
    if kind == "decimal":
        return Decimal(blob["v"])
    if kind == "bytes":
        return bytes.fromhex(blob["v"])
    return str(blob["v"])


def stat_to_json(value: Any) -> str | None:
    """Encode one profile statistic, preserving its type.

    Profile min/max are compared across runs to detect a narrowing range. Storing
    them as text makes every such comparison a `str` vs `float` `TypeError`, which
    `drift` swallows — so the check silently stops reporting anything at all.
    """
    return None if value is None else json.dumps(_value_to_json(value), separators=(",", ":"))


def stat_from_json(raw: str | None) -> Any:
    """Decode a statistic written by `stat_to_json`.

    Tolerates the bare strings written before statistics were typed, so an existing
    store keeps working rather than failing to open.
    """
    if raw is None:
        return None
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return _value_from_json(blob) if isinstance(blob, dict) and "t" in blob else raw


def dataset_to_json(ds: DatasetId) -> str:
    """Encode a dataset identity as compact JSON."""
    return json.dumps({"ns": ds.namespace, "name": ds.name}, separators=(",", ":"))


def dataset_from_json(raw: str) -> DatasetId:
    """Decode an identity written by `dataset_to_json`."""
    blob = json.loads(raw)
    return DatasetId(namespace=blob["ns"], name=blob["name"])


def key_to_json(key: KeyPredicate) -> str:
    """Encode a partition key, tagging each value with its type.

    The tag is the point: an untagged `2026-03-14T00:00:00` comes back a string and
    compares unequal to the datetime a fresh plan computes, so the planner silently
    matches nothing.
    """
    return json.dumps(
        [[name, _value_to_json(value)] for name, value in key.bindings], separators=(",", ":")
    )


def key_from_json(raw: str) -> KeyPredicate:
    """Decode a key written by `key_to_json`, restoring each value's type."""
    return KeyPredicate(bindings=tuple((n, _value_from_json(v)) for n, v in json.loads(raw)))


def spec_to_json(spec: PartitionSpec) -> str:
    """Encode a partition spec: field names, kinds, and time grains."""
    return json.dumps(
        [
            {"name": f.name, "kind": f.kind, "grain": f.grain.label if f.grain else None}
            for f in spec.fields
        ],
        separators=(",", ":"),
    )


def spec_from_json(raw: str) -> PartitionSpec:
    """Decode a spec written by `spec_to_json`.

    Rebuilt through `PartitionSpec.of`, so a stored spec with duplicate field names
    is rejected on load rather than producing a spec the planner cannot reason about.
    """
    fields = []
    for blob in json.loads(raw):
        if blob["kind"] == "time":
            fields.append(PartitionField.time(blob["name"], Grain.parse(blob["grain"])))
        else:
            fields.append(PartitionField.value(blob["name"]))
    return PartitionSpec.of(*fields)


def _field_mapping_to_json(fm: FieldMapping) -> dict[str, Any]:
    if isinstance(fm, TimeWindow):
        return {
            "k": "window",
            "src": fm.source,
            "lo": fm.lo,
            "hi": fm.hi,
            "in": fm.in_grain.label,
            "out": fm.out_grain.label,
        }
    if isinstance(fm, Passthrough):
        return {"k": "pass", "src": fm.source}
    return {"k": "unbounded"}


def _field_mapping_from_json(blob: dict[str, Any]) -> FieldMapping:
    kind = blob["k"]
    if kind == "window":
        return TimeWindow(
            blob["src"], blob["lo"], blob["hi"], Grain.parse(blob["in"]), Grain.parse(blob["out"])
        )
    if kind == "pass":
        return Passthrough(blob["src"])
    return UNBOUNDED


def mapping_to_json(mapping: PartitionMapping) -> str:
    """Encode one edge's partition mapping, field by field."""
    return json.dumps(
        [[name, _field_mapping_to_json(fm)] for name, fm in mapping.fields], separators=(",", ":")
    )


def mapping_from_json(raw: str) -> PartitionMapping:
    """Decode a mapping written by `mapping_to_json`."""
    return PartitionMapping(
        fields=tuple((n, _field_mapping_from_json(b)) for n, b in json.loads(raw))
    )
