"""JSON encoding for the IR.

Partition values are not all strings. A `dt` binding is a datetime, a bucket number
is an int, and `ANY` is neither. Round-tripping through untagged JSON would turn
`datetime(2026, 3, 14)` into `"2026-03-14T00:00:00"` and quietly break every
comparison against a freshly computed key, so values carry their type.
"""

from __future__ import annotations

import json
from datetime import datetime
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
]


def _value_to_json(value: Any) -> dict[str, Any]:
    if value is ANY:
        return {"t": "any"}
    if isinstance(value, datetime):
        return {"t": "datetime", "v": value.isoformat()}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "int", "v": value}
    if isinstance(value, float):
        return {"t": "float", "v": value}
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
    if kind == "int":
        return int(blob["v"])
    if kind == "float":
        return float(blob["v"])
    if kind == "bool":
        return bool(blob["v"])
    return str(blob["v"])


def dataset_to_json(ds: DatasetId) -> str:
    return json.dumps({"ns": ds.namespace, "name": ds.name}, separators=(",", ":"))


def dataset_from_json(raw: str) -> DatasetId:
    blob = json.loads(raw)
    return DatasetId(namespace=blob["ns"], name=blob["name"])


def key_to_json(key: KeyPredicate) -> str:
    return json.dumps(
        [[name, _value_to_json(value)] for name, value in key.bindings], separators=(",", ":")
    )


def key_from_json(raw: str) -> KeyPredicate:
    return KeyPredicate(bindings=tuple((n, _value_from_json(v)) for n, v in json.loads(raw)))


def spec_to_json(spec: PartitionSpec) -> str:
    return json.dumps(
        [
            {"name": f.name, "kind": f.kind, "grain": f.grain.label if f.grain else None}
            for f in spec.fields
        ],
        separators=(",", ":"),
    )


def spec_from_json(raw: str) -> PartitionSpec:
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
    return json.dumps(
        [[name, _field_mapping_to_json(fm)] for name, fm in mapping.fields], separators=(",", ":")
    )


def mapping_from_json(raw: str) -> PartitionMapping:
    return PartitionMapping(
        fields=tuple((n, _field_mapping_from_json(b)) for n, b in json.loads(raw))
    )
