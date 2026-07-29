"""JSON round-tripping for the IR.

The failure this module exists to prevent is silent. Round-tripping a partition key
through untagged JSON turns `datetime(2026, 3, 14)` into `"2026-03-14T00:00:00"`,
which compares unequal to a freshly computed key — so the planner rebuilds a
partition it already has, forever, and nothing raises.

Every test here is therefore an equality assertion after a round trip, not a check
that the JSON looks right.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fathom.core.codec import (
    dataset_from_json,
    dataset_to_json,
    key_from_json,
    key_to_json,
    mapping_from_json,
    mapping_to_json,
    spec_from_json,
    spec_to_json,
)
from fathom.core.grains import Grain
from fathom.core.partitions import UNBOUNDED, PartitionMapping, Passthrough, TimeWindow
from fathom.core.types import ANY, DatasetId, KeyPredicate, PartitionField, PartitionSpec


@pytest.mark.parametrize(
    "dataset",
    [
        DatasetId("duckdb", "raw.events"),
        DatasetId("s3://lake", "raw/events"),
        DatasetId("file", "/tmp/x"),
        DatasetId("abfss://c@acct", "path/to/thing"),
    ],
)
def test_dataset_round_trips(dataset):
    assert dataset_from_json(dataset_to_json(dataset)) == dataset


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 3, 14),
        datetime(2026, 3, 14, 7, 30, tzinfo=UTC),
        "eu",
        42,
        -1,
        3.5,
        True,
        False,
        None,
        ANY,
    ],
)
def test_partition_values_keep_their_type(value):
    """A string that used to be a datetime compares unequal to the real thing."""
    key = KeyPredicate(bindings=(("dt", value),))
    restored = key_from_json(key_to_json(key))

    assert restored == key
    assert restored.get("dt") is value if value is ANY else restored.get("dt") == value
    assert type(restored.get("dt")) is type(value)


def test_any_survives_as_the_sentinel_not_a_string():
    """`ANY` is a singleton; a decoded copy would silently stop subsuming anything."""
    key = KeyPredicate(bindings=(("region", ANY),))
    assert key_from_json(key_to_json(key)).get("region") is ANY


def test_null_is_distinct_from_any():
    """A null partition is a real partition, and not the same as an unconstrained one."""
    null = KeyPredicate(bindings=(("region", None),))
    unconstrained = KeyPredicate(bindings=(("region", ANY),))
    assert key_from_json(key_to_json(null)) != key_from_json(key_to_json(unconstrained))


def test_bool_does_not_decode_as_int():
    """`isinstance(True, int)` is true in Python, so the encoder has to check bool first."""
    key = KeyPredicate(bindings=(("flag", True),))
    assert type(key_from_json(key_to_json(key)).get("flag")) is bool


def test_empty_key_round_trips():
    assert key_from_json(key_to_json(KeyPredicate())) == KeyPredicate()


@pytest.mark.parametrize(
    "spec",
    [
        PartitionSpec(),
        PartitionSpec.of(PartitionField.time("dt", Grain.DAY)),
        PartitionSpec.of(PartitionField.time("dt", Grain.MONTH), PartitionField.value("region")),
        PartitionSpec.of(PartitionField.time("hour", Grain.HOUR)),
    ],
)
def test_spec_round_trips_with_its_grains(spec):
    restored = spec_from_json(spec_to_json(spec))
    assert restored == spec
    assert [f.grain for f in restored.fields] == [f.grain for f in spec.fields]


@pytest.mark.parametrize(
    "mapping",
    [
        PartitionMapping(),
        PartitionMapping.of(dt=TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)),
        PartitionMapping.of(dt=TimeWindow("dt", -3, 6, Grain.DAY, Grain.MONTH)),
        PartitionMapping.of(region=Passthrough("region")),
        PartitionMapping.of(dt=UNBOUNDED),
        PartitionMapping.of(
            dt=TimeWindow("src_dt", 0, 6, Grain.HOUR, Grain.DAY), region=Passthrough("r")
        ),
    ],
)
def test_mapping_round_trips(mapping):
    assert mapping_from_json(mapping_to_json(mapping)) == mapping


def test_mapping_preserves_window_bounds_and_source():
    """A dropped offset would narrow the mapping, which is the unsafe direction."""
    window = TimeWindow("source_dt", -2, 5, Grain.HOUR, Grain.MONTH)
    restored = mapping_from_json(mapping_to_json(PartitionMapping.of(dt=window))).get("dt")

    assert isinstance(restored, TimeWindow)
    assert (restored.source, restored.lo, restored.hi) == ("source_dt", -2, 5)
    assert (restored.in_grain, restored.out_grain) == (Grain.HOUR, Grain.MONTH)


def test_encoding_is_stable_across_calls():
    """Stored JSON is compared as text in the store's uniqueness keys."""
    key = KeyPredicate.of(region="eu", dt=datetime(2026, 3, 14))
    assert key_to_json(key) == key_to_json(KeyPredicate.of(dt=datetime(2026, 3, 14), region="eu"))


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["dt", "region", "bucket"]),
            st.one_of(
                st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1)),
                st.text(max_size=12),
                st.integers(min_value=-1000, max_value=1000),
                st.just(ANY),
                st.none(),
            ),
        ),
        max_size=3,
        unique_by=lambda pair: pair[0],
    )
)
def test_any_key_round_trips(bindings):
    key = KeyPredicate(bindings=tuple(bindings))
    assert key_from_json(key_to_json(key)) == key


# -- profile statistics --------------------------------------------------------


@pytest.mark.parametrize("value", [1, -4, 0.5, "eu", True, datetime(2026, 3, 14), None])
def test_statistics_keep_their_type(value):
    """A float min stored as text makes every range comparison a swallowed TypeError."""
    from fathom.core.codec import stat_from_json, stat_to_json

    restored = stat_from_json(stat_to_json(value))
    assert restored == value
    assert type(restored) is type(value)


def test_statistics_decode_pre_typed_stores():
    """An existing store must keep opening, not fail on rows written before tagging."""
    from fathom.core.codec import stat_from_json

    assert stat_from_json("plain text from an older schema") == "plain text from an older schema"
    assert stat_from_json(None) is None


def test_numeric_statistics_stay_comparable_after_a_round_trip():
    from fathom.core.codec import stat_from_json, stat_to_json

    low = stat_from_json(stat_to_json(1.5))
    high = stat_from_json(stat_to_json(9.5))
    assert low < high  # the comparison `drift` makes, and the one text breaks
