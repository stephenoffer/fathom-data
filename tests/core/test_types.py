"""The vocabulary, and the ways people write it down.

Every constructor here exists because somebody had a string and needed a type. The
tests are mostly about the boundary between the two: what `parse` accepts, what it
refuses, and whether the refusal tells the reader what to write instead.

An error message is API. These pin the parts of it a user reads — the offending
value, the accepted alternatives, and the suggestion — without pinning the prose
around them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fathom.core.grains import Grain
from fathom.core.types import (
    ANY,
    UNPARTITIONED,
    Capabilities,
    ChangeSource,
    ColumnRef,
    DatasetId,
    ErasureMode,
    KeyPredicate,
    LineageSource,
    PartitionField,
    PartitionSpec,
    Pushdown,
    covered_by,
    subsumes,
)

# -- dataset identity ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("duckdb/raw.events", DatasetId("duckdb", "raw.events")),
        ("s3://lake/raw/events", DatasetId("s3://lake", "raw/events")),
        (
            "snowflake://xy12345/db.schema.orders",
            DatasetId("snowflake://xy12345", "db.schema.orders"),
        ),
        ("file:///tmp/lake/events", DatasetId("file", "/tmp/lake/events")),
        ("  duckdb/raw.events  ", DatasetId("duckdb", "raw.events")),
    ],
)
def test_parse_reads_back_what_str_prints(text: str, expected: DatasetId):
    assert DatasetId.parse(text) == expected


@pytest.mark.parametrize(
    "dataset",
    [
        DatasetId("duckdb", "raw.events"),
        DatasetId("s3://lake", "raw/events"),
        DatasetId("file", "/tmp/lake/events"),
    ],
)
def test_str_and_parse_round_trip(dataset: DatasetId):
    """Anything printed in a plan can be pasted into the next command."""
    assert DatasetId.parse(str(dataset)) == dataset


def test_a_bare_name_is_refused_with_the_two_ways_to_fix_it():
    with pytest.raises(ValueError) as exc:
        DatasetId.parse("raw.events")
    message = str(exc.value)
    assert "raw.events" in message
    assert "system/name" in message
    assert "normalize" in message  # the other route, for a bare table name


def test_an_empty_identity_is_refused():
    with pytest.raises(ValueError, match="cannot be empty"):
        DatasetId.parse("   ")


def test_repr_is_short_enough_to_print_a_collection_of():
    assert repr(DatasetId("duckdb", "raw.events")) == "DatasetId('duckdb', 'raw.events')"


def test_column_refs_round_trip():
    ref = ColumnRef(DatasetId("duckdb", "raw.events"), "amount")
    assert ColumnRef.parse(str(ref)) == ref


def test_a_column_ref_without_a_column_is_refused():
    with pytest.raises(ValueError, match="dataset#column"):
        ColumnRef.parse("duckdb/raw.events")


# -- partition fields ----------------------------------------------------------


def test_a_time_field_without_a_grain_says_what_to_write():
    with pytest.raises(ValueError) as exc:
        PartitionField(name="dt", kind="time")
    message = str(exc.value)
    assert "grain: day" in message  # the fathom.yml spelling
    assert "PartitionField.time" in message  # the Python spelling


def test_a_value_field_with_a_grain_offers_the_field_it_probably_meant():
    with pytest.raises(ValueError) as exc:
        PartitionField(name="dt", kind="value", grain=Grain.DAY)
    assert "PartitionField.time('dt', 'day')" in str(exc.value)


def test_an_unknown_kind_explains_the_two_that_exist():
    with pytest.raises(ValueError) as exc:
        PartitionField(name="dt", kind="temporal")  # type: ignore[arg-type]
    assert "'time'" in str(exc.value) and "'value'" in str(exc.value)


def test_field_constructors_take_grain_names_as_well_as_grains():
    assert PartitionField.time("dt", "daily") == PartitionField.time("dt", Grain.DAY)


def test_fields_parse_from_their_compact_form():
    assert PartitionField.parse("dt:day") == PartitionField.time("dt", Grain.DAY)
    assert PartitionField.parse("region") == PartitionField.value("region")


def test_an_empty_field_name_is_refused_with_both_forms():
    with pytest.raises(ValueError, match="name:grain"):
        PartitionField.parse(":day")


# -- partition specs -----------------------------------------------------------


def test_specs_parse_from_one_line():
    spec = PartitionSpec.parse("dt:day, region")
    assert spec.names == ("dt", "region")
    assert spec.field("dt").kind == "time"
    assert spec.field("region").kind == "value"


def test_specs_round_trip_through_their_compact_form():
    spec = PartitionSpec.parse("dt:month, region, tenant")
    assert PartitionSpec.parse(str(spec)) == spec


def test_an_empty_spec_parses_to_unpartitioned():
    assert PartitionSpec.parse("") == UNPARTITIONED
    assert PartitionSpec.parse("  ,  ") == UNPARTITIONED


def test_an_unpartitioned_spec_prints_as_such_rather_than_as_nothing():
    assert str(UNPARTITIONED) == "<unpartitioned>"
    assert repr(UNPARTITIONED) == "UNPARTITIONED"


def test_membership_reads_the_way_it_looks():
    spec = PartitionSpec.parse("dt:day, region")
    assert "region" in spec
    assert "tenant" not in spec


def test_time_fields_are_separable_from_value_fields():
    spec = PartitionSpec.parse("dt:day, region")
    assert [f.name for f in spec.time_fields] == ["dt"]


def test_a_duplicate_field_names_the_field():
    with pytest.raises(ValueError, match="duplicate partition field 'dt'"):
        PartitionSpec.parse("dt:day, dt:month")


def test_requiring_a_missing_field_lists_the_ones_that_exist():
    spec = PartitionSpec.parse("dt:day, region")
    with pytest.raises(KeyError) as exc:
        spec.require("date")
    message = str(exc.value)
    assert "'dt'" in message and "'region'" in message
    assert "Did you mean 'dt'?" in message


def test_requiring_a_field_of_an_unpartitioned_dataset_explains_the_real_problem():
    """The field is not missing — the spec is, and that is the actionable fact."""
    with pytest.raises(KeyError, match="unpartitioned"):
        UNPARTITIONED.require("dt")


# -- key predicates ------------------------------------------------------------


def test_predicates_parse_from_the_cli_syntax():
    spec = PartitionSpec.parse("dt:day, region")
    key = KeyPredicate.parse("dt=2026-03-14,region=eu", spec)
    assert key.get("dt") == datetime(2026, 3, 14)
    assert key.get("region") == "eu"


def test_a_time_value_is_truncated_to_its_declared_grain():
    """A key seeded at 17:42 must equal the same day's key read from a path."""
    spec = PartitionSpec.parse("dt:month")
    assert KeyPredicate.parse("dt=2026-03-14T17:42:00", spec).get("dt") == datetime(2026, 3, 1)


def test_without_a_spec_values_stay_strings():
    assert KeyPredicate.parse("dt=2026-03-14").get("dt") == "2026-03-14"


def test_a_star_binds_the_field_to_any():
    assert KeyPredicate.parse("region=*").get("region") is ANY


def test_a_binding_without_an_equals_sign_shows_the_form_it_wanted():
    with pytest.raises(ValueError, match="field=value"):
        KeyPredicate.parse("region")


def test_a_bad_datetime_names_the_field_and_its_grain():
    spec = PartitionSpec.parse("dt:day")
    with pytest.raises(ValueError) as exc:
        KeyPredicate.parse("dt=last-tuesday", spec)
    message = str(exc.value)
    assert "'dt'" in message and "day grain" in message and "2026-03-14" in message


def test_predicates_round_trip_through_their_printed_form():
    spec = PartitionSpec.parse("dt:day, region")
    key = KeyPredicate.parse("dt=2026-03-14,region=eu", spec)
    assert KeyPredicate.parse(str(key).replace("/", ","), spec) == key


def test_an_empty_string_is_the_whole_dataset():
    assert KeyPredicate.parse("") == KeyPredicate()
    assert str(KeyPredicate.parse("")) == "<whole dataset>"


def test_binding_order_does_not_change_identity():
    assert KeyPredicate.of(region="eu", dt="x") == KeyPredicate.of(dt="x", region="eu")


# -- the ordering the planner is defined against -------------------------------


def test_the_whole_dataset_covers_a_single_partition():
    whole = KeyPredicate.of(dt=ANY, region=ANY)
    one = KeyPredicate.of(dt="2026-03-14", region="eu")
    assert subsumes(whole, one)
    assert not subsumes(one, whole)


def test_coverage_is_how_the_worklist_knows_it_has_stopped_growing():
    already = [KeyPredicate.of(dt=ANY, region="eu")]
    assert covered_by(already, KeyPredicate.of(dt="2026-03-14", region="eu"))
    assert not covered_by(already, KeyPredicate.of(dt="2026-03-14", region="us"))


# -- capabilities --------------------------------------------------------------


def test_every_capability_member_explains_itself():
    """A capability matrix of bare constants makes the reader guess the consequence."""
    for enum in (LineageSource, ChangeSource, Pushdown, ErasureMode):
        for member in enum:
            assert member.description, f"{enum.__name__}.{member.name} has no description"
            assert member.description.endswith("."), f"{member.name} is not a sentence"


def test_describe_maps_the_wire_value_to_the_explanation():
    assert ChangeSource.describe()["list_diff"].startswith("Every object is listed")


def test_capability_members_still_format_as_their_value():
    """They are StrEnums, and messages interpolate them."""
    assert f"{Pushdown.SKETCHES}" == "sketches"


def test_a_summary_names_what_the_adapter_cannot_do():
    caps = Capabilities(LineageSource.QUERY_LOG, ChangeSource.LIST_DIFF)
    summary = caps.summary()
    assert "no pushdown" in summary
    assert "erasure unsupported" in summary
    assert "dataset-level lineage" in summary


def test_the_verbose_explanation_covers_every_capability():
    caps = Capabilities(
        LineageSource.NATIVE,
        ChangeSource.SNAPSHOT_DIFF,
        pushdown=Pushdown.SKETCHES,
        erasure=ErasureMode.DELETE_VECTOR,
        column_lineage=True,
        partition_aware=True,
        freshness_lag=timedelta(hours=3),
    )
    lines = caps.explain()
    assert len(lines) == 7  # four enums, columns, partitions, and the lag
    assert any("Metadata can be this stale" in line for line in lines)


def test_a_lag_is_omitted_when_the_adapter_declares_none():
    assert len(Capabilities(LineageSource.DECLARED, ChangeSource.WATERMARK).explain()) == 6
