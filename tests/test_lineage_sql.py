"""SQL lineage extraction across dialects.

The value of routing every engine through one parser is that these cases only have
to be right once. The value of refusing to guess is that the cases we get wrong
degrade to `UNBOUNDED` instead of to a wrong answer.
"""

from __future__ import annotations

import pytest

from fathom.grains import Grain
from fathom.ids import normalize_table
from fathom.lineage import extract
from fathom.partitions import UNBOUNDED, Passthrough, TimeWindow
from fathom.types import PartitionField, PartitionSpec

DAY = PartitionSpec.of(PartitionField.time("dt", Grain.DAY))
MONTH = PartitionSpec.of(PartitionField.time("dt", Grain.MONTH))
DAY_REGION = PartitionSpec.of(PartitionField.time("dt", Grain.DAY), PartitionField.value("region"))


def ids(system="duckdb"):
    return (
        normalize_table("raw.events", system=system),
        normalize_table("mart.out", system=system),
    )


def test_simple_ctas_yields_an_identity_mapping():
    raw, out = ids()
    got = extract(
        "CREATE TABLE mart.out AS SELECT dt, amount FROM raw.events",
        dialect="duckdb",
        specs={raw: DAY, out: DAY},
    )
    assert len(got) == 1
    mapping = got[0].mappings[raw].get("dt")
    assert isinstance(mapping, TimeWindow)
    assert (mapping.lo, mapping.hi, mapping.out_grain) == (0, 0, Grain.DAY)


def test_date_trunc_is_recognized_as_a_rollup():
    raw, out = ids()
    got = extract(
        "INSERT INTO mart.out SELECT DATE_TRUNC('month', dt) AS dt, SUM(amount) AS amount "
        "FROM raw.events GROUP BY 1",
        dialect="duckdb",
        specs={raw: DAY, out: MONTH},
    )
    mapping = got[0].mappings[raw].get("dt")
    assert isinstance(mapping, TimeWindow)
    assert (mapping.in_grain, mapping.out_grain) == (Grain.DAY, Grain.MONTH)


def test_value_field_carried_through_is_a_passthrough():
    raw, out = ids()
    got = extract(
        "CREATE TABLE mart.out AS SELECT dt, region, amount FROM raw.events",
        dialect="duckdb",
        specs={raw: DAY_REGION, out: DAY_REGION},
    )
    assert got[0].mappings[raw].get("region") == Passthrough("region")


def test_transformed_value_field_widens():
    """`upper(region)` is not the same value, so it is not a passthrough."""
    raw, out = ids()
    got = extract(
        "CREATE TABLE mart.out AS SELECT dt, UPPER(region) AS region FROM raw.events",
        dialect="duckdb",
        specs={raw: DAY_REGION, out: DAY_REGION},
    )
    assert got[0].mappings[raw].get("region") is UNBOUNDED


def test_column_edges_are_recovered():
    raw, _ = ids()
    got = extract(
        "CREATE TABLE mart.out AS SELECT dt, amount AS revenue FROM raw.events",
        dialect="duckdb",
    )
    assert ("amount", "revenue") in got[0].column_edges[raw]


def test_ambiguous_unqualified_columns_are_not_attributed():
    """With two candidate tables and no qualifier, silence beats a coin flip."""
    got = extract(
        "CREATE TABLE mart.out AS SELECT dt, amount FROM raw.events, raw.other",
        dialect="duckdb",
    )
    assert all(edges == () for edges in got[0].column_edges.values())
    assert any("not attributed" in n for n in got[0].notes)


def test_qualified_columns_survive_multiple_sources():
    got = extract(
        "CREATE TABLE mart.out AS SELECT e.dt AS dt, o.amount AS amount "
        "FROM raw.events e JOIN raw.other o ON e.id = o.id",
        dialect="duckdb",
    )
    events = normalize_table("raw.events", system="duckdb")
    other = normalize_table("raw.other", system="duckdb")
    assert ("dt", "dt") in got[0].column_edges[events]
    assert ("amount", "amount") in got[0].column_edges[other]


def test_merge_widens_and_says_why():
    raw, out = ids()
    got = extract(
        "MERGE INTO mart.out t USING raw.events s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET t.amount = s.amount",
        dialect="duckdb",
        specs={raw: DAY, out: DAY},
    )
    assert got[0].mappings[raw].get("dt") is UNBOUNDED
    assert any("MERGE" in n for n in got[0].notes)


def test_unparseable_sql_is_reported_not_raised():
    """One malformed entry in a query log must not abort the whole ingest."""
    got = extract("this is not sql at all !!!", dialect="duckdb")
    assert got and got[0].is_empty
    assert any("unparseable" in n for n in got[0].notes)


def test_select_without_a_target_is_ignored():
    assert extract("SELECT 1", dialect="duckdb") == []


@pytest.mark.parametrize(
    ("dialect", "trunc"),
    [
        ("duckdb", "DATE_TRUNC('month', dt)"),
        ("snowflake", "DATE_TRUNC('month', dt)"),
        ("spark", "DATE_TRUNC('MONTH', dt)"),
        ("trino", "DATE_TRUNC('month', dt)"),
        ("postgres", "DATE_TRUNC('month', dt)"),
        ("clickhouse", "toStartOfMonth(dt)"),
        # BigQuery takes its arguments the other way round.
        ("bigquery", "DATE_TRUNC(dt, MONTH)"),
    ],
)
def test_rollup_is_recognized_across_dialects(dialect, trunc):
    raw = normalize_table("raw.events", system=dialect)
    out = normalize_table("mart.out", system=dialect)
    got = extract(
        f"CREATE TABLE mart.out AS SELECT {trunc} AS dt FROM raw.events",
        dialect=dialect,
        system=dialect,
        specs={raw: DAY, out: MONTH},
    )
    mapping = got[0].mappings[raw].get("dt")
    assert isinstance(mapping, TimeWindow), f"{dialect} produced {mapping}"
    assert mapping.out_grain is Grain.MONTH


def test_wrong_argument_order_widens_instead_of_inventing_a_grain():
    """BigQuery syntax written the DuckDB way parses to nonsense. Refuse it."""
    raw = normalize_table("raw.events", system="bigquery")
    out = normalize_table("mart.out", system="bigquery")
    got = extract(
        "CREATE TABLE mart.out AS SELECT DATE_TRUNC('month', dt) AS dt FROM raw.events",
        dialect="bigquery",
        system="bigquery",
        specs={raw: DAY, out: MONTH},
    )
    assert got[0].mappings[raw].get("dt") is UNBOUNDED


def test_unknown_specs_produce_unbounded_but_still_link():
    """Without partition specs we still know the edge exists, just not its shape."""
    raw, _ = ids()
    got = extract("CREATE TABLE mart.out AS SELECT * FROM raw.events", dialect="duckdb")
    assert raw in got[0].sources
    assert got[0].mappings[raw].is_unbounded
