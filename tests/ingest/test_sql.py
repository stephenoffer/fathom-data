"""SQL lineage extraction across dialects.

The value of routing every engine through one parser is that these cases only have
to be right once. The value of refusing to guess is that the cases we get wrong
degrade to `UNBOUNDED` instead of to a wrong answer.
"""

from __future__ import annotations

import pytest

from fathom.core.grains import Grain
from fathom.core.ids import normalize_table
from fathom.core.partitions import UNBOUNDED, Passthrough, TimeWindow
from fathom.core.types import PartitionField, PartitionSpec
from fathom.ingest import extract

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


def test_cte_names_do_not_become_datasets():
    """A CTE parses as a table but is not one.

    Left in, every `WITH x AS (...)` mints a phantom dataset that collects the edges
    the real source table should have owned, and the extra "source" makes the
    statement look ambiguous so column lineage is dropped too.
    """
    sql = """
    WITH recent AS (SELECT dt, region, amount FROM raw.events WHERE dt > '2026-01-01')
    INSERT INTO mart.out SELECT dt, region, SUM(amount) AS amount FROM recent GROUP BY dt, region
    """
    (extraction,) = extract(sql, dialect="duckdb", system="duckdb")

    assert extraction.sources == (normalize_table("raw.events", system="duckdb"),)
    assert all("recent" not in ds.name for ds in extraction.sources)


def test_a_qualified_cte_name_is_still_a_real_table():
    """`raw.recent` is a table even when a CTE happens to share its bare name."""
    sql = """
    WITH recent AS (SELECT dt FROM raw.events)
    INSERT INTO mart.out SELECT r.dt AS dt FROM raw.recent r
    """
    (extraction,) = extract(sql, dialect="duckdb", system="duckdb")
    assert normalize_table("raw.recent", system="duckdb") in extraction.sources


def test_a_joins_time_column_does_not_bind_the_other_sides_partitions():
    """`SELECT a.dt` says how a maps to the target and nothing about b.

    Both tables partition by `dt`, but the target's `dt` is taken from `events`. If
    the extractor hands `dims` the same identity window, a change to one day of
    `dims` invalidates one day of the target — when in truth it can affect any of
    them. That is under-invalidation, so `dims` must widen to UNBOUNDED.
    """
    events = normalize_table("raw.events", system="duckdb")
    dims = normalize_table("raw.dims", system="duckdb")
    target = normalize_table("mart.out", system="duckdb")
    sql = """
    INSERT INTO mart.out
    SELECT e.dt AS dt, SUM(e.amount) AS amount
    FROM raw.events e JOIN raw.dims d ON e.region = d.region
    GROUP BY e.dt
    """
    (extraction,) = extract(
        sql,
        dialect="duckdb",
        system="duckdb",
        specs={events: DAY, dims: DAY, target: DAY},
    )

    assert extraction.mappings[events].get("dt") == TimeWindow("dt", 0, 0, Grain.DAY, Grain.DAY)
    assert extraction.mappings[dims].get("dt") is UNBOUNDED


def test_a_volatile_query_id_does_not_mint_a_new_edge_each_run():
    """Evidence is an edge's identity in the store.

    A warehouse query log identifies each *execution*. Folding that id into the
    evidence makes one dependency a brand-new edge on every run, so an hourly model
    accumulates thousands of rows a year, `fan_in` counts executions instead of
    inputs, and `edge_between` joins them all — which widens every plan that
    crosses the edge.
    """
    from fathom.adapters.base import QueryEvent
    from fathom.graph.query import fan_in
    from fathom.ingest.events import graph_from_queries
    from fathom.store import Store

    target = normalize_table("mart.out", system="duckdb")
    store = Store(":memory:")
    for run in range(24):
        event = QueryEvent(
            sql="INSERT INTO mart.out SELECT dt FROM raw.events",
            dialect="duckdb",
            query_id=f"job-{run:04d}",
        )
        built = graph_from_queries(
            [event], dialect="duckdb", system="duckdb", evidence_label="bigquery:query_log"
        )
        store.save_graph(built.graph)

    loaded = store.load_graph()
    assert len(loaded.edges) == 1
    assert fan_in(loaded, target) == 1
