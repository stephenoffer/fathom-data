"""Building a graph from a dbt manifest.

dbt supplies the dependency graph and the relation names. Parsing `compiled_code`
supplies the column edges and partition mappings dbt does not record.

Writes a small manifest in the shape dbt actually produces, then plans against it.

    python examples/03_dbt_project.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from fathom import KeyPredicate
from fathom.ids import normalize_table
from fathom.integrations import ingest_dbt, parse_manifest

MANIFEST = {
    "metadata": {
        "adapter_type": "bigquery",
        "project_name": "analytics",
        "dbt_version": "1.9.0",
    },
    "sources": {
        "source.analytics.raw.events": {
            "resource_type": "source",
            "database": "prod",
            "schema": "raw",
            "name": "events",
            # dbt cannot express grain for every adapter, so this is the escape hatch.
            "config": {"meta": {"fathom": {"partition": [{"field": "dt", "grain": "day"}]}}},
        }
    },
    "nodes": {
        "model.analytics.gold_monthly": {
            "resource_type": "model",
            "database": "prod",
            "schema": "gold",
            "name": "gold_monthly",
            "alias": "monthly",
            "depends_on": {"nodes": ["source.analytics.raw.events"]},
            "config": {
                "materialized": "incremental",
                "partition_by": {"field": "dt", "data_type": "date", "granularity": "month"},
            },
            "columns": {"dt": {"name": "dt", "data_type": "date"}},
            # BigQuery takes DATE_TRUNC's arguments the other way round.
            "compiled_code": (
                "select date_trunc(dt, MONTH) as dt, sum(amount) as revenue "
                "from prod.raw.events group by 1"
            ),
        },
        "test.analytics.not_null_revenue": {
            "resource_type": "test",
            "database": "prod",
            "schema": "gold",
            "name": "not_null_revenue",
            "depends_on": {"nodes": ["model.analytics.gold_monthly"]},
            "config": {},
        },
    },
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "target"
        target.mkdir()
        (target / "manifest.json").write_text(json.dumps(MANIFEST))

        parsed = parse_manifest(MANIFEST)
        print(f"Adapter: {parsed.system}  (selects identifier case folding)")
        print("Datasets dbt knows about:")
        for dataset in parsed.datasets:
            spec = parsed.specs.get(dataset)
            shown = (
                ", ".join(f"{f.name}:{f.grain.label if f.grain else 'value'}" for f in spec.fields)
                if spec
                else "unpartitioned"
            )
            print(f"  {dataset}  [{shown}]")
        print("  (the `not_null_revenue` test is not a data dependency, so it is skipped)")

        result = ingest_dbt(str(target))
        print(f"\n{result.summary()}")
        for note in result.notes:
            print(f"  ! {note}")

        print("\nEdges, with detail recovered from the compiled SQL:")
        for edge in result.graph.edges:
            print(f"  {edge}")
            for src_col, dst_col in edge.columns:
                print(f"      {src_col} -> {dst_col}")

        source = normalize_table("prod.raw.events", system="bigquery")
        model = normalize_table("prod.gold.monthly", system="bigquery")

        plan = result.graph.invalidate({source: [KeyPredicate.of(dt=datetime(2026, 3, 14))]})
        print("\nOne dirty day in the source:")
        print("  " + plan.summary().replace("\n", "\n  "))

        assert plan.partitions(model) == frozenset({KeyPredicate.of(dt=datetime(2026, 3, 1))})
        print("\nThe day mapped to exactly its month, from a mapping dbt never recorded.")


if __name__ == "__main__":
    main()
