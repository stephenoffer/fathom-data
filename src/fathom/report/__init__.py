"""Out to people, and out to other tools.

    render         Mermaid, Graphviz, D2, PlantUML, Cytoscape, JSON, Markdown
    emit           OpenLineage, DataHub, Atlas, OpenMetadata payloads
    orchestrators  a plan as an Airflow, Dagster, or Prefect DAG file
    notify         a finding routed to a person, deduplicated and quieted
    telemetry      fathom's own metrics and traces, in Prometheus and OTLP form
    compliance     records of processing, subject access, model training summaries

Every function here is pure: artifacts in, a string or a dict out. No clients, no
credentials, no network. Posting is the caller's problem, which is what makes this
testable, safe to run inside an agent, and impossible to blame for an outage.

`emit` is the adoption strategy in one module. A team already running a catalog will
not replace it; they will accept richer events into what they have. `orchestrators`
is the same argument for the scheduler: nobody migrates off Airflow to try a planner,
and the last mile between "here is the plan" and "here is the DAG" is exactly where a
good plan stops being used.

Neither imports the thing it targets. A library that pulls in Airflow to be imported
is a library people vendor around.
"""

from . import compliance, emit, notify, orchestrators, render, telemetry

__all__ = [
    "compliance",
    "emit",
    "notify",
    "orchestrators",
    "render",
    "telemetry",
]
