# Examples

Each example is a self-contained script that builds its own data, runs, and prints
what happened. They are executed by `tests/test_examples.py` in CI, so they cannot
drift from the code.

```bash
python examples/00_five_minutes.py
python examples/01_local_lakehouse.py
python examples/02_shadow_mode.py
python examples/03_dbt_project.py
python examples/04_erasure.py
python examples/05_cloud_storage.py
python examples/06_worth_keeping.py
```

| Example | Shows |
|---|---|
| `00_five_minutes` | The whole idea with nothing installed: a graph, a plan, and what widening costs |
| `01_local_lakehouse` | The whole loop: detect, plan, apply, and verify against a full rebuild |
| `02_shadow_mode` | Grading the planner, including catching a deliberately wrong plan |
| `03_dbt_project` | Building a graph from a dbt manifest, with mappings recovered from compiled SQL |
| `04_erasure` | Locating a subject across derived tables, and why ordering matters |
| `05_cloud_storage` | Delta on object storage, using `memory://` in place of S3 |
| `06_worth_keeping` | The three questions outside the warehouse: what never arrived, who reads it, and what was published |

## Reading them in order

`00` is the one to read first. It needs nothing but the base install — no warehouse,
no object store, no config file — and it shows the three things worth understanding
before anything else: what a partition mapping is, what a plan looks like, and what
happens when nothing could be proven.

`01` is next. It ends by asserting that an incremental rebuild is byte-identical to a
full one, which is the claim everything else rests on.

`02` is the one to read before using any of this in production.

`06` is the one to read if you are trying to work out which tables to switch off. It
ends by *declining* to recommend deleting a table that is expensive and unread, and
the comments say why — which is the more useful half of that feature.
