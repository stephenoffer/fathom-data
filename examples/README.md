# Examples

Each example is a self-contained script that builds its own data, runs, and prints
what happened. They are executed by `tests/test_examples.py` in CI, so they cannot
drift from the code.

```bash
python examples/01_local_lakehouse.py
python examples/02_shadow_mode.py
python examples/03_dbt_project.py
python examples/04_erasure.py
python examples/05_cloud_storage.py
```

| Example | Shows |
|---|---|
| `01_local_lakehouse` | The whole loop: detect, plan, apply, and verify against a full rebuild |
| `02_shadow_mode` | Grading the planner, including catching a deliberately wrong plan |
| `03_dbt_project` | Building a graph from a dbt manifest, with mappings recovered from compiled SQL |
| `04_erasure` | Locating a subject across derived tables, and why ordering matters |
| `05_cloud_storage` | Delta on object storage, using `memory://` in place of S3 |

## Reading them in order

`01` is the one to read first — it ends by asserting that an incremental rebuild is
byte-identical to a full one, which is the claim everything else rests on.

`02` is the one to read before using any of this in production.
