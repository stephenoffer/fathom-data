# S3, GCS, Azure, and other object storage

Every adapter that touches bytes goes through one filesystem layer, so `s3://`,
`gs://`, `abfss://`, and a local path are the same code path. Delta and Iceberg
tables on object storage read exactly like local ones.

## Install

```bash
pip install 'fathom-data[cloud]'    # s3fs + gcsfs + adlfs
pip install 'fathom-data[s3]'       # or just one
```

## Credentials

Never in the config file. Use environment references:

```yaml
storage_options:
  s3:
    key: "${AWS_ACCESS_KEY_ID}"
    secret: "${AWS_SECRET_ACCESS_KEY}"
  gs:
    token: "${GOOGLE_APPLICATION_CREDENTIALS}"
  abfss:
    account_name: "${AZURE_STORAGE_ACCOUNT}"
    account_key: "${AZURE_STORAGE_KEY}"
```

Better still, omit `storage_options` entirely and let the SDK's default credential
chain work — instance profiles, workload identity, managed identity. A missing
variable with no default raises an error naming the variable, so this fails loudly
rather than silently connecting anonymously.

### S3-compatible endpoints

```yaml
storage_options:
  s3:
    endpoint_url: "${S3_ENDPOINT}"       # MinIO, R2, Ceph
    key: "${S3_KEY}"
    secret: "${S3_SECRET}"
```

## Layouts

### Hive-partitioned (self-describing)

```
s3://lake/events/dt=2026-03-14/region=eu/part-0.parquet
```

```yaml
  - name: s3://lake/events
    adapter: storage
    partition: [{field: dt, grain: day}, {field: region}]
```

Nothing else needed; the path parses itself.

### Anything else needs a template

```
s3://lake/events/2026/03/14/part-0.parquet
```

```yaml
    template: "events/{yyyy}/{MM}/{dd}"
```

Placeholders `{yyyy}`, `{MM}`, `{dd}`, `{HH}` assemble a timestamp; `{name}`
captures a value field.

Without a template a non-Hive path binds nothing and the dataset widens. Guessing
which segment is a month is exactly the inference that silently corrupts a plan.

## Change detection

**Use a table format if you can.** Delta and Iceberg give `SNAPSHOT_DIFF`: cost
proportional to commits, exact partition tuples.

```yaml
  - name: s3://lake/events
    adapter: delta        # or iceberg, or omit and let it sniff
```

The generic storage adapter uses `LIST_DIFF`, which is correct and scales badly. It
is fine under a few hundred thousand objects and ruinous above — a naive LIST over
a 100M-object bucket costs hours and real money.

### Resume tokens

The token is a high-water modification time **plus the etags of objects on that
boundary**. Timestamps are coarse and object stores happily write several objects in
the same second, so a strict `>` would miss one while a `>=` would re-report the
newest forever. Carrying the boundary etags gives both properties: nothing missed,
and a quiet dataset converges to reporting nothing.

Objects with no modification time at all are reported every run by design. Skipping
them silently would be worse.

## Profiling

Parquet footers are read over the network at metadata cost — no data pages, no
egress for the rows themselves:

```bash
fathom profile s3://lake/events
```

Scope it to what changed, which is what makes it affordable:

```python
project.profile(dataset, partition=KeyPredicate.of(dt=datetime(2026, 3, 14)))
```

## Performance

**Filesystems are cached per protocol and options.** A new client per dataset means
a TLS handshake and credential resolution for every partition you touch. Rotating
credentials mid-process needs `fathom.adapters.fs.clear_cache()`.

**Prefer `SNAPSHOT_DIFF`.** The difference between reading three commit files and
listing a bucket is several orders of magnitude.

**Scope listings.** One dataset per meaningful prefix. Pointing at a bucket root and
relying on templates to sort it out means listing everything on every run.

## Errors are never silent

```
cannot read s3://lake/events: NoCredentialsError: Unable to locate credentials
  no credentials found. Set AWS_PROFILE / AWS_ACCESS_KEY_ID, or pass
  storage_options={'key': ..., 'secret': ...}
```

A missing prefix returns empty; an access failure raises. Reporting the second as
the first sends people to debug their SQL, which is why the distinction is enforced
at the filesystem boundary rather than left to each adapter.

## Provider quirks that are handled for you

Three things every cloud spells differently, and getting any of them wrong reports
"nothing changed" rather than raising:

- **etags** — `ETag` on S3 (quoted), `md5Hash` on GCS, `content_md5` on Azure
- **modification times** — floats, ISO strings, or datetimes, sometimes absent
- **directories** — a fiction on object storage; a prefix is a directory when
  anything lives under it, and a file is never one

## Erasure

Object storage makes erasure harder than it looks, and the adapter reports which
case applies rather than assuming:

- **Delta or Iceberg** → `DELETE_VECTOR`. Positional deletes, then compaction of
  only the affected partitions. This is where partition scoping pays for itself.
- **Versioned buckets** → deleting an object leaves prior versions, and lifecycle
  rules may be retaining them. Crypto-shredding is the only reliable answer.
- **Object Lock or WORM** → impossible by design. The plan refuses and reports
  incomplete rather than issuing deletes that do nothing.
- **Cross-region replication** → delete markers do not always propagate the way
  people assume. Out of scope for the proof, and the [erase guide](../guide/erase.md)
  says so.

## HDFS and local

Both work through the same layer:

```yaml
  - name: hdfs://namenode:8020/warehouse/events
  - name: /mnt/data/events
```

`memory://` also works, which is what the test suite uses to exercise the
object-storage code path without a network.
