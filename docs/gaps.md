# Feature gaps

What fathom cannot yet answer, grouped by who is asking. Each entry names the
question, why the current design cannot reach it, and what closes it.

The organising claim is unchanged: a model, a checkpoint, a prompt, an eval set, and
a vector index are datasets, so the graph already works on them. The gaps are places
where an asset exists that has no `DatasetId`, or a question exists that the graph
could answer and no code asks.

Entries marked **Closed** name the module that answers them. The remaining six —
multimodal assets, a REST API, catalog write-back, orchestrator submission,
ticketing, and scale — are still open.

---

## Research and frontier AI

### 1. Training runs are invisible

**Closed** — `ai/train/experiments.py`.

`ai/training.py` records what a model was trained *on*. It does not record the run:
hyperparameters, the sweep it belonged to, the ablation it was compared against, or
the scaling-law fit that justified the size. When a model regresses, the first
question is "what changed between this run and the last one", and today that is a
spreadsheet.

**Closes it:** runs, sweeps, trials, and ablations as first-class assets; run
comparison; scaling-law fitting and extrapolation; seed and determinism tracking.

### 2. Checkpoints have no topology

**Closed** — `ai/train/checkpoints.py`.

A checkpoint is currently one asset. In practice it is sharded across ranks, written
by a specific parallelism configuration, and resumable only by a job with matching
topology. Resuming a 512-GPU run into a 256-GPU cluster is a real operation with a
real failure mode, and nothing here models it.

**Closes it:** shard lineage, parallelism descriptors (data/tensor/pipeline/expert),
resume compatibility checks, checkpoint retention policy, and rollback.

### 3. Fine-tuning lineage is flat

**Closed** — `ai/train/finetuning.py`.

A LoRA adapter derives from a base model. A merged model derives from both. A
distilled student derives from a teacher. A DPO run derives from a preference set
that derives from annotator output. None of those relationships are expressible, so
"which base model is inside this deployment" cannot be answered.

**Closes it:** base/adapter/merge/distill/quantize relationships, preference-data
lineage, annotator provenance, and the licence propagation that follows from them.

### 4. Contamination checking does not scale

**Closed** — `ai/quality/contamination.py`.

`ai/evals.py` checks contamination as graph reachability, which is correct and
cheap. It cannot catch the case where the eval text was copied into a corpus with no
lineage edge — the common case for scraped data.

**Closes it:** MinHash and n-gram overlap, shingle indexes, near-duplicate detection,
and a contamination report that distinguishes "shares an ancestor" from "shares text".

### 5. Tokenizers are untracked

**Closed** — `ai/train/tokenizers.py`.

Changing a tokenizer changes every token count, every context budget, and every
cached embedding. It is a schema change for text, and it is invisible.

**Closes it:** tokenizer assets, vocabulary diffing, token-count re-estimation, and
budget checks that fail when a vocabulary moves under a fixed context window.

### 6. Only tabular and text data exist

An image corpus, an audio dataset, and a video archive are datasets with partitions,
drift, and personal data in them. The profiler assumes Parquet columns.

**Closes it:** modality-aware assets, perceptual-hash dedup, sample-rate and
resolution drift, and per-modality quality checks.

### 7. Serving is not in the graph

**Closed** — `ai/serve/deployments.py`.

The graph ends at the model. Deployments, routing policies, quantized variants, and
KV-cache reuse are all downstream transformations that change what users see, and
none of them are recorded.

**Closes it:** deployment assets, traffic splits, quantization regression suites,
rollout and rollback lineage, and inference cost attribution.

### 8. Data mixtures are not decisions anyone can audit

**Closed** — `ai/train/mixtures.py`.

Deciding a corpus is 40% web, 30% code, 20% books, 10% math is the single highest-
leverage choice in pretraining, and it lives in a config file nobody versions
against outcomes.

**Closes it:** mixture assets, weight provenance, ablation linkage, and re-mixture
planning that prices what changing a weight actually costs.

### 9. Safety findings do not become regressions

**Closed** — `ai/quality/safety.py`.

A red-team finding is fixed once and forgotten. Nothing turns it into a permanent
test, so the same jailbreak reappears three releases later.

**Closes it:** safety suites as eval sets, refusal-rate regression, and a finding
lifecycle that ends in a test rather than a ticket.

---

## Enterprise

### 10. There is no access control

**Closed** — `govern/access/rbac.py`.

Anyone who can run the CLI can read every profile, every label, and every erasure
proof — including the subject digests. In a regulated environment that is
disqualifying on its own.

**Closes it:** roles, permissions, dataset-scoped grants, column-level masking, and
a deny-by-default posture on governance artefacts.

### 11. Nothing is audited

**Closed** — `govern/access/audit.py`.

Who ran the erasure? Who confirmed the label that let PII into the training set? The
store records outcomes, not actors.

**Closes it:** an append-only, hash-chained audit log; actor identity; and
tamper-evidence that a regulator can verify.

### 12. One store, one tenant

**Closed** — `govern/access/tenancy.py`.

A platform team running fathom for twenty product teams has no isolation boundary.

**Closes it:** tenant scoping through the store, per-tenant policy, and
cross-tenant reference detection.

### 13. Changes need no approval

**Closed** — `govern/approvals.py`.

Editing a partition spec silently changes what every future rebuild covers. There is
no gate, no reviewer, and no record.

**Closes it:** change proposals, required approvals by risk class, and enforcement
that a spec change affecting a contracted dataset cannot merge unreviewed.

### 14. Incidents are not modelled

**Closed** — `observe/incidents.py`.

A drift finding is a line of output. It is not an incident with an owner, a
timeline, a blast radius, or a postmortem.

**Closes it:** incident lifecycle, automatic blast-radius computation from the graph,
notification routing, and a postmortem template populated from lineage.

### 15. Nothing reaches a human

**Closed** — `report/notify.py`.

Every check exits non-zero and that is all. No Slack, no PagerDuty, no email, no
webhook.

**Closes it:** a notifier abstraction with routing rules, deduplication, escalation,
and quiet hours.

### 16. Keys are assumed, not managed

**Closed** — `govern/access/keys.py`.

Crypto-shredding is named as the answer for versioned storage, but there is no key
registry, no rotation, and no way to prove a key was destroyed.

**Closes it:** a key registry, per-subject and per-tenant keys, rotation, and
destruction proofs that compose with erasure proofs.

### 17. There is no API

Everything is a CLI or a Python import. A platform team cannot put a service in
front of it, and no other system can query the graph.

**Closes it:** a typed request/response layer covering the read surface, with
pagination, filtering, and a stable envelope — transport-agnostic so it can be
served by anything.

### 18. Operations cannot see it

**Closed** — `report/telemetry.py`.

No metrics, no traces, no health endpoint. Running this as a service means running it
blind.

**Closes it:** OpenTelemetry spans, Prometheus metrics, structured logs, and health
and readiness checks.

### 19. Catalogs are one-way

`report/emit.py` writes OpenLineage and DataHub. It cannot read from a catalog, so an
organisation with Collibra or Alation as the system of record must maintain two.

**Closes it:** bidirectional catalog sync for DataHub, Collibra, Alation, Atlan, and
Amundsen, with conflict rules.

### 20. Orchestrators are export-only

A plan can be rendered as a DAG. It cannot be submitted, watched, or reconciled
against what actually ran.

**Closes it:** submission and status for Airflow, Dagster, Prefect, and Temporal, and
reconciliation of planned versus executed.

### 21. Findings do not become work

A contract breach should open a ticket assigned to the producing team.

**Closes it:** Jira, ServiceNow, and Linear adapters with idempotent creation and
lifecycle sync.

### 22. The store cannot be upgraded

**Closed** — `store/migrations.py`.

Schema version 1 is checked and refused if newer. There is no migration path, no
backup, and no export.

**Closes it:** forward migrations, backup and restore, export and import, and
compaction.

### 23. It has never been run at scale

Every traversal is a fresh walk. Every profile is a fresh query. There is no cache,
no index beyond SQLite's, and no parallelism.

**Closes it:** graph indexing, memoised traversal, parallel profiling and detection,
batched store writes, and a benchmark suite that fails on regression.

---

## What this does not propose

- **A web UI.** Out of scope for a library; the API layer is what a UI would sit on.
- **A hosted service.** Same reason.
- **Reimplementing an orchestrator.** Submitting and reconciling is the boundary.
- **Model training.** fathom records what trained; it does not train.
