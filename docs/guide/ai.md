# AI assets — models, features, vectors, prompts, evals

A model is produced from named inputs, in slices that can be rebuilt independently,
with a version history somebody will need explained, and it can contain a person's
data. So is a feature view, a vector index, a prompt, an eval set, and an agent run.

Give them `DatasetId`s and the graph, the planner, the profiler, the policy engine,
and the eraser already work on them. There is no second graph for ML, which matters
because a second graph is a graph that disagrees with the first one.

```python
from fathom.ai import assets

assets.model("fraud.scorer", registry="internal")   # model://internal/fraud.scorer
assets.feature_view("user.activity", store="feast") # feature://feast/user.activity
assets.vector_index("docs", store="pgvector")       # index://pgvector/docs
assets.prompt("triage.system")                      # prompt://local/triage.system
assets.eval_set("bench.v1")                         # eval://local/bench.v1
```

Each kind carries a conventional partition spec, which is what makes it plannable:

| Kind | Partitioned by | So that |
|---|---|---|
| model, adapter, prompt, eval set | `version` | a retrain is a partition rebuild |
| feature view | `dt`, `entity` | a backfill costs what changed |
| vector index | `shard`, `dt` | re-embedding scales with the delta, not the corpus |
| corpus | `dt`, `source` | a new crawl invalidates one day |

Defaults, not rules. Declaring a different spec on the graph overrides it, and you
should whenever the real materialization differs — a spec that lies costs more than
no spec, because the planner trusts it.

## Recording a training run

```python
from fathom.ai import training

run = training.TrainingRun(model=MODEL, version="v3", code_version="9f2a1c")
run.add_input(GOLD, partitions=plan.partitions(GOLD), columns=["amount", "region"])
training.record_training_run(graph, run)
```

Every input edge is `UNBOUNDED`, and that is not a limitation being papered over. A
model is a function of all its training data at once, so no partition of the model
can be attributed to a partition of the input. One changed day correctly invalidates
the whole model rather than some fictional slice of it.

`pin_from_plan(plan, dataset)` bridges the two: whatever the planner said was fresh
is exactly what the run should record as the slice it consumed.

### What that buys

```python
training.stale_models(graph, {RAW: [KeyPredicate.of(dt=yesterday)]})
# [model://internal/fraud.scorer]
```

Retraining becomes an invalidation question rather than a weekly cron that runs
whether or not anything moved.

```python
bom = training.data_bill_of_materials(graph, MODEL)
bom.is_complete   # False
bom.gaps          # ['the edge from gold.training_set carries no column detail, …']
```

A bill of materials that hides what it could not determine is worse than none,
because it will be read as complete.

```python
print(training.training_data_summary(graph, MODEL))
```

Generates the prose an EU AI Act filing asks general-purpose model providers to
publish — from lineage, so it cannot go stale in a document nobody re-reads.

### Reproducibility

```python
training.unpinned_inputs(run)   # inputs recorded without a snapshot or partition set
training.is_reproducible(run)   # every input pinned, and a code version recorded
training.run_digest(run)        # inputs + code + hyperparameters, no timestamps
```

`run_digest` deliberately excludes when the run happened. Two runs with the same
digest should produce the same model, and *when* they ran does not bear on that. A
digest that changed every run would be a timestamp with extra steps.

## Vectors — the expensive one

Every chunk that goes through an embedding endpoint is billed per token. Most RAG
pipelines reindex the corpus on a schedule and pay that bill in full whether or not
anything changed. A corpus of ten million chunks reindexed nightly is not a decision
anybody made; it is what happens when nothing knows which chunks moved.

```python
from fathom.ai import vectors

current = [vectors.chunk_of(doc, i, text, shard=shard) for ...]
plan = vectors.reindex_plan(INDEX, indexed=stored_digests, current=current)
print(plan.summary())
# index://pgvector/docs: 3 of 100 chunk(s) (~300 tokens), 97% skipped

vectors.estimate_savings(plan, price_per_million_tokens=0.13)["cost_avoided"]
```

The comparison is on **content**, not modification time. A pipeline that rewrites
every document nightly still only pays for the ones that actually differ.

### Two rules pipelines get wrong

**An embedding-model version change invalidates every vector.** Vectors from two
model versions are not comparable, so similarity search across a partially reindexed
store returns confidently wrong neighbours. `reindex_plan` short-circuits to a full
reindex with the reason attached rather than producing a cheap-looking plan and a
broken index.

**A deleted document leaves its vectors behind.** Vector stores are append-mostly and
deletion is frequently an afterthought, so a subject erased from the corpus stays
retrievable through the index.

```python
vectors.orphan_vectors(indexed_keys, current)          # source gone, vector remains
vectors.retrievable_after_erasure(indexed_keys, erased) # what the erasure missed
```

## Context windows

A retrieval-augmented answer is a derived dataset with a lifespan of one request.
Record its inputs or three questions become unanswerable the moment it ends.

```python
from fathom.ai import rag

manifest = rag.ContextManifest(run_id=request_id, model=MODEL, sink=ENDPOINT)
manifest.add(chunk_key, score=0.91, dataset=CORPUS, token_estimate=200)

rag.provenance(graph, manifest)            # the tables behind the chunks
rag.enforce_context(graph, manifest, labels, SinkPolicy.no_pii(ENDPOINT))
```

That last line is the `label` verb pointed at a prompt. The sink is a third-party
model endpoint rather than a table, and the check is the same one: did something
forbidden reach a place not cleared for it.

**The limit, stated:** this records what was *retrieved*, not what the model *used*.
Nothing here can tell you which chunk a token came from, so `unused_context` is a
cost signal and not an attribution claim.

## Evals

An eval score is a claim about data the model has not seen. Contamination is a
reachability property, so lineage can check it.

```python
from fathom.ai import evals

report = evals.contamination(graph, MODEL, EVAL_SET)
report.severity   # 'clean' | 'suspect' | 'contaminated'
```

Three levels of evidence, reported separately because they mean different things:

| Evidence | Verdict | Note |
|---|---|---|
| shared ancestry | `suspect` | common upstream; frequently benign |
| reachability | `contaminated` | one set is derived from the other |
| identifier overlap | `contaminated` | conclusive, and the only one needing data |

Without identifiers the verdict tops out at `suspect`. That is the honest ceiling for
a graph-only check, and it is stated rather than rounded up.

```python
evals.contaminated_models(graph)   # every provable leak in the registry
evals.holdout_integrity(graph, EVAL_SET)
```

## Feature views

Two failures, neither of which shows up as an error anywhere.

**Target leakage** — a feature computed from a column derived from the label. The
model scores beautifully in training and collapses in production, because at serving
time that column does not exist yet.

```python
from fathom.ai import features
features.leaky_features(graph, VIEW, label=ColumnRef(GOLD, "is_fraud"))
```

Needs column lineage on the path. Without it this returns nothing — a false negative,
stated as such rather than approximated with a guess.

**Training/serving skew** — offline and online copies computed by different code
paths. That is a profile question, so it reuses the same drift comparison the `check`
verb runs, pointed at two materializations rather than at two days.

```python
features.skew(offline_profile, online_profile).summary()
features.serving_risks(graph, views)   # stale views, ranked by whether a model serves from them
```

## Erasure that reaches the model

Deleting a subject's rows from every table is the part tooling handles well. It is
also not the whole request. If those rows fed a training run, the model retains them,
and no amount of deletion downstream of the weights changes that.

```python
from fathom.ai import unlearning

unlearning.is_deletion_sufficient(graph, ORIGIN)   # False, once a model was trained
unlearning.exposures(graph, ORIGIN)                # every asset, and by which route
unlearning.retraining_required(graph, ORIGIN)      # the models whose only remedy is a retrain
```

`obligations()` states, per asset, what would actually discharge the request:

| Route | Remedy | Complete? |
|---|---|---|
| training, fine-tune | retrain without the subject | yes |
| training, if encrypted per subject | crypto-shred the key | yes |
| training, approximate unlearning | influence-based removal | **no — a claim, not a proof** |
| retrieval | delete the vectors | yes |
| context, evaluation | purge retained logs | if logs are the only copy |

```python
print(unlearning.completeness_statement(graph, ORIGIN, subject_digest=digest))
```

```
# Erasure completeness — subject a1b2c3d4e5f6…

**This erasure is not complete.**

1 model(s) were trained on data derived from this subject. Deleting the subject's
rows from storage does not remove their contribution to those models' parameters…
```

Wire it into an ordinary erasure plan with `unlearning.extend_plan(graph, plan)`,
which rewrites the AI targets with the reason that actually applies and keeps the
plan reporting `is_complete = False`. That is the point: the plan must not read as
finished while a model still holds the data.

Nothing here claims to remove a subject from a model. It claims to tell you, exactly,
what you have not yet done.

## Prompts and agents

```python
from fathom.ai import prompts, agents

template = prompts.PromptTemplate(dataset=assets.prompt("triage.system"))
template.commit("Answer for {{user_name}} using {{context}}.")
template.bind("user_name", RAW_USERS)     # now an edge in the graph
prompts.unbound_variables(template)       # data entering from somewhere unseen
```

Prompt versions are content-addressed and whitespace-normalized, so reformatting is
not a version and changing a word is. A variable bound to a table carries that
table's labels into the prompt — which is how personal data reaches a third-party
endpoint without anyone writing a line of code that sends it.

```python
run = agents.AgentRun(agent=assets.agent("triage"), run_id=rid)
run.call(assets.tool("sql"), reads=[GOLD])
run.call(assets.tool("webhook"), egress=True)

agents.reach(graph, run)                  # everything it could have seen, transitively
agents.first_time_access(run, history)    # the cheapest anomaly signal there is
agents.least_privilege_gap(granted, history)  # the revocation list, backed by evidence
agents.risk_report(graph, run, labels=labels, history=history).summary()
```

`exfiltration_paths` says *could have*, deliberately: it reports that sensitive data
was in reach and an egress tool was called in the same run — not that one flowed into
the other, which nothing outside the agent's own process can establish.

## Attribution

```python
from fathom.ai import attribution

diagnosis = attribution.attribute_column(
    graph, ColumnRef(GOLD, "revenue"), before=stored, after=fresh
)
print(attribution.blame_report(diagnosis))
```

Ranks upstream columns by drift of their own, breaking ties on proximity. A dataset
with no profile is reported as **unchecked**, never as clean — "we did not look" and
"we looked and it was fine" are different answers, and conflating them sends people
down the wrong path.

## See also

- [`label`](label.md) — the policy machinery these reuse
- [`erase`](erase.md) — the table half of an erasure request
- [Concepts](concepts.md) — why the partition mapping is what makes any of this cheap
