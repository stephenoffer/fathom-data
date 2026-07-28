# label — what a column means, and what policy applies

Nobody hand-annotates forty thousand columns. Inference proposes; a human confirms;
confirmation is sticky.

## Inference

```bash
fathom profile        # labels are inferred from profiles, so profile first
fathom label
```

```
  file:///lake/users email_address            email              75%  inferred:name
  file:///lake/users email_address            pii                75%  implied
  file:///lake/users latitude                 latitude           90%  inferred:name+stats
  file:///lake/users amount_cents             minor_units        95%  inferred:name+stats
  file:///lake/users user_id                  user_identifier    70%  inferred:name
```

Confidence never reaches 1.0 from inference. These are proposals.

### Two honest limits

**Footer-only inference is name-driven.** Parquet footers give min/max, null counts,
and types — not value vocabularies. A column called `email` is proposed as an email
by its name and type, at a confidence that says so.

**Statistics corroborate or refute, and refutation is the useful direction.** A
column named `latitude` whose values reach 4,000 is not a latitude, and the guess is
dropped rather than downgraded:

```python
infer(profile(ColumnProfile(name="latitude", dtype="double", min=0.0, max=4000.0)))
# {} — the name matched, the data said no
```

Where statistics agree, confidence rises and the origin records why
(`inferred:name+stats`).

### Detected labels

| Label | Implies PII |
|---|---|
| `email`, `phone`, `national_id`, `date_of_birth` | yes |
| `person_name`, `postal_address`, `ip_address`, `device_id` | yes |
| `currency_code`, `monetary_amount`, `minor_units` | no |
| `latitude`, `longitude`, `user_identifier` | no |

Anything in the first two rows adds `pii` automatically.

## Propagation

Labels flow downstream along the same edges dirtiness does. A column derived from
an email column is still email-derived.

```python
from fathom import infer, propagate

seeds = infer(profile)
labels = propagate(graph, seeds)
```

Confidence decays per hop, so a label six transformations from its evidence does not
read as strongly as one at the source.

### Edges without column lineage

Where an edge carries no column detail — a Spark job using the DataFrame API, an
opaque UDF — the label lands on the target as **unattributed** rather than being
dropped:

```
file:///lake/derived (column unknown): forbidden label 'pii' (confidence 86%)
```

Losing track of PII because a job used an API we cannot parse is worse than an
imprecise warning. Enforcement reads unattributed as "may contain", which is the
safe direction.

## Enforcement

```yaml
policies:
  - dataset: PROD.ML.TRAINING_SET
    forbid: [pii]
    reason: not cleared for personal data
```

```bash
fathom label
```

```
policy: 1 violation(s)
  snowflake://ac1/PROD.ML.TRAINING_SET feature_1: forbidden label 'pii'
    (confidence 71%) — not cleared for personal data
```

Exits 1, so it drops into CI directly.

`require` works the other way, failing when an expected label is absent:

```yaml
  - dataset: PROD.ML.TRAINING_SET
    require: ["consent:training"]
```

### Confidence gating

Weak inferences do not block pipelines. `min_confidence` defaults to 0.5, and a
**confirmed** label always counts regardless — a human already decided.

```python
project.enforce(labels)                                  # default gate
enforce(labels, policies, min_confidence=0.8)            # stricter
```

## Confirming a proposal

```python
project.store.set_label(
    dataset, "email_address", "pii",
    confidence=1.0, origin="reviewed by data-gov", confirmed=True,
)
```

Confirmation is sticky. Re-running inference cannot undo it, and propagation cannot
downgrade it. That is deliberate: the whole design depends on human review being
worth doing once.

## Adding a detector

Detectors are name-and-type patterns with an optional statistical check:

```python
from fathom.policy import _Detector, _DETECTORS  # to be made public in a later release
```

Two rules if you add one. Keep confidence below 0.9 for a name-only match, and add a
`_range_supports` case if the footer statistics can refute it. The `postal_address`
detector is the cautionary example: an early version matched `email_address` because
it looked for `address` anywhere in the name.
