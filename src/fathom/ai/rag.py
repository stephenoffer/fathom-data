"""Provenance for what goes into a context window.

A retrieval-augmented answer is a derived dataset with a lifespan of one request.
It has inputs — a prompt, some retrieved chunks, some tool results — and if you do
not record them, three questions become unanswerable the moment the request ends:

- *Where did this answer come from?* The citation problem. Users ask, and the honest
  answer requires knowing which documents were actually in the window, not which
  ones the retriever might plausibly have found.
- *Did anything leave that should not have?* A chunk carrying personal data, placed
  in a prompt sent to a third-party endpoint, is a transfer. It is invisible unless
  labels travel with the chunk.
- *What was the answer really made of?* A retrieved chunk came from a document, which
  came from a corpus, which came from tables. `provenance` walks all of it.

A `ContextManifest` is that record. It is small, it is written once per request, and
it is the thing that makes the other three answerable a year later.

The deliberate limit: this records what was *retrieved*, not what the model *used*.
Nothing here can tell you which chunk a token came from. `unused_context` is
therefore a cost signal, not an attribution claim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.types import ColumnRef, DatasetId
from ..core.util import digest as _digest
from ..govern.policy import LabelSet, SinkPolicy, Violation, labels_over
from ..graph.model import Graph, link
from ..graph.query import ancestors
from .assets import AssetKind, spec_for

__all__ = [
    "ContextManifest",
    "Retrieval",
    "citation_coverage",
    "context_cost",
    "context_digest",
    "context_tokens",
    "enforce_context",
    "labels_in_context",
    "provenance",
    "record_context",
    "redaction_targets",
    "sources_of",
    "manifest_from_json",
    "manifest_to_json",
    "never_retrieved",
    "retrieval_index",
    "top_sources",
    "uncited_claims",
    "unattributed_retrievals",
    "unused_context",
    "wasted_cost",
]


@dataclass(frozen=True)
class Retrieval:
    """One chunk that made it into the window, and where it came from."""

    chunk_key: str
    score: float = 0.0
    document: str = ""
    dataset: DatasetId | None = None
    token_estimate: int = 0
    rank: int = 0

    @property
    def source_document(self) -> str:
        """The document behind this chunk, derived from the key when not given."""
        return self.document or self.chunk_key.rsplit("#", 1)[0]

    def __str__(self) -> str:
        return f"#{self.rank} {self.chunk_key} ({self.score:.3f})"


@dataclass
class ContextManifest:
    """Everything that entered one model call.

    `sink` is the identity of the endpoint the context was sent to. Recording it is
    what turns a policy check from "does this data carry a PII label" into "did PII
    reach a system not cleared for it", which is the question that actually has legal
    weight.
    """

    run_id: str = ""
    model: DatasetId | None = None
    sink: DatasetId | None = None
    prompt: DatasetId | None = None
    prompt_tokens: int = 0
    retrievals: list[Retrieval] = field(default_factory=list)
    tools: list[DatasetId] = field(default_factory=list)
    occurred: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add(
        self,
        chunk_key: str,
        *,
        score: float = 0.0,
        document: str = "",
        dataset: DatasetId | None = None,
        token_estimate: int = 0,
    ) -> Retrieval:
        """Record one retrieved chunk, ranked by insertion order."""
        item = Retrieval(
            chunk_key=chunk_key,
            score=score,
            document=document,
            dataset=dataset,
            token_estimate=token_estimate,
            rank=len(self.retrievals) + 1,
        )
        self.retrievals.append(item)
        return item

    @property
    def datasets(self) -> list[DatasetId]:
        """Distinct datasets contributing to this context."""
        found = {r.dataset for r in self.retrievals if r.dataset is not None}
        found.update(self.tools)
        if self.prompt is not None:
            found.add(self.prompt)
        return sorted(found, key=str)

    def summary(self) -> str:
        """The manifest as text: chunks, documents, and token count."""
        return (
            f"context {self.run_id or '(unnamed)'}: {len(self.retrievals)} chunk(s) from "
            f"{len(sources_of(self))} document(s), ~{context_tokens(self):,} tokens"
        )


def sources_of(manifest: ContextManifest) -> list[str]:
    """Distinct source documents represented in a context."""
    return sorted({r.source_document for r in manifest.retrievals if r.source_document})


def context_tokens(manifest: ContextManifest) -> int:
    """Total estimated tokens in the window, prompt included."""
    return manifest.prompt_tokens + sum(r.token_estimate for r in manifest.retrievals)


def context_digest(manifest: ContextManifest) -> str:
    """A stable hash of what was in the window.

    Two calls with the same digest saw the same context, which is how you tell a
    non-deterministic model apart from a non-deterministic retriever.
    """
    return _digest.of_json(
        {
            "model": str(manifest.model) if manifest.model else "",
            "prompt": str(manifest.prompt) if manifest.prompt else "",
            "chunks": sorted(r.chunk_key for r in manifest.retrievals),
            "tools": sorted(str(t) for t in manifest.tools),
        }
    )


def record_context(graph: Graph, manifest: ContextManifest) -> Graph:
    """Write a context manifest into the graph as edges into the model call.

    Retrieval edges are unbounded: nothing here can say which slice of a corpus
    produced which part of an answer, and a mapping claiming otherwise would be a
    fabrication the planner would then trust.
    """
    if manifest.model is None:
        return graph
    model_spec = spec_for(AssetKind.MODEL)
    graph.add_dataset(manifest.model, model_spec)
    evidence = f"context:{manifest.run_id}" if manifest.run_id else "context"

    for source in manifest.datasets:
        link(graph, source, manifest.model, evidence=evidence, dst_spec=model_spec)
    if manifest.sink is not None:
        link(graph, manifest.model, manifest.sink, evidence=f"{evidence}:sink")
    return graph


def provenance(graph: Graph, manifest: ContextManifest) -> list[DatasetId]:
    """Everything upstream of everything in the context, transitively.

    The full answer to "what is this response made of" — not the three chunks the
    retriever returned, but the tables those chunks were built from.
    """
    out: set[DatasetId] = set()
    for source in manifest.datasets:
        out.add(source)
        out.update(ancestors(graph, source))
    return sorted(out, key=str)


def labels_in_context(
    graph: Graph, manifest: ContextManifest, labels: LabelSet
) -> dict[str, list[ColumnRef]]:
    """Labels carried by anything that reached this context, grouped by label name.

    Walks the full upstream closure rather than only the retrieved datasets: a chunk
    derived from a table containing email addresses carries that label whether or not
    the chunk itself was ever labelled.
    """
    return labels_over(labels, provenance(graph, manifest))


def unattributed_retrievals(manifest: ContextManifest) -> list[Retrieval]:
    """Retrieved chunks with no dataset recorded.

    `provenance` can only walk from a dataset, so these contribute nothing to the
    closure and no label check can see through them. They are the blind spot in every
    context policy check, and `Retrieval.dataset` defaults to `None` — so the blind
    spot is what you get by default rather than something you opt into.
    """
    return [r for r in manifest.retrievals if r.dataset is None]


def enforce_context(
    graph: Graph,
    manifest: ContextManifest,
    labels: LabelSet,
    policy: SinkPolicy,
    *,
    min_confidence: float = 0.5,
) -> list[Violation]:
    """Check what reached a context against the policy of the endpoint it was sent to.

    This is the `label` verb pointed at a prompt. The sink is a third-party model
    endpoint rather than a table, and the check is the same one: did something
    forbidden reach a place not cleared for it.
    """
    present = labels_in_context(graph, manifest, labels)
    violations: list[Violation] = []

    # "I could not check" must not render as "nothing forbidden". A chunk recorded
    # without its dataset is invisible to `provenance`, so a context carrying
    # personal data to a third-party endpoint returned zero violations purely
    # because nobody said where the chunk came from.
    untraceable = unattributed_retrievals(manifest)
    if untraceable:
        violations.append(
            Violation(
                dataset=policy.dataset,
                column=", ".join(r.chunk_key for r in untraceable[:5]),
                label="unattributed",
                rule=(
                    f"{len(untraceable)} retrieved chunk(s) carry no dataset, so no "
                    "policy could be checked against them"
                ),
                confidence=1.0,
                reason=policy.reason,
            )
        )
    for name in sorted(policy.forbid & set(present)):
        for ref in present[name]:
            confidence = max(
                (label.confidence for label in labels.get(ref, set()) if label.name == name),
                default=0.0,
            )
            confirmed = any(
                label.confirmed for label in labels.get(ref, set()) if label.name == name
            )
            if not confirmed and confidence < min_confidence:
                continue
            violations.append(
                Violation(
                    dataset=policy.dataset,
                    column=f"{ref.dataset}#{ref.column}",
                    label=name,
                    rule="forbidden label reached a model context",
                    confidence=confidence,
                    reason=policy.reason,
                )
            )
    return violations


def redaction_targets(
    graph: Graph, manifest: ContextManifest, labels: LabelSet, *, forbid: Iterable[str] = ("pii",)
) -> list[Retrieval]:
    """Retrieved chunks whose source carries a forbidden label.

    What to drop from the window before sending it, when dropping is preferable to
    refusing the whole request.
    """
    banned = set(forbid)
    tainted: set[DatasetId] = set()
    for ref, values in labels.items():
        if any(label.name in banned for label in values):
            tainted.add(ref.dataset)

    out: list[Retrieval] = []
    for item in manifest.retrievals:
        if item.dataset is None:
            continue
        closure = {item.dataset, *ancestors(graph, item.dataset)}
        if closure & tainted:
            out.append(item)
    return out


def unused_context(manifest: ContextManifest, cited: Iterable[str]) -> list[Retrieval]:
    """Retrieved chunks that no citation referenced.

    A cost signal rather than an attribution claim: the model may well have used
    something it did not cite. Consistently high values mean the retriever's `k` is
    set higher than the answers need, and every one of those tokens is billed.
    """
    used = set(cited)
    return [r for r in manifest.retrievals if r.chunk_key not in used]


def citation_coverage(manifest: ContextManifest, cited: Iterable[str]) -> float:
    """Fraction of the window that was cited. Between 0 and 1."""
    if not manifest.retrievals:
        return 0.0
    used = {key for key in cited if any(r.chunk_key == key for r in manifest.retrievals)}
    return len(used) / len(manifest.retrievals)


def uncited_claims(cited: Iterable[str], manifest: ContextManifest) -> list[str]:
    """Citations naming chunks that were never in the window.

    A hallucinated citation is worse than none, and it is trivially detectable the
    moment the window is recorded.
    """
    present = {r.chunk_key for r in manifest.retrievals}
    return sorted(key for key in cited if key not in present)


def context_cost(
    manifest: ContextManifest, *, price_per_million_input_tokens: float = 0.0
) -> float:
    """What this context cost to send, at a given input price."""
    return context_tokens(manifest) / 1_000_000 * price_per_million_input_tokens


def wasted_cost(
    manifest: ContextManifest,
    cited: Iterable[str],
    *,
    price_per_million_input_tokens: float = 0.0,
) -> float:
    """What the uncited part of the window cost."""
    unused = unused_context(manifest, cited)
    tokens = sum(r.token_estimate for r in unused)
    return tokens / 1_000_000 * price_per_million_input_tokens


def manifest_to_json(manifest: ContextManifest, *, indent: int | None = 2) -> str:
    """Serialize a manifest for storage alongside the response it produced."""
    return json.dumps(
        {
            "run_id": manifest.run_id,
            "occurred": manifest.occurred.isoformat(),
            "model": str(manifest.model) if manifest.model else None,
            "sink": str(manifest.sink) if manifest.sink else None,
            "prompt": str(manifest.prompt) if manifest.prompt else None,
            "digest": context_digest(manifest),
            "tokens": context_tokens(manifest),
            "retrievals": [
                {
                    "rank": r.rank,
                    "chunk": r.chunk_key,
                    "document": r.source_document,
                    "score": r.score,
                    "dataset": str(r.dataset) if r.dataset else None,
                    "tokens": r.token_estimate,
                }
                for r in manifest.retrievals
            ],
            "tools": [str(t) for t in manifest.tools],
        },
        indent=indent,
        sort_keys=True,
    )


def manifest_from_json(raw: str) -> ContextManifest:
    """Rebuild a manifest written by `manifest_to_json`."""
    blob = json.loads(raw)

    def _id(value: str | None) -> DatasetId | None:
        if not value:
            return None
        namespace, _, name = value.rpartition("/")
        return DatasetId(namespace=namespace, name=name)

    manifest = ContextManifest(
        run_id=blob.get("run_id", ""),
        model=_id(blob.get("model")),
        sink=_id(blob.get("sink")),
        prompt=_id(blob.get("prompt")),
        occurred=datetime.fromisoformat(blob["occurred"]),
    )
    for entry in blob.get("retrievals", ()):
        manifest.add(
            entry["chunk"],
            score=float(entry.get("score", 0.0)),
            document=entry.get("document", ""),
            dataset=_id(entry.get("dataset")),
            token_estimate=int(entry.get("tokens", 0)),
        )
    manifest.tools = [t for t in (_id(v) for v in blob.get("tools", ())) if t is not None]
    return manifest


def top_sources(manifests: Sequence[ContextManifest], *, limit: int = 10) -> list[tuple[str, int]]:
    """Which documents get retrieved most across many requests.

    The list to profile and label first: they are in front of users constantly, so
    an error or a policy problem there has the widest reach.
    """
    counts: dict[str, int] = {}
    for manifest in manifests:
        for document in sources_of(manifest):
            counts[document] = counts.get(document, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def never_retrieved(
    manifests: Sequence[ContextManifest], corpus_documents: Iterable[str]
) -> list[str]:
    """Corpus documents no request ever retrieved.

    Embedded, stored, and paid for, contributing nothing. Usually a chunking or
    metadata-filter problem rather than genuinely useless content.
    """
    seen: set[str] = set()
    for manifest in manifests:
        seen.update(sources_of(manifest))
    return sorted(set(corpus_documents) - seen)


def retrieval_index(manifests: Sequence[ContextManifest]) -> Mapping[str, list[str]]:
    """Map each document to the run ids that retrieved it.

    The reverse lookup an incident needs: a document turns out to be wrong, and this
    names every answer that was built on it.
    """
    out: dict[str, list[str]] = {}
    for manifest in manifests:
        for document in sources_of(manifest):
            out.setdefault(document, []).append(manifest.run_id)
    return {k: sorted(v) for k, v in sorted(out.items())}
