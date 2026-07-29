"""Embeddings and vector indexes as partitioned datasets.

Re-embedding is metered. Every chunk that goes through an embedding endpoint is
billed per token, and the default behaviour of most RAG pipelines — reindex the
corpus on a schedule — pays that bill in full whether or not anything changed. A
corpus of ten million chunks reindexed nightly is not a technical decision anybody
made; it is what happens when nothing knows which chunks moved.

This module makes that a planning problem instead. A vector index is a dataset
partitioned by shard and day, the corpus feeding it is a dataset partitioned the
same way, and the edge between them carries a partition mapping. One changed source
day resolves to one index partition, and `estimate_savings` puts a number on the
difference.

Two rules that fall out of the model and are worth stating because pipelines get
them wrong:

- **An embedding-space version change invalidates every vector.** Vectors from two
  model versions are not comparable, so a similarity search across a partially
  reindexed store returns confidently wrong neighbours. `requires_full_reindex`
  exists to make that unmissable rather than a footnote.
- **A deleted document leaves its vectors behind.** Vector stores are append-mostly
  and deletion is frequently an afterthought, so a subject erased from the corpus is
  still retrievable through the index. `orphan_vectors` finds them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..core.partitions import PartitionMapping
from ..core.types import ANY, DatasetId, KeyPredicate, PartitionSpec
from ..core.util import digest as _digest
from ..core.util.text import token_estimate
from ..graph.model import Graph, InvalidationPlan, link
from ..observe.profile import ColumnProfile, Profile
from .assets import AssetKind, kind_of, spec_for

__all__ = [
    "Chunk",
    "EmbeddingCost",
    "ReindexPlan",
    "centroid_shift",
    "chunk_of",
    "deletion_targets",
    "dimension_change",
    "graph_reindex_plan",
    "norm_shift",
    "retrievable_after_erasure",
    "chunk_digest",
    "chunk_provenance",
    "documents_in",
    "embedding_drift",
    "estimate_cost",
    "estimate_savings",
    "indexes_for_space",
    "orphan_vectors",
    "record_index",
    "reindex_plan",
    "requires_full_reindex",
    "space_of",
    "stale_chunks",
    "stale_shards",
    "vector_profile",
    "version_mismatch",
]


@dataclass(frozen=True)
class Chunk:
    """One embedded span of one document.

    `content_digest` is what decides whether re-embedding is needed. Comparing text
    rather than modification time matters here: a document rewritten to identical
    content costs nothing to skip and real money to re-embed.
    """

    document: str
    ordinal: int = 0
    content_digest: str = ""
    token_estimate: int = 0
    shard: str = ""

    @property
    def key(self) -> str:
        """Stable identity for this chunk: document and ordinal."""
        return f"{self.document}#{self.ordinal}"

    def __str__(self) -> str:
        return self.key


def chunk_digest(text: str) -> str:
    """A stable content hash for one chunk of text."""
    return _digest.short(_digest.of_text(text, normalize=False), 32)


def chunk_of(document: str, ordinal: int, text: str, *, shard: str = "") -> Chunk:
    """Build a chunk with its digest and token estimate filled in."""
    return Chunk(
        document=document,
        ordinal=ordinal,
        content_digest=chunk_digest(text),
        token_estimate=token_estimate(text),
        shard=shard,
    )


# -- wiring --------------------------------------------------------------------


def record_index(
    graph: Graph,
    *,
    index: DatasetId,
    corpus: DatasetId,
    space: DatasetId,
    corpus_spec: PartitionSpec | None = None,
) -> Graph:
    """Wire a corpus and an embedding space into a vector index.

    Two edges, and they behave differently on purpose. The corpus edge carries a
    real partition mapping, because a changed source day maps to a specific index
    partition. The embedding-space edge is unbounded, because changing the embedding
    model invalidates the entire index and nothing narrower would be true.
    """
    index_spec = spec_for(AssetKind.VECTOR_INDEX)
    graph.add_dataset(index, index_spec)
    graph.add_dataset(corpus, corpus_spec or spec_for(AssetKind.CORPUS))

    link(
        graph,
        corpus,
        index,
        evidence="embedding:corpus",
        mapping=PartitionMapping.rollup(graph.spec(corpus), index_spec),
    )
    link(
        graph,
        space,
        index,
        evidence="embedding:space",
        src_spec=spec_for(AssetKind.EMBEDDING_SPACE),
    )
    return graph


def space_of(graph: Graph, index: DatasetId) -> DatasetId | None:
    """The embedding space an index was built in, or None when it was never recorded."""
    for edge in graph.in_edges(index):
        if kind_of(edge.src) is AssetKind.EMBEDDING_SPACE:
            return edge.src
    return None


def indexes_for_space(graph: Graph, space: DatasetId) -> list[DatasetId]:
    """Every index built in one embedding space.

    All of them are invalidated together when the space's model version moves, which
    is the blast radius of an embedding-model upgrade.
    """
    return sorted(
        {e.dst for e in graph.out_edges(space) if kind_of(e.dst) is AssetKind.VECTOR_INDEX},
        key=str,
    )


def documents_in(chunks: Iterable[Chunk]) -> list[str]:
    """Distinct source documents behind a set of chunks."""
    return sorted({chunk.document for chunk in chunks})


# -- staleness -----------------------------------------------------------------


def stale_chunks(indexed: Mapping[str, str], current: Iterable[Chunk]) -> list[Chunk]:
    """Chunks whose content changed, or which were never embedded.

    `indexed` maps chunk key to the content digest recorded at embedding time. The
    comparison is on content, so a pipeline that rewrites every document nightly
    still only pays for the ones that actually differ.
    """
    out: list[Chunk] = []
    for chunk in current:
        previous = indexed.get(chunk.key)
        if previous is None or previous != chunk.content_digest:
            out.append(chunk)
    return sorted(out, key=lambda c: c.key)


def orphan_vectors(indexed: Iterable[str], current: Iterable[Chunk]) -> list[str]:
    """Indexed chunk keys with no corresponding chunk in the corpus any more.

    These are the vectors that keep answering questions about documents that were
    deleted. Every erasure request that stops at the corpus leaves some of these.
    """
    live = {chunk.key for chunk in current}
    return sorted(key for key in indexed if key not in live)


def version_mismatch(index_version: str, space_version: str) -> bool:
    """True when an index holds vectors from a different model version than the space.

    Not a warning. Mixed-version vectors make nearest-neighbour results meaningless
    while looking entirely healthy from the outside.

    One side being unknown counts as a mismatch. Requiring *both* to be known before
    reporting one meant the commonest real case — an index built before anyone
    recorded a version, now facing a named model — was waved through, leaving the
    index holding two embedding spaces at once. Unknown means unproven, and the safe
    direction is to re-embed. Two unknowns are a project not using versioning at all,
    which is not evidence of a change.
    """
    if not index_version and not space_version:
        return False
    return index_version != space_version


def requires_full_reindex(before_version: str, after_version: str) -> bool:
    """True when an embedding-model change invalidates every existing vector."""
    return version_mismatch(before_version, after_version)


def stale_shards(chunks: Iterable[Chunk]) -> list[KeyPredicate]:
    """Index partitions covering a set of stale chunks.

    Chunks with no shard recorded widen to `ANY`, which reindexes more than needed
    and never less — the same direction the planner errs in everywhere else.
    """
    keys: set[KeyPredicate] = set()
    for chunk in chunks:
        keys.add(KeyPredicate(bindings=(("shard", chunk.shard or ANY), ("dt", ANY))))
    return sorted(keys, key=str)


# -- planning ------------------------------------------------------------------


@dataclass
class ReindexPlan:
    """What has to be re-embedded, and what it will cost."""

    index: DatasetId
    chunks: list[Chunk] = field(default_factory=list)
    partitions: list[KeyPredicate] = field(default_factory=list)
    total_chunks: int = 0
    full_reindex: bool = False
    reason: str = ""
    # Indexed keys with no chunk in the corpus any more. Re-embedding is only half a
    # reindex: these vectors keep answering questions about documents that no longer
    # exist, which is how a deletion — including an erasure request — stays visible
    # through retrieval. Reporting only what to add would call that plan complete.
    orphans: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        """Tokens this plan will send to the embedding endpoint."""
        return sum(chunk.token_estimate for chunk in self.chunks)

    @property
    def skipped(self) -> int:
        """Chunks the plan avoids re-embedding."""
        return max(0, self.total_chunks - len(self.chunks))

    @property
    def savings(self) -> float:
        """Fraction of the corpus this plan avoids re-embedding."""
        if self.total_chunks <= 0:
            return 0.0
        return max(0.0, 1.0 - len(self.chunks) / self.total_chunks)

    def summary(self) -> str:
        """The plan as text, including orphaned vectors still to delete."""
        stale = f"; {len(self.orphans):,} orphaned vector(s) to delete" if self.orphans else ""
        if self.full_reindex:
            return (
                f"{self.index}: FULL reindex of {self.total_chunks:,} chunk(s) "
                f"(~{self.tokens:,} tokens) — {self.reason}{stale}"
            )
        return (
            f"{self.index}: {len(self.chunks):,} of {self.total_chunks:,} chunk(s) "
            f"(~{self.tokens:,} tokens), {self.savings:.0%} skipped{stale}"
        )


def reindex_plan(
    index: DatasetId,
    *,
    indexed: Mapping[str, str],
    current: Sequence[Chunk],
    index_version: str = "",
    space_version: str = "",
) -> ReindexPlan:
    """Decide what to re-embed, content by content.

    A model-version change short-circuits everything: no per-chunk comparison is
    meaningful across embedding spaces, so the plan becomes a full reindex with the
    reason attached rather than a cheap-looking plan that produces a broken index.
    """
    orphans = orphan_vectors(indexed, current)

    if requires_full_reindex(index_version, space_version):
        return ReindexPlan(
            index=index,
            chunks=list(current),
            partitions=[KeyPredicate(bindings=(("dt", ANY), ("shard", ANY)))],
            total_chunks=len(current),
            full_reindex=True,
            reason=(
                f"embedding space moved from {index_version!r} to {space_version!r}; "
                "vectors from two model versions are not comparable"
            ),
            orphans=orphans,
        )

    stale = stale_chunks(indexed, current)
    return ReindexPlan(
        index=index,
        chunks=stale,
        partitions=stale_shards(stale),
        total_chunks=len(current),
        full_reindex=False,
        orphans=orphans,
    )


def graph_reindex_plan(
    graph: Graph, dirty: Mapping[DatasetId, Iterable[KeyPredicate]]
) -> InvalidationPlan:
    """Which index partitions a set of changed corpus partitions invalidates.

    The graph-level counterpart to `reindex_plan`: use this when the corpus is a
    partitioned dataset in the graph and you want the answer without enumerating
    chunks.
    """
    return graph.invalidate(dirty)


# -- cost ----------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingCost:
    """What a reindex costs, in tokens and money."""

    chunks: int = 0
    tokens: int = 0
    price_per_million_tokens: float = 0.0

    @property
    def cost(self) -> float:
        """What this plan costs at the model's per-token price."""
        return self.tokens / 1_000_000 * self.price_per_million_tokens

    def __str__(self) -> str:
        return f"{self.chunks:,} chunk(s), ~{self.tokens:,} tokens, ~${self.cost:,.2f}"


def estimate_cost(
    chunks: Iterable[Chunk], *, price_per_million_tokens: float = 0.0
) -> EmbeddingCost:
    """Cost of embedding a set of chunks at a given price point."""
    items = list(chunks)
    return EmbeddingCost(
        chunks=len(items),
        tokens=sum(chunk.token_estimate for chunk in items),
        price_per_million_tokens=price_per_million_tokens,
    )


def estimate_savings(
    plan: ReindexPlan, *, price_per_million_tokens: float = 0.0
) -> dict[str, float]:
    """What the plan costs against what a full reindex would have.

    The number to put in front of whoever signs off on the embedding bill.
    """
    incremental = estimate_cost(plan.chunks, price_per_million_tokens=price_per_million_tokens)
    # A full reindex costs the same per chunk; scale rather than requiring the caller
    # to hand over every chunk in the corpus a second time.
    per_chunk = incremental.tokens / len(plan.chunks) if plan.chunks else 0.0
    full_tokens = int(per_chunk * plan.total_chunks)
    full_cost = full_tokens / 1_000_000 * price_per_million_tokens
    return {
        "chunks_reembedded": float(len(plan.chunks)),
        "chunks_total": float(plan.total_chunks),
        "tokens_incremental": float(incremental.tokens),
        "tokens_full": float(full_tokens),
        "cost_incremental": incremental.cost,
        "cost_full": full_cost,
        "cost_avoided": max(0.0, full_cost - incremental.cost),
        "savings_ratio": plan.savings,
    }


# -- drift ---------------------------------------------------------------------


def vector_profile(
    index: DatasetId,
    *,
    partition: KeyPredicate | None = None,
    count: int = 0,
    dimensions: int = 0,
    mean_norm: float | None = None,
    centroid: Sequence[float] = (),
) -> Profile:
    """Express vector statistics as an ordinary `Profile`.

    Reusing the profile type means embedding drift goes through the same store,
    the same history, and the same `check` verb as column drift, instead of becoming
    a parallel monitoring stack that nobody wires up.
    """
    columns = [
        ColumnProfile(name="_count", dtype="int64", row_count=count, min=count, max=count),
        ColumnProfile(
            name="_dimensions", dtype="int64", row_count=count, min=dimensions, max=dimensions
        ),
    ]
    if mean_norm is not None:
        columns.append(
            ColumnProfile(
                name="_mean_norm", dtype="double", row_count=count, min=mean_norm, max=mean_norm
            )
        )
    for position, value in enumerate(centroid):
        columns.append(
            ColumnProfile(
                name=f"_centroid_{position:04d}",
                dtype="double",
                row_count=count,
                min=value,
                max=value,
            )
        )
    return Profile(
        dataset=index,
        partition=partition or KeyPredicate(),
        row_count=count,
        columns=tuple(columns),
        source="vector",
    )


def centroid_shift(before: Profile, after: Profile) -> float | None:
    """Euclidean distance between two profiles' centroids, or None when absent.

    A moving centroid is the cheapest available signal that a corpus changed
    character — new topics, a different language mix, a scraper that started
    picking up boilerplate.
    """
    total = 0.0
    seen = 0
    for column in before.columns:
        if not column.name.startswith("_centroid_"):
            continue
        other = after.column(column.name)
        if other is None or column.min is None or other.min is None:
            continue
        try:
            total += (float(other.min) - float(column.min)) ** 2
        except (TypeError, ValueError):
            continue
        seen += 1
    return total**0.5 if seen else None


def norm_shift(before: Profile, after: Profile) -> float | None:
    """Relative change in mean vector norm, or None when it was not recorded."""
    b, a = before.column("_mean_norm"), after.column("_mean_norm")
    if b is None or a is None or b.min is None or a.min is None:
        return None
    try:
        base = float(b.min)
        return None if base == 0 else (float(a.min) - base) / base
    except (TypeError, ValueError):
        return None


def dimension_change(before: Profile, after: Profile) -> tuple[int, int] | None:
    """Dimensionality before and after, when it changed.

    A dimension change is never a drift signal. It means the embedding model was
    swapped and the index is now internally inconsistent.
    """
    b, a = before.column("_dimensions"), after.column("_dimensions")
    if b is None or a is None or b.min is None or a.min is None:
        return None
    try:
        before_dim, after_dim = int(b.min), int(a.min)
    except (TypeError, ValueError):
        return None
    return None if before_dim == after_dim else (before_dim, after_dim)


def embedding_drift(
    before: Profile, after: Profile, *, centroid_tolerance: float = 0.1
) -> list[str]:
    """Human-readable embedding drift findings between two vector profiles."""
    findings: list[str] = []

    dimensions = dimension_change(before, after)
    if dimensions is not None:
        findings.append(
            f"[error] dimensionality changed {dimensions[0]} -> {dimensions[1]}; "
            "the index holds vectors from two different models"
        )

    shift = centroid_shift(before, after)
    if shift is not None and shift > centroid_tolerance:
        findings.append(f"[warn] centroid moved {shift:.4f}, above the {centroid_tolerance} floor")

    norm = norm_shift(before, after)
    if norm is not None and abs(norm) > 0.1:
        findings.append(f"[warn] mean vector norm moved {norm:+.1%}")

    if before.row_count and after.row_count:
        delta = (after.row_count - before.row_count) / before.row_count
        if abs(delta) > 0.25:
            findings.append(
                f"[warn] vector count moved {delta:+.1%} "
                f"({before.row_count:,} -> {after.row_count:,})"
            )
    return findings


# -- erasure -------------------------------------------------------------------


def chunk_provenance(graph: Graph, index: DatasetId) -> list[DatasetId]:
    """Everything upstream of a vector index — the documents a retrieved chunk came from."""
    from ..graph.query import ancestors

    return ancestors(graph, index)


def deletion_targets(indexed: Iterable[str], documents: Iterable[str]) -> list[str]:
    """Indexed chunk keys belonging to a set of documents.

    What an erasure request has to remove from the vector store. Matching is on the
    document prefix of the chunk key, so it holds regardless of how the pipeline
    chose to chunk.
    """
    wanted = set(documents)
    return sorted(key for key in indexed if key.rsplit("#", 1)[0] in wanted)


def retrievable_after_erasure(indexed: Iterable[str], erased_documents: Iterable[str]) -> list[str]:
    """Chunks still in the index for documents that were supposed to be erased.

    Non-empty means the erasure is incomplete and the data is still reachable
    through search, which is the failure mode this whole module exists to surface.
    """
    return deletion_targets(indexed, erased_documents)
