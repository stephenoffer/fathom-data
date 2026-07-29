"""Contamination and near-duplicate detection by content, not just by lineage.

`evals.py` answers contamination as reachability: if the eval set and the training
corpus share an ancestor, the score measures memorisation. That is correct, cheap,
and misses the case that actually happens — the eval text was scraped into the
corpus with no edge between them at all.

This module closes that by comparing text. MinHash over shingles for corpus-scale
near-duplicate detection, exact n-gram overlap for the precise question, and a
verdict that keeps the two separate:

- **`SUSPECT`** means the estimate crossed a threshold. It is a reason to check.
- **`CONTAMINATED`** means verified overlap was found and can be shown.

Rounding the first up to the second is how a benchmark result gets thrown away over
a false positive, and rounding the second down to the first is how a contaminated
model ships. The threshold is a parameter, the verification is not optional, and
`report` shows the matching text so a human can settle it.

Everything here is pure Python and dependency-free. It is not the fastest possible
implementation, and for corpora past a few hundred million documents you want the
sketches computed by whatever already reads the data. `Sketch` is serialisable for
exactly that reason.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "ContaminationReport",
    "ContaminationVerdict",
    "DuplicateCluster",
    "Match",
    "Sketch",
    "cluster_duplicates",
    "containment",
    "estimate_jaccard",
    "exact_overlap",
    "find_duplicates",
    "jaccard",
    "longest_common_substring",
    "minhash",
    "ngram_overlap",
    "ngrams",
    "normalize_text",
    "report",
    "shingles",
    "sketch_of",
    "suspicious_pairs",
    "verify",
]

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")

# A 64-bit mask keeps the hashes in a range that stays exact in Python ints without
# growing without bound.
_MASK = (1 << 61) - 1


def normalize_text(text: str, *, fold_case: bool = True, strip_punctuation: bool = True) -> str:
    """Canonicalise before comparing.

    Contamination survives reformatting. An eval question reindented, recased, and
    stripped of its question mark is still that question, and a comparison that
    misses it is worse than no comparison because it reports clean.
    """
    out = text
    if strip_punctuation:
        out = _PUNCTUATION.sub(" ", out)
    out = _WHITESPACE.sub(" ", out).strip()
    return out.lower() if fold_case else out


def ngrams(text: str, n: int = 13) -> list[str]:
    """Word n-grams. Thirteen is the conventional default for text contamination."""
    words = normalize_text(text).split()
    if n <= 0 or len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def shingles(text: str, *, size: int = 5) -> set[str]:
    """Overlapping word shingles, the unit MinHash estimates over."""
    return set(ngrams(text, size))


def _hash(value: str, seed: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8, salt=seed.to_bytes(8, "little"))
    return int.from_bytes(digest.digest(), "little") & _MASK


@dataclass(frozen=True)
class Sketch:
    """A MinHash signature. Serialisable so it can be computed where the data lives."""

    identifier: str
    signature: tuple[int, ...]
    shingle_size: int = 5
    tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "signature": list(self.signature),
            "shingle_size": self.shingle_size,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> Sketch:
        """Rebuild from a serialised sketch, rejecting a malformed one loudly.

        A sketch read back with a truncated signature would silently compare against
        everything at a similarity nobody asked for, so the shape is checked here.
        """
        signature = blob.get("signature")
        if not isinstance(signature, (list, tuple)) or not signature:
            raise ValueError("a sketch needs a non-empty signature")
        return cls(
            identifier=str(blob["identifier"]),
            signature=tuple(int(v) for v in signature),
            shingle_size=int(blob.get("shingle_size", 5)),
            tokens=int(blob.get("tokens", 0)),
        )


def minhash(text: str, *, permutations: int = 128, shingle_size: int = 5) -> tuple[int, ...]:
    """MinHash signature over the text's shingles.

    More permutations narrows the estimate's error at linear cost. 128 gives roughly
    ±0.09 at one standard deviation, which is fine for triage and not fine for a
    verdict — which is why a verdict requires verification.
    """
    parts = shingles(text, size=shingle_size)
    if not parts:
        return tuple([_MASK] * permutations)
    return tuple(min(_hash(part, seed) for part in parts) for seed in range(permutations))


def sketch_of(
    identifier: str, text: str, *, permutations: int = 128, shingle_size: int = 5
) -> Sketch:
    return Sketch(
        identifier=identifier,
        signature=minhash(text, permutations=permutations, shingle_size=shingle_size),
        shingle_size=shingle_size,
        tokens=len(normalize_text(text).split()),
    )


def estimate_jaccard(left: Sketch, right: Sketch) -> float:
    """Estimated Jaccard similarity from two signatures.

    Signatures of different lengths or shingle sizes are not comparable, and
    comparing them anyway produces a number that looks fine and means nothing.
    """
    if len(left.signature) != len(right.signature):
        raise ValueError(
            f"signatures have different lengths ({len(left.signature)} vs "
            f"{len(right.signature)}); they are not comparable"
        )
    if left.shingle_size != right.shingle_size:
        raise ValueError(
            f"signatures use different shingle sizes ({left.shingle_size} vs "
            f"{right.shingle_size}); they are not comparable"
        )
    if not left.signature:
        return 0.0
    matches = sum(1 for a, b in zip(left.signature, right.signature, strict=True) if a == b)
    return matches / len(left.signature)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Exact Jaccard over two shingle sets."""
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def containment(needle: Iterable[str], haystack: Iterable[str]) -> float:
    """Fraction of `needle` present in `haystack`.

    The right measure for contamination. Jaccard is symmetric and punishes size
    differences, so a short eval set inside an enormous corpus scores near zero on
    Jaccard while being fully contained.
    """
    a, b = set(needle), set(haystack)
    return len(a & b) / len(a) if a else 0.0


def ngram_overlap(left: str, right: str, *, n: int = 13) -> float:
    """Containment of `left`'s n-grams in `right`'s."""
    return containment(ngrams(left, n), ngrams(right, n))


def exact_overlap(left: str, right: str, *, n: int = 13) -> list[str]:
    """The n-grams the two texts literally share. This is the evidence."""
    shared = set(ngrams(left, n)) & set(ngrams(right, n))
    return sorted(shared)


def longest_common_substring(left: str, right: str, *, minimum: int = 40) -> str:
    """Longest shared run of characters, for showing a human what matched.

    Quadratic in the worst case, so it runs only on pairs that already crossed a
    threshold — never across a corpus.
    """
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return ""
    previous = [0] * (len(b) + 1)
    best = 0
    best_end = 0
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best, best_end = current[j], i
        previous = current
    return a[best_end - best : best_end] if best >= minimum else ""


# -- duplicate detection -------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """Two documents that look alike, and how strongly."""

    left: str
    right: str
    similarity: float
    verified: bool = False
    evidence: str = ""


def find_duplicates(
    sketches: Sequence[Sketch], *, threshold: float = 0.8, bands: int = 16
) -> list[Match]:
    """Near-duplicate pairs, using LSH banding to avoid the quadratic comparison.

    Banding trades recall for speed: pairs that agree in no band are never compared.
    With 128 permutations and 16 bands the cutover sits near 0.75, so a threshold far
    below that will silently miss pairs — the function raises rather than pretending.
    """
    if not sketches:
        return []
    width = len(sketches[0].signature)
    if bands <= 0 or width % bands != 0:
        raise ValueError(f"{bands} bands does not divide a {width}-permutation signature")

    rows = width // bands
    approximate_cutover = (1.0 / bands) ** (1.0 / rows)
    if threshold < approximate_cutover - 0.15:
        raise ValueError(
            f"threshold {threshold} is far below this banding's sensitivity "
            f"(~{approximate_cutover:.2f}); increase bands or raise the threshold, "
            "because pairs below it will be missed rather than reported"
        )

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, sketch in enumerate(sketches):
        for band in range(bands):
            key = (band, sketch.signature[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(index)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                candidates.add((min(left, right), max(left, right)))

    found = []
    for left, right in sorted(candidates):
        similarity = estimate_jaccard(sketches[left], sketches[right])
        if similarity >= threshold:
            found.append(
                Match(
                    left=sketches[left].identifier,
                    right=sketches[right].identifier,
                    similarity=round(similarity, 4),
                )
            )
    return sorted(found, key=lambda m: -m.similarity)


@dataclass(frozen=True)
class DuplicateCluster:
    """A group of mutually near-identical documents."""

    members: tuple[str, ...]
    representative: str = ""

    @property
    def redundant(self) -> tuple[str, ...]:
        """Everything but the representative — what deduplication would remove."""
        return tuple(m for m in self.members if m != self.representative)


def cluster_duplicates(matches: Iterable[Match]) -> list[DuplicateCluster]:
    """Transitively group matched pairs, via union-find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for match in matches:
        union(match.left, match.right)

    groups: dict[str, list[str]] = {}
    for member in parent:
        groups.setdefault(find(member), []).append(member)

    return [
        DuplicateCluster(members=tuple(sorted(members)), representative=sorted(members)[0])
        for members in groups.values()
        if len(members) > 1
    ]


def suspicious_pairs(
    evals: Sequence[Sketch], corpus: Sequence[Sketch], *, threshold: float = 0.6
) -> list[Match]:
    """Eval documents that look like corpus documents.

    Directional on purpose: the question is whether the eval leaked into training,
    not whether the two sets resemble each other.
    """
    found = []
    for item in evals:
        for document in corpus:
            similarity = estimate_jaccard(item, document)
            if similarity >= threshold:
                found.append(
                    Match(
                        left=item.identifier,
                        right=document.identifier,
                        similarity=round(similarity, 4),
                    )
                )
    return sorted(found, key=lambda m: -m.similarity)


# -- verdicts ------------------------------------------------------------------


class ContaminationVerdict(StrEnum):
    """Deliberately three-valued.

    Collapsing `SUSPECT` into either neighbour is the mistake: upward throws away a
    good benchmark on an estimate, downward ships a contaminated model.
    """

    CLEAN = "clean"
    SUSPECT = "suspect"
    CONTAMINATED = "contaminated"
    UNKNOWN = "unknown"  # nothing to compare; not the same as clean


@dataclass(frozen=True)
class ContaminationReport:
    """What was found, and whether it was verified."""

    eval_set: str
    verdict: ContaminationVerdict
    matches: tuple[Match, ...] = ()
    checked: int = 0
    threshold: float = 0.6
    method: str = "minhash"

    @property
    def verified_matches(self) -> tuple[Match, ...]:
        return tuple(m for m in self.matches if m.verified)

    def summary(self) -> str:
        lines = [
            f"{self.eval_set}: {self.verdict.value.upper()} "
            f"({len(self.matches)} match(es) over {self.checked} comparison(s), "
            f"{self.method} at {self.threshold})"
        ]
        if self.verdict is ContaminationVerdict.SUSPECT:
            lines.append(
                "  SUSPECT is an estimate, not a finding. Run `verify` on these pairs "
                "before discarding a benchmark result."
            )
        for match in self.matches[:5]:
            mark = "verified" if match.verified else "estimated"
            lines.append(f"  {match.left} ~ {match.right}: {match.similarity:.2f} ({mark})")
            if match.evidence:
                lines.append(f'    "{match.evidence[:100]}"')
        if self.verdict is ContaminationVerdict.UNKNOWN:
            lines.append("  nothing was compared; this is not a clean result")
        return "\n".join(lines)


def verify(
    matches: Iterable[Match], texts: Mapping[str, str], *, n: int = 13, minimum_run: int = 40
) -> list[Match]:
    """Turn estimated matches into verified ones by comparing actual text.

    A match survives verification only if a literal shared run exists. That evidence
    goes into the report so a human can look at it and decide, which is the only way
    a contamination finding ever gets acted on.
    """
    out = []
    for match in matches:
        left, right = texts.get(match.left), texts.get(match.right)
        if left is None or right is None:
            out.append(match)
            continue
        overlap = ngram_overlap(left, right, n=n)
        evidence = longest_common_substring(left, right, minimum=minimum_run)
        out.append(
            Match(
                left=match.left,
                right=match.right,
                similarity=round(overlap, 4),
                verified=bool(evidence),
                evidence=evidence,
            )
        )
    return out


def report(
    eval_set: str,
    evals: Sequence[Sketch],
    corpus: Sequence[Sketch],
    *,
    threshold: float = 0.6,
    texts: Mapping[str, str] | None = None,
) -> ContaminationReport:
    """Check an eval set against a corpus, verifying if the text is available."""
    if not evals or not corpus:
        return ContaminationReport(
            eval_set=eval_set,
            verdict=ContaminationVerdict.UNKNOWN,
            checked=0,
            threshold=threshold,
        )

    matches = suspicious_pairs(evals, corpus, threshold=threshold)
    checked = len(evals) * len(corpus)

    if texts:
        matches = verify(matches, texts)
        confirmed = [m for m in matches if m.verified]
        verdict = (
            ContaminationVerdict.CONTAMINATED
            if confirmed
            else (ContaminationVerdict.SUSPECT if matches else ContaminationVerdict.CLEAN)
        )
    else:
        verdict = ContaminationVerdict.SUSPECT if matches else ContaminationVerdict.CLEAN

    return ContaminationReport(
        eval_set=eval_set,
        verdict=verdict,
        matches=tuple(matches),
        checked=checked,
        threshold=threshold,
        method="minhash+ngram" if texts else "minhash",
    )
