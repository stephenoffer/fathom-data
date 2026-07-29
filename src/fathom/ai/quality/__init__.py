"""Checking AI assets by their content rather than only by their lineage.

`ai/evals.py` answers contamination as reachability, which is correct, cheap, and
blind to the case that actually happens: the eval text was scraped into the corpus
with no edge between them at all. Lineage cannot see a copy nobody recorded.

This package compares the bytes. It is deliberately separate from `evals` because
the two answer the same question by different means and have different costs — a
graph traversal is free and a corpus scan is not, so the traversal runs first and
this runs when the traversal says nothing and the result still looks too good.

- **`contamination`** — MinHash near-duplicate detection, exact n-gram overlap, and
  a three-valued verdict that keeps an estimate distinct from a finding.
"""

from .contamination import (
    ContaminationReport,
    ContaminationVerdict,
    DuplicateCluster,
    Match,
    Sketch,
    cluster_duplicates,
    containment,
    estimate_jaccard,
    exact_overlap,
    find_duplicates,
    jaccard,
    longest_common_substring,
    minhash,
    ngram_overlap,
    ngrams,
    normalize_text,
    report,
    shingles,
    sketch_of,
    suspicious_pairs,
    verify,
)
from .safety import (
    Finding,
    FindingState,
    Harm,
    ProbeResult,
    RefusalReport,
    RegressionReport,
    SafetyProbe,
    SafetySuite,
    Severity,
    close,
    coverage_by_harm,
    grade,
    guard,
    refusal_report,
    regressions,
    suite_edges,
    summarize_findings,
    unguarded,
)

__all__ = [
    "ContaminationReport",
    "ContaminationVerdict",
    "DuplicateCluster",
    "Finding",
    "FindingState",
    "Harm",
    "Match",
    "ProbeResult",
    "RefusalReport",
    "RegressionReport",
    "SafetyProbe",
    "SafetySuite",
    "Severity",
    "Sketch",
    "close",
    "cluster_duplicates",
    "containment",
    "coverage_by_harm",
    "estimate_jaccard",
    "exact_overlap",
    "find_duplicates",
    "grade",
    "guard",
    "jaccard",
    "longest_common_substring",
    "minhash",
    "ngram_overlap",
    "ngrams",
    "normalize_text",
    "refusal_report",
    "regressions",
    "report",
    "shingles",
    "sketch_of",
    "suite_edges",
    "summarize_findings",
    "suspicious_pairs",
    "unguarded",
    "verify",
]
