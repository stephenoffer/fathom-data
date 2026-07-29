"""What the data actually looks like, and whether it moved.

    profile       footer-only statistical fingerprints, and drift between two of them
    schema        structural change between two profiles, which breaks queries outright
    quality       expectations over a profile, generated from what was observed
    seasonal      the same, bucketed by a cycle, for data that knows Tuesday from Sunday
    completeness  whether the partitions that should exist actually do
    joins         join keys, and the two ways one quietly ruins a table
    regression    did the rewrite change the numbers, and by how much
    usage         who reads a dataset, and what follows from nobody reading it
    shadow        the planner graded against a full rebuild, publishing its own misses
    incidents     findings grouped into one owned incident, and the postmortem after
    freshness     transitive age — a table is only as fresh as its oldest input

All are cheap by construction. A profile reads Parquet footers rather than data
pages, an expectation is checked against a profile rather than the table, and both
freshness and completeness are answered from metadata alone. That is what makes
running them on every dirty partition affordable instead of nightly and partial.

`completeness` is the odd one out and the reason it exists: everything else reads
data that arrived and asks whether it looks right. Only completeness can see the
partition that never arrived, which has no profile to drift and no rows to fail an
expectation.
"""

from . import (
    completeness,
    freshness,
    incidents,
    joins,
    profile,
    quality,
    regression,
    schema,
    seasonal,
    shadow,
    usage,
)

__all__ = [
    "completeness",
    "freshness",
    "incidents",
    "joins",
    "profile",
    "quality",
    "regression",
    "schema",
    "seasonal",
    "shadow",
    "usage",
]
