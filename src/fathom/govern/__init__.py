"""Obligations that travel with the data rather than with the code.

    policy            what a column means, propagated along edges, enforced at sinks
    reidentification  columns that identify nobody alone and everybody together
    erasure           where a subject's data physically lives, and how to destroy it
    licenses          what the data may be used for, combined most-restrictive-first
    consent           which purposes its subjects agreed to, and where it may be stored
    contracts         what one team promised another about a dataset, and who a breach hurts
    replicas          copies no adapter can see, declared so a proof can name them

The records built on all four — Article 30 entries, subject access responses, model
training summaries — live in `report.compliance`, because they read the AI graph too
and this package must stay below it.

One thing distinguishes this package from the rest of the library and is worth
stating once: **these combine restrictively.** Dirtiness and labels take the union
as they flow downstream — anything an input taints, the output is tainted by.
Licences, purposes, and residency take the intersection: the output may be used only
where every input may. Getting that backwards fails open, which is why it is a
package boundary and not a convention.
"""

from . import (
    access,
    approvals,
    consent,
    contracts,
    erasure,
    licenses,
    policy,
    reidentification,
    replicas,
)

__all__ = [
    "access",
    "approvals",
    "consent",
    "contracts",
    "erasure",
    "licenses",
    "policy",
    "reidentification",
    "replicas",
]
