"""Who may act, and what they actually did.

`govern/` answers what obligations travel with data — licence, consent, residency.
This package answers the two questions a security review asks before any of that
matters: who is allowed to see this, and can you show me what happened.

- **`rbac`** — roles, scoped grants, sensitivity ceilings, and column-level
  visibility. Deny by default, and governance artefacts deny harder: a broad read on
  a dataset does not confer the right to read an erasure proof, because a proof
  names a person.
- **`tenancy`** — ownership and sharing across tenants in one deployment. Lineage
  crossing a boundary is usually correct, so crossings are classified three ways
  rather than two: a check that reports every shared dependency as a violation is a
  check that gets switched off.
- **`keys`** — the key registry crypto-shredding needs to be more than a suggestion.
  Destroying a key covering two subjects is refused rather than warned about.
- **`audit`** — an append-only, hash-chained record of actions and their actors,
  including the denied ones. Tamper-evident rather than tamper-proof, and `verify`
  reports *where* a chain broke, because a log known to be altered somewhere is
  nearly useless while one altered at a known point leaves everything before it
  trustworthy.

Neither module authenticates anyone. Establishing who a principal is belongs to
whatever already does that; these decide what they may do once you know.
"""

from .audit import (
    GENESIS,
    AuditEntry,
    AuditLog,
    AuditOutcome,
    ChainBreak,
    digest_of,
    entries_for,
    entries_since,
    record,
    replay,
    summarize,
    verify,
    who_touched,
)
from .keys import (
    Key,
    KeyRegistry,
    KeyState,
    ShredProof,
    ShredRefused,
    covered_by,
    destroy,
    destroy_for_subject,
    keys_for_subject,
    register,
    rotate,
    shred_proof,
    subjects_covered,
    verify_destroyed,
)
from .rbac import (
    GOVERNANCE_ACTIONS,
    AccessDenied,
    AccessPolicy,
    Action,
    Decision,
    Effect,
    Grant,
    Principal,
    Role,
    Sensitivity,
    can,
    columns_visible_to,
    explain,
    grant_for,
    is_governance_action,
    mask_profile,
    merge_roles,
    principals_who_can,
    redact,
    require,
    role_with,
    sensitivity_of,
)
from .tenancy import (
    Crossing,
    CrossingKind,
    IsolationReport,
    Tenant,
    TenantMap,
    classify_crossings,
    crossings,
    datasets_of,
    is_shared_with,
    leaks,
    owner_of,
    scope_graph,
    share,
    shared_with,
    tenant_summary,
    unowned,
)

__all__ = [
    "GENESIS",
    "GOVERNANCE_ACTIONS",
    "AccessDenied",
    "AccessPolicy",
    "Action",
    "AuditEntry",
    "AuditLog",
    "AuditOutcome",
    "ChainBreak",
    "Decision",
    "Effect",
    "Grant",
    "Principal",
    "Role",
    "Sensitivity",
    "Crossing",
    "CrossingKind",
    "IsolationReport",
    "Key",
    "KeyRegistry",
    "KeyState",
    "ShredProof",
    "ShredRefused",
    "Tenant",
    "TenantMap",
    "classify_crossings",
    "covered_by",
    "crossings",
    "datasets_of",
    "destroy",
    "destroy_for_subject",
    "is_shared_with",
    "keys_for_subject",
    "leaks",
    "owner_of",
    "register",
    "rotate",
    "scope_graph",
    "share",
    "shared_with",
    "shred_proof",
    "subjects_covered",
    "tenant_summary",
    "unowned",
    "verify_destroyed",
    "can",
    "columns_visible_to",
    "digest_of",
    "entries_for",
    "entries_since",
    "explain",
    "grant_for",
    "is_governance_action",
    "mask_profile",
    "merge_roles",
    "principals_who_can",
    "record",
    "redact",
    "replay",
    "require",
    "role_with",
    "sensitivity_of",
    "summarize",
    "verify",
    "who_touched",
]
