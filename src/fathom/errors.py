"""Error types.

Two rules, both learned from tools that get this wrong:

**Never swallow an access failure as "nothing found."** A missing prefix and an
expired credential look identical from the call site, and reporting the second as
the first turns a five-minute fix into an afternoon of confusion — the user sees
"no lineage extracted" and starts debugging their SQL.

**Every message names the thing and the next action.** A stack trace ending in
`NoCredentialsError` is true and useless. `cannot read s3://lake/events: no AWS
credentials found (set AWS_PROFILE, or pass storage_options)` is actionable.
"""

from __future__ import annotations

__all__ = [
    "AdapterUnavailable",
    "ConfigError",
    "FathomError",
    "PlanError",
    "StorageAccessError",
]


class FathomError(Exception):
    """Base for everything this library raises deliberately."""


class StorageAccessError(FathomError):
    """A filesystem or object store could not be read.

    Distinct from `FileNotFoundError`, which callers legitimately treat as "absent".
    This means we could not tell, which is never safe to interpret as empty.
    """

    def __init__(self, uri: str, reason: str, hint: str = "") -> None:
        self.uri = uri
        self.reason = reason
        self.hint = hint
        message = f"cannot read {uri}: {reason}"
        if hint:
            message += f"\n  {hint}"
        super().__init__(message)


class AdapterUnavailable(FathomError):
    """An adapter needs an optional dependency that is not installed."""

    def __init__(self, adapter: str, package: str, extra: str) -> None:
        super().__init__(
            f"the {adapter} adapter needs {package}: pip install 'fathom-data[{extra}]'"
        )


class ConfigError(FathomError):
    """A project configuration file is malformed or internally inconsistent."""


class PlanError(FathomError):
    """A plan could not be built from the graph and seeds given."""


# Provider exception names mapped to advice. Matching on class name rather than
# importing every SDK keeps this dependency-free.
_HINTS: dict[str, str] = {
    "NoCredentialsError": (
        "no credentials found. Set AWS_PROFILE / AWS_ACCESS_KEY_ID, or pass "
        "storage_options={'key': ..., 'secret': ...}"
    ),
    "PartialCredentialsError": "credentials are incomplete; check the key and secret pair",
    "ClientError": (
        "the storage service rejected the request. Common causes: wrong region, "
        "a bucket policy denying ListBucket, or requester-pays not enabled"
    ),
    "AccessDenied": "access denied; the principal needs list and read on this prefix",
    "DefaultCredentialsError": (
        "no Google credentials found. Run `gcloud auth application-default login`, "
        "or set GOOGLE_APPLICATION_CREDENTIALS"
    ),
    "ClientAuthenticationError": (
        "Azure authentication failed; check the account key, SAS token, or managed identity"
    ),
    "EndpointConnectionError": "could not reach the endpoint; check the URL and network access",
}


def hint_for(exc: BaseException) -> str:
    """Advice for a provider exception, or an empty string when we have none."""
    for cls in type(exc).__mro__:
        hint = _HINTS.get(cls.__name__)
        if hint:
            return hint
    return ""
