"""Error types, and the two rules they exist to enforce.

Both rules were learned from tools that get them wrong, so both get a test:

- a missing prefix and an expired credential look identical from the call site, and
  reporting the second as the first turns a five-minute fix into an afternoon
- a stack trace ending in `NoCredentialsError` is true and useless
"""

from __future__ import annotations

import pytest

from fathom.core.errors import (
    AdapterUnavailable,
    ConfigError,
    FathomError,
    PlanError,
    StorageAccessError,
    hint_for,
)


@pytest.mark.parametrize(
    "exc",
    [
        StorageAccessError("s3://b/k", "denied"),
        AdapterUnavailable("iceberg", "pyiceberg", "iceberg"),
        ConfigError("bad"),
        PlanError("no"),
    ],
)
def test_everything_deliberate_shares_one_base(exc):
    """One `except FathomError` has to catch everything this library raises on purpose."""
    assert isinstance(exc, FathomError)


def test_storage_errors_name_the_thing_and_the_next_action():
    error = StorageAccessError("s3://lake/events", "no credentials", "set AWS_PROFILE")
    message = str(error)

    assert "s3://lake/events" in message
    assert "no credentials" in message
    assert "set AWS_PROFILE" in message
    assert error.uri == "s3://lake/events"


def test_a_storage_error_without_a_hint_still_names_the_uri():
    assert str(StorageAccessError("gs://b/k", "timeout")) == "cannot read gs://b/k: timeout"


def test_storage_access_is_not_file_not_found():
    """Callers legitimately treat absence as empty; they must not treat a denial as empty."""
    assert not issubclass(StorageAccessError, FileNotFoundError)


def test_missing_extra_names_the_install_command():
    message = str(AdapterUnavailable("iceberg", "pyiceberg", "iceberg"))
    assert "pyiceberg" in message
    assert "pip install 'fathom-data[iceberg]'" in message


class NoCredentialsError(Exception):
    """Stands in for botocore's, which is not a dependency here."""


class SubclassOfProviderError(NoCredentialsError):
    pass


def test_hints_are_matched_on_class_name_not_by_importing_every_sdk():
    hint = hint_for(NoCredentialsError())
    assert "AWS_PROFILE" in hint or "credentials" in hint


def test_hints_walk_the_mro():
    """Provider SDKs subclass their own errors; the advice still has to apply."""
    assert hint_for(SubclassOfProviderError()) == hint_for(NoCredentialsError())


def test_an_unrecognized_exception_gets_no_invented_advice():
    assert hint_for(ValueError("something else entirely")) == ""


@pytest.mark.parametrize(
    "name",
    ["NoCredentialsError", "ClientError", "AccessDenied", "DefaultCredentialsError"],
)
def test_the_common_provider_failures_are_covered(name):
    exc = type(name, (Exception,), {})()
    assert hint_for(exc), f"{name} produces no advice"
