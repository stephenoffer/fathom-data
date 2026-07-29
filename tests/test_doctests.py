"""Every `Example:` block in the package, executed.

The docstring examples are the first thing a new user copies, and a copied example
that raises is worse than no example at all — it costs the reader their confidence
in the rest of the documentation before they have written a line of their own.

`tests/test_docs.py` already checks that documented commands and config keys exist.
This does the same job one level down, for the code samples inside docstrings: every
`>>>` in `src/fathom` runs here, and its printed output must match.

Modules that need an optional dependency are skipped rather than failed, because the
base install deliberately does not carry them.
"""

from __future__ import annotations

import doctest
import importlib
import pathlib
import pkgutil

import pytest

import fathom

PACKAGE = pathlib.Path(fathom.__file__).parent


def module_names() -> list[str]:
    """Every importable module in the package, deepest last."""
    found = [fathom.__name__]
    for info in pkgutil.walk_packages([str(PACKAGE)], prefix="fathom."):
        found.append(info.name)
    return sorted(found)


@pytest.mark.parametrize("name", module_names())
def test_docstring_examples_run(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except ImportError as exc:  # an optional adapter dependency is absent
        pytest.skip(f"{name} needs an optional dependency: {exc}")

    results = doctest.testmod(
        module,
        verbose=False,
        report=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    relative = name.removeprefix("fathom.").replace(".", "/")
    assert not results.failed, (
        f"{results.failed} of {results.attempted} docstring example(s) in {name} "
        "produced output that does not match. Run "
        f"`python -m pytest --doctest-modules {PACKAGE / relative}.py` to see the diff."
    )
