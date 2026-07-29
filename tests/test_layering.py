"""The package layering, enforced.

An organized tree stays organized for about a month unless something checks. These
tests are that something. They read the imports out of every module and assert the
three rules that keep the structure meaningful:

1. **Layers only import downward.** `core` knows nothing; `graph` knows `core`;
   `ai` may know all of them. An upward import is what turns a layered library into
   a ball of mutual dependencies, and it always arrives as one innocuous convenience.
2. **Directories stay navigable.** No more than twelve modules or ten
   subdirectories at any single level. Depth is free; breadth is not.
3. **Every package explains itself.** A package without a docstring is a package
   whose reason for existing was never written down, which is how the next person
   puts a file in the wrong one.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "fathom"

# Ascending. Each layer may import from itself and anything above it in this list.
LAYERS = [
    "core",
    "graph",
    "observe",
    "govern",
    "ai",
    "adapters",
    "ingest",
    "store",
    "report",
    "cli",
]

# `adapters` and `ingest` sit beside each other rather than above: an adapter reads
# and an ingest module writes, and neither is built on the other.
ALLOWED_EXTRA: dict[str, set[str]] = {
    "ingest": {"adapters"},
    "adapters": set(),
    "store": {"adapters"},
    "report": {"adapters", "ingest", "store"},
    "cli": set(LAYERS),
}

MAX_MODULES = 12
MAX_SUBDIRS = 10


def modules() -> list[pathlib.Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in str(p))


def layer_of(path: pathlib.Path) -> str | None:
    parts = path.relative_to(PACKAGE).parts
    return parts[0] if len(parts) > 1 and parts[0] in LAYERS else None


def imported_layers(path: pathlib.Path) -> set[str]:
    """Layers this module imports from, resolving relative imports against its own."""
    own = path.relative_to(PACKAGE).parts[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(own[: len(own) - node.level + 1])
                target = base + (node.module.split(".") if node.module else [])
            elif (node.module or "").startswith("fathom."):
                target = (node.module or "").split(".")[1:]
            else:
                continue
            if target and target[0] in LAYERS:
                found.add(target[0])
    return found


def allowed_for(layer: str) -> set[str]:
    below = set(LAYERS[: LAYERS.index(layer)])
    if layer in ALLOWED_EXTRA:
        below = {name for name in below if name not in set(LAYERS[LAYERS.index(layer) :])}
        below -= {"adapters", "ingest", "store", "report", "cli"}
        below |= ALLOWED_EXTRA[layer]
    return below | {layer}


@pytest.mark.parametrize(
    "path", [p for p in modules() if layer_of(p)], ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_layers_import_downward_only(path: pathlib.Path):
    layer = layer_of(path)
    assert layer is not None
    violations = imported_layers(path) - allowed_for(layer)
    assert not violations, (
        f"{path.relative_to(PACKAGE)} is in `{layer}` and imports from {sorted(violations)}, "
        f"which sits above it. Move the shared piece down, or the consumer up."
    )


def test_core_depends_on_nothing():
    """The vocabulary cannot depend on its speakers."""
    for path in modules():
        if layer_of(path) != "core":
            continue
        assert imported_layers(path) <= {"core"}, f"{path.relative_to(PACKAGE)} leaves core"


@pytest.mark.parametrize(
    "directory",
    sorted({p.parent for p in modules()}, key=str),
    ids=lambda p: str(p.relative_to(PACKAGE.parent)),
)
def test_directories_stay_navigable(directory: pathlib.Path):
    files = [p for p in directory.glob("*.py")]
    subdirs = [p for p in directory.iterdir() if p.is_dir() and p.name != "__pycache__"]
    assert len(files) <= MAX_MODULES, f"{directory.name}/ has {len(files)} modules; split it"
    assert len(subdirs) <= MAX_SUBDIRS, f"{directory.name}/ has {len(subdirs)} subpackages"


@pytest.mark.parametrize(
    "init",
    sorted(PACKAGE.rglob("__init__.py"), key=str),
    ids=lambda p: str(p.relative_to(PACKAGE.parent)),
)
def test_every_package_says_what_it_is_for(init: pathlib.Path):
    tree = ast.parse(init.read_text())
    doc = ast.get_docstring(tree)
    assert doc, f"{init.relative_to(PACKAGE)} has no docstring"
    assert len(doc.splitlines()) > 1, (
        f"{init.relative_to(PACKAGE)} has a one-line docstring; say what belongs here "
        "and what does not, so the next module lands in the right package"
    )


@pytest.mark.parametrize("path", modules(), ids=lambda p: str(p.relative_to(PACKAGE.parent)))
def test_every_module_is_documented(path: pathlib.Path):
    assert ast.get_docstring(ast.parse(path.read_text())), f"{path.name} has no module docstring"


def test_the_tests_mirror_the_source_tree():
    """A test directory per package, so a new module has an obvious place to be tested."""
    packages = {p.parent.name for p in PACKAGE.rglob("__init__.py") if p.parent != PACKAGE}
    tested = {p.name for p in (ROOT / "tests").iterdir() if p.is_dir() and p.name != "__pycache__"}
    # Leaf packages under adapters/ and core/ are covered by their parent's directory.
    missing = {"graph", "observe", "govern", "ai", "adapters", "ingest", "report", "cli"} - tested
    assert not missing, f"no test directory for {sorted(missing)}"
    assert packages  # the walk found something, so an empty `tested` would be a real failure
