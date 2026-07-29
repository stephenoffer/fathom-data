"""The command line, and the project file behind it.

    main      the four verbs plus `ingest`, `changed`, `profile`, and `shadow`
    config    parsing and validating `fathom.yml`
    project   turning that config into graphs, adapters, and specs

Kept in its own package because a library that imports click to be imported is a
library people vendor around.
"""

from .main import main

__all__ = ["main"]
