"""The intermediate representation everything else speaks.

Nothing in this package imports from anywhere else in `fathom`. That is the whole
point of it being a package: identity, time grains, the partition lattice, and the
error types are the vocabulary, and a vocabulary that depends on its speakers is not
a vocabulary.

    types       datasets, columns, partition specs, key predicates, capabilities
    grains      hour/day/month/year arithmetic, always rounding outward
    partitions  the mapping lattice — compose, join, apply, and the soundness rule
    ids         OpenLineage-compatible identity normalization
    paths       object paths to partition keys
    codec       JSON round-tripping that preserves partition value types
    errors      the exceptions, each carrying the next action
    util/       digests, Markdown, clocks, text measurement
"""

from . import codec, errors, grains, ids, partitions, paths, types, util

__all__ = ["codec", "errors", "grains", "ids", "partitions", "paths", "types", "util"]
