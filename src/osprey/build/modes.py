"""The channel-finder mode registry.

Kept in the build-time kernel so every caller — the build pipeline, the
project materializer, the channel-finder services and the benchmark harness —
reads the same set of mode names without importing the ``cli`` build-profile
loader.
"""

from __future__ import annotations

#: The registry of channel-finder paradigms — the single source of truth for
#: which paradigm names exist. Everything else that enumerates paradigms
#: derives from this tuple: the ``enable_<paradigm>`` template flags in
#: :mod:`osprey.cli.templates.manager` are built by iterating it, and
#: :data:`osprey.registry.mcp.CHANNEL_FINDER_TOOLS_BY_PIPELINE` is checked
#: against it at import. Adding a paradigm here is the one edit that opens
#: the name up everywhere.
#:
#: Two things derive a *narrower* set than the whole tuple, each subtracting
#: ``graph`` because a graph store is a service rather than a database file
#: (``tests/build/test_modes.py`` pins both subtractions so the exclusion
#: stays deliberate):
#:
#: - :data:`osprey.services.channel_finder.benchmarks.runner.PARADIGM_CONFIG_KEYS`
#:   — every entry is a ``database.path`` config key, so a paradigm whose store
#:   is not a database file has no entry to name.
#: - :data:`osprey.cli.channel_finder_cmd.FILE_DATABASE_PARADIGMS`, behind the
#:   ``validate --pipeline`` ``click.Choice`` — ``validate`` opens a database
#:   file, so it has nothing to offer a paradigm without a file.
VALID_CHANNEL_FINDER_MODES: tuple[str, ...] = (
    "in_context",
    "hierarchical",
    "middle_layer",
    "graph",
)
