"""Channel-finder paradigm/tier defaults.

The single source of truth for the tier a build lands on for a given
channel-finder paradigm, and the tier/paradigm conflict rule. Kept in the
build-time kernel so every caller — the build pipeline, the project
materializer, and the benchmark harness — derives the tier the same way
without importing the full ``cli`` build-profile loader.
"""

from __future__ import annotations

#: The registry of channel-finder paradigms — the single source of truth for
#: which paradigm names exist. Everything else that enumerates paradigms
#: derives from this tuple: :data:`osprey.cli.templates.scaffolding._ALL_PARADIGMS`
#: is an alias of it, the ``enable_<paradigm>`` template flags in
#: :mod:`osprey.cli.templates.manager` are built by iterating it, and
#: :data:`osprey.registry.mcp.CHANNEL_FINDER_TOOLS_BY_PIPELINE` is checked
#: against it at import. Adding a paradigm here is the one edit that opens
#: the name up everywhere.
#:
#: Two things derive a *narrower* set than the whole tuple, each subtracting
#: ``graph`` because a graph store is a service rather than a database file
#: (``tests/build/test_mode_registry_single_source.py`` pins both subtractions
#: so the exclusion stays deliberate):
#:
#: - :data:`osprey.services.channel_finder.benchmarks.runner.PARADIGM_CONFIG_KEYS`
#:   — every entry is a ``database.path`` config key, so a paradigm whose store
#:   is not a database file has no entry to name.
#: - :data:`osprey.cli.channel_finder_cmd.FILE_DATABASE_PARADIGMS`, behind the
#:   two ``click.Choice`` lists in that module — ``validate`` opens a database
#:   file and ``generate`` writes one, so neither has anything to offer a
#:   paradigm without a file.
VALID_CHANNEL_FINDER_MODES: tuple[str, ...] = (
    "in_context",
    "hierarchical",
    "middle_layer",
    "graph",
)


def default_tier_for_mode(channel_finder_mode: str | None) -> int:
    """Paradigm-aware default tier: ``in_context`` → 1, otherwise → 3.

    The single source of truth for the tier a build lands on when no explicit
    ``tier`` is pinned. Every code path that needs a concrete tier from a
    channel-finder paradigm (the build pipeline, the project materializer)
    MUST derive it here so the rule cannot drift between callers.

    ``graph`` inherits the tier-3 default. It ships no tiered database of its
    own, so for graph the tier selects only the tier-3 benchmark query set —
    which is why :func:`tier_mode_conflict` rejects pinning that tier
    explicitly while this function still derives it.
    """
    return 1 if channel_finder_mode == "in_context" else 3


def tier_mode_conflict(tier: int | None, channel_finder_mode: str | None) -> str | None:
    """Return a rule-naming error message if ``tier``/paradigm conflict, else None.

    Two rules, checked in this order:

    1. ``graph`` has no tiered artifacts at all — its store is a seeded graph
       service, not a database file, so no tier selects anything for it and
       *any* explicit ``tier`` is a conflict. This arm sits ahead of the tier-1
       rule so a ``tier: 1`` graph profile is told to drop ``tier`` rather than
       to switch to ``in_context``, which would be the wrong fix.
    2. Tier 1 ships only the ``in_context`` paradigm DB. Pairing an explicit
       ``tier: 1`` with a ``hierarchical``/``middle_layer`` paradigm would
       otherwise fail deep in :func:`materialize_tier_artifacts` as an opaque
       ``FileNotFoundError`` (``tier1/<paradigm>.json`` does not exist).

    Callers surface this message so the failure is legible on every
    configuration path. Omitting ``tier`` is always accepted: callers that need
    a concrete tier derive it from :func:`default_tier_for_mode`, and callers
    that write a profile drop the derived tier again wherever this function
    reports a conflict with it (``tests/e2e/sdk_helpers.py`` does exactly that
    for ``graph``).
    """
    if tier is not None and channel_finder_mode == "graph":
        return f"channel_finder_mode: graph has no tiered artifacts; omit tier (got tier: {tier})"
    if tier == 1 and channel_finder_mode not in (None, "in_context"):
        return (
            "tier 1 requires channel_finder_mode: in_context "
            f"(got channel_finder_mode: {channel_finder_mode!r})"
        )
    return None
