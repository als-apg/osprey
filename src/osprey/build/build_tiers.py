"""Channel-finder paradigm/tier defaults.

The single source of truth for the tier a build lands on for a given
channel-finder paradigm, and the tier/paradigm conflict rule. Kept in the
build-time kernel so every caller — the build pipeline, the project
materializer, and the benchmark harness — derives the tier the same way
without importing the full ``cli`` build-profile loader.
"""

from __future__ import annotations

VALID_CHANNEL_FINDER_MODES: tuple[str, ...] = (
    "in_context",
    "hierarchical",
    "middle_layer",
)


def default_tier_for_mode(channel_finder_mode: str | None) -> int:
    """Paradigm-aware default tier: ``in_context`` → 1, otherwise → 3.

    The single source of truth for the tier a build lands on when no explicit
    ``tier`` is pinned. Every code path that needs a concrete tier from a
    channel-finder paradigm (the build pipeline, the project materializer)
    MUST derive it here so the rule cannot drift between callers.
    """
    return 1 if channel_finder_mode == "in_context" else 3


def tier_mode_conflict(tier: int | None, channel_finder_mode: str | None) -> str | None:
    """Return a rule-naming error message if ``tier``/paradigm conflict, else None.

    Tier 1 ships only the ``in_context`` paradigm DB. Pairing an explicit
    ``tier: 1`` with a ``hierarchical``/``middle_layer`` paradigm would
    otherwise fail deep in :func:`materialize_tier_artifacts` as an opaque
    ``FileNotFoundError`` (``tier1/<paradigm>.json`` does not exist). Callers
    surface this message so the failure is legible on every configuration path.
    """
    if tier == 1 and channel_finder_mode not in (None, "in_context"):
        return (
            "tier 1 requires channel_finder_mode: in_context "
            f"(got channel_finder_mode: {channel_finder_mode!r})"
        )
    return None
