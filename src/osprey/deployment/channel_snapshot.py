"""Build-time channel snapshot decision for web channel suggestions.

Web panels offer a typeahead over the control-system addresses a project knows
about. A browser can reach neither the project's Channel Finder database nor
the Turtle corpus of the graph paradigm, so the build emits a static snapshot
of those addresses next to the rendered service files instead.

This module answers the single question the build needs: *should a snapshot be
written, and what goes in it?* :func:`compute_channel_snapshot` returns one
:class:`SnapshotDecision`; every consumer reads that object rather than
re-deriving the predicate, so the compose fragment, the mount, and the file on
disk can never disagree about whether a snapshot exists.

Membership is not decided here. Which channels the facility has is
:func:`~osprey.channel_roster.registered_channels`' answer, resolved and read
once per build and shared with every other consumer of that same question --
so a typeahead can no longer suggest a different set of channels than the one
the rest of the deployment was built from. What stays here is presentation: the
``web.channel_suggestions`` feature switch, the size guard that keeps the
typeahead responsive, and the rule that an empty list is not worth writing.
Those are properties of a browser widget rather than of the facility, so the
roster never sees them.

The decision fails soft in almost every direction. A project that configures no
channel source at all, one whose source is empty, unreadable, or too large to
be useful as a typeahead, gets no snapshot at all. The build itself is never
blocked, because a missing autocomplete list is a degraded panel, not a broken
deployment. The one exception is a ``pipeline_mode`` naming a paradigm that
does not exist: that is a configuration mistake rather than a degraded panel,
so it stops the build.

Path preconditions are the roster's (:mod:`osprey.channel_roster.sources`): a
relative ``database.path`` is resolved against the process working directory,
which the build sets to the project root before generating compose files, and a
relative ``services.graphdb.ttl_path`` is render-relative instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from osprey.channel_roster import registered_channels
from osprey.utils.logger import get_logger

logger = get_logger("deployment.channel_snapshot")

#: Upper bound on snapshot size when the project does not set one. Well past any
#: real facility catalog, and small enough that the browser-side typeahead stays
#: responsive on the largest one anybody has pointed at OSPREY so far.
DEFAULT_MAX_CHANNELS = 50000

#: Named in the skip message so a project that trips the guard knows what to raise.
MAX_CHANNELS_CONFIG_KEY = "web.channel_suggestions.max_channels"

#: Named in the skip message when the feature is switched off explicitly.
ENABLED_CONFIG_KEY = "web.channel_suggestions.enabled"


@dataclass(frozen=True)
class SnapshotDecision:
    """Whether to emit a channel snapshot, and what it contains.

    Attributes:
        emit: True when a snapshot file should be written.
        channels: Sorted, deduplicated control-system addresses. Empty unless
            ``emit`` is True — a decision not to emit carries no payload.
        count: How many distinct addresses the source yielded. Reported even
            when ``emit`` is False, so a caller can say why nothing was written.
        source_path: Resolved path of the database or corpus the addresses came
            from, or None when no source was configured.
    """

    emit: bool
    channels: list[str] = field(default_factory=list)
    count: int = 0
    source_path: Path | None = None


def _suggestions_section(config: dict) -> dict:
    """Read the ``web.channel_suggestions`` block, tolerating any shape.

    An absent block is not an opt-out: the feature is on by default, and the
    keys are only written explicitly by newer generated configs.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The block as a dict, or an empty dict when it is absent or not a mapping.
    """
    web = config.get("web") or {}
    if not isinstance(web, dict):
        return {}
    section = web.get("channel_suggestions") or {}
    return section if isinstance(section, dict) else {}


def _max_channels(section: dict) -> int:
    """Resolve the size guard, falling back to the default on an unusable value."""
    raw = section.get("max_channels", DEFAULT_MAX_CHANNELS)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring unusable {MAX_CHANNELS_CONFIG_KEY} value {raw!r}; "
            f"using the default of {DEFAULT_MAX_CHANNELS}."
        )
        return DEFAULT_MAX_CHANNELS


def compute_channel_snapshot(config: dict) -> SnapshotDecision:
    """Decide whether the build should emit a channel snapshot, and with what.

    A snapshot is emitted when ``web.channel_suggestions.enabled`` is not
    switched off, :func:`~osprey.channel_roster.registered_channels` returns a
    roster, and that roster holds at least one and at most
    ``web.channel_suggestions.max_channels`` distinct addresses.

    The addresses — not the human-facing channel names — are what a panel writes
    into a control-system request, so those are what the snapshot carries, and
    they are the roster's membership verbatim: whichever source the project's
    paradigm names, read once for the whole build.

    A roster that could not be built at all — no source configured, graph mode
    naming no corpus, a source that cannot be read — degrades to no snapshot.
    The reason is logged at debug here, because a build that reads no channels
    is a degraded panel; a source that resolved and then failed is warned about
    by the roster reader that failed to read it.

    Args:
        config: Full project configuration dictionary, as the build already
            holds it.

    Returns:
        The decision. An unreadable or malformed source yields ``emit=False``
        rather than raising.

    Raises:
        PipelineModeError: If ``channel_finder.pipeline_mode`` names a paradigm
            that does not exist.
    """
    section = _suggestions_section(config)

    if not section.get("enabled", True):
        logger.debug(f"{ENABLED_CONFIG_KEY} is off; not emitting a channel snapshot.")
        return SnapshotDecision(emit=False)

    roster = registered_channels(config)

    if roster.source is None:
        # No roster at all. Membership is absent rather than empty, and the
        # absence carries the sentence that says which of the two it is. Only
        # a source that resolved and then failed to read names a path; one
        # that was never configured has none to name.
        if roster.absence is None:
            return SnapshotDecision(emit=False)
        logger.debug(f"{roster.absence.message()} Not emitting a channel snapshot.")
        return SnapshotDecision(emit=False, source_path=roster.absence.path)

    source_path = roster.source.path
    channels = sorted(set(roster.addresses))
    count = len(channels)

    if count == 0:
        # An empty snapshot is not a smaller suggestion list, it is a typeahead
        # that never suggests anything — so there is nothing worth writing.
        logger.debug(
            f"The channel source at {source_path} holds no channels; "
            "not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False, source_path=source_path)

    max_channels = _max_channels(section)
    if count > max_channels:
        logger.warning(
            f"The channel source at {source_path} holds {count} channels, above the "
            f"{MAX_CHANNELS_CONFIG_KEY} limit of {max_channels}; not emitting a channel "
            "snapshot. Raise that limit to include it."
        )
        return SnapshotDecision(emit=False, count=count, source_path=source_path)

    logger.debug(f"Channel snapshot: {count} channels from {source_path}.")
    return SnapshotDecision(emit=True, channels=channels, count=count, source_path=source_path)
