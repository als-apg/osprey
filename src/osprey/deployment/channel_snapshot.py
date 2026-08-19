"""Build-time channel snapshot decision for web channel suggestions.

Web panels offer a typeahead over the control-system addresses a project knows
about. The addresses come from the project's Channel Finder database, which
lives on the build host and is not reachable from a browser, so the build emits
a static snapshot of them next to the rendered service files instead.

This module answers the single question the build needs: *should a snapshot be
written, and what goes in it?* :func:`compute_channel_snapshot` returns one
:class:`SnapshotDecision`; every consumer reads that object rather than
re-deriving the predicate, so the compose fragment, the mount, and the file on
disk can never disagree about whether a snapshot exists.

The decision fails soft in every direction. A project with no Channel Finder
database, an empty one, one too large to be useful as a typeahead, or one that
cannot be read at all simply gets no snapshot — the build itself is never
blocked, because a missing autocomplete list is a degraded panel, not a broken
deployment.

Working-directory precondition: a relative ``database.path`` is resolved against
the process working directory, which the build sets to the project root before
generating compose files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        count: How many distinct addresses the database yielded. Reported even
            when ``emit`` is False, so a caller can say why nothing was written.
        source_path: Resolved path of the database the addresses came from, or
            None when no database was configured or reachable.
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


def _load_channel_records(pipeline_type: str, db_config: dict, db_path: Path) -> list[dict]:
    """Load a channel database and return its channel records.

    Constructing any of the database classes loads the file, so a bad path or
    malformed content surfaces here as an exception for the caller to absorb.
    The database classes are imported lazily to keep the deployment import graph
    free of the Channel Finder service.

    Args:
        pipeline_type: Pipeline name from ``detect_pipeline_config``.
        db_config: The pipeline's ``database`` block (``path``, and ``type`` for
            the in-context pipeline).
        db_path: Resolved path to the database file.

    Returns:
        Channel records, each carrying at least a ``channel`` key and usually an
        ``address``.

    Raises:
        ValueError: If ``pipeline_type`` is not a known pipeline.
    """
    database: Any
    if pipeline_type == "in_context":
        from osprey.services.channel_finder.databases import (
            FlatChannelDatabase,
            TemplateChannelDatabase,
        )

        db_type = db_config.get("type", "template")
        if db_type == "flat":
            database = FlatChannelDatabase(str(db_path))
        else:
            database = TemplateChannelDatabase(str(db_path))
    elif pipeline_type == "hierarchical":
        from osprey.services.channel_finder.databases import HierarchicalChannelDatabase

        database = HierarchicalChannelDatabase(str(db_path))
    elif pipeline_type == "middle_layer":
        from osprey.services.channel_finder.databases import MiddleLayerDatabase

        database = MiddleLayerDatabase(str(db_path))
    else:
        raise ValueError(f"Unknown channel finder pipeline '{pipeline_type}'")

    records: list[dict] = database.get_all_channels()
    return records


def compute_channel_snapshot(config: dict) -> SnapshotDecision:
    """Decide whether the build should emit a channel snapshot, and with what.

    A snapshot is emitted when the project configures a Channel Finder database,
    ``web.channel_suggestions.enabled`` is not switched off, the database is
    readable, and it holds at least one and at most
    ``web.channel_suggestions.max_channels`` distinct addresses.

    The addresses — not the human-facing channel names — are what a panel writes
    into a control-system request, so those are what the snapshot carries. A
    database whose records name both keeps its ``address``; one that names only
    ``channel`` (the hierarchical and middle-layer pipelines synthesize an
    address from the channel name) contributes that instead.

    Args:
        config: Full project configuration dictionary, as the build already
            holds it.

    Returns:
        The decision. Never raises: an unreadable or malformed database is
        logged as a warning and yields ``emit=False``.
    """
    section = _suggestions_section(config)

    if not section.get("enabled", True):
        logger.debug(f"{ENABLED_CONFIG_KEY} is off; not emitting a channel snapshot.")
        return SnapshotDecision(emit=False)

    from osprey.services.channel_finder.utils.detection import detect_pipeline_config

    pipeline_type, db_config = detect_pipeline_config(config)
    if not pipeline_type or not db_config:
        logger.debug("No channel finder database is configured; not emitting a channel snapshot.")
        return SnapshotDecision(emit=False)

    db_path = Path(db_config["path"])
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    try:
        records = _load_channel_records(pipeline_type, db_config, db_path)
        addresses = {record.get("address", record["channel"]) for record in records}
        channels = sorted(address for address in addresses if address)
    except Exception as e:
        logger.warning(
            f"Could not read the channel database at {db_path} ({e}); "
            "not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False, source_path=db_path)

    count = len(channels)

    if count == 0:
        # An empty snapshot is not a smaller suggestion list, it is a typeahead
        # that never suggests anything — so there is nothing worth writing.
        logger.debug(
            f"The channel database at {db_path} holds no channels; not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False, source_path=db_path)

    max_channels = _max_channels(section)
    if count > max_channels:
        logger.warning(
            f"The channel database at {db_path} holds {count} channels, above the "
            f"{MAX_CHANNELS_CONFIG_KEY} limit of {max_channels}; not emitting a channel "
            "snapshot. Raise that limit to include it."
        )
        return SnapshotDecision(emit=False, count=count, source_path=db_path)

    logger.debug(f"Channel snapshot: {count} channels from {db_path}.")
    return SnapshotDecision(emit=True, channels=channels, count=count, source_path=db_path)
