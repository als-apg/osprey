"""Enumerate a facility's channels from a Channel Finder paradigm database.

One of the two roster readers. Membership is what the database says it is --
every record it holds, addressed the way every other reader of these files
addresses them (``address``, falling back to ``channel``). The write-limits
file never touches that set: it gates a subset of the channels a facility has,
and reading it as a roster is the bug this package exists to remove. A channel
absent from the limits file is still a channel; a limits key naming no record
adds nothing.

Direction is the half a paradigm database cannot state, because no paradigm
carries one. It is derived, in priority order:

1. **The channel limits file**, when one is configured and readable -- the
   deployment's enforced writability ground truth, read through
   :class:`~osprey_connectors.control_system.limits_validator.LimitsValidator`
   so that "writable" means here exactly what it means to the runtime write
   path that refuses the write, defaults block included. This is the same
   priority order, and the same authority, that the knowledge graph's
   ``ttl_generator/direction.py`` applies to the corpus it mints; that module
   re-implements the merge because it is deliberately import-light, while this
   one calls the validator itself.
2. **The address grammar** -- a final colon-separated token of ``SP`` is a
   setpoint, everything else reads.
3. **Neither**, when no limits file is available *and* no address in the
   database is a setpoint: the directions stay ``None`` and the result carries
   a :attr:`~osprey.channel_roster.records.RosterAbsenceReason.DIRECTION_UNDERIVABLE`
   absence naming the database. Membership is still real -- that pair is the
   one case the record types let coexist. Guessing "read" for the whole
   facility would be indistinguishable, to every consumer downstream, from a
   facility that genuinely has nothing settable.

The database classes are imported lazily, as ``channel_snapshot`` imports them,
to keep the Channel Finder service out of the build's import graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

from .records import (
    ChannelDirection,
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
)

logger = get_logger("channel_roster.database")

#: Config key naming the channel limits database, shared with the runtime write
#: path and with the knowledge graph's direction assignment.
LIMITS_DATABASE_CONFIG_KEY = "control_system.limits_checking.database_path"

#: Final address token the grammar fallback reads as a setpoint.
WRITE_SUBFIELD = "SP"

#: What separates an address into its tokens.
ADDRESS_SEPARATOR = ":"


def resolve_limits_path(config: Mapping[str, Any]) -> Path | None:
    """Find the channel limits file this deployment enforces, if any.

    A relative ``database_path`` is authored beside the config it appears in --
    the rule
    :meth:`~osprey_connectors.control_system.limits_validator.LimitsValidator.resolve_database_path`
    applies at runtime and
    :func:`~osprey.deployment.compose_generator.resolve_limits_mount` applies at
    render -- so it is anchored on the recorded ``config_dir`` when the build
    has one, and on the working directory otherwise, which is the project root
    the build sets before generating anything.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The anchored path, or None when the key names nothing usable. Existence
        is not probed here: an unreadable file is a direction-source question,
        answered in :func:`_load_writable_addresses`.
    """
    control_system = config.get("control_system") or {}
    if not isinstance(control_system, dict):
        return None
    limits_checking = control_system.get("limits_checking") or {}
    if not isinstance(limits_checking, dict):
        return None

    raw = limits_checking.get("database_path")
    if not isinstance(raw, str) or not raw.strip():
        return None

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    config_dir = config.get("config_dir")
    anchor = Path(config_dir) if isinstance(config_dir, str) and config_dir.strip() else Path.cwd()
    return anchor / path


def _load_writable_addresses(limits_path: Path) -> frozenset[str] | None:
    """Read the addresses a limits file declares writable, defaults-merged.

    Delegates to
    :meth:`~osprey_connectors.control_system.limits_validator.LimitsValidator._load_limits_database`
    rather than re-reading the JSON, so the ``defaults`` block, the metadata
    keys and the per-entry validation are applied by the same code the write
    path applies them with. The demo file grants writability by *omitting*
    ``writable`` so entries inherit ``defaults.writable: true``; a reader that
    looked only for explicit ``writable: true`` would find none at all.

    Args:
        limits_path: Path to a ``channel_limits.json``-shaped file.

    Returns:
        The writable addresses, or None when the file is missing or cannot be
        parsed -- direction then falls through to the address grammar, which is
        a degraded answer rather than a wrong one.
    """
    from osprey_connectors.control_system.limits_validator import LimitsValidator

    if not limits_path.is_file():
        # Probed here rather than left to the validator: a deployment that
        # enforces no limits is an ordinary one, and the validator answers an
        # absent file by logging the resolved path as an error.
        logger.debug(
            f"No channel limits database at {limits_path}; deriving channel directions "
            "from the address grammar instead."
        )
        return None

    try:
        limits, _raw = LimitsValidator._load_limits_database(str(limits_path))
    except ValueError as exc:
        # Named by its config key rather than by the file it resolved to: a
        # build resolves a relative path into its own staging tree, so the
        # resolved name is a `build/.tmp/...` file an operator cannot edit,
        # while the key is the line in config.yml they can.
        logger.warning(
            f"The channel limits database {LIMITS_DATABASE_CONFIG_KEY} names could not be "
            f"read ({exc}); deriving channel directions from the address grammar instead."
        )
        return None
    return frozenset(address for address, entry in limits.items() if entry.writable)


def _database_class(pipeline_type: str, db_config: Mapping[str, Any]) -> Any:
    """Return the database class that opens this paradigm's file.

    Split from the load so the paradigm NAME is validated before anything
    touches the filesystem: a typo'd ``pipeline_mode`` is a configuration
    mistake with a fix, and it must not be reported as "your source is not
    staged yet" just because the path it named does not exist either.

    Args:
        pipeline_type: Pipeline name from ``detect_pipeline_config``.
        db_config: The pipeline's ``database`` block (``type`` picks between the
            in-context pipeline's flat and template databases).

    Returns:
        The class to construct with the resolved path.

    Raises:
        PipelineModeError: If ``pipeline_type`` is not a known paradigm.
    """
    if pipeline_type == "in_context":
        from osprey.services.channel_finder.databases import (
            FlatChannelDatabase,
            TemplateChannelDatabase,
        )

        if db_config.get("type", "template") == "flat":
            return FlatChannelDatabase
        return TemplateChannelDatabase
    if pipeline_type == "hierarchical":
        from osprey.services.channel_finder.databases import HierarchicalChannelDatabase

        return HierarchicalChannelDatabase
    if pipeline_type == "middle_layer":
        from osprey.services.channel_finder.databases import MiddleLayerDatabase

        return MiddleLayerDatabase

    from osprey.services.channel_finder.core.exceptions import PipelineModeError

    raise PipelineModeError(f"Unknown channel finder pipeline '{pipeline_type}'")


def _load_channel_records(
    pipeline_type: str, db_config: Mapping[str, Any], db_path: Path
) -> list[dict]:
    """Load a paradigm database and return its channel records.

    Constructing any of the database classes loads the file, so malformed
    content surfaces here as an exception for the caller to absorb.

    Args:
        pipeline_type: Pipeline name from ``detect_pipeline_config``.
        db_config: The pipeline's ``database`` block (``type`` for the
            in-context pipeline; the path is already resolved).
        db_path: Resolved path to the database file.

    Returns:
        Channel records, each carrying at least a ``channel`` key and usually an
        ``address``.

    Raises:
        PipelineModeError: If ``pipeline_type`` is not a known paradigm.
    """
    database = _database_class(pipeline_type, db_config)(str(db_path))
    records: list[dict] = database.get_all_channels()
    return records


def _record_addresses(records: list[dict]) -> list[str]:
    """Return each record's address once, in database order.

    A record that names both an address and a channel keeps its ``address``;
    one that names only ``channel`` (the hierarchical and middle-layer
    pipelines synthesize an address from the channel name) contributes that.
    The same address twice is one channel, so the second occurrence is dropped
    rather than minting a duplicate record.

    Args:
        records: Raw records as a paradigm database yields them.

    Returns:
        The distinct, non-empty addresses, first occurrence first.

    Raises:
        KeyError: If a record names neither an address nor a channel.
    """
    seen: dict[str, None] = {}
    for record in records:
        address = record.get("address", record["channel"])
        if address:
            seen.setdefault(address, None)
    return list(seen)


def _database_path_config_key(pipeline_type: str) -> str:
    """The config key that declared this paradigm's database file.

    Named in the absence a missing database earns, because that key is what an
    operator edits -- the resolved path a build reports is a file inside its own
    staging tree.
    """
    return f"channel_finder.pipelines.{pipeline_type}.database.path"


def _is_setpoint(address: str) -> bool:
    """Whether the address grammar reads *address* as a setpoint."""
    return address.rsplit(ADDRESS_SEPARATOR, 1)[-1] == WRITE_SUBFIELD


def read_database_roster(
    config: Mapping[str, Any],
    source: RosterSource,
    pipeline_type: str,
    db_config: Mapping[str, Any],
) -> RosterResult:
    """Enumerate the channels of a Channel Finder paradigm database.

    Args:
        config: Full project configuration dictionary -- read only for the
            limits file that may answer direction, never for membership.
        source: The resolved database source, from
            :mod:`osprey.channel_roster.sources`.
        pipeline_type: Pipeline name from ``detect_pipeline_config``.
        db_config: That pipeline's ``database`` block.

    Returns:
        A result carrying one record per distinct address in database order.
        Directions are limits-derived, grammar-derived, or ``None`` alongside a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.DIRECTION_UNDERIVABLE`
        absence, per the priority order in the module docstring. A database that
        is not there yields no records and a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.MISSING_SOURCE`
        absence -- the fail-soft half of the pair. One that is there and cannot
        be read yields no records and a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.CORRUPT_SOURCE`
        absence naming it, and one that reads cleanly and holds no channel an
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.EMPTY_SOURCE`
        one -- so a consumer reports a broken or unpopulated source instead of
        an empty facility.

    Raises:
        PipelineModeError: If ``pipeline_type`` names a paradigm that does not
            exist. That is a configuration mistake rather than a degraded
            source, so it stops the build rather than becoming an absence.
    """
    from osprey.services.channel_finder.core.exceptions import PipelineModeError

    # The paradigm name is checked before the file: a mode that does not exist
    # is a configuration mistake, and it would otherwise be reported as an
    # unstaged source whenever its path happens to be absent too.
    _database_class(pipeline_type, db_config)

    if not source.path.exists():
        # Absent, not broken: the two get opposite treatment from consumers, so
        # they are different reasons here rather than one reason plus a second
        # ``stat`` at every seam. ``exists`` rather than ``is_file``: a path
        # that names a directory IS there and cannot be read, which is the
        # other half of the pair.
        logger.warning(
            f"The channel database {source.for_display()} is not there, so this build "
            "enumerates no channels from it."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.MISSING_SOURCE,
                path=source.path,
                spelled=source.spelled,
                config_keys=(_database_path_config_key(pipeline_type),),
            )
        )

    try:
        addresses = _record_addresses(_load_channel_records(pipeline_type, db_config, source.path))
    except PipelineModeError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read the channel database at {source.path} ({exc}).")
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.CORRUPT_SOURCE,
                path=source.path,
                spelled=source.spelled,
                detail=str(exc),
            )
        )

    if not addresses:
        logger.warning(
            f"The channel database at {source.path} was read and enumerates no channels."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.EMPTY_SOURCE,
                path=source.path,
                spelled=source.spelled,
            )
        )

    limits_path = resolve_limits_path(config)
    writable = _load_writable_addresses(limits_path) if limits_path is not None else None

    absence: RosterAbsence | None = None
    directions: dict[str, ChannelDirection | None]
    if writable is not None:
        directions = {
            address: ("write" if address in writable else "read") for address in addresses
        }
    elif any(_is_setpoint(address) for address in addresses):
        directions = {
            address: ("write" if _is_setpoint(address) else "read") for address in addresses
        }
    else:
        # No limits file and not one setpoint address in the whole database:
        # there is no rule left that could tell a settable channel from a
        # readable one, and "read" would be a guess wearing a fact's clothes.
        directions = dict.fromkeys(addresses)
        absence = RosterAbsence(reason=RosterAbsenceReason.DIRECTION_UNDERIVABLE, path=source.path)

    records = tuple(
        ChannelRecord(address=address, source=source, direction=direction)
        for address, direction in directions.items()
    )
    return RosterResult(records=records, source=source, absence=absence)
