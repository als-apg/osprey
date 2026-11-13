"""What the deployment's archive holds and how it is configured, read-only.

The directory service builds the archiver connector per target
(:meth:`~osprey.services.pv_directory.service.DirectoryService.archive`); this
module turns its answers into the facts the panel and the agent read:

* :func:`read_extent` asks the connector which channels the store records and
  the span of its samples, and answers unknown, with the reason, when it cannot.
* :func:`deployed_configuration` is the archive's configuration as rendered:
  the archiver type and, for the deployment's own store, the knobs that shape
  it. Never a login, a password variable or an address.
* :func:`served_targets` says, per configured control target, whether the
  archive serves it.

The directory's rows are joined with the store's channels in one place, the
service's ``archived`` keyword and facet.

"Archived" means "in the newest samples the store holds": what the recorder
writes now.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from osprey.cli.build_profile_archiver import VAArchiverConfig, freshness_max_age_s
from osprey.services.pv_directory.vocabulary import service_failed
from osprey_connectors.archive_targets import archive_targets
from osprey_connectors.simulation.archive import SeedKnobs
from osprey_connectors.types import (
    MONGODB_ARCHIVER,
    TARGET_VA,
    configured_targets,
    resolve_archiver_type,
)

__all__ = [
    "NO_EXTENT_REASON",
    "Extent",
    "deployed_configuration",
    "freshness_bound_s",
    "own_store",
    "read_extent",
    "served_targets",
]

logger = logging.getLogger(__name__)

#: The reason a connectors release whose archiver has no ``store_extent`` answers.
NO_EXTENT_REASON = "installed osprey-connectors cannot list the archive's channels (upgrade it)"

#: The recorder knob the configuration shows beside the seed's.
_RECORDER_CADENCE = "recorder_cadence_sec"


@dataclass(frozen=True)
class Extent:
    """What the store holds, as far as the archiver can say; ``None`` is unknown."""

    channels: frozenset[str] | None = None
    oldest: datetime | None = None
    newest: datetime | None = None
    reason: str | None = None


async def read_extent(connector: Any, deadline: float) -> Extent:
    """The store's extent from *connector*, or unknown with the reason.

    A connector without ``store_extent`` (an older connectors release) answers
    :data:`NO_EXTENT_REASON`; a read that misses *deadline* or raises answers
    its reason. Never raises.
    """
    read = getattr(connector, "store_extent", None)
    if read is None:
        return Extent(reason=NO_EXTENT_REASON)
    try:
        answer = await asyncio.wait_for(read(), deadline)
    except TimeoutError:
        return Extent(reason=f"no answer within {deadline:g} s")
    except Exception as failure:
        logger.warning("the archive's extent could not be read", exc_info=True)
        return Extent(reason=service_failed(failure))
    channels = getattr(answer, "channels", None)
    return Extent(
        channels=frozenset(channels) if channels is not None else None,
        oldest=getattr(answer, "oldest", None),
        newest=getattr(answer, "newest", None),
        reason=getattr(answer, "reason", None),
    )


def own_store(config: Mapping[str, Any], target: str) -> bool:
    """Whether *target*'s archive is the deployment's own store (the VA's recorder)."""
    archiver = config.get("archiver")
    return (
        target == TARGET_VA
        and resolve_archiver_type(archiver) == MONGODB_ARCHIVER
        and target in archive_targets(config)
    )


def deployed_configuration(config: Mapping[str, Any], target: str) -> dict[str, Any]:
    """The archive's configuration as rendered, for *target*: what it is and its shape.

    ``archiver.type`` always; for the deployment's own store on ``va``, the
    ``va_archiver`` knobs the seed and the recorder read. Nothing that says
    where the store is or how to log in.
    """
    shown: dict[str, Any] = {"archiver": {"type": resolve_archiver_type(config.get("archiver"))}}
    if not own_store(config, target):
        return shown
    knobs = _knobs(config)
    if knobs is not None:
        shown["va_archiver"] = knobs
    return shown


def freshness_bound_s(config: Mapping[str, Any]) -> int | None:
    """How old the newest sample of the deployment's own store may be before it reads late.

    The bound of the archive-freshness check the build derives from the
    recorder's cadence. ``None`` when the knobs do not read.
    """
    knobs = _knobs(config)
    if knobs is None:
        return None
    return freshness_max_age_s(VAArchiverConfig(recorder_cadence_sec=knobs[_RECORDER_CADENCE]))


def served_targets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each control target a session can be on, and whether the archive serves it."""
    served = archive_targets(config)
    return [
        {"target": target, "archive": target in served}
        for target in configured_targets(config.get("control_system"))
    ]


def _knobs(config: Mapping[str, Any]) -> dict[str, int] | None:
    """The store's shape as rendered: the seed's knobs and the recorder's cadence."""
    try:
        seed = SeedKnobs.from_config(config)
    except ValueError:
        return None
    block = config.get("va_archiver")
    rendered = block if isinstance(block, Mapping) else {}
    knobs = {
        "retention_days": seed.retention_days,
        "hot_span_hours": seed.hot_span_hours,
        "hot_cadence_sec": seed.hot_cadence_sec,
        "tail_cadence_sec": seed.tail_cadence_sec,
    }
    cadence = rendered.get(_RECORDER_CADENCE, VAArchiverConfig.recorder_cadence_sec)
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
        return None
    knobs[_RECORDER_CADENCE] = cadence
    return knobs
