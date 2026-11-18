"""How the deployment's archive is configured, and which targets it serves, read-only.

The directory service builds the archive per target
(:meth:`~osprey.services.pv_directory.service.DirectoryService.archive`) and
asks it what it holds; this module adds the facts the panel and the agent read
beside those answers:

* :func:`deployed_configuration` is the archive's configuration as rendered:
  the archiver type and, for the deployment's own store, the knobs that shape
  it. Never a login, a password variable or an address.
* :func:`served_targets` says, per configured control target, whether the
  archive serves it, from :func:`history_targets`.

The directory's rows are joined with the store's channels in one place, the
service's ``archived`` keyword and facet.

"Archived" means "in the newest samples the store holds": what the recorder
writes now.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.cli.build_profile_archiver import VAArchiverConfig, freshness_max_age_s
from osprey.services.pv_directory._connectors import service_targets
from osprey_connectors.archiver_selection import resolve_archiver_type
from osprey_connectors.connector_types import MONGODB_ARCHIVER, TARGET_VA
from osprey_connectors.simulation.archive import SeedKnobs
from osprey_connectors.targets import configured_targets

__all__ = [
    "deployed_configuration",
    "freshness_bound_s",
    "history_targets",
    "own_store",
    "served_targets",
]

#: The recorder knob the configuration shows beside the seed's.
_RECORDER_CADENCE = "recorder_cadence_sec"


def history_targets(config: Mapping[str, Any]) -> frozenset[str]:
    """The targets whose history the deployment's archive holds.

    The archive row of the connectors' ``service_targets``, without the mock;
    none on a connectors release without the PV directory modules.
    """
    if service_targets is None:
        return frozenset()
    held: frozenset[str] = service_targets(config).history_targets()
    return held


def own_store(config: Mapping[str, Any], target: str) -> bool:
    """Whether *target*'s archive is the deployment's own store (the VA's recorder)."""
    archiver = config.get("archiver")
    return (
        target == TARGET_VA
        and resolve_archiver_type(archiver) == MONGODB_ARCHIVER
        and target in history_targets(config)
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
    """Each control target a session can be on, and whether the archive holds its history."""
    served = history_targets(config)
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
