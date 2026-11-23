"""What the facility file says the simulator serves: its own addresses and what a scenario stills.

``status_address`` is ``<code>:SIM:<model>:STATUS`` for one physics model and
``server_health_addresses`` the model's IOC health addresses, one per signal of
``SERVER_HEALTH_SIGNALS``. ``resolved_still`` is the sorted list of float
addresses a scenario's ``still`` names. The build refuses a channel on one of
these addresses and validates scenarios against the resolved ``still``; the
simulator view writes the same values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from osprey.facility import STILL_ALL
from osprey.facility.records import ChannelSelection, FacilityRecords
from osprey_connectors.simulation.values import DEFAULT_VALUE_TYPE
from osprey_connectors.simulation.view import SCENARIOS_DIR, SERVER_HEALTH_SIGNALS

__all__ = [
    "resolved_still",
    "server_health_addresses",
    "status_address",
]


def status_address(code: str, model: str) -> str:
    """The status address the simulator serves for one physics model.

    Args:
        code: The facility's identity ``code``.
        model: The model's name.

    Returns:
        ``<code>:SIM:<model>:STATUS``.
    """
    return f"{code}:SIM:{model}:STATUS"


def server_health_addresses(code: str, model: str) -> list[str]:
    """The IOC health addresses the simulator serves for one model, texture included.

    Args:
        code: The facility's identity ``code``.
        model: The model's name.

    Returns:
        ``<code>:SIM:<model>:<signal>`` for each signal of the view's
        ``SERVER_HEALTH_SIGNALS``.
    """
    return [f"{code}:SIM:{model}:{signal}" for signal in SERVER_HEALTH_SIGNALS]


def _float_addresses(channels: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(channel["id"])
        for channel in channels
        if channel.get("value_type", DEFAULT_VALUE_TYPE) == DEFAULT_VALUE_TYPE
    }


def resolved_still(records: FacilityRecords, scenario: Mapping[str, Any]) -> list[str]:
    """The float addresses a scenario's ``still`` names, sorted, each once.

    ``all`` is every float channel, an address is itself, and a selection is the
    float channels its query matches, over every model of the file.

    Args:
        records: The facility file's records.
        scenario: A scenario stating ``still``.

    Returns:
        The addresses the scenario stills.

    Raises:
        FacilityBuildError: ``value-invalid``: a selection matches no float
            channel, or a resolved reading is one whose ``noise`` or ``couple``
            the scenario sets.
    """
    from osprey.facility.errors import FacilityBuildError

    name = str(scenario["name"])

    def stop(detail: str, remedy: str) -> FacilityBuildError:
        return FacilityBuildError(
            "value-invalid",
            name,
            [f"{SCENARIOS_DIR}/{name}.yaml"],
            remedy,
            record_kind="scenario",
            detail=detail,
        )

    still = scenario["still"]
    resolved: set[str] = set()
    if still == STILL_ALL:
        resolved = _float_addresses(records.of_kind("channel"))
    else:
        for entry in still:
            if isinstance(entry, str):
                resolved.add(entry)
                continue
            matched = _float_addresses(records.select_channels(ChannelSelection(**entry)))
            if not matched:
                raise stop(
                    f"`still` selection {dict(entry)!r} matches no float channel",
                    "correct this selection in `still`",
                )
            resolved |= matched
    for slot in ("noise", "couple"):
        clashes = sorted(resolved.intersection(scenario.get(slot) or ()))
        if clashes:
            raise stop(
                f"`{slot}.{clashes[0]}` sets the motion of a reading `still` stills",
                "a scenario either stills a reading or sets its motion",
            )
    return sorted(resolved)
