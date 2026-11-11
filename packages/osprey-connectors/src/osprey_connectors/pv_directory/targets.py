"""Which source answers each directory section on each control target, in one place.

A deployment carries **one** ``pv_directory:`` block and a session can be pointed
at up to three machines. A section answers only from the machine the session is
on, so :func:`pv_directory_targets` says which source that is per target, and a
reader holding ``None`` says "not available for this target": never another
machine's data under the selected machine's name.

================ ================================ ==========================
row              ``va``                           ``live`` and ``standin``
================ ================================ ==========================
``pv_directory`` ``simulator_view`` when          the section's ``type``
                 :func:`has_simulator`
``alarm_state``  ``alarm_replay`` when            the sub-block's ``type``
                 ``replay_file`` is set
``alarm_log``    ``None``                         the sub-block's ``type``
``put_log``      ``None``                         the sub-block's ``type``
``logbook``      ``None``                         ``ariel``, when configured
================ ================================ ==========================

The facility's services belong to the facility's machine and the stand-in
rehearses that machine, as :mod:`osprey_connectors.archive_targets` has it, so
the two share :data:`~osprey_connectors.archive_targets.APPLIANCE_TARGETS` and
a ``standin`` spec is marked ``rehearsal``. A simulator spec carries none of the
facility's settings, and a facility spec no ``replay_file``. ``replay_file`` is
passed verbatim: only the framework reads its ``demo:`` spelling.

Pure, and never raises: the answer is a function of the mapping alone. A block
that is not a mapping, or a ``type`` or ``replay_file`` that is not a non-blank
string, names no source; refusing such a block at build is ``resolve_section``'s
job in the framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from osprey_connectors.archive_targets import APPLIANCE_TARGETS
from osprey_connectors.pv_directory.contract import BackendSpec, Source
from osprey_connectors.simulator import virtual_accelerator_port
from osprey_connectors.types import (
    CONTROL_TARGETS,
    TARGET_STANDIN,
    TARGET_VA,
    configured_targets,
)

__all__ = ["has_simulator", "pv_directory_targets"]

_SECTION_KEY = "pv_directory"
_SIMULATOR_VIEW = "simulator_view"
_ALARM_REPLAY = "alarm_replay"
_ARIEL = "ariel"

_Kind = Literal["pv_directory", "alarm_log", "alarm_state", "put_log"]
_KINDS: tuple[_Kind, ...] = ("pv_directory", "alarm_log", "alarm_state", "put_log")


def has_simulator(config: Mapping[str, Any]) -> bool:
    """Whether the build has a simulator.

    The port alone answers no for an in-process virtual accelerator; the
    connector type alone answers no for a facility baseline that keeps its
    simulator. The simulator view cannot decide, because every build writes it.

    Args:
        config: The full config mapping, as loaded from ``config.yml``. The build
            validator, which sees an overlay, passes the overlay with
            ``services.virtual_accelerator.port`` filled in from the profile.
    """
    if not isinstance(config, Mapping):
        return False
    return virtual_accelerator_port(config) is not None or TARGET_VA in configured_targets(
        config.get("control_system")
    )


def pv_directory_targets(
    config: Mapping[str, Any], *, ariel_configured: bool
) -> dict[str, dict[str, BackendSpec | Source | None]]:
    """The source of every section on every target, per the module table.

    Args:
        config: The full config mapping, as loaded from ``config.yml``: not the
            ``pv_directory:`` section, because the ``va`` half turns on whether
            the deployment built a simulator.
        ariel_configured: Whether the deployment has a logbook to read.

    Returns:
        ``{kind: {target: BackendSpec | None}, "logbook": {target: Source | None}}``
        with every target of :data:`~osprey_connectors.types.CONTROL_TARGETS` in
        every row. With no section, every entry is ``None``.
    """
    section = config.get(_SECTION_KEY) if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        return {row: dict.fromkeys(CONTROL_TARGETS) for row in (*_KINDS, "logbook")}

    display = _mapping(section.get("display"))
    directory_fold = {
        **_present(description_property=_text(section.get("description_property"))),
        **_present(exact_properties=display.get("exact_properties")),
        **_caveats(section),
    }
    alarm_state = _mapping(section.get("alarm_state"))
    replay_file = _text(alarm_state.get("replay_file"))

    table: dict[str, dict[str, BackendSpec | Source | None]] = {
        "pv_directory": _facility_row("pv_directory", section, directory_fold),
        "alarm_log": _facility_row("alarm_log", _mapping(section.get("alarm_log"))),
        "alarm_state": _facility_row("alarm_state", alarm_state),
        "put_log": _facility_row("put_log", _mapping(section.get("put_log"))),
        "logbook": {target: _logbook(target, ariel_configured) for target in CONTROL_TARGETS},
    }
    if has_simulator(config):
        table["pv_directory"][TARGET_VA] = _spec(
            "pv_directory", _SIMULATOR_VIEW, TARGET_VA, directory_fold
        )
    if replay_file:
        table["alarm_state"][TARGET_VA] = _spec(
            "alarm_state",
            _ALARM_REPLAY,
            TARGET_VA,
            {"replay_file": replay_file, **_caveats(alarm_state)},
        )
    return table


def _facility_row(
    kind: _Kind, block: Mapping[str, Any], fold: Mapping[str, Any] | None = None
) -> dict[str, BackendSpec | Source | None]:
    """A row with *block*'s ``type`` on the facility's targets and nothing elsewhere."""
    service = _text(block.get("type"))
    settings = {**_mapping(block.get("settings")), **(_caveats(block) if fold is None else fold)}
    return {
        target: _spec(kind, service, target, settings)
        if service and target in APPLIANCE_TARGETS
        else None
        for target in CONTROL_TARGETS
    }


def _spec(kind: _Kind, service: str, target: str, settings: Mapping[str, Any]) -> BackendSpec:
    return BackendSpec(
        kind=kind,
        type=service,
        target=target,
        settings=_copied(settings),
        rehearsal=target == TARGET_STANDIN,
    )


def _logbook(target: str, ariel_configured: bool) -> Source | None:
    if not ariel_configured or target not in APPLIANCE_TARGETS:
        return None
    return Source(
        target=target,  # type: ignore[arg-type]
        service=_ARIEL,
        label=_ARIEL,
        rehearsal=target == TARGET_STANDIN,
    )


def _caveats(block: Mapping[str, Any]) -> dict[str, Any]:
    return _present(caveats=block.get("caveats"))


def _present(**values: Any) -> dict[str, Any]:
    """The keyword arguments the section set; a key it did not set is not folded."""
    return {key: value for key, value in values.items() if value is not None}


def _copied(settings: Mapping[str, Any]) -> dict[str, Any]:
    """*settings* with its folded lists copied, so a spec never aliases the config."""
    return {key: list(v) if isinstance(v, list) else v for key, v in settings.items()}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
