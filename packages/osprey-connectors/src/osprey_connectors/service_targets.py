"""Which source answers each directory section on each control target, in one place.

A deployment carries **one** ``pv_directory:`` block and a session can be pointed
at up to three machines. A section answers only from the machine the session is
on, so :func:`service_targets` says which source that is per target, and a
reader holding ``None`` says "not available for this target": never another
machine's data under the selected machine's name.

================ ================================ ==========================
row              ``va``                           ``live`` and ``standin``
================ ================================ ==========================
``directory``    ``simulator_view`` when          the section's ``type``
                 ``simulator.has_simulator``
``alarm_state``  ``alarm_replay`` when            the sub-block's ``type``
                 ``replay_file`` is set
``alarm_log``    ``None``                         the sub-block's ``type``
``put_log``      ``None``                         the sub-block's ``type``
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
from osprey_connectors.service_answers import BackendSpec
from osprey_connectors.simulator import has_simulator
from osprey_connectors.types import CONTROL_TARGETS, TARGET_STANDIN, TARGET_VA, ControlTarget

__all__ = ["service_targets"]

_SECTION_KEY = "pv_directory"
_SIMULATOR_VIEW = "simulator_view"
_ALARM_REPLAY = "alarm_replay"

_Kind = Literal["directory", "alarm_log", "alarm_state", "put_log"]
_KINDS: tuple[_Kind, ...] = ("directory", "alarm_log", "alarm_state", "put_log")


def service_targets(config: Mapping[str, Any]) -> dict[str, dict[str, BackendSpec | None]]:
    """The source of every section on every target, per the module table.

    Args:
        config: The full config mapping, as loaded from ``config.yml``: not the
            ``pv_directory:`` section, because the ``va`` half turns on whether
            the deployment built a simulator.

    Returns:
        ``{kind: {target: BackendSpec | None}}`` with every target of
        :data:`~osprey_connectors.types.CONTROL_TARGETS` in every row. With no
        section, every entry is ``None``.
    """
    section = config.get(_SECTION_KEY) if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        return {row: dict.fromkeys(CONTROL_TARGETS) for row in _KINDS}

    display = _mapping(section.get("display"))
    directory_fold = {
        **_present(description_property=_text(section.get("description_property"))),
        **_present(exact_properties=display.get("exact_properties")),
        **_caveats(section),
    }
    alarm_state = _mapping(section.get("alarm_state"))
    replay_file = _text(alarm_state.get("replay_file"))

    table: dict[str, dict[str, BackendSpec | None]] = {
        "directory": _facility_row("directory", section, directory_fold),
        "alarm_log": _facility_row("alarm_log", _mapping(section.get("alarm_log"))),
        "alarm_state": _facility_row("alarm_state", alarm_state),
        "put_log": _facility_row("put_log", _mapping(section.get("put_log"))),
    }
    if has_simulator(config):
        table["directory"][TARGET_VA] = _spec(
            "directory", _SIMULATOR_VIEW, TARGET_VA, directory_fold
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
) -> dict[str, BackendSpec | None]:
    """A row with *block*'s ``type`` on the facility's targets and nothing elsewhere."""
    service = _text(block.get("type"))
    settings = {**_mapping(block.get("settings")), **(_caveats(block) if fold is None else fold)}
    return {
        target: _spec(kind, service, target, settings)
        if service and target in APPLIANCE_TARGETS
        else None
        for target in CONTROL_TARGETS
    }


def _spec(
    kind: _Kind, service: str, target: ControlTarget, settings: Mapping[str, Any]
) -> BackendSpec:
    return BackendSpec(
        kind=kind,
        type=service,
        target=target,
        settings=_copied(settings),
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
