"""Which source answers each service kind on each control target, in one place.

A deployment carries **one** ``pv_directory:`` block and **one** ``archiver:``
block, and a session can be pointed at up to three machines: the facility's own
(``live``), the virtual accelerator (``va``) and the stand-in soft IOC
(``standin``). A service answers only from the machine the session is on, so
:func:`service_targets` says which source that is per target, as one
:class:`ServiceTable`, and a reader holding ``None`` says "not available for
this target": never another machine's data under the selected machine's name.

================ ================================ ==========================
row              ``va``                           ``live`` and ``standin``
================ ================================ ==========================
``directory``    ``simulator_view`` when          the section's ``type``
                 ``simulator.has_simulator``
``alarm_state``  ``alarm_replay`` when            the sub-block's ``type``
                 ``replay_file`` is set
``alarm_log``    ``None``                         the sub-block's ``type``
``put_log``      ``None``                         the sub-block's ``type``
``archive``      ``mongodb_archiver`` when the    a facility archive when its
                 deployment built a simulator;    block names an address;
                 the mock                         the mock
================ ================================ ==========================

The facility's services belong to the facility's machine and the stand-in
rehearses that machine, so they share :data:`APPLIANCE_TARGETS`, and a
``standin`` spec is marked ``rehearsal``. A simulator spec carries none of the
facility's settings, and a facility spec no ``replay_file``. ``replay_file`` is
passed verbatim: only the framework reads its ``demo:`` spelling.

The archive row
---------------

Every archiver belongs to one of three classes, and the class decides:

* ``mongodb_archiver`` is **the deployment's own store**. On a deployment that
  built a virtual accelerator it serves ``va``, whatever machine its sessions
  start on and whether or not a stand-in was built beside it; on one that built
  no simulator it serves nothing.
* Every other archiver except the mock is **the facility's archive** (the
  EPICS archiver appliance, MYA, DOOCS, or a type this table does not know)
  and serves ``live`` and ``standin`` when its block names an address, nothing
  when it does not.
* The mock, or no block at all, is opened on **every** target. It synthesizes
  and has no past, so it holds no target's history
  (:meth:`ServiceTable.history_targets` leaves it out); whether pairing it with
  a machine is honest is the archiver pairing's question, asked at build and at
  the target switch, not this table's.

**The facility's archive belongs to the facility's machine, and the stand-in
rehearses that machine.** It is not something a deployment stands up for its
simulation; it holds the real machine's past, and that past is the one a
stand-in session is shown, because what the stand-in rehearses is the
procedure against the real machine's history. It needs an address to count: a
block that cannot be dialled is not a deployment saying it has an archive. The
address is read per type (:data:`_ADDRESS_SETTINGS`): the appliance's ``url``;
MYA and DOOCS find their server without a setting (the client library's
configured ``myquery_server``, the facility's name service), so their block
always names one.

**The mongodb store belongs to the virtual accelerator.** The deployment seeds
it with the simulator's synthesized history and the recorder samples the
simulator into it (``osprey.services.archiver_recorder``); the stand-in is
never recorded there and its offsets are never seeded there. So the store is
the ``va`` target's exactly when the deployment built a simulator — read
through :func:`osprey_connectors.deployment.simulator.virtual_accelerator_port`, the
build's own statement that it stood one up — and no other target's ever.
``control_system.type`` is not consulted: it names the machine a session
starts on, and a deployment baselined on its stand-in or on the facility's
machine still seeds and records the simulator it built.

A block the factory would refuse — both settings spellings at once, or a
settings value that is not a mapping — serves no archive on any target, so it
claims none; that is the mock's block too.

Pure, and never raises: the answer is a function of the mapping alone. No
environment variable, no file and no control-context record is consulted, which
is what lets a supervisor call it at every restart and get the deployment's
answer rather than the calling process's. A block that is not a mapping, or a
``type`` or ``replay_file`` that is not a non-blank string, names no source;
refusing such a block at build is ``resolve_section``'s job in the framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import get_args

from osprey_connectors.archiver_selection import resolve_archiver_settings, resolve_archiver_type
from osprey_connectors.connector_types import (
    CONTROL_TARGETS,
    DOOCS_ARCHIVER,
    EPICS_ARCHIVER,
    MOCK_ARCHIVER,
    MONGODB_ARCHIVER,
    MYA_ARCHIVER,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    ControlTarget,
)
from osprey_connectors.deployment.simulator import has_simulator, virtual_accelerator_port
from osprey_connectors.model import FrozenModel, ReadOnlyMapping
from osprey_connectors.service_answers import BackendSpec, ServiceKind

__all__ = ["APPLIANCE_TARGETS", "ServiceTable", "service_targets"]

_SECTION_KEY = "pv_directory"
_SIMULATOR_VIEW = "simulator_view"
_ALARM_REPLAY = "alarm_replay"

#: The top-level config section selecting and configuring the archiver.
_ARCHIVER_SECTION_KEY = "archiver"

#: The archiver types that are the deployment's own store. Every other type
#: except the mock is the facility's archive.
_STORE_ARCHIVERS: frozenset[str] = frozenset({MONGODB_ARCHIVER})

#: The setting under which a facility archive names where it is, per type.
#: ``None``: the connector finds its server without one (MYA through the client
#: library's configured ``myquery_server``, DOOCS through the facility's name
#: service), so its block always names an address. A type not listed names it
#: under :data:`_DEFAULT_ADDRESS_SETTING`.
_ADDRESS_SETTINGS: Mapping[str, str | None] = MappingProxyType(
    {EPICS_ARCHIVER: "url", MYA_ARCHIVER: None, DOOCS_ARCHIVER: None}
)

#: The address setting of a facility archive :data:`_ADDRESS_SETTINGS` does not list.
_DEFAULT_ADDRESS_SETTING = "url"

#: The targets the facility's services and archive serve: the real machine and
#: the stand-in that rehearses it.
APPLIANCE_TARGETS: frozenset[ControlTarget] = frozenset({TARGET_LIVE, TARGET_STANDIN})

#: The targets the deployment's own store serves: the virtual accelerator.
_STORE_TARGETS: frozenset[ControlTarget] = frozenset({TARGET_VA})

#: No archive for any target.
_NO_TARGETS: frozenset[ControlTarget] = frozenset()

_KINDS: tuple[ServiceKind, ...] = get_args(ServiceKind)

#: One row of the table: the spec each target opens, ``None`` where none.
_Row = ReadOnlyMapping[ControlTarget, BackendSpec | None]


class ServiceTable(FrozenModel):
    """The source of every service kind on every control target.

    One field per :data:`~osprey_connectors.service_answers.ServiceKind`, each
    holding every target of :data:`~osprey_connectors.connector_types.CONTROL_TARGETS`:
    the spec a reader on that target builds, or ``None`` when the kind is not
    available there. ``archive`` is what a reader on a target opens, the mock
    on every target included.
    """

    directory: _Row
    alarm_state: _Row
    alarm_log: _Row
    put_log: _Row
    archive: _Row

    def row(self, kind: ServiceKind) -> Mapping[ControlTarget, BackendSpec | None]:
        """The row of *kind*: its spec on every target."""
        rows: Mapping[ServiceKind, Mapping[ControlTarget, BackendSpec | None]] = {
            "directory": self.directory,
            "alarm_state": self.alarm_state,
            "alarm_log": self.alarm_log,
            "put_log": self.put_log,
            "archive": self.archive,
        }
        return rows[kind]

    def history_targets(self) -> frozenset[ControlTarget]:
        """The targets whose history the archive holds: every spec but the mock's.

        A target absent from the set has no archive of its own: a caller holding
        it has nothing to offer as its past, never a default to fall back on.
        """
        return frozenset(
            target
            for target, spec in self.archive.items()
            if spec is not None and spec.type != MOCK_ARCHIVER
        )


def service_targets(config: object) -> ServiceTable:
    """The source of every service kind on every target, per the module table.

    Args:
        config: The full config mapping, as loaded from ``config.yml``: not the
            ``pv_directory:`` or ``archiver:`` section, because the ``va`` half
            turns on whether the deployment built a simulator. Anything that is
            not a mapping has no section, which for the archive is the mock's.

    Returns:
        The table, every target of
        :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` in every row.
        With no ``pv_directory:`` section, every entry of the four directory
        rows is ``None``.
    """
    archive = _archive_row(config)
    section = config.get(_SECTION_KEY) if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        none: dict[ControlTarget, BackendSpec | None] = dict.fromkeys(CONTROL_TARGETS)
        return ServiceTable(
            directory=none, alarm_state=none, alarm_log=none, put_log=none, archive=archive
        )

    display = _mapping(section.get("display"))
    directory_fold = {
        **_present(description_property=_text(section.get("description_property"))),
        **_present(exact_properties=display.get("exact_properties")),
        **_caveats(section),
    }
    alarm_state_block = _mapping(section.get("alarm_state"))
    replay_file = _text(alarm_state_block.get("replay_file"))

    directory = _facility_row("directory", section, directory_fold)
    alarm_state = _facility_row("alarm_state", alarm_state_block)
    if has_simulator(config):
        directory[TARGET_VA] = _spec("directory", _SIMULATOR_VIEW, TARGET_VA, directory_fold)
    if replay_file:
        alarm_state[TARGET_VA] = _spec(
            "alarm_state",
            _ALARM_REPLAY,
            TARGET_VA,
            {"replay_file": replay_file, **_caveats(alarm_state_block)},
        )
    return ServiceTable(
        directory=directory,
        alarm_state=alarm_state,
        alarm_log=_facility_row("alarm_log", _mapping(section.get("alarm_log"))),
        put_log=_facility_row("put_log", _mapping(section.get("put_log"))),
        archive=archive,
    )


def _archive_row(config: object) -> dict[ControlTarget, BackendSpec | None]:
    """The ``archive`` row, per the module's archive table."""
    if not isinstance(config, Mapping):
        # No section is the mock's.
        config = {}
    section = config.get(_ARCHIVER_SECTION_KEY)
    archiver_type = resolve_archiver_type(section)
    try:
        settings = resolve_archiver_settings(section)
    except ValueError:
        # The factory refuses this block with the same error, so no archive is
        # ever built from it and there is no store to attribute.
        return dict.fromkeys(CONTROL_TARGETS)
    if archiver_type == MOCK_ARCHIVER:
        served: frozenset[ControlTarget] = frozenset(CONTROL_TARGETS)
    elif archiver_type in _STORE_ARCHIVERS:
        served = _STORE_TARGETS if virtual_accelerator_port(config) is not None else _NO_TARGETS
    else:
        served = APPLIANCE_TARGETS if _names_an_address(archiver_type, settings) else _NO_TARGETS
    return {
        target: _spec("archive", archiver_type, target, settings) if target in served else None
        for target in CONTROL_TARGETS
    }


def _names_an_address(archiver_type: str, settings: Mapping[str, object]) -> bool:
    """Whether a facility archive's *settings* state where it is.

    A non-blank string under the type's :data:`_ADDRESS_SETTINGS` entry, or
    always for a type whose connector finds its server without one. The
    archive's own reader decides whether the address is well formed; this
    only establishes that the deployment said one.
    """
    key = _ADDRESS_SETTINGS.get(archiver_type, _DEFAULT_ADDRESS_SETTING)
    if key is None:
        return True
    address = settings.get(key)
    return isinstance(address, str) and bool(address.strip())


def _facility_row(
    kind: ServiceKind, block: Mapping[str, object], fold: Mapping[str, object] | None = None
) -> dict[ControlTarget, BackendSpec | None]:
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
    kind: ServiceKind, service: str, target: ControlTarget, settings: Mapping[str, object]
) -> BackendSpec:
    return BackendSpec(
        kind=kind,
        type=service,
        target=target,
        settings=_copied(settings),
        rehearsal=target == TARGET_STANDIN,
    )


def _caveats(block: Mapping[str, object]) -> dict[str, object]:
    return _present(caveats=block.get("caveats"))


def _present(**values: object) -> dict[str, object]:
    """The keyword arguments the section set; a key it did not set is not folded."""
    return {key: value for key, value in values.items() if value is not None}


def _copied(settings: Mapping[str, object]) -> dict[str, object]:
    """*settings* with its folded lists copied, so a spec never aliases the config."""
    return {key: list(v) if isinstance(v, list) else v for key, v in settings.items()}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
