"""Which control targets the deployment's archive serves, decided in one place.

A deployment carries **one** ``archiver:`` block, and a session can be pointed
at up to three machines — the facility's own (``live``), the virtual
accelerator (``va``) and the stand-in soft IOC (``standin``). The block holds
the past of some of them and not others. :func:`archive_targets` says which,
so that a reader offering history on a target can ask whether the store it
would open is that target's own, and answer "no archive" when it is not —
never another machine's past under the selected machine's name.

Every archiver belongs to one of two classes, and the class decides:

* ``mongodb_archiver`` is **the deployment's own store**. On a deployment that
  built a virtual accelerator it serves ``va``, whatever machine its sessions
  start on and whether or not a stand-in was built beside it; on one that built
  no simulator it serves nothing.
* Every other archiver except the mock is **the facility's archive** (the
  EPICS archiver appliance, MYA, DOOCS, or a type this table does not know)
  and serves ``live`` and ``standin`` when its block names an address, nothing
  when it does not.
* The mock, or no block at all, serves nothing.

**The facility's archive belongs to the facility's machine, and the stand-in
rehearses that machine.** It is not something a deployment stands up for its
simulation; it holds the real machine's past, and that past is the one a
stand-in session is shown, because what the stand-in rehearses is the
procedure against the real machine's history. It needs an address to count: a
block that cannot be dialled is not a deployment saying it has an archive. The
address is read per type (:data:`ADDRESS_SETTINGS`): the appliance's ``url``;
MYA and DOOCS find their server without a setting (the client library's
configured ``myquery_server``, the facility's name service), so their block
always names one.

**The mongodb store belongs to the virtual accelerator.** The deployment seeds
it with the simulator's synthesized history and the recorder samples the
simulator into it (``osprey.services.archiver_recorder``); the stand-in is
never recorded there and its offsets are never seeded there. So the store is
the ``va`` target's exactly when the deployment built a simulator — read
through :func:`osprey_connectors.simulator.virtual_accelerator_port`, the
build's own statement that it stood one up — and no other target's ever.
``control_system.type`` is not consulted: it names the machine a session
starts on, and a deployment baselined on its stand-in or on the facility's
machine still seeds and records the simulator it built.

Nothing is the honest answer for the mock, which synthesizes and has no past,
and for a block the connector factory would refuse — both settings spellings
at once, or a settings value that is not a mapping — which serves no archive,
so it claims none.

Pure: the answer is a function of the mapping alone. No environment variable,
no file and no control-context record is consulted, which is what lets a
supervisor call it at every restart and get the deployment's answer rather than
the calling process's.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from osprey_connectors.simulator import virtual_accelerator_port
from osprey_connectors.types import (
    DOOCS_ARCHIVER,
    EPICS_ARCHIVER,
    MOCK_ARCHIVER,
    MONGODB_ARCHIVER,
    MYA_ARCHIVER,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    resolve_archiver_settings,
    resolve_archiver_type,
)

#: The top-level config section selecting and configuring the archiver.
ARCHIVER_SECTION_KEY = "archiver"

#: The archiver types that are the deployment's own store. Every other type
#: except the mock is the facility's archive.
STORE_ARCHIVERS: frozenset[str] = frozenset({MONGODB_ARCHIVER})

#: The setting under which a facility archive names where it is, per type.
#: ``None``: the connector finds its server without one (MYA through the client
#: library's configured ``myquery_server``, DOOCS through the facility's name
#: service), so its block always names an address. A type not listed names it
#: under :data:`DEFAULT_ADDRESS_SETTING`.
ADDRESS_SETTINGS: Mapping[str, str | None] = MappingProxyType(
    {EPICS_ARCHIVER: "url", MYA_ARCHIVER: None, DOOCS_ARCHIVER: None}
)

#: The address setting of a facility archive :data:`ADDRESS_SETTINGS` does not list.
DEFAULT_ADDRESS_SETTING = "url"

#: The targets the facility's archive serves: the real machine and the
#: stand-in that rehearses it.
APPLIANCE_TARGETS: frozenset[str] = frozenset({TARGET_LIVE, TARGET_STANDIN})

#: The targets the deployment's own store serves: the virtual accelerator.
STORE_TARGETS: frozenset[str] = frozenset({TARGET_VA})

#: No archive for any target.
NO_TARGETS: frozenset[str] = frozenset()

__all__ = [
    "ADDRESS_SETTINGS",
    "APPLIANCE_TARGETS",
    "ARCHIVER_SECTION_KEY",
    "DEFAULT_ADDRESS_SETTING",
    "NO_TARGETS",
    "STORE_ARCHIVERS",
    "STORE_TARGETS",
    "archive_targets",
    "archive_withheld",
]


def archive_targets(config: Mapping[str, Any]) -> frozenset[str]:
    """The control targets whose history this deployment's ``archiver`` block holds.

    Args:
        config: The full config mapping, as loaded from ``config.yml`` — not the
            ``archiver:`` section, because the answer for the mongodb store
            turns on whether the deployment built a simulator, which the
            section alone does not carry.

    Returns:
        The targets per the module table — :data:`APPLIANCE_TARGETS`,
        :data:`STORE_TARGETS` or :data:`NO_TARGETS`. A target absent from the
        set has no archive: a caller holding it has nothing to offer, never a
        default to fall back on.
    """
    section = config.get(ARCHIVER_SECTION_KEY) if isinstance(config, Mapping) else None
    archiver_type = resolve_archiver_type(section)
    if archiver_type == MOCK_ARCHIVER:
        return NO_TARGETS

    try:
        settings = resolve_archiver_settings(section)
    except ValueError:
        # The factory refuses this block with the same error, so no connector
        # is ever built from it and there is no store to attribute.
        return NO_TARGETS

    if archiver_type in STORE_ARCHIVERS:
        return STORE_TARGETS if virtual_accelerator_port(config) is not None else NO_TARGETS
    return APPLIANCE_TARGETS if _names_an_address(archiver_type, settings) else NO_TARGETS


def archive_withheld(config: Mapping[str, Any], target: str) -> bool:
    """Whether a session on *target* must be told it has no archive.

    The agent's reader of the one ``archiver:`` block asks this before it
    opens the connector, and the answer is the table's: an attributed block
    whose targets do not include *target* is another machine's past, and is
    withheld. The mock is not withheld here; whether it is an honest pairing is
    the archiver pairing's question, asked at build and at the target switch.

    Args:
        config: The full config mapping, as loaded from ``config.yml``.
        target: The control target the session is on.

    Returns:
        ``True`` when the block is not the mock and does not serve *target*.
    """
    section = config.get(ARCHIVER_SECTION_KEY) if isinstance(config, Mapping) else None
    if resolve_archiver_type(section) == MOCK_ARCHIVER:
        return False
    return target not in archive_targets(config)


def _names_an_address(archiver_type: str, settings: Mapping[str, Any]) -> bool:
    """Whether a facility archive's *settings* state where it is.

    A non-blank string under the type's :data:`ADDRESS_SETTINGS` entry, or
    always for a type whose connector finds its server without one. The
    connector's own reader decides whether the address is well formed; this
    only establishes that the deployment said one.
    """
    key = ADDRESS_SETTINGS.get(archiver_type, DEFAULT_ADDRESS_SETTING)
    if key is None:
        return True
    address = settings.get(key)
    return isinstance(address, str) and bool(address.strip())
