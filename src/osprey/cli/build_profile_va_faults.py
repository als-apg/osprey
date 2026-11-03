"""The ``virtual_accelerator.live_standin:`` refusals.

``live_standin: <port>`` stands a SECOND soft-IOC up and gives the deployment a
THIRD control target, ``standin``, configured from a block of its own
(``control_system.connector.live_standin``). ``live`` keeps meaning the
facility's authored ``epics`` block throughout — the stand-in never takes that
label — so a facility may stand the rehearsal up beside its real machine, and a
deployment may equally be baselined on the stand-in itself
(``control_system.type: live_standin``).

That makes the stand-in a port the deployment spends and a target the
deployment claims, and both can be claimed twice. The rules live here rather
than inside :meth:`BuildProfile.validate` for the reason
:func:`~osprey.cli.build_profile_archiver.va_archiver_errors` does: a block's
rules belong beside the block, but they are *reported* from validate's single
accumulator so a facility fixing a profile meets every problem it has in one
pass.

One of them is about the third target rather than about ports:

* :func:`standin_baseline_errors` — a deployment baselined on ``live_standin``
  that builds no stand-in. The baseline names a machine this build does not
  stand up, so every session would dial a port nothing serves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# The one nested-tree walker, borrowed rather than repeated for the same reason
# the path-tree builder below is.
from osprey.deployment.reach import dotted_get
from osprey.port_layout import VA_PVA_PORT_CONFIG_KEY
from osprey_connectors.types import (
    EPICS,
    LIVE_STANDIN,
    VIRTUAL_ACCELERATOR,
    resolve_control_system_type,
)

# The one path-tree builder in this package, borrowed rather than repeated: a
# `config:` block addresses the same leaf through a dotted key or a nested
# mapping, and a second implementation of "which leaves does this reach" is a
# second answer free to disagree with the renderer's.
from .build_profile_archiver import _expand_dotted
from .build_profile_schema import VAConfig

#: Key of the VA gateway table, in the nested spelling a rendered config reads.
#: Mirrors ``_VA_CONNECTOR_PATH`` in ``build_injectors``; the two ends of the
#: same block, one written by the injector and one refused here.
_VA_GATEWAYS_KEY = f"control_system.connector.{VIRTUAL_ACCELERATOR}.gateways"

#: Key of the deployment baseline's control-system section. The *type* inside
#: it is read through :func:`resolve_control_system_type` rather than by dotted
#: lookup, so an absent ``type:`` resolves the way every other reader resolves
#: it (to the mock) instead of to a second answer invented here.
_CONTROL_SYSTEM_KEY = "control_system"


def live_standin_errors(
    live_standin: int,
    va_port: int,
    claimed_ports: Mapping[str, int],
    config: Any,
) -> list[str]:
    """Every reason a profile's ``live_standin`` port cannot be built.

    Args:
        live_standin: The port ``virtual_accelerator.live_standin`` names.
        va_port: The baseline soft-IOC's port, which the stand-in may not share.
        claimed_ports: Dotted key → port for every other port this profile
            spends, from :meth:`BuildProfile._claimed_ports`.
        config: The profile's resolved ``config:`` block.

    Returns:
        The accumulated failures, empty when the stand-in validates.
    """
    errors: list[str] = []

    if not (1 <= live_standin <= 65535):
        errors.append(f"virtual_accelerator.live_standin must be in 1..65535 (got {live_standin})")
    elif live_standin == va_port:
        # The two soft-IOCs are two containers publishing on the host, so one
        # port cannot serve both — and the collision is worse than a refused
        # bind: the sandbox and the "live machine" would be the same endpoint.
        errors.append(
            f"virtual_accelerator.live_standin must differ from "
            f"virtual_accelerator.port (both {va_port})"
        )
    else:
        # Checked only for a usable port: an out-of-range value names no
        # endpoint, so "collides with" would be a second complaint about one
        # fault. Sorted so the report is stable whatever order the blocks
        # were read in.
        errors.extend(
            f"virtual_accelerator.live_standin ({live_standin}) "
            f"collides with {key} ({claimed_ports[key]})"
            for key in sorted(claimed_ports)
            if claimed_ports[key] == live_standin
        )
        errors.extend(_gateway_collision_errors(live_standin, config))
    return errors


def pva_port_errors(va: VAConfig, claimed_ports: Mapping[str, int], config: Any) -> list[str]:
    """Every reason a profile's ``virtual_accelerator.pva_port`` cannot be built.

    The rendered key is the build's to write, so a ``config:`` block that spells
    it is refused whether or not the profile sets the port.

    Args:
        va: The profile's virtual-accelerator block.
        claimed_ports: Dotted key → port for every other port this profile
            spends, from :meth:`BuildProfile._claimed_ports`.
        config: The profile's resolved ``config:`` block.

    Returns:
        The accumulated failures, empty when the port validates or is unset.
    """
    errors: list[str] = []

    if dotted_get(_expand_dotted(config), VA_PVA_PORT_CONFIG_KEY) is not None:
        errors.append(
            f"config: sets {VA_PVA_PORT_CONFIG_KEY}, which the build writes from "
            "virtual_accelerator.pva_port and would overwrite. "
            "Set virtual_accelerator.pva_port instead."
        )

    pva_port = va.pva_port
    if pva_port is None:
        return errors
    if isinstance(pva_port, bool) or not isinstance(pva_port, int):
        errors.append(f"virtual_accelerator.pva_port must be a port number (got {pva_port!r})")
        return errors
    if not (1 <= pva_port <= 65535):
        errors.append(f"virtual_accelerator.pva_port must be in 1..65535 (got {pva_port})")
        return errors

    # Both servers bind TCP in one container and publish on one host interface.
    if pva_port == va.port:
        errors.append(
            f"virtual_accelerator.pva_port must differ from "
            f"virtual_accelerator.port (both {pva_port})"
        )
    if pva_port == va.live_standin:
        errors.append(
            f"virtual_accelerator.pva_port must differ from "
            f"virtual_accelerator.live_standin (both {pva_port})"
        )
    errors.extend(
        f"virtual_accelerator.pva_port ({pva_port}) collides with {key} ({claimed_ports[key]})"
        for key in sorted(claimed_ports)
        if claimed_ports[key] == pva_port
    )
    return errors


def _gateway_collision_errors(live_standin: int, config: Any) -> list[str]:
    """Refuse a hand-authored VA gateway sitting on the stand-in's port.

    The VA gateways are how a session dials the *simulation*; the stand-in is
    what ``live`` dials. A profile that points both at one endpoint has written
    a deployment where switching target changes the label and nothing else,
    which is the single thing the stand-in exists to make impossible.

    Read spelling-independently, because the renderer honors a dotted key and a
    nested mapping alike and either could be the one that lands.
    """
    node = dotted_get(_expand_dotted(config), _VA_GATEWAYS_KEY)
    if not isinstance(node, dict):
        return []

    errors: list[str] = []
    for role in sorted(node):
        row = node[role]
        if not isinstance(row, dict):
            continue
        port = row.get("port")
        if isinstance(port, int) and not isinstance(port, bool) and port == live_standin:
            dotted = f"{_VA_GATEWAYS_KEY}.{role}.port"
            errors.append(
                f"virtual_accelerator.live_standin ({live_standin}) collides with the "
                f"profile's `config:` {dotted} ({port}) — the virtual accelerator and "
                f"its live stand-in are two endpoints, never one"
            )
    return errors


def standin_baseline_errors(config: Any, va: VAConfig | None) -> list[str]:
    """Refuse a deployment baselined on a stand-in it does not build.

    ``control_system.type: live_standin`` is a legal baseline — the deployment
    that runs the soft IOC may also start every session on it — but only where
    the deployment actually stands one up. Without
    ``virtual_accelerator.live_standin`` the baseline names a machine no
    container serves, and the failure surfaces as a connector dialing a port
    nothing is listening on, one ``osprey up`` later.

    Args:
        config: The profile's resolved ``config:`` block.
        va: The parsed ``virtual_accelerator:`` block, or ``None`` when the
            profile declares none — which is itself a way to reach this fault.

    Returns:
        The single failure, or an empty list.
    """
    if _baseline_type(config) != LIVE_STANDIN:
        return []
    if va is not None and va.live_standin is not None:
        return []
    return [
        f"control_system.type: {LIVE_STANDIN} with no "
        f"virtual_accelerator.live_standin — the baseline names a machine this "
        f"deployment does not stand up, so every session would dial a port nothing "
        f"serves. Set `virtual_accelerator.live_standin` to the port the stand-in "
        f"should serve, or set `control_system.type` back to the connector that "
        f"reaches this machine (`{EPICS}` for a facility's own)."
    ]


def _baseline_type(config: Any) -> str:
    """The control-system type a profile's ``config:`` block selects.

    Read spelling-independently — the renderer honors a dotted key and a nested
    mapping alike — and resolved through the connector vocabulary's own
    resolver, so a profile that says nothing about its control system gets the
    same answer here as the factory gives it at runtime.
    """
    baseline: str = resolve_control_system_type(
        dotted_get(_expand_dotted(config), _CONTROL_SYSTEM_KEY)
    )
    return baseline
