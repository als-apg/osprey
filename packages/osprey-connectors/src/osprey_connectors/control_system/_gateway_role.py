"""The gateway role a connector selects, and why a readonly run cannot connect.

The one place the role rule lives. It sits in the control-system layer, the
bottom of the connectors layering, so ``EPICSConnector.connect()`` takes it
from below and :mod:`osprey_connectors.ipc` (the host report and the pool's
derivation) imports it from here rather than restating it.
"""

from __future__ import annotations

from typing import Any

ROLE_READ_ONLY = "read_only"
ROLE_WRITE_ACCESS = "write_access"


def _selected_role(gateways: dict[str, Any], *, writes_enabled: bool, readonly_run: bool) -> str:
    """The gateway role a connector for this gateways table selects — the one place the rule lives.

    ``EPICSConnector.connect()``, the connector host's report and the pool's
    derivation all take their role from here:

    * writes allowed, a ``write_access`` gateway configured and not a readonly
      run → ``write_access``;
    * a ``write_access`` gateway and no ``read_only`` one, and not a readonly
      run → ``write_access``: a machine reached only through a write gateway
      reads through it while its writes are off, and every write is refused in
      software;
    * otherwise → ``read_only``.

    A readonly run never selects ``write_access``. On a block with only a write
    gateway the ``read_only`` role it selects has no endpoint, and the run
    refuses to connect rather than reaching the network some other way.

    Takes the posture as an argument rather than resolving it: the caller holds
    the target, and one gateways table is all this selection is entitled to know
    about.
    """
    if readonly_run:
        return ROLE_READ_ONLY
    write_gateway = gateways.get(ROLE_WRITE_ACCESS)
    # Spelled out rather than ROLE_READ_ONLY: this is where the connector's
    # read_only gateway key is consumed, and the config-key manifest cites it.
    read_gateway = gateways.get("read_only", {})
    if write_gateway and (writes_enabled or not read_gateway):
        return ROLE_WRITE_ACCESS
    return ROLE_READ_ONLY


def readonly_gateway_refusal(kind: str | None = None) -> str:
    """Why a readonly run on a block with only a write gateway does not connect.

    A connect error, not a write refusal: a readonly run never selects
    ``write_access`` (:func:`_selected_role`), so on such a block it has nothing
    to read through, and the remedy is the deployment admin's — add a read
    gateway. ``EPICSConnector.connect()`` and the pool's config refusal both say
    this. *kind* names the machine, as in every refusal.
    """
    machine = f"This {kind} machine" if kind else "This machine"
    return (
        f"{machine} has no read-only gateway, and a read-only run never reads through "
        "its write gateway; the deployment admin can add a read-only gateway."
    )
