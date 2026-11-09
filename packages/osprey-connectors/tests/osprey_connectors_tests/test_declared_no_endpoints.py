"""A Channel Access block says "nothing to probe" only by an explicit ``gateways: []``.

The derivation carries that statement to child verification, which names it in
plain words. A missing table is not the same statement: a forgotten table must
never read as a direct connection, so it derives exactly what it did before.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.ipc import derive_endpoints, verify_host_report

_NOTHING_CONFIGURED = {
    "selected_role": None,
    "mode": None,
    "host": None,
    "port": None,
    "_epics_configured": False,
}
_ROWS = {"read_only": {"address": "gw.example.org", "port": 5064, "use_name_server": False}}


def _config(gateways: Any) -> dict[str, Any]:
    """An EPICS deployment whose block spells *gateways*, or none for ``...``."""
    block: dict[str, Any] = {"timeout_s": 5.0, "probe_channel": "LIVE:PROBE"}
    if gateways is not ...:
        block["gateways"] = gateways
    return {"control_system": {"type": "epics", "connector": {"epics": block}}}


@pytest.mark.parametrize(
    ("gateways", "declared"), [([], True), (..., False), ({}, False), (None, False), (_ROWS, False)]
)
def test_only_an_explicit_empty_list_declares_no_endpoints(gateways: Any, declared: bool) -> None:
    derivation = derive_endpoints(
        _config(gateways), "live", writes_enabled=False, readonly_run=True
    )

    assert derivation.declares_no_endpoints is declared


def test_child_verification_names_a_target_that_declares_no_endpoints() -> None:
    derivation = derive_endpoints(_config([]), "live", writes_enabled=False, readonly_run=True)

    verification = verify_host_report(derivation, _NOTHING_CONFIGURED)

    assert verification.ok is True
    assert "declares no endpoints" in verification.detail


def test_a_target_that_declares_no_endpoints_still_refuses_a_child_on_a_gateway() -> None:
    """The symmetry holds: nothing declared, yet the child came up on an endpoint."""
    derivation = derive_endpoints(_config([]), "live", writes_enabled=False, readonly_run=True)
    report = {
        "selected_role": "read_only",
        "mode": "addr_list",
        "host": "ambient.example.org",
        "port": 5064,
        "_epics_configured": True,
    }

    verification = verify_host_report(derivation, report)

    assert verification.ok is False
    assert verification.field == "_epics_configured"


def test_gateway_rows_are_verified_as_before() -> None:
    derivation = derive_endpoints(_config(_ROWS), "live", writes_enabled=False, readonly_run=True)

    verification = verify_host_report(derivation, _NOTHING_CONFIGURED)

    assert verification.ok is False
    assert verification.field == "_epics_configured"
