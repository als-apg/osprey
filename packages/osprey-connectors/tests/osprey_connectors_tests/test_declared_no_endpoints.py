"""A Channel Access block says "nothing to probe" only by an explicit ``gateways: []``.

:func:`declared_gateways` is the one reader of that statement, and the
derivation takes its verdict (target eligibility and the switch gate's
reachability rung move onto it with the connectors floor bump, #2592). The
derivation carries it to child verification, which names it in plain words. A
missing table is not the same statement: a forgotten table must never read as a
direct connection, so it derives exactly what it did before.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.ipc import (
    GATEWAYS_MISSING,
    GATEWAYS_NONE_DECLARED,
    GATEWAYS_ROWS,
    declared_gateways,
    derive_endpoints,
    verify_host_report,
)

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
    ("block", "shape"),
    [
        ({"gateways": []}, GATEWAYS_NONE_DECLARED),
        ({}, GATEWAYS_MISSING),
        ({"gateways": None}, GATEWAYS_MISSING),
        ({"gateways": {}}, GATEWAYS_MISSING),
        ({"gateways": [{"address": "gw"}]}, GATEWAYS_MISSING),
        (None, GATEWAYS_MISSING),
        ("not a block", GATEWAYS_MISSING),
        ({"gateways": _ROWS}, GATEWAYS_ROWS),
    ],
)
def test_declared_gateways_classifies_the_three_spellings(block: Any, shape: str) -> None:
    assert declared_gateways(block) == shape


@pytest.mark.parametrize(
    ("gateways", "shape"),
    [([], GATEWAYS_NONE_DECLARED), (..., GATEWAYS_MISSING), (_ROWS, GATEWAYS_ROWS)],
)
def test_the_derivation_carries_the_classifier_verdict(gateways: Any, shape: str) -> None:
    derivation = derive_endpoints(
        _config(gateways), "live", writes_enabled=False, readonly_run=True
    )

    assert derivation.gateways_declared == shape


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
