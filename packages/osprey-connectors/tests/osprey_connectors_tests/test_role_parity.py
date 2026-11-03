"""The gateway role is decided in one function, and every caller answers with it.

:func:`osprey_connectors.ipc._verification._selected_role` holds the rule: writes
allowed through a configured write gateway; a machine reached only through a write
gateway reads through it while off; a readonly run never selects the write gateway.
``EPICSConnector.connect()``, the connector host's report (``_host._rule_role``) and the
pool's derivation take their role from it rather than restating it, so for every
gateways table, write posture and run mode they agree with it. A readonly run on a
write-gateway-only block selects ``read_only``, which has no endpoint there: the pool
refuses to spawn it and ``connect()`` refuses to connect, both with one connect error
saying the machine has no read-only gateway and the deployment admin can add one —
not the ``readonly_run`` write refusal.
"""

import inspect
import os

import pytest

import osprey_connectors.control_system._epics_connector as epics_connector_module
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.ipc import _host, _pool
from osprey_connectors.ipc._pool import ConnectorHostPool
from osprey_connectors.ipc._verification import _selected_role, derive_endpoints
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._epics_fakes import (  # noqa: F401 - fixtures, used by name
    InlineCaLoop,
    clean_epics_env,
    install_fake_aioca,
)

READ = {"address": "ro.example.org", "port": 5064}
WRITE = {"address": "rw.example.org", "port": 5065}

GATEWAYS = {
    "both": {"read_only": dict(READ), "write_access": dict(WRITE)},
    "read-only": {"read_only": dict(READ)},
    "write-only": {"write_access": dict(WRITE)},
    "empty-write": {"read_only": dict(READ), "write_access": {}},
    # An empty read row: _selected_role's truthiness test and derive_endpoints' row
    # filter must agree that it is no read gateway.
    "empty-read": {"read_only": {}, "write_access": dict(WRITE)},
}

CASES = [
    pytest.param(shape, writes, readonly, id=f"{shape}-{'on' if writes else 'off'}-{mode}")
    for shape in GATEWAYS
    for writes in (False, True)
    for readonly, mode in ((False, "run"), (True, "readonly"))
]


def _expected(shape: str, writes: bool, readonly: bool) -> str:
    return _selected_role(GATEWAYS[shape], writes_enabled=writes, readonly_run=readonly)


# -- The rule itself ------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "writes", "readonly", "role"),
    [
        ("both", True, False, "write_access"),
        ("both", False, False, "read_only"),
        ("both", True, True, "read_only"),
        ("write-only", False, False, "write_access"),
        ("write-only", True, False, "write_access"),
        ("write-only", False, True, "read_only"),
        ("write-only", True, True, "read_only"),
        ("read-only", True, False, "read_only"),
        ("empty-write", True, False, "read_only"),
        ("empty-read", False, False, "write_access"),
        ("empty-read", True, False, "write_access"),
        ("empty-read", False, True, "read_only"),
    ],
)
def test_the_rule(shape, writes, readonly, role):
    assert _selected_role(GATEWAYS[shape], writes_enabled=writes, readonly_run=readonly) == role


# -- Every caller answers with it ------------------------------------------------


@pytest.mark.parametrize(("shape", "writes", "readonly"), CASES)
def test_the_host_report_rule_is_the_selected_role(shape, writes, readonly):
    assert _host._rule_role(GATEWAYS[shape], writes, readonly) == _expected(shape, writes, readonly)


def test_the_host_report_rule_selects_nothing_for_no_gateways():
    assert _host._rule_role({}, True, False) is None


@pytest.mark.parametrize(("shape", "writes", "readonly"), CASES)
def test_the_pool_derivation_is_the_selected_role(monkeypatch, tmp_path, shape, writes, readonly):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.setattr(ConnectorHostPool, "_writes_allowed", staticmethod(lambda *_: writes))
    section = {
        "type": "epics",
        "writes": "allowed",
        "connector": {"epics": {"gateways": GATEWAYS[shape]}},
    }
    pool = ConnectorHostPool(section)
    key = ("live", "readonly" if readonly else None)

    derivation, _, _ = pool._derive(key, pool._section)
    expected = _expected(shape, writes, readonly)

    assert derivation.selected_role == expected
    refusal = pool._config_refusal(key, pool._section, derivation)
    if GATEWAYS[shape].get(expected):
        assert refusal is None
    else:
        assert refusal is not None
        assert "no read-only gateway" in refusal
        assert "deployment admin" in refusal
        # A connect error, not a write refusal: the chip cannot fix it.
        assert refusal_phrase("readonly_run") not in refusal


@pytest.mark.usefixtures("clean_epics_env")
@pytest.mark.asyncio
@pytest.mark.parametrize(("shape", "writes", "readonly"), CASES)
async def test_connect_selects_the_selected_role(monkeypatch, shape, writes, readonly):
    install_fake_aioca(monkeypatch)
    loop = InlineCaLoop()
    monkeypatch.setattr(epics_connector_module, "get_ca_loop", lambda: loop)
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda _self: writes))
    if readonly:
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    else:
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    gateways = GATEWAYS[shape]
    expected = _expected(shape, writes, readonly)
    connector = EPICSConnector()

    if gateways.get(expected):
        await connector.connect({"gateways": gateways})
        assert connector._selected_role == expected
        assert os.environ["EPICS_CA_ADDR_LIST"] == gateways[expected]["address"]
    else:
        with pytest.raises(ConnectionError) as caught:
            await connector.connect({"gateways": gateways})
        assert "no read-only gateway" in str(caught.value)
        assert "deployment admin" in str(caught.value)
        # A connect error, not a write refusal: the chip cannot fix it.
        assert refusal_phrase("readonly_run") not in str(caught.value)
        assert loop.pointed_at == []


# -- An omitted posture is the tier ---------------------------------------------


@pytest.mark.parametrize(
    ("tier", "type_key", "role"),
    [
        ("never", True, "read_only"),
        ("allowed", False, "write_access"),
    ],
    ids=["tier-never-key-on", "tier-allowed-key-off"],
)
def test_an_omitted_writes_enabled_follows_the_tier_not_the_type_key(tier, type_key, role):
    """derive_endpoints' default is the tier; a contradicting per-type key is ignored."""
    config = {
        "control_system": {
            "type": "epics",
            "writes": tier,
            "connector": {
                "epics": {
                    "writes_enabled": type_key,
                    "gateways": {"read_only": dict(READ), "write_access": dict(WRITE)},
                }
            },
        }
    }

    derivation = derive_endpoints(config, "live", readonly_run=False)

    assert derivation.selected_role == role


# -- Only _selected_role contains it ---------------------------------------------


def test_the_pool_restates_no_role():
    assert not hasattr(_pool, "_read_through_only_gateway")


@pytest.mark.parametrize(
    "caller", [_host._rule_role, EPICSConnector.connect], ids=["host-report", "connect"]
)
def test_each_caller_delegates_to_the_selected_role(caller):
    source = inspect.getsource(caller)

    assert "_selected_role(" in source
    # The read-through clause and the readonly guard are not restated.
    assert "not read_gateway" not in source
    assert 'gateways.get("read_only")' not in source
    assert "not readonly_run" not in source
