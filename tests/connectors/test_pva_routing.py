"""PVA routing and connection plumbing on the EPICS connector.

The connector speaks two transports: Channel Access (pyepics) for every
address, and PVAccess (p4p) for addresses matching one of the
``control_system.connector.epics.pva_channels`` globs. This file covers the
routing decision and the ``connect()`` plumbing behind it: the glob match, the
eager p4p client context, the PVA gateway environment, and — the property that
protects every existing deployment — that an absent or empty glob list leaves
the connector exactly as pure-CA as it was before.

Convention (matching ``test_epics_connector.py``): inject fake driver modules
(pyepics and p4p) instead of importing the real ones, start from an environment
with no EPICS_* variable set (the suite-wide ``restore_environ`` fixture undoes
the connector's direct ``os.environ`` writes), and assert on the
concrete payload — the env value, the context constructor argument, the
ImportError text — never merely that a call "didn't raise". p4p is not
installed on every dev machine, so the fake is what makes these tests run
everywhere.
"""

import os
import sys

import pytest

from osprey.connectors.control_system.epics_connector import EPICSConnector
from tests.connectors._epics_fakes import (
    EPICS_PVA_VARS,
    clean_epics_env,  # noqa: F401 - fixture, used by name
    fake_pyepics,  # noqa: F401 - fixture, used by name
    install_fake_p4p,
    patch_writes_enabled,
)

PVA_VARS = EPICS_PVA_VARS

# connect() runs against a stand-in pyepics: the real one would load libca and
# keep the shutdown-hook change for every later test in the worker.
pytestmark = pytest.mark.usefixtures("fake_pyepics")


def _routing_connector(*globs: str) -> EPICSConnector:
    """A connector with routing globs injected, skipping connect()."""
    connector = EPICSConnector()
    connector._pva_channel_globs = list(globs)
    return connector


# ---------------------------------------------------------------------------
# _is_pva_channel — the routing decision
# ---------------------------------------------------------------------------


class TestIsPvaChannel:
    def test_no_globs_routes_everything_to_channel_access(self):
        """A freshly constructed connector routes nothing over PVA."""
        connector = EPICSConnector()

        assert connector._pva_channel_globs == []
        assert connector._is_pva_channel("SR:CAM1:IMAGE") is False

    def test_matching_glob_routes_over_pva(self):
        connector = _routing_connector("SR:CAM*:IMAGE")

        assert connector._is_pva_channel("SR:CAM1:IMAGE") is True

    def test_non_matching_address_stays_on_channel_access(self):
        connector = _routing_connector("SR:CAM*:IMAGE")

        assert connector._is_pva_channel("SR:BEAM:CURRENT") is False

    def test_any_of_several_globs_matches(self):
        connector = _routing_connector("BL*:DET:*", "SR:CAM?:IMAGE")

        assert connector._is_pva_channel("BL7:DET:ARRAY") is True
        assert connector._is_pva_channel("SR:CAM3:IMAGE") is True
        assert connector._is_pva_channel("SR:CAM12:IMAGE") is False  # '?' is one char

    def test_match_is_case_sensitive(self):
        """fnmatchcase, not fnmatch: routing must not depend on the host OS.

        Plain ``fnmatch.fnmatch`` normcases both sides, which lowercases on
        Windows — the same address would then route differently per platform.
        """
        connector = _routing_connector("SR:CAM*:IMAGE")

        assert connector._is_pva_channel("sr:cam1:image") is False
        assert connector._is_pva_channel("SR:cam1:IMAGE") is False
        assert connector._is_pva_channel("SR:CAM1:IMAGE") is True

    def test_glob_is_anchored_at_both_ends(self):
        """fnmatch matches the WHOLE address — no accidental substring routing."""
        connector = _routing_connector("SR:CAM1:IMAGE")

        assert connector._is_pva_channel("SR:CAM1:IMAGE") is True
        assert connector._is_pva_channel("PREFIX:SR:CAM1:IMAGE") is False
        assert connector._is_pva_channel("SR:CAM1:IMAGE:SUFFIX") is False


# ---------------------------------------------------------------------------
# connect() — glob parsing and the eager p4p client context
# ---------------------------------------------------------------------------


class TestConnectPvaContext:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_globs_parsed_and_context_created_eagerly(self, monkeypatch):
        """Non-empty pva_channels: p4p is stashed and the client Context is built now.

        Eager creation (rather than on first read) is what removes the
        lazy-init race between the concurrent asyncio.to_thread reads that
        read_multiple_channels gathers.
        """
        patch_writes_enabled(monkeypatch, False)
        p4p_mod, context_cls = install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect({"pva_channels": ["SR:CAM*:IMAGE", "BL*:DET:*"]})

        assert connector._pva_channel_globs == ["SR:CAM*:IMAGE", "BL*:DET:*"]
        assert connector._p4p is p4p_mod
        context_cls.assert_called_once_with("pva")
        assert connector._pva_context is context_cls.return_value
        assert connector._is_pva_channel("BL7:DET:ARRAY") is True

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_single_glob_string_is_accepted(self, monkeypatch):
        """A scalar YAML value is normalized to a one-element glob list."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect({"pva_channels": "SR:CAM1:IMAGE"})

        assert connector._pva_channel_globs == ["SR:CAM1:IMAGE"]
        assert connector._is_pva_channel("SR:CAM1:IMAGE") is True

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_blank_entries_are_dropped(self, monkeypatch):
        """Whitespace-only list entries never become a routing pattern."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect({"pva_channels": ["  SR:CAM1:IMAGE  ", "", "   "]})

        assert connector._pva_channel_globs == ["SR:CAM1:IMAGE"]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_missing_p4p_raises_with_install_hint(self, monkeypatch):
        """PVA channels configured but p4p absent: fail fast, naming the remedy."""
        patch_writes_enabled(monkeypatch, False)
        monkeypatch.setitem(sys.modules, "p4p", None)
        monkeypatch.setitem(sys.modules, "p4p.client", None)
        monkeypatch.setitem(sys.modules, "p4p.client.thread", None)

        connector = EPICSConnector()
        with pytest.raises(ImportError) as excinfo:
            await connector.connect({"pva_channels": ["SR:CAM1:IMAGE"]})

        message = str(excinfo.value)
        assert "p4p is required" in message
        assert "pva_channels" in message
        assert "pip install p4p" in message


# ---------------------------------------------------------------------------
# connect() — no pva_channels means no PVA anything
# ---------------------------------------------------------------------------


class TestNoPvaConfigured:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_absent_pva_channels_never_imports_p4p(self, monkeypatch):
        """No pva_channels key: connect() must not touch p4p at all.

        sys.modules['p4p'] is nulled, so *any* import attempt would raise —
        connecting successfully is the proof that none was made.
        """
        patch_writes_enabled(monkeypatch, False)
        monkeypatch.setitem(sys.modules, "p4p", None)

        connector = EPICSConnector()
        await connector.connect({"gateways": {"read_only": {"address": "ro", "port": 5064}}})

        assert connector._connected is True
        assert connector._pva_channel_globs == []
        assert connector._p4p is None
        assert connector._pva_context is None
        assert connector._is_pva_channel("SR:CAM1:IMAGE") is False

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_empty_pva_channels_list_is_the_same_no_op(self, monkeypatch):
        patch_writes_enabled(monkeypatch, False)
        monkeypatch.setitem(sys.modules, "p4p", None)

        connector = EPICSConnector()
        await connector.connect({"pva_channels": []})

        assert connector._p4p is None
        assert connector._pva_context is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_empty_pva_channels_leaves_pva_env_untouched(self, monkeypatch):
        """A pva_gateway block with no pva_channels sets no environment variable.

        The glob list is the single switch: with it empty, the connector is
        byte-for-byte the pure-CA connector it was before this feature.
        """
        patch_writes_enabled(monkeypatch, False)
        monkeypatch.setitem(sys.modules, "p4p", None)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": [],
                "pva_gateway": {"address": "pvagw.example.com", "port": 5075},
            }
        )

        for var in PVA_VARS:
            assert var not in os.environ

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_channel_access_gateway_still_configured_alongside_pva(self, monkeypatch):
        """PVA routing is additive: the CA gateway env is set exactly as before."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "gateways": {"read_only": {"address": "cagw.example.com", "port": 5064}},
                "pva_channels": ["SR:CAM1:IMAGE"],
                "pva_gateway": {"address": "pvagw.example.com", "port": 5075},
            }
        )

        assert os.environ["EPICS_CA_ADDR_LIST"] == "cagw.example.com"
        assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"
        assert os.environ["EPICS_CA_AUTO_ADDR_LIST"] == "NO"
        assert os.environ["EPICS_PVA_ADDR_LIST"] == "pvagw.example.com:5075"


# ---------------------------------------------------------------------------
# connect() — pva_gateway environment
# ---------------------------------------------------------------------------


class TestPvaGatewayEnv:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_addr_list_branch_carries_the_port(self, monkeypatch):
        """PVA has no client-side server-port var — the port rides in the entry."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["SR:CAM1:IMAGE"],
                "pva_gateway": {"address": "pvagw.example.com", "port": 5085},
            }
        )

        assert os.environ["EPICS_PVA_ADDR_LIST"] == "pvagw.example.com:5085"
        assert "EPICS_PVA_NAME_SERVERS" not in os.environ

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_addr_list_branch_appends_no_port_unless_set(self, monkeypatch):
        """EPICS_PVA_ADDR_LIST entries are UDP search targets (default 5076).

        5075 is the PVA TCP port. Appending it here made p4p search on the
        wrong port and every read time out, so an unset port appends nothing
        and p4p's own default applies.
        """
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["SR:CAM1:IMAGE"],
                "pva_gateway": {"address": "pvagw.example.com"},
            }
        )

        assert os.environ["EPICS_PVA_ADDR_LIST"] == "pvagw.example.com"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_addr_list_accepts_a_space_separated_host_list(self, monkeypatch):
        """A many-server facility lists its hosts in ``address``; passed verbatim."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["*:image"],
                "pva_gateway": {"address": "10.0.0.1 10.0.0.2 cam3.example.com"},
            }
        )

        assert os.environ["EPICS_PVA_ADDR_LIST"] == "10.0.0.1 10.0.0.2 cam3.example.com"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_explicit_port_is_appended_to_every_listed_host(self, monkeypatch):
        """``port`` names the search port of each host, not a suffix on the string."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["*:image"],
                "pva_gateway": {"address": "10.0.0.1 cam2.example.com", "port": 5086},
            }
        )

        assert os.environ["EPICS_PVA_ADDR_LIST"] == "10.0.0.1:5086 cam2.example.com:5086"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_name_server_branch_still_defaults_to_5075(self, monkeypatch):
        """Name servers are TCP endpoints, where 5075 is the right default."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["SR:CAM1:IMAGE"],
                "pva_gateway": {"address": "pvagw.example.com", "use_name_server": True},
            }
        )

        assert os.environ["EPICS_PVA_NAME_SERVERS"] == "pvagw.example.com:5075"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_name_server_branch_sets_and_clears_env(self, monkeypatch):
        """use_name_server routes via EPICS_PVA_NAME_SERVERS and clears ADDR_LIST."""
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)
        monkeypatch.setenv("EPICS_PVA_ADDR_LIST", "stale.example.com:5075")

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["SR:CAM1:IMAGE"],
                "pva_gateway": {
                    "address": "localhost",
                    "port": 5085,
                    "use_name_server": True,
                },
            }
        )

        assert os.environ["EPICS_PVA_NAME_SERVERS"] == "localhost:5085"
        assert "EPICS_PVA_ADDR_LIST" not in os.environ

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_auto_addr_list_disabled_whenever_gateway_present(self, monkeypatch):
        """Containment parity with EPICS_CA_AUTO_ADDR_LIST.

        Without this, p4p broadcast-discovers the local subnet from a
        deployment that was deliberately pinned to a gateway.
        """
        patch_writes_enabled(monkeypatch, False)
        install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect(
            {
                "pva_channels": ["SR:CAM1:IMAGE"],
                "pva_gateway": {
                    "address": "localhost",
                    "port": 5085,
                    "use_name_server": True,
                },
            }
        )

        assert os.environ["EPICS_PVA_AUTO_ADDR_LIST"] == "NO"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_no_gateway_block_leaves_env_alone_but_still_builds_context(self, monkeypatch):
        """PVA channels without a gateway: default p4p discovery, context still eager."""
        patch_writes_enabled(monkeypatch, False)
        _, context_cls = install_fake_p4p(monkeypatch)

        connector = EPICSConnector()
        await connector.connect({"pva_channels": ["SR:CAM1:IMAGE"]})

        for var in PVA_VARS:
            assert var not in os.environ
        context_cls.assert_called_once_with("pva")
