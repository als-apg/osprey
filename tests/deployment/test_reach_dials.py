"""Each Reach Contract consumer dials what its own client would.

The ``dial`` of every consumer in :data:`osprey.deployment.reach.REACH_CONTRACTS`
goes through the client's real resolver — the qmd base URL, the graph
connection, the ARIEL DSN, the OTLP endpoint, the bridge URL, the panel URL,
the VA gateway fill — so these tests pin the resolver-against-config seam:
a moved key moves the dial, an env override wins where the client honours
one, and a client with nothing to dial says ``None``.
"""

from __future__ import annotations

import pytest

from osprey.bluesky_bridge_connection import SECOND_LANE_KEYS, lane_env_prefix
from osprey.deployment.reach import REACH_CONTRACTS, reach_dials


def _dial(service: str, config: dict, index: int = 0):
    consumer = REACH_CONTRACTS[service].consumers[index]
    assert consumer.dial is not None, f"{service} consumer {consumer.name} declares no dial"
    return consumer.dial(config)


class TestQmd:
    def test_follows_the_published_port_on_loopback(self):
        assert _dial("qmd", {"services": {"qmd": {"port": 9180}}}) == ("127.0.0.1", 9180)

    def test_both_consumers_share_the_resolver(self):
        config = {"services": {"qmd": {"port": 9180}}}
        assert _dial("qmd", config, 0) == _dial("qmd", config, 1)

    def test_no_block_is_nothing_to_dial(self):
        assert _dial("qmd", {}) is None

    def test_a_malformed_block_is_nothing_to_dial(self):
        """The resolver refuses the block; the client would too."""
        assert _dial("qmd", {"services": {"qmd": {"port": "eighty"}}}) is None


class TestGraphdb:
    def test_derives_bolt_from_the_published_port(self):
        assert _dial("graphdb", {"services": {"graphdb": {"port_host": 7688}}}) == (
            "localhost",
            7688,
        )

    def test_an_explicit_uri_is_dialed_as_written(self):
        config = {"services": {"graphdb": {"uri": "bolt://graph.example:7000"}}}
        assert _dial("graphdb", config) == ("graph.example", 7000)

    def test_no_store_is_nothing_to_dial(self):
        assert _dial("graphdb", {}) is None


class TestPostgresql:
    def test_follows_the_published_port(self):
        config = {"ariel": {}, "services": {"postgresql": {"port_host": 15432}}}
        assert _dial("postgresql", config) == ("localhost", 15432)

    def test_an_explicit_uri_wins(self):
        config = {"ariel": {"database": {"uri": "postgresql://u:p@db.example:6543/ariel"}}}
        assert _dial("postgresql", config) == ("db.example", 6543)


class TestOpenobserve:
    CONFIG = {
        "claude_code": {"telemetry": {"enabled": True, "backend": "openobserve"}},
        "services": {"openobserve": {"port": 15080}},
    }

    def test_follows_the_published_port(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OTEL_OPENOBSERVE_HOST", raising=False)
        monkeypatch.delenv("OSPREY_OTEL_OPENOBSERVE_PORT", raising=False)
        monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
        host, port = _dial("openobserve", self.CONFIG)
        assert port == 15080

    def test_the_compose_authors_host_wins(self, monkeypatch):
        """A host-networked persona container is told 127.0.0.1 by its compose
        block; the exporter, and so the knock, go there."""
        monkeypatch.setenv("OSPREY_OTEL_OPENOBSERVE_HOST", "127.0.0.1")
        monkeypatch.delenv("OSPREY_OTEL_OPENOBSERVE_PORT", raising=False)
        assert _dial("openobserve", self.CONFIG) == ("127.0.0.1", 15080)

    def test_an_explicit_endpoint_is_dialed_as_written(self):
        config = {
            "claude_code": {
                "telemetry": {"enabled": True, "endpoint": "http://collector.example:4318"}
            }
        }
        assert _dial("openobserve", config) == ("collector.example", 4318)

    def test_telemetry_off_is_nothing_to_dial(self):
        assert _dial("openobserve", {"claude_code": {"telemetry": {"enabled": False}}}) is None


class TestBluesky:
    def test_follows_the_published_port(self, monkeypatch):
        monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
        assert _dial("bluesky", {"services": {"bluesky": {"port": 18090}}}) == (
            "127.0.0.1",
            18090,
        )

    def test_the_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", "http://bridge:9/")
        assert _dial("bluesky", {"services": {"bluesky": {"port": 18090}}}) == ("bridge", 9)


@pytest.mark.parametrize(
    ("lane", "prefix"),
    [(lane, lane_env_prefix(lane)) for lane in sorted(SECOND_LANE_KEYS.values())],
)
class TestSecondLane:
    """A second plan lane resolves like ``resolve_bridge_url(lane)``: its own
    ``<PREFIX>_BRIDGE_URL`` outright, else its published port on loopback —
    and, having no ``bluesky.bridge_url`` and no default of its own, nothing
    at all without a port.

    Parametrized from :data:`SECOND_LANE_KEYS`, the one registry of lane keys,
    so a lane added there for a new control target — the stand-in's — is dialed
    here without this file naming it.
    """

    def test_follows_the_lanes_published_port(self, lane, prefix, monkeypatch):
        monkeypatch.delenv(f"{prefix}_BRIDGE_URL", raising=False)
        assert _dial(lane, {"services": {lane: {"port": 18190, "target": "live"}}}) == (
            "127.0.0.1",
            18190,
        )

    def test_the_lanes_own_env_override_wins(self, lane, prefix, monkeypatch):
        monkeypatch.setenv(f"{prefix}_BRIDGE_URL", "http://lane-bridge:9/")
        assert _dial(lane, {"services": {lane: {"port": 18190}}}) == ("lane-bridge", 9)

    def test_lane_ones_override_is_not_this_lanes(self, lane, prefix, monkeypatch):
        """``BLUESKY_BRIDGE_URL`` names lane 1's bridge; a second lane dialing it
        would send the other machine's plans to the baseline's bridge."""
        monkeypatch.delenv(f"{prefix}_BRIDGE_URL", raising=False)
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", "http://lane-one:10080")
        assert _dial(lane, {"services": {lane: {"port": 18190}}}) == ("127.0.0.1", 18190)

    def test_no_port_is_nothing_to_dial(self, lane, prefix, monkeypatch):
        monkeypatch.delenv(f"{prefix}_BRIDGE_URL", raising=False)
        assert _dial(lane, {"services": {lane: {"target": "live"}}}) is None
        assert _dial(lane, {}) is None


@pytest.mark.parametrize(
    ("service", "panel"), [("bluesky_web", "bluesky"), ("event_dispatcher", "events")]
)
class TestPanels:
    def test_dials_the_panel_url(self, service, panel):
        config = {"web": {"panels": {panel: {"url": "http://127.0.0.1:8097"}}}}
        assert _dial(service, config) == ("127.0.0.1", 8097)

    def test_no_url_is_nothing_to_dial(self, service, panel):
        assert _dial(service, {"web": {"panels": {panel: {"enabled": True}}}}) is None


class TestVirtualAccelerator:
    def test_follows_the_published_port_on_the_gateway_address(self):
        config = {
            "services": {"virtual_accelerator": {"port": 15064}},
            "control_system": {
                "connector": {
                    "virtual_accelerator": {"gateways": {"read_only": {"address": "localhost"}}}
                }
            },
        }
        assert _dial("virtual_accelerator", config) == ("localhost", 15064)

    def test_an_explicit_gateway_port_wins(self):
        """fill_gateway_ports' own precedence: a gateway that names its port
        is pointed at a VA this project does not deploy."""
        config = {
            "services": {"virtual_accelerator": {"port": 15064}},
            "control_system": {
                "connector": {
                    "virtual_accelerator": {
                        "gateways": {"read_only": {"address": "va.example", "port": 5074}}
                    }
                }
            },
        }
        assert _dial("virtual_accelerator", config) == ("va.example", 5074)

    def test_the_connector_default_when_nothing_is_configured(self):
        from osprey_connectors.control_system.va_connector import DEFAULT_VA_PORT

        assert _dial("virtual_accelerator", {}) == ("localhost", DEFAULT_VA_PORT)


class TestLiveStandin:
    """The ``standin`` target's own block, dialed the EPICS connector's way.

    ``control_system.connector.live_standin`` is what the ``live_standin`` type
    is configured from, and the facility's ``epics`` block — the ``live``
    target — is never read here: the stand-in is a third machine, not a mode of
    the first.
    """

    @staticmethod
    def _config(standin: dict, epics: dict | None = None) -> dict:
        connector: dict = {"live_standin": standin}
        if epics is not None:
            connector["epics"] = epics
        return {"control_system": {"connector": connector}}

    def test_follows_the_read_only_gateway(self):
        config = self._config(
            {"gateways": {"read_only": {"address": "localhost", "port": 5074}}},
            epics={"gateways": {"read_only": {"address": "gateway.facility.org", "port": 5064}}},
        )
        assert _dial("live_standin", config) == ("localhost", 5074)

    def test_a_gateway_naming_no_port_falls_back_to_the_ca_default(self):
        """The connector's own fallback (``_configure_epics_env``), so the dial
        is the endpoint it would really use rather than a guess."""
        from osprey.deployment.reach import _EPICS_DEFAULT_CA_PORT

        config = self._config({"gateways": {"read_only": {"address": "localhost"}}})
        assert _dial("live_standin", config) == ("localhost", _EPICS_DEFAULT_CA_PORT)

    def test_no_gateways_is_nothing_to_dial(self):
        assert _dial("live_standin", self._config({"probe_channel": "X:Y"})) is None
        assert _dial("live_standin", {}) is None

    def test_the_facility_block_is_not_the_standins(self):
        """An ``epics`` block alone is a deployment with a live machine and no
        stand-in connector — nothing for this dial to answer with."""
        epics_only = {
            "control_system": {
                "connector": {
                    "epics": {"gateways": {"read_only": {"address": "localhost", "port": 5074}}}
                }
            }
        }
        assert _dial("live_standin", epics_only) is None


def test_every_consumer_declares_a_dial():
    """The health category can only knock on what the registry can dial;
    a consumer without one would be silently absent from the readout."""
    missing = [
        (contract.service, consumer.name)
        for contract in REACH_CONTRACTS.values()
        for consumer in contract.consumers
        if consumer.dial is None
    ]
    assert missing == []


def test_reach_dials_lists_only_live_consumers():
    config = {"ariel": {"search_modules": {"hybrid": {"enabled": True}}}}
    assert [(c.service, d) for c, _k, d in reach_dials(config) if c.service == "qmd"] == [
        ("qmd", None)
    ]
    assert all(c.service != "bluesky" for c, _k, _d in reach_dials(config))
