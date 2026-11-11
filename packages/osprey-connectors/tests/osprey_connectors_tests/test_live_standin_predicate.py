"""The shared stand-in predicates — the derivations every reader agrees on.

:func:`live_standin_active` is the ``standin`` target's deployed-container
gate: the target eligibility check, the recorder's enablement gate and the
build's gateway derivation all ask it whether the endpoint a session would dial
is this deployment's own stand-in. These tests pin its three conjuncts, and in
particular the two cases where a loopback endpoint is *not* a stand-in: an SSH
tunnel into a real gateway, and a stale port left behind after a deployment
went live.
"""

from __future__ import annotations

import pytest

from osprey_connectors.standin import (
    LIVE_STANDIN_PORT_KEY,
    live_standin_active,
    live_standin_port,
)

STANDIN_PORT = 5074


def config_with_standin(port: object = STANDIN_PORT) -> dict:
    """A rendered config whose services block carries a stand-in on *port*."""
    return {
        "control_system": {"type": "virtual_accelerator"},
        "services": {"live_standin": {"path": "./services/virtual_accelerator", "port": port}},
    }


class TestActive:
    """Endpoints that are the deployment's own stand-in."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LocalHost", "::1", "127.0.0.5"])
    def test_loopback_host_on_the_stated_port_is_the_standin(self, host: str) -> None:
        """Every spelling of "this host" reaches the same verdict."""
        assert live_standin_active(
            config_with_standin(), endpoint_host=host, endpoint_port=STANDIN_PORT
        )

    def test_a_render_carrying_only_the_projected_port_is_the_standin(self) -> None:
        """No ``deployed_services`` conjunct: the projected port is the whole
        evidence, so an attached render answers as the single-user one does."""
        render = {"services": {"live_standin": {"port": STANDIN_PORT}}}

        assert live_standin_active(render, endpoint_host="127.0.0.1", endpoint_port=STANDIN_PORT)

    def test_the_port_may_be_stated_as_text(self) -> None:
        """A YAML-quoted port still names the port it names."""
        assert live_standin_active(
            config_with_standin("5074"), endpoint_host="127.0.0.1", endpoint_port=STANDIN_PORT
        )


class TestNotActive:
    """Endpoints that are a real machine, and must be labelled as one."""

    def test_no_services_block_at_all(self) -> None:
        """A deployment that stood no stand-in up has none to claim."""
        assert not live_standin_active(
            {"control_system": {"type": "epics"}},
            endpoint_host="127.0.0.1",
            endpoint_port=STANDIN_PORT,
        )

    def test_ssh_tunnel_to_localhost_without_a_standin_block(self) -> None:
        """A forwarded real gateway is loopback and nothing else — still LIVE."""
        tunnelled = {"services": {"postgresql": {"port": 5432}}}

        assert not live_standin_active(tunnelled, endpoint_host="localhost", endpoint_port=5064)

    @pytest.mark.parametrize(
        "port",
        [
            # The label follows the endpoint, never a leftover services block.
            pytest.param(5064, id="stale-port-after-the-deployment-went-live"),
            # No port dialled means no endpoint to match against.
            pytest.param(None, id="unresolved-endpoint-port"),
        ],
    )
    def test_a_loopback_endpoint_off_the_standin_port_is_not_active(self, port) -> None:
        assert not live_standin_active(
            config_with_standin(), endpoint_host="127.0.0.1", endpoint_port=port
        )

    @pytest.mark.parametrize(
        "host",
        [
            # An SSH-tunnel-style named gateway is off this host, so it is not ours.
            pytest.param("cagw.example.com", id="non-loopback-host-on-the-matching-port"),
            # A host this module cannot read is a machine it cannot vouch for.
            pytest.param("", id="unreadable-empty"),
            pytest.param("   ", id="unreadable-blank"),
            pytest.param("not a host", id="unreadable-not-a-host"),
            pytest.param("127.0.0.1:5074", id="unreadable-host-with-port"),
            pytest.param("[::1]", id="unreadable-bracketed-ipv6"),
        ],
    )
    def test_a_host_that_is_not_ours_to_vouch_for_fails_toward_live_machine(
        self, host: str
    ) -> None:
        assert not live_standin_active(
            config_with_standin(), endpoint_host=host, endpoint_port=STANDIN_PORT
        )


class TestPort:
    """The port accessor the build and the recorder gate read on its own."""

    def test_present(self) -> None:
        assert live_standin_port(config_with_standin()) == STANDIN_PORT

    def test_absent(self) -> None:
        assert live_standin_port({"services": {}}) is None

    def test_absent_from_a_config_with_no_services_section(self) -> None:
        assert live_standin_port({"control_system": {"type": "epics"}}) is None

    @pytest.mark.parametrize("value", ["", None, "not-a-port", True, {"port": 5074}, [5074]])
    def test_values_that_name_no_port(self, value: object) -> None:
        """Including ``true``, which asks for a stand-in without saying where."""
        assert live_standin_port(config_with_standin(value)) is None

    def test_the_dotted_key_is_the_one_the_build_projects(self) -> None:
        """Pinned because the reach contract projects this exact spelling."""
        assert LIVE_STANDIN_PORT_KEY == "services.live_standin.port"
