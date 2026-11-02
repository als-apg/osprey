"""The shared simulator predicates — the derivations every reader agrees on.

:func:`virtual_accelerator_port` is the deployment's statement that it built a
simulator, and the key the archive's attribution reads to call the store the
``va`` target's. :func:`virtual_accelerator_active` is the recorder's gate: is
the endpoint the ``va`` target dials this deployment's own simulator container?
These tests pin its three conjuncts, and in particular the cases where a
``va`` block is *not* the deployment's simulator: a gateway on another host,
and a port other than the one the deployment published.
"""

from __future__ import annotations

import pytest

from osprey_connectors.simulator import (
    VIRTUAL_ACCELERATOR_PORT_KEY,
    virtual_accelerator_active,
    virtual_accelerator_port,
)

VA_PORT = 5064


def config_with_simulator(port: object = VA_PORT) -> dict:
    """A rendered config whose services block carries a simulator on *port*."""
    return {
        "control_system": {"type": "live_standin"},
        "services": {
            "virtual_accelerator": {"path": "./services/virtual_accelerator", "port": port}
        },
    }


class TestActive:
    """Endpoints that are the deployment's own simulator."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LocalHost", "::1", "127.0.0.5"])
    def test_loopback_host_on_the_published_port_is_the_simulator(self, host: str) -> None:
        assert virtual_accelerator_active(
            config_with_simulator(), endpoint_host=host, endpoint_port=VA_PORT
        )

    def test_whatever_machine_the_sessions_start_on(self) -> None:
        """``control_system.type`` is not read: a stand-in or live baseline still
        built the simulator, and the endpoint is judged on its own."""
        for baseline in ("live_standin", "epics", "mock", "virtual_accelerator"):
            config = config_with_simulator()
            config["control_system"]["type"] = baseline
            assert virtual_accelerator_active(
                config, endpoint_host="127.0.0.1", endpoint_port=VA_PORT
            ), baseline

    def test_an_attached_render_carrying_only_the_projected_port_is_the_simulator(self) -> None:
        attached_render = {"services": {"virtual_accelerator": {"port": VA_PORT}}}

        assert virtual_accelerator_active(
            attached_render, endpoint_host="127.0.0.1", endpoint_port=VA_PORT
        )

    def test_the_port_may_be_stated_as_text(self) -> None:
        assert virtual_accelerator_active(
            config_with_simulator("5064"), endpoint_host="127.0.0.1", endpoint_port=VA_PORT
        )


class TestNotActive:
    """Endpoints that are a real machine until the config proves otherwise."""

    def test_no_simulator_built(self) -> None:
        """An external IOC a ``va`` block happens to dial is not this deployment's."""
        assert not virtual_accelerator_active(
            {"control_system": {"type": "virtual_accelerator"}},
            endpoint_host="127.0.0.1",
            endpoint_port=VA_PORT,
        )

    def test_a_gateway_moved_off_the_host(self) -> None:
        assert not virtual_accelerator_active(
            config_with_simulator(), endpoint_host="va.example.org", endpoint_port=VA_PORT
        )

    def test_a_gateway_on_another_port(self) -> None:
        """The stand-in's port, for one: the simulator beside it is not on it."""
        assert not virtual_accelerator_active(
            config_with_simulator(), endpoint_host="127.0.0.1", endpoint_port=5074
        )

    @pytest.mark.parametrize("host", ["", "   ", "not a host", "127.0.0.1:5064", "[::1]"])
    def test_unreadable_host_fails_toward_a_real_machine(self, host: str) -> None:
        assert not virtual_accelerator_active(
            config_with_simulator(), endpoint_host=host, endpoint_port=VA_PORT
        )

    def test_unresolved_endpoint_port(self) -> None:
        assert not virtual_accelerator_active(
            config_with_simulator(), endpoint_host="127.0.0.1", endpoint_port=None
        )


class TestPort:
    """The accessor the archive's attribution and the recorder's gate share."""

    def test_present(self) -> None:
        assert virtual_accelerator_port(config_with_simulator()) == VA_PORT

    def test_absent(self) -> None:
        assert virtual_accelerator_port({"services": {}}) is None

    def test_absent_from_a_config_with_no_services_section(self) -> None:
        assert virtual_accelerator_port({"control_system": {"type": "epics"}}) is None

    @pytest.mark.parametrize("value", ["", None, "not-a-port", True, {"port": 5064}, [5064]])
    def test_values_that_name_no_port(self, value: object) -> None:
        assert virtual_accelerator_port(config_with_simulator(value)) is None

    def test_the_dotted_key_is_the_one_the_build_writes_and_projects(self) -> None:
        assert VIRTUAL_ACCELERATOR_PORT_KEY == "services.virtual_accelerator.port"
