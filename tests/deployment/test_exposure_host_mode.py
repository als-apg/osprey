"""Exposure and endpoint reporting for services on the host network namespace.

A host-namespace service publishes no port, so neither the wildcard-publication
check that arms the service-token rules nor the endpoint summary can see it in
any ``ports:`` block. Both learn about it from what the build rendered: the
exposure check from the bind env var in the compose file, the summary from the
same config derivation the port preflight uses.
"""

from __future__ import annotations

import json
import logging

import pytest

from osprey.deployment.container_lifecycle import (
    _binds_off_host,
    _host_network_bound_off_host,
    _reconcile_exposure,
)
from osprey.deployment.deploy_summary import format_endpoint_summary


def _write_compose(tmp_path, name, services):
    """Write a minimal rendered compose file and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps({"services": services}))  # JSON is valid YAML
    return str(path)


def _dispatcher(bind="127.0.0.1", *, host=True, environment=None):
    """A rendered event-dispatcher block, host-networked by default."""
    service = {"image": "osprey/event-dispatcher:local"}
    if host:
        service["network_mode"] = "host"
    else:
        service["networks"] = ["osprey-network"]
    if environment is None:
        environment = {"FASTMCP_PORT": "8020"}
        if bind is not None:
            environment["FASTMCP_HOST"] = bind
    service["environment"] = environment
    return service


def _worker(bind="127.0.0.1", *, host=True):
    """A rendered dispatch-worker block, host-networked by default."""
    service = {"image": "project:local", "environment": {"DISPATCH_WORKER_PORT": "9190"}}
    if host:
        service["network_mode"] = "host"
    if bind is not None:
        service["environment"]["DISPATCH_WORKER_BIND"] = bind
    return service


class TestBindClassification:
    """Which bind addresses keep a host-namespace listener on this machine."""

    @pytest.mark.parametrize("address", ["127.0.0.1", "127.0.0.53", "localhost", "::1", "[::1]"])
    def test_loopback_spellings_stay_private(self, address):
        assert _binds_off_host(address) is False

    @pytest.mark.parametrize("address", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.4"])
    def test_wildcards_and_real_interfaces_are_reachable(self, address):
        assert _binds_off_host(address) is True

    def test_unusable_values_fail_closed(self):
        # An empty bind means every interface; a name or an uninterpolated
        # variable cannot be resolved here, and the guard this feeds is
        # fail-closed, so neither may be read as "private".
        assert _binds_off_host("") is True
        assert _binds_off_host("   ") is True
        assert _binds_off_host("${DISPATCHER_BIND}") is True
        assert _binds_off_host("dispatcher.example.org") is True


class TestHostNetworkBindScan:
    """Reading the rendered files for host-mode services that bind off-host."""

    def test_loopback_default_is_not_reported(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"event-dispatcher": _dispatcher()}
        )
        assert _host_network_bound_off_host([compose]) == []

    def test_overridden_bind_is_reported_with_its_address(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"event-dispatcher": _dispatcher("0.0.0.0")}
        )
        assert _host_network_bound_off_host([compose]) == [("event-dispatcher", "0.0.0.0")]

    def test_bridge_mode_is_never_scanned(self, tmp_path):
        # On the compose network the bind env says nothing about host reach —
        # 0.0.0.0 there is what makes the service resolvable by service name,
        # and the network itself is the boundary. Only network_mode: host counts.
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {"event-dispatcher": _dispatcher("0.0.0.0", host=False)},
        )
        assert _host_network_bound_off_host([compose]) == []

    def test_host_mode_without_any_bind_var_fails_closed(self, tmp_path):
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {"event-dispatcher": _dispatcher(environment={"FASTMCP_PORT": "8020"})},
        )
        assert _host_network_bound_off_host([compose]) == [("event-dispatcher", "")]

    def test_environment_as_a_key_value_list_is_read(self, tmp_path):
        # Compose accepts both spellings; a hand-edited render may hold either.
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {
                "event-dispatcher": {
                    "network_mode": "host",
                    "environment": ["FASTMCP_PORT=8020", "FASTMCP_HOST=0.0.0.0"],
                }
            },
        )
        assert _host_network_bound_off_host([compose]) == [("event-dispatcher", "0.0.0.0")]

    def test_services_are_collected_across_files_and_sorted(self, tmp_path):
        dispatcher = _write_compose(
            tmp_path, "dispatcher.yml", {"event-dispatcher": _dispatcher("0.0.0.0")}
        )
        workers = _write_compose(
            tmp_path,
            "worker.yml",
            {"dispatch-worker-1": _worker("0.0.0.0"), "dispatch-worker-2": _worker()},
        )
        assert _host_network_bound_off_host([dispatcher, workers]) == [
            ("dispatch-worker-1", "0.0.0.0"),
            ("event-dispatcher", "0.0.0.0"),
        ]

    def test_unreadable_and_malformed_files_are_skipped(self, tmp_path):
        missing = str(tmp_path / "never-rendered.yml")
        empty = tmp_path / "empty.yml"
        empty.write_text("")
        listy = tmp_path / "listy.yml"
        listy.write_text("- not a compose document\n")
        assert _host_network_bound_off_host([missing, str(empty), str(listy)]) == []


class TestReconcileExposure:
    """The third reachability clause, beside publication and the web stack."""

    def test_host_mode_on_loopback_stays_private(self, tmp_path):
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {"event-dispatcher": _dispatcher(), "dispatch-worker-1": _worker()},
        )
        assert _reconcile_exposure({}, [compose]) is False

    def test_overridden_dispatcher_bind_arms_the_token_rules(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"event-dispatcher": _dispatcher("0.0.0.0")}
        )
        assert _reconcile_exposure({}, [compose]) is True

    def test_a_concrete_interface_counts_as_reachable(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"event-dispatcher": _dispatcher("192.168.1.10")}
        )
        assert _reconcile_exposure({}, [compose]) is True

    def test_overridden_worker_bind_arms_the_token_rules(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"dispatch-worker-1": _worker("0.0.0.0")}
        )
        assert _reconcile_exposure({}, [compose]) is True

    def test_mixed_stack_is_exposed_by_its_one_off_host_service(self, tmp_path, caplog):
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {
                "event-dispatcher": _dispatcher(),
                "dispatch-worker-1": _worker(),
                "dispatch-worker-2": _worker("0.0.0.0"),
            },
        )
        with caplog.at_level(logging.WARNING):
            assert _reconcile_exposure({}, [compose]) is True
        # The warning names the one service that made it reachable, and only it.
        assert "dispatch-worker-2 runs on the host network and binds 0.0.0.0" in caplog.text
        assert "dispatch-worker-1" not in caplog.text

    def test_published_loopback_ports_beside_host_mode_stay_private(self, tmp_path):
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {
                "postgresql": {"ports": ["127.0.0.1:5432:5432"]},
                "event-dispatcher": _dispatcher(),
            },
        )
        assert _reconcile_exposure({}, [compose]) is False

    def test_wildcard_publication_still_exposes(self, tmp_path):
        # The first clause, unchanged: a bridge-mode service publishing on every
        # interface is reachable however the host-mode services are bound.
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {"openobserve": {"ports": ["5080:5080"]}, "event-dispatcher": _dispatcher()},
        )
        assert _reconcile_exposure({}, [compose]) is True

    def test_web_terminal_stack_still_exposes(self, tmp_path):
        # The second clause, unchanged: the web stack is host-networked and its
        # nginx binds every interface, with nothing published to read.
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"event-dispatcher": _dispatcher()}
        )
        config = {"modules": {"web_terminals": {"enabled": True}}}
        assert _reconcile_exposure(config, [compose]) is True


class TestEndpointSummary:
    """Host-network bound ports appear in the summary, which publishes none."""

    def test_host_mode_ports_are_listed_with_their_source(self):
        config = {
            "project_name": "demo",
            "services": {
                "event_dispatcher": {"network": "host", "port": 8020},
                "dispatch_worker": {"network": "host", "worker_port_base": 9190, "worker_count": 2},
            },
        }
        summary = format_endpoint_summary(config, [])
        assert "event-dispatcher     http://127.0.0.1:8020  (host network)" in summary
        assert "dispatch-worker-1    http://127.0.0.1:9190  (host network)" in summary
        assert "dispatch-worker-2    http://127.0.0.1:9191  (host network)" in summary

    def test_the_dispatcher_bind_override_is_where_it_answers(self):
        config = {
            "services": {"event_dispatcher": {"network": "host", "port": 8020, "bind": "0.0.0.0"}}
        }
        summary = format_endpoint_summary(config, [])
        # A wildcard bind is shown on loopback, the address it always answers
        # on — the same reduction the published bindings get.
        assert "event-dispatcher     http://127.0.0.1:8020  (host network)" in summary

    def test_published_and_host_bound_services_are_listed_together(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"postgresql": {"ports": ["127.0.0.1:5432:5432"]}}
        )
        config = {"services": {"event_dispatcher": {"network": "host", "port": 8020}}}
        summary = format_endpoint_summary(config, [compose])
        lines = [line for line in summary.splitlines() if "127.0.0.1" in line]
        assert [line.split()[0] for line in lines] == ["event-dispatcher", "postgresql"]
        assert "5432" in summary and "(host network)" not in lines[1]

    def test_bridge_mode_summary_is_unchanged(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"postgresql": {"ports": ["127.0.0.1:5432:5432"]}}
        )
        config = {"services": {"event_dispatcher": {"port": 8020}}}
        summary = format_endpoint_summary(config, [compose])
        assert "host network" not in summary
        assert "event-dispatcher" not in summary

    def test_an_unusable_config_still_summarizes_what_it_can(self, tmp_path):
        compose = _write_compose(
            tmp_path, "docker-compose.yml", {"postgresql": {"ports": ["127.0.0.1:5432:5432"]}}
        )
        summary = format_endpoint_summary({"services": None}, [compose])
        assert "postgresql" in summary
