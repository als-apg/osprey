"""Exposure and endpoint reporting for services on the host network namespace.

A host-namespace service publishes no port, so neither the wildcard-publication
check that arms the service-token rules nor the endpoint summary can see it in
any ``ports:`` block. Both learn about it from what the build rendered: the
exposure check from the variable each service declares with ``bind_env:`` (read
off the compose file its ``services.<key>`` block renders), the summary from the
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


def _write_service_compose(tmp_path, key, services):
    """Write the compose file ``services.<key>`` renders, and return its path."""
    directory = tmp_path / "build" / "services" / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "docker-compose.yml"
    path.write_text(json.dumps({"services": services}))  # JSON is valid YAML
    return str(path)


def _services(**blocks):
    """Rendered ``services.<key>`` blocks, each pointing at its own template dir."""
    return {key: {"path": f"./services/{key}", **block} for key, block in blocks.items()}


#: The pair's blocks as the build declares them on the host network.
_PAIR = _services(
    event_dispatcher={"network": "host", "bind_env": "FASTMCP_HOST"},
    dispatch_worker={"network": "host", "bind_env": "DISPATCH_WORKER_BIND"},
)


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
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
        )
        assert _host_network_bound_off_host([compose], _PAIR) == []

    def test_overridden_bind_is_reported_with_its_address(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher("0.0.0.0")}
        )
        assert _host_network_bound_off_host([compose], _PAIR) == [
            ("event-dispatcher", "binds 0.0.0.0")
        ]

    def test_bridge_mode_is_never_scanned(self, tmp_path):
        # On the compose network the bind env says nothing about host reach —
        # 0.0.0.0 there is what makes the service resolvable by service name,
        # and the network itself is the boundary. Only network_mode: host counts.
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher("0.0.0.0", host=False)}
        )
        assert _host_network_bound_off_host([compose], _PAIR) == []

    def test_host_mode_without_any_bind_var_fails_closed(self, tmp_path):
        compose = _write_service_compose(
            tmp_path,
            "event_dispatcher",
            {"event-dispatcher": _dispatcher(environment={"FASTMCP_PORT": "8020"})},
        )
        assert _host_network_bound_off_host([compose], _PAIR) == [
            (
                "event-dispatcher",
                "renders no FASTMCP_HOST, so its bind address cannot be read",
            )
        ]

    def test_environment_as_a_key_value_list_is_read(self, tmp_path):
        # Compose accepts both spellings; a hand-edited render may hold either.
        compose = _write_service_compose(
            tmp_path,
            "event_dispatcher",
            {
                "event-dispatcher": {
                    "network_mode": "host",
                    "environment": ["FASTMCP_PORT=8020", "FASTMCP_HOST=0.0.0.0"],
                }
            },
        )
        assert _host_network_bound_off_host([compose], _PAIR) == [
            ("event-dispatcher", "binds 0.0.0.0")
        ]

    def test_services_are_collected_across_files_and_sorted(self, tmp_path):
        dispatcher = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher("0.0.0.0")}
        )
        workers = _write_service_compose(
            tmp_path,
            "dispatch_worker",
            {"dispatch-worker-1": _worker("0.0.0.0"), "dispatch-worker-2": _worker()},
        )
        assert _host_network_bound_off_host([dispatcher, workers], _PAIR) == [
            ("dispatch-worker-1", "binds 0.0.0.0"),
            ("event-dispatcher", "binds 0.0.0.0"),
        ]

    def test_unreadable_and_malformed_files_are_skipped(self, tmp_path):
        missing = str(tmp_path / "never-rendered.yml")
        empty = tmp_path / "empty.yml"
        empty.write_text("")
        listy = tmp_path / "listy.yml"
        listy.write_text("- not a compose document\n")
        assert _host_network_bound_off_host([missing, str(empty), str(listy)], _PAIR) == []

    def test_a_declared_loopback_bind_stays_private(self, tmp_path):
        compose = _write_service_compose(
            tmp_path,
            "site_api",
            {"site-api": {"network_mode": "host", "environment": {"SITE_BIND": "127.0.0.1"}}},
        )
        services = _services(site_api={"network": "host", "bind_env": "SITE_BIND"})
        assert _host_network_bound_off_host([compose], services) == []

    def test_a_declared_off_host_bind_is_reported(self, tmp_path):
        compose = _write_service_compose(
            tmp_path,
            "site_api",
            {"site-api": {"network_mode": "host", "environment": {"SITE_BIND": "10.0.0.4"}}},
        )
        services = _services(site_api={"network": "host", "bind_env": "SITE_BIND"})
        assert _host_network_bound_off_host([compose], services) == [("site-api", "binds 10.0.0.4")]

    def test_a_declared_variable_the_render_lacks_fails_closed(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "site_api", {"site-api": {"network_mode": "host", "environment": {}}}
        )
        services = _services(site_api={"network": "host", "bind_env": "SITE_BIND"})
        assert _host_network_bound_off_host([compose], services) == [
            ("site-api", "renders no SITE_BIND, so its bind address cannot be read")
        ]

    def test_an_undeclared_host_service_fails_closed(self, tmp_path):
        compose = _write_service_compose(
            tmp_path,
            "site_api",
            {"site-api": {"network_mode": "host", "environment": {"SITE_BIND": "127.0.0.1"}}},
        )
        services = _services(site_api={"network": "host"})
        assert _host_network_bound_off_host([compose], services) == [
            (
                "site-api",
                "declares neither `listens: false` nor `bind_env:`, "
                "so its bind address cannot be read",
            )
        ]

    def test_listens_false_is_not_reachable(self, tmp_path):
        bridge = _write_service_compose(
            tmp_path, "teams_bridge", {"teams-bridge": {"network_mode": "host"}}
        )
        archive = _write_service_compose(tmp_path, "archive", {"archive": {"network_mode": "host"}})
        services = _services(
            teams_bridge={"network": "host", "listens": False},
            archive={"network": "host", "listens": False},
        )
        assert _host_network_bound_off_host([bridge, archive], services) == []

    def test_a_file_no_block_renders_is_undeclared(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "stray", {"stray": {"network_mode": "host", "environment": {}}}
        )
        services = _services(site_api={"network": "host", "listens": False})
        ((service, clause),) = _host_network_bound_off_host([compose], services)
        assert service == "stray"
        assert clause.startswith("declares neither")

    def test_blocks_sharing_a_path_with_different_declarations_fail_closed(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "shared", {"shared": {"network_mode": "host", "environment": {}}}
        )
        services = {
            "quiet": {"path": "./services/shared", "network": "host", "listens": False},
            "loud": {"path": "./services/shared", "network": "host", "bind_env": "SITE_BIND"},
        }
        ((service, clause),) = _host_network_bound_off_host([compose], services)
        assert service == "shared"
        assert clause.startswith("declares neither")

    def test_a_facility_variable_is_read(self, tmp_path):
        compose = _write_service_compose(
            tmp_path,
            "site_poller",
            {"site-poller": {"network_mode": "host", "environment": {"SITE_BIND": "0.0.0.0"}}},
        )
        services = _services(site_poller={"network": "host", "bind_env": "SITE_BIND"})
        assert _host_network_bound_off_host([compose], services) == [
            ("site-poller", "binds 0.0.0.0")
        ]

    def test_no_services_mapping_reads_every_host_service_as_undeclared(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
        )
        ((_service, clause),) = _host_network_bound_off_host([compose], None)
        assert clause.startswith("declares neither")


class TestReconcileExposure:
    """The third reachability clause, beside publication and the web stack."""

    def test_host_mode_on_loopback_stays_private(self, tmp_path):
        files = [
            _write_service_compose(
                tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
            ),
            _write_service_compose(tmp_path, "dispatch_worker", {"dispatch-worker-1": _worker()}),
        ]
        assert _reconcile_exposure({"services": _PAIR}, files) is False

    def test_overridden_dispatcher_bind_arms_the_token_rules(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher("0.0.0.0")}
        )
        assert _reconcile_exposure({"services": _PAIR}, [compose]) is True

    def test_a_concrete_interface_counts_as_reachable(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher("192.168.1.10")}
        )
        assert _reconcile_exposure({"services": _PAIR}, [compose]) is True

    def test_overridden_worker_bind_arms_the_token_rules(self, tmp_path):
        compose = _write_service_compose(
            tmp_path, "dispatch_worker", {"dispatch-worker-1": _worker("0.0.0.0")}
        )
        assert _reconcile_exposure({"services": _PAIR}, [compose]) is True

    def test_mixed_stack_is_exposed_by_its_one_off_host_service(self, tmp_path, caplog):
        files = [
            _write_service_compose(
                tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
            ),
            _write_service_compose(
                tmp_path,
                "dispatch_worker",
                {"dispatch-worker-1": _worker(), "dispatch-worker-2": _worker("0.0.0.0")},
            ),
        ]
        with caplog.at_level(logging.WARNING):
            assert _reconcile_exposure({"services": _PAIR}, files) is True
        # The warning names the one service that made it reachable, and only it.
        assert "dispatch-worker-2 runs on the host network and binds 0.0.0.0" in caplog.text
        assert "dispatch-worker-1" not in caplog.text

    def test_an_undeclared_host_service_names_what_is_missing(self, tmp_path, caplog):
        compose = _write_service_compose(
            tmp_path, "site_api", {"site-api": {"network_mode": "host"}}
        )
        config = {"services": _services(site_api={"network": "host"})}
        with caplog.at_level(logging.WARNING):
            assert _reconcile_exposure(config, [compose]) is True
        assert (
            "site-api runs on the host network and declares neither `listens: false` "
            "nor `bind_env:`" in caplog.text
        )

    def test_outbound_only_services_leave_the_start_private(self, tmp_path):
        files = [
            _write_service_compose(tmp_path, "archive", {"archive": {"network_mode": "host"}}),
            _write_service_compose(
                tmp_path, "teams_bridge", {"teams-bridge": {"network_mode": "host"}}
            ),
        ]
        config = {
            "services": _services(
                archive={"network": "host", "listens": False},
                teams_bridge={"network": "host", "listens": False},
            )
        }
        assert _reconcile_exposure(config, files) is False

    def test_published_loopback_ports_beside_host_mode_stay_private(self, tmp_path):
        files = [
            _write_service_compose(
                tmp_path, "postgresql", {"postgresql": {"ports": ["127.0.0.1:5432:5432"]}}
            ),
            _write_service_compose(
                tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
            ),
        ]
        assert _reconcile_exposure({"services": _PAIR}, files) is False

    def test_wildcard_publication_still_exposes(self, tmp_path):
        # The first clause, unchanged: a bridge-mode service publishing on every
        # interface is reachable however the host-mode services are bound.
        files = [
            _write_service_compose(
                tmp_path, "openobserve", {"openobserve": {"ports": ["5080:5080"]}}
            ),
            _write_service_compose(
                tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
            ),
        ]
        assert _reconcile_exposure({"services": _PAIR}, files) is True

    def test_web_terminal_stack_still_exposes(self, tmp_path):
        # The second clause, unchanged: the web stack is host-networked and its
        # nginx binds every interface, with nothing published to read.
        compose = _write_service_compose(
            tmp_path, "event_dispatcher", {"event-dispatcher": _dispatcher()}
        )
        config = {"modules": {"web_terminals": {"enabled": True}}, "services": _PAIR}
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
