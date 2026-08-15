"""Tests for the deploy-time host-port conflict preflight."""

from __future__ import annotations

import json
import socket

import pytest

from osprey.deployment import host_ports
from osprey.deployment.host_ports import (
    HostPortBinding,
    PortConflict,
    derive_host_network_bindings,
    find_port_conflicts,
    format_conflict_report,
    parse_host_port_bindings,
)


def _host_config(**services):
    """A rendered-config stand-in carrying only the service blocks under test."""
    return {"services": services}


def _write_compose(tmp_path, name, services):
    """Write a minimal rendered compose file and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps({"services": services}))  # JSON is valid YAML
    return str(path)


@pytest.fixture
def listening_port():
    """Bind, listen, and yield a loopback ``(socket, port)``; close on teardown."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        yield sock, port
    finally:
        sock.close()


class TestParsing:
    """Parsing of the port string forms that occur in this repo's templates."""

    def test_all_string_forms(self, tmp_path):
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {
                # IP:HOST:CONTAINER
                "postgresql": {"ports": ["127.0.0.1:5432:5432"]},
                # IP:HOST:CONTAINER/proto
                "virtual-accelerator": {"ports": ["127.0.0.1:5064:5064/tcp"]},
                # HOST:CONTAINER (published on all interfaces)
                "bluesky-tiled": {"ports": ["8091:8000"]},
                # Container-only, no host publication -> skipped
                "internal": {"ports": ["9000"]},
            },
        )

        bindings = parse_host_port_bindings([compose])
        by_service = {b.service: b for b in bindings}

        assert set(by_service) == {"postgresql", "virtual-accelerator", "bluesky-tiled"}

        assert by_service["postgresql"].host_ip == "127.0.0.1"
        assert by_service["postgresql"].host_port == 5432
        assert by_service["postgresql"].container_port == 5432

        va = by_service["virtual-accelerator"]
        assert (va.host_ip, va.host_port, va.container_port) == ("127.0.0.1", 5064, 5064)

        tiled = by_service["bluesky-tiled"]
        assert (tiled.host_ip, tiled.host_port, tiled.container_port) == ("0.0.0.0", 8091, 8000)
        assert tiled.compose_file == compose

    def test_dict_long_form(self, tmp_path):
        compose = _write_compose(
            tmp_path,
            "docker-compose.yml",
            {"svc": {"ports": [{"host_ip": "127.0.0.1", "published": 8020, "target": 8020}]}},
        )
        (binding,) = parse_host_port_bindings([compose])
        assert (binding.host_ip, binding.host_port, binding.container_port) == (
            "127.0.0.1",
            8020,
            8020,
        )

    def test_unreadable_or_portless_files_are_skipped(self, tmp_path):
        no_ports = _write_compose(tmp_path, "a.yml", {"svc": {"image": "x"}})
        missing = str(tmp_path / "does-not-exist.yml")
        assert parse_host_port_bindings([no_ports, missing]) == []


class TestDuplicateDetection:
    """Static duplicate detection, isolated from whatever the host is running."""

    @pytest.fixture(autouse=True)
    def _no_external_listeners(self, monkeypatch):
        # Force the connect-probe to report every address free so these tests
        # exercise only the intra-set duplicate logic, independent of any real
        # container that happens to hold a well-known port on this host.
        monkeypatch.setattr(host_ports, "_port_is_free", lambda host_ip, host_port: True)

    def test_two_services_same_host_port(self):
        bindings = [
            HostPortBinding("postgresql", "127.0.0.1", 5432, 5432, "a.yml"),
            HostPortBinding("other-db", "127.0.0.1", 5432, 5432, "b.yml"),
        ]
        conflicts = find_port_conflicts(bindings, project_name="proj")
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.kind == "duplicate"
        assert conflict.host_port == 5432
        assert conflict.service == "other-db"
        assert "postgresql" in conflict.holder
        assert conflict.remedy == "services.other-db.port"

    def test_distinct_ports_no_conflict(self):
        bindings = [
            HostPortBinding("postgresql", "127.0.0.1", 5432, 5432, "a.yml"),
            HostPortBinding("openobserve", "127.0.0.1", 5080, 5080, "a.yml"),
        ]
        assert find_port_conflicts(bindings, project_name="proj") == []


class TestExternalConflict:
    def test_listener_detected_then_cleared(self, monkeypatch, listening_port):
        # No container attributes the port -> reported as an unknown holder.
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: "")
        sock, port = listening_port
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        conflicts = find_port_conflicts([binding], project_name="proj")
        assert len(conflicts) == 1
        assert conflicts[0].kind == "external"
        assert conflicts[0].host_port == port
        assert conflicts[0].remedy == "services.postgresql.port_host"
        assert "unknown" in conflicts[0].holder

        # Free the port and re-probe: clean.
        sock.close()
        assert find_port_conflicts([binding], project_name="proj") == []

    def test_own_project_container_is_exempt(self, monkeypatch, listening_port):
        _, port = listening_port
        ps_json = json.dumps(
            {
                "Names": "proj-ariel-postgres",
                "Ports": f"127.0.0.1:{port}->5432/tcp",
                "Labels": "com.docker.compose.project=proj",
            }
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: ps_json)
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        assert find_port_conflicts([binding], project_name="proj") == []

    def test_foreign_stack_is_attributed(self, monkeypatch, listening_port):
        _, port = listening_port
        ps_json = json.dumps(
            {
                "Names": "other-ariel-postgres",
                "Ports": f"127.0.0.1:{port}->5432/tcp",
                "Labels": "com.docker.compose.project=other",
            }
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: ps_json)
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        conflicts = find_port_conflicts([binding], project_name="proj")
        assert len(conflicts) == 1
        assert conflicts[0].kind == "external"
        assert "other-ariel-postgres" in conflicts[0].holder
        assert "other" in conflicts[0].holder

    def test_foreign_stack_podman_ps_shape(self, monkeypatch, listening_port):
        _, port = listening_port
        # Podman emits a JSON array with list Names, list Ports dicts, dict Labels.
        ps_json = json.dumps(
            [
                {
                    "Names": ["other-openobserve"],
                    "Ports": [{"host_ip": "127.0.0.1", "host_port": port, "container_port": 5080}],
                    "Labels": {"com.docker.compose.project": "other"},
                }
            ]
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: ps_json)
        binding = HostPortBinding("openobserve", "127.0.0.1", port, 5080, "a.yml")

        conflicts = find_port_conflicts([binding], project_name="proj")
        assert len(conflicts) == 1
        assert "other-openobserve" in conflicts[0].holder


class TestReport:
    def test_foreign_stack_report_suggests_sharing_the_stack(self):
        conflicts = [
            PortConflict(
                host_port=5432,
                bind_address="127.0.0.1",
                service="postgresql",
                kind="external",
                holder="container 'other-ariel-postgres' (compose project 'other')",
                remedy="services.postgresql.port_host",
            )
        ]
        report = format_conflict_report(conflicts)

        assert "1 conflict" in report
        assert "5432" in report
        assert "other-ariel-postgres" in report
        assert "services.postgresql.port_host" in report
        # Foreign-stack collisions point at attaching to the shared stack.
        assert "shared services stack" in report
        # The cancelled port_block knob is never referenced.
        assert "port_block" not in report

    def test_duplicate_report_only_suggests_config_change(self):
        conflicts = [
            PortConflict(
                host_port=8091,
                bind_address="127.0.0.1",
                service="tiled",
                kind="duplicate",
                holder="service 'bluesky-bridge'",
                remedy="services.bluesky.tiled_port",
            )
        ]
        report = format_conflict_report(conflicts)

        assert "services.bluesky.tiled_port" in report
        # No foreign stack, so no shared-stack suggestion and no port_block.
        assert "shared services stack" not in report
        assert "port_block" not in report
        # Nothing here binds on the host namespace, so that paragraph stays out.
        assert "host network namespace" not in report


class TestHostNetworkDerivation:
    """Ports of services that bind on the host namespace and publish nothing."""

    def test_bridge_and_absent_network_derive_nothing(self):
        # Bridge mode writes no `network` key at all; an explicit spelling is
        # accepted too. Neither binds a host port outside its compose network.
        assert derive_host_network_bindings(_host_config()) == []
        assert (
            derive_host_network_bindings(
                _host_config(
                    event_dispatcher={"port": 8020},
                    dispatch_worker={"worker_port_base": 9190, "worker_count": 3},
                )
            )
            == []
        )
        assert (
            derive_host_network_bindings(
                _host_config(
                    event_dispatcher={"network": "bridge", "port": 8020},
                    dispatch_worker={"network": "bridge", "worker_port_base": 9190},
                )
            )
            == []
        )

    def test_missing_or_null_blocks_derive_nothing(self):
        # A null stanza parses to None, and a config may predate the services
        # entirely; neither may raise.
        assert derive_host_network_bindings(None) == []
        assert derive_host_network_bindings({}) == []
        assert derive_host_network_bindings({"services": None}) == []
        assert derive_host_network_bindings(_host_config(event_dispatcher=None)) == []

    def test_dispatcher_binds_its_configured_port_on_loopback(self):
        (binding,) = derive_host_network_bindings(
            _host_config(event_dispatcher={"network": "host", "port": 8055})
        )
        assert binding.service == "event-dispatcher"
        assert (binding.host_ip, binding.host_port) == ("127.0.0.1", 8055)
        assert binding.host_network is True
        # It comes from the rendered config, not from any compose file.
        assert "compose" not in binding.compose_file

    def test_dispatcher_defaults_and_bind_override(self):
        (default_port,) = derive_host_network_bindings(
            _host_config(event_dispatcher={"network": "host"})
        )
        assert default_port.host_port == 8020

        (overridden,) = derive_host_network_bindings(
            _host_config(event_dispatcher={"network": "host", "port": 8020, "bind": "0.0.0.0"})
        )
        assert overridden.host_ip == "0.0.0.0"

    def test_one_port_per_worker_by_the_stride_rule(self):
        bindings = derive_host_network_bindings(
            _host_config(
                dispatch_worker={
                    "network": "host",
                    "worker_port_base": 9500,
                    "worker_port_stride": 1,
                    "worker_count": 3,
                }
            )
        )
        assert [b.service for b in bindings] == [
            "dispatch-worker-1",
            "dispatch-worker-2",
            "dispatch-worker-3",
        ]
        # port(i) = base + (i - 1) * stride, i 1-based.
        assert [b.host_port for b in bindings] == [9500, 9501, 9502]
        assert {b.host_ip for b in bindings} == {"127.0.0.1"}
        assert all(b.host_network for b in bindings)

    def test_worker_defaults_match_the_template_fallbacks(self):
        # Base, stride and count may all be absent from a hand-authored config;
        # the derivation falls back exactly the way the compose template does.
        bindings = derive_host_network_bindings(_host_config(dispatch_worker={"network": "host"}))
        assert [(b.service, b.host_port) for b in bindings] == [("dispatch-worker-1", 9190)]

    def test_custom_stride_spaces_the_workers(self):
        bindings = derive_host_network_bindings(
            _host_config(
                dispatch_worker={
                    "network": "host",
                    "worker_port_base": 9190,
                    "worker_port_stride": 10,
                    "worker_count": 3,
                }
            )
        )
        assert [b.host_port for b in bindings] == [9190, 9200, 9210]

    def test_both_halves_on_host_derive_both(self):
        bindings = derive_host_network_bindings(
            _host_config(
                event_dispatcher={"network": "host", "port": 8020},
                dispatch_worker={"network": "host", "worker_port_base": 9190, "worker_count": 2},
            )
        )
        assert [(b.service, b.host_port) for b in bindings] == [
            ("event-dispatcher", 8020),
            ("dispatch-worker-1", 9190),
            ("dispatch-worker-2", 9191),
        ]


class TestHostNetworkConflicts:
    """The derived bindings join the same duplicate and external checks."""

    @pytest.fixture
    def _no_external_listeners(self, monkeypatch):
        monkeypatch.setattr(host_ports, "_port_is_free", lambda host_ip, host_port: True)

    def test_published_port_wins_and_the_worker_is_told_to_move(self, _no_external_listeners):
        published = HostPortBinding("openobserve", "127.0.0.1", 9190, 5080, "a.yml")
        config = _host_config(dispatch_worker={"network": "host", "worker_port_base": 9190})

        conflicts = find_port_conflicts([published], project_name="proj", config=config)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.kind == "duplicate"
        assert conflict.service == "dispatch-worker-1"
        assert conflict.host_network is True
        assert "openobserve" in conflict.holder
        # One key moves the whole worker block, so the index is not in it.
        assert conflict.remedy == "dispatch.worker_port_base"

    def test_dispatcher_and_worker_on_one_port_collide(self, _no_external_listeners):
        # Two derived bindings, no compose file involved: the pair share one
        # host namespace, so a dispatcher port inside the worker range is fatal.
        config = _host_config(
            event_dispatcher={"network": "host", "port": 9191},
            dispatch_worker={"network": "host", "worker_port_base": 9190, "worker_count": 2},
        )
        conflicts = find_port_conflicts([], project_name="proj", config=config)

        assert [c.service for c in conflicts] == ["dispatch-worker-2"]
        assert conflicts[0].kind == "duplicate"
        assert "event-dispatcher" in conflicts[0].holder
        assert conflicts[0].remedy == "dispatch.worker_port_base"

    def test_bridge_mode_derives_nothing_to_collide_with(self, listening_port):
        # Same port, same config shape, only the axis differs: with no host
        # placement there is no derived binding, so nothing is probed at all.
        _, port = listening_port
        config = _host_config(event_dispatcher={"port": port})
        assert find_port_conflicts([], project_name="proj", config=config) == []

    def test_external_listener_on_a_derived_dispatcher_port(self, monkeypatch, listening_port):
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: "")
        _, port = listening_port
        config = _host_config(event_dispatcher={"network": "host", "port": port})

        conflicts = find_port_conflicts([], project_name="proj", config=config)

        assert len(conflicts) == 1
        assert conflicts[0].kind == "external"
        assert conflicts[0].service == "event-dispatcher"
        assert conflicts[0].host_network is True
        assert conflicts[0].remedy == "services.event_dispatcher.port"

    def test_own_host_network_container_is_exempt(self, monkeypatch, listening_port):
        # A host-network container publishes no port map, so `ps` cannot
        # attribute its listener by port — an idempotent redeploy has to be
        # recognised by the container's name instead.
        _, port = listening_port
        ps_json = json.dumps(
            {
                "Names": "proj-event-dispatcher",
                "Ports": "",
                "Labels": "com.docker.compose.project=proj",
            }
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: ps_json)
        config = _host_config(event_dispatcher={"network": "host", "port": port})

        assert find_port_conflicts([], project_name="proj", config=config) == []

    def test_another_projects_host_network_container_still_conflicts(
        self, monkeypatch, listening_port
    ):
        _, port = listening_port
        ps_json = json.dumps(
            {
                "Names": "other-event-dispatcher",
                "Ports": "",
                "Labels": "com.docker.compose.project=other",
            }
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: ps_json)
        config = _host_config(event_dispatcher={"network": "host", "port": port})

        conflicts = find_port_conflicts([], project_name="proj", config=config)
        assert len(conflicts) == 1
        assert conflicts[0].kind == "external"

    def test_no_config_leaves_published_checking_unchanged(self, _no_external_listeners):
        bindings = [
            HostPortBinding("postgresql", "127.0.0.1", 5432, 5432, "a.yml"),
            HostPortBinding("other-db", "127.0.0.1", 5432, 5432, "b.yml"),
        ]
        conflicts = find_port_conflicts(bindings, project_name="proj")
        assert [c.service for c in conflicts] == ["other-db"]
        assert conflicts[0].host_network is False


class TestHostNetworkReport:
    def test_report_names_the_binding_and_explains_the_namespace(self):
        conflicts = [
            PortConflict(
                host_port=9190,
                bind_address="127.0.0.1",
                service="dispatch-worker-1",
                kind="external",
                holder="an unknown host process",
                remedy="dispatch.worker_port_base",
                host_network=True,
            )
        ]
        report = format_conflict_report(conflicts)

        assert "dispatch-worker-1" in report
        assert "(host network)" in report
        assert "dispatch.worker_port_base" in report
        # The paragraph that explains why a port with no published mapping is
        # nonetheless contested by a second project on this host.
        assert "host network namespace" in report
        assert "its own ports" in report
