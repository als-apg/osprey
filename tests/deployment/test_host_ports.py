"""Tests for the deploy-time host-port conflict preflight."""

from __future__ import annotations

import json
import socket

import pytest

from osprey.deployment import host_ports
from osprey.deployment.compose_generator import repo_identity
from osprey.deployment.graphdb_service import (
    CONTAINER_BOLT_PORT,
    CONTAINER_HTTP_PORT,
    GRAPHDB_HTTP_PORT_CONFIG_KEY,
    GRAPHDB_PORT_CONFIG_KEY,
    GRAPHDB_SERVICE_NAME,
)
from osprey.deployment.host_ports import (
    HostPortBinding,
    PortConflict,
    derive_host_network_bindings,
    find_port_conflicts,
    format_conflict_report,
    parse_host_port_bindings,
)
from osprey.port_layout import (
    CA_DEFAULT_PORT,
    DEFAULT_PORT_BASE,
    PORT_BASE_CONFIG_KEY,
    SLOTS_BY_NAME,
    block_range,
    default_port,
)

#: A second base, far from the default, for the two-deployments-on-one-host
#: scenario. Every expectation below is derived from it rather than written out,
#: so a layout change moves the test with the code instead of against it.
SECOND_BASE = 20000

#: The identity label the render stamps on every container it creates. Spelled
#: here rather than read off a golden: this file must pin what the preflight
#: does with the label, which is a different question from whether the exemplar
#: profile carries one.
REPO_ID_LABEL = "com.osprey.repo-id"


def _host_config(**services):
    """A rendered-config stand-in carrying only the service blocks under test."""
    return {"services": services}


def _based_config(base, **services):
    """A rendered-config stand-in that also names a port base."""
    return {"deployment": {"port_base": base}, "services": services}


def _ps_row(name, project, port=None, repo_id=None):
    """One docker-shaped ``ps`` row, with the labels this deployment reads.

    Args:
        name: Container name.
        project: ``com.docker.compose.project`` label value.
        port: Published host port, or ``None`` for a host-network container,
            which maps nothing and so is attributed by name instead.
        repo_id: ``com.osprey.repo-id`` label value, or ``None`` to leave the
            label off entirely, as a container an older OSPREY created does.

    Returns:
        A dict ready to be JSON-encoded as runtime ``ps`` output.
    """
    labels = [f"com.docker.compose.project={project}"]
    if repo_id is not None:
        labels.append(f"{REPO_ID_LABEL}={repo_id}")
    return {
        "Names": name,
        "Ports": f"127.0.0.1:{port}->5432/tcp" if port is not None else "",
        "Labels": ",".join(labels),
    }


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
                "bluesky-tiled": {"ports": ["10070:8000"]},
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
        assert (tiled.host_ip, tiled.host_port, tiled.container_port) == ("0.0.0.0", 10070, 8000)
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
                host_port=10070,
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

    def test_a_live_standin_collision_names_the_standin_key(self):
        """A deployment with a stand-in publishes two virtual-accelerator ports,
        so the remedy has to name the one that moves the contested container."""
        conflicts = [
            PortConflict(
                host_port=5074,
                bind_address="127.0.0.1",
                service="live-standin",
                kind="duplicate",
                holder="service 'virtual-accelerator'",
                remedy=host_ports._remedy_for_service("live-standin"),
            )
        ]
        report = format_conflict_report(conflicts)

        assert "Set a different services.live_standin.port." in report
        # Never the other instance's key: moving it would leave this collision.
        assert "services.virtual_accelerator.port" not in report

    def test_the_report_carries_no_em_dash_asides(self):
        """The house style for printed copy, pinned where the guard cannot see it.

        ``tests/cli/test_printed_copy_style.py`` reads the arguments of printing
        calls, and this report reaches the terminal as a variable instead: it is
        built here, returned, and handed to ``output.fail`` by
        ``container_lifecycle._report_port_conflicts``. So the em-dash rule has
        to be pinned at the source, or it is not pinned at all.
        """
        conflicts = [
            PortConflict(
                host_port=5432,
                bind_address="127.0.0.1",
                service="postgresql",
                kind="external",
                holder="container 'other-ariel-postgres' (compose project 'other')",
                remedy="services.postgresql.port_host",
            ),
            PortConflict(
                host_port=9190,
                bind_address="127.0.0.1",
                service="dispatch-worker-1",
                kind="duplicate",
                holder="service 'dispatch-worker-0'",
                remedy="dispatch.worker_port_base",
                host_network=True,
            ),
        ]

        # Both branches of every paragraph, so no wording is exempt by luck.
        # Each call carries a positive anchor too: an absence assertion alone
        # stays green on a report that degenerated to an empty string, which
        # would read as clean copy while printing nothing at all.
        both = format_conflict_report(conflicts)
        assert "shared services stack" in both and "host network namespace" in both
        assert "—" not in both

        foreign_only = format_conflict_report(conflicts[:1])
        assert "shared services stack" in foreign_only
        assert "—" not in foreign_only

        host_network_only = format_conflict_report(conflicts[1:])
        assert "host network namespace" in host_network_only
        assert "—" not in host_network_only

        # The block framing, the one-knob paragraph and the Channel Access
        # exception are printed copy too, and reach the terminal the same way.
        block_framed = format_conflict_report(
            [
                PortConflict(
                    host_port=default_port("postgres"),
                    bind_address="127.0.0.1",
                    service="postgresql",
                    kind="external",
                    holder="container 'other-postgres' (compose project 'other')",
                    remedy=PORT_BASE_CONFIG_KEY,
                    port_base=DEFAULT_PORT_BASE,
                ),
                PortConflict(
                    host_port=CA_DEFAULT_PORT,
                    bind_address="127.0.0.1",
                    service="virtual-accelerator",
                    kind="external",
                    holder="container 'other-va' (compose project 'other')",
                    remedy="services.virtual_accelerator.port",
                    port_base=DEFAULT_PORT_BASE,
                    channel_access=True,
                ),
            ]
        )
        assert "This deployment's block is ports" in block_framed
        assert "moves all of them at once" in block_framed
        assert "Channel Access protocol" in block_framed
        assert "—" not in block_framed


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
        (defaulted,) = derive_host_network_bindings(
            _host_config(event_dispatcher={"network": "host"})
        )
        assert defaulted.host_port == default_port("dispatcher")

        (overridden,) = derive_host_network_bindings(
            _host_config(event_dispatcher={"network": "host", "port": 8020, "bind": "0.0.0.0"})
        )
        assert overridden.host_ip == "0.0.0.0"

    def test_the_dispatcher_default_follows_this_deployments_base(self):
        """The one rule: a port comes from the base the deployment resolved.

        A config that moved its block and left the dispatcher key absent binds
        inside its OWN block, so preflighting the default base would probe a
        port this deployment never touches and miss the one it does.
        """
        (binding,) = derive_host_network_bindings(
            _based_config(SECOND_BASE, event_dispatcher={"network": "host"})
        )
        assert binding.host_port == default_port("dispatcher", base=SECOND_BASE)
        assert binding.host_port != default_port("dispatcher")

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
        # the derivation falls back exactly the way the compose template does,
        # which is now the layout's worker band rather than a literal.
        bindings = derive_host_network_bindings(_host_config(dispatch_worker={"network": "host"}))
        assert [(b.service, b.host_port) for b in bindings] == [
            ("dispatch-worker-1", default_port("worker", 1))
        ]

    def test_the_worker_band_follows_this_deployments_base(self):
        """Worker w reads off the port at whatever base the deployment set."""
        bindings = derive_host_network_bindings(
            _based_config(SECOND_BASE, dispatch_worker={"network": "host", "worker_count": 3})
        )
        assert [b.host_port for b in bindings] == [
            default_port("worker", w, base=SECOND_BASE) for w in (1, 2, 3)
        ]

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

    def test_a_facility_host_mode_service_derives_from_its_port_key(self):
        """The docs invite a facility to put its own service on the host
        network; the preflight and the deploy summary must see it there.
        Derived from ``services.<name>.port`` — the same per-service
        convention the generic remedy key already assumes."""
        (binding,) = derive_host_network_bindings(
            _host_config(my_ioc_gw={"network": "host", "port": 5075})
        )
        assert binding.service == "my_ioc_gw"
        assert (binding.host_ip, binding.host_port) == ("127.0.0.1", 5075)
        assert binding.host_network is True
        # The config-key spelling is what makes the generic remedy correct.
        assert host_ports._remedy_for_service(binding.service) == "services.my_ioc_gw.port"

    def test_a_facility_service_bind_override_is_honoured(self):
        (binding,) = derive_host_network_bindings(
            _host_config(my_ioc_gw={"network": "host", "port": 5075, "bind": "0.0.0.0"})
        )
        assert binding.host_ip == "0.0.0.0"

    def test_a_portless_host_mode_service_is_announced_not_silently_skipped(self, caplog):
        """A host-mode block with no usable port key cannot be derived — and
        the whole failure mode this derivation exists for is a silent gap, so
        the escape is said out loud."""
        with caplog.at_level("WARNING"):
            bindings = derive_host_network_bindings(_host_config(my_ioc_gw={"network": "host"}))
        assert bindings == []
        assert "my_ioc_gw" in caplog.text
        assert "preflight" in caplog.text

    def test_the_outbound_only_bridges_neither_derive_nor_warn(self, caplog):
        """The bundled bridges legitimately run host-mode with no listening
        socket; a warning for them would be noise on a valid config."""
        with caplog.at_level("WARNING"):
            bindings = derive_host_network_bindings(
                _host_config(nextcloud_bridge={"network": "host"}, gchat_bridge={"network": "host"})
            )
        assert bindings == []
        assert "host network" not in caplog.text

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

    def test_graphdb_on_host_derives_both_of_its_ports(self):
        bolt, http = derive_host_network_bindings(_host_config(graphdb={"network": "host"}))

        assert (bolt.service, bolt.host_port) == (
            GRAPHDB_SERVICE_NAME,
            default_port("graphdb_bolt"),
        )
        assert (http.service, http.host_port) == (
            GRAPHDB_SERVICE_NAME,
            default_port("graphdb_http"),
        )
        # Both bind loopback, both are host-network, and neither came from a
        # compose file — the generic `.port` branch would have found neither.
        assert {b.host_ip for b in (bolt, http)} == {"127.0.0.1"}
        assert all(b.host_network for b in (bolt, http))
        assert all("compose" not in b.compose_file for b in (bolt, http))

    def test_graphdb_host_ports_move_but_container_ports_do_not(self):
        """The label a derived binding carries is the port INSIDE the container,
        which the image fixes, not the host port the project moved it to. A
        host-port label would resolve the wrong remedy key and the wrong URL
        scheme the moment either port was overridden."""
        bolt, http = derive_host_network_bindings(
            _host_config(graphdb={"network": "host", "port_host": 17687, "http_port_host": 17474})
        )
        assert (bolt.host_port, bolt.container_port) == (17687, CONTAINER_BOLT_PORT)
        assert (http.host_port, http.container_port) == (17474, CONTAINER_HTTP_PORT)

        # Defaults are per-key, so moving one port leaves the other alone.
        (only_http_moved,) = [
            b
            for b in derive_host_network_bindings(
                _host_config(graphdb={"network": "host", "http_port_host": 17474})
            )
            if b.container_port == CONTAINER_HTTP_PORT
        ]
        assert only_http_moved.host_port == 17474

    def test_graphdb_default_bindings_carry_canonical_container_ports(self):
        bindings = derive_host_network_bindings(_host_config(graphdb={"network": "host"}))
        assert [(b.host_port, b.container_port) for b in bindings] == [
            (default_port("graphdb_bolt"), CONTAINER_BOLT_PORT),
            (default_port("graphdb_http"), CONTAINER_HTTP_PORT),
        ]

    def test_graphdb_is_excluded_from_the_generic_port_key_loop(self, caplog):
        """graphdb is derived by its own branch, so the generic loop must not
        also visit it — it reads `services.graphdb.port`, a key that does not
        exist, and would warn that a fully covered service escapes the
        preflight on every host-mode render."""
        with caplog.at_level("WARNING"):
            bindings = derive_host_network_bindings(_host_config(graphdb={"network": "host"}))
        assert len(bindings) == 2
        assert "graphdb" not in caplog.text
        assert "services.graphdb.port" not in caplog.text

    def test_graphdb_off_the_host_network_derives_nothing(self):
        assert derive_host_network_bindings(_host_config(graphdb={"port_host": 7687})) == []
        assert derive_host_network_bindings(_host_config(graphdb={"network": "bridge"})) == []


class TestRemedyResolution:
    """Which config key a contested port is sent to.

    Two answers, and the difference is the whole point of the block: a port
    still sitting where the layout puts it moves with every other port in the
    deployment, so its remedy is the one base knob. A port the project placed by
    hand does not move with the block, so its remedy stays the key that placed
    it. A service that publishes two host ports still needs the container port
    to tell its two keys apart once it is off-layout.
    """

    def test_every_named_slot_exists_in_the_layout(self):
        """The service-to-slot table is the only place the two vocabularies
        meet, and a misspelled slot would be a lookup error at the exact moment
        an operator is being told what to change."""
        unknown = {
            slot
            for slots in host_ports._SERVICE_LAYOUT_SLOTS.values()
            for slot in slots
            if slot not in SLOTS_BY_NAME
        }
        assert unknown == set()

    def test_every_framework_service_with_a_remedy_key_maps_to_a_slot(self):
        """A service the framework publishes and can move belongs in the table.
        The exceptions are named here so adding one is a deliberate act: the
        virtual accelerator is the Channel Access exception, and the graph
        store's own fallback key covers a binding no container port matched."""
        exceptions = {"virtual-accelerator"}
        missing = set(host_ports._SERVICE_REMEDY_KEYS) - set(host_ports._SERVICE_LAYOUT_SLOTS)
        assert missing == exceptions

    def test_a_framework_port_at_the_layout_slot_names_the_base(self):
        """Every framework-owned service, at its slot, at two different bases."""
        cases = [
            ("postgresql", "postgres"),
            ("mongodb", "mongo"),
            ("openobserve", "openobserve"),
            ("event-dispatcher", "dispatcher"),
            ("bluesky-bridge", "bluesky"),
            ("tiled", "tiled"),
            ("bluesky-web", "bluesky_web"),
            ("live-standin", "va_standin"),
            ("qmd", "qmd"),
        ]
        for base in (DEFAULT_PORT_BASE, SECOND_BASE):
            for service, slot in cases:
                port = default_port(slot, base=base)
                assert (
                    host_ports._remedy_for_service(service, None, port, base)
                    == PORT_BASE_CONFIG_KEY
                ), f"{service} at {port} (base {base})"

    def test_an_indexed_slot_names_the_base_across_its_whole_band(self):
        """Worker w and virtual-accelerator instance n are in-block wherever
        they sit in their band, so one base moves the fan-out as a unit."""
        for worker in (1, 7, 39):
            port = default_port("worker", worker, base=SECOND_BASE)
            remedy = host_ports._remedy_for_service(
                f"dispatch-worker-{worker}", None, port, SECOND_BASE
            )
            assert remedy == PORT_BASE_CONFIG_KEY

        for instance in (0, 9):
            port = default_port("va_standin", instance, base=SECOND_BASE)
            remedy = host_ports._remedy_for_service("live-standin", None, port, SECOND_BASE)
            assert remedy == PORT_BASE_CONFIG_KEY

    def test_a_port_just_outside_a_band_is_not_the_bases_to_move(self):
        """The band is checked, not the block: one past the last worker is not
        a worker port, and telling that operator to move the base would leave
        whatever actually binds there exactly where it is."""
        past_the_band = default_port("worker", 39, base=SECOND_BASE) + 1
        assert (
            host_ports._remedy_for_service("dispatch-worker-40", None, past_the_band, SECOND_BASE)
            == "dispatch.worker_port_base"
        )

    def test_a_hand_placed_port_keeps_the_key_that_placed_it(self):
        """An override does not move with the block, so naming the base would
        be an instruction that changes nothing about the contested port."""
        cases = [
            ("postgresql", "services.postgresql.port_host"),
            ("mongodb", "services.mongodb.port_host"),
            ("openobserve", "services.openobserve.port"),
            ("event-dispatcher", "services.event_dispatcher.port"),
            ("dispatch-worker-1", "dispatch.worker_port_base"),
            ("dispatch-worker-7", "dispatch.worker_port_base"),
            ("bluesky-bridge", "services.bluesky.port"),
            ("bluesky-va-bridge", "services.bluesky_va.port"),
            ("bluesky-live-bridge", "services.bluesky_live.port"),
            ("tiled", "services.bluesky.tiled_port"),
            ("bluesky-web", "services.bluesky_web.port"),
            ("live-standin", "services.live_standin.port"),
        ]
        # Well outside the block, which is what "placed by hand" looks like.
        off_layout = 5555
        assert (
            not block_range(DEFAULT_PORT_BASE)[0] <= off_layout <= block_range(DEFAULT_PORT_BASE)[1]
        )
        for service, expected in cases:
            assert (
                host_ports._remedy_for_service(service, None, off_layout, DEFAULT_PORT_BASE)
                == expected
            )

    def test_the_channel_access_port_is_never_sent_to_the_base(self):
        """Instance 1 of the virtual accelerator is the documented exception:
        5064 is outside the block by design, and no base moves it."""
        for base in (DEFAULT_PORT_BASE, SECOND_BASE):
            assert (
                host_ports._remedy_for_service("virtual-accelerator", None, CA_DEFAULT_PORT, base)
                == "services.virtual_accelerator.port"
            )

    def test_a_facility_service_keeps_its_own_generic_key(self):
        """The facility band is a deployment's to spend. Nothing the framework
        publishes lives there, so a collision is the facility's own key even
        though the port is inside the block."""
        facility_port = default_port("facility", base=SECOND_BASE)
        assert (
            host_ports._remedy_for_service("my_ioc_gw", None, facility_port, SECOND_BASE)
            == "services.my_ioc_gw.port"
        )

    def test_without_a_base_the_older_per_service_answers_stand(self):
        """A caller with no config cannot ask the layout question at all, and a
        guessed base would hand out a knob that moves a block this deployment
        may not own."""
        assert host_ports._remedy_for_service("postgresql") == "services.postgresql.port_host"
        assert (
            host_ports._remedy_for_service("postgresql", None, default_port("postgres"), None)
            == "services.postgresql.port_host"
        )

    def test_each_graphdb_container_port_resolves_its_own_key(self):
        """The graph store publishes two host ports, so once they are off the
        layout its service name alone cannot say which key moves the contested
        one. The container port can: the image fixes it."""
        for container_port, expected in (
            (CONTAINER_BOLT_PORT, GRAPHDB_PORT_CONFIG_KEY),
            (CONTAINER_HTTP_PORT, GRAPHDB_HTTP_PORT_CONFIG_KEY),
        ):
            assert host_ports._remedy_for_service(GRAPHDB_SERVICE_NAME, container_port) == expected
        # The two keys are genuinely different; a same-key result would make the
        # per-port table pointless without failing any other assertion here.
        assert GRAPHDB_PORT_CONFIG_KEY != GRAPHDB_HTTP_PORT_CONFIG_KEY

    def test_both_graphdb_ports_at_the_layout_slots_name_the_base(self):
        """On layout the two keys collapse into one: the pair moves together."""
        for slot, container_port in (
            ("graphdb_bolt", CONTAINER_BOLT_PORT),
            ("graphdb_http", CONTAINER_HTTP_PORT),
        ):
            port = default_port(slot, base=SECOND_BASE)
            remedy = host_ports._remedy_for_service(
                GRAPHDB_SERVICE_NAME, container_port, port, SECOND_BASE
            )
            assert remedy == PORT_BASE_CONFIG_KEY

    def test_one_moved_graphdb_port_resolves_its_own_key_not_the_base(self):
        """A project that moved only the Browser port has one in-block port and
        one hand-placed one, and each has to name what actually moves it."""
        bolt = default_port("graphdb_bolt", base=SECOND_BASE)
        assert (
            host_ports._remedy_for_service(
                GRAPHDB_SERVICE_NAME, CONTAINER_BOLT_PORT, bolt, SECOND_BASE
            )
            == PORT_BASE_CONFIG_KEY
        )
        assert (
            host_ports._remedy_for_service(
                GRAPHDB_SERVICE_NAME, CONTAINER_HTTP_PORT, 17474, SECOND_BASE
            )
            == GRAPHDB_HTTP_PORT_CONFIG_KEY
        )

    def test_a_graphdb_port_hand_placed_on_the_other_slot_is_not_the_bases(self):
        """The two graph-store slots are one port apart, so a bolt port moved by
        hand onto the HTTP slot's default sits exactly where a layout port would
        be. Matching on the host port alone would send that operator to
        deployment.port_base, which moves the whole block and leaves this one
        binding untouched. The container port is what tells the slots apart."""
        on_the_http_slot = default_port("graphdb_http", base=SECOND_BASE)
        assert (
            host_ports._remedy_for_service(
                GRAPHDB_SERVICE_NAME, CONTAINER_BOLT_PORT, on_the_http_slot, SECOND_BASE
            )
            == GRAPHDB_PORT_CONFIG_KEY
        )
        # And the mirror image: the Browser port parked on the bolt slot.
        on_the_bolt_slot = default_port("graphdb_bolt", base=SECOND_BASE)
        assert (
            host_ports._remedy_for_service(
                GRAPHDB_SERVICE_NAME, CONTAINER_HTTP_PORT, on_the_bolt_slot, SECOND_BASE
            )
            == GRAPHDB_HTTP_PORT_CONFIG_KEY
        )

    def test_an_unparseable_container_port_still_matches_on_the_host_port(self):
        """A binding whose container port did not parse cannot be filtered by
        it, and answering 'no slot' there would be worse than the best-effort
        answer the host port alone gives."""
        assert (
            host_ports._remedy_for_service(
                GRAPHDB_SERVICE_NAME,
                None,
                default_port("graphdb_bolt", base=SECOND_BASE),
                SECOND_BASE,
            )
            == PORT_BASE_CONFIG_KEY
        )

    def test_a_lane_pair_is_still_matched_on_the_host_port_alone(self):
        """Both Bluesky lanes are the same image on the same container port, so
        the slot filter must not be applied to them: only the host port
        separates lane 1 from lane 2."""
        for slot in ("bluesky", "bluesky_second_lane"):
            port = default_port(slot, base=SECOND_BASE)
            assert (
                host_ports._remedy_for_service("bluesky-bridge", 8000, port, SECOND_BASE)
                == PORT_BASE_CONFIG_KEY
            )

    def test_every_disambiguated_slot_belongs_to_a_service_that_owns_two(self):
        """The container-port filter only earns its keep where a service owns
        more than one slot; anywhere else it would silently narrow a match that
        the host port had already settled."""
        multi_slot = {
            slot
            for slots in host_ports._SERVICE_LAYOUT_SLOTS.values()
            if len(slots) > 1
            for slot in slots
        }
        assert set(host_ports._SLOT_CONTAINER_PORTS) <= multi_slot

    def test_the_override_beats_the_host_port_the_project_published_on(self):
        """Resolution keys on the container port, so a project that moved both
        published ports still gets the key that moves the contested one."""
        assert (
            host_ports._remedy_for_service(GRAPHDB_SERVICE_NAME, CONTAINER_HTTP_PORT)
            == GRAPHDB_HTTP_PORT_CONFIG_KEY
        )
        # 17474 is a HOST port, not a container port: it must not match.
        assert host_ports._remedy_for_service(GRAPHDB_SERVICE_NAME, 17474) == (
            GRAPHDB_PORT_CONFIG_KEY
        )

    def test_an_unmatched_graphdb_binding_falls_back_to_a_key_that_exists(self):
        """Whatever fails to match the per-port table — an unparseable container
        port, or a port the table does not know — must still name a real config
        key. The generic `services.<name>.port` fallback does not exist for this
        service, so it would send an operator to edit nothing."""
        for container_port in (None, 7473, 0):
            remedy = host_ports._remedy_for_service(GRAPHDB_SERVICE_NAME, container_port)
            assert remedy == GRAPHDB_PORT_CONFIG_KEY
            assert remedy != f"services.{GRAPHDB_SERVICE_NAME}.port"

    def test_conflicts_on_the_two_ports_print_the_two_keys(self, monkeypatch):
        monkeypatch.setattr(host_ports, "_port_is_free", lambda host_ip, host_port: True)
        bindings = [
            # A prior claim on each address, then graphdb losing both. Both host
            # ports are hand-placed, which is what keeps the two keys distinct.
            HostPortBinding("other", "127.0.0.1", 17687, 17687, "a.yml"),
            HostPortBinding("other", "127.0.0.1", 17474, 17474, "a.yml"),
            HostPortBinding("graphdb", "127.0.0.1", 17687, CONTAINER_BOLT_PORT, "b.yml"),
            HostPortBinding("graphdb", "127.0.0.1", 17474, CONTAINER_HTTP_PORT, "b.yml"),
        ]
        conflicts = find_port_conflicts(bindings, project_name="proj")
        assert [(c.host_port, c.remedy) for c in conflicts] == [
            (17687, GRAPHDB_PORT_CONFIG_KEY),
            (17474, GRAPHDB_HTTP_PORT_CONFIG_KEY),
        ]
        report = format_conflict_report(conflicts)
        assert GRAPHDB_PORT_CONFIG_KEY in report
        assert GRAPHDB_HTTP_PORT_CONFIG_KEY in report

    def test_derived_and_parsed_graphdb_bindings_resolve_identically(self):
        """Task 3.3's HTTP predicate reads the same label, so a derived binding
        that disagreed with the parsed one would render bolt as a browser URL."""
        derived = derive_host_network_bindings(
            _host_config(graphdb={"network": "host", "port_host": 17687, "http_port_host": 17474})
        )
        parsed = [
            HostPortBinding("graphdb", "127.0.0.1", 17687, CONTAINER_BOLT_PORT, "a.yml"),
            HostPortBinding("graphdb", "127.0.0.1", 17474, CONTAINER_HTTP_PORT, "a.yml"),
        ]
        assert [(b.service, b.host_port, b.container_port) for b in derived] == [
            (b.service, b.host_port, b.container_port) for b in parsed
        ]
        assert [host_ports._remedy_for_service(b.service, b.container_port) for b in derived] == [
            host_ports._remedy_for_service(b.service, b.container_port) for b in parsed
        ]

    def test_a_container_port_never_decides_another_services_key(self):
        """The per-port table is keyed on the pair, so graphdb's container port
        appearing on some other service changes nothing about that service."""
        cases = [
            ("postgresql", "services.postgresql.port_host"),
            ("openobserve", "services.openobserve.port"),
            ("virtual-accelerator", "services.virtual_accelerator.port"),
            ("my_ioc_gw", "services.my_ioc_gw.port"),
        ]
        for service, expected in cases:
            assert host_ports._remedy_for_service(service) == expected
            assert host_ports._remedy_for_service(service, CONTAINER_BOLT_PORT) == expected


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


class TestBlockFramedReport:
    """The report reads as one block of a thousand ports, not as loose numbers."""

    def _conflict(self, port, remedy, base=DEFAULT_PORT_BASE, **kwargs):
        """One external conflict at ``port``, framed by ``base``."""
        return PortConflict(
            host_port=port,
            bind_address="127.0.0.1",
            service="postgresql",
            kind="external",
            holder="an unknown host process",
            remedy=remedy,
            port_base=base,
            **kwargs,
        )

    def test_the_block_is_named_before_the_conflicts(self):
        first, last = block_range(SECOND_BASE)
        report = format_conflict_report(
            [
                self._conflict(
                    default_port("postgres", base=SECOND_BASE),
                    PORT_BASE_CONFIG_KEY,
                    base=SECOND_BASE,
                )
            ]
        )
        assert f"ports {first}-{last}" in report
        assert f"{PORT_BASE_CONFIG_KEY} {SECOND_BASE}" in report
        # The block line precedes the per-port lines it frames.
        assert report.index(f"ports {first}-{last}") < report.index("cannot bind")

    def test_a_report_without_a_base_prints_no_block_line(self):
        """A conflict found with no config to resolve carries no base, and
        inventing one would frame the report around a block this deployment may
        not own."""
        report = format_conflict_report(
            [self._conflict(5432, "services.postgresql.port_host", base=None)]
        )
        assert "This deployment's block" not in report
        assert "5432" in report

    def test_the_one_knob_is_explained_once_not_per_line(self):
        conflicts = [
            self._conflict(default_port("postgres"), PORT_BASE_CONFIG_KEY),
            self._conflict(default_port("mongo"), PORT_BASE_CONFIG_KEY),
        ]
        report = format_conflict_report(conflicts)
        assert report.count("moves all of them at once") == 1
        # And it is not offered when no conflict names the base.
        hand_placed = format_conflict_report(
            [self._conflict(5432, "services.postgresql.port_host")]
        )
        assert "moves all of them at once" not in hand_placed

    def test_the_channel_access_exception_names_the_one_key_that_moves_it(self):
        report = format_conflict_report(
            [
                self._conflict(
                    CA_DEFAULT_PORT, "services.virtual_accelerator.port", channel_access=True
                )
            ]
        )
        assert f"Port {CA_DEFAULT_PORT} is outside the block." in report
        assert "services.virtual_accelerator.port" in report
        assert f"{PORT_BASE_CONFIG_KEY} does not move it" in report

    def test_an_in_block_conflict_says_nothing_about_channel_access(self):
        report = format_conflict_report(
            [self._conflict(default_port("postgres"), PORT_BASE_CONFIG_KEY)]
        )
        assert "Channel Access" not in report


class TestCheckoutAttribution:
    """Whose container holds the port, decided by repo-id first, project second."""

    def _preflight(self, monkeypatch, listening_port, row, repo_id):
        """Run the preflight against one fake ``ps`` row on a live listener."""
        _, port = listening_port
        row["Ports"] = f"127.0.0.1:{port}->5432/tcp"
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: json.dumps(row))
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")
        return find_port_conflicts([binding], project_name="proj", repo_id=repo_id)

    def test_the_identity_is_derived_from_the_config_when_none_is_passed(
        self, monkeypatch, listening_port, tmp_path
    ):
        """The branch every real deploy takes: container_lifecycle passes no
        repo_id, so the preflight has to work the checkout out for itself from
        the same config it is preflighting, via the same repo_identity the
        render stamped into the label."""
        _, port = listening_port
        identity = repo_identity(tmp_path)
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                _ps_row("proj-postgres", "proj", port=port, repo_id=identity)
            ),
        )
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        # `project_root` is what a rendered config carries, and resolve_repo_root
        # honours it when it names a directory that exists here.
        config = {"project_root": str(tmp_path)}
        assert find_port_conflicts([binding], project_name="proj", config=config) == []

    def test_a_derived_identity_that_disagrees_is_still_a_conflict(
        self, monkeypatch, listening_port, tmp_path
    ):
        """The control for the test above: the same derivation, a container from
        a different checkout. A derivation that silently produced '' would make
        this pass on the project name and hide the collision."""
        _, port = listening_port
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                _ps_row(
                    "proj-postgres", "proj", port=port, repo_id=repo_identity(tmp_path / "other")
                )
            ),
        )
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        conflicts = find_port_conflicts(
            [binding], project_name="proj", config={"project_root": str(tmp_path)}
        )
        assert len(conflicts) == 1
        assert conflicts[0].kind == "external"

    def test_an_unresolvable_repo_root_degrades_to_the_project_check(
        self, monkeypatch, listening_port
    ):
        """When the checkout cannot be identified at all, _deployment_identity
        answers '' and attribution falls back to what it always was, rather than
        refusing the deploy over a question it could not ask."""
        _, port = listening_port

        def _no_root(config=None, config_path=None):
            raise OSError("no repo root here")

        monkeypatch.setattr(host_ports, "resolve_repo_root", _no_root)
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                _ps_row("proj-postgres", "proj", port=port, repo_id="def456")
            ),
        )
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        # The project name matches, and with no identity of our own that is the
        # only answer available: our own container, not a conflict.
        assert find_port_conflicts([binding], project_name="proj", config={}) == []

    def test_the_project_check_still_finds_a_foreign_stack_without_an_identity(
        self, monkeypatch, listening_port
    ):
        """The other half of the fallback: degrading must not exempt everything."""
        _, port = listening_port

        def _no_root(config=None, config_path=None):
            raise OSError("no repo root here")

        monkeypatch.setattr(host_ports, "resolve_repo_root", _no_root)
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                _ps_row("other-postgres", "other", port=port, repo_id="def456")
            ),
        )
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        conflicts = find_port_conflicts([binding], project_name="proj", config={})
        assert len(conflicts) == 1
        assert "other-postgres" in conflicts[0].holder

    def test_a_matching_repo_id_is_our_own_container(self, monkeypatch, listening_port):
        conflicts = self._preflight(
            monkeypatch,
            listening_port,
            _ps_row("proj-postgres", "proj", repo_id="abc123"),
            "abc123",
        )
        assert conflicts == []

    def test_a_matching_repo_id_wins_over_a_project_name_that_disagrees(
        self, monkeypatch, listening_port
    ):
        """The label is what the destructive verbs select on, so a status that
        answered differently would describe a different set of containers."""
        conflicts = self._preflight(
            monkeypatch,
            listening_port,
            _ps_row("renamed-postgres", "renamed", repo_id="abc123"),
            "abc123",
        )
        assert conflicts == []

    def test_another_checkout_of_the_same_project_is_a_real_conflict(
        self, monkeypatch, listening_port
    ):
        """Two clones of one deployment share a compose project name, so the
        project check alone would wave this collision through."""
        conflicts = self._preflight(
            monkeypatch,
            listening_port,
            _ps_row("proj-postgres", "proj", repo_id="def456"),
            "abc123",
        )
        assert len(conflicts) == 1
        assert conflicts[0].kind == "external"
        # The checkout is named, or the holder reads as this deployment's own.
        assert "def456" in conflicts[0].holder
        assert "proj" in conflicts[0].holder

    def test_our_own_unlabelled_container_is_still_recognised(self, monkeypatch, listening_port):
        """A container an older OSPREY created carries no repo-id at all, and
        the project check is the only answer available for it."""
        assert (
            self._preflight(monkeypatch, listening_port, _ps_row("proj-postgres", "proj"), "abc123")
            == []
        )

    def test_an_unlabelled_container_of_another_project_still_conflicts(
        self, monkeypatch, listening_port
    ):
        """The fallback is the project check, not a blanket exemption: reading
        no label must not turn every foreign listener into our own."""
        conflicts = self._preflight(
            monkeypatch, listening_port, _ps_row("other-postgres", "other"), "abc123"
        )
        assert len(conflicts) == 1
        assert "other-postgres" in conflicts[0].holder

    def test_an_unidentifiable_checkout_falls_back_to_the_project_name(
        self, monkeypatch, listening_port
    ):
        """When this checkout's own identity cannot be worked out, attribution
        degrades to what it always was rather than calling everything foreign."""
        assert (
            self._preflight(
                monkeypatch, listening_port, _ps_row("proj-postgres", "proj", repo_id="def456"), ""
            )
            == []
        )

    def test_the_podman_label_shape_carries_the_repo_id_too(self, monkeypatch, listening_port):
        _, port = listening_port
        ps_json = json.dumps(
            [
                {
                    "Names": ["proj-postgres"],
                    "Ports": [{"host_ip": "127.0.0.1", "host_port": port, "container_port": 5432}],
                    "Labels": {
                        "com.docker.compose.project": "proj",
                        REPO_ID_LABEL: "def456",
                    },
                }
            ]
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: ps_json)
        binding = HostPortBinding("postgresql", "127.0.0.1", port, 5432, "a.yml")

        conflicts = find_port_conflicts([binding], project_name="proj", repo_id="abc123")
        assert len(conflicts) == 1
        assert "def456" in conflicts[0].holder

    def test_a_host_network_container_of_another_checkout_still_conflicts(
        self, monkeypatch, listening_port
    ):
        """A host-network container maps no port, so it is found by name. Two
        checkouts produce the same container name as well as the same project
        name, which is why ownership is settled by the label there too."""
        _, port = listening_port
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                _ps_row("proj-event-dispatcher", "proj", repo_id="def456")
            ),
        )
        config = _host_config(event_dispatcher={"network": "host", "port": port})

        conflicts = find_port_conflicts([], project_name="proj", config=config, repo_id="abc123")
        assert [c.service for c in conflicts] == ["event-dispatcher"]

    def test_our_own_host_network_container_is_still_an_idempotent_redeploy(
        self, monkeypatch, listening_port
    ):
        """The same row as above, read from the checkout that created it."""
        _, port = listening_port
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                _ps_row("proj-event-dispatcher", "proj", repo_id="def456")
            ),
        )
        config = _host_config(event_dispatcher={"network": "host", "port": port})

        assert find_port_conflicts([], project_name="proj", config=config, repo_id="def456") == []


class TestTwoDeploymentsOnOneHost:
    """The acceptance scenario, with a faked port scan instead of containers.

    A default-base deployment is running. A second project, built at
    ``port_base: 20000``, is preflighted against it. Everything the second
    project binds is inside its own block and free, so the only collision left
    is the one port a base cannot move.
    """

    #: What the first deployment holds: its whole block, plus Channel Access.
    def _held_ports(self):
        first, last = block_range(DEFAULT_PORT_BASE)
        return set(range(first, last + 1)) | {CA_DEFAULT_PORT}

    def _second_project(self):
        """Published bindings of a second project built at ``SECOND_BASE``."""
        return [
            HostPortBinding(
                service, "127.0.0.1", default_port(slot, base=SECOND_BASE), container, "b.yml"
            )
            for service, slot, container in (
                ("postgresql", "postgres", 5432),
                ("mongodb", "mongo", 27017),
                ("openobserve", "openobserve", 5080),
                ("graphdb", "graphdb_bolt", CONTAINER_BOLT_PORT),
                ("graphdb", "graphdb_http", CONTAINER_HTTP_PORT),
                ("tiled", "tiled", 8000),
                ("bluesky-bridge", "bluesky", 8000),
            )
        ] + [
            # The Channel Access exception: instance 1 is NOT in the block, so
            # it did not move with the base and lands on the running one.
            HostPortBinding(
                "virtual-accelerator", "127.0.0.1", CA_DEFAULT_PORT, CA_DEFAULT_PORT, "b.yml"
            )
        ]

    def _run(self, monkeypatch):
        held = self._held_ports()
        monkeypatch.setattr(
            host_ports, "_port_is_free", lambda host_ip, host_port: host_port not in held
        )
        monkeypatch.setattr(
            host_ports,
            "_run_runtime_ps",
            lambda config=None: json.dumps(
                [
                    _ps_row("first-postgres", "first", port=p, repo_id="firstrepo01")
                    for p in sorted(held)
                ]
            ),
        )
        return find_port_conflicts(
            self._second_project(),
            project_name="second",
            config={"deployment": {"port_base": SECOND_BASE}},
            repo_id="secondrepo1",
        )

    def test_only_the_channel_access_port_collides(self, monkeypatch):
        conflicts = self._run(monkeypatch)
        assert [(c.host_port, c.service) for c in conflicts] == [
            (CA_DEFAULT_PORT, "virtual-accelerator")
        ]
        assert conflicts[0].remedy == "services.virtual_accelerator.port"
        assert conflicts[0].channel_access is True
        assert conflicts[0].port_base == SECOND_BASE

    def test_the_report_frames_the_second_block_and_explains_the_exception(self, monkeypatch):
        report = format_conflict_report(self._run(monkeypatch))
        first, last = block_range(SECOND_BASE)
        assert f"ports {first}-{last}" in report
        assert f"Port {CA_DEFAULT_PORT} is outside the block." in report
        assert "services.virtual_accelerator.port" in report
        # The base moved everything it could, so it is not offered again.
        assert "moves all of them at once" not in report

    def test_the_same_project_at_the_default_base_collides_everywhere(self, monkeypatch):
        """The control: without the second base every published port is taken,
        which is the state the block is what fixes."""
        held = self._held_ports()
        monkeypatch.setattr(
            host_ports, "_port_is_free", lambda host_ip, host_port: host_port not in held
        )
        monkeypatch.setattr(host_ports, "_run_runtime_ps", lambda config=None: "")
        bindings = [
            HostPortBinding(
                b.service,
                b.host_ip,
                b.host_port - SECOND_BASE + DEFAULT_PORT_BASE,
                b.container_port,
                b.compose_file,
            )
            if b.host_port != CA_DEFAULT_PORT
            else b
            for b in self._second_project()
        ]
        conflicts = find_port_conflicts(
            bindings,
            project_name="second",
            config={"deployment": {"port_base": DEFAULT_PORT_BASE}},
            repo_id="secondrepo1",
        )
        assert len(conflicts) == len(bindings)
        # Every in-block one names the single knob that moves them together.
        in_block = [c for c in conflicts if not c.channel_access]
        assert {c.remedy for c in in_block} == {PORT_BASE_CONFIG_KEY}
