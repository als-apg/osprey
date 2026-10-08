"""Bridge-network consumers of the project image reach the stores by service name.

The ``services.<store>`` blocks in ``config.yml`` are written for the HOST side:
a published port on the host's loopback. From inside a bridge-networked
container that loopback is the container's own, where nothing listens. So every
bundled template that runs the project image on the bridge — the ARIEL sync
daemon and the dispatch workers — is rendered with the in-network address of
each store this deployment runs:

* one ``OSPREY_QMD_<CORPUS>_URL`` per qmd sidecar, naming its compose service
  and the port it listens on inside the network;
* ``ARIEL_DATABASE_HOST``/``PORT`` for the ARIEL store.

Each is gated on the store being deployed here (a name that resolves only on a
network where this deployment runs it), and under ``network: host`` the qmd
URLs are omitted — the published address in the config block is already right
from the host namespace.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from tests.deployment.test_compose_generator import _render_service_template

#: (template, config key of the consumer, the compose service whose env is read)
CONSUMERS = [
    ("ariel_sync/docker-compose.yml.j2", "ariel_sync", "ariel-sync"),
    ("dispatch_worker/docker-compose.yml.j2", "dispatch_worker", "dispatch-worker-1"),
]

#: The render context the qmd fragment's resolver produces for two sidecars.
QMD_CONTEXT = {
    "port": 10060,
    "bind_address": "127.0.0.1",
    "interval_seconds": 30,
    "first_index_grace_seconds": 3600,
    "corpora": [],
    "network_env": {
        "OSPREY_QMD_OKF_URL": "http://qmd-okf:10060",
        "OSPREY_QMD_ARIEL_URL": "http://qmd-ariel:10061",
    },
}


def _env(
    template: str,
    key: str,
    compose_name: str,
    *,
    network: str | None = None,
    deployed: tuple[str, ...] = (),
    postgresql: dict[str, Any] | None = None,
    bind_address: str | None = None,
) -> dict[str, Any]:
    """Render *template* and return the consumer's ``environment:`` mapping."""
    block: dict[str, Any] = {}
    if key == "dispatch_worker":
        block.update({"worker_count": 1, "workspace_mode": "isolated"})
    if network is not None:
        block["network"] = network
    services: dict[str, Any] = {key: block}
    if postgresql is not None:
        services["postgresql"] = postgresql
    rendered = _render_service_template(
        template,
        "proj-a",
        deployment={} if bind_address is None else {"bind_address": bind_address},
        services=services,
        deployed_services=[key, *deployed],
        osprey_qmd=QMD_CONTEXT,
        osprey_ariel_mirror_source=None,
        osprey_container_ariel_mirror_dir=None,
    )
    return yaml.safe_load(rendered)["services"][compose_name]["environment"]


@pytest.mark.parametrize(("template", "key", "compose_name"), CONSUMERS)
class TestQmdSidecarDial:
    def test_bridge_consumer_gets_each_sidecars_in_network_url(
        self, template: str, key: str, compose_name: str
    ) -> None:
        env = _env(template, key, compose_name, deployed=("qmd",))
        assert env["OSPREY_QMD_OKF_URL"] == "http://qmd-okf:10060"
        assert env["OSPREY_QMD_ARIEL_URL"] == "http://qmd-ariel:10061"

    def test_host_consumer_gets_none(self, template: str, key: str, compose_name: str) -> None:
        env = _env(template, key, compose_name, network="host", deployed=("qmd",))
        assert not any(name.startswith("OSPREY_QMD_") for name in env)

    def test_no_url_without_a_deployed_sidecar(
        self, template: str, key: str, compose_name: str
    ) -> None:
        env = _env(template, key, compose_name)
        assert not any(name.startswith("OSPREY_QMD_") for name in env)

    def test_a_render_without_qmd_context_emits_nothing(
        self, template: str, key: str, compose_name: str
    ) -> None:
        block: dict[str, Any] = {"worker_count": 1} if key == "dispatch_worker" else {}
        rendered = _render_service_template(
            template,
            "proj-a",
            services={key: block},
            deployed_services=[key, "qmd"],
            osprey_ariel_mirror_source=None,
            osprey_container_ariel_mirror_dir=None,
        )
        env = yaml.safe_load(rendered)["services"][compose_name]["environment"]
        assert not any(name.startswith("OSPREY_QMD_") for name in env)


@pytest.mark.parametrize(("template", "key", "compose_name"), CONSUMERS)
class TestArielStoreDial:
    def test_bridge_consumer_dials_the_network_alias(
        self, template: str, key: str, compose_name: str
    ) -> None:
        env = _env(template, key, compose_name, deployed=("postgresql",))
        assert env["ARIEL_DATABASE_HOST"] == "ariel-postgres"
        assert env["ARIEL_DATABASE_PORT"] == "5432"

    def test_host_consumer_dials_the_published_port(
        self, template: str, key: str, compose_name: str
    ) -> None:
        env = _env(
            template,
            key,
            compose_name,
            network="host",
            deployed=("postgresql",),
            postgresql={"port_host": 15432},
        )
        assert env["ARIEL_DATABASE_HOST"] == "127.0.0.1"
        assert env["ARIEL_DATABASE_PORT"] == "15432"

    @pytest.mark.parametrize(
        ("bind", "host"),
        [("10.0.0.5", "10.0.0.5"), ("0.0.0.0", "127.0.0.1"), ("::", "[::1]")],
    )
    def test_host_consumer_dials_the_interface_the_store_publishes_on(
        self, template: str, key: str, compose_name: str, bind: str, host: str
    ) -> None:
        # The store publishes on `deployment.bind_address`: a pinned interface
        # is reached there, a wildcard on its family's loopback.
        env = _env(
            template,
            key,
            compose_name,
            network="host",
            deployed=("postgresql",),
            bind_address=bind,
        )
        assert env["ARIEL_DATABASE_HOST"] == host

    def test_bridge_consumer_ignores_the_bind(
        self, template: str, key: str, compose_name: str
    ) -> None:
        env = _env(template, key, compose_name, deployed=("postgresql",), bind_address="10.0.0.5")
        assert env["ARIEL_DATABASE_HOST"] == "ariel-postgres"

    @pytest.mark.parametrize("network", [None, "host"])
    def test_external_store_gets_no_override(
        self, template: str, key: str, compose_name: str, network: str | None
    ) -> None:
        env = _env(template, key, compose_name, network=network)
        assert "ARIEL_DATABASE_HOST" not in env
        assert "ARIEL_DATABASE_PORT" not in env


@pytest.mark.parametrize(
    ("deployment", "address"),
    [
        (None, "127.0.0.1"),
        ({}, "127.0.0.1"),
        ({"bind_address": "10.0.0.5"}, "10.0.0.5"),
        ({"bind_address": "0.0.0.0"}, "127.0.0.1"),
        ({"bind_address": "::"}, "[::1]"),
        ({"bind_address": "fd00::5"}, "[fd00::5]"),
    ],
)
def test_the_host_dial_address_follows_the_published_interface(
    deployment: dict[str, Any] | None, address: str
) -> None:
    from osprey.deployment.compose_generator import _host_dial_address

    config: dict[str, Any] = {} if deployment is None else {"deployment": deployment}
    assert _host_dial_address(config) == address
