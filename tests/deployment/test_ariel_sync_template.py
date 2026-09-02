"""Render tests for the bundled ``ariel_sync`` compose template.

``ariel_sync`` is the deployed half of the ARIEL logbook mirror: a long-running
container that polls the facility logbook and writes what it finds into the
deployment's own ARIEL store. Three things about that shape are load-bearing and
are pinned here.

**It is a daemon, not a one-shot.** ``osprey ariel sync --watch`` runs an initial
sync and then stays in the ingestion loop, so the mirror keeps up with the
logbook between deploys instead of freezing at the last ``osprey up``. That is
why it carries ``restart: unless-stopped``: the container is meant to be running
whenever the deployment is. It publishes nothing and answers no probe — it only
makes outbound calls — so it declares neither ``ports:`` nor ``healthcheck:``,
and the question "is the mirror still moving?" is answered by the
``ariel_last_ingestion`` health row reading the store, not by container state.

**It reaches the store at whichever address is right from where it runs.** The
``ariel.database`` block in ``config.yml`` is written for the HOST side, which
from inside a bridge-networked container names that container's own loopback.
When this deployment co-deploys the store, the template says the address
outright — the ``ariel-postgres`` network alias and postgres's container port on
the bridge, the host loopback and the PUBLISHED port under ``network: host`` —
following the ``OSPREY_ARCHIVER_MONGODB_HOST``/``PORT`` precedent in the dispatch
worker. Those two variables apply only in ``resolve_ariel_dsn``'s DERIVED rung,
so a deployment that authored its own ``ariel.database.uri`` (an external store)
is never redirected, and a deployment that co-deploys no store gets neither
variable nor a ``depends_on`` naming a service compose would refuse to resolve.

**It binds the deployment's qmd mirror when there is one.** The enhancement
module writing that mirror runs inside THIS container, so without the bind its
output lands in the container's writable layer, where the qmd sidecar never
indexes it and the next recreate discards it. Both mirror keys are injected as
``None`` when the deployment runs no qmd export, so every mirror-shaped line is
gated on the source's truthiness rather than defaulted to an empty string, which
would render a mount with an empty host half.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

# The same helper every other bundled-service render test builds its context
# with, rather than a second hand-rolled context: a render is only worth what
# the context behind it is worth, and this one is derived from the production
# image/port resolvers instead of literals.
from tests.deployment.test_compose_generator import (
    _layout_ports_for,
    _render_service_template,
)

#: Path of the template under test, relative to ``templates/services/``.
TEMPLATE = "ariel_sync/docker-compose.yml.j2"

#: The compose service key. Hyphenated like every other bundled service key
#: (``archiver-recorder``, ``gchat-bridge``, ``virtual-accelerator``), while the
#: config/profile key that selects it stays ``ariel_sync``.
SERVICE_KEY = "ariel-sync"


def _render(
    *,
    project_name: str = "proj-a",
    ariel_sync: dict[str, Any] | None = None,
    postgresql: dict[str, Any] | None = None,
    postgres_deployed: bool = True,
    mirror_source: str | None = None,
    container_mirror_dir: str | None = None,
    env_chain: list[str] | None = None,
    deployment: dict[str, Any] | None = None,
) -> str:
    """Render the packaged template for one deployment shape.

    Every knob moves independently on purpose: a config can declare a
    ``services.postgresql`` block it never deploys, and the mirror keys are
    injected for every render whether or not a qmd export writes one.

    Args:
        project_name: Project the render is namespaced to.
        ariel_sync: This service's own config block (``network``, ``image``,
            ``env``).
        postgresql: The store's config block, for its ``port_host``.
        postgres_deployed: Whether ``postgresql`` is in ``deployed_services``.
        mirror_source: ``osprey_ariel_mirror_source`` — the host bind source, or
            ``None`` when no export writes a mirror.
        container_mirror_dir: ``osprey_container_ariel_mirror_dir`` — the mount
            target inside the container.
        env_chain: ``osprey_env_chain``, the deployment's env files in ascending
            precedence.
        deployment: The ``deployment`` block, which moves the whole port layout.

    Returns:
        The rendered compose document as text.
    """
    services: dict[str, Any] = {"ariel_sync": ariel_sync if ariel_sync is not None else {}}
    if postgresql is not None:
        services["postgresql"] = postgresql

    deployed = ["ariel_sync"]
    if postgres_deployed:
        deployed.append("postgresql")

    overrides: dict[str, Any] = {
        "services": services,
        "deployed_services": deployed,
        # Always present in a production render — `None` is the "no export
        # writes a mirror" answer, not an absent key.
        "osprey_ariel_mirror_source": mirror_source,
        "osprey_container_ariel_mirror_dir": container_mirror_dir,
    }
    if env_chain is not None:
        overrides["osprey_env_chain"] = env_chain
    if deployment is not None:
        overrides["deployment"] = deployment
    return _render_service_template(TEMPLATE, project_name, **overrides)


def _service(rendered: str) -> dict[str, Any]:
    """The one service block of a rendered document."""
    doc = yaml.safe_load(rendered)
    assert list(doc["services"]) == [SERVICE_KEY], doc["services"]
    return doc["services"][SERVICE_KEY]


def _env(rendered: str) -> dict[str, Any]:
    """The service's ``environment:`` mapping."""
    return _service(rendered).get("environment", {})


# ── The daemon shape ────────────────────────────────────────────────────────


def test_command_runs_the_watch_daemon() -> None:
    """``--watch`` is what makes the deployed mirror stay fresh.

    Without it the container runs one sync and exits, and the mirror freezes
    until the next ``osprey up`` — the exact staleness this service exists to
    remove.
    """
    assert _service(_render())["command"] == ["osprey", "ariel", "sync", "--watch"]


def test_restart_policy_is_unless_stopped() -> None:
    """A daemon is supposed to be running whenever the deployment is."""
    assert _service(_render())["restart"] == "unless-stopped"


def test_publishes_no_ports_and_declares_no_healthcheck() -> None:
    """The poller opens no listening socket, so it has nothing to publish and
    nothing an HTTP or TCP probe could ask.

    Its liveness question is answered by the ``ariel_last_ingestion`` health row
    reading the store, exactly as the archiver recorder's is.
    """
    service = _service(_render())
    assert "ports" not in service
    assert "healthcheck" not in service


@pytest.mark.parametrize("network", ["bridge", "host"])
def test_container_name_is_project_namespaced(network: str) -> None:
    """``container_name`` is a HOST-GLOBAL docker identifier.

    Two projects deploying a mirror on one host must not collide on one name,
    in either network mode.
    """
    block = {"network": network}
    name_a = _service(_render(project_name="proj-a", ariel_sync=block))["container_name"]
    name_b = _service(_render(project_name="proj-b", ariel_sync=block))["container_name"]

    assert name_a == "proj-a-ariel-sync"
    assert name_b == "proj-b-ariel-sync"
    assert not name_a.startswith("osprey-")


def test_image_is_the_project_worker_image_overridable_by_env() -> None:
    """The mirror runs the PROJECT image, like the dispatch worker.

    Same image as the workers and the web terminals, so the whole slice is one
    failure domain; ``OSPREY_WORKER_IMAGE`` overrides with a prebuilt tag.
    """
    rendered = _render()
    assert "image: ${OSPREY_WORKER_IMAGE:-" in rendered
    assert _service(rendered)["image"].startswith("${OSPREY_WORKER_IMAGE:-")


def test_service_image_key_overrides_the_default_fallback() -> None:
    """``services.ariel_sync.image`` is the inner fallback, as everywhere else."""
    rendered = _render(ariel_sync={"image": "registry.example/osprey:pinned"})
    assert _service(rendered)["image"] == "${OSPREY_WORKER_IMAGE:-registry.example/osprey:pinned}"


def test_declares_no_build_block() -> None:
    """The project image is built once by ``osprey up``.

    A second ``build:`` for the same tag would race the sole builder.
    """
    assert "build" not in _service(_render())


# ── Config, identity and recreate triggers ──────────────────────────────────


def test_mounts_the_staged_config_and_names_it_to_the_process() -> None:
    """The deploy-time config overlays the copy baked into the image.

    ``CONFIG_FILE`` points at that staged copy so the sync — and any subprocess
    it spawns — resolves config independently of CWD.
    """
    rendered = _render()
    service = _service(rendered)

    assert (
        "./build/services/ariel_sync/config.yml:/app/proj-a/build/config.yml:ro"
        in service["volumes"]
    )
    env = _env(rendered)
    assert env["CONFIG_FILE"] == "/app/proj-a/build/config.yml"
    assert env["OSPREY_PROJECT_DIR"] == "/app/proj-a"
    assert env["TZ"] == "UTC"


def test_carries_both_recreate_digest_labels() -> None:
    """Compose diffs the DOCUMENT, never the files a container reads.

    Without these two labels an edited config or env chain leaves the running
    container on the values it parsed at startup.
    """
    labels = _service(_render())["labels"]
    assert labels["osprey.config.digest"] == "${OSPREY_CONFIG_DIGEST:-}"
    assert labels["osprey.env.digest"] == "${OSPREY_ENV_DIGEST:-}"
    assert labels["osprey.project.name"] == "proj-a"


def test_env_chain_is_listed_in_ascending_precedence_when_present() -> None:
    """The chain files are read host-side by the compose CLI.

    Ascending precedence, so a later entry wins — the same order the
    invocation's own ``--env-file`` flags carry.
    """
    rendered = _render(env_chain=[".env.shared", ".env"])
    assert _service(rendered)["env_file"] == [".env.shared", ".env"]


def test_no_env_file_block_without_a_chain() -> None:
    """A deployment with no chain files gets no ``env_file:`` key at all.

    An entry naming a missing path errors ``compose up`` outright.
    """
    assert "env_file" not in _service(_render())


# ── Store address: it follows the deployment's topology ─────────────────────


def test_bridge_with_co_deployed_store_dials_the_network_alias() -> None:
    """On the bridge the store answers at its alias, on its CONTAINER port.

    ``ariel-postgres`` is pinned as a network alias in the postgresql template
    precisely so this name survives the per-project ``container_name``.
    """
    rendered = _render(postgresql={"port_host": 19800})
    env = _env(rendered)
    assert env["ARIEL_DATABASE_HOST"] == "ariel-postgres"
    assert env["ARIEL_DATABASE_PORT"] == "5432"


def test_host_mode_with_co_deployed_store_dials_the_published_port() -> None:
    """Under ``network: host`` the container's loopback IS the host's.

    The store is reached where it PUBLISHES, never on 5432 inside its
    container.
    """
    rendered = _render(ariel_sync={"network": "host"}, postgresql={"port_host": 19800})
    env = _env(rendered)
    assert env["ARIEL_DATABASE_HOST"] == "localhost"
    assert env["ARIEL_DATABASE_PORT"] == "19800"


def test_host_mode_store_port_falls_back_to_the_layout() -> None:
    """A ``services.postgresql`` block that omits ``port_host`` still renders a
    real port — the layout value every other reader of this store resolves,
    which moves with ``deployment.port_base``.
    """
    deployment = {"port_base": 21000}
    rendered = _render(
        ariel_sync={"network": "host"},
        postgresql={},
        deployment=deployment,
    )
    expected = _layout_ports_for(deployment)["postgres"]
    assert _env(rendered)["ARIEL_DATABASE_PORT"] == str(expected)


def test_host_mode_store_port_survives_an_absent_postgresql_block() -> None:
    """A deployment can deploy the store without authoring a config block for
    it; the render must not raise on the missing parent.
    """
    rendered = _render(ariel_sync={"network": "host"}, postgresql=None)
    expected = _layout_ports_for({})["postgres"]
    assert _env(rendered)["ARIEL_DATABASE_PORT"] == str(expected)


@pytest.mark.parametrize("network", ["bridge", "host"])
def test_co_deployed_store_is_health_gated(network: str) -> None:
    """The first ingest would otherwise race the store's startup.

    Health-gated rather than start-gated: "container started" is not "answering
    queries", and on a fresh volume postgres's init makes that window seconds
    long.
    """
    service = _service(_render(ariel_sync={"network": network}, postgresql={"port_host": 19800}))
    assert service["depends_on"] == {"postgresql": {"condition": "service_healthy"}}


@pytest.mark.parametrize("network", ["bridge", "host"])
def test_external_store_gets_neither_depends_on_nor_overrides(network: str) -> None:
    """A store this deployment does not run is already addressed correctly.

    Its ``ariel.database.uri`` names a real host, so overriding the address
    would point the mirror at a name that resolves to nothing — and a
    ``depends_on`` naming an undeployed service errors compose outright.
    """
    rendered = _render(ariel_sync={"network": network}, postgres_deployed=False)
    service = _service(rendered)
    env = service.get("environment", {})

    assert "depends_on" not in service
    assert "ARIEL_DATABASE_HOST" not in env
    assert "ARIEL_DATABASE_PORT" not in env


# ── The qmd mirror bind ─────────────────────────────────────────────────────


def test_mirror_is_bound_read_write_and_named_to_the_entrypoint() -> None:
    """The exporter runs inside THIS container and writes as the dropped user.

    Read-write like the web terminals' bind of the same directory, and named to
    the entrypoint so the mirror's group is joined before the privilege drop —
    ``group_add:`` alone does not survive ``gosu``.
    """
    rendered = _render(
        mirror_source="./var/ariel_mirror",
        container_mirror_dir="/app/proj-a/var/ariel_mirror",
    )
    service = _service(rendered)

    assert "./var/ariel_mirror:/app/proj-a/var/ariel_mirror" in service["volumes"]
    assert _env(rendered)["OSPREY_ARIEL_MIRROR_DIR"] == "/app/proj-a/var/ariel_mirror"


def test_host_mode_keeps_the_mirror_bind_and_its_entrypoint_variable() -> None:
    """The two conditionals are independent, and this is the shape that pairs
    them: a single-host facility deployment on the host namespace, with a qmd
    export writing a mirror.

    It is also the only shape where the host namespace and the entrypoint's
    ``OSPREY_ARIEL_MIRROR_DIR`` group join meet, so the mount, the variable and
    the network mode are asserted together rather than in separate renders.
    """
    rendered = _render(
        ariel_sync={"network": "host"},
        mirror_source="./var/ariel_mirror",
        container_mirror_dir="/app/proj-a/var/ariel_mirror",
    )
    service = _service(rendered)

    assert service["network_mode"] == "host"
    assert "./var/ariel_mirror:/app/proj-a/var/ariel_mirror" in service["volumes"]
    assert _env(rendered)["OSPREY_ARIEL_MIRROR_DIR"] == "/app/proj-a/var/ariel_mirror"


def test_absolute_mirror_source_is_bound_verbatim() -> None:
    """A mirror outside the deployment repo keeps its absolute host path."""
    rendered = _render(
        mirror_source="/srv/shared/ariel_mirror",
        container_mirror_dir="/srv/shared/ariel_mirror",
    )
    assert "/srv/shared/ariel_mirror:/srv/shared/ariel_mirror" in _service(rendered)["volumes"]


def test_no_mirror_lines_when_no_export_writes_one() -> None:
    """Both keys arrive as ``None``, not ``""``.

    Gating on truthiness is what keeps a deployment with no qmd export from
    rendering a mount with an empty host half — which compose accepts at render
    time and rejects at ``up``.
    """
    rendered = _render(mirror_source=None, container_mirror_dir=None)
    service = _service(rendered)

    assert service["volumes"] == [
        "./build/services/ariel_sync/config.yml:/app/proj-a/build/config.yml:ro"
    ]
    assert "OSPREY_ARIEL_MIRROR_DIR" not in _env(rendered)
    assert "None" not in rendered


# ── The shared axes ─────────────────────────────────────────────────────────


def test_bridge_joins_the_project_network_and_declares_it() -> None:
    """The default shape, rendered by the shared macro rather than by hand."""
    rendered = _render()
    doc = yaml.safe_load(rendered)
    assert doc["services"][SERVICE_KEY]["networks"] == ["osprey-network"]
    assert "osprey-network" in doc["networks"]


def test_host_mode_shares_the_host_namespace_and_declares_no_network() -> None:
    """Nothing in the file joins a network, so none is declared."""
    rendered = _render(ariel_sync={"network": "host"})
    doc = yaml.safe_load(rendered)
    service = doc["services"][SERVICE_KEY]

    assert service["network_mode"] == "host"
    assert "networks" not in service
    assert "networks" not in doc


def test_env_passthrough_axis_hands_declared_host_variables_through() -> None:
    """``services.ariel_sync.env`` names host variables the container needs —
    the proxy trio for reaching a facility logbook, for instance.
    """
    rendered = _render(ariel_sync={"env": ["HTTPS_PROXY", "NO_PROXY"]})
    env = _env(rendered)
    assert env["HTTPS_PROXY"] == "${HTTPS_PROXY}"
    assert env["NO_PROXY"] == "${NO_PROXY}"
