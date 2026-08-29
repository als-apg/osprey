"""The ``services.graphdb`` config schema: defaults, resolution, and its preflight.

The graph store is a Neo4j container that holds a *disposable mirror* of an
RDF/Turtle corpus: the TTL on disk is the source of truth, and re-seeding
rebuilds the graph from it. Several surfaces have to agree on the same handful
of numbers — the compose fragment that publishes the two ports, the deploy-time
port-conflict sweep that names the key to move one of them, the health category
that dials the store, and the seeder that connects over bolt — so the schema is
resolved once here rather than re-spelled at each of them.

The name is deliberately backend-neutral. ``graphdb`` says "the graph store this
deployment runs", not "Neo4j", so a triple-store variant can be slotted in later
without renaming a config key an operator has already written down.

Config shape, as it appears in a project's ``config.yml``::

    services:
      graphdb:
        path: ./services/graphdb
        image: neo4j:5.26-community
        port_host: 10802                           # bolt, the graphdb_bolt slot
        http_port_host: 10803                      # HTTP, the graphdb_http slot
        ttl_path: ./data/demo_machine.ttl          # optional; corpus to seed
        heap_initial_size: 512m
        heap_max_size: 1G
        pagecache_size: 512m

Two published ports, not one, and both are load-bearing: the driver speaks bolt
and nothing else, while the operator-facing Browser and the health probe speak
HTTP. Each protocol keeps its own container port and its own host slot, which
is why they get one dotted-key constant each — a port-conflict refusal has to
name *which* of the two an operator must move.

``bind_address`` is deliberately NOT a per-service key. Every deployed service
publishes on the project-wide ``deployment.bind_address`` (default
``127.0.0.1``), and the graph store is no exception; :func:`resolve_bind_address`
— imported from :mod:`osprey.deployment.qmd_service`, which owns that one
reading — is reused here so a project cannot end up with the store exposed on an
interface the rest of the stack is not.

There is no ``password`` key, and adding one would accomplish nothing: the
container reads ``GRAPHDB_PASSWORD`` from the project ``.env``, which ``osprey
up`` mints when it is unset. A password written into ``config.yml`` would be
read by nobody and would silently lose to the minted value.

``uri`` and ``username`` may also appear in this block, and are deliberately
absent from :class:`GraphdbServiceConfig`. They describe an *external* store —
one this deployment connects to but does not run — so they say nothing about how
a container is rendered, published, or preflighted, which is what that schema
answers. They are read by :func:`resolve_graphdb_connection` instead, which
resolves the other half of this block: what to dial, as whom, with what
password. Keeping the two resolvers separate is what lets one deployment run a
store it does not connect to, and another connect to a store it does not run.

Security posture
----------------
The store publishes on ``deployment.bind_address`` like every other service,
and it authenticates: the composite ``NEO4J_AUTH`` carries the minted
``GRAPHDB_PASSWORD``. So moving the bind address off loopback exposes an
authenticated endpoint rather than an open one — but it is a single shared
credential with full write access to the graph, and the graph is rebuilt from
the TTL rather than backed up, so the blast radius of losing it is the corpus's
integrity rather than merely its confidentiality.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.qmd_service import DEFAULT_BIND_ADDRESS, resolve_bind_address
from osprey.port_layout import default_port, resolve_port_base

__all__ = [
    "CONTAINER_BOLT_PORT",
    "CONTAINER_HTTP_PORT",
    "DEFAULT_BIND_ADDRESS",
    "DEFAULT_HEAP_INITIAL_SIZE",
    "DEFAULT_HEAP_MAX_SIZE",
    "DEFAULT_HTTP_PORT",
    "DEFAULT_IMAGE",
    "DEFAULT_PAGECACHE_SIZE",
    "DEFAULT_PASSWORD",
    "DEFAULT_PORT",
    "DEFAULT_USERNAME",
    "GRAPHDB_HTTP_PORT_CONFIG_KEY",
    "GRAPHDB_PASSWORD_ENV",
    "GRAPHDB_PORT_CONFIG_KEY",
    "GRAPHDB_SEED_COMMAND",
    "GRAPHDB_SERVICE_NAME",
    "GraphdbConnection",
    "GraphdbServiceConfig",
    "preflight_graphdb_config",
    "resolve_bind_address",
    "resolve_graphdb_connection",
    "resolve_graphdb_service_config",
]

#: Compose service name, the name that appears in ``deployed_services`` and the
#: name the container's network alias pins. Spelled once so the preflight, its
#: refusal text, and the registries that key off it cannot drift apart.
GRAPHDB_SERVICE_NAME = "graphdb"

#: Image the store runs by default, overridable per project.
#:
#: Pinned to the 5.26 LTS line rather than tracking a newer Neo4j: neosemantics
#: (n10s) — the plugin that imports RDF, without which the store has no way to
#: load the corpus at all — is installed at container start from its plugin
#: manifest (https://neo4j-labs.github.io/neosemantics/versions.json), and 5.26
#: is the newest server line that manifest covers; it does so patch-by-patch,
#: where older lines get at most a single frozen entry. The pin is therefore a
#: statement about the plugin's release cadence, not about Neo4j's, and it can
#: move as soon as n10s ships for a newer server. Check the manifest — and the
#: supported-set pin test beside this module's config tests — before bumping.
DEFAULT_IMAGE = "neo4j:5.26-community"

#: Container-internal port the Neo4j image serves bolt on. Fixed by the image,
#: so it is the same number in every deployment no matter which host port that
#: deployment publishes it as — which is why a binding, a remedy-key lookup or a
#: browser-vs-binary decision keyed on "which of the store's two ports is this"
#: keys on this rather than on the host port below.
CONTAINER_BOLT_PORT = 7687

#: Container-internal port the Neo4j image serves HTTP on — Browser and
#: healthcheck. Image-fixed for the same reason as :data:`CONTAINER_BOLT_PORT`.
CONTAINER_HTTP_PORT = 7474

#: Published host port for the bolt protocol, which is the only thing the neo4j
#: driver speaks — the ``graphdb_bolt`` layout slot **at the layout's default
#: base**, which is right only for a caller with no config to resolve a base
#: from. :func:`resolve_graphdb_service_config` derives it at the base its
#: config resolved instead; this constant is the dataclass field's default.
#: No longer the container-internal port: that is :data:`CONTAINER_BOLT_PORT`.
DEFAULT_PORT = default_port("graphdb_bolt")

#: Published host port for HTTP: the Neo4j Browser an operator opens, and the
#: endpoint the container healthcheck probes. Not what the health category uses
#: — that dials the store over bolt through :func:`resolve_graphdb_connection`,
#: so its remedies name ``port_host``/``uri`` rather than this key. The
#: ``graphdb_http`` layout slot at the default base, on the same terms as
#: :data:`DEFAULT_PORT`; the image's own HTTP port is
#: :data:`CONTAINER_HTTP_PORT`.
DEFAULT_HTTP_PORT = default_port("graphdb_http")

#: JVM heap the server starts with, and the ceiling it may grow to. Neo4j sizes
#: nothing automatically inside a container — left unset it reads the *host's*
#: memory and picks numbers the container's own limit will not honour — so both
#: ends are spelled explicitly. These are the values the working prototype ran a
#: 8k-triple corpus on; a substantially larger graph wants them raised.
DEFAULT_HEAP_INITIAL_SIZE = "512m"
DEFAULT_HEAP_MAX_SIZE = "1G"

#: Off-heap cache for graph data and indexes. Separate from the heap and *not*
#: taken out of it: the container's memory budget is roughly heap max plus page
#: cache plus about half a gigabyte of JVM overhead.
DEFAULT_PAGECACHE_SIZE = "512m"

#: Account the store authenticates as. Not a default an operator overrides for a
#: store this deployment runs: the container's composite ``NEO4J_AUTH`` is
#: ``neo4j/${GRAPHDB_PASSWORD}``, so ``neo4j`` is the only account that exists on
#: it. It is a *default* only for an external store reached through an explicit
#: ``uri``, where the operator names whichever account they provisioned.
DEFAULT_USERNAME = "neo4j"

#: Environment variable both ends of the credential read: the compose service
#: interpolates it into ``NEO4J_AUTH``, and this module reads it back to dial the
#: store. Spelled once so the mint, the template and the resolver cannot drift.
GRAPHDB_PASSWORD_ENV = "GRAPHDB_PASSWORD"

#: Password used when ``GRAPHDB_PASSWORD`` is unset, matching the
#: ``${GRAPHDB_PASSWORD:-ospreygraph}`` fallback the compose service carries — so
#: a project stays dialable before the first ``osprey up`` mints a real one, and
#: so both ends agree on what "unset" means. Deliberately weak and deliberately
#: registered in the forbidden-values table: it is a placeholder, and a
#: deployment still running on it should be caught saying so.
DEFAULT_PASSWORD = "ospreygraph"

#: Config key an operator edits to move the published bolt port. Spelled once so
#: the deploy-time port-conflict remedy and the schema cannot drift apart.
GRAPHDB_PORT_CONFIG_KEY = "services.graphdb.port_host"

#: Config key an operator edits to move the published HTTP port. The graph store
#: is the first single compose service to publish two host ports, which is why
#: the port-conflict remedy has to resolve per container port rather than per
#: service — a conflict on 7474 that told the operator to move ``port_host``
#: would send them to edit the wrong number.
GRAPHDB_HTTP_PORT_CONFIG_KEY = "services.graphdb.http_port_host"

#: The verb that fills or repairs the graph by hand, named by every surface that
#: has to send an operator to it: the deploy-time staging step's warnings and the
#: health category's empty-graph remedy. Not a config key, but it lives here for
#: the same reason those do — two modules that spell one operator-facing string
#: separately agree only by coincidence, and an operator following a stale copy
#: of it is told to run a command that no longer exists.
GRAPHDB_SEED_COMMAND = "osprey knowledge seed-graph"

#: A JVM memory size: digits with an optional unit suffix, as Neo4j's own
#: settings spell them (``512m``, ``1G``, ``2048k``, or bare bytes).
_MEMORY_SIZE = re.compile(r"^\d+[kKmMgG]?$")


@dataclass(frozen=True)
class GraphdbServiceConfig:
    """Resolved ``services.graphdb`` settings for one deployment.

    Attributes:
        image: Container image to run. Overridable for a mirror or a pinned
            digest; see :data:`DEFAULT_IMAGE` for why the default is held back.
        port_host: Published host port for bolt, the driver protocol.
        http_port_host: Published host port for HTTP — Browser and health probe.
        bind_address: Host interface both ports are published on, taken from the
            project-wide ``deployment.bind_address``.
        heap_initial_size: JVM heap the server starts with.
        heap_max_size: Ceiling the JVM heap may grow to.
        pagecache_size: Off-heap cache for graph data and indexes.
        ttl_path: Corpus to seed the graph from, or ``None`` (the default) to
            bring the store up bootstrapped but empty. Kept verbatim: it is
            resolved against the config file's directory by its consumers, per
            the ``facility_knowledge.bundle_path`` rule, and resolving it here
            against the process's cwd instead would give a deploy and a CLI verb
            two different files from one config value.
    """

    image: str = DEFAULT_IMAGE
    port_host: int = DEFAULT_PORT
    http_port_host: int = DEFAULT_HTTP_PORT
    bind_address: str = DEFAULT_BIND_ADDRESS
    heap_initial_size: str = DEFAULT_HEAP_INITIAL_SIZE
    heap_max_size: str = DEFAULT_HEAP_MAX_SIZE
    pagecache_size: str = DEFAULT_PAGECACHE_SIZE
    ttl_path: str | None = None


@dataclass(frozen=True)
class GraphdbConnection:
    """Everything needed to open a driver session against the graph store.

    Three fields rather than one DSN string, because that is the shape the neo4j
    driver actually takes: ``GraphDatabase.driver(uri, auth=(username,
    password))``. Credentials therefore never enter the URI — no
    ``bolt://user:pass@host`` to leak into a log line, an error message, or a
    ``docker compose`` label. This is the archiver's ``password_env`` split
    rather than the Postgres embedded-DSN one.

    Consumers should keep to driver APIs common to the 5.x and 6.x lines
    (``execute_query``, ``execute_read``): the repo's ``neo4j>=5.20`` floor
    resolves to the 6.x driver, while the server it dials is pinned to 5.26.

    Attributes:
        uri: Address to dial — an explicit external one, or the derived
            ``bolt://localhost:{port_host}``.
        username: Account to authenticate as. See :data:`DEFAULT_USERNAME` for
            why it is configurable only on the explicit path.
        password: The store's password. Not part of ``repr``, so a dataclass that
            lands in a traceback or a debug log does not carry the credential
            with it; it still participates in equality and comparison.
    """

    uri: str
    username: str
    password: str = field(repr=False)


def resolve_graphdb_service_config(
    config: Mapping[str, Any] | None,
) -> GraphdbServiceConfig | None:
    """Resolve the ``services.graphdb`` block, or ``None`` when it is absent.

    Absence is a state, not a failure: most deployments run no graph store, and
    callers are expected to take their non-graph path silently rather than probe
    a store that was never deployed. Only a present-but-partial block gets
    defaults filled in. The one case where absence *is* a failure — a block-less
    ``graphdb`` in ``deployed_services`` — is caught by
    :func:`preflight_graphdb_config`, which knows the deployment's intent and
    this function does not.

    Args:
        config: A loaded project config mapping, or ``None``.

    Returns:
        The resolved settings, or ``None`` if ``config`` is ``None`` or carries
        no ``services.graphdb`` block.

    Raises:
        ValueError: If a present key is malformed — a port that is not a
            positive integer in range, a blank ``image`` or ``ttl_path``, or a
            memory size the JVM would not parse. Silently defaulting instead
            would publish a port nobody is dialing, or hand Neo4j a heap setting
            it rejects at startup — both far harder to trace back to a typo in
            ``config.yml`` than a refusal at resolution time.
    """
    block = _graphdb_block(config)
    if block is None:
        return None
    return GraphdbServiceConfig(
        image=_text(block.get("image"), DEFAULT_IMAGE, "services.graphdb.image"),
        # Both defaults come from the base THIS config resolved, never from the
        # layout's own: two deployments on one host differ only by their
        # ``deployment.port_base``, and a store defaulted to the module's base
        # would publish on top of the other deployment's graph store.
        port_host=_port(
            block.get("port_host"),
            default_port("graphdb_bolt", base=resolve_port_base(config)),
            GRAPHDB_PORT_CONFIG_KEY,
        ),
        http_port_host=_port(
            block.get("http_port_host"),
            default_port("graphdb_http", base=resolve_port_base(config)),
            GRAPHDB_HTTP_PORT_CONFIG_KEY,
        ),
        bind_address=resolve_bind_address(config),
        heap_initial_size=_memory_size(
            block.get("heap_initial_size"),
            DEFAULT_HEAP_INITIAL_SIZE,
            "services.graphdb.heap_initial_size",
        ),
        heap_max_size=_memory_size(
            block.get("heap_max_size"), DEFAULT_HEAP_MAX_SIZE, "services.graphdb.heap_max_size"
        ),
        pagecache_size=_memory_size(
            block.get("pagecache_size"), DEFAULT_PAGECACHE_SIZE, "services.graphdb.pagecache_size"
        ),
        ttl_path=_optional_text(block.get("ttl_path"), "services.graphdb.ttl_path"),
    )


def resolve_graphdb_connection(
    graphdb_section: Mapping[str, Any] | None,
    services: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    base: int | None = None,
) -> GraphdbConnection:
    """Resolve what to dial, as whom, and with which password.

    ``services.graphdb.uri`` is optional: with it unset the address is derived
    from the port the store is published on, so a project that moves
    ``port_host`` stays connectable without editing a second, duplicated copy of
    the same number. Setting it points this deployment at a store it does not
    run — the external-store case, which has no local port to derive from.

    Precedence:
      1. An explicit ``uri`` wins, paired with ``username`` (default
         :data:`DEFAULT_USERNAME`). The address is used as written, never
         rebuilt from the local port.
      2. Otherwise ``bolt://localhost:{port_host}`` as :data:`DEFAULT_USERNAME`,
         with ``port_host`` defaulting to the ``graphdb_bolt`` slot at ``base``.
         A configured ``username`` is deliberately *not* honored here: the
         container authenticates on the composite ``neo4j/${GRAPHDB_PASSWORD}``,
         so any other account would be one the store does not have.

    The password is read the same way on both paths — from
    :data:`GRAPHDB_PASSWORD_ENV`, falling back to :data:`DEFAULT_PASSWORD` —
    because an external store is credentialed by the operator setting that
    variable, exactly as the deployed one is credentialed by ``osprey up``
    minting it. There is no ``password`` config key, and adding one would be read
    by nobody.

    Args:
        graphdb_section: The ``services.graphdb`` block from config.yml, or
            ``None`` when the project has none — which resolves to the shipped
            defaults rather than refusing, since a caller that needs "is there a
            store at all?" answered asks :func:`resolve_graphdb_service_config`.
        services: The ``services.graphdb`` mapping to read ``port_host`` from,
            for a caller that holds the deployed service's settings separately
            from the connection keys. Defaults to ``graphdb_section``, which is
            where both normally live in the same block.
        env: Where to read :data:`GRAPHDB_PASSWORD_ENV` from. Defaults to the
            process environment, which is what every in-container consumer
            wants. A caller acting ON a project rather than IN it — the deploy
            that seeds a store it is bringing up — passes that project's own
            parsed ``.env`` instead: run from another directory, the ambient
            value belongs to some other deployment, and using it would either
            fail confusingly or reach a store this call was never pointed at.
        base: The port base this deployment resolved, from
            :func:`osprey.port_layout.resolve_port_base`. Only consulted when no
            ``port_host`` is set, in which case the derived address dials the
            ``graphdb_bolt`` slot of *this* deployment's block — the same number
            :func:`resolve_graphdb_service_config` publishes it on, which is the
            whole point: dial and publish must agree. ``None`` means the
            layout's own default base, which is right only for a caller with no
            config to resolve one from.

    Returns:
        The address and credentials to open a driver with.

    Raises:
        ValueError: If ``uri`` or ``username`` is present but blank or not a
            string, or if ``port_host`` is not a TCP port — the same refusals
            :func:`resolve_graphdb_service_config` gives for the same keys, so a
            typo cannot resolve here to ``bolt://localhost:True`` while refusing
            two lines earlier in the deploy.
    """
    section = graphdb_section or {}

    uri = _optional_text(section.get("uri"), "services.graphdb.uri")
    if uri is not None:
        username = _text(section.get("username"), DEFAULT_USERNAME, "services.graphdb.username")
        return GraphdbConnection(uri=uri, username=username, password=_password(env))

    port = _port(
        (services if services is not None else section).get("port_host"),
        default_port("graphdb_bolt", base=base),
        GRAPHDB_PORT_CONFIG_KEY,
    )
    return GraphdbConnection(
        uri=f"bolt://localhost:{port}",
        username=DEFAULT_USERNAME,
        password=_password(env),
    )


def _password(env: Mapping[str, str] | None) -> str:
    """Read the store's password from an environment mapping.

    Args:
        env: The mapping to read from, or ``None`` for the process environment.

    Returns:
        The value of :data:`GRAPHDB_PASSWORD_ENV`, or :data:`DEFAULT_PASSWORD`
        when it is unset.
    """
    return (env if env is not None else os.environ).get(GRAPHDB_PASSWORD_ENV, DEFAULT_PASSWORD)


def preflight_graphdb_config(config: Mapping[str, Any] | None) -> None:
    """Refuse a deploy that lists ``graphdb`` but describes no graph store.

    The block is required rather than defaulted, which is a deliberate departure
    from how an absent block is treated everywhere else in this module. A
    defaulted store would come up on ports nothing was told about and with no
    corpus to seed, and the operator's first signal would be an empty graph that
    every query answers with zero rows — a shape that reads as "the data is
    wrong" rather than "the service was never configured". Naming the missing
    block at deploy time costs one dictionary lookup and points straight at the
    fix.

    Args:
        config: A loaded project config mapping, or ``None``.

    Raises:
        DeploymentPreconditionError: If ``graphdb`` appears in
            ``deployed_services`` with no ``services.graphdb`` block. Nothing has
            been started when this raises.
        ValueError: Propagated from :func:`resolve_graphdb_service_config` for a
            malformed ``services.graphdb`` block.
    """
    if not _is_deployed(config):
        return
    if resolve_graphdb_service_config(config) is not None:
        return
    raise DeploymentPreconditionError(
        reason=(
            f"'{GRAPHDB_SERVICE_NAME}' is listed in deployed_services, but config.yml "
            "carries no 'services.graphdb' block. That block is where the graph store's "
            "ports, image and seed corpus are declared, so without it the deploy has "
            "nothing to render a compose fragment from and nothing to seed the graph "
            "with."
        ),
        remedy=(
            "Add a 'services.graphdb' block to config.yml, for example:\n"
            "    services:\n"
            "      graphdb:\n"
            "        path: ./services/graphdb\n"
            f"        port_host: {DEFAULT_PORT}\n"
            f"        http_port_host: {DEFAULT_HTTP_PORT}\n"
            "        ttl_path: ./data/demo_machine.ttl\n"
            f"Or drop '{GRAPHDB_SERVICE_NAME}' from deployed_services if this deployment "
            "should not run a graph store."
        ),
    )


def _graphdb_block(config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return the ``services.graphdb`` mapping, or ``None`` when it is absent.

    Args:
        config: A loaded project config mapping, or ``None``.

    Returns:
        The block, or ``None`` when any level of the path is missing or is not a
        mapping — which includes the bare ``graphdb:`` key-with-no-value that
        YAML parses as ``None``.
    """
    if not config:
        return None
    services = config.get("services")
    if not isinstance(services, Mapping):
        return None
    block = services.get(GRAPHDB_SERVICE_NAME)
    return block if isinstance(block, Mapping) else None


def _is_deployed(config: Mapping[str, Any] | None) -> bool:
    """True when ``graphdb`` is listed in ``deployed_services``.

    Args:
        config: A loaded project config mapping, or ``None``.

    Returns:
        Whether this deployment intends to run the graph store. Entries are
        compared as strings, matching how every other reader of the list
        normalizes it.
    """
    if not config:
        return False
    deployed = config.get("deployed_services") or []
    if isinstance(deployed, (str, bytes)) or not isinstance(deployed, Iterable):
        return False
    return GRAPHDB_SERVICE_NAME in {str(service) for service in deployed}


def _port(value: Any, default: int, key: str) -> int:
    """Coerce a config value to a publishable TCP port, or fall back.

    Args:
        value: The raw value read from config, possibly ``None``.
        default: Value to use when the key is absent.
        key: Dotted config key, named in the error message.

    Returns:
        ``default`` when ``value`` is ``None``, otherwise the port itself.

    Raises:
        ValueError: If ``value`` is present but not an integer in 1–65535.
            ``bool`` is rejected explicitly — it is an ``int`` subclass, so
            ``port_host: true`` would otherwise resolve to port 1.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{key} must be an integer TCP port between 1 and 65535, got {value!r}")
    return int(value)


def _memory_size(value: Any, default: str, key: str) -> str:
    """Coerce a config value to a JVM memory size, or fall back to ``default``.

    Args:
        value: The raw value read from config, possibly ``None``.
        default: Value to use when the key is absent.
        key: Dotted config key, named in the error message.

    Returns:
        ``default`` when ``value`` is ``None``, otherwise the stripped size.

    Raises:
        ValueError: If ``value`` is present but not digits with an optional
            ``k``/``m``/``g`` suffix. The shape is checked here because the
            server does not fail loudly on a bad one — a size it cannot parse
            leaves the JVM refusing to start with a message that names the
            setting rather than the file it came from.
    """
    if value is None:
        return default
    if not isinstance(value, str) or not _MEMORY_SIZE.match(value.strip()):
        raise ValueError(
            f"{key} must be a JVM memory size — digits with an optional k/m/g suffix, "
            f"such as '512m' or '1G' — got {value!r}"
        )
    return value.strip()


def _text(value: Any, default: str, key: str) -> str:
    """Coerce a config value to a non-blank string, or fall back to ``default``.

    Args:
        value: The raw value read from config, possibly ``None``.
        default: Value to use when the key is absent.
        key: Dotted config key, named in the error message.

    Returns:
        ``default`` when ``value`` is ``None``, otherwise the stripped text.

    Raises:
        ValueError: If ``value`` is present but not a non-blank string. A blank
            one is a half-finished edit, and taking the default for it would
            hide the edit rather than honour it.
    """
    return _optional_text(value, key) or default


def _optional_text(value: Any, key: str) -> str | None:
    """Coerce a config value to a non-blank string, or ``None`` when unset.

    Args:
        value: The raw value read from config, possibly ``None``.
        key: Dotted config key, named in the error message.

    Returns:
        ``None`` when ``value`` is ``None``, otherwise the stripped text.

    Raises:
        ValueError: If ``value`` is present but not a non-blank string.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    return value.strip()
