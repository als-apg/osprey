"""The ``services.qmd`` config schema: defaults, resolution, and its base URL.

The qmd sidecar is a container that indexes the deployment's markdown corpora
and answers semantic queries over HTTP. Three surfaces need to agree on the
same three numbers — the compose fragment that publishes the port, the sidecar
entrypoint that drives the update loop, and the Python client that queries it —
so the schema is resolved once here rather than re-spelled at each of them.

Config shape, as it appears in a project's ``config.yml``::

    services:
      qmd:
        path: ./services/qmd
        port: 8180
        interval: 30

``bind_address`` is deliberately NOT a per-service key. Every deployed service
publishes on the project-wide ``deployment.bind_address`` (default
``127.0.0.1``), and the qmd sidecar is no exception; :func:`resolve_bind_address`
reads that one key so a project cannot end up with the sidecar exposed on an
interface the rest of the stack is not. See the "security posture" note below
for why that default is load-bearing rather than cosmetic.

Security posture
----------------
The sidecar endpoint is **loopback-trusted-unauthenticated**, per the repo's
internal-API baseline: it carries no token, no TLS, and no per-caller identity,
and it will answer any request that reaches it. That is safe exactly as long as
only the host's own containers can reach it, which is what the default
``127.0.0.1`` publish guarantees. Moving ``deployment.bind_address`` off
loopback publishes an unauthenticated search endpoint over the whole corpus to
whatever can route to that interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Host port the sidecar publishes. Deliberately NOT qmd's own default of
#: 8181: the daemon binds IPv6 loopback only (``[::1]:8181``, hardcoded, no
#: ``--host`` flag), so the container's entrypoint runs qmd on the internal
#: 8181 and fronts it with a forwarder that owns the routable port. Keeping the
#: two numbers distinct means "8181" always names the private daemon and this
#: port always names the endpoint clients talk to.
DEFAULT_PORT = 8180

#: Interface every deployed service publishes on when ``deployment`` is absent.
#: Matches the ``| default('127.0.0.1')`` the service compose templates spell.
DEFAULT_BIND_ADDRESS = "127.0.0.1"

#: Seconds between fallback corpus sweeps, and NOT the expected freshness lag —
#: a corpus writer touches its ``.qmd-touch`` marker, which triggers an update
#: within a poll rather than waiting out a full interval. The interval only
#: catches writers that forgot to touch the marker.
#:
#: Tune it against the measured cost of the sweep it schedules: a no-op ``qmd
#: update`` that finds nothing changed costs about **0.3 s at 2k documents and
#: 12.5 s at 135k** (scan cost grows with corpus size, not with the number of
#: changes). At 135k documents this default leaves the update loop busy roughly
#: 42% of the time discovering nothing, so a large corpus should raise it —
#: marker-driven updates keep freshness regardless.
DEFAULT_INTERVAL_SECONDS = 30

#: Config key an operator edits to move the published host port. Spelled once
#: so the deploy-time port-conflict remedy and the schema cannot drift apart.
PORT_CONFIG_KEY = "services.qmd.port"


@dataclass(frozen=True)
class QMDServiceConfig:
    """Resolved ``services.qmd`` settings for one deployment.

    Attributes:
        port: Published host port of the sidecar's HTTP endpoint.
        bind_address: Host interface the port is published on, taken from the
            project-wide ``deployment.bind_address``.
        interval_seconds: Fallback corpus-sweep period for the sidecar's
            update loop.
    """

    port: int = DEFAULT_PORT
    bind_address: str = DEFAULT_BIND_ADDRESS
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS

    @property
    def base_url(self) -> str:
        """URL a client on the host reaches the sidecar at.

        Always loopback, never ``bind_address``: a wildcard publish
        (``0.0.0.0``) is an address to *listen* on, not one to connect to, and
        a host-side caller reaches every published port over loopback anyway.
        """
        return f"http://127.0.0.1:{self.port}"


def resolve_qmd_service_config(config: Mapping[str, Any] | None) -> QMDServiceConfig | None:
    """Resolve the ``services.qmd`` block, or ``None`` when it is absent.

    Absence is a state, not a failure: a deployment with no qmd sidecar is a
    supported configuration, and callers are expected to take their non-qmd
    path silently rather than probe an endpoint that was never deployed. Only a
    present-but-partial block gets defaults filled in.

    Args:
        config: A loaded project config mapping, or ``None``.

    Returns:
        The resolved settings, or ``None`` if ``config`` is ``None`` or carries
        no ``services.qmd`` block.

    Raises:
        ValueError: If ``port`` or ``interval`` is present but not a positive
            integer. A silently-defaulted typo would point the client at the
            wrong endpoint or stall the update loop, which is far harder to
            diagnose than a refusal at resolution time.
    """
    if not config:
        return None
    services = config.get("services")
    if not isinstance(services, Mapping):
        return None
    block = services.get("qmd")
    if not isinstance(block, Mapping):
        return None
    return QMDServiceConfig(
        port=_positive_int(block.get("port"), DEFAULT_PORT, PORT_CONFIG_KEY),
        bind_address=resolve_bind_address(config),
        interval_seconds=_positive_int(
            block.get("interval"), DEFAULT_INTERVAL_SECONDS, "services.qmd.interval"
        ),
    )


def resolve_bind_address(config: Mapping[str, Any] | None) -> str:
    """Return ``deployment.bind_address``, defaulting to loopback.

    Args:
        config: A loaded project config mapping, or ``None``.

    Returns:
        The configured interface, or :data:`DEFAULT_BIND_ADDRESS` when the
        ``deployment`` block is absent or sets no address.
    """
    if not config:
        return DEFAULT_BIND_ADDRESS
    deployment = config.get("deployment")
    if not isinstance(deployment, Mapping):
        return DEFAULT_BIND_ADDRESS
    address = deployment.get("bind_address")
    if not isinstance(address, str) or not address.strip():
        return DEFAULT_BIND_ADDRESS
    return address.strip()


def _positive_int(value: Any, default: int, key: str) -> int:
    """Coerce a config value to a positive int, or fall back to ``default``.

    Args:
        value: The raw value read from config, possibly ``None``.
        default: Value to use when the key is absent.
        key: Dotted config key, named in the error message.

    Returns:
        ``default`` when ``value`` is ``None``, otherwise the value itself.

    Raises:
        ValueError: If ``value`` is present but not a positive integer.
            ``bool`` is rejected explicitly — it is an ``int`` subclass, so
            ``port: true`` would otherwise resolve to port 1.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return int(value)
