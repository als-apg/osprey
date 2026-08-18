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
        models_dir: /srv/osprey/qmd-models   # optional; see below

``models_dir`` is the one key that changes how the sidecar is *built*. By
default the image bakes its three GGUF models in at build time, which needs
build-host access to huggingface.co. On a network that has none, an operator
stages the same three files once on the host and names the directory here; the
build then skips the downloads and the models arrive at runtime over a
read-only bind mount instead. Default unset — behaviour is unchanged.

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
from pathlib import Path
from typing import Any

from osprey.deployment.errors import DeploymentPreconditionError

#: Host port the sidecar publishes. Deliberately NOT qmd's own default of
#: 8181: the daemon binds loopback only (``listen(port, "localhost")``,
#: hardcoded, no ``--host`` flag), so the container's entrypoint runs qmd on the
#: internal 8181 and fronts it with a forwarder that owns the routable port.
#: Keeping the two numbers distinct means "8181" always names the private
#: daemon and this port always names the endpoint clients talk to.
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

#: Config key naming a host directory of pre-staged GGUF model files. Spelled
#: once so the schema, the deploy-time preflight and its refusal text agree.
MODELS_DIR_CONFIG_KEY = "services.qmd.models_dir"

#: The exact filenames the three models must carry, in the order the sidecar
#: loads them: embedding, reranker, query expansion.
#:
#: These are not the huggingface filenames. node-llama-cpp — which qmd loads
#: its models through — caches a repo file as ``hf_<org>_<basename>``, and that
#: cached name is the *only* one it recognises as "already present". A file
#: under any other name, including the bare huggingface basename, leaves qmd
#: believing the model is absent, and it re-downloads: on a locked-down network
#: that hangs, and into a read-only mount it fails outright. Hence a preflight
#: that checks names rather than merely counting files.
#:
#: Kept in step with the model URIs pinned in
#: ``src/osprey/templates/services/qmd/Dockerfile`` — the same three files it
#: fetches and SHA256-verifies at build time.
MODEL_FILENAMES = (
    "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf",
    "hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf",
    "hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf",
)


@dataclass(frozen=True)
class QMDServiceConfig:
    """Resolved ``services.qmd`` settings for one deployment.

    Attributes:
        port: Published host port of the sidecar's HTTP endpoint.
        bind_address: Host interface the port is published on, taken from the
            project-wide ``deployment.bind_address``.
        interval_seconds: Fallback corpus-sweep period for the sidecar's
            update loop.
        models_dir: Absolute host directory holding the pre-staged GGUF models,
            or ``None`` (the default) to bake them into the image at build
            time. Validated for shape here and for contents by
            :func:`preflight_qmd_models_dir` at deploy time.
    """

    port: int = DEFAULT_PORT
    bind_address: str = DEFAULT_BIND_ADDRESS
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    models_dir: str | None = None

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
            integer, or if ``models_dir`` is present but not an absolute path.
            A silently-defaulted typo would point the client at the wrong
            endpoint or stall the update loop, which is far harder to diagnose
            than a refusal at resolution time.
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
        models_dir=_absolute_path(block.get("models_dir"), MODELS_DIR_CONFIG_KEY),
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


def preflight_qmd_models_dir(config: Mapping[str, Any] | None) -> None:
    """Refuse a deploy whose pre-staged model directory would not satisfy qmd.

    Runs only when ``services.qmd.models_dir`` is set; an unset key means the
    image bakes its own models and there is nothing to check.

    What this catches is a *silent* failure, which is why it refuses rather
    than warns. The mount is read-only and the models are absent, so the
    sidecar starts, decides it must download them, and cannot — on a host
    configured this way because it has no route to huggingface.co, that is a
    hang rather than an error. The container's health check allows an hour of
    start-up for the model load it normally does, so the first honest signal an
    operator gets is an unhealthy container an hour later, with the cause an
    hour behind it. Checking three filenames on the host beforehand costs
    nothing and turns that into an immediate refusal naming the fix.

    Args:
        config: A loaded project config mapping, or ``None``.

    Raises:
        DeploymentPreconditionError: If the key is set and the directory is
            missing, or does not hold all three models under the exact names
            qmd recognises. Nothing has been started when this raises.
        ValueError: Propagated from :func:`resolve_qmd_service_config` for a
            malformed ``services.qmd`` block.
    """
    resolved = resolve_qmd_service_config(config)
    if resolved is None or resolved.models_dir is None:
        return

    directory = Path(resolved.models_dir)
    if not directory.is_dir():
        raise DeploymentPreconditionError(
            reason=(
                f"{MODELS_DIR_CONFIG_KEY} names a host directory that does not "
                f"exist: {directory}. Setting it says the qmd sidecar's models are "
                "staged on this host instead of downloaded during the image build, "
                "so there is nothing for the build to skip and nothing for the "
                "container to load."
            ),
            remedy=_models_staging_remedy(directory),
        )

    missing = [name for name in MODEL_FILENAMES if not _staged(directory / name)]
    if missing:
        raise DeploymentPreconditionError(
            reason=(
                f"{MODELS_DIR_CONFIG_KEY} is set to {directory}, but "
                f"{len(missing)} of the sidecar's {len(MODEL_FILENAMES)} models are "
                "not staged there under the names qmd looks for (missing, or present "
                "but empty):\n"
                + "\n".join(f"    {name}" for name in missing)
                + "\n\nqmd loads models through node-llama-cpp, which recognises a "
                "staged model only by its `hf_<org>_<basename>` cache name. Under any "
                "other name — the bare download name included — the file reads as "
                "absent and the sidecar tries to fetch it into a read-only mount, on "
                "a host configured this way precisely because it cannot reach "
                "huggingface.co."
            ),
            remedy=_models_staging_remedy(directory),
        )


def _models_staging_remedy(directory: Path) -> str:
    """The staging instructions both model-directory refusals end on.

    Args:
        directory: The configured host directory, named so the operator does
            not have to re-read config.yml to act on the message.

    Returns:
        A multi-line remedy listing all three filenames verbatim.
    """
    listing = "\n".join(f"    {name}" for name in MODEL_FILENAMES)
    return (
        f"Stage all three model files in {directory}, named exactly:\n"
        f"{listing}\n"
        "Copy them from a host that can reach huggingface.co, or from the qmd model\n"
        "cache of a machine that has already built the sidecar image; the source URLs\n"
        "and their SHA256 pins are in the qmd service's Dockerfile. Rename each to the\n"
        "`hf_<org>_<basename>` form above — the download names differ — then re-run\n"
        "the deploy.\n"
        f"To go back to build-time downloads, remove {MODELS_DIR_CONFIG_KEY} from "
        "config.yml."
    )


def _staged(path: Path) -> bool:
    """True when ``path`` is a non-empty regular file.

    Emptiness counts as absence: an interrupted copy leaves a zero-byte file
    that would otherwise pass a bare existence check and fail in the container.
    """
    return path.is_file() and path.stat().st_size > 0


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


def _absolute_path(value: Any, key: str) -> str | None:
    """Coerce a config value to an absolute host path, or ``None`` when unset.

    Args:
        value: The raw value read from config, possibly ``None``.
        key: Dotted config key, named in the error message.

    Returns:
        ``None`` when ``value`` is ``None``, otherwise the stripped path.

    Raises:
        ValueError: If ``value`` is present but not a non-empty absolute path.
            Absoluteness is required rather than resolved-for: a relative path
            means one directory to this process and another to the container
            runtime, which resolves it against the compose file instead, and
            neither of them expands ``~``. Accepting one would let the check
            below pass against a directory the mount never sees.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty absolute host path, got {value!r}")
    path = value.strip()
    if not Path(path).is_absolute():
        raise ValueError(
            f"{key} must be an absolute host path, got {path!r}. A relative path "
            "resolves differently here than in the container runtime, and neither "
            "expands '~'."
        )
    return path
