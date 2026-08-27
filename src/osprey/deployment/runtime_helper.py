"""Simple container runtime detection for Docker and Podman.

Detects available container runtime and provides command helpers.
Requires modern runtimes: Docker Desktop 4.0+ or Podman 4.0+.

Examples:
    Basic usage::

        from osprey.deployment.runtime_helper import get_runtime_command

        cmd = get_runtime_command(config)
        # Returns: ['docker', 'compose'] or ['podman', 'compose']
"""

import hashlib
import os
import re
import shutil
import subprocess
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

#: Memoized detections, keyed on the runtimes a call would probe — which folds in
#: both ``CONTAINER_RUNTIME`` and the config's ``container_runtime``. Keying on
#: those inputs rather than memoizing a single answer per process keeps a
#: config-less call from pinning the runtime for a later call that names one.
_runtime_cmd_cache: dict[tuple[str, ...], list[str]] = {}

#: Memoized compose provider probes, keyed on the compose argv base that was
#: probed (``("podman", "compose")`` and friends). Keyed the same way as
#: ``_runtime_cmd_cache`` so a probe of one runtime never answers for another,
#: and cleared by the same :func:`reset_runtime_cache` test hook.
_compose_provider_cache: dict[tuple[str, ...], "ComposeProviderInfo"] = {}


def _runtimes_to_try(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Ordered runtimes to probe, from the ``CONTAINER_RUNTIME`` env var then config.

    An explicit ``docker``/``podman`` from either source (env wins) pins the list
    to that one runtime; anything else — including ``auto``, an absent key, and a
    config-less call — falls back to trying docker then podman.
    """
    selected = config.get("container_runtime", "auto") if config else None

    env_runtime = os.getenv("CONTAINER_RUNTIME")
    if env_runtime:
        selected = env_runtime

    if selected and selected.lower() in ("docker", "podman"):
        return (selected.lower(),)
    return ("docker", "podman")


def get_runtime_command(config: Mapping[str, Any] | None = None) -> list[str]:
    """Get container compose command.

    Checks CONTAINER_RUNTIME env var, then config.container_runtime,
    auto-detects if 'auto' or not set. The detection is memoized per resolved
    runtime preference, so callers passing different configs (or none) each get
    the runtime their own inputs select.

    Args:
        config: Optional configuration mapping

    Returns:
        Command list: ['docker', 'compose'] or ['podman', 'compose']

    Raises:
        RuntimeError: If no container runtime with compose support found
    """
    candidates = _runtimes_to_try(config)

    cached = _runtime_cmd_cache.get(candidates)
    if cached is not None:
        return cached.copy()

    # Try each runtime
    for runtime in candidates:
        if not shutil.which(runtime):
            continue

        try:
            # Check if compose subcommand works
            result = subprocess.run([runtime, "compose", "version"], capture_output=True, timeout=5)

            if result.returncode != 0:
                continue

            # Also verify daemon is actually running
            # This prevents selecting Docker when only Docker CLI is installed but daemon isn't running
            ps_result = subprocess.run([runtime, "ps"], capture_output=True, timeout=5)

            if ps_result.returncode == 0:
                cmd = [runtime, "compose"]
                _runtime_cmd_cache[candidates] = cmd
                return cmd.copy()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # No runtime found - check if any are installed but not running
    docker_installed = shutil.which("docker") is not None
    podman_installed = shutil.which("podman") is not None

    if docker_installed or podman_installed:
        # Runtimes are installed but not running
        error_parts = ["Container runtime installed but not running:\n"]
        if docker_installed:
            error_parts.append("\n" + _get_docker_not_running_message())
        if podman_installed:
            error_parts.append("\n" + _get_podman_not_running_message())
        raise RuntimeError("".join(error_parts))
    else:
        # No runtime installed at all
        raise RuntimeError(
            "No container runtime found. Install Docker Desktop 4.0+ or Podman 4.0+\n"
            "Docker: https://docs.docker.com/get-docker/\n"
            "Podman: https://podman.io/getting-started/installation"
        )


def reset_runtime_cache() -> None:
    """Clear every memoized runtime command so the next call re-detects.

    Primarily used for test isolation: a test that fakes the docker/podman
    probe pins the detected runtime for the rest of the process otherwise.
    """
    _runtime_cmd_cache.clear()
    _compose_provider_cache.clear()


class ComposeProvider(StrEnum):
    """Which compose implementation actually runs behind a compose argv base.

    The runtime binary does not answer this: ``podman compose`` is a dispatcher
    that hands the work to whichever external provider the host has configured,
    so a ``podman`` argv can be served by podman-compose *or* by Docker Compose
    v2. The two providers need different invocation shapes, so callers branch on
    this rather than on ``cmd[0]``.
    """

    DOCKER_V2 = "docker-compose-v2"
    PODMAN_COMPOSE = "podman-compose"


#: Oldest podman-compose OSPREY's compose invocation is compatible with. Chosen
#: because it is what EPEL 8 ships stock (EPEL 9 ships 1.5.0), so facilities on
#: distribution packages are covered without building anything from source.
PODMAN_COMPOSE_MIN_VERSION: tuple[int, ...] = (1, 0, 6)

#: Timeout for the banner probe. Generous compared to the runtime detection
#: probes because ``podman compose version`` starts a Python provider process.
_PROVIDER_PROBE_TIMEOUT = 15

#: ``podman-compose version 1.0.6``, however it is embedded in the surrounding
#: dispatcher chatter. Requires the literal hyphenated name, so the docker
#: pattern below cannot match it.
_PODMAN_COMPOSE_BANNER = re.compile(
    r"podman-compose\s+version:?\s+v?(\d+(?:\.\d+)*)", re.IGNORECASE
)

#: ``Docker Compose version v2.24.5``. The space between the two words is
#: load-bearing: Compose v1 announced itself as ``docker-compose version 1.29.2``
#: and is not a supported provider, so it must fall through to the fail-closed
#: branch rather than being read as a v2.
_DOCKER_COMPOSE_BANNER = re.compile(
    r"docker\s+compose\s+version:?\s+v?(\d+(?:\.\d+)*)", re.IGNORECASE
)


@dataclass(frozen=True)
class ComposeProviderInfo:
    """The provider behind a compose argv base, with the version it reported.

    Attributes:
        provider: Which supported implementation answered the probe.
        version: Parsed version as an int tuple, directly comparable
            (``info.version >= PODMAN_COMPOSE_MIN_VERSION``).
        version_text: The version exactly as the provider printed it.
        banner: The trimmed stdout+stderr of the probe, kept for diagnostics.
    """

    provider: ComposeProvider
    version: tuple[int, ...]
    version_text: str
    banner: str


class UnsupportedComposeProviderError(RuntimeError):
    """Raised when the compose provider is not one OSPREY can invoke correctly.

    A ``RuntimeError`` subclass so the existing "no usable runtime" handling in
    this module and its callers keeps catching it.
    """


def _format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _unsupported_provider_message(base: Sequence[str], detail: str, banner: str) -> str:
    """Fail-closed text: what was wrong, what is supported, what was reported."""
    lines = [
        f"Unsupported compose provider behind `{' '.join(base)}`: {detail}",
        "",
        "OSPREY supports Docker Compose v2 and podman-compose "
        f"{_format_version(PODMAN_COMPOSE_MIN_VERSION)} or newer. See the container "
        "runtime compatibility statement in the deployment guide "
        "(docs/source/how-to/deploy-project/index.rst) for the supported providers and the "
        "invocation shape each one gets.",
    ]
    if banner:
        lines.extend(["", "The provider reported:", textwrap.indent(banner, "  ")])
    return "\n".join(lines)


def detect_compose_provider(
    cmd: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> ComposeProviderInfo:
    """Probe which compose implementation serves a compose argv base.

    Runs ``<base> version`` and reads the banner from stdout *and* stderr —
    ``podman compose`` prints its "executing external compose provider" notice
    on stderr and the provider's own banner on stdout, and which stream carries
    what varies by version, so both are scanned as one text.

    Recognition is by banner, never by ``cmd[0]``: a host whose podman is
    configured to delegate to Docker Compose v2 reports the Docker banner and is
    reported as :attr:`ComposeProvider.DOCKER_V2`, because the invocation shape
    follows the provider that parses the argv, not the binary that forwards it.

    Anything else fails closed rather than guessing: an unrecognized banner, a
    podman-compose older than :data:`PODMAN_COMPOSE_MIN_VERSION`, a Compose v1,
    or a probe that could not be run at all. Guessing would mean emitting an
    argv the provider silently mis-parses, which surfaces much later as a
    deploy that came up against the wrong files.

    Memoized per argv base and cleared by :func:`reset_runtime_cache`. Only
    successful probes are cached, so a fail-closed run is re-probed after the
    host is fixed rather than staying poisoned for the process.

    Args:
        cmd: Compose argv base to probe, e.g. ``["podman", "compose"]``.
            Defaults to :func:`get_runtime_command`'s answer for ``config``.
        config: Optional configuration mapping, used only when ``cmd`` is
            omitted.

    Returns:
        The detected provider and its version.

    Raises:
        UnsupportedComposeProviderError: The provider is not supported, its
            version is below the floor, or the probe could not be run.
    """
    base = list(cmd) if cmd is not None else get_runtime_command(config)
    key = tuple(base)

    cached = _compose_provider_cache.get(key)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [*base, "version"],
            capture_output=True,
            text=True,
            timeout=_PROVIDER_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedComposeProviderError(
            _unsupported_provider_message(
                base, f"the version probe timed out after {_PROVIDER_PROBE_TIMEOUT}s", ""
            )
        ) from exc
    except OSError as exc:
        raise UnsupportedComposeProviderError(
            _unsupported_provider_message(base, f"the version probe could not run ({exc})", "")
        ) from exc

    banner = "\n".join(
        stream.strip() for stream in (result.stdout or "", result.stderr or "") if stream.strip()
    )

    podman_match = _PODMAN_COMPOSE_BANNER.search(banner)
    if podman_match:
        version = _parse_version(podman_match.group(1))
        if version < PODMAN_COMPOSE_MIN_VERSION:
            raise UnsupportedComposeProviderError(
                _unsupported_provider_message(
                    base,
                    f"podman-compose {podman_match.group(1)} is older than the supported "
                    f"floor {_format_version(PODMAN_COMPOSE_MIN_VERSION)}",
                    banner,
                )
            )
        info = ComposeProviderInfo(
            provider=ComposeProvider.PODMAN_COMPOSE,
            version=version,
            version_text=podman_match.group(1),
            banner=banner,
        )
        _compose_provider_cache[key] = info
        return info

    docker_match = _DOCKER_COMPOSE_BANNER.search(banner)
    if docker_match:
        version = _parse_version(docker_match.group(1))
        if version < (2,):
            raise UnsupportedComposeProviderError(
                _unsupported_provider_message(
                    base,
                    f"Docker Compose {docker_match.group(1)} predates Compose v2",
                    banner,
                )
            )
        info = ComposeProviderInfo(
            provider=ComposeProvider.DOCKER_V2,
            version=version,
            version_text=docker_match.group(1),
            banner=banner,
        )
        _compose_provider_cache[key] = info
        return info

    if "podman-compose" in banner.lower():
        # The dispatcher named podman-compose but no version came back, so the
        # floor cannot be checked — the one thing that must not be assumed.
        raise UnsupportedComposeProviderError(
            _unsupported_provider_message(
                base, "podman-compose reported no parseable version", banner
            )
        )

    detail = "the version banner was not recognized"
    if result.returncode != 0:
        detail += f" (probe exited {result.returncode})"
    raise UnsupportedComposeProviderError(_unsupported_provider_message(base, detail, banner))


#: Environment variable carrying :func:`env_chain_digest`'s answer into a
#: compose invocation. The rendered templates interpolate it into an
#: ``osprey.env.digest`` container label, so the name is part of the compose
#: contract and is spelled here once for every producer and consumer of it.
ENV_DIGEST_VAR = "OSPREY_ENV_DIGEST"

#: The same contract as :data:`ENV_DIGEST_VAR`, for the other file a container
#: reads its settings out of: the rendered config the build decided on. Carried
#: into an ``osprey.config.digest`` label by every service template.
CONFIG_DIGEST_VAR = "OSPREY_CONFIG_DIGEST"


def env_chain_digest(repo_root: Path | str) -> str:
    """Fingerprint of the env chain's *contents* under ``repo_root``.

    THE recipe, spelled once so every reproducer agrees on it:

    * take :func:`osprey.utils.dotenv.chain_files` for ``repo_root`` — the
      existing members of ``.env.shared``, ``.env``, in that order;
    * feed each file's raw bytes, unparsed and in that order, into one
      ``sha256``; the files are concatenated into a single hash, not hashed
      individually;
    * return the hex digest — or the **empty string** when the chain has no
      existing member at all, which is a valid project shape rather than an
      error.

    Nothing else is mixed in: not the file names, not their modification times,
    not ``repo_root``. Two deployments whose chain files hold identical bytes
    have identical digests, and a comment-only edit changes the digest just as a
    value edit does. Parsing instead of hashing raw bytes would be a second,
    subtly different answer to "did the env change?", so the bytes are hashed as
    they sit on disk.

    Membership is not encoded, deliberately: moving a key from ``.env.shared``
    into ``.env`` byte-for-byte leaves the digest unchanged. Chain-set drift is
    a build-time question, answered by the ``osprey up`` preflight, not by this
    content fingerprint.

    :param repo_root: Directory the chain lives in (the deployment repo root).
    :return: Hex sha256 of the concatenated chain, or ``""`` for an empty chain.
    """
    from osprey.utils.dotenv import chain_files

    paths = chain_files(Path(repo_root))
    if not paths:
        return ""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def as_built_config_digest(repo_root: Path | str) -> str:
    """Fingerprint of the *contents* of the config this repo's build decided on.

    :func:`env_chain_digest`'s twin, for the other file whose contents reach a
    container without passing through the compose document. Every service
    directory receives the rendered project configuration as its own
    ``config.yml`` and mounts it, so ``osprey set`` changes a FILE — and compose
    decides whether to recreate a container by comparing service definitions,
    which such a change leaves identical. Without a label carrying this, the
    container keeps serving the configuration it parsed at startup, and ``set``
    -> ``build`` -> ``up`` reports success having changed nothing an operator
    can observe.

    Hashes ``<repo_root>/build/config.yml`` — the single as-built config
    (:func:`osprey.deployment.container_lifecycle.as_built_config_path`), which
    is what ``up`` starts from and what every per-service copy is rendered from.
    One file rather than the copies: they are one document, and hashing the
    source keeps this answer independent of how many services a deployment has.

    Raw bytes, like the env chain, and for the same reason — parsing would be a
    second, subtly different answer to "did the config change?". A **comment**
    edit therefore moves the digest and recreates containers. That is the honest
    trade: the alternative is a semantic diff that has to stay in step with
    every reader of the file, and a spurious recreate costs a restart while a
    missed one leaves a container lying about its own configuration.

    :param repo_root: The deployment repo root (the compose project directory).
    :return: Hex sha256 of the as-built config, or ``""`` when the repo has no
        rendered build yet — a valid state for every verb that runs before one.
    """
    from osprey.deployment.container_lifecycle import as_built_config_path

    path = as_built_config_path(repo_root)
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_env(
    config: dict | None,
    base_env: dict[str, str] | None = None,
    *,
    ignore_orphans: bool = False,
) -> dict[str, str]:
    """Build the environment for a runtime (docker/podman compose) invocation.

    Pins ``COMPOSE_PROJECT_NAME`` to :func:`compose_generator.resolve_project_name`
    so the volume namespace compose derives (``<COMPOSE_PROJECT_NAME>_<name>``)
    always matches the project name baked into container labels. Left unset,
    compose falls back to ``basename(cwd)`` for the volume namespace, which can
    silently diverge from the label project name (e.g. an explicit
    ``project_name`` in config, or a deploy invoked from a different cwd than
    ``project_root``). Every subprocess/exec call that shells out to a runtime
    command should build its env through this function rather than passing
    ``os.environ`` (or a copy of it) directly.

    Also injects :data:`ENV_DIGEST_VAR` (:func:`env_chain_digest` of the repo
    the invocation is pinned to), and injects it unconditionally rather than
    on request. The rendered templates carry an ``osprey.env.digest`` label
    interpolated from it, and a label is compose's own recreate trigger: podman-
    compose does not diff a container's environment against the env files it was
    built from, so an edit to ``.env`` reaches a *running* container only if
    something in the compose document itself changed. Folding the chain's
    content hash into a label makes that edit a document change, so the affected
    containers are recreated on both providers.

    Unconditional is the point. The value only works as a recreate trigger if
    every invocation that can reach ``up`` computes the SAME one — a single call
    site that skipped it would interpolate the label to the empty string, look
    like a document change to compose, and recreate the whole stack for no
    reason. That is also why the digest is not a caller's argument: it is
    derived here, from the same repo root compose is pinned to, for all of them.

    ``ignore_orphans=True`` additionally sets ``COMPOSE_IGNORE_ORPHANS``, and
    is for the web-terminal call sites only. There it is structural, not
    cosmetic: a web-terminal deploy runs TWO compose invocations against the
    single project this function pins — the backend services under
    ``build/services/*.yml`` and the web stack under
    ``docker-compose.web.yml`` — so each one necessarily sees the other's
    containers as orphans of the shared project, and says so at length, twice
    per deploy. The warning's own suggested remedy (``--remove-orphans``) is
    the one thing that must never be run there: it would delete the other
    stack, which is exactly why both call sites in
    :mod:`~osprey.deployment.web_terminals.provision` forbid the flag. A
    warning whose only advertised fix is destructive, in a situation the
    design guarantees, is pure noise — so it is turned off at the source
    rather than left for every operator to learn to ignore.

    It must stay opt-in because the single-stack paths are the mirror image:
    plain ``osprey up`` reconciles with ``up --remove-orphans`` and ``clean``
    tears down with ``down --remove-orphans``, and docker compose hard-errors
    on the combination ("cannot combine COMPOSE_IGNORE_ORPHANS and
    --remove-orphans") rather than letting one win. A default-on env var
    would turn every one of those deploys into that error.

    Args:
        config: Configuration dictionary used to resolve the project name.
            Falsy values (``None``, ``{}``) resolve the same as an empty dict,
            i.e. the ``"unnamed-project"`` fallback.
        base_env: Environment to layer the pin onto. Defaults to
            ``os.environ``. Never mutated — a fresh copy is always returned.
        ignore_orphans: Set ``COMPOSE_IGNORE_ORPHANS=1`` for a compose
            invocation that shares its project with another one by design.
            Never combine with a ``--remove-orphans`` argv.

    Returns:
        A new environment dict with ``COMPOSE_PROJECT_NAME`` and
        :data:`ENV_DIGEST_VAR` set (and ``COMPOSE_IGNORE_ORPHANS`` when
        requested).
    """
    from osprey.deployment.compose_generator import resolve_project_name, resolve_repo_root

    env = dict(base_env if base_env is not None else os.environ)
    env["COMPOSE_PROJECT_NAME"] = resolve_project_name(config or {})
    # Derived from the repo compose_base_cmd pins as --project-directory, so the
    # digest covers the same chain files compose itself resolves ./.env* against.
    repo_root = resolve_repo_root(config)
    env[ENV_DIGEST_VAR] = env_chain_digest(repo_root)
    # Unconditional for the same reason the env digest is: the value only works
    # as a recreate trigger if every invocation that can reach `up` computes the
    # same one, and a call site that skipped it would interpolate the label to
    # empty, read as a document change, and recreate the whole stack for nothing.
    env[CONFIG_DIGEST_VAR] = as_built_config_digest(repo_root)
    if ignore_orphans:
        env["COMPOSE_IGNORE_ORPHANS"] = "1"
    return env


def with_plain_progress(cmd: list[str]) -> list[str]:
    """Add ``--progress plain`` to a docker compose argv so it cannot garble.

    Takes an already-resolved compose argv rather than resolving one itself,
    deliberately: every call site reaches its runtime through the
    ``get_runtime_command`` name bound in *its own* module, which is the seam
    the deploy tests monkeypatch. A helper that called ``get_runtime_command``
    internally would resolve this module's binding instead, walk straight past
    that patch, and run live runtime detection inside unit tests.

    Compose's default (``auto``) selects a TTY renderer on an interactive
    terminal, which repaints each frame by moving the cursor up N lines and
    rewriting them — without erasing to end of line. Long project and service names
    (``my-control-assistant-bluesky-queueserver`` and friends) wrap on a
    normal-width terminal, the renderer's N is then wrong by the number of
    wrapped lines, and successive frames overprint each other into
    unreadable fragments. ``plain`` is append-only, so it has no frame
    arithmetic to get wrong.

    Docker only, deliberately. ``--progress`` is a docker compose v2 flag;
    ``podman compose`` delegates to whichever provider the host has, and a
    provider that rejects an unknown flag would turn a cosmetic improvement
    into a failed deploy. Podman hosts keep compose's own default, and lose
    little: ``auto`` already resolves to ``plain`` whenever stdout is not a
    terminal, which covers CI and systemd — the places a production deploy's
    output is actually captured.

    ``--progress`` is a global flag, valid ahead of every compose subcommand,
    so callers append their ``-f``/``--env-file`` arguments and the subcommand
    afterwards unchanged.

    Args:
        cmd: A compose argv base, e.g. ``["docker", "compose"]``, as returned
            by :func:`get_runtime_command`. Extended in place and returned, so
            ``with_plain_progress(get_runtime_command(config))`` reads as one
            expression at the call site.

    Returns:
        The same list, with the flag appended when the runtime is docker.
    """
    if cmd and cmd[0] == "docker":
        cmd.extend(("--progress", "plain"))
    return cmd


def verify_runtime_is_running(config: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    """Verify that the detected container runtime is actually running.

    Args:
        config: Optional configuration mapping

    Returns:
        Tuple of (is_running: bool, error_message: str)
        If running: (True, "")
        If not running: (False, helpful error message)
    """
    try:
        cmd = get_runtime_command(config)
        runtime = cmd[0]  # 'docker' or 'podman'

        # Try a simple ps command to verify daemon is accessible
        result = subprocess.run([runtime, "ps"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            return True, ""

        # Check for common error patterns
        stderr = result.stderr.lower()

        if "cannot connect to the docker daemon" in stderr or "docker daemon" in stderr:
            return False, _get_docker_not_running_message()

        if "cannot connect to podman" in stderr or "connection refused" in stderr:
            return False, _get_podman_not_running_message()

        # Generic error
        return False, f"{runtime.capitalize()} is installed but not responding:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        runtime = "container runtime"
        try:
            cmd = get_runtime_command(config)
            runtime = cmd[0]
        except Exception:
            pass  # Best-effort runtime name detection for error message
        return False, f"{runtime.capitalize()} command timed out. The service may not be running."
    except RuntimeError as e:
        # No runtime found at all
        return False, str(e)
    except Exception as e:
        return False, f"Error checking container runtime: {e}"


def _get_docker_not_running_message() -> str:
    """Get platform-specific message for Docker not running."""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        return (
            "Docker Desktop is not running.\n\n"
            "To fix this:\n"
            "1. Open Docker Desktop from Applications\n"
            "2. Wait for Docker to start (whale icon in menu bar should be steady)\n"
            "3. Try your command again\n\n"
            "If Docker Desktop is not installed:\n"
            "https://docs.docker.com/desktop/install/mac-install/"
        )
    elif system == "Windows":
        return (
            "Docker Desktop is not running.\n\n"
            "To fix this:\n"
            "1. Start Docker Desktop from the Start menu\n"
            "2. Wait for Docker to start (system tray icon should be running)\n"
            "3. Try your command again\n\n"
            "If Docker Desktop is not installed:\n"
            "https://docs.docker.com/desktop/install/windows-install/"
        )
    else:  # Linux
        return (
            "Docker daemon is not running.\n\n"
            "To fix this:\n"
            "1. Start Docker: sudo systemctl start docker\n"
            "2. Enable on boot: sudo systemctl enable docker\n"
            "3. Check status: sudo systemctl status docker\n\n"
            "If permission issues, add user to docker group:\n"
            "sudo usermod -aG docker $USER\n"
            "(then log out and back in)"
        )


def _get_podman_not_running_message() -> str:
    """Get platform-specific message for Podman not running."""
    import platform

    system = platform.system()

    if system in ["Darwin", "Windows"]:  # macOS or Windows
        return (
            "Podman machine is not running.\n\n"
            "To fix this:\n"
            "1. Start Podman: podman machine start\n"
            "2. Check status: podman machine list\n\n"
            "If no machine exists:\n"
            "1. Initialize: podman machine init\n"
            "2. Start: podman machine start"
        )
    else:  # Linux
        return (
            "Podman service is not responding.\n\n"
            "To fix this:\n"
            "1. Check status: systemctl --user status podman.socket\n"
            "2. Start if needed: systemctl --user start podman.socket\n\n"
            "For rootful Podman:\n"
            "sudo systemctl start podman.socket"
        )


def get_ps_command(
    config: Mapping[str, Any] | None = None, all_containers: bool = False
) -> list[str]:
    """Get container ps command.

    Args:
        config: Optional configuration mapping
        all_containers: Include stopped containers (-a flag)

    Returns:
        Command list for ps operation with JSON format
    """
    cmd = get_runtime_command(config)
    runtime = cmd[0]  # 'docker' or 'podman'

    ps_cmd = [runtime, "ps"]
    if all_containers:
        ps_cmd.append("-a")
    ps_cmd.extend(["--format", "json"])

    return ps_cmd


def _inspect_image_id(cmd: list[str], env: dict[str, str] | None) -> str | None:
    """Run an inspect ``cmd`` that prints one image ID; normalize or return None.

    A non-zero exit (missing image/container) or empty output yields ``None`` so
    callers can treat an absent target as "nothing to reconcile" rather than
    raising. The ``sha256:`` prefix is stripped so IDs from an image inspect and
    a container inspect compare equal regardless of which form the runtime emits.
    """
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return None
    image_id = result.stdout.strip()
    if not image_id:
        return None
    return image_id.removeprefix("sha256:")


def get_image_id(runtime: str, image: str, env: dict[str, str] | None = None) -> str | None:
    """Image ID a local image reference currently resolves to, or ``None``.

    Runs ``<runtime> image inspect --format {{.Id}} <image>``. A reference not
    present locally (e.g. a tag that was never pulled) returns ``None`` instead
    of raising, so a caller reconciling image drift can skip it.
    """
    return _inspect_image_id([runtime, "image", "inspect", "--format", "{{.Id}}", image], env)


def get_container_image_id(
    runtime: str, container: str, env: dict[str, str] | None = None
) -> str | None:
    """Image ID a container was created from, or ``None`` if it doesn't exist.

    Runs ``<runtime> container inspect --format {{.Image}} <container>``. A
    missing container returns ``None`` (a no-op for the caller), so reconciling
    a service whose container hasn't been created yet is harmless.
    """
    return _inspect_image_id(
        [runtime, "container", "inspect", "--format", "{{.Image}}", container], env
    )


# ---------------------------------------------------------------------------
# podman + Docker Compose v2: the empty-credential registry refusal
#
# `podman compose` is a dispatcher, and one of the providers it will dispatch
# to is the Docker Compose v2 CLI plugin. detect_compose_provider reports that
# host as DOCKER_V2 and OSPREY invokes it correctly -- the argv shape is not the
# problem. Image BUILDS are: a `FROM` that has to fetch its base image reaches
# the registry with an empty credential instead of with none at all, and the
# registry answers 401. The same host pulls the same image fine through `podman
# pull` and through `podman compose pull`; only the build path carries the
# empty credential, which is why this cannot be fixed with `podman login`.
#
# An anonymous pull would have worked. Supplying ""/"" is strictly worse than
# supplying nothing, and it is what converts a working public pull into a
# refusal whose wording ("incorrect username or password") points at a password
# nobody typed.


#: Substrings that together identify the empty-credential refusal. All must be
#: present: the 401 wording alone is also what a genuinely wrong `podman login`
#: produces, and mistaking that for this bug would send an operator to change a
#: provider when they should fix their password. Requiring the pull-time frame
#: ("fetching manifest") pins it to the build path this actually affects.
_REGISTRY_AUTH_DENIED_MARKERS: tuple[str, ...] = (
    "fetching manifest",
    "unable to retrieve auth token",
    "incorrect username or password",
)


#: What to do about it. Both entries are real escapes, in the order a facility
#: would prefer them: podman-compose is the provider OSPREY's own CI pins for
#: the podman lane, and docker is the other supported runtime outright.
_REGISTRY_AUTH_REMEDY = (
    "podman handed the registry an EMPTY credential for this pull, and the registry "
    "rejected it -- the wording names a password, but none was ever sent. This is the "
    "podman-with-Docker-Compose-v2 provider pairing, not a wrong login: `podman login` "
    "does not fix it, because the credential does not come from podman's auth file.\n"
    "Either:\n"
    "  - install podman-compose and pin it as podman's provider, in containers.conf:\n"
    "\n"
    "        [engine]\n"
    '        compose_providers = ["/absolute/path/to/podman-compose"]\n'
    "\n"
    "  - or run this deployment on docker (start the Docker daemon, or set\n"
    "    container_runtime: docker), which uses its own compose provider.\n"
    "\n"
    "Pre-pulling the base images (`podman pull <image>`) also unblocks a single run: "
    "a build that needs no fetch never presents the credential."
)


def diagnose_build_failure(output: str) -> str | None:
    """Explain a failed build's output, or ``None`` when there is nothing to add.

    Reads a captured build log (or any fragment of one) and recognizes the
    empty-credential registry refusal described above, which otherwise reaches
    the operator as a 401 about a password they never entered.

    Substring matching rather than a parse: the text arrives wrapped in the
    dispatcher's own chatter, in whichever line-ending and ANSI decoration the
    provider emitted, and every one of those variants is the same failure.

    Deliberately conservative. It answers only the signature it can name, so a
    build that failed for any other reason -- a missing wheel, a compile error,
    an unreachable private registry -- passes through untouched rather than
    collecting a confident and wrong remedy.

    :param output: Captured build output, whole or partial.
    :returns: Operator-facing remedy text, or ``None`` for no opinion.
    """
    if not output:
        return None
    if all(marker in output for marker in _REGISTRY_AUTH_DENIED_MARKERS):
        return _REGISTRY_AUTH_REMEDY
    return None


#: The one action that resolves the pairing, rendered in the CLI's remedy slot
#: rather than inside the advisory so it is not said twice.
PODMAN_COMPOSE_PROVIDER_REMEDY = (
    "install podman-compose and pin it in containers.conf "
    '([engine] compose_providers = ["/absolute/path/to/podman-compose"]), '
    "or deploy on docker"
)


def podman_compose_provider_advisory(runtime: str, provider: ComposeProvider) -> str | None:
    """Warn ahead of a build that this host's compose provider breaks base-image pulls.

    Fires only for podman served by Docker Compose v2 -- the pairing
    :func:`detect_compose_provider` reports as
    :attr:`ComposeProvider.DOCKER_V2` behind a ``podman`` argv. podman served by
    podman-compose is the configuration OSPREY's CI podman lane pins, and docker
    served by Docker Compose v2 is the ordinary docker case; both stay silent.

    A warning and not a refusal, on purpose. The pairing is one OSPREY declares
    supported and invokes correctly, and the break lands only on builds that
    must FETCH a base image -- a host whose images are already local, or which
    pulls from a registry that accepts the empty credential, completes normally.
    Refusing those would be causing an outage to describe one. What the operator
    needs before a long build is the name of the thing that is about to go
    wrong; :func:`diagnose_build_failure` says it again, in place, if it does.

    Takes the resolved runtime and provider rather than probing for them. The
    deploy has already paid for both by the time a preflight runs, and probing
    again from here would reach past the module-bound ``get_runtime_command``
    that the lifecycle tests patch -- shelling out for an answer the caller is
    holding. Being a pure function of the pairing also means it cannot fail, so
    unlike its sibling preflights it needs no totality contract of its own: the
    caller owns "the host could not be interrogated", which is where the two
    exceptions involved were already being reported.

    :param runtime: The resolved container runtime (``docker`` or ``podman``).
    :param provider: The compose provider detected behind it.
    :returns: Operator-facing advisory text, or ``None`` when the pairing is
        anything else.
    """
    if runtime != "podman" or provider is not ComposeProvider.DOCKER_V2:
        return None

    return (
        "`podman compose` on this host is served by Docker Compose v2, not by "
        "podman-compose. Image builds that must fetch a base image reach the registry "
        "with an empty credential on that pairing and are refused with a 401 naming a "
        "password that was never sent. Images already present locally are unaffected, "
        "so a build with nothing to fetch still succeeds."
    )
