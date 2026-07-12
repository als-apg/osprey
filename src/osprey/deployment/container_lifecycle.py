"""Container lifecycle operations — start, stop, restart, and rebuild.

Manages the lifecycle of containerized service deployments using
Docker or Podman compose.
"""

import os
import secrets
import subprocess
from collections.abc import Callable
from pathlib import Path

from osprey.deployment.compose_generator import (
    clean_deployment,
    prepare_compose_files,
)
from osprey.deployment.runtime_helper import (
    get_runtime_command,
    verify_runtime_is_running,
)
from osprey.utils.config import ConfigBuilder
from osprey.utils.dotenv import parse_dotenv_file
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")

# Services that fail closed on an unset bearer token (no insecure default in
# their compose templates), so a deploy must supply real secrets. Maps each
# deployed-service name to the token env var(s) it requires. event_dispatcher
# and dispatch_worker both list the same pair because they share one .env:
# the dispatcher forwards routed requests to workers using DISPATCH_WORKER_TOKEN,
# so either service alone still needs both vars minted. bluesky needs its own
# promote token (see ``security.require_armed`` in
# ``osprey.services.bluesky_bridge``) plus the API key the bridge presents to
# the co-deployed Tiled catalog. The Tiled key hangs off "bluesky" rather than a
# "tiled" key of its own because "tiled" is never a member of
# ``deployed_services``, so the membership guard in ``_ensure_service_tokens``
# would skip it and the key would never mint.
_SERVICE_TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "event_dispatcher": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "dispatch_worker": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "bluesky": ("BLUESKY_PROMOTE_TOKEN", "BLUESKY_TILED_API_KEY"),
}

# Token vars that are safe to auto-mint even when the agent's code execution is
# unsandboxed on the host (see _local_exec_arming_unsafe below). Every *other*
# declared token var is withheld under that config — a var nobody has triaged
# fails CLOSED by omission.
#
# This is an allowlist, not a blocklist, because a security gate must fail
# closed on the paths its author did not enumerate. Under a blocklist, a service
# later added to _SERVICE_TOKEN_VARS with an arming token would mint it under
# writes_enabled + local exec, warn about nothing, and break no test.
#
# The gate is per *variable*, not per service: one service can declare both an
# arming token and non-arming credentials. The bar for adding a var here is that
# it grants no write-capable route the agent itself can walk. Under local exec,
# agent-authored code can read any minted token straight out of .env/config.yml
# and call the route it gates directly, bypassing the in-tool writes_enabled
# re-check. BLUESKY_PROMOTE_TOKEN is therefore absent: it gates the bridge's
# POST /runs/{id}/promote.
_LOCAL_EXEC_SAFE_VARS = {
    "BLUESKY_TILED_API_KEY",  # outbound credential to the Tiled catalog; gates no bridge route
    "EVENT_DISPATCHER_TOKEN",  # inbound webhook boundary, not a write path the agent walks
    "DISPATCH_WORKER_TOKEN",  # worker-routing boundary, same
}


def _default_token() -> str:
    """The default secret recipe, also the one ``.env.template`` documents."""
    return secrets.token_urlsafe(32)


# Per-variable overrides of the default recipe. A var absent here gets
# ``_default_token``.
#
# BLUESKY_TILED_API_KEY: Tiled validates its ``--api-key`` during server startup
# and raises ``ValueError("The API key must only contain alphanumeric
# characters")`` for anything else, so a rejected key makes the container exit
# before it ever listens. ``token_urlsafe``'s alphabet includes ``-`` and ``_``,
# which land in roughly 7 of 10 values, so that recipe crash-loops Tiled on most
# deploys — non-deterministically. ``token_hex(32)`` draws from ``[0-9a-f]``:
# alphanumeric by construction, and the same 256 bits of entropy.
#
# Generate from an alphanumeric alphabet rather than stripping ``-``/``_`` out of
# a urlsafe value, which would shorten the secret by a variable amount and drop
# entropy silently.
_VAR_GENERATORS: dict[str, Callable[[], str]] = {
    "BLUESKY_TILED_API_KEY": lambda: secrets.token_hex(32),
}


def _generate_token(var: str) -> str:
    """Mint one secret for ``var`` using its registered recipe."""
    return _VAR_GENERATORS.get(var, _default_token)()


def _local_exec_arming_unsafe(config: dict) -> bool:
    """True when writes are enabled AND python-executor code runs unsandboxed on the host.

    ``control_system.writes_enabled`` (default False) and
    ``execution.execution_method`` (default "container") are read directly from
    the raw config dict, mirroring how every other consumer of these two flags
    resolves them (e.g. ``osprey.connectors.control_system.base``,
    ``osprey.cli.templates.claude_code``, the ``osprey_writes_check`` hook, and
    ``osprey.mcp_server.python_executor.executor._read_config``).

    The container execution_method runs agent code fs/network-isolated from the
    project's ``.env`` and has no equivalent exposure.
    """
    writes_enabled = bool(config.get("control_system", {}).get("writes_enabled", False))
    execution_method = config.get("execution", {}).get("execution_method", "container")
    return writes_enabled and execution_method == "local"


def _ensure_service_tokens(
    config: dict, expose_network: bool, env_path: Path | None = None
) -> None:
    """Self-provision required fail-closed service tokens into the project ``.env``.

    For any token var (per ``_SERVICE_TOKEN_VARS``, keyed by the deployed
    services present) unset in BOTH the process env and the project ``.env``,
    generate a strong random value (``_generate_token``: ``token_urlsafe(32)``
    unless the var registers a different alphabet in ``_VAR_GENERATORS``) and
    append it to ``.env``
    (``chmod 0o600``, matching the build-time convention). Existing values are
    never overwritten, so re-running ``deploy up`` is idempotent. No-op unless
    a token-requiring service is actually deployed.

    A token that is *present but explicitly empty* (e.g. ``TOKEN=`` exported in
    the shell) is left untouched — generating would silently override a
    deliberate value. For a loopback deploy the server simply fails closed; for
    an exposed deploy (``--expose`` / bind 0.0.0.0) we refuse rather than bind a
    fail-open-at-bind server to all interfaces.

    When ``_local_exec_arming_unsafe(config)`` — see that function's docstring —
    the mint is restricted to the ``_LOCAL_EXEC_SAFE_VARS`` allowlist: any other
    declared var is skipped (never minted; an existing value in .env/env is left
    untouched but not read either). For the withheld ``BLUESKY_PROMOTE_TOKEN``
    the bridge's own ``require_armed()`` then keeps returning 503, i.e. this
    deploy never arms it, rather than arming it with a token any local
    agent-code execution can trivially read back out of ``.env``.

    The restriction is per var, not per service: the allowlisted vars a service
    declares (e.g. bluesky's ``BLUESKY_TILED_API_KEY``, which grants catalog
    access, not the ability to move the machine) still mint under the same
    unsafe config. And because it is an allowlist, a token var added later
    without being triaged fails closed rather than arming silently.
    """
    deployed_services = config.get("deployed_services")
    services = {str(s) for s in (deployed_services or [])}
    unsafe_local_exec = _local_exec_arming_unsafe(config)

    # Iterate the map (not the deployed `services` set) so var order is
    # deterministic regardless of set iteration order or config.yml ordering.
    required_vars: list[str] = []
    seen: set[str] = set()
    warned: set[str] = set()
    for svc_name, token_vars in _SERVICE_TOKEN_VARS.items():
        if svc_name not in services:
            continue
        for var in token_vars:
            if unsafe_local_exec and var not in _LOCAL_EXEC_SAFE_VARS:
                if var not in warned:
                    warned.add(var)
                    logger.warning(
                        "Refusing to arm %r: control_system.writes_enabled is true but "
                        "execution.execution_method is 'local' (agent code runs unsandboxed "
                        "on the host, cwd=project_root). Local execution can read %s "
                        "straight out of .env/config.yml and call the write-capable route "
                        "it gates directly, bypassing the in-tool writes_enabled re-check. "
                        "NOT minting %s — this deploy will not arm it (an existing value "
                        "in .env or the environment is left in place and still arms the "
                        "service). Set execution.execution_method: container to use this "
                        "feature with writes_enabled: true.",
                        svc_name,
                        var,
                        var,
                    )
                continue
            if var not in seen:
                seen.add(var)
                required_vars.append(var)

    if not required_vars:
        return

    if env_path is None:
        env_path = Path(".env")

    existing = parse_dotenv_file(env_path) if env_path.is_file() else {}

    generated: dict[str, str] = {}
    for name in required_vars:
        # Process env wins over .env (matches docker compose --env-file).
        present = name in os.environ or name in existing
        if present:
            continue  # keep the user's value (even an empty one — see docstring)
        generated[name] = _generate_token(name)

    if generated:
        prefix = ""
        if env_path.is_file():
            text = env_path.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                prefix = "\n"
        block = "".join(f"{k}={v}\n" for k, v in generated.items())
        with env_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{prefix}# Auto-generated service auth tokens (osprey deploy up)\n{block}")
        os.chmod(env_path, 0o600)
        # Log the path and which keys — NEVER the values.
        logger.key_info(
            "Generated auth token(s) %s in %s (gitignored) — keep them secret",
            ", ".join(generated),
            env_path.resolve(),
        )

    if expose_network:
        post = parse_dotenv_file(env_path) if env_path.is_file() else {}
        for name in required_vars:
            effective = os.environ.get(name, post.get(name, ""))
            if not effective.strip():
                raise RuntimeError(
                    f"{name} is empty; refusing to --expose (bind 0.0.0.0) with an "
                    f"empty token. Set {name} in .env to a strong secret."
                )


def deploy_up(config_path, detached=False, dev_mode=False, expose_network=False):
    """Start services using container runtime (Docker or Podman).

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param dev_mode: Development mode for local framework testing
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    """
    config, compose_files = prepare_compose_files(config_path, dev_mode, expose_network)

    if not config.get("deployed_services"):
        logger.key_info(
            "No services configured for this project — deployed_services is empty in "
            "config.yml. Skipping osprey deploy up."
        )
        return

    # Verify container runtime is actually running
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Self-provision fail-closed service tokens into .env (before the --env-file
    # check below) so a fresh deploy is secure by default. The dispatch worker
    # mounts the same .env for provider auth; a deploy where .env already
    # carries provider keys renders with osprey_env_present=True and picks up
    # the appended tokens, and a tokens-only .env carries no provider secret to
    # mount in the first place.
    _ensure_service_tokens(config, expose_network)

    # Set up environment for containers
    env = os.environ.copy()
    if dev_mode:
        env["DEV_MODE"] = "true"
        logger.key_info("Development mode: DEV_MODE environment variable set for containers")

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))

    # Only add --env-file if .env exists, otherwise let docker-compose use defaults
    env_file = Path(".env")
    if env_file.exists():
        cmd.extend(["--env-file", ".env"])
    else:
        logger.warning(
            "No .env file found - services will start with default/empty environment variables"
        )
        logger.info("To configure API keys: cp .env.example .env && edit .env")

    cmd.append("up")
    if dev_mode:
        # `osprey deploy up --dev` re-bakes the local osprey checkout into a fresh
        # wheel on every run, but compose reuses the cached image tag (e.g.
        # osprey-dispatch:local) unless a rebuild is forced — so without --build
        # the container keeps running the stale code from the first build.
        cmd.append("--build")
    if detached:
        cmd.append("-d")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    if detached:
        subprocess.run(cmd, env=env, check=True)
    else:
        os.execvpe(cmd[0], cmd, env)


def deploy_down(config_path, dev_mode=False):
    """Stop services using container runtime (Docker or Podman).

    :param config_path: Path to the configuration file
    :type config_path: str
    """
    try:
        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(config_path)
            config = config.raw_config
    except Exception as e:
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e

    deployed_services = config.get("deployed_services", [])
    deployed_service_names = (
        [str(service) for service in deployed_services] if deployed_services else []
    )

    # Try to use existing compose files (suppress warnings for status check)
    from osprey.deployment.compose_generator import find_existing_compose_files

    compose_files = find_existing_compose_files(config, deployed_service_names, quiet=True)

    # If no existing compose files found, rebuild them
    if not compose_files:
        logger.info("No existing compose files found, rebuilding...")
        _, compose_files = prepare_compose_files(config_path, dev_mode)
    else:
        logger.info("Using existing compose files for 'down' operation:")
        for f in compose_files:
            logger.info(f"  - {f}")

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))

    # Only add --env-file if .env exists
    env_file = Path(".env")
    if env_file.exists():
        cmd.extend(["--env-file", ".env"])

    cmd.append("down")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


def deploy_restart(config_path, detached=False, expose_network=False):
    """Restart services using container runtime (Docker or Podman).

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    """
    config, compose_files = prepare_compose_files(config_path, expose_network=expose_network)

    # Verify container runtime is actually running
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Honor the same fail-closed/expose guard as deploy_up when re-rendering with
    # a (possibly newly exposed) bind address.
    _ensure_service_tokens(config, expose_network)

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))
    cmd.extend(["--env-file", ".env", "restart"])

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    subprocess.run(cmd)

    # If detached mode requested, detach after restart
    if detached:
        logger.info("Services restarted. Running in detached mode.")


def rebuild_deployment(config_path, detached=False, dev_mode=False, expose_network=False):
    """Rebuild deployment from scratch (clean + up).

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param dev_mode: Development mode for local framework testing
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    """
    config, compose_files = prepare_compose_files(config_path, dev_mode, expose_network)

    # Verify container runtime is actually running (for the rebuild phase)
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Self-provision fail-closed service tokens (see deploy_up) before rebuilding.
    _ensure_service_tokens(config, expose_network)

    # Clean first
    clean_deployment(compose_files, config)

    # Set up environment for containers
    env = os.environ.copy()
    if dev_mode:
        env["DEV_MODE"] = "true"
        logger.key_info("Development mode: DEV_MODE environment variable set for containers")

    # Then start up
    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))

    # Only add --env-file if .env exists
    env_file = Path(".env")
    if env_file.exists():
        cmd.extend(["--env-file", ".env"])
    else:
        logger.warning(
            "No .env file found - services will start with default/empty environment variables"
        )
        logger.info("To configure API keys: cp .env.example .env && edit .env")

    cmd.extend(["up", "--build"])
    if detached:
        cmd.append("-d")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    os.execvpe(cmd[0], cmd, env)
