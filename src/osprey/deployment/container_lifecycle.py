"""Container lifecycle operations — start, stop, restart, and rebuild.

Manages the lifecycle of containerized service deployments using
Docker or Podman compose.
"""

import os
import secrets
import subprocess
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
# so either service alone still needs both vars minted. bluesky is a single
# fail-closed bridge instance needing only its own promote token — see
# ``security.require_armed`` in ``osprey.services.bluesky_bridge``.
_SERVICE_TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "event_dispatcher": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "dispatch_worker": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "bluesky": ("BLUESKY_PROMOTE_TOKEN",),
}

# Services whose fail-closed token must NOT be auto-armed when the agent's code
# execution is unsandboxed on the host (see _local_exec_arming_unsafe below).
# Their bearer token gates a write-capable HTTP route (bluesky-bridge's
# POST /runs/{id}/promote); under local exec, agent-authored code can read that
# token straight out of .env/config.yml and call the route directly, bypassing
# the in-tool writes_enabled re-check that's supposed to gate it. The
# event-dispatch services aren't in this set: their tokens gate an inbound
# webhook/worker-routing boundary, not a write path the agent itself walks.
_LOCAL_EXEC_GATED_SERVICES = {"bluesky"}


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
    generate a strong random value (``secrets.token_urlsafe(32)``, the same
    recipe the ``.env.template`` documents) and append it to ``.env``
    (``chmod 0o600``, matching the build-time convention). Existing values are
    never overwritten, so re-running ``deploy up`` is idempotent. No-op unless
    a token-requiring service is actually deployed.

    A token that is *present but explicitly empty* (e.g. ``TOKEN=`` exported in
    the shell) is left untouched — generating would silently override a
    deliberate value. For a loopback deploy the server simply fails closed; for
    an exposed deploy (``--expose`` / bind 0.0.0.0) we refuse rather than bind a
    fail-open-at-bind server to all interfaces.

    A service in ``_LOCAL_EXEC_GATED_SERVICES`` is skipped entirely (no token
    minted, existing values in .env/env left untouched but not read either) when
    ``_local_exec_arming_unsafe(config)`` — see that function's docstring. The
    bridge's own ``require_armed()`` then keeps returning 503, i.e. it never
    arms, rather than being armed with a token any local agent-code execution
    can trivially read back out of ``.env``.
    """
    deployed_services = config.get("deployed_services")
    services = {str(s) for s in (deployed_services or [])}
    unsafe_local_exec = _local_exec_arming_unsafe(config)

    # Iterate the map (not the deployed `services` set) so var order is
    # deterministic regardless of set iteration order or config.yml ordering.
    required_vars: list[str] = []
    seen: set[str] = set()
    for svc_name, token_vars in _SERVICE_TOKEN_VARS.items():
        if svc_name not in services:
            continue
        if svc_name in _LOCAL_EXEC_GATED_SERVICES and unsafe_local_exec:
            logger.warning(
                "Refusing to arm %r: control_system.writes_enabled is true but "
                "execution.execution_method is 'local' (agent code runs unsandboxed "
                "on the host, cwd=project_root). Local execution can read %s "
                "straight out of .env/config.yml and call the write-capable route "
                "it gates directly, bypassing the in-tool writes_enabled re-check. "
                "NOT minting %s — the service stays unarmed. Set "
                "execution.execution_method: container to use this feature with "
                "writes_enabled: true.",
                svc_name,
                ", ".join(token_vars),
                ", ".join(token_vars),
            )
            continue
        for var in token_vars:
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
        generated[name] = secrets.token_urlsafe(32)

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
