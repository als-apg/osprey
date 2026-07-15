"""Container lifecycle operations — start, stop, restart, and rebuild.

Manages the lifecycle of containerized service deployments using
Docker or Podman compose.
"""

import os
import secrets
import string
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlsplit

from osprey.deployment.compose_generator import (
    _copy_local_framework_for_override,
    clean_deployment,
    prepare_compose_files,
    resolve_project_name,
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
#
# openobserve is included for a DIFFERENT reason than the others. Its compose
# template *does* carry an insecure ``${ZO_ROOT_USER_PASSWORD:-Complexpass#123}``
# default, so a deploy does not fail without a real secret — left alone it
# silently comes up on a shared, publicly-known password. Because that store
# captures full agent conversation transcripts (every telemetry content gate
# defaults ON), minting a per-deploy ``ZO_ROOT_USER_PASSWORD`` replaces the
# shared default with a strong secret that BOTH the container and the agent's
# telemetry resolver read from the same ``.env`` value (single source of truth).
# The agent's config references ``${ZO_ROOT_USER_PASSWORD:-Complexpass#123}``, so
# it stays launchable before the first deploy and picks up the minted value on
# its next launch. The email is a username, not a secret (it has a sensible
# non-secret default), so only the password is minted. See
# ``_generate_openobserve_password`` for why a plain token recipe won't do.
_SERVICE_TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "event_dispatcher": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "dispatch_worker": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "bluesky": ("BLUESKY_PROMOTE_TOKEN", "BLUESKY_TILED_API_KEY"),
    "openobserve": ("ZO_ROOT_USER_PASSWORD",),
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
    "ZO_ROOT_USER_PASSWORD",  # OpenObserve admin/ingest cred; gates an observability store, not a control-system write path
}


def _default_token() -> str:
    """The default secret recipe, also the one ``.env.template`` documents."""
    return secrets.token_urlsafe(32)


def _generate_openobserve_password() -> str:
    """Mint a ``ZO_ROOT_USER_PASSWORD`` that satisfies OpenObserve's policy.

    OpenObserve refuses to start unless the root password is 8–128 characters
    with at least one lowercase letter, one uppercase letter, one digit, and
    one special (non-alphanumeric) character — otherwise the container
    crash-loops at startup. ``_default_token``'s ``token_urlsafe`` draws from
    ``[A-Za-z0-9_-]``, which carries no character a strict policy counts as
    "special", so that recipe crash-loops the container non-deterministically:
    the same class of failure as ``BLUESKY_TILED_API_KEY``'s Tiled-alphabet
    constraint (see ``_VAR_GENERATORS`` below).

    Build a value that guarantees all four required classes instead, drawing
    every character from ``secrets`` (never ``random``): a 44-char alphanumeric
    core (>=256 bits of entropy on its own, meeting the module's CSPRNG bar)
    plus one guaranteed member of each class, then shuffled so the class
    positions are not fixed. The special is drawn from ``@%*^`` — punctuation
    every reasonable policy counts as "special", and each of which is safe both
    in a ``.env`` value (unlike ``#``, ``$``, quotes, backslash, ``=`` or a
    space, which break dotenv parsing) and in the base64 Basic-auth header the
    resolver computes from it.
    """
    alphabet = string.ascii_letters + string.digits
    chars = [secrets.choice(alphabet) for _ in range(44)]
    chars += [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("@%*^"),
    ]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


# Per-variable overrides of the default recipe. A var absent here gets
# ``_default_token``.
#
# CSPRNG bar: any recipe registered here MUST draw from ``secrets`` (never
# ``random``, a hashed timestamp, or any other non-cryptographic source) and
# MUST yield at least 256 bits of entropy — the same bar ``_default_token``'s
# ``token_urlsafe(32)`` meets. A registered recipe exists to change the
# *alphabet* for a downstream consumer's parsing rules (see
# BLUESKY_TILED_API_KEY below), never to weaken the randomness.
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
    # OpenObserve rejects a root password that misses any of its four required
    # character classes and crash-loops — see _generate_openobserve_password.
    "ZO_ROOT_USER_PASSWORD": _generate_openobserve_password,
}


def _generate_token(var: str) -> str:
    """Mint one secret for ``var`` using its registered recipe."""
    return _VAR_GENERATORS.get(var, _default_token)()


def _validate_ariel_dsn(value: str) -> bool:
    """True if ``value`` parses as a URI whose password is cleanly encoded.

    Ports pySC's discipline of never trusting a hand-assembled connection
    string (F3): a DSN's password segment sits between ``:`` and ``@`` in the
    URI's authority component, so an *unescaped* reserved character (``@ : /
    ? #``) inside the password either steals characters from the wrong field
    (an unescaped ``/`` truncates the authority early, eating the host into
    the path — caught below by the missing/wrong ``hostname``) or silently
    changes what the URI means without raising a parse error. Requiring every
    reserved character to appear only in its ``%XX`` form, and requiring that
    form to percent-decode without error, is what "parses cleanly" means here.
    """
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return False
    try:
        _ = parsed.port
    except ValueError:
        # An unescaped reserved character in the password (/ ? #) truncates
        # the authority component early, leaving a non-numeric fragment where
        # the port belongs — the tell that the real host was swallowed into
        # the path/query/fragment instead of being parsed as part of netloc.
        # (``.hostname`` alone does not catch this: the truncated netloc
        # still yields a plausible-looking, but wrong, hostname.)
        return False
    password = parsed.password
    if password is None:
        return True
    if any(reserved in password for reserved in "@:/?#"):
        return False
    try:
        unquote(password, errors="strict")
    except UnicodeDecodeError:
        return False
    return True


def _validate_openobserve_password(value: str) -> bool:
    """True if ``value`` satisfies OpenObserve's root-password policy.

    OpenObserve refuses to start unless ``ZO_ROOT_USER_PASSWORD`` is 8–128
    characters with at least one lowercase letter, one uppercase letter, one
    digit, and one special (non-alphanumeric) character — a non-conforming
    value crash-loops the container at startup with an OpenObserve-internal
    error. Rejecting it here turns that opaque crash-loop into a clear
    deploy-time failure for an *operator-supplied* password (a minted one
    already conforms — see ``_generate_openobserve_password``), mirroring the
    ``BLUESKY_TILED_API_KEY``/Tiled-alphabet check.
    """
    if not 8 <= len(value) <= 128:
        return False
    return (
        any(c.islower() for c in value)
        and any(c.isupper() for c in value)
        and any(c.isdigit() for c in value)
        and any(not c.isalnum() for c in value)
    )


# Per-variable validators applied to the *effective* value of a required var
# at the deploy boundary (see ``_ensure_service_tokens``), regardless of
# whether that value was freshly minted, carried over from an existing
# ``.env``, supplied by the operator, or overridden in the process
# environment. A var absent from this map has no registered constraint and
# ``_validate_var`` returns True for it — this fails OPEN, the deliberate
# inverse of the ``_LOCAL_EXEC_SAFE_VARS`` allowlist above.
#
# ``_LOCAL_EXEC_SAFE_VARS`` fails CLOSED on an unenumerated var because
# minting there is a privilege grant — the bar for opting a var *out* of that
# safety restriction must be high. Validating a var nobody has triaged yet is
# the opposite kind of decision: opting a var *into* a format constraint is
# additive hardening, not a prerequisite the deploy must clear, so withholding
# it by default must not block an otherwise-working deploy. Adding an entry
# here is opt-in per var, exactly like ``_VAR_GENERATORS``.
_VAR_VALIDATORS: dict[str, Callable[[str], bool]] = {
    # Tiled rejects a non-alphanumeric --api-key at startup (see
    # _VAR_GENERATORS above); reject it here too so an *operator-supplied*
    # key (never minted, so _VAR_GENERATORS never runs on it) fails at deploy
    # time instead of crash-looping the Tiled container.
    "BLUESKY_TILED_API_KEY": str.isalnum,
    "ARIEL_DSN": _validate_ariel_dsn,
    # OpenObserve crash-loops on a root password that misses any required
    # character class; validate an operator-supplied value at deploy time.
    "ZO_ROOT_USER_PASSWORD": _validate_openobserve_password,
}

# Human-readable constraint text shown in the RuntimeError _ensure_service_tokens
# raises on a _VAR_VALIDATORS failure — never the offending value itself. A
# var validated with no entry here falls back to a generic description.
_VAR_VALIDATOR_DESCRIPTIONS: dict[str, str] = {
    "BLUESKY_TILED_API_KEY": (
        "must be alphanumeric — Tiled rejects any other character in --api-key at startup"
    ),
    "ARIEL_DSN": (
        "must parse as a URI whose password contains no unescaped reserved "
        "character (@ : / ? #); percent-encode the password"
    ),
    "ZO_ROOT_USER_PASSWORD": (
        "must be 8–128 characters with at least one lowercase letter, one "
        "uppercase letter, one digit, and one special character — OpenObserve "
        "rejects any weaker root password at startup"
    ),
}


def _effective_value(var: str, dotenv: dict[str, str]) -> str:
    """The value ``_ensure_service_tokens`` treats as authoritative for ``var``.

    Process env wins over ``.env`` (matches ``docker compose --env-file``);
    absent from both yields ``""``.
    """
    return os.environ.get(var, dotenv.get(var, ""))


def _validate_var(var: str, value: str) -> bool:
    """Check ``value`` against ``var``'s registered constraint, if any.

    Returns True (pass) for any var with no registered validator in
    ``_VAR_VALIDATORS`` — see that dict's docstring for the fail-open
    rationale.
    """
    validator = _VAR_VALIDATORS.get(var)
    if validator is None:
        return True
    return validator(value)


def _raise_invalid_var(var: str) -> None:
    """Raise the standard "invalid var" RuntimeError, never the value."""
    constraint = _VAR_VALIDATOR_DESCRIPTIONS.get(
        var, "does not satisfy its registered format constraint"
    )
    raise RuntimeError(f"{var} is invalid: {constraint}. Refusing to deploy. (Value not shown.)")


# Vars checked against their _VAR_VALIDATORS constraint when present, but
# NEVER minted — distinct from _SERVICE_TOKEN_VARS (which mints an unset var
# for a deployed service) and from a bare _VAR_VALIDATORS entry alone (which,
# by itself, only fires for a var some deployed service's required_vars
# already pulled in). A var earns a place here when it carries a registered
# format constraint but no osprey-native service in this deploy system's
# world (_SERVICE_TOKEN_VARS / find_service_config's compose templates) ever
# requires it: ARIEL_DSN is provisioned by the separate osprey-build-deploy
# skill's facility-scaffolding pipeline (its own generated
# docker-compose.yml/.env.template, brought up by the facility's own
# scripts/deploy.sh via a raw `docker compose`/`podman compose` call — never
# through `osprey deploy up` or this module), so no _SERVICE_TOKEN_VARS entry
# will ever declare it. This is defense-in-depth, not enforcement: if an
# operator or other tooling nonetheless places ARIEL_DSN into *this*
# project's effective env, it is validated like any other var — never
# fabricated when absent, never auto-minted when malformed, just rejected
# with a named-var/no-value error.
_VALIDATE_ONLY_VARS: set[str] = {"ARIEL_DSN"}


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

    Independently of all of the above, every var in ``_VALIDATE_ONLY_VARS``
    (e.g. ``ARIEL_DSN``) is checked against its ``_VAR_VALIDATORS`` constraint
    when present in the effective env — but never minted, and never required:
    this runs even when no deployed service pulls in any ``_SERVICE_TOKEN_VARS``
    entry at all, since a validate-only var's presence does not depend on
    ``deployed_services`` membership.
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

    # Unlike required_vars (below), _VALIDATE_ONLY_VARS carries no minting
    # obligation, so nothing here short-circuits on an empty required_vars —
    # env_path must resolve regardless of whether any service needs a token.
    if env_path is None:
        env_path = Path(".env")

    if required_vars:
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
                fh.write(
                    f"{prefix}# Auto-generated service auth tokens (osprey deploy up)\n{block}"
                )
            os.chmod(env_path, 0o600)
            # Log the path and which keys — NEVER the values.
            logger.key_info(
                "Generated auth token(s) %s in %s (gitignored) — keep them secret",
                ", ".join(generated),
                env_path.resolve(),
            )

    # Validate the effective value of every required var — whichever of
    # process env, an existing .env, or a value just minted above the caller
    # actually sees — against its registered _VAR_VALIDATORS constraint (if
    # any). Unconditional: runs on every deploy path (deploy_up, deploy_restart,
    # rebuild_deployment), not only under --expose, so a malformed
    # operator-supplied value is caught on the default loopback deploy too.
    # Re-parse .env since the mint step above may have just appended to it.
    post = parse_dotenv_file(env_path) if env_path.is_file() else {}
    for name in required_vars:
        effective = _effective_value(name, post)
        if expose_network and not effective.strip():
            raise RuntimeError(
                f"{name} is empty; refusing to --expose (bind 0.0.0.0) with an "
                f"empty token. Set {name} in .env to a strong secret."
            )
        if effective and not _validate_var(name, effective):
            _raise_invalid_var(name)

    # Validate-only vars (ARIEL_DSN): checked when present in the same
    # effective-value sense as required_vars above, but never minted when
    # absent and never required to unblock a deploy — see _VALIDATE_ONLY_VARS'
    # docstring for why no _SERVICE_TOKEN_VARS entry can express this today.
    for name in _VALIDATE_ONLY_VARS:
        effective = _effective_value(name, post)
        if not effective:
            continue  # absent — never fabricated, never minted
        if not _validate_var(name, effective):
            _raise_invalid_var(name)


def _ensure_bluesky_substrate_env(config: dict, env_path: Path | None = None) -> None:
    """Auto-configure the bluesky bridge's EPICS-substrate scan devices for a
    VA-backed Bluesky stack, making ``osprey deploy up`` turn-key.

    Additive and non-breaking, mirroring ``_ensure_service_tokens``'s
    "existing value wins, append what's missing" convention: when the
    deployed project is a VA-backed Bluesky stack (BOTH ``"bluesky"`` and
    ``"virtual_accelerator"`` present in ``deployed_services``), derive
    ``BLUESKY_EPICS_SUBSTRATE``/``BLUESKY_EPICS_MOTORS``/``_DETECTORS`` from
    the built project's own ``data/channel_limits.json`` (the canonical
    derivation lives in
    ``osprey.services.bluesky_bridge.substrate_devices.derive_substrate_env``,
    shared with ``tests/e2e/_orm_stack.py``) and append any of those keys not
    already present in the project ``.env``. Any value already set — in the
    process env or an existing ``.env`` — is left untouched, so an
    operator-configured or e2e-harness-configured substrate env is always
    preserved.

    A no-op for any deploy that is not both bluesky- and
    virtual-accelerator-backed (e.g. a plain agent deploy, or bluesky without
    the VA): nothing is read or written. This makes the bridge substrate-mode
    with real channel names available regardless of
    ``control_system.type`` -- the bridge's own connector backend follows
    that setting separately.

    Never raises into a deploy: a missing/unreadable ``channel_limits.json``
    or a derivation that yields no correctors/BPMs logs a warning and is
    skipped, leaving the bridge to fall back to its own demo-runner default
    (or a manually-set substrate env) exactly as before this function
    existed.

    :param config: Raw deploy config (``deployed_services`` membership).
    :param env_path: Project ``.env`` path; defaults to ``Path(".env")``
        (matching ``_ensure_service_tokens``), i.e. resolved against the
        current working directory -- ``osprey deploy`` always chdirs into
        the project directory first. Overridable for tests.
    """
    deployed_services = config.get("deployed_services")
    services = {str(s) for s in (deployed_services or [])}
    if "bluesky" not in services or "virtual_accelerator" not in services:
        return

    # The substrate runner drives real Channel Access devices, which only
    # exist behind a real or virtual IOC. A ``mock`` control system speaks no
    # CA, so a mock deploy must stay on the bridge's demo runner -- the
    # documented "mock = safe browse/demo, virtual_accelerator = real run"
    # contract. Arming substrate here would win over an explicit demo_runner
    # (see ``bluesky_bridge.app``'s substrate-vs-demo precedence) and leave the
    # bridge trying to resolve scan devices that only the mock demo provides.
    # Only auto-configure substrate for a control system that actually speaks CA.
    control_system_type = str(config.get("control_system", {}).get("type", "mock")).strip().lower()
    if control_system_type == "mock":
        logger.info(
            "control_system.type is 'mock'; leaving the bluesky bridge on its demo "
            "runner and skipping BLUESKY_EPICS_SUBSTRATE auto-configuration "
            "(substrate mode needs a real or virtual IOC to speak CA to)."
        )
        return

    if env_path is None:
        env_path = Path(".env")
    project_dir = env_path.resolve().parent

    from osprey.services.bluesky_bridge.substrate_devices import derive_substrate_env

    try:
        derived = derive_substrate_env(project_dir)
    except Exception:
        logger.warning(
            "Could not auto-configure bluesky bridge scan devices from %s "
            "(derivation raised unexpectedly). Skipping BLUESKY_EPICS_SUBSTRATE "
            "auto-configuration -- set BLUESKY_EPICS_MOTORS/_DETECTORS manually "
            "if you want the bridge to run in EPICS-substrate mode.",
            project_dir / "data" / "channel_limits.json",
            exc_info=True,
        )
        return
    if not derived:
        logger.warning(
            "Could not auto-configure bluesky bridge scan devices from %s "
            "(missing, unreadable, or yields no SR correctors/BPMs). Skipping "
            "BLUESKY_EPICS_SUBSTRATE auto-configuration -- set "
            "BLUESKY_EPICS_MOTORS/_DETECTORS manually if you want the bridge "
            "to run in EPICS-substrate mode.",
            project_dir / "data" / "channel_limits.json",
        )
        return

    existing = parse_dotenv_file(env_path) if env_path.is_file() else {}
    generated = {k: v for k, v in derived.items() if k not in os.environ and k not in existing}
    if not generated:
        return

    prefix = ""
    if env_path.is_file():
        text = env_path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            prefix = "\n"
    block = "".join(f"{k}={v}\n" for k, v in generated.items())
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{prefix}# Auto-configured bluesky bridge scan devices (osprey deploy up)\n{block}"
        )
    logger.key_info(
        "Auto-configured bluesky bridge scan devices %s in %s from the project's "
        "own channel_limits.json",
        ", ".join(generated),
        env_path.resolve(),
    )


def _resolve_claude_cli_version(config: dict) -> str:
    """The ``CLAUDE_CLI_VERSION`` build arg for the project image.

    Uses the project's ``claude_code.cli_version`` pin when set, else the
    framework default scaffolding bakes into the rendered ``Dockerfile`` — the
    single source of that fallback (``osprey.cli.templates.scaffolding``), so the
    build-time CLI pin never drifts from the value the Dockerfile documents.
    """
    version = config.get("claude_code", {}).get("cli_version")
    if version:
        return str(version)
    # Lazy import: keep the CLI-templates package off the deploy import path
    # unless the fallback is actually needed.
    from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

    return _DEFAULT_CLAUDE_CLI_VERSION


def _resolve_pip_spec() -> str:
    """The ``OSPREY_PIP_SPEC`` build arg for the project image.

    An operator ``OSPREY_PIP_SPEC`` export wins (e.g. a ``git+https`` URL that
    pins an unreleased build). Otherwise pin the running framework version
    (``osprey-framework==<version>``), matching the dispatch image's production
    install, so a project image built without ``--dev`` ships a deterministic
    release rather than tracking whatever ``osprey-framework`` resolves to at
    build time. Under ``--dev`` a locally-built wheel is staged into the build
    context and the Dockerfile installs that instead, ignoring this spec.
    """
    spec = os.environ.get("OSPREY_PIP_SPEC")
    if spec:
        return spec
    try:
        from osprey import __version__ as osprey_version
    except Exception:
        osprey_version = ""
    return f"osprey-framework=={osprey_version}" if osprey_version else "osprey-framework"


def _worker_image_target(config: dict, env: dict) -> str:
    """The image the dispatch worker will actually run.

    Resolution mirrors the worker compose service's
    ``${OSPREY_WORKER_IMAGE:-<services.dispatch_worker.image | default>}``:
    an ``OSPREY_WORKER_IMAGE`` env override wins, then a profile-pinned
    ``services.dispatch_worker.image``, else the ``<project>:local`` project
    image that :func:`_build_project_image` builds.
    """
    override = env.get("OSPREY_WORKER_IMAGE")
    if override:
        return str(override)
    worker_cfg = config.get("services", {}).get("dispatch_worker", {})
    explicit = worker_cfg.get("image") if isinstance(worker_cfg, dict) else None
    if explicit:
        return str(explicit)
    return f"{resolve_project_name(config)}:local"


def _project_image_build_cmd(config: dict, runtime: str, project_root: str) -> list[str]:
    """Construct the ``<runtime> build`` argv that produces ``<project>:local``.

    :param config: Raw deploy config (project name, ``claude_code.cli_version``).
    :param runtime: Base container command (``docker`` or ``podman``).
    :param project_root: Build context — the project root that holds the
        rendered ``Dockerfile`` (and, under ``--dev``, the staged wheel).
    :return: The full build command as an argv list.
    """
    project_name = resolve_project_name(config)
    dockerfile = os.path.join(project_root, "Dockerfile")
    return [
        runtime,
        "build",
        "-t",
        f"{project_name}:local",
        "-f",
        dockerfile,
        "--build-arg",
        f"CLAUDE_CLI_VERSION={_resolve_claude_cli_version(config)}",
        "--build-arg",
        f"OSPREY_PIP_SPEC={_resolve_pip_spec()}",
        project_root,
    ]


def _build_project_image(config: dict, dev_mode: bool, env: dict) -> None:
    """Build the ``<project>:local`` image the dispatch worker references.

    The dispatch worker's compose service intentionally has no ``build:`` block
    (a second builder for the same tag would race the event-dispatcher — see
    ``dispatch_worker/docker-compose.yml.j2``), so nothing in ``compose up``
    produces the image it runs. This builds it once, from the project root
    (context) and the rendered project ``Dockerfile``, before ``compose up``.

    No-op unless the worker is deployed and its effective image is the local
    ``<project>:local`` tag: an ``OSPREY_WORKER_IMAGE`` override or a
    profile-pinned ``services.dispatch_worker.image`` means a prebuilt image is
    wanted, so there is nothing to build. The event-dispatcher's own
    ``osprey-dispatch:local`` build (its compose ``build:`` block) is untouched.

    Under ``dev_mode`` a wheel is built from the local osprey checkout and staged
    into the build context (mirroring the dispatch image's ``--dev`` convention);
    the Dockerfile's wheel-drop branch then installs it so unreleased code is
    baked in. The staged wheel is removed afterward so it cannot poison a later
    non-dev build (whose wheel-drop branch fires on any ``*.whl`` in the context).

    :param config: Raw deploy config.
    :param dev_mode: Whether ``--dev`` was passed (stage a local wheel).
    :param env: Environment for the build subprocess (also read for
        ``OSPREY_WORKER_IMAGE``).
    """
    services = {str(s) for s in (config.get("deployed_services") or [])}
    if "dispatch_worker" not in services:
        return

    target = _worker_image_target(config, env)
    project_image = f"{resolve_project_name(config)}:local"
    if target != project_image:
        logger.key_info(
            "Dispatch worker uses image %r (OSPREY_WORKER_IMAGE / pinned "
            "services.dispatch_worker.image) — skipping %s build.",
            target,
            project_image,
        )
        return

    runtime = get_runtime_command(config)[0]
    project_root = os.getcwd()

    staged_wheels: list[Path] = []
    if dev_mode:
        before = set(Path(project_root).glob("*.whl"))
        _copy_local_framework_for_override(project_root)
        staged_wheels = list(set(Path(project_root).glob("*.whl")) - before)

    try:
        cmd = _project_image_build_cmd(config, runtime, project_root)
        logger.key_info("Building dispatch worker project image %s:", project_image)
        logger.info("Running command:\n    %s", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
    finally:
        for whl in staged_wheels:
            try:
                whl.unlink()
            except OSError:
                logger.warning("Could not remove staged dev wheel %s", whl)


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

    # Auto-configure the bluesky bridge's EPICS-substrate scan devices for a
    # VA-backed Bluesky stack (additive; no-op unless both bluesky and
    # virtual_accelerator are deployed) -- see _ensure_bluesky_substrate_env.
    _ensure_bluesky_substrate_env(config)

    # Set up environment for containers
    env = os.environ.copy()
    if dev_mode:
        env["DEV_MODE"] = "true"
        logger.key_info("Development mode: DEV_MODE environment variable set for containers")

    # Build the <project>:local image the dispatch worker references. The worker
    # has no compose build block (that would race the event-dispatcher on the
    # shared tag), so this is the only thing that produces its image. No-op
    # unless the worker is deployed on the local project image. Run before
    # `compose up` (which, non-detached, os.execvpe-replaces this process).
    _build_project_image(config, dev_mode, env)

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

    # Rebuild the dispatch worker's <project>:local image too (see deploy_up).
    _build_project_image(config, dev_mode, env)

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
