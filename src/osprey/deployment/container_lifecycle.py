"""Container lifecycle operations — start, stop, restart, and rebuild.

Manages the lifecycle of containerized service deployments using
Docker or Podman compose.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from osprey.deployment.compose_generator import (
    _copy_local_framework_for_override,
    clean_deployment,
    prepare_compose_files,
    resolve_project_name,
)
from osprey.deployment.deploy_summary import log_endpoint_summary
from osprey.deployment.errors import ComposeInterpolationError
from osprey.deployment.facility_config import normalize_facility_config
from osprey.deployment.host_ports import (
    find_port_conflicts,
    format_conflict_report,
    parse_host_port_bindings,
)
from osprey.deployment.runtime_helper import (
    get_runtime_command,
    runtime_env,
    verify_runtime_is_running,
)
from osprey.deployment.service_tokens import (
    _VAR_GENERATORS,  # noqa: F401  (re-exported for tests)
    _VAR_VALIDATORS,  # noqa: F401  (re-exported for tests)
    _effective_value,
    _generate_openobserve_password,  # noqa: F401  (re-exported for tests)
    _generate_token,
    _raise_invalid_var,
    _validate_openobserve_password,  # noqa: F401  (re-exported for tests)
    _validate_var,
)
from osprey.deployment.staleness import warn_if_project_stale
from osprey.deployment.web_terminals.provision import (
    deploy_down_web_terminals,
    deploy_up_web_terminals,
    preflight_web_terminals,
)
from osprey.deployment.wheel_build import _staged_dev_artifact_paths
from osprey.errors import BuildProfileError
from osprey.utils.config import ConfigBuilder
from osprey.utils.dotenv import (
    append_profile_env,
    atomic_write,
    compose_unsafe_vars,
    derive_project_env,
    parse_dotenv_file,
)
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")

# Services that fail closed on an unset bearer token (no insecure default in
# their compose templates), so a deploy must supply real secrets. Maps each
# deployed-service name to the token env var(s) it requires. event_dispatcher
# and dispatch_worker both list the same pair because they share one .env:
# the dispatcher forwards routed requests to workers using DISPATCH_WORKER_TOKEN,
# so either service alone still needs both vars minted. bluesky needs its own
# launch token (see ``security.require_armed`` in
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
#
# postgresql follows the openobserve rationale: its compose template carries an
# insecure ``${ARIEL_DB_PASSWORD:-ariel}`` default, so left alone the ARIEL
# store comes up on a shared, publicly-known password. Minting a per-deploy
# ``ARIEL_DB_PASSWORD`` replaces it with a strong secret that the container
# (POSTGRES_PASSWORD) and the agent's DSN (derived from the ``services.postgresql``
# block by ``resolve_ariel_dsn``, which substitutes the same
# ``${ARIEL_DB_PASSWORD:-ariel}``) read from the same ``.env`` value. An explicit
# ``ariel.database.uri`` still wins when a project sets one.
# NOTE: Postgres only reads POSTGRES_PASSWORD when *initializing* a fresh data
# volume — a pre-existing volume keeps its original password (same
# stale-volume caveat as openobserve; recreate the volume to adopt the minted
# value).
_SERVICE_TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "event_dispatcher": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "dispatch_worker": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "bluesky": ("BLUESKY_LAUNCH_TOKEN", "BLUESKY_TILED_API_KEY"),
    "openobserve": ("ZO_ROOT_USER_PASSWORD",),
    "postgresql": ("ARIEL_DB_PASSWORD",),
}

# Vars checked against their _VAR_VALIDATORS constraint when present, but
# NEVER minted — distinct from _SERVICE_TOKEN_VARS (which mints an unset var
# for a deployed service) and from a bare _VAR_VALIDATORS entry alone (which,
# by itself, only fires for a var some deployed service's required_vars
# already pulled in). A var earns a place here when it carries a registered
# format constraint but no osprey-native service in this deploy system's
# world (_SERVICE_TOKEN_VARS / find_service_config's compose templates) ever
# requires it: ARIEL_DSN is provisioned by the separate osprey-build-deploy
# skill's facility-scaffolding pipeline (its own generated
# docker-compose.yml/.env.template for that facility's ARIEL stack) — not
# minted by this deploy system's token path, so no _SERVICE_TOKEN_VARS entry
# will ever declare it. This is defense-in-depth, not enforcement: if an
# operator or other tooling nonetheless places ARIEL_DSN into *this*
# project's effective env, it is validated like any other var — never
# fabricated when absent, never auto-minted when malformed, just rejected
# with a named-var/no-value error.
_VALIDATE_ONLY_VARS: set[str] = {"ARIEL_DSN"}

# Document-plane CURVE certificate layout, relative to the project directory.
# Fixed by the read-only mounts the bluesky compose template declares
# (``../../data/bluesky_curve/<side>`` -> ``/app/curve``) together with the
# certificate paths it hardcodes into each container's environment, so this
# constant and that template are one contract: change either and the bridge
# refuses to bind its proxy.
#
# Split per side rather than per file so neither container can read the
# secret it is not supposed to hold. The bridge (which binds the proxy, and
# so holds the SERVER half) gets the proxy's secret certificate plus a
# ``clients/`` directory of the publisher public keys it will accept; the
# queueserver (the publishing CLIENT) gets the publisher's secret certificate
# plus the proxy's public one.
_BLUESKY_CURVE_DIR = Path("data") / "bluesky_curve"

# Control-plane (RE Manager 0MQ control socket) CURVE keypair. Unlike the
# document plane's certificates these are z85 key *strings*, not files:
# ``start-re-manager`` and ``bluesky-queueserver-api`` both read them straight
# out of the environment, so the compose template passes them through from the
# project ``.env`` under these OSPREY-namespaced names.
_QSERVER_ZMQ_PRIVATE_KEY_VAR = "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY"
_QSERVER_ZMQ_PUBLIC_KEY_VAR = "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"


def _append_env_block(env_path: Path, comment: str, values: dict[str, str]) -> None:
    """Append a commented block of ``KEY=value`` lines to the project ``.env``.

    The single write convention for every deploy-time provisioner that adds to
    the project ``.env``: keep any trailing newline sane, append rather than
    rewrite (an operator's own entries are never touched), and ``chmod 0o600``
    afterwards.

    The ``chmod`` is unconditional even for blocks whose own values are not
    secret (the EPICS-substrate device names, say). The project ``.env`` is
    secrets-destined by construction — service auth tokens and the RE manager's
    control-socket private key land in the same file — so the mode is a
    property of the file, not of any one block. Keeping it here means no call
    site has to judge whether its own values are sensitive, and a provisioner
    that happens to be the one that *creates* ``.env`` cannot leave a
    world-readable file behind for a later block to fill with secrets.

    Args:
        env_path: Project ``.env`` path.
        comment: Header comment for the appended block, written verbatim after
            a ``# `` prefix.
        values: Variables to append, in the order they should be written.
    """
    prefix = ""
    if env_path.is_file():
        text = env_path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            prefix = "\n"
    block = "".join(f"{k}={v}\n" for k, v in values.items())
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{prefix}# {comment}\n{block}")
    os.chmod(env_path, 0o600)


def _profile_env_path(project_dir: Path) -> Path | None:
    """The profile ``.env`` that owns this project's secrets, or ``None``.

    The profile is located through the project's own manifest
    (``build_args.profile_path_abs``, the path ``osprey build`` resolved), and
    the ``.env`` sits at the profile *root* — a persona delta shares the root
    profile's secrets rather than keeping its own beside itself.

    ``None`` means there is no profile to write to: a preset-built project, a
    manifest predating the key, or a persona delta whose root has gone missing.
    The path is not required to exist — a profile directory that has since been
    deleted resolves here and fails at the write, which is where the degraded
    path names it.
    """
    from osprey.cli.profile_root import resolve_profile_root
    from osprey.cli.templates.manifest import manifest_profile_path

    profile_path = manifest_profile_path(project_dir)
    if profile_path is None:
        return None
    try:
        root_dir, _ = resolve_profile_root(profile_path)
    except BuildProfileError:
        # A persona delta whose root profile is gone. Reporting it as "no
        # profile" is honest — we cannot say which directory the secrets belong
        # in — and the manifest flag still records that they were not synced.
        logger.warning(
            "Cannot locate the profile root for %s (its profile.yml is missing), so the "
            "service secrets this deploy uses stay in the project .env only.",
            profile_path,
        )
        return None
    return root_dir / ".env"


def _sync_secrets_to_profile(env_path: Path, entries: dict[str, str]) -> None:
    """Persist the secrets this deploy is running with into the owning profile.

    The profile ``.env`` is the source of truth for facility secrets, but a
    deploy is where several of them first come into existence (minted tokens,
    volume-initializing passwords). Writing them back is what makes the project
    reproducible: a rebuild from the same profile comes up on the same secrets
    instead of minting a second set the running containers do not trust.

    **Append-only.** A key already in the profile keeps its value — it is pinned
    by the docker volume initialized with it and by every container already
    trusting it — and a supplied value that disagrees is reported by name (never
    by value) for the operator to resolve. The project ``.env`` is then
    re-derived from the profile in ``deploy`` mode: profile-carried keys refresh
    in place, keys the project lacks are appended, and nothing is deleted — a
    deploy must not prune a project it did not build.

    Degraded path — no profile recorded, or one that cannot be written (a
    registry-mode host, a profile directory that has moved or been deleted): the
    secrets stay in the project ``.env`` where they were minted, a warning names
    the path that failed, and ``secrets_synced_to_profile: false`` goes into the
    manifest so ``osprey build --force`` says so before wiping the only copy.

    Never raises into a deploy: every failure here degrades to that path.

    :param env_path: The project ``.env`` the deploy is writing.
    :param entries: The effective value of each secret, keyed by var name.
    """
    from osprey.cli.templates.manifest import write_secrets_synced_to_profile

    project_dir = env_path.resolve().parent
    try:
        _sync_secrets_to_profile_inner(env_path, project_dir, entries)
    except Exception:
        logger.warning(
            "Could not persist this deploy's service secrets to the profile .env; they "
            "remain in %s only. Re-running osprey deploy up after fixing the profile "
            "will sync them.",
            env_path.resolve(),
            exc_info=True,
        )
        try:
            write_secrets_synced_to_profile(project_dir, False)
        except OSError:
            # Whatever stopped the write-back may equally stop the stamp (a
            # read-only project dir, a full disk). The warning above is already
            # out; losing the manifest flag on top of it must not be what
            # finally raises into the deploy.
            logger.debug(
                "Could not record secrets_synced_to_profile in the manifest at %s",
                project_dir,
                exc_info=True,
            )


def _sync_secrets_to_profile_inner(
    env_path: Path, project_dir: Path, entries: dict[str, str]
) -> None:
    """The write-back half of :func:`_sync_secrets_to_profile`, free to raise."""
    from osprey.cli.templates.manifest import write_secrets_synced_to_profile

    profile_env = _profile_env_path(project_dir)
    if profile_env is None:
        # Not a failure: a preset-built project has no profile to own its
        # secrets. The flag still records that the project .env is the only
        # copy, which is what the pre-wipe warning needs to know.
        write_secrets_synced_to_profile(project_dir, False)
        return

    try:
        result = append_profile_env(profile_env, entries)
    except OSError as exc:
        logger.warning(
            "Could not write the service secrets to the profile .env at %s (%s). They "
            "were minted into %s instead and exist nowhere else — copy them into the "
            "profile, or re-run osprey deploy up once the profile is reachable.",
            profile_env,
            exc,
            env_path.resolve(),
        )
        write_secrets_synced_to_profile(project_dir, False)
        return

    for conflict in result.conflicts:
        # Values are never logged: the point is which var disagrees, not what
        # either side holds.
        logger.warning(
            "%s differs between this deploy and the profile .env at %s. The profile's "
            "value was kept (a minted secret is pinned by the volume and containers "
            "that adopted it); this deploy keeps using its own. Reconcile them by hand "
            "if the two stacks are meant to share the secret.",
            conflict.key,
            profile_env,
        )

    if result.added:
        logger.key_info(
            "Persisted service secret(s) %s to the profile .env at %s — a rebuild from "
            "this profile will come up on the same secrets",
            ", ".join(result.added),
            profile_env,
        )

    _derive_project_env_from_profile(env_path, profile_env)
    write_secrets_synced_to_profile(project_dir, not result.conflicts)


def _derive_project_env_from_profile(env_path: Path, profile_env: Path) -> None:
    """Refresh the project ``.env`` from the profile, overlay-only.

    Deploy-mode derivation: there is no render to compare against here, so the
    existing project file supplies the whole shape and the profile only updates
    the keys it carries and appends the ones the project is missing. Written
    through a temp file and ``os.replace`` — the whole file is rewritten, so a
    crash mid-write must not truncate an operator's ``.env``.
    """
    profile_text = profile_env.read_text(encoding="utf-8") if profile_env.is_file() else ""
    if not profile_text.strip():
        return
    existing_text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    derived = derive_project_env(profile_text, "", existing_text, mode="deploy")
    if derived == existing_text:
        return

    atomic_write(env_path, derived)


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

    Minting is unconditional for every var a deployed service declares: these
    tokens authenticate *network callers* to a service's own HTTP boundary and
    are not a hardware-safety layer. Whether a scan or a write is permitted is
    decided at the connector (``writes_enabled`` plus the per-put channel
    limits), which every write path — agent-side and bridge-side alike — must
    still clear. No deploy-time value is read here for safety semantics.

    Whatever the effective values turn out to be — minted here, already in the
    project ``.env``, or exported in the shell — they are then persisted to the
    ``.env`` of the profile the project was built from, so a rebuild reproduces
    this stack rather than minting a second set of tokens its containers do not
    trust. See :func:`_sync_secrets_to_profile` for the append-only rule and the
    degraded path when there is no reachable profile.

    Independently of the above, every var in ``_VALIDATE_ONLY_VARS``
    (e.g. ``ARIEL_DSN``) is checked against its ``_VAR_VALIDATORS`` constraint
    when present in the effective env — but never minted, and never required:
    this runs even when no deployed service pulls in any ``_SERVICE_TOKEN_VARS``
    entry at all, since a validate-only var's presence does not depend on
    ``deployed_services`` membership.
    """
    deployed_services = config.get("deployed_services")
    services = {str(s) for s in (deployed_services or [])}

    # Iterate the map (not the deployed `services` set) so var order is
    # deterministic regardless of set iteration order or config.yml ordering.
    required_vars: list[str] = []
    seen: set[str] = set()
    for svc_name, token_vars in _SERVICE_TOKEN_VARS.items():
        if svc_name not in services:
            continue
        for var in token_vars:
            if var not in seen:
                seen.add(var)
                required_vars.append(var)

    # Unlike required_vars (below), _VALIDATE_ONLY_VARS carries no minting
    # obligation, so nothing here short-circuits on an empty required_vars —
    # env_path must resolve regardless of whether any service needs a token.
    if env_path is None:
        env_path = Path(".env")

    # Scan the WHOLE file, ahead of everything else, because compose mangles a
    # `$` on BOTH routes out of this file and every deploy uses both:
    #
    #   * `--env-file .env` (see _env_file_args) makes it the substitution
    #     source for the compose DOCUMENT, so every `environment: - X=${X}`
    #     entry — how most services here receive their secrets — resolves
    #     through it;
    #   * the dispatch worker additionally gets the file entire, via
    #     `env_file: ../../.env` in its compose template.
    #
    # Measured on Docker Compose v2.34: `h0rse$battery` substitutes to `h0rse`,
    # and `secret$HOME` to the deploy host's home path with NO warning at all.
    # podman-compose eats only the braced `${...}` form — a different mangling
    # of the same file, which is why the rule is "no `$`" and not "no
    # interpolating `$`".
    # So the check cannot be per-var — _validate_var below only ever sees the
    # deployed services' _SERVICE_TOKEN_VARS plus _VALIDATE_ONLY_VARS, while
    # the provider API key, the olog password and the wiki-search token all
    # travel in this file and are in neither set.
    #
    # Before the mint, deliberately. Scanning after would leave freshly minted
    # tokens appended to .env while the raise skips _sync_secrets_to_profile,
    # so the next `osprey build` re-derives .env from the profile, drops them,
    # and mints a second set. Nothing is lost by checking early: every minted
    # value is `$`-free by construction (see the alphabets in service_tokens),
    # which the generator self-rejection test pins.
    #
    # The file — not the effective value — is what matters: compose reads it
    # off disk, so a process-env override never reaches a container this way.
    if env_path.is_file():
        offenders = compose_unsafe_vars(parse_dotenv_file(env_path))
        if offenders:
            raise ComposeInterpolationError(offenders, env_path)

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
            _append_env_block(
                env_path,
                "Auto-generated service auth tokens (osprey deploy up)",
                generated,
            )
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
            _raise_invalid_var(name, effective)

    # Validate-only vars (ARIEL_DSN): checked when present in the same
    # effective-value sense as required_vars above, but never minted when
    # absent and never required to unblock a deploy — see _VALIDATE_ONLY_VARS'
    # docstring for why no _SERVICE_TOKEN_VARS entry can express this today.
    for name in _VALIDATE_ONLY_VARS:
        effective = _effective_value(name, post)
        if not effective:
            continue  # absent — never fabricated, never minted
        if not _validate_var(name, effective):
            _raise_invalid_var(name, effective)

    # Persist the *effective* secrets — not just the ones minted a moment ago —
    # into the profile that owns them, so a rebuild reproduces this stack
    # instead of minting a second set of tokens the running containers reject.
    # Values already in the profile win; see _sync_secrets_to_profile.
    if required_vars:
        effective_secrets = {
            name: value for name in required_vars if (value := _effective_value(name, post))
        }
        if effective_secrets:
            _sync_secrets_to_profile(env_path, effective_secrets)


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
    skipped, leaving the bridge and the queueserver worker with an empty
    device namespace — browse-only, exactly as if no substrate env had ever
    been set — unless an operator supplies one by hand.

    Deliberately *not* written back to the profile ``.env``, unlike the service
    secrets ``_ensure_service_tokens`` syncs there. These vars are derived
    configuration, not state only a deploy can produce: their source is the
    project's own ``channel_limits.json``. The write-back exists to preserve
    what nothing else can reproduce, so the profile is the wrong place to pin a
    device list this function derives.

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

    # The substrate devices are ophyd-async Channel Access devices, which only
    # exist behind a real or virtual IOC. A ``mock`` control system speaks no
    # CA, so there is nothing to point them at -- the documented "mock =
    # browse-only, virtual_accelerator = real run" contract. Deriving a
    # substrate here would hand the queueserver worker device names it can
    # never connect to, turning a clean browse-only deployment into one whose
    # environment fails to open. Only auto-configure substrate for a control
    # system that actually speaks CA.
    control_system_type = str(config.get("control_system", {}).get("type", "mock")).strip().lower()
    if control_system_type == "mock":
        logger.info(
            "control_system.type is 'mock': this deployment is browse-only -- plans can "
            "be listed and composed but never executed. Skipping "
            "BLUESKY_EPICS_SUBSTRATE auto-configuration (scan devices need an "
            "EPICS-like connector to speak Channel Access to). Flip it with "
            "`osprey config set-control-system virtual_accelerator`."
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

    _append_env_block(
        env_path,
        "Auto-configured bluesky bridge scan devices (osprey deploy up)",
        generated,
    )
    logger.key_info(
        "Auto-configured bluesky bridge scan devices %s in %s from the project's "
        "own channel_limits.json",
        ", ".join(generated),
        env_path.resolve(),
    )


def _bluesky_curve_paths(project_dir: Path) -> dict[str, Path]:
    """The document-plane certificate paths for one project, by role.

    Args:
        project_dir: Project directory (the parent of the project ``.env``).

    Returns:
        Mapping of role name to path. ``bridge``/``queueserver``/
        ``bridge_clients`` are directories (the compose mount sources); the
        four remaining entries are the certificate files each side reads.
    """
    base = project_dir / _BLUESKY_CURVE_DIR
    bridge = base / "bridge"
    queueserver = base / "queueserver"
    clients = bridge / "clients"
    return {
        "bridge": bridge,
        "bridge_clients": clients,
        "queueserver": queueserver,
        # Server half — held only by the bridge, which binds the proxy.
        "proxy_secret": bridge / "proxy.key_secret",
        # Client half — held only by the queueserver, which publishes.
        "publisher_secret": queueserver / "publisher.key_secret",
        # Public halves, crossed over: each side pins the other's identity.
        "proxy_public": queueserver / "proxy.key",
        "publisher_public": clients / "publisher.key",
    }


def _ensure_bluesky_document_plane_certs(config: dict, env_path: Path | None = None) -> None:
    """Generate the document plane's CURVE certificates into the project tree.

    The bridge binds a ``bluesky.callbacks.zmq`` Proxy that the queueserver's
    Publisher pushes run documents at. Both containers are dual-homed onto
    ``osprey-network`` (they need the Virtual Accelerator and Tiled), so
    network placement alone cannot keep a rogue container from injecting
    forged run documents — key authentication is what does, and that needs
    certificate material on disk before ``compose up`` mounts it.

    Generation is **unconditional** for any deploy carrying the ``bluesky``
    service, and deliberately NOT gated on the connector being EPICS-like the
    way substrate derivation is. A mock deploy never opens the RE worker
    environment, so empty certificate directories would cost it nothing today
    — but they are not free: the moment an operator flips the connector to
    ``virtual_accelerator`` and restarts without a full redeploy, the bridge
    finds no secret certificate and refuses to bind the plane at all (see
    ``DocumentPlaneConfig.from_env``), which reads as "live rows silently
    stopped working" rather than as a missing deploy step. Missing certificates
    degrade to no document publishing with a loud log, never to an
    unauthenticated socket, so generating them everywhere buys capability with
    no safety cost.

    Idempotent and existing-material-wins: a complete set is left exactly as
    it is, so redeploying never rotates keys out from under a running stack.
    An *incomplete* set is regenerated whole — a certificate's secret half
    cannot be recovered from its public one, so a half-present set is
    unusable and repairing it in place is not possible.

    Never raises into a deploy: any failure logs a warning and leaves the
    stack to come up without a document plane (no live rows), which is the
    same degradation the bridge already handles.

    Args:
        config: Raw deploy config (``deployed_services`` membership).
        env_path: Project ``.env`` path, used only to locate the project
            directory; defaults to ``Path(".env")`` like every other
            provisioning step here.
    """
    services = {str(s) for s in (config.get("deployed_services") or [])}
    if "bluesky" not in services:
        return

    if env_path is None:
        env_path = Path(".env")
    paths = _bluesky_curve_paths(env_path.resolve().parent)

    certs = ("proxy_secret", "publisher_secret", "proxy_public", "publisher_public")
    present = [name for name in certs if paths[name].is_file()]
    if len(present) == len(certs):
        logger.info(
            "Bluesky document-plane CURVE certificates already present in %s — keeping them",
            paths["bridge"].parent,
        )
        return
    if present:
        logger.warning(
            "Bluesky document-plane CURVE certificates in %s are incomplete (%d of %d "
            "present) — regenerating the whole set, since a certificate's secret half "
            "cannot be rebuilt from its public one. Every container currently running "
            "against the old set must be recreated to pick these up: pyzmq's CURVE "
            "authenticator reads the accepted-client directory once, when the socket is "
            "configured, so dropping a key in beside a running bridge changes nothing "
            "and the publisher stays refused.",
            paths["bridge"].parent,
            len(present),
            len(certs),
        )

    try:
        _generate_bluesky_curve_certs(paths)
    except Exception:
        logger.warning(
            "Could not generate the bluesky document-plane CURVE certificates under %s. "
            "The stack will still come up, but the bridge will refuse to bind its "
            "document proxy, so scans will report no live rows.",
            paths["bridge"].parent,
            exc_info=True,
        )
        return

    logger.key_info(
        "Generated bluesky document-plane CURVE certificates under %s (gitignored, like "
        "the project .env). Both containers read them at startup, so certificates written "
        "or replaced while the stack is running take effect only after the bridge and "
        "queueserver containers are recreated.",
        paths["bridge"].parent,
    )


def _generate_bluesky_curve_certs(paths: dict[str, Path]) -> None:
    """Mint both document-plane keypairs and place them per ``paths``.

    Certificates are minted into a scratch directory and then moved into
    place, so a failure part-way through cannot leave one side holding a key
    that no longer pairs with the other's.

    Args:
        paths: Role-to-path mapping from :func:`_bluesky_curve_paths`.
    """
    # Lazy import: pyzmq arrives with the core bluesky dependency, but keep it
    # off the import path of every non-bluesky deploy.
    from zmq.auth import create_certificates

    for key in ("bridge", "bridge_clients", "queueserver"):
        paths[key].mkdir(parents=True, exist_ok=True)
        os.chmod(paths[key], 0o700)

    with tempfile.TemporaryDirectory(dir=paths["bridge"].parent) as scratch:
        scratch_dir = Path(scratch)
        proxy_public, proxy_secret = create_certificates(scratch_dir, "proxy")
        publisher_public, publisher_secret = create_certificates(scratch_dir, "publisher")
        for source, target in (
            (proxy_secret, paths["proxy_secret"]),
            (publisher_secret, paths["publisher_secret"]),
            (proxy_public, paths["proxy_public"]),
            (publisher_public, paths["publisher_public"]),
        ):
            os.replace(source, target)

    # `create_certificates` writes world-readable files. The public halves are
    # not secrets and stay that way; the secret halves get .env's 0600.
    os.chmod(paths["proxy_secret"], 0o600)
    os.chmod(paths["publisher_secret"], 0o600)


def _ensure_bluesky_control_plane_keys(config: dict, env_path: Path | None = None) -> None:
    """Mint the RE Manager's control-socket CURVE keypair into the project ``.env``.

    The manager's 0MQ control socket is the one route to plan execution that
    does not pass the bridge's launch-token gate, and that gate is only
    meaningful if it is the sole way in. The socket publishes no port, but the
    queueserver container has to sit on ``osprey-network`` as well as the
    internal one, so key authentication — not network placement — is what
    keeps another container on that network from driving the manager directly.

    The two halves must pair, which is why this cannot ride on
    ``_ensure_service_tokens``: ``_VAR_GENERATORS``' recipes take no arguments
    and mint each variable independently, so there is no way to express "this
    value is derived from that one" — and two independently random keys are
    exactly a broken CURVE handshake. A pair-aware hook in that map would have
    to reach across variables mid-loop; a separate step is the smaller change,
    and it leaves that function's minting, ``--expose`` and validation
    semantics untouched.

    **Both halves are secrets.** The public key is not the usual
    "public keys are safe to publish" case: upstream ``bluesky-queueserver``
    clients authenticate with a fixed, publicly-known client keypair, so
    holding the manager's public key is by itself enough to reach its control
    socket. Both values therefore go into the 0600 ``.env`` and neither is
    ever logged — only the variable names are.

    Existing values win, in the same "process env, then ``.env``" precedence
    ``_ensure_service_tokens`` uses, with one repair: when only the private
    key is set, the public half is *derived* from it rather than minted, so an
    operator who supplied their own manager key gets a matching pair instead
    of a silent handshake failure. The reverse (a public key with no private
    half) cannot be repaired — a secret key is not recoverable from a public
    one — so it warns rather than fabricating a key that would not pair.

    **Plaintext is not a supported mode.** An explicitly empty private key is
    treated as unset, not as "encryption off": the compose template expands
    this variable with a ``:?`` guard, which fails on empty exactly as it does
    on unset, so an empty value cannot produce a running stack — only a
    confusing abort deep inside ``compose``. Refusing here turns that into an
    actionable message. The operator override story is to supply a valid pair
    of their own, never to switch encryption off.

    Raises into a deploy only for that one case — a deliberate empty value,
    which is unsatisfiable. Everything else (a malformed key, an unavailable
    ``zmq``) logs a warning and returns, leaving the ``:?`` guard to stop the
    deploy rather than this function.

    Args:
        config: Raw deploy config (``deployed_services`` membership).
        env_path: Project ``.env`` path; defaults to ``Path(".env")``.
    """
    services = {str(s) for s in (config.get("deployed_services") or [])}
    if "bluesky" not in services:
        return

    if env_path is None:
        env_path = Path(".env")
    existing = parse_dotenv_file(env_path) if env_path.is_file() else {}

    def _is_set(name: str) -> bool:
        """Present with a non-empty value.

        Unlike ``_ensure_service_tokens``, an empty value is NOT honoured as a
        deliberate "leave it alone" here: a plaintext control socket is not a
        supported mode for this stack, and the compose template's ``:?`` guard
        rejects empty exactly as it rejects unset.
        """
        return bool(_effective_value(name, existing).strip())

    # An explicitly empty value for EITHER half is unsatisfiable, so refuse it
    # here where the message can be actionable. Empty is not "encryption off":
    # for the private key the compose template's `:?` guard rejects it exactly
    # as it rejects unset, and an empty public half leaves the bridge unable to
    # authenticate to a manager that does have a key. Minting over either one
    # would contradict a value the operator deliberately set.
    for var in (_QSERVER_ZMQ_PRIVATE_KEY_VAR, _QSERVER_ZMQ_PUBLIC_KEY_VAR):
        if (var in os.environ or var in existing) and not _effective_value(var, existing).strip():
            raise RuntimeError(
                f"{var} is set but empty. An empty value would run the bluesky RE manager's "
                "control socket in plaintext, which is not a supported mode: the manager is "
                "the one route to plan execution that does not pass the bridge's launch-token "
                "gate, and its container shares a network with every other service. Either "
                f"unset {var} and let `osprey deploy up` mint a keypair, or set "
                f"{_QSERVER_ZMQ_PRIVATE_KEY_VAR} to the private half of a CURVE keypair of "
                "your own."
            )

    # A value containing `$` is unusable no matter how well-formed it is: the
    # project .env goes through docker compose's interpolation, which would
    # rewrite it on the way to the container (see
    # `_assert_env_interpolation_safe`). Checked BEFORE the pairing check
    # below, so an operator whose key has this problem hears about the real
    # cause rather than a mismatched-pair message about a value neither side
    # ever sees. Existing values are never overwritten, so refusing is the only
    # honest option here — minting over an operator's deliberate key is not.
    for var in (_QSERVER_ZMQ_PRIVATE_KEY_VAR, _QSERVER_ZMQ_PUBLIC_KEY_VAR):
        _assert_env_interpolation_safe(var, _effective_value(var, existing))

    has_private = _is_set(_QSERVER_ZMQ_PRIVATE_KEY_VAR)
    has_public = _is_set(_QSERVER_ZMQ_PUBLIC_KEY_VAR)
    if has_private and has_public:
        # Both present is the "operator supplied their own" path, and it is the
        # one shape the compose `:?` guards cannot check: two well-formed keys
        # that simply do not belong together satisfy every guard and then fail
        # at runtime as a control-socket timeout with nothing pointing at the
        # keys. Verify the pairing here, where the message can name the cause.
        _verify_qserver_keypair_pairs(
            _effective_value(_QSERVER_ZMQ_PRIVATE_KEY_VAR, existing),
            _effective_value(_QSERVER_ZMQ_PUBLIC_KEY_VAR, existing),
        )
        return
    if has_public and not has_private:
        logger.warning(
            "%s is set but %s is not. A CURVE public key cannot be turned back into its "
            "private half, and minting a fresh private key would silently orphan the "
            "public one you set, so nothing is generated here — the deploy will stop at "
            "the compose template's fail-closed guard on the private key rather than "
            "start the manager in plaintext. Set %s to the private half of that keypair, "
            "or unset %s to let `osprey deploy up` mint a fresh pair.",
            _QSERVER_ZMQ_PUBLIC_KEY_VAR,
            _QSERVER_ZMQ_PRIVATE_KEY_VAR,
            _QSERVER_ZMQ_PRIVATE_KEY_VAR,
            _QSERVER_ZMQ_PUBLIC_KEY_VAR,
        )
        return

    try:
        generated = _derive_qserver_keypair(
            _effective_value(_QSERVER_ZMQ_PRIVATE_KEY_VAR, existing)
        )
    except EnvInterpolationUnsafeError:
        # NOT swallowed like the failures below: the compose `:?` guard cannot
        # catch a value that is present-but-rewritten, so degrading this to a
        # warning would hand the operator a control-socket timeout with nothing
        # pointing at the key. See EnvInterpolationUnsafeError.
        raise
    except Exception:
        logger.warning(
            "Could not provision the bluesky RE manager's control-socket CURVE keypair "
            "(%s / %s). No key is written, so the compose template's fail-closed guard "
            "on the private key will stop the deploy rather than start the manager in "
            "plaintext.",
            _QSERVER_ZMQ_PRIVATE_KEY_VAR,
            _QSERVER_ZMQ_PUBLIC_KEY_VAR,
            exc_info=True,
        )
        return

    if has_private and _QSERVER_ZMQ_PRIVATE_KEY_VAR in os.environ:
        # The private half lives only in this shell, so writing its derived
        # public half to .env would leave a half-state on disk: the next deploy
        # from a clean shell would find a public key with no private one, hit
        # the orphan branch above, and abort at the compose guard. Export it for
        # this deploy instead and re-derive it every time — the pair then always
        # comes from the same source, and nothing on disk can go stale.
        os.environ.update(generated)
        logger.key_info(
            "Derived %s from the %s in this environment for this deploy only; not written "
            "to %s, since the private half is not there either and a lone public key on "
            "disk would fail the next deploy from a clean shell.",
            ", ".join(generated),
            _QSERVER_ZMQ_PRIVATE_KEY_VAR,
            env_path.resolve(),
        )
        return

    _append_env_block(
        env_path,
        "Auto-generated bluesky RE manager control-socket keypair (osprey deploy up)",
        generated,
    )
    # Names only, never values -- and that applies to BOTH halves, not just the
    # private one (see the function docstring on why the public key is bearer
    # material here). Matches _ensure_service_tokens' "log which keys, never
    # what they are" convention.
    logger.key_info(
        "Generated bluesky RE manager control-socket keypair %s in %s (gitignored). "
        "Treat both values as secrets — the public half alone is enough to reach the "
        "manager's control socket.",
        ", ".join(generated),
        env_path.resolve(),
    )


def _verify_qserver_keypair_pairs(private_key: str, public_key: str) -> None:
    """Refuse a configured keypair whose two halves do not belong together.

    Both halves being present is the operator-supplied path, and a mismatched
    pair is the one failure the compose ``:?`` guards cannot catch: two
    well-formed z85 keys satisfy every guard, so the deploy comes up and the
    bridge's every control-socket call then times out with nothing in any log
    pointing at the keys. Checking the pairing at deploy time turns that into
    a named cause.

    Args:
        private_key: The effective ``BLUESKY_QSERVER_ZMQ_PRIVATE_KEY``.
        public_key: The effective ``BLUESKY_QSERVER_ZMQ_PUBLIC_KEY``.

    Raises:
        RuntimeError: The public key is not the one derived from the private
            key. Names both variables and never echoes either value.
    """
    import zmq

    try:
        derived = zmq.curve_public(private_key.encode()).decode()
    except Exception:
        # Malformed private key: same policy as everywhere else here — warn and
        # leave the compose guard to stop the deploy. Nothing can be verified
        # against a key that will not parse, and raising would pre-empt the
        # clearer failure the operator is about to get from their own value.
        logger.warning(
            "Could not check that %s pairs with %s: the private key does not parse as a "
            "CURVE key. The deploy continues, but the manager will reject every "
            "control-socket call until that value is a valid z85 private key.",
            _QSERVER_ZMQ_PRIVATE_KEY_VAR,
            _QSERVER_ZMQ_PUBLIC_KEY_VAR,
            exc_info=True,
        )
        return

    if derived != public_key.strip():
        raise RuntimeError(
            f"{_QSERVER_ZMQ_PUBLIC_KEY_VAR} is not the public half of "
            f"{_QSERVER_ZMQ_PRIVATE_KEY_VAR}. Both values are well-formed, so every "
            "compose guard would pass and the stack would start — but the bridge could "
            "never complete a CURVE handshake with the RE manager, and every queue call "
            "would fail as an unexplained control-socket timeout. Set "
            f"{_QSERVER_ZMQ_PUBLIC_KEY_VAR} to the public half of that keypair, or unset "
            "both and let `osprey deploy up` mint a matched pair."
        )


# How many keypairs to draw before giving up on finding one free of `$`.
# `$` is one of Z85's 85 characters, so a single 40-character key carries one
# with probability 1 - (84/85)**40 ≈ 38%; a draw is a PAIR (private + public,
# 80 characters together) and is rejected if EITHER half does, i.e. ≈61% of the
# time. The chance of 40 consecutive rejections is therefore ≈3e-9 — the bound
# exists so a broken generator cannot spin forever, not because exhaustion is
# expected.
_QSERVER_KEYPAIR_MINT_ATTEMPTS = 40


class EnvInterpolationUnsafeError(RuntimeError):
    """A ``.env`` value docker compose would rewrite on its way to the container.

    Its own type, not a bare ``RuntimeError``, for one reason: the keypair
    provisioning path deliberately swallows exceptions into a warning and lets
    the compose ``:?`` guard stop the deploy. That is right for "zmq is missing"
    or "this key is malformed", where the guard's message is as good as any.
    It is wrong here — the guard cannot fire on a value that IS set, and the
    operator would be left with an opaque control-socket timeout — so this one
    is re-raised.
    """


def _assert_env_interpolation_safe(var: str, value: str) -> None:
    """Refuse a ``.env`` value docker compose would silently rewrite.

    Docker Compose INTERPOLATES the env-file it is given: a ``$`` followed by an
    identifier is expanded as a variable reference, and an unset variable
    expands to the empty string. So a secret containing ``$`` reaches the
    container SHORTER AND DIFFERENT than what was written, with nothing but a
    ``level=warning ... variable is not set`` line to say so. podman-compose
    mangles a different set (the braced ``${...}`` form only, resolved against
    the same file's earlier entries), which is a second reason to refuse ``$``
    rather than to model any one implementation.

    That is not hypothetical here. The RE manager's control-socket keys are
    Z85, whose alphabet includes ``$``: a real deploy wrote a valid 40-character
    public key and the bridge received 38 characters, failing with "the key must
    be a 40 byte z85 encoded string" — intermittently, because it depends on
    what the mint happened to draw.

    Every OTHER secret this module mints already avoids the problem by
    construction (``token_urlsafe``/``token_hex`` alphabets, and OpenObserve's
    recipe explicitly excludes ``$`` for exactly this family of reasons), so
    this guard exists for values whose alphabet is not ours to choose: an
    operator-supplied key, or a public half derived from one. Freshly minted
    pairs are re-drawn instead of refused — see `_derive_qserver_keypair`.

    Rejects ANY ``$``, not only the sequences compose currently expands: which
    forms are treated as references is compose's business and may widen, and a
    key is opaque random material, so there is no cost to being strict.

    Raises:
        RuntimeError: ``value`` contains ``$``.
    """
    if "$" not in value:
        return
    raise EnvInterpolationUnsafeError(
        f"{var} contains a '$', which docker compose would expand as a variable "
        "reference when it reads the project .env — the container would receive a "
        "truncated, invalid key and the bluesky RE manager's control socket would "
        "fail to authenticate, with only a 'variable is not set' warning to explain "
        "it. This value cannot be used as-is. Supply a CURVE keypair whose halves "
        f"contain no '$' (unset {var} to let `osprey deploy up` mint one), or set "
        f"{_QSERVER_ZMQ_PRIVATE_KEY_VAR} to a private key whose derived public half "
        "is also '$'-free."
    )


def _derive_qserver_keypair(private_key: str) -> dict[str, str]:
    """The control-plane variables to append, given any private key already set.

    Freshly minted pairs are drawn until BOTH halves are free of ``$`` (see
    `_assert_env_interpolation_safe` for why that character is disqualifying).
    Rejection sampling rather than escaping: writing ``$$`` into ``.env`` would
    be correct only for compose, and every other reader of that file — osprey's
    own ``parse_dotenv_file``, the "existing value wins" check on the next
    deploy, an operator — would see a literal ``$$`` and a different key. It
    costs well under one bit against a 256-bit secret.

    A public half DERIVED from an operator's private key cannot be re-drawn, so
    that path is checked and refused instead.

    Args:
        private_key: The effective ``BLUESKY_QSERVER_ZMQ_PRIVATE_KEY``, or an
            empty string when none is set. An empty value never reaches here —
            the caller refuses it outright, since plaintext is not a supported
            mode.

    Returns:
        The public half alone when a private key is already set (derived from
        it, so the operator's own manager key is kept); a freshly minted pair
        otherwise.

    Raises:
        RuntimeError: A derived public half contains ``$``, or no ``$``-free
            pair could be drawn within `_QSERVER_KEYPAIR_MINT_ATTEMPTS`.
    """
    import zmq

    if private_key.strip():
        public = zmq.curve_public(private_key.encode()).decode()
        # Not re-drawable: this half is determined by the operator's own key.
        _assert_env_interpolation_safe(_QSERVER_ZMQ_PUBLIC_KEY_VAR, public)
        return {_QSERVER_ZMQ_PUBLIC_KEY_VAR: public}

    for _ in range(_QSERVER_KEYPAIR_MINT_ATTEMPTS):
        public_bytes, secret_bytes = zmq.curve_keypair()
        public, secret = public_bytes.decode(), secret_bytes.decode()
        if "$" not in public and "$" not in secret:
            return {
                _QSERVER_ZMQ_PRIVATE_KEY_VAR: secret,
                _QSERVER_ZMQ_PUBLIC_KEY_VAR: public,
            }
    raise EnvInterpolationUnsafeError(
        f"could not mint a bluesky RE manager keypair free of '$' in "
        f"{_QSERVER_KEYPAIR_MINT_ATTEMPTS} attempts — every draw contained a character "
        "docker compose would expand out of the project .env. This should be "
        "vanishingly unlikely; suspect the key generator rather than bad luck."
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


def _resolve_pip_spec(dev_mode: bool = False) -> str:
    """The ``OSPREY_PIP_SPEC`` build arg for the project image.

    An operator ``OSPREY_PIP_SPEC`` export wins (e.g. a ``git+https`` URL that
    pins an unreleased build). Otherwise pin the released framework version
    (``osprey-framework==<version>``), matching the dispatch image's production
    install, so a project image built without ``--dev`` ships a deterministic
    release rather than tracking whatever ``osprey-framework`` resolves to at
    build time.

    Under ``dev_mode`` a locally-built wheel has been staged into the build
    context and the Dockerfile installs that instead, ignoring this spec — so the
    release check is skipped. It has to be: a development checkout is exactly
    where ``--dev`` is used, and refusing there would block the workflow this
    error recommends. The caller passes the *effective* dev mode, so a ``--dev``
    run whose wheel staging failed still gets the check and refuses rather than
    quietly installing released code.

    :param dev_mode: Whether a local wheel was staged for this build.
    :raises UnreleasedVersionPinError: if a PyPI install would be pinned to a
        version that was never published.
    """
    spec = os.environ.get("OSPREY_PIP_SPEC")
    if spec:
        return spec

    from osprey.deployment.errors import UnreleasedVersionPinError
    from osprey.version import get_release_version, is_release, unreleased_pin_reason

    if dev_mode:
        # The staged wheel is what gets installed; this value is inert.
        return f"osprey-framework=={get_release_version()}"

    if not is_release():
        raise UnreleasedVersionPinError(
            unreleased_pin_reason(),
            "Use `osprey deploy up --dev` to build and stage a wheel from this "
            "checkout, or set OSPREY_PIP_SPEC to pin explicitly.",
        )
    return f"osprey-framework=={get_release_version()}"


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


def _project_image_build_cmd(
    config: dict, runtime: str, project_root: str, dev_mode: bool = False
) -> list[str]:
    """Construct the ``<runtime> build`` argv that produces ``<project>:local``.

    Carries the same ``com.osprey.project`` label
    :func:`osprey.deployment.web_terminals.persona_images._persona_image_build_cmd`
    stamps on its persona images, so a later ``nuke`` can verify a tag belongs
    to this deployment before removing it. Under ``dev_mode`` an
    ``OSPREY_DEV=1`` build-arg is added (mirroring the persona dev path), so
    the Dockerfile's dev branch can key off it.

    :param config: Raw deploy config (project name, ``claude_code.cli_version``).
    :param runtime: Base container command (``docker`` or ``podman``).
    :param project_root: Build context — the project root that holds the
        rendered ``Dockerfile`` (and, under ``--dev``, the staged wheel).
    :param dev_mode: Whether ``--dev`` was passed (adds ``OSPREY_DEV=1``).
    :return: The full build command as an argv list.
    """
    project_name = resolve_project_name(config)
    dockerfile = os.path.join(project_root, "Dockerfile")
    cmd = [
        runtime,
        "build",
        "-t",
        f"{project_name}:local",
        "-f",
        dockerfile,
        "--label",
        f"com.osprey.project={project_name}",
        "--build-arg",
        f"CLAUDE_CLI_VERSION={_resolve_claude_cli_version(config)}",
        "--build-arg",
        f"OSPREY_PIP_SPEC={_resolve_pip_spec(dev_mode=dev_mode)}",
    ]
    if dev_mode:
        cmd.extend(["--build-arg", "OSPREY_DEV=1"])
    cmd.append(project_root)
    return cmd


def _warn_unignored_build_dir(project_root: str) -> None:
    """Warn if a ``build/`` dir would bloat the ``--dev`` build context.

    A rendered project accumulates a ``build/`` directory (compose files,
    service contexts). When it isn't excluded via the project's
    ``.dockerignore``, ``<runtime> build``'s context tar sweeps it up on every
    ``--dev`` build — slow, and a foot-gun a one-line ``.dockerignore`` entry
    fixes. Plain-text line check only (no docker introspection); a missing
    ``.dockerignore`` counts as not-matching.
    """
    if not (Path(project_root) / "build").exists():
        return
    dockerignore = Path(project_root) / ".dockerignore"
    lines = dockerignore.read_text(encoding="utf-8").splitlines() if dockerignore.is_file() else []
    if any(line.strip().rstrip("/") == "build" for line in lines):
        return
    logger.warning(
        "The project's build/ directory is not excluded from the --dev image "
        "build context (no matching entry in %s), so it will be sent to the "
        "container runtime on every build. Add a line 'build/' to .dockerignore, "
        "or re-render the project with `osprey build --force`.",
        dockerignore,
    )


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
    ``<project>-dispatch:local`` build (its compose ``build:`` block) is untouched.

    Under ``dev_mode`` a wheel is built from the local osprey checkout and staged
    into the build context (mirroring the dispatch image's ``--dev`` convention);
    the Dockerfile's wheel-drop branch then installs it so unreleased code is
    baked in. The ``OSPREY_DEV=1`` build-arg is passed only when that staging
    actually succeeded — a failed wheel build keeps the pinned install
    fail-loud instead of silently relaxing it to the latest published release.
    The staged wheel is removed afterward so it cannot poison a later
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

    # OSPREY_DEV=1 (the pin-relaxing build arg) is passed only when the wheel
    # was actually staged: on a failed build/staging the Dockerfile must keep
    # its fail-loud pinned install rather than silently falling back to the
    # latest published release under a flag that means "run my local code".
    staged_artifacts: list[Path] = []
    wheel_staged = False
    if dev_mode:
        _warn_unignored_build_dir(project_root)
        before = _staged_dev_artifact_paths(project_root)
        wheel_staged = bool(_copy_local_framework_for_override(project_root))
        staged_artifacts = sorted(_staged_dev_artifact_paths(project_root) - before)
        if not wheel_staged:
            logger.warning(
                "Dev-wheel staging failed for the project image build; building "
                "without OSPREY_DEV so the Dockerfile keeps its pinned "
                "osprey-framework install."
            )

    try:
        cmd = _project_image_build_cmd(config, runtime, project_root, dev_mode and wheel_staged)
        logger.key_info("Building dispatch worker project image %s:", project_image)
        logger.info("Running command:\n    %s", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
    finally:
        # Remove BOTH staged artifacts (wheel + requirements manifest) so
        # neither can poison a later non-dev build in this context.
        for artifact in staged_artifacts:
            try:
                artifact.unlink()
            except OSError:
                logger.warning("Could not remove staged dev artifact %s", artifact)


def _env_file_args() -> list[str]:
    """``["--env-file", ".env"]`` if ``.env`` exists in the cwd, else ``[]``.

    Shared by :func:`deploy_up`'s plain-services compose invocation and
    :func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`'s
    two invocations (``deploy_up`` passes the result in), so the "no .env"
    warning (and the fallback-to-defaults behavior it describes) is only
    defined once.
    """
    if Path(".env").exists():
        return ["--env-file", ".env"]
    logger.warning(
        "No .env file found - services will start with default/empty environment variables"
    )
    logger.info("To configure API keys: cp .env.example .env && edit .env")
    return []


def _check_shared_disk_preflight(config: dict) -> None:
    """Abort before any compose invocation if a configured shared-disk host path is missing.

    Ports the retired ``deploy.sh`` step-2b check (see the ``osprey-build-deploy``
    skill's now-removed ``templates/core/scripts/deploy.sh``): a container that
    bind-mounts a host path that doesn't exist still starts, then fails only at
    first read/write with an obscure in-container error. Checking on the host,
    before ``compose`` ever runs, turns that into an immediate, actionable
    deploy-time error instead of a confusing runtime one.

    Skipped entirely when ``modules.shared_disk`` is absent/disabled, or when
    ``host_path`` isn't configured — there's nothing to check in either case,
    mirroring the shell version's ``IF MODULE shared_disk.enabled`` guard.

    :param config: Raw deploy config.
    :raises RuntimeError: if ``modules.shared_disk.enabled`` is set and
        ``host_path`` is configured but does not exist (or isn't a directory)
        on this host.
    """
    shared_disk = (config.get("modules") or {}).get("shared_disk") or {}
    if not shared_disk.get("enabled"):
        return

    host_path = shared_disk.get("host_path")
    if not host_path:
        return

    if not Path(host_path).is_dir():
        raise RuntimeError(
            f"modules.shared_disk.host_path does not exist on this server: {host_path}\n"
            "Mount the filesystem (check /etc/fstab) or correct modules.shared_disk.host_path."
        )


def _web_terminals_enabled(config: dict) -> bool:
    """True if ``modules.web_terminals.enabled`` is set on ``config``.

    Same read as :func:`osprey.deployment.web_terminals.lint.lint_web_terminals`'s
    own enabled-gate — the one place ``deploy_up`` decides whether a
    web-terminal reconcile is part of this deploy. Coerces a present-but-null
    ``modules`` or ``modules.web_terminals`` stanza (e.g. a bare ``web_terminals:``
    key in YAML, which parses to ``None``) to an empty dict rather than letting
    ``.get`` on ``None`` raise ``AttributeError`` — mirroring lint's own
    ``_as_dict`` coercion, which treats that same null stanza as disabled.
    """
    modules = config.get("modules") or {}
    web_terminals = modules.get("web_terminals") or {}
    return bool(web_terminals.get("enabled"))


def _preflight_host_ports(config, compose_files):
    """Abort the deploy if a published host port is already taken.

    Parses the published bindings out of the rendered compose files and checks
    them for intra-deploy duplicates and external listeners (see
    :mod:`osprey.deployment.host_ports`). On any conflict the report is logged
    and a :class:`RuntimeError` is raised so the caller aborts before running
    a single container-touching command.

    :param config: Loaded configuration dictionary
    :type config: dict
    :param compose_files: Rendered compose file paths for this deploy
    :type compose_files: list[str]
    :raises RuntimeError: If any host-port conflict is found
    """
    bindings = parse_host_port_bindings(compose_files)
    conflicts = find_port_conflicts(bindings, resolve_project_name(config), config)
    if not conflicts:
        return
    logger.error(format_conflict_report(conflicts))
    raise RuntimeError(
        f"host port preflight failed: {len(conflicts)} "
        f"conflict{'' if len(conflicts) == 1 else 's'} (see report above)"
    )


def deploy_up(config_path, detached=False, dev_mode=False, expose_network=False):
    """Start services using container runtime (Docker or Podman).

    When ``modules.web_terminals.enabled`` is set, the web-terminal stack is
    reconciled too (rendering its artifacts and including
    ``docker-compose.web.yml`` in the compose invocation) — see
    :func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`.
    That reconcile always runs detached,
    independent of ``detached``, and takes over from the plain services path
    below.

    Idempotent from any prior state: every path first clears this project's
    own non-running containers (``compose rm -f`` — a wedged ``created``
    container from an aborted deploy holds its published host ports on Docker
    Desktop, blocking the next ``up``), and the plain path's ``up`` carries
    ``--remove-orphans`` to reconcile away services dropped from the config.
    Running containers and volumes are never touched by either measure.

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

    # Advisory staleness check BEFORE anything deploys: a project rendered by
    # an older framework/preset self-describes an out-of-date service set in
    # config.yml, so the deploy below would "succeed" at the wrong goal with
    # no error anywhere. Never blocks (see warn_if_project_stale).
    warn_if_project_stale(Path(config_path).resolve().parent)

    web_terminals_enabled = _web_terminals_enabled(config)

    # A web-terminals-only deploy (no backend services) is valid, so the
    # early-return below must not fire on empty deployed_services in that case.
    if not config.get("deployed_services") and not web_terminals_enabled:
        logger.key_info(
            "No services configured for this project — deployed_services is empty in "
            "config.yml. Skipping osprey deploy up."
        )
        # Still say what is (not) reachable — an unexpectedly empty deploy is
        # the classic stale-render shape, and "web terminal (not configured)"
        # is the line that makes it diagnosable.
        log_endpoint_summary(config, compose_files)
        return

    # Shared-disk host path (if configured) must exist before either path
    # below reaches a compose invocation -- see _check_shared_disk_preflight.
    # Runs once, here, so it covers both the web-terminals branch and the
    # plain services branch further down rather than duplicating the check
    # in each.
    _check_shared_disk_preflight(config)

    # Verify container runtime is actually running
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Fail fast on a host-port collision (a foreign stack, or a second project
    # on the same host) with an actionable diagnosis, rather than letting
    # `compose up` collapse mid-start on a bare "address already in use". Runs
    # before any container-touching command below, so an abort here leaves the
    # host untouched. A port held by this project's own container is not a
    # conflict, so an idempotent redeploy stays green.
    _preflight_host_ports(config, compose_files)

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

    # Provision the bluesky scan stack's 0MQ key material before compose mounts
    # it: the RE manager's control-socket keypair into .env, and the document
    # plane's CURVE certificates into data/bluesky_curve/. Both are no-ops
    # without the bluesky service, and both are additive (existing material
    # always wins) so a redeploy never rotates keys under a running stack.
    # Neither is gated on the connector -- see
    # _ensure_bluesky_document_plane_certs for why a browse-only deploy still
    # gets its certificates.
    _ensure_bluesky_control_plane_keys(config)
    _ensure_bluesky_document_plane_certs(config)

    # Set up environment for containers
    env = os.environ.copy()
    if dev_mode:
        env["DEV_MODE"] = "true"
        logger.key_info("Development mode: DEV_MODE environment variable set for containers")

    # Fail-fast web-terminal preflight (persona render + credential gate)
    # BEFORE the minutes-long image build below: a deploy that is doomed to
    # abort on a missing provider secret must say so in seconds, not after
    # the whole project image has been built. deploy_up_web_terminals re-runs
    # the same (idempotent) steps later, unchanged. Nothing needs to survive
    # from here to the `up`: whatever preflight (or a hand-edit) did to
    # `.env.auth` is digested into the sidecar's rendered service definition
    # by deploy_up_web_terminals' own re-render, and compose recreates on the
    # definition change.
    if web_terminals_enabled:
        preflight_web_terminals(config, env)

    # Build the <project>:local image the dispatch worker references. The worker
    # has no compose build block (that would race the event-dispatcher on the
    # shared tag), so this is the only thing that produces its image. No-op
    # unless the worker is deployed on the local project image. Run before
    # `compose up` (which, non-detached, os.execvpe-replaces this process).
    _build_project_image(config, dev_mode, env)

    if web_terminals_enabled:
        deploy_up_web_terminals(config, compose_files, dev_mode, env, _env_file_args())
        log_endpoint_summary(config, compose_files)
        return

    # Pin COMPOSE_PROJECT_NAME so this deploy owns its own compose project (and
    # volume namespace); without it compose derives the project from the first
    # -f file's directory, collapsing every deploy on the host into the shared
    # "services" project whose up/down cross-adopts sibling stacks.
    run_env = runtime_env(config, env)

    base_cmd = get_runtime_command(config)
    for compose_file in compose_files:
        base_cmd.extend(("-f", compose_file))
    base_cmd.extend(_env_file_args())

    # Self-heal before reconciling: an aborted prior deploy can leave this
    # project's containers wedged in created/exited state, and Docker Desktop
    # reserves published host ports at container CREATE time — so a stale
    # created container blocks the next `up` on its own port ("address already
    # in use" with nothing listening). `rm -f` removes only non-running
    # containers (running ones are untouched, and it exits 0 as a no-op when
    # there is nothing stopped), so a healthy stack's reconcile stays
    # zero-churn. Volumes are never touched — destroying state stays the job
    # of clean/rebuild. Best-effort: if it fails, `up` surfaces the real error.
    rm_cmd = base_cmd + ["rm", "-f"]
    logger.info(f"Running command:\n    {' '.join(rm_cmd)}")
    subprocess.run(rm_cmd, env=run_env)

    if dev_mode:
        # `osprey deploy up --dev` re-bakes the local osprey checkout into a fresh
        # wheel on every run, but compose reuses the cached image tag (e.g.
        # <project>-dispatch:local) unless it is rebuilt — so a dev deploy must build.
        # Build in its OWN step, then `up --no-build`: a single `up --build` can
        # build a local-only tag and then fail container-create with
        # "No such image" under Docker's containerd image store. Non-dev stays a
        # plain `up` so compose's implicit build-on-up still covers a build-only
        # service that has no published upstream tag to pull.
        build_cmd = base_cmd + ["build"]
        logger.info(f"Running command:\n    {' '.join(build_cmd)}")
        subprocess.run(build_cmd, env=run_env, check=True)

    # --remove-orphans reconciles away containers whose service left the
    # config since the last deploy (including a formerly-enabled web-terminal
    # stack). Safe here because this single invocation's -f list defines the
    # ENTIRE compose project; the web path must never use it — its two
    # invocations share one project name, so each would destroy the other
    # stack's containers as "orphans".
    cmd = base_cmd + ["up", "--remove-orphans"]
    if dev_mode:
        cmd.append("--no-build")
    if detached:
        cmd.append("-d")

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    if detached:
        subprocess.run(cmd, env=run_env, check=True)
        log_endpoint_summary(config, compose_files)
    else:
        # execvpe replaces this process, so the summary must print first —
        # compose's own output follows it.
        log_endpoint_summary(config, compose_files)
        os.execvpe(cmd[0], cmd, run_env)


def deploy_down(config_path, dev_mode=False):
    """Stop services using container runtime (Docker or Podman).

    When ``modules.web_terminals.enabled`` is set, the web-terminal stack is
    torn down first via its own compose invocation
    (:func:`osprey.deployment.web_terminals.provision.deploy_down_web_terminals`)
    — the services ``-f`` list below can never carry
    ``docker-compose.web.yml`` (its relative paths are project-root-relative,
    the services files' resolve against ``build/services/``), and the
    services ``down`` execvpe-replaces this process, so the web ``down``
    must happen before it.

    :param config_path: Path to the configuration file
    :type config_path: str
    """
    try:
        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(config_path)
            config = normalize_facility_config(config.raw_config)
    except Exception as e:
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e

    if _web_terminals_enabled(config):
        env_file_args = ["--env-file", ".env"] if Path(".env").exists() else []
        deploy_down_web_terminals(config, os.environ.copy(), env_file_args)

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
    # execvpe (not execvp) so the COMPOSE_PROJECT_NAME pin reaches compose:
    # `down` must target the same project `up` created, or it either misses this
    # deploy's containers or (unpinned) tears down the shared "services" project.
    os.execvpe(cmd[0], cmd, runtime_env(config, os.environ.copy()))


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

    # Same reason the token mint above runs on this path: the bluesky compose
    # template expands the manager's private key with a `:?` guard, so an unset
    # value aborts the whole `compose restart` rather than degrading. A project
    # whose .env predates this feature has no key, and restarting it must not
    # be the thing that breaks it. The document-plane certificates are NOT
    # provisioned here -- they are bind-mount sources, and `compose restart`
    # reuses each container's existing config rather than re-resolving mounts,
    # so generating them would change nothing until the next `deploy up`.
    _ensure_bluesky_control_plane_keys(config)

    cmd = get_runtime_command(config)
    for compose_file in compose_files:
        cmd.extend(("-f", compose_file))
    cmd.extend(["--env-file", ".env", "restart"])

    logger.info(f"Running command:\n    {' '.join(cmd)}")
    subprocess.run(cmd, env=runtime_env(config, os.environ.copy()))

    # If detached mode requested, detach after restart
    if detached:
        logger.info("Services restarted. Running in detached mode.")


def rebuild_deployment(config_path, detached=False, dev_mode=False, expose_network=False):
    """Rebuild deployment from scratch (clean + up).

    Tears down this project's containers, volumes, and images via
    :func:`clean_deployment`, then delegates the start-up to :func:`deploy_up`
    so every up-path behavior — the web-terminals branch, the dev-mode
    build/up split, the stale-container preflight — stays defined in exactly
    one place. ``clean``'s ``down --rmi all`` removes the images, so the
    delegated ``up`` rebuilds/pulls everything fresh via compose's own
    build-on-up, no explicit ``build`` step needed here. The web-terminal
    stack's per-user volumes are declared only in ``docker-compose.web.yml``
    (never in the services compose files ``clean`` operates on), so a rebuild
    recreates web containers but preserves user volumes.

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

    # Verify container runtime is actually running (for the clean phase;
    # deploy_up re-verifies for its own).
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    clean_deployment(compose_files, config)

    deploy_up(config_path, detached=detached, dev_mode=dev_mode, expose_network=expose_network)
