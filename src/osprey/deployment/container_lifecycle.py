"""Container lifecycle operations — start, stop, restart, and rebuild.

Manages the lifecycle of containerized service deployments using
Docker or Podman compose.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NamedTuple

import yaml

from osprey.cli.phase_reporter import current_reporter
from osprey.cli.phase_reporter import report_step as _report_step
from osprey.deployment.build_progress import BuildModel, with_plain_build_progress
from osprey.deployment.compose_generator import (
    COMPOSE_ENV_FILENAME,
    REPO_ID_LABEL,
    _copy_local_framework_for_override,
    clean_deployment,
    compose_base_cmd,
    prepare_compose_files,
    repo_identity,
    resolve_project_name,
    resolve_repo_root,
)
from osprey.deployment.deploy_summary import log_endpoint_summary
from osprey.deployment.errors import (
    ComposeInterpolationError,
    DevModeUnavailableError,
    NoRenderedBuildError,
)
from osprey.deployment.host_ports import (
    find_port_conflicts,
    format_conflict_report,
    parse_host_port_bindings,
)
from osprey.deployment.runtime_helper import (
    get_runtime_command,
    runtime_env,
    verify_runtime_is_running,
    with_plain_progress,
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
from osprey.deployment.staleness import BUILD_DIRNAME, warn_if_project_stale
from osprey.deployment.subprocess_capture import run_captured
from osprey.deployment.web_terminals.provision import (
    deploy_down_web_terminals,
    deploy_up_web_terminals,
    preflight_web_terminals,
)
from osprey.deployment.wheel_build import _staged_dev_artifact_paths
from osprey.utils.config import config_anchored_at, load_project_config
from osprey.utils.dotenv import (
    compose_unsafe_vars,
    parse_dotenv_file,
)
from osprey.utils.logger import get_logger
from osprey.utils.workspace import container_image_context

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
#
# mongodb is the archiver store, and follows the same shape one step further:
# its compose template carries an insecure ``${MONGO_ROOT_PASSWORD:-osprey}``
# default, and the minted value has to reach THREE readers, not two — the
# container (MONGO_INITDB_ROOT_PASSWORD), the archiver_recorder service writing
# samples in-network, and the agent's own connector, whose config block names
# the variable rather than carrying a value
# (``archiver.mongodb_archiver.password_env: MONGO_ROOT_PASSWORD``). All three
# resolve the same ``.env`` entry, which is what keeps one store openable by
# the process that writes it and the process that reads it. The seeder and
# ``osprey sim apply`` read it from the project ``.env`` explicitly rather than
# from an ambient environment.
# NOTE: mongod, like postgres, reads its root credentials only when
# initializing a fresh data volume — the same stale-volume caveat applies.
_SERVICE_TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "event_dispatcher": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "dispatch_worker": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "bluesky": ("BLUESKY_LAUNCH_TOKEN", "BLUESKY_TILED_API_KEY"),
    "openobserve": ("ZO_ROOT_USER_PASSWORD",),
    "postgresql": ("ARIEL_DB_PASSWORD",),
    "mongodb": ("MONGO_ROOT_PASSWORD",),
}

# Vars checked against their _VAR_VALIDATORS constraint when present, but
# NEVER minted — distinct from _SERVICE_TOKEN_VARS (which mints an unset var
# for a deployed service) and from a bare _VAR_VALIDATORS entry alone (which,
# by itself, only fires for a var some deployed service's required_vars
# already pulled in). A var earns a place here when it carries a registered
# format constraint but no osprey-native service in this deploy system's
# world (_SERVICE_TOKEN_VARS / find_service_config's compose templates) ever
# requires it: ARIEL_DSN names an ARIEL Postgres this deploy does not run. A
# deploy that runs its own gets the URI from `resolve_ariel_dsn`, derived from
# `services.postgresql` and the minted ARIEL_DB_PASSWORD, so the var is only
# ever set by an operator pointing at a database provisioned elsewhere — not
# minted by this deploy system's token path, so no _SERVICE_TOKEN_VARS entry
# will ever declare it. This is defense-in-depth, not enforcement: if an
# operator or other tooling nonetheless places ARIEL_DSN into *this*
# project's effective env, it is validated like any other var — never
# fabricated when absent, never auto-minted when malformed, just rejected
# with a named-var/no-value error.
_VALIDATE_ONLY_VARS: set[str] = {"ARIEL_DSN"}


class VolumeInitializedStore(NamedTuple):
    """Where one store's identity lives on the host, for the stale-volume check.

    Four names that must agree with the service's compose template, which
    ``TestRegistryAgreement`` in the per-store mint tests pins field by field:

    * ``service`` — the ``deployed_services`` key, and the name shown to the
      operator;
    * ``volume`` — the *bare* volume name the template declares. Compose
      namespaces it with the project name on the host, so the host-side name is
      ``<project>_<volume>``;
    * ``container`` — the suffix the template appends to the project name in
      ``container_name``. The container is the only host-side record of the
      credential its volume was initialized with;
    * ``cred_env`` — what that credential is called *inside* the container,
      which is not the ``.env`` var name for two of the three stores
      (``MONGO_ROOT_PASSWORD`` arrives as ``MONGO_INITDB_ROOT_PASSWORD``,
      ``ARIEL_DB_PASSWORD`` as ``POSTGRES_PASSWORD``).
    """

    service: str
    volume: str
    container: str
    cred_env: str


#: Minted vars a service reads ONLY when it initializes a fresh data volume.
#: For every other token, minting a new value and restarting is enough; for
#: these the container keeps whatever it was born with, so a fresh mint
#: beside a surviving volume produces a credential mismatch that shows up as an
#: authentication failure at start rather than as anything about ``.env``.
#:
#: The dangerous case is exactly "minted, not found": an operator who deleted
#: ``.env`` (or just its minted section) while the volumes lived. Every var here
#: is interpolated with a ``:-`` DEFAULT in its compose template, so none
#: carries a ``:?`` guard that would abort the deploy and say so — which is why
#: ``_preflight_stale_store_volumes`` is the only thing standing between that
#: operator and a login that will not work for reasons nothing has named.
_VOLUME_INITIALIZED_VARS: dict[str, VolumeInitializedStore] = {
    "ZO_ROOT_USER_PASSWORD": VolumeInitializedStore(
        service="openobserve",
        volume="openobserve_data",
        container="-openobserve",
        cred_env="ZO_ROOT_USER_PASSWORD",
    ),
    "ARIEL_DB_PASSWORD": VolumeInitializedStore(
        service="postgresql",
        volume="ariel_postgres_data",
        container="-ariel-postgres",
        cred_env="POSTGRES_PASSWORD",
    ),
    "MONGO_ROOT_PASSWORD": VolumeInitializedStore(
        service="mongodb",
        volume="archiver_mongodb_data",
        container="-archiver-mongodb",
        cred_env="MONGO_INITDB_ROOT_PASSWORD",
    ),
}

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


def _ensure_service_tokens(
    config: dict,
    expose_network: bool,
    env_path: Path | None = None,
) -> set[str]:
    """Self-provision required fail-closed service tokens into ``env_path``.

    :returns: The ``_VOLUME_INITIALIZED_VARS`` names this call actually minted —
        the subset whose brand-new value a pre-existing data volume would
        ignore. Empty on the common paths (nothing minted, or nothing minted
        that a volume adopts), and the input to
        ``_preflight_stale_store_volumes``.

    ``osprey up`` passes the deployment repo's own ``.env`` explicitly; a caller
    that passes nothing falls back to a cwd-relative ``.env`` (see the
    assignment below). Everything the rest of this docstring says about "the
    ``.env``" is about the file the caller named.

    For any token var (per ``_SERVICE_TOKEN_VARS``, keyed by the deployed
    services present) unset in BOTH the process env and that ``.env``,
    generate a strong random value (``_generate_token``: ``token_urlsafe(32)``
    unless the var registers a different alphabet in ``_VAR_GENERATORS``) and
    append it to ``.env``
    (``chmod 0o600``, matching the build-time convention). Existing values are
    never overwritten, so re-running ``osprey up`` is idempotent. No-op unless
    a token-requiring service is actually deployed.

    A token that is *present but explicitly empty* (e.g. ``TOKEN=`` exported in
    the shell) is left untouched — generating would silently override a
    deliberate value. For a loopback deploy the server simply fails closed; for
    a deployment the build rendered as reachable off-host (a wildcard bind, or
    the host-networked web-terminal stack) we refuse rather than bind a
    fail-open-at-bind server to all interfaces.

    Minting is unconditional for every var a deployed service declares: these
    tokens authenticate *network callers* to a service's own HTTP boundary and
    are not a hardware-safety layer. Whether a scan or a write is permitted is
    decided at the connector (``writes_enabled`` plus the per-put channel
    limits), which every write path — agent-side and bridge-side alike — must
    still clear. No deploy-time value is read here for safety semantics.

    A mint lands once and stays there. Nothing is copied anywhere afterwards:
    the file written here IS the deployment's one secret store, sitting at the
    repo root beside the ``profile.yml`` that describes the stack, and it is the
    same file compose is pointed at (``--env-file <repo>/.env``). A second copy
    owned by the profile must never be written: two stores drift, and a rebuild
    reading the wrong one mints secrets the running containers reject. One
    directory, one file, no second copy to keep in step.

    Independently of the above, every var in ``_VALIDATE_ONLY_VARS``
    (e.g. ``ARIEL_DSN``) is checked against its ``_VAR_VALIDATORS`` constraint
    when present in the effective env — but never minted, and never required:
    this runs even when no deployed service pulls in any ``_SERVICE_TOKEN_VARS``
    entry at all, since a validate-only var's presence does not depend on
    ``deployed_services`` membership.
    """
    minted_volume_initialized: set[str] = set()
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
    #     `env_file: ./.env` in its compose template — resolved against the
    #     pinned compose project directory, which IS the repo root, so it is
    #     this same file and not a copy inside the render.
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
    # tokens appended to a .env the deploy then refuses to use, so the operator
    # fixes the offending value and the next run mints a second set beside the
    # first. Nothing is lost by checking early: every minted value is `$`-free
    # by construction (see the alphabets in service_tokens), which the generator
    # self-rejection test pins.
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

            # A mint is the moment the volume-initialized vars become a hazard:
            # the value is new, and any volume that already exists will keep
            # ignoring it. Report them upward rather than warning here — whether
            # this is the ordinary first deploy (no volume yet, nothing wrong)
            # or a mismatch that cannot work is not a guess: it is one
            # label-filtered `volume ls` away, which is what
            # _preflight_stale_store_volumes asks before anything starts.
            minted_volume_initialized |= {
                name for name in generated if name in _VOLUME_INITIALIZED_VARS
            }

    # Validate the effective value of every required var — whichever of
    # process env, an existing .env, or a value just minted above the caller
    # actually sees — against its registered _VAR_VALIDATORS constraint (if
    # any). Unconditional: runs on every deploy path, not only for a deployment
    # that is reachable off-host, so a malformed operator-supplied value is
    # caught on the default loopback deploy too.
    # Re-parse .env since the mint step above may have just appended to it.
    post = parse_dotenv_file(env_path) if env_path.is_file() else {}
    for name in required_vars:
        effective = _effective_value(name, post)
        if expose_network and not effective.strip():
            raise RuntimeError(
                f"{name} is empty; refusing to start a deployment that is reachable "
                f"off-host with an empty token. Set {name} in .env to a strong secret."
            )
        if effective and not _validate_var(name, effective):
            _raise_invalid_var(name, effective)

    # Validate-only vars (ARIEL_DSN): checked when present in the same
    # effective-value sense as required_vars above, but never minted when
    # absent and never required to unblock a deploy — see _VALIDATE_ONLY_VARS'
    # docstring for why no _SERVICE_TOKEN_VARS entry can express this.
    for name in _VALIDATE_ONLY_VARS:
        effective = _effective_value(name, post)
        if not effective:
            continue  # absent — never fabricated, never minted
        if not _validate_var(name, effective):
            _raise_invalid_var(name, effective)

    return minted_volume_initialized


def _volume_initialized_vars_that_would_be_minted(config: dict, env_path: Path) -> set[str]:
    """Which store credentials a mint on this config *would* generate.

    The same rule ``_ensure_service_tokens`` mints by — absent from both the
    process env and the ``.env`` — evaluated without writing anything. It exists
    for ``restart``, which has to run the stale-volume check BEFORE its ``down``:
    the ``down`` removes the store containers, and with them the only host-side
    copy of the credential each surviving volume was initialized with. Asking
    after the stop would find every volume unrecoverable, having just made it so.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    services = {str(s) for s in (config.get("deployed_services") or [])}
    on_disk = parse_dotenv_file(env_path) if env_path.is_file() else {}
    return {
        var
        for var, store in _VOLUME_INITIALIZED_VARS.items()
        if store.service in services and var not in os.environ and var not in on_disk
    }


def _harvest_original_credential(probe, project: str, store: VolumeInitializedStore) -> str | None:
    """The credential *store*'s volume was initialized with, if still readable.

    ``None`` whenever the container is gone, carries no such variable, or holds
    an empty one — every case in which the volume cannot be reopened and the
    only way forward is to discard it.
    """
    env = probe.env_of_container(f"{project}{store.container}")
    if not env:
        return None
    return env.get(store.cred_env) or None


def _stale_store_volumes(
    probe,
    project: str,
    minted: set[str],
    services: set[str],
    env_path: Path,
) -> tuple[list[tuple[str, VolumeInitializedStore]], dict[str, str | None]]:
    """Stores whose surviving data volume will reject the credential in ``.env``.

    Read-only, and asked of the runtime rather than assumed: a credential beside
    no volume is the ordinary first deploy, and one beside a surviving volume
    may be unusable. Only the runtime knows which.

    A store qualifies on either of two independent grounds, because neither
    alone covers the ground the other does:

    * *this run minted the credential* — the value is brand new and a volume
      that already exists never adopted it. The only rule that can speak for a
      store whose container is gone, since there is then nothing to compare
      against;
    * *the store's own container disagrees with ``.env``* — a running container
      holds the credential the volume actually has, so a ``.env`` that differs
      cannot authenticate no matter who wrote it or how long ago. This is the
      rule that recognises a deployment already in the broken state, where a
      previous run's mint is sitting in ``.env`` and nothing new is minted.

    Deliberately NOT a third ground: a surviving volume with no container to
    compare against and nothing minted. That is simply a stopped deployment —
    the normal way one sits between sessions — and refusing it would block
    every start.

    Returns the qualifying ``(env_var, store)`` pairs in
    ``_VOLUME_INITIALIZED_VARS`` order (so a multi-store report reads the same
    way every time), together with each one's harvested original credential —
    ``None`` where the volume can no longer be reopened.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    try:
        existing = {resource.name for resource in probe.volumes_for_project(project)}
    except RuntimeError as exc:
        # A runtime that cannot answer must not become a deploy-blocking verdict
        # of its own: fall through to the start, where the store's own
        # authentication failure still reports the mismatch the old way.
        logger.warning("Could not check this deployment's data volumes: %s", exc)
        return [], {}

    on_disk = parse_dotenv_file(env_path) if env_path.is_file() else {}
    stale: list[tuple[str, VolumeInitializedStore]] = []
    originals: dict[str, str | None] = {}

    for var, store in _VOLUME_INITIALIZED_VARS.items():
        if store.service not in services:
            continue
        if f"{project}_{store.volume}" not in existing:
            continue

        original = _harvest_original_credential(probe, project, store)
        # The value compose will actually substitute: a shell export outranks
        # --env-file, the same precedence _ensure_service_tokens mints by and
        # _preflight_env_shadowing warns about.
        effective = _effective_value(var, on_disk)
        disagrees = original is not None and bool(effective) and original != effective

        if var in minted or disagrees:
            stale.append((var, store))
            originals[var] = original

    return stale, originals


def _preflight_stale_store_volumes(
    config: dict,
    minted: set[str],
    env_path: Path,
    *,
    reuse_stores: bool = False,
) -> None:
    """Abort before anything starts when a fresh credential meets an old volume.

    A refusal rather than a warning, and here rather than at the store's own
    health probe, for two reasons that both come down to what ``compose up``
    does on its way past:

    * it recreates the store container — and that container's environment is
      the only host-side record of the credential its volume was initialized
      with. Starting the stack overwrites it with the value that cannot work,
      turning a recoverable mismatch into a permanently orphaned volume;
    * it builds the project image first, so the store's own "Authentication
      failed" arrives minutes later and names only whichever store was probed
      first, leaving the rest to be found one restart at a time.

    With *reuse_stores*, adopt the volumes instead: restore each store's
    original credential to ``env_path`` in place of the value just minted. That
    is only possible while the store's container survives, so a run that cannot
    reopen every stale volume still refuses rather than starting a stack that
    is half-adopted and half-doomed.

    :raises RuntimeError: if any deployed store has a surviving volume that will
        reject the credential in ``env_path``, and this run is not able (or not
        asked) to adopt it.
    """
    services = {str(s) for s in (config.get("deployed_services") or [])}
    if not any(store.service in services for store in _VOLUME_INITIALIZED_VARS.values()):
        return  # the fast path: no volume-initialized store is deployed at all

    # Imported here, not at module scope: reset.py imports THIS module, so a
    # top-level import would close a cycle. RuntimeProbe is the codebase's one
    # container-runtime conversation seam — every listing label-filtered, every
    # removal naming exactly one resource — and nothing here removes anything.
    from osprey.deployment.reset import RuntimeProbe

    project = resolve_project_name(config)
    probe = RuntimeProbe(get_runtime_command(config)[0], run=subprocess.run)

    stale, originals = _stale_store_volumes(probe, project, minted, services, env_path)
    if not stale:
        return

    if reuse_stores and all(originals.values()):
        _adopt_original_credentials(env_path, stale, originals)
        return

    raise RuntimeError(
        _stale_store_report(project, env_path, stale, originals, reuse_stores=reuse_stores)
    )


def _adopt_original_credentials(
    env_path: Path,
    stale: list[tuple[str, VolumeInitializedStore]],
    originals: dict[str, str | None],
) -> None:
    """Put each store's original credential into ``.env``, replacing any mint.

    Both shapes occur, because the two start paths reach here at opposite sides
    of the mint. ``up`` has already minted, so the file carries a fresh
    assignment that must be *rewritten in place* — appending instead would leave
    two assignments of one name, and which wins would depend on the reader
    (compose takes the last; a human takes the first they see). ``restart``
    checks before its ``down`` and therefore before any mint, so the file may
    not carry the name at all and the value has to be appended.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    on_disk = parse_dotenv_file(env_path) if env_path.is_file() else {}

    absent: dict[str, str] = {}
    for var, _store in stale:
        original = originals[var]
        if original is None:  # unreachable via the caller's `all(...)` guard
            continue
        if var in on_disk:
            text = re.sub(
                rf"^{re.escape(var)}=.*$",
                f"{var}={original}",
                text,
                flags=re.MULTILINE,
            )
        else:
            absent[var] = original
    env_path.write_text(text, encoding="utf-8")
    if absent:
        _append_env_block(
            env_path,
            "Credentials adopted from pre-existing data volumes (osprey --reuse-stores)",
            absent,
        )

    written = parse_dotenv_file(env_path)
    missed = [var for var, _ in stale if written.get(var) != originals[var]]
    if missed:
        raise RuntimeError(
            f"Could not restore {', '.join(sorted(missed))} in {env_path.resolve()}. "
            "Edit the file by hand, or re-run without --reuse-stores."
        )

    logger.key_info(
        "Adopted %d pre-existing data volume(s): restored the credential(s) %s they were "
        "initialized with into %s. Values are never printed.",
        len(stale),
        ", ".join(var for var, _ in stale),
        env_path.resolve(),
    )


def _stale_store_report(
    project: str,
    env_path: Path,
    stale: list[tuple[str, VolumeInitializedStore]],
    originals: dict[str, str | None],
    *,
    reuse_stores: bool,
) -> str:
    """The refusal text: what is stale, and which way out each store leaves open.

    Names variables and volumes, never values — the same rule every other
    secret-touching line on this path follows.
    """
    lines = [
        f"{len(stale)} store(s) of this deployment have a data volume that predates the "
        f"credentials just generated in {env_path.resolve()}.",
        "Each store reads its root credential only while initializing an empty volume, so it "
        "keeps the password it was created with and will reject the new one:",
        "",
    ]
    for var, store in stale:
        lines.append(f"    {store.service:<12} {project}_{store.volume}")
        if originals[var] is None:
            lines.append(f"    {'':<12} its container is gone — the original is unrecoverable")
        else:
            lines.append(f"    {'':<12} original recoverable from {project}{store.container}")
    lines.append("")

    recoverable = [var for var, _ in stale if originals[var] is not None]
    if reuse_stores:
        lines.append(
            "--reuse-stores cannot reopen every volume: a store whose container has been "
            "recreated no longer holds the credential its volume was initialized with."
        )
        if recoverable:
            lines.append(
                f"The other {len(recoverable)} could be adopted, but starting a stack that is "
                "half-adopted and half-doomed only moves the failure later."
            )
    elif len(recoverable) == len(stale):
        lines.append("--reuse-stores     keep the data: restore the original credential(s) to .env")
    lines.append("`osprey reset` re-initializes the whole stack, discarding the data in it.")
    lines += ["", "Nothing was started and no image was built."]
    return "\n".join(lines)


def _refuse_invented_history(config: dict) -> None:
    """Abort a deploy that would stand up a machine with a fabricated past.

    The honesty rule's deploy-time site. ``osprey build`` refuses to write the
    pairing and the MCP server refuses to start on it, but neither covers the
    deploy of a project whose ``config.yml`` was edited after the build — and a
    deploy is where the pairing becomes a running stack other people trust.

    Keyed on ``control_system.type`` alone, exactly as the other two sites are,
    rather than on ``deployed_services`` like ``_ensure_bluesky_substrate_env``
    below: that function is auto-configuring a service and so only cares whether
    the service is here, while this one is asking what the deployment claims
    about itself. An attached project deploying no VA container of its own still
    points its agent at one.

    Both keys are resolved through nested sections only, the way ``ConfigBuilder``
    reads a rendered ``config.yml`` — see
    :func:`~osprey.connectors.honesty.pairing_in_rendered_config`.

    :param config: Raw deploy config (the rendered project's ``config.yml``).
    :raises RuntimeError: if the config pairs a virtual accelerator with the
        mock archiver, an unset ``archiver.type`` included.
    """
    from osprey.connectors.honesty import VA_MOCK_ARCHIVER_WHY, pairing_in_rendered_config
    from osprey.connectors.types import VIRTUAL_ACCELERATOR

    pairing = pairing_in_rendered_config(config)
    if not pairing.is_invented_history:
        return

    raise RuntimeError(
        f"This deployment's control_system.type is {VIRTUAL_ACCELERATOR!r} and its "
        f"archiver.type is {pairing.archiver_phrase} — {VA_MOCK_ARCHIVER_WHY}\n"
        f"Fix config.yml before deploying: under its `archiver:` section set `type:` "
        f"to a connector reading a store this stack writes, or set the `type:` under "
        f"`control_system:` to 'mock' for an honestly storeless deployment. To have "
        f"the stack deploy its own store, rebuild from a profile carrying a "
        f"`va_archiver:` block — the control-assistant preset ships one, and "
        f"`osprey up` then brings up the store and its recorder beside the "
        f"virtual accelerator."
    )


def _ensure_bluesky_substrate_env(config: dict, env_path: Path | None = None) -> None:
    """Auto-configure the bluesky bridge's EPICS-substrate scan devices for a
    VA-backed Bluesky stack, making ``osprey up`` turn-key.

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
        current working directory -- ``osprey up`` always chdirs into the
        repo root first. Overridable for tests.
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
            "`osprey set connector=virtual_accelerator` -- that pairing needs the "
            "archiver pointed at a real store, and on a mock-archiver profile the "
            "next `osprey build` refuses and names the fix."
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
    environment, so empty certificate directories would cost such a deploy nothing
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
    and it leaves that function's minting, exposure and validation semantics
    untouched.

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
                f"unset {var} and let `osprey up` mint a keypair, or set "
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
            "or unset %s to let `osprey up` mint a fresh pair.",
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
            "both and let `osprey up` mint a matched pair."
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
        f"contain no '$' (unset {var} to let `osprey up` mint one), or set "
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
            "Use `osprey up --dev` to build and stage a wheel from this "
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
    # The Dockerfile lives inside the context's render, not at its root: the
    # context is a deployment REPO (see workspace.container_image_context), and a
    # repo keeps its build output one level down.
    dockerfile = os.path.join(project_root, BUILD_DIRNAME, "Dockerfile")
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
    with_plain_build_progress(cmd)
    cmd.append(project_root)
    return cmd


def _build_project_image(
    config: dict, dev_mode: bool, env: dict, build_context: Path | str | None = None
) -> None:
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
    :param build_context: The container repo this image is built from —
        ``<repo>/build/.image/<project>``, which ``osprey build`` rendered
        against the ``/app/<project>`` path the container sees rather than this
        host's (:func:`osprey.utils.workspace.container_image_context`). Passed
        explicitly because it is the one thing here that is NOT the compose
        project directory. Building from ``<repo>/build`` instead would produce
        an image whose every recorded path — ``project_root``, and the
        ``OSPREY_CONFIG`` each MCP server is handed — names this machine.
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
    project_root = str(
        Path(build_context) if build_context is not None else resolve_repo_root(config)
    )

    # OSPREY_DEV=1 (the pin-relaxing build arg) is passed only when the wheel
    # was actually staged: on a failed build/staging the Dockerfile must keep
    # its fail-loud pinned install rather than silently falling back to the
    # latest published release under a flag that means "run my local code".
    staged_artifacts: list[Path] = []
    wheel_staged = False
    if dev_mode:
        # No build-context bloat guard here, deliberately. `build/` IS the
        # deployment being shipped in this context, so the usual advice — exclude
        # `build/` via `.dockerignore` — would produce an image with no config,
        # no .mcp.json and no Claude Code artifacts.
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
        logger.debug("Building dispatch worker project image %s:", project_image)
        logger.debug("Running command:\n    %s", " ".join(cmd))
        # Watched for the duration of the build and no longer; the step line
        # below is what reports the finished image.
        with (report := single_image_build_reporter(project_image)):
            run_captured(
                cmd,
                env=env,
                spool_name="build-project-image",
                repo_root=resolve_repo_root(config),
                on_line=report,
            )
        _report_step(f"project image {project_image}")
    finally:
        # Remove BOTH staged artifacts (wheel + requirements manifest) so
        # neither can poison a later non-dev build in this context.
        for artifact in staged_artifacts:
            try:
                artifact.unlink()
            except OSError:
                logger.warning("Could not remove staged dev artifact %s", artifact)


def _env_file_args(repo_root: Path | str | None = None) -> list[str]:
    """``["--env-file", "<repo>/.env"]`` if that file exists, else ``[]``.

    Thin binding of :func:`osprey.deployment.compose_generator.compose_env_file_args`
    into this module, kept because
    :func:`osprey.deployment.web_terminals.provision.web_stack_compose_cmd`
    resolves the fragment through this name when a caller has none to thread
    down. The rule and its "no .env" warning have one definition, in the module
    that owns the invocation contract.

    :param repo_root: The deployment repo whose ``.env`` this is. ``None``
        falls back to the working directory, which every repo-scoped verb has
        already chdir'd into.
    """
    from osprey.deployment.compose_generator import compose_env_file_args

    return compose_env_file_args(repo_root if repo_root is not None else Path.cwd())


#: A shell-variable name as compose's interpolator recognises one.
_ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _compose_interpolated_vars(text: str) -> set[str]:
    """Every variable name a compose document *substitutes* from the environment.

    Covers the forms compose's interpolator recognises — ``$VAR``, ``${VAR}``,
    and each of the braced modifiers (``${VAR:-d}``, ``${VAR-d}``, ``${VAR:?m}``,
    ``${VAR?m}``, ``${VAR:+a}``, ``${VAR+a}``) — because the modifier changes
    what compose substitutes when the name is *unset*, not whether the name is
    read. A variable with a default is still shadowed by an export.

    ``$$`` is skipped: it is compose's escape for a literal ``$``, so the name
    behind it is never resolved and warning about it would be a false positive.
    Scanning continues *inside* the braces, so a default that itself interpolates
    (``${A:-${B}}``) contributes both names.

    A scanner rather than a single regex, because the braced form nests and the
    ``$$`` escape has to be consumed before the character after it can be judged.
    Deliberately syntactic — the whole document is scanned as text rather than
    parsed as YAML, since compose substitutes before it parses, so a reference in
    a comment-shaped or otherwise unparsed position is still a reference.

    :param text: The raw contents of one rendered compose file.
    :return: The referenced variable names.
    """
    names: set[str] = set()
    index = 0
    end = len(text)
    while index < end:
        if text[index] != "$":
            index += 1
            continue
        nxt = text[index + 1] if index + 1 < end else ""
        if nxt == "$":
            index += 2  # `$$` — an escaped literal dollar, not a reference
            continue
        match = _ENV_VAR_NAME.match(text, index + 2 if nxt == "{" else index + 1)
        if match:
            names.add(match.group(0))
            index = match.end()
        else:
            index += 1
    return names


def _preflight_env_shadowing(
    compose_files: list[str],
    repo_root: Path | str,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Warn when a shell export shadows the ``.env`` value compose would use.

    ``<repo>/.env`` is the deployment's one secret store, and every compose
    invocation is pointed at it explicitly (see
    :func:`osprey.deployment.compose_generator.compose_env_file_args`). But
    ``--env-file`` is the *lowest* precedence source for document interpolation:
    a variable also exported in the calling shell wins, and compose says nothing
    about it. So a stale export silently starts the stack on a value its volumes
    were never initialized with — an openobserve that will not accept the
    password in ``.env``, a database whose data directory answers to the pinned
    value while the container is handed another one.

    The same precedence the mint already designs to (``_ensure_service_tokens``
    skips a var the process env sets, so no divergent value is ever written
    into ``.env``): the export is honoured, the store is left intact, and
    nothing on either path tells the operator the two disagree. This is that
    telling.

    Only variables the rendered compose files actually interpolate are compared,
    against the file that will be passed as ``--env-file``. An exported name the
    store does not pin is not a divergence (there is nothing for it to
    contradict), and a pinned name no compose file reads cannot change what
    starts.

    **A warning, never a refusal.** Exporting a variable over the store is a
    legitimate gesture — a one-off run against another host's credentials, a
    rotation in progress — and a deploy that refused it would break the escape
    hatch to protect people from using it.

    **Names only, never values.** Which variable diverged is the actionable
    fact; what either side holds is a secret, and a warning is the one place a
    secret would leak into a terminal, a CI log and a bug report at once. Same
    rule as :func:`osprey.utils.dotenv.compose_unsafe_vars` and the build's own
    divergence warning.

    :param compose_files: The compose files this deploy will start, in ``-f``
        order. Relative entries resolve against *repo_root*, exactly as
        :func:`osprey.deployment.compose_generator.compose_base_cmd` resolves
        them. Unreadable entries are skipped — a missing compose file is the
        start's own error to raise, not this advisory's.
    :param repo_root: The deployment repo root; both the compose project
        directory and the home of the ``.env`` compose is pointed at.
    :param environ: The process environment to compare against. ``None`` reads
        the live one, overlaid with the shell values the CLI's entry-time
        ``.env`` load replaced (:func:`~osprey.utils.config.dotenv_shell_overrides`)
        — the comparison is against what the operator's shell actually
        exported, which the in-process override would otherwise have erased.
    :return: The shadowed variable names, sorted. Empty when there is no
        divergence, no ``.env``, or nothing interpolated.
    """
    root = Path(repo_root).expanduser().absolute()
    env_file = root / COMPOSE_ENV_FILENAME
    if not env_file.is_file():
        # No store to be shadowed: compose is given no --env-file at all, so the
        # process env is the only source and nothing is being overridden.
        return []

    pinned = parse_dotenv_file(env_file)
    if not pinned:
        return []

    if environ is None:
        from osprey.utils.config import dotenv_shell_overrides

        process_env: Mapping[str, str] = {**os.environ, **dotenv_shell_overrides()}
    else:
        process_env = environ

    referenced: set[str] = set()
    for compose_file in compose_files:
        path = Path(compose_file)
        path = path if path.is_absolute() else root / path
        try:
            referenced |= _compose_interpolated_vars(path.read_text(encoding="utf-8"))
        except OSError:
            continue

    shadowed = sorted(
        name
        for name in referenced
        if name in pinned and name in process_env and process_env[name] != pinned[name]
    )
    if not shadowed:
        return []

    logger.warning(
        "Shell export shadows this deployment's .env: %s\n"
        "  Exported here with a value that differs from the one pinned in %s. Compose "
        "substitutes the EXPORTED value into the compose files — --env-file is the "
        "lower-precedence source — so the stack starts on the shell's value while the "
        "store keeps the one its volumes were initialized with. Values are never "
        "printed.\n"
        "  To start on the pinned value: unset %s. To adopt the exported one: edit "
        ".env to match — but a volume that already exists keeps the credential it was "
        "created with, whichever value the container is handed.",
        ", ".join(shadowed),
        env_file,
        " ".join(shadowed),
    )
    return shadowed


def _check_shared_disk_preflight(config: dict) -> None:
    """Abort before any compose invocation if a configured shared-disk host path is missing.

    A container that bind-mounts a host path that doesn't exist still starts,
    then fails only at first read/write with an obscure in-container error.
    Checking on the host, before ``compose`` ever runs, turns that into an
    immediate, actionable deploy-time error instead of a confusing runtime one.

    Skipped entirely when ``modules.shared_disk`` is absent/disabled, or when
    ``host_path`` isn't configured — there's nothing to check in either case.

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


# ---------------------------------------------------------------------------
# Staged archiver bring-up
# ---------------------------------------------------------------------------

# The archiver's two compose service keys. The store's key and its
# deployed_services name are the same word; the recorder's are NOT — config and
# deployed_services call it ``archiver_recorder`` (the directory under
# ``services/``), while its compose service key is hyphenated. Getting that
# wrong turns the quiesce below into a "no such service" error, so both spellings
# are named here rather than being spelled inline at each use.
_ARCHIVER_STORE_SERVICE = "mongodb"
_ARCHIVER_RECORDER_SERVICE = "archiver-recorder"
_ARCHIVER_RECORDER_DEPLOY_NAME = "archiver_recorder"

# The install target every archiver error names. pymongo is an optional extra,
# and a deploy that hits the seeder without it must say what to install rather
# than surfacing a bare ImportError.
_ARCHIVER_EXTRA = "osprey-framework[archiver-mongodb]"

# How long the staged store gets to start answering before the deploy gives up.
# Generous because the very first start of a fresh volume creates the admin user
# and preallocates WiredTiger's journal before mongod accepts a connection —
# which is exactly what the compose healthcheck's own ``start_period`` covers.
_ARCHIVER_HEALTH_TIMEOUT_S = 180.0
_ARCHIVER_HEALTH_POLL_S = 2.0

# MongoDB's AuthenticationFailed error code.
_MONGO_AUTHENTICATION_FAILED = 18

# How long an authentication refusal is read as "the store is still creating its
# root user" rather than "this password is wrong".
#
# Three numbers have to stay in this order, and the ordering IS the design:
#
#   15 s   the mongodb compose template's healthcheck ``start_period`` — mongod's
#          own declared allowance for initializing a fresh volume
#   45 s   this grace window
#  180 s   _ARCHIVER_HEALTH_TIMEOUT_S, the full reachability budget
#
# Below the start_period, a fresh volume's normal initialization would be called
# a wrong password and abort the first deploy of every new project. Above the
# reachability budget, the grace would never expire and a genuinely stale volume
# would burn the whole budget for a diagnosis available in seconds. Move the
# healthcheck's start_period and this has to move with it.
_ARCHIVER_AUTH_GRACE_S = 45.0

# Minimum gap between seed progress lines. A base seed writes thousands of
# chunks; reporting each one would bury the deploy log, and reporting none would
# leave a multi-minute step looking like a hang.
_ARCHIVER_PROGRESS_INTERVAL_S = 15.0


def _archiver_store_deployed(config: dict) -> bool:
    """True when this deploy runs the archiver store itself.

    Membership in ``deployed_services`` is the gate for every archiver step
    below: a project that reads a store someone *else* runs (an attached
    facility project, which sets ``va_archiver.host``) must never have its
    history seeded or reseeded by a local deploy.
    """
    return _ARCHIVER_STORE_SERVICE in (config.get("deployed_services") or [])


def _preflight_archiver_pymongo(config: dict) -> None:
    """Abort a store-deploying run that cannot talk to the store it deploys.

    pymongo is an optional extra, and without it the seeder cannot run — but the
    only place that becomes visible is minutes later, after the project image has
    been built and the store container is already up. Checking here, beside the
    token mint, turns that into an immediate error naming the install.

    :param config: Raw deploy config.
    :raises RuntimeError: if the store is deployed and pymongo is absent.
    """
    if not _archiver_store_deployed(config):
        return
    try:
        import pymongo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "This project deploys the archiver store (services.mongodb), and seeding "
            "its history needs pymongo, which is not installed.\n"
            f"Install it with: pip install '{_ARCHIVER_EXTRA}'"
        ) from exc


def _archiver_seed_inputs(config: dict, project_dir: Path):
    """The channel set, engine and boot values one base seed is built from.

    Every import here is function-local. The seeder pulls in numpy and the
    simulation package, and hoisting either into this module's import path would
    put a scientific-stack import on every deploy-verb invocation, archiver
    or not.

    The channel set is the build-generated manifest the Virtual Accelerator and
    the recorder both read, resolved exactly as they resolve it
    (``VA_CHANNELS_FILE``, relative names against ``data/simulation/``), so the
    seeded history covers precisely the channels the live half serves. It is read
    from the project's own ``.env`` first, because that is where the build wrote
    it; the ambient value is the fallback for a deploy whose environment carries
    it instead.

    :returns: ``(channels, engine, boot_values)``. ``engine`` is ``None`` and
        ``boot_values`` empty for a project with no machine model — every channel
        is then procedural, which is a valid configuration, not a fault.
    """
    from osprey.services.virtual_accelerator.manifest.build import build_manifest
    from osprey.services.virtual_accelerator.manifest.loaders import (
        load_machine_json_channels,
        load_manifest_file,
    )
    from osprey.simulation.apply import resolve_simulation_file
    from osprey.simulation.engine import SimulationEngine, resolve_state_dir

    env = parse_dotenv_file(project_dir / ".env") if (project_dir / ".env").is_file() else {}
    named = (env.get("VA_CHANNELS_FILE") or os.environ.get("VA_CHANNELS_FILE") or "").strip()
    if named:
        manifest_path = Path(named)
        if not manifest_path.is_absolute():
            # The render's data dir, not the source zone: the build generates
            # the manifest into `build/data/simulation` only, and that is the
            # directory the VA and recorder containers mount as /data/simulation
            # — so it is the one place a relative VA_CHANNELS_FILE can name the
            # same channel set the live half serves.
            manifest_path = project_dir / BUILD_DIRNAME / "data" / "simulation" / manifest_path
        channels = load_manifest_file(manifest_path)
    else:
        channels = build_manifest()["channels"]

    machine_path, _, _, _ = resolve_simulation_file(config, project_dir)
    if machine_path is None or not machine_path.is_file():
        return channels, None, {}

    engine = SimulationEngine.from_file(
        machine_path, state_dir=resolve_state_dir(config, project_dir)
    )
    # The same map the Virtual Accelerator boots its records from: machine.json's
    # static channel values, skipping the handful of derived channels that carry
    # an expression instead. Anchoring the procedural generator on it is what
    # makes a seeded sample and a recorded one describe the same machine.
    boot_values = {
        address: entry["value"]
        for address, entry in load_machine_json_channels(machine_path).items()
        if "value" in entry
    }
    return channels, engine, boot_values


def _reapply_active_scenarios(config: dict, project_dir: Path, engine) -> None:
    """Re-apply the active scenario set onto a freshly rebuilt base.

    A reseed rewrites the whole base series, which erases the event windows the
    active scenarios had written into it. Without this the deployment would come
    back up claiming a fault is active while its history showed a clean machine —
    precisely the divergence the stored archiver exists to remove.

    The logbook is deliberately left alone: a knob change rebuilds the archive,
    not the narrative, and purging ARIEL's entries here would destroy history
    this deploy was never asked to touch.

    A failure here is not recoverable by retrying the deploy, and the error says
    so. By this point the seeder has written the manifest, so the next
    ``osprey up`` reads MATCH, skips the reseed, and skips this step with it —
    leaving a clean base under a faulted machine permanently. The manifest is
    deliberately NOT deleted to force a retry: that would throw away a
    multi-minute seed to redo work that succeeded. The one step that failed is
    the one the operator is told to re-run.

    :raises RuntimeError: if the re-apply fails, naming the command that fixes it.
    """
    from osprey.simulation.apply import apply_scenarios, persisted_scenario_anchor
    from osprey.simulation.engine import DEFAULT_SCENARIO

    if engine is None:
        logger.info("No machine model in this project; no scenarios to re-apply after the reseed")
        return

    names = engine.active_scenarios()
    # The anchor the running world is already on: re-anchoring here would slide
    # the live VA's events, the logbook and the archive's windows to a T0 nobody
    # asked for, as a side effect of a deploy meant to rebuild only the store.
    anchor = persisted_scenario_anchor(config, project_dir)
    try:
        result = apply_scenarios(project_dir, names, seed_logbook=False, now=anchor)
    except Exception as exc:
        # `nominal` is implicit, so the recovery command names the faults — which
        # is what the operator activated and what `sim apply` expects back.
        faults = [name for name in names if name != DEFAULT_SCENARIO] or [DEFAULT_SCENARIO]
        raise RuntimeError(
            "The base series was rebuilt but the active scenarios could not be "
            "re-applied, so the archive currently shows a clean machine while the "
            f"simulation runs {list(names)!r}.\n"
            f"Re-run `osprey sim apply {' '.join(faults)}` from {project_dir} to put the "
            f"event windows back. Cause: {exc}"
        ) from exc
    logger.key_info(f"Re-applied scenarios {list(result.active)!r} onto the rebuilt archive")


def _wait_for_archiver_store(
    collection, deadline: float, store_hint: str = "the archiver store"
) -> None:
    """Block until the staged store answers, or fail with what it was asked.

    The compose healthcheck already gates the recorder on the same condition;
    this is the host side of it, and it probes with the credentials the seeder is
    about to use — so an authentication failure surfaces here, with the variable
    named, rather than as an unexplained seeding error.

    :param collection: The archive collection, for its client.
    :param deadline: :func:`time.monotonic` instant to give up at.
    :param store_hint: How to name this store in an error — who is connecting
        where, which is what tells an operator whose credentials were refused.
    :raises RuntimeError: if the store rejects these credentials, or is still
        unreachable at ``deadline``.
    """
    from pymongo.errors import OperationFailure, PyMongoError

    client = collection.database.client
    auth_deadline = time.monotonic() + _ARCHIVER_AUTH_GRACE_S
    last: Exception | None = None
    while True:
        try:
            client.admin.command("ping")
            return
        except OperationFailure as exc:
            # "Authentication failed" means two opposite things depending on
            # WHEN it arrives, and the difference decides whether a first deploy
            # works at all.
            #
            # On a FRESH volume, mongod accepts connections before it has created
            # the root user from MONGO_INITDB_ROOT_USERNAME/PASSWORD. A probe
            # landing in that window is refused — and the store is perfectly
            # healthy a second later. Treating that as terminal aborts the very
            # first deploy of every new project, intermittently, depending on
            # which side of the race the probe lands. (It is the same window the
            # compose healthcheck covers with its own ``start_period``.)
            #
            # Once the store has had time to initialize, the SAME error means the
            # stale-volume shape instead: mongod reads its root credentials only
            # when initializing a fresh volume, so a rotated MONGO_ROOT_PASSWORD
            # leaves a running store that will refuse this password forever. That
            # is worth failing fast on rather than burning the full budget.
            #
            # So the grace window is what separates them: retry while the store
            # could still be initializing, then treat a refusal as final.
            if exc.code != _MONGO_AUTHENTICATION_FAILED:
                raise
            if time.monotonic() < auth_deadline:
                last = exc
                time.sleep(_ARCHIVER_HEALTH_POLL_S)
                continue
            raise RuntimeError(
                f"The archiver store rejected the credentials in {store_hint}: {exc}\n"
                "mongod reads its root credentials only when initializing a FRESH data "
                "volume, so a store whose volume predates the current MONGO_ROOT_PASSWORD "
                "keeps the password it was created with. Either restore the original "
                "value, or remove the `archiver_mongodb_data` volume to re-initialize the "
                "store — which discards the seeded and recorded history with it."
            ) from exc
        except PyMongoError as exc:
            last = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"The archiver store did not become reachable within "
                    f"{_ARCHIVER_HEALTH_TIMEOUT_S:.0f}s: {exc}\n"
                    f"Check `{_ARCHIVER_STORE_SERVICE}` in `osprey status` and its "
                    "container logs. A store on a pre-existing volume keeps the credentials "
                    "it was initialized with, so a rotated MONGO_ROOT_PASSWORD needs the "
                    "volume recreated (which discards the archive)."
                ) from last
            time.sleep(_ARCHIVER_HEALTH_POLL_S)


def _seed_progress_reporter():
    """A :func:`~osprey.simulation.archiver_seed.seed_base` progress callback.

    Each firing becomes a step line under the verb's open phase, so a first
    deploy's multi-minute seed reports as it goes instead of stalling silently.
    Rate-limited rather than per-chunk: the callback fires once per chunk, which
    on a large archive is many times a second, and the point is to prove the step
    is moving — not to narrate every insert. The channel count and the seeded
    span are fixed for the run and already announced before seeding starts, so
    the line carries the one quantity that grows.
    """
    last = [time.monotonic()]

    def report(seed_report) -> None:
        now = time.monotonic()
        if now - last[0] < _ARCHIVER_PROGRESS_INTERVAL_S:
            return
        last[0] = now
        _report_step(
            f"seeding archive: {seed_report.documents:,} documents written "
            f"across {seed_report.channels:,} channels"
        )

    return report


def _archiver_store_connection(config: dict, project_dir: Path) -> dict | None:
    """Connection parameters for the store this deploy is bringing up.

    Delegates to :func:`~osprey.simulation.apply.archiver_store_config` so the
    deploy-time seeder and ``osprey sim apply`` open one store the same way, then
    fills in the one difference between the two callers. ``sim apply`` reads the
    password from the project ``.env`` and never from the ambient environment,
    because it is routinely run from somewhere else and must not pick up a
    foreign deployment's credential. ``osprey up`` is the process that hands
    compose its environment: when the password is exported rather than written to
    ``.env``, the exported value is what the store container is created with, so
    it is also what the seeder must authenticate with.

    :returns: The parameters, or ``None`` when the project declares no connection
        block for the store it deploys — nothing can be seeded, and saying so is
        better than guessing a host.
    """
    from osprey.simulation.apply import archiver_store_config

    store = archiver_store_config(config, project_dir)
    if store is None:
        return None
    if not store["password"]:
        store["password"] = os.environ.get(store["password_env"], "")
    return store


def _stage_archiver_store(config, compose_files, env, project_dir, *, keep_base=False) -> None:
    """Start the archiver store on its own and seed its base history.

    Staged ahead of the full bring-up for one reason: the recorder writes into a
    collection the seeder creates, with the indexes and the compressor the seeder
    chooses. Starting both at once would race the collection into existence with
    whatever options the first writer happened to imply.

    What happens depends on what the store's seed manifest says about the knobs
    now in force:

    * **match** — the store already holds the archive this profile describes.
      Nothing is written, and the deploy continues immediately.
    * **absent** — no manifest: a first deploy, or a volume that ``clean``/
      ``rebuild`` wiped. Seed it.
    * **mismatch** — the knobs moved, so the store's coverage no longer describes
      what the profile asks for. Report what changed, then rebuild — unless
      ``keep_base`` says to leave it alone.

    Both rebuild paths quiesce the recorder first. It is one operation — stop the
    writer, drop the collection, rebuild it, re-apply the active scenarios — and
    splitting it by state would leave the mismatch path stopping a writer the
    absent path lets run into a collection being dropped underneath it. Stopping
    a service that was never started is a no-op, and neither deploy branch needs
    anything undone afterwards: the full ``up`` that follows starts it again.

    :param config: Raw deploy config.
    :param compose_files: Rendered compose file paths for this deploy.
    :param env: The environment the deploy hands compose.
    :param project_dir: Root of the built project (holds ``.env`` and ``data/``).
    :param keep_base: Leave a mismatched base in place instead of rebuilding it.
    :raises RuntimeError: if the store cannot be reached or authenticated.
    """
    from osprey.simulation.apply import archiver_collection
    from osprey.simulation.archiver_seed import (
        SeedKnobs,
        SeedState,
        compare_fingerprint,
        seed_base,
        seed_fingerprint,
    )

    store = _archiver_store_connection(config, project_dir)
    if store is None:
        logger.warning(
            "This project deploys the archiver store but declares no "
            "`archiver.mongodb_archiver` connection block, so its history cannot be "
            "seeded. The store still starts; archiver reads will find it empty."
        )
        return
    if not store["password"]:
        raise RuntimeError(
            f"The archiver store's password variable {store['password_env']} is set "
            f"neither in {project_dir / '.env'} nor in this environment, so the seeder "
            "cannot authenticate against the store it is about to start."
        )

    knobs = SeedKnobs.from_config(config)
    services = config.get("services") or {}
    compression = str((services.get(_ARCHIVER_STORE_SERVICE) or {}).get("compression") or "zstd")

    run_env = runtime_env(config, env)
    # The invocation contract, same as _start_stack: -f paths and --env-file
    # anchored on the repo root, so staging is correct from any working
    # directory rather than only from a root the caller chdir'd into.
    base_cmd = compose_base_cmd(
        get_runtime_command(config),
        compose_files,
        project_dir,
        _env_file_args(project_dir),
    )

    up_cmd = base_cmd + ["up", "-d", _ARCHIVER_STORE_SERVICE]
    logger.debug(f"Running command:\n    {' '.join(up_cmd)}")
    run_captured(up_cmd, env=run_env, spool_name="archiver-store-up", repo_root=project_dir)
    _report_step("archiver store started")

    # Assembled while the store boots, and before the health budget starts: this
    # reads a manifest and a machine model off disk, and charging that time
    # against the store's start-up allowance would make a slow disk look like an
    # unreachable server.
    channels, engine, boot_values = _archiver_seed_inputs(config, project_dir)
    fingerprint = seed_fingerprint(
        knobs, (str(channel["address"]) for channel in channels), compression=compression
    )

    store_hint = f"{store['username']}@{store['host']}:{store['port']}"
    with archiver_collection(store) as collection:
        _wait_for_archiver_store(
            collection, time.monotonic() + _ARCHIVER_HEALTH_TIMEOUT_S, store_hint
        )
        comparison = compare_fingerprint(collection, fingerprint)

        if comparison.state is SeedState.MATCH:
            logger.key_info("Archive already covers the configured window; skipping the base seed")
            return

        if comparison.state is SeedState.MISMATCH:
            logger.key_info("The archive's knobs changed since it was seeded:")
            logger.key_info(comparison.describe())
            if keep_base:
                logger.warning(
                    "Keeping the existing base (--keep-archiver-base). The stored history "
                    "still describes the OLD knobs, so archiver reads will not match what "
                    "this profile declares."
                )
                return
            logger.key_info(
                "Rebuilding the base series. Recorded history is discarded with it — the "
                "recorder's samples share the collection being dropped. Pass "
                "--keep-archiver-base to skip this."
            )

        if _ARCHIVER_RECORDER_DEPLOY_NAME in (config.get("deployed_services") or []):
            stop_cmd = base_cmd + ["stop", _ARCHIVER_RECORDER_SERVICE]
            logger.debug(f"Running command:\n    {' '.join(stop_cmd)}")
            quiesce = run_captured(
                stop_cmd,
                env=run_env,
                spool_name="archiver-recorder-stop",
                repo_root=project_dir,
                check=False,
            )
            _report_step("archiver recorder quiesced")
            # Not fatal — the rebuild is still the right thing to do — but a
            # recorder that would not stop is writing into the collection about
            # to be dropped, so the breach of the quiesce invariant has to be
            # visible rather than swallowed by a return code nobody reads.
            if quiesce.returncode != 0:
                logger.warning(
                    f"Could not stop `{_ARCHIVER_RECORDER_SERVICE}` (exit "
                    f"{quiesce.returncode}). If it is still running it may write "
                    "samples into the collection being rebuilt; check "
                    "`osprey status` if the reseeded archive looks wrong."
                )

        collection.drop()
        logger.key_info(
            f"Seeding {len(channels):,} channels over {knobs.retention_days} days "
            f"({knobs.hot_span_hours}h at {knobs.hot_cadence_sec}s, then "
            f"{knobs.tail_cadence_sec}s). This takes minutes on a first deploy."
        )
        report = seed_base(
            collection,
            channels,
            knobs,
            t0=datetime.now(UTC),
            engine=engine,
            boot_values=boot_values,
            compression=compression,
            progress=_seed_progress_reporter(),
        )
        logger.key_info(report.describe())

    # Outside the store connection: re-applying opens its own, and holding this
    # one across it would keep an idle client alive for the whole rewrite.
    _reapply_active_scenarios(config, project_dir, engine)


# ---------------------------------------------------------------------------
# Staged ARIEL bring-up
# ---------------------------------------------------------------------------

# ARIEL's store, under both names it goes by: the compose service key and the
# ``deployed_services`` entry are the same word here (unlike the archiver's
# recorder), but naming it once keeps the two uses from drifting apart.
_ARIEL_STORE_SERVICE = "postgresql"

# How long the staged store gets to accept a connection. Shorter than the
# archiver's budget because postgres does not open its port until the cluster is
# initialized and ready — a first start of a fresh volume runs initdb behind a
# closed socket, so what this waits out is a refusal, not a slow answer.
_ARIEL_HEALTH_TIMEOUT_S = 90.0
_ARIEL_HEALTH_POLL_S = 2.0

# The one command that finishes the job by hand, named in every warning below.
# `quickstart` rather than `migrate`: it runs the migration AND reports what it
# found, so an operator following the message once ends up where this staging
# would have left them.
_ARIEL_RECOVERY_HINT = "osprey ariel quickstart"


def _ariel_store_deployed(config: dict) -> bool:
    """True when this deploy runs ARIEL's store itself, for a project that uses it.

    Both halves matter. Membership in ``deployed_services`` keeps a project
    pointed at a Postgres someone *else* runs from having its schema created or
    its logbook written by a local deploy. An ``ariel:`` section is what makes a
    deployed Postgres ARIEL's at all — the same store can be deployed for
    something else entirely, and a deploy has no business migrating a database
    whose schema it cannot claim to own.
    """
    return _ARIEL_STORE_SERVICE in (config.get("deployed_services") or []) and bool(
        config.get("ariel")
    )


def _ariel_store_config(config: dict, project_dir: Path) -> dict:
    """The project's ``ariel:`` section with its DSN resolved for THIS project.

    The password is read from ``<project_dir>/.env`` by name, never from the
    ambient environment — the same rule the archiver seeder follows, and for the
    same reason: a deploy is routinely driven from another directory, where an
    exported ``ARIEL_DB_PASSWORD`` belongs to somebody else's deployment.

    Resolving it here (rather than letting each consumer derive its own) also
    means the migration and the seed that follows are pinned to one DSN, so they
    cannot disagree about which database this deploy is talking to.
    """
    from osprey.services.ariel_search.config import resolve_ariel_dsn
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = Path(project_dir) / ".env"
    env = parse_dotenv_file(env_path) if env_path.is_file() else {}

    ariel = dict(config.get("ariel") or {})
    services = (config.get("services") or {}).get(_ARIEL_STORE_SERVICE) or {}
    dsn = resolve_ariel_dsn(ariel, services, env=env)
    ariel["database"] = {**(ariel.get("database") or {}), "uri": dsn}
    return ariel


def _wait_for_ariel_store(ariel_config: dict, deadline: float) -> None:
    """Block until the staged store accepts these credentials, or give up.

    Probes with the DSN the migration is about to use, so a wrong password
    surfaces here — named, and beside the deploy step that owns it — rather than
    as a migration error that reads like a schema problem.

    :param ariel_config: ARIEL config with its DSN already resolved.
    :param deadline: :func:`time.monotonic` instant to give up at.
    :raises RuntimeError: if the store is still unreachable at ``deadline``.
    """
    import psycopg

    dsn = str((ariel_config.get("database") or {}).get("uri"))
    last: Exception | None = None
    while True:
        try:
            with psycopg.connect(dsn, connect_timeout=5):
                return
        except Exception as exc:  # noqa: BLE001 — every failure here is "not yet"
            last = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"ARIEL's database did not accept a connection within "
                    f"{_ARIEL_HEALTH_TIMEOUT_S:.0f}s: {last}"
                ) from last
            time.sleep(_ARIEL_HEALTH_POLL_S)


def _migrate_ariel_store(ariel_config: dict) -> None:
    """Create ARIEL's schema, idempotently.

    Wrapped rather than called inline so the deploy has one seam for "the schema
    now exists", and so the async boundary lives in exactly one place.
    """
    import asyncio

    from osprey.services.ariel_search.cli_operations import run_migrate

    asyncio.run(run_migrate(ariel_config))


def _stage_ariel_store(config, compose_files, env, project_dir) -> None:
    """Start ARIEL's store, create its schema, and seed a first narrative.

    The logbook counterpart of :func:`_stage_archiver_store`, and staged ahead of
    the full bring-up for the same kind of reason: everything that reads ARIEL —
    the panel's own server and the ``ariel`` MCP server inside every web terminal
    — connects at start-up and reports a database with no tables as no database
    at all. Creating the schema after those consumers are running would leave a
    stack that is only correct once somebody restarts it.

    Three steps, each conditional on the one before:

    * **migrate** — always, and idempotent: this is what a store this deploy
      brought up owes the consumers it is about to start.
    * **seed** — only into an EMPTY logbook, and only what the active scenarios
      already narrate (see
      :func:`osprey.simulation.apply.seed_active_logbook`). A deploy may fill a
      blank; it may not rewrite history.

    Failure here warns and returns rather than aborting the deploy. The logbook is
    one panel among many, and a control room whose channels, scans and archive are
    all up should not be denied them because its search tab could not be
    provisioned — but the warning names the command that finishes the job, so the
    gap is never silent.

    :param config: Raw deploy config.
    :param compose_files: Rendered compose file paths for this deploy.
    :param env: The environment the deploy hands compose.
    :param project_dir: Root of the built project (holds ``.env``).
    """
    if not _ariel_store_deployed(config):
        return

    from osprey.simulation import apply as simulation_apply

    run_env = runtime_env(config, env)
    base_cmd = compose_base_cmd(
        get_runtime_command(config),
        compose_files,
        project_dir,
        _env_file_args(project_dir),
    )

    up_cmd = base_cmd + ["up", "-d", _ARIEL_STORE_SERVICE]
    logger.debug(f"Running command:\n    {' '.join(up_cmd)}")
    run_captured(up_cmd, env=run_env, spool_name="ariel-store-up", repo_root=project_dir)
    _report_step("ARIEL store started")

    ariel_config = _ariel_store_config(config, project_dir)
    try:
        _wait_for_ariel_store(ariel_config, time.monotonic() + _ARIEL_HEALTH_TIMEOUT_S)
        _migrate_ariel_store(ariel_config)
    except Exception as exc:  # noqa: BLE001 — reported, never fatal (see docstring)
        logger.warning(
            f"ARIEL's schema could not be created, so its panel and MCP tools will "
            f"report the database as unavailable. Everything else in this deploy is "
            f"unaffected. Run `{_ARIEL_RECOVERY_HINT}` from {project_dir} once the "
            f"database is reachable. Cause: {exc}"
        )
        return
    _report_step("ARIEL schema ready")

    try:
        seeded = simulation_apply.seed_active_logbook(config, project_dir, ariel_config)
    except Exception as exc:  # noqa: BLE001 — reported, never fatal (see docstring)
        logger.warning(
            f"ARIEL's schema is in place but its logbook could not be seeded, so the "
            f"panel will come up empty. Run `osprey sim apply` from {project_dir} to "
            f"write the active scenarios' entries. Cause: {exc}"
        )
        return
    if seeded:
        logger.key_info(f"Seeded {seeded} logbook entries from the active scenarios")


def deploy_up(
    config_path,
    detached=False,
    dev_mode=False,
    expose_network=False,
    keep_archiver_base=False,
    reuse_stores=False,
):
    """Start services using container runtime (Docker or Podman).

    The legacy entry point: it RE-RENDERS the compose files from the config it
    is handed and then starts them. A deployment repo starts from
    :func:`up_as_built` instead, which renders nothing.

    :param config_path: Path to the configuration file
    :type config_path: str
    :param detached: Run in detached mode
    :type detached: bool
    :param dev_mode: Development mode for local framework testing
    :type dev_mode: bool
    :param expose_network: Expose services to all network interfaces (0.0.0.0)
    :type expose_network: bool
    :param keep_archiver_base: Leave a mismatched archiver base in place instead
        of rebuilding it (see :func:`_stage_archiver_store`)
    :type keep_archiver_base: bool
    """
    config, compose_files = prepare_compose_files(config_path, dev_mode, expose_network)
    project_dir = Path(config_path).resolve().parent

    # Before every other check, and before any branch below has touched the
    # host: a deployment whose archive is fabricated does not get to start.
    _refuse_invented_history(config)

    # The compose project directory for every invocation below, resolved from
    # the config path rather than inherited from the working directory — see
    # compose_base_cmd for what it pins.
    repo_root = resolve_repo_root(config, config_path)

    # Advisory staleness check BEFORE anything deploys: a project rendered by
    # an older framework/preset self-describes an out-of-date service set in
    # config.yml, so the deploy below would "succeed" at the wrong goal with
    # no error anywhere. Never blocks (see warn_if_project_stale).
    warn_if_project_stale(project_dir)

    # Anchored on the config this verb was HANDED, not on a derived
    # `<repo>/build/config.yml`: a legacy flat project's own directory is its
    # render, so deriving the path would name a file that does not exist there.
    # Same contract as the as-built path — see `_start_as_built`.
    with config_anchored_at(config_path):
        _start_stack(
            config,
            compose_files,
            repo_root,
            detached=detached,
            dev_mode=dev_mode,
            expose_network=expose_network,
            # ANCHORED, deliberately: the provisioners this reaches mint REAL
            # SECRETS, and `_start_stack`'s default leaves them writing a
            # cwd-relative `.env`. The repo root is resolved four lines up and
            # every compose invocation below reads `<repo>/.env` with
            # --env-file, so an unanchored mint writes tokens the stack never
            # reads — a deploy that comes up with its fail-closed tokens unset,
            # which looks secure and says nothing.
            env_path=repo_root / COMPOSE_ENV_FILENAME,
            keep_archiver_base=keep_archiver_base,
            reuse_stores=reuse_stores,
        )


class BuildProgressReporter:
    """One build's captured stream, parsed once into one :class:`BuildModel`.

    A build site wants a live view of what each service is doing right now, and
    some sites want a step line each time an image finishes — and the one thing
    none of them may do is parse the stream twice. So this is every half at once
    over a single model:

    * the ``on_line`` **callable** handed to :func:`run_captured`, and
    * the **context manager** that registers that model with the phase reporter
      for the duration of the run::

          with (report := compose_build_step_reporter()):
              run_captured(..., on_line=report)

    Registration is scoped to the block: a build that has returned or raised
    stops being watched immediately, however it left. Used as a bare callable
    (no ``with``) it still reports its step lines — nothing here requires a
    watcher, which is what keeps library callers and tests working.

    Constructed through :func:`compose_build_step_reporter` or
    :func:`single_image_build_reporter` rather than directly: the two of them
    are what the arguments below mean at a real call site.

    ``label`` names the service for a single-image build, whose BuildKit
    headers carry no service name of their own; a ``compose build`` names every
    service itself and passes none. ``step_prefix`` is the step line's wording
    (``service image <ref>``); a site whose finished images are already
    reported elsewhere passes ``None`` and gets the live view alone.
    """

    def __init__(self, label: str | None = None, *, step_prefix: str | None = None) -> None:
        self._step_prefix = step_prefix
        #: The one model. Exposed so a caller can snapshot it directly.
        self.model = BuildModel(label, on_finished_image=self._report_image)
        self._watch: AbstractContextManager[BuildModel] | None = None

    def __call__(self, line: str) -> None:
        """Absorb one line of the child's output, timestamped on arrival."""
        self.model.feed(line, time.monotonic())

    def __enter__(self) -> "BuildProgressReporter":
        self._watch = current_reporter().watch_build(self.model)
        self._watch.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop watching, whatever left the block — and never swallow it."""
        watch, self._watch = self._watch, None
        if watch is not None:
            watch.__exit__(exc_type, exc, tb)

    def _report_image(self, ref: str) -> None:
        """Report one finished image as a step of the open phase.

        Fires for every image BuildKit names, including one on a vertex too
        anonymous to attribute to a service row — "an image is done" is a fact
        about the build, and a step line for it must not go missing because the
        stream's shape defeated attribution.
        """
        if self._step_prefix is not None:
            _report_step(f"{self._step_prefix} {ref}")


def compose_build_step_reporter() -> BuildProgressReporter:
    """Per-image progress steps for a ``compose build``, from BuildKit's own stream.

    ``compose build`` builds every service image in one parallel invocation —
    the longest step of a dev deploy, half an hour cold — and a single step
    line printed at the end reads as a hang for the whole build. Splitting the
    build per service would give progress at the price of the parallelism, so
    the progress is derived instead: fed to :func:`run_captured` as ``on_line``,
    this watches the captured stream, reports each image the moment BuildKit
    finishes it, and — inside its ``with`` block — keeps the phase reporter's
    live view of what every service is working on.

    Deduplicated per image, because BuildKit repeats the ``naming to`` line
    (bare, then with a duration). The implicit ``docker.io/library/`` prefix is
    stripped as registry noise; a real registry in the tag is the operator's
    own naming and stays. Both behaviours live in :class:`BuildModel`.
    """
    return BuildProgressReporter(step_prefix="service image")


def single_image_build_reporter(image_tag: str) -> BuildProgressReporter:
    """The live view for one image's own ``<runtime> build``, labeled by its tag.

    :func:`compose_build_step_reporter`'s counterpart for the sites that build a
    single image directly — the dispatch worker's project image, a persona
    image, the auth sidecar. Two things differ from the compose case, and both
    are silent when a call site gets them wrong:

    * A single-image build's BuildKit headers name no service (``#10 [ 2/13]``
      where compose writes ``#10 [va 2/13]``), so an unlabeled model parses the
      whole build into nothing — no rows, no error. The image tag is that label,
      and its row then reads as the image the build is producing.
    * No step lines. Each of these sites already reports its finished image as a
      step of its own once the build returns, so a line from the parser would
      report the same image twice. This is the live view alone.

    Used exactly like its compose sibling::

        with (report := single_image_build_reporter(tag)):
            run_captured(..., on_line=report)
    """
    return BuildProgressReporter(image_tag, step_prefix=None)


def _start_stack(
    config: dict,
    compose_files: list[str],
    repo_root: Path | str,
    *,
    detached: bool = False,
    dev_mode: bool = False,
    expose_network: bool = False,
    env_path: Path | None = None,
    build_context: Path | str | None = None,
    keep_archiver_base: bool = False,
    reuse_stores: bool = False,
) -> None:
    """Provision, preflight and start one already-resolved stack.

    Everything both start paths do once their compose files exist, in the one
    order they must happen in. :func:`deploy_up` reaches here having just
    rendered those files; :func:`up_as_built` reaches here having read the ones
    a build already wrote. Nothing below can tell the difference, which is the
    point: there is one deploy sequence, and the two verbs differ only in where
    the inputs came from.

    When ``modules.web_terminals.enabled`` is set, the web-terminal stack is
    reconciled too (rendering its artifacts and including
    ``docker-compose.web.yml`` in the compose invocation) — see
    :func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`.
    That reconcile always runs detached, independent of ``detached``, and takes
    over from the plain services path below.

    Idempotent from any prior state: every path first clears this project's own
    non-running containers (``compose rm -f`` — a wedged ``created`` container
    from an aborted deploy holds its published host ports on Docker Desktop,
    blocking the next ``up``), and the plain path's ``up`` carries
    ``--remove-orphans`` to reconcile away services dropped from the config.
    Running containers and volumes are never touched by either measure.

    Args:
        config: Loaded deploy config.
        compose_files: The services stack's compose files, in ``-f`` order.
        repo_root: The compose project directory every invocation is pinned to.
        detached: Run in detached mode.
        dev_mode: Stage the local framework into the images.
        expose_network: Whether this deployment publishes on all interfaces.
            Read as a statement of fact about what will be started, not as a
            request: it gates the empty-token refusal, and the as-built path
            derives it from the rendered bindings rather than from a flag.
        env_path: The ``.env`` every provisioner writes. ``None`` leaves each
            one on its own default, which is the working directory.
        build_context: Directory holding the rendered ``Dockerfile`` for the
            project image. ``None`` resolves the repo root — correct only for a
            project directory that IS its own render.
        keep_archiver_base: Leave a mismatched archiver base in place instead
            of rebuilding it (see :func:`_stage_archiver_store`).
    """
    web_terminals_enabled = _web_terminals_enabled(config)

    # A web-terminals-only deploy (no backend services) is valid, so the
    # early-return below must not fire on empty deployed_services in that case.
    if not config.get("deployed_services") and not web_terminals_enabled:
        logger.key_info(
            "No services configured for this project — deployed_services is empty in "
            "config.yml. Skipping osprey up."
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
    minted_store_vars = _ensure_service_tokens(config, expose_network, env_path)

    # Fail fast on the optional dependency the staged archiver bring-up below
    # cannot proceed without. Here, beside the mint that provisions the store's
    # own credential, rather than at the seeder: a missing extra must abort in
    # seconds, before the minutes-long image build, not after it.
    _preflight_archiver_pymongo(config)

    # Refuse a deploy whose store credentials a surviving data volume will
    # reject. Before every container-touching step below — the image build and
    # the archiver staging included — because `compose up` recreating a store
    # container destroys the only host-side copy of the credential that volume
    # was initialized with. See _preflight_stale_store_volumes.
    #
    # But AFTER the pymongo check above, which is a local import and costs
    # nothing: this one is the first thing on the path to ask the container
    # runtime a question, and a deploy doomed by a missing extra must abort
    # without having touched the host at all.
    _preflight_stale_store_volumes(
        config, minted_store_vars or set(), env_path or Path(".env"), reuse_stores=reuse_stores
    )

    # Auto-configure the bluesky bridge's EPICS-substrate scan devices for a
    # VA-backed Bluesky stack (additive; no-op unless both bluesky and
    # virtual_accelerator are deployed) -- see _ensure_bluesky_substrate_env.
    _ensure_bluesky_substrate_env(config, env_path)

    # Provision the bluesky scan stack's 0MQ key material before compose mounts
    # it: the RE manager's control-socket keypair into .env, and the document
    # plane's CURVE certificates into data/bluesky_curve/. Both are no-ops
    # without the bluesky service, and both are additive (existing material
    # always wins) so a redeploy never rotates keys under a running stack.
    # Neither is gated on the connector -- see
    # _ensure_bluesky_document_plane_certs for why a browse-only deploy still
    # gets its certificates.
    _ensure_bluesky_control_plane_keys(config, env_path)
    _ensure_bluesky_document_plane_certs(config, env_path)

    # Advisory, and last of the .env preflights so it reads the file every
    # provisioner above has finished writing: a shell export beats --env-file
    # for compose's document interpolation, so an export that disagrees with the
    # store silently starts the stack on a value its volumes never adopted.
    # Names the variables, never the values -- see _preflight_env_shadowing.
    # Before the image build below, so the operator sees it in seconds rather
    # than after the minutes a build takes.
    _preflight_env_shadowing(compose_files, repo_root)

    # Set up environment for containers. The shell values the CLI's entry-time
    # `.env` load replaced are restored on top: compose documents the OPPOSITE
    # precedence (a shell export beats --env-file), and the warning above just
    # told the operator the exported value is the one compose will substitute —
    # handing compose the overridden copy would make that a lie and silently
    # start the stack on the store's value instead. Variables `.env` added that
    # the shell never set are untouched (their process value IS the file's).
    from osprey.utils.config import dotenv_shell_overrides

    env = {**os.environ, **dotenv_shell_overrides()}
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
        preflight_web_terminals(config)

    # Build the <project>:local image the dispatch worker references. The worker
    # has no compose build block (that would race the event-dispatcher on the
    # shared tag), so this is the only thing that produces its image. No-op
    # unless the worker is deployed on the local project image. Run before
    # `compose up` (which, non-detached, os.execvpe-replaces this process).
    _build_project_image(config, dev_mode, env, build_context)

    # Stage the archiver store and seed its history BEFORE the branch split, so
    # both deploy paths inherit it and the recorder — started by whichever `up`
    # follows — finds a collection that already exists with the right indexes.
    # No-op unless this project deploys the store itself. Anchored on the repo
    # root: the single root `.env` is the secret store the seeder authenticates
    # from, and every compose invocation on this path reads it with --env-file.
    if _archiver_store_deployed(config):
        _stage_archiver_store(
            config, compose_files, env, Path(repo_root), keep_base=keep_archiver_base
        )

    # Same placement and the same reason for ARIEL's store: the schema has to
    # exist before the consumers that read it start, and both deploy paths need
    # it. Ordered AFTER the archiver so the logbook is seeded against a machine
    # whose history is already in place — the two halves of one narrative, in the
    # order they document each other.
    _stage_ariel_store(config, compose_files, env, Path(repo_root))

    if web_terminals_enabled:
        deploy_up_web_terminals(config, compose_files, dev_mode, env, _env_file_args(repo_root))
        log_endpoint_summary(config, compose_files)
        return

    # Pin COMPOSE_PROJECT_NAME so this deploy owns its own compose project (and
    # volume namespace); without it compose derives the project from the first
    # -f file's directory, collapsing every deploy on the host into the shared
    # "services" project whose up/down cross-adopts sibling stacks.
    run_env = runtime_env(config, env)

    base_cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        _env_file_args(repo_root),
    )

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
    logger.debug(f"Running command:\n    {' '.join(rm_cmd)}")
    run_captured(rm_cmd, env=run_env, spool_name="compose-rm", repo_root=repo_root, check=False)
    _report_step("cleared stopped containers")

    if dev_mode:
        # `osprey up --dev` re-bakes the local osprey checkout into a fresh
        # wheel on every run, but compose reuses the cached image tag (e.g.
        # <project>-dispatch:local) unless it is rebuilt — so a dev deploy must build.
        # Build in its OWN step, then `up --no-build`: a single `up --build` can
        # build a local-only tag and then fail container-create with
        # "No such image" under Docker's containerd image store. Non-dev stays a
        # plain `up` so compose's implicit build-on-up still covers a build-only
        # service that has no published upstream tag to pull.
        build_cmd = base_cmd + ["build"]
        logger.debug(f"Running command:\n    {' '.join(build_cmd)}")
        # Watched for the duration of the build and no longer: the live view
        # (and its heartbeats) must go quiet the moment compose returns, before
        # the closing step line below.
        with (report := compose_build_step_reporter()):
            run_captured(
                build_cmd,
                env=run_env,
                spool_name="compose-build",
                repo_root=repo_root,
                on_line=report,
            )
        _report_step("service images built")

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

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    if detached:
        run_captured(cmd, env=run_env, spool_name="compose-up", repo_root=repo_root)
        _report_step("containers started")
        log_endpoint_summary(config, compose_files)
    else:
        # execvpe replaces this process, so the summary must print first —
        # compose's own output follows it.
        log_endpoint_summary(config, compose_files)
        # The terminal belongs to compose from the next line on, and this
        # process will not exist to take anything down: whatever is rendering
        # has to stop here, and the open start phase — which never closes on
        # this path — has to be committed as a permanent line while there is
        # still a process to write it. Unconditional: a no-op on a reporter
        # with nothing to hand over, and inside the phase because a hand-off
        # after it would find nothing left to commit.
        current_reporter().hand_off()
        os.execvpe(cmd[0], cmd, run_env)


def as_built_config_path(repo_root: Path | str) -> Path:
    """The rendered config that describes what this repo runs.

    ``<repo>/build/config.yml``: one place, derived from the zone layout rather
    than searched for, so a stray nested render can never answer for the repo.

    The one answer to "what did the build decide?", so it is what the verbs that
    ACT on the deployment start from and what the verbs that merely REPORT on it
    read — a status that consulted a different file than ``up`` started from
    would describe a deployment nobody is running.
    """
    return Path(repo_root) / BUILD_DIRNAME / "config.yml"


def as_built_compose_files(config: dict, repo_root: Path | str) -> list[str]:
    """The compose files a build left in ``build/``, in ``-f`` order.

    Read, never rendered: this is the whole difference between ``osprey up`` and
    the legacy ``deploy up``. The locations come from the rendered config's own
    ``build_dir`` — resolved against the repo root, which is both the compose
    project directory and the root that config's ``project_root`` names — so
    nothing here hardcodes a layout that the render could disagree with.

    The repo root is handed to the lookup as its explicit ``base`` because
    ``find_existing_compose_files`` resolves ``build_dir`` and each service's
    template directory relatively: every repo-scoped verb must be correct from
    any directory inside the repo, not only from its root. Never answer that by
    moving the working directory instead: that borrows global process state for
    one lookup, and re-anchors every relative path the caller still holds.

    The paths come back RELATIVE to the repo root, which is what
    :func:`~osprey.deployment.compose_generator.compose_base_cmd` resolves for
    each ``-f`` it builds. A caller that OPENS these files itself has no such
    anchor and must join them onto the repo root first — resolving them against
    the working directory finds nothing, and finding nothing looks exactly like
    a deployment that declares nothing.
    """
    from osprey.deployment.compose_generator import find_existing_compose_files

    services = [str(service) for service in (config.get("deployed_services") or [])]
    return find_existing_compose_files(config, services, base=repo_root)


def _published_on_all_interfaces(compose_files: list[str]) -> list[str]:
    """Services in these compose files that publish a port on a wildcard address.

    Read out of the rendered files rather than out of the config, because the
    rendered files are what compose will act on. A build decided each binding
    when it rendered ``deployment.bind_address`` into every ``ports:`` entry, so
    at start time this is a fact to be discovered, not a setting to be applied.
    """
    from osprey.deployment.host_ports import _WILDCARD_HOSTS

    return sorted(
        {
            binding.service
            for binding in parse_host_port_bindings(compose_files)
            if binding.host_ip in _WILDCARD_HOSTS
        }
    )


def _reconcile_exposure(config: dict, compose_files: list[str]) -> bool:
    """Whether this start is reachable off-host, read off what was rendered.

    Exposure is a property of the build. The bind address is baked into every
    ``ports:`` entry the build wrote, and nothing is re-rendered on a start — so
    the only honest answer is the one the rendered files already give, and there
    is no flag that could change it. Making a deployment reachable is
    ``osprey set deployment.bind_address=0.0.0.0`` followed by a build.

    The answer feeds the empty-token refusal in :func:`_ensure_service_tokens`,
    which is why it is computed here rather than inferred there: an exposed
    deployment gets the fail-closed token rules on the strength of what it
    publishes, not on the strength of anything the operator typed.

    Two independent ways to be reachable, and both count. A services binding on
    a wildcard address is the obvious one. The other is the web-terminal stack:
    every service in it runs ``network_mode: host`` and its nginx "binds every
    interface, unrestricted" (``docker-compose.web.yml.j2``), so a web deploy is
    off-host reachable with no published port anywhere in ``compose_files`` —
    which is exactly the case a wildcard-binding check alone would call private.

    Args:
        config: The rendered config, read for whether the web stack is part of
            this deployment.
        compose_files: The services stack's rendered compose files.

    Returns:
        Whether this deployment is reachable off-host, for the token guard.
    """
    wildcard_services = _published_on_all_interfaces(compose_files)
    host_networked = _web_terminals_enabled(config)
    reasons = []
    if wildcard_services:
        reasons.append(f"{', '.join(wildcard_services)} publish on 0.0.0.0")
    if host_networked:
        reasons.append("the web-terminal stack runs on the host network")

    if reasons:
        logger.warning(
            "This deployment is reachable from the network (%s). Starting it under "
            "the fail-closed service-token rules that reachability calls for.",
            "; ".join(reasons),
        )

    return bool(reasons)


def up_as_built(
    repo_root: Path | str,
    *,
    detached: bool = False,
    dev_mode: bool = False,
    keep_archiver_base: bool = False,
    reuse_stores: bool = False,
) -> None:
    """Start a deployment repo from ``build/`` exactly as it was built.

    The three-zone start path. It reads ``build/config.yml`` and the compose
    files beside it and starts them, re-rendering nothing from ``profile.yml``,
    so the services that come up are the ones the last ``osprey build``
    produced and the ones ``osprey status`` will report. The caller has already
    settled whether starting this build is allowed — that is the drift gate in
    the CLI, not a decision this function repeats.

    The web-terminal stack is the documented exception, and it is exactly one
    thing wide. :func:`deploy_up_web_terminals` writes ``docker-compose.web.yml``
    with its nginx config and landing page into ``build/`` on every start.
    Those three artifacts follow the *roster* — a user added or removed since
    the build has to take effect at the next start, which is what makes the
    roster verbs usable without a rebuild — so they are derived here rather than
    at build time.

    Nothing else on this path reads ``profile.yml`` or writes into ``build/``.
    Persona projects in particular are rendered by ``osprey build``, one per
    delta in ``personas/``, and a start only checks they are there
    (:func:`verify_persona_renders`) and refuses when they are not: a start that
    rendered them would put a fresh persona beside a deployment built from an
    older profile. The services stack is untouched by any of it.

    Everything after the inputs are resolved is :func:`_start_stack`. What is
    specific to a deployment repo:

    * the single root ``.env`` is the secret store every provisioner writes, and
      minted secrets are NOT copied anywhere else (there is nowhere else);
    * the image build context is ``<repo>/build``, where the build wrote the
      ``Dockerfile`` — not the repo root, which holds only source;
    * exposure is read off the rendered bindings (see :func:`_reconcile_exposure`).

    Args:
        repo_root: The deployment repo — the directory holding ``profile.yml``.
        detached: Run in detached mode.
        dev_mode: Stage the local osprey checkout into the images.
        keep_archiver_base: Leave a mismatched archiver base in place instead
            of rebuilding it (see :func:`_stage_archiver_store`).

    Raises:
        NoRenderedBuildError: When ``build/`` holds no rendered config to start.
        RuntimeError: From the preflights, including the ``--dev`` one.
    """
    repo_root = Path(repo_root)
    config, compose_files, exposed = _resolve_as_built_inputs(repo_root, dev_mode=dev_mode)
    _start_as_built(
        repo_root,
        config,
        compose_files,
        exposed,
        detached=detached,
        dev_mode=dev_mode,
        keep_archiver_base=keep_archiver_base,
        reuse_stores=reuse_stores,
    )


def _sidecar_dev_flavor(compose_files: list[str], repo_root: Path) -> tuple[list[str], list[str]]:
    """Split the rendered compose files with framework-installing image builds by flavor.

    Dev-ness is a property of the RENDER: ``osprey build --dev`` stages a wheel
    from the local checkout into each service build context and emits the
    ``OSPREY_DEV`` build arg; a plain build emits the pinned
    ``osprey-framework==<version>`` install and nothing else. The rendered YAML
    is therefore the single source of truth for which flavor ``build/`` holds —
    read here rather than kept in side-channel bookkeeping that could disagree
    with it.

    Only services that install the framework at image-build time matter (their
    ``build.args`` carry ``OSPREY_VERSION``). Pure-image services and images
    built at start time (dispatch worker, personas) take their dev-ness from the
    start verb and are none of this function's business.

    Args:
        compose_files: The rendered compose files, as returned by
            :func:`as_built_compose_files`. Relative entries — the pinned
            invocation contract spells them repo-relative — are resolved
            against *repo_root*, not the working directory, so the answer does
            not depend on where the caller stands.
        repo_root: The deployment repo the relative entries anchor on.

    Returns:
        ``(pinned, dev)`` — the compose files whose framework-installing builds
        were rendered without / with the dev flavor. A file that cannot be read
        or parsed lands in neither list; the compose invocation that follows
        will surface it on its own terms.
    """
    pinned: list[str] = []
    dev: list[str] = []
    for path in compose_files:
        file = Path(path)
        if not file.is_absolute():
            file = repo_root / file
        try:
            doc = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        services = doc.get("services") or {}
        if not isinstance(services, Mapping):
            continue
        for service in services.values():
            build = (service or {}).get("build") if isinstance(service, Mapping) else None
            args = build.get("args") if isinstance(build, Mapping) else None
            if isinstance(args, Mapping) and "OSPREY_VERSION" in args:
                (dev if str(args.get("OSPREY_DEV", "")) == "1" else pinned).append(path)
                break
    return pinned, dev


def _resolve_as_built_inputs(repo_root: Path, *, dev_mode: bool) -> tuple[dict, list[str], bool]:
    """Read what a start would run, and refuse before anything is touched.

    Every decision a start makes from the filesystem alone: whether there is a
    build, whether ``--dev`` can be honored, what compose files the build left,
    and whether this deployment is reachable off-host
    (:func:`_reconcile_exposure`). Nothing here starts, stops or writes
    anything, which is what lets :func:`restart_deployment` call it *first* —
    a restart that is going to be refused must be refused while the stack it
    would have stopped is still running.

    Args:
        repo_root: The deployment repo.
        dev_mode: Whether the local osprey checkout is to be staged.

    Returns:
        The rendered config, its compose files in ``-f`` order, and whether the
        deployment is reachable off-host.

    Raises:
        NoRenderedBuildError: When ``build/`` holds no rendered config, or none
            of the services it declares was rendered.
        DevModeUnavailableError: When ``--dev`` was asked of a render that holds
            pinned (non-dev) sidecar image builds — `up` never re-renders, so
            honoring it is impossible without a dev build.
        RuntimeError: From the ``--dev`` preflight.
    """
    config_path = as_built_config_path(repo_root)
    if not config_path.is_file():
        raise NoRenderedBuildError(
            reason=f"No build found at {config_path.parent}.",
            remedy="Render one:\n    osprey build",
        )

    if dev_mode:
        # The same fail-before-any-work check the rendering path gets from
        # prepare_compose_files. Nothing re-renders here, so it has to be made
        # explicitly — and it must stay first: a --dev deploy that cannot stage
        # a wheel comes up successfully running the released code instead.
        from osprey.deployment.wheel_build import preflight_dev_mode

        preflight_dev_mode()

    config = load_project_config(str(config_path), wrap_errors=True)

    # Before anything below can matter: a deployment whose archive is fabricated
    # does not get to start. build/config.yml is what the last build wrote, but
    # nothing stops a hand-edit between build and start, and a start is where
    # the pairing becomes a running stack other people trust. Hosted here so
    # both callers — `osprey up` and `osprey restart` — refuse identically,
    # before the restart's `down` has stopped anything.
    _refuse_invented_history(config)

    compose_files = as_built_compose_files(config, repo_root)

    if not compose_files and config.get("deployed_services"):
        raise NoRenderedBuildError(
            reason=(
                f"{config_path} declares services "
                f"({', '.join(str(s) for s in config['deployed_services'])}) but no compose "
                f"files were rendered for them under {repo_root / BUILD_DIRNAME}."
            ),
            remedy="Re-render build/:\n    osprey build",
        )

    # The render's flavor must match the start's. `up` never re-renders, so a
    # `--dev` start of a pinned render would build sidecar images from contexts
    # that hold no wheel and no OSPREY_DEV arg — the containers would come up
    # running the pinned release (or fail on an unreleased pin) with nothing
    # saying the local checkout was never involved. Same failure mode
    # DevModeUnavailableError exists for, so it refuses the same way.
    pinned, dev_flavored = _sidecar_dev_flavor(compose_files, repo_root)
    if dev_mode and pinned:
        names = ", ".join(sorted(Path(p).parent.name for p in pinned))
        raise DevModeUnavailableError(
            reason=(
                f"build/ was rendered without --dev, and `up` never re-renders. "
                f"The {names} image build(s) would install the pinned release "
                f"instead of this checkout."
            ),
            remedy=(
                "Render a dev build first:\n"
                "    osprey build --dev\n"
                "or chain it in one step:\n"
                "    osprey up --build --dev"
            ),
        )
    if not dev_mode and dev_flavored:
        # The mirror case starts fine — `up` starts what was built, and what
        # was built is a dev render — but the operator should know the images
        # bake a local checkout, not the published release.
        logger.warning(
            "build/ is a dev render (osprey build --dev): the service images "
            "bake in the local osprey checkout, not the published release. "
            "Run `osprey build` for a release render."
        )

    return config, compose_files, _reconcile_exposure(config, compose_files)


def _start_as_built(
    repo_root: Path,
    config: dict,
    compose_files: list[str],
    exposed: bool,
    *,
    detached: bool,
    dev_mode: bool,
    keep_archiver_base: bool = False,
    reuse_stores: bool = False,
) -> None:
    """Start already-resolved as-built inputs, with a deployment repo's rules.

    The second half of :func:`up_as_built`, split out because
    :func:`restart_deployment` runs a ``down`` between the two halves and must
    reach this one without re-reading (and re-warning about) the build. What
    makes it repo-specific rather than generic is the two arguments it pins on
    :func:`_start_stack`: the repo's single ``.env``, and the build directory as
    the image build context.

    The whole start runs with this repo's render anchored as the process config.
    A start is not only a sequence of ``compose`` invocations — it synthesizes
    and seeds an archive in THIS process, and that work reads the config through
    unqualified lookups. Compose hands every container ``CONFIG_FILE``; without
    the anchor the host is the one participant in that contract left resolving
    against a working directory that holds no ``config.yml`` at all, and its
    lookups degrade to defaults without failing. See
    :func:`~osprey.utils.config.config_anchored_at`.
    """
    with config_anchored_at(as_built_config_path(repo_root)):
        _start_stack(
            config,
            compose_files,
            repo_root,
            detached=detached,
            dev_mode=dev_mode,
            expose_network=exposed,
            # Explicit, never the cwd-relative default: the token mint writes
            # real secrets, and every compose invocation on this path reads
            # `<repo>/.env` with --env-file. A mint that landed anywhere else
            # would leave the stack starting with its fail-closed tokens unset —
            # secure looking, and silent.
            env_path=repo_root / COMPOSE_ENV_FILENAME,
            build_context=container_image_context(repo_root, resolve_project_name(config)),
            keep_archiver_base=keep_archiver_base,
            reuse_stores=reuse_stores,
        )


def _repo_label_filter(repo_root: Path) -> str:
    """The runtime ``--filter`` selecting only this checkout's resources.

    ``label=com.osprey.repo-id=<identity>``, derived through
    :func:`~osprey.deployment.compose_generator.repo_identity` — the single
    spelling of that identity, shared with the render that baked it in.
    """
    return f"label={REPO_ID_LABEL}={repo_identity(repo_root)}"


def _down_by_label(config: dict | None, repo_root: Path) -> None:
    """Stop this repo's containers by label, when ``build/`` cannot say which.

    The recovery path for a deployment whose ``build/`` was wiped (or never
    re-rendered) while its stack was running. Without it those containers are
    unreachable by any OSPREY verb: ``compose down`` needs the compose files
    that declared them, and they are exactly what is missing.

    What it removes and what it leaves:

    * containers labelled for THIS checkout only — stopped, then removed, so a
      published host port is released the way a ``compose down`` releases it;
    * volumes are kept, exactly as a ``compose down`` keeps them. Destroying
      state stays ``osprey reset``'s job;
    * the compose network is left in place. Nothing labels a network with the
      repo identity, so it cannot be selected here; the next ``osprey up``
      reuses it.

    Honest limit, and it is the reason this function says so out loud: labels
    are applied at container CREATE time. A stack started before OSPREY labelled
    its containers carries no ``com.osprey.repo-id`` at all and is invisible
    here — it stays running, and nothing about that is inferable from a silent
    "done".

    Args:
        config: The rendered config when there is one, for runtime selection
            only. ``None`` when ``build/`` holds no config at all, which is the
            case this path most exists for.
        repo_root: The deployment repo whose containers are to be stopped.

    Raises:
        RuntimeError: When the runtime cannot list, stop or remove them.
    """
    # get_runtime_command answers with a COMPOSE argv (["docker", "compose"]);
    # the label sweep is plain-runtime work, so it takes the binary off the
    # front rather than resolving a runtime a second way.
    runtime = get_runtime_command(config)[0]
    label_filter = _repo_label_filter(repo_root)

    listing = subprocess.run(
        [runtime, "ps", "-aq", "--filter", label_filter],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(
            f"Could not list this deployment's containers: `{runtime} ps` exited "
            f"{listing.returncode}. {(listing.stderr or '').strip()}"
        )

    container_ids = listing.stdout.split()
    if not container_ids:
        logger.key_info(
            "Nothing to stop. There are no compose files in %s to run a `down` from, and "
            "no container is labelled %s for this repo. Containers get that label when "
            "they are CREATED, so a stack started before this repo's containers were "
            "labelled is not visible here and may still be running — `osprey build` "
            "restores build/, after which `osprey down` works normally.",
            repo_root / BUILD_DIRNAME,
            label_filter.removeprefix("label="),
        )
        return

    logger.key_info(
        "No compose files in %s — stopping the %d container(s) labelled %s instead. "
        "Volumes are kept, as they are by a compose `down`; the compose network is not "
        "labelled and stays.",
        repo_root / BUILD_DIRNAME,
        len(container_ids),
        label_filter.removeprefix("label="),
    )

    # stop, then rm: the same two steps, in the same order, that `compose down`
    # performs — so containers get their shutdown grace period rather than the
    # SIGKILL a bare `rm -f` would deliver.
    for verb in ("stop", "rm"):
        cmd = [runtime, verb, *container_ids]
        logger.debug(f"Running command:\n    {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"`{runtime} {verb}` failed (exit {result.returncode}) — this "
                f"deployment's containers may still be running. "
                f"{(result.stderr or '').strip()}"
            )

    # The recovery path reports what it did for the same reason the compose path
    # does: without it, a `down` that fell back to labels closes its phase with a
    # bare ✓ and never says that anything was found, let alone how much.
    _report_step(f"removed {len(container_ids)} labelled container(s)")


def down_deployment(repo_root: Path | str) -> None:
    """Stop a deployment repo's stack, keeping every volume.

    The three-zone stop path, and the mirror of :func:`up_as_built`: it acts on
    what a build left in ``build/`` and renders nothing. Where the legacy
    :func:`deploy_down` re-renders the compose files when it cannot find them,
    this refuses to render at stop time — the compose files that declared
    the running containers are a fact about the past, and re-deriving them from
    a since-edited profile can only produce a ``down`` aimed at a stack that was
    never started. :func:`_down_by_label` covers that case instead.

    Order matters, and it is web-stack-first. A web-terminal deployment runs two
    compose invocations against one project, so the services ``down`` does not
    carry ``docker-compose.web.yml`` in its ``-f`` list and would leave the web
    containers running — holding the host-global container names
    (``<prefix>-nginx``, ``<prefix>-web-<user>``) that the next web deploy on
    this host, from any project, then collides with.

    Volumes are never removed, on either path. Per-user terminal state, the
    databases, the artifact store: all of it survives a ``down`` and is
    ``osprey reset``'s to destroy, deliberately and with a confirmation.

    Unlike the legacy path this does not ``execvpe``-replace the process: the
    label fallback runs after the compose invocation on the same call, and
    :func:`restart_deployment` needs to still be here afterwards to start the
    stack again.

    Args:
        repo_root: The deployment repo — the directory holding ``profile.yml``.

    Raises:
        RuntimeError: When the compose ``down``, or the label sweep standing in
            for it, fails.
    """
    repo_root = Path(repo_root)
    config_path = as_built_config_path(repo_root)
    config = (
        load_project_config(str(config_path), wrap_errors=True) if config_path.is_file() else None
    )
    compose_files = as_built_compose_files(config, repo_root) if config else []

    if config and _web_terminals_enabled(config):
        deploy_down_web_terminals(config, os.environ.copy(), _env_file_args(repo_root))

    if not compose_files:
        _down_by_label(config, repo_root)
        return

    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        _env_file_args(repo_root),
    )
    cmd.append("down")

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    # Captured, not streamed: the phase line this runs under is only readable if
    # compose's own output goes to the spool instead of the same terminal. The
    # pinned COMPOSE_PROJECT_NAME reaches compose through `env=` — unpinned,
    # `down` targets the shared "services" project rather than this deploy's.
    result = run_captured(
        cmd,
        env=runtime_env(config, os.environ.copy()),
        spool_name="compose-down",
        repo_root=repo_root,
        check=False,
    )
    if getattr(result, "returncode", 0):
        # check=False and raise here, rather than letting run_captured raise:
        # a failed `down` has a specific consequence to state. The reporter
        # already holds the spool path, so the failure line replays it.
        raise RuntimeError(
            f"`compose down` failed (exit {result.returncode}) — this deployment's "
            "containers may still be running."
        )
    _report_step("containers stopped")


def restart_deployment(
    repo_root: Path | str,
    *,
    detached: bool = False,
    dev_mode: bool = False,
    keep_archiver_base: bool = False,
    reuse_stores: bool = False,
) -> None:
    """Stop this deployment and start it again from ``build/``.

    ``down`` then ``up``, not ``compose restart``. The difference is the whole
    point of the change: ``compose restart`` restarts each container against the
    definition it was CREATED with, so a rebuilt image, an edited compose file
    or a newly minted token reaches nothing until something recreates the
    container. This recreates them, which is what an operator who typed
    ``restart`` after a build is asking for.

    It starts ``build/`` as built, and re-renders nothing from ``profile.yml``,
    exactly like :func:`up_as_built` — including the one documented exception:
    a web-terminal deployment re-renders that stack's compose file, nginx
    config and landing page at every start, because those three follow the user
    roster rather than the build. The drift gate lives in the CLI, ahead of this
    call.

    The order inside is deliberate. Everything the start would refuse is read
    and settled BEFORE the ``down``, so a restart that cannot start does not
    stop the running stack on its way to saying so.

    Service tokens are re-minted on the way back up, into the repo's own
    ``.env`` — the same file every compose invocation is pointed at — because
    the start half is :func:`_start_as_built` unchanged.

    Args:
        repo_root: The deployment repo.
        detached: Run in detached mode.
        dev_mode: Stage the local osprey checkout into the images.
        keep_archiver_base: Leave a mismatched archiver base in place instead
            of rebuilding it (see :func:`_stage_archiver_store`).

    Raises:
        NoRenderedBuildError: When ``build/`` holds nothing to start. Nothing is
            stopped.
        RuntimeError: From the preflights or from the stop itself.
    """
    repo_root = Path(repo_root)
    config, compose_files, exposed = _resolve_as_built_inputs(repo_root, dev_mode=dev_mode)

    # BEFORE the stop, unlike every other start path, and that ordering is the
    # whole point: `down` removes the store containers, and a removed container
    # takes with it the only host-side copy of the credential its data volume
    # was initialized with. Run at the usual point — after the stop, inside
    # `_start_as_built` — this check would report every stale volume as
    # unrecoverable, having itself destroyed the evidence moments earlier.
    # Nothing is minted yet here, so the set is the one a mint *would* generate.
    env_path = repo_root / COMPOSE_ENV_FILENAME
    _preflight_stale_store_volumes(
        config,
        _volume_initialized_vars_that_would_be_minted(config, env_path),
        env_path,
        reuse_stores=reuse_stores,
    )

    down_deployment(repo_root)
    _start_as_built(
        repo_root,
        config,
        compose_files,
        exposed,
        detached=detached,
        dev_mode=dev_mode,
        keep_archiver_base=keep_archiver_base,
        reuse_stores=reuse_stores,
    )


def deploy_down(config_path, dev_mode=False):
    """Stop services using container runtime (Docker or Podman).

    When ``modules.web_terminals.enabled`` is set, the web-terminal stack is
    torn down first via its own compose invocation
    (:func:`osprey.deployment.web_terminals.provision.deploy_down_web_terminals`):
    the two stacks are separate invocations of one pinned base, and the
    services ``down`` ends this process, so the web ``down`` must happen
    before it.

    :param config_path: Path to the configuration file
    :type config_path: str
    :raises SystemExit: Always — with the ``compose down`` child's own exit code.
    """
    config = load_project_config(config_path, wrap_errors=True)
    repo_root = resolve_repo_root(config, config_path)

    if _web_terminals_enabled(config):
        deploy_down_web_terminals(config, os.environ.copy(), _env_file_args(repo_root))

    deployed_services = config.get("deployed_services", [])
    deployed_service_names = (
        [str(service) for service in deployed_services] if deployed_services else []
    )

    # Try to use existing compose files (suppress warnings for status check)
    from osprey.deployment.compose_generator import find_existing_compose_files

    compose_files = find_existing_compose_files(
        config, deployed_service_names, quiet=True, base=repo_root
    )

    # If no existing compose files found, rebuild them
    if not compose_files:
        logger.info("No existing compose files found, rebuilding...")
        _, compose_files = prepare_compose_files(config_path, dev_mode)
    else:
        logger.info("Using existing compose files for 'down' operation:")
        for f in compose_files:
            logger.info(f"  - {f}")

    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)), compose_files, repo_root
    )
    cmd.append("down")

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    # The COMPOSE_PROJECT_NAME pin must reach compose: `down` must target the
    # same project `up` created, or it either misses this deploy's containers
    # or (unpinned) tears down the shared "services" project. That pin is why
    # this used to `execvpe` rather than `execvp`; the captured run carries the
    # same environment through `env=`.
    run_env = runtime_env(config, os.environ.copy())
    # check=False, then exit on the child's own code: `down` is the last thing
    # this process does, and propagating the code verbatim keeps the exit
    # status callers see identical to the exec'd child's.
    proc = run_captured(
        cmd, env=run_env, spool_name="compose-down", repo_root=repo_root, check=False
    )
    if proc.returncode == 0:
        _report_step("containers stopped")
    # A signal-killed child reports a NEGATIVE returncode, which sys.exit would
    # mask to its low byte (-15 -> 241). Shells spell "killed by signal N" as
    # 128+N, and 128+N is exactly what the exec'd child's own status used to be.
    sys.exit(proc.returncode if proc.returncode >= 0 else 128 - proc.returncode)


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

    # config.yml is bind-mounted, so a restart is how an edit takes effect —
    # including an edit into the pairing deploy_up refuses. Same guard, same
    # reason. (`rebuild_deployment` reaches it through deploy_up.)
    _refuse_invented_history(config)

    # Verify container runtime is actually running
    is_running, error_msg = verify_runtime_is_running(config)
    if not is_running:
        raise RuntimeError(error_msg)

    # Honor the same fail-closed/expose guard as deploy_up when re-rendering with
    # a (possibly newly exposed) bind address.
    #
    # ANCHORED, deliberately: this mints REAL SECRETS, and without an explicit
    # env_path it falls back to a cwd-relative `.env`. Run from anywhere but the
    # repo root that wrote tokens into a stray file — where the operator would
    # never find them and the next deploy would mint different ones — and the
    # containers pinned to the originals would not come back up.
    _ensure_service_tokens(
        config, expose_network, resolve_repo_root(config, config_path) / COMPOSE_ENV_FILENAME
    )

    # Same reason the token mint above runs on this path: the bluesky compose
    # template expands the manager's private key with a `:?` guard, so an unset
    # value aborts the whole `compose restart` rather than degrading. A project
    # whose .env carries no key must not be broken by a restart. The
    # document-plane certificates are NOT
    # provisioned here -- they are bind-mount sources, and `compose restart`
    # reuses each container's existing config rather than re-resolving mounts,
    # so generating them would change nothing until the next `osprey up`.
    # Anchored for the same reason as the token mint above — this mints a
    # keypair, and a cwd-relative `.env` would strand it.
    _ensure_bluesky_control_plane_keys(
        config, resolve_repo_root(config, config_path) / COMPOSE_ENV_FILENAME
    )

    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        resolve_repo_root(config, config_path),
    )
    cmd.append("restart")

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
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
