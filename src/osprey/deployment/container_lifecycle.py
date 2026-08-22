"""Container lifecycle operations — start, stop, restart, and rebuild.

Manages the lifecycle of containerized service deployments using
Docker or Podman compose.
"""

import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple

import yaml

from osprey.cli import output
from osprey.cli.phase_reporter import current_reporter
from osprey.cli.phase_reporter import report_group as _report_group
from osprey.cli.phase_reporter import report_step as _report_step
from osprey.deployment.build_progress import BuildModel, with_plain_build_progress
from osprey.deployment.compose_generator import (
    COMPOSE_ENV_FILENAME,
    REPO_ID_LABEL,
    _copy_local_framework_for_override,
    clean_deployment,
    compose_base_cmd,
    compose_provider_env,
    prepare_compose_files,
    repo_identity,
    resolve_image_defaults,
    resolve_project_name,
    resolve_repo_root,
)
from osprey.deployment.deploy_summary import log_endpoint_summary
from osprey.deployment.errors import (
    ArchiverClientMissingError,
    ComposeInterpolationError,
    DevModeUnavailableError,
    NoRenderedBuildError,
)
from osprey.deployment.graphdb_service import (
    GRAPHDB_SEED_COMMAND,
    GRAPHDB_SERVICE_NAME,
    preflight_graphdb_config,
)
from osprey.deployment.host_ports import (
    find_port_conflicts,
    format_conflict_report,
    parse_host_port_bindings,
)
from osprey.deployment.qmd_service import preflight_qmd_models_dir
from osprey.deployment.runtime_helper import (
    ComposeProvider,
    UnsupportedComposeProviderError,
    detect_compose_provider,
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
    _is_forbidden_value,
    _raise_forbidden_var,
    _raise_invalid_var,
    _validate_openobserve_password,  # noqa: F401  (re-exported for tests)
    _validate_var,
)
from osprey.deployment.staleness import BUILD_DIRNAME, warn_if_project_stale
from osprey.deployment.subprocess_capture import run_captured
from osprey.deployment.web_terminals.env_production import migrate_users_env
from osprey.deployment.web_terminals.provision import (
    deploy_down_web_terminals,
    deploy_up_web_terminals,
    preflight_web_terminals,
)
from osprey.deployment.wheel_build import _staged_dev_artifact_paths
from osprey.utils.config import config_anchored_at, load_project_config
from osprey.utils.dotenv import (
    ENV_CHAIN_FILENAMES,
    ENV_LOCAL_FILENAME,
    ENV_SHARED_FILENAME,
    atomic_write,
    chain_files,
    compose_unsafe_vars,
    compose_unsafe_vars_in_chain,
    merge_chain,
    parse_dotenv_file,
    write_env_merged,
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
#
# graphdb is the Neo4j knowledge-graph store, and follows the openobserve
# rationale with one wrinkle the other three do not have: its compose template
# passes credentials as the COMPOSITE ``NEO4J_AUTH: "neo4j/${GRAPHDB_PASSWORD:-…}"``,
# so the minted value is only the password half. Left alone the store comes up
# on the template's published default, guarding the facility knowledge graph
# the agent answers from. The minted value reaches two readers from this one
# ``.env`` entry — the container, via that composite, and the agent's
# connection resolver, whose ``services.graphdb`` block names the variable
# rather than carrying a value (the archiver's ``password_env`` pattern, not
# the postgres embedded-DSN one, because the neo4j driver takes auth
# separately as ``driver(uri, auth=(user, password))``).
# NOTE: neo4j, like postgres and mongod, adopts its initial password only when
# initializing a fresh data volume — the same stale-volume caveat applies.
_SERVICE_TOKEN_VARS: dict[str, tuple[str, ...]] = {
    "event_dispatcher": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "dispatch_worker": ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"),
    "bluesky": ("BLUESKY_LAUNCH_TOKEN", "BLUESKY_TILED_API_KEY"),
    "openobserve": ("ZO_ROOT_USER_PASSWORD",),
    "postgresql": ("ARIEL_DB_PASSWORD",),
    "mongodb": ("MONGO_ROOT_PASSWORD",),
    "graphdb": ("GRAPHDB_PASSWORD",),
}

# Non-secret settings written into ``.env`` for a deployed service, as
# ``service -> ((var, default), ...)``. Keyed by ``deployed_services`` membership
# exactly like ``_SERVICE_TOKEN_VARS``, and never overwriting a value that is
# already there — but a DIFFERENT concern, which is why it is a different map.
#
# ``_SERVICE_TOKEN_VARS`` declares machine-generated secrets, and every name it
# reaches is minted through ``_VAR_GENERATORS`` under that module's CSPRNG bar.
# A constant is not a secret and has no recipe, so registering one there would
# mean writing a generator that returns a fixed string — a value the bar has no
# meaning for, sitting in the map an operator reads to learn what osprey
# considers secret.
#
# The reason to write a default down at all is discoverability. Both halves of
# the OpenObserve login are needed to reach the store, and the email previously
# existed only as a ``:-`` fallback inline in the compose template and the
# rendered config.yml. An operator looking for their credentials found a
# password in ``.env``, no email anywhere, and nothing naming the file that
# would have told them. Writing the default makes ``.env`` the whole answer,
# and — because it is the same value the ``:-`` fallback resolves to — records
# the login rather than changing it.
_SERVICE_DEFAULT_VARS: dict[str, tuple[tuple[str, str], ...]] = {
    "openobserve": (
        # Must stay equal to the ``${ZO_ROOT_USER_EMAIL:-…}`` fallback in
        # ``osprey/templates/services/openobserve/docker-compose.yml.j2``; the
        # mint tests parse the template and pin the two together.
        ("ZO_ROOT_USER_EMAIL", "root@example.com"),
        # The telemetry INGEST account: the identity the agent authenticates as
        # when it ships telemetry, so root's credentials never travel into a
        # rendered ``config.yml`` or a web-terminal environment. An account
        # name, not a secret — written down for exactly the discoverability
        # reason the root email is, and never overwritten when an operator
        # names their own.
        #
        # Unlike the root email this one has NO ``:-`` fallback in the
        # openobserve compose template, and must not grow one. The store reads
        # ``ZO_ROOT_USER_*`` to initialize itself and knows nothing about this
        # account until the post-start provisioner creates it through the API;
        # the fallback that matters belongs to the *agent's* telemetry config,
        # which is where a reader of ``config.yml`` looks for it.
        ("ZO_INGEST_USER_EMAIL", "ingest@example.com"),
    ),
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
#
# GRAPHDB_PASSWORD earns its place on a SECOND ground, which is why this is a
# set of two rather than a set of ARIEL_DSN-shaped vars. It does have a
# _SERVICE_TOKEN_VARS entry — but only under the "graphdb" key, so the mint and
# the validation above reach it only on a deploy that runs its own graph store.
# The other supported shape is an EXTERNAL store: an explicit
# `services.graphdb.uri` pointing at a Neo4j this deploy does not run, with
# "graphdb" absent from `deployed_services`. On that path the operator writes
# the password into the project `.env` themselves, and it is precisely the
# operator-supplied values — never the minted ones — that the format constraint
# exists to catch. Without this entry the var would be invisible to the deploy
# boundary in exactly the configuration where it is most likely to be malformed.
#
# Membership in both sets is deliberate and costs nothing: on a deploy that
# does run graphdb the validate-only loop re-checks an already-checked value
# with the same pure validator and the same pure forbidden-value lookup. Both
# loops ask both questions in the same order, so which one reaches a given
# value first does not change the diagnosis the operator reads.
_VALIDATE_ONLY_VARS: set[str] = {"ARIEL_DSN", "GRAPHDB_PASSWORD"}


class VolumeInitializedStore(NamedTuple):
    """Where one store's identity lives on the host, for the stale-volume check.

    Five names that must agree with the service's compose template, which
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
      which is not the ``.env`` var name for most of these stores
      (``MONGO_ROOT_PASSWORD`` arrives as ``MONGO_INITDB_ROOT_PASSWORD``,
      ``ARIEL_DB_PASSWORD`` as ``POSTGRES_PASSWORD``);
    * ``cred_prefix`` — the fixed text the template puts *in front of* the
      ``.env`` value when it builds ``cred_env``, stripped back off on the way
      in. Empty for every store whose container variable holds the credential
      and nothing else; ``"neo4j/"`` for graphdb, whose ``NEO4J_AUTH`` is a
      ``<user>/<password>`` composite. Without the strip, the harvested value
      would carry the prefix while ``.env`` does not, and the two places that
      consume it would both be wrong in the same direction: the staleness
      comparison would call every healthy graphdb volume stale, and
      ``--reuse-stores`` would then "restore" ``neo4j/<password>`` into
      ``GRAPHDB_PASSWORD``, re-composing to ``neo4j/neo4j/<password>`` and
      locking the operator out of a store that was working.
    """

    service: str
    volume: str
    container: str
    cred_env: str
    cred_prefix: str = ""


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
    # Neo4j reads NEO4J_AUTH only while initializing an empty /data, and that
    # volume carries the server's auth store — so it belongs here for the same
    # reason the three above do. What is different is only the spelling: the
    # template composes `neo4j/${GRAPHDB_PASSWORD:-…}`, so the container-side
    # value is the bare password with a fixed user prefix on it, which
    # `cred_prefix` takes back off. The plugins volume is deliberately NOT
    # named: it holds a downloaded jar, carries no credential, and a store is
    # listed here for the volume whose survival makes a credential unusable.
    "GRAPHDB_PASSWORD": VolumeInitializedStore(
        service="graphdb",
        volume="graphdb_data",
        container="-graphdb",
        cred_env="NEO4J_AUTH",
        cred_prefix="neo4j/",
    ),
}


class StoreIssuedCredential(NamedTuple):
    """Where one credential this deploy does NOT choose comes from.

    Two names, both of which a provisioner needs and neither of which any other
    registry in this module records:

    * ``service`` — the ``deployed_services`` key whose store issues the
      credential, and the name shown to the operator;
    * ``identity_var`` — the ``.env`` variable naming the *account* the secret
      belongs to. A store-issued secret is meaningless without the identity it
      was issued for: both halves are presented together on every request, and
      a provisioner asking the store for "the token" has to say whose.
    """

    service: str
    identity_var: str


#: Secrets the STORE issues and this deploy only records — the inverse of every
#: other credential registry in this module, and the reason it is a registry of
#: its own rather than an entry in one of them.
#:
#: ``_SERVICE_TOKEN_VARS`` + ``_VAR_GENERATORS`` say "osprey chooses this value
#: before the stack starts". ``_VOLUME_INITIALIZED_VARS`` says "osprey chose it,
#: and a surviving volume may not have adopted it". Neither sentence is true
#: here: OpenObserve mints a service account's token itself (a 16-character
#: alphanumeric string, read back with ``GET /api/{org}/service_accounts/{email}``
#: once the container is live), so the value simply does not exist at the moment
#: every other credential in this file is written.
#:
#: What each of the three obvious registrations would actually do, since a
#: reader's instinct is that one of them must fit:
#:
#: * ``_VAR_GENERATORS`` would FABRICATE a token the store never issued. The
#:   ``.env`` would look perfectly healthy and every telemetry request would
#:   401 — the failure this registry exists to make impossible.
#: * ``_VAR_VALIDATORS`` grades osprey's own recipes. ``_validate_openobserve_password``
#:   (8–128 characters, four character classes) rejects a legitimate
#:   16-character alphanumeric server token outright, so registering it would
#:   refuse every working deploy; and a value the store chose is not osprey's
#:   to grade in the first place.
#: * ``_VOLUME_INITIALIZED_VARS`` misfires twice, either half of it fatally.
#:   Its harvest reads the credential out of the store container's *environment*
#:   (``cred_env``), and a service-account token was never in any container's
#:   environment — it lives inside the volume's own database — so
#:   ``_harvest_original_credential`` would answer ``None`` forever and
#:   ``--reuse-stores`` would report an adoptable store as unrecoverable. Worse,
#:   ``_volume_initialized_vars_that_would_be_minted`` predicts a mint for every
#:   registered var carrying no non-empty value, and ``deploy_restart`` hands
#:   that prediction to ``_preflight_stale_store_volumes`` as ``minted`` — so a
#:   restart with a surviving ``openobserve`` volume and a not-yet-harvested
#:   token would abort as stale, over a credential that is absent exactly as
#:   designed.
#:
#: What the registration IS for: this name reaches ``.env`` from a deploy-time
#: writer, just later in the run than the others, and that is precisely what
#: ``_deploy_written_env_vars`` has to know. Being in the census makes a
#: ``env.pinned`` declaration on it refuse (``_preflight_pinned_writers`` — the
#: deploy writes it, so no chain can own it), and stops the required-env probe
#: from telling an operator to supply a value only the store can produce.
_STORE_ISSUED_VARS: dict[str, StoreIssuedCredential] = {
    "ZO_INGEST_SA_TOKEN": StoreIssuedCredential(
        service="openobserve",
        identity_var="ZO_INGEST_USER_EMAIL",
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


def _report_fact(message: str, /, *, wrote: tuple[str, object] | None = None) -> None:
    """Report ``message`` under this module's logger.

    The promotion contract lives in :func:`osprey.cli.output.report_fact`; this
    binds it to the lifecycle logger so call sites pass the line alone.

    Args:
        message: The finished inline line, built by the caller.
        wrote: The closing-ledger row, passed through as
            :func:`osprey.cli.output.report_fact` documents it.
    """
    output.report_fact(logger, message, wrote=wrote)


def _warn_fact(summary: str, detail: str | None = None, remedy: str | None = None, /) -> None:
    """Warn under this module's logger, in the run's own column.

    The promotion contract lives in :func:`osprey.cli.output.warn_fact`; this
    binds it to the lifecycle logger so call sites pass the words alone.

    Args:
        summary: What went sideways, in one line.
        detail: What an operator needs in order to judge it.
        remedy: The one thing to do about it, if there is an honest one.
    """
    output.warn_fact(logger, summary, detail, remedy)


#: What the knob diff below is a diff of. Named once because both halves of the
#: report -- the block the operator reads and the record the sinks keep -- open
#: with it, and a heading that said two different things would read as two
#: different findings.
_KNOB_DIFF_TITLE = "The archive's knobs changed since it was seeded:"


def _report_knob_diff(comparison) -> None:
    """Print the archive's knob diff, and keep its record for the log sinks.

    A knob diff is a block of facts about one thing, which is the shape
    :func:`osprey.cli.output.section` renders: the heading, then one indented
    ``knob   stored -> expected`` row per field that moved. The rows are built
    from the comparison's own ``differences`` rather than from its one-line
    ``describe()``, so the column the operator scans is a real column instead of
    a sentence that happens to line up.

    The record stays exactly what it was -- the heading and ``describe()``, at
    the level they always had -- so a log file still reads the way it read
    before. The altitude gate drops both on a normal run, which is what keeps
    the printed block from arriving twice.

    Args:
        comparison: The seeder's fingerprint verdict, carrying the
            ``(key, stored, expected)`` triples for the fields that moved.
    """
    output.section(
        _KNOB_DIFF_TITLE,
        [(key, f"{stored!r} -> {expected!r}") for key, stored, expected in comparison.differences],
    )
    logger.key_info(_KNOB_DIFF_TITLE)
    logger.key_info(comparison.describe())


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
    services present) unset — or present with an empty value — in BOTH the
    process env and that ``.env``,
    generate a strong random value (``_generate_token``: ``token_urlsafe(32)``
    unless the var registers a different alphabet in ``_VAR_GENERATORS``) and
    append it to ``.env``
    (``chmod 0o600``, matching the build-time convention). Existing non-empty
    values are never overwritten, so re-running ``osprey up`` is idempotent.
    No-op unless a token-requiring service is actually deployed.

    **An empty value is not a decision here.** Osprey's general rule is that an
    explicitly empty variable is the operator saying *leave this alone* — a
    blank ``NO_PROXY=`` is a real setting, and a missing line is not the same
    thing. These vars are the deliberate exception, because every name this loop
    can reach is a machine-generated secret with no meaningful empty state. An
    empty one does not switch a service's authentication off; it runs the
    service with no authentication at all, and for the three store passwords it
    hands the container back the published default its compose template
    interpolates with ``:-``. Nothing is expressible as ``TOKEN=`` that leaving
    the var unset does not express better, so the operator gives up nothing and
    a deploy stops coming up unauthenticated with nobody told. The same
    judgement is already made one function over, for the RE manager's
    control-socket keys — see ``_is_set`` inside
    :func:`_ensure_bluesky_qserver_zmq_keys`, which spells out that an empty
    value is not honoured there either. This brings the token path into
    agreement with it rather than adding a third convention.

    The exception is bounded by ``required_vars``, and that bound is what makes
    it safe: it is exactly ``_SERVICE_TOKEN_VARS`` filtered by
    ``deployed_services`` membership (built below, from a module constant no
    runtime path mutates), so the only names whose empty spelling is overridden
    are the machine-minted secrets that map declares. Nothing else travelling in
    this ``.env`` can reach the predicate — the proxy settings (``HTTP_PROXY``,
    ``NO_PROXY``), where a blank value *is* the setting, belong to neither
    ``_SERVICE_TOKEN_VARS`` nor ``_VALIDATE_ONLY_VARS``, and the validate-only
    loop below never mints anything at all.

    Whatever a value's origin, it is then checked at the boundary: one that
    matches a compose template's published default is refused by name
    (``_is_forbidden_value``), and one that fails a registered format constraint
    is refused too. A minted value satisfies both by construction, so either
    finding is always about a value an operator supplied.

    Minting over a blank also rewrites this process's copy of it. The deploy
    loads the project ``.env`` over its own environment before reaching here, so
    a blank line in that file leaves a stale empty variable behind that would
    outrank the newly appended line for compose's interpolation. The file is the
    source of truth and the environment is its mirror, so both are updated
    together; a name that was never in the environment stays out of it.

    One empty spelling is still left alone, because minting cannot repair it: a
    var the ``.env`` does not carry at all, exported empty in the deploying
    shell. That export is the operator's own, it outranks any ``--env-file``
    line a mint would write, and overwriting a shell the deploy does not own is
    not this function's business — so writing one would report a mint that
    changes nothing. It keeps the older treatment: on a loopback deploy the service
    fails closed on its own; on a deployment the build rendered as reachable
    off-host (a wildcard bind, or the host-networked web-terminal stack) we
    refuse rather than bind a fail-open-at-bind server to all interfaces.

    Minting is unconditional for every var a deployed service declares: these
    tokens authenticate *network callers* to a service's own HTTP boundary and
    are not a hardware-safety layer. Whether a plan or a write is permitted is
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
    (e.g. ``ARIEL_DSN``) is checked against both its ``_VAR_FORBIDDEN_VALUES``
    entry and its ``_VAR_VALIDATORS`` constraint
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

    # The rest of the chain rides the SAME two routes: `.env.shared` is one of
    # the `--env-file` fragments on the docker shape and a member of the
    # worker's `env_file:` list, so a `$` there is mangled exactly like one in
    # `.env`. The podman shape re-scans the chain in
    # _write_env_merged_for_compose, but that runs only on the podman branch —
    # this scan is the one every provider gets. (`.env` is re-scanned as a
    # chain member; the file-specific refusal above keeps its precise path.)
    chain_offenders = compose_unsafe_vars_in_chain(env_path.parent)
    if chain_offenders:
        raise ComposeInterpolationError(
            chain_offenders,
            " + ".join(str(path) for path in chain_files(env_path.parent)),
        )

    if required_vars:
        existing = parse_dotenv_file(env_path) if env_path.is_file() else {}

        generated: dict[str, str] = {}
        for name in required_vars:
            # Process env wins over .env (matches docker compose --env-file).
            # "Present" means present with a NON-EMPTY value: for these vars an
            # empty spelling is a blank, not a decision (see the docstring).
            #
            # The second clause is not the presence rule — it is the one empty
            # spelling a mint cannot repair. A name absent from the `.env` but
            # exported empty in the deploying shell came from the shell alone,
            # and that export outranks the `--env-file` line this would append
            # for compose's interpolation, so minting would report a token no
            # container ever receives. Left alone, and caught below instead.
            # (A name the `.env` *does* carry is a different case: the deploy
            # has already loaded that file over this process's environment, so
            # the empty copy sitting in `os.environ` is the file's own value
            # mirrored, not a second opinion — see the write-back below.)
            exported_only = name in os.environ and name not in existing
            present = bool(_effective_value(name, existing).strip()) or exported_only
            if present:
                continue  # keep the operator's value
            generated[name] = _generate_token(name)

        if generated:
            _append_env_block(
                env_path,
                "Auto-generated service auth tokens (osprey deploy up)",
                generated,
            )
            # Only the names minted OVER an empty `.env` entry are in the
            # environment already: the deploy loads the project `.env` over this
            # process before it gets here, so the blank line just replaced left a
            # stale empty copy behind. That copy would outrank the file for
            # compose's interpolation and for the validation below — a token
            # minted, reported, and then not delivered. Replace it, so the file
            # and its in-process mirror say the same thing. Names absent from
            # the environment stay absent; they reach the containers through
            # `--env-file` as they always have.
            for name, value in generated.items():
                if name in os.environ:
                    os.environ[name] = value
            # Report the count inline and the key names in the ledger — NEVER
            # the values.
            _report_fact(
                f"minted {len(generated)} service auth token(s) → .env",
                wrote=(
                    ".env",
                    f"{', '.join(generated)} (gitignored; keep them secret)",
                ),
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

    # Write the non-secret service settings (_SERVICE_DEFAULT_VARS) for every
    # deployed service that declares one. Its OWN block, outside `if
    # required_vars:` and outside `if generated:`, and that placement is the
    # whole upgrade path: the mint is idempotent, so a project deployed before
    # this existed has a `.env` carrying a password and no email, generates
    # nothing on its next deploy, and would never receive the line if this rode
    # along with the minted block. Exactly the deployments that need it are the
    # ones with nothing left to mint.
    #
    # Presence is read the same way the mint reads it — process env over `.env`,
    # non-empty — so an operator's value is never overwritten and a re-run
    # appends nothing. Reported separately from the token ledger, and the VALUE
    # is shown: these are account names, not secrets, and an operator who cannot
    # see what was written is back where they started.
    defaults: dict[str, str] = {}
    existing_defaults = parse_dotenv_file(env_path) if env_path.is_file() else {}
    for svc_name, var_defaults in _SERVICE_DEFAULT_VARS.items():
        if svc_name not in services:
            continue
        for var, default in var_defaults:
            if _effective_value(var, existing_defaults).strip():
                continue  # keep the operator's value
            defaults[var] = default

    if defaults:
        _append_env_block(
            env_path,
            "Service account names (osprey up) — not secrets",
            defaults,
        )
        # Same repair the mint makes, for the same reason: the deploy loaded
        # this `.env` over the process environment before we got here, so a name
        # the file spelled EMPTY left an empty copy in `os.environ` that
        # outranks the `--env-file` line for compose's interpolation. Without
        # this the container receives the blank while `.env` shows the default —
        # written down and not delivered. Names absent from the environment stay
        # absent and reach the containers through `--env-file` as usual.
        for var, value in defaults.items():
            if var in os.environ:
                os.environ[var] = value
        _report_fact(
            f"wrote {len(defaults)} service account name(s) → .env",
            wrote=(
                ".env",
                ", ".join(f"{var}={value}" for var, value in defaults.items()),
            ),
        )

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
        # Still reachable, and now for exactly one shape. A blank carried by the
        # `.env` was minted over above, so an empty effective value at this
        # point means the var is exported empty in a shell whose `.env` never
        # mentions it — the one spelling the mint deliberately leaves alone. The
        # remedy therefore names the environment, not `.env`: adding a line to
        # the file would not change what compose resolves while that export
        # stands.
        if expose_network and not effective.strip():
            raise RuntimeError(
                f"{name} is empty; refusing to start a deployment that is reachable "
                f"off-host with an empty token. Unset {name} in the environment and let "
                f"`osprey up` mint one, or export a strong secret."
            )
        # Ahead of the format check so the more specific diagnosis wins. A
        # registered forbidden value is well-formed by construction — that is
        # exactly why no validator can catch it — but a var could later register
        # one that is not, and "your password is public" is the finding an
        # operator needs first either way. Narrow by construction: a var reaches
        # here holding either a value minted moments ago or one an operator
        # supplied, so a match means the operator pinned the template's published
        # default as their real password.
        if effective and _is_forbidden_value(name, effective):
            _raise_forbidden_var(name)
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
        # Same two checks in the same order as the required_vars loop above, and
        # for a sharper reason here: a value reaching THIS loop was never minted,
        # so it is operator-supplied by construction. Pinning a template's
        # published default is exactly the mistake an operator wiring up an
        # external store makes — they copy the fallback out of the compose file —
        # and the forbidden map is the only check that can see it, since such a
        # value is well-formed by every format rule its var registers.
        if _is_forbidden_value(name, effective):
            _raise_forbidden_var(name)
        if not _validate_var(name, effective):
            _raise_invalid_var(name, effective)

    return minted_volume_initialized


def _volume_initialized_vars_that_would_be_minted(config: dict, env_path: Path) -> set[str]:
    """Which store credentials a mint on this config *would* generate.

    The same rule ``_ensure_service_tokens`` mints by — no non-empty value in
    either the process env or the ``.env``, and not left alone as an export the
    mint cannot reach — evaluated without writing anything. It exists
    for ``restart``, which has to run the stale-volume check BEFORE its ``down``:
    the ``down`` removes the store containers, and with them the only host-side
    copy of the credential each surviving volume was initialized with. Asking
    after the stop would find every volume unrecoverable, having just made it so.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    services = {str(s) for s in (config.get("deployed_services") or [])}
    on_disk = parse_dotenv_file(env_path) if env_path.is_file() else {}

    def _would_mint(var: str) -> bool:
        # Kept in step with the predicate in _ensure_service_tokens: a blank
        # entry in the .env is minted over, so predicting "not minted" for one
        # would let restart skip the very store whose credential is about to
        # change. The export carve-out is mirrored for the same reason.
        exported_only = var in os.environ and var not in on_disk
        return not exported_only and not _effective_value(var, on_disk).strip()

    return {
        var
        for var, store in _VOLUME_INITIALIZED_VARS.items()
        if store.service in services and _would_mint(var)
    }


def _harvest_original_credential(probe, project: str, store: VolumeInitializedStore) -> str | None:
    """The credential *store*'s volume was initialized with, if still readable.

    ``None`` whenever the container is gone, carries no such variable, or holds
    an empty one — every case in which the volume cannot be reopened and the
    only way forward is to discard it.

    The returned value is always in ``.env`` terms, never container terms:
    ``store.cred_prefix`` is stripped here rather than at either call site,
    because this is the single point both of them read the credential through —
    the staleness comparison against the ``.env`` value, and the
    ``--reuse-stores`` write back into it. Stripping in one place is what keeps
    those two from disagreeing about what the credential is. A composite that
    is nothing but its prefix (``"neo4j/"``, an empty password) reduces to the
    empty string and is reported as unreadable, which is what it is.
    """
    env = probe.env_of_container(f"{project}{store.container}")
    if not env:
        return None
    raw = env.get(store.cred_env)
    if not raw:
        return None
    return raw.removeprefix(store.cred_prefix) or None


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

    _report_fact(
        f"Adopted {len(stale)} pre-existing data volume(s): restored the credential(s) "
        f"{', '.join(var for var, _ in stale)} they were initialized with into "
        f"{env_path.resolve()}. Values are never printed."
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
    """Auto-configure the bluesky bridge's EPICS-substrate plan devices for a
    VA-backed Bluesky stack, making ``osprey up`` turn-key.

    Additive and non-breaking, mirroring ``_ensure_service_tokens``'s
    "existing value wins, append what's missing" convention: when the
    deployed project is a VA-backed Bluesky stack (BOTH ``"bluesky"`` and
    ``"virtual_accelerator"`` present in ``deployed_services``), derive
    ``BLUESKY_EPICS_SUBSTRATE``/``BLUESKY_EPICS_SETPOINTS``/``_READBACKS`` from
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
        _report_fact(
            "control_system.type is 'mock': this deployment is browse-only -- plans can "
            "be listed and composed but never executed. Skipping "
            "BLUESKY_EPICS_SUBSTRATE auto-configuration (plan devices need an "
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
            "Could not auto-configure bluesky bridge plan devices from %s "
            "(derivation raised unexpectedly). Skipping BLUESKY_EPICS_SUBSTRATE "
            "auto-configuration -- set BLUESKY_EPICS_SETPOINTS/_READBACKS manually "
            "if you want the bridge to run in EPICS-substrate mode.",
            project_dir / "data" / "channel_limits.json",
            exc_info=True,
        )
        return
    if not derived:
        logger.warning(
            "Could not auto-configure bluesky bridge plan devices from %s "
            "(missing, unreadable, or yields no SR correctors/BPMs). Skipping "
            "BLUESKY_EPICS_SUBSTRATE auto-configuration -- set "
            "BLUESKY_EPICS_SETPOINTS/_READBACKS manually if you want the bridge "
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
        "Auto-configured bluesky bridge plan devices (osprey deploy up)",
        generated,
    )
    _report_fact(
        "bluesky plan devices auto-configured → .env",
        wrote=(
            ".env",
            f"{', '.join(generated)} from the project's channel_limits.json",
        ),
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
        logger.debug(
            "Bluesky document-plane CURVE certificates are already present in %s. Keeping them.",
            paths["bridge"].parent,
        )
        return
    if present:
        logger.warning(
            "Bluesky document-plane CURVE certificates in %s are incomplete (%d of %d "
            "present). Regenerating the whole set, since a certificate's secret half "
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
            "document proxy, so plan runs will report no live rows.",
            paths["bridge"].parent,
            exc_info=True,
        )
        return

    _report_fact(
        f"bluesky CURVE certificates → {_BLUESKY_CURVE_DIR}/",
        wrote=(
            f"{_BLUESKY_CURVE_DIR}/",
            "document-plane certificates (gitignored); containers read them at "
            "startup, so replacements need the bridge and queueserver recreated",
        ),
    )
    # The transcript keeps the resolved location: a log read away from the
    # project needs the absolute path the relative line cannot carry.
    logger.key_info(
        f"Bluesky document-plane CURVE certificates live under {paths['bridge'].parent}"
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
            "public one you set, so nothing is generated here. The deploy will stop at "
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
        _report_fact(
            f"derived {', '.join(generated)} from this shell's "
            f"{_QSERVER_ZMQ_PRIVATE_KEY_VAR} for this deploy only (not written to .env)"
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
    _report_fact(
        "bluesky control-socket keypair minted → .env",
        wrote=(
            ".env",
            f"{', '.join(generated)} (gitignored); treat both halves as secrets, since "
            "the public half alone reaches the manager's control socket",
        ),
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


#: Env-var spellings for "on" and "off", matching the other framework switches.
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _resolve_prebuilt_images(config: dict) -> bool:
    """Whether this host already has the images and must not build them.

    Some deployment hosts cannot build at all — no build tooling, no registry
    reachable, images side-loaded from a tarball instead. There the dev-mode
    compose build is not slow but impossible, and the only thing that can run
    is ``up`` against tags that are already present.

    Resolution order:
      1. ``OSPREY_PREBUILT_IMAGES`` environment variable (truthy: 1/true/yes/on;
         falsy: 0/false/no/off)
      2. Top-level ``prebuilt_images`` key in the deploy config
      3. Default: false — deploys build as they always have

    Env var wins in both directions, so a host whose config pins the switch can
    still be made to build for one shell, and vice versa.

    :param config: Loaded deploy config.
    """
    env = os.environ.get("OSPREY_PREBUILT_IMAGES", "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in _FALSY:
        return False
    return bool(config.get("prebuilt_images", False))


def _worker_image_target(config: dict, env: dict) -> str:
    """The image the dispatch worker will actually run.

    Resolution mirrors the worker compose service's
    ``${OSPREY_WORKER_IMAGE:-<services.dispatch_worker.image | default>}``:
    an ``OSPREY_WORKER_IMAGE`` env override wins, then a profile-pinned
    ``services.dispatch_worker.image``, else the project image that
    :func:`_build_project_image` builds — carrying whatever the registry and
    tag axes resolved to.

    That last layer is READ from
    :func:`~osprey.deployment.compose_generator.resolve_image_defaults` rather
    than restated here. It is the same function the template's innermost
    ``default(...)`` renders from, so a prediction made here and the reference
    compose ends up carrying cannot drift onto different axes.
    """
    override = env.get("OSPREY_WORKER_IMAGE")
    if override:
        return str(override)
    worker_cfg = config.get("services", {}).get("dispatch_worker", {})
    explicit = worker_cfg.get("image") if isinstance(worker_cfg, dict) else None
    if explicit:
        return str(explicit)
    return resolve_image_defaults(config)["worker"]


def _project_image_build_target(config: dict, env: dict) -> str | None:
    """The image tag :func:`_build_project_image` will build, or ``None``.

    The gate in front of that build, split out so a caller can ask whether the
    build will happen at all without running it. Three ways it does not: this
    host says its images are prebuilt (:func:`_resolve_prebuilt_images`), this
    deployment does not deploy the dispatch worker, or the worker's effective
    image (:func:`_worker_image_target`) is a prebuilt one rather than this
    project's own image.

    The prebuilt switch is asked first and outranks the rest, because it is a
    statement about the HOST rather than about the deployment: a machine with no
    build tooling, or one running images side-loaded from a tarball or pulled
    from a registry, cannot build this tag whatever the config says about it.
    Building it anyway is worse than slow — under an ``images.registry`` the tag
    is registry-qualified, so a local build overwrites the very image that was
    pulled, with a locally-assembled one nothing published.

    That question is the difference between a definite refusal and a
    hypothetical one for anything the build itself would raise. The
    unreleased-pin refusal is the case in point: it is a fact about a build that
    is going to happen, and reporting it for a deployment that builds nothing
    would send the operator to fix something that was never going to run.

    :param config: Raw deploy config.
    :param env: Environment the build would run with, read for
        ``OSPREY_WORKER_IMAGE``.
    :return: The project image's tag, or ``None`` when nothing is built.
    """
    if _resolve_prebuilt_images(config):
        return None
    services = {str(s) for s in (config.get("deployed_services") or [])}
    if "dispatch_worker" not in services:
        return None
    project_image = resolve_image_defaults(config)["worker"]
    if _worker_image_target(config, env) != project_image:
        return None
    return project_image


def _unreleased_pin_problem(config: dict, env: dict, dev_mode: bool) -> tuple[str, str] | None:
    """Whether the project-image build would refuse over an unreleased pin.

    :func:`_resolve_pip_spec` asked as a question, behind
    :func:`_project_image_build_target` so it is only asked about a build this
    deployment will actually make.

    Only the DEFINITE case is reported. Under ``--dev`` the spec is inert when a
    wheel is staged and live when the staging fails, and which of those happens
    is not knowable from here — so a ``--dev`` start is left to the build's own
    refusal rather than pre-judged, which would refuse the very workflow the
    error recommends. An ``OSPREY_PIP_SPEC`` export answers the question outright
    and is honoured here exactly as the build honours it.

    Pure: it reads the process environment and this build's version metadata.

    :param config: Raw deploy config.
    :param env: Environment the build would run with.
    :param dev_mode: Whether ``--dev`` was passed.
    :return: The refusal as a ``(problem, remedy)`` pair — the one probed
        refusal that already carries its two halves apart, so the collected
        report can keep them apart too — or ``None``.
    """
    if dev_mode or _project_image_build_target(config, env) is None:
        return None
    from osprey.deployment.errors import UnreleasedVersionPinError

    try:
        _resolve_pip_spec(dev_mode=False)
    except UnreleasedVersionPinError as exc:
        return (f"{exc.summary}: {exc.reason}", exc.remedy)
    return None


def _project_image_build_cmd(
    config: dict, runtime: str, project_root: str, dev_mode: bool = False
) -> list[str]:
    """Construct the ``<runtime> build`` argv that produces the project image.

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
        # The tag compose will reference, axes included: building one name and
        # rendering another would leave the worker pointing at nothing.
        resolve_image_defaults(config)["worker"],
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
    """Build the project image the dispatch worker references.

    The dispatch worker's compose service intentionally has no ``build:`` block
    (a second builder for the same tag would race the event-dispatcher — see
    ``dispatch_worker/docker-compose.yml.j2``), so nothing in ``compose up``
    produces the image it runs. This builds it once, from the project root
    (context) and the rendered project ``Dockerfile``, before ``compose up``.

    No-op unless the worker is deployed and its effective image is this
    project's own axis-derived tag: an ``OSPREY_WORKER_IMAGE`` override or a
    profile-pinned ``services.dispatch_worker.image`` means a prebuilt image is
    wanted, so there is nothing to build. The event-dispatcher's own
    ``<project>-dispatch`` build (its compose ``build:`` block) is untouched.

    A host that declares its images prebuilt (``prebuilt_images`` /
    ``OSPREY_PREBUILT_IMAGES``) skips this build outright, for the same reason
    the compose build below it is skipped: the tags are expected to be on the
    host already. That switch is the whole answer for a host with no build
    tooling — and, under an ``images.registry``, for a host that pulled this
    exact tag and must not have it overwritten by a local rebuild carrying the
    same name.

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
    project_image = _project_image_build_target(config, env)
    if project_image is None:
        # Only SOME of the no-op cases are worth a line. A deployment without
        # the dispatch worker was never going to build this image and has
        # nothing to explain; a deployment that has the worker but points it at
        # another image took a decision the operator should see reflected back;
        # and a prebuilt host gets the same line the compose build reports for
        # the same switch — but only where the switch actually took a build
        # away, which is the worker running this project's own tag. Saying it
        # for a deployment that has no worker would report a skip of something
        # that was never going to happen.
        services = {str(s) for s in (config.get("deployed_services") or [])}
        if _resolve_prebuilt_images(config):
            if "dispatch_worker" in services and (
                _worker_image_target(config, env) == resolve_image_defaults(config)["worker"]
            ):
                _report_group("images")
                _report_step("skipped image build (prebuilt images)")
            return
        if "dispatch_worker" in services:
            _report_fact(
                f"Dispatch worker uses image {_worker_image_target(config, env)!r} "
                f"(OSPREY_WORKER_IMAGE / pinned services.dispatch_worker.image). "
                f"Skipping the {resolve_image_defaults(config)['worker']} build."
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
        _report_group("images")
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


#: The merged env chain, written under the render zone for the delivery shapes
#: that can be handed only ONE env file. The leading dot is load-bearing rather
#: than cosmetic: every sweep that keeps secrets out of a build artifact or an
#: image context already matches ``.env*``, so naming the file this way puts it
#: inside all of them without a second list to keep in step. It is a machine
#: artifact — regenerated from the chain on every compose invocation that needs
#: it, never a place to edit a value.
ENV_MERGED_RELPATH = Path(BUILD_DIRNAME) / ".env.merged"


def _write_env_merged_for_compose(repo_root: Path | str) -> Path:
    """Regenerate ``<repo>/build/.env.merged`` from the env chain, and return it.

    The whole chain collapsed into one file at :data:`ENV_MERGED_RELPATH`, at
    mode ``0600``, for a compose provider that reads a single ``--env-file``
    and would otherwise see only the last file it was handed — the merge has to
    happen ahead of it for ``.env`` to keep winning over ``.env.shared``.

    Regenerated on every call rather than when it looks stale: it is derived
    from files a few kilobytes long, so recomputing it is cheaper than being
    wrong about it. The two ways a cached copy goes wrong are both ordinary —
    a project built but never brought up has no merged file at all, and a
    hand-edited ``.env`` leaves a stale one that would start the stack on the
    superseded value with nothing to show for it.

    The chain-wide ``$``-scan runs BEFORE the write, so a value compose would
    mangle stops the deploy while the chain is still the only copy of it,
    rather than after it has been duplicated into a second file on disk. The
    refusal carries variable names only — never values.

    :param repo_root: The deployment repo the chain lives in.
    :return: The merged file's path, for splicing into ``--env-file``.
    :raises ComposeInterpolationError: A chain value contains ``$``.
    """
    root = Path(repo_root)
    offenders = compose_unsafe_vars_in_chain(root)
    if offenders:
        raise ComposeInterpolationError(offenders, " + ".join(str(p) for p in chain_files(root)))
    return write_env_merged(root, root / ENV_MERGED_RELPATH)


def _compose_provider(config: dict | None) -> ComposeProvider:
    """Which compose implementation this deployment's invocations are shaped for.

    Resolved here rather than passed in from the CLI, because the answer is a
    property of the HOST — and of the runtime this same module already resolves
    through ``get_runtime_command`` — not of what the operator typed. Both names
    are the ones bound in this module, which is the seam the deploy tests patch.
    The probe is memoized per compose argv base, so a verb that asks more than
    once (``down`` does) pays for one probe.

    An unsupported provider stops the verb here rather than being guessed at
    (see :func:`~osprey.deployment.runtime_helper.detect_compose_provider`).
    That is the whole reason to probe: a provider handed the wrong argv shape
    does not report an unsupported deployment, it starts a stack whose paths
    resolve against the wrong directory.

    :param config: The deploy config, for the runtime it names.
    :return: The provider to shape argv and environment for.
    :raises UnsupportedComposeProviderError: The host's compose provider is not
        one OSPREY can invoke correctly.
    """
    return detect_compose_provider(get_runtime_command(config), config).provider


def _env_file_args(
    repo_root: Path | str | None = None,
    provider: ComposeProvider | None = None,
) -> list[str]:
    """The ``--env-file`` fragment for one compose invocation, provider-shaped.

    THE seam every lifecycle verb resolves its env-file argv through, and
    therefore the one place the merged chain has to be (re)written: whichever
    verb runs next — ``up``, ``down``, ``restart``, ``status``, a roster verb —
    passes through here first, so the file handed to compose is regenerated
    from the chain on that same invocation.

    Two shapes, because the providers disagree about what a repeated
    ``--env-file`` means:

    * :attr:`~osprey.deployment.runtime_helper.ComposeProvider.PODMAN_COMPOSE`
      keeps only the LAST file it is given, so a chain passed as several
      fragments would silently drop everything but ``.env``. It gets the single
      merged file instead (see :func:`_write_env_merged_for_compose`).
    * Anything else — today's Docker Compose v2 shape, and the default when no
      probe result is threaded down — resolves through
      :func:`osprey.deployment.compose_generator.compose_env_file_args`, so the
      rule and its "no .env" warning keep one definition, in the module that
      owns the invocation contract.

    A repo with no chain file at all takes the default branch either way: there
    is nothing to merge, and delegating keeps the one warning about it (and the
    empty fragment compose needs, since it hard-fails on an ``--env-file`` that
    is not there) in that single definition.

    :param repo_root: The deployment repo whose chain this is. ``None`` falls
        back to the working directory, which every repo-scoped verb has already
        chdir'd into.
    :param provider: The detected compose provider, when the caller has probed
        for one. ``None`` selects the default shape.
    :raises ComposeInterpolationError: A chain value contains ``$`` and the
        provider shape requires the merged file.
    """
    from osprey.deployment.compose_generator import compose_env_file_args

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if provider is ComposeProvider.PODMAN_COMPOSE and chain_files(root):
        return ["--env-file", str(_write_env_merged_for_compose(root))]
    return compose_env_file_args(root)


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


#: Where the env chain's per-key fingerprints from the last deploy are kept.
#:
#: Machine zone, never committed and never hand-edited: it is derived from the
#: chain the way ``build/.env.merged`` is, and lands in the same directory for
#: the same reason — every sweep that keeps a secret out of a build artifact or
#: an image context already matches ``.env*``, so the leading dot puts this file
#: inside all of them without a second list to keep in step.
#:
#: It holds **digests, never values**. The stale-pin question is only ever asked
#: as "are these two the same string?", which a per-key ``sha256`` answers
#: exactly, so nothing is gained by keeping a second plaintext copy of the
#: chain's secrets on disk — and a file that cannot leak a value is one fewer
#: thing to reason about when a build directory is tarred up for a bug report.
ENV_CHAIN_STATE_RELPATH = Path(BUILD_DIRNAME) / ".env.chain-state.json"

#: Schema marker for :data:`ENV_CHAIN_STATE_RELPATH`. A file written by an older
#: shape is discarded rather than migrated: the whole record is re-derivable
#: from the next deploy, so the cost of dropping it is one deploy's worth of
#: stale-pin history, against the cost of carrying migration code for a
#: machine-generated cache.
_ENV_CHAIN_STATE_VERSION = 1


def _value_digest(value: str) -> str:
    """Fingerprint ONE env value, so equality can be tested without keeping it.

    Not a second spelling of
    :func:`osprey.deployment.runtime_helper.env_chain_digest`, and not
    substitutable for it in either direction — the two answer different
    questions at different granularities. That one hashes the chain's whole
    files into a single digest, which is what a container label needs ("did
    anything at all change?") and is deliberately blind to which key moved.
    The stale-pin comparison is per-key by nature, so it needs a digest per
    value; deriving one from the other is not possible, and consolidating them
    would silently break whichever caller lost its granularity.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_chain_state(repo_root: Path) -> dict[str, str]:
    """The shared-file key digests recorded by the previous deploy.

    Empty for every ordinary "no history yet" case — no file, an unreadable one,
    malformed JSON, a version this build does not recognise. A missing record
    means the layering questions below are answered "no evidence", which is the
    right answer when there genuinely is none, so none of these are errors.

    :param repo_root: The deployment repo root.
    :return: ``{KEY: sha256-of-value}`` as ``.env.shared`` stood at the last
        deploy, or ``{}`` when there is no usable record.
    """
    try:
        raw = json.loads((repo_root / ENV_CHAIN_STATE_RELPATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != _ENV_CHAIN_STATE_VERSION:
        return {}
    digests = raw.get("shared_value_digests")
    if not isinstance(digests, dict):
        return {}
    return {str(key): str(value) for key, value in digests.items()}


def _write_chain_state(repo_root: Path, shared_value_digests: Mapping[str, str]) -> None:
    """Record the shared-file key digests for the next deploy to compare against.

    Best-effort by design: this is an advisory's memory, not deployment state.
    A repo whose ``build/`` cannot be written is a repo with bigger problems
    than an unreported stale pin, and turning that into the reason a deploy
    fails would be a preflight causing the outage it exists to prevent.

    :param repo_root: The deployment repo root.
    :param shared_value_digests: ``{KEY: sha256-of-value}`` to carry forward.
    """
    path = repo_root / ENV_CHAIN_STATE_RELPATH
    payload = {
        "version": _ENV_CHAIN_STATE_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "shared_value_digests": dict(shared_value_digests),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        logger.debug("Could not record env chain state at %s: %s", path, exc)


def _read_profile_yaml(path: Path) -> dict:
    """One profile file, parsed, or an empty mapping.

    The tolerant half of :func:`_profile_document`, split out because the same
    "nothing to read" rule has to cover both layers it reads — the tracked
    ``profile.yml`` and this host's variant overlay.

    :param path: The file to read.
    :return: The parsed mapping, or ``{}`` when the file is absent, unreadable,
        unparseable, or not a mapping at its top level.
    """
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return document if isinstance(document, dict) else {}


def _selected_variant_profile(repo_root: Path) -> Path | None:
    """The overlay profile this HOST builds with, or ``None``.

    ``.env.variant`` names one of the repo's tracked ``profiles/<name>.yml``
    layers, and the build merges it over ``profile.yml`` before resolving
    anything (see :mod:`osprey.cli.variant_selection`). The selection is read
    back here — never re-derived — so the deploy cannot decide a different
    variant than the build did.

    Tolerant, like every other reader behind :func:`_profile_document`: a
    setting that names a variant this repo does not define is ``osprey build``'s
    refusal to raise, and raising it from a probe would abort a deploy over a
    file the deploy does not build from.

    :param repo_root: The deployment repo root, holding the setting and
        ``profiles/``.
    :return: The overlay to merge, or ``None`` when this host builds the tracked
        profile as-is.
    """
    from osprey.cli.variant_selection import resolve_variant_selection
    from osprey.errors import BuildProfileError

    try:
        return resolve_variant_selection(repo_root).path
    except (BuildProfileError, OSError):
        return None


def _profile_document(repo_root: Path | str) -> dict:
    """This deployment's profile as THIS host builds it, or an empty mapping.

    The one tolerant reader behind every deploy-time probe that asks the profile
    a question (:func:`pinned_env_keys`, :func:`_required_env_problems`). Read
    off the repo's own file rather than the preset it was materialized from, for
    the same reason
    :func:`osprey.deployment.web_terminals.auth_credentials._profile_env_defaults`
    does: only the repo's file is honestly "what this deployment declared". No
    ``extends:`` chain is resolved, because an emitted profile is already flat
    and a deployment root that hand-writes one keeps its parent's declarations
    unread.

    What is NOT taken as written is the host's variant layer. One repo can carry
    several ``profiles/<name>.yml`` overlays and ``.env.variant`` picks the one
    THIS host builds; the build merges it over ``profile.yml`` before resolving
    anything, by the same deep merge (:func:`~osprey.cli.build_profile_merge._deep_merge`)
    used here — so string lists union rather than replace, and an ``env.pinned``
    or ``env.required`` name the overlay adds is one the built stack really has.
    Reading the tracked file alone would leave every declaration a variant
    contributes unenforced at ``osprey up``: the build renders a project whose
    profile pins a name, and the deploy would go on believing nothing was
    pinned. The merge covers the whole document rather than the ``env`` block
    alone, for the same reason — the deploy block a variant overrides
    (:func:`_ci_only_env_names`) decides which of those names a host reads.

    Every "nothing to read" shape answers the same way, with an empty mapping:
    no ``profile.yml``, an unreadable one, one that does not parse, or one whose
    top level is not a mapping — and the same rule absorbs an unusable overlay,
    leaving the tracked document standing. A profile that does not parse is the
    staleness check's finding to report and ``osprey build``'s to fail on, so it
    must not become a refusal raised from a probe here.

    :param repo_root: The deployment repo root, holding ``profile.yml``.
    :return: The parsed document, or ``{}`` when there is nothing to read.
    """
    from osprey.cli.repo_resolver import PROFILE_FILENAME

    root = Path(repo_root).expanduser().absolute()
    document = _read_profile_yaml(root / PROFILE_FILENAME)

    overlay_path = _selected_variant_profile(root)
    if overlay_path is None:
        return document
    overlay = _read_profile_yaml(overlay_path)
    if not overlay:
        return document

    from osprey.cli.build_profile_merge import _deep_merge

    return _deep_merge(document, overlay)


def _ci_only_env_names(document: Mapping[str, object]) -> set[str]:
    """The declared variable names that belong to CI, not to a deploy host.

    The registry credential (``deploy.registry.token_env_var``) and each
    external project's pull credential (``deploy.external_projects[].token_env_var``)
    are read where images are pushed and pulled from a registry — the pipeline,
    and a host that pulls prebuilt images. A host that builds its own images
    never reads them, and refusing its start over a registry token it has no use
    for would make every developer deploy of a facility profile unstartable.

    They are named in the profile's deploy block and declared under
    ``env.required`` beside everything else (the deploy block has no env channel
    of its own — see :mod:`osprey.cli.build_profile_deploy`), so the exclusion
    has to be made here rather than by the declaration.

    :param document: A profile document as read by :func:`_profile_document`.
    :return: The CI-only names. Empty when the profile has no deploy block.
    """
    deploy = document.get("deploy")
    if not isinstance(deploy, dict):
        return set()
    blocks: list[object] = [deploy.get("registry")]
    external = deploy.get("external_projects")
    if isinstance(external, list):
        blocks.extend(external)
    names: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        token = block.get("token_env_var")
        if isinstance(token, str) and token:
            names.add(token)
    return names


def _required_env_problems(repo_root: Path | str, env: Mapping[str, str]) -> list[tuple[str, str]]:
    """One finding per ``env.required`` name nothing gives this deploy a value for.

    ``env.required`` is the profile's statement that the agent cannot run
    without these variables. Nothing on the deploy path enforced it: the stack
    started, and the missing variable surfaced later as an empty substitution, a
    container restarting, or a tool failing to authenticate at the first request
    — a failure mode that costs a whole deploy to observe and reads as anything
    but "you never filled in the ``.env``".

    A name counts as provided when ANY source the stack reads carries a
    non-empty value for it: this repo's env chain (``.env.shared`` under
    ``.env``, via :func:`~osprey.utils.dotenv.merge_chain`) or the process
    environment. The union rather than a precedence order, because the question
    here is existence, not which value wins — and both halves are needed: the
    chain is read from ``repo_root``, which under ``osprey up --repo`` is not
    the directory whose ``.env`` the CLI loaded into its own environment at
    entry, while an exported value reaches compose without being in any file.

    An empty value is a missing value. ``.env.example`` renders every required
    name as a bare ``NAME=``, so the copied-but-not-filled-in file is the exact
    shape this probe exists to catch, and treating its blanks as answers would
    make the probe silent for the operator who most needs it.

    Two classes of declared name are excluded, both because they are not the
    operator's to supply on this host:

    * the CI credentials (:func:`_ci_only_env_names`);
    * everything a deploy-time writer mints (:func:`_deploy_written_env_vars`,
      the same census :func:`_preflight_pinned_writers` refuses a pin against).
      When the service that needs one is deployed, the mint two steps above has
      already written it and there is nothing to report; when it is not, nothing
      in the stack reads it, and a refusal would be about a service this deploy
      does not run. The exemplar profile declares its dispatch tokens this way,
      so this is the ordinary shape rather than an edge case.

    Pure: it reads ``profile.yml`` and the env chain, and writes nothing.

    :param repo_root: The deployment repo root, holding ``profile.yml`` and the
        env chain.
    :param env: The environment the stack would be started with.
    :return: ``(problem, remedy)`` pairs in the order the profile declares the
        names, so the report reads in the order of the file the operator opens
        to fix it. Empty when the profile declares nothing (or nothing missing).
    """
    document = _profile_document(repo_root)
    env_section = document.get("env")
    if not isinstance(env_section, dict):
        return []
    declared = env_section.get("required")
    if not isinstance(declared, list):
        return []

    excluded = _ci_only_env_names(document) | _deploy_written_env_vars()
    chain = merge_chain(Path(repo_root))

    problems: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in declared:
        if not isinstance(name, str) or not name or name in seen or name in excluded:
            continue
        seen.add(name)
        if (chain.get(name) or "").strip() or (env.get(name) or "").strip():
            continue
        problems.append(
            (
                f"{name} is required by this deployment's profile (env.required), but no "
                f"value reaches the stack: it is unset in .env, .env.shared and the "
                f"environment, or set there to an empty value.",
                f"Set {name} in .env at the repo root. .env.example lists every variable "
                f"this deployment reads; copy it first (cp .env.example .env) if this repo "
                f"has no .env yet.",
            )
        )
    return problems


def pinned_env_keys(repo_root: Path | str) -> set[str]:
    """The variable names this deployment declares under ``env.pinned``.

    A pinned name is one the repo's own env chain owns outright: the value the
    stack runs on is meant to come from ``.env.shared``/``.env`` under this
    root, not from a shell export, not from a local override of a shared
    default, and not from anything that writes the chain from outside. The
    declaration lives in the profile because that is the file an operator edits
    and reviews; the deploy reads it back here so the preflights can say which
    of the things they already detect are merely notable and which contradict
    something the deployment declared.

    Read off the repo's own ``profile.yml`` through :func:`_profile_document`,
    which is also where the "nothing to read" shapes are absorbed.

    Every "nothing declared" shape answers the same way, with an empty set: no
    ``profile.yml``, no ``env`` block, no ``pinned`` key, or a document this
    cannot parse. A profile that does not parse is the staleness check's
    finding to report and ``osprey build``'s to fail on, so it must not become a
    refusal raised from here — and an empty set leaves every consumer at the
    behaviour it had before pins existed.

    :param repo_root: The deployment repo root, holding ``profile.yml``.
    :return: The declared names, deduplicated. Empty when nothing is declared.
    """
    env_section = _profile_document(repo_root).get("env")
    if not isinstance(env_section, dict):
        return set()
    declared = env_section.get("pinned")
    if not isinstance(declared, list):
        return set()
    return {name for name in declared if isinstance(name, str) and name}


def _classify_local_overrides(
    shared: Mapping[str, str],
    local: Mapping[str, str],
    previous_shared_digests: Mapping[str, str],
    pinned: Collection[str] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Split the keys ``.env`` overrides into deliberate, stale and pinned.

    The first two start from the same fact — a key ``.env.shared`` and ``.env``
    disagree about — and they deserve opposite volumes, which is the whole point
    of reporting them separately:

    * A **deliberate override** is the chain working exactly as designed. This
      host wants its own value, so ``.env`` carries it and wins. It is worth one
      line so the operator can see the list without opening two files, and worth
      no more than that.
    * A **stale pin** is the failure mode the layering makes possible. The
      shared defaults moved on, and ``.env`` still holds the value they carried
      *before* that change — the signature of a pin written to match the default
      of the day and then forgotten, rather than a value anybody chose for this
      host. The stack starts on the superseded value, looking healthy.

    The distinction is exact, not heuristic: a key is stale only when the
    recorded shared value from the last deploy differs from today's shared
    value **and** equals today's local value. An override whose local value
    matches neither is a value someone chose; an override on a shared key that
    has not moved is simply current.

    The third bucket is not a severity of the same question but a different one.
    A **pinned override** is a local value for a name the deployment declared it
    owns (``env.pinned``, read by :func:`pinned_env_keys`), so no local value for
    it is deliberate by definition: the declaration says the shared half decides,
    and the local file is contradicting it. It is therefore classified first and
    exclusively — a pinned key never also appears as deliberate or as stale,
    because the answer to "is this override intended?" was already given in the
    profile. Its consumer refuses (:func:`_preflight_pinned_overrides`); the two
    unpinned buckets keep local-wins and their existing volumes.

    That third question is asked of ``.env`` alone, and this is the one place
    the three buckets do not share a starting fact. Deliberate and stale both
    begin at "the two files disagree", which requires the shared file to have
    said something; a pin does not, because a pinned name ``.env.shared`` never
    anchors is precisely the shape where the local file is the stack's only
    source for a variable the deployment says it does not decide. Only exact
    agreement clears it — a local line that restates the shared value changes
    nothing about what starts.

    All three comparisons are string equality, so they are performed on digests
    (see :data:`ENV_CHAIN_STATE_RELPATH`) rather than on the secrets themselves.

    :param shared: Parsed ``.env.shared`` as it is now.
    :param local: Parsed ``.env`` as it is now.
    :param previous_shared_digests: Per-key digests of ``.env.shared`` as it
        stood at the previous deploy (see :func:`_read_chain_state`).
    :param pinned: The names declared under ``env.pinned``. Empty — the default,
        and what a deployment that declares nothing yields — leaves the other
        two buckets exactly as they were before pins existed.
    :return: ``(deliberate, stale, pinned)``, each sorted, together covering
        every overridden key exactly once — plus, in the pinned bucket, every
        declared name the local file sets that ``.env.shared`` does not anchor.
    """
    declared = set(pinned)
    deliberate: list[str] = []
    stale: list[str] = []
    contradicted: list[str] = []
    for key in sorted(local):
        if key in declared:
            # Asked of the LOCAL file alone, because the declaration is what
            # makes a local line wrong — not the shared file's answer to it. A
            # pinned name `.env.shared` does not carry (or a chain with no
            # shared file at all) has nothing to disagree with, so requiring a
            # disagreement here would let the one shape the pin most needs to
            # stop — the local file as the ONLY source of a name the deployment
            # says the shared half owns — start the stack unremarked.
            if key not in shared or local[key] != shared[key]:
                contradicted.append(key)
            continue
        if key not in shared or local[key] == shared[key]:
            continue
        was = previous_shared_digests.get(key)
        if (
            was is not None
            and _value_digest(shared[key]) != was
            and _value_digest(local[key]) == was
        ):
            stale.append(key)
        else:
            deliberate.append(key)
    return deliberate, stale, contradicted


def _preflight_pinned_overrides(repo_root: Path | str) -> list[str]:
    """Refuse the deploy when ``.env`` sets a name the profile pins.

    ``env.pinned`` is a deployment saying, in the file operators edit and
    review, that these variables are decided by the shared half of the chain and
    nowhere else. Local-wins is the chain's normal and correct behaviour, so
    without a declaration an override is simply this host's business — but *with*
    one, the same override means the stack runs on a value that contradicts what
    the deployment says it runs on, and every reading of the shared file goes on
    describing something that is not what started. Nothing else on the path
    notices: the override is well-formed, the stack comes up healthy, and
    :func:`_report_chain_overrides` would file it under "working as intended".

    Whether ``.env.shared`` names the variable too does not enter into it (see
    :func:`_classify_local_overrides`). A pinned name the shared file leaves
    unset, in a repo that has no ``.env.shared`` at all, is the same deploy from
    the stack's point of view — one running on a local value for a variable the
    deployment declared local files do not decide — and it is the shape the
    declaration is least able to survive, because there is no shared value the
    operator could compare against to notice.

    A refusal rather than a warning, on the same grounds as
    :func:`_preflight_env_chain_drift`: an override the deployment already
    declared out of bounds has no outcome anybody asked for, and both remedies
    (drop the local line, or drop the pin) are one edit away.

    Evaluated on the **pre-mint** chain, and called from :func:`_start_stack`
    ahead of :func:`_ensure_service_tokens`, for two reasons that point the same
    way. The mint writes into ``.env``, so a check downstream of it would be
    reading a file this command has already changed and could refuse on a line
    the operator never wrote; and a deploy that is going to abort here should
    abort having provisioned nothing.

    **Names only, never values** — same rule as every report in this chain.

    :param repo_root: The deployment repo root, holding the chain and the
        profile that declares the pins.
    :return: The empty list, when nothing is contradicted.
    :raises RuntimeError: ``.env`` overrides at least one pinned name.
    """
    root = Path(repo_root).expanduser().absolute()
    declared = pinned_env_keys(root)
    if not declared:
        return []

    shared_path = root / ENV_SHARED_FILENAME
    local_path = root / COMPOSE_ENV_FILENAME
    shared = parse_dotenv_file(shared_path) if shared_path.is_file() else {}
    local = parse_dotenv_file(local_path) if local_path.is_file() else {}
    _, _, contradicted = _classify_local_overrides(shared, local, {}, declared)
    if not contradicted:
        return []

    logger.error(
        "Pinned variable overridden: %s sets %s, which this deployment pins under env.pinned.\n"
        "  A pinned name is one the shared half of the chain owns outright. The local file wins "
        "for every unpinned variable, and would win for these too, so the stack would start on "
        "values that contradict what the deployment declares it runs on — with nothing in the "
        "logs and a healthy stack to show for it. Values are never printed.\n"
        "  To start on the declared value: remove the listed name(s) from %s. To let this host "
        "choose its own: drop them from env.pinned in the profile and re-run `osprey build`.",
        COMPOSE_ENV_FILENAME,
        ", ".join(contradicted),
        local_path,
    )
    raise RuntimeError(
        f"env chain preflight failed: {COMPOSE_ENV_FILENAME} overrides "
        f"{len(contradicted)} variable(s) pinned by this deployment (see report above). "
        f"Remove them from {COMPOSE_ENV_FILENAME}, or drop them from env.pinned."
    )


def _deploy_written_env_vars() -> set[str]:
    """Every variable name a deploy-time writer can put into ``.env``.

    The census the pin declaration is checked against, and deliberately the
    whole one rather than the subset this deploy's services would reach. A pin
    is a static statement in the profile, so whether it is a legal thing to say
    must not depend on which services happen to be enabled today — otherwise
    the contradiction stays invisible until the deploy that first turns one on.

    Six writers, all of them appending to (or rewriting) the local file on the
    ordinary start path:

    * :data:`_SERVICE_TOKEN_VARS` — the minted secrets;
    * :data:`_SERVICE_DEFAULT_VARS` — the non-secret settings written down for
      discoverability;
    * the bluesky substrate devices, derived from the built project's channel
      limits (:func:`_ensure_bluesky_substrate_env`);
    * the RE manager's control-socket keypair
      (:func:`_ensure_bluesky_control_plane_keys`);
    * the credentials ``--reuse-stores`` restores from surviving data volumes
      (:func:`_adopt_original_credentials`, over
      :data:`_VOLUME_INITIALIZED_VARS`);
    * the secrets a live store issues and the deploy harvests back into ``.env``
      (:data:`_STORE_ISSUED_VARS`). Later in the run than the rest — the value
      does not exist until the container is up — but written by this command all
      the same, which is the only thing this census asks.

    :return: The names, as one set.
    """
    written = {var for names in _SERVICE_TOKEN_VARS.values() for var in names}
    written |= {var for pairs in _SERVICE_DEFAULT_VARS.values() for var, _default in pairs}
    written |= {_QSERVER_ZMQ_PRIVATE_KEY_VAR, _QSERVER_ZMQ_PUBLIC_KEY_VAR}
    written |= set(_VOLUME_INITIALIZED_VARS)
    written |= set(_STORE_ISSUED_VARS)

    # Imported rather than spelled again, from the module that defines them for
    # the bridge. Guarded because this runs on every deploy while the writer
    # that uses these names runs only on a bluesky one: an environment where the
    # import fails is an environment where that writer cannot run either, so the
    # census loses nothing it could have enforced.
    try:
        from osprey.services.bluesky_bridge.devices._specs_from_env import (
            READBACKS_ENV,
            SETPOINTS_ENV,
            SUBSTRATE_ENV,
        )
    except ImportError:  # pragma: no cover - the bridge package is always present
        logger.debug("Bluesky substrate env names unavailable; omitted from the writer census")
    else:
        written |= {SUBSTRATE_ENV, SETPOINTS_ENV, READBACKS_ENV}
    return written


def _preflight_pinned_writers(repo_root: Path | str) -> list[str]:
    """Refuse a pin on a name the deploy itself writes.

    ``env.pinned`` says a variable's value comes from this repo's chain and
    nowhere else. A machine-minted name cannot satisfy that: the deploy writes
    it into ``.env`` — on the first start, or whenever it is missing — so the
    declaration and the provisioner contradict each other by construction, and
    the two enforcement checks either side of this one would then be policing a
    value ``osprey up`` put there itself. Refusing the *declaration* is what
    makes pinned-and-minted an impossible state rather than a race between two
    parts of the same command.

    Checked against the whole writer census (:func:`_deploy_written_env_vars`),
    not the services this deploy enables, so the answer is a property of the
    profile and does not change with the config. Runs before anything reads or
    writes the chain, because a profile making an unsatisfiable statement is
    wrong independently of what the files currently hold.

    :param repo_root: The deployment repo root, holding the profile.
    :return: The empty list, when nothing pinned is machine-written.
    :raises RuntimeError: ``env.pinned`` names at least one written variable.
    """
    declared = pinned_env_keys(repo_root)
    if not declared:
        return []
    offenders = sorted(declared & _deploy_written_env_vars())
    if not offenders:
        return []

    logger.error(
        "Pinned variable is machine-minted: env.pinned declares %s, which the deploy "
        "writes into %s itself.\n"
        "  A pin says the chain under this root is the variable's only source, and these "
        "names are provisioned by `osprey up` — the service token mint, the service "
        "defaults, the bluesky provisioners, or the credential `--reuse-stores` restores "
        "from a surviving data volume. The declaration cannot hold against a writer in "
        "the same command.\n"
        "  %s",
        ", ".join(offenders),
        COMPOSE_ENV_FILENAME,
        " ".join(f"unpin {name}; it is machine-minted." for name in offenders),
    )
    raise RuntimeError(
        f"env.pinned declares {len(offenders)} machine-minted variable(s) (see report "
        f"above): {', '.join(offenders)}. "
        + " ".join(f"unpin {name}; it is machine-minted." for name in offenders)
    )


def _report_chain_overrides(repo_root: Path) -> tuple[list[str], list[str]]:
    """Report what ``.env`` overrides in ``.env.shared``, at two severities.

    The severity split *is* the feature. A deployment repo is expected to
    override shared defaults — that is what the local file is for — so warning
    about every override would train operators to scroll past the one case that
    is not deliberate. Deliberate overrides get a single info line naming the
    keys; the warning is reserved for stale pins (see
    :func:`_classify_local_overrides`).

    Answering the stale-pin question needs the shared file's *previous*
    contents, so this both reads and advances the record at
    :data:`ENV_CHAIN_STATE_RELPATH`. It advances it with one deliberate
    exception: a key currently reported as stale keeps its recorded digest
    rather than adopting today's shared value. Without that, the warning would
    fire exactly once — the next deploy would find the shared value unchanged
    since the record and fall silent while the pin was still stale. Holding the
    entry back makes the report persist until the operator resolves it, either
    by dropping the key from ``.env`` or by setting it to a value of their own.

    Recorded at preflight rather than after ``compose up`` returns: this runs at
    the last gate before the stack is started, and the non-detached start path
    replaces this process outright, so there is no "afterwards" to write from on
    every path. The record therefore means "the chain as of the most recent
    deploy that got this far", which is what the comparison needs.

    **Names only, never values** — on both severities, same rule as the
    shadowing warning below.

    A third kind of override, one of a name declared under ``env.pinned``, is
    not reported here at either volume. It has already refused the deploy, on
    the pre-mint chain, in :func:`_preflight_pinned_overrides`; it is classified
    out here so that a caller reaching this function on its own cannot file a
    contradicted pin under "working as intended".

    :param repo_root: The deployment repo root, holding the chain.
    :return: ``(deliberate, stale)`` key names, each sorted.
    """
    shared_path = repo_root / ENV_SHARED_FILENAME
    local_path = repo_root / COMPOSE_ENV_FILENAME
    shared = parse_dotenv_file(shared_path) if shared_path.is_file() else {}
    local = parse_dotenv_file(local_path) if local_path.is_file() else {}
    previous = _read_chain_state(repo_root)

    deliberate, stale, _ = _classify_local_overrides(
        shared, local, previous, pinned_env_keys(repo_root)
    )

    if deliberate:
        _report_fact(
            f"Local {COMPOSE_ENV_FILENAME} overrides {ENV_SHARED_FILENAME} for "
            f"{len(deliberate)} variable(s): {', '.join(deliberate)}. That is the chain "
            "working as intended, with the local file winning. This is reported once, by "
            "name, and never with a value."
        )

    if stale:
        logger.warning(
            "Stale pin: %s still holds the value %s carried BEFORE it changed: %s\n"
            "  These were last deployed with the shared default of the day copied into the "
            "local file. The shared file has since moved on; the local pin has not, so the "
            "stack starts on the superseded value and looks healthy doing it. Values are "
            "never printed.\n"
            "  To adopt the shared value: remove the listed name(s) from %s. To keep a local "
            "value on purpose: set it to the value this host wants. An override that is "
            "not just the old default is reported at info level, not here.",
            COMPOSE_ENV_FILENAME,
            ENV_SHARED_FILENAME,
            ", ".join(stale),
            local_path,
        )

    carried = {key: _value_digest(value) for key, value in shared.items()}
    for key in stale:
        carried[key] = previous[key]
    _write_chain_state(repo_root, carried)

    return deliberate, stale


def _rendered_chain_members(compose_files: list[str], repo_root: Path) -> set[str] | None:
    """Which chain files the rendered stack was built to read, or ``None``.

    Chain membership is fixed at render time — the rendered compose documents
    name the files that existed when ``osprey build`` ran — so the rendered
    ``env_file:`` lists are the record of what the build decided. This reads
    that record back out of them.

    The **fallback** record, used only when a render left no explicit membership
    marker (see
    :func:`osprey.deployment.compose_generator.read_rendered_env_chain`), which
    is what a project rendered before that marker existed looks like. Inferring
    from ``env_file:`` is strictly weaker, and the weakness is the reason the
    marker exists: see the ``None`` case below.

    ``None`` means *no record*, which is not the same as "an empty chain". A
    project that deploys no service with an ``env_file:`` block at all produces
    exactly the same evidence as one rendered against an empty chain, and a
    preflight that refused on that ambiguity would block deploys it has no
    finding against. So an absent record is silence, and only a *disagreement*
    between two known sets is drift. The explicit marker has no such blind spot
    — it records the empty chain as the empty list — so a project rendered with
    one is checked even in that case.

    Compose files are parsed as YAML here rather than scanned as text: an
    ``env_file:`` entry is structural (it is a list under a service), unlike the
    interpolation scan next door, where compose substitutes before it parses and
    a reference in an unparsed position still counts. Unreadable or unparseable
    files are skipped, same reasoning as the shadowing preflight — a broken
    render is the start's own error to raise.

    :param compose_files: The compose files this deploy will start, in ``-f``
        order. Relative entries resolve against *repo_root*.
    :param repo_root: The deployment repo root.
    :return: The chain file *names* the render binds, or ``None`` when no
        rendered document names one.
    """
    members: set[str] = set()
    for compose_file in compose_files:
        path = Path(compose_file)
        path = path if path.is_absolute() else repo_root / path
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        for service in (document.get("services") or {}).values():
            if not isinstance(service, dict):
                continue
            entries = service.get("env_file")
            if isinstance(entries, str):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for entry in entries:
                # Compose accepts a bare path or a mapping with a `path:` key.
                spelled = entry.get("path") if isinstance(entry, dict) else entry
                if isinstance(spelled, str) and Path(spelled).name in ENV_CHAIN_FILENAMES:
                    members.add(Path(spelled).name)
    return members or None


def _preflight_env_chain_drift(compose_files: list[str], repo_root: Path | str) -> None:
    """Refuse the deploy when the chain on disk is not the chain that was rendered.

    Membership is a **build-time** decision. The rendered documents name the
    chain files that existed when the project was built — in the worker's
    ``env_file:`` list, and in what the interpolation delivery was shaped for —
    so a file added or removed afterwards changes nothing about what the stack
    reads. Adding ``.env.shared`` to a repo built without one does not put its
    values into the containers; removing it from a repo built with one points a
    rendered ``env_file:`` at a path that is gone.

    Neither failure announces itself. The first comes up healthy on values the
    operator believes they just set; the second fails inside compose with a path
    error that names the file but not the reason it is expected. Both are one
    re-render away from correct, so this refuses and names ``osprey build``.

    A refusal rather than a warning, unlike the shadowing preflight next door,
    because the two situations are not alike: exporting over the store is a
    legitimate gesture with a legitimate outcome, while a chain that does not
    match its render has no outcome anybody asked for.

    Scoped to the SHARED half. ``.env`` is exempt on both sides: the deploy
    creates it (the token mint, a few steps above this call) and refuses without
    it, so its appearance between a build and a start is this command's own
    doing rather than a drift an operator introduced.

    What the render decided is read from the marker it wrote
    (:func:`osprey.deployment.compose_generator.read_rendered_env_chain`),
    falling back to what the rendered ``env_file:`` lists imply
    (:func:`_rendered_chain_members`) for a project rendered before that marker
    existed. Either way, a render that recorded nothing is silence rather than
    a refusal — see the fallback's own note on why "no record" must not be read
    as "an empty chain".

    :param compose_files: The compose files this deploy will start, in ``-f``
        order.
    :param repo_root: The deployment repo root, holding the chain.
    :raises RuntimeError: The rendered membership and the on-disk chain differ.
    """
    from osprey.deployment.compose_generator import read_rendered_env_chain

    root = Path(repo_root).expanduser().absolute()
    # The render's own marker is authoritative — it records what the build
    # decided, including the empty chain, which nothing can be inferred from.
    # Only a render that predates the marker falls back to the weaker reading.
    recorded = read_rendered_env_chain(root)
    rendered = (
        set(recorded) if recorded is not None else _rendered_chain_members(compose_files, root)
    )
    if rendered is None:
        return

    # `.env` is compared away on BOTH sides, because its presence is never a
    # build-time decision to drift from. It is the deployment's mandatory secret
    # store: `osprey up` refuses outright when it is missing, and mints the
    # service tokens INTO it a few steps above this check — so on a project
    # built before it existed (the ordinary `init` -> `build` -> `up` order,
    # where `init` writes only the shared half) the deploy would be reporting
    # its own mint as operator drift and refusing the first start of every new
    # deployment. What the render genuinely fixes, and what this therefore still
    # compares, is the SHARED half: adding `.env.shared` to a project built
    # without one puts none of its values into the containers, and removing one
    # points a rendered `env_file:` at a path that is gone.
    rendered = rendered - {ENV_LOCAL_FILENAME}
    on_disk = {path.name for path in chain_files(root)} - {ENV_LOCAL_FILENAME}
    if on_disk == rendered:
        return

    added = sorted(on_disk - rendered)
    removed = sorted(rendered - on_disk)
    detail = []
    if added:
        detail.append(f"  Added since the build, and NOT read by this stack: {', '.join(added)}")
    if removed:
        detail.append(f"  Expected by the build but gone from disk: {', '.join(removed)}")

    logger.error(
        "Env chain drift: this project was built to read %s, but %s is on disk now.\n"
        "%s\n"
        "  Which env files the stack reads is decided when the project is rendered, not when "
        "it starts. A file added afterwards contributes nothing, and one removed "
        "afterwards leaves the render pointing at a path that is gone. Either way the stack "
        "would start on a set of values nobody chose.\n"
        "  Run `osprey build` to re-render against the chain as it stands, then `osprey up`.",
        ", ".join(sorted(rendered)),
        ", ".join(sorted(on_disk)) if on_disk else "no chain file at all",
        "\n".join(detail),
    )
    raise RuntimeError(
        "env chain preflight failed: the chain on disk does not match the one this project "
        "was built to read (see report above). Run `osprey build`, then `osprey up`."
    )


def _preflight_env_shadowing(
    compose_files: list[str],
    repo_root: Path | str,
    *,
    environ: Mapping[str, str] | None = None,
    provider: ComposeProvider | None = None,
) -> list[str]:
    """Warn — or, for a pinned name, refuse — when a shell export and the env chain disagree.

    The chain under ``<repo>`` — ``.env.shared`` then ``.env``, later winning —
    is the deployment's one secret store, and every compose invocation is
    pointed at it explicitly (see :func:`_env_file_args`). But the env file is
    not the last word on document interpolation, and *which* word it is depends
    on the provider (see below). Either way a stale export means the stack can
    start on a value its volumes were never initialized with — an openobserve
    that will not accept the password in the chain, a database whose data
    directory answers to the chain's value while the container is handed
    another one.

    The same precedence the mint already designs to (``_ensure_service_tokens``
    skips a var the process env sets, so no divergent value is ever written
    into the chain): the export is honoured, the store is left intact, and
    nothing on either path tells the operator the two disagree. This is that
    telling.

    **Which side wins is provider-scoped, and inverted between the two.** Docker
    Compose substitutes the exported value — the env file is its lowest
    precedence source — while podman-compose reads its env file *first* and
    ignores the export. So the same divergence means "the stack runs on your
    shell's value" under one provider and "your export is being silently
    dropped" under the other, and a single wording could only be right for one
    of them. With no *provider* given, both are named rather than guessing:
    the ambiguity is the honest report, and the remedy that resolves it (unset
    the name) is the same either way.

    Only variables the rendered compose files actually interpolate are compared,
    against the merged chain — what the deployment's env files collectively say
    after ``.env`` has been laid over ``.env.shared``, which is the value
    compose is handed however the fragment is spelled. An exported name the
    chain does not set is not a divergence *for an unpinned name* (there is
    nothing for it to contradict, and the export is the only source there is),
    and a chain key no compose file reads cannot change what starts.

    For a **pinned** name that same shape is the refusal at its sharpest rather
    than a non-event: the declaration says the chain is the variable's only
    source, so an export the compose files interpolate is a shell supplying a
    value the deployment says a shell does not supply — and with nothing in the
    chain to disagree with, no later reading of the repo would show it. So the
    refusal below is scoped to what compose interpolates and what the shell
    exports, not to what the chain happens to carry.

    The chain's own layering is reported separately, at two severities, by
    :func:`_report_chain_overrides` — called from here so that one preflight
    covers everything the env chain can quietly get wrong.

    **A warning, never a refusal — for every name the deployment has not
    pinned.** Exporting a variable over the store is a legitimate gesture — a
    one-off run against another host's credentials, a rotation in progress — and
    a deploy that refused it would break the escape hatch to protect people from
    using it. A name declared under ``env.pinned`` is the case where the
    deployment has already said the escape hatch does not apply to it: the chain
    is that variable's only source, and under Docker Compose an export is the
    one way a value reaches the stack from around the chain without touching a
    file. So a divergent export of a pinned name refuses, and the remedy is to
    unset it.

    The refusal does not vary by provider, though the wording does. Under
    podman-compose the export reaches nothing rather than winning, but a repo
    deployed under both providers must not be startable on one host and refused
    on the other for the same shell state, and an export silently dropped is
    still a shell disagreeing with a name the deployment says it owns. What
    changes per provider is which of those two things the operator is told.

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
        directory and the home of the env chain compose is pointed at.
    :param environ: The process environment to compare against. ``None`` reads
        the live one, overlaid with the shell values the CLI's entry-time
        ``.env`` load replaced (:func:`~osprey.utils.config.dotenv_shell_overrides`)
        — the comparison is against what the operator's shell actually
        exported, which the in-process override would otherwise have erased.
    :param provider: The detected compose provider, when the caller has probed
        for one, selecting which precedence the warning states. ``None`` names
        both.
    :return: The shadowed variable names, sorted. Empty when there is no
        divergence, no chain file, or nothing interpolated.
    :raises RuntimeError: A name declared under ``env.pinned`` is shadowed.
    """
    root = Path(repo_root).expanduser().absolute()
    _report_chain_overrides(root)

    chain = chain_files(root)
    if not chain:
        # No store to be shadowed: compose is given no --env-file at all, so the
        # process env is the only source and nothing is being overridden.
        return []

    # The values the chain resolves to, NOT the `env.pinned` declaration read
    # further down — the two are different questions about the same variables,
    # and this one is "what does the store say". A chain that resolves to
    # nothing is not an early exit: it means every name a compose file reads is
    # decided by the shell, which is exactly what a pin forbids.
    chain_values = merge_chain(root)

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
        if name in chain_values and name in process_env and process_env[name] != chain_values[name]
    )

    # The pinned half asks a WIDER question than the advisory one, and has to:
    # an unpinned name the chain never sets is not a divergence (there is
    # nothing for the export to contradict, and the export is the only source
    # there is), while a PINNED name the chain never sets is the divergence at
    # its worst. The declaration says the chain decides that variable, so an
    # export compose interpolates is a shell deciding it instead — and with no
    # chain entry to compare against, nothing downstream would ever notice.
    # Exact agreement with the chain still clears it, as it does above.
    declared = pinned_env_keys(root)
    refused = sorted(
        name
        for name in referenced
        if name in declared and name in process_env and process_env[name] != chain_values.get(name)
    )
    advisory = [name for name in shadowed if name not in declared]
    if not advisory and not refused:
        return []

    if provider is ComposeProvider.PODMAN_COMPOSE:
        mechanics = (
            "podman-compose resolves interpolation from the env file BEFORE the calling "
            "shell, so the ENV-FILE value is the one substituted and the export you set "
            "reaches nothing."
        )
        remedy = (
            f"To change what starts: edit the chain; an export cannot reach this stack. "
            f"To stop the disagreement: unset {' '.join(advisory)}."
        )
    elif provider is ComposeProvider.DOCKER_V2:
        mechanics = (
            "Docker Compose substitutes the EXPORTED value into the compose files "
            "(--env-file is its lower-precedence source), so the stack starts on the "
            "shell's value while the chain keeps the one its volumes were initialized "
            "with."
        )
        remedy = (
            f"To start on the chain's value: unset {' '.join(advisory)}. To adopt the "
            f"exported one: edit the chain to match; a volume that already exists "
            f"keeps the credential it was created with, whichever value the container is "
            f"handed."
        )
    else:
        mechanics = (
            "The supported providers invert this: Docker Compose substitutes the EXPORTED "
            "value (--env-file is its lower-precedence source), while podman-compose reads "
            "its env file first and ignores the export. The provider was not probed for "
            "this message, so both are named rather than one guessed."
        )
        remedy = f"Either way, to leave the chain as the only source: unset {' '.join(advisory)}."

    # Everything the escape hatch still covers is reported first, so a refusal
    # below does not take the rest of the picture down with it.
    if advisory:
        _warn_fact(
            f"shell export disagrees with this deployment's env chain: {', '.join(advisory)}",
            f"exported here with a value that differs from the one the chain resolves to "
            f"({' + '.join(str(path) for path in chain)}). {mechanics} Values are never printed.",
            remedy,
        )

    if refused:
        logger.error(
            "Pinned variable shadowed by a shell export: %s\n"
            "  This deployment pins these names under env.pinned, which says the chain "
            "(%s) is their only source. %s Either way a shell is deciding, or being silently "
            "dropped by, a variable the deployment declared its own. Values are never "
            "printed.\n"
            "  To deploy: unset %s. To let a shell decide this name after all: drop it from "
            "env.pinned in the profile and re-run `osprey build`.",
            ", ".join(refused),
            " + ".join(str(path) for path in chain),
            mechanics,
            " ".join(refused),
        )
        raise RuntimeError(
            f"env chain preflight failed: {len(refused)} pinned variable(s) are shadowed by a "
            f"shell export (see report above). Unset them, or drop them from env.pinned."
        )

    return shadowed


def _declared_passthrough_names(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """The host variables each service declares under ``services.<name>.env``.

    The per-service env axis: a list of host environment variable NAMES a
    service hands to its container, written by the author in either the nested
    (``services.<name>.config.env``) or the dotted
    (``config: {"services.<name>.env": [...]}``) spelling. Both spellings are
    merged into this one block during the build, so the rendered ``config.yml``
    the deploy is handed is the single merged truth — the same block
    ``templates/services/_env_axis.j2`` renders from.

    Every "declares nothing" shape answers with an empty mapping rather than
    raising: no ``services`` block, a service block that is not a mapping, an
    ``env`` value that is not a list, and non-string or empty entries inside one.
    The names are validated where the profile is parsed
    (``build_profile_schema.env_names_errors``), so a malformed value here means
    a hand-edited ``config.yml``, which is the deploy's to survive rather than
    to refuse over.

    :param config: The deployment's rendered config.
    :return: ``{VARIABLE_NAME: [service, ...]}``, services in config order, so a
        report can say which service asked for a name as well as which name.
    """
    services = config.get("services") if isinstance(config, Mapping) else None
    if not isinstance(services, Mapping):
        return {}

    declared: dict[str, list[str]] = {}
    for service, block in services.items():
        if not isinstance(block, Mapping):
            continue
        names = block.get("env")
        if isinstance(names, str) or not isinstance(names, Sequence):
            continue
        for name in names:
            if isinstance(name, str) and name:
                declared.setdefault(name, []).append(str(service))
    return declared


def _preflight_declared_env_unset(
    compose_files: list[str],
    repo_root: Path | str,
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Warn when a service declares a host variable nothing on this host sets.

    ``services.<name>.env`` is a promise to hand a host variable to a container,
    and the renderer keeps it with a bare ``NAME: ${NAME}``. Deliberately bare:
    a ``:-`` default would mean inventing a value the renderer knows nothing
    about, and a ``:?`` would abort the whole compose document over one name
    that only a single service asked for (see ``_env_axis.j2``). What that
    leaves is compose substituting the EMPTY STRING — so an unset name reaches
    the container set-and-empty, not absent, and code reading it as
    ``os.environ.get(NAME, default)`` gets ``""`` rather than its default.

    That is the right runtime behaviour and a terrible thing to discover from
    inside a container. The declaration is the only evidence anyone intended the
    variable to carry something, and this is the one moment the deploy holds
    both halves — what was declared, and what this host actually has. So it says
    so, by name, here.

    **Advisory, never a refusal.** A declared name with nothing behind it is a
    perfectly ordinary state: an optional site proxy, a knob left at its
    in-container default, a variable this host does not need and the next one
    does. Refusing would make the axis unusable for exactly the optional
    passthroughs it is best at, and it would contradict the empty-when-unset
    semantics the renderer is built around. The complement — a name the
    deployment cannot run without — is what ``env.required`` is for, and
    :func:`_required_env_problems` is where that one does refuse.

    Scoped from both ends, so the report stays worth reading:

    * only names some rendered compose file actually interpolates, via
      :func:`_compose_interpolated_vars` — a service declared but not deployed
      contributes no compose document, so nothing it declares is reported;
    * only names nothing gives a value: neither this repo's env chain
      (``.env.shared`` under ``.env``) nor the process environment. The union
      rather than a precedence order, for the same reason
      :func:`_required_env_problems` takes it — the question is existence, not
      which value wins.

    An empty value counts as no value, again as in :func:`_required_env_problems`:
    ``NAME=`` in the chain and no ``NAME`` at all put the identical empty string
    in the container, so reporting one and not the other would split a single
    outcome across two answers.

    **Names only, never values** — the rule every report on this chain holds to.

    :param compose_files: The compose files this deploy will start, in ``-f``
        order. Relative entries resolve against *repo_root*. Unreadable entries
        are skipped: a missing render is the start's own error to raise.
    :param repo_root: The deployment repo root, holding the env chain.
    :param config: The deployment's rendered config, carrying the declarations.
    :param environ: The environment to check against. ``None`` reads the live
        one overlaid with the shell values the CLI's entry-time ``.env`` load
        replaced, matching what the stack is actually started with.
    :return: The declared-but-unset names, sorted. Empty when nothing is
        declared, nothing is interpolated, or everything declared has a value.
    """
    declared = _declared_passthrough_names(config)
    if not declared:
        return []

    root = Path(repo_root).expanduser().absolute()
    referenced: set[str] = set()
    for compose_file in compose_files:
        path = Path(compose_file)
        path = path if path.is_absolute() else root / path
        try:
            referenced |= _compose_interpolated_vars(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    if not referenced:
        return []

    if environ is None:
        from osprey.utils.config import dotenv_shell_overrides

        process_env: Mapping[str, str] = {**os.environ, **dotenv_shell_overrides()}
    else:
        process_env = environ

    chain = merge_chain(root)
    unset = sorted(
        name
        for name in declared
        if name in referenced
        and not (chain.get(name) or "").strip()
        and not (process_env.get(name) or "").strip()
    )
    if not unset:
        return []

    asked_by = "; ".join(f"{name} -> {', '.join(declared[name])}" for name in unset)
    _warn_fact(
        f"declared for passthrough, but unset on this host: {', '.join(unset)}",
        "A service's `env:` list names host variables it hands to its container "
        f"({asked_by}). Nothing sets these here: not .env, not .env.shared, not the "
        "environment. So compose substitutes the EMPTY STRING, and each container "
        "starts with the variable set and empty rather than absent. Values are "
        "never printed.",
        "Set them in .env at the repo root if the containers are meant to receive "
        "something, or drop them from the service's `env:` list in the build profile "
        "and re-run `osprey build` if they are not.",
    )
    return unset


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


def _report_port_conflicts(conflicts):
    """Render the host-port conflict report as the CLI's one failure shape.

    :func:`~osprey.deployment.host_ports.format_conflict_report` already writes
    the three parts a failure has, in the order :func:`osprey.cli.output.fail`
    wants them: its first line is what failed, its last is the one thing to do
    about it, and every line between is why. Splitting it there is what lets the
    report keep its own wording while wearing the ``✗``/``→`` shape every other
    refusal wears. Blank spacer lines are dropped -- the indent under the summary
    is what makes the body subordinate, so the spacers have nothing left to do.

    The report goes to stderr, because :func:`~osprey.cli.output.fail` always
    does: a caller piping stdout somewhere keeps its failures on the terminal.
    The record moves to the INFO family with it. An ``ERROR`` record renders
    through the altitude gate, so keeping one would print the whole report a
    second time underneath the block above it; the failure itself still reaches
    the caller as the :class:`RuntimeError` raised next to this call.

    :param conflicts: Conflicts from
        :func:`~osprey.deployment.host_ports.find_port_conflicts`
    """
    report = format_conflict_report(conflicts)
    summary, *rest = report.splitlines()
    remedy = rest.pop() if rest else None
    output.fail(summary, "\n".join(line for line in rest if line.strip()) or None, remedy)
    logger.key_info(report)


def _preflight_host_ports(config, compose_files):
    """Abort the deploy if a published host port is already taken.

    Parses the published bindings out of the rendered compose files and checks
    them for intra-deploy duplicates and external listeners (see
    :mod:`osprey.deployment.host_ports`). On any conflict the report is printed
    in the CLI's one failure shape and a :class:`RuntimeError` is raised so the
    caller aborts before running a single container-touching command.

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
    _report_port_conflicts(conflicts)
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

# The install target every archiver error names. pymongo is a core dependency,
# so a deploy that hits the seeder without it is running from an incomplete
# environment -- which is worth saying plainly rather than surfacing a bare
# ImportError, or pointing at an extra that no longer exists.
_ARCHIVER_INSTALL_TARGET = "osprey-framework"

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

    pymongo is a core dependency, but an environment can still be missing it —
    a partial install, a stale venv — and without it the seeder cannot run. The
    only place that would otherwise become visible is minutes later, after the
    project image has been built and the store container is already up. Checking
    here, beside the token mint, turns that into an immediate error naming the
    install.

    The reason names ``sys.executable`` rather than only the package, because
    the environment is the whole confusion: see
    :class:`~osprey.deployment.errors.ArchiverClientMissingError`.

    :param config: Raw deploy config.
    :raises ArchiverClientMissingError: if the store is deployed and pymongo is
        absent from the interpreter running this process.
    """
    if not _archiver_store_deployed(config):
        return
    try:
        import pymongo  # noqa: F401
    except ImportError as exc:
        raise ArchiverClientMissingError(
            reason=(
                "This project deploys the archiver store (services.mongodb), and "
                "OSPREY seeds its history from this process — "
                f"{sys.executable} — not from the project's build/.venv and not "
                "in the project image. pymongo is not importable there. A "
                "`dependencies:` entry in the build profile installs it into "
                "those, not into this interpreter."
            ),
            remedy=(
                "Reinstall osprey where the CLI runs:\n"
                f"    pip install --upgrade {_ARCHIVER_INSTALL_TARGET}"
            ),
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
        logger.debug("No machine model in this project; no scenarios to re-apply after the reseed")
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
    # The reseed's closing line. The archive rewrite that ran inside
    # `apply_scenarios` has no step of its own, so its counts ride here rather
    # than on a second line -- but only when there was a rewrite to report: a
    # `None` archiver means it never ran, and a `skipped` one already says so
    # in words the reseed has no room for.
    step = f"scenarios re-applied: {', '.join(result.active)}"
    archiver = result.archiver
    if archiver is not None and not archiver.skipped:
        step += f" ({archiver.describe()})"
    _report_step(step)


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


def _stage_archiver_store(
    config, compose_files, env, project_dir, *, keep_base=False, provider=None
) -> None:
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
    :param provider: The compose provider this deploy resolved, so the staging
        invocation is shaped like the ``up`` that follows it. ``None`` is the
        docker shape.
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

    run_env = runtime_env(config, {**env, **compose_provider_env(provider, project_dir)})
    # The invocation contract, same as _start_stack: one provider-shaped base,
    # anchored on the repo root, so staging is correct from any working
    # directory rather than only from a root the caller chdir'd into. The store
    # this brings up is the same compose project the `up` below it starts, so a
    # shape that differed here would create its container against a different
    # project directory than the one that goes on to run it.
    base_cmd = compose_base_cmd(
        get_runtime_command(config),
        compose_files,
        project_dir,
        _env_file_args(project_dir, provider),
        provider,
    )

    _report_group("archiver")
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
            _report_step("archive already seeded, skipping the base seed")
            return

        if comparison.state is SeedState.MISMATCH:
            _report_knob_diff(comparison)
            if keep_base:
                logger.warning(
                    "Keeping the existing base (--keep-archiver-base). The stored history "
                    "still describes the OLD knobs, so archiver reads will not match what "
                    "this profile declares."
                )
                return
            _report_fact(
                "Rebuilding the base series. Recorded history is discarded with it. The "
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
        # Before the work, not after it: this is the only warning an operator
        # gets that the next thing to happen is measured in minutes.
        _report_step(
            f"seeding the archive base: {len(channels):,} channels over "
            f"{knobs.retention_days} days (minutes on a first deploy)"
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
        _report_step(f"archive base: {report.describe()}")

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


def _stage_ariel_store(config, compose_files, env, project_dir, *, provider=None) -> None:
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
    one panel among many, and a control room whose channels, plan runs and archive are
    all up should not be denied them because its search tab could not be
    provisioned — but the warning names the command that finishes the job, so the
    gap is never silent.

    :param config: Raw deploy config.
    :param compose_files: Rendered compose file paths for this deploy.
    :param env: The environment the deploy hands compose.
    :param project_dir: Root of the built project (holds ``.env``).
    :param provider: The compose provider this deploy resolved, so the staging
        invocation is shaped like the ``up`` that follows it. ``None`` is the
        docker shape.
    """
    if not _ariel_store_deployed(config):
        return

    from osprey.simulation import apply as simulation_apply

    run_env = runtime_env(config, {**env, **compose_provider_env(provider, project_dir)})
    # The invocation contract, same as _stage_archiver_store: one
    # provider-shaped base, anchored on the repo root, so the store this brings
    # up joins the same compose project the `up` below it starts.
    base_cmd = compose_base_cmd(
        get_runtime_command(config),
        compose_files,
        project_dir,
        _env_file_args(project_dir, provider),
        provider,
    )

    _report_group("ariel")
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
        _report_step(f"logbook seeded: {seeded} entries")


# ---------------------------------------------------------------------------
# Staged graph-store bring-up
# ---------------------------------------------------------------------------

# How long the staged graph store gets to accept a bolt connection. The same
# budget ARIEL's store gets, and for the same reason: Neo4j opens its port only
# once the store is recovered and the plugins the image fetches at every start
# are loaded, so what this waits out is a refusal rather than a slow answer.
_GRAPHDB_HEALTH_TIMEOUT_S = 90.0
_GRAPHDB_HEALTH_POLL_S = 2.0

# Trivial round-trip that proves the store is up, authenticated and answering
# Cypher. Deliberately not a count of anything: the wait is about reachability,
# and a query whose cost grows with the graph would make a big store look down.
_GRAPHDB_PING_CYPHER = "RETURN 1 AS ok"

# The one command that finishes the job by hand, named in every warning below
# and by the graphdb health category's remedy text, so an operator reading
# either is pointed at the same verb. Imported from the module both sides
# already share their vocabulary through, rather than spelled here a second
# time: two copies would agree only until one of them was edited.
_GRAPHDB_RECOVERY_HINT = GRAPHDB_SEED_COMMAND


def _graphdb_store_deployed(config: dict) -> bool:
    """True when this deploy runs the graph store itself.

    Membership in ``deployed_services`` is the whole test. A project pointed at
    a Neo4j someone *else* runs (the explicit ``services.graphdb.uri`` case) must
    never have that store bootstrapped or imported into by a local deploy — the
    graph is a mirror, and rebuilding somebody else's mirror is not this deploy's
    call to make.
    """
    return GRAPHDB_SERVICE_NAME in (config.get("deployed_services") or [])


def _graphdb_connection(config: dict, project_dir: Path):
    """Resolve what to dial, as whom, and with which password, for THIS project.

    The password is read from ``<project_dir>/.env`` by name, never from the
    ambient environment — the same rule :func:`_ariel_store_config` follows, and
    for the same reason: a deploy is routinely driven from another directory,
    where an exported ``GRAPHDB_PASSWORD`` belongs to somebody else's deployment.

    :param config: Raw deploy config.
    :param project_dir: Root of the built project (holds ``.env``).
    :returns: The resolved
        :class:`~osprey.deployment.graphdb_service.GraphdbConnection`.
    """
    from osprey.deployment.graphdb_service import resolve_graphdb_connection
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = Path(project_dir) / ".env"
    env = parse_dotenv_file(env_path) if env_path.is_file() else {}

    block = (config.get("services") or {}).get(GRAPHDB_SERVICE_NAME) or {}
    return resolve_graphdb_connection(block, env=env)


def _graphdb_config_dir(project_dir: Path) -> Path:
    """The directory holding this project's ``config.yml``.

    What a relative ``services.graphdb.ttl_path`` resolves against — see
    :func:`_graphdb_ttl_text`. Derived from the project root the deploy was
    handed rather than from :func:`osprey_connectors.workspace.resolve_config_path`,
    which falls back to the process working directory: a deploy driven with
    ``--repo`` from somewhere else would otherwise resolve the corpus against
    whatever directory the operator happened to be standing in.

    Both project shapes are covered by one rule, the same one
    ``resolve_config_path`` applies to a working directory: the render one zone
    down in ``build/`` when it is there, and the project root itself otherwise —
    a container's project directory, or a legacy flat project, IS its own render.
    """
    from osprey.utils.workspace import rendered_config_path

    rendered = rendered_config_path(project_dir)
    return rendered.parent if rendered.is_file() else Path(project_dir)


def _graphdb_ttl_text(ttl_path: str, project_dir: Path) -> tuple[Path, str]:
    """Read the configured TTL corpus, resolving its path the documented way.

    ``ttl_path`` follows the ``facility_knowledge.bundle_path`` rule — ``~``
    expanded, an absolute value used as written, a relative one resolved against
    the ``config.yml`` directory — because ``osprey knowledge seed-graph`` reads
    the same key by the same rule. Two resolutions of one key would mean the
    deploy and the verb could seed a store from different files while both
    reporting success.

    :param ttl_path: The configured ``services.graphdb.ttl_path`` value.
    :param project_dir: Root of the built project.
    :returns: The resolved path and its text.
    :raises OSError: If the corpus is not on disk or cannot be read — reported
        by the caller's envelope, never fatal.
    """
    from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path

    resolved = resolve_bundle_path(ttl_path, _graphdb_config_dir(project_dir))
    return resolved, resolved.read_text(encoding="utf-8")


def _wait_for_graphdb_store(connection, deadline: float) -> None:
    """Block until the staged store answers Cypher on these credentials, or give up.

    Probes with the connection the bootstrap is about to use, so a rejected
    password surfaces here — named, and beside the deploy step that owns it —
    rather than as a bootstrap error that reads like a plugin problem.

    :param connection: The resolved
        :class:`~osprey.deployment.graphdb_service.GraphdbConnection`.
    :param deadline: :func:`time.monotonic` instant to give up at.
    :raises RuntimeError: if the store is still unreachable at ``deadline``.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    last: Exception | None = None
    while True:
        try:
            with graph_seeder.open_session(
                connection.uri, connection.username, connection.password
            ) as session:
                session.run(_GRAPHDB_PING_CYPHER).consume()
                return
        except Exception as exc:  # noqa: BLE001 — every failure here is "not yet"
            last = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"the graph store did not accept a connection within "
                    f"{_GRAPHDB_HEALTH_TIMEOUT_S:.0f}s: {last}"
                ) from last
            time.sleep(_GRAPHDB_HEALTH_POLL_S)


def _bootstrap_and_seed_graphdb(config: dict, project_dir: Path, connection) -> None:
    """Bootstrap the staged store and, on a first bring-up only, import its corpus.

    One session for the whole sequence, and each step gated on the one before:

    * **bootstrap** — always, and idempotent. A store whose graph config is not
      osprey's (:attr:`~osprey.services.facility_knowledge.seeder.graph_seeder.BootstrapStatus.DIFFERS`)
      is reported and left alone: n10s refuses to re-initialize a configured
      store, so importing into it would load a corpus under assumptions about the
      graph's shape that do not hold. Only ``--force`` can resolve that, and only
      by wiping — which a deploy does not get to do to data it did not write.
    * **seed** — only into an EMPTY graph (zero ``(:Resource)`` nodes, n10s's own
      bookkeeping not counted), and only when a corpus is configured. A deploy may
      fill a blank; it may not overwrite a graph somebody seeded.
    * **marker** — only after the import reports success. n10s commits in batches,
      so a failed import leaves triples behind; a marker written regardless would
      label that half-graph a good seed and every later run would report it
      unchanged and skip it forever.

    :param config: Raw deploy config.
    :param project_dir: Root of the built project.
    :param connection: The resolved graph-store connection.
    """
    from osprey.deployment.graphdb_service import resolve_graphdb_service_config
    from osprey.services.facility_knowledge.seeder import graph_seeder

    settings = resolve_graphdb_service_config(config)
    ttl_path = settings.ttl_path if settings is not None else None

    with graph_seeder.open_session(
        connection.uri, connection.username, connection.password
    ) as session:
        bootstrapped = graph_seeder.bootstrap(session)
        if not bootstrapped.ok:
            logger.warning(
                f"The graph store came up, but its {bootstrapped.message}. Nothing was "
                f"imported, so its contents are whatever was already there. Run "
                f"`{_GRAPHDB_RECOVERY_HINT} --force` from {project_dir} to wipe it and "
                f"re-seed under osprey's settings."
            )
            return
        _report_step("graph store bootstrapped")

        if graph_seeder.resource_count(session) > 0:
            # The normal second-deploy path: a graph that already carries a
            # corpus is left exactly as it is, and silently — this is not a
            # problem, and a warning here would cry wolf on every redeploy.
            return

        if ttl_path is None:
            _report_fact(
                "graph store bootstrapped but not seeded: no services.graphdb.ttl_path "
                f"is configured. Set one and run `{_GRAPHDB_RECOVERY_HINT}` to import a "
                "corpus."
            )
            return

        resolved, text = _graphdb_ttl_text(ttl_path, project_dir)
        imported = graph_seeder.import_ttl(session, text)
        if not imported.ok:
            logger.warning(
                f"The graph store came up but importing {resolved} failed "
                f"({imported.termination_status}: {imported.extra_info}), so graph queries "
                f"will return little or nothing. Everything else in this deploy is "
                f"unaffected. Fix the corpus and run `{_GRAPHDB_RECOVERY_HINT} --force` "
                f"from {project_dir}."
            )
            return

        graph_seeder.write_marker(session, graph_seeder.ttl_sha256(text))
        _report_step(f"graph seeded: {imported.triples_loaded} triples")


def _stage_graphdb_store(config, compose_files, env, project_dir, *, provider=None) -> None:
    """Start the graph store, bootstrap it, and seed it on a first bring-up.

    The knowledge-graph counterpart of :func:`_stage_ariel_store`, and staged
    ahead of the full bring-up for the same kind of reason: a store that comes up
    unbootstrapped and empty answers every query with zero rows, which reads as
    "the data is wrong" rather than "nothing was ever loaded". Doing it after the
    consumers start would leave a stack that is only correct once somebody
    restarts it.

    Failure here warns and returns rather than aborting the deploy. The graph is
    one search surface among many, and a control room whose channels, plan runs
    and archive are all up should not be denied them because its graph could not
    be provisioned — but every warning names
    ``osprey knowledge seed-graph`` (see :data:`_GRAPHDB_RECOVERY_HINT`), so the
    gap is never silent.

    :param config: Raw deploy config.
    :param compose_files: Rendered compose file paths for this deploy.
    :param env: The environment the deploy hands compose.
    :param project_dir: Root of the built project (holds ``.env``).
    :param provider: The compose provider this deploy resolved, so the staging
        invocation is shaped like the ``up`` that follows it. ``None`` is the
        docker shape.
    """
    if not _graphdb_store_deployed(config):
        return

    run_env = runtime_env(config, {**env, **compose_provider_env(provider, project_dir)})
    # The invocation contract, same as _stage_ariel_store: one provider-shaped
    # base, anchored on the repo root, so the store this brings up joins the same
    # compose project the `up` below it starts.
    base_cmd = compose_base_cmd(
        get_runtime_command(config),
        compose_files,
        project_dir,
        _env_file_args(project_dir, provider),
        provider,
    )

    _report_group("graphdb")
    up_cmd = base_cmd + ["up", "-d", GRAPHDB_SERVICE_NAME]
    logger.debug(f"Running command:\n    {' '.join(up_cmd)}")
    run_captured(up_cmd, env=run_env, spool_name="graphdb-store-up", repo_root=project_dir)
    _report_step("graph store started")

    try:
        connection = _graphdb_connection(config, project_dir)
        _wait_for_graphdb_store(connection, time.monotonic() + _GRAPHDB_HEALTH_TIMEOUT_S)
        _bootstrap_and_seed_graphdb(config, project_dir, connection)
    except Exception as exc:  # noqa: BLE001 — reported, never fatal (see docstring)
        logger.warning(
            f"The graph store could not be bootstrapped or seeded, so graph queries will "
            f"return nothing. Everything else in this deploy is unaffected. Run "
            f"`{_GRAPHDB_RECOVERY_HINT}` from {project_dir} once the store is reachable. "
            f"Cause: {exc}"
        )
        return


def _stage_openobserve_identity(config, compose_files, env, project_dir, *, provider=None) -> None:
    """Start the telemetry store, then provision the account that writes to it.

    Staged like ARIEL's store and the archiver's, and for a reason neither of
    them has: the ingest credential is issued by the STORE, so it cannot be
    written before the store is running. Doing it here rather than after the
    bring-up is what makes it work on both start shapes at once, because an
    attached ``osprey up`` replaces this process with compose and never comes
    back to run anything afterwards.

    The store is brought up on its own first, exactly as ``up`` will leave it:
    the same compose project, the same env chain, so the ``up`` below reconciles
    a container that is already correct rather than recreating it.

    :param config: Raw deploy config.
    :param compose_files: Rendered compose file paths for this deploy.
    :param env: The environment the deploy hands compose. A harvested token is
        mirrored into it when it already carries the name (see
        ``openobserve_provision._mirror_into_environment``).
    :param project_dir: Root of the built project (holds ``.env``).
    :param provider: The compose provider this deploy resolved, so the staging
        invocation is shaped like the ``up`` that follows it.
    """
    from osprey.deployment import openobserve_provision

    if not openobserve_provision.store_deployed(config):
        return

    run_env = runtime_env(config, {**env, **compose_provider_env(provider, project_dir)})
    base_cmd = compose_base_cmd(
        get_runtime_command(config),
        compose_files,
        project_dir,
        _env_file_args(project_dir, provider),
        provider,
    )

    _report_group("openobserve")
    up_cmd = base_cmd + ["up", "-d", openobserve_provision.SERVICE]
    logger.debug(f"Running command:\n    {' '.join(up_cmd)}")
    run_captured(up_cmd, env=run_env, spool_name="openobserve-store-up", repo_root=project_dir)
    _report_step("telemetry store started")

    outcome = openobserve_provision.provision_ingest_identity(
        config, Path(project_dir) / COMPOSE_ENV_FILENAME, env=env
    )
    if outcome.action == "verified":
        _report_step("ingest identity verified")
    elif outcome.wrote_token:
        _report_step("ingest identity provisioned")


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


def _collect_unmet_preconditions(
    config: dict,
    repo_root: Path | str,
    env: dict,
    *,
    dev_mode: bool,
    web_terminals_enabled: bool,
) -> None:
    """Probe every cheaply checkable start precondition, and report them all.

    A start has several preconditions that are answerable from this
    deployment's own files, before anything is built or started. Raised one at a
    time, they cost the operator a whole deploy attempt per finding: refuse on
    the first, fix it, re-run, wait, meet the second. This asks all of them and
    raises once, so a deployment with four problems is described in one refusal.

    Every probe is pure — it reads files and this build's version metadata,
    writes nothing, starts no process, and asks the container runtime nothing.
    That is not a stylistic preference at this position in the sequence: the
    steps around it provision credentials and are ordered against each other, so
    a probe with a side effect would silently be a step in that order.

    The refusals stay where they are. Each probe's raising counterpart runs
    later, unchanged, and would refuse the same start on its own — this is a
    reporting pass in front of them, not a replacement for them.

    Args:
        config: Raw deploy config.
        repo_root: The deployment repo whose renders and secret files are read.
        env: The environment the project-image build would run with.
        dev_mode: Whether ``--dev`` was passed.
        web_terminals_enabled: Whether the web tier is part of this deploy;
            three of the probes are about artifacts only it has.

    Raises:
        UnmetPreconditionsError: One or more probes reported a problem. Nothing
            has been built or started.
    """
    from osprey.deployment.errors import UnmetPreconditionsError
    from osprey.deployment.web_terminals.provision import web_terminal_preflight_problems

    findings: list[tuple[str, str]] = []
    if web_terminals_enabled:
        findings.extend(web_terminal_preflight_problems(config, repo_root=repo_root))
    # Ahead of the pin below because it is about the chain every service reads
    # rather than about one image's build, and the deploy meets it first. Sees
    # the chain as it stands AFTER every mint above, so a token the deploy just
    # provisioned is not reported as missing -- see _required_env_problems.
    findings.extend(_required_env_problems(repo_root, env))
    if (pin := _unreleased_pin_problem(config, env, dev_mode)) is not None:
        findings.append(pin)

    if findings:
        raise UnmetPreconditionsError(findings)


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
        _report_fact(
            "No services are configured for this project: deployed_services is empty in "
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

    # Same shape, for the other configured host path: when services.qmd.models_dir
    # is set, the sidecar's models come from that directory over a read-only mount
    # instead of from the image. A missing or mis-named file there does not fail
    # the deploy — it makes the sidecar try to download the model it cannot find,
    # on a host that was configured this way because it has no route out. Checked
    # here, before the build the setting is meant to shorten.
    preflight_qmd_models_dir(config)

    # And the same shape once more for the graph store: a `graphdb` in
    # deployed_services with no `services.graphdb` block describes no store at
    # all — no ports, no image, no corpus. Refused here, beside the other
    # config-only preflights and before anything touches the host, rather than
    # left to surface as a store that comes up on ports nothing was told about
    # and answers every query with zero rows.
    preflight_graphdb_config(config)

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

    # Refuse a pin the deploy itself would write over. Ahead of the override
    # refusal below because it is a statement about the profile alone: a
    # declaration that contradicts a provisioner is wrong whatever the chain
    # currently holds, and saying so first keeps the operator from resolving an
    # override on a name that should never have been pinned.
    _preflight_pinned_writers(repo_root)

    # Refuse a local file that overrides a name the deployment pins. On the
    # PRE-MINT chain, so the refusal is about lines the operator wrote rather
    # than about the ones the mint below is going to append, and so a deploy
    # doomed by a contradicted pin aborts having provisioned nothing.
    _preflight_pinned_overrides(repo_root)

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

    # Auto-configure the bluesky bridge's EPICS-substrate plan devices for a
    # VA-backed Bluesky stack (additive; no-op unless both bluesky and
    # virtual_accelerator are deployed) -- see _ensure_bluesky_substrate_env.
    _ensure_bluesky_substrate_env(config, env_path)

    # Provision the bluesky plan stack's 0MQ key material before compose mounts
    # it: the RE manager's control-socket keypair into .env, and the document
    # plane's CURVE certificates into data/bluesky_curve/. Both are no-ops
    # without the bluesky service, and both are additive (existing material
    # always wins) so a redeploy never rotates keys under a running stack.
    # Neither is gated on the connector -- see
    # _ensure_bluesky_document_plane_certs for why a browse-only deploy still
    # gets its certificates.
    _ensure_bluesky_control_plane_keys(config, env_path)
    _ensure_bluesky_document_plane_certs(config, env_path)

    # Carry a pre-rename .env.production onto .env.users before anything reads
    # it -- the web-terminal preflight below is the first reader, and it would
    # otherwise refuse a deploy whose secrets are sitting right there under the
    # old name. Scoped to web-terminal deploys because they are the only ones
    # that own the artifact; see migrate_users_env for the both-files rule.
    if web_terminals_enabled:
        migrate_users_env(repo_root)

    # Refuse a chain that is not the one this project was rendered against:
    # membership is fixed at build time, so a file added or removed since then
    # changes nothing about what the stack reads. Ahead of the advisory below
    # because there is no point reporting on a chain the render does not match.
    _preflight_env_chain_drift(compose_files, repo_root)

    # Which compose implementation is behind the resolved runtime, answered once
    # for every invocation this start makes: the env-file fragment, the argv,
    # the subprocess environment and the precedence the warning below states are
    # all downstream of it.
    #
    # Below every refusal that reads only this deployment's own files, because
    # this one asks the HOST a question (it runs `<runtime> compose version`).
    # A deploy that is going to abort on a port collision, a missing dependency
    # or a drifted env chain should not first go and start a provider process to
    # be told something it will never use. Above the advisory below, which is
    # the first thing whose wording depends on the answer.
    provider = _compose_provider(config)

    # Advisory, and last of the .env preflights so it reads the files every
    # provisioner above has finished writing: the shell and the env chain can
    # disagree, and which side compose substitutes is provider-scoped, so a
    # stale export silently starts the stack on a value its volumes never
    # adopted (or is silently dropped). Also reports what the local file
    # overrides in the shared defaults -- one info line for the deliberate
    # ones, a warning for a stale pin. Names the variables, never the values;
    # see _preflight_env_shadowing. Before the image build below, so the
    # operator sees it in seconds rather than after the minutes a build takes.
    _preflight_env_shadowing(compose_files, repo_root, provider=provider)

    # The other half of the same question, and the same volume: a service can
    # also declare a host variable under `services.<name>.env` that nothing on
    # this host sets. The renderer hands it over as a bare `${NAME}` by design,
    # so compose substitutes the empty string and the container starts with the
    # variable set-and-empty -- a state only visible from inside it. Advisory,
    # because a declared-but-absent passthrough is an ordinary thing (an
    # optional proxy, a knob this host does not need); `env.required` is where a
    # deployment says a variable is not optional, and that one refuses. Beside
    # the advisory above so one place covers everything the env chain can
    # quietly get wrong, and above the image build for the same reason.
    _preflight_declared_env_unset(compose_files, repo_root, config)

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
        _report_fact("dev mode: DEV_MODE set for the containers")

    # Report every remaining precondition this deployment's own files can
    # answer, in one refusal rather than one deploy attempt each. The position
    # is pinned from both sides.
    #
    # Above: it must follow every provisioner that WRITES what the probes read,
    # or it would refuse what the deploy is about to provide. Two of those are
    # load-bearing here — `_ensure_service_tokens` and the bluesky key material
    # mint `.env` on a fresh repo (a deploy that has no `.env` at all has one by
    # now), and `migrate_users_env` carries a pre-rename deployment's secrets
    # onto `.env.users`, whose own comment names the web-terminal preflight as
    # that file's first reader. This pass now reads it one step earlier, so it
    # inherits the same constraint.
    #
    # Below: it must precede the first minutes-long step, which is the project
    # image build. Reporting in seconds is the whole point.
    #
    # Deliberately OUTSIDE the pass, and left to raise on their own:
    #  * everything `_ensure_service_tokens` refuses upstream — the
    #    forbidden-published-default credential and the `expose_network`
    #    empty-token raise — because that step MINTS. It cannot be probed
    #    without either duplicating the mint's rules or moving the mint, and it
    #    has already run by the time control reaches here.
    #  * the `--dev` failed-staging case. Whether a wheel stages is not knowable
    #    without staging it, and the staging is paired with its cleanup inside
    #    `_build_project_image`'s try/finally — hoisting it out to get a definite
    #    answer would leave a staged wheel behind on a refusal.
    #  * `preflight_web_terminals`' minting half (`_provision_auth_secrets`) and
    #    the auth-sidecar image check ordered against it, for the same reason.
    _collect_unmet_preconditions(
        config,
        repo_root,
        env,
        dev_mode=dev_mode,
        web_terminals_enabled=web_terminals_enabled,
    )

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
        preflight_web_terminals(config, repo_root=Path(repo_root))

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
            config,
            compose_files,
            env,
            Path(repo_root),
            keep_base=keep_archiver_base,
            provider=provider,
        )

    # Same placement and the same reason for ARIEL's store: the schema has to
    # exist before the consumers that read it start, and both deploy paths need
    # it. Ordered AFTER the archiver so the logbook is seeded against a machine
    # whose history is already in place — the two halves of one narrative, in the
    # order they document each other.
    _stage_ariel_store(config, compose_files, env, Path(repo_root), provider=provider)

    # And the graph store, on the same placement and for the same reason: the
    # corpus has to be in the graph before the surfaces that query it start, and
    # both deploy paths need it. No-op unless this project deploys the store
    # itself; never aborts (see _stage_graphdb_store).
    _stage_graphdb_store(config, compose_files, env, Path(repo_root), provider=provider)

    # And the telemetry store, for a third reason: its ingest credential is one
    # the STORE issues, so the only moment it can be harvested is with the store
    # running. Here rather than after the bring-up below because the attached
    # `up` never returns (it replaces this process with compose), so anything
    # that must happen on both start shapes has to happen before the branch.
    # No-op unless this project deploys the store itself.
    _stage_openobserve_identity(config, compose_files, env, Path(repo_root), provider=provider)

    if web_terminals_enabled:
        # The provider goes down with the fragment it shaped: that fragment is
        # already provider-specific (one merged file, or the repeated chain),
        # and the web path builds its own argv and its own subprocess
        # environment around it. Handed the fragment alone, it would splice a
        # podman-shaped --env-file into a docker-shaped argv and run it with no
        # COMPOSE_PROJECT_DIR at all. The repo root goes down with them for the
        # same reason: all three describe ONE invocation, and the web path
        # re-resolving the root can only answer the working directory — which
        # is where the podman shape would then write its merged document.
        deploy_up_web_terminals(
            config,
            compose_files,
            dev_mode,
            env,
            _env_file_args(repo_root, provider),
            provider,
            repo_root=Path(repo_root),
        )
        log_endpoint_summary(config, compose_files)
        return

    # Pin COMPOSE_PROJECT_NAME so this deploy owns its own compose project (and
    # volume namespace); without it compose derives the project from the first
    # -f file's directory, collapsing every deploy on the host into the shared
    # "services" project whose up/down cross-adopts sibling stacks. The provider
    # overlay rides in on the same env: for the shape that cannot take the
    # project directory in argv, this is where it is pinned instead.
    run_env = runtime_env(config, {**env, **compose_provider_env(provider, repo_root)})

    base_cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        _env_file_args(repo_root, provider),
        provider,
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
    _report_group("services")
    rm_cmd = base_cmd + ["rm", "-f"]
    logger.debug(f"Running command:\n    {' '.join(rm_cmd)}")
    run_captured(rm_cmd, env=run_env, spool_name="compose-rm", repo_root=repo_root, check=False)
    _report_step("cleared stopped containers")

    prebuilt = _resolve_prebuilt_images(config)
    if dev_mode and prebuilt:
        # Nothing to build: the tags are expected to be on the host already, and
        # the `up --no-build` below runs against them. A tag that is in fact
        # missing surfaces as compose's own "No such image", which names the
        # image that has to be loaded — better than anything a preflight here
        # could say.
        _report_step("skipped image build (prebuilt images)")
    elif dev_mode:
        # `osprey up --dev` re-bakes the local osprey checkout into a fresh
        # wheel on every run, but compose reuses the cached image tag (e.g.
        # <project>-dispatch:local) unless it is rebuilt — so a dev deploy must build.
        # Build in its OWN step, then `up --no-build`: a single `up --build` can
        # build a local-only tag and then fail container-create with
        # "No such image" under Docker's containerd image store. Non-dev has no
        # build step of its own, so unless the host says its images are prebuilt
        # it stays a plain `up` and compose's implicit build-on-up still covers a
        # build-only service that has no published upstream tag to pull.
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
    if dev_mode or prebuilt:
        # Non-dev never builds in a step of its own, so on a prebuilt host the
        # only build left to suppress is compose's implicit build-on-up. Without
        # this a pull-only mirror deploy would answer a missing tag by building
        # a locally-tagged impostor from the template's `build:` block instead
        # of failing on the image that never arrived. The switch reaches
        # compose's implicit builds only: explicit persona and auth-sidecar
        # builds stay governed by `image_source`.
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
        # after it would find nothing left to commit. The ledger flushes first
        # for the same reason: no summary card ever prints on this path, and
        # what this run wrote to disk has to be said by the process that wrote
        # it.
        output.flush_ledger()
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


# The env vars the host-network templates render to name the interface a
# service binds. Read off the RENDERED files rather than off the config keys
# behind them: the rendered value is what the process binds, and a hand-authored
# override (``services.event_dispatcher.bind``) reaches this check without this
# module having to know the key exists.
_HOST_BIND_ENV_VARS = ("FASTMCP_HOST", "DISPATCH_WORKER_BIND")

# The one ``network_mode`` spelling that puts a service in the host's network
# namespace. Mirrors host_ports._HOST_NETWORK_MODE, on the rendered side.
_HOST_NETWORK_MODE = "host"


def _binds_off_host(address: str) -> bool:
    """Whether binding *address* makes a host-namespace service reachable off-host.

    Loopback (``127.0.0.0/8``, ``::1``, ``localhost``) keeps the listening
    socket on this machine; a wildcard or a concrete interface address does not.
    Anything unparseable — a name, an uninterpolated ``${VAR}`` — is called
    reachable, because this answer arms a fail-closed guard and the safe way to
    be wrong is to arm it when it was not needed.
    """
    text = str(address).strip().strip("[]")
    if not text:
        return True  # an empty bind means every interface
    if text == "localhost":
        return False
    try:
        return not ipaddress.ip_address(text).is_loopback
    except ValueError:
        return True


def _rendered_environment(service: Mapping) -> dict[str, str]:
    """A rendered service's ``environment:`` block as a mapping.

    Compose accepts both the mapping spelling the templates emit and a
    ``KEY=value`` list; a rendered file that was hand-edited may hold either.
    """
    environment = service.get("environment")
    if isinstance(environment, Mapping):
        return {str(key): "" if value is None else str(value) for key, value in environment.items()}
    if isinstance(environment, list):
        pairs = {}
        for entry in environment:
            if isinstance(entry, str) and "=" in entry:
                key, _, value = entry.partition("=")
                pairs[key.strip()] = value.strip()
        return pairs
    return {}


def _host_network_bound_off_host(compose_files: list[str]) -> list[tuple[str, str]]:
    """Services in these compose files that bind off-host on the host network.

    A service in the host's network namespace publishes nothing — compose has no
    port map to publish — so :func:`_published_on_all_interfaces`, which reads
    ``ports:`` entries, cannot see it at all. What decides its reach is the
    interface it binds, which the host-mode templates render into a bind env var
    (:data:`_HOST_BIND_ENV_VARS`). Loopback keeps it on this machine; anything
    else is off-host reachable with no published port anywhere in the render.

    Read out of the rendered files for the same reason the wildcard check is: a
    start re-renders nothing, so the files are both what the build decided and
    what compose will act on — including an edit made to them after the build.

    A host-mode service naming no bind var at all counts as reachable: its
    binding is then whatever the process defaults to, which this cannot see.

    Returns:
        ``(service, bind address)`` pairs, sorted by service. The address is the
        rendered value, or ``""`` when the service names no bind var.
    """
    found: dict[str, str] = {}
    for compose_file in compose_files:
        try:
            doc = yaml.safe_load(Path(compose_file).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            # Debug, not a warning: the wildcard check just parsed this same
            # file and has already said so at warning level.
            logger.debug(
                f"Could not read compose file {compose_file} for the exposure check: {exc}"
            )
            continue
        services = doc.get("services") if isinstance(doc, Mapping) else None
        if not isinstance(services, Mapping):
            continue
        for name, service in services.items():
            if not isinstance(service, Mapping):
                continue
            if str(service.get("network_mode") or "").strip() != _HOST_NETWORK_MODE:
                continue
            environment = _rendered_environment(service)
            binds = [environment[var] for var in _HOST_BIND_ENV_VARS if var in environment]
            off_host = [bind for bind in binds if _binds_off_host(bind)]
            if binds and not off_host:
                continue
            found[str(name)] = off_host[0] if off_host else ""
    return sorted(found.items())


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

    Three independent ways to be reachable, and each one counts on its own. A
    services binding on a wildcard address is the obvious one. The second is the
    web-terminal stack: every service in it runs ``network_mode: host`` and its
    nginx "binds every interface, unrestricted" (``docker-compose.web.yml.j2``),
    so a web deploy is off-host reachable with no published port anywhere in
    ``compose_files`` — which is exactly the case a wildcard-binding check alone
    would call private.

    The third is a services-stack service the build put on the host network. It
    publishes no port either, so only the interface it was rendered to bind says
    whether it is reachable (:func:`_host_network_bound_off_host`). The
    templates bind loopback there, which is what keeps the default host-mode
    deployment private — but the bind is overridable, and an overridden one is
    as reachable as any wildcard publication.

    Args:
        config: The rendered config, read for whether the web stack is part of
            this deployment.
        compose_files: The services stack's rendered compose files.

    Returns:
        Whether this deployment is reachable off-host, for the token guard.
    """
    wildcard_services = _published_on_all_interfaces(compose_files)
    host_networked = _web_terminals_enabled(config)
    host_bound = _host_network_bound_off_host(compose_files)
    reasons = []
    if wildcard_services:
        reasons.append(f"{', '.join(wildcard_services)} publish on 0.0.0.0")
    if host_networked:
        reasons.append("the web-terminal stack runs on the host network")
    for service, address in host_bound:
        where = f"binds {address}" if address else "names no bind address"
        reasons.append(f"{service} runs on the host network and {where}")

    if reasons:
        _warn_fact(
            "this deployment is reachable from the network",
            f"{'; '.join(reasons)}; starting under the fail-closed service-token rules "
            "that reachability calls for",
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

    The four-zone start path. It reads ``build/config.yml`` and the compose
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
        _warn_on_timezone_drift(config)
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


def _warn_on_timezone_drift(config: dict) -> None:
    """Promote a host-vs-facility timezone divergence at the start of an up.

    The same divergence :func:`osprey_connectors.config._warn_on_tz_drift`
    records when a timestamp is first stamped — but that record is a library
    warning the altitude gate keeps off the terminal while a lifecycle verb
    runs, and it fires minutes into the start, wherever the first timestamp
    happens to be minted. The operator-facing copy of it belongs to the verb:
    stated once, up front, from the same two values.

    Only an explicit ``system.timezone`` counts, for the connectors check's
    reason: the implicit UTC default would make every host with ``$TZ`` set
    look divergent.
    """
    configured = (config.get("system") or {}).get("timezone")
    host_tz = os.environ.get("TZ")
    if not configured or not host_tz or host_tz == configured:
        return
    _warn_fact(
        "host timezone differs from the facility setting",
        f"$TZ={host_tz} but system.timezone={configured}; OS-level log timestamps "
        "will differ from the times the agent reports",
        "set both to the same zone (system.timezone is what the agent reports)",
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
        _report_fact(
            f"nothing to stop: no compose files in {BUILD_DIRNAME}/ and no containers "
            "labelled for this repo"
        )
        # The caveat stays in the transcript at the level it always had: the
        # label is applied at CREATE time, so a stack started before this
        # repo's containers were labelled is invisible to the sweep above.
        logger.key_info(
            f"There are no compose files in {repo_root / BUILD_DIRNAME} to run a "
            f"`down` from, and no container carries "
            f"{label_filter.removeprefix('label=')}. Containers get that label when "
            "they are created, so a stack started before this repo's containers were "
            "labelled is not visible here and may still be running. `osprey build` "
            "restores build/, after which `osprey down` works normally."
        )
        return

    _report_fact(
        f"no compose files in {BUILD_DIRNAME}/; stopping the {len(container_ids)} "
        "container(s) labelled for this repo instead (volumes and the compose "
        "network are kept)"
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

    The four-zone stop path, and the mirror of :func:`up_as_built`: it acts on
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
        # The refusal is caught, not propagated: a `down` whose compose provider
        # is unsupported must still reach the label sweep below, which talks to
        # the container runtime directly and is the whole recovery path for a
        # deployment compose cannot act on. When there ARE compose files the
        # same refusal is raised again below — this only keeps it from
        # pre-empting the fallback.
        try:
            deploy_down_web_terminals(
                config,
                os.environ.copy(),
                _env_file_args(repo_root, _compose_provider(config)),
                repo_root=repo_root,
            )
        except UnsupportedComposeProviderError as exc:
            logger.warning(
                "Skipping the web-terminal stack's compose teardown: %s\n"
                "  Its containers carry this deployment's own label, so the sweep "
                "below removes them if there is nothing else to stop.",
                exc,
            )

    if not compose_files:
        _down_by_label(config, repo_root)
        return

    provider = _compose_provider(config)

    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        _env_file_args(repo_root, provider),
        provider,
    )
    cmd.append("down")

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    # Captured, not streamed: the phase line this runs under is only readable if
    # compose's own output goes to the spool instead of the same terminal. The
    # pinned COMPOSE_PROJECT_NAME reaches compose through `env=` — unpinned,
    # `down` targets the shared "services" project rather than this deploy's.
    result = run_captured(
        cmd,
        env=runtime_env(config, {**os.environ, **compose_provider_env(provider, repo_root)}),
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
    provider = _compose_provider(config)

    if _web_terminals_enabled(config):
        deploy_down_web_terminals(
            config,
            os.environ.copy(),
            _env_file_args(repo_root, provider),
            repo_root=repo_root,
        )

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
        logger.debug("No existing compose files found, rebuilding...")
        _, compose_files = prepare_compose_files(config_path, dev_mode)
    else:
        logger.debug("Using existing compose files for 'down' operation:")
        for f in compose_files:
            logger.debug(f"  - {f}")

    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        _env_file_args(repo_root, provider),
        provider,
    )
    cmd.append("down")

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    # The COMPOSE_PROJECT_NAME pin must reach compose: `down` must target the
    # same project `up` created, or it either misses this deploy's containers
    # or (unpinned) tears down the shared "services" project. That pin is why
    # this used to `execvpe` rather than `execvp`; the captured run carries the
    # same environment through `env=`.
    run_env = runtime_env(config, {**os.environ, **compose_provider_env(provider, repo_root)})
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

    repo_root = resolve_repo_root(config, config_path)
    provider = _compose_provider(config)
    cmd = compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        compose_files,
        repo_root,
        _env_file_args(repo_root, provider),
        provider,
    )
    cmd.append("restart")

    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    subprocess.run(
        cmd, env=runtime_env(config, {**os.environ, **compose_provider_env(provider, repo_root)})
    )

    # After the restart, not before: the ingest credential is issued by a store
    # that has to be answering to issue it, and `compose restart` takes it down
    # and brings it back. The provisioner waits for readiness itself, so this
    # covers the case a restart exists to repair here too, which is a data
    # volume that was replaced under a `.env` still holding the old token.
    #
    # What a restart CANNOT do is deliver a newly harvested token to the
    # containers: `compose restart` reuses each container's existing definition,
    # so a value written now reaches them at the next `osprey up`. Provisioning
    # anyway is what makes the store and the `.env` agree in the meantime, which
    # is the half a later start cannot reconstruct.
    from osprey.deployment import openobserve_provision

    openobserve_provision.provision_ingest_identity(config, repo_root / COMPOSE_ENV_FILENAME)

    # If detached mode requested, detach after restart
    if detached:
        logger.debug("Services restarted. Running in detached mode.")


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

    clean_deployment(compose_files, config, repo_root=resolve_repo_root(config, config_path))

    deploy_up(config_path, detached=detached, dev_mode=dev_mode, expose_network=expose_network)
