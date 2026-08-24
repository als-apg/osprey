"""Renders the multi-user web-terminal deployment artifacts from a facility config's
``modules.web_terminals`` stanza.

Three artifacts come out of one facility config: the docker-compose overlay (one
service per user + nginx), the nginx routing fragment, and the static landing page.
All port arithmetic is delegated to :func:`osprey.deployment.web_terminals.ports.allocate_ports`
— this module only builds the per-service context list and hands it to Jinja2.
"""

from __future__ import annotations

import posixpath
import re
import shutil
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader

from osprey.deployment.compose_generator import (
    configured_ariel_mirror_path,
    repo_identity,
    repo_relative_mount_source,
    resolve_repo_root,
    user_audit_relpath,
)
from osprey.deployment.web_terminals.auth_credentials import (
    TERMINAL_SECRET_VAR_PREFIX,
    terminal_secret_var,
)
from osprey.deployment.web_terminals.personas import (
    SUPPORTED_MCP_TOPOLOGY,
    USERNAME_CHARSET_RE,
    as_dict,
    config_archiver_password_env,
    config_needs_ariel_mirror,
    config_needs_ariel_password,
    config_needs_dispatcher_token,
    config_needs_facility_bundle,
    config_needs_graphdb_password,
    config_needs_launch_token,
    effective_image_source,
    entry_requires_login,
    env_var_suffix,
    env_var_suffix_collisions,
    resolve_personas,
)
from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.utils.facility import resolve_facility_name
from osprey.utils.workspace import AUDIT_DIR_RELPATH, agent_data_base_dir

# Package-relative location of the .j2 sources (Tasks 1.3/1.6). Resolved via
# importlib.resources, NOT Path(__file__).parent, so this works from an installed
# wheel too (hatchling ships all of src/osprey as package data).
_TEMPLATE_PACKAGE_PATH = "templates/modules/web_terminals"

_COMPOSE_TEMPLATE = "docker-compose.web.yml.j2"
_NGINX_TEMPLATE = "nginx.conf.j2"
_LANDING_TEMPLATE = "landing.html.j2"

# Output paths are relative to the rendered docker-compose.web.yml.j2 itself, per
# that template's own mount-path contract (nginx/nginx.conf, nginx/landing.html).
_COMPOSE_OUTPUT = "docker-compose.web.yml"
_NGINX_OUTPUT = "nginx/nginx.conf"
_LANDING_OUTPUT = "nginx/landing.html"

# Per-container constant: every per-user app's service families (web + every
# registry companion family) bind this host, never a routable interface —
# nginx's reverse proxy is the only off-host path. Not
# config-driven: unlike the per-family ports, there is no config knob
# for this, since a facility that wants a per-user port reachable directly
# off-host would defeat the single-origin chokepoint this module exists to
# provide.
_LOOPBACK_BIND_HOST = "127.0.0.1"

# Default nginx image when `modules.web_terminals.nginx_image` is unset. Kept
# byte-identical to docker-compose.web.yml.j2's own `| default(...)` fallback so
# an absent config value renders the same image from either side.
_DEFAULT_NGINX_IMAGE = "nginx:1.27-alpine"

#: The authentication methods this deployment can actually serve: ``none`` (no
#: auth at all), ``password`` (OSPREY-managed scrypt hashes) and ``oidc``
#: (Authlib client against a facility IdP). Anything else is a hard config
#: error rather than a forward-compatible passthrough — nginx would emit the
#: ``auth_request`` seam against a sidecar that cannot answer for that method,
#: which does fail closed but only as an unactionable 403 on every request.
SUPPORTED_AUTH_METHODS = ("none", "password", "oidc")

#: Port the auth sidecar listens on when ``auth.port`` is unset. Deliberately
#: outside the per-user port families (which start at ``*_base_port`` and grow
#: with the roster) so a default-port sidecar can't collide with user N's
#: terminal; lint joins the effective value into its port-collision check.
_DEFAULT_AUTH_PORT = 9070

#: How long an unlocked-user entry stays valid when ``auth.session_lifetime``
#: is unset, in seconds (12 hours — one operator shift plus slack).
_DEFAULT_SESSION_LIFETIME = 12 * 60 * 60

#: Env-var *names* (never values) the sidecar reads its OIDC client credentials
#: from when the config doesn't name its own. The credentials themselves live
#: in ``.env.auth`` and never enter config.yml.
_DEFAULT_OIDC_CLIENT_ID_ENV = "OSPREY_AUTH_OIDC_CLIENT_ID"
_DEFAULT_OIDC_CLIENT_SECRET_ENV = "OSPREY_AUTH_OIDC_CLIENT_SECRET"

#: Label key the auth sidecar's service carries the sha256 digest of
#: ``.env.auth``'s content under (repo label convention: dotted ``osprey.*``
#: keys, as in the service templates' ``osprey.project.name``). Compose bakes
#: ``env_file`` content into a container at CREATION time, and a
#: service-definition change is the only recreate trigger every compose
#: implementation honours — podman-compose in particular never recreates on a
#: content-only file change. Stamping the digest into the definition turns an
#: ``.env.auth`` edit into exactly that trigger. The digest is not a secret:
#: the file's values are high-entropy mints and IdP-issued credentials, and
#: only the hash of the whole file — never a variable's value — is emitted.
AUTH_ENV_DIGEST_LABEL = "osprey.auth.env.digest"

#: Request header nginx carries a user's operator secret to that user's terminal
#: in. The app authenticates every HTTP/WS request against it, so nginx — the
#: single origin in front of every terminal — is what turns "reached the proxy"
#: into "is the operator". Spelled once here because the rendered snippet and
#: the app's own middleware are the two ends of one contract.
TERMINAL_SECRET_HEADER = "X-Osprey-Terminal-Secret"

#: Prefix of the per-user environment variable one user's operator secret is
#: minted into (deploy ``.env``) and read back out of (nginx's envsubst pass).
#: Re-exported from
#: :data:`~osprey.deployment.web_terminals.auth_credentials.TERMINAL_SECRET_VAR_PREFIX`
#: rather than spelled again: the mint writes the variable and this module reads
#: it, and a prefix defined twice is a prefix that can be minted one way and
#: referenced another. Kept under a render-side name because the templates and
#: the render errors below refer to it by that name.
TERMINAL_SECRET_ENV_PREFIX = TERMINAL_SECRET_VAR_PREFIX

#: Output-relative directory the per-user nginx *templates* land in — mounted
#: read-only at ``/etc/nginx/templates``, where the base image's entrypoint
#: envsubsts every file into ``NGINX_ENVSUBST_OUTPUT_DIR``. Deliberately a
#: directory of its own, separate from the rendered ``nginx/nginx.conf``: the
#: entrypoint processes the whole directory, so nothing that is not a per-user
#: snippet may live in it.
NGINX_TEMPLATES_OUTPUT_DIR = "nginx/templates"

#: How one per-user snippet is named. The user's own name, not its env-var
#: suffix, so the include the nginx template emits inside that user's location
#: can be spelled from the same ``svc.user`` the location itself is keyed by.
_SECRET_TEMPLATE_PREFIX = "secret-"
_SECRET_TEMPLATE_SUFFIX = ".conf.template"

#: What a roster name may look like once it is a FILENAME and an nginx
#: ``include`` argument. Wider than
#: :data:`~osprey.deployment.web_terminals.personas.USERNAME_CHARSET_RE`, which
#: the deploy path now applies in every auth method (the terminal-secret mint
#: runs whatever ``auth.method`` says, and ``_validate_usernames`` gates it), so
#: on that path this rule is the looser of two and never the one that refuses.
#: It is kept separate because ``render_web_terminals`` is also called with no
#: mint behind it — by lint, by tests, by any caller passing
#: ``terminal_secrets`` for a roster it did not provision — and what turning a
#: name into a FILE makes newly dangerous is narrower than what makes it a bad
#: identity: a name that stops being one path component, ``../`` escaping the
#: templates directory the write seam clears, or a separator splitting the
#: include into two arguments. Each refusal names its own rule, so an operator
#: reading one knows which of the two they broke.
_SECRET_TEMPLATE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

#: What the DERIVED variable suffix must look like. The filename rule above
#: admits ``.`` — legal in a filename and in an nginx ``include`` — while
#: :func:`~osprey.deployment.web_terminals.personas.env_var_suffix` maps only
#: ``-`` to ``_``, so ``alice.b`` would derive ``OSPREY_TERMINAL_SECRET_ALICE.B``.
#: That is not a legal environment-variable name: envsubst does not recognize
#: the reference, leaves it in the snippet verbatim, and nginx refuses to start
#: with ``invalid variable name`` — a crash-looping proxy whose error names
#: neither the user nor the config key. Checked separately from the filename so
#: each refusal can say which of the two rules was broken.
_ENV_VAR_SUFFIX_RE = re.compile(r"[A-Z0-9_]+")


def terminal_secret_env_var(username: str) -> str:
    """The environment variable one roster user's operator secret travels in.

    A thin alias of
    :func:`~osprey.deployment.web_terminals.auth_credentials.terminal_secret_var`,
    which is where the name is actually derived. There is one derivation, not
    two agreeing ones: the mint writes this variable into the deploy ``.env``
    and the rendered artifacts reference it, and a name derived twice is a name
    that can be minted one way and read another — which fails as a terminal
    that refuses every request rather than as anything visible at render time.

    The render-side spelling is kept because the templates, the render refusals
    and their tests all name it, and because this module is what a reader of
    the artifacts arrives at first.

    Args:
        username: The roster name, exactly as configured.

    Returns:
        ``OSPREY_TERMINAL_SECRET_<SUFFIX>``, with the suffix from
        :func:`~osprey.deployment.web_terminals.personas.env_var_suffix`.
    """
    return terminal_secret_var(username)


def _session_cookie_name(web_port: int) -> str:
    """The browser cookie one user's app hands out, as nginx must spell it.

    Read from the app's own
    :func:`~osprey.interfaces.common_middleware.session_cookie_name` rather than
    re-spelled here. The perimeter forwards exactly this cookie on the locations
    where it injects no operator secret, so a name assembled independently would
    forward a cookie the app does not read — a terminal that cannot hold a
    session, with nothing in either end's config to show why.

    Imported inside the function, like the theme reader further down: this
    module is on the deploy path and the interfaces package pulls in the web
    stack, which a render has no other reason to load.

    Args:
        web_port: The user's allocated ``web`` port — the value the app itself
            publishes as ``OSPREY_WEB_PORT`` and derives the cookie name from.

    Returns:
        ``osprey_terminal_session_<port>``.
    """
    from osprey.interfaces.common_middleware import session_cookie_name

    return session_cookie_name(web_port)


def _secret_template_path(username: str) -> str:
    """Output-relative path of one user's rendered nginx template snippet."""
    return (
        f"{NGINX_TEMPLATES_OUTPUT_DIR}/{_SECRET_TEMPLATE_PREFIX}{username}{_SECRET_TEMPLATE_SUFFIX}"
    )


def _secret_template_content(username: str) -> str:
    """One user's snippet: a single ``proxy_set_header`` naming that user's variable.

    Built here rather than as a ``.j2`` under ``templates/`` because it is one
    directive with one substitution, and because the file is an *nginx* template
    — its ``${...}`` is resolved by the container's envsubst pass at start, not
    by Jinja2 at render time. That is the whole mechanism: the secret's value
    never enters a rendered artifact, so nothing on disk in ``build/`` (or in a
    git status, or in an image layer) holds it.

    The comment lines carry no ``$`` for the same reason — envsubst does not
    know what a comment is, and would substitute a bare ``$NAME`` inside one.

    The substitution is QUOTED, which decides how a blank variable fails. envsubst
    resolves a present-but-empty variable to nothing, and the compose carrier
    guarantees the variable is always present (``${VAR:-}``), so a blank entry in
    the deploy ``.env`` reaches nginx as an empty substitution. Unquoted that
    emits ``proxy_set_header X-Osprey-Terminal-Secret ;`` — not a directive nginx
    accepts, so the proxy refuses to start and EVERY user's terminal is down over
    one blank variable, with an error naming a generated file rather than the
    roster. Quoted it emits an empty header value, which only that user's app
    refuses (it fails closed on a credential that does not match), so the blast
    radius of one blank variable is the one user it belongs to. The trade is
    deliberate: a quieter failure for a narrower one, on a condition the deploy
    gate (``_provision_terminal_secrets``) already refuses ahead of the render.
    """
    return (
        f"# Rendered by OSPREY (osprey.deployment.web_terminals.render).\n"
        f"# Injects {username}'s operator secret into requests proxied to that\n"
        f"# user's terminal. Included only from that user's authenticated\n"
        f"# location, so no terminal ever sees another user's secret. The value\n"
        f"# is substituted by nginx's entrypoint at container start from\n"
        f"# {terminal_secret_env_var(username)}, and appears in no rendered artifact.\n"
        f"proxy_set_header {TERMINAL_SECRET_HEADER} "
        f'"${{{terminal_secret_env_var(username)}}}";\n'
    )


def clear_nginx_templates_dir(dest: Path | str) -> list[Path]:
    """Empty the rendered nginx-templates directory beneath *dest*.

    Called by the write seam **before** it writes a render's output, and
    unconditionally — including on a render that carries no secrets at all.
    Without it a decommissioned user's snippet survives every later re-render:
    the base image's entrypoint envsubsts *every* file in the mounted templates
    directory, so nginx would keep emitting a ``proxy_set_header`` for a user
    who is no longer on the roster, from a variable the deploy ``.env`` may
    still hold. Stale routing is caught by the roster verbs re-rendering
    ``nginx.conf``; a stale *snippet* is not, because nothing else ever
    overwrites it.

    The whole subtree goes, not just the files sitting directly in it: the
    entrypoint walks the mounted directory recursively (a template in a
    subdirectory is processed into the matching subdirectory of the output),
    so a leftover one level down is exactly as live as one at the top. Removing
    the directory and recreating it empty is safe because it is nothing but
    render output in the disposable ``build/`` zone — unlike its parent, which
    holds ``nginx.conf`` and ``landing.html`` and is left untouched.

    The directory is recreated even when there was nothing to remove, because
    the nginx service bind-mounts it: a missing source is materialized by the
    container runtime as a root-owned directory (or refused outright), which is
    a confusing failure for a deployment that simply has no snippets yet.

    Args:
        dest: The artifacts destination the render mapping's relative paths are
            written beneath (``<repo>/build`` on every deploy path).

    Returns:
        Every file removed, in sorted order — empty on a first render, where the
        directory does not exist yet.
    """
    directory = Path(dest) / NGINX_TEMPLATES_OUTPUT_DIR
    removed: list[Path] = []
    if directory.is_dir():
        removed = sorted(path for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return removed


def _terminal_secret_artifacts(
    services: list[dict[str, Any]], terminal_secrets: dict[str, str], *, auth_method: str
) -> dict[str, str]:
    """Build one nginx template snippet per roster user with a GATED location.

    Every roster user is CHECKED — the four refusals below cover the whole
    roster, because every user's container reads its own secret whatever the
    perimeter does with it — but only a user whose ``/u/<user>/`` location
    actually ``include``s a snippet gets one written. That is exactly the
    predicate nginx.conf.j2 emits the include under (``auth_method != "none"
    and not svc.login_exempt``), spelled here so the two cannot disagree.

    A snippet for anyone else is a plaintext operator secret substituted into
    the nginx container's filesystem for no reader: with authentication off
    nginx injects no header at all (the browser reaches the app through that
    user's own ``?token=`` exchange), and a ``login: false`` entry is served
    ungated with the header explicitly CLEARED. Not written, so the invariant
    the nginx ``-T`` tests reason about — every file under
    ``/etc/nginx/osprey`` is included by exactly one location — holds by
    construction rather than by luck.

    Args:
        services: The resolved per-user service entries (``user`` and
            ``login_exempt`` are read).
        terminal_secrets: ``{username: secret}`` as provisioned into the deploy
            ``.env``. Only the KEYS are used — a value is read solely to refuse
            an absent one, and is never rendered anywhere.
        auth_method: The parsed ``auth.method`` (see :func:`_auth_tls_context`).
            ``"none"`` means no location is gated, so no snippet is written.

    Returns:
        ``{output-relative path: content}``, one entry per gated roster user —
        empty when nothing is gated.

    Raises:
        ValueError: On any of four fail-closed conditions — a roster name that
            is not usable as a single filename (see
            :data:`_SECRET_TEMPLATE_NAME_RE`), a name whose derived variable is
            not a legal environment-variable name (see
            :data:`_ENV_VAR_SUFFIX_RE`), two names that derive the SAME
            variable, or a user with no non-blank secret. Each of the four
            fails invisibly if it is allowed through: an illegal variable name
            is left verbatim by envsubst and kills nginx at start with
            ``invalid variable name``; two users sharing a variable share one
            secret, which is the isolation this carrier exists to establish; a
            blank secret renders a header that substitutes to empty, which nginx
            accepts and the app then refuses on every request — a terminal that
            401s behind a healthy-looking proxy, with nothing in the render to
            explain it. Every message names the offending user and variable.
    """
    unusable = [
        service["user"]
        for service in services
        if not _SECRET_TEMPLATE_NAME_RE.fullmatch(service["user"])
    ]
    if unusable:
        named = ", ".join(repr(name) for name in unusable)
        raise ValueError(
            f"modules.web_terminals.users {named} cannot carry an operator secret: "
            "each roster name becomes a single snippet filename under "
            f"{NGINX_TEMPLATES_OUTPUT_DIR}/ and the argument of the nginx `include` "
            "that reads it, so a name that is not one plain path component would "
            f"write outside that directory or split the directive. Rename them to "
            f"match {_SECRET_TEMPLATE_NAME_RE.pattern!r}"
        )
    illegal = [
        service["user"]
        for service in services
        if not _ENV_VAR_SUFFIX_RE.fullmatch(env_var_suffix(service["user"]))
    ]
    if illegal:
        named = ", ".join(f"{name!r} ({terminal_secret_env_var(name)})" for name in illegal)
        raise ValueError(
            f"modules.web_terminals.users {named} derive an illegal environment-variable "
            "name for their operator secret. env_var_suffix() only uppercases and maps "
            "'-' to '_', so any other punctuation survives into the variable: nginx's "
            "envsubst leaves such a reference verbatim and nginx then refuses to start "
            "with `invalid variable name`, which says nothing about the roster. Rename "
            f"them so the derived suffix matches {_ENV_VAR_SUFFIX_RE.pattern!r}"
        )
    collisions = env_var_suffix_collisions([service["user"] for service in services])
    if collisions:
        detail = "; ".join(
            f"{', '.join(repr(name) for name in names)} all key "
            f"{TERMINAL_SECRET_ENV_PREFIX}{suffix}"
            for suffix, names in collisions.items()
        )
        raise ValueError(
            f"modules.web_terminals.users collide on their operator-secret variable "
            f"({detail}): colliding names would share one secret, so nginx would inject "
            "the same credential into both terminals and neither could tell the two "
            "operators apart. Rename one of them. Checked whatever auth.method says, "
            "unlike the sidecar's own collision gate — this carrier is rendered even "
            "with authentication off, and the isolation it provides is the same"
        )
    missing = [
        service["user"]
        for service in services
        if not str(terminal_secrets.get(service["user"], "") or "").strip()
    ]
    if missing:
        named = ", ".join(f"{name} ({terminal_secret_env_var(name)})" for name in missing)
        raise ValueError(
            f"modules.web_terminals.users {named} have no operator secret. Every roster "
            "user needs one: nginx injects it as the "
            f"{TERMINAL_SECRET_HEADER} header on that user's location, and a terminal "
            "reached without it refuses every request. Mint the missing variables into "
            "the deploy .env (osprey up does this in preflight) and re-render"
        )
    return {
        _secret_template_path(service["user"]): _secret_template_content(service["user"])
        for service in services
        if auth_method != "none" and not service.get("login_exempt")
    }


def _container_agent_data_dir(config: Any, container_project_dir: str) -> str:
    """Where a per-user container's agent-data volume mounts, as that container sees it.

    The mount target has to be the directory the process inside the container
    actually writes to, and that directory is decided by ``agent_data.base_dir``
    — the same key :func:`osprey.utils.workspace.resolve_agent_data_root`
    resolves at runtime, anchored on the config's ``project_root`` (in-container:
    *container_project_dir*). Derived through the shared
    :func:`~osprey.utils.workspace.agent_data_base_dir` reader for that reason:
    a literal here is a second, silent spelling of the same setting, and a
    project that relocates its agent-data root would mount the volume at the old
    path while the agent writes to the new one — an unbacked directory in the
    container's writable layer, discarded at the next recreate.

    The join happens here rather than in the template because an absolute
    ``base_dir`` names the same absolute path on both sides and must NOT be
    re-anchored under the project directory; a template-level
    ``{{ dir }}/{{ base }}`` concatenation cannot make that distinction.

    Known limit: a ``~``-relative ``base_dir`` is anchored here like any other
    relative path, while ``anchored_path`` expands it against the container's
    ``HOME`` — which the compose template sets to ``/data/claude-config``. Such
    a project's agent data therefore lands in the claude-config volume rather
    than the agent-data one: misfiled, but persisted by the claude-config volume,
    so this is not the unbacked-directory data loss above (the agent-data volume
    then mounts over an empty ``~`` directory and holds nothing). Left unhandled rather
    than expanded here because resolving it would mean spelling that ``HOME``
    path a second time in this module, free to drift from the template that
    actually sets it — the same duplication this function exists to remove.
    """
    base = PurePosixPath(agent_data_base_dir(config))
    if base.is_absolute():
        return base.as_posix()
    return (PurePosixPath(container_project_dir) / base).as_posix()


def _container_bundle_dir(config: Any, container_project_dir: str) -> str | None:
    """Where the deployment's knowledge bundle mounts, as this container sees it.

    The per-service counterpart of :func:`_container_agent_data_dir`, and derived
    the same way and for the same reason: the mount target has to be the
    directory the processes inside the container actually read — the OKF panel
    and the ``facility_knowledge`` MCP server both resolve it from
    ``facility_knowledge.bundle_path``, anchored on the config's ``project_root``,
    which in-container is *container_project_dir*. A literal here would be a
    second spelling of that key, free to drift.

    Per-service rather than one shared path, because personas differ: two
    personas built from different projects have different
    ``container_project_dir`` values, so the same configured
    ``data/facility_knowledge`` resolves to two different in-container paths and
    a single hardcoded target would mount the bundle where only one of them
    looks. An ABSOLUTE ``bundle_path`` names the same absolute path on both
    sides and is NOT re-anchored — the same distinction
    :func:`_container_agent_data_dir` makes, and the same reason the join happens
    in Python rather than as a template concatenation.

    Known limit, shared with its sibling: the path is read from the DEPLOY
    config, not from each persona's own ``config.yml``. A persona that relocates
    its bundle relative to what the deploy config declares would have the
    directory bound at the deploy's path while its readers look at the persona's
    — visible as an empty bundle, not as silent data loss, since nothing writes
    to an unmounted target here. Reading it per persona would mean this function
    touching disk, which :func:`render_web_terminals` is contractually forbidden
    from doing.

    :return: The in-container mount target, or ``None`` when the deployment
        configures no bundle at all.
    """
    knowledge = as_dict(as_dict(config).get("facility_knowledge"))
    raw = knowledge.get("bundle_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    base = PurePosixPath(raw.strip())
    if base.is_absolute():
        return base.as_posix()
    return (PurePosixPath(container_project_dir) / base).as_posix()


def _container_mirror_dir(config: Any, container_project_dir: str) -> str | None:
    """Where the deployment's ARIEL qmd mirror mounts, as this container sees it.

    The mirror counterpart of :func:`_container_bundle_dir`, derived the same
    way and for the same reason: the target has to be the directory the
    exporter inside the container actually WRITES — it resolves
    ``ariel.enhancement_modules.qmd_export.mirror_path`` against the project
    root, which in-container is *container_project_dir* — or the bind covers
    nothing and the container's own logbook enhancements land in its writable
    layer, unindexed by the sidecar and gone at the next recreate. Read through
    the one reader every other consumer of the key uses
    (:func:`~osprey.deployment.compose_generator.configured_ariel_mirror_path`),
    ``settings`` winning over the module block exactly as the exporter merges
    them. Absolute values are not re-anchored, as for the bundle. Same known
    limit as its sibling: the path is read from the DEPLOY config, not each
    persona's own, and the lint reports a persona that relocated it.

    :return: The in-container mount target, or ``None`` when the deployment
        writes no mirror at all.
    """
    from osprey.deployment.compose_generator import configured_ariel_mirror_path

    raw = configured_ariel_mirror_path(as_dict(config))
    if raw is None:
        return None
    base = PurePosixPath(raw)
    if base.is_absolute():
        return base.as_posix()
    return (PurePosixPath(container_project_dir) / base).as_posix()


def _container_audit_dir(container_project_dir: str) -> str:
    """Where a per-user container's audit zone mounts, as that container sees it.

    The python executor appends its refusal log under
    ``<project root>/<AUDIT_DIR_RELPATH>``
    (:func:`osprey.services.python_executor.refusal_audit.audit_dir`), and the
    project root in-container is *container_project_dir*. Derived from the
    same constant the writer uses rather than spelled here, for the reason
    :func:`_container_agent_data_dir` gives: a path the container does not
    write to is a mount over an empty directory, while the log — the one
    record of what the agent tried and was refused — accumulates in the
    writable layer and vanishes at the next recreate.
    """
    return (PurePosixPath(container_project_dir) / AUDIT_DIR_RELPATH).as_posix()


def render_web_terminals(
    config: Any,
    auth_env_digest: str | None = None,
    dispatcher_personas: set[str] | None = None,
    ariel_personas: set[str] | None = None,
    launch_token_personas: set[str] | None = None,
    graphdb_personas: set[str] | None = None,
    facility_bundle_personas: set[str] | None = None,
    facility_bundle_gid: int | None = None,
    ariel_mirror_personas: set[str] | None = None,
    ariel_mirror_gid: int | None = None,
    audit_gid: int | None = None,
    archiver_password_personas: dict[str, str] | None = None,
    terminal_secrets: dict[str, str] | None = None,
) -> dict[str, str]:
    """Render the compose overlay, nginx fragment, and landing page for one facility config.

    Args:
        config: The parsed facility config, read defensively as nested dicts (same
            convention as :func:`osprey.deployment.web_terminals.lint.lint_web_terminals`
            — no assumption that ``config`` is a particular schema/dataclass type).
            Deterministic: the same inputs always render the same three artifacts,
            with no clock/random/filesystem inputs — which is why the digest below
            arrives as a parameter instead of being read from disk here.
        auth_env_digest: sha256 hex digest of ``.env.auth``'s current content,
            emitted as the :data:`AUTH_ENV_DIGEST_LABEL` label on the auth
            sidecar's service so a content change becomes a service-DEFINITION
            change — the one recreate trigger every compose implementation
            honours. The deploy path computes it via
            :func:`osprey.deployment.web_terminals.artifacts.write_web_terminal_artifacts`;
            ``None`` (the default, and the ``osprey scaffold web-terminals
            render`` path, which has no project root to digest) emits no label —
            harmless, because every ``osprey up`` re-renders through the
            artifacts seam with a current digest before its ``up``, so a
            label-less render can never reach a running stack stale.
        dispatcher_personas: Persona names whose project declares the EVENTS
            panel, and whose users therefore need the event dispatcher's bearer
            token. Resolved from disk by
            :func:`osprey.deployment.web_terminals.personas.personas_needing_dispatcher_token`
            and passed in for the same reason ``auth_env_digest`` is: this
            function reads no filesystem. Each such user gets
            ``EVENT_DISPATCHER_TOKEN`` in its OWN ``environment:`` block,
            interpolated by compose from the deploy ``.env`` — never written
            into this rendered artifact, and never into the shared
            ``.env.users``, which every user's container reads (see
            :func:`osprey.deployment.web_terminals.env_production._build_env_production_subset`).
            That split IS the tier boundary: a read-only persona must not hold a
            credential that can fire triggers. ``None`` (the default, and the
            ``osprey scaffold web-terminals render`` path, which has no project
            root to resolve against) emits no token line at all.
        ariel_personas: Persona names whose project configures ARIEL, and whose
            users therefore need the Postgres password ``osprey up`` minted into
            the deploy ``.env`` (see
            :func:`osprey.deployment.web_terminals.personas.personas_needing_ariel_password`).
            Same placement and same reason as ``dispatcher_personas``: both ARIEL
            consumers inside the container — the panel's server and the ``ariel``
            MCP server — resolve their DSN through
            :func:`osprey.services.ariel_search.config.resolve_ariel_dsn`, which
            reads ``ARIEL_DB_PASSWORD`` from the environment and otherwise falls
            back to the shipped default, so a container that never receives it
            authenticates with the wrong password against a Postgres initialized
            with the minted one. ``None`` emits no line.
        launch_token_personas: Persona names entitled to arm a Bluesky queue
            start, resolved from disk by
            :func:`osprey.deployment.web_terminals.personas.personas_needing_launch_token`.
            Same placement and same reason as ``dispatcher_personas`` — see that
            argument for why a per-persona credential belongs in the per-user
            ``environment:`` block and never in the shared ``.env.users``.
            The tier boundary bites hardest here: ``BLUESKY_LAUNCH_TOKEN`` is
            what lets the ``bluesky`` MCP server arm a queue start without a
            second confirmation, so a read-only persona holding it would be able
            to move hardware. ``None`` emits no line.
        graphdb_personas: Persona names whose project configures a graph store,
            and whose users therefore need the Neo4j password ``osprey up``
            minted into the deploy ``.env`` (see
            :func:`osprey.deployment.web_terminals.personas.personas_needing_graphdb_password`).
            Same placement and same reason as ``dispatcher_personas``: the
            consumer is the ``graph`` MCP server inside the container, which
            resolves what to dial and with which password through
            :func:`osprey.deployment.graphdb_service.resolve_graphdb_connection`
            — that reads ``GRAPHDB_PASSWORD`` from the environment and otherwise
            falls back to the shipped default, so a container that never
            receives it authenticates with the wrong password against a store
            initialized with the minted one, and every graph query fails on
            auth. Unlike ``launch_token_personas`` this set is not a tier
            boundary: Neo4j has one write-capable account, and read-only-ness is
            enforced by that server's read transactions rather than by which
            persona holds the credential. ``None`` emits no line.
        facility_bundle_personas: Persona names whose project names a
            ``facility_knowledge.bundle_path``, and whose container therefore
            gets the deployment's knowledge bundle bind-mounted (see
            :func:`osprey.deployment.web_terminals.personas.personas_needing_facility_bundle`).
            Resolved from disk and passed in for the same reason the credential
            sets are. Unlike them this grants a MOUNT rather than a secret, so it
            is not a tier boundary: the bundle is the same corpus every entitled
            persona reads, and the tier that matters for it is decided by what
            the agent inside is allowed to write. ``None`` emits no mount for
            any user.
        facility_bundle_gid: Group id of the bundle directory on the host,
            resolved from disk by
            :func:`osprey.deployment.compose_generator.shared_corpus_gid` and
            emitted as ``group_add:`` on every entitled service. This is the
            half of the sharing story that setgid alone cannot supply: the
            directory's setgid bit decides which group new files are OWNED by,
            but a container's process is a MEMBER of only the groups its image
            gives it, so without this the group bits grant nothing and access
            holds only where the deploying user's uid happens to equal the
            image's — an accident that breaks on any other host. ``None`` (the
            directory does not exist yet, or a platform with no gids) emits no
            ``group_add``, which is the honest render rather than a guessed
            group.
        ariel_mirror_personas: Persona names whose project runs an ARIEL qmd
            export that writes a mirror, and whose container therefore gets the
            deployment's mirror directory bind-mounted at the path its own
            exporter writes (see
            :func:`osprey.deployment.web_terminals.personas.personas_needing_ariel_mirror`).
            Same placement and same reason as ``facility_bundle_personas``:
            a mount, not a secret, resolved from disk and passed in. Without it
            an entry enhanced inside that container is written into the
            container's writable layer — never indexed by the sidecar, gone at
            the next recreate. ``None`` emits no mount for any user.
        ariel_mirror_gid: Group id of the mirror directory on the host, for
            the same ``group_add:`` reason as ``facility_bundle_gid``. ``None``
            emits no group for it.
        audit_gid: Group id of the deployment's audit zone on the host
            (``var/audit``), under which the deploy provisions one directory
            per roster user and this render binds it at the user's container
            audit path. Every per-user service joins it, because every
            container writes its refusal log there. ``None`` emits no group
            for it.
        archiver_password_personas: ``{persona_name: env_var_name}`` for the
            personas whose archiver connector authenticates with a password,
            resolved from disk by
            :func:`osprey.deployment.web_terminals.personas.personas_needing_archiver_password`.
            Same placement and same reason as ``dispatcher_personas``; a map
            rather than a set because the connector reads the variable its own
            ``archiver.<type>.password_env`` names, and the line emitted into
            the user's ``environment:`` block carries exactly that name (the
            control-assistant preset spells it ``MONGO_ROOT_PASSWORD``, which
            ``osprey up`` mints). Without it the agent's every archiver read
            fails with "Environment variable '…' is not set" while the same
            project works on the single-user host path, which reads the whole
            deploy ``.env``. ``None`` emits no line.
        terminal_secrets: ``{username: operator secret}`` as provisioned into
            the deploy ``.env``, resolved from disk and passed in for the same
            reason ``auth_env_digest`` is. Supplying it adds one
            ``nginx/templates/secret-<user>.conf.template`` to the returned
            mapping (see :data:`NGINX_TEMPLATES_OUTPUT_DIR`) for each roster
            user whose location is GATED — the only ones an ``include`` ever
            reads (see :func:`_terminal_secret_artifacts`); the three artifacts
            render byte-identically either way. **Only the keys are used.** A value is read solely to refuse an absent one — no
            secret value enters any rendered artifact, which is why the snippet
            is an nginx *template* naming
            :func:`terminal_secret_env_var`'s variable rather than a rendered
            conf holding the secret. ``None`` (the default, and the ``osprey
            scaffold web-terminals render`` path, which has no deploy ``.env``
            to read) emits no snippets at all: the render stays exactly what it
            was before this carrier existed. The per-service variable NAME and
            external origin are threaded on every render regardless — neither is
            a secret, and the compose reference resolves to empty when the
            deploy ``.env`` holds nothing.

    Returns:
        Mapping of output-relative-path to rendered content: the three artifacts
        ``docker-compose.web.yml``, ``nginx/nginx.conf`` and
        ``nginx/landing.html``, plus — only when ``terminal_secrets`` is
        supplied — one ``nginx/templates/secret-<user>.conf.template`` per
        roster user with a gated location. The write seam clears that directory
        first (see
        :func:`clear_nginx_templates_dir`), so a decommissioned user's snippet
        cannot survive a re-render.

    Raises:
        ValueError: If ``modules.web_terminals.nginx_port`` is missing/not an int,
            if a configured user can't resolve a full port-family set, if
            ``deploy.fqdn`` is missing while the deployment needs an external
            origin — at least one user configured (the host baked into
            ``OSPREY_TERMINAL_LANDING_URL``) or authentication enabled (see
            :func:`_external_origin`), if the ``auth``/``tls`` stanzas are
            incoherent (unknown ``auth.method``, ``tls.enabled`` without a
            cert/key pair, or authentication over cleartext HTTP without
            ``auth.allow_insecure_http``), if
            a roster entry's persona reference can't be resolved (see
            :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
            ``strict`` contract — render always resolves strictly), or if
            ``modules.web_terminals.mcp.topology`` is set to anything other than
            ``per_container_stdio`` (see :func:`_check_mcp_topology`), or if
            ``terminal_secrets`` is supplied and a roster user has no non-blank
            secret in it, or a roster name is not usable as one snippet filename
            (both :func:`_terminal_secret_artifacts`), or if
            ``auth_env_digest`` is not a sha256 hex digest — the value is
            templated verbatim into compose YAML, so anything but lowercase hex
            is rejected here rather than trusted not to carry YAML structure.
    """
    if auth_env_digest and not re.fullmatch(r"[0-9a-f]{64}", auth_env_digest):
        raise ValueError("auth_env_digest must be a sha256 hex digest")
    root = as_dict(config)
    facility = as_dict(root.get("facility"))
    registry = as_dict(root.get("registry"))
    web_terminals = as_dict(as_dict(root.get("modules")).get("web_terminals"))
    facility_prefix = facility.get("prefix") or ""

    _check_mcp_topology(web_terminals)

    resolved_users = resolve_personas(web_terminals, registry, facility_prefix, strict=True)

    base_ports = base_ports_from_config(web_terminals)
    services = []
    for entry in resolved_users:
        user_ports = allocate_ports(base_ports, entry["index"])
        services.append(
            {
                "user": entry["name"],
                "image": entry["image"],
                "project": entry["project"],
                "container_project_dir": entry["container_project_dir"],
                # Where this user's agent-data volume mounts INSIDE the
                # container, resolved from the same key the process inside it
                # resolves its own agent-data root from.
                # Read from `agent_data.base_dir` through the shared helper
                # rather than spelled as a literal here, because the container
                # resolves its own agent-data root from that key
                # (workspace.resolve_agent_data_root, anchored on the config's
                # project_root, which in-container is container_project_dir).
                # A literal here silently decouples the two: the volume mounts
                # at one path while memory, sessions and artifacts are written
                # to another, which is a plain directory in the container's
                # writable layer — so every user's data is discarded the next
                # time the container is recreated, with nothing to see at
                # mount time or in any log.
                "container_agent_data_dir": _container_agent_data_dir(
                    root, entry["container_project_dir"]
                ),
                "extra_mounts": entry["extra_mounts"],
                # Optional per-user window/tab title -> OSPREY_WEB_APP_NAME. None
                # (the common case: resolve_personas omits the key unless a
                # non-empty string was set) leaves the template's `{% if
                # svc.display_name %}` guard false, so no env line is emitted and
                # the app falls back to config web.app_name.
                "display_name": entry.get("display_name"),
                # Optional per-user default theme -> OSPREY_WEB_THEME. Same
                # shape as display_name above: absent (the common case) leaves
                # the template's `{% if svc.theme %}` guard false, so no env
                # line is emitted and the app falls back to config web.theme.
                "theme": entry.get("theme"),
                # Optional per-user OIDC identity -> the auth sidecar's
                # OSPREY_AUTH_OIDC_SUBJECT_<SUFFIX>. Same shape again: absent
                # leaves the compose template's `{% if svc.oidc_subject %}`
                # guard false. Without this passthrough an `oidc` deployment
                # would render no subject mapping at all and every IdP identity
                # would be unmappable.
                "oidc_subject": entry.get("oidc_subject"),
                # Whether this entry opted out of the login wall (`login:
                # false` on the roster entry). Read through the shared
                # predicate rather than off the raw key so the template's gate
                # and credential provisioning can never disagree about who has
                # a login. With auth off the template never consults it.
                "login_exempt": not entry_requires_login(entry),
                # The env-var name that subject is emitted under, resolved HERE
                # through env_var_suffix() — the single definition of the
                # username->env-var mapping, shared with credential
                # provisioning, the sidecar's own lookup and lint. The template
                # emits this verbatim rather than re-deriving `upper|replace`,
                # which would be a second copy of the mapping free to drift from
                # the one that keys the password hashes.
                "oidc_subject_env": f"OSPREY_AUTH_OIDC_SUBJECT_{env_var_suffix(entry['name'])}",
                # The env-var NAME this user's operator secret travels in — the
                # credential nginx injects as a request header and the app
                # authenticates every request against. A name, never a value:
                # the compose service references it as
                # `${OSPREY_TERMINAL_SECRET_<SUFFIX>:-}` so the secret is
                # interpolated at container-create time from the deploy .env,
                # and nginx substitutes the same variable into this user's
                # snippet at start. Resolved HERE through the shared
                # terminal_secret_env_var(), for the same reason
                # `oidc_subject_env` is: a template-side `upper|replace` would
                # be a second copy of the mapping, free to drift from the one
                # the secret was minted under.
                "terminal_secret_env": terminal_secret_env_var(entry["name"]),
                # The name of the ONE cookie this user's app reads, resolved
                # HERE through the app's own
                # :func:`~osprey.interfaces.common_middleware.session_cookie_name`
                # so the perimeter and the app cannot disagree about it. nginx
                # forwards this cookie and nothing else on the locations where it
                # injects no operator secret: one cookie jar serves the whole
                # origin, so the browser's raw `Cookie` header names every other
                # terminal (and, with auth on, the sidecar's own session) that
                # browser has unlocked — and these containers run
                # agent-generated code.
                "session_cookie": _session_cookie_name(user_ports["web"]),
                **user_ports,
                # One env line per companion family, derived from the web-server
                # registry (PANEL_ENV_VARS) so a newly registered companion is
                # multi-user-wired without touching this module or the template.
                # The "web" family is not in this list — the template exports
                # its OSPREY_TERMINAL_WEB_PORT/OSPREY_WEB_PORT pair explicitly.
                "panel_env": [
                    {"name": env_var, "port": user_ports[family]}
                    for family, env_var in PANEL_ENV_VARS.items()
                ],
                # Whether this user's container gets the event dispatcher's
                # bearer (see the `dispatcher_personas` arg). A persona-less
                # roster entry — the zero-migration path, where the web image
                # IS the deploy project — is answered from this same config,
                # with no disk read, so the determinism contract holds either
                # way.
                "wants_dispatcher": (
                    entry["persona"] in (dispatcher_personas or set())
                    if entry.get("persona")
                    else config_needs_dispatcher_token(root)
                ),
                # Whether this user's container gets the ARIEL Postgres password
                # (see the `ariel_personas` arg). Persona-less entries are
                # answered from this same config, with no disk read, exactly as
                # above.
                "wants_ariel_db": (
                    entry["persona"] in (ariel_personas or set())
                    if entry.get("persona")
                    else config_needs_ariel_password(root)
                ),
                # Whether this user's container gets the Bluesky queue's launch
                # token (see the `launch_token_personas` arg). Persona-less
                # entries are answered from this same config, with no disk read,
                # exactly as above.
                "wants_launch_token": (
                    entry["persona"] in (launch_token_personas or set())
                    if entry.get("persona")
                    else config_needs_launch_token(root)
                ),
                # Whether this user's container gets the graph store's Neo4j
                # password (see the `graphdb_personas` arg). Persona-less
                # entries are answered from this same config, with no disk read,
                # exactly as above.
                "wants_graphdb": (
                    entry["persona"] in (graphdb_personas or set())
                    if entry.get("persona")
                    else config_needs_graphdb_password(root)
                ),
                # The NAME of the variable this user's archiver connector
                # authenticates with (see the `archiver_password_personas`
                # arg), or None for no grant. Persona-less entries are
                # answered from this same config, with no disk read, exactly
                # as above.
                "archiver_password_env": (
                    (archiver_password_personas or {}).get(entry["persona"])
                    if entry.get("persona")
                    else config_archiver_password_env(root)
                ),
                # Where the deployment's knowledge bundle mounts inside THIS
                # user's container, or None when the user is not entitled or the
                # deployment configures no bundle. One key rather than a
                # boolean + a shared path: personas differ in
                # container_project_dir, so the target is per service and the
                # template must not be able to emit a mount without one.
                # Persona-less entries are answered from this same config with
                # no disk read, exactly as the four grants above.
                "container_bundle_dir": (
                    _container_bundle_dir(root, entry["container_project_dir"])
                    if (
                        entry["persona"] in (facility_bundle_personas or set())
                        if entry.get("persona")
                        else config_needs_facility_bundle(root)
                    )
                    else None
                ),
                # Where the deployment's ARIEL qmd mirror mounts inside THIS
                # user's container, or None when the user is not entitled or
                # the deployment writes no mirror. Same shape and same reason
                # as container_bundle_dir: per service, and the template
                # cannot emit the mount without a target.
                "container_mirror_dir": (
                    _container_mirror_dir(root, entry["container_project_dir"])
                    if (
                        entry["persona"] in (ariel_mirror_personas or set())
                        if entry.get("persona")
                        else config_needs_ariel_mirror(root)
                    )
                    else None
                ),
                # Where this user's refusal audit log is written inside the
                # container, and the host directory that backs it. Every user,
                # unconditionally: the python executor runs in every persona
                # and its audit zone is the one path here that the roster's
                # tier does not gate. The source is per user so two containers
                # writing the same fixed filename cannot clobber each other.
                "container_audit_dir": _container_audit_dir(entry["container_project_dir"]),
                "audit_source": repo_relative_mount_source(user_audit_relpath(entry["name"])),
            }
        )

    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        raise ValueError("modules.web_terminals.nginx_port is required and must be an int")

    image_source = effective_image_source(web_terminals)

    auth_tls_ctx = _auth_tls_context(web_terminals)
    if auth_tls_ctx["tls_enabled"] and not (auth_tls_ctx["tls_cert"] and auth_tls_ctx["tls_key"]):
        raise ValueError(
            "modules.web_terminals.tls.enabled is true but tls.cert/tls.key are not both "
            "set — the gated `listen 443 ssl` seam needs both paths to emit a "
            "coherent ssl_certificate/ssl_certificate_key pair"
        )
    _check_tls_host_cert_dir(auth_tls_ctx)
    if (
        auth_tls_ctx["auth_method"] != "none"
        and not auth_tls_ctx["tls_enabled"]
        and not auth_tls_ctx["auth_allow_insecure_http"]
    ):
        raise ValueError(
            f"modules.web_terminals.auth.method is {auth_tls_ctx['auth_method']!r} but "
            "tls.enabled is false — session cookies would cross the network in "
            "cleartext, so login is refused rather than rendered insecurely. Enable "
            "tls, or set auth.allow_insecure_http: true to accept that risk (only "
            "sensible behind a TLS-terminating proxy or on an isolated network)"
        )
    _check_roster_charset(services, auth_tls_ctx["auth_method"])
    _check_roster_env_var_collisions(services, auth_tls_ctx["auth_method"])

    # Built (and refused) ahead of the Jinja pass so a roster user with no
    # operator secret stops the render before any artifact exists, rather than
    # after three of them have been produced.
    secret_templates = (
        _terminal_secret_artifacts(
            services, terminal_secrets, auth_method=auth_tls_ctx["auth_method"]
        )
        if terminal_secrets is not None
        else {}
    )

    tls_enabled = auth_tls_ctx["tls_enabled"]
    # The one origin every absolute URL in this deployment is built from (see
    # _external_origin). Derived only when something actually needs it: a
    # roster-less config with no auth has no absolute URL to build, and must
    # keep rendering without a deploy.fqdn.
    external_origin = (
        _external_origin(root, nginx_port, tls_enabled=tls_enabled)
        if services or auth_tls_ctx["auth_method"] != "none"
        else ""
    )
    landing_url = _landing_url(root, nginx_port, tls_enabled=tls_enabled) if services else ""

    # Every service is told the deployment's external origin, because the app
    # inside each container checks a mutating request's `Origin` against it —
    # the CSRF half of the carrier. One deployment-wide value rather than a
    # per-service one: there is exactly one origin (see _external_origin), and a
    # service that believed in a different one would reject the very links the
    # landing page hands out. It is derived here rather than deferred to nginx's
    # per-request `$host` because a container reads its environment once, at
    # start — and because `$host` carries no port, so a deployment behind a
    # non-443 plain-HTTP nginx would check against an origin no browser sends.
    for service in services:
        service["external_origin"] = external_origin

    # The bundle's host path AS CONFIGURED. Read here rather than inside the
    # per-service loop: every entitled user mounts the same one directory, and
    # only the target differs per persona.
    raw_bundle_path = as_dict(root.get("facility_knowledge")).get("bundle_path")
    bundle_path = raw_bundle_path.strip() if isinstance(raw_bundle_path, str) else ""
    mirror_path = configured_ariel_mirror_path(root)

    compose_ctx = {
        "facility_prefix": facility_prefix,
        "registry_url": registry.get("url") or "",
        "image_source": image_source,
        "services": services,
        "nginx_port": nginx_port,
        "landing_url": landing_url,
        "facility_timezone": facility.get("timezone") or "UTC",
        "bind_host": _LOOPBACK_BIND_HOST,
        # Read defensively (as everywhere else in this module); the template
        # carries the same default, so an absent/blank value still renders the
        # public tag. Facilities whose images all come from a private registry
        # override this so the nginx image is pullable too.
        "nginx_image": web_terminals.get("nginx_image") or _DEFAULT_NGINX_IMAGE,
        # The auth sidecar's service (image, listen port, session lifetime, the
        # env-var names its OIDC credentials arrive under) is rendered from the
        # same parsed stanza the nginx seam reads, so the two ends of the
        # auth_request contract can't drift apart.
        "external_origin": external_origin,
        # Empty string (not None) when no digest was supplied, so the template
        # can gate the label block on plain truthiness.
        "auth_env_digest": auth_env_digest or "",
        "auth_env_digest_label": AUTH_ENV_DIGEST_LABEL,
        # Which CHECKOUT this stack belongs to, baked into every container and
        # volume label the template emits. Derived through the same helper the
        # services stack renders from, so one deployment cannot end up with two
        # identities depending on which of its two compose files is read.
        "repo_id": repo_identity(resolve_repo_root(config)),
        # The ONE host directory every entitled user's bundle mount reads from —
        # the deployment's bundle, not a per-user copy. That is what makes the
        # corpus shared: a concept alice drafts is the same file bob's container
        # and the qmd sidecar see. Empty string (not None) when the deployment
        # configures no bundle, so the template gates on plain truthiness.
        #
        # Spelled through the shared bind-source rule, with no repo root to
        # resolve against because this function reads no filesystem: a relative
        # `bundle_path` becomes `./<path>` (resolved by compose against the
        # pinned project directory, which IS the repo root) and an absolute one
        # is emitted verbatim.
        "facility_bundle_source": (repo_relative_mount_source(bundle_path) if bundle_path else ""),
        # The group every entitled container joins, so the shared mode on that
        # directory actually reaches it. Emitted as a STRING: compose's
        # `group_add` accepts group names as well as numeric ids, and quoting
        # keeps a numeric one from being read as anything else.
        "facility_bundle_gid": str(facility_bundle_gid) if facility_bundle_gid is not None else "",
        # The ONE host directory every entitled user's mirror mount writes into
        # — the deployment's mirror, the same one the qmd sidecar indexes and
        # the host exporter fills — spelled through the same bind-source rule
        # as the bundle above and for the same reasons. Empty string when the
        # deployment writes no mirror.
        "ariel_mirror_source": (
            repo_relative_mount_source(mirror_path) if mirror_path is not None else ""
        ),
        "ariel_mirror_gid": str(ariel_mirror_gid) if ariel_mirror_gid is not None else "",
        # The group of the deployment's audit zone, joined by every per-user
        # service; its per-user subdirectories are provisioned by the deploy
        # with the same inherited group.
        "audit_gid": str(audit_gid) if audit_gid is not None else "",
        **auth_tls_ctx,
    }

    nginx_ctx = {
        "nginx_port": nginx_port,
        "services": services,
        "bind_host": _LOOPBACK_BIND_HOST,
        "external_origin": external_origin,
        **auth_tls_ctx,
    }
    landing_cfg = as_dict(web_terminals.get("landing"))
    landing_ctx = {
        "facility_name": resolve_facility_name(root, ""),
        "groups": _build_groups(landing_cfg, resolved_users),
        "theme_blocks": _landing_theme_blocks(root),
        "notices": _build_notices(landing_cfg, root),
        "footer": _landing_footer(landing_cfg),
    }

    template_dir = files("osprey").joinpath(_TEMPLATE_PACKAGE_PATH)
    with as_file(template_dir) as template_path:
        # Compose + nginx emit YAML/conf, not HTML — autoescape=False, matching
        # compose_generator.py's own Jinja2 convention. Landing emits HTML and is
        # rendered with autoescape=True; landing.html.j2 also `|e`-escapes every
        # interpolation itself as defense-in-depth.
        conf_env = Environment(loader=FileSystemLoader(str(template_path)), autoescape=False)
        html_env = Environment(loader=FileSystemLoader(str(template_path)), autoescape=True)

        rendered_compose = conf_env.get_template(_COMPOSE_TEMPLATE).render(**compose_ctx)
        rendered_nginx = conf_env.get_template(_NGINX_TEMPLATE).render(**nginx_ctx)
        rendered_landing = html_env.get_template(_LANDING_TEMPLATE).render(**landing_ctx)

    return {
        _COMPOSE_OUTPUT: rendered_compose,
        _NGINX_OUTPUT: rendered_nginx,
        _LANDING_OUTPUT: rendered_landing,
        # Additive, and only when the caller holds the secrets: a render with
        # none in hand (the scaffold path) returns exactly the three artifacts
        # it always did, byte for byte.
        **secret_templates,
    }


#: The design-system custom properties the landing page's inline stylesheet
#: uses. Read by name out of the resolved theme rather than copied as a whole
#: emitted block: the landing page is a standalone file with no stylesheet to
#: fall back on, so it wants only what it uses — and a rename must fail the
#: render loudly (see ``theme_css_variables``) instead of leaving a deployed
#: page quietly off-palette.
_LANDING_THEME_VARIABLES = (
    "--bg-primary",
    "--bg-panel",
    "--bg-elevated",
    "--text-primary",
    "--text-secondary",
    "--text-muted",
    "--color-accent",
    "--border-default",
    "--border-accent",
    # The downward card shadow, not --shadow-panel: that one is a drawer edge
    # shadow (`-4px 0 ...`), which reads as a left-hand glow on a card.
    "--shadow-dropdown",
)

#: What a baked custom-property value is allowed to look like. Values come from
#: the framework's own token tree rather than from operator config, so this is a
#: belt-and-braces guard on a string that lands inside a ``<style>`` element,
#: where HTML escaping would not save us from a ``</style>`` sequence.
_SAFE_CSS_VALUE_RE = re.compile(r"^[#A-Za-z0-9 ,.()%/_-]+$")


def _landing_theme_blocks(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the landing page's inline theme blocks from the deployment's ``web.theme``.

    The landing page is served by nginx as a flat file with no application
    behind it — it cannot link ``tokens.css``, and it has no theme picker and no
    ``localStorage``. So its palette is resolved here, at render time, and baked
    into the page:

    - ``web.theme`` naming a **family** (``"desy"``) yields TWO blocks: that
      family's dark values under ``:root``, and its light values under a
      ``prefers-color-scheme: light`` media query. Light/dark then follows each
      viewer's OS, which is the same thing the terminals behind this page do
      when configured with a bare family.
    - ``web.theme`` naming a **concrete id** (``"desy-light"``) yields ONE
      block and no media query — the deployment pinned a mode, and the landing
      page honors that pin exactly as the terminals do.

    The landing page deliberately uses the deployment-level theme and never a
    per-user one: it is shown before anyone has identified themselves.

    Args:
        root: The parsed facility config.

    Returns:
        Block dicts for ``landing.html.j2``: ``{"media": str|None,
        "color_scheme": str, "variables": [(name, value), ...]}``.

    Raises:
        ValueError: If a resolved value fails :data:`_SAFE_CSS_VALUE_RE`.
        MissingThemeVariableError: If the resolved theme does not define one of
            :data:`_LANDING_THEME_VARIABLES`.
    """
    from osprey.interfaces.design_system.theme_config import (
        family_of,
        load_theme_registry,
        resolve_pinned_mode,
        resolve_theme_id,
        theme_css_variables,
    )

    configured = str(as_dict(root.get("web")).get("theme") or "main")
    entries, defaults = load_theme_registry()
    resolved_id = resolve_theme_id(configured, entries, defaults, config_key="web.theme")
    pinned_mode = resolve_pinned_mode(configured, entries)

    #: (media query or None, color-scheme, theme id) per block to emit.
    wanted: list[tuple[str | None, str, str]]
    if pinned_mode is not None:
        wanted = [(None, pinned_mode, resolved_id)]
    else:
        family = family_of(resolved_id, entries) or "main"
        family_modes = defaults[family]
        wanted = [
            (None, "dark", family_modes["dark"]),
            ("(prefers-color-scheme: light)", "light", family_modes["light"]),
        ]

    blocks: list[dict[str, Any]] = []
    for media, color_scheme, theme_id in wanted:
        variables = theme_css_variables(theme_id, _LANDING_THEME_VARIABLES)
        for name, value in variables.items():
            if not _SAFE_CSS_VALUE_RE.match(value):
                raise ValueError(
                    f"theme {theme_id!r} value for {name!r} is not safe to inline "
                    f"into the landing page's <style>: {value!r}"
                )
        blocks.append(
            {
                "media": media,
                "color_scheme": color_scheme,
                "variables": list(variables.items()),
            }
        )
    return blocks


#: ``modules.web_terminals.external_origin``, as a whole-string match: a scheme,
#: a host, an optional port, and nothing else. No path (not even ``/``), no
#: query, no fragment, no credentials, no trailing slash — this value is compared
#: character-for-character against the ``Origin`` header a browser sends, and a
#: browser never puts any of those in one. Only ``http`` and ``https`` are
#: accepted: those are the two schemes this perimeter can serve, and a typo like
#: ``htps://`` would otherwise render an origin nothing can ever match.
_EXTERNAL_ORIGIN_RE = re.compile(r"https?://[A-Za-z0-9._~%-]+(?::\d+)?\Z")


def _configured_external_origin(root: dict[str, Any]) -> str:
    """``modules.web_terminals.external_origin`` as configured, validated.

    Returns the empty string when the key is absent or blank — the derived
    origin then applies (:func:`_external_origin`).

    Raises:
        ValueError: If the value is not a string, or is not
            ``scheme://host[:port]`` and nothing else. Refused HERE rather than
            trusted, because nothing downstream would report it: the value is
            baked into every container as ``OSPREY_TERMINAL_EXTERNAL_ORIGIN``
            and compared against the browser's ``Origin`` as a whole string, so
            a trailing slash or a stray path segment produces a deployment whose
            pages all load and whose every write answers 403.
    """
    web_terminals = as_dict(as_dict(root.get("modules")).get("web_terminals"))
    if "external_origin" not in web_terminals:
        return ""
    value = web_terminals.get("external_origin")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"modules.web_terminals.external_origin {value!r} is not a string; it must "
            "be the origin a browser reaches this deployment on, e.g. "
            "'https://terminals.example.org'"
        )
    origin = value.strip()
    if not origin:
        return ""
    if not _EXTERNAL_ORIGIN_RE.fullmatch(origin):
        raise ValueError(
            f"modules.web_terminals.external_origin {origin!r} is not an origin. It must "
            "be scheme://host[:port] with nothing after the host — no path, no trailing "
            "slash, no query (e.g. 'https://terminals.example.org', "
            "'http://terminals.example.org:8443'). Each terminal compares it against the "
            "browser's Origin header as a whole string, so anything else renders a "
            "deployment whose pages load and whose every write is refused"
        )
    return origin


def _external_origin(root: dict[str, Any], nginx_port: int, *, tls_enabled: bool) -> str:
    """Build the one origin every absolute URL this deployment emits is derived from.

    Three consumers need an absolute URL that a browser will actually resolve:
    the landing link baked into each container (:func:`_landing_url`), the auth
    sidecar's OIDC ``redirect_uri``, and the ``OSPREY_TERMINAL_EXTERNAL_ORIGIN``
    each per-user app checks a mutating request's ``Origin`` against. All three
    must agree exactly — an IdP rejects a ``redirect_uri`` that isn't
    character-for-character the registered one, a landing link on a different
    origin would drop the session cookie, and an ``Origin`` that does not match
    is refused — so they come from here rather than being assembled three times.

    ``modules.web_terminals.external_origin`` WINS when set, and is returned
    verbatim. It exists because the derivation below describes only the topology
    where a browser talks to THIS nginx directly. A facility load balancer or
    ingress proxy terminating TLS in front of it (the shape
    ``auth.allow_insecure_http`` is documented for) is a supported deployment in
    which the browser's origin is the front terminator's — a different host, and
    usually a different scheme — and nothing derivable from this config can name
    it. Without the override every write from such a deployment is refused while
    the pages themselves load, which reads as a broken app rather than as a
    configuration gap.

    Otherwise the origin is derived. Scheme and port follow ``tls.enabled``:
    with TLS on, the 443 server is the sole content server (the plain listener
    only redirects to it), and 443 is left implicit, which is both the canonical
    origin serialization and the form an IdP client registration is normally
    written in. With TLS off the origin is plain HTTP on the published
    ``nginx_port``. A default port left explicit here is not a mismatch: both
    ends of the ``Origin`` check normalize it away
    (:func:`osprey.interfaces.common_middleware._normalize_origin`).

    The derived host comes from ``deploy.fqdn``: the schema documents that field
    as reachable from developers' laptops (used in client-mode profiles),
    whereas ``deploy.host`` is only guaranteed SSH-resolvable (may be a bare
    `~/.ssh/config` alias, not a browser-reachable hostname).

    Args:
        root: The parsed facility config (``modules.web_terminals.external_origin``
            and ``deploy.fqdn`` are read).
        nginx_port: The published plain-HTTP port, used only when TLS is off and
            no origin is configured.
        tls_enabled: The parsed ``tls.enabled`` (see :func:`_auth_tls_context`).

    Raises:
        ValueError: If ``modules.web_terminals.external_origin`` is set to
            something that is not an origin (see
            :func:`_configured_external_origin`), or if it is unset and
            ``deploy.fqdn`` is missing or blank.
    """
    configured = _configured_external_origin(root)
    if configured:
        return configured
    deploy = as_dict(root.get("deploy"))
    host = str(deploy.get("fqdn") or "").strip()
    if not host:
        raise ValueError(
            "deploy.fqdn is required to render modules.web_terminals landing_url "
            "(OSPREY_TERMINAL_LANDING_URL) when at least one user is configured "
            "or authentication is enabled. Set it, or — when a load balancer in "
            "front of this nginx is what browsers actually reach — set "
            "modules.web_terminals.external_origin to that address"
        )
    return f"https://{host}" if tls_enabled else f"http://{host}:{nginx_port}"


def deployment_external_origin(config: Any) -> str:
    """The origin a browser reaches this deployment's web terminals on.

    :func:`_external_origin` as a question a caller holding nothing but the
    rendered config can ask. Everything an operator is handed to open — the
    landing link, the auth sidecar's OIDC ``redirect_uri``, and the per-user
    login URL :func:`terminal_login_url` builds — comes from this one
    derivation, so a link printed by one verb cannot land on a different origin
    than the one the containers check a mutating request's ``Origin`` against.

    Args:
        config: The rendered deployment config (``build/config.yml`` as loaded).

    Returns:
        ``modules.web_terminals.external_origin`` verbatim when it is set;
        otherwise ``https://<fqdn>`` with TLS on and
        ``http://<fqdn>:<nginx_port>`` without.

    Raises:
        ValueError: If ``modules.web_terminals.nginx_port`` is missing or is not
            an int; if ``modules.web_terminals.external_origin`` is set to
            something that is not an origin; or if it is unset and
            ``deploy.fqdn`` is missing or blank — the values an origin cannot be
            assembled without.
    """
    root = as_dict(config)
    web_terminals = as_dict(as_dict(root.get("modules")).get("web_terminals"))
    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        raise ValueError("modules.web_terminals.nginx_port is required and must be an int")
    tls_enabled = bool(_auth_tls_context(web_terminals)["tls_enabled"])
    return _external_origin(root, nginx_port, tls_enabled=tls_enabled)


def terminal_login_url(config: Any, username: str, secret: str) -> str:
    """One user's ``?token=`` login URL — the way in when nginx authenticates nobody.

    With ``auth.method: none`` (the default) nginx injects no operator secret and
    runs no login flow, so the per-user app's own gate is the only one standing:
    a browser arrives with no cookie, and the single way to get one is a GET to
    that user's terminal carrying the operator secret as ``?token=``. The app
    exchanges it for the session cookie and redirects to the token-stripped URL.
    Nothing else in a multi-user deployment hands that URL out, which is why this
    exists as a verb (``osprey users login-url``) rather than as a line printed
    once during a deploy that may have scrolled away.

    The secret is percent-encoded into the query. It is generated by
    ``token_urlsafe`` and so needs no escaping today; encoding it anyway means an
    operator who pinned a secret of their own by hand in the deploy ``.env`` gets
    a URL that still means what it says.

    Args:
        config: The rendered deployment config, for the origin
            (:func:`deployment_external_origin`).
        username: The roster user this URL is for. Rendered into the path as the
            nginx location key, exactly as the perimeter spells it.
        secret: That user's operator secret, read from the deploy ``.env``. Never
            logged by anything here — the caller owns where the URL goes.

    Returns:
        ``<origin>/u/<user>/?token=<secret>``.

    Raises:
        ValueError: Whatever :func:`deployment_external_origin` raises.
    """
    origin = deployment_external_origin(config)
    return f"{origin}/u/{username}/?token={quote(secret, safe='')}"


def _landing_url(root: dict[str, Any], nginx_port: int, *, tls_enabled: bool = False) -> str:
    """The absolute origin baked into every service's ``OSPREY_TERMINAL_LANDING_URL``.

    Per-user containers only get this value once, at container start (env vars, not
    request time) — unlike nginx.conf.j2's per-request ``$host`` redirect target,
    resolving it can't be deferred to the browser. It is the deployment's external
    origin verbatim (:func:`_external_origin`), which is what keeps a "back to
    landing" link and an OIDC ``redirect_uri`` on the same origin by construction.
    """
    return _external_origin(root, nginx_port, tls_enabled=tls_enabled)


def _user_card(resolved_user: dict[str, Any]) -> dict[str, Any]:
    """Build one auto-populated ``users``-group landing card from a resolved roster entry.

    The base shape is unchanged (``{label, url}``); a ``sublabel`` key is added
    only when the entry resolved to a persona, so a no-persona roster keeps
    producing exactly the same two-key items landing.html.j2 rendered before.

    One case drops the badge even though a persona is in effect: when the
    persona name and the roster name are the same word. The badge exists to say
    which tier a login belongs to, and a card reading ``ariel`` above a pill
    reading ``ARIEL`` says nothing the label did not — the common shape for a
    single-tenant service persona, where the roster entry and the persona are
    named after the same thing. Compared case-insensitively because the badge is
    rendered uppercase, so ``Ariel``/``ariel`` would look duplicated too. Two
    different personas sharing a section still each show their own badge, which
    is the case the badge is actually for.

    Args:
        resolved_user: One :func:`osprey.deployment.web_terminals.personas.resolve_personas`
            entry (``name`` and ``persona`` are read).

    Returns:
        ``{"label", "url"}`` for a persona-less user, plus ``"sublabel"`` (the
        persona name) when ``persona`` is a non-empty string that differs from
        the user's own name.
    """
    name = resolved_user["name"]
    card: dict[str, Any] = {"label": name, "url": f"/u/{name}/"}
    persona = resolved_user.get("persona")
    if isinstance(persona, str) and persona and persona.casefold() != name.casefold():
        card["sublabel"] = persona
    return card


def _build_groups(
    landing_cfg: dict[str, Any], resolved_users: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Transform config ``landing.groups`` into the template's ``groups`` shape:
    plain dicts with a ``label`` and an ``items`` key, since landing.html.j2
    uses bracket subscript (``group["items"]``) throughout.

    ``{type: "users"}`` auto-populates one card per configured user, using the
    relative ``/u/<user>/`` path that nginx.conf.j2 (bind-nginx-reverse-proxy)
    reverse-proxies to that user's loopback upstream — so, unlike ``landing_url``,
    no deploy-host needs baking into the landing cards themselves. When a user
    resolves to a persona (:func:`resolve_personas` returns a non-``None``
    ``persona``), that card also carries an optional ``sublabel`` holding the
    persona name, shown as a secondary badge on the card; users with no persona
    in effect (every bare-string roster) omit the key entirely, so
    landing.html.j2's ``{% if item["sublabel"] %}`` guard renders them without
    one. ``{type: "links", label, links}`` passes ``links`` straight through as
    ``items`` (link cards never carry a ``sublabel``). Unrecognized/malformed
    group entries are dropped rather than raising: lint is the authoritative
    gate on schema well-formedness, this is just the render-time adapter.

    **One ``users`` entry can expand to more than one section.** A roster is not
    always a list of people: a deployment may also run a standalone service
    behind its own login — an ARIEL logbook terminal beside the operators, say.
    Those belong under their own heading rather than mixed in with the staff, so
    a persona may declare ``landing_group: <heading>`` in the catalog
    (:func:`resolve_personas` carries it onto each of its users). Users carrying
    one are lifted out of the default section into a section of that name,
    emitted after it in first-appearance roster order, and marked
    ``variant: "tray"`` — the cue landing.html.j2 renders as an accent-edged
    panel, so the page reads as people on top, services below. Everything else
    about those users is untouched; this is presentation only.

    ``{type: "users"}`` also takes an optional ``label`` overriding the default
    ``"Terminals"`` heading on the section that keeps the ungrouped users, so a
    deployment that splits its roster can name both halves. Sections other than
    the default never carry ``variant``, so a config that declares no
    ``landing_group`` anywhere renders byte-identically to before.

    Args:
        landing_cfg: The already-dict-coerced ``modules.web_terminals.landing``
            section (only ``groups`` is read).
        resolved_users: :func:`osprey.deployment.web_terminals.personas.resolve_personas`
            output, in roster order — each entry's ``name`` becomes the card label
            and ``/u/<name>/`` url, its ``persona`` (when not ``None``) the
            optional ``sublabel``, and its ``landing_group`` (when present) the
            section it is lifted into.
    """
    groups_raw = landing_cfg.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        groups_raw = [{"type": "users"}]  # schema default when `landing.groups` is omitted

    groups: list[dict[str, Any]] = []
    for entry in groups_raw:
        entry = as_dict(entry)
        group_type = entry.get("type")
        if group_type == "users":
            groups.extend(_user_groups(resolved_users, entry.get("label")))
        elif group_type == "links":
            links = entry.get("links")
            items = [as_dict(link) for link in links] if isinstance(links, list) else []
            groups.append({"label": entry.get("label") or "", "items": items})
    return groups


def _user_groups(resolved_users: list[dict[str, Any]], default_label: Any) -> list[dict[str, Any]]:
    """Split one ``{type: "users"}`` entry into its default section plus a tray
    section per distinct persona ``landing_group``.

    The default section is emitted FIRST and always — even empty, so
    landing.html.j2's own empty-items suppression stays the single rule that
    decides whether a heading appears, rather than this function second-guessing
    it. Tray sections follow in the order their group name is first seen walking
    the roster, so the page order is the roster's order and an operator can
    predict it from the config alone.

    Args:
        resolved_users: :func:`resolve_personas` output, in roster order.
        default_label: The ``{type: "users"}`` entry's own ``label``, used as the
            heading for the ungrouped users. Anything that is not a non-empty
            string falls back to ``"Terminals"``.

    Returns:
        ``[{label, items}, {label, items, variant: "tray"}, ...]``.
    """
    label = default_label if isinstance(default_label, str) and default_label else "Terminals"
    default_items: list[dict[str, Any]] = []
    # dict, not defaultdict+sorted: insertion order IS first-appearance order,
    # which is the ordering guarantee this function documents.
    trays: dict[str, list[dict[str, Any]]] = {}
    for user in resolved_users:
        group = user.get("landing_group")
        card = _user_card(user)
        if isinstance(group, str) and group:
            trays.setdefault(group, []).append(card)
        else:
            default_items.append(card)

    groups: list[dict[str, Any]] = [{"label": label, "items": default_items}]
    groups.extend(
        {"label": name, "items": items, "variant": "tray"} for name, items in trays.items()
    )
    return groups


def _auth_tls_context(web_terminals: dict[str, Any]) -> dict[str, Any]:
    """Read the ``web_terminals.auth``/``web_terminals.tls`` stanzas into the context
    keys the nginx seam and the auth sidecar's compose service consume.

    This is the single place ``modules.web_terminals.auth`` is parsed: the nginx
    template, the compose overlay and the lint rules all read the keys returned
    here rather than the raw config, so there is one definition of what an
    ``auth`` stanza means. **This function renders nothing** — it derives values;
    the templates turn ``tls_enabled``/``auth_method`` into actual `listen 443
    ssl` / `auth_request` directives.

    Both stanzas remain entirely optional and default to the inert posture: no
    authentication, no TLS. Every value is read defensively (wrong-typed entries
    fall back to their default, as everywhere else in this module — lint is the
    authoritative gate on schema well-formedness) with one exception: an
    ``auth.method`` string naming a method that does not exist raises, because
    the resulting deployment would emit an auth seam nothing can answer.

    Args:
        web_terminals: The already-unwrapped ``modules.web_terminals`` dict (as
            passed to :func:`render_web_terminals`'s Jinja contexts).

    Returns:
        A dict with the auth keys ``auth_method`` (one of
        :data:`SUPPORTED_AUTH_METHODS`, defaults to ``"none"``), ``auth_port``
        (int, the sidecar's listen port), ``auth_session_lifetime`` (int
        seconds), ``auth_allow_insecure_http`` (bool), ``auth_image`` (str or
        ``None`` — required only in registry image-source mode),
        ``auth_oidc_issuer`` (str or ``None``),
        ``auth_oidc_client_id_env``/``auth_oidc_client_secret_env`` (the env-var
        *names* the sidecar reads its OIDC client credentials from — never the
        credentials themselves) and ``auth_oidc_claim`` (str or ``None``, the
        ID-token claim carrying the identity to map onto a roster user); plus
        the TLS keys ``tls_enabled`` (bool) and
        ``tls_cert``/``tls_key`` (str path or ``None``, read only when
        ``tls_enabled``).

    Raises:
        ValueError: If ``auth.method`` is a non-empty string outside
            :data:`SUPPORTED_AUTH_METHODS`.
    """
    auth = as_dict(web_terminals.get("auth"))
    tls = as_dict(web_terminals.get("tls"))
    oidc = as_dict(auth.get("oidc"))

    method = auth.get("method")
    auth_method = method if isinstance(method, str) and method else "none"
    if auth_method not in SUPPORTED_AUTH_METHODS:
        raise ValueError(
            f"modules.web_terminals.auth.method {auth_method!r} is not a supported "
            f"authentication method; expected one of {', '.join(SUPPORTED_AUTH_METHODS)}"
        )

    return {
        "auth_method": auth_method,
        "auth_port": _positive_int(auth.get("port"), _DEFAULT_AUTH_PORT),
        "auth_session_lifetime": _positive_int(
            auth.get("session_lifetime"), _DEFAULT_SESSION_LIFETIME
        ),
        "auth_allow_insecure_http": bool(auth.get("allow_insecure_http", False)),
        "auth_image": auth.get("image") or None,
        "auth_oidc_issuer": oidc.get("issuer") or None,
        "auth_oidc_client_id_env": _non_empty_str(
            oidc.get("client_id_env"), _DEFAULT_OIDC_CLIENT_ID_ENV
        ),
        "auth_oidc_client_secret_env": _non_empty_str(
            oidc.get("client_secret_env"), _DEFAULT_OIDC_CLIENT_SECRET_ENV
        ),
        # Which ID-token claim carries the identity that maps onto a roster
        # user. Left as None when unset (rather than defaulted here) so the
        # compose service emits no OSPREY_AUTH_OIDC_CLAIM at all and the
        # sidecar's own documented default applies — one default, in one place.
        "auth_oidc_claim": _non_empty_str(oidc.get("claim"), "") or None,
        "tls_enabled": bool(tls.get("enabled", False)),
        "tls_cert": tls.get("cert"),
        "tls_key": tls.get("key"),
        # The HOST directory holding the certificate and key. Optional, and the
        # only key in this stanza that names a path on the deploy host rather
        # than one inside the nginx container: setting it makes the overlay
        # bind-mount that directory read-only at the container-side directory
        # `tls.cert` sits in, which is the step that otherwise has to be done by
        # hand. Left unset, nothing is mounted and the operator supplies the
        # mount themselves — the escape hatch for a facility whose certificates
        # arrive by some route a directory bind cannot express.
        "tls_host_cert_dir": _non_empty_str(tls.get("host_cert_dir"), "") or None,
        # Where that directory lands inside the container: the parent of
        # `tls.cert`. Derived rather than configured so the mount and the
        # `ssl_certificate` directive cannot name different places.
        "tls_mount_target": _tls_mount_target(tls),
    }


def _check_tls_host_cert_dir(ctx: dict[str, Any]) -> None:
    """Refuse a ``tls.host_cert_dir`` that cannot deliver both files.

    One read-only bind mount covers the certificate *and* the key only when the
    two sit in the same directory inside the container, because the mount lands
    at the parent of ``tls.cert``. A key elsewhere would leave nginx reading a
    path nothing supplies — the exact silent failure this key exists to remove —
    so it is refused at render time rather than at nginx start.

    A relative ``host_cert_dir`` is refused for the same reason: compose
    resolves it against the project directory, so a deployment moved to another
    machine would mount a different (or missing) directory without saying so.

    Nothing is checked when the key is unset: that is the supported posture in
    which the operator supplies the mount themselves.
    """
    host_dir = ctx["tls_host_cert_dir"]
    if not host_dir:
        return
    if not ctx["tls_enabled"]:
        raise ValueError(
            "modules.web_terminals.tls.host_cert_dir is set but tls.enabled is false — "
            "the certificate directory would be mounted into an nginx that serves no "
            "HTTPS. Enable tls, or drop host_cert_dir"
        )
    if not posixpath.isabs(host_dir):
        raise ValueError(
            f"modules.web_terminals.tls.host_cert_dir {host_dir!r} must be an absolute "
            "path on the deploy host: compose resolves a relative bind source against "
            "the project directory, so the same config would mount a different "
            "directory elsewhere"
        )
    cert_dir = ctx["tls_mount_target"]
    key_dir = posixpath.dirname(_non_empty_str(ctx["tls_key"], "")) or None
    if cert_dir != key_dir:
        raise ValueError(
            "modules.web_terminals.tls.cert and tls.key must sit in the same directory "
            f"when host_cert_dir is set (cert is in {cert_dir!r}, key in {key_dir!r}): "
            "host_cert_dir is mounted at the certificate's directory, so a key outside "
            "it would not be present in the container"
        )


def _tls_mount_target(tls: dict[str, Any]) -> str | None:
    """The in-container directory ``tls.host_cert_dir`` is mounted at.

    The parent of ``tls.cert``, so a mounted directory always lands exactly
    where nginx's ``ssl_certificate`` looks. Returns ``None`` when there is
    nothing to mount or no certificate path to derive it from; the caller
    validates that ``tls.key`` shares the directory, which is what makes a
    single mount sufficient for both files.
    """
    cert = _non_empty_str(tls.get("cert"), "")
    if not cert:
        return None
    return posixpath.dirname(cert) or None


def _positive_int(value: Any, default: int) -> int:
    """A config value read as a positive int, falling back to ``default``.

    ``bool`` is excluded explicitly: it passes ``isinstance(..., int)``, and
    ``auth.port: true`` becoming port 1 would be a baffling deployment.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _non_empty_str(value: Any, default: str) -> str:
    """A config value read as a non-empty string, falling back to ``default``."""
    return value if isinstance(value, str) and value.strip() else default


def _check_roster_charset(services: list[dict[str, Any]], auth_method: str) -> None:
    """Fail-closed render gate: with authentication on, every roster name must
    match the username charset.

    The charset is enforced in two other places, and a deployment can miss both:
    lint is skippable (``--no-lint``), and ``auth_credentials`` only runs on the
    password-mode deploy path. Nothing on the render path itself checked, so a
    name carrying ``&`` or ``?`` rendered an nginx auth location whose verify
    query held *two* user parameters
    (``/_osprey_auth/bob&user=alice`` -> ``...verify?user=bob&user=alice``) —
    the exact ambiguity per-user internal locations exist to prevent. Since the
    username is the authorization identity, an unusable one must stop the render
    rather than produce a seam whose meaning depends on how the sidecar's query
    parser breaks a tie.

    Scoped to authentication being on, deliberately: with ``auth.method: none``
    the username is a routing label, not an identity, and raising here would
    break an otherwise valid render. Lint still reports it in that case.

    Args:
        services: The resolved per-user service entries (each with a ``user``).
        auth_method: The parsed ``auth.method`` (see :func:`_auth_tls_context`).

    Raises:
        ValueError: If authentication is on and any roster name is outside the
            charset. The message names every offender.
    """
    if auth_method == "none":
        return
    # fullmatch, not match: `$` also matches *before* a trailing newline, so
    # `match` would accept "alice\n" — a name that then renders into an nginx
    # location key mid-directive.
    offenders = [
        service["user"]
        for service in services
        if not USERNAME_CHARSET_RE.fullmatch(service["user"])
    ]
    if offenders:
        raise ValueError(
            f"modules.web_terminals.users {', '.join(repr(name) for name in offenders)} "
            f"must match {USERNAME_CHARSET_RE.pattern!r} because "
            f"modules.web_terminals.auth.method is {auth_method!r}: an authenticated "
            "deployment makes the username the authorization identity, and it is "
            "rendered literally into an nginx location key and into the auth "
            "subrequest's ?user= value, where a name containing '&' or '?' would "
            "silently authorize a different user"
        )


def _check_roster_env_var_collisions(services: list[dict[str, Any]], auth_method: str) -> None:
    """Fail-closed render gate: with authentication on, no two roster names may
    share one per-user env-var suffix.

    :func:`~osprey.deployment.web_terminals.personas.env_var_suffix` is total and
    lossy — ``alice-b`` and ``alice_b`` both key ``..._ALICE_B`` — so a colliding
    pair would share a single ``OSPREY_AUTH_PW_HASH_ALICE_B``: one user's
    password would open the other's terminal, which is the precise isolation
    this feature exists to establish.

    Caught here rather than only downstream because the downstream signals are
    both poor. Credential provisioning raises on the password-mode deploy path
    only, and in ``oidc`` mode the collision surfaces (if at all) as a
    whole-deployment 503 from the sidecar — an outage whose message says nothing
    about which two roster names caused it. Render time knows both names.

    Args:
        services: The resolved per-user service entries (each with a ``user``).
        auth_method: The parsed ``auth.method`` (see :func:`_auth_tls_context`).

    Raises:
        ValueError: If authentication is on and two or more names collide. The
            message names each colliding group and the suffix they share.
    """
    if auth_method == "none":
        return
    collisions = env_var_suffix_collisions([service["user"] for service in services])
    if collisions:
        detail = "; ".join(
            f"{', '.join(repr(name) for name in names)} all key {suffix!r}"
            for suffix, names in collisions.items()
        )
        raise ValueError(
            f"modules.web_terminals.users collide on their per-user env-var suffix "
            f"({detail}) and modules.web_terminals.auth.method is {auth_method!r}: "
            "colliding names would share one OSPREY_AUTH_PW_HASH_* entry, so one "
            "user's password would open the other's terminal. Rename one of them"
        )


def _check_mcp_topology(web_terminals: dict[str, Any]) -> None:
    """Fail closed on any ``modules.web_terminals.mcp.topology`` value other than
    the one wired topology, ``per_container_stdio``.

    Only two of the framework's eight MCP servers (``channel-finder`` and
    ``facility-knowledge``) are safely shareable across a shared HTTP tier
    without per-user-state corruption — not enough to justify building and
    securing a whole shared tier. ``shared_http`` is therefore a *recognized
    but rejected* schema value: it lints as an ERROR and raises here at render
    time. See ``references/modules/web-terminals.md`` for the rationale.

    This check is scoped to the shared **framework**-MCP tier only. It has
    nothing to do with, and never rejects, a facility's own
    ``claude_code.servers`` custom ``url``/HTTP entries — those are a
    separate, already-supported path (resolved by
    :func:`osprey.registry.mcp.resolve_servers` into each project's own
    ``.mcp.json``) that this module never reads or touches.

    Args:
        web_terminals: The already-unwrapped ``modules.web_terminals`` dict.

    Raises:
        ValueError: If ``mcp.topology`` is set to anything other than
            ``per_container_stdio`` (including ``shared_http`` and any other
            unrecognized value).
    """
    mcp_cfg = as_dict(web_terminals.get("mcp"))
    topology = mcp_cfg.get("topology") or SUPPORTED_MCP_TOPOLOGY
    if topology != SUPPORTED_MCP_TOPOLOGY:
        raise ValueError(
            f"modules.web_terminals.mcp.topology {topology!r} is not wired yet for "
            "the shared framework-MCP tier; per_container_stdio is the only "
            "supported topology (a facility's own claude_code.servers custom "
            "`url` entries are a separate, already-supported path and are "
            "unaffected by this restriction)."
        )


#: Landing footer used when ``landing.footer`` is unset. A deployment that wants
#: no footer at all sets it to an empty string — absence means "we did not say",
#: which is different from "we said none".
_DEFAULT_LANDING_FOOTER = (
    "OSPREY multi-user web terminal stack. Experimental system. Proceed with caution."
)

#: Package-relative notice rendered when ``landing.notices`` is absent entirely.
#: Shipped inside the wheel so a config that says nothing about notices still
#: carries the safety copy; ``notices: []`` is how a deployment asks for none.
_PACKAGED_NOTICE = "notices/working-safely.md"

#: Markdown extensions enabled for notice documents. Deliberately empty: notices
#: are prose with headings, lists and emphasis, and every extension is another
#: syntax a facility author has to know about before their file renders the way
#: they meant it to.
_NOTICE_MD_EXTENSIONS: list[str] = []


def _notice_slug(stem: str) -> str:
    """Element id for a notice, derived from its filename stem.

    Gives each section a stable deep-link (``…/#local-procedures``) so an
    operator can be pointed at one notice rather than at the page.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "notice"


def _split_notice(text: str, fallback_title: str) -> tuple[str, str]:
    """Split a notice document into its ``<summary>`` label and its body.

    The document's leading ``# H1`` is the label, so adding a notice needs no
    config beyond the path — the file names itself. A document with no H1 keeps
    all of its text as body and falls back to the filename-derived title, which
    is wrong-looking enough to be noticed without dropping content.
    """
    lines = text.lstrip("\n").splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip(), "\n".join(lines[1:])
    return fallback_title, text


def _render_notice(text: str, *, stem: str) -> dict[str, str]:
    """Convert one notice document to the template's notice shape."""
    import markdown as md

    title, body = _split_notice(text, stem.replace("-", " ").replace("_", " ").strip())
    return {
        "id": _notice_slug(stem),
        "title": title,
        # HTML by construction, so landing.html.j2 emits it with `|safe`. The
        # author of a notice already controls config.yml, so this is not a trust
        # boundary being crossed — but it IS the one value on the page that is
        # not escaped, which is why notices are files a deployment opts into
        # rather than strings anything else can reach.
        "body_html": md.markdown(body, extensions=_NOTICE_MD_EXTENSIONS),
    }


def _packaged_notice() -> dict[str, str]:
    """The shipped default notice, read from package data."""
    notice_path = files("osprey").joinpath(_TEMPLATE_PACKAGE_PATH, _PACKAGED_NOTICE)
    return _render_notice(
        notice_path.read_text(encoding="utf-8"),
        stem=PurePosixPath(_PACKAGED_NOTICE).stem,
    )


def _build_notices(landing_cfg: dict[str, Any], root: dict[str, Any]) -> list[dict[str, str]]:
    """Build the landing page's collapsible notice sections.

    Three cases, deliberately distinct:

    * ``notices`` absent — render the packaged default, so a config that says
      nothing still ships the safety copy.
    * ``notices: []`` — render none. The deployment said so explicitly.
    * ``notices: [path, ...]`` — render those, in order. A listed path that does
      not exist is SKIPPED and reported by :mod:`.lint`, never silently replaced
      with the packaged default: substituting OSPREY's safety text for a
      facility's missing ``local-procedures.md`` would be worse than a gap.
    """
    if "notices" not in landing_cfg:
        return [_packaged_notice()]

    raw = landing_cfg.get("notices")
    if not isinstance(raw, list):
        return []

    repo_root = resolve_repo_root(root)
    notices: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            continue
        doc = repo_root / entry
        if not doc.is_file():
            continue
        notices.append(_render_notice(doc.read_text(encoding="utf-8"), stem=doc.stem))
    return notices


def _landing_footer(landing_cfg: dict[str, Any]) -> str:
    """The landing footer line. Absent means the shipped default; empty means none."""
    footer = landing_cfg.get("footer")
    if footer is None:
        return _DEFAULT_LANDING_FOOTER
    return str(footer)
