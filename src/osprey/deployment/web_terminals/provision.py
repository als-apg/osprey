"""Host-side provisioning of multi-user web-terminal deployments.

Extracts the web-terminal-only provisioning logic ``osprey up`` runs
before and around the compose invocation for a ``modules.web_terminals``
deploy: per-persona local image builds (over the persona projects ``osprey
build`` rendered), ``.env.production`` generation for local-mode deploys, ``.env.auth``
credential provisioning and its fail-closed gate when authentication is on,
rootless-podman ``loginctl`` linger, the advisory post-up ``verify.sh`` smoke
check, and the dual-compose (backend-services + web stack) orchestration itself.

``osprey.deployment.container_lifecycle.deploy_up`` delegates the whole
web-terminal branch to :func:`deploy_up_web_terminals` here; everything generic
or shared with the plain (non-web) deploy path stays in ``container_lifecycle``.
"""

import fnmatch
import os
import shutil
import subprocess
from importlib.resources import as_file, files
from pathlib import Path

import yaml

from osprey.cli.phase_reporter import report_step as _report_step
from osprey.deployment.build_progress import with_plain_build_progress
from osprey.deployment.compose_generator import (
    _stage_dev_wheel_for_context,
    compose_base_cmd,
    resolve_project_name,
    resolve_repo_root,
)
from osprey.deployment.runtime_helper import (
    get_container_image_id,
    get_image_id,
    get_runtime_command,
    runtime_env,
    with_plain_progress,
)
from osprey.deployment.subprocess_capture import run_captured
from osprey.deployment.web_terminals.artifacts import (
    check_bash_launch_token_conflict,
    web_compose_file,
    write_web_terminal_artifacts,
)
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    AuthCredentialsResult,
    AuthSecretsResult,
    ensure_auth_credentials,
    ensure_auth_session_secrets,
    raise_if_env_auth_would_be_interpolated,
)
from osprey.deployment.web_terminals.env_production import ensure_env_production
from osprey.deployment.web_terminals.persona_images import (
    build_persona_images,
    verify_persona_renders,
)
from osprey.deployment.web_terminals.personas import (
    effective_image_source,
    normalize_users,
    resolve_personas,
)
from osprey.deployment.web_terminals.postup_hooks import (
    enable_linger,
    reload_nginx_config,
    run_verify_script,
    warn_if_web_stack_unreachable,
)
from osprey.deployment.web_terminals.render import _auth_tls_context
from osprey.deployment.web_terminals.seeding import seed_user_containers
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")


def preflight_web_terminals(config: dict) -> None:
    """Fail-fast web-terminal preflight, run BEFORE any image build.

    ``deploy_up`` invokes this ahead of its (minutes-long) project-image
    build so the deploy aborts that need no image at all — an
    unresolvable persona catalog, a persona that would hold
    ``BLUESKY_LAUNCH_TOKEN`` while its agent may also run a shell
    (:func:`~osprey.deployment.web_terminals.artifacts.check_bash_launch_token_conflict`),
    a missing provider auth secret
    (:func:`ensure_env_production`'s fail-closed gate), a registry-mode
    deployment with no ``auth.image`` (:func:`_require_auth_sidecar_image`) and
    an auth credential that could not be established
    (:func:`_provision_auth_secrets`) — surface in seconds. Local mode first
    checks that every referenced persona has a rendered project (the credential
    sweep reads each rendered persona's ``config.yml``), exactly as the main
    provisioning path does, which is why the repo root is passed to both: the
    renders it looks for are the ones ``osprey build`` wrote into that repo's
    ``build/``.

    Auth provisioning runs LAST, and in both image-source modes: the sidecar's
    credentials are needed whatever the images come from, but minting them for
    a deploy that :func:`ensure_env_production` is about to abort would write
    (and print) passwords for a stack that never comes up.

    Every step is idempotent, so :func:`deploy_up_web_terminals` re-running
    the same sequence later is a cheap no-op: the persona check reads and
    writes nothing, and an existing ``.env.production`` is returned as-is. Auth
    provisioning is idempotent in the same sense — an established hash or secret
    is never rewritten. Assumes the project-root cwd every
    other step of ``osprey up`` already relies on.

    Nothing is returned: whatever this preflight (or an operator's hand-edit)
    did to ``.env.auth`` is picked up by the deploy's own re-render, which
    stamps the file's content digest into the sidecar's service definition
    (see :func:`~osprey.deployment.web_terminals.artifacts.write_web_terminal_artifacts`)
    — compose itself then recreates the sidecar on the definition change, so
    no authorship signal needs to survive from here to the ``up``.
    """
    repo_root = str(resolve_repo_root(config))
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) == "local":
        facility_prefix = (config.get("facility") or {}).get("prefix") or ""
        registry_cfg = config.get("registry") or {}
        resolved_users = resolve_personas(web_terminals, registry_cfg, facility_prefix, strict=True)
        verify_persona_renders(config, resolved_users, repo_root=Path(repo_root))
    # BEFORE ensure_env_production, for the same reason auth provisioning runs
    # last: a persona that would hold BLUESKY_LAUNCH_TOKEN while its shipped
    # settings still permit `Bash` is a deploy that must not come up, and
    # discovering that after the image build would have already minted (and
    # printed) credentials for it. The render seam and `decommission_user` check
    # again — see check_bash_launch_token_conflict for why all three call sites
    # are load-bearing.
    check_bash_launch_token_conflict(config, repo_root)
    ensure_env_production(config, repo_root)
    # BEFORE the mint, deliberately: a registry-mode deploy that forgot
    # auth.image is already doomed, and minting here first would write (and
    # print, once) a password for a stack that never comes up. Moving this call
    # below _provision_auth_secrets silently loses that property.
    _require_auth_sidecar_image(web_terminals)
    _provision_auth_secrets(web_terminals, repo_root)


def _provision_auth_secrets(web_terminals: dict, repo_root: str) -> None:
    """Establish every auth credential this deployment needs, then gate on it.

    A no-op when ``modules.web_terminals.auth.method`` is ``none`` (the
    default): nothing is read, written or warned about, so an unauthenticated
    deployment behaves byte-for-byte as it did before authentication existed.

    Otherwise, in this order:

    1. **Mint** — :func:`~osprey.deployment.web_terminals.auth_credentials.ensure_auth_credentials`
       for the roster (``password`` mode only: it is the sole method that
       consults a stored hash, and an ``oidc`` deployment must not accumulate
       passwords nobody will ever type), then
       :func:`~osprey.deployment.web_terminals.auth_credentials.ensure_auth_session_secrets`,
       which runs in EVERY method. The latter is not optional in ``oidc`` mode
       for two independent reasons: the sidecar 503s without a cookie-signing
       secret, and the sidecar's compose service declares ``env_file:
       .env.auth`` — ``compose up`` hard-fails when that file does not exist,
       so this call is also what brings the file into being on a fresh root.
       A password the mint has to invent is shown once — but only to a terminal;
       see :func:`_mint_echo` for where it goes otherwise and why.
    2. **Gate** — :func:`_raise_if_auth_provisioning_incomplete`, as a
       post-mint invariant. Deliberately after the mint, never instead of it:
       the question is not "was a credential configured?" but "does one exist
       now?", and only the mint can answer that.

    Nothing wraps the mint in ``try``/``except``: ``ensure_auth_credentials``
    raises (writing nothing) on a roster it cannot key — a charset violation,
    or two usernames colliding onto one credential variable — and that raise IS
    the deploy abort. ``osprey up`` never runs lint, so swallowing it would
    silently deploy a stack where one operator's password opens another's
    terminal.

    The method is read through
    :func:`~osprey.deployment.web_terminals.render._auth_tls_context` rather
    than off the raw stanza, so the deploy path and the rendered artifacts can
    never disagree about what ``auth`` means — and an unsupported method raises
    here as well, ahead of the render that would otherwise be the first to
    notice.

    Whether the mint added content is deliberately NOT reported: the deploy's
    re-render digests ``.env.auth`` into the sidecar's service definition, so
    compose itself detects any content change — authored here or hand-edited —
    without an authorship signal.

    :param web_terminals: The already-unwrapped ``modules.web_terminals`` dict.
    :param repo_root: Directory holding ``.env``, ``.env.auth`` and
        ``.gitignore``.
    :raises RuntimeError: If a credential could not be established (see the
        gate), or if the roster cannot be keyed (charset violation or two
        usernames colliding onto one credential variable, both raised by
        ``ensure_auth_credentials``).
    """
    auth_method = _auth_tls_context(web_terminals)["auth_method"]
    if auth_method == "none":
        return

    _warn_if_env_auth_not_gitignored(Path(repo_root))

    credentials = None
    if auth_method == "password":
        usernames = [entry["name"] for entry in normalize_users(web_terminals.get("users"))]
        credentials = ensure_auth_credentials(usernames, repo_root, echo=_mint_echo())
        _report_unshown_mints(credentials)
    session_secrets = ensure_auth_session_secrets(repo_root)
    _raise_if_auth_provisioning_incomplete(auth_method, credentials, session_secrets)
    # After the mint rather than before it, uniquely on this path: on a fresh
    # root .env.auth does not exist until ensure_auth_session_secrets creates
    # it, so there would be nothing to scan. Safe here because the mint is
    # idempotent and appends only `$`-free values — a refusal leaves the file
    # exactly as a re-run would rebuild it, unlike the credential-removing
    # verbs (see the function's own docstring for why they scan up front).
    raise_if_env_auth_would_be_interpolated(repo_root)


def _stdout_is_a_terminal() -> bool:
    """Whether stdout is a terminal a person is watching right now.

    Its own function because it is the whole basis of the decision below, and
    because a test needs one seam to move rather than a captured stream to fake.
    """
    import sys

    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        # A closed or exotic stream. "Not a terminal" is the safe reading: it
        # withholds a secret rather than printing one somewhere unknown.
        return False


def _mint_echo():
    """The sink :func:`ensure_auth_credentials` shows a minted password through.

    A freshly minted web-terminal password exists in cleartext exactly once, in
    whatever this writes to. On a terminal that is a person reading it, which is
    the whole design: the plaintext is never stored, only its hash.

    When stdout is NOT a terminal it is a file, a pipe, or — the case this
    exists for — a CI job log that is retained and readable by everyone with
    access to the project. So there the sink drops the line on the floor, and
    :func:`_report_unshown_mints` says so afterwards. Nothing is echoed on a
    best-effort basis: a password either goes to a live terminal or nowhere.
    """
    if _stdout_is_a_terminal():
        return print
    return lambda _line: None


def _report_unshown_mints(credentials: AuthCredentialsResult) -> None:
    """Say that passwords were minted but deliberately not shown.

    Deliberately does NOT tell the operator to go read them out of
    ``.env.auth``: that file holds password *hashes*, and the plaintext this
    deploy generated is unrecoverable from the moment it is not printed. The two
    remedies named are the two that exist — rotate to a password of the
    operator's choosing, or supply one up front and re-deploy, which the
    credential path prefers over minting.

    No-op when nothing was minted, and never reached on a terminal, where the
    passwords are shown once as they are generated.
    """
    if not credentials.minted or _stdout_is_a_terminal():
        return
    users = ", ".join(sorted(credentials.minted))
    logger.warning(
        "Minted a login password for web-terminal user(s) %s and did NOT print it: stdout "
        "is not a terminal here, and this output can be retained (a CI job log is readable "
        "by everyone with access to the project). The plaintext is gone — only the hash is "
        "stored in %s. Set a password you choose with `osprey users passwd <user>`, or put "
        "OSPREY_AUTH_PW_<USER> in .env before the first deploy and it will be hashed in "
        "instead of a random one being minted.",
        users,
        credentials.env_auth_path,
    )


def _raise_if_auth_provisioning_incomplete(
    auth_method: str,
    credentials: AuthCredentialsResult | None,
    secrets: AuthSecretsResult,
) -> None:
    """Fail-closed deploy gate: abort before compose when a credential is missing.

    Both provisioning functions *report* a write failure through their
    ``missing`` tuple rather than raising, so this is the single place the
    deploy actually stops — early enough that no container has been created and
    the operator is left with the stack they already had, rather than one whose
    login nobody can pass. BOTH tuples are checked: skipping either would let an
    unwritable ``.env.auth`` log a complaint and proceed to compose anyway.

    A missing password hash can only arise in ``password`` mode, the only mode
    that provisions one at all — hence the ``None`` credentials result
    elsewhere. A missing signing secret counts in every method: without it the
    sidecar answers every request with 503 and the whole deployment is locked
    out.

    :param auth_method: The parsed ``auth.method`` (never ``"none"`` here).
    :param credentials: The :func:`ensure_auth_credentials` result, or ``None``
        when the method provisions no password hashes.
    :param secrets: The :func:`ensure_auth_session_secrets` result.
    :raises RuntimeError: If either ``missing`` tuple is non-empty. The message
        names each affected user and variable — never a password or a hash.
    """
    problems: list[str] = []
    if credentials is not None and credentials.missing:
        problems.append(
            "no login password could be established for web-terminal user(s) "
            f"{', '.join(sorted(credentials.missing))}"
        )
    if secrets.missing:
        problems.append(f"no value could be established for {', '.join(sorted(secrets.missing))}")
    if not problems:
        return
    raise RuntimeError(
        f"Cannot deploy with modules.web_terminals.auth.method: {auth_method} — "
        f"{'; '.join(problems)}. {secrets.env_auth_path} could not be written "
        "(check the file's permissions and the project root's, then re-run). "
        "Aborting before any container is started: bringing the stack up would "
        "leave the affected user(s) unable to log in at all."
    )


def _warn_if_env_auth_not_gitignored(repo_root: Path) -> None:
    """Warn — never block — when ``.gitignore`` does not cover ``.env.auth``.

    Projects scaffolded before authentication existed have a ``.gitignore``
    that lists ``.env`` and ``.env.production`` but not ``.env.auth``, and
    gitignore matches literal names rather than prefixes, so ``.env`` does not
    cover it. The file this preflight is about to write holds password hashes
    and the sidecar's signing secrets; committing it would publish them.

    Coverage is decided by the LAST matching pattern, the way git decides it: a
    later ``!.env.auth`` un-ignores what an earlier ``.env*`` covered, and that
    file IS tracked, so it must still warn.

    Warn-only by design: the file is written with mode 0600 either way, the
    credentials are only *at risk* if someone stages them, and refusing to
    deploy over the contents of a file the operator may not even track would
    make the auth feature unusable outside a git checkout for no security gain.

    A project root with no ``.gitignore`` at all is left alone: that is
    ordinarily a directory nobody commits from, and a deploy is not the place
    to press an operator into starting a gitignore they never had.
    """
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.is_file():
        return
    try:
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        # Unreadable .gitignore — nothing to say that would be trustworthy.
        return
    # git resolves a path against the LAST pattern that matches it, not the
    # first — so `.env*` followed by `!.env.auth` leaves the file TRACKED. Scan
    # every line and keep the polarity of the last match rather than returning
    # on the first: returning early would go quiet on exactly the arrangement
    # that re-exposes the credentials.
    ignored = False
    for line in lines:
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        if negated:
            pattern = pattern[1:]
        # Match the way git does for a root-anchored plain file: leading and
        # trailing "/" are structural, and a pattern may be a glob, so `.env*`
        # counts as covering the file just as `.env.auth` does.
        if fnmatch.fnmatch(AUTH_ENV_FILENAME, pattern.strip("/")):
            ignored = not negated
    if ignored:
        return
    logger.warning(
        "%s does not ignore %s. That file holds this deployment's web-terminal "
        "password hashes and session-signing secrets — add a %s line to %s so it "
        "cannot be committed.",
        gitignore_path,
        AUTH_ENV_FILENAME,
        AUTH_ENV_FILENAME,
        gitignore_path.name,
    )


# ---------------------------------------------------------------------------
# Auth sidecar: image production and the shared force-recreate primitive
# ---------------------------------------------------------------------------

#: The auth sidecar's compose service key, as rendered by
#: ``docker-compose.web.yml.j2``. Every invocation that addresses the sidecar by
#: name reads it from here rather than restating the literal, so the deploy path
#: and the rendered artifact cannot drift apart.
AUTH_SERVICE_NAME = "auth"

#: Build-context directory (relative to the project root) the bundled sidecar
#: Dockerfile is materialized into. Under ``build/`` with every other service's
#: context, so it is regenerable and already gitignored.
AUTH_BUILD_CONTEXT = Path("build") / "services" / "auth_sidecar"

#: Files copied out of the bundled template package to form that context. The
#: ``.dockerignore`` is not optional: the Dockerfile COPYs it as the guaranteed
#: sibling that keeps its optional ``*.wh[l]`` glob matching.
_AUTH_CONTEXT_FILES = ("Dockerfile", ".dockerignore")

#: Package-relative location of those bundled files. Resolved through
#: importlib.resources rather than ``Path(__file__)`` so it works from an
#: installed wheel too, matching render.py's convention for the .j2 sources.
#:
#: They live beside the module's other templates rather than under
#: ``templates/services/``: that tree is catalog-bound (every directory there
#: must be a registered, project-copied build artifact — see
#: ``BuildArtifactCatalog``), and the sidecar is a web-terminals stack member,
#: not a service a project declares under ``services:``.
_AUTH_TEMPLATE_PACKAGE_PATH = "templates/modules/web_terminals/auth_sidecar"


def auth_sidecar_local_tag(config: dict) -> str:
    """The image tag a local-mode deploy builds for the sidecar.

    Must equal what ``docker-compose.web.yml.j2`` renders when ``auth.image``
    is unset (``{{ auth_image | default(facility_prefix ~ "-assistant-auth:local",
    true) }}``) — a mismatch would leave compose referencing a tag nothing
    built, failing at ``up`` with an opaque "no such image".
    """
    facility_prefix = (config.get("facility") or {}).get("prefix") or ""
    return f"{facility_prefix}-assistant-auth:local"


def _require_auth_sidecar_image(web_terminals: dict) -> None:
    """Registry-mode gate: ``auth.image`` must name a published sidecar image.

    Registry mode builds nothing locally, and the compose overlay falls back to
    the ``:local`` tag when ``auth.image`` is unset — a tag that mode never
    produces. Without this the deploy would reach ``compose pull``/``up`` and
    die on a tag no registry has, with nothing to say which key was missing.
    Publishing the image is the facility CI's contract, exactly like
    ``.env.production``; OSPREY only checks that the deployment names one.

    A no-op when authentication is off (no sidecar is rendered at all) and in
    local mode (:func:`build_auth_sidecar_image` produces the tag there).

    :raises RuntimeError: Registry mode, authentication on, ``auth.image`` unset.
    """
    if _auth_tls_context(web_terminals)["auth_method"] == "none":
        return
    if effective_image_source(web_terminals) == "local":
        return
    if _auth_tls_context(web_terminals)["auth_image"]:
        return
    raise RuntimeError(
        "modules.web_terminals.auth.image is not set. A registry-mode deploy "
        "(modules.web_terminals.image_source: registry, the default) never builds "
        "the auth sidecar's image locally, so it must name a published one — "
        "building it and pushing it is the facility CI's job, like .env.production. "
        "Either set modules.web_terminals.auth.image, or set "
        "modules.web_terminals.image_source: local to have `osprey up` build "
        "the sidecar image on this host."
    )


def _materialize_auth_build_context(repo_root: Path, dev_mode: bool) -> Path:
    """Write the bundled sidecar Dockerfile into this project's build tree.

    The sidecar has no rendered project of its own (unlike a persona) and no
    compose ``build:`` block (unlike a backend service), so its build context is
    materialized here from the template package. Overwritten on every deploy: it
    holds no operator-owned content, only the bundled files plus whatever
    ``--dev`` stages beside them.

    :returns: The build-context directory.
    """
    context_dir = repo_root / AUTH_BUILD_CONTEXT
    context_dir.mkdir(parents=True, exist_ok=True)
    template_dir = files("osprey").joinpath(_AUTH_TEMPLATE_PACKAGE_PATH)
    for name in _AUTH_CONTEXT_FILES:
        with as_file(template_dir.joinpath(name)) as source:
            shutil.copyfile(source, context_dir / name)
    # Same wheel-drop convention every service build context uses: under --dev
    # the locally-built wheel lands beside the Dockerfile and its optional COPY
    # glob picks it up. Fail-loud rather than silently deploying the pinned PyPI
    # release under a flag that means "run my local code".
    _stage_dev_wheel_for_context(str(context_dir), dev_mode)
    return context_dir


def build_auth_sidecar_image(config: dict, dev_mode: bool, env: dict[str, str]) -> None:
    """Build the sidecar's ``<facility_prefix>-assistant-auth:local`` image.

    The local-mode counterpart of :func:`_require_auth_sidecar_image`, and the
    only producer of that tag: the web compose overlay declares the sidecar with
    an ``image:`` and no ``build:`` block, so nothing else would ever create it.
    Called from :func:`deploy_up_web_terminals`'s local branch, beside
    :func:`build_persona_images` and before any compose invocation.

    The tag is deliberately local-only, which is exactly why the web stack's
    ``pull`` runs in registry mode ONLY (see :func:`deploy_up_web_terminals`'s
    MODE BRANCH): ``compose pull`` hard-fails on a tag no registry can serve, so
    a local-mode pull would abort the deploy on this image alone.

    Three no-op cases, each deliberate:

    * authentication off — no sidecar service is rendered, so nothing to build;
    * registry mode — the image comes from ``auth.image`` (preflight already
      required it);
    * ``auth.image`` set in local mode — the deployment pinned an external
      image, and building over that pin would produce a tag nothing references.

    :param config: Raw deploy config.
    :param dev_mode: Whether ``--dev`` was passed — stages a locally-built wheel
        into the build context, exactly like every other service image.
    :param env: Environment for the build subprocess.
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    auth_ctx = _auth_tls_context(web_terminals)
    if auth_ctx["auth_method"] == "none":
        return
    if effective_image_source(web_terminals) != "local":
        return
    if auth_ctx["auth_image"]:
        logger.info(
            "Skipping the local auth-sidecar build: modules.web_terminals.auth.image "
            "pins %s, so the deployment supplies the image.",
            auth_ctx["auth_image"],
        )
        return

    repo_root = resolve_repo_root(config)
    context_dir = _materialize_auth_build_context(repo_root, dev_mode)

    from osprey import __version__ as osprey_version

    tag = auth_sidecar_local_tag(config)
    project_label = resolve_project_name(config)
    cmd = [
        get_runtime_command(config)[0],
        "build",
        "-t",
        tag,
        "-f",
        str(context_dir / "Dockerfile"),
        # OSPREY_PROJECT_NAME is what stamps `com.osprey.project` on the image
        # (the Dockerfile's final metadata-only layer), the same ownership label
        # persona images carry — `nuke` verifies it before removing a tag, so an
        # unlabeled image here would be left behind by a full teardown.
        "--build-arg",
        f"OSPREY_PROJECT_NAME={project_label}",
        "--build-arg",
        f"OSPREY_VERSION={osprey_version}",
    ]
    if dev_mode:
        cmd.extend(["--build-arg", "OSPREY_DEV=1"])
    with_plain_build_progress(cmd)
    cmd.append(str(context_dir))

    logger.key_info("Building auth sidecar image %s:", tag)
    logger.debug("Running command:\n    %s", " ".join(cmd))
    # Function-level import, like `compose_build_step_reporter` below:
    # container_lifecycle imports this module at its own top level, so the
    # favour cannot be returned there.
    from osprey.deployment.container_lifecycle import single_image_build_reporter

    # Watched for the duration of the build and no longer; the step line below
    # is what reports the finished image.
    with (report := single_image_build_reporter(tag)):
        run_captured(
            cmd, env=env, spool_name="build-auth-sidecar", repo_root=repo_root, on_line=report
        )
    _report_step(f"auth sidecar image {tag}")


def web_stack_compose_cmd(
    config: dict,
    env_file_args: list[str] | None = None,
    *,
    repo_root: Path | str | None = None,
) -> list[str]:
    """The web stack's pinned ``<runtime> compose ... -f build/docker-compose.web.yml`` argv.

    One builder for every invocation against the web compose project — the
    deploy path's ``pull``/``up``/recreate, ``deploy_down_web_terminals``, and
    the roster verbs' nginx reload and sidecar recreate — so no caller can fork
    the argv and drift from it. It carries the same pinned base as the services
    stack (:func:`~osprey.deployment.compose_generator.compose_base_cmd`): the
    repo root as project directory, the rendered compose file addressed under
    ``build/``, and the repo-root ``.env``.

    That pin is what makes the file's own relative paths correct. The rendered
    compose file lives in ``build/`` while the secrets it names (``.env.auth``,
    ``.env.production``) live at the repo root, and compose resolves both
    against the project directory rather than against the file's own location —
    so pinning the root is what lets one rendered file reference both zones.

    :param env_file_args: The ``--env-file`` fragment the caller already
        resolved. ``None`` resolves it here through
        ``container_lifecycle._env_file_args``, for the roster verbs that are
        entered straight from the CLI and never had one to thread down — so the
        rule (and its "no .env" warning) keeps a single definition.
    :param repo_root: The deployment repo. Defaults to the one resolved from
        ``config``.
    """
    root = Path(repo_root) if repo_root is not None else resolve_repo_root(config)
    if env_file_args is None:
        # Function-local import: container_lifecycle imports this module at its
        # top level, so importing it back at module scope would be a cycle
        # (same reason persona_images imports _resolve_pip_spec locally).
        from osprey.deployment.container_lifecycle import _env_file_args

        env_file_args = _env_file_args(root)
    return compose_base_cmd(
        with_plain_progress(get_runtime_command(config)),
        [web_compose_file(root)],
        root,
        env_file_args,
    )


def force_recreate_auth_sidecar(
    config: dict,
    env_file_args: list[str] | None = None,
    *,
    repo_root: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Recreate the auth sidecar container so it re-reads ``.env.auth``.

    Compose bakes ``env_file`` CONTENT into a container at creation time, and
    podman-compose never recreates a container whose service definition is
    unchanged — so after ``.env.auth`` changes, neither a plain ``up -d`` nor a
    restart puts the new credentials in force. Only a recreate does. (The
    deploy path needs no call here: it re-renders the compose file, whose
    digest label turns the content change into a service-definition change
    that compose itself acts on — see
    :func:`~osprey.deployment.web_terminals.artifacts.write_web_terminal_artifacts`.)

    One function because its callers — the paths that change ``.env.auth``
    WITHOUT issuing a plain ``up`` the label could act through — must all do
    it identically:

    1. the lifecycle verbs (``decommission_user``/``prune_users``), which purge
       a departed user's entries and must kill that user's sessions at once;
    2. ``osprey users passwd <user>``, so a rotated password takes effect
       immediately rather than at the next full deploy.

    Re-renders the artifacts (via :func:`write_web_terminal_artifacts`) before
    the recreate, so the container it creates carries the digest label of the
    exact ``.env.auth`` it bakes. Without this, every caller above recreates
    from a compose file rendered BEFORE its file mutation, leaving label !=
    sha256(baked env): the next ``osprey up`` re-renders the current digest,
    sees a definition change, and spuriously bounces an already-current sidecar
    — and, worse, a byte-exact revert of ``.env.auth`` to the pre-mutation
    content would render a digest MATCHING the stale label, so podman-compose
    would skip the recreate and silently keep serving the reverted-away
    credentials. The render is best-effort: if it fails, the recreate still
    runs against the existing compose file (a recreate always re-bakes the
    CURRENT file content — only the label lags, at the cost of one spurious
    bounce at the next deploy), because skipping the recreate over a render
    error would leave a purged user's password in force.

    Scoped to the single ``auth`` service: recreating the whole stack would
    bounce every live terminal for a change none of them can see.

    A no-op (with a warning) when the repo has no rendered
    ``build/docker-compose.web.yml`` — there is no stack to recreate, and
    failing here would turn "nothing was deployed yet" into a deploy error.
    One repo answers that probe, the pre-recreate re-render, and the compose
    argv: all three are taken from the ``repo_root`` resolved once at the top
    of this function, so the probe can only ever address the file the recreate
    runs against. Split, they diverge quietly — a probe made against the
    working directory answered "no" for a repo whose stack was rendered and
    running, so ``osprey users passwd`` warned, recreated nothing, and reported
    a rotation that was never put in force.

    :param config: Raw deploy config (selects the runtime and the pinned
        compose project).
    :param env_file_args: The ``["--env-file", ".env"]`` (or ``[]``) argv
        fragment the caller resolved, exactly as the ``up``/``down`` paths take
        it; ``None`` lets :func:`web_stack_compose_cmd` resolve it, which is what
        the lifecycle verbs (``passwd``, ``decommission``, ``prune``) pass since
        they are entered straight from the CLI.
    :param repo_root: The deployment repo this recreate acts on — the one whose
        ``build/`` is probed and re-rendered and whose root compose pins as the
        project directory. Passed by every caller that already resolved a repo
        from its ``config_path`` (the roster verbs in
        :mod:`~osprey.deployment.web_terminals.lifecycle`), which is what lets
        them run from any working directory. ``None`` falls back to
        :func:`~osprey.deployment.compose_generator.resolve_repo_root` on
        ``config`` alone, whose last resort is the working directory — correct
        only for a caller standing in the repo.
    :param env: Base environment for the subprocess; defaults to this process's
        own. Pinned with ``COMPOSE_PROJECT_NAME`` here either way, so the
        recreate addresses the same compose project the stack was brought up in.
    """
    root = Path(repo_root) if repo_root is not None else resolve_repo_root(config)
    compose_file = web_compose_file(root)
    if not compose_file.exists():
        logger.warning(
            "No rendered web stack at %s — skipping the auth sidecar recreate. "
            "Any %s change takes effect at the next `osprey up`.",
            compose_file,
            AUTH_ENV_FILENAME,
        )
        return
    # Restore the label invariant before recreating: the compose file on disk
    # was rendered BEFORE this caller's .env.auth mutation, so its digest label
    # describes the file's previous content. See the docstring for both
    # consequences; best-effort per the same docstring. Handed the same `root`
    # the probe and the argv use — a re-render into a different repo's build/
    # would leave the file this recreate actually reads carrying the stale
    # label it exists to refresh.
    try:
        write_web_terminal_artifacts(config, root)
    except (ValueError, OSError) as exc:
        logger.warning(
            "Could not re-render the web-terminal artifacts before the auth sidecar "
            "recreate (%s). Recreating against the existing docker-compose.web.yml — "
            "the recreated container still bakes the current %s, but its digest label "
            "lags until the next `osprey up` re-renders (costing one extra "
            "sidecar recreate there).",
            exc,
            AUTH_ENV_FILENAME,
        )
    # Deliberately NOT scanned here, though this is the one line every path
    # that re-bakes .env.auth passes through. A recreate is always the step
    # that puts an ALREADY-APPLIED file change into force, so refusing at this
    # point would leave the file and the running sidecar disagreeing — for
    # `decommission`/`prune`, the file would say a departed user is gone while
    # the sidecar still accepts their password, which is the precise divergence
    # those verbs exist to close. The scan runs ahead of each mutation instead;
    # see auth_credentials.raise_if_env_auth_would_be_interpolated.
    _force_recreate_services(
        web_stack_compose_cmd(config, env_file_args, repo_root=root),
        runtime_env(config, env if env is not None else dict(os.environ), ignore_orphans=True),
        [AUTH_SERVICE_NAME],
        repo_root=root,
    )


def _force_recreate_services(
    web_cmd: list[str],
    run_env: dict[str, str],
    services: list[str],
    *,
    repo_root: Path | str | None = None,
) -> None:
    """Run one service-scoped ``up -d --force-recreate`` for ``services``.

    The single place that argv is built. Always service-scoped: a bare
    ``up -d --force-recreate`` would bounce every live terminal in the stack,
    which is exactly what both callers (the post-``up`` reconcile and
    :func:`force_recreate_auth_sidecar`) exist to avoid.

    :param repo_root: The deployment repo whose ``var/logs/`` holds this
        recreate's spooled output. Both callers already resolved it.
    """
    cmd = web_cmd + ["up", "-d", "--force-recreate", *services]
    logger.debug(f"Running command:\n    {' '.join(cmd)}")
    run_captured(cmd, env=run_env, spool_name="compose-force-recreate", repo_root=repo_root)
    _report_step(f"recreated {', '.join(services)}")


def deploy_up_web_terminals(
    config: dict,
    compose_files: list[str],
    dev_mode: bool,
    env: dict[str, str],
    env_file_args: list[str],
) -> None:
    """Reconcile the web-terminal stack (plus any co-deployed backend services).

    Renders and writes the web-terminal artifacts (``docker-compose.web.yml``,
    ``nginx/nginx.conf``, ``nginx/landing.html``) under the project root via
    :func:`write_web_terminal_artifacts`, ensures ``.env.production`` exists
    (:func:`ensure_env_production` — see below), then reconciles TWO
    INDEPENDENT compose invocations rather than merging everything behind one
    ``-f`` list:

    1. The backend-services stack (``compose_files``, possibly just the
       network-only top-level file when no service is actually deployed —
       see the ``deployed_services`` guard below), exactly as the plain
       non-web path would run it (``up`` (``--build`` under ``dev_mode``)
       ``-d``, no ``pull``).
    2. The web-terminal stack (``docker-compose.web.yml`` alone): ``up -d``,
       preceded by ``pull`` in registry mode only (see the mode branch
       below).

    MODE BRANCH — ``modules.web_terminals.image_source`` (default
    ``"registry"``), read directly here rather than threaded through as a
    parameter so this function stays the single place that decides the
    mode-dependent step order:

    - **registry** (the default): :func:`ensure_env_production`
      only exists-checks in this mode (raises if ``.env.production`` is
      missing — a registry deploy expects CI to have produced it already),
      then the web stack runs ``pull`` before ``up -d``.
    - **local**: :func:`ensure_env_production` generates ``.env.production``
      from ``.env`` when absent. Then :func:`build_persona_images` builds
      every referenced persona's ``<project>-<persona>:local`` image —
      called with :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
      ``strict=True`` output, so an unresolvable persona reference (unknown
      catalog entry, or ``local`` mode with no catalog/``default_persona``
      configured at all) raises HERE, before any compose invocation, rather
      than surfacing as an opaque "no such image" failure at ``compose up``.
      Local mode's web-stack invocation then runs bare ``up -d`` — NO
      ``pull`` anywhere on this path. This is load-bearing, not an
      optimization: ``compose pull`` hard-fails (exit 1) on a local-only
      tag it can't find in any registry, the same trap the backend-services
      sub-invocation below already avoids for a service with no published
      upstream tag.

    WHY TWO INVOCATIONS, NOT ONE ``-f a -f b -f build/docker-compose.web.yml``:
    not path resolution. Compose resolves every *relative* path in
    every merged ``-f`` file (bind-mount sources, ``build:`` contexts,
    ``env_file:``) against ONE directory, and both invocations below pin
    that directory to the same repo root
    (:func:`~osprey.deployment.compose_generator.compose_base_cmd`), with every
    template spelled against it. Two files, one base — which is what makes the
    web file's ``env_file: .env.auth`` (repo root) and its ``./build/nginx/…``
    mounts (render zone) both resolve correctly from a single pin.

    What keeps them apart is the command SEQUENCE, not the paths: the web stack
    runs ``pull`` before ``up`` in registry mode, which the services stack must
    never do (a service with only a ``build:`` block has no upstream tag, and
    ``compose pull`` hard-fails on it), while ``dev_mode`` gives the services
    stack a ``build`` step the web stack has no use for. Merging the two behind
    one ``-f`` list would mean re-expressing that split as per-service argument
    lists; it is possible under the pinned base, and deliberately not done here.
    Splitting costs nothing functionally: the web stack runs every service under
    ``network_mode: host`` (see ``docker-compose.web.yml.j2``), so it never
    needed to join ``osprey-network`` from the services file anyway.

    The services sub-invocation only runs when a real service is deployed
    (``config["deployed_services"]`` non-empty): ``compose_files`` always
    includes the top-level ``build/services/docker-compose.yml`` (a bare
    network declaration, no ``services:`` key) even for a web-terminals-only
    deploy, and ``compose up`` on a file with zero services fails outright
    with ``no service selected`` — this is exactly why the *plain* non-web
    path's own early-return guards on ``deployed_services`` before ever
    reaching ``up``. It also never runs ``pull``: unlike the web stack's
    images (always registry-hosted — ``nginx:*-alpine`` and
    ``<registry>/web-terminal:latest``), a deployed service like
    ``event_dispatcher`` may declare only a ``build:`` block with no
    published upstream tag, and ``compose pull`` hard-fails (exit 1) on a
    buildable service compose can't find remotely — ``compose up`` builds it
    locally instead, exactly like the plain non-web path already relies on.

    This path always runs detached, regardless of the caller's ``--detached``
    flag: ``deploy_up``'s non-detached path ``os.execvpe``-replaces the current
    process (see below), which would make it impossible for a caller to run
    anything — e.g. the post-up hook this function ends with — after
    ``compose up`` returns. A web-terminal deploy needs that hook, so it can
    never take the execvpe path.

    Idempotency comes from compose's own reconciliation (``pull`` (registry
    mode only) + ``up -d`` for the web stack; plain ``up -d`` for the
    services stack, mirroring the non-web path): no bespoke digest/state
    diffing, and deliberately no ``--force-recreate`` on either invocation,
    so a no-op second run recreates zero containers. Each invocation is
    preceded by a ``rm -f`` stale-container preflight (see the inline
    comments for why it is service-scoped and why ``--remove-orphans`` is
    forbidden on this path); running containers are untouched, so the
    zero-recreate property holds. Under ``dev_mode`` the
    services ``up -d`` also carries ``--build``, mirroring the non-web
    path's dev-mode ``--build``: without it, a co-deployed backend service's
    cached image tag would keep running the stale code from its first
    build. The web stack never needs ``--build`` — none of its images have
    a ``build:`` block (registry mode) or is otherwise built with this
    module's own dev-wheel machinery (local mode — see
    :func:`build_persona_images`).

    :param config: Raw deploy config.
    :param compose_files: Compose files ``prepare_compose_files`` already
        resolved — always at least the top-level network-only file, even for
        a web-terminals-only deploy (see the ``deployed_services`` guard
        above for why that alone doesn't get an ``up`` invocation).
    :param dev_mode: Whether ``--dev`` was passed; appends ``--build`` to the
        services stack's ``up -d`` invocation when set, and is threaded into
        :func:`build_persona_images` (local mode) for its own dev-wheel
        staging.
    :param env: Base environment for the pull/up subprocesses (already has
        ``DEV_MODE`` applied by the caller when relevant); pinned with
        ``COMPOSE_PROJECT_NAME`` via :func:`runtime_env` before use here so
        both invocations share one project namespace — and so the volume
        namespace compose derives matches the project name baked into
        container labels.
    :param env_file_args: The ``["--env-file", ".env"]`` (or ``[]``) argv
        fragment ``deploy_up`` resolved via ``_env_file_args``; passed in
        rather than recomputed here so the "no .env" warning stays defined in
        one place and this module needs no import back into
        ``container_lifecycle``.
    """
    # The deployment repo: the compose project directory both invocations below
    # are pinned to, the root the artifacts are rendered under (into build/),
    # and the anchor for every secret file this path reads.
    repo_root = resolve_repo_root(config)

    write_web_terminal_artifacts(config, repo_root)

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    local_mode = effective_image_source(web_terminals) == "local"

    if local_mode:
        # strict=True: an unresolvable persona reference is a misconfiguration
        # that must surface HERE -- before compose ever runs -- not as an
        # opaque unbuilt-tag failure at `compose up` (see docstring's MODE
        # BRANCH section). `osprey up` never runs lint, so this strict
        # resolve is the only preflight standing between a broken persona
        # catalog and that opaque failure.
        facility_prefix = (config.get("facility") or {}).get("prefix") or ""
        registry_cfg = config.get("registry") or {}
        resolved_users = resolve_personas(web_terminals, registry_cfg, facility_prefix, strict=True)
        # Every referenced persona must already have the project `osprey build`
        # rendered for it, so the image build below always finds a complete
        # context. This renders nothing: a gap is a stale or partial build, and
        # the refusal names the rebuild that fixes it.
        verify_persona_renders(config, resolved_users, repo_root=repo_root)
        # ensure_env_production AFTER that check because the check REFUSES: its
        # claude_code credential sweep reads each rendered persona's config.yml,
        # so a deploy missing one has to be stopped before the sweep meets a
        # directory that is not there. Still BEFORE any compose invocation: a
        # missing/ungeneratable .env.production would otherwise surface as an
        # opaque compose "env file not found" failure only once `up` runs.
        ensure_env_production(config, repo_root)
        build_persona_images(config, resolved_users, dev_mode, env)
        # The sidecar's own local-only tag, produced by the same rule and at the
        # same point as the persona images: nothing else builds it, and the web
        # stack's `up` below would otherwise die on a tag that does not exist.
        build_auth_sidecar_image(config, dev_mode, env)
    else:
        # Registry mode pulls prebuilt images and has no render to check; the
        # same before-compose rule holds.
        ensure_env_production(config, repo_root)

    run_env = runtime_env(config, env, ignore_orphans=True)

    # ---- backend services (own compose project directory: build/services/) --
    # Skipped when no real service is deployed -- see docstring for why
    # `up` on the network-only top-level file alone would fail outright.
    if config.get("deployed_services"):
        services_base = compose_base_cmd(
            with_plain_progress(get_runtime_command(config)),
            compose_files,
            repo_root,
            env_file_args,
        )
        # Stale-container preflight (see deploy_up): clear this stack's own
        # wedged created/exited containers — a created container from an
        # aborted deploy holds its published host ports on Docker Desktop and
        # blocks the next `up`. `rm -f` is service-scoped to THIS invocation's
        # -f files, never touches running containers or volumes, and no-ops
        # (exit 0) on a clean stack. Deliberately NOT `--remove-orphans`
        # anywhere on this path: both invocations share one
        # COMPOSE_PROJECT_NAME, so orphan-removal in either would destroy the
        # OTHER stack's containers as "orphans" of the shared project.
        services_rm = services_base + ["rm", "-f"]
        logger.debug(f"Running command:\n    {' '.join(services_rm)}")
        run_captured(
            services_rm,
            env=run_env,
            spool_name="compose-services-rm",
            repo_root=repo_root,
            check=False,
        )
        _report_step("cleared stale service containers")
        if dev_mode:
            # Mirrors the plain non-web path's dev-mode build (see deploy_up):
            # without a rebuild, a co-deployed service's cached image tag keeps
            # running the stale code from its first build. Build in its own step,
            # then `up --no-build`, to dodge the `up --build` containerd
            # image-store race.
            # Function-level import, like `_env_file_args` below:
            # container_lifecycle imports this module at its own top level, so
            # the favour cannot be returned there.
            from osprey.deployment.container_lifecycle import compose_build_step_reporter

            services_build = services_base + ["build"]
            logger.debug(f"Running command:\n    {' '.join(services_build)}")
            # Watched only for as long as the build runs — same scope as the
            # plain path's build (see _start_stack).
            with (report := compose_build_step_reporter()):
                run_captured(
                    services_build,
                    env=run_env,
                    spool_name="build-services",
                    repo_root=repo_root,
                    on_line=report,
                )
            _report_step("built service images")
        services_cmd = services_base + ["up"]
        if dev_mode:
            services_cmd.append("--no-build")
        services_cmd.append("-d")
        logger.debug(f"Running command:\n    {' '.join(services_cmd)}")
        run_captured(
            services_cmd, env=run_env, spool_name="compose-services-up", repo_root=repo_root
        )
        _report_step("backend services started")

    # ---- web-terminal stack (same pinned project directory: the repo root) --
    web_cmd = web_stack_compose_cmd(config, env_file_args, repo_root=repo_root)

    # Same stale-container preflight as the services stack above (and same
    # no-`--remove-orphans` constraint — see that comment).
    web_rm = web_cmd + ["rm", "-f"]
    logger.debug(f"Running command:\n    {' '.join(web_rm)}")
    run_captured(web_rm, env=run_env, spool_name="compose-web-rm", repo_root=repo_root, check=False)
    _report_step("cleared stale web-terminal containers")

    if not local_mode:
        # Registry mode only: local-only tags have no upstream to pull from,
        # and `compose pull` hard-fails (exit 1) on one -- see docstring's
        # MODE BRANCH section. This is the load-bearing guard the task exists
        # to add; never run `pull` unconditionally here again.
        pull_cmd = web_cmd + ["pull"]
        logger.debug(f"Running command:\n    {' '.join(pull_cmd)}")
        run_captured(pull_cmd, env=run_env, spool_name="compose-web-pull", repo_root=repo_root)
        _report_step("pulled web-terminal images")

    up_cmd = web_cmd + ["up", "-d"]
    logger.debug(f"Running command:\n    {' '.join(up_cmd)}")
    run_captured(up_cmd, env=run_env, spool_name="compose-web-up", repo_root=repo_root)
    _report_step("web-terminal stack started")

    # Post-`up` recreate for podman's same-tag image drift. A changed
    # `.env.auth` needs nothing here: the render above stamped its content
    # digest into the sidecar's service definition, so the `up -d` itself
    # already recreated the sidecar on any content change.
    _reconcile_web_stack_recreates(config, web_cmd, run_env, repo_root=repo_root)

    # Hot-reload nginx: `up -d` never restarts a running nginx whose
    # bind-mounted nginx.conf/landing.html CONTENT changed — the container
    # definition is unchanged, so compose reconciles nothing and the freshly
    # rendered routes silently never take effect. `nginx -s reload` is
    # zero-downtime and a no-op when the config is unchanged.
    reload_nginx_config(web_cmd, run_env)

    # -----------------------------------------------------------------------
    # POST-UP HOOK — web-terminal reconcile complete (`compose up -d`
    # succeeded, containers running). Linger runs first so a rootless-podman
    # host survives the deploy operator's session ending before seeding's
    # (longer-running, per-user) exec calls are attempted; seeding itself
    # tolerates per-user failures and logs rather than aborting the deploy.
    # verify.sh runs once containers are seeded, so its health probes see the
    # fully-reconciled state -- see run_verify_script for why its result is
    # advisory only and never raises from here. The host-reachability probe
    # runs last and is likewise advisory (see warn_if_web_stack_unreachable:
    # on Docker Desktop a fully-healthy stack can still be unreachable from
    # the host). It gets this invocation's `web_cmd`/`run_env` so it can bounce
    # the stack to re-register a stale Docker Desktop port forward -- the one
    # failure mode `up -d` structurally cannot fix on its own, because the
    # containers' definitions are unchanged and compose reconciles nothing.
    # -----------------------------------------------------------------------
    enable_linger(config, run_env)
    seed_user_containers(config, env=run_env)
    run_verify_script(str(repo_root), run_env)
    warn_if_web_stack_unreachable(config, web_cmd=web_cmd, run_env=run_env)


def _reconcile_web_stack_recreates(
    config: dict,
    web_cmd: list[str],
    run_env: dict[str, str],
    *,
    repo_root: Path | str | None = None,
) -> None:
    """Issue the one post-``up`` force-recreate this deploy needs: image drift.

    **Image drift** (podman only). ``podman-compose`` 1.0.6 treats a
    container as up-to-date whenever its service definition is unchanged, so
    a same-tag digest change (a re-pulled ``:latest``, or any moving tag)
    never triggers a recreate — ``up -d`` leaves the container running the
    previous image. This inspects, for each service the rendered
    ``docker-compose.web.yml`` declares, the image ID its ``image:``
    reference now resolves to versus the image ID its running container was
    created from. Gated to podman: ``docker compose`` already recreates a
    service after its image is re-pulled, so a second forced recreate would
    needlessly bounce a live terminal. Every inspection is advisory — a
    service whose image or container can't be inspected (missing image,
    container not yet created) is skipped, never aborting the deploy.

    A stale ``.env.auth`` is NOT this function's problem: the render
    step digests the file into the auth sidecar's service definition (the
    ``osprey.auth.env.digest`` label), and a definition change is the recreate
    trigger every compose implementation honours — verified on Docker Compose
    v2.34 and podman-compose 1.0.6/1.6.0, whose config hashes both cover
    service labels. Image drift stays here because no label can express "the
    same tag now names different bytes".

    An all-match run issues no compose command at all.

    :param config: Raw deploy config (selects the runtime).
    :param web_cmd: The web stack's ``<runtime> compose -f docker-compose.web.yml
        [--env-file ...]`` base argv, reused verbatim so the recreate runs
        against the same compose project as the preceding ``up``.
    :param run_env: The ``COMPOSE_PROJECT_NAME``-pinned environment shared by
        every web-stack invocation.
    :param repo_root: The deployment repo — the rendered compose file this
        reads and the spooled output of any recreate it issues both hang off
        it. Defaults to the one resolved from ``config``.
    """
    root = Path(repo_root) if repo_root is not None else None
    changed: list[str] = []

    runtime = get_runtime_command(config)[0]
    if runtime == "podman":
        # The rendered compose file (written at the top of
        # deploy_up_web_terminals) is the single source of truth for which
        # services exist and what image / container name each carries -- reading
        # it here keeps this reconcile free of any per-service name
        # reconstruction and covers nginx and every per-user terminal uniformly.
        # Resolved lazily, inside the branch that needs it: the docker path
        # touches neither the compose file nor a recreate.
        root = root if root is not None else resolve_repo_root(config)
        compose_file = web_compose_file(root)
        try:
            compose_doc = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "image-drift reconcile skipped: could not read %s: %s", compose_file, exc
            )
            compose_doc = None

        services = (compose_doc or {}).get("services") or {}
        for service_name, service in services.items():
            service = service or {}
            image = service.get("image")
            container = service.get("container_name")
            if not image or not container:
                continue
            tag_id = get_image_id(runtime, image, env=run_env)
            running_id = get_container_image_id(runtime, container, env=run_env)
            if tag_id is None or running_id is None:
                # Missing image or not-yet-created container -- advisory skip.
                continue
            if tag_id != running_id:
                changed.append(service_name)

    if not changed:
        return

    _force_recreate_services(web_cmd, run_env, changed, repo_root=root)


def deploy_down_web_terminals(
    config: dict,
    env: dict[str, str],
    env_file_args: list[str],
) -> None:
    """Tear down the web-terminal stack — the mirror of
    :func:`deploy_up_web_terminals`'s second compose invocation.

    ``deploy_down``'s services invocation does not carry the web compose file
    in its ``-f`` list (the two stacks stay separate invocations — see the WHY
    TWO INVOCATIONS note on :func:`deploy_up_web_terminals`), so the web stack
    needs this dedicated ``down``. Without it the web containers outlive every
    ``osprey down`` — and because their ``container_name``s are fixed
    host-global identifiers (``<prefix>-web-<user>``, ``<prefix>-nginx``),
    the NEXT web-terminals deploy on the host, from any project, dies at
    ``up`` with a container-name Conflict instead of reconciling.

    A no-op when the repo has no rendered ``build/docker-compose.web.yml``
    (nothing was ever deployed from here, or ``build/`` was wiped).
    Volumes are deliberately kept, mirroring the services
    ``down`` (no ``--volumes``): per-user claude-config/agent-data volumes
    are the durable user state ``osprey users remove`` manages.

    Best-effort: a failing web ``down`` is logged loudly but never raises —
    the caller's services ``down`` (which execvpe-replaces the process) must
    still run, or a broken web stack would leave the backend services
    running too.

    :param config: Raw deploy config (resolves the pinned compose project).
    :param env: Base environment to layer the ``COMPOSE_PROJECT_NAME`` pin
        onto, exactly like the ``up`` path's invocations.
    :param env_file_args: ``["--env-file", ".env"]`` (or ``[]``) argv
        fragment, resolved by the caller.
    """
    root = resolve_repo_root(config)
    if not web_compose_file(root).exists():
        return
    down_cmd = web_stack_compose_cmd(config, env_file_args, repo_root=root)
    down_cmd.append("down")
    logger.debug(f"Running command:\n    {' '.join(down_cmd)}")
    result = subprocess.run(
        down_cmd,
        env=runtime_env(config, env, ignore_orphans=True),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            "web-terminal stack down failed (rc=%s) — its containers may still be running:\n%s",
            result.returncode,
            result.stderr,
        )
