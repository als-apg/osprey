"""Host-side provisioning of multi-user web-terminal deployments.

Extracts the web-terminal-only provisioning logic ``osprey deploy up`` runs
before and around the compose invocation for a ``modules.web_terminals``
deploy: per-persona local image builds (rendering any missing persona project
on demand), ``.env.production`` generation for local-mode deploys, rootless-
podman ``loginctl`` linger, the advisory post-up ``verify.sh`` smoke check, and
the dual-compose (backend-services + web stack) orchestration itself.

``osprey.deployment.container_lifecycle.deploy_up`` delegates the whole
web-terminal branch to :func:`deploy_up_web_terminals` here; everything generic
or shared with the plain (non-web) deploy path stays in ``container_lifecycle``.
"""

import os
import subprocess
from pathlib import Path

from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.web_terminals.artifacts import write_web_terminal_artifacts
from osprey.deployment.web_terminals.env_production import ensure_env_production
from osprey.deployment.web_terminals.persona_images import (
    auto_render_missing_personas,
    build_persona_images,
)
from osprey.deployment.web_terminals.personas import effective_image_source, resolve_personas
from osprey.deployment.web_terminals.postup_hooks import (
    enable_linger,
    reload_nginx_config,
    run_verify_script,
    warn_if_web_stack_unreachable,
)
from osprey.deployment.web_terminals.seeding import seed_user_containers
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")


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

    - **registry** (default, today's path): :func:`ensure_env_production`
      only exists-checks in this mode (raises if ``.env.production`` is
      missing — a registry deploy expects CI to have produced it already),
      then the web stack runs ``pull`` before ``up -d``, exactly as before
      this task.
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

    WHY TWO INVOCATIONS, NOT ONE ``-f a -f b -f docker-compose.web.yml``:
    compose resolves every *relative* path in *every* merged ``-f`` file
    (bind-mount sources, ``build:`` contexts, ``env_file:``) against the
    directory of the FIRST ``-f`` file — the "compose project directory" —
    never against the file's own directory. ``compose_files`` are written
    under ``build/services/`` and their own templates already lean on that
    rule (see the comment atop ``event_dispatcher/docker-compose.yml.j2``);
    ``docker-compose.web.yml`` is written to the project ROOT by
    :func:`write_web_terminal_artifacts` and its relative paths
    (``env_file: .env.production``, ``./nginx/nginx.conf``, ``./nginx/
    landing.html``) are project-root-relative. Merging both behind one
    ``-f`` list makes compose resolve the web file's paths against
    ``build/services/`` instead — real deploys failed immediately with
    ``env file .../build/services/.env.production not found``.

    A single merged invocation with ``--project-directory <project_root>``
    was considered and rejected: pinning the project directory to the root
    fixes the web file but breaks EVERY service template's own relative
    paths the same way in the other direction (verified with a real
    ``compose ... --project-directory . config``: ``event-dispatcher``'s
    ``build.context`` resolved to ``<root>/event_dispatcher`` instead of the
    real ``build/services/event_dispatcher``). Two invocations sidestep the
    conflict entirely — each compose file gets the project directory (its
    own) it was actually written to resolve against — and cost nothing
    functionally: the web stack runs every service under
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
    write_web_terminal_artifacts(config)

    # project_root mirrors every other cwd-relative assumption already baked
    # into this function (write_web_terminal_artifacts's own dest_dir="."
    # default, _env_file_args' Path(".env")): `osprey deploy up` is always
    # invoked from the project root.
    project_root = os.getcwd()

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    local_mode = effective_image_source(web_terminals) == "local"

    if local_mode:
        # strict=True: an unresolvable persona reference is a misconfiguration
        # that must surface HERE -- before compose ever runs -- not as an
        # opaque unbuilt-tag failure at `compose up` (see docstring's MODE
        # BRANCH section). `osprey deploy up` never runs lint, so this strict
        # resolve is the only preflight standing between a broken persona
        # catalog and that opaque failure.
        facility_prefix = (config.get("facility") or {}).get("prefix") or ""
        registry_cfg = config.get("registry") or {}
        resolved_users = resolve_personas(web_terminals, registry_cfg, facility_prefix, strict=True)
        # Render any referenced persona whose project isn't on disk yet, so the
        # image build below always finds a complete context -- the `osprey build`
        # + `osprey deploy up` demo promise, no manual per-persona builds.
        auto_render_missing_personas(config, resolved_users, env)
        # ensure_env_production AFTER auto-render (its claude_code credential
        # sweep reads each rendered persona's config.yml -- on a first deploy
        # those exist only once auto-render has run) but still BEFORE any
        # compose invocation: a missing/ungeneratable .env.production would
        # otherwise surface as an opaque compose "env file not found" failure
        # only once `up` runs.
        ensure_env_production(config, project_root)
        build_persona_images(config, resolved_users, dev_mode, env)
    else:
        # Registry mode has no auto-render; the same before-compose rule holds.
        ensure_env_production(config, project_root)

    run_env = runtime_env(config, env)

    # ---- backend services (own compose project directory: build/services/) --
    # Skipped when no real service is deployed -- see docstring for why
    # `up` on the network-only top-level file alone would fail outright.
    if config.get("deployed_services"):
        services_base = get_runtime_command(config)
        for compose_file in compose_files:
            services_base.extend(("-f", compose_file))
        services_base.extend(env_file_args)
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
        logger.info(f"Running command:\n    {' '.join(services_rm)}")
        subprocess.run(services_rm, env=run_env)
        if dev_mode:
            # Mirrors the plain non-web path's dev-mode build (see deploy_up):
            # without a rebuild, a co-deployed service's cached image tag keeps
            # running the stale code from its first build. Build in its own step,
            # then `up --no-build`, to dodge the `up --build` containerd
            # image-store race.
            services_build = services_base + ["build"]
            logger.info(f"Running command:\n    {' '.join(services_build)}")
            subprocess.run(services_build, env=run_env, check=True)
        services_cmd = services_base + ["up"]
        if dev_mode:
            services_cmd.append("--no-build")
        services_cmd.append("-d")
        logger.info(f"Running command:\n    {' '.join(services_cmd)}")
        subprocess.run(services_cmd, env=run_env, check=True)

    # ---- web-terminal stack (own compose project directory: project root) --
    web_cmd = get_runtime_command(config)
    web_cmd.extend(("-f", "docker-compose.web.yml"))
    web_cmd.extend(env_file_args)

    # Same stale-container preflight as the services stack above (and same
    # no-`--remove-orphans` constraint — see that comment).
    web_rm = web_cmd + ["rm", "-f"]
    logger.info(f"Running command:\n    {' '.join(web_rm)}")
    subprocess.run(web_rm, env=run_env)

    if not local_mode:
        # Registry mode only: local-only tags have no upstream to pull from,
        # and `compose pull` hard-fails (exit 1) on one -- see docstring's
        # MODE BRANCH section. This is the load-bearing guard the task exists
        # to add; never run `pull` unconditionally here again.
        pull_cmd = web_cmd + ["pull"]
        logger.info(f"Running command:\n    {' '.join(pull_cmd)}")
        subprocess.run(pull_cmd, env=run_env, check=True)

    up_cmd = web_cmd + ["up", "-d"]
    logger.info(f"Running command:\n    {' '.join(up_cmd)}")
    subprocess.run(up_cmd, env=run_env, check=True)

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
    # the host).
    # -----------------------------------------------------------------------
    enable_linger(config, run_env)
    seed_user_containers(config, env=run_env)
    run_verify_script(project_root, run_env)
    warn_if_web_stack_unreachable(config)


def deploy_down_web_terminals(
    config: dict,
    env: dict[str, str],
    env_file_args: list[str],
) -> None:
    """Tear down the web-terminal stack — the mirror of
    :func:`deploy_up_web_terminals`'s second compose invocation.

    ``deploy_down``'s services invocation can never carry
    ``docker-compose.web.yml`` in its ``-f`` list (the web file's relative
    paths are project-root-relative while the services files resolve against
    ``build/services/`` — see the WHY TWO INVOCATIONS note on
    :func:`deploy_up_web_terminals`), so the web stack needs this dedicated
    ``down``. Without it the web containers outlive every
    ``osprey deploy down`` — and because their ``container_name``s are fixed
    host-global identifiers (``<prefix>-web-<user>``, ``<prefix>-nginx``),
    the NEXT web-terminals deploy on the host, from any project, dies at
    ``up`` with a container-name Conflict instead of reconciling.

    A no-op when no rendered ``docker-compose.web.yml`` exists at the project
    root (nothing was ever deployed from here, or the render predates web
    terminals). Volumes are deliberately kept, mirroring the services
    ``down`` (no ``--volumes``): per-user claude-config/agent-data volumes
    are the durable user state ``osprey deploy decommission`` manages.

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
    if not Path("docker-compose.web.yml").exists():
        return
    down_cmd = get_runtime_command(config)
    down_cmd.extend(("-f", "docker-compose.web.yml"))
    down_cmd.extend(env_file_args)
    down_cmd.append("down")
    logger.info(f"Running command:\n    {' '.join(down_cmd)}")
    result = subprocess.run(
        down_cmd, env=runtime_env(config, env), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.warning(
            "web-terminal stack down failed (rc=%s) — its containers may still be running:\n%s",
            result.returncode,
            result.stderr,
        )
