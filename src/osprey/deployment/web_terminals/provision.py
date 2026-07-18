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

import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

from osprey.deployment.compose_generator import (
    _copy_local_framework_for_override,
    resolve_project_name,
)
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.web_terminals.artifacts import write_web_terminal_artifacts
from osprey.deployment.web_terminals.ports import effective_image_source, resolve_personas
from osprey.deployment.web_terminals.seeding import seed_user_containers
from osprey.utils.config import ConfigBuilder
from osprey.utils.dotenv import parse_dotenv_file
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")


def _resolve_persona_claude_cli_version(project_path: str) -> str | None:
    """The persona's own ``CLAUDE_CLI_VERSION`` build-arg value, or ``None`` to omit it.

    Read from ``<project_path>/config.yml``'s ``claude_code.cli_version`` —
    NEVER the facility config passed to :func:`build_persona_images`, since
    each persona project is its own independently-rendered project with its
    own pin. Unlike :func:`_resolve_claude_cli_version` (the dispatch-worker/
    facility-project path), there is deliberately no framework-default
    fallback here: when the persona's own ``config.yml`` doesn't set a pin,
    the build-arg is omitted entirely so the Dockerfile's own rendered
    ``ARG CLAUDE_CLI_VERSION=<default>`` stands, rather than silently
    overriding it with a value that specific persona project never declared.

    A missing/unreadable ``config.yml`` degrades the same way — logged and
    treated as unset — since a malformed persona catalog entry is lint's job
    to reject, not this builder's.

    :param project_path: The persona's rendered project directory.
    :return: The pinned version string, or ``None`` if unset/unreadable.
    """
    config_path = Path(project_path) / "config.yml"
    if not config_path.is_file():
        return None
    try:
        with quiet_logger(["registry", "CONFIG"]):
            persona_config = ConfigBuilder(str(config_path)).raw_config
    except Exception as exc:
        logger.warning("Could not read %s for CLAUDE_CLI_VERSION: %s", config_path, exc)
        return None
    version = persona_config.get("claude_code", {}).get("cli_version")
    return str(version) if version else None


def _persona_image_build_cmd(
    runtime: str,
    project_path: str,
    project: str,
    persona: str,
    project_label: str,
) -> list[str]:
    """Construct the ``<runtime> build`` argv that produces ``<project>-<persona>:local``.

    Mirrors :func:`_project_image_build_cmd`'s argv shape (same
    ``OSPREY_PIP_SPEC`` build-arg, same runtime/tag/``-f``/context layout)
    generalized to an arbitrary persona project directory, plus the
    ``com.osprey.project`` label Task 3.6's ``nuke`` needs to verify before
    removing this tag.

    :param runtime: Base container command (``docker`` or ``podman``).
    :param project_path: The persona's rendered project directory — both the
        build context and where its ``Dockerfile`` lives.
    :param project: The persona catalog entry's resolved ``project`` name
        (tag prefix, matching :func:`osprey.deployment.web_terminals.ports.resolve_personas`'s
        ``<project>-<persona>:local`` naming).
    :param persona: The persona catalog key (tag suffix).
    :param project_label: THIS DEPLOYMENT's project name
        (:func:`resolve_project_name` on the facility config being deployed),
        NOT the persona's own ``project`` — the ``com.osprey.project`` label
        must identify the deployment that built the image, matching every
        other container label this module sets, so a later ``nuke`` can
        verify it before removing the tag.
    :return: The full build command as an argv list.
    """
    # Lazy import to avoid an import cycle: container_lifecycle imports
    # deploy_up_web_terminals from this module at module top level, so this
    # module must not import container_lifecycle at import time. _resolve_pip_spec
    # is a generic helper shared with the dispatch-worker build path, so it
    # stays in container_lifecycle.
    from osprey.deployment.container_lifecycle import _resolve_pip_spec

    dockerfile = os.path.join(project_path, "Dockerfile")
    cmd = [
        runtime,
        "build",
        "-t",
        f"{project}-{persona}:local",
        "-f",
        dockerfile,
        "--label",
        f"com.osprey.project={project_label}",
    ]
    cli_version = _resolve_persona_claude_cli_version(project_path)
    if cli_version:
        cmd.extend(["--build-arg", f"CLAUDE_CLI_VERSION={cli_version}"])
    cmd.extend(["--build-arg", f"OSPREY_PIP_SPEC={_resolve_pip_spec()}"])
    cmd.append(project_path)
    return cmd


def _referenced_personas(config: dict, resolved_users: list[dict]) -> list[dict[str, str]]:
    """The distinct set of personas ``resolved_users`` actually reference, one build unit each.

    Multiple users sharing one persona collapse to a single entry (first-seen
    order) — the ONE-BUILD-PER-PERSONA-PER-RUN contract
    :func:`build_persona_images` relies on. Each returned dict carries the
    ``persona`` name, the ``project`` tag prefix
    :func:`osprey.deployment.web_terminals.ports.resolve_personas` already
    resolved for that entry (reused as-is, not re-derived, so this stays in
    sync with that function's own default-persona/no-persona fallback
    rules), and ``project_path`` looked up from the persona catalog — the one
    field ``resolve_personas`` doesn't carry through, since it belongs to the
    build path, not the render path.

    An entry whose ``persona`` is ``None`` (zero-migration, no persona system
    in effect for that user) or whose catalog lookup misses (a stale/bad
    reference a lenient, lifecycle-style resolution left in place) is skipped
    rather than raised — well-formedness is lint's job; this function only
    decides what to build from what already resolved. A referenced persona
    whose catalog entry is missing/empty ``project_path`` is also skipped,
    but is logged as a warning (not silent): lint (Task 2.4) is the well-
    formedness gate for a config that never runs it, so a local deploy that
    bypasses lint would otherwise fail opaquely at ``compose up`` on an
    unbuilt tag with no clue why.

    :param config: Raw deploy config (read for ``modules.web_terminals.personas``).
    :param resolved_users: :func:`osprey.deployment.web_terminals.ports.resolve_personas`'s
        output.
    :return: One ``{"persona", "project", "project_path", "build_profile"}``
        dict per distinct referenced persona, in first-seen order
        (``build_profile`` is the catalog entry's value, or ``""`` when it has
        none or a non-string one).
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    personas_raw = web_terminals.get("personas")
    personas_catalog: dict = personas_raw if isinstance(personas_raw, dict) else {}

    seen: set[str] = set()
    referenced: list[dict[str, str]] = []
    for entry in resolved_users:
        persona_name = entry.get("persona")
        if not isinstance(persona_name, str) or persona_name in seen:
            continue
        catalog_entry = personas_catalog.get(persona_name)
        if not isinstance(catalog_entry, dict):
            continue
        project_path = catalog_entry.get("project_path")
        if not isinstance(project_path, str) or not project_path:
            seen.add(persona_name)
            logger.warning(
                "Persona %r is referenced but its catalog entry has no "
                "project_path configured — skipping its local build. "
                "compose up will fail on the unbuilt %r tag.",
                persona_name,
                entry.get("image"),
            )
            continue
        seen.add(persona_name)
        # Trust resolve_personas' contract that every resolved entry carries a
        # non-empty "project" — no persona_name fallback here, which would
        # silently diverge from the tag resolve_personas itself resolved
        # (<project>-<persona>:local) if that contract were ever violated.
        project = cast(str, entry.get("project"))
        # The bundled preset auto-render builds the project from; carried here so
        # _auto_render_missing_personas need not re-walk the catalog. Normalized
        # to "" when absent or non-str, which that caller treats as "no usable
        # build_profile" identically to the raw value.
        build_profile = catalog_entry.get("build_profile")
        referenced.append(
            {
                "persona": persona_name,
                "project": project,
                "project_path": project_path,
                "build_profile": build_profile if isinstance(build_profile, str) else "",
            }
        )
    return referenced


def _auto_render_missing_personas(
    config: dict, resolved_users: list[dict], env: dict[str, str]
) -> None:
    """Render each referenced persona whose project directory is absent (local mode).

    The demo promise is ``osprey build`` + ``osprey deploy up`` = a full
    two-persona stack with no manual per-persona builds. Every persona
    :func:`build_persona_images` is about to build needs a rendered project on
    disk (its ``Dockerfile`` and ``config.yml`` are the build context); this
    fills the gap by rendering any referenced persona whose ``project_path``
    directory does not yet exist, running BEFORE :func:`build_persona_images`
    so the image build finds a complete context.

    Operates on exactly :func:`_referenced_personas`'s distinct set — the same
    personas that will be built — so render and build never diverge. For each:

    * **Directory absent**: render it with ``osprey build <project> --preset
      <build_profile> -o <parent(project_path)> --skip-deps``. ``osprey build``
      writes ``<output_dir>/<project_name>``, so rendering ``<project>`` into
      ``project_path``'s PARENT lands it exactly at ``project_path`` (the demo
      keeps ``project_path``'s basename equal to the catalog ``project``).
      ``--skip-deps`` keeps the render network-free — the persona image installs
      dependencies itself via ``OSPREY_PIP_SPEC`` at build time. A catalog entry
      with no usable ``build_profile`` cannot be rendered and raises here.
    * **Directory present but incomplete** (missing ``config.yml`` OR
      ``Dockerfile``): a partial/aborted render. Raise rather than silently
      rebuild over it — the operator must remove it (or finish it) so an
      auto-render never has to reason about a half-written tree.
    * **Directory present and complete**: a no-op. A rendered project is
      user-owned (its ``config.yml`` may carry local edits); auto-render never
      overwrites one.

    :param config: Raw deploy config, forwarded to :func:`_referenced_personas`
        (which reads ``modules.web_terminals.personas``).
    :param resolved_users: :func:`osprey.deployment.web_terminals.ports.resolve_personas`'s
        output for this deploy.
    :param env: Environment for the ``osprey build`` subprocess(es).
    :raises ValueError: A referenced persona whose project_path is a partial
        render, or one that must be rendered but whose catalog entry lacks a
        usable ``build_profile``.
    """
    for unit in _referenced_personas(config, resolved_users):
        persona_name = unit["persona"]
        project = unit["project"]
        project_path = Path(unit["project_path"])

        if project_path.exists():
            missing = [
                name for name in ("config.yml", "Dockerfile") if not (project_path / name).is_file()
            ]
            if missing:
                raise ValueError(
                    f"Persona {persona_name!r} has a partial render at "
                    f"{project_path} (missing {', '.join(missing)}). Remove that "
                    "directory and re-run `osprey deploy up` to re-render it, or "
                    "rebuild it with `osprey build`."
                )
            # Complete render is user-owned -- never overwrite it.
            continue

        build_profile = unit["build_profile"]
        if not isinstance(build_profile, str) or not build_profile:
            raise ValueError(
                f"Persona {persona_name!r} has no rendered project at "
                f"{project_path} and its catalog entry has no usable "
                "build_profile, so it cannot be auto-rendered. Set "
                f"modules.web_terminals.personas.{persona_name}.build_profile "
                "to a bundled preset name, or render the project manually with "
                "`osprey build`."
            )

        # Re-enter the CLI through the RUNNING interpreter, never a bare
        # "osprey": PATH may resolve to a different install whose bundled
        # presets diverge from (or predate) this one's.
        cmd = [
            sys.executable,
            "-m",
            "osprey",
            "build",
            project,
            "--preset",
            build_profile,
            "-o",
            str(project_path.parent),
            "--skip-deps",
        ]
        logger.key_info("Auto-rendering persona %r project at %s", persona_name, project_path)
        logger.info("Running command:\n    %s", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)


def build_persona_images(
    config: dict, resolved_users: list[dict], dev_mode: bool, env: dict
) -> None:
    """Build every REFERENCED persona's ``<project>-<persona>:local`` image (local mode only).

    Generalizes :func:`_build_project_image`'s local-build pattern (build
    context + ``-f`` Dockerfile + ``OSPREY_PIP_SPEC`` build-arg + dev-wheel
    staging) from the single dispatch-worker project image to an arbitrary
    number of persona project directories — one build per DISTINCT persona
    :func:`_referenced_personas` finds in ``resolved_users``, even when
    several users share it. :func:`_build_project_image` itself is untouched:
    the dispatch worker's ``<project>:local`` image is a different tag,
    disjoint from every ``<persona.project>-<persona>:local`` tag this
    function produces, and is built independently.

    No-op when ``modules.web_terminals.image_source`` is not ``"local"``
    (the default, ``"registry"``): a registry-mode deploy pulls every
    persona's image instead — nothing here to build. In local mode, a
    missing ``modules.web_terminals.personas`` catalog or
    ``default_persona`` raises ``ValueError``, mirroring the same guard
    :func:`osprey.deployment.web_terminals.ports.resolve_personas` enforces
    under ``strict=True`` and the lint rule it mirrors — ``osprey deploy up``
    never runs lint, so this fail-closed check must live on the build path
    too, not only on whichever render/resolve call happened to run first.

    Each build's context is the persona's own ``project_path`` (its
    Dockerfile lives there too — see :func:`_persona_image_build_cmd`), its
    ``CLAUDE_CLI_VERSION`` build-arg comes from THAT project's own
    ``config.yml`` (never the facility config — see
    :func:`_resolve_persona_claude_cli_version`), and under ``dev_mode`` a
    locally-built wheel is staged into that same ``project_path`` and removed
    afterward, exactly mirroring :func:`_build_project_image`'s per-context
    dev-wheel convention (so a later non-dev persona build's wheel-drop
    branch, which fires on any ``*.whl`` in ITS OWN context, never sees a
    stale wheel from this run).

    :param config: Raw deploy config (the facility config being deployed —
        read for ``modules.web_terminals`` and for
        :func:`resolve_project_name`'s ``com.osprey.project`` label value).
    :param resolved_users: :func:`osprey.deployment.web_terminals.ports.resolve_personas`'s
        output for this deploy.
    :param dev_mode: Whether ``--dev`` was passed (stage a local wheel into
        each persona's build context).
    :param env: Environment for the build subprocesses.
    :raises ValueError: Local mode with no ``personas`` catalog or no
        ``default_persona`` configured.
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) != "local":
        return  # registry mode pulls every persona's image; nothing to build

    personas_raw = web_terminals.get("personas")
    personas_catalog: dict = personas_raw if isinstance(personas_raw, dict) else {}
    default_persona = web_terminals.get("default_persona")
    if not personas_catalog or not isinstance(default_persona, str) or not default_persona:
        raise ValueError(
            "modules.web_terminals.image_source: local requires both a "
            "modules.web_terminals.personas catalog and default_persona to be "
            "configured"
        )

    referenced = _referenced_personas(config, resolved_users)
    if not referenced:
        return

    runtime = get_runtime_command(config)[0]
    project_label = resolve_project_name(config)

    for unit in referenced:
        persona_name = unit["persona"]
        project = unit["project"]
        project_path = unit["project_path"]

        staged_wheels: list[Path] = []
        if dev_mode:
            before = set(Path(project_path).glob("*.whl"))
            _copy_local_framework_for_override(project_path)
            staged_wheels = list(set(Path(project_path).glob("*.whl")) - before)

        try:
            cmd = _persona_image_build_cmd(
                runtime, project_path, project, persona_name, project_label
            )
            logger.key_info("Building persona image %s-%s:local:", project, persona_name)
            logger.info("Running command:\n    %s", " ".join(cmd))
            subprocess.run(cmd, env=env, check=True)
        finally:
            for whl in staged_wheels:
                try:
                    whl.unlink()
                except OSError:
                    logger.warning("Could not remove staged dev wheel %s", whl)


def _copy_named_env_var(var_name: str | None, source: dict[str, str], dest: dict[str, str]) -> None:
    """Copy ``source[var_name]`` into ``dest[var_name]`` iff both are present.

    ``var_name`` is itself a config-declared *name* (e.g. ``llm.api_key_env_var``
    resolves to ``"CBORG_API_KEY"``), not a literal value — this is the
    ``${env.${config.X.Y_env_var}}`` indirection the ``.gitlab-ci.yml`` template
    (see :func:`_build_env_production_subset`) uses for every secret it
    assembles. A ``var_name`` that is unset (module misconfigured) or absent
    from ``source`` (operator never set it) is silently skipped — never
    fabricated, matching every other var-presence check in this module.
    """
    if not var_name or var_name not in source:
        return
    dest[var_name] = source[var_name]


def _build_env_production_subset(config: dict, dotenv: dict[str, str]) -> dict[str, str]:
    """Build the module-conditional CI subset for a local-mode ``.env.production``.

    Mirrors the shipped facility-scaffolding CI template's own
    ``.env.production`` assembly step (``osprey-build-deploy`` skill's
    ``templates/core/.gitlab-ci.yml``, ``osprey-build`` job, lines 80-98) —
    the same var set that job composes from masked CI variables, here sourced
    from the operator's local ``.env`` instead of ``$CI_*`` secrets:

    - ``llm.api_key_env_var`` — the LLM provider key, unconditional.
    - ``modules.olog.{username,password}_env_var`` — only if ``modules.olog.enabled``.
    - ``modules.wiki_search.token_env_var`` — only if ``modules.wiki_search.enabled``.
    - ``modules.event_dispatcher.token_env_var`` — only if
      ``modules.event_dispatcher.enabled``. Deliberately NOT
      ``sidecar_token_env_var`` — see the exclusion list below.
    - ``ARIEL_DSN`` — only if ``modules.ariel.enabled``, from
      ``modules.ariel.dsn`` directly. Unlike every other entry above, the CI
      template substitutes this value straight from facility-config
      (``ARIEL_DSN=${config.modules.ariel.dsn}``, no ``${env.*}``
      indirection) because the DSN is itself a literal config value, not the
      *name* of an env var holding one — so it is read from ``config``, not
      ``dotenv``.
    - ``TZ`` — always, from ``facility.timezone`` (default ``"UTC"``, matching
      the facility-config schema's own documented default), likewise a
      literal config value with no ``${env.*}`` indirection in the template.

    NEVER included, by construction (this function never reads these config
    paths at all): the CI/registry provider token (``ci.token_env_var`` /
    legacy ``gitlab.token_env_var``), ``registry.token_env_var``, the
    dispatcher's ``sidecar_token_env_var``, or any ``registry.external_projects``
    entry's ``token_env_var``. Those all gate build/push/CI-registry access,
    not anything the running containers need, and none of them belongs in a
    per-deploy runtime secrets file. This is the security spec for this
    function: a var absent from the enumerated list above can never appear in
    the returned dict, regardless of what the input ``.env`` contains.

    :param config: Raw deploy config (facility fields merged in — see
        ``modules.web_terminals.image_source`` in :func:`ensure_env_production`).
    :param dotenv: The operator's ``.env``, already parsed via
        :func:`osprey.utils.dotenv.parse_dotenv_file`.
    :return: The subset to write into ``.env.production``, in stable
        (insertion) order.
    """
    subset: dict[str, str] = {}

    llm = config.get("llm") or {}
    _copy_named_env_var(llm.get("api_key_env_var"), dotenv, subset)

    modules = config.get("modules") or {}

    olog = modules.get("olog") or {}
    if olog.get("enabled"):
        _copy_named_env_var(olog.get("username_env_var"), dotenv, subset)
        _copy_named_env_var(olog.get("password_env_var"), dotenv, subset)

    wiki_search = modules.get("wiki_search") or {}
    if wiki_search.get("enabled"):
        _copy_named_env_var(wiki_search.get("token_env_var"), dotenv, subset)

    event_dispatcher = modules.get("event_dispatcher") or {}
    if event_dispatcher.get("enabled"):
        # NEVER sidecar_token_env_var -- see the exclusion list above.
        _copy_named_env_var(event_dispatcher.get("token_env_var"), dotenv, subset)

    ariel = modules.get("ariel") or {}
    if ariel.get("enabled"):
        dsn = ariel.get("dsn")
        if dsn:
            subset["ARIEL_DSN"] = str(dsn)

    facility = config.get("facility") or {}
    subset["TZ"] = str(facility.get("timezone") or "UTC")

    return subset


def ensure_env_production(config: dict, project_root: str | Path) -> Path:
    """Ensure ``<project_root>/.env.production`` exists, generating it when possible.

    ``docker-compose.web.yml`` (see :func:`deploy_up_web_terminals`) declares
    ``env_file: .env.production`` unconditionally, so compose hard-fails before
    a single container starts if that file is missing. This resolves it up
    front, with different rules per ``modules.web_terminals.image_source``
    (default ``"registry"``):

    - **Already present** (either mode): returned as-is, untouched. This is
      always checked first, so an operator-authored or previously-generated
      file is never clobbered.
    - **Registry mode, absent**: raises. Registry-mode deploys expect a CI
      pipeline (the ``osprey-build-deploy`` skill's ``.gitlab-ci.yml``,
      ``osprey-build`` job) to have produced this file already — this
      function only exists-checks in that mode, it never generates, because
      there is no local ``.env`` this system is licensed to treat as the
      authoritative source of CI-provisioned secrets.
    - **Local mode, absent, ``.env`` present**: generated via
      :func:`_build_env_production_subset` (the module-conditional CI subset)
      and written with mode ``0600`` from the moment the file is created — the
      same permission convention :func:`_ensure_service_tokens` uses for
      minted tokens.
    - **Local mode, absent, ``.env`` absent too**: raises, before any compose
      invocation — there is nothing to generate from and no file to fall back
      on.

    Every secret value this generates comes solely from the parsed ``.env``
    (never the ambient process/shell environment, unlike
    :func:`_ensure_service_tokens`'s ``_effective_value``): ``.env`` is the
    canonical local secrets store for this deploy, so reading only from it
    keeps the generated file deterministic and independent of whatever
    happens to be exported in the caller's shell.

    :param config: Raw deploy config.
    :param project_root: Project root directory; ``.env.production`` and
        ``.env`` are both resolved relative to it.
    :return: Path to the existing or newly-generated ``.env.production``.
    :raises RuntimeError: per the absent-file rules above, with an actionable
        message naming the missing file(s) and how to resolve it.
    """
    root = Path(project_root)
    env_production_path = root / ".env.production"
    if env_production_path.is_file():
        return env_production_path

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) != "local":
        raise RuntimeError(
            f"{env_production_path} not found. Registry-mode web-terminal deploys "
            "(modules.web_terminals.image_source: registry, the default) expect "
            "this file to be produced by CI (see the osprey-build-deploy skill's "
            ".gitlab-ci.yml osprey-build job) and shipped alongside the pulled "
            "image context -- osprey deploy up does not generate it in this mode. "
            "Either supply .env.production directly, or set "
            "modules.web_terminals.image_source: local to generate it from .env."
        )

    env_path = root / ".env"
    if not env_path.is_file():
        raise RuntimeError(
            f"Neither {env_production_path} nor {env_path} was found. Local-mode "
            "web-terminal deploys (modules.web_terminals.image_source: local) need "
            "one of them: create .env.production directly, or create .env so "
            "osprey deploy up can derive .env.production's module-conditional CI "
            "subset from it."
        )

    dotenv = parse_dotenv_file(env_path)
    subset = _build_env_production_subset(config, dotenv)

    lines = "".join(f"{key}={value}\n" for key, value in subset.items())
    # Create with mode 0600 from the FIRST byte on disk, not write-then-chmod:
    # write_text() would create the file at the process umask (typically
    # 0644) and write every secret before a later os.chmod tightened
    # permissions, leaving a window on a multi-user host where a co-tenant
    # could read it. os.open with O_CREAT + an explicit mode is atomic --
    # there is no instant the file exists at a wider mode.
    fd = os.open(env_production_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(lines)
    # Belt-and-suspenders: also covers the file already existing (e.g. a
    # leftover from a prior run) with a wider mode O_CREAT wouldn't have
    # reset on its own.
    os.chmod(env_production_path, 0o600)

    logger.key_info(
        "Generated %s from %s (mode 0600): %s",
        env_production_path,
        env_path,
        ", ".join(subset),
    )

    return env_production_path


def _enable_linger(config: dict, run_env: dict[str, str]) -> None:
    """Enable rootless-podman linger so web-terminal containers survive logout.

    Rootless podman runs containers under the deploy user's ``systemd --user``
    session, which systemd-logind tears down (along with everything under it)
    the moment that user's last login session ends. ``loginctl enable-linger
    <user>`` asks logind to keep the session alive across logout and reboot
    instead, which is what makes a rootless-podman web-terminal deploy survive
    the operator closing their SSH session. Docker containers run under the
    docker daemon rather than a per-user systemd session, so there is nothing
    to enable there.

    This is a best-effort persistence step, not a deploy precondition: every
    way it can fail to apply (wrong runtime, no ``loginctl`` on ``PATH``, no
    systemd, no permission) is logged and swallowed rather than raised, so a
    host that can't support linger still completes its deploy.

    :param config: Raw deploy config, used only to detect podman vs. docker
        via :func:`get_runtime_command`.
    :param run_env: The ``COMPOSE_PROJECT_NAME``-pinned environment the caller
        already built via :func:`runtime_helper.runtime_env`; reused here so
        the ``loginctl`` subprocess sees the same ``PATH`` as the compose
        calls around it.
    """
    if get_runtime_command(config)[0] != "podman":
        return  # linger is a rootless-podman/systemd concept; docker has no analog

    if shutil.which("loginctl") is None:
        logger.warning("loginctl not found on PATH — skipping podman linger enable")
        return

    try:
        deploy_user = getpass.getuser()
    except (KeyError, OSError) as exc:
        # getuser() falls back to pwd.getpwuid(os.getuid()) when USER/LOGNAME
        # etc. are all unset, which raises KeyError (3.12 and earlier) or
        # OSError (3.13+) for a uid with no passwd entry -- e.g. an LDAP/NSS
        # user under a stripped-env systemd/cron context. Best-effort means
        # best-effort: give up on linger rather than aborting the deploy.
        logger.warning(f"Could not determine deploy user for linger: {exc}")
        return

    try:
        status = subprocess.run(
            ["loginctl", "show-user", deploy_user, "--property=Linger"],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode == 0 and status.stdout.strip() == "Linger=yes":
            logger.debug(f"Linger already enabled for {deploy_user} — nothing to do")
            return
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not check linger status for {deploy_user}: {exc}")
        # Fall through -- a failed status check doesn't mean enabling would fail.

    try:
        enable = subprocess.run(
            ["loginctl", "enable-linger", deploy_user],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if enable.returncode == 0:
            logger.info(f"Enabled systemd linger for {deploy_user} (podman persistence)")
        else:
            logger.warning(
                f"loginctl enable-linger {deploy_user} failed (exit {enable.returncode}): "
                f"{enable.stderr.strip()}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not enable linger for {deploy_user}: {exc}")


def _run_verify_script(project_root: str, run_env: dict[str, str]) -> None:
    """Best-effort, advisory post-up smoke check via the scaffolded ``scripts/verify.sh``.

    An ``osprey-build-deploy``-scaffolded project ships ``scripts/verify.sh``
    (see the ``osprey-build-deploy`` skill's ``templates/core/scripts/
    verify.sh``): a health-check script parameterized per-facility with a
    probe for each enabled module. Historically it was operator-run-by-hand
    only; this makes ``osprey deploy up`` run it automatically as the last
    step of the post-up hook, once ``compose up -d`` has already succeeded
    and containers are running, so an operator gets an immediate health
    signal without a separate manual step.

    Silently skipped (no log line at all) when ``<project_root>/scripts/
    verify.sh`` doesn't exist — an older project scaffolded before this file
    existed, or a non-``osprey-build-deploy`` project, must deploy exactly as
    before.

    The script's own convention (see its header) is to ALWAYS exit 0 —
    verification is advisory, never deploy-blocking — but this runs it via
    ``bash`` (rather than executing the path directly) and ignores whatever
    exit code it reports either way, so a site-customized copy that doesn't
    honor that convention still can never fail ``osprey deploy up``: this
    step runs after compose already reported success, so a nonzero exit is a
    signal to look closer, not evidence the deploy failed. Output streams
    straight to the operator's terminal (stdout/stderr are inherited, not
    captured) exactly like every other compose subprocess call in this
    module, so the health report appears live rather than being buffered and
    dumped at the end.

    :param project_root: The project root whose ``scripts/verify.sh`` (if
        any) to run; also the script's working directory, so its own
        ``./scripts/...``-relative assumptions resolve the same as when an
        operator runs it by hand from the project root.
    :param run_env: Environment for the subprocess — the same
        ``COMPOSE_PROJECT_NAME``-pinned env the compose calls in this module
        use, so any ``${COMPOSE_PROJECT_NAME}``-derived container name the
        script probes matches what compose actually named.
    """
    verify_path = Path(project_root) / "scripts" / "verify.sh"
    if not verify_path.is_file():
        return

    logger.key_info("Running post-up smoke check: %s", verify_path)
    try:
        result = subprocess.run(["bash", str(verify_path)], cwd=project_root, env=run_env)
    except OSError as exc:
        logger.warning("Could not run %s: %s", verify_path, exc)
        return

    if result.returncode == 0:
        logger.key_info("%s completed (exit 0)", verify_path)
    else:
        logger.warning(
            "%s exited %s -- advisory only, this does NOT fail the deploy. "
            "Review the output above.",
            verify_path,
            result.returncode,
        )


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
      called with :func:`osprey.deployment.web_terminals.ports.resolve_personas`'s
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
    so a no-op second run recreates zero containers. Under ``dev_mode`` the
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

    # ensure_env_production BEFORE any compose invocation, in both modes: a
    # missing/ungeneratable .env.production would otherwise surface as an
    # opaque compose "env file not found" failure only once `up` runs.
    ensure_env_production(config, project_root)

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
        _auto_render_missing_personas(config, resolved_users, env)
        build_persona_images(config, resolved_users, dev_mode, env)

    run_env = runtime_env(config, env)

    # ---- backend services (own compose project directory: build/services/) --
    # Skipped when no real service is deployed -- see docstring for why
    # `up` on the network-only top-level file alone would fail outright.
    if config.get("deployed_services"):
        services_base = get_runtime_command(config)
        for compose_file in compose_files:
            services_base.extend(("-f", compose_file))
        services_base.extend(env_file_args)
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

    # -----------------------------------------------------------------------
    # POST-UP HOOK — web-terminal reconcile complete (`compose up -d`
    # succeeded, containers running). Linger runs first so a rootless-podman
    # host survives the deploy operator's session ending before seeding's
    # (longer-running, per-user) exec calls are attempted; seeding itself
    # tolerates per-user failures and logs rather than aborting the deploy.
    # verify.sh runs last, once containers are seeded, so its health probes
    # see the fully-reconciled state -- see _run_verify_script for why its
    # result is advisory only and never raises from here.
    # -----------------------------------------------------------------------
    _enable_linger(config, run_env)
    seed_user_containers(config, env=run_env)
    _run_verify_script(project_root, run_env)
