"""Per-persona local image builds for multi-user web-terminal deployments.

``image_source: local`` deploys build one ``<project>-<persona>:local`` image
per referenced persona before any compose invocation (rendering a missing
persona project on demand via ``osprey build``). Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`;
registry-mode deploys never enter this module.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from osprey.deployment.compose_generator import (
    _copy_local_framework_for_override,
    _staged_dev_artifact_paths,
    resolve_project_name,
)
from osprey.deployment.runtime_helper import get_runtime_command
from osprey.deployment.web_terminals.personas import effective_image_source
from osprey.utils.config import ConfigBuilder
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
    dev_mode: bool = False,
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
        (tag prefix, matching :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
        ``<project>-<persona>:local`` naming).
    :param persona: The persona catalog key (tag suffix).
    :param dev_mode: Whether ``--dev`` was passed (adds an ``OSPREY_DEV=1``
        build-arg, mirroring :func:`_project_image_build_cmd`'s dev path).
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
    if dev_mode:
        cmd.extend(["--build-arg", "OSPREY_DEV=1"])
    cmd.append(project_path)
    return cmd


def _referenced_personas(config: dict, resolved_users: list[dict]) -> list[dict[str, str]]:
    """The distinct set of personas ``resolved_users`` actually reference, one build unit each.

    Multiple users sharing one persona collapse to a single entry (first-seen
    order) — the ONE-BUILD-PER-PERSONA-PER-RUN contract
    :func:`build_persona_images` relies on. Each returned dict carries the
    ``persona`` name, the ``project`` tag prefix
    :func:`osprey.deployment.web_terminals.personas.resolve_personas` already
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
    :param resolved_users: :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
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
        # auto_render_missing_personas need not re-walk the catalog. Normalized
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


def auto_render_missing_personas(
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
    :param resolved_users: :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
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
    :func:`osprey.deployment.web_terminals.personas.resolve_personas` enforces
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
    stale wheel from this run). The ``OSPREY_DEV=1`` build-arg is passed only
    when that persona's staging actually succeeded — a failed wheel build
    keeps the pinned install fail-loud.

    :param config: Raw deploy config (the facility config being deployed —
        read for ``modules.web_terminals`` and for
        :func:`resolve_project_name`'s ``com.osprey.project`` label value).
    :param resolved_users: :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
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

        # OSPREY_DEV=1 (the pin-relaxing build arg) is passed only when the
        # wheel actually landed in THIS persona's context: on a failed
        # build/staging the Dockerfile must keep its fail-loud pinned install
        # rather than silently falling back to the latest published release
        # (mirroring _build_project_image's fail-closed dev path).
        staged_artifacts: list[Path] = []
        wheel_staged = False
        if dev_mode:
            before = _staged_dev_artifact_paths(project_path)
            wheel_staged = bool(_copy_local_framework_for_override(project_path))
            staged_artifacts = sorted(_staged_dev_artifact_paths(project_path) - before)
            if not wheel_staged:
                logger.warning(
                    "Dev-wheel staging failed for persona %r; building without "
                    "OSPREY_DEV so the Dockerfile keeps its pinned "
                    "osprey-framework install.",
                    persona_name,
                )

        try:
            cmd = _persona_image_build_cmd(
                runtime,
                project_path,
                project,
                persona_name,
                project_label,
                dev_mode and wheel_staged,
            )
            logger.key_info("Building persona image %s-%s:local:", project, persona_name)
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
