"""Per-persona local image builds for multi-user web-terminal deployments.

``image_source: local`` deploys build one ``<project>-<persona>:local`` image
per referenced persona before any compose invocation (rendering a missing
persona project on demand via ``osprey build``). Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`;
registry-mode deploys never enter this module.
"""

import os
import posixpath
import subprocess
import sys
from pathlib import Path
from typing import cast

from osprey.build.claude_code_resolver import load_provider_spec
from osprey.deployment.compose_generator import (
    _copy_local_framework_for_override,
    resolve_project_name,
)
from osprey.deployment.runtime_helper import get_runtime_command
from osprey.deployment.web_terminals.personas import effective_image_source
from osprey.deployment.wheel_build import _staged_dev_artifact_paths
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
    cmd.extend(["--build-arg", f"OSPREY_PIP_SPEC={_resolve_pip_spec(dev_mode=dev_mode)}"])
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
        # The persona delta auto-render builds the project from; carried here so
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


# A catalog `build_profile` names a profile FILE. Whether the operator instead
# wrote a bundled preset NAME is told from the STRING alone — never by probing
# the filesystem — so a mistyped path is reported as a path rather than silently
# retried as a preset name (and vice versa).
_PROFILE_FILE_SUFFIXES = (".yml", ".yaml")


def _is_profile_file_reference(build_profile: str) -> bool:
    """Whether ``build_profile`` reads as a path rather than a bundled preset name.

    Bundled preset names are bare slugs (``control-assistant``), so any ``/``
    or any ``.yml``/``.yaml`` suffix means a path. Only ``/`` counts as a
    separator: catalog values are configuration, written posix-style, and are
    read on the same machine that renders the personas. Used only to pick which
    rejection an unusable value earns — a preset name and a path that leaves the
    profile are different mistakes with different fixes.
    """
    return "/" in build_profile or build_profile.endswith(_PROFILE_FILE_SUFFIXES)


def persona_build_profile_shape_problem(build_profile: str) -> str | None:
    """Why ``build_profile`` can never name a persona delta, or ``None`` if it can.

    The single shape rule for a local-mode catalog entry, shared by the two
    places that enforce it: ``osprey lint`` (the well-formedness gate) and
    :func:`_resolve_persona_profile` (the deploy path, which ``osprey deploy up``
    reaches without ever running lint). Sharing the predicate is what keeps the
    two from disagreeing — a config that lints clean and then fails the deploy is
    worse than either verdict alone, because the gate promised the problem away.

    Decided from the STRING alone, with no filesystem access at all: lint holds
    only a config, not the deployed project, so it can never learn where the
    profile root is or whether a file exists there. That bounds what this can
    answer — *shape*, not existence — and the deploy path checks existence
    separately, where it has a root to check against. Path normalization is
    likewise lexical (``os.path.normpath``, never ``Path.resolve``), so the
    verdict does not depend on which machine lint runs on.

    Returns:
        A clause naming what is wrong, phrased to follow "its ``build_profile``"
        in the caller's own sentence, or ``None`` when the shape is usable. The
        caller supplies the remedy, since only the deploy path can name the
        absolute file the value has to resolve to.
    """
    from osprey.cli.profile_root import PERSONA_DIRNAME

    if not _is_profile_file_reference(build_profile):
        return (
            f"is the bundled preset name {build_profile!r}. A persona project is never "
            "rendered from a preset: it is rendered from a delta over the profile the "
            "deployed project was built from, which is what gives it that profile's data "
            "tree, secrets and conventions."
        )

    if posixpath.isabs(build_profile):
        return (
            f"is the absolute path {build_profile!r}. A persona delta belongs to the "
            "deployed project's own profile and is named relative to it."
        )

    # Lexical normalization, so `personas/../ops.yml` is judged on where it
    # actually lands rather than on the directory name it passes through.
    if posixpath.dirname(posixpath.normpath(build_profile)) != PERSONA_DIRNAME:
        return (
            f"is {build_profile!r}, which is not a direct child of the profile's "
            f"{PERSONA_DIRNAME}/ directory. A persona file is read as a delta only when "
            f"its parent directory IS {PERSONA_DIRNAME}/, so anything else would build a "
            "hollow project from the delta alone, or from a profile this deployment does "
            "not own."
        )

    return None


def _parent_profile_root(project_root: Path) -> Path | None:
    """The profile ROOT of the project being deployed, or ``None`` if unknowable.

    Located through the deployed project's own manifest
    (``build_args.profile_path_abs``, the profile path ``osprey build``
    resolved) and then through root discovery, so a project that was itself
    built from a persona delta reports the root one level above ``personas/``
    rather than the delta's own directory.

    ``None`` means the question cannot be answered — a manifest that records no
    profile, or one naming a profile that has since moved or been deleted. The
    caller turns that into its own error: there is no fallback anchor, because
    guessing one is how a persona ends up rendered from a profile that is not
    this deployment's. The recorded profile has to still BE there, too: a
    directory that no longer holds a profile would otherwise be reported as the
    root, and every persona under it as an individually missing file — one
    problem told as several, none of them the real one — which is what
    ``require_profile_file=True`` asks of the shared resolver.

    Every way the answer is no collapses to ``None`` here, deliberately: the
    caller's error says the same thing for all of them, and a persona delta
    whose root profile is gone leaves no directory we can honestly call this
    deployment's profile root either.
    """
    from osprey.cli.profile_root import resolve_project_profile

    return resolve_project_profile(project_root, require_profile_file=True).root


def _persona_delta_remedy(persona_name: str, profile_root: Path) -> str:
    """The one sentence every unusable-``build_profile`` error ends with.

    Names both spellings the operator needs — the catalog value to write and the
    file it has to resolve to — from one place, so the several ways an entry can
    be wrong cannot end up recommending different fixes. Ends by naming
    ``/osprey-build-interview``, because the operator most likely to read this is
    one whose project predates the persona-delta layout: they have a variant
    build in some older shape and need it converted, which is a bigger job than
    editing one catalog value.
    """
    from osprey.cli.profile_root import PERSONA_DIRNAME

    delta = profile_root / PERSONA_DIRNAME / f"{persona_name}.yml"
    return (
        f"Set modules.web_terminals.personas.{persona_name}.build_profile to "
        f"'{PERSONA_DIRNAME}/{persona_name}.yml' — the delta at {delta} — which is what "
        "`osprey profile new` writes for every persona in the catalog. If this "
        "deployment has no such delta because its variant build predates the layout, "
        "run /osprey-build-interview: it converts an existing variant into a persona "
        "delta over the profile this project is built from."
    )


def _resolve_persona_profile(build_profile: str, persona_name: str, profile_root: Path) -> Path:
    """Resolve a catalog ``build_profile`` to the persona delta it must name.

    A persona project is rendered from a delta over the very profile this
    deployment was built from — that is what gives the persona the same data
    tree, the same ``.env`` secrets and the same convention artifacts as its
    host. So the catalog value is accepted in exactly one shape: a relative path
    to a direct child of ``<profile root>/personas/``, which is what ``osprey
    profile new`` writes (``personas/<persona>.yml``).

    The shape rule itself lives in :func:`persona_build_profile_shape_problem`,
    which ``osprey lint`` enforces on the same values — one predicate, so a
    config cannot lint clean and then fail here. What this adds is the half lint
    cannot do: anchoring the accepted relative path on the profile root and
    checking that the file is actually there.

    :param build_profile: The catalog entry's value, non-empty.
    :param persona_name: Catalog key, named in every error.
    :param profile_root: This deployment's profile root
        (:func:`_parent_profile_root`).
    :return: The absolute path to the persona delta.
    :raises ValueError: For any value that is not an existing delta inside
        ``<profile root>/personas/``, naming both the value and the file the
        operator has to point at instead.
    """
    remedy = _persona_delta_remedy(persona_name, profile_root)

    problem = persona_build_profile_shape_problem(build_profile)
    if problem is not None:
        raise ValueError(
            f"Persona {persona_name!r} cannot be auto-rendered: its build_profile "
            f"{problem} {remedy}"
        )

    # Shape is settled above, so what is left is the filesystem half. The
    # `.resolve()` is both for the message (an operator chasing a missing file
    # wants the absolute path, not the relative spelling they already wrote) and
    # for the anchoring check below, which needs the real target.
    candidate = (profile_root / build_profile).resolve()
    if not candidate.is_file():
        raise ValueError(
            f"Persona {persona_name!r} names the build_profile {build_profile!r}, but no "
            f"file exists at {candidate}. " + remedy
        )

    # The property everything else rests on: the file handed to `osprey build`
    # must anchor BACK at the profile root it was resolved from. The shape rule
    # is lexical — it is shared with lint, which has no filesystem — and cannot
    # see a symlink, while `.resolve()` follows one. So a delta symlinked out of
    # `personas/`, or a symlinked `personas/` directory, would otherwise pass
    # every check above and then be read by root discovery somewhere else
    # entirely: as a standalone profile (no data tree, no `.env`, no
    # conventions), or as a delta over a DIFFERENT facility's profile. Both
    # build a hollow or wrong persona and report nothing, which is the one
    # outcome worse than any rejection above.
    #
    # Asked of root discovery rather than by comparing directories, because the
    # danger is the wrong ROOT, not the symlink: a symlinked `personas/` makes
    # candidate and the expected directory resolve to the SAME place, so a
    # containment check on the parent passes while the root is still wrong.
    from osprey.cli.profile_root import PERSONA_DIRNAME, resolve_profile_root
    from osprey.errors import BuildProfileError

    try:
        anchored_root, is_delta = resolve_profile_root(candidate)
    except BuildProfileError as e:
        # The delta sits in a `personas/` directory with no profile beside it.
        # That error already names the file that has to exist, and says it
        # better than a second sentence here could.
        raise ValueError(
            f"Persona {persona_name!r} cannot be auto-rendered from {build_profile!r}: {e}"
        ) from e

    if not is_delta or anchored_root != profile_root:
        landed = (
            f"a delta over the profile at {anchored_root}"
            if is_delta
            else f"a standalone profile at {anchored_root}"
        )
        raise ValueError(
            f"Persona {persona_name!r} names the build_profile {build_profile!r}, which "
            f"resolves to {candidate} — {landed}, not a delta over the profile this "
            f"deployment was built from ({profile_root}). A symlinked delta, or a "
            f"symlinked {PERSONA_DIRNAME}/ directory, does this: the persona would be "
            "built without that profile's data tree, secrets and conventions, or over "
            f"somebody else's. Keep the delta a real file inside {profile_root / PERSONA_DIRNAME}. "
            + remedy
        )
    return candidate


def _check_existing_render(persona_name: str, project_path: Path) -> None:
    """Accept a persona's already-rendered project, or say why it is unusable.

    A render that is already on disk is the operator's — it is never overwritten
    and never re-rendered — so this is the only chance to notice that it will
    not actually come up, and both ways it can fail are quiet until the
    container is already running.

    A directory missing ``config.yml`` or ``Dockerfile`` is a partial or aborted
    render: it is refused rather than rebuilt over, so an auto-render never has
    to reason about a half-written tree.

    A complete render still has its model selection checked. The web terminal
    runs the same resolver strict at startup, so a model the profile has since
    moved away from would ship a container that crash-loops behind the reverse
    proxy and surface only as a 502.

    Raises:
        ValueError: Naming the persona, the directory, and the way out.
    """
    missing = [name for name in ("config.yml", "Dockerfile") if not (project_path / name).is_file()]
    if missing:
        raise ValueError(
            f"Persona {persona_name!r} has a partial render at "
            f"{project_path} (missing {', '.join(missing)}). Remove that "
            "directory and re-run `osprey deploy up` to re-render it, or "
            "rebuild it with `osprey build`."
        )

    try:
        load_provider_spec(project_path)
    except ValueError as e:
        raise ValueError(
            f"Persona {persona_name!r} render at {project_path} has a "
            f"model configuration its provider cannot serve:\n  {e}\n"
            f"Fix {project_path / 'config.yml'}, or remove that "
            "directory and re-run `osprey deploy up` to re-render it "
            "from this deployment's profile."
        ) from e


def auto_render_missing_personas(
    config: dict,
    resolved_users: list[dict],
    env: dict[str, str],
    project_root: Path,
) -> None:
    """Render each referenced persona whose project directory is absent (local mode).

    The demo promise is ``osprey build`` + ``osprey deploy up`` = a full
    two-persona stack with no manual per-persona builds. Every persona
    :func:`build_persona_images` is about to build needs a rendered project on
    disk (its ``Dockerfile`` and ``config.yml`` are the build context); this
    fills the gap by rendering any referenced persona whose ``project_path``
    directory does not yet exist, running BEFORE :func:`build_persona_images`
    so the image build finds a complete context.

    Every persona is rendered from a delta inside THIS deployment's own profile
    — ``<profile root>/personas/<persona>.yml``, located through the deployed
    project's manifest (:func:`_parent_profile_root`) and validated by
    :func:`_resolve_persona_profile`. That single anchor is the whole point:
    the delta merges over the same profile the deployed project was built from,
    so every persona in the stack shares its data tree, its convention
    artifacts, and — because the build derives the project ``.env`` from the
    profile root's ``.env``, and root discovery puts a delta's root one level
    above ``personas/`` — its secrets. Nothing about the parent's *build
    invocation* is replayed here: the profile is the source of truth, so what
    the operator changed is already in the file the delta merges over.

    The rendered profile is always an existing file passed positionally, so a
    persona subprocess build never takes the ``--preset`` path and therefore
    never materializes a profile of its own. A persona profile is created by
    ``osprey profile new``, beside its host, and by nothing else.

    Operates on exactly :func:`_referenced_personas`'s distinct set — the same
    personas that will be built — so render and build never diverge. For each:

    * **Directory absent**: render it with ``osprey build <project>
      <profile root>/personas/<persona>.yml -o <parent(project_path)>
      --skip-deps``. ``osprey build`` writes ``<output_dir>/<project_name>``, so
      rendering ``<project>`` into ``project_path``'s PARENT lands it exactly at
      ``project_path`` (the demo keeps ``project_path``'s basename equal to the
      catalog ``project``). ``--skip-deps`` keeps the render network-free — the
      persona image installs dependencies itself via ``OSPREY_PIP_SPEC`` at
      build time. A catalog entry with no usable ``build_profile``, or one whose
      value is not a delta in that directory, cannot be rendered and raises
      here.
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
    :param project_root: The deployed project's root — the directory holding
        the ``.osprey-manifest.json`` that records which profile this deployment
        was built from. Required, and only consulted when something actually has
        to be rendered.
    :raises ValueError: A referenced persona whose project_path is a partial
        render or whose existing render has an unservable model; or, on the
        render path, a deployment whose profile cannot be located, a catalog
        entry with no ``build_profile``, and any ``build_profile`` that is not
        an existing delta inside ``<profile root>/personas/``.
    """
    # Resolved on the first persona that actually needs rendering: a deploy
    # whose persona projects are all present must not fail over a profile it
    # never has to read.
    profile_root: Path | None = None
    for unit in _referenced_personas(config, resolved_users):
        persona_name = unit["persona"]
        project = unit["project"]
        project_path = Path(unit["project_path"])

        if project_path.exists():
            _check_existing_render(persona_name, project_path)
            continue

        if profile_root is None:
            profile_root = _parent_profile_root(project_root)
        if profile_root is None:
            raise ValueError(
                f"Persona {persona_name!r} has no rendered project at {project_path} and "
                "cannot be auto-rendered: the profile this deployment was built from "
                f"cannot be located, so there is nowhere to read its persona delta from. "
                f"The manifest in {project_root} records no profile path, or the profile "
                "it names has moved. Rebuild this project from its profile — `osprey "
                "build <name> <path-to-profile>/profile.yml` — and re-run `osprey deploy "
                "up`, or render each persona project yourself with `osprey build`."
            )

        build_profile = unit["build_profile"]
        if not isinstance(build_profile, str) or not build_profile:
            raise ValueError(
                f"Persona {persona_name!r} has no rendered project at {project_path} and "
                "its catalog entry has no build_profile, so it cannot be auto-rendered. "
                + _persona_delta_remedy(persona_name, profile_root)
            )

        # Always an existing profile FILE, passed positionally: `--preset` is
        # what would make this subprocess materialize a profile of its own.
        profile_path = _resolve_persona_profile(build_profile, persona_name, profile_root)

        # Re-enter the CLI through the RUNNING interpreter, never a bare
        # "osprey": PATH may resolve to a different install whose build
        # behaviour diverges from (or predates) this one's.
        cmd = [
            sys.executable,
            "-m",
            "osprey",
            "build",
            project,
            str(profile_path),
            "-o",
            str(project_path.parent),
            "--skip-deps",
        ]
        logger.key_info("Auto-rendering persona %r project at %s", persona_name, project_path)
        logger.info("  ↳ from %s", profile_path)
        logger.debug("Running command:\n    %s", " ".join(cmd))
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
            logger.debug("Running command:\n    %s", " ".join(cmd))
            subprocess.run(cmd, env=env, check=True)
        finally:
            # Remove BOTH staged artifacts (wheel + requirements manifest) so
            # neither can poison a later non-dev build in this context.
            for artifact in staged_artifacts:
                try:
                    artifact.unlink()
                except OSError:
                    logger.warning("Could not remove staged dev artifact %s", artifact)
