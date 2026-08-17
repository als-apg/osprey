"""Per-persona local image builds for multi-user web-terminal deployments.

``image_source: local`` deploys build one ``<project>-<persona>:local`` image
per referenced persona before any compose invocation. Each of those builds needs
a rendered persona project as its context, and ``osprey build`` is what writes
one: :func:`verify_persona_renders` checks that every referenced persona has
one and refuses the start when it does not. Nothing here renders — a start runs
what the last build produced. Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`;
registry-mode deploys never enter this module.
"""

import os
import posixpath
import re
from pathlib import Path
from typing import cast

from osprey.build.claude_code_resolver import load_provider_spec
from osprey.build.claude_code_telemetry import ObservabilityCredentialError
from osprey.cli.phase_reporter import report_group as _report_group
from osprey.cli.phase_reporter import report_step as _report_step
from osprey.deployment.build_progress import with_plain_build_progress
from osprey.deployment.compose_generator import (
    _copy_local_framework_for_override,
    resolve_project_name,
    resolve_repo_root,
)
from osprey.deployment.runtime_helper import get_runtime_command
from osprey.deployment.subprocess_capture import run_captured
from osprey.deployment.web_terminals.personas import effective_image_source
from osprey.deployment.wheel_build import _staged_dev_artifact_paths
from osprey.utils.config import ConfigBuilder
from osprey.utils.dotenv import ENV_LOCAL_FILENAME
from osprey.utils.log_filter import quiet_logger
from osprey.utils.logger import get_logger
from osprey.utils.workspace import BUILD_DIR_NAME, IMAGE_DIR_NAME

logger = get_logger("deployment.lifecycle")

#: Environment-variable names quoted inside an unresolved-placeholder message,
#: with any ``:-default`` suffix dropped. The credential check reports the
#: literal ``${VAR}`` it refused to encode, which is the only place the name of
#: the variable an operator has to set is available to this module.
_PLACEHOLDER_NAME_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")


def _persona_image_context(project_path: str | Path) -> Path:
    """The directory a persona's image is built from, given its rendered project.

    One rule for every OSPREY image: the build context is
    ``build/.image/<the image's project name>/`` — a deployment REPO rendered
    against the ``/app/<project name>`` path that container sees itself at, with
    ``profile.yml`` at its root and the render below it in ``build/``. The
    persona's flat render at ``build/<repo>-<persona>/`` is the HOST's copy: its
    ``config.yml`` records this machine's ``project_root``, so an image built
    from it ships MCP servers, hooks and an agent-data root that name the
    building host. Its ``.image`` sibling is the same render made against the
    container's own path.

    Derived from the render's location rather than from the repo root, because
    the catalog's ``project_path`` is what every caller here has (and is an
    externally-pinned contract — ``osprey init`` writes ``build/<repo>-<persona>``
    and the dispatch e2e pins it), and the two directories are siblings by
    construction: :func:`osprey.utils.workspace.container_image_context` puts the
    container copy at ``<render>/../.image/<render name>``. Relative in, relative
    out — repo-scoped verbs run from the repo root.

    :param project_path: The persona's rendered project directory, as the
        catalog names it.
    :return: The build context for that persona's image.
    """
    render = Path(project_path)
    return render.parent / IMAGE_DIR_NAME / render.name


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
    context: str,
    project: str,
    persona: str,
    project_label: str,
    dev_mode: bool = False,
) -> list[str]:
    """Construct the ``<runtime> build`` argv that produces ``<project>-<persona>:local``.

    Mirrors :func:`_project_image_build_cmd`'s argv shape (same
    ``OSPREY_PIP_SPEC`` build-arg, same runtime/tag/``-f``/context layout)
    generalized to an arbitrary persona image context, plus the
    ``com.osprey.project`` label ``osprey reset`` verifies before removing this
    tag. That shape includes where the ``Dockerfile`` is: the
    context is a deployment repo, so its recipe — and its ``config.yml`` — sit
    one level down, in the render.

    :param runtime: Base container command (``docker`` or ``podman``).
    :param context: The persona's image build context — its container repo,
        :func:`_persona_image_context` of the rendered project.
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
        other container label this module sets, so a later ``osprey reset`` can
        verify it before removing the tag.
    :return: The full build command as an argv list.
    """
    # Lazy import to avoid an import cycle: container_lifecycle imports
    # deploy_up_web_terminals from this module at module top level, so this
    # module must not import container_lifecycle at import time. _resolve_pip_spec
    # is a generic helper shared with the dispatch-worker build path, so it
    # stays in container_lifecycle.
    from osprey.deployment.container_lifecycle import _resolve_pip_spec

    render = os.path.join(context, BUILD_DIR_NAME)
    cmd = [
        runtime,
        "build",
        "-t",
        f"{project}-{persona}:local",
        "-f",
        os.path.join(render, "Dockerfile"),
        "--label",
        f"com.osprey.project={project_label}",
    ]
    cli_version = _resolve_persona_claude_cli_version(render)
    if cli_version:
        cmd.extend(["--build-arg", f"CLAUDE_CLI_VERSION={cli_version}"])
    cmd.extend(["--build-arg", f"OSPREY_PIP_SPEC={_resolve_pip_spec(dev_mode=dev_mode)}"])
    if dev_mode:
        cmd.extend(["--build-arg", "OSPREY_DEV=1"])
    with_plain_build_progress(cmd)
    cmd.append(context)
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
    but is logged as a warning (not silent): lint is the well-formedness gate,
    so a local deploy that bypasses lint would otherwise fail opaquely at
    ``compose up`` on an unbuilt tag with no clue why.

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
                "project_path configured. Skipping its local build. "
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
        # The delta this entry says its project was rendered from; carried here
        # so verify_persona_renders need not re-walk the catalog to explain an
        # absent one. Normalized to "" when absent or non-str, which that caller
        # treats as "no usable build_profile" identically to the raw value.
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
    :func:`_resolve_persona_profile` (the deploy path, which ``osprey up``
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


def _canonical_delta_reference(persona_name: str) -> str:
    """The one ``build_profile`` value a persona is supposed to carry.

    One spelling, shared by the remedy that recommends it and the check that
    asks whether recommending it would be circular — a catalog already holding
    this value needs to hear about a missing FILE, not about the value.
    """
    from osprey.cli.profile_root import PERSONA_DIRNAME

    return f"{PERSONA_DIRNAME}/{persona_name}.yml"


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
    reference = _canonical_delta_reference(persona_name)
    delta = profile_root / reference
    return (
        f"Set modules.web_terminals.personas.{persona_name}.build_profile to "
        f"'{reference}' — the delta at {delta} — which is what "
        "`osprey init` writes for every persona in the catalog. If this "
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
    init`` writes (``personas/<persona>.yml``).

    The shape rule itself lives in :func:`persona_build_profile_shape_problem`,
    which ``osprey lint`` enforces on the same values — one predicate, so a
    config cannot lint clean and then fail here. What this adds is the half lint
    cannot do: anchoring the accepted relative path on the profile root and
    checking that the file is actually there.

    The start path calls this only to explain an ABSENT render
    (:func:`verify_persona_renders`), never to produce one: it is what separates
    "you have not built yet" from "no build can ever write this", and the two
    have different remedies. Sending an operator whose catalog names a preset
    through ``osprey build`` would only bring them back here.

    :param build_profile: The catalog entry's value, non-empty.
    :param persona_name: Catalog key, named in every error.
    :param profile_root: This deployment's profile root — under the four-zone
        layout, the deployment repo itself.
    :return: The absolute path to the persona delta.
    :raises ValueError: For any value that is not an existing delta inside
        ``<profile root>/personas/``, naming both the value and the file the
        operator has to point at instead.
    """
    remedy = _persona_delta_remedy(persona_name, profile_root)

    problem = persona_build_profile_shape_problem(build_profile)
    if problem is not None:
        raise ValueError(
            f"Persona {persona_name!r} has no render and cannot get one: its build_profile "
            f"{problem} {remedy}"
        )

    # Shape is settled above, so what is left is the filesystem half. The
    # `.resolve()` is both for the message (an operator chasing a missing file
    # wants the absolute path, not the relative spelling they already wrote) and
    # for the anchoring check below, which needs the real target.
    candidate = (profile_root / build_profile).resolve()
    if not candidate.is_file():
        # Two different situations wear the same missing file, and the shared
        # remedy fits only one of them. When the catalog points somewhere ELSE,
        # the fix is to repoint it at the canonical delta. When it already names
        # the canonical delta, that advice is circular — "set this value to the
        # value it already holds" — and the real fact is that the source zone
        # lost a file the catalog still references.
        if build_profile == _canonical_delta_reference(persona_name):
            raise ValueError(
                f"Persona {persona_name!r} names the build_profile {build_profile!r}, which "
                f"is the right spelling, but no file exists at {candidate}. The catalog "
                "references a delta this repo does not have: either the delta was removed "
                "and the entry outlived it, or the entry was added and the delta never "
                f"written. Restore {candidate}, or drop "
                f"modules.web_terminals.personas.{persona_name} from the catalog and the "
                "roster entries that reference it. If the delta was never written because "
                "this deployment's variant build predates the layout, run "
                "/osprey-build-interview: it converts an existing variant into a persona "
                "delta over the profile this repo is built from."
            )
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
            f"Persona {persona_name!r} has no render and cannot get one from {build_profile!r}: {e}"
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


def _check_existing_render(persona_name: str, project_path: Path, repo_root: Path) -> None:
    """Accept a persona's rendered project, or say why it is unusable.

    The render is ``osprey build``'s and the start is only reading it, so this
    is the last chance to notice that it will not actually come up: both ways it
    can fail are quiet until the container is already running.

    A directory missing ``config.yml`` or ``Dockerfile`` is a partial or aborted
    render. It is refused rather than patched over, and the remedy is a rebuild:
    a start writes nothing into ``build/``.

    The image is built from the container copy of that render
    (:func:`_persona_image_context`), so its ``Dockerfile`` and ``config.yml``
    are checked too — a build produces both copies in one atomic swap, so a
    repo with one and not the other was built by a version that did not know
    about images, and the container build would fail with a docker error about
    a missing path instead of a sentence naming the persona.

    A complete render still has its model selection checked. The web terminal
    runs the same resolver strict at startup, so a model the profile has since
    moved away from would ship a container that crash-loops behind the reverse
    proxy and surface only as a 502. That check reads the persona's own render
    but expands ``${VAR}`` against the REPO root's ``.env``: personas are
    rendered into the disposable build zone and a deployment keeps its secrets
    at its root, so resolving against the render would leave a
    ``default_model: ${OPS_MODEL}`` as a literal placeholder and refuse a
    persona the deployment can in fact serve.

    That same resolve also reads the persona's telemetry block, so a missing or
    unresolved observability credential surfaces here too — and it is reported
    as the credential problem it is. The remedy for a model the provider cannot
    serve is an edit to the profile; the remedy for an unset observability
    credential is a variable in the deployment's own ``.env``, and an operator
    handed the wrong one of those goes and changes something that was never
    wrong.

    Args:
        persona_name: The persona whose render is being checked.
        project_path: Its rendered project directory under ``build/``.
        repo_root: The deployment repo — the zone holding ``.env``.

    Raises:
        ValueError: Naming the persona, the directory, and the way out.
    """
    context = _persona_image_context(project_path)
    required = ("config.yml", "Dockerfile")
    missing = [name for name in required if not (project_path / name).is_file()]
    # Kept as its own list, not appended to the one above: the two live under
    # different roots, and a flat join would print the image context's files as
    # though they were paths under the render.
    missing_in_context = [
        name for name in required if not (context / BUILD_DIR_NAME / name).is_file()
    ]
    if missing or missing_in_context:
        where = []
        if missing:
            where.append(", ".join(missing))
        if missing_in_context:
            where.append(f"{', '.join(missing_in_context)} under its image context {context}")
        raise ValueError(
            f"Persona {persona_name!r} has a partial render at "
            f"{project_path} (missing {'; '.join(where)}). "
            "Re-render it with `osprey build`, which replaces build/ whole."
        )

    try:
        load_provider_spec(project_path, env_dir=repo_root)
    except ObservabilityCredentialError as e:
        # Ahead of the ValueError arm below on purpose: this IS a ValueError,
        # so the broad handler would otherwise claim a credential the profile
        # never names is a bad model, and send the operator to edit profile.yml
        # over a variable that belongs in the deployment's secret store. Only
        # the credential failures land here — an unusable endpoint stays a
        # general telemetry misconfiguration and keeps the broad frame.
        env_path = repo_root / ENV_LOCAL_FILENAME
        names = sorted(set(_PLACEHOLDER_NAME_RE.findall(str(e))))
        if names:
            # Names only, never values: a placeholder is unresolved precisely
            # because nothing on this host holds the value to print.
            fix = " && ".join(f'echo "{var}=<value>" >> {env_path}' for var in names)
            remedy = f"Set {', '.join(names)} in {env_path} — {fix} — "
        else:
            remedy = (
                "Give the telemetry block's openobserve.user and "
                f"openobserve.password real values, via {env_path} — "
            )
        raise ValueError(
            f"Persona {persona_name!r} render at {project_path} names "
            f"observability credentials this deployment cannot resolve:\n  {e}\n"
            f"{remedy}then re-render with `osprey build`. The model "
            "configuration is not what is wrong here. Note that `osprey up` "
            "mints the observability store's own ZO_ROOT_USER_PASSWORD into "
            "that file when it starts the store, so a deployment that has "
            "never been started will not have one there yet."
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Persona {persona_name!r} render at {project_path} has a "
            f"model configuration its provider cannot serve:\n  {e}\n"
            "Fix the model in this deployment's profile — its profile.yml or "
            f"the persona's own delta — and re-render with `osprey build`."
        ) from e


def verify_persona_renders(
    config: dict,
    resolved_users: list[dict],
    repo_root: Path,
) -> None:
    """Check every referenced persona has a usable rendered project, or refuse.

    Every persona :func:`build_persona_images` is about to build needs a
    rendered project on disk — its ``Dockerfile`` and ``config.yml`` are the
    image build context. ``osprey build`` writes them, one per delta in
    ``personas/`` (:func:`osprey.cli.build_cmd._render_persona_projects`), and a
    start writes none: this runs BEFORE the image builds so a gap is a refusal
    with a remedy rather than an opaque failure inside a container build.

    A start must never render the missing ones itself, and the absence is the
    point. ``build/`` is a complete account of what a deploy will run, so a
    persona project appearing at start time makes that account false exactly
    when it matters: the operator whose delta changed would get a fresh persona
    standing beside a deployment rendered from the earlier profile. The
    drift gate already refuses that start — the persona deltas are folded into
    the build fingerprint — so the honest reading of a missing render is "this
    build is stale or partial", and the honest remedy is the one that replaces
    ``build/`` whole.

    Operates on exactly :func:`_referenced_personas`'s distinct set — the same
    personas that will be built — so what is checked and what is built never
    diverge. Note that the CATALOG decides which personas are deployed while
    ``personas/`` decides which are rendered, so a repo may hold deltas nothing
    references; those are built and simply not checked here.

    For each:

    * **Directory present and complete**: accepted, after
      :func:`_check_existing_render` confirms it will actually come up.
    * **Directory present but incomplete**: refused there — see that function.
    * **Directory absent**: refused here. The catalog entry is validated first,
      because a value that can never name a delta would send the operator
      through a build that succeeds and a start that refuses again.

    :param config: Raw deploy config, forwarded to :func:`_referenced_personas`
        (which reads ``modules.web_terminals.personas``).
    :param resolved_users: :func:`osprey.deployment.web_terminals.personas.resolve_personas`'s
        output for this deploy.
    :param repo_root: The deployment repo — the directory holding
        ``profile.yml``, ``personas/`` and ``build/``. It IS the profile root:
        under the four-zone layout the source zone is the repo root, so there
        is no separate profile directory to locate and nothing to read a
        manifest for.
    :raises ValueError: A referenced persona with no render, a partial render,
        or a render whose model its provider cannot serve; and, on the absent
        path, any ``build_profile`` that is not an existing delta inside
        ``<repo>/personas/``.
    """
    for unit in _referenced_personas(config, resolved_users):
        persona_name = unit["persona"]
        project_path = Path(unit["project_path"])

        if project_path.exists():
            _check_existing_render(persona_name, project_path, repo_root)
            continue

        build_profile = unit["build_profile"]
        if not isinstance(build_profile, str) or not build_profile:
            raise ValueError(
                f"Persona {persona_name!r} has no rendered project at {project_path}, and "
                "its catalog entry has no build_profile naming the delta a build would "
                "render it from. " + _persona_delta_remedy(persona_name, repo_root)
            )

        # Raises when no build could ever produce this render — a preset name, a
        # path that leaves personas/, a delta that is not there. Checked before
        # the remedy below so the operator is not sent through a build that
        # cannot change the outcome.
        delta = _resolve_persona_profile(build_profile, persona_name, repo_root)
        rendered_at = repo_root / BUILD_DIR_NAME / f"{repo_root.name}-{delta.stem}"

        raise ValueError(
            f"Persona {persona_name!r} has no rendered project at {project_path}. "
            "Persona projects are rendered by `osprey build`, never by a start — "
            "`osprey up` runs what the last build produced. Run `osprey build` in "
            f"{repo_root} and start again.\n"
            f"A build of this repo renders {delta.name} to {rendered_at}. If that "
            "directory is there and this one is not, the catalog names a path no build "
            f"writes: set modules.web_terminals.personas.{persona_name}.project_path to "
            f"'{BUILD_DIR_NAME}/{rendered_at.name}' and its project to "
            f"'{rendered_at.name}', which is what `osprey init` writes."
        )


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
    under ``strict=True`` and the lint rule it mirrors — ``osprey up``
    never runs lint, so this fail-closed check must live on the build path
    too, not only on whichever render/resolve call happened to run first.

    Each build's context is the persona's own container repo — the ``.image``
    sibling of the ``project_path`` render, see :func:`_persona_image_context` —
    its ``CLAUDE_CLI_VERSION`` build-arg comes from THAT project's own
    ``config.yml`` (never the facility config — see
    :func:`_resolve_persona_claude_cli_version`), and under ``dev_mode`` a
    locally-built wheel is staged into that same context and removed
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
    # The deployment repo whose var/logs/ takes each build's spooled output.
    repo_root = resolve_repo_root(config)

    for unit in referenced:
        persona_name = unit["persona"]
        project = unit["project"]
        # The container repo the build renders for this persona — NOT the flat
        # host render the catalog names. See _persona_image_context.
        context = str(_persona_image_context(unit["project_path"]))

        # OSPREY_DEV=1 (the pin-relaxing build arg) is passed only when the
        # wheel actually landed in THIS persona's context: on a failed
        # build/staging the Dockerfile must keep its fail-loud pinned install
        # rather than silently falling back to the latest published release
        # (mirroring _build_project_image's fail-closed dev path).
        staged_artifacts: list[Path] = []
        wheel_staged = False
        if dev_mode:
            before = _staged_dev_artifact_paths(context)
            wheel_staged = bool(_copy_local_framework_for_override(context))
            staged_artifacts = sorted(_staged_dev_artifact_paths(context) - before)
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
                context,
                project,
                persona_name,
                project_label,
                dev_mode and wheel_staged,
            )
            image_tag = f"{project}-{persona_name}:local"
            # No "building X:" announcement: the live build region carries the
            # progress while it runs, and the step line below reports the
            # finished image.
            logger.debug("Running command:\n    %s", " ".join(cmd))
            # Function-level import for the same cycle reason as
            # `_resolve_pip_spec` above: container_lifecycle imports this module
            # at its own top level.
            from osprey.deployment.container_lifecycle import single_image_build_reporter

            # Watched for the duration of the build and no longer; the step line
            # below is what reports the finished image. The group declaration is
            # idempotent, so the per-persona loop declares it each pass.
            _report_group("personas")
            with (report := single_image_build_reporter(image_tag)):
                run_captured(
                    cmd,
                    env=env,
                    spool_name=f"build-persona-{project}-{persona_name}",
                    repo_root=repo_root,
                    on_line=report,
                )
            # One line per image, after its own build: the slowest step of a
            # local-mode deploy, and the only progress an operator gets while
            # a handful of multi-minute builds run one after another.
            _report_step(f"persona image {image_tag}")
        finally:
            # Remove BOTH staged artifacts (wheel + requirements manifest) so
            # neither can poison a later non-dev build in this context.
            for artifact in staged_artifacts:
                try:
                    artifact.unlink()
                except OSError:
                    logger.warning("Could not remove staged dev artifact %s", artifact)
