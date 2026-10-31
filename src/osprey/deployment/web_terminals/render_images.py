"""Per-render local image builds for multi-user web-terminal deployments.

``image_source: local`` deploys build one ``<project>:local`` image per
referenced render before any compose invocation (the tag names the
render's rendered project — see
:func:`osprey.deployment.web_terminals.roster.resolve_cards` for the
naming rules). Each of those builds needs
a rendered render project as its context, and ``osprey build`` is what writes
one: :func:`verify_renders` checks that every referenced render has
one and refuses the start when it does not. Nothing here renders — a start runs
what the last build produced. Called from
:func:`osprey.deployment.web_terminals.provision.deploy_up_web_terminals`, and,
for a deployment whose dispatch workers run at a tier, from ``osprey up``
beside the project image (:func:`osprey.deployment.container_lifecycle._build_referenced_renders`);
registry-mode deploys build nothing here.
"""

import os
import posixpath
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

from osprey.agent_runner.provider_env import load_provider_spec
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
from osprey.deployment.web_terminals.render_naming import (
    RENDER_KIND_TIER,
    render_project,
)
from osprey.deployment.web_terminals.roster import (
    RENDER_CATALOG_REQUIRED,
    dispatch_tier_name,
    effective_image_source,
    resolve_cards,
    served_renders,
)
from osprey.deployment.wheel_build import _staged_dev_artifact_paths
from osprey.docs_links import setup_remedy
from osprey.utils.workspace import (
    IMAGE_DIR_NAME,
    LAYER_DIR_KINDS,
    LAYER_DIRNAMES,
    TIERS_DIRNAME,
)
from osprey_connectors.config import ConfigBuilder
from osprey_connectors.log_filter import quiet_logger
from osprey_connectors.logger import get_logger
from osprey_connectors.workspace import BUILD_DIR_NAME

logger = get_logger("deployment.lifecycle")


def _render_image_context(project_path: str | Path) -> Path:
    """The directory a render's image is built from, given its rendered project.

    One rule for every OSPREY image: the build context is
    ``build/.image/<the image's project name>/`` — a deployment REPO rendered
    against the ``/app/<project name>`` path that container sees itself at, with
    ``profile.yml`` at its root and the render below it in ``build/``. The
    render's flat render at ``build/<project_name>-<render>/`` is the HOST's copy: its
    ``config.yml`` records this machine's ``project_root``, so an image built
    from it ships MCP servers, hooks and an agent-data root that name the
    building host. Its ``.image`` sibling is the same render made against the
    container's own path.

    Derived from the render's location rather than from the repo root, because
    the catalog's ``project_path`` is what every caller here has (and is an
    externally-pinned contract — ``osprey build`` writes
    ``build/<project_name>-<render>`` into the rendered config and the dispatch
    e2e pins it), and the two directories are siblings by
    construction: :func:`osprey.utils.workspace.container_image_context` puts the
    container copy at ``<render>/../.image/<render name>``. Relative in, relative
    out — repo-scoped verbs run from the repo root.

    :param project_path: The render's rendered project directory, as the
        catalog names it.
    :return: The build context for that render's image.
    """
    render = Path(project_path)
    return render.parent / IMAGE_DIR_NAME / render.name


def _resolve_render_claude_cli_version(project_path: str) -> str | None:
    """The render's own ``CLAUDE_CLI_VERSION`` build-arg value, or ``None`` to omit it.

    Read from ``<project_path>/config.yml``'s ``claude_code.cli_version`` —
    NEVER the facility config passed to :func:`build_render_images`, since
    each render project is its own independently-rendered project with its
    own pin. Unlike :func:`_resolve_claude_cli_version` (the dispatch-worker/
    facility-project path), there is deliberately no framework-default
    fallback here: when the render's own ``config.yml`` doesn't set a pin,
    the build-arg is omitted entirely so the Dockerfile's own rendered
    ``ARG CLAUDE_CLI_VERSION=<default>`` stands, rather than silently
    overriding it with a value that specific render project never declared.

    A missing/unreadable ``config.yml`` degrades the same way — logged and
    treated as unset — since a malformed render catalog entry is lint's job
    to reject, not this builder's.

    :param project_path: The render's rendered project directory.
    :return: The pinned version string, or ``None`` if unset/unreadable.
    """
    config_path = Path(project_path) / "config.yml"
    if not config_path.is_file():
        return None
    try:
        with quiet_logger(["registry", "CONFIG"]):
            render_config = ConfigBuilder(str(config_path)).raw_config
    except Exception as exc:
        logger.warning("Could not read %s for CLAUDE_CLI_VERSION: %s", config_path, exc)
        return None
    from osprey.agent_runner.launcher import pinned_cli_version

    return pinned_cli_version(render_config.get("claude_code") or {})


def _render_image_build_cmd(
    runtime: str,
    context: str,
    image_tag: str,
    project_label: str,
    config: dict,
    dev_mode: bool = False,
) -> list[str]:
    """Construct the ``<runtime> build`` argv that produces *image_tag*.

    Mirrors :func:`_project_image_build_cmd`'s argv shape (same
    ``OSPREY_PIP_SPEC`` build-arg, same runtime/tag/``-f``/context layout)
    generalized to an arbitrary render image context, plus the
    ``com.osprey.project`` label ``osprey reset`` verifies before removing this
    tag. That shape includes where the ``Dockerfile`` is: the
    context is a deployment repo, so its recipe — and its ``config.yml`` — sit
    one level down, in the render.

    :param runtime: Base container command (``docker`` or ``podman``).
    :param context: The render's image build context — its container repo,
        :func:`_render_image_context` of the rendered project.
    :param config: THIS DEPLOYMENT's facility config, for the site build args
        (:func:`osprey.deployment.container_lifecycle.site_image_build_args`).
        The facility config and not the render's own, unlike
        ``CLAUDE_CLI_VERSION`` above: a site CA, a package index and the
        offline mode are properties of the HOST every image on it is built on,
        so a render catalog cannot sensibly disagree with the deployment
        about them.
    :param image_tag: The tag to build, exactly as
        :func:`osprey.deployment.web_terminals.roster.resolve_cards`
        resolved it into the entry's ``image`` — the single derivation site
        for render tag naming, never recomputed here.
    :param dev_mode: Whether ``--dev`` was passed (adds an ``OSPREY_DEV=1``
        build-arg, mirroring :func:`_project_image_build_cmd`'s dev path).
    :param project_label: THIS DEPLOYMENT's project name
        (:func:`resolve_project_name` on the facility config being deployed),
        NOT the render's own ``project`` — the ``com.osprey.project`` label
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
    from osprey.deployment.container_lifecycle import _resolve_pip_spec, site_image_build_args
    from osprey.version import pins_prerelease

    render = os.path.join(context, BUILD_DIR_NAME)
    cmd = [
        runtime,
        "build",
        "-t",
        image_tag,
        "-f",
        os.path.join(render, "Dockerfile"),
        "--label",
        f"com.osprey.project={project_label}",
    ]
    cli_version = _resolve_render_claude_cli_version(render)
    if cli_version:
        cmd.extend(["--build-arg", f"CLAUDE_CLI_VERSION={cli_version}"])
    pip_spec = _resolve_pip_spec(dev_mode=dev_mode)
    cmd.extend(["--build-arg", f"OSPREY_PIP_SPEC={pip_spec}"])
    if pins_prerelease(pip_spec):
        # Same reason as the project image: a beta framework resolves only
        # beside its beta connectors, which pip admits only under --pre.
        cmd.extend(["--build-arg", "OSPREY_PIP_PRE=1"])
    if dev_mode:
        cmd.extend(["--build-arg", "OSPREY_DEV=1"])
    cmd.extend(site_image_build_args(config, context))
    with_plain_build_progress(cmd)
    cmd.append(context)
    return cmd


#: The roster name the dispatch tier is resolved under. Never a card: it only
#: lets :func:`resolve_cards` name the tier's image the way it names every
#: other image, so the worker and a person on the same tier agree on the tag.
_DISPATCH_RESOLUTION_NAME = "dispatch-worker"


def _pinned_tier_entry(
    config: Mapping[str, Any], tier: str, *, strict: bool = True
) -> dict[str, Any] | None:
    """A tier no card runs, resolved as a single roster entry pinned to it.

    The resolution behind the dispatch tier (:func:`resolve_dispatch_tier`)
    and behind a tier only ``default_tier`` or a role names: the tier resolves
    against the deployment's own ``modules.web_terminals`` (its render
    projection, default tier, image source, image tag) and ``registry``, with
    no shared apps and no role table, so its image is named by
    :func:`resolve_cards`, the one function that names every image, and a
    person later given that tier pulls exactly this tag.

    :return: The resolved entry, or ``None`` when it names no image.
    :raises ValueError: Any refusal :func:`resolve_cards` raises for it under
        *strict*.
    """
    modules = config.get("modules")
    web_terminals_raw = modules.get("web_terminals") if isinstance(modules, Mapping) else None
    web_terminals = dict(web_terminals_raw) if isinstance(web_terminals_raw, Mapping) else {}
    # A one-person roster written in place (not a roster read): the tier resolves alone.
    web_terminals.update(users=[{"name": _DISPATCH_RESOLUTION_NAME, "index": 0, "tier": tier}])
    web_terminals.pop("shared", None)
    web_terminals.pop("authorization", None)
    registry = config.get("registry")
    (entry,) = resolve_cards(
        web_terminals, dict(registry) if isinstance(registry, Mapping) else {}, strict=strict
    )
    return entry if isinstance(entry.get("image"), str) else None


def resolve_dispatch_tier(
    config: Mapping[str, Any], *, strict: bool = True
) -> dict[str, Any] | None:
    """The dispatch tier's render, resolved by the one function that names every image.

    The tier resolves as a single roster entry pinned to it, against the
    deployment's own ``modules.web_terminals`` (its render projection, default
    tier, image source, image tag) and ``registry``, with no shared apps and no
    role table: so local mode names ``<p>-<tier>:local`` and registry mode
    names exactly what a person on that tier pulls. Nothing here spells an
    image itself.

    :param config: Raw deploy config.
    :param strict: Refuse a tier that does not resolve. ``False`` answers
        ``None`` instead, for callers that only report.
    :return: The resolved entry (``render``, ``image``, ``project``,
        ``container_project_dir``, ...), or ``None`` when no tier is set.
    :raises ValueError: Under ``strict``, a tier with no render in
        ``modules.web_terminals.renders``, or any refusal
        :func:`resolve_cards` raises for it.
    """
    tier = dispatch_tier_name(config)
    if tier is None:
        return None
    modules = config.get("modules")
    web_terminals_raw = modules.get("web_terminals") if isinstance(modules, Mapping) else None
    web_terminals = dict(web_terminals_raw) if isinstance(web_terminals_raw, Mapping) else {}
    renders = web_terminals.get("renders")
    render = renders.get(tier) if isinstance(renders, Mapping) else None
    if not isinstance(render, Mapping) or render.get("kind") != RENDER_KIND_TIER:
        if not strict:
            return None
        raise ValueError(
            f"dispatch.tier names {tier!r}, but this build has no render of that tier in "
            "modules.web_terminals.renders, so no image can run the dispatch workers. "
            f"Check that tiers/{tier}.yml exists and the web terminals are enabled, then "
            "run `osprey build` and start again."
        )
    try:
        return _pinned_tier_entry(config, tier, strict=strict)
    except ValueError:
        if strict:
            raise
        return None


def dispatch_worker_tier_image(config: Mapping[str, Any], env: Mapping[str, Any]) -> str | None:
    """The image a tiered dispatch worker runs, or ``None`` when no tier is set.

    A tier decides the worker's image outright, so anything else that would
    name one is refused rather than silently losing: an ``OSPREY_WORKER_IMAGE``
    override, or a ``services.dispatch_worker.image`` pin that names another
    image.

    :param config: Raw deploy config.
    :param env: The environment the start runs with (read for
        ``OSPREY_WORKER_IMAGE``).
    :raises ValueError: Either contradiction above, or a tier that does not
        resolve (:func:`resolve_dispatch_tier`).
    """
    tier = dispatch_tier_name(config)
    if tier is None:
        return None
    if env.get("OSPREY_WORKER_IMAGE"):
        raise ValueError(
            f"OSPREY_WORKER_IMAGE is set, but dispatch.tier {tier!r} already decides the "
            "dispatch workers' image (the tier's render). Unset OSPREY_WORKER_IMAGE, or "
            "remove dispatch.tier from the profile and run `osprey build` to run the "
            "workers on an image of your choosing."
        )
    entry = cast(dict, resolve_dispatch_tier(config, strict=True))
    image = cast(str, entry["image"])
    services = config.get("services")
    worker = services.get("dispatch_worker") if isinstance(services, Mapping) else None
    pinned = worker.get("image") if isinstance(worker, Mapping) else None
    if pinned and pinned != image:
        raise ValueError(
            f"services.dispatch_worker.image pins {pinned!r}, but dispatch.tier {tier!r} "
            f"runs the workers on the tier's render {image!r}. Remove one of the two from "
            "the profile and run `osprey build`."
        )
    return image


#: The tags built so far in the open :func:`render_build_session`, or ``None``
#: outside one.
_SESSION_BUILT_TAGS: ContextVar[set[str] | None] = ContextVar(
    "render_build_session_tags", default=None
)


@contextmanager
def render_build_session() -> Iterator[None]:
    """One start's worth of render builds: each tag is built at most once inside it.

    ``osprey up`` builds every referenced render beside the project image, and
    the web stack's own pass later asks for the same set; inside one session
    that second ask finds every tag already built and builds nothing.
    """
    token = _SESSION_BUILT_TAGS.set(set())
    try:
        yield
    finally:
        _SESSION_BUILT_TAGS.reset(token)


def _referenced_renders(
    config: dict, resolved_users: list[dict], *, serves_roster: bool = True
) -> list[dict[str, str]]:
    """One build unit per render this deployment serves.

    Which renders: :func:`~osprey.deployment.web_terminals.roster.served_renders`
    — every card's render, ``default_tier``, every role's tier and the dispatch
    tier — resolved strictly, so a roster whose binding does not resolve is
    refused here as at render. A role tier nobody holds today is built too:
    single sign-on may hand it out tomorrow, and a start must not meet an
    unbuilt tag then. This function decides nothing about membership; it is the
    one projection from a served render to the image that runs it.

    Each served render's ``image`` and ``project`` come from the one function
    that names every image,
    :func:`~osprey.deployment.web_terminals.roster.resolve_cards`, and are
    carried through rather than re-derived: from the entry of a card in
    ``resolved_users`` that runs the render, else (the dispatch tier) from
    :func:`resolve_dispatch_tier`, else (a tier only ``default_tier`` or a role
    names) from a single entry pinned to it, the same way. An app no card in
    ``resolved_users`` runs, and a name the projection does not list, give no
    unit: there is nothing to name its image from, and lint reports the name.

    A served render whose projection entry has no ``project_path`` is skipped
    with a warning (not silently): lint is the well-formedness gate, so a local
    deploy that bypasses lint would otherwise fail opaquely at ``compose up`` on
    an unbuilt tag with no clue why.

    :param config: Raw deploy config (read for ``modules.web_terminals`` and
        ``services.dispatch_worker.tier``).
    :param resolved_users: :func:`osprey.deployment.web_terminals.roster.resolve_cards`'s
        strict output for this deploy.
    :param serves_roster: Whether this deploy stands up the web terminals. A
        deploy without them serves no roster, so the dispatch tier is its one
        render.
    :return: One ``{"render", "project", "project_path", "build_profile",
        "image"}`` dict per served render, in served order (``build_profile``
        is the file the render is built from, ``tiers/<name>.yml``).
    :raises ValueError: A roster or dispatch tier that does not resolve.
    """
    by_render: dict[str, dict] = {}
    for card in resolved_users:
        render_name = card.get("render")
        if isinstance(render_name, str):
            by_render.setdefault(render_name, card)
    # Resolved before the walk, so a dispatch tier this build has no render of
    # is refused rather than skipped as a name the projection does not list.
    dispatch_entry = resolve_dispatch_tier(config, strict=True)

    served_set = served_renders(config, strict=True)
    if not serves_roster:
        dispatch_tier = dispatch_tier_name(config)
        served_set = {name: s for name, s in served_set.items() if name == dispatch_tier}

    referenced: list[dict[str, str]] = []
    for render_name, served in served_set.items():
        if served is None:
            continue
        entry = by_render.get(render_name)
        if entry is None and dispatch_entry is not None and render_name == dispatch_entry["render"]:
            entry = dispatch_entry
        if entry is None and served.kind == RENDER_KIND_TIER:
            entry = _pinned_tier_entry(config, render_name)
        if entry is None:
            continue
        if served.project_path is None:
            logger.warning(
                "Render %r is referenced but its modules.web_terminals.renders entry "
                "has no project_path. Skipping its local build. "
                "compose up will fail on the unbuilt %r tag.",
                render_name,
                entry.get("image"),
            )
            continue
        # Every strictly resolved entry carries a non-empty "project" and
        # "image". No fallback derivation here, which would silently diverge
        # from the tag resolve_cards itself resolved.
        project = cast(str, entry.get("project"))
        image = cast(str, entry.get("image"))
        # The file this render is built from; carried here so
        # verify_renders need not re-walk the projection to explain an
        # absent render.
        referenced.append(
            {
                "render": render_name,
                "project": project,
                "project_path": served.project_path,
                "build_profile": _canonical_delta_reference(render_name),
                "image": image,
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
    read on the same machine that renders the renders. Used only to pick which
    rejection an unusable value earns — a preset name and a path that leaves the
    profile are different mistakes with different fixes.
    """
    return "/" in build_profile or build_profile.endswith(_PROFILE_FILE_SUFFIXES)


def render_build_profile_shape_problem(build_profile: str) -> str | None:
    """Why ``build_profile`` can never name a login file, or ``None`` if it can.

    The single shape rule for the file a render is built from, shared by the two
    places that enforce it: ``osprey lint`` (the well-formedness gate) and
    :func:`_resolve_render_profile` (the deploy path, which ``osprey up``
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
    if not _is_profile_file_reference(build_profile):
        return (
            f"is the bundled preset name {build_profile!r}. A login render is never "
            "rendered from a preset: it is rendered from a login file over the profile "
            "the deployed project was built from, which is what gives it that profile's data "
            "tree, secrets and conventions."
        )

    if posixpath.isabs(build_profile):
        return (
            f"is the absolute path {build_profile!r}. A login file belongs to the "
            "deployed project's own profile and is named relative to it."
        )

    # Lexical normalization, so `tiers/../ops.yml` is judged on where it
    # actually lands rather than on the directory name it passes through.
    if posixpath.dirname(posixpath.normpath(build_profile)) not in LAYER_DIRNAMES:
        layer_dirs = " or ".join(f"{dirname}/" for dirname in LAYER_DIR_KINDS)
        return (
            f"is {build_profile!r}, which is not a direct child of the profile's "
            f"{layer_dirs} directory. A login file is read as a layer only when "
            "its parent directory IS one of those, so anything else would build a "
            "hollow project from the layer alone, or from a profile this deployment does "
            "not own."
        )

    return None


def _canonical_delta_reference(render_name: str) -> str:
    """The one ``build_profile`` value a render is supposed to carry.

    ``tiers/<name>.yml``, the login file it is rendered from. One spelling, shared by the remedy that recommends it and the check that asks
    whether recommending it would be circular — a catalog already holding this
    value needs to hear about a missing FILE, not about the value.
    """
    return f"{TIERS_DIRNAME}/{render_name}.yml"


#: What an operator whose variant build predates the layer layout does
#: next. The deploy path's refusals and the lint gate's end with it, so a config
#: rejected by either one is sent to the same fix.
PREDATES_DELTA_REMEDY = (
    "A variant build that predates the layer layout has no delta yet. "
    + setup_remedy(
        "the setup converts an existing variant into a layer over the "
        "profile this deployment is built from"
    )
)


def _layer_delta_remedy(render_name: str, profile_root: Path) -> str:
    """The one sentence every unusable-``build_profile`` error ends with.

    Names both spellings the operator needs — the catalog value to write and the
    file it has to resolve to — from one place, so the several ways an entry can
    be wrong cannot end up recommending different fixes. Ends with
    :data:`PREDATES_DELTA_REMEDY`, which links the guided setup's page, because the
    operator most likely to read this is one whose project predates the
    layer layout: they have a variant build in some older shape and
    need it converted, which is a bigger job than editing one catalog value.
    """
    reference = _canonical_delta_reference(render_name)
    delta = profile_root / reference
    return (
        f"Set modules.web_terminals.renders.{render_name}.build_profile to "
        f"'{reference}' — the delta at {delta} — which is what "
        "`osprey init` writes for every render in the catalog. " + PREDATES_DELTA_REMEDY
    )


def _resolve_render_profile(build_profile: str, render_name: str, profile_root: Path) -> Path:
    """Resolve a catalog ``build_profile`` to the layer it must name.

    A render project is rendered from a delta over the very profile this
    deployment was built from — that is what gives the render the same data
    tree, the same ``.env`` secrets and the same convention artifacts as its
    host. So the catalog value is accepted in exactly one shape: a relative path
    to a direct child of ``<profile root>/tiers/``, which is what ``osprey
    init`` writes (``tiers/<render>.yml``).

    The shape rule itself lives in :func:`render_build_profile_shape_problem`,
    which ``osprey lint`` enforces on the same values — one predicate, so a
    config cannot lint clean and then fail here. What this adds is the half lint
    cannot do: anchoring the accepted relative path on the profile root and
    checking that the file is actually there.

    The start path calls this only to explain an ABSENT render
    (:func:`verify_renders`), never to produce one: it is what separates
    "you have not built yet" from "no build can ever write this", and the two
    have different remedies. Sending an operator whose catalog names a preset
    through ``osprey build`` would only bring them back here.

    :param build_profile: The catalog entry's value, non-empty.
    :param render_name: Catalog key, named in every error.
    :param profile_root: This deployment's profile root — under the four-zone
        layout, the deployment repo itself.
    :return: The absolute path to the layer.
    :raises ValueError: For any value that is not an existing delta inside
        ``<profile root>/tiers/``, naming both the value and the file the
        operator has to point at instead.
    """
    remedy = _layer_delta_remedy(render_name, profile_root)

    problem = render_build_profile_shape_problem(build_profile)
    if problem is not None:
        raise ValueError(
            f"Catalog entry {render_name!r} has no render and cannot get one: its build_profile "
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
        if build_profile == _canonical_delta_reference(render_name):
            raise ValueError(
                f"Render {render_name!r} names the build_profile {build_profile!r}, which "
                f"is the right spelling, but no file exists at {candidate}. The catalog "
                "references a delta this repo does not have: either the delta was removed "
                "and the entry outlived it, or the entry was added and the delta never "
                f"written. Restore {candidate}, or drop "
                f"modules.web_terminals.renders.{render_name} from the catalog and the "
                "roster entries that reference it. " + PREDATES_DELTA_REMEDY
            )
        raise ValueError(
            f"Render {render_name!r} names the build_profile {build_profile!r}, but no "
            f"file exists at {candidate}. " + remedy
        )

    # The property everything else rests on: the file handed to `osprey build`
    # must anchor BACK at the profile root it was resolved from. The shape rule
    # is lexical — it is shared with lint, which has no filesystem — and cannot
    # see a symlink, while `.resolve()` follows one. So a delta symlinked out of
    # `tiers/`, or a symlinked `tiers/` directory, would otherwise pass
    # every check above and then be read by root discovery somewhere else
    # entirely: as a standalone profile (no data tree, no `.env`, no
    # conventions), or as a delta over a DIFFERENT facility's profile. Both
    # build a hollow or wrong render and report nothing, which is the one
    # outcome worse than any rejection above.
    #
    # Asked of root discovery rather than by comparing directories, because the
    # danger is the wrong ROOT, not the symlink: a symlinked `tiers/` makes
    # candidate and the expected directory resolve to the SAME place, so a
    # containment check on the parent passes while the root is still wrong.
    from osprey.cli.profile_root import resolve_profile_root
    from osprey_connectors.errors import BuildProfileError

    try:
        anchored_root, anchored_kind = resolve_profile_root(candidate)
    except BuildProfileError as e:
        # The delta sits in a `tiers/` directory with no profile beside it.
        # That error already names the file that has to exist, and says it
        # better than a second sentence here could.
        raise ValueError(
            f"Catalog entry {render_name!r} has no render and cannot get one from {build_profile!r}: {e}"
        ) from e

    if not anchored_kind.is_layer or anchored_root != profile_root:
        landed = (
            f"a delta over the profile at {anchored_root}"
            if anchored_kind.is_layer
            else f"a standalone profile at {anchored_root}"
        )
        layer_dir = posixpath.dirname(posixpath.normpath(build_profile))
        raise ValueError(
            f"Render {render_name!r} names the build_profile {build_profile!r}, which "
            f"resolves to {candidate} — {landed}, not a delta over the profile this "
            f"deployment was built from ({profile_root}). A symlinked delta, or a "
            f"symlinked {layer_dir}/ directory, does this: the render would be "
            "built without that profile's data tree, secrets and conventions, or over "
            f"somebody else's. Keep the delta a real file inside {profile_root / layer_dir}. "
            + remedy
        )
    return candidate


def _check_existing_render(render_name: str, project_path: Path, repo_root: Path) -> None:
    """Accept a render's rendered project, or say why it is unusable.

    The render is ``osprey build``'s and the start is only reading it, so this
    is the last chance to notice that it will not actually come up: both ways it
    can fail are quiet until the container is already running.

    A directory missing ``config.yml`` or ``Dockerfile`` is a partial or aborted
    render. It is refused rather than patched over, and the remedy is a rebuild:
    a start writes nothing into ``build/``.

    The image is built from the container copy of that render
    (:func:`_render_image_context`), so its ``Dockerfile`` and ``config.yml``
    are checked too — a build produces both copies in one atomic swap, so a
    repo with one and not the other was built by a version that did not know
    about images, and the container build would fail with a docker error about
    a missing path instead of a sentence naming the render.

    A complete render still has its model selection checked. The web terminal
    runs the same resolver strict at startup, so a model the profile has since
    moved away from would ship a container that crash-loops behind the reverse
    proxy and surface only as a 502. That check reads the render's own render
    but expands ``${VAR}`` against the REPO root's ``.env``: renders are
    rendered into the disposable build zone and a deployment keeps its secrets
    at its root, so resolving against the render would leave a
    ``default_model: ${OPS_MODEL}`` as a literal placeholder and refuse a
    render the deployment can in fact serve.

    Args:
        render_name: The render whose render is being checked.
        project_path: Its rendered project directory under ``build/``.
        repo_root: The deployment repo — the zone holding ``.env``.

    Raises:
        ValueError: Naming the render, the directory, and the way out.
    """
    context = _render_image_context(project_path)
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
            f"Render {render_name!r} has a partial render at "
            f"{project_path} (missing {'; '.join(where)}). "
            "Re-render it with `osprey build`, which replaces build/ whole."
        )

    try:
        load_provider_spec(project_path, env_dir=repo_root)
    except ValueError as e:
        raise ValueError(
            f"Render {render_name!r} render at {project_path} has a "
            f"model configuration its provider cannot serve:\n  {e}\n"
            "Fix the model in this deployment's profile — its profile.yml or "
            f"the render's own delta — and re-render with `osprey build`."
        ) from e


def verify_renders(
    config: dict,
    resolved_users: list[dict],
    repo_root: Path,
    *,
    serves_roster: bool = True,
) -> None:
    """Check every referenced tier and app has a usable rendered project, or refuse.

    Every render :func:`build_render_images` is about to build needs a
    rendered project on disk: its ``Dockerfile`` and ``config.yml`` are the
    image build context. ``osprey build`` writes them, one per file in
    ``tiers/`` and ``apps/`` (:func:`osprey.cli.build_cmd._render_projects`),
    and a start writes none: this runs BEFORE the image builds so a gap is a
    refusal with a remedy rather than an opaque failure inside a container build.

    A start must never render the missing ones itself, and the absence is the
    point. ``build/`` is a complete account of what a deploy will run, so a
    render appearing at start time makes that account false exactly when it
    matters: the operator whose tier file changed would get a fresh tier standing
    beside a deployment rendered from the earlier profile. The drift gate
    already refuses that start (the tier and app files are folded into the build
    fingerprint), so the honest reading of a missing render is "this build is
    stale or partial", and the honest remedy is the one that replaces ``build/``
    whole.

    Operates on exactly :func:`_referenced_renders`'s distinct set, the same
    renders that will be built, so what is checked and what is built never
    diverge. The roster decides which renders are deployed while ``tiers/`` and
    ``apps/`` decide which exist, so a repo may hold files nothing references;
    those are built and simply not checked here.

    For each:

    * **Directory present and complete**: accepted, after
      :func:`_check_existing_render` confirms it will actually come up.
    * **Directory present but incomplete**: refused there; see that function.
    * **Directory absent**: refused here. The file it is built from is checked
      first, because a render whose file is gone or does not anchor at this
      repo would send the operator through a build that succeeds and a start
      that refuses again.

    :param config: Raw deploy config, forwarded to :func:`_referenced_renders`
        (which reads ``modules.web_terminals.renders``).
    :param resolved_users: :func:`osprey.deployment.web_terminals.roster.resolve_cards`'s
        output for this deploy.
    :param repo_root: The deployment repo: the directory holding
        ``profile.yml``, ``tiers/``, ``apps/`` and ``build/``. It IS the profile
        root.
    :param serves_roster: Forwarded to :func:`_referenced_renders`.
    :raises ValueError: A referenced render that is absent, partial, or whose
        model its provider cannot serve; and, on the absent path, a render whose
        login file is missing or does not anchor at ``repo_root``.
    """
    for unit in _referenced_renders(config, resolved_users, serves_roster=serves_roster):
        render_name = unit["render"]
        project_path = Path(unit["project_path"])

        if project_path.exists():
            _check_existing_render(render_name, project_path, repo_root)
            continue

        build_profile = unit["build_profile"]
        if not isinstance(build_profile, str) or not build_profile:
            raise ValueError(
                f"Render {render_name!r} has no rendered project at {project_path}, and "
                "its catalog entry has no build_profile naming the delta a build would "
                "render it from. " + _layer_delta_remedy(render_name, repo_root)
            )

        layer = (repo_root / build_profile).resolve()
        if not layer.is_file():
            # The config names a render whose file this repo no longer has: a
            # build cannot write it, so sending the operator to one would only
            # bring them back here.
            raise ValueError(
                f"The login {render_name!r} has no rendered project at {project_path}, "
                f"and no build can write one: the deployment's config names it, but "
                f"{layer} does not exist. Restore {layer}, or remove {render_name!r} from "
                "the roster that references it, then run `osprey build` and start again."
            )

        # Raises when no build could ever produce this render: a file that does
        # not anchor back at this repo's profile. Checked before the remedy below
        # so the operator is not sent through a build that cannot change the
        # outcome.
        source = _resolve_render_profile(build_profile, render_name, repo_root)
        _, rendered_relpath = render_project(resolve_project_name(config), source.stem)
        rendered_at = repo_root / rendered_relpath

        raise ValueError(
            f"The login {render_name!r} has no rendered project at {project_path}. "
            "Logins are rendered by `osprey build`, never by a start: "
            "`osprey up` runs what the last build produced. Run `osprey build` in "
            f"{repo_root} and start again.\n"
            f"A build of this repo renders {build_profile} to {rendered_at}. If that "
            "directory is there and this one is not, the deployment's config was "
            "rendered by an older build: `osprey build` writes each render's project "
            "and project_path from project_name, and a fresh build makes them agree."
        )


def build_render_images(
    config: dict,
    resolved_users: list[dict],
    dev_mode: bool,
    env: dict,
    *,
    serves_roster: bool = True,
) -> None:
    """Build every REFERENCED render's ``<project>:local`` image (local mode only).

    Generalizes :func:`_build_project_image`'s local-build pattern (build
    context + ``-f`` Dockerfile + ``OSPREY_PIP_SPEC`` build-arg + dev-wheel
    staging) from the single dispatch-worker project image to an arbitrary
    number of render project directories — one build per DISTINCT render
    :func:`_referenced_renders` finds in ``resolved_users`` and the dispatch
    tier, even when several users share it, and at most one per tag across
    every call inside one :func:`render_build_session`. :func:`_build_project_image` itself is untouched:
    the dispatch worker's ``<project>:local`` image is a different tag,
    disjoint from every render tag this function produces — a disjointness
    the ``render_project_shadows_worker_image`` lint rule enforces now that
    render tags carry no ``-<render>`` suffix of their own — and is built
    independently.

    No-op when ``modules.web_terminals.image_source`` is not ``"local"``
    (the default, ``"registry"``): a registry-mode deploy pulls every
    render's image instead — nothing here to build. In local mode, a
    missing ``modules.web_terminals.renders`` projection or
    ``default_tier`` raises the same refusal
    :func:`osprey.deployment.web_terminals.roster.resolve_cards` raises
    under ``strict=True`` in every image source — ``osprey up``
    never runs lint, so this fail-closed check must live on the build path
    too, not only on whichever render/resolve call happened to run first.

    Each build's context is the render's own container repo — the ``.image``
    sibling of the ``project_path`` render, see :func:`_render_image_context` —
    its ``CLAUDE_CLI_VERSION`` build-arg comes from THAT project's own
    ``config.yml`` (never the facility config — see
    :func:`_resolve_render_claude_cli_version`), and under ``dev_mode`` a
    locally-built wheel is staged into that same context and removed
    afterward, exactly mirroring :func:`_build_project_image`'s per-context
    dev-wheel convention (so a later non-dev render build's wheel-drop
    branch, which fires on any ``*.whl`` in ITS OWN context, never sees a
    stale wheel from this run). The ``OSPREY_DEV=1`` build-arg is passed only
    when that render's staging actually succeeded — a failed wheel build
    keeps the pinned install fail-loud.

    :param config: Raw deploy config (the facility config being deployed —
        read for ``modules.web_terminals`` and for
        :func:`resolve_project_name`'s ``com.osprey.project`` label value).
    :param resolved_users: :func:`osprey.deployment.web_terminals.roster.resolve_cards`'s
        output for this deploy.
    :param dev_mode: Whether ``--dev`` was passed (stage a local wheel into
        each render's build context).
    :param env: Environment for the build subprocesses.
    :param serves_roster: Forwarded to :func:`_referenced_renders`.
    :raises ValueError: Local mode with no ``renders`` projection or no
        ``default_tier`` configured.
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if effective_image_source(web_terminals) != "local":
        return  # registry mode pulls every render's image; nothing to build

    renders_raw = web_terminals.get("renders")
    renders: dict = renders_raw if isinstance(renders_raw, dict) else {}
    default_tier = web_terminals.get("default_tier")
    if not renders or not isinstance(default_tier, str) or not default_tier:
        raise ValueError(RENDER_CATALOG_REQUIRED)

    referenced = _referenced_renders(config, resolved_users, serves_roster=serves_roster)
    if not referenced:
        return

    runtime = get_runtime_command(config)[0]
    project_label = resolve_project_name(config)
    # The deployment repo whose var/logs/ takes each build's spooled output.
    repo_root = resolve_repo_root(config)

    # Renders sharing one render (same catalog `project` AND `project_path`)
    # resolve to the same tag and the same content — one build serves them all.
    # Inside a render_build_session the record spans every call of the start.
    session_tags = _SESSION_BUILT_TAGS.get()
    built_tags: set[str] = session_tags if session_tags is not None else set()

    for unit in referenced:
        render_name = unit["render"]
        if unit["image"] in built_tags:
            logger.debug(
                "Render %r shares the already-built image %r; skipping its build.",
                render_name,
                unit["image"],
            )
            continue
        # The container repo the build renders for this render — NOT the flat
        # host render the catalog names. See _render_image_context.
        context = str(_render_image_context(unit["project_path"]))

        # OSPREY_DEV=1 (the pin-relaxing build arg) is passed only when the
        # wheel actually landed in THIS render's context: on a failed
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
                    "Dev-wheel staging failed for render %r; building without "
                    "OSPREY_DEV so the Dockerfile keeps its pinned "
                    "osprey-framework install.",
                    render_name,
                )

        cmd: list[str] = []
        try:
            image_tag = unit["image"]
            cmd = _render_image_build_cmd(
                runtime,
                context,
                image_tag,
                project_label,
                config,
                dev_mode and wheel_staged,
            )
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
            # idempotent, so the per-render loop declares it each pass.
            _report_group("renders")
            with (report := single_image_build_reporter(image_tag)):
                run_captured(
                    cmd,
                    env=env,
                    spool_name=f"build-render-{image_tag.split(':', 1)[0]}",
                    repo_root=repo_root,
                    on_line=report,
                )
            # One line per image, after its own build: the slowest step of a
            # local-mode deploy, and the only progress an operator gets while
            # a handful of multi-minute builds run one after another.
            _report_step(f"render image {image_tag}")
            built_tags.add(image_tag)
        finally:
            # Remove BOTH staged artifacts (wheel + requirements manifest) so
            # neither can poison a later non-dev build in this context.
            for artifact in staged_artifacts:
                try:
                    artifact.unlink()
                except OSError:
                    logger.warning("Could not remove staged dev artifact %s", artifact)
            # Same treatment for the site CA the build args stage here.
            from osprey.deployment.container_lifecycle import clear_staged_site_ca

            clear_staged_site_ca(cmd, context)
