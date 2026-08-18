"""Render-tree persistence for the build pipeline.

Everything that writes into the freshly rendered tree that isn't a service
injector: config overrides, convention-tree application + its
scaffold-ownership registration, and persisting profile MCP servers / custom
categories into ``config.yml``.

Nothing here preserves anything across a build, and nothing here makes a git
repository. The render is ``build/`` — output, wiped and re-made whole by every
build — and the durable state sits outside it, in the repo's own ``.env``,
``var/`` and ``.git``. A pre-clear that preserves is therefore meaningless
(there is nothing in the render to keep) and a nested repository is actively
wrong (the repo is the git boundary).
"""

from __future__ import annotations

import shutil
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .build_profile_model import BuildProfile
    from .profile_conventions import ConventionCopy

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .phase_reporter import report_step

logger = get_logger("build")


def _apply_config_overrides(project_path: Path, config_dict: dict[str, Any]) -> None:
    """Apply dot-notation config overrides to the project's config.yml."""
    from osprey.utils.config_writer import config_update_fields

    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found at %s. Skipping the config overrides.", config_path)
        return
    config_update_fields(config_path, config_dict)


@dataclass
class ConventionApplication:
    """The outcome of applying a profile's convention tree to a project.

    Attributes:
        by_category: Number of copies made per convention source directory.
        seeded_users: Roster users whose profile context directory was created.
        departed_users: Profile context directories with no roster user left.
        owned_artifacts: ``scaffold.user_owned`` names for what the conventions
            actually copied (``rules/facility-ops``, ``skills/orbit-check``,
            ``services/foo`` — skills and services as whole directories, per
            their shape). Ownership registration works from this list — never
            from a directory scan, which would claim framework artifacts
            sharing the destination. Already post-exclude: a copy the resolved
            profile excluded is neither made nor listed.
    """

    by_category: dict[str, int] = field(default_factory=dict)
    seeded_users: list[str] = field(default_factory=list)
    departed_users: list[str] = field(default_factory=list)
    owned_artifacts: list[str] = field(default_factory=list)

    @property
    def copied(self) -> int:
        """Total number of files and directories copied."""
        return sum(self.by_category.values())


def _resolve_context_roster(project_path: Path) -> list[str]:
    """Read the web-terminal roster from the built project's resolved config.

    The roster is read back from ``config.yml`` rather than from the profile's
    ``config:`` block so it reflects everything that shaped it — app template,
    preset, profile, and ``--set`` overrides, all already applied by the time
    the conventions land. A disabled module has no roster: a persona build
    switches ``modules.web_terminals`` off, which is what makes it skip
    per-user context entirely.

    Where the subtree comes from is this function's own; deriving names from it
    is :func:`~osprey.deployment.web_terminals.personas.roster_user_names`, the
    same call ``osprey init`` makes when it seeds the per-user context
    slots this copy fills.
    """
    from osprey.deployment.web_terminals.personas import as_dict, roster_user_names
    from osprey.utils.config_writer import _load

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return []

    modules = as_dict(as_dict(_load(config_path)).get("modules"))
    return roster_user_names(modules.get("web_terminals"))


def _profile_known_root_entries(profile: BuildProfile, profile_path: Path | None) -> list[str]:
    """Profile-root entry names this profile legitimizes beyond the defaults.

    Feeds the unknown-root-entry warning's ``extra_known``: the profile's own
    filename plus the first path component of every profile-relative path the
    profile declares — ``data:``, ``env.file``, ``dispatch.triggers``, and
    each ``services.*.template``. A path that points above the profile root
    contributes nothing (its first component is not a root entry), and a
    bundled ``dispatch.triggers`` name harmlessly exempts a directory that
    does not exist.
    """
    known: list[str] = []
    if profile_path is not None:
        known.append(profile_path.name)

    candidates: list[str | None] = [profile.data, profile.env.file]
    if profile.dispatch is not None:
        candidates.append(profile.dispatch.triggers)
    candidates.extend(service.template for service in profile.services.values())

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        parts = PurePosixPath(candidate.strip()).parts
        if parts and parts[0] != ".." and parts[0] not in known:
            known.append(parts[0])
    return known


def _artifact_name(copy: ConventionCopy) -> str:
    """Category-qualified name of a convention artifact (``agents/orbit-writer``).

    The vocabulary persona ``exclude:`` entries use to name convention-dir
    artifacts. The name keeps the full path below the convention's destination
    — markdown categories legitimately nest (``commands/osprey/scan``), and a
    basename would collapse same-stem artifacts in different namespaces — so
    for the ownable classes it *is* the ownership canonical, taken from
    :func:`ownership_canonical` rather than re-derived: one naming scheme, not
    two. Mirror files keep their full destination path for the same reason.
    """
    from .profile_conventions import PROJECT_MIRROR_DIR, convention_for

    if copy.category == PROJECT_MIRROR_DIR:
        return f"{copy.category}/{copy.destination}"
    canonical = ownership_canonical(copy)
    if canonical is not None:
        return canonical
    # What is left is the classes outside the ownership system — MCP servers
    # and per-user context (directories per user, plus the base.md baseline as
    # the one file copy) — so the only thing separating name from destination
    # is the source directory's own spelling (``mcp_servers/`` for
    # ``_mcp_servers/``).
    convention = convention_for(copy.category)
    assert convention is not None  # planned copies always come from the table
    rel = PurePosixPath(copy.destination).relative_to(convention.destination).as_posix()
    return f"{copy.category}/{rel}"


def ownership_canonical(copy: ConventionCopy) -> str | None:
    """``scaffold.user_owned`` name for a convention copy, or ``None``.

    Ownership follows the DESTINATION, not the channel: anything landing under
    ``.claude/`` — including a ``project/`` mirror file that reaches there —
    is owned, because that is the tree a later render could rewrite or prune.
    ``services/`` directories are owned for the same reason. Everything else
    is outside the ownership system: MCP servers register through
    ``claude_code.servers``, per-user context (its ``base.md`` baseline
    included) is re-copied from the profile on every build after the framework
    installs its fallback, and mirror files landing outside ``.claude/`` are
    likewise final once the copy loop has run — no post-build render rewrites
    or prunes anything outside ``.claude/``, so there is nothing there for
    ownership to defend.

    This function answers only *whether* a copy is owned; how the name is
    spelled is :func:`~osprey.cli.profile_conventions.ownership_name`, the
    same rule the regen/prune reader applies, so writer and reader cannot
    disagree about what a registered name looks like.

    Public API: the web terminal's Scaffold Gallery derives ownership names
    through this same function so the spelling cannot drift from the build's.
    """
    from .profile_conventions import ownership_name

    dest = copy.destination
    if dest.startswith(".claude/") or copy.category == "services":
        return ownership_name(dest, is_directory=copy.is_directory)
    return None


def _apply_conventions(
    profile_dir: Path,
    project_path: Path,
    context_users: list[str] | None = None,
    *,
    extra_known: Iterable[str] = (),
    excluded: Collection[str] = frozenset(),
) -> ConventionApplication:
    """Apply the profile's convention directories into the built project.

    Each convention directory has exactly one destination (see
    :mod:`osprey.cli.profile_conventions`): markdown artifacts copy file by
    file, skills / services / MCP servers / per-user context copy as whole
    directories, and ``project/`` mirrors verbatim onto the project root.
    Directory copies *replace* their destination wholesale (``rmtree`` +
    ``copytree``) — nothing from a previous build survives inside one. Copies
    dereference symlinks and preserve mode and mtime (``copy2`` semantics);
    source validation rejects symlinks it sees escaping the profile, but its
    scan does not recurse into symlinked directories — treat it as pipeline
    coherence, not containment. The hard boundaries are the package-directory
    refusal below and the per-copy project-root containment check.

    Per-user context is the one roster-derived category. A roster user with no
    profile directory gets one seeded (in the *profile* — it is the source of
    truth, and the operator fills it in there); a profile directory for someone
    no longer on the roster is warned about and skipped, never copied and never
    deleted.

    The profile root must be operator-owned. A directory inside the installed
    osprey package — where ``--profile <bundled-preset>.yml`` would point —
    is refused by construction, regardless of how the build was invoked:
    reading conventions there would pick up neighbouring presets' files, and
    seeding per-user context would write into the wheel. The preset hash makes
    the same judgment by folding no convention material
    (:func:`~osprey.cli.build_profile_merge.compute_preset_hash`).

    Args:
        profile_dir: The profile root.
        project_path: Root of the built project.
        context_users: Resolved web-terminal roster. ``None`` or empty skips
            the per-user category.
        extra_known: Profile-root entry names the caller knows about beyond
            the defaults (a non-default ``data:`` directory or profile
            filename), exempted from the unknown-entry warning.
        excluded: Category-qualified artifact names a resolved persona delta
            excludes (``agents/orbit-writer``). An excluded copy is skipped
            entirely — whatever it would have shadowed keeps the framework
            render — and never enters ``owned_artifacts``.

    Returns:
        What was applied, for the build's progress reporting and ownership
        registration.

    Raises:
        BuildProfileError: If any convention source is misshapen or the
            ``project/`` mirror writes a build-owned path.
    """
    import osprey

    from .profile_conventions import (
        CONVENTION_DIRS,
        PER_USER_CONTEXT_DIRNAME,
        PROJECT_MIRROR_DIR,
        partition_context_users,
        plan_convention_copies,
        validate_convention_sources,
        warn_unknown_root_entries,
    )

    package_root = Path(osprey.__file__).resolve().parent
    if profile_dir.resolve().is_relative_to(package_root):
        raise BuildProfileError(
            f"Profile directory {profile_dir} is inside the installed osprey package — "
            "the build never reads convention material from, or seeds context into, "
            "package directories. Materialize an operator-owned profile first: "
            "osprey init <dir> --preset <name>."
        )

    warn_unknown_root_entries(profile_dir, extra_known)
    validate_convention_sources(profile_dir)

    per_user_source = PER_USER_CONTEXT_DIRNAME
    result = ConventionApplication()
    roster = context_users or []
    if roster:
        _, missing, departed = partition_context_users(profile_dir, roster)
        result.departed_users = departed
        result.seeded_users = missing
        for user in missing:
            seeded = profile_dir / per_user_source / user
            seeded.mkdir(parents=True, exist_ok=True)
            # An empty directory does not survive a git checkout, and a
            # profile is something operators keep in version control.
            (seeded / ".gitkeep").touch()

    entry_nouns = {convention.source: convention.entry_noun for convention in CONVENTION_DIRS}
    excluded_names = frozenset(excluded)
    project_root = project_path.resolve()
    for copy in plan_convention_copies(profile_dir, context_users=roster or None):
        name = _artifact_name(copy)
        if name in excluded_names:
            # The resolved persona excludes it: nothing is copied, so whatever
            # it would have shadowed keeps the framework render.
            logger.debug("Convention: %s excluded by the resolved profile", name)
            continue

        destination = (project_path / copy.destination).resolve()
        if not destination.is_relative_to(project_root):
            # Reachable when a destination path already exists as a symlink
            # pointing elsewhere (a previous build, or a hand-crafted project,
            # left e.g. `.claude/rules` linking outside the tree): resolve()
            # follows it, and a build must never write outside the project it
            # was asked to build.
            raise BuildProfileError(
                f"Convention destination escapes the project root: {copy.destination}"
            )

        shadowed = destination.exists()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy.is_directory:
            if shadowed:
                shutil.rmtree(destination)
            shutil.copytree(copy.source, destination)
        else:
            shutil.copy2(copy.source, destination)

        if shadowed:
            # Anything already at the destination was rendered by this build
            # (step 11 runs on a fresh or force-cleared tree).
            logger.debug(
                "  ↷ profile overrides framework %s '%s'", entry_nouns[copy.category], name
            )
        canonical = ownership_canonical(copy)
        if canonical is not None:
            result.owned_artifacts.append(canonical)

        result.by_category[copy.category] = result.by_category.get(copy.category, 0) + 1
        logger.debug("Convention: %s/… → %s", copy.category, copy.destination)

    if result.departed_users:
        logger.warning(
            "  %s/ has context for %d user(s) not on the roster: %s\n"
            "     They were skipped. Add them to modules.web_terminals.users, or remove "
            "the directory from the profile.",
            per_user_source,
            len(result.departed_users),
            ", ".join(result.departed_users),
        )
    if result.seeded_users:
        # A write into the operator's own profile, outside build/: it stays in
        # the default view, as a step under the render phase.
        report_step(f"seeded {len(result.seeded_users)} empty context dir(s) in the profile")
        logger.debug(
            "Seeded empty context directories: %s",
            ", ".join(f"{per_user_source}/{user}" for user in result.seeded_users),
        )
    if PROJECT_MIRROR_DIR in result.by_category:
        # As above: this one lands on the project root.
        report_step(
            f"mirrored {result.by_category[PROJECT_MIRROR_DIR]} file(s) onto the project root"
        )
    return result


def _register_convention_artifacts(project_path: Path, applied: ConventionApplication) -> int:
    """Register applied convention artifacts as user_owned in config.yml.

    The single-writer contract: a profile convention file wins over regen —
    :func:`~osprey.cli.templates.claude_code.is_user_owned` skips it and prune
    never unlinks it — and the Scaffold Gallery shows it as user-owned rather
    than "untracked." That holds for every ownable class alike: markdown
    artifacts per file, skills and services as whole directories (their shape).

    Registration works strictly from what the conventions copied
    (:attr:`ConventionApplication.owned_artifacts`) — never from scanning the
    destination directories, which also hold framework-rendered artifacts that
    must stay framework-owned.

    Deliberately writes ``config.yml`` only — no ``.osprey-manifest.json``
    ``user_owned`` entry, even for artifacts shadowing a catalog-listed
    framework render, where a ``framework_hash`` could be computed. The
    profile, not the project, is where a shadowed artifact is reconciled when
    the framework version moves; a claim-time drift signal would presume the
    project holds the durable edit. ``check_user_owned_drift`` therefore stays
    silent for convention-derived ownership by design.
    """
    from osprey.services.build_artifacts.ownership import update_config_add_user_owned

    if not (project_path / "config.yml").exists():
        return 0

    registered = 0
    for name in applied.owned_artifacts:
        if update_config_add_user_owned(project_path, name):
            registered += 1

    return registered


def _persist_mcp_servers(project_path: Path, mcp_servers: dict[str, Any]) -> None:
    """Persist profile MCP server definitions into config.yml's claude_code.servers.

    Servers are written in the format that ``_custom_server_from_spec()`` parses,
    so ``regenerate_claude_code()`` can reconstruct them into the rendered
    ``.mcp.json`` and ``settings.json``.  Placeholders like ``{project_root}``
    are preserved as-is — resolution happens during regen.
    """
    from osprey.utils.config_writer import _load, _save, anchored_put

    from .build_profile import McpServerDef

    config_path = project_path / "config.yml"
    data = _load(config_path)

    # Collect the server specs first, then register them via anchored_put:
    # writing an empty ``servers:`` map and filling it afterwards would leave
    # any section comment trailing ``claude_code:`` anchored above the new
    # entries (rendering them under the next section's banner).
    specs: dict[str, dict[str, Any]] = {}
    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue

        spec: dict[str, Any] = {}
        if server.url:
            # Written explicitly (even for the default) so the rendered
            # config.yml states its wire transport instead of implying it —
            # _custom_server_from_spec() reads this key back during regen.
            spec["transport"] = server.transport
            spec["url"] = server.url
        else:
            # Stdio servers get no transport key: launching via 'command' IS
            # the transport, and load_profile rejects a declared one.
            if server.command:
                spec["command"] = server.command
            if server.args:
                spec["args"] = list(server.args)
            if server.env:
                spec["env"] = dict(server.env)
        if server.port is not None and server.url:
            # Emit a derived network block so non-Claude consumers
            # (compose-port checkers, integration-tests probes) can read
            # host/docker URLs without re-deriving them. The endpoint path
            # follows the declared url (an SSE server's stream is not /mcp).
            # NOTE: docker_url uses the MCP server's YAML key (`name`) as the
            # container hostname. This assumes the operator names the
            # docker-compose service identically to the mcp_servers entry
            # (e.g. mcp_servers.matlab → service: matlab). If they diverge,
            # docker_url will point at a non-existent host.
            path = urlparse(server.url).path or "/mcp"
            spec["network"] = {
                "port": int(server.port),
                "host_url": f"http://localhost:{server.port}{path}",
                "docker_url": f"http://{name}:{server.port}{path}",
            }
        if server.permissions:
            spec["permissions"] = dict(server.permissions)

        specs[name] = spec

    if specs:
        if "claude_code" not in data:
            from ruamel.yaml import CommentedMap

            data["claude_code"] = CommentedMap()
        cc = data["claude_code"]
        if "servers" in cc:
            for name, spec in specs.items():
                anchored_put(cc["servers"], name, spec)
        else:
            anchored_put(cc, "servers", specs)

    _save(config_path, data)


def _persist_artifact_server(project_path: Path, overrides: dict[str, Any]) -> None:
    """Merge profile ``artifact_server`` overrides into config.yml's block.

    Scalar subkeys (``host``, ``port``, ``auto_launch``) replace the rendered
    defaults; ``categories`` entries are written into the block's
    ``categories`` submap.
    """
    from ruamel.yaml import CommentedMap

    from osprey.utils.config_writer import _load, _save, anchored_put

    config_path = project_path / "config.yml"
    data = _load(config_path)

    if "artifact_server" not in data:
        # Root-level append: new top-level sections land at the file tail
        # by design (the template documents appended sections there).
        data["artifact_server"] = CommentedMap()
    block = data["artifact_server"]

    for key, value in overrides.items():
        if key != "categories":
            anchored_put(block, key, value)
    categories = {
        key: {"label": spec["label"], "color": spec["color"]}
        for key, spec in overrides.get("categories", {}).items()
    }
    if categories:
        if "categories" in block:
            for key, entry in categories.items():
                anchored_put(block["categories"], key, entry)
        else:
            anchored_put(block, "categories", categories)

    _save(config_path, data)
