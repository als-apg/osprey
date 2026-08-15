"""Profile-directory conventions — the canonical profile → project mapping.

A build profile is the source of truth for the project it builds, so every
artifact class it can carry has exactly one channel into the project. This
module is that channel table, plus the validation that keeps it honest:

* :data:`CONVENTION_DIRS` — the mapping itself (``rules/`` → ``.claude/rules/``
  and friends), including ``project/``, the verbatim path mirror that exists as
  an escape hatch for anything without a dedicated channel.
* :func:`validate_convention_sources` — up-front source validation, so a
  misshapen convention directory fails with an actionable message before the
  build writes anything.
* :func:`reserved_path_channel` — build-owned destinations the ``project/``
  mirror may not write, each naming the channel that *does* own them. These
  enforce pipeline coherence (exactly one writer per artifact), not sandboxing:
  the profile is operator-trusted.
* :func:`plan_convention_copies` — the pure source → destination plan a build
  carries out.
* :func:`ownership_name` — the destination → ``scaffold.user_owned`` name rule,
  shared by the build that registers ownership and the render that honors it.

Everything here is pure: it reads the profile tree and returns plans, names, or
errors. Nothing writes into a project.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path, PurePosixPath

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger
from osprey.utils.workspace import BUILD_DIR_NAME, STATE_DIR_NAME

logger = get_logger("build")


class EntryShape(Enum):
    """How a convention directory's entries map into the project.

    Attributes:
        MARKDOWN: Individual markdown artifacts, copied file by file (nested
            subdirectories are preserved — Claude Code namespaces commands that
            way). Anything that is not a ``.md`` file is rejected up front.
        FILE: Individual files copied file by file, with no extension
            constraint — the artifact class is defined by its destination, not
            by a markup format. Hooks are executable ``.py`` scripts named by
            filename in ``settings.json``, so the name a hook is known by
            includes its suffix.
        DIRECTORY: One whole directory per named thing (a skill, an MCP server,
            a service, a web-terminal user), copied as a unit so the artifact's
            own shape — not this module — decides what belongs in it.
        MIRROR: A verbatim path mirror onto the project root.
    """

    MARKDOWN = "markdown"
    FILE = "file"
    DIRECTORY = "directory"
    MIRROR = "mirror"


@dataclass(frozen=True)
class ConventionDir:
    """One row of the profile → project mapping table.

    Attributes:
        source: Directory name at the profile root.
        destination: Project-relative destination prefix; empty for the
            ``project/`` mirror, which lands on the project root itself.
        shape: How entries below ``source`` map into ``destination``.
        entry_noun: What one entry *is*, for error messages ("skill", "service").
        description: Human-readable purpose, for docs and error messages.
        per_user: Entries are web-terminal users resolved from the build's
            roster rather than free-form names, so the build seeds missing ones
            and skips departed ones (see :func:`partition_context_users`).
    """

    source: str
    destination: str
    shape: EntryShape
    entry_noun: str
    description: str
    per_user: bool = False


CONVENTION_DIRS: tuple[ConventionDir, ...] = (
    ConventionDir(
        source="rules",
        destination=".claude/rules",
        shape=EntryShape.MARKDOWN,
        entry_noun="rule",
        description="Agent behavior rules",
    ),
    ConventionDir(
        source="skills",
        destination=".claude/skills",
        shape=EntryShape.DIRECTORY,
        entry_noun="skill",
        description="Agent skills (one directory per skill)",
    ),
    ConventionDir(
        source="agents",
        destination=".claude/agents",
        shape=EntryShape.MARKDOWN,
        entry_noun="agent",
        description="Subagent definitions",
    ),
    ConventionDir(
        source="commands",
        destination=".claude/commands",
        shape=EntryShape.MARKDOWN,
        entry_noun="command",
        description="Slash commands",
    ),
    ConventionDir(
        source="output-styles",
        destination=".claude/output-styles",
        shape=EntryShape.MARKDOWN,
        entry_noun="output style",
        description="Output styles",
    ),
    ConventionDir(
        source="hooks",
        destination=".claude/hooks",
        shape=EntryShape.FILE,
        entry_noun="hook",
        description="Claude Code hook scripts",
    ),
    ConventionDir(
        source="web-terminal-context",
        destination="docker/web-terminal-context",
        shape=EntryShape.DIRECTORY,
        entry_noun="web-terminal user",
        description="Per-user web-terminal context (one directory per user)",
        per_user=True,
    ),
    ConventionDir(
        source="mcp_servers",
        destination="_mcp_servers",
        shape=EntryShape.DIRECTORY,
        entry_noun="MCP server",
        description="Facility MCP server sources (one directory per server)",
    ),
    ConventionDir(
        source="services",
        destination="services",
        shape=EntryShape.DIRECTORY,
        entry_noun="service",
        description="Facility-owned compose services (one directory per service)",
    ),
    ConventionDir(
        source="project",
        destination="",
        shape=EntryShape.MIRROR,
        entry_noun="file",
        description="Verbatim mirror onto the project root (escape hatch)",
    ),
)

_BY_SOURCE: dict[str, ConventionDir] = {c.source: c for c in CONVENTION_DIRS}

_PER_USER_CONVENTION: ConventionDir = next(c for c in CONVENTION_DIRS if c.per_user)

#: Profile-root directory holding one subdirectory of context per web-terminal
#: user. Public because ``osprey init`` seeds those subdirectories from
#: the roster and must name the same directory this table maps.
PER_USER_CONTEXT_DIRNAME: str = _PER_USER_CONVENTION.source

CONVENTION_SOURCES: tuple[str, ...] = tuple(c.source for c in CONVENTION_DIRS)

PROJECT_MIRROR_DIR = "project"

#: The disposable output zone. Every ``osprey build`` wipes and re-renders it,
#: so nothing durable may live there. Spelled here under this module's own name
#: — the repo root it judges is the directory the zone sits in — but ALIASED to
#: :data:`osprey.utils.workspace.BUILD_DIR_NAME` rather than restated. Three
#: modules named this directory with three independent string literals, which
#: is three chances for the zone layout to disagree with itself; there is now
#: one literal and two names for it.
BUILD_OUTPUT_DIR = BUILD_DIR_NAME

#: The durable state zone (``var/agent_data``, ``var/audit``) — git-ignored,
#: never rendered, and never wiped by a build. Aliased for the same reason.
STATE_DIR = STATE_DIR_NAME

#: SOURCE zone: tracked, user-edited. ``profile.yml`` plus the material it
#: names, and the CI files a deployment ships (``scripts/verify.sh`` beside
#: ``ci-extra.yml``; ``.gitlab-ci.yml`` is dot-prefixed and exempt already).
_SOURCE_ZONE_ENTRIES: frozenset[str] = frozenset(
    {"profile.yml", "triggers.yml", "data", "personas", "scripts", "ci-extra.yml"}
)

#: SECRETS zone: git-ignored, durable. Dot-prefixed, so already exempt from the
#: warning — listed anyway so this table reads as the whole layout rather than
#: leaving one zone to the dotfile exemption by coincidence.
_SECRETS_ZONE_ENTRIES: frozenset[str] = frozenset({".env", ".env.example"})

#: OUTPUT and STATE zones: git-ignored and generated. Both sit at the repo root
#: the profile is read from, so the warning must not read them as typos of a
#: convention directory — a build would otherwise flag its own output.
_GENERATED_ZONE_ENTRIES: frozenset[str] = frozenset({BUILD_OUTPUT_DIR, STATE_DIR})

#: Repo-root entries that are neither convention directories nor typos — the
#: four-zone layout in full, plus every convention source. The repo root *is*
#: the profile root, so a deployment's own generated zones are entries this
#: module has to recognize rather than warn about (SC-9).
KNOWN_ROOT_ENTRIES: frozenset[str] = (
    _SOURCE_ZONE_ENTRIES
    | _SECRETS_ZONE_ENTRIES
    | _GENERATED_ZONE_ENTRIES
    | frozenset(CONVENTION_SOURCES)
)

#: Name prefixes exempt from the unknown-root-entry warning. Dot-prefixed
#: entries (``.git/``, ``.DS_Store``) and ``docs/`` are exempt too — a profile
#: is a directory someone keeps in version control and documents.
_ROOT_WARNING_EXEMPT_PREFIXES = ("README", "LICENSE")
_ROOT_WARNING_EXEMPT_NAMES = frozenset({"docs"})


@dataclass(frozen=True)
class ReservedPath:
    """A build-owned project path the ``project/`` mirror may not write.

    Attributes:
        path: Project-relative posix path.
        channel: The profile channel that *does* own it, phrased for an error
            message ("the profile's ``config:`` block").
    """

    path: str
    channel: str


RESERVED_PROJECT_PATHS: tuple[ReservedPath, ...] = (
    ReservedPath("config.yml", "the profile's `config:` block"),
    ReservedPath(
        ".osprey-manifest.json",
        "the build itself — it stamps the manifest from the resolved profile",
    ),
    ReservedPath(
        ".claude/settings.json",
        "the profile's `config:` keys (`claude_code.permissions`, `claude_code.hooks`, "
        "`artifacts.hooks`)",
    ),
    ReservedPath(
        ".claude/hooks/hook_config.json",
        "the framework render — it is generated from the enabled `mcp_servers:` and "
        "`config.control_system.write_tools`, and it is the runtime configuration of "
        "the write-safety layer (osprey_writes_check.py reads its `write_tools`)",
    ),
    ReservedPath(".mcp.json", "the profile's `mcp_servers:` block"),
    ReservedPath(".env", "the profile's `.env` file and `env:` keys"),
    ReservedPath(".env.example", "the profile's `.env.example` file and `env:` keys"),
    ReservedPath("CLAUDE.md", "the profile's `claude_md_template:` key"),
    ReservedPath("data/simulation/channel_manifest.json", "the profile's `data/` directory"),
    ReservedPath("data/simulation/channel_limits.json", "the profile's `data/` directory"),
)

#: The table indexed by path — the exact reservations mapped to the channel
#: that owns each. Public because the declared-hook validator needs the same
#: exact-only view (the prefix rules in :func:`reserved_path_channel` reserve
#: `.claude/hooks/` itself, which is precisely where a declared hook lives).
RESERVED_PATH_CHANNELS: dict[str, str] = {r.path: r.channel for r in RESERVED_PROJECT_PATHS}

#: The reserved paths as a bare set, for the ownership rules that ask only
#: "is this path generated?" and have their own wording for the answer. Derived
#: here, beside the table, so those rules cannot disagree about what the table
#: says: the *exact* entries and nothing else. The prefix rules in
#: :func:`reserved_path_channel` say a ``.claude/`` subtree belongs to its
#: convention directory — which is what makes those artifacts ownable in the
#: first place — so a rule applying them would refuse the whole channel.
RESERVED_EXACT_PATHS: frozenset[str] = frozenset(RESERVED_PATH_CHANNELS)


@dataclass(frozen=True)
class ConventionCopy:
    """One planned copy from a profile convention directory into a project.

    Attributes:
        category: Convention source directory the copy came from.
        source: Absolute path in the profile.
        destination: Project-relative posix destination path.
        is_directory: True when the whole directory is copied as a unit.
    """

    category: str
    source: Path
    destination: str
    is_directory: bool


def convention_for(source_name: str) -> ConventionDir | None:
    """Return the convention row for a profile-root directory name, if any."""
    return _BY_SOURCE.get(source_name)


def destination_for(source_rel: str | PurePosixPath) -> str:
    """Map a profile-relative convention path to its project destination.

    Args:
        source_rel: Path relative to the profile root, e.g. ``rules/safety.md``
            or ``project/docs/runbook.md``.

    Returns:
        The project-relative posix destination (empty string for the
        ``project/`` mirror root itself).

    Raises:
        BuildProfileError: If the first segment is not a convention directory.
    """
    parts = PurePosixPath(source_rel).parts
    if not parts:
        raise BuildProfileError("Empty convention path — expected a profile-relative path.")
    convention = _BY_SOURCE.get(parts[0])
    if convention is None:
        known = ", ".join(f"{name}/" for name in CONVENTION_SOURCES)
        raise BuildProfileError(
            f"{parts[0]!r} is not a profile convention directory.\nConvention directories: {known}"
        )
    return "/".join(segment for segment in (convention.destination, *parts[1:]) if segment)


#: How a refusal names the framework render as an artifact's writer. One
#: constant because it is one answer: :func:`reserved_path_channel` returns it
#: for a path the render owns, and ``scaffold unclaim``'s message falls back to
#: it for an artifact that has no reserved channel at all. Two spellings would
#: send an operator looking for two different mechanisms.
FRAMEWORK_RENDER_CHANNEL = "the framework render — carry the change as a `config:` key instead"


@lru_cache(maxsize=1)
def _framework_rendered_outputs() -> frozenset[str]:
    """Project-relative paths the framework render owns under ``.claude/``.

    Deferred import: the build-artifact catalog pulls in the scaffold-ownership
    stack, which this module's mapping table has no need of.
    """
    from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

    catalog = BuildArtifactCatalog.default()
    return frozenset(
        artifact.output_path
        for artifact in catalog.all_artifacts()
        if artifact.output_path == "CLAUDE.md" or artifact.output_path.startswith(".claude/")
    )


def reserved_path_channel(dest_rel: str) -> str | None:
    """Return the channel owning ``dest_rel``, or ``None`` if the mirror may write it.

    Only ``.claude/`` prefixes are protected: each such subtree already has a
    dedicated convention channel, so a mirror write there would put a second
    writer on the same artifact and confuse its ownership registration.
    Convention destinations outside ``.claude/`` need no reservation — copies
    apply in sorted destination order, and a directory sorts before every path
    beneath it, so a mirror file inside one lands after (and survives) the
    wholesale directory copy.

    Args:
        dest_rel: Project-relative posix path the ``project/`` mirror would write.

    Returns:
        A phrase naming the owning channel, suitable for an error message, or
        ``None`` when nothing else writes that path.
    """
    normalized = PurePosixPath(dest_rel).as_posix()

    # ORDER IS LOAD-BEARING: an exact reservation beats the convention prefix
    # below. `.claude/hooks/hook_config.json` sits inside the `hooks/` channel's
    # destination but is generated from the resolved config, not hand-authored —
    # if the prefix loop answered first, the channel would claim it and a
    # hand-written copy would freeze the write-safety layer's own configuration.
    # Pinned by test_exact_reservation_beats_the_convention_prefix.
    exact = RESERVED_PATH_CHANNELS.get(normalized)
    if exact is not None:
        return exact

    # A .claude/ subtree with its own convention directory is that directory's
    # to write — whether or not the framework happens to render a file there.
    for convention in CONVENTION_DIRS:
        prefix = convention.destination
        if prefix.startswith(".claude/") and normalized.startswith(f"{prefix}/"):
            return f"the profile's `{convention.source}/` convention directory"

    if normalized in _framework_rendered_outputs():
        return FRAMEWORK_RENDER_CHANNEL

    return None


def convention_slot_for(dest_rel: str) -> str | None:
    """The profile-relative slot a convention directory would write ``dest_rel`` from.

    The inverse of :func:`destination_for`, and the reason a refusal can tell an
    operator where to *move* a file rather than only that it may not stay. The
    ``project/`` mirror is excluded by construction (its destination is the
    project root, which would match everything).

    Args:
        dest_rel: Project-relative posix path.

    Returns:
        ``hooks/custom.py`` for ``.claude/hooks/custom.py``, or ``None`` when no
        convention directory owns that destination.
    """
    normalized = PurePosixPath(dest_rel).as_posix()
    for convention in CONVENTION_DIRS:
        prefix = convention.destination
        if prefix and normalized.startswith(f"{prefix}/"):
            return f"{convention.source}/{normalized[len(prefix) + 1 :]}"
    return None


def ownership_name(dest_rel: str, *, is_directory: bool) -> str:
    """The ``scaffold.user_owned`` name a project-relative destination is owned under.

    The one spelling rule the ownership system has, shared by both of its
    halves: the build registers what its conventions copied under this name
    (:func:`~osprey.cli.build_persistence.ownership_canonical`), and regen and
    prune ask whether a path they are about to write is owned under it
    (:func:`~osprey.cli.templates.claude_code.is_user_owned`). Derived here or
    nowhere — a spelling that drifted between writer and reader would fail
    silently, un-owning every artifact of the affected class rather than
    raising anything.

    The name is the destination, minus two things that are not part of an
    artifact's identity: the ``.claude/`` prefix (a rule is
    ``rules/facility``), and ``.md`` on a file. Anything else keeps its
    suffix, which is what the artifact is actually known by — ``settings.json``
    wires each hook as ``.claude/hooks/<filename>``, so
    ``hooks/osprey_limits.py`` is its name. A directory keeps its name whole:
    it may legitimately end in ``.md``, and it owns every file beneath it.

    Args:
        dest_rel: Project-relative posix destination path.
        is_directory: True when the destination is a whole directory copied as
            a unit (a skill, a service).

    Returns:
        The ownership name. Whether that destination's *class* is ownable at
        all is a separate question, answered by ``ownership_canonical``.
    """
    name = dest_rel[len(".claude/") :] if dest_rel.startswith(".claude/") else dest_rel
    if not is_directory and name.endswith(".md"):
        name = name[: -len(".md")]
    return name


def _iter_files(root: Path, *, include_hidden: bool = False) -> Iterator[Path]:
    """Yield every file below ``root``.

    Dot-prefixed entries are skipped by default (``.DS_Store`` is not an
    artifact), but the ``project/`` mirror keeps them: ``.claude/settings.json``
    is exactly the kind of path the reserved list has to catch, and a
    ``.gitignore`` is a legitimate thing to mirror.
    """
    for path in sorted(root.rglob("*")):
        if not include_hidden and any(
            part.startswith(".") for part in path.relative_to(root).parts
        ):
            continue
        if path.is_file():
            yield path


def _mirror_violations(mirror_dir: Path) -> list[tuple[str, str]]:
    """Return ``(project-relative path, owning channel)`` for reserved writes."""
    violations: list[tuple[str, str]] = []
    for path in _iter_files(mirror_dir, include_hidden=True):
        rel = path.relative_to(mirror_dir).as_posix()
        channel = reserved_path_channel(rel)
        if channel is not None:
            violations.append((rel, channel))
    return violations


def _format_mirror_violations(violations: Sequence[tuple[str, str]]) -> str:
    """Report every reserved write, naming the exact move for the ones that have one.

    A channel name alone tells an operator their file is in the wrong place but
    not where to put it. Where a convention directory owns the destination the
    move is exactly derivable (:func:`convention_slot_for`), so the message
    states it as a path they can act on — this is the only guidance a profile
    that mirrored an artifact before its channel existed will get. An exactly
    reserved path gets no move advice: its reservation means no convention slot
    accepts it either, so the named channel is the only way to carry the change.
    """
    lines = [
        f"{PROJECT_MIRROR_DIR}/ mirror writes {len(violations)} build-owned "
        f"path(s) — each already has a channel:"
    ]
    for rel, channel in violations:
        lines.append(f"  {PROJECT_MIRROR_DIR}/{rel} → {rel} is written by {channel}")
        slot = None if rel in RESERVED_PATH_CHANNELS else convention_slot_for(rel)
        if slot is not None:
            lines.append(
                f"      Move it: {PROJECT_MIRROR_DIR}/{rel} → {slot} "
                "(both paths relative to the profile root)."
            )
    lines.append(
        f"Remove them from {PROJECT_MIRROR_DIR}/ and carry the change through "
        "the named channel; the mirror is for files the build does not own."
    )
    return "\n".join(lines)


def validate_project_mirror(mirror_dir: Path) -> None:
    """Reject a ``project/`` mirror that writes build-owned paths.

    Args:
        mirror_dir: The profile's ``project/`` directory (missing is fine).

    Raises:
        BuildProfileError: Listing every reserved path the mirror would write.
    """
    if not mirror_dir.is_dir():
        return
    violations = _mirror_violations(mirror_dir)
    if violations:
        raise BuildProfileError(_format_mirror_violations(violations))


def _symlink_escapes(path: Path, profile_dir: Path) -> bool:
    """True when ``path`` is a symlink resolving outside the profile directory."""
    if not path.is_symlink():
        return False
    return not path.resolve().is_relative_to(profile_dir.resolve())


def _validate_markdown_dir(convention: ConventionDir, root: Path) -> list[str]:
    errors: list[str] = []
    for path in _iter_files(root):
        rel = f"{convention.source}/{path.relative_to(root).as_posix()}"
        if path.suffix != ".md":
            errors.append(
                f"{rel} is not a markdown file: {convention.source}/ holds one "
                f".md file per {convention.entry_noun}.\n"
                f"  Put files that are copied verbatim under {PROJECT_MIRROR_DIR}/ instead."
            )
    return errors


def _validate_reserved_destinations(convention: ConventionDir, root: Path) -> list[str]:
    """Reject convention files whose destination a build-owned artifact already claims.

    A channel owns its destination *subtree*, but not every path inside one is
    the channel's to write: ``.claude/hooks/hook_config.json`` lands under
    ``hooks/`` yet is generated from the resolved config. Shipping a
    hand-written copy would be registered as user-owned like any other hook,
    which makes regen skip it — and since that file is what
    ``osprey_writes_check.py`` reads its ``write_tools`` from, a stale or empty
    copy silently disarms the write-safety layer for every later build.

    Checked per file rather than only at claim time because the two routes to
    the same frozen file are independent: a profile can simply *contain* the
    file without anyone running a claim.
    """
    errors: list[str] = []
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        source_rel = f"{convention.source}/{rel}"
        channel = RESERVED_PATH_CHANNELS.get(destination_for(source_rel))
        if channel is not None:
            errors.append(
                f"{source_rel} is generated, not authored: it lands at "
                f"{destination_for(source_rel)}, which is written by {channel}.\n"
                f"  Remove it from the profile and change the keys named above — a "
                f"copy here would be registered as yours and never regenerate."
            )
    return errors


def _validate_directory_dir(convention: ConventionDir, root: Path) -> list[str]:
    errors: list[str] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            errors.append(
                f"{convention.source}/{entry.name} is a file: {convention.source}/ "
                f"holds one directory per {convention.entry_noun}.\n"
                f"  Move it into {convention.source}/{entry.stem}/ "
                f"(the whole directory is copied as a unit)."
            )
    return errors


def validate_convention_sources(profile_dir: Path) -> None:
    """Validate every convention directory a profile carries, up front.

    Checks that each convention directory is a directory, that its entries have
    the shape the destination expects, that no entry symlinks outside the
    profile (a profile must be self-contained to be reproducible), and that the
    ``project/`` mirror writes no build-owned path.

    Args:
        profile_dir: The profile root directory.

    Raises:
        BuildProfileError: Reporting every problem found, not just the first.
    """
    errors: list[str] = []

    for convention in CONVENTION_DIRS:
        root = profile_dir / convention.source
        if not root.exists():
            continue
        if not root.is_dir():
            errors.append(
                f"{convention.source} is a file, but it names a profile "
                f"convention directory ({convention.description}).\n"
                f"  Make it a directory, or rename the file."
            )
            continue

        escaping = [
            f"{convention.source}/{path.relative_to(root).as_posix()}"
            for path in sorted(root.rglob("*"))
            if _symlink_escapes(path, profile_dir)
        ]
        errors += [
            f"{rel} is a symlink pointing outside the profile.\n"
            f"  A profile must be self-contained — copy the target in instead."
            for rel in escaping
        ]

        if convention.shape is EntryShape.MARKDOWN:
            errors += _validate_markdown_dir(convention, root)
            errors += _validate_reserved_destinations(convention, root)
        elif convention.shape is EntryShape.DIRECTORY:
            errors += _validate_directory_dir(convention, root)
        elif convention.shape is EntryShape.MIRROR:
            violations = _mirror_violations(root)
            if violations:
                errors.append(_format_mirror_violations(violations))
        else:
            # EntryShape.FILE constrains no extension — its destination, not a
            # markup format, is what defines the artifact class. What it does
            # constrain is *which* destinations: a generated file inside the
            # subtree is not the channel's to carry.
            errors += _validate_reserved_destinations(convention, root)

    if errors:
        raise BuildProfileError(
            f"Profile convention directories are invalid ({len(errors)} problem(s)):\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def partition_context_users(
    profile_dir: Path, roster: Iterable[str]
) -> tuple[list[str], list[str], list[str]]:
    """Split per-user context directories against the build's resolved roster.

    Args:
        profile_dir: The profile root directory.
        roster: Web-terminal user names the build resolved.

    Returns:
        ``(matched, missing, departed)`` — users with a context directory to
        copy, roster users whose directory the build should seed, and context
        directories for users no longer on the roster (warn and skip).
    """
    root = profile_dir / _PER_USER_CONVENTION.source
    present = (
        {
            entry.name
            for entry in root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        }
        if root.is_dir()
        else set()
    )
    wanted = set(roster)
    return sorted(wanted & present), sorted(wanted - present), sorted(present - wanted)


def plan_convention_copies(
    profile_dir: Path, *, context_users: Iterable[str] | None = None
) -> list[ConventionCopy]:
    """Plan every copy a profile's convention directories imply.

    Args:
        profile_dir: The profile root directory.
        context_users: Resolved web-terminal roster. ``None`` or empty skips
            the per-user context category entirely — a persona build disables
            ``modules.web_terminals``, so it has no roster to derive from.

    Returns:
        Planned copies, sorted by destination. Directory-shaped categories
        yield one entry per named directory; markdown and mirror categories
        yield one entry per file.
    """
    matched = (
        set(partition_context_users(profile_dir, context_users)[0])
        if context_users is not None
        else set()
    )

    copies: list[ConventionCopy] = []
    for convention in CONVENTION_DIRS:
        root = profile_dir / convention.source
        if not root.is_dir():
            continue

        if convention.shape is EntryShape.DIRECTORY:
            for entry in root.iterdir():
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                if convention.per_user and entry.name not in matched:
                    continue
                copies.append(
                    ConventionCopy(
                        category=convention.source,
                        source=entry,
                        destination=destination_for(f"{convention.source}/{entry.name}"),
                        is_directory=True,
                    )
                )
            continue

        for path in _iter_files(root, include_hidden=convention.shape is EntryShape.MIRROR):
            rel = path.relative_to(root).as_posix()
            copies.append(
                ConventionCopy(
                    category=convention.source,
                    source=path,
                    destination=destination_for(f"{convention.source}/{rel}"),
                    is_directory=False,
                )
            )

    return sorted(copies, key=lambda copy: copy.destination)


def unknown_root_entries(profile_dir: Path, extra_known: Iterable[str] = ()) -> list[str]:
    """Return profile-root entries that are neither known nor exempt.

    Exempt: dot-prefixed entries, ``README*``, ``LICENSE*``, and ``docs/`` — a
    profile is a directory someone keeps in version control and documents.

    Args:
        profile_dir: The profile root directory.
        extra_known: Additional names the caller knows about (a profile whose
            ``data:`` key or profile filename differs from the default).

    Returns:
        Sorted names, each a candidate typo of a convention directory.
    """
    known = KNOWN_ROOT_ENTRIES | set(extra_known)
    return sorted(
        entry.name
        for entry in profile_dir.iterdir()
        if entry.name not in known
        and not entry.name.startswith(".")
        and entry.name not in _ROOT_WARNING_EXEMPT_NAMES
        and not entry.name.startswith(_ROOT_WARNING_EXEMPT_PREFIXES)
    )


def warn_unknown_root_entries(profile_dir: Path, extra_known: Iterable[str] = ()) -> list[str]:
    """Warn about unrecognized repo-root entries and return their names.

    A misspelled convention directory (``rule/`` for ``rules/``) is otherwise
    silent: nothing reads it, so nothing complains, and the artifacts simply
    never reach the build.

    The other way to arrive here is not a typo at all: a facility's own
    directory (``ioc/``, ``nginx/``) sitting beside ``profile.yml``. Under the
    four-zone layout the repo root *is* the profile root, so there is nowhere
    to nest such a directory away to — the remedy is to move it into the
    channel that carries it, or to accept that nothing copies it, which for
    repo-local material is the correct outcome. The two causes read identically
    from here — one entry or twenty, all unknown — so the message names both
    remedies rather than guessing which one applies.
    """
    unknown = unknown_root_entries(profile_dir, extra_known)
    if unknown:
        logger.warning(
            "  Repo root has %d unrecognized top-level entry/entries: %s\n"
            "     Nothing copies them into the build — check for a typo.\n"
            "     Convention directories: %s\n"
            "     %s is the repo root and the profile root at once: profile.yml and\n"
            "     the material it names sit here, beside the generated %s/ and %s/\n"
            "     zones. Nothing nests — there is no profile/ subdirectory to move\n"
            "     them into.\n"
            "     If an entry is meant to reach the deployment, move it into the\n"
            "     channel that carries it — a convention directory above, or %s/ for\n"
            "     a verbatim copy. If it is repo-local material the deployment does\n"
            "     not need, leaving it here costs nothing but this warning.",
            len(unknown),
            ", ".join(unknown),
            ", ".join(f"{name}/" for name in CONVENTION_SOURCES),
            profile_dir,
            BUILD_OUTPUT_DIR,
            STATE_DIR,
            PROJECT_MIRROR_DIR,
        )
    return unknown
