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
* :func:`is_reserved_write` and :func:`is_protected_key` — the *protected set*:
  the paths and config keys a running agent may not rewrite, consulted by every
  framework writer. A separate question from the reservations above, which ask
  which build channel owns a path (see :func:`is_reserved_write`).
* :func:`is_setup_patch_capable` — the posture read shared by every gate that
  has to know whether a rendered persona can still reach the setup tool.
* :func:`plan_convention_copies` — the pure source → destination plan a build
  carries out.
* :func:`ownership_name` — the destination → ``scaffold.user_owned`` name rule,
  shared by the build that registers ownership and the render that honors it.

Everything here is pure: it reads the profile tree and returns plans, names, or
errors. Nothing writes into a project.
"""

from __future__ import annotations

import posixpath
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger
from osprey.utils.workspace import BUILD_DIR_NAME, STATE_DIR_NAME

# Free at module level: ``osprey.utils.logger`` above already pulls
# ``osprey_connectors.config`` into this module's import closure, so naming it
# here adds no load time to the CLI's lazy-command budget. Imported rather than
# restated — see :data:`PROTECTED_CONFIG_KEYS`.
from osprey_connectors.config import RUNTIME_WRITE_PATH_KEYS

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
        description="Per-user web-terminal context (one directory per user, plus a shared base.md)",
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

#: The one loose file the per-user convention directory accepts: the shared
#: baseline every seeded user's ``CLAUDE.md`` starts from. It rides the same
#: channel as the per-user directories (landing beside them at the convention's
#: destination) but is roster-independent — a profile that ships it overrides
#: the framework's fallback for every user at once. Public because ``osprey
#: init`` materializes it and seeding documents it, and all three must spell
#: the same filename.
CONTEXT_BASELINE_FILENAME: str = "base.md"


def context_baseline_slot(root: Path) -> Path:
    """The baseline slot inside a per-user context directory.

    Args:
        root: A per-user context directory — a profile's
            ``web-terminal-context/``, the framework's template copy of it, or
            an app bundle's.

    Returns:
        The path of the shared baseline in that directory, whether or not a
        file is there.

    Four sites act on this one slot — the validator that accepts it as the
    convention's only loose file, the plan that copies it, ``osprey init``
    that seeds it, and the bundle/framework lookup that picks the text to seed
    it from — and each one used to join the filename itself. They spell it here
    instead, so a slot the validator accepts is exactly the slot the build
    copies and ``osprey init`` writes.
    """
    return root / CONTEXT_BASELINE_FILENAME


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

#: The boot unit ``osprey scaffold systemd`` writes beside the profile, spelled
#: as a literal because :mod:`~osprey.cli.deploy_scaffold_templates` — which
#: owns the name as ``SYSTEMD_UNIT_NAME`` — imports *from* this module, so
#: importing it back would close a cycle. The two spellings are pinned to each
#: other by a test rather than by an import.
_SYSTEMD_UNIT_ENTRY = "osprey.service"

#: SOURCE zone: tracked, user-edited. ``profile.yml`` plus the material it
#: names, and the files a deployment's scaffolding verbs emit into the root:
#: the CI pair (``scripts/verify.sh`` beside ``ci-extra.yml``;
#: ``.gitlab-ci.yml`` is dot-prefixed and exempt already) and the systemd boot
#: unit. Those are OSPREY's own output, so a build that flagged them would be
#: warning about a file OSPREY told the operator to create.
#: ``profiles/`` holds the host-variant overlays
#: (:mod:`~osprey.cli.variant_selection`) — tracked like everything else here,
#: since which hosts a deployment has is part of what the deployment is. Only
#: the *choice* between them is host-local, and that lives in an ignored
#: dot-file the warning never sees.
_SOURCE_ZONE_ENTRIES: frozenset[str] = frozenset(
    {
        "profile.yml",
        "triggers.yml",
        "data",
        "personas",
        "profiles",
        "scripts",
        "ci-extra.yml",
        _SYSTEMD_UNIT_ENTRY,
    }
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
    ReservedPath(
        "docker/web-terminal-context/base.md",
        "the profile's `web-terminal-context/base.md` slot — the shared baseline "
        "sits beside the per-user directories, not in the mirror",
    ),
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

#: The same exact reservations keyed by their casefolded path, for
#: :func:`is_reserved_write`'s lookup only. Private, and derived rather than
#: authored, because the exported tables above are read verbatim by the
#: ownership rules and the ``project/`` mirror: those consumers compare real
#: path strings and must keep the shipped spelling. Case folding belongs to the
#: *question* "may an agent write here", which has to answer the same for
#: ``.claude/Settings.json`` on a case-insensitive filesystem.
#: Pinned by test_reserved_exact_table_is_unchanged_by_the_pattern_table and
#: test_case_variants_of_an_exact_reservation_are_refused.
_RESERVED_CHANNELS_BY_FOLDED_PATH: dict[str, str] = {
    path.casefold(): channel for path, channel in RESERVED_PATH_CHANNELS.items()
}


@dataclass(frozen=True)
class ReservedPattern:
    """A protected project path matched by shape rather than by exact name.

    Attributes:
        pattern: Project-relative posix glob, written in lower case and
            matched by :func:`fnmatch.fnmatchcase` against a *casefolded* path.
            ``*`` spans path separators, so ``.claude/skills/**`` covers a
            whole subtree. Both sides are casefolded rather than left to
            :func:`fnmatch.fnmatch`'s platform rules, which gets the two
            properties this set needs at once: the answer is the same on every
            host (``fnmatch`` alone would differ between Linux and macOS), and
            it is closed over case on the hosts where it has to be — on a
            case-insensitive filesystem ``.CLAUDE/Skills/x`` opens the very
            file ``.claude/skills/x`` names.
        channel: The channel that *does* write it, phrased for a refusal.
    """

    pattern: str
    channel: str


#: Project paths no agent-side writer may touch, matched by shape. These are
#: whole classes of artifact rather than named files, which is why they cannot
#: live in :data:`RESERVED_PROJECT_PATHS` — that table is the exact set the
#: ownership rules and the ``project/`` mirror validation read, and widening it
#: would refuse the very channels that author these artifacts in the first place.
#:
#: Each entry answers the question "may a running agent rewrite this?", not
#: "which build channel owns it?". The two differ in both directions: an agent
#: may still author ``.claude/agents/`` and ``.claude/commands/`` material even
#: though the mirror may not, and it may not touch a settings overlay or a
#: limits table that the mirror is free to ship.
#:
#: Pinned by test_pattern_reserved_write_names_its_channel and
#: test_unreserved_writes_stay_writable.
RESERVED_PATH_PATTERNS: tuple[ReservedPattern, ...] = (
    ReservedPattern(
        ".claude/hooks/osprey_*.py",
        "the profile's `hooks/` convention directory — the `osprey_` hooks are the "
        "write-safety layer itself, and an agent editing one would be disarming the "
        "check that guards its own writes",
    ),
    ReservedPattern(
        ".claude/skills/**",
        "the profile's `skills/` convention directory — a skill is instruction text "
        "the agent would otherwise be rewriting for itself",
    ),
    ReservedPattern(
        ".claude/rules/**",
        "the profile's `rules/` convention directory — a rule is instruction text "
        "the agent would otherwise be rewriting for itself",
    ),
    ReservedPattern(
        ".claude/settings.local.json",
        "the profile's `config:` keys (`claude_code.permissions`, `claude_code.hooks`) — "
        "a local settings overlay silently widens the permissions the build rendered",
    ),
    ReservedPattern(
        "data/channel_limits.json",
        "the profile's `data/` directory — this is the limits table every setpoint "
        "is checked against before it reaches the control system",
    ),
)


#: Config keys no agent-side writer may set, keyed by the file that carries
#: them. A key belongs here when it does any of the following: gates writes,
#: approval, or limits; anchors a filesystem path the safety layers derive their
#: allow and deny areas from; or is rendered into ``.claude/settings.json`` or
#: ``.mcp.json``, where it becomes the agent's own permission surface.
#:
#: A pattern is a dotted key path in which a ``*`` segment stands for exactly
#: one level, and a trailing ``*`` covers the family prefix itself along with
#: everything below it — see :func:`is_protected_key` for the full rule.
#:
#: :data:`~osprey_connectors.config.RUNTIME_WRITE_PATH_KEYS` is *imported and
#: unpacked*, never restated: that tuple is already the single list of keys
#: whose value names a path something writes at run time, and a second copy here
#: would drift the moment a key is added there. Those keys qualify under the
#: second clause of the inclusion rule — repointing one moves the area a safety
#: layer treats as writable. Pinned by test_every_runtime_write_path_key_is_protected.
PROTECTED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "config.yml": (
        "control_system.*",
        "approval.*",
        "hooks.*",
        "claude_code.permissions.*",
        "claude_code.hooks.*",
        "claude_code.servers.*",
        "artifacts.hooks",
        "agent_data.*",
        "file_paths.*",
        "artifacts.*",
        *RUNTIME_WRITE_PATH_KEYS,
    ),
    ".mcp.json": (
        "mcpServers.*.command",
        "mcpServers.*.args",
        "mcpServers.*.env.*",
    ),
}

#: Exact keys a family pattern in :data:`PROTECTED_CONFIG_KEYS` covers but that
#: do not belong in the protected set. Consulted *after* family matching, so an
#: entry can only ever subtract.
#:
#: The inclusion rule a protected key has to meet is that setting it changes
#: what the agent may do: it gates writes, approval or limits; or it anchors a
#: filesystem path a safety layer derives an allow or deny zone from; or it is
#: rendered into ``.claude/settings.json`` or ``.mcp.json``. ``hooks.debug``
#: meets none of the three -- it toggles diagnostic verbosity in the hook
#: scripts and nothing else, is rendered into no artifact, and moves no zone.
#: The ``hooks.*`` family is there for the hook *wiring*, and it over-matched
#: this one key, which is a shipped operator control: the Web Terminal's Hook
#: Debug switch sets it through ``PATCH /api/config``.
#:
#: Keys are segment tuples, not dotted strings, for the same reason everything
#: else in this module is (see :func:`is_protected_key_path`), and membership is
#: exact: no wildcards, no prefixes, no ancestor rule. That asymmetry is
#: deliberate. A typo in a *pattern* can only over-protect, which is safe; a
#: typo in an exemption that behaved like a pattern could un-protect a whole
#: family, so an exemption is allowed to name one key and nothing else. In
#: particular, exempting ``("hooks", "debug")`` does not exempt ``("hooks",)``:
#: replacing the whole block still rewrites the wiring, so it stays protected.
#: Pinned in both directions by test_profile_conventions.py.
PROTECTED_KEY_EXEMPTIONS: dict[str, frozenset[tuple[str, ...]]] = {
    "config.yml": frozenset({("hooks", "debug")}),
}


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


#: What a refusal says when the path it was handed is not project-relative at
#: all — it is absolute, or it still climbs above the project root once
#: normalized (``foo/../../etc/passwd``). Such a path names a file the
#: protected set has no way to reason about, so the answer is a refusal rather
#: than ``None``: a writer must never be able to turn "outside the project"
#: into "not protected, go ahead". Pinned by
#: test_a_path_that_climbs_out_of_the_project_is_refused.
NOT_PROJECT_RELATIVE_CHANNEL = (
    "nothing here — the path is not project-relative (it is absolute, or it climbs "
    "above the project root); pass the path relative to the project root"
)


def is_reserved_write(project_rel: str) -> str | None:
    """Return the channel owning ``project_rel``, or ``None`` if a writer may write it.

    The protected set every framework writer consults before it puts bytes into
    a built project — the scaffold gallery, the Claude-setup panel, the
    ``setup_patch`` tool. It answers a different question from
    :func:`reserved_path_channel`, and the two must not be conflated:

    * :func:`reserved_path_channel` asks *which build channel owns this path*,
      so that the profile's ``project/`` mirror cannot become a second writer on
      an artifact some convention directory already produces. It covers every
      ``.claude/`` subtree that has a channel.
    * This function asks *may a running agent rewrite this*. It is narrower
      under ``.claude/`` — authoring a subagent or a slash command is the point
      of the panel, so those subtrees stay open — and wider outside it, because
      a settings overlay, an ``osprey_`` hook or the limits table can change
      what the agent is allowed to do even when no build channel is involved.

    Two normalizations make the answer depend on which *file* is named rather
    than on how the writer spelled it:

    * The path is run through :func:`posixpath.normpath` first, so ``./x``,
      ``a//b`` and ``foo/../.claude/skills/x`` all ask the same question as
      their plain spellings. Without it a writer dodges the whole set by
      spelling a path the long way round. A path that is absolute, or that
      still starts with ``..`` after normalization, is not project-relative;
      it is refused with :data:`NOT_PROJECT_RELATIVE_CHANNEL` rather than
      allowed, because "not a path I can judge" must not read as "writable".
    * Both the path and the patterns are casefolded before matching. The
      framework runs on case-insensitive filesystems, where
      ``.CLAUDE/Skills/x`` and ``.claude/settings.LOCAL.json`` open exactly the
      protected files, so a case-sensitive match would be a protected set only
      on Linux. Folding both sides keeps the answer identical on every host.

    Args:
        project_rel: Project-relative posix path the writer would write.

    Returns:
        A phrase naming the owning channel, suitable for a refusal message, or
        ``None`` when the path is not protected. A channel rather than a bare
        boolean, so a refusal can tell an operator the way in.
    """
    normalized = posixpath.normpath(PurePosixPath(project_rel).as_posix())
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        return NOT_PROJECT_RELATIVE_CHANNEL
    folded = normalized.casefold()

    # ORDER IS LOAD-BEARING: the exact table answers first. Its entries carry
    # the precise channel ("the profile's `config:` block"); a pattern widened
    # onto one of them would answer with a coarser phrase and send an operator
    # to the wrong place. Pinned by test_exact_reservation_beats_the_pattern_table.
    exact = _RESERVED_CHANNELS_BY_FOLDED_PATH.get(folded)
    if exact is not None:
        return exact

    for reserved in RESERVED_PATH_PATTERNS:
        if fnmatchcase(folded, reserved.pattern.casefold()):
            return reserved.channel

    return None


def _key_path_matches(pattern: str, key_path: Sequence[str]) -> bool:
    """Whether ``key_path`` is covered by one :data:`PROTECTED_CONFIG_KEYS` pattern.

    ``key_path`` is one entry per key *segment*, never a dotted string: a raw
    key may legitimately contain a ``.`` (an MCP server named ``evil.srv``) and
    splitting it again would turn one segment into two. The *pattern* is still
    split on ``.``, which is sound because patterns are authored in this module
    and no segment of one contains a dot.

    Segment-wise:

    * a literal segment must match exactly, and a ``*`` segment matches any one
      segment (``mcpServers.*.command`` covers every server's command line);
    * a *trailing* ``*`` matches the rest of the key, including nothing at all,
      so ``control_system.*`` covers ``control_system`` itself as well as
      everything below it — a writer that replaced the whole block with a scalar
      would otherwise slip past a rule that only guarded its children;
    * a key that is a strict *ancestor* of a pattern is covered too, because
      writing the ancestor rewrites the protected descendant. Only prefixes of
      the pattern qualify, never unrelated siblings under the same root:
      ``services`` is protected because a runtime-write path lives beneath it,
      while ``services.orbit.enabled`` is not.
    """
    pattern_parts = pattern.split(".")
    wildcard_tail = pattern_parts[-1] == "*"
    if wildcard_tail:
        pattern_parts = pattern_parts[:-1]
    key_parts = tuple(key_path)

    shared = min(len(pattern_parts), len(key_parts))
    for index in range(shared):
        if pattern_parts[index] != "*" and pattern_parts[index] != key_parts[index]:
            return False
    if len(key_parts) <= len(pattern_parts):
        return True  # the pattern itself, or an ancestor of it
    return wildcard_tail


def is_protected_key_path(target_file: str, key_path: Sequence[str]) -> bool:
    """Whether the key at ``key_path`` is one an agent-side writer may not set.

    The sound form of :func:`is_protected_key`, and the one every *document*
    walk uses. Because the key arrives already split into segments, a raw key
    containing a literal ``.`` stays one segment: an ``.mcp.json`` server named
    ``evil.srv`` is ``("mcpServers", "evil.srv", "command")``, which
    ``mcpServers.*.command`` covers. Re-splitting a dotted rendering of that
    same key would produce four segments, and the pattern — whose ``*`` stands
    for exactly one segment — would match nothing at all.

    Args:
        target_file: The file the key lives in. Given as a name or a path — only
            the final component is consulted, so a writer that already holds an
            absolute target does not have to shorten it first.
        key_path: The key as a sequence of segments, e.g.
            ``("control_system", "writes_enabled")``.

    Returns:
        ``True`` when some pattern in :data:`PROTECTED_CONFIG_KEYS` covers the
        key and :data:`PROTECTED_KEY_EXEMPTIONS` does not name it exactly. A
        file with no table protects nothing: the writers that consult this are
        already restricted to the two files that have one, and inventing
        protection for a third would refuse writes no rule describes.
    """
    name = PurePosixPath(target_file).name
    patterns = PROTECTED_CONFIG_KEYS.get(name)
    if not patterns:
        return False
    key_parts = tuple(key_path)
    if not any(_key_path_matches(pattern, key_parts) for pattern in patterns):
        return False
    # Exemptions are consulted last, and only subtract: a key the families do
    # not cover is already unprotected, so an exemption for it would be dead
    # rather than dangerous. Exact membership -- see PROTECTED_KEY_EXEMPTIONS
    # for why an exemption deliberately has none of a pattern's reach.
    return key_parts not in PROTECTED_KEY_EXEMPTIONS.get(name, frozenset())


def is_protected_key(target_file: str, dotted: str) -> bool:
    """Whether ``dotted`` is a key an agent-side writer may not set in ``target_file``.

    The dotted front door, for a caller that holds a key path already written
    the way an operator or a ``config:`` block writes it. ``dotted`` is split on
    ``.``, so it cannot express a raw key that *contains* a dot — for that, and
    for anything walking a parsed document, use :func:`is_protected_key_path`
    (which :func:`protected_view` does).

    Args:
        target_file: The file the key lives in (see :func:`is_protected_key_path`).
        dotted: Dotted key path, e.g. ``control_system.writes_enabled``.

    Returns:
        ``True`` when some pattern in :data:`PROTECTED_CONFIG_KEYS` covers the key.
    """
    return is_protected_key_path(target_file, dotted.split("."))


def flatten_key_paths(doc: Mapping[str, Any]) -> dict[tuple[str, ...], Any]:
    """Flatten a nested document to ``key path -> value``, one tuple per leaf.

    Mappings are descended; everything else is a leaf, lists included — a list
    is compared whole because its *contents* are the privilege (a widened
    ``control_system.write_tools`` is a bigger change than any one element).
    An empty mapping is kept as a leaf rather than descended into nothing, so
    emptying a block reads as the change it is instead of vanishing silently.

    Keys are tuples of segments, not dotted strings, and that is the whole
    point: a document's own key may contain a ``.``, and a dotted rendering
    makes two different documents look identical (a server named ``evil.srv``
    versus a nested ``evil`` → ``srv``). The tuple form is injective, so it is
    what protection and the PUT diff are built on.

    Args:
        doc: Parsed document (a loaded ``config.yml`` or ``.mcp.json``).

    Returns:
        One entry per leaf, keyed by its segment path.
    """
    flat: dict[tuple[str, ...], Any] = {}

    def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in node.items():
            key_path = (*prefix, str(key))
            if isinstance(value, Mapping) and value:
                _walk(value, key_path)
            else:
                flat[key_path] = value

    _walk(doc, ())
    return flat


def flatten_dotted(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The dotted rendering of :func:`flatten_key_paths` — for display, not for matching.

    Use it to *show* an operator which keys a document carries. Do not decide
    anything on it: joining segments with ``.`` is lossy whenever a raw key
    contains a dot, and two distinct leaves can collapse onto one entry.
    :func:`protected_view` and :func:`is_protected_key_path` work on segment
    paths for exactly that reason.

    Args:
        doc: Parsed document (a loaded ``config.yml`` or ``.mcp.json``).

    Returns:
        One entry per leaf, keyed by its dotted path.
    """
    return {".".join(key_path): value for key_path, value in flatten_key_paths(doc).items()}


def protected_view(target_file: str, doc: Mapping[str, Any]) -> dict[tuple[str, ...], Any]:
    """The protected keys of ``doc``, flattened — the primitive a whole-file PUT diffs.

    A writer that replaces a whole file cannot be judged key by key: it has no
    patch to inspect. Comparing this view of the old bytes against the same view
    of the new ones catches every class of change at once, because the view is a
    flat mapping and mappings compare by both keys and values — an added key, a
    deleted one, a changed value, a changed list, and a subtree reshaped into a
    scalar (the flattened children disappear while the parent appears as a leaf).
    Anything outside the protected set is absent from both views, so an ordinary
    edit compares equal and is not refused.

    Keyed by *segment path* (see :func:`flatten_key_paths`), never by dotted
    string. A dotted view would let a writer hide a privilege change behind a
    key that contains a dot: an ``.mcp.json`` server named ``evil.srv`` renders
    as ``mcpServers.evil.srv.command``, which no pattern covers and which can
    collide with a genuinely nested key — the added launcher would drop out of
    the view and the PUT would compare equal. Join the segments with ``.`` when
    a refusal has to name the key to an operator.

    Args:
        target_file: The file the document was loaded from (see
            :func:`is_protected_key_path`).
        doc: The parsed document.

    Returns:
        The flattened document restricted to its protected keys, keyed by
        segment path.
    """
    return {
        key_path: value
        for key_path, value in flatten_key_paths(doc).items()
        if is_protected_key_path(target_file, key_path)
    }


#: The workspace MCP tool that patches a built project's ``config.yml`` and
#: ``.mcp.json`` — the one write surface that can change what the agent itself
#: is allowed to do. A persona keeps or loses the setup capability by whether
#: the render leaves this name outside ``permissions.deny``; see
#: :func:`is_setup_patch_capable`.
SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"


def permission_entries(value: Any) -> list[str]:
    """The string entries of a permissions list — read only when it *is* a list.

    Anything else yields no entries, **a bare string included**. A lone
    ``deny: mcp__…`` reads like "one entry spelled without its list", but the
    render does not read it that way and neither may this: ``settings.json.j2``
    iterates the value, so a bare-string deny renders one deny entry per
    *character* and names no tool at all, while the render's subtraction
    (``d not in remove_deny``) against a bare-string ``remove_deny`` is
    *substring* containment, which lifts denies nobody wrote. Guessing "one
    entry" would diverge from the render in both directions at once.

    Reading a loose spelling as nothing is safe here only because no such
    document can reach a render:
    :func:`osprey.cli.build_profile_load._reject_permission_list_shapes`
    refuses at profile-parse time any ``claude_code.permissions.*`` value that
    is not a list of non-empty strings, in either spelling. So every list this
    sees on the build path is list-shaped, and this and the render agree entry
    for entry. The tolerance that remains is for documents that never came from
    a profile at all — a hand-assembled dict, a settings.json read off disk.
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [entry for entry in value if isinstance(entry, str)]


def _deny_entries(config: Any) -> list[str]:
    """The *effective* deny list ``config`` carries, however its document spells it.

    Two shapes reach this, and they are told apart by which block is present:

    * a **persona config** (``config.yml``-shaped), whose deny lives under
      ``claude_code.permissions.deny`` — the shape the render context and the
      persona-roster guard hold, and the primary input. This shape is NOT yet
      composed: a profile delta cannot subtract from an inherited list, so a
      tier that lifts a base deny carries the inherited ``deny`` and its own
      ``claude_code.permissions.remove_deny`` side by side, and the
      subtraction happens when ``settings.json`` is rendered. So it is
      performed here too, the same way and for the same reason: reading the
      raw ``deny`` alone would report a lifted tier as denied while its
      rendered ``settings.json`` lets the tool through — a *disagreement with
      the render*, which is the one thing this must not have. Which consumer
      that would hurt depends on the consumer (see
      :func:`is_setup_patch_capable`); that it is wrong does not;
    * a rendered ``.claude/settings.json``, which has no ``claude_code:``
      wrapper and carries ``permissions.deny`` at the top level. Read only when
      no ``claude_code`` block is present, which a persona config always has
      when it says anything about permissions at all. Without this branch a
      settings document handed here would find nothing and read as *capable*
      while the tool is in fact denied — a mis-call must not fail open. This
      shape is already composed, so no subtraction is applied to it and a
      stray top-level ``remove_deny`` is ignored.

    The subtraction mirrors the render exactly — that is its whole
    justification, not a lean in a safe direction. See
    :func:`osprey.cli.templates.claude_code._rendered_deny_list`, which builds
    the rendered array as ``[d for d in deny_defaults if d not in remove_deny]``
    plus the same filter over the facility deny: membership is **exact**, not
    glob. So a wildcard deny (``mcp__osprey_workspace__*``) is NOT lifted by an
    exact ``remove_deny`` of one tool name — the render leaves that wildcard in
    the deny array, and so does this.

    Kill-switch entries cannot be lifted here, and not merely by convention:
    they are generated at render time from ``writes_enabled: false`` into their
    own context key and never appear in a ``config.yml`` deny list, so there is
    nothing for this subtraction to reach. The render keeps them out of the
    filtered lists for exactly that reason.

    Anything not shaped like a mapping yields no entries; see
    :func:`permission_entries` for how each list is read, and for why a list
    spelled as a bare string is read as no entries here rather than as one —
    the render does not read it as one either, and a profile that spells it
    that way is refused before it can be built.
    """
    root = config if isinstance(config, Mapping) else {}
    claude_code = root.get("claude_code")
    persona_shaped = isinstance(claude_code, Mapping)
    if persona_shaped:
        permissions = claude_code.get("permissions")
    elif claude_code is None:
        permissions = root.get("permissions")
    else:
        permissions = None
    if not isinstance(permissions, Mapping):
        return []
    deny = permission_entries(permissions.get("deny"))
    if not persona_shaped:
        return deny
    lifted = set(permission_entries(permissions.get("remove_deny")))
    return [entry for entry in deny if entry not in lifted]


def is_setup_patch_capable(config: Any) -> bool:
    """Whether ``config``'s render leaves the setup capability in the agent's hands.

    The posture question every later gate asks about a persona: can this render
    still reach :data:`SETUP_PATCH_TOOL`? Consumed by the container's render
    context, by the persona-roster guard, and by the lint belt, so that all
    three answer it the same way instead of each re-reading the deny list.

    ``config`` is a **rendered persona config** — the ``config.yml``-shaped
    document whose deny list lives at ``claude_code.permissions.deny``, read
    together with the ``remove_deny`` beside it so that a tier which lifts an
    inherited deny reports as capable. A rendered ``.claude/settings.json``
    (deny at the top level, already composed) is read too; see
    :func:`_deny_entries` for how the two are told apart, how the subtraction
    mirrors the render, and why a kill-switch deny is out of its reach.

    The deny is what *removes* the capability, so every way of not having one —
    no ``claude_code`` block, no ``permissions``, no ``deny`` key, an empty or
    null deny — leaves the persona capable. This is not a permissive default
    dressed up as a rule: an unwritten deny is exactly the state of a project
    that never gated the tool, and the readonly tier expresses itself by
    *adding* the entry.

    Entries are matched with :func:`fnmatch.fnmatchcase`, which covers the exact
    literal the presets write and also a wildcard deny (``mcp__osprey_workspace__*``)
    that takes the tool away just as completely. The asymmetry against the exact
    ``remove_deny`` subtraction is not a lean in a safe direction but a pair of
    mirrors, each held against the thing it has to agree with: the glob **match**
    mirrors Claude Code's own deny matching, where a deny pattern gates every
    tool name it matches; the exact **subtraction** mirrors the render's
    ``d not in remove_deny``
    (:func:`osprey.cli.templates.claude_code._rendered_deny_list`).

    **Parity is the criterion, not conservatism**, because the two consumers
    pull in opposite directions. The container's Dockerfile *grants* the
    ``build/config.yml`` chown on ``True``; the persona-roster guard *refuses*
    a ``default_persona`` or ``login: false`` on ``True``. So there is no safe
    direction to lean: an answer biased toward ``False`` waves a capable
    persona past the roster guard, and one biased toward ``True`` hands an
    image's ``config.yml`` to an agent that cannot in fact patch it. The only
    answer that serves both consumers is the render's own, which is why every
    reading rule here is stated as a mirror of the render rather than as a
    fail-safe. The parity itself is pinned by
    ``test_the_predicate_matches_the_rendered_deny_list``.

    Args:
        config: The rendered persona config (see above). Tolerant of a missing
            or misshapen block — an unreadable document reports no deny, and an
            unreadable ``remove_deny`` lifts nothing.

    Returns:
        ``True`` when nothing in the effective deny list names the tool.

    Pinned in ``tests/cli/test_profile_conventions.py`` by
    ``test_setup_patch_capable_is_false_when_the_tool_is_denied``,
    ``…_is_true_when_another_tool_is_denied``,
    ``…_is_true_with_no_deny_block``,
    ``…_is_true_with_no_claude_code_block``,
    ``…_is_true_for_an_empty_or_null_deny``,
    ``…_is_false_under_a_wildcard_deny``,
    ``…_reads_a_rendered_settings_json_shape``,
    ``…_tolerates_a_misshapen_document``; for the subtraction by
    ``…_is_true_when_remove_deny_lifts_the_tool``,
    ``…_is_false_when_remove_deny_lifts_another_tool``,
    ``test_a_misshapen_remove_deny_lifts_nothing``,
    ``test_an_exact_remove_deny_does_not_lift_a_wildcard_deny`` and
    ``test_a_settings_shaped_document_ignores_a_stray_remove_deny``; for the
    bare-string spellings by ``test_a_bare_string_deny_is_no_deny_at_all`` and
    ``test_a_bare_string_remove_deny_lifts_nothing``, each of which also names
    the profile-time refusal that keeps those spellings off the build path; and
    for the parity the whole docstring rests on by
    ``test_the_predicate_matches_the_rendered_deny_list``.
    """
    return not any(fnmatchcase(SETUP_PATCH_TOOL, entry) for entry in _deny_entries(config))


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
        if convention.per_user and entry == context_baseline_slot(root) and entry.is_file():
            # The shared baseline, not a user: it has its own slot at the
            # convention root and is planned as a file copy.
            continue
        if not entry.is_dir():
            errors.append(
                _per_user_file_error(convention, entry)
                if convention.per_user
                else (
                    f"{convention.source}/{entry.name} is a file: {convention.source}/ "
                    f"holds one directory per {convention.entry_noun}.\n"
                    f"  Move it into {convention.source}/{entry.stem}/ "
                    f"(the whole directory is copied as a unit)."
                )
            )
    return errors


def _per_user_file_error(convention: ConventionDir, entry: Path) -> str:
    """Report a loose file in a per-user convention directory.

    A per-user directory's names are not free-form: the build matches every one
    of them against the resolved roster, and a directory naming nobody on it is
    skipped (:func:`partition_context_users`). "Move it into ``<stem>/``" — the
    advice every other directory-shaped convention gets — would therefore tell
    an operator to invent a user, and their file would quietly stop being read
    on the next build. So the rule is stated instead, and the file is pointed at
    the routes that carry it: a named user's directory, or — for the shared
    baseline — the :data:`CONTEXT_BASELINE_FILENAME` slot at the convention
    root, which the validator has already accepted by the time this error fires
    for anything else. Both routes are offered, because the message cannot tell
    which one the file wants: a loose file here is as likely to be baseline text
    under the wrong name as it is to be one user's.
    """
    header = (
        f"{convention.source}/{entry.name} is a file: {convention.source}/ holds one "
        f"directory per {convention.entry_noun} (plus the shared "
        f"{CONTEXT_BASELINE_FILENAME} baseline), and each one is named for a user on "
        f"the resolved roster. A directory named anything else is skipped as a user "
        f"who has left, so a directory invented to hold this file would not be read."
    )
    return (
        f"{header}\n"
        f"  Move it into the directory of the user it belongs to "
        f"({convention.source}/<user>/{entry.name}), naming a user the build resolves.\n"
        f"  Or, if it is the text every user starts from, rename it to "
        f"{convention.source}/{CONTEXT_BASELINE_FILENAME} — the shared baseline slot, "
        f"the one loose file {convention.source}/ accepts."
    )


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
            if convention.per_user:
                baseline = context_baseline_slot(root)
                if baseline.is_file():
                    # Roster-independent by design: the baseline overrides the
                    # framework's fallback for every seeded user at once, so it
                    # copies even on a build with no roster (a persona render),
                    # like every other convention artifact.
                    copies.append(
                        ConventionCopy(
                            category=convention.source,
                            source=baseline,
                            destination=destination_for(
                                f"{convention.source}/{CONTEXT_BASELINE_FILENAME}"
                            ),
                            is_directory=False,
                        )
                    )
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
