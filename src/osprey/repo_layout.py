"""The deployment root's layout, declared once.

A deployment repo is laid out in entries at its root: ``profile.yml`` and the
source material it names, the convention directories the build renders, the
secrets, the generated zones, the repo's own Claude Code files and the shell
``osprey init`` writes once. Every table that names one of those entries — the
zone tables, the known-root set, the paths executed code may not write, init's
categories, the image-exclusion set, the layer directories — is a view of
:data:`ROOT_ENTRIES`, computed by the module that owns the table at its own
import. A layout change is an edit to a row here, never to a list of call sites.

Each row separates three things, so that a path can change without changing a
key or a label. ``key`` is the entry's identity: hash and drift labels,
profile list prefixes, artifact names and render attributes are keyed by it.
``path`` is its location relative to the deployment root, and may be nested
(``agent/rules``). The display forms (:func:`display_path`,
:func:`listing_name`) are how a message or a listing names it.

A former location that is still read as an entry is a retired path of the live
row (:attr:`RootEntry.retired_paths`), never a row of its own, so nothing about
the successor is restated.

A former location that is no longer read at all, only refused until its
contents are moved, is a moved-from path of the live row or child
(:attr:`RootEntry.moved_from`, :attr:`RootChild.moved_from`). Those paths are
relative to the deployment root, whatever their successor is, and
:func:`retired_moves` is the one table of them. A former top-level directory
whose contents all moved elsewhere, and which no move names whole, is a
dissolved name (:func:`dissolved_names`).

A source the build copies into the render carries its place there
(:attr:`RootEntry.render`, :attr:`RootChild.render`), so the render is
assembled from declared sources only (:func:`rendered_sources`).

Children are named paths inside a row (:data:`ROOT_CHILDREN`): the CI verifier
inside ``scripts``, the canopy library inside ``web``. A child follows its
parent wherever the parent resolves, and is in no zone table. A child's key
names its role, lowercase words joined by hyphens (``verify-script``,
``canopy-library``), and never spells its file: the key survives the child
moving or being renamed.

Each row states what it is for once (:attr:`RootEntry.purpose`). A convention
row's purpose is the description the convention directory carries; no table
restates it.

The contract every reader relies on
-----------------------------------
:data:`ROOT_ENTRIES` and :data:`ROOT_CHILDREN` are the only module state. Every
function reads them when called; nothing here builds an index or a derived
constant at import. A test that needs another layout rebinds
:data:`ROOT_ENTRIES` to a :func:`with_paths` result before any consumer is
imported, and every consumer's tables follow.

Declaration order is observable: each ordered table is the rows of one role,
in declaration order. A row that carries two ordered roles ties those two
orders together at that row: ``web-terminal-context`` is both a convention and
an entry init replaces, so every other row of either role stays on its own
side of it. Rows that share no ordered role may be interleaved freely.

This module is a leaf below ``osprey.cli`` and loads no YAML library, so the
runtime layers, the python executor and the repo resolver can import it within
their import budgets. It reads the render directory and the env-chain file
names from :mod:`osprey_connectors`, which owns them; every other root name is
spelled here, and the module that exports the name aliases the row.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from osprey.bluesky_bridge_connection import LANE_KEYS
from osprey_connectors.dotenv import ENV_LOCAL_FILENAME, ENV_SHARED_FILENAME
from osprey_connectors.workspace import BUILD_DIR_NAME

__all__ = [
    "ROOT_CHILDREN",
    "ROOT_ENTRIES",
    "EntryKind",
    "InitRole",
    "RenderedSource",
    "RootChild",
    "RootEntry",
    "RootMove",
    "Zone",
    "check_layout",
    "child",
    "child_display_path",
    "child_path",
    "convention_rows",
    "display_path",
    "dissolved_names",
    "entry",
    "former_locations",
    "image_excluded_rows",
    "layer_rows",
    "listing_name",
    "moved_from_top_level_names",
    "non_convention_source_rows",
    "paths",
    "rendered_sources",
    "retired_moves",
    "retired_paths",
    "rows_in_zone",
    "rows_with_init",
    "top_level_names",
    "watchdogs_source_dir",
    "with_paths",
    "zoned_rows",
]


class EntryKind(StrEnum):
    """What an entry is on disk."""

    FILE = "file"
    DIRECTORY = "directory"
    #: A directory or a file standing in for one (``.git`` is a directory in a
    #: clone and a gitfile in a worktree). Listed without a trailing slash.
    EITHER = "either"


class Zone(StrEnum):
    """The zone an entry belongs to. An entry no zone table lists has none."""

    #: Tracked and operator-edited: the profile and everything it names.
    #: Executed code may not write here.
    SOURCE = "source"
    #: Git-ignored and durable.
    SECRETS = "secrets"
    #: Git-ignored and written by OSPREY: the render and the state zone.
    GENERATED = "generated"
    #: The repo's own Claude Code files, for whoever edits the repo; never
    #: rendered and never in an image.
    CLAUDE_CODE = "claude-code"


class InitRole(StrEnum):
    """What ``osprey init`` does with an entry. An entry it never names has none."""

    #: Owned by a materialization: held aside and replaced by ``--force``.
    REPLACED = "replaced"
    #: Authored when absent and never rewritten.
    WRITE_ONCE = "write-once"
    #: Emitted by the scaffolding engine under its marker contract.
    SCAFFOLDED = "scaffolded"
    #: Copied out of the app bundle on a repo's first-ever materialization only.
    SEEDED = "seeded"
    #: The convention directory holding the facility rule init renders.
    FACILITY_RULE = "facility-rule"
    #: Named by init's promise and never written by it.
    UNTOUCHED = "untouched"


@dataclass(frozen=True, slots=True)
class RootEntry:
    """One entry at the deployment root and every role it plays."""

    #: Stable identity. Labels, profile list prefixes and render attributes are
    #: keyed by it, so none of them changes when the entry moves.
    key: str
    #: Location relative to the deployment root: POSIX, no leading or trailing
    #: slash, possibly nested.
    path: str
    kind: EntryKind
    #: What the entry is for, stated once. A convention row's purpose is the
    #: description its convention directory carries.
    purpose: str
    zone: Zone | None = None
    #: The build renders it as a convention directory.
    convention: bool = False
    init: InitRole | None = None
    #: Never copied into a container image by the fixed-name rule. The repo's
    #: Claude Code files, the ``.env`` names and ``.dockerignore`` are left out
    #: by the staging rule's other clauses, not by this flag.
    image_excluded: bool = False
    #: The kind word a layer directory's files are read as, else ``None``.
    layer_kind: str | None = None
    #: Former locations still read as this entry, relative to the root.
    retired_paths: tuple[str, ...] = ()
    #: Former locations never read again, only refused until their contents
    #: move to :attr:`path`. Relative to the root: POSIX, possibly nested.
    moved_from: tuple[str, ...] = ()
    #: Where the build copies this source to, relative to the render directory,
    #: else ``None``: the build does not copy it as a declared source.
    render: str | None = None


@dataclass(frozen=True, slots=True)
class RootChild:
    """A named path inside a row, which follows its parent wherever it resolves."""

    #: The ``key`` of the row this child sits inside.
    parent: str
    #: The child's role, lowercase words joined by hyphens; never its file name.
    key: str
    #: Location relative to the parent: POSIX, no leading or trailing slash.
    path: str
    purpose: str
    #: Former locations never read again, only refused until their contents
    #: move here. Relative to the deployment root, not to the parent.
    moved_from: tuple[str, ...] = ()
    #: Where the build copies this child to, relative to the render directory,
    #: else ``None``.
    render: str | None = None
    #: ``(old, subpath)`` pairs: the former location *old*, relative to the
    #: deployment root, whose contents move to this child's path joined with
    #: *subpath*.
    moved_into: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class RootMove:
    """One former location and the live path its contents belong at now."""

    #: The former location, relative to the deployment root.
    old: str
    #: The successor, relative to the deployment root.
    new: str
    #: The successor row's kind; a child declares none, so it is ``EITHER``.
    kind: EntryKind


class RenderedSource(NamedTuple):
    """One source the build copies into the render, and where it lands there."""

    #: The row's key, or ``<row>:<child>`` for a child.
    key: str
    #: The source, relative to the deployment root.
    source_rel: str
    #: Where it lands, relative to the render directory.
    render_rel: str


_FILE = EntryKind.FILE
_DIR = EntryKind.DIRECTORY

#: Every entry a table of the deployment root names. Entries known only through
#: an exemption rule (a dot prefix, a ``README``/``LICENSE`` prefix, ``docs``) and
#: entries a profile key names have no row.
ROOT_ENTRIES: tuple[RootEntry, ...] = (
    RootEntry(
        "profile.yml",
        "profile.yml",
        _FILE,
        "The deployment's profile; its presence makes a directory a deployment root.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        "providers.yml",
        "providers.yml",
        _FILE,
        "The provider catalog the build renders into the API providers.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        "facility",
        "facility",
        _DIR,
        "The facility description: devices, channels, simulation and knowledge.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
        moved_from=("data/facility",),
        render="data/facility",
    ),
    RootEntry(
        "web",
        "web",
        _DIR,
        "The web content the build serves: canopy panels and landing pages.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        "tiers",
        "tiers",
        _DIR,
        "The login files laid over the profile, one per login.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
        layer_kind="tier",
        retired_paths=("personas",),  # retired-spelling
    ),
    RootEntry(
        "apps",
        "apps",
        _DIR,
        "The applications, one directory each.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        ".osprey",
        ".osprey",
        _DIR,
        "What OSPREY records about the repo for itself, such as the saved preset output.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        "triggers.yml",
        "triggers.yml",
        _FILE,
        "The resolved trigger configuration the dispatch block points at.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        "rules",
        "agent/rules",
        _DIR,
        "Agent behavior rules",
        zone=Zone.SOURCE,
        convention=True,
        init=InitRole.FACILITY_RULE,
        moved_from=("rules",),
    ),
    RootEntry(
        "skills",
        "agent/skills",
        _DIR,
        "Agent skills (one directory per skill)",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("skills",),
    ),
    RootEntry(
        "agents",
        "agent/agents",
        _DIR,
        "Subagent definitions",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("agents",),
    ),
    RootEntry(
        "commands",
        "agent/commands",
        _DIR,
        "Slash commands",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("commands",),
    ),
    RootEntry(
        "output-styles",
        "agent/output-styles",
        _DIR,
        "Output styles",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("output-styles",),
    ),
    RootEntry(
        "hooks",
        "agent/hooks",
        _DIR,
        "Claude Code hook scripts",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("hooks",),
    ),
    RootEntry(
        "web-terminal-context",
        "agent/context",
        _DIR,
        "Per-user web-terminal context (one directory per user, plus a shared base.md)",
        zone=Zone.SOURCE,
        convention=True,
        init=InitRole.REPLACED,
        moved_from=("web-terminal-context",),
    ),
    RootEntry(
        ".env.example",
        ".env.example",
        _FILE,
        "The template of the secrets the deployment needs.",
        zone=Zone.SECRETS,
        init=InitRole.REPLACED,
    ),
    RootEntry(
        "watchdogs",
        "agent/watchdogs",
        _DIR,
        "The watchdogs' instructions and references, one directory per watchdog.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
        moved_from=("watchdogs",),
    ),
    RootEntry(
        "code_review",
        "agent/code_review",
        _DIR,
        "The code reviewer's prompt and material.",
        zone=Zone.SOURCE,
        init=InitRole.REPLACED,
        moved_from=("data/code_review",),
    ),
    RootEntry(
        "mcp_servers",
        "agent/mcp_servers",
        _DIR,
        "Facility MCP server sources (one directory per server)",
        zone=Zone.SOURCE,
        convention=True,
        init=InitRole.SEEDED,
        moved_from=("mcp_servers",),
    ),
    RootEntry(
        "services",
        "ops/services",
        _DIR,
        "Facility-owned compose services (one directory per service)",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("services",),
    ),
    RootEntry(
        "project",
        "ops/project",
        _DIR,
        "Verbatim mirror onto the project root (escape hatch)",
        zone=Zone.SOURCE,
        convention=True,
        moved_from=("project",),
    ),
    RootEntry(
        "profiles",
        "profiles",
        _DIR,
        "The host-variant overlays of the profile.",
        zone=Zone.SOURCE,
    ),
    RootEntry(
        "scripts",
        "ops/scripts",
        _DIR,
        "Scripts the scaffolding verbs emit, such as the CI verifier and the boot hook.",
        zone=Zone.SOURCE,
        moved_from=("scripts",),
    ),
    RootEntry(
        "osprey.service",
        "ops/osprey.service",
        _FILE,
        "The systemd boot unit the scaffolding verb emits.",
        zone=Zone.SOURCE,
        moved_from=("osprey.service",),
    ),
    RootEntry(
        ".dockerignore",
        ".dockerignore",
        _FILE,
        "The repo files kept out of the deployment's images.",
        zone=Zone.SOURCE,
    ),
    RootEntry(
        ".gitignore",
        ".gitignore",
        _FILE,
        "The repo's ignore rules for the generated and secrets zones.",
        init=InitRole.WRITE_ONCE,
    ),
    RootEntry(
        ENV_SHARED_FILENAME,
        ENV_SHARED_FILENAME,
        _FILE,
        "The tracked, non-secret half of the environment chain.",
        init=InitRole.WRITE_ONCE,
    ),
    RootEntry(
        "README.md",
        "README.md",
        _FILE,
        "The repo's own readme.",
        init=InitRole.WRITE_ONCE,
    ),
    RootEntry(
        "ci-extra.yml",
        "ops/ci-extra.yml",
        _FILE,
        "The deployment's own additions to the generated CI pipeline.",
        zone=Zone.SOURCE,
        init=InitRole.WRITE_ONCE,
        moved_from=("ci-extra.yml",),
    ),
    RootEntry(
        ".gitlab-ci.yml",
        ".gitlab-ci.yml",
        _FILE,
        "The CI pipeline the scaffolding engine emits.",
        init=InitRole.SCAFFOLDED,
    ),
    RootEntry(
        ENV_LOCAL_FILENAME,
        ENV_LOCAL_FILENAME,
        _FILE,
        "The host's secrets, the local half of the environment chain.",
        zone=Zone.SECRETS,
        init=InitRole.UNTOUCHED,
    ),
    RootEntry(
        ".git",
        ".git",
        EntryKind.EITHER,
        "Git's own directory, or the gitfile of a worktree.",
        init=InitRole.UNTOUCHED,
        image_excluded=True,
    ),
    RootEntry(
        "var",
        "var",
        _DIR,
        "The durable state zone, which survives every build.",
        zone=Zone.GENERATED,
        init=InitRole.UNTOUCHED,
        image_excluded=True,
    ),
    RootEntry(
        BUILD_DIR_NAME,
        BUILD_DIR_NAME,
        _DIR,
        "The render, re-created by every build.",
        zone=Zone.GENERATED,
        init=InitRole.UNTOUCHED,
        image_excluded=True,
    ),
    RootEntry(
        ".osprey-compose.yml",
        ".osprey-compose.yml",
        _FILE,
        "The merged compose document a deploy writes.",
        image_excluded=True,
    ),
    RootEntry(
        "CLAUDE.md",
        "CLAUDE.md",
        _FILE,
        "Claude Code instructions for whoever edits the repo.",
        zone=Zone.CLAUDE_CODE,
    ),
    RootEntry(
        "CLAUDE.local.md",
        "CLAUDE.local.md",
        _FILE,
        "Host-local Claude Code instructions for whoever edits the repo.",
        zone=Zone.CLAUDE_CODE,
    ),
    RootEntry(
        ".claude",
        ".claude",
        _DIR,
        "Claude Code settings for whoever edits the repo.",
        zone=Zone.CLAUDE_CODE,
    ),
    RootEntry(
        ".mcp.json",
        ".mcp.json",
        _FILE,
        "Claude Code MCP servers for whoever edits the repo.",
        zone=Zone.CLAUDE_CODE,
    ),
)

#: Named paths inside a row. A child resolves under its row's path at the repo
#: root.
ROOT_CHILDREN: tuple[RootChild, ...] = (
    RootChild(
        "scripts",
        "verify-script",
        "verify.sh",
        "The CI verifier the scaffolding engine emits.",
    ),
    RootChild(
        "scripts",
        "boot-hook",
        "osprey-boot-hook.sh",
        "The boot hook the systemd scaffolding emits.",
    ),
    RootChild(
        "web",
        "canopy-library",
        "canopy/library",
        "The facility's canopy panel library.",
        moved_from=("data/canopy/library",),
        render="data/canopy/library",
    ),
    RootChild(
        "web",
        "canopy-collections",
        "canopy/collections",
        "The facility's canopy collections.",
        moved_from=("data/canopy/collections",),
        render="data/canopy/collections",
    ),
    RootChild(
        "web",
        "canopy-theme",
        "canopy/theme",
        "The facility's canopy theme.",
        moved_from=("data/canopy/theme",),
        render="data/canopy/theme",
    ),
    RootChild(
        "web",
        "landing",
        "landing",
        "The pages the web terminal's landing screen shows.",
        moved_from=("data/landing",),
        render="data/landing",
    ),
    RootChild(
        "var",
        "runtime",
        "runtime",
        "Material the deployment mints at start, such as certificates; never hashed or staged.",
        moved_from=("data/.runtime",),
        moved_into=tuple((f"data/{lane}_curve", f"{lane}_curve") for lane in LANE_KEYS),
    ),
)


# --- lookup ----------------------------------------------------------------------


def entry(key: str) -> RootEntry:
    """The row whose ``key`` is *key*.

    Raises:
        KeyError: No row has that key.
    """
    for row in ROOT_ENTRIES:
        if row.key == key:
            return row
    raise KeyError(f"no deployment-root entry has the key {key!r}")


def child(parent: str, key: str) -> RootChild:
    """The child *key* of the row *parent*.

    Raises:
        KeyError: The row has no such child.
    """
    for row in ROOT_CHILDREN:
        if row.parent == parent and row.key == key:
            return row
    raise KeyError(f"the deployment-root entry {parent!r} has no child {key!r}")


# --- selection -------------------------------------------------------------------


def rows_in_zone(zone: Zone) -> tuple[RootEntry, ...]:
    """The rows of *zone*, in declaration order."""
    return tuple(row for row in ROOT_ENTRIES if row.zone is zone)


def convention_rows() -> tuple[RootEntry, ...]:
    """The rows the build renders as convention directories, in declaration order."""
    return tuple(row for row in ROOT_ENTRIES if row.convention)


def rows_with_init(role: InitRole) -> tuple[RootEntry, ...]:
    """The rows ``osprey init`` treats as *role*, in declaration order."""
    return tuple(row for row in ROOT_ENTRIES if row.init is role)


def zoned_rows() -> tuple[RootEntry, ...]:
    """The rows that belong to a zone, in declaration order."""
    return tuple(row for row in ROOT_ENTRIES if row.zone is not None)


def non_convention_source_rows() -> tuple[RootEntry, ...]:
    """The source-zone rows the build does not render as conventions, in declaration order."""
    return tuple(row for row in ROOT_ENTRIES if row.zone is Zone.SOURCE and not row.convention)


def image_excluded_rows() -> tuple[RootEntry, ...]:
    """The rows never copied into a container image by their fixed name."""
    return tuple(row for row in ROOT_ENTRIES if row.image_excluded)


def layer_rows() -> tuple[RootEntry, ...]:
    """The live layer directories, in declaration order (their listing order)."""
    return tuple(row for row in ROOT_ENTRIES if row.layer_kind is not None)


def paths(rows: Iterable[RootEntry]) -> tuple[str, ...]:
    """The live path of each row, in the order given."""
    return tuple(row.path for row in rows)


def retired_paths(rows: Iterable[RootEntry]) -> tuple[str, ...]:
    """Every retired path of the rows, in the order given."""
    return tuple(retired for row in rows for retired in row.retired_paths)


def retired_moves() -> tuple[RootMove, ...]:
    """Every moved-from path with its successor, rows first, each in declaration order.

    A child's successor is its parent row's path joined with the child's own.
    """
    return _moves(ROOT_ENTRIES, ROOT_CHILDREN)


def _moves(rows: Iterable[RootEntry], children: Iterable[RootChild]) -> tuple[RootMove, ...]:
    """The moves *rows* and *children* declare; a child of no row in *rows* declares none.

    A child's :attr:`RootChild.moved_into` pairs follow its own moved-from
    paths, each successor being the child's path joined with the pair's subpath.
    """
    rows = tuple(rows)
    children = tuple(children)
    parents = {row.key: row.path for row in rows}
    return (
        *(RootMove(old, row.path, row.kind) for row in rows for old in row.moved_from),
        *(
            move
            for item in children
            if item.parent in parents
            for move in (
                *(
                    RootMove(old, f"{parents[item.parent]}/{item.path}", EntryKind.EITHER)
                    for old in item.moved_from
                ),
                *(
                    RootMove(old, f"{parents[item.parent]}/{item.path}/{sub}", EntryKind.EITHER)
                    for old, sub in item.moved_into
                ),
            )
        ),
    )


def rendered_sources() -> tuple[RenderedSource, ...]:
    """Every declared source the build copies into the render, rows first, each in declaration order.

    A child's source is its parent row's path joined with the child's own.
    """
    return _rendered(ROOT_ENTRIES, ROOT_CHILDREN)


def _rendered(
    rows: Iterable[RootEntry], children: Iterable[RootChild]
) -> tuple[RenderedSource, ...]:
    rows = tuple(rows)
    parents = {row.key: row.path for row in rows}
    return (
        *(RenderedSource(row.key, row.path, row.render) for row in rows if row.render is not None),
        *(
            RenderedSource(
                f"{item.parent}:{item.key}", f"{parents[item.parent]}/{item.path}", item.render
            )
            for item in children
            if item.parent in parents and item.render is not None
        ),
    )


def dissolved_names() -> tuple[str, ...]:
    """The top-level names moved-from paths sit under that no move names whole, in first-seen order.

    What an unmigrated deployment holds at its root only as a container of
    moved content: the name itself is no live entry and no move's old path,
    so nothing succeeds it and it is emptied rather than moved.
    """
    moves = retired_moves()
    whole = {move.old for move in moves}
    live = set(top_level_names(ROOT_ENTRIES))
    return tuple(
        name
        for name in dict.fromkeys(move.old.split("/", 1)[0] for move in moves)
        if name not in whole and name not in live
    )


def former_locations(path: str) -> tuple[str, ...]:
    """Each moved-from path the repo-relative live *path* had, in :func:`retired_moves` order.

    A move to *path* gives its own old path; a move to a directory holding
    *path* gives the old directory joined with the rest of *path*.
    """
    return tuple(
        move.old + path[len(move.new) :]
        for move in retired_moves()
        if path == move.new or path.startswith(f"{move.new}/")
    )


def moved_from_top_level_names(rows: Iterable[RootEntry]) -> tuple[str, ...]:
    """The first segment of each row's moved-from paths, once each, in first-seen order.

    What an unmigrated deployment still holds at its root for these rows, and
    nothing else: a first segment a row of *rows* still occupies (a moved-from
    path under ``agent`` while a row lives under ``agent``) is a live name, not
    a moved-from one.
    """
    rows = tuple(rows)
    live = set(top_level_names(rows))
    return tuple(
        name
        for name in dict.fromkeys(old.split("/", 1)[0] for row in rows for old in row.moved_from)
        if name not in live
    )


def top_level_names(rows: Iterable[RootEntry]) -> tuple[str, ...]:
    """The first segment of each row's path, once each, in first-seen order.

    What a nested entry occupies at the deployment root: ``agent/rules`` and
    ``agent/watchdogs`` both answer ``agent``.
    """
    return tuple(dict.fromkeys(row.path.split("/", 1)[0] for row in rows))


# --- display and location --------------------------------------------------------


def display_path(key: str, *parts: str) -> str:
    """How a message names the row *key*, or a path inside it, relative to the root.

    ``display_path("rules", "x.md")`` is the row's path joined with ``x.md``.
    """
    return "/".join((entry(key).path, *parts))


def listing_name(row: RootEntry) -> str:
    """The row as a listing names it: a directory carries a trailing slash."""
    return f"{row.path}/" if row.kind is EntryKind.DIRECTORY else row.path


def child_path(parent_dir: Path, parent: str, key: str) -> Path:
    """The child *key* joined to *parent_dir*, wherever the row *parent* resolved."""
    return parent_dir.joinpath(*child(parent, key).path.split("/"))


def child_display_path(parent: str, key: str, *parts: str) -> str:
    """How a message names the child, or a path inside it, relative to the root.

    The parent row's path joined with the child's own, then with *parts*:
    ``child_display_path("web", "landing", "notice.md")``.
    """
    return display_path(parent, child(parent, key).path, *parts)


def watchdogs_source_dir(repo_root: Path) -> Path:
    """The directory under *repo_root* that holds the authored ``<name>/`` watchdogs.

    The source side only: the render's watchdogs directory and the profile key
    that enables them are named elsewhere and do not follow this row.
    """
    return repo_root.joinpath(*entry("watchdogs").path.split("/"))


# --- substitution and invariants -------------------------------------------------


def with_paths(new_paths: Mapping[str, str]) -> tuple[RootEntry, ...]:
    """A copy of :data:`ROOT_ENTRIES` with the named rows' paths replaced.

    Keys, roles and order are unchanged. The result is checked by
    :func:`check_layout` against the current children.

    Raises:
        KeyError: A key names no row.
        ValueError: The substituted layout breaks an invariant.
    """
    unknown = set(new_paths) - {row.key for row in ROOT_ENTRIES}
    if unknown:
        raise KeyError(f"no deployment-root entry has the key(s) {sorted(unknown)}")
    rows = tuple(
        replace(row, path=new_paths[row.key]) if row.key in new_paths else row
        for row in ROOT_ENTRIES
    )
    check_layout(rows, ROOT_CHILDREN)
    return rows


def _relative_posix_problem(path: str) -> str | None:
    if not path:
        return "is empty"
    if path.startswith("/") or path.endswith("/"):
        return "has a leading or trailing slash"
    if "\\" in path:
        return "is not a POSIX path"
    if any(segment in ("", ".", "..") for segment in path.split("/")):
        return "has an empty, '.' or '..' segment"
    return None


def _nested_pairs(values: Iterable[str]) -> list[tuple[str, str]]:
    """Each ``(inner, outer)`` pair where *inner* lies strictly inside *outer*, once."""
    unique = list(dict.fromkeys(values))
    return [
        (inner, outer)
        for inner in unique
        for outer in unique
        if PurePosixPath(outer) in PurePosixPath(inner).parents
    ]


def check_layout(rows: Iterable[RootEntry], children: Iterable[RootChild] = ()) -> None:
    """Raise when *rows* and *children* do not describe one consistent layout.

    Keys are unique; paths are relative POSIX paths and unique; no path lies
    inside another, except one retired path inside another retired path; a
    convention or layer row is a source-zone directory; every child names a
    directory row as its parent, and a parent's children have unique keys and
    unique paths, none inside another; an image-excluded row is a single
    segment. A moved-from path is a relative
    POSIX path declared once, is no live or retired path, holds no live path
    (its successor included), lies inside no successor of its own, and lies
    inside another live path only when that is a directory row that is neither
    a convention nor a layer. No moved-from path lies inside another
    moved-from path or inside a retired path. A render path is a relative
    POSIX path declared once and lies inside no other render path.

    Raises:
        ValueError: Naming every broken invariant, each overlapping pair once.
    """
    rows = tuple(rows)
    problems: list[str] = []

    def _duplicates(values: Iterable[str]) -> list[str]:
        return sorted(value for value, count in Counter(values).items() if count > 1)

    if dup := _duplicates(row.key for row in rows):
        problems.append(f"duplicate keys {dup}")

    live = [row.path for row in rows]
    retired = [path for row in rows for path in row.retired_paths]
    for label, path in [*(("path", p) for p in live), *(("retired path", p) for p in retired)]:
        if (problem := _relative_posix_problem(path)) is not None:
            problems.append(f"{label} {path!r} {problem}")
    if dup := _duplicates([*live, *retired]):
        problems.append(f"paths declared twice {dup}")
    retired_only = set(retired) - set(live)
    problems.extend(
        f"{inner!r} lies inside {outer!r}"
        for inner, outer in _nested_pairs([*live, *retired])
        if not (inner in retired_only and outer in retired_only)
    )
    for row in rows:
        if not (row.convention or row.layer_kind is not None):
            continue
        if row.zone is not Zone.SOURCE:
            problems.append(f"{row.key!r} is a convention or layer row outside the source zone")
        if row.kind is not EntryKind.DIRECTORY:
            problems.append(f"{row.key!r} is a convention or layer row that is not a directory")
    problems.extend(
        f"{row.key!r} is kept out of an image as one top-level name of the root, "
        f"so its path {row.path!r} must be a single segment"
        for row in rows
        if row.image_excluded and "/" in row.path
    )

    directories = {row.key for row in rows if row.kind is EntryKind.DIRECTORY}
    children = tuple(children)
    for item in children:
        if item.parent not in directories:
            problems.append(f"child {item.key!r} names {item.parent!r}, not a directory row")
        if (problem := _relative_posix_problem(item.path)) is not None:
            problems.append(f"child path {item.path!r} {problem}")
    if dup := _duplicates(f"{item.parent}:{item.key}" for item in children):
        problems.append(f"duplicate children {dup}")
    for parent in dict.fromkeys(item.parent for item in children):
        own = [item.path for item in children if item.parent == parent]
        if dup := _duplicates(own):
            problems.append(f"children of {parent!r} share the path(s) {dup}")
        problems.extend(
            f"child path {inner!r} of {parent!r} lies inside {outer!r}"
            for inner, outer in _nested_pairs(own)
        )

    problems.extend(_moved_problems(rows, children))
    problems.extend(_render_problems(rows, children))

    if problems:
        raise ValueError("inconsistent deployment-root layout: " + "; ".join(problems))


def _moved_problems(rows: tuple[RootEntry, ...], children: tuple[RootChild, ...]) -> list[str]:
    """What is wrong with the moved-from paths of *rows* and *children*."""
    row_paths = {row.key: row.path for row in rows}
    moves = [(move.old, move.new) for move in _moves(rows, children)]
    # Every live path at the root, mapped to whether an old path may lie inside it.
    holders = {
        row.path: row.kind is EntryKind.DIRECTORY and not row.convention and row.layer_kind is None
        for row in rows
    }
    for item in children:
        if item.parent in row_paths:
            holders.setdefault(f"{row_paths[item.parent]}/{item.path}", False)
    retired = {path for row in rows for path in row.retired_paths}

    problems: list[str] = []
    if dup := sorted(old for old, count in Counter(old for old, _ in moves).items() if count > 1):
        problems.append(f"moved-from paths declared twice {dup}")
    problems.extend(
        f"moved-from path {inner!r} lies inside the moved-from path {outer!r}"
        for inner, outer in _nested_pairs(old for old, _ in moves)
    )
    for old, new in dict.fromkeys(moves):
        if (problem := _relative_posix_problem(old)) is not None:
            problems.append(f"moved-from path {old!r} {problem}")
            continue
        if old in holders:
            problems.append(f"moved-from path {old!r} is a live path")
            continue
        if old in retired:
            problems.append(f"moved-from path {old!r} is also a retired path")
        problems.extend(
            f"moved-from path {old!r} lies inside the retired path {outer!r}"
            for outer in sorted(retired)
            if PurePosixPath(outer) in PurePosixPath(old).parents
        )
        here = PurePosixPath(old)
        if here in PurePosixPath(new).parents:
            problems.append(f"moved-from path {old!r} holds its successor {new!r}")
        elif PurePosixPath(new) in here.parents:
            problems.append(f"moved-from path {old!r} lies inside its successor {new!r}")
        for live, may_hold in holders.items():
            if live == new:
                continue
            if here in PurePosixPath(live).parents:
                problems.append(f"moved-from path {old!r} holds the live path {live!r}")
            elif PurePosixPath(live) in here.parents and not may_hold:
                problems.append(
                    f"moved-from path {old!r} lies inside {live!r}, "
                    "which is not a plain directory row"
                )
    return problems


def _render_problems(rows: tuple[RootEntry, ...], children: tuple[RootChild, ...]) -> list[str]:
    """What is wrong with the render paths of *rows* and *children*."""
    rendered = _rendered(rows, children)
    problems = [
        f"render path {source.render_rel!r} of {source.key!r} {problem}"
        for source in rendered
        if (problem := _relative_posix_problem(source.render_rel)) is not None
    ]
    renders = [source.render_rel for source in rendered]
    if dup := sorted(path for path, count in Counter(renders).items() if count > 1):
        problems.append(f"render paths declared twice {dup}")
    problems.extend(
        f"render path {inner!r} lies inside the render path {outer!r}"
        for inner, outer in _nested_pairs(renders)
    )
    return problems
