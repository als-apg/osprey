"""The deployment-root declaration: its invariants, its accessors, and its reach.

The derivation section applies the rules every table module follows to the
declaration and runs the pins of ``tests/test_repo_layout_tables.py`` against the
result, so the declaration provably carries every value those tables hold,
order included, before any table is computed from it.
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import tests.test_repo_layout_tables as table_pins
from osprey import repo_layout
from osprey.repo_layout import (
    EntryKind,
    InitRole,
    RootChild,
    RootEntry,
    RootMove,
    Zone,
    check_layout,
    child_display_path,
    child_path,
    convention_rows,
    display_path,
    entry,
    image_excluded_rows,
    layer_rows,
    listing_name,
    non_convention_source_rows,
    paths,
    retired_moves,
    retired_paths,
    rows_in_zone,
    rows_with_init,
    top_level_names,
    watchdogs_source_dir,
    with_paths,
    zoned_rows,
)

#: A layout with entries nested under two new top-level directories: four
#: agent-facing entries under ``agent/`` (one renamed) and the scripts under
#: ``ops/``.
NESTED_PATHS: dict[str, str] = {
    "rules": "agent/rules",
    "watchdogs": "agent/watchdogs",
    "mcp_servers": "agent/mcp_servers",
    "web-terminal-context": "agent/context",
    "scripts": "ops/scripts",
}


@pytest.fixture
def nested_layout(monkeypatch: pytest.MonkeyPatch) -> tuple[RootEntry, ...]:
    rows = with_paths(NESTED_PATHS)
    monkeypatch.setattr(repo_layout, "ROOT_ENTRIES", rows)
    return rows


# --- invariants ------------------------------------------------------------------


def test_the_declared_layout_is_consistent() -> None:
    check_layout(repo_layout.ROOT_ENTRIES, repo_layout.ROOT_CHILDREN)


def test_keys_and_paths_are_unique_and_relative_posix() -> None:
    rows = repo_layout.ROOT_ENTRIES
    assert len({row.key for row in rows}) == len(rows)
    assert len({row.path for row in rows}) == len(rows)
    for path in [*paths(rows), *retired_paths(rows), *(c.path for c in repo_layout.ROOT_CHILDREN)]:
        assert path and not path.startswith("/") and not path.endswith("/"), path
        assert "\\" not in path and ".." not in path.split("/"), path


def test_every_key_equals_its_path_its_name_or_the_path_it_moved_from() -> None:
    assert all(
        row.key in (row.path, row.path.rsplit("/", 1)[-1]) or row.key in row.moved_from
        for row in repo_layout.ROOT_ENTRIES
    )


def test_no_retired_path_is_a_live_path() -> None:
    live = set(paths(repo_layout.ROOT_ENTRIES))
    assert live.isdisjoint(retired_paths(repo_layout.ROOT_ENTRIES))


def test_only_the_layer_directory_carries_a_retired_path() -> None:
    carriers = {row.key: row.retired_paths for row in repo_layout.ROOT_ENTRIES if row.retired_paths}
    assert carriers == {"tiers": ("personas",)}  # retired-spelling


def test_every_row_states_its_purpose() -> None:
    for item in (*repo_layout.ROOT_ENTRIES, *repo_layout.ROOT_CHILDREN):
        assert item.purpose and item.purpose == item.purpose.strip(), item.key


def test_a_convention_purpose_is_the_convention_description() -> None:
    """The convention directory's description is the row's purpose, stated nowhere else."""
    assert {row.path: row.purpose for row in convention_rows()} == {
        pin["source"]: pin["description"] for pin in table_pins._CONVENTION_DIR_ROWS
    }


#: The rows that are directories; ``.git`` is the one row that may be either, and
#: every other row is a file.
_DIRECTORY_KEYS = frozenset(
    {
        "data",
        "tiers",
        "apps",
        ".osprey",
        "rules",
        "skills",
        "agents",
        "commands",
        "output-styles",
        "hooks",
        "web-terminal-context",
        "watchdogs",
        "code_review",
        "mcp_servers",
        "services",
        "project",
        "profiles",
        "scripts",
        "var",
        "build",
        ".claude",
    }
)


def test_every_row_declares_its_kind() -> None:
    kinds = {row.key: row.kind for row in repo_layout.ROOT_ENTRIES}
    assert {key for key, kind in kinds.items() if kind is EntryKind.DIRECTORY} == _DIRECTORY_KEYS
    assert {key for key, kind in kinds.items() if kind is EntryKind.EITHER} == {".git"}


def test_child_keys_name_roles() -> None:
    keys = {(item.parent, item.key) for item in repo_layout.ROOT_CHILDREN}
    assert keys == {
        ("scripts", "verify-script"),
        ("scripts", "boot-hook"),
        ("data", "facility"),
        ("data", "canopy-library"),
        ("data", "canopy-collections"),
        ("var", "runtime"),
    }
    for _, key in keys:
        assert key == key.lower() and "_" not in key and "/" not in key and "." not in key, key


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: (*rows, rows[0]), "duplicate keys"),
        (lambda rows: (*rows, replace(rows[0], key="other")), "paths declared twice"),
        (lambda rows: (replace(rows[0], path="/profile.yml"), *rows[1:]), "leading or trailing"),
        (lambda rows: (replace(rows[0], path="profile.yml/"), *rows[1:]), "leading or trailing"),
        (lambda rows: (replace(rows[0], path=""), *rows[1:]), "'' is empty"),
        (lambda rows: (replace(rows[0], path="a//b"), *rows[1:]), "empty, '.' or '..' segment"),
        (lambda rows: (replace(rows[0], path="a/../b"), *rows[1:]), "'..' segment"),
        (lambda rows: (replace(rows[0], path="a/./b"), *rows[1:]), "empty, '.' or '..' segment"),
        (lambda rows: (replace(rows[0], path="a\\b"), *rows[1:]), "not a POSIX path"),
        (
            lambda rows: (*rows, RootEntry("inside", "data/inside", EntryKind.DIRECTORY, "x.")),
            "'data/inside' lies inside 'data'",
        ),
        (
            lambda rows: tuple(
                replace(row, retired_paths=("data/old",)) if row.key == "tiers" else row
                for row in rows
            ),
            "'data/old' lies inside 'data'",
        ),
        (
            lambda rows: tuple(
                replace(row, retired_paths=("agent/rules/old",)) if row.key == "tiers" else row
                for row in rows
            ),
            "'agent/rules/old' lies inside 'agent/rules'",
        ),
        (
            lambda rows: (
                tuple(
                    replace(row, retired_paths=("old",)) if row.key == "tiers" else row
                    for row in rows
                )
                + (RootEntry("inside", "old/inside", EntryKind.DIRECTORY, "x."),)
            ),
            "'old/inside' lies inside 'old'",
        ),
        (
            lambda rows: tuple(
                replace(row, retired_paths=("a\\b",)) if row.key == "tiers" else row for row in rows
            ),
            "retired path 'a\\\\b' is not a POSIX path",
        ),
        (
            lambda rows: tuple(
                replace(row, retired_paths=("profile.yml",)) if row.key == "tiers" else row
                for row in rows
            ),
            "paths declared twice ['profile.yml']",
        ),
        (
            lambda rows: tuple(
                replace(row, zone=Zone.SECRETS) if row.key == "rules" else row for row in rows
            ),
            "'rules' is a convention or layer row outside the source zone",
        ),
        (
            lambda rows: tuple(
                replace(row, kind=EntryKind.FILE) if row.key == "skills" else row for row in rows
            ),
            "'skills' is a convention or layer row that is not a directory",
        ),
        (
            lambda rows: tuple(
                replace(row, kind=EntryKind.FILE) if row.key == "tiers" else row for row in rows
            ),
            "'tiers' is a convention or layer row that is not a directory",
        ),
    ],
)
def test_a_broken_layout_is_refused(
    change: Callable[[tuple[RootEntry, ...]], tuple[RootEntry, ...]], message: str
) -> None:
    with pytest.raises(ValueError, match="inconsistent deployment-root layout") as caught:
        check_layout(change(repo_layout.ROOT_ENTRIES), repo_layout.ROOT_CHILDREN)
    assert message in str(caught.value)


def test_a_retired_path_may_lie_inside_another_retired_path() -> None:
    rows = tuple(
        replace(row, retired_paths=("old", "old/inner")) if row.key == "tiers" else row
        for row in repo_layout.ROOT_ENTRIES
    )
    check_layout(rows, repo_layout.ROOT_CHILDREN)


def test_a_misplaced_convention_file_is_reported_for_its_kind_alone() -> None:
    rows = tuple(
        replace(row, kind=EntryKind.FILE) if row.key == "skills" else row
        for row in repo_layout.ROOT_ENTRIES
    )
    with pytest.raises(ValueError) as caught:
        check_layout(rows, repo_layout.ROOT_CHILDREN)
    assert "outside the source zone" not in str(caught.value)


def test_an_image_excluded_row_must_be_one_segment() -> None:
    rows = tuple(
        replace(row, path="state/var") if row.key == "var" else row
        for row in repo_layout.ROOT_ENTRIES
    )
    with pytest.raises(ValueError) as caught:
        check_layout(rows, repo_layout.ROOT_CHILDREN)
    assert "'state/var' must be a single segment" in str(caught.value)


def test_an_overlap_is_reported_once() -> None:
    rows = (*repo_layout.ROOT_ENTRIES, RootEntry("inside", "data/inside", EntryKind.FILE, "x"))
    with pytest.raises(ValueError) as caught:
        check_layout(rows, repo_layout.ROOT_CHILDREN)
    assert str(caught.value).count("data/inside") == 1


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (RootChild("profile.yml", "x", "x", "x"), "names 'profile.yml', not a directory row"),
        (RootChild("data", "elsewhere", "../x", "x"), "child path '../x' has an empty"),
        (RootChild("data", "facility", "other", "x"), "duplicate children ['data:facility']"),
        (
            RootChild("data", "second-facility", "facility", "x"),
            "children of 'data' share the path(s) ['facility']",
        ),
        (
            RootChild("data", "facility-part", "facility/part", "x"),
            "child path 'facility/part' of 'data' lies inside 'facility'",
        ),
    ],
)
def test_a_broken_child_is_refused(extra: RootChild, message: str) -> None:
    with pytest.raises(ValueError, match="inconsistent deployment-root layout") as caught:
        check_layout(repo_layout.ROOT_ENTRIES, (*repo_layout.ROOT_CHILDREN, extra))
    assert message in str(caught.value)


def test_children_of_different_parents_may_share_a_path() -> None:
    check_layout(
        repo_layout.ROOT_ENTRIES,
        (*repo_layout.ROOT_CHILDREN, RootChild("tiers", "facility", "facility", "x")),
    )


# --- moved paths -----------------------------------------------------------------


#: Planted ``moved_from`` declarations over :data:`NESTED_PATHS`, beside the live
#: ones, so the projection and its invariants are exercised on a wider layout.
PLANTED_MOVES: dict[str, tuple[str, ...]] = {
    "rules": ("rules",),
    "web-terminal-context": ("web-terminal-context", "context"),
    "scripts": ("scripts",),
}


def _planted_rows(
    moves: dict[str, tuple[str, ...]] = PLANTED_MOVES,
    new_paths: dict[str, str] = NESTED_PATHS,
) -> tuple[RootEntry, ...]:
    return tuple(
        replace(row, moved_from=moves[row.key]) if row.key in moves else row
        for row in with_paths(new_paths)
    )


def _planted_children() -> tuple[RootChild, ...]:
    """The live children, whose runtime child moved out of ``data`` to ``var/runtime``."""
    return repo_layout.ROOT_CHILDREN


def _move_row(rows: tuple[RootEntry, ...], key: str, *old: str) -> tuple[RootEntry, ...]:
    return tuple(replace(row, moved_from=old) if row.key == key else row for row in rows)


#: What the agent is made of, grouped under ``agent/`` at the root: each row's
#: moved-from path and its successor, in declaration order.
AGENT_MOVES = (
    ("rules", "agent/rules"),
    ("skills", "agent/skills"),
    ("agents", "agent/agents"),
    ("commands", "agent/commands"),
    ("output-styles", "agent/output-styles"),
    ("hooks", "agent/hooks"),
    ("web-terminal-context", "agent/context"),
    ("watchdogs", "agent/watchdogs"),
    ("data/code_review", "agent/code_review"),
    ("mcp_servers", "agent/mcp_servers"),
)

#: The top-level names an unmigrated repo holds for those rows: ``data`` is still
#: a live name, so ``data/code_review`` contributes none.
AGENT_GROUP = (
    "rules",
    "skills",
    "agents",
    "commands",
    "output-styles",
    "hooks",
    "web-terminal-context",
    "watchdogs",
    "mcp_servers",
)

#: What runs the deployment, grouped under ``ops/`` at the root: each row's
#: moved-from path, its successor and its kind, in declaration order.
OPS_MOVES = (
    ("services", "ops/services", EntryKind.DIRECTORY),
    ("project", "ops/project", EntryKind.DIRECTORY),
    ("scripts", "ops/scripts", EntryKind.DIRECTORY),
    ("osprey.service", "ops/osprey.service", EntryKind.FILE),
    ("ci-extra.yml", "ops/ci-extra.yml", EntryKind.FILE),
)

#: The top-level names an unmigrated repo holds for the ``ops/`` rows.
OPS_GROUP = tuple(old for old, _new, _kind in OPS_MOVES)


def test_the_live_moves_group_the_agent_and_ops_trees_and_the_runtime_child() -> None:
    assert retired_moves() == (
        *(RootMove(old, new, EntryKind.DIRECTORY) for old, new in AGENT_MOVES),
        *(RootMove(old, new, kind) for old, new, kind in OPS_MOVES),
        RootMove("data/.runtime", "var/runtime", EntryKind.EITHER),
    )
    assert [(item.parent, item.key) for item in repo_layout.ROOT_CHILDREN if item.moved_from] == [
        ("var", "runtime")
    ]


def test_the_moved_from_top_level_names_are_the_first_segments_once_each() -> None:
    assert repo_layout.moved_from_top_level_names(repo_layout.ROOT_ENTRIES) == (
        *AGENT_GROUP,
        *OPS_GROUP,
    )
    rows = _move_row(repo_layout.ROOT_ENTRIES, "watchdogs", "old/watchdogs", "old/more")
    assert repo_layout.moved_from_top_level_names(rows) == (
        *AGENT_GROUP[:-2],
        "old",
        "mcp_servers",
        *OPS_GROUP,
    )


def test_a_moved_from_first_segment_that_is_still_live_is_no_moved_from_name() -> None:
    assert "data" in top_level_names(repo_layout.ROOT_ENTRIES)
    assert "data" not in repo_layout.moved_from_top_level_names(repo_layout.ROOT_ENTRIES)


def test_the_retired_moves_project_planted_rows_in_declaration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _planted_rows()
    monkeypatch.setattr(repo_layout, "ROOT_ENTRIES", rows)
    monkeypatch.setattr(repo_layout, "ROOT_CHILDREN", _planted_children())
    expected_rows = [RootMove(old, row.path, row.kind) for row in rows for old in row.moved_from]
    assert retired_moves() == (
        *expected_rows,
        RootMove("data/.runtime", "var/runtime", EntryKind.EITHER),
    )
    assert RootMove("rules", "agent/rules", EntryKind.DIRECTORY) in expected_rows
    old_order = [move.old for move in expected_rows]
    assert old_order.index("context") == old_order.index("web-terminal-context") + 1


def test_the_retired_moves_leave_the_retired_paths_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = retired_paths(repo_layout.ROOT_ENTRIES)
    monkeypatch.setattr(repo_layout, "ROOT_ENTRIES", _planted_rows())
    assert retired_paths(repo_layout.ROOT_ENTRIES) == declared == ("personas",)  # retired-spelling
    assert "personas" not in {move.old for move in retired_moves()}  # retired-spelling


def test_a_planted_moved_layout_passes_the_invariants() -> None:
    check_layout(_planted_rows(), _planted_children())


def test_a_moved_path_may_lie_inside_a_plain_directory_row() -> None:
    """``data`` is a directory row that is no convention and no layer."""
    assert entry("code_review").moved_from == ("data/code_review",)
    check_layout(_planted_rows(), _planted_children())


def test_a_data_child_old_path_is_refused_while_the_child_lives() -> None:
    children = (*_planted_children(), RootChild("data", "code-review", "code_review", "x."))
    with pytest.raises(ValueError, match="inconsistent deployment-root layout") as caught:
        check_layout(_planted_rows(), children)
    assert "moved-from path 'data/code_review' is a live path" in str(caught.value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda rows: _move_row(rows, "rules", "profile.yml"),
            "moved-from path 'profile.yml' is a live path",
        ),
        (
            lambda rows: _move_row(rows, "rules", "agent/watchdogs"),
            "moved-from path 'agent/watchdogs' is a live path",
        ),
        (
            lambda rows: _move_row(rows, "rules", "data/facility"),
            "moved-from path 'data/facility' is a live path",
        ),
        (
            lambda rows: _move_row(rows, "rules", "agent"),
            "moved-from path 'agent' holds its successor 'agent/rules'",
        ),
        (
            lambda rows: _move_row(rows, "scripts", "agent"),
            "moved-from path 'agent' holds the live path 'agent/rules'",
        ),
        (
            lambda rows: _move_row(rows, "rules", "agent/rules/old"),
            "moved-from path 'agent/rules/old' lies inside its successor 'agent/rules'",
        ),
        (
            lambda rows: _move_row(rows, "skills", "rules"),
            "moved-from paths declared twice ['rules']",
        ),
        (
            lambda rows: _move_row(rows, "rules", "rules", "rules"),
            "moved-from paths declared twice ['rules']",
        ),
        (
            lambda rows: _move_row(rows, "rules", "personas"),  # retired-spelling
            "moved-from path 'personas' is also a retired path",  # retired-spelling
        ),
        (
            lambda rows: _move_row(rows, "rules", "tiers/rules"),
            "moved-from path 'tiers/rules' lies inside 'tiers'",
        ),
        (
            lambda rows: _move_row(rows, "rules", "profile.yml/rules"),
            "moved-from path 'profile.yml/rules' lies inside 'profile.yml'",
        ),
        (
            lambda rows: _move_row(rows, "rules", "data/facility/rules"),
            "moved-from path 'data/facility/rules' lies inside 'data/facility'",
        ),
        (
            lambda rows: _move_row(rows, "rules", "/rules"),
            "moved-from path '/rules' has a leading or trailing slash",
        ),
        (
            lambda rows: _move_row(rows, "rules", "a/../rules"),
            "moved-from path 'a/../rules' has an empty, '.' or '..' segment",
        ),
    ],
)
def test_a_broken_moved_path_is_refused(
    change: Callable[[tuple[RootEntry, ...]], tuple[RootEntry, ...]], message: str
) -> None:
    with pytest.raises(ValueError, match="inconsistent deployment-root layout") as caught:
        check_layout(change(_planted_rows()), _planted_children())
    assert message in str(caught.value)


@pytest.mark.parametrize(
    ("moved_from", "message"),
    [
        (("var",), "moved-from path 'var' is a live path"),
        (("var/runtime/old",), "moved-from path 'var/runtime/old' lies inside its successor"),
        (("rules",), "moved-from paths declared twice ['rules']"),
        (("data/canopy/library",), "moved-from path 'data/canopy/library' is a live path"),
        (("data/.runtime/x/../y",), "has an empty, '.' or '..' segment"),
    ],
)
def test_a_broken_moved_child_is_refused(moved_from: tuple[str, ...], message: str) -> None:
    children = tuple(
        replace(item, moved_from=moved_from) if item.key == "runtime" else item
        for item in _planted_children()
    )
    with pytest.raises(ValueError, match="inconsistent deployment-root layout") as caught:
        check_layout(_planted_rows(), children)
    assert message in str(caught.value)


def test_a_child_old_path_holding_its_successor_is_refused() -> None:
    children = (
        *_planted_children(),
        RootChild("data", "deep", "x/y", "x.", moved_from=("data/x",)),
    )
    with pytest.raises(ValueError) as caught:
        check_layout(_planted_rows(), children)
    assert "moved-from path 'data/x' holds its successor 'data/x/y'" in str(caught.value)


# --- lookup and accessors --------------------------------------------------------


def test_entry_returns_the_row_and_refuses_an_unknown_key() -> None:
    assert entry("tiers").layer_kind == "tier"
    with pytest.raises(KeyError, match="'nowhere'"):
        entry("nowhere")


def test_accessors_join_the_declared_paths(tmp_path: Path) -> None:
    assert display_path("rules") == "agent/rules"
    assert display_path("rules", "x.md") == "agent/rules/x.md"
    assert child_display_path("scripts", "verify-script") == "ops/scripts/verify.sh"
    assert child_display_path("scripts", "boot-hook") == "ops/scripts/osprey-boot-hook.sh"
    assert child_display_path("data", "facility") == "data/facility"
    assert child_display_path("data", "canopy-library") == "data/canopy/library"
    assert child_display_path("data", "canopy-collections") == "data/canopy/collections"
    assert child_display_path("var", "runtime") == "var/runtime"
    assert display_path("code_review", "reviewer.md") == "agent/code_review/reviewer.md"
    assert watchdogs_source_dir(tmp_path) == tmp_path / "agent" / "watchdogs"


def test_a_data_child_joins_wherever_the_data_root_resolves(tmp_path: Path) -> None:
    data_root = tmp_path / "elsewhere"
    assert child_path(data_root, "data", "facility") == data_root / "facility"
    assert child_path(data_root, "data", "canopy-library") == data_root / "canopy" / "library"
    with pytest.raises(KeyError, match="no child 'nowhere'"):
        child_path(data_root, "data", "nowhere")


def test_listing_name_marks_only_directories() -> None:
    assert listing_name(entry("rules")) == "agent/rules/"
    assert listing_name(entry("profile.yml")) == "profile.yml"
    assert listing_name(entry(".git")) == ".git"


def test_selection_follows_declaration_order() -> None:
    assert paths(rows_with_init(InitRole.UNTOUCHED)) == (".env", ".git", "var", "build")
    assert paths(rows_in_zone(Zone.GENERATED)) == ("var", "build")
    assert paths(layer_rows()) == ("tiers",)
    assert top_level_names(convention_rows())[:2] == ("agent", "ops")
    assert paths(zoned_rows())[:2] == ("profile.yml", "providers.yml")
    assert ".gitignore" not in paths(zoned_rows())
    assert "agent/rules" not in paths(non_convention_source_rows())
    assert paths(non_convention_source_rows())[-1] == "ops/ci-extra.yml"


# --- substitution ----------------------------------------------------------------


def test_with_paths_replaces_only_the_named_paths() -> None:
    rows = with_paths({"scripts": "tools/scripts"})
    before = {row.key: row for row in repo_layout.ROOT_ENTRIES}
    assert [row.key for row in rows] == list(before)
    for row in rows:
        expected = (
            replace(before[row.key], path="tools/scripts")
            if row.key == "scripts"
            else before[row.key]
        )
        assert row == expected
    assert entry("scripts").path == "ops/scripts", "the registry itself is not changed"


def test_with_paths_refuses_an_unknown_key_and_a_broken_layout() -> None:
    with pytest.raises(KeyError, match="nowhere"):
        with_paths({"nowhere": "x"})
    with pytest.raises(ValueError, match="paths declared twice"):
        with_paths({"skills": "agent/rules"})
    with pytest.raises(ValueError, match="'agent/rules/skills' lies inside 'agent/rules'"):
        with_paths({"skills": "agent/rules/skills"})


def test_a_nested_layout_passes_the_invariants(nested_layout: tuple[RootEntry, ...]) -> None:
    check_layout(nested_layout, repo_layout.ROOT_CHILDREN)
    assert {row.key for row in nested_layout} == {row.key for row in repo_layout.ROOT_ENTRIES}


@pytest.mark.usefixtures("nested_layout")
def test_lookups_follow_a_rebound_registry(tmp_path: Path) -> None:
    assert entry("rules").path == "agent/rules"
    assert display_path("rules", "x.md") == "agent/rules/x.md"
    assert listing_name(entry("web-terminal-context")) == "agent/context/"
    assert child_display_path("scripts", "verify-script") == "ops/scripts/verify.sh"
    assert child_display_path("data", "facility") == "data/facility"
    assert watchdogs_source_dir(tmp_path) == tmp_path / "agent" / "watchdogs"
    assert paths(rows_with_init(InitRole.SEEDED)) == ("agent/mcp_servers",)


def test_a_nested_layout_changes_the_top_level_names(monkeypatch: pytest.MonkeyPatch) -> None:
    def known_names() -> set[str]:
        zoned = zoned_rows()
        return {*top_level_names(zoned), *retired_paths(zoned)}

    declared = known_names()
    regrouped = {"profiles": "config/profiles", "triggers.yml": "config/triggers.yml"}
    monkeypatch.setattr(repo_layout, "ROOT_ENTRIES", with_paths(regrouped))
    moved = set(regrouped)
    assert moved | {"agent", "ops"} <= declared
    assert "config" not in declared
    assert known_names() == (declared - moved) | {"config"}


# --- reach -----------------------------------------------------------------------


def test_importing_the_declaration_loads_no_cli_module_and_no_yaml() -> None:
    probe = (
        "import sys\n"
        "import osprey.repo_layout\n"
        "print('\\n'.join(sorted(m for m in sys.modules\n"
        "    if m.startswith('osprey.cli') or m.split('.')[0] in ('yaml', 'ruamel', '_ruamel_yaml'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == []


# --- derivation: the declaration reproduces the pinned tables --------------------


def _derived_tables() -> dict[str, types.SimpleNamespace]:
    """Every pinned table, computed from the declaration by the derivation rules."""
    source = rows_in_zone(Zone.SOURCE)
    source_plain = non_convention_source_rows()
    zoned = zoned_rows()

    # Render attributes keyed by the row's path, walked in declaration order; a
    # path on one side only is an error. The source is the row's path and the
    # description its purpose, so the mapping carries neither.
    by_path = {
        pin["source"]: {
            field: value for field, value in pin.items() if field not in ("source", "description")
        }
        for pin in table_pins._CONVENTION_DIR_ROWS
    }
    assert set(paths(convention_rows())) == set(by_path), (
        "convention rows and render attributes disagree"
    )
    render = {row.key: by_path[row.path] for row in convention_rows()}
    keys = list(render)
    convention_dirs = tuple(
        types.SimpleNamespace(**render[row.key], source=row.path, description=row.purpose)
        for row in convention_rows()
    )

    write_once = rows_with_init(InitRole.WRITE_ONCE)
    ci_emitted = (
        *paths(rows_with_init(InitRole.SCAFFOLDED)),
        child_display_path("scripts", "verify-script"),
    )
    seeded = rows_with_init(InitRole.SEEDED)
    (facility_rule,) = rows_with_init(InitRole.FACILITY_RULE)
    untouched = tuple(listing_name(row) for row in rows_with_init(InitRole.UNTOUCHED))
    layers = layer_rows()

    return {
        "profile_conventions": types.SimpleNamespace(
            _SOURCE_ZONE_ENTRIES=frozenset(
                (*top_level_names(source_plain), *retired_paths(source_plain))
            ),
            _SECRETS_ZONE_ENTRIES=frozenset(paths(rows_in_zone(Zone.SECRETS))),
            _GENERATED_ZONE_ENTRIES=frozenset(paths(rows_in_zone(Zone.GENERATED))),
            REPO_CLAUDE_CODE_ENTRIES=frozenset(paths(rows_in_zone(Zone.CLAUDE_CODE))),
            KNOWN_ROOT_ENTRIES=frozenset((*top_level_names(zoned), *retired_paths(zoned))),
            CONVENTION_SOURCES=tuple(keys),
            SELECTED_CONVENTION_SOURCES={
                render[key]["list_field"]: key
                for key in keys
                if render[key]["list_field"] is not None
            },
            CONVENTION_DIRS=convention_dirs,
        ),
        "profile_cmd": types.SimpleNamespace(
            MATERIALIZED_SOURCE_ENTRIES=paths(rows_with_init(InitRole.REPLACED)),
        ),
        "init_cmd": types.SimpleNamespace(
            WRITE_ONCE_FILES=dict.fromkeys(paths(write_once)),
            WRITE_ONCE_DIRS=dict.fromkeys(paths(seeded)),
            CI_EMITTED_PATHS=ci_emitted,
            FACILITY_RULE_DIR=facility_rule.key,
            PRESERVED_BY_FORCE=(
                *untouched,
                *paths(write_once),
                *ci_emitted,
                *(listing_name(row) for row in seeded),
                listing_name(facility_rule),
            ),
        ),
        "build_cmd": types.SimpleNamespace(
            _NON_SOURCE_ROOT_ENTRIES=frozenset(paths(image_excluded_rows())),
        ),
        "workspace": types.SimpleNamespace(
            LAYER_DIR_KINDS={row.path: row.layer_kind for row in layers},
            RETIRED_LAYER_DIRS={
                retired: row.path for row in layers for retired in row.retired_paths
            },
            LAYER_DIRNAMES=(*paths(layers), *retired_paths(layers)),
        ),
        "executor": types.SimpleNamespace(
            PROFILE_SOURCE_ENTRIES=(
                *top_level_names(source),
                *retired_paths(source),
                *repo_layout.moved_from_top_level_names(source),
            ),
        ),
    }


_TABLE_PINS = sorted(
    name
    for name, value in vars(table_pins).items()
    if name.startswith("test_") and isinstance(value, types.FunctionType)
)


@pytest.mark.parametrize("pin", _TABLE_PINS)
def test_the_declaration_reproduces_the_pinned_table(pin: str) -> None:
    """Run one table pin with its table modules replaced by the derived values."""
    original = getattr(table_pins, pin)
    derived = _derived_tables()
    # Every module the pin names is a table module and must be replaced. The
    # helpers assertion rewriting adds are not identifiers and are skipped.
    named_modules = {
        name
        for name in original.__code__.co_names
        if name.isidentifier() and isinstance(original.__globals__.get(name), types.ModuleType)
    }
    assert named_modules <= set(derived), f"{pin} reads table modules with no derivation"
    scope = {**original.__globals__, **derived}
    types.FunctionType(original.__code__, scope, original.__name__)()


def test_the_table_pins_are_found() -> None:
    assert "test_profile_source_entries" in _TABLE_PINS


def test_a_pin_reading_an_unreplaced_module_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness fails a pin whose table module it does not derive, never runs it."""

    def pin() -> None:
        assert unreplaced.VALUE == 1  # type: ignore[name-defined]  # noqa: F821

    monkeypatch.setattr(table_pins, "test_unreplaced", pin, raising=False)
    monkeypatch.setitem(pin.__globals__, "unreplaced", types.ModuleType("unreplaced"))
    with pytest.raises(AssertionError, match="test_unreplaced reads table modules"):
        test_the_declaration_reproduces_the_pinned_table("test_unreplaced")
