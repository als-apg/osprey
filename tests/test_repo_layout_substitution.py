"""Moving deployment-root entries by editing only the layout declaration.

A fresh interpreter rebinds the deployment-root registry to a layout where the
agent's material sits under ``agent/`` (``agent/rules``, ``agent/watchdogs``,
``agent/mcp_servers`` and the per-user context as ``agent/context``) and the
emitted scripts sit in ``ops/scripts``, before any consumer is imported. It then
imports the consumers and reports what they computed and did on a deployment
root in ``tmp_path`` laid out that way. Nothing in the repository moves.

Expected: every location follows the move, and nothing keyed by identity does.
The known-root set and the write boundary name the new top-level names, and the
boundary keeps every old name the live layout declares as moved from; a rule's copy,
artifact name, selection and claim are unchanged but for its profile path;
messages name the new paths; watchdogs are read from their new source and from
the unchanged render; fold labels, convention keys, render directories and the
config key stay as they are.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from osprey import repo_layout

#: Each moved deployment-root entry, by key, and where it sits in the moved layout.
_MOVED: dict[str, str] = {
    "rules": "agent/rules",
    "watchdogs": "agent/watchdogs",
    "mcp_servers": "agent/mcp_servers",
    "web-terminal-context": "agent/context",
    "scripts": "ops/scripts",
}

#: The names the moved entries occupied at the root before the move.
_OLD_NAMES = frozenset(_MOVED)

#: The old names the live layout already declares as moved-from paths: an
#: unmigrated tree may still hold them, so the write boundary keeps them.
_DECLARED_OLD = _OLD_NAMES & {move.old for move in repo_layout.retired_moves()}

#: What each moved entry occupies at the root in the moved layout.
_NEW_TOP_LEVEL = frozenset(path.split("/", 1)[0] for path in _MOVED.values())

#: The text of the authored watchdog, and of its render, so a read names its side.
_SOURCE_INSTRUCTIONS = "Watch from the authored source.\n"
_RENDER_INSTRUCTIONS = "Watch from the render.\n"

#: A deployment root in the moved layout: one rule (and one rule no list selects),
#: one watchdog, one MCP-server package, one user's context, the CI verifier, and
#: the watchdog's render under ``build/``.
_MOVED_ROOT: dict[str, str] = {
    "profile.yml": "name: moved\nwatchdogs:\n  sentinel:\n    enabled: true\n    engine: agent\n",
    "facility/identity.yaml": "name: moved\n",
    "web/landing/notice.md": "# notice\n",
    "agent/rules/x.md": "# x\n",
    "agent/rules/y.md": "# y\n",
    "agent/watchdogs/sentinel/instructions.md": _SOURCE_INSTRUCTIONS,
    "agent/mcp_servers/beamline/__init__.py": "",
    "agent/context/alice/CLAUDE.md": "# alice\n",
    "ops/scripts/verify.sh": "#!/bin/sh\n",
    "build/watchdogs/sentinel/instructions.md": _RENDER_INSTRUCTIONS,
}

#: The same deployment root with a wrong file under the moved rules directory.
_BAD_RULE_ROOT: dict[str, str] = {
    "profile.yml": "name: moved\n",
    "agent/rules/notes.txt": "not a rule\n",
}

#: A root holding the moved entries at their old names: in the moved layout
#: those names are no longer known.
_OLD_ROOT: dict[str, str] = {
    "profile.yml": "name: old\n",
    **{f"{name}/.keep": "" for name in sorted(_OLD_NAMES)},
}

#: Runs in a fresh interpreter: rebinds the registry, then imports the consumers.
_PROBE = r"""
import json
import sys
from pathlib import Path

consumers_before = sorted(
    name for name in sys.modules
    if name.startswith("osprey.") and name != "osprey.repo_layout"
)
from osprey import repo_layout

repo_layout.ROOT_ENTRIES = repo_layout.with_paths(json.loads(sys.argv[1]))

from osprey.cli import build_persistence, init_cmd, profile_conventions, scaffold_cmd
from osprey.cli import deploy_scaffold_templates
from osprey.cli.build_cmd import _NON_SOURCE_ROOT_ENTRIES, _stage_source_zone
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import profile_inputs, resolve_profile_document
from osprey.cli.build_profile_schema import McpServerDef
from osprey.cli.build_profile_watchdogs import copy_enabled_watchdogs
from osprey.deployment import compose_generator
from osprey.facility import settings as facility_settings
from osprey.mcp_server.python_executor import executor
from osprey.services.watchdog import config as watchdog_config
from osprey.services.watchdog.definitions import load_profile_watchdogs, load_watchdogs
from osprey_connectors.errors import BuildProfileError

moved, bad_rule, old, project, image, render = (Path(arg) for arg in sys.argv[2:8])


def rel(path, root):
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def files_under(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# --- the planned copies and what a build applies --------------------------------
copies = profile_conventions.plan_convention_copies(moved, context_users=["alice"])
planned = [
    {
        "category": c.category,
        "source": rel(c.source, moved),
        "destination": c.destination,
        "artifact": build_persistence._artifact_name(c),
        "owned": build_persistence.ownership_canonical(c),
    }
    for c in copies
]
applied = build_persistence._apply_conventions(
    moved,
    project,
    context_users=["alice", "bob"],
    selected=profile_conventions.convention_selections({"rules": ["rules/x"]}),
)

# --- a validation error ---------------------------------------------------------
try:
    profile_conventions.validate_convention_sources(bad_rule)
    validation_error = None
except BuildProfileError as exc:
    validation_error = str(exc)

# --- the claim target -----------------------------------------------------------
claim = scaffold_cmd.resolve_claim_target("rules/x")

# --- the fold -------------------------------------------------------------------
profile_path = moved / "profile.yml"
document = resolve_profile_document(_read_profile_document(profile_path), profile_path, warn=False)
fold = [
    [i.key, i.label, rel(i.path, moved)] for i in profile_inputs(document)
]

# --- watchdogs: authored source and render ---------------------------------------
from_source = load_profile_watchdogs(moved)
from_render = load_watchdogs(
    {watchdog_config.WATCHDOGS_BLOCK: {"sentinel": {"enabled": True, "engine": "agent"}}},
    moved / "build",
)
copied = copy_enabled_watchdogs(document.raw.get("watchdogs"), moved, render)

# --- the MCP-server pairing check -----------------------------------------------
build_zone = f"/build/{profile_conventions.convention_for('mcp_servers').destination}"
servers = {
    name: McpServerDef(command="python", args=["-m", module], env={"PYTHONPATH": build_zone})
    for name, module in (("paired", "beamline"), ("unpaired", "absent"))
}
unpaired = init_cmd._missing_server_packages(servers, moved)

# --- the image --------------------------------------------------------------------
_stage_source_zone(moved, image, build_inputs=())

print(json.dumps({
    "consumers_before": consumers_before,
    "tables": {
        "KNOWN_ROOT_ENTRIES": sorted(profile_conventions.KNOWN_ROOT_ENTRIES),
        "_SOURCE_ZONE_ENTRIES": sorted(profile_conventions._SOURCE_ZONE_ENTRIES),
        "CONVENTION_DIRS": [
            [c.key, c.source, c.destination] for c in profile_conventions.CONVENTION_DIRS
        ],
        "CONVENTION_SOURCES": list(profile_conventions.CONVENTION_SOURCES),
        "SELECTED_CONVENTION_SOURCES": profile_conventions.SELECTED_CONVENTION_SOURCES,
        "PER_USER_CONTEXT_DIRNAME": profile_conventions.PER_USER_CONTEXT_DIRNAME,
        "PROFILE_SOURCE_ENTRIES": list(executor.PROFILE_SOURCE_ENTRIES),
        "MCP_SERVER_SOURCE_DIR": init_cmd.MCP_SERVER_SOURCE_DIR,
        "FACILITY_RULE_DIR": init_cmd.FACILITY_RULE_DIR,
        "WRITE_ONCE_DIRS": dict(init_cmd.WRITE_ONCE_DIRS),
        "WRITE_ONCE_FILES": list(init_cmd.WRITE_ONCE_FILES),
        "CI_EMITTED_PATHS": list(init_cmd.CI_EMITTED_PATHS),
        "REPO_VERIFY_PATH": list(init_cmd.REPO_VERIFY_PATH),
        "VERIFY_PATH": deploy_scaffold_templates.VERIFY_PATH,
        "BOOT_HOOK_PATH": deploy_scaffold_templates.BOOT_HOOK_PATH,
        "_NON_SOURCE_ROOT_ENTRIES": sorted(_NON_SOURCE_ROOT_ENTRIES),
    },
    "render_constants": {
        "WATCHDOGS_BLOCK": watchdog_config.WATCHDOGS_BLOCK,
        "VOCABULARY_SOURCE": facility_settings.VOCABULARY_SOURCE,
        "settings._RENDER_FACILITY_DIR": facility_settings._RENDER_FACILITY_DIR,
        "profile_conventions._RENDER_FACILITY_DIR": profile_conventions._RENDER_FACILITY_DIR,
        "BUILD_OUTPUT_DIR": profile_conventions.BUILD_OUTPUT_DIR,
        "compose SERVICES_DIR": compose_generator.SERVICES_DIR,
    },
    "convention_source": {
        key: profile_conventions.convention_source(key)
        for key in ("rules", "mcp_servers", "web-terminal-context")
    },
    "unknown_moved": profile_conventions.unknown_root_entries(moved),
    "unknown_old": profile_conventions.unknown_root_entries(old),
    "protected": sorted(rel(p, moved) for p in executor.resolve_protected_roots(moved, entries=())),
    "planned": planned,
    "applied": {
        "owned": sorted(applied.owned_artifacts),
        "seeded": list(applied.seeded_users),
        "project": files_under(project),
        "profile": files_under(moved),
    },
    "selection": {
        "split": list(profile_conventions.split_convention_selection("rules", ["rules/x", "safety"])),
    },
    "validation_error": validation_error,
    "claim": {
        "name": claim.name,
        "project_rel": claim.project_rel,
        "profile_rel": claim.profile_rel,
    },
    "fold": fold,
    "watchdogs": {
        "source": [[d.name, d.instructions] for d in from_source],
        "render": [[d.name, d.instructions] for d in from_render],
        "copied": copied,
        "render_files": files_under(render),
    },
    "unpaired": [list(pair) for pair in unpaired],
    "image": sorted(p.name for p in image.iterdir()),
    "image_files": files_under(image),
}))
"""

#: The same probe without the rebinding, for the values that must not move.
_BASELINE = r"""
import json

from osprey.cli import profile_conventions
from osprey.cli.build_cmd import _NON_SOURCE_ROOT_ENTRIES
from osprey.deployment import compose_generator
from osprey.facility import settings as facility_settings
from osprey.services.watchdog import config as watchdog_config

print(json.dumps({
    "render_constants": {
        "WATCHDOGS_BLOCK": watchdog_config.WATCHDOGS_BLOCK,
        "VOCABULARY_SOURCE": facility_settings.VOCABULARY_SOURCE,
        "settings._RENDER_FACILITY_DIR": facility_settings._RENDER_FACILITY_DIR,
        "profile_conventions._RENDER_FACILITY_DIR": profile_conventions._RENDER_FACILITY_DIR,
        "BUILD_OUTPUT_DIR": profile_conventions.BUILD_OUTPUT_DIR,
        "compose SERVICES_DIR": compose_generator.SERVICES_DIR,
    },
    "CONVENTION_DIRS": [
        [c.key, c.source, c.destination] for c in profile_conventions.CONVENTION_DIRS
    ],
    "SELECTED_CONVENTION_SOURCES": profile_conventions.SELECTED_CONVENTION_SOURCES,
    "_NON_SOURCE_ROOT_ENTRIES": sorted(_NON_SOURCE_ROOT_ENTRIES),
    "KNOWN_ROOT_ENTRIES": sorted(profile_conventions.KNOWN_ROOT_ENTRIES),
    "PROFILE_SOURCE_ENTRIES": list(
        __import__(
            "osprey.mcp_server.python_executor.executor", fromlist=["x"]
        ).PROFILE_SOURCE_ENTRIES
    ),
}))
"""


def _write(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _run(*args: str) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, "-c", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """What the consumers computed and did under the moved layout."""
    base = tmp_path_factory.mktemp("root-substitution")
    moved = _write(base / "moved", _MOVED_ROOT)
    bad_rule = _write(base / "bad-rule", _BAD_RULE_ROOT)
    old = _write(base / "old", _OLD_ROOT)
    scratch = [base / name for name in ("project", "image", "render")]
    for directory in scratch:
        directory.mkdir()
    return _run(_PROBE, json.dumps(_MOVED), *map(str, (moved, bad_rule, old, *scratch)))


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    """The same tables in a fresh interpreter whose registry is not rebound."""
    return _run(_BASELINE)


def test_the_registry_is_rebound_before_any_consumer_is_imported(probe: dict[str, Any]) -> None:
    assert [
        name
        for name in probe["consumers_before"]
        if name.startswith(
            ("osprey.cli", "osprey.mcp_server", "osprey.services", "osprey.facility")
        )
    ] == []


# --- the known-root set -----------------------------------------------------------------


def test_the_known_root_set_gains_the_first_segments_and_loses_the_moved_names(
    probe: dict[str, Any], baseline: dict[str, Any]
) -> None:
    known = set(probe["tables"]["KNOWN_ROOT_ENTRIES"])
    before = set(baseline["KNOWN_ROOT_ENTRIES"])
    assert known - before == _NEW_TOP_LEVEL - before
    assert _NEW_TOP_LEVEL <= known
    assert before - known == _OLD_NAMES - _DECLARED_OLD


def test_the_source_zone_table_names_the_first_segment_of_the_moved_scripts(
    probe: dict[str, Any],
) -> None:
    entries = set(probe["tables"]["_SOURCE_ZONE_ENTRIES"])
    assert "ops" in entries
    assert {"scripts", "watchdogs", "ops/scripts", "agent/watchdogs"}.isdisjoint(entries)


def test_a_root_laid_out_that_way_raises_no_unknown_entry(probe: dict[str, Any]) -> None:
    assert probe["unknown_moved"] == []


def test_the_old_names_are_unknown_entries_in_the_moved_layout(probe: dict[str, Any]) -> None:
    assert sorted(probe["unknown_old"]) == sorted(_OLD_NAMES)


# --- the write boundary -----------------------------------------------------------------


def test_the_write_boundary_protects_the_moved_paths_and_the_content_rows(
    probe: dict[str, Any],
) -> None:
    protected = set(probe["protected"])
    assert _NEW_TOP_LEVEL <= protected
    assert {"facility", "web", "data"} <= protected
    assert _DECLARED_OLD <= protected
    assert (_OLD_NAMES - _DECLARED_OLD).isdisjoint(protected)


def test_the_protected_source_table_lists_the_moved_paths(probe: dict[str, Any]) -> None:
    entries = probe["tables"]["PROFILE_SOURCE_ENTRIES"]
    assert _NEW_TOP_LEVEL <= set(entries)
    assert _DECLARED_OLD <= set(entries)
    assert (_OLD_NAMES - _DECLARED_OLD).isdisjoint(entries)


# --- the convention table: location moves, identity and render do not -------------------


def test_each_convention_keeps_its_key_and_destination_and_takes_its_new_source(
    probe: dict[str, Any], baseline: dict[str, Any]
) -> None:
    expected = [
        [key, _MOVED.get(key, source), destination]
        for key, source, destination in baseline["CONVENTION_DIRS"]
    ]
    assert probe["tables"]["CONVENTION_DIRS"] == expected
    assert probe["tables"]["CONVENTION_SOURCES"] == [key for key, _, _ in expected]
    assert probe["convention_source"] == {
        "rules": "agent/rules",
        "mcp_servers": "agent/mcp_servers",
        "web-terminal-context": "agent/context",
    }


def test_list_fields_still_select_by_key(probe: dict[str, Any], baseline: dict[str, Any]) -> None:
    assert probe["tables"]["SELECTED_CONVENTION_SOURCES"] == baseline["SELECTED_CONVENTION_SOURCES"]
    assert probe["tables"]["PER_USER_CONTEXT_DIRNAME"] == "web-terminal-context"


def test_a_rule_copy_lands_at_the_same_destination_under_the_same_name(
    probe: dict[str, Any],
) -> None:
    rule = next(c for c in probe["planned"] if c["source"] == "agent/rules/x.md")
    assert rule == {
        "category": "rules",
        "source": "agent/rules/x.md",
        "destination": ".claude/rules/x.md",
        "artifact": "rules/x",
        "owned": "rules/x",
    }


def test_a_profile_list_entry_spelled_with_the_key_selects_the_rule(
    probe: dict[str, Any],
) -> None:
    assert probe["selection"]["split"] == [["safety"], ["rules/x"]]
    assert "rules/x" in probe["applied"]["owned"]
    assert ".claude/rules/x.md" in probe["applied"]["project"]
    assert ".claude/rules/y.md" not in probe["applied"]["project"]


def test_the_claim_target_keeps_its_name_and_names_the_moved_profile_path(
    probe: dict[str, Any],
) -> None:
    assert probe["claim"] == {
        "name": "rules/x",
        "project_rel": ".claude/rules/x.md",
        "profile_rel": "agent/rules/x.md",
    }


def test_a_validation_error_names_the_wrong_file_at_its_moved_path(
    probe: dict[str, Any],
) -> None:
    error = probe["validation_error"]
    assert error is not None
    assert "agent/rules/notes.txt is not a markdown file" in error
    assert "agent/rules/" in error
    assert " rules/notes.txt" not in error


# --- per-user context and MCP-server packages -------------------------------------------


def test_the_per_user_context_is_read_and_seeded_at_the_moved_path(
    probe: dict[str, Any],
) -> None:
    context = next(c for c in probe["planned"] if c["category"] == "web-terminal-context")
    assert context["source"] == "agent/context/alice"
    assert context["artifact"] == "web-terminal-context/alice"
    assert "docker/web-terminal-context/alice/CLAUDE.md" in probe["applied"]["project"]
    assert probe["applied"]["seeded"] == ["bob"]
    assert "agent/context/bob/.gitkeep" in probe["applied"]["profile"]
    assert not any(path.startswith("web-terminal-context/") for path in probe["applied"]["profile"])


def test_mcp_server_packages_are_read_and_seeded_at_the_moved_path(
    probe: dict[str, Any],
) -> None:
    server = next(c for c in probe["planned"] if c["category"] == "mcp_servers")
    assert server["source"] == "agent/mcp_servers/beamline"
    assert server["artifact"] == "mcp_servers/beamline"
    assert probe["unpaired"] == [["unpaired", "absent"]]
    assert probe["tables"]["WRITE_ONCE_DIRS"] == {"agent/mcp_servers": "agent/mcp_servers"}


def test_the_init_keys_stay_keys_and_the_init_paths_follow(probe: dict[str, Any]) -> None:
    tables = probe["tables"]
    assert tables["MCP_SERVER_SOURCE_DIR"] == "mcp_servers"
    assert tables["FACILITY_RULE_DIR"] == "rules"
    assert set(tables["WRITE_ONCE_DIRS"]) == {"agent/mcp_servers"}
    assert tables["WRITE_ONCE_FILES"] == [
        ".gitignore",
        ".env.shared",
        "README.md",
        "ops/ci-extra.yml",
    ]


# --- watchdogs: source moves, render stays -----------------------------------------------


def test_the_profile_watchdogs_are_read_from_the_moved_source(probe: dict[str, Any]) -> None:
    assert probe["watchdogs"]["source"] == [["sentinel", _SOURCE_INSTRUCTIONS]]


def test_the_rendered_watchdogs_are_still_read_from_the_build(probe: dict[str, Any]) -> None:
    assert probe["watchdogs"]["render"] == [["sentinel", _RENDER_INSTRUCTIONS]]


def test_a_build_copies_the_moved_watchdog_into_the_unchanged_render_directory(
    probe: dict[str, Any],
) -> None:
    assert probe["watchdogs"]["copied"] == ["sentinel"]
    assert probe["watchdogs"]["render_files"] == ["watchdogs/sentinel/instructions.md"]


# --- the fold ---------------------------------------------------------------------------


def test_fold_labels_keep_the_key_and_input_keys_show_the_moved_paths(
    probe: dict[str, Any],
) -> None:
    fold = {label: [key, path] for key, label, path in probe["fold"]}
    assert fold["convention:rules"] == ["agent/rules/", "agent/rules"]
    assert fold["convention:mcp_servers"] == ["agent/mcp_servers/", "agent/mcp_servers"]
    assert fold["convention:web-terminal-context"] == ["agent/context/", "agent/context"]
    assert fold["watchdog:sentinel"] == ["agent/watchdogs/sentinel/", "agent/watchdogs/sentinel"]


def test_conventions_fold_in_key_order(probe: dict[str, Any]) -> None:
    labels = [label for _, label, _ in probe["fold"] if (label or "").startswith("convention:")]
    assert labels == sorted(labels)


# --- the scripts children ---------------------------------------------------------------


def test_the_scripts_children_display_under_the_moved_directory(probe: dict[str, Any]) -> None:
    tables = probe["tables"]
    assert tables["VERIFY_PATH"] == "ops/scripts/verify.sh"
    assert tables["BOOT_HOOK_PATH"] == "ops/scripts/osprey-boot-hook.sh"
    assert tables["REPO_VERIFY_PATH"] == ["ops", "scripts", "verify.sh"]
    assert tables["CI_EMITTED_PATHS"] == [".gitlab-ci.yml", "ops/scripts/verify.sh"]


# --- the image --------------------------------------------------------------------------


def test_the_image_carries_the_moved_sources_and_still_leaves_out_the_excluded_entries(
    probe: dict[str, Any], baseline: dict[str, Any]
) -> None:
    assert probe["tables"]["_NON_SOURCE_ROOT_ENTRIES"] == baseline["_NON_SOURCE_ROOT_ENTRIES"]
    assert probe["image"] == ["agent", "facility", "ops", "profile.yml", "web"]
    assert {
        "agent/rules/x.md",
        "agent/watchdogs/sentinel/instructions.md",
        "agent/mcp_servers/beamline/__init__.py",
        "agent/context/alice/CLAUDE.md",
        "ops/scripts/verify.sh",
    } <= set(probe["image_files"])


# --- names that do not follow -----------------------------------------------------------


def test_the_config_key_and_the_render_constants_are_unchanged(
    probe: dict[str, Any], baseline: dict[str, Any]
) -> None:
    assert probe["render_constants"] == baseline["render_constants"]
    assert probe["render_constants"]["WATCHDOGS_BLOCK"] == "watchdogs"
