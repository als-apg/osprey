"""Nested deployment-root entries work in init and the deploy scaffolding.

A fresh interpreter rebinds the deployment-root registry to a layout where the
source-zone entries ``watchdogs`` and ``web-terminal-context`` sit in an
``agent/`` group and ``osprey.service``, ``ci-extra.yml``, ``services`` and
``scripts`` sit in an ``ops/`` group, before any consumer is imported. It then
drives the consumers on deployment roots in ``tmp_path`` laid out that way and
reports what they did. Nothing in the repository moves.

Expected: ``init --force`` holds nested entries aside and puts them back
(after a failure, after a kill) leaving neither a stash nor an empty group
behind; write-once files land inside their group; the systemd unit is written
at its row path and installed and started under its file name; the CI pipeline
includes ``ci-extra.yml`` and builds service images from where the rows put
them; and the health check prose names the directory the check sits in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

from osprey.cli.repo_resolver import HELD_SOURCE_ZONE_DIRNAME

#: Each nested deployment-root entry, by key, and where it sits in the nested layout.
_NESTED: dict[str, str] = {
    "watchdogs": "agent/watchdogs",
    "web-terminal-context": "agent/context",
    "osprey.service": "ops/osprey.service",
    "ci-extra.yml": "ops/ci-extra.yml",
    "services": "ops/services",
    "scripts": "ops/scripts",
}

#: Runs in a fresh interpreter: rebinds the registry, imports the consumers, and
#: prints one JSON object describing what they did under ``sys.argv[2]``.
_PROBE = r"""
import asyncio
import json
import sys
import types
from pathlib import Path

from osprey import repo_layout

repo_layout.ROOT_ENTRIES = repo_layout.with_paths(json.loads(sys.argv[1]))

from click.testing import CliRunner

from osprey.cli import deploy_scaffold, deploy_scaffold_templates, init_cmd, scaffold_cmd
from osprey.cli.repo_resolver import HELD_SOURCE_ZONE_DIRNAME
from osprey.health.core import systemd_unit

base = Path(sys.argv[2])
report = {}


def tree(root):
    if not root.exists():
        return None
    return sorted(
        path.relative_to(root).as_posix() + ("/" if path.is_dir() else "")
        for path in root.rglob("*")
    )


def zone(name):
    root = base / name
    for relative, text in {
        "profile.yml": "name: nested\n",
        "agent/watchdogs/sentinel/instructions.md": "held\n",
        "agent/context/alice/CLAUDE.md": "# alice\n",
        "agent/notes.md": "not an entry\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


# --- init --force: hold aside, then a replacement that succeeds ---------------
target = zone("replaced")
with init_cmd._replacing_source_zone(target, active=True):
    report["held_during"] = tree(target / HELD_SOURCE_ZONE_DIRNAME)
    report["repo_during"] = tree(target)
    fresh = target / "agent" / "watchdogs" / "fresh" / "instructions.md"
    fresh.parent.mkdir(parents=True)
    fresh.write_text("new\n", encoding="utf-8")
report["after_success"] = tree(target)

# --- the same with a replacement that brings back no nested entry -------------
target = zone("emptied")
(target / "agent" / "notes.md").unlink()
with init_cmd._replacing_source_zone(target, active=True):
    pass
report["after_empty_success"] = tree(target)

# --- init --force: a replacement that fails half-way --------------------------
target = zone("failed")
try:
    with init_cmd._replacing_source_zone(target, active=True):
        half = target / "agent" / "watchdogs" / "half" / "instructions.md"
        half.parent.mkdir(parents=True)
        half.write_text("half\n", encoding="utf-8")
        raise RuntimeError("materialization failed")
except RuntimeError:
    pass
report["after_failure"] = tree(target)

# --- a failure whose group directory the materializer removed -----------------
target = zone("failed-no-group")
(target / "agent" / "notes.md").unlink()
try:
    with init_cmd._replacing_source_zone(target, active=True):
        (target / "agent").rmdir()
        raise RuntimeError("materialization failed")
except RuntimeError:
    pass
report["after_failure_no_group"] = tree(target)

# --- a killed run: the stash is what it left, the repo has no group -----------
target = base / "killed"
held = target / HELD_SOURCE_ZONE_DIRNAME
for relative, text in {
    "profile.yml": "name: nested\n",
    "agent/watchdogs/sentinel/instructions.md": "held\n",
    "agent/context/alice/CLAUDE.md": "# alice\n",
}.items():
    path = held / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
init_cmd._reinstate_held_source_zone(target)
report["after_reinstate"] = tree(target)

# --- write-once files land inside their group ---------------------------------
target = base / "write-once"
target.mkdir()
report["write_once_keys"] = sorted(init_cmd.WRITE_ONCE_FILES)
report["write_if_absent"] = init_cmd._write_if_absent(target / "ops" / "ci-extra.yml", "x\n")
report["write_if_absent_again"] = init_cmd._write_if_absent(
    target / "ops" / "ci-extra.yml", "y\n"
)
report["write_once_tree"] = tree(target)

# --- the systemd unit: written at its path, known by its name -----------------
deploy_scaffold_templates.resolve_shell_command = lambda command: "/opt/osprey/bin/osprey"
repo = base / "systemd"
repo.mkdir()
(repo / "profile.yml").write_text("name: nested\n", encoding="utf-8")
result = CliRunner().invoke(scaffold_cmd.scaffold, ["systemd", "--repo", str(repo)])
report["systemd_exit"] = result.exit_code
report["systemd_output"] = result.output
report["systemd_tree"] = tree(repo)
report["unit_name"] = deploy_scaffold_templates.SYSTEMD_UNIT_NAME
report["unit_path"] = deploy_scaffold_templates.SYSTEMD_UNIT_PATH
report["output_name"] = deploy_scaffold.SYSTEMD_OUTPUT_NAME
report["unit_text"] = (repo / "ops" / "osprey.service").read_text(encoding="utf-8")
report["hook_text"] = (repo / "ops" / "scripts" / "osprey-boot-hook.sh").read_text(
    encoding="utf-8"
)

# --- the health row: finds the unit at its path, asks for it by name ----------
systemd_unit.shutil = types.SimpleNamespace(which=lambda name: "/usr/bin/systemctl")
empty = base / "no-unit"
empty.mkdir()
report["health_absent"] = asyncio.run(systemd_unit._check_systemd_unit(empty))[0].message
row = asyncio.run(systemd_unit._check_systemd_unit(repo))[0]
report["health_present"] = [row.status.value, row.message, row.details]

# --- the CI pipeline and the health check ------------------------------------
ci = deploy_scaffold_templates.CIContext(
    facility_name="nested",
    osprey_version="0.0.0",
    requirement="osprey-framework",
    registry_url="registry.example.org/nested",
    registry_host="registry.example.org",
    registry_token_var="REGISTRY_TOKEN",
    deploy_host="host.example.org",
    deploy_user="osprey",
    deploy_path="/srv/nested",
    service_images=["beamline"],
)
report["ci_text"] = deploy_scaffold_templates.render("gitlab-ci.yml.j2", ci)
services_root = base / "services"
dockerfile = services_root / "ops" / "services" / "beamline" / "Dockerfile"
dockerfile.parent.mkdir(parents=True)
dockerfile.write_text("FROM scratch\n", encoding="utf-8")
report["service_images"] = deploy_scaffold_templates.service_image_names(
    {"services": {"beamline": {}}}, services_root
)
report["verify_text"] = deploy_scaffold_templates.render(
    "verify.sh.j2",
    deploy_scaffold_templates.build_verify_context({"name": "nested"}, "0.0.0"),
)

print(json.dumps(report))
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """What the consumers did under the nested layout."""
    base = tmp_path_factory.mktemp("nested-root-entries")
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(_NESTED), str(base)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        env={**os.environ, "HOME": str(base / "home"), "XDG_CONFIG_HOME": ""},
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


# --- init --force -----------------------------------------------------------------


def test_hold_aside_moves_nested_entries_into_their_group_in_the_stash(
    probe: dict[str, Any],
) -> None:
    held = probe["held_during"]
    assert "agent/watchdogs/sentinel/instructions.md" in held
    assert "agent/context/alice/CLAUDE.md" in held
    assert "agent/notes.md" not in held
    # The group itself stays in the repo while it holds something else.
    assert "agent/notes.md" in probe["repo_during"]
    assert not any(path.startswith("agent/watchdogs") for path in probe["repo_during"])


def test_a_successful_replacement_leaves_no_stash(probe: dict[str, Any]) -> None:
    after = probe["after_success"]
    assert "agent/watchdogs/fresh/instructions.md" in after
    assert "agent/watchdogs/sentinel/instructions.md" not in after
    assert "agent/notes.md" in after
    assert not any(path.startswith(HELD_SOURCE_ZONE_DIRNAME) for path in after)


def test_a_replacement_that_brings_back_no_nested_entry_prunes_the_group(
    probe: dict[str, Any],
) -> None:
    # profile.yml is part of the source zone too, so nothing at all is left.
    assert probe["after_empty_success"] == []


def test_a_failed_replacement_puts_nested_entries_back_and_leaves_no_stash(
    probe: dict[str, Any],
) -> None:
    assert probe["after_failure"] == [
        "agent/",
        "agent/context/",
        "agent/context/alice/",
        "agent/context/alice/CLAUDE.md",
        "agent/notes.md",
        "agent/watchdogs/",
        "agent/watchdogs/sentinel/",
        "agent/watchdogs/sentinel/instructions.md",
        "profile.yml",
    ]


def test_a_failed_replacement_recreates_a_group_the_materializer_removed(
    probe: dict[str, Any],
) -> None:
    assert probe["after_failure_no_group"] == [
        "agent/",
        "agent/context/",
        "agent/context/alice/",
        "agent/context/alice/CLAUDE.md",
        "agent/watchdogs/",
        "agent/watchdogs/sentinel/",
        "agent/watchdogs/sentinel/instructions.md",
        "profile.yml",
    ]


def test_reinstating_a_killed_run_creates_groups_and_prunes_the_stash(
    probe: dict[str, Any],
) -> None:
    # No group is left behind in the stash to be reported as unrecognized,
    # so the stash goes entirely.
    assert probe["after_reinstate"] == [
        "agent/",
        "agent/context/",
        "agent/context/alice/",
        "agent/context/alice/CLAUDE.md",
        "agent/watchdogs/",
        "agent/watchdogs/sentinel/",
        "agent/watchdogs/sentinel/instructions.md",
        "profile.yml",
    ]


def test_write_once_files_are_written_inside_their_group(probe: dict[str, Any]) -> None:
    assert "ops/ci-extra.yml" in probe["write_once_keys"]
    assert probe["write_if_absent"] is True
    assert probe["write_if_absent_again"] is False
    assert probe["write_once_tree"] == ["ops/", "ops/ci-extra.yml"]


# --- systemd ----------------------------------------------------------------------


def test_the_unit_is_written_at_its_path_and_named_by_its_file_name(
    probe: dict[str, Any],
) -> None:
    assert probe["unit_path"] == "ops/osprey.service"
    assert probe["output_name"] == "ops/osprey.service"
    assert probe["unit_name"] == "osprey.service"
    assert "ops/osprey.service" in probe["systemd_tree"]
    assert "ops/scripts/osprey-boot-hook.sh" in probe["systemd_tree"]
    assert "osprey.service" not in probe["systemd_tree"]


def test_the_unit_header_copies_the_path_and_enables_the_name(probe: dict[str, Any]) -> None:
    text = probe["unit_text"]
    assert "#   cp ops/osprey.service ~/.config/systemd/user/" in text
    assert "#   systemctl --user enable --now osprey.service\n" in text
    assert "ops/osprey.service" not in text.replace("cp ops/osprey.service", "")


def test_the_boot_hook_starts_the_unit_by_name(probe: dict[str, Any]) -> None:
    text = probe["hook_text"]
    assert "systemctl --user start osprey.service" in text
    assert "ops/osprey.service" not in text


def test_scaffold_systemd_prints_the_path_to_copy_and_the_name_to_enable(
    probe: dict[str, Any],
) -> None:
    assert probe["systemd_exit"] == 0, probe["systemd_output"]
    output = probe["systemd_output"]
    assert "ops/osprey.service (created)" in output
    assert "ops/osprey.service ~/.config/systemd/user/" in output
    assert "systemctl --user enable --now osprey.service\n" in output


def test_the_health_row_finds_the_unit_at_its_path(probe: dict[str, Any]) -> None:
    assert "ops/osprey.service" in probe["health_absent"]
    status, message, details = probe["health_present"]
    # Found in the repo, so the row reaches the installed check, which asks for
    # the unit by the name the user manager knows it by.
    assert status == "warning"
    assert message == "osprey.service is scaffolded but not installed"
    assert "/ops/osprey.service to " in details
    assert details.split(" to ", 1)[1].split(",")[0].endswith("/systemd/user/osprey.service")
    assert "enable --now osprey.service`" in details


# --- CI ---------------------------------------------------------------------------


def test_the_pipeline_includes_ci_extra_where_the_row_puts_it(probe: dict[str, Any]) -> None:
    text = probe["ci_text"]
    assert "  - local: ops/ci-extra.yml\n" in text
    assert "          - ops/ci-extra.yml\n" in text
    assert "jobs go in ops/ci-extra.yml," in text
    assert " ci-extra.yml" not in text.replace("ops/ci-extra.yml", "")


def test_the_pipeline_builds_service_images_from_the_services_row(
    probe: dict[str, Any],
) -> None:
    assert probe["service_images"] == ["beamline"]
    text = probe["ci_text"]
    assert '"ops/services/$SERVICE"' in text
    assert '"services/$SERVICE"' not in text


def test_the_health_check_names_the_directory_it_is_emitted_into(
    probe: dict[str, Any],
) -> None:
    text = probe["verify_text"]
    assert "into the repo's ops/scripts/ directory" in text
    assert "./ops/scripts/verify.sh" in text
    assert " scripts/" not in text.replace("ops/scripts/", "")
