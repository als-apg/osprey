"""Tests for the profile → project convention mapping table.

Covers the canonical mapping itself, the ``project/`` mirror's reserved-path
rejection (each error must name the channel that owns the path), the up-front
source validation, and the unknown-root-entry typo warning.

The ``zone``-named group at the end pins the three-zone repo layout: the repo
root is the profile root, so the layout's own directories must be recognized
rather than warned about, and the warning's remedy must not send an operator
looking for a nested ``profile/`` directory that does not exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.cli.profile_conventions import (
    BUILD_OUTPUT_DIR,
    CONTEXT_BASELINE_FILENAME,
    CONVENTION_DIRS,
    CONVENTION_SOURCES,
    KNOWN_ROOT_ENTRIES,
    PER_USER_CONTEXT_DIRNAME,
    PROJECT_MIRROR_DIR,
    RESERVED_PROJECT_PATHS,
    STATE_DIR,
    ConventionDir,
    EntryShape,
    convention_for,
    convention_slot_for,
    destination_for,
    partition_context_users,
    plan_convention_copies,
    reserved_path_channel,
    unknown_root_entries,
    validate_convention_sources,
    validate_project_mirror,
    warn_unknown_root_entries,
)
from osprey.errors import BuildProfileError


def _write(path: Path, content: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    """A minimal, valid profile root."""
    root = tmp_path / "my-profile"
    root.mkdir()
    _write(root / "profile.yml", "name: my-profile\n")
    return root


# ── Mapping table ────────────────────────────────────────────────────


def test_mapping_table_is_exactly_the_specified_set():
    assert {c.source: c.destination for c in CONVENTION_DIRS} == {
        "rules": ".claude/rules",
        "skills": ".claude/skills",
        "agents": ".claude/agents",
        "commands": ".claude/commands",
        "output-styles": ".claude/output-styles",
        "hooks": ".claude/hooks",
        "web-terminal-context": "docker/web-terminal-context",
        "mcp_servers": "_mcp_servers",
        "services": "services",
        "project": "",
    }


def test_only_web_terminal_context_is_roster_derived():
    assert [c.source for c in CONVENTION_DIRS if c.per_user] == ["web-terminal-context"]


def test_skills_services_and_servers_copy_as_whole_directories():
    shapes = {c.source: c.shape for c in CONVENTION_DIRS}
    assert shapes["skills"] is EntryShape.DIRECTORY
    assert shapes["services"] is EntryShape.DIRECTORY
    assert shapes["mcp_servers"] is EntryShape.DIRECTORY
    assert shapes["web-terminal-context"] is EntryShape.DIRECTORY
    assert shapes["rules"] is EntryShape.MARKDOWN
    assert shapes["project"] is EntryShape.MIRROR


def test_hooks_are_file_shaped_not_markdown():
    """Hooks are executable scripts, so the markdown suffix rule cannot apply."""
    convention = convention_for("hooks")
    assert convention is not None
    assert convention.shape is EntryShape.FILE
    assert convention.destination == ".claude/hooks"


def test_convention_for_resolves_and_misses():
    resolved = convention_for("rules")
    assert isinstance(resolved, ConventionDir)
    assert resolved.destination == ".claude/rules"
    assert convention_for("rule") is None


@pytest.mark.parametrize(
    ("source_rel", "expected"),
    [
        ("rules/safety.md", ".claude/rules/safety.md"),
        ("agents/orbit-writer.md", ".claude/agents/orbit-writer.md"),
        ("commands/ops/restart.md", ".claude/commands/ops/restart.md"),
        ("output-styles/terse.md", ".claude/output-styles/terse.md"),
        ("hooks/osprey_limits.py", ".claude/hooks/osprey_limits.py"),
        ("skills/orbit-check", ".claude/skills/orbit-check"),
        ("mcp_servers/matlab", "_mcp_servers/matlab"),
        ("services/archiver", "services/archiver"),
        ("web-terminal-context/alice", "docker/web-terminal-context/alice"),
        ("web-terminal-context/alice/context.md", "docker/web-terminal-context/alice/context.md"),
        ("project/docs/runbook.md", "docs/runbook.md"),
        ("project/Makefile", "Makefile"),
    ],
)
def test_destination_for_maps_each_category(source_rel: str, expected: str):
    assert destination_for(source_rel) == expected


def test_destination_for_mirror_root_is_the_project_root():
    assert destination_for("project") == ""


def test_destination_for_rejects_a_non_convention_root():
    with pytest.raises(BuildProfileError) as excinfo:
        destination_for("overlays/safety.md")
    message = str(excinfo.value)
    assert "'overlays'" in message
    assert "rules/" in message  # names the real convention dirs


# ── Reserved paths ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reserved", "channel_hint"),
    [
        ("config.yml", "`config:`"),
        (".osprey-manifest.json", "build"),
        (".claude/settings.json", "claude_code.permissions"),
        (".mcp.json", "`mcp_servers:`"),
        (".env", "`env:`"),
        (".env.example", "`env:`"),
        ("CLAUDE.md", "claude_md_template"),
        ("data/simulation/channel_manifest.json", "`data/`"),
        ("data/simulation/channel_limits.json", "`data/`"),
        ("docker/web-terminal-context/base.md", "`web-terminal-context/base.md`"),
    ],
)
def test_reserved_path_names_its_channel(reserved: str, channel_hint: str):
    channel = reserved_path_channel(reserved)
    assert channel is not None
    assert channel_hint in channel


def test_every_reserved_entry_is_reported():
    for entry in RESERVED_PROJECT_PATHS:
        assert reserved_path_channel(entry.path) == entry.channel


def test_framework_rendered_claude_output_is_reserved():
    """A framework render with no convention channel of its own is still reserved."""
    channel = reserved_path_channel(".claude/statusline.py")
    assert channel is not None and "framework render" in channel


def test_convention_owned_claude_subtree_is_reserved_and_names_the_dir():
    channel = reserved_path_channel(".claude/rules/facility.md")
    assert channel is not None and "`rules/`" in channel
    channel = reserved_path_channel(".claude/skills/my-skill/SKILL.md")
    assert channel is not None and "`skills/`" in channel


def test_exact_reservation_beats_the_convention_prefix():
    """`.claude/hooks/hook_config.json` is inside the hooks/ subtree but not the channel's.

    The ordering inside ``reserved_path_channel`` is what decides this — the
    exact-match lookup runs before the convention-prefix loop. If that ever
    flips, the hooks/ channel would claim a file the framework generates from
    the resolved config, so the ordering is asserted rather than assumed.
    """
    channel = reserved_path_channel(".claude/hooks/hook_config.json")
    assert channel is not None
    assert "`hooks/`" not in channel, "the channel must not claim a generated file"
    assert "generated from" in channel
    # The message names the keys that DO control it, and what depends on it.
    assert "mcp_servers:" in channel
    assert "control_system.write_tools" in channel
    assert "write-safety" in channel


@pytest.mark.parametrize(
    "dest",
    [
        ".claude/hooks/osprey_writes_check.py",  # a framework-rendered hook
        ".claude/hooks/facility_guard.py",  # one the framework never renders
    ],
)
def test_hooks_destination_steers_to_the_hooks_channel(dest: str):
    """`.claude/hooks/` now has a channel, so the mirror refusal must name it.

    Before the ``hooks/`` convention existed, a framework-rendered hook fell
    through to the generic "carry it as a `config:` key" message and a custom
    one was not reserved at all. Both now route to the same place, which is the
    only writer of that subtree.
    """
    channel = reserved_path_channel(dest)
    assert channel is not None and "`hooks/`" in channel


@pytest.mark.parametrize(
    "allowed",
    ["docs/runbook.md", "Makefile", ".gitignore", "scripts/deploy.sh", "data/facility.json"],
)
def test_unreserved_paths_are_writable_by_the_mirror(allowed: str):
    assert reserved_path_channel(allowed) is None


def test_validate_project_mirror_accepts_a_clean_tree(profile_dir: Path):
    mirror = profile_dir / "project"
    _write(mirror / "docs" / "runbook.md")
    _write(mirror / ".gitignore", "_agent_data/\n")
    validate_project_mirror(mirror)


def test_validate_project_mirror_reports_every_violation(profile_dir: Path):
    mirror = profile_dir / "project"
    _write(mirror / "config.yml")
    _write(mirror / ".claude" / "settings.json")
    _write(mirror / "docs" / "runbook.md")

    with pytest.raises(BuildProfileError) as excinfo:
        validate_project_mirror(mirror)

    message = str(excinfo.value)
    assert "project/config.yml" in message
    assert "project/.claude/settings.json" in message
    assert "`config:` block" in message
    assert "runbook.md" not in message


@pytest.mark.parametrize(
    ("mirrored", "slot"),
    [
        (".claude/hooks/facility_guard.py", "hooks/facility_guard.py"),
        (".claude/hooks/osprey_writes_check.py", "hooks/osprey_writes_check.py"),
        (".claude/commands/ops/restart.md", "commands/ops/restart.md"),
    ],
)
def test_mirror_refusal_states_the_exact_move(profile_dir: Path, mirrored: str, slot: str):
    """The refusal is a migration instruction: both paths, not just a channel name.

    A profile that carried a hook through `project/.claude/hooks/` before the
    `hooks/` channel existed hits this on its next build, and this line is the
    only guidance it gets — so it has to name the file and where it goes.
    """
    mirror = profile_dir / "project"
    _write(mirror / mirrored)

    with pytest.raises(BuildProfileError) as excinfo:
        validate_project_mirror(mirror)

    assert f"Move it: project/{mirrored} → {slot}" in str(excinfo.value)


def test_mirror_refusal_omits_a_move_when_no_directory_owns_the_path(profile_dir: Path):
    """`.mcp.json` is generated from a profile KEY — there is no slot to move it to."""
    mirror = profile_dir / "project"
    _write(mirror / ".mcp.json")

    with pytest.raises(BuildProfileError) as excinfo:
        validate_project_mirror(mirror)

    message = str(excinfo.value)
    assert "`mcp_servers:`" in message
    assert "Move it:" not in message


def test_convention_slot_for_inverts_destination_for():
    """The move the refusal prints is the mapping table read backwards."""
    for source_rel in ("hooks/facility_guard.py", "rules/safety.md", "skills/orbit-check"):
        assert convention_slot_for(destination_for(source_rel)) == source_rel
    # The mirror lands on the project root, so it can never be the answer.
    assert convention_slot_for("docs/runbook.md") is None
    assert convention_slot_for("config.yml") is None


def test_validate_project_mirror_tolerates_a_missing_mirror(profile_dir: Path):
    validate_project_mirror(profile_dir / "project")


# ── Source validation ────────────────────────────────────────────────


def test_validate_convention_sources_accepts_a_well_formed_profile(profile_dir: Path):
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "agents" / "orbit-writer.md")
    _write(profile_dir / "commands" / "ops" / "restart.md")
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md")
    _write(profile_dir / "skills" / "orbit-check" / "scripts" / "check.py")
    _write(profile_dir / "services" / "archiver" / "docker-compose.yml")
    _write(profile_dir / "mcp_servers" / "matlab" / "server.py")
    _write(profile_dir / "web-terminal-context" / "alice" / "context.md")
    _write(profile_dir / "hooks" / "facility_guard.py")
    _write(profile_dir / "project" / "docs" / "runbook.md")
    validate_convention_sources(profile_dir)


def test_hooks_accept_any_extension_unlike_the_markdown_shapes(profile_dir: Path):
    """The suffix rule is the markdown shape's, not every file shape's.

    The same ``.py`` file that is a legitimate hook is rejected under
    ``rules/``, which is what makes the two shapes distinct rather than a
    stylistic difference.
    """
    _write(profile_dir / "hooks" / "facility_guard.py", "print('guard')\n")
    _write(profile_dir / "hooks" / "notes.md", "# why this hook exists\n")
    validate_convention_sources(profile_dir)

    _write(profile_dir / "rules" / "facility_guard.py")
    with pytest.raises(BuildProfileError, match="rules/facility_guard.py"):
        validate_convention_sources(profile_dir)


def test_validate_convention_sources_accepts_an_empty_profile(profile_dir: Path):
    validate_convention_sources(profile_dir)


def test_convention_directory_that_is_a_file_is_rejected(profile_dir: Path):
    _write(profile_dir / "rules")
    with pytest.raises(BuildProfileError, match="rules is a file"):
        validate_convention_sources(profile_dir)


def test_non_markdown_artifact_is_rejected_with_the_mirror_as_the_way_out(profile_dir: Path):
    _write(profile_dir / "rules" / "safety.txt")
    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)
    message = str(excinfo.value)
    assert "rules/safety.txt" in message
    assert "project/" in message


def test_profile_shipped_hook_config_is_rejected(profile_dir: Path):
    """The write-safety layer's own config may not be carried by the channel.

    ``osprey_writes_check.py`` reads its ``write_tools`` from this file, and a
    profile copy would be registered user-owned and skipped by every later
    regen — so an empty list here disarms the write gate permanently and
    silently. The build refuses instead.
    """
    _write(
        profile_dir / "hooks" / "hook_config.json",
        '{"server_prefixes": [], "approval_prefixes": [], "write_tools": []}\n',
    )

    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)

    message = str(excinfo.value)
    assert "hooks/hook_config.json" in message
    assert "generated, not authored" in message
    # Names the keys that DO control it, so the refusal is actionable.
    assert "mcp_servers:" in message
    assert "control_system.write_tools" in message


def test_a_normal_hook_beside_it_is_still_accepted(profile_dir: Path):
    """The refusal is one generated path, not a retreat from the channel."""
    _write(profile_dir / "hooks" / "facility_guard.py", "print('guard')\n")
    _write(profile_dir / "hooks" / "osprey_writes_check.py", "# facility edit\n")
    validate_convention_sources(profile_dir)


def test_loose_file_in_a_directory_shaped_convention_is_rejected(profile_dir: Path):
    _write(profile_dir / "skills" / "orbit-check.md")
    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)
    message = str(excinfo.value)
    assert "skills/orbit-check.md" in message
    assert "one directory per skill" in message


def test_loose_file_in_the_per_user_convention_states_the_roster_rule(profile_dir: Path):
    """The generic "move it into ``<stem>/``" advice manufactures a phantom user.

    Directory names under the per-user convention are matched against the
    resolved roster, so a directory invented to hold a loose file names nobody:
    the build skips it as departed, and the operator's file stops being read
    while they believe they followed the instructions. The message has to name
    the rule, and must not offer the move that breaks it.
    """
    _write(profile_dir / PER_USER_CONTEXT_DIRNAME / "shift-notes.md")

    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)

    message = str(excinfo.value)
    assert f"{PER_USER_CONTEXT_DIRNAME}/shift-notes.md is a file" in message
    assert "named for a user on the resolved roster" in message
    assert "skipped as a user who has left" in message
    assert f"{PER_USER_CONTEXT_DIRNAME}/<user>/shift-notes.md" in message
    # The advice that would quietly stop the file being read.
    assert f"Move it into {PER_USER_CONTEXT_DIRNAME}/shift-notes/" not in message


def test_base_md_at_the_convention_root_is_accepted_and_planned(profile_dir: Path):
    """``base.md`` is the per-user convention's one loose file: the shared
    baseline every seeded user starts from, overriding the framework's
    fallback at the same destination."""
    _write(profile_dir / PER_USER_CONTEXT_DIRNAME / CONTEXT_BASELINE_FILENAME, "# baseline\n")

    validate_convention_sources(profile_dir)

    copies = plan_convention_copies(profile_dir, context_users=["alice"])
    baselines = [c for c in copies if c.destination == "docker/web-terminal-context/base.md"]
    assert len(baselines) == 1
    assert not baselines[0].is_directory
    assert baselines[0].category == PER_USER_CONTEXT_DIRNAME


def test_base_md_copies_even_without_a_roster(profile_dir: Path):
    """The baseline is roster-independent: a persona render (no roster) still
    carries the profile's text, like every other convention artifact."""
    _write(profile_dir / PER_USER_CONTEXT_DIRNAME / CONTEXT_BASELINE_FILENAME, "# baseline\n")

    copies = plan_convention_copies(profile_dir, context_users=None)
    assert any(c.destination == "docker/web-terminal-context/base.md" for c in copies)


def test_a_base_md_directory_is_not_read_as_the_baseline(profile_dir: Path):
    """Only the FILE is the baseline slot — a directory named ``base.md`` is
    just a user directory naming nobody on the roster, skipped like any
    other departed user."""
    (profile_dir / PER_USER_CONTEXT_DIRNAME / CONTEXT_BASELINE_FILENAME).mkdir(parents=True)

    validate_convention_sources(profile_dir)
    copies = plan_convention_copies(profile_dir, context_users=["alice"])
    assert not any(c.destination == "docker/web-terminal-context/base.md" for c in copies)


def test_mirrored_base_md_is_rejected_naming_the_slot(profile_dir: Path):
    """The mirror may not write the baseline's destination: the slot is its one
    channel, so two writers can never race on build ordering."""
    _write(profile_dir / PROJECT_MIRROR_DIR / "docker" / "web-terminal-context" / "base.md")

    with pytest.raises(BuildProfileError) as excinfo:
        validate_project_mirror(profile_dir / PROJECT_MIRROR_DIR)

    message = str(excinfo.value)
    assert f"`{PER_USER_CONTEXT_DIRNAME}/{CONTEXT_BASELINE_FILENAME}` slot" in message


def test_the_other_directory_conventions_keep_their_move_advice(profile_dir: Path):
    """Only the per-user convention constrains its directory names.

    A skill directory is free-form, so ``skills/<stem>/`` is exactly the right
    move there — the per-user wording must not have spread to the conventions
    it does not describe.
    """
    _write(profile_dir / "skills" / "orbit-check.md")

    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)

    message = str(excinfo.value)
    assert "Move it into skills/orbit-check/" in message
    assert "roster" not in message


def test_hidden_entries_do_not_trip_source_validation(profile_dir: Path):
    _write(profile_dir / "rules" / ".DS_Store")
    _write(profile_dir / "skills" / ".DS_Store")
    validate_convention_sources(profile_dir)


def test_symlink_escaping_the_profile_is_rejected(profile_dir: Path, tmp_path: Path):
    outside = _write(tmp_path / "outside" / "safety.md")
    (profile_dir / "rules").mkdir()
    (profile_dir / "rules" / "safety.md").symlink_to(outside)
    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)
    assert "self-contained" in str(excinfo.value)


def test_symlink_inside_the_profile_is_allowed(profile_dir: Path):
    target = _write(profile_dir / "rules" / "safety.md")
    _write(profile_dir / "agents" / "orbit-writer.md")
    (profile_dir / "agents" / "safety.md").symlink_to(target)
    validate_convention_sources(profile_dir)


def test_source_validation_covers_the_project_mirror(profile_dir: Path):
    _write(profile_dir / "project" / ".mcp.json")
    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)
    assert "`mcp_servers:`" in str(excinfo.value)


def test_source_validation_reports_problems_together(profile_dir: Path):
    _write(profile_dir / "rules" / "safety.txt")
    _write(profile_dir / "skills" / "loose.md")
    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)
    message = str(excinfo.value)
    assert "rules/safety.txt" in message
    assert "skills/loose.md" in message


# ── Copy planning ────────────────────────────────────────────────────


def test_plan_covers_each_category_with_the_right_granularity(profile_dir: Path):
    _write(profile_dir / "rules" / "facility-ops.md")
    _write(profile_dir / "commands" / "ops" / "restart.md")
    _write(profile_dir / "skills" / "orbit-check" / "SKILL.md")
    _write(profile_dir / "skills" / "orbit-check" / "reference.md")
    _write(profile_dir / "services" / "archiver" / "docker-compose.yml")
    _write(profile_dir / "mcp_servers" / "matlab" / "server.py")
    _write(profile_dir / "project" / "docs" / "runbook.md")

    plan = {copy.destination: copy for copy in plan_convention_copies(profile_dir)}

    assert plan[".claude/rules/facility-ops.md"].is_directory is False
    assert plan[".claude/commands/ops/restart.md"].category == "commands"
    # A skill is claimed as a whole directory, not file by file.
    assert plan[".claude/skills/orbit-check"].is_directory is True
    assert ".claude/skills/orbit-check/SKILL.md" not in plan
    assert plan["services/archiver"].is_directory is True
    assert plan["_mcp_servers/matlab"].is_directory is True
    assert plan["docs/runbook.md"].source == profile_dir / "project" / "docs" / "runbook.md"


def test_plan_copies_hooks_file_by_file_keeping_their_suffix(profile_dir: Path):
    _write(profile_dir / "hooks" / "facility_guard.py")
    _write(profile_dir / "hooks" / ".DS_Store")

    plan = {copy.destination: copy for copy in plan_convention_copies(profile_dir)}

    assert plan[".claude/hooks/facility_guard.py"].is_directory is False
    assert plan[".claude/hooks/facility_guard.py"].category == "hooks"
    # File-shaped categories skip dot-prefixed entries, same as the markdown ones.
    assert not any(dest.endswith(".DS_Store") for dest in plan)


def test_plan_mirrors_dot_prefixed_files_but_skips_them_elsewhere(profile_dir: Path):
    _write(profile_dir / "project" / ".gitignore")
    _write(profile_dir / "rules" / ".DS_Store")
    destinations = {copy.destination for copy in plan_convention_copies(profile_dir)}
    assert ".gitignore" in destinations
    assert not any(dest.endswith(".DS_Store") for dest in destinations)


def test_plan_skips_per_user_context_without_a_roster(profile_dir: Path):
    _write(profile_dir / "web-terminal-context" / "alice" / "context.md")
    assert plan_convention_copies(profile_dir) == []
    assert plan_convention_copies(profile_dir, context_users=[]) == []


def test_plan_copies_only_roster_users(profile_dir: Path):
    _write(profile_dir / "web-terminal-context" / "alice" / "context.md")
    _write(profile_dir / "web-terminal-context" / "departed" / "context.md")

    plan = plan_convention_copies(profile_dir, context_users=["alice", "bob"])

    assert [copy.destination for copy in plan] == ["docker/web-terminal-context/alice"]
    assert plan[0].is_directory is True


def test_plan_is_sorted_by_destination(profile_dir: Path):
    _write(profile_dir / "rules" / "b.md")
    _write(profile_dir / "rules" / "a.md")
    _write(profile_dir / "project" / "Makefile")
    destinations = [copy.destination for copy in plan_convention_copies(profile_dir)]
    assert destinations == sorted(destinations)


def test_partition_context_users_splits_matched_missing_and_departed(profile_dir: Path):
    _write(profile_dir / "web-terminal-context" / "alice" / "context.md")
    _write(profile_dir / "web-terminal-context" / "departed" / "context.md")

    matched, missing, departed = partition_context_users(profile_dir, ["alice", "bob"])

    assert matched == ["alice"]
    assert missing == ["bob"]
    assert departed == ["departed"]


def test_partition_context_users_without_the_directory(profile_dir: Path):
    assert partition_context_users(profile_dir, ["alice"]) == ([], ["alice"], [])


# ── Unknown root entries ─────────────────────────────────────────────


def test_unknown_root_entry_flags_a_typo(profile_dir: Path):
    _write(profile_dir / "rule" / "safety.md")
    _write(profile_dir / "rules" / "safety.md")
    assert unknown_root_entries(profile_dir) == ["rule"]


@pytest.mark.parametrize(
    "exempt",
    [".git", ".gitignore", "README.md", "README", "LICENSE", "LICENSE.txt", "docs"],
)
def test_unknown_root_entry_exemptions(profile_dir: Path, exempt: str):
    if exempt in {".git", "docs"}:
        (profile_dir / exempt).mkdir()
    else:
        _write(profile_dir / exempt)
    assert unknown_root_entries(profile_dir) == []


def test_known_root_entries_are_not_flagged(profile_dir: Path):
    _write(profile_dir / "triggers.yml")
    _write(profile_dir / ".env")
    _write(profile_dir / ".env.example")
    (profile_dir / "personas").mkdir()
    (profile_dir / "data").mkdir()
    for source in CONVENTION_SOURCES:
        (profile_dir / source).mkdir()
    assert unknown_root_entries(profile_dir) == []


def test_hooks_directory_is_not_read_as_a_typo(profile_dir: Path):
    """Named explicitly: before the channel existed, `hooks/` WAS an unknown entry."""
    _write(profile_dir / "hooks" / "facility_guard.py")
    assert unknown_root_entries(profile_dir) == []


def test_extra_known_names_suppress_the_flag(profile_dir: Path):
    (profile_dir / "facility-data").mkdir()
    assert unknown_root_entries(profile_dir) == ["facility-data"]
    assert unknown_root_entries(profile_dir, extra_known=["facility-data"]) == []


def test_warn_unknown_root_entries_logs_the_convention_list(
    profile_dir: Path, caplog: pytest.LogCaptureFixture
):
    (profile_dir / "rule").mkdir()
    with caplog.at_level(logging.WARNING):
        assert warn_unknown_root_entries(profile_dir) == ["rule"]
    assert "rule" in caplog.text
    assert "output-styles/" in caplog.text


def test_warn_unknown_root_entries_is_quiet_on_a_clean_profile(
    profile_dir: Path, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        assert warn_unknown_root_entries(profile_dir) == []
    assert caplog.text == ""


# ── Three-zone repo root ─────────────────────────────────────────────
#
# The repo root IS the profile root: source, secrets, output and state sit in
# one directory. Every test below carries `zone` in its name so the layout's
# properties can be run as one group.


@pytest.fixture
def zone_repo(tmp_path: Path) -> Path:
    """A complete three-zone deployment repo, hand-built from the layout.

    Source at the root (no nesting), the git-ignored `.env`, and both generated
    zones present — the shape `osprey init` emits and every build reads.
    """
    repo = tmp_path / "als-exemplar"
    repo.mkdir()

    # SOURCE — tracked, user-edited.
    _write(repo / "profile.yml", "name: als-exemplar\n")
    _write(repo / "triggers.yml")
    (repo / "data").mkdir()
    (repo / "personas").mkdir()
    _write(repo / "rules" / "safety.md")
    _write(repo / "web-terminal-context" / "operator" / "CONTEXT.md")
    _write(repo / "README.md")
    _write(repo / ".gitignore", "/build/\n/var/\n/.env*\n!.env.example\n")
    _write(repo / ".gitlab-ci.yml")
    _write(repo / "ci-extra.yml")
    _write(repo / "scripts" / "verify.sh")

    # SECRETS — ignored, durable.
    _write(repo / ".env")
    _write(repo / ".env.example")

    # OUTPUT — ignored, disposable.
    _write(repo / "build" / "config.yml")

    # STATE — ignored, durable.
    (repo / "var" / "agent_data").mkdir(parents=True)
    (repo / "var" / "audit").mkdir(parents=True)

    return repo


def test_zone_repo_root_warns_about_nothing(zone_repo: Path):
    """SC-9: a freshly init'ed repo builds with zero unknown-root-entry warnings."""
    assert unknown_root_entries(zone_repo) == []


def test_zone_repo_root_is_silent_through_the_warning_path(
    zone_repo: Path, caplog: pytest.LogCaptureFixture
):
    """SC-9 through the caller's door — the build calls `warn_`, not `unknown_`."""
    with caplog.at_level(logging.WARNING):
        assert warn_unknown_root_entries(zone_repo) == []
    assert caplog.text == ""


@pytest.mark.parametrize("entry", ["build", "var", "scripts", "ci-extra.yml"])
def test_zone_entry_is_known_on_its_own(profile_dir: Path, entry: str):
    """Each zone entry stands alone: the layout is not all-or-nothing."""
    if entry.endswith(".yml"):
        _write(profile_dir / entry)
    else:
        (profile_dir / entry).mkdir()
    assert unknown_root_entries(profile_dir) == []


def test_zone_generated_dirnames_are_the_ones_the_table_knows():
    """The exported spellings and the warning's table cannot drift apart: init
    creates these directories and build wipes one, all from these constants."""
    assert BUILD_OUTPUT_DIR in KNOWN_ROOT_ENTRIES
    assert STATE_DIR in KNOWN_ROOT_ENTRIES


def test_zone_layout_still_flags_a_genuinely_stray_entry(zone_repo: Path):
    """The other half of SC-9: recognizing the zones must not blind the check."""
    (zone_repo / "ioc").mkdir()
    _write(zone_repo / "rule" / "safety.md")
    assert unknown_root_entries(zone_repo) == ["ioc", "rule"]


def test_zone_remedy_does_not_instruct_nesting(zone_repo: Path, caplog: pytest.LogCaptureFixture):
    """The remedy must not tell operators to nest the profile in a
    profile/ directory. Source lives at the repo root."""
    (zone_repo / "ioc").mkdir()
    with caplog.at_level(logging.WARNING):
        warn_unknown_root_entries(zone_repo)

    assert "profile/profile.yml" not in caplog.text
    assert "nested profile/" not in caplog.text
    assert "repo root and the profile root at once" in caplog.text


def test_zone_remedy_names_the_channel_to_move_an_entry_into(
    zone_repo: Path, caplog: pytest.LogCaptureFixture
):
    """An operator with material that *should* reach the deployment needs the
    way in, since nesting it away is not an answer."""
    (zone_repo / "ioc").mkdir()
    with caplog.at_level(logging.WARNING):
        warn_unknown_root_entries(zone_repo)

    assert f"{PROJECT_MIRROR_DIR}/" in caplog.text
    assert "channel that carries it" in caplog.text
    assert "repo-local material" in caplog.text
    assert "leaving it here costs nothing" in caplog.text


def test_zone_remedy_keeps_the_typo_answer(profile_dir: Path, caplog: pytest.LogCaptureFixture):
    """The original cause still gets its own answer: the convention list."""
    (profile_dir / "rule").mkdir()
    with caplog.at_level(logging.WARNING):
        assert warn_unknown_root_entries(profile_dir) == ["rule"]

    assert "check for a typo" in caplog.text
    assert "rules/" in caplog.text


def test_zone_remedy_names_the_repo_it_is_judging(
    zone_repo: Path, caplog: pytest.LogCaptureFixture
):
    """Without the path, an operator building several repos cannot tell which
    one the advice is about."""
    (zone_repo / "ioc").mkdir()
    with caplog.at_level(logging.WARNING):
        warn_unknown_root_entries(zone_repo)

    assert str(zone_repo) in caplog.text
