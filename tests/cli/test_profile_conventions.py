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

from osprey.cli import profile_conventions as conventions
from osprey.cli.profile_conventions import (
    BUILD_OUTPUT_DIR,
    CONTEXT_BASELINE_FILENAME,
    CONVENTION_DIRS,
    CONVENTION_SOURCES,
    KNOWN_ROOT_ENTRIES,
    NOT_PROJECT_RELATIVE_CHANNEL,
    PER_USER_CONTEXT_DIRNAME,
    PROJECT_MIRROR_DIR,
    PROTECTED_CONFIG_KEYS,
    PROTECTED_KEY_EXEMPTIONS,
    RESERVED_EXACT_PATHS,
    RESERVED_PATH_CHANNELS,
    RESERVED_PATH_PATTERNS,
    RESERVED_PROJECT_PATHS,
    SETUP_PATCH_TOOL,
    STATE_DIR,
    ConventionDir,
    EntryShape,
    ReservedPattern,
    convention_for,
    convention_slot_for,
    destination_for,
    flatten_dotted,
    flatten_key_paths,
    is_protected_key,
    is_protected_key_path,
    is_reserved_write,
    is_setup_patch_capable,
    ownership_name,
    partition_context_users,
    plan_convention_copies,
    protected_view,
    reserved_path_channel,
    unknown_root_entries,
    validate_convention_sources,
    validate_project_mirror,
    warn_unknown_root_entries,
)
from osprey.errors import BuildProfileError
from osprey_connectors.config import RUNTIME_WRITE_PATH_KEYS


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


def test_loose_file_advice_offers_the_baseline_slot_too(profile_dir: Path):
    """Both routes, because the message cannot tell which one the file wants.

    A loose file here is as likely to be the shared baseline under the wrong
    name as it is to be one user's context, and the baseline slot is the only
    other place the convention carries a file. Offering just the per-user route
    tells an operator with baseline text to file it under one user, where the
    other users never see it.
    """
    _write(profile_dir / PER_USER_CONTEXT_DIRNAME / "shift-notes.md")

    with pytest.raises(BuildProfileError) as excinfo:
        validate_convention_sources(profile_dir)

    message = str(excinfo.value)
    assert f"rename it to {PER_USER_CONTEXT_DIRNAME}/{CONTEXT_BASELINE_FILENAME}" in message
    assert f"{PER_USER_CONTEXT_DIRNAME}/<user>/shift-notes.md" in message


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


def test_zone_scaffolded_boot_unit_is_not_flagged(zone_repo: Path):
    """`osprey scaffold systemd` writes the unit beside the profile.

    The repo root IS the profile root, so an unrecognized unit would make every
    build after the scaffold warn about a file OSPREY itself told the operator
    to create — a warning that can never be cleared except by deleting the unit.
    """
    from osprey.cli.deploy_scaffold_templates import SYSTEMD_UNIT_NAME

    _write(zone_repo / SYSTEMD_UNIT_NAME)
    assert unknown_root_entries(zone_repo) == []


def test_zone_boot_unit_spelling_matches_the_verb_that_writes_it():
    """The two literals cannot drift apart.

    `profile_conventions` spells the unit name itself rather than importing it,
    because the module that owns the name imports this one. A rename on either
    side that missed the other would bring the every-build warning straight
    back, so the equality is asserted instead of assumed.
    """
    from osprey.cli.deploy_scaffold import SYSTEMD_OUTPUT_NAME

    assert SYSTEMD_OUTPUT_NAME in KNOWN_ROOT_ENTRIES


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


# ── Protected set: reserved writes ───────────────────────────────────
#
# The protected set is what every framework writer (the scaffold gallery, the
# Claude-setup panel, the `setup_patch` MCP tool) consults before it writes into
# a built project. It is a *different* question from the `project/` mirror's
# reservations above — that one asks which build channel owns a path, this one
# asks whether a running agent may rewrite the safety configuration it lives
# under — so the two tables are pinned separately.


def test_reserved_exact_table_is_unchanged_by_the_pattern_table():
    """The mirror/ownership consumers read the exact table and must not shift.

    ``ownership.generated_project_paths()`` returns ``RESERVED_EXACT_PATHS``
    verbatim and ``scaffold``'s claim refusal reads the same set, so a pattern
    added for the protected set must not leak into it.
    """
    assert RESERVED_EXACT_PATHS == frozenset(r.path for r in RESERVED_PROJECT_PATHS)
    assert RESERVED_PATH_CHANNELS == {r.path: r.channel for r in RESERVED_PROJECT_PATHS}
    assert len(RESERVED_PROJECT_PATHS) == 11


@pytest.mark.parametrize(
    ("target", "channel_hint"),
    [
        (".claude/hooks/osprey_writes_check.py", "`hooks/`"),
        (".claude/hooks/osprey_limits.py", "`hooks/`"),
        (".claude/skills/foo/SKILL.md", "`skills/`"),
        (".claude/skills/foo/scripts/run.py", "`skills/`"),
        (".claude/rules/facility.md", "`rules/`"),
        (".claude/rules/nested/deep.md", "`rules/`"),
        (".claude/settings.local.json", "claude_code.permissions"),
        ("data/channel_limits.json", "`data/`"),
        ("data/bluesky_devices.yml", "`data/`"),
    ],
)
def test_pattern_reserved_write_names_its_channel(target: str, channel_hint: str):
    """A refusal has to say who *does* write the path, or it is a dead end."""
    channel = is_reserved_write(target)
    assert channel is not None
    assert channel_hint in channel


@pytest.mark.parametrize(
    "allowed",
    [
        ".claude/agents/orbit.md",
        ".claude/commands/scan.md",
        ".claude/output-styles/terse.md",
        ".claude/hooks/facility_guard.py",
        "docs/runbook.md",
        "data/facility.json",
        "data/simulation/channel_manifest.json.bak",
        "notebooks/analysis.ipynb",
    ],
)
def test_unreserved_writes_stay_writable(allowed: str):
    """The matcher is falsifiable: near-misses of every pattern still pass.

    ``.claude/agents/`` and ``.claude/commands/`` are deliberately *not* in the
    protected set — an agent authoring its own subagent or slash command is the
    point of the panel. ``facility_guard.py`` is a profile-authored hook, not an
    ``osprey_`` one, so the write-safety layer does not own it.
    """
    assert is_reserved_write(allowed) is None


def test_every_exact_reservation_is_a_reserved_write():
    for entry in RESERVED_PROJECT_PATHS:
        assert is_reserved_write(entry.path) == entry.channel


def test_exact_reservation_beats_the_pattern_table(monkeypatch: pytest.MonkeyPatch):
    """ORDER pin: the exact table answers before the pattern table.

    The exact entries carry the precise channel ("the profile's `config:`
    block"); a pattern that happened to cover the same path would answer with a
    coarser one and send an operator to the wrong place. No shipped pattern
    overlaps an exact entry today, so the ordering is pinned with a decoy —
    otherwise a flip of the two lookups would go unnoticed until a pattern
    widened onto an exact path.
    """
    decoy = ReservedPattern("config.*", "a pattern that must never answer first")
    monkeypatch.setattr(conventions, "RESERVED_PATH_PATTERNS", (decoy, *RESERVED_PATH_PATTERNS))
    assert is_reserved_write("config.yml") == RESERVED_PATH_CHANNELS["config.yml"]
    # The decoy still works, so the test cannot pass by matching nothing.
    assert is_reserved_write("config.toml") == decoy.channel


def test_skill_files_stay_ownable_while_being_reserved_writes():
    """The two questions must not be conflated.

    A profile ships skills through its ``skills/`` convention directory and the
    build registers them as user-owned, which is unchanged. What is new is that
    an *agent-side* writer may not rewrite one — a skill is instruction text the
    agent would otherwise be editing for itself.
    """
    assert ".claude/skills/foo/SKILL.md" not in RESERVED_EXACT_PATHS
    assert ownership_name(".claude/skills/foo", is_directory=True) == "skills/foo"
    assert is_reserved_write(".claude/skills/foo/SKILL.md") is not None


def test_channel_limits_overlay_still_validates_in_the_mirror(profile_dir: Path):
    """``project/data/channel_limits.json`` is a legitimate profile overlay.

    The protected set stops a *running agent* from rewriting the limits table;
    it must not stop the profile that authored it from shipping one, or the
    build breaks for every deployment carrying facility limits in the mirror.
    """
    mirror = profile_dir / "project" / "data"
    _write(mirror / "channel_limits.json", "{}\n")

    validate_project_mirror(profile_dir)  # does not raise
    assert reserved_path_channel("data/channel_limits.json") is None
    assert is_reserved_write("data/channel_limits.json") is not None


def test_reserved_write_normalizes_its_input():
    assert is_reserved_write("./.claude/rules/facility.md") is not None


# ── Protected set: paths spelled the way an attacker would spell them ───
#
# The negatives above are all well-formed near-misses. These pin the other
# half: a path that names a protected file while *not* being spelled like one.
# Each of these opens exactly the file the pattern names, so each must refuse.


@pytest.mark.parametrize(
    ("target", "channel_hint"),
    [
        (".CLAUDE/skills/foo/SKILL.md", "`skills/`"),
        (".claude/Skills/foo/SKILL.md", "`skills/`"),
        (".claude/RULES/facility.md", "`rules/`"),
        (".claude/settings.LOCAL.json", "claude_code.permissions"),
        (".claude/Settings.Local.Json", "claude_code.permissions"),
        (".claude/hooks/OSPREY_limits.py", "`hooks/`"),
        ("DATA/Channel_Limits.json", "`data/`"),
    ],
)
def test_case_variants_of_a_pattern_are_refused(target: str, channel_hint: str):
    """macOS and Windows open the same file whatever the case.

    A case-sensitive match would make the protected set a protected set only on
    Linux: on the filesystems the framework actually ships to, ``.CLAUDE/skills``
    *is* ``.claude/skills``, and a writer that capitalized a letter would have
    rewritten the skill the pattern exists to protect.
    """
    channel = is_reserved_write(target)
    assert channel is not None
    assert channel_hint in channel


@pytest.mark.parametrize(
    ("target", "exact"),
    [
        ("Config.yml", "config.yml"),
        (".claude/Settings.json", ".claude/settings.json"),
        (".MCP.json", ".mcp.json"),
    ],
)
def test_case_variants_of_an_exact_reservation_are_refused(target: str, exact: str):
    """The folded lookup answers with the *same* channel the exact spelling does."""
    assert is_reserved_write(target) == RESERVED_PATH_CHANNELS[exact]


def test_case_folding_lives_only_in_the_lookup():
    """The exported tables keep the shipped spelling.

    ``RESERVED_EXACT_PATHS`` and ``RESERVED_PATH_CHANNELS`` are read verbatim by
    the ownership rules and the ``project/`` mirror, which compare real path
    strings — folding a key there would un-reserve every entry for them. The
    fold belongs to ``is_reserved_write``'s question alone.
    """
    assert ".claude/settings.json" in RESERVED_EXACT_PATHS
    assert ".claude/settings.json".casefold() not in RESERVED_EXACT_PATHS - {
        ".claude/settings.json"
    }
    assert all(r.path in RESERVED_PATH_CHANNELS for r in RESERVED_PROJECT_PATHS)
    # No two reservations differ only by case, or the folded lookup would drop one.
    folded = {path.casefold() for path in RESERVED_EXACT_PATHS}
    assert len(folded) == len(RESERVED_EXACT_PATHS)
    # The mirror's own question stays case-sensitive: it validates profile files
    # by their real names, and is not the agent-write question at all.
    assert reserved_path_channel(".CLAUDE/settings.json") is None


@pytest.mark.parametrize(
    "target",
    [
        "foo/../.claude/skills/x",
        "docs/./../.claude/rules/facility.md",
        ".claude/agents/../skills/x",
        ".claude//skills//x",
        "./data/channel_limits.json",
    ],
)
def test_traversal_spellings_still_reach_the_protected_file(target: str):
    """``..`` is normalized away before matching, like ``./`` and ``//``.

    Without it a writer opens a protected file by routing through a directory
    that is not protected — the set would be a naming convention rather than a
    rule about files.
    """
    assert is_reserved_write(target) is not None


@pytest.mark.parametrize(
    "target",
    ["../outside.md", "foo/../../etc/passwd", "..", "/etc/passwd", "/srv/project/config.yml"],
)
def test_a_path_that_climbs_out_of_the_project_is_refused(target: str):
    """A path the set cannot judge must never read as a writable one.

    Nothing in the protected set describes a file outside the project, so a
    matcher answering ``None`` would hand the writer a free pass on exactly the
    paths it understands least. The refusal names the contract instead.
    """
    assert is_reserved_write(target) == NOT_PROJECT_RELATIVE_CHANNEL


def test_normalization_does_not_widen_the_set():
    """The falsifiable half: normalizing must not start refusing ordinary paths."""
    assert is_reserved_write("./docs/runbook.md") is None
    assert is_reserved_write("docs/../notebooks/analysis.ipynb") is None
    assert is_reserved_write(".claude/skills/../agents/orbit.md") is None
    # A traversal that lands back *inside* the project is an ordinary path.
    assert is_reserved_write(".claude/skills/../../data/facility.json") is None


# ── Protected set: config keys ───────────────────────────────────────


PROTECTED_KEY_FAMILIES = [
    # (target file, a member of the family, two negative controls)
    (
        "config.yml",
        "control_system.limits_checking.enabled",
        "control_systems.enabled",
        "services.control_system.limits",
    ),
    ("config.yml", "approval.mode", "approvals.mode", "ui.approval.mode"),
    # `hooks.enabled` rather than `hooks.debug`: the latter is the one member of
    # this family carved out by PROTECTED_KEY_EXEMPTIONS, and a family probe has
    # to stand on a key the family actually protects.
    ("config.yml", "hooks.enabled", "hooks_enabled", "web.hooks.enabled"),
    (
        "config.yml",
        "claude_code.permissions.deny",
        "claude_code.permission.deny",
        "permissions.deny",
    ),
    ("config.yml", "claude_code.hooks.x", "claude_code.hooksx", "claude_code.model"),
    (
        "config.yml",
        "claude_code.servers.foo.command",
        "claude_code.server.foo",
        "servers.foo.command",
    ),
    ("config.yml", "artifacts.hooks", "artifact.hooks", "docs.artifacts.hooks"),
    ("config.yml", "agent_data.base_dir", "agent_datax.base_dir", "services.agent_data.base_dir"),
    ("config.yml", "file_paths.x", "file_path.x", "services.file_paths.x"),
    ("config.yml", "artifacts.x", "artifact.x", "services.artifacts.x"),
    (
        "config.yml",
        "services.bluesky.devices_file",
        "service.bluesky.devices_file",
        "services.bluesky.devices",
    ),
    (".mcp.json", "mcpServers.foo.command", "mcpServers.foo.disabled", "servers.foo.command"),
]


@pytest.mark.parametrize(
    ("target_file", "protected", "negative_a", "negative_b"),
    PROTECTED_KEY_FAMILIES,
)
def test_protected_key_families(target_file: str, protected: str, negative_a: str, negative_b: str):
    """One key per declared family, plus two near-misses that must stay open.

    Without the negative controls a matcher that returned ``True`` for
    everything would satisfy the positive half, and the protected set would
    silently freeze the whole config file.
    """
    assert is_protected_key(target_file, protected) is True
    assert is_protected_key(target_file, negative_a) is False
    assert is_protected_key(target_file, negative_b) is False


@pytest.mark.parametrize(
    ("target_file", "dotted"),
    [
        (".mcp.json", "mcpServers.foo.args"),
        (".mcp.json", "mcpServers.foo.env.API_KEY"),
        (".mcp.json", "mcpServers.foo.env"),
    ],
)
def test_mcp_server_launch_surface_is_protected(target_file: str, dotted: str):
    """A server's command line and environment are how it gets its privileges."""
    assert is_protected_key(target_file, dotted) is True


def test_every_runtime_write_path_key_is_protected():
    """The runtime-write keys are imported, not restated, so they cannot drift.

    Each names a path something writes at run time; the safety layers derive
    their allow/deny zones from those paths, so repointing one moves the zone.
    """
    for key in RUNTIME_WRITE_PATH_KEYS:
        assert is_protected_key("config.yml", key) is True
    assert set(RUNTIME_WRITE_PATH_KEYS) <= set(PROTECTED_CONFIG_KEYS["config.yml"])


def test_protection_does_not_cross_files():
    """Each file has its own table; a key protected in one is not in the other."""
    assert is_protected_key(".mcp.json", "control_system.writes_enabled") is False
    assert is_protected_key("config.yml", "mcpServers.foo.command") is False


def test_unknown_target_file_protects_nothing():
    """Only the two files the writers may patch have tables at all."""
    assert is_protected_key("README.md", "control_system.writes_enabled") is False


def test_target_file_may_be_given_as_a_path():
    assert is_protected_key("/srv/project/config.yml", "approval.mode") is True


def test_an_ancestor_of_a_protected_key_is_protected():
    """Writing the parent rewrites the protected child.

    ``mcpServers`` carries no pattern of its own, but replacing it wholesale
    replaces every server's command line, so the ancestor is refused too.
    """
    assert is_protected_key(".mcp.json", "mcpServers") is True
    assert is_protected_key("config.yml", "claude_code") is True
    # An ancestor is only the *prefix* of a pattern, never an unrelated sibling
    # under the same root — `services.channel_finder…` is a protected runtime
    # write path, but `services` as a namespace is not sealed off.
    assert is_protected_key("config.yml", "services.orbit.enabled") is False


# ── Protected set: exemptions ────────────────────────────


def test_hooks_debug_is_exempt_from_the_hooks_family():
    """`hooks.debug` fails the inclusion rule, and `hooks.*` over-matched it.

    It gates no write, approval or limit, anchors no path a safety layer derives
    a zone from, and is rendered into no artifact — it toggles diagnostic
    verbosity in the hook scripts. It is also a shipped operator control (the
    Web Terminal's Hook Debug switch PATCHes it), so leaving it inside the
    family would break a feature to protect nothing.
    """
    assert is_protected_key("config.yml", "hooks.debug") is False
    assert is_protected_key_path("config.yml", ("hooks", "debug")) is False
    assert protected_view("config.yml", {"hooks": {"debug": True}}) == {}


def test_the_hooks_exemption_subtracts_one_key_and_nothing_else():
    """The rest of the family stays protected — including the block itself.

    Exempting a leaf must not exempt its parent: replacing the whole `hooks`
    block rewrites the wiring, which is what the family is there for.
    """
    assert is_protected_key("config.yml", "hooks") is True
    assert is_protected_key("config.yml", "hooks.enabled") is True
    assert is_protected_key("config.yml", "hooks.debug_extra") is True
    assert is_protected_key("config.yml", "hooks.debug.level") is True
    assert is_protected_key("config.yml", "debug") is False  # not in the family at all


def test_an_exemption_is_exact_and_cannot_widen():
    """Exact tuple membership — no wildcards, no prefixes, no ancestor rule.

    A typo in a *pattern* can only over-protect, which is safe; a typo in an
    exemption that behaved like a pattern could un-protect a whole family. So an
    exemption names one key, and a `*` in one is a literal segment, not a
    wildcard. Both properties are asserted against the table as it stands.
    """
    for keys in PROTECTED_KEY_EXEMPTIONS.values():
        for key_path in keys:
            assert "*" not in key_path, f"{key_path} — exemptions carry no patterns"

    # A hypothetical `hooks.*` exemption would only ever match a literal `*`
    # segment; nothing in the family is un-protected by proximity to an entry.
    assert is_protected_key("config.yml", "hooks.debugging") is True
    assert is_protected_key("config.yml", "hooks.Debug") is True  # exact case, too


def test_every_exemption_actually_subtracts_something():
    """An exemption for a key no family covers is dead weight, not protection.

    Guards the table against a stale entry: if a family pattern is narrowed
    later so it no longer reaches an exempted key, this fails rather than
    leaving a line that reads like it is doing work.
    """
    for target_file, keys in PROTECTED_KEY_EXEMPTIONS.items():
        assert target_file in PROTECTED_CONFIG_KEYS, f"{target_file} has no protected-key table"
        for key_path in keys:
            covered = any(
                _pattern_covers(pattern, key_path) for pattern in PROTECTED_CONFIG_KEYS[target_file]
            )
            assert covered, f"{key_path} is exempt from nothing — no pattern covers it"


def _pattern_covers(pattern: str, key_path: tuple[str, ...]) -> bool:
    """Whether the family pattern would cover this key if it were not exempt."""
    from osprey.cli.profile_conventions import _key_path_matches

    return _key_path_matches(pattern, key_path)


# ── Protected set: the PUT diff primitive ────────────────────────────


def _config_doc() -> dict:
    return {
        "control_system": {"limits_checking": {"enabled": True}, "write_tools": ["set_pv"]},
        "approval": {"mode": "always"},
        "logging": {"level": "INFO"},
    }


def test_flatten_dotted_leaves_lists_and_scalars_whole():
    flat = flatten_dotted(_config_doc())
    assert flat["control_system.limits_checking.enabled"] is True
    assert flat["control_system.write_tools"] == ["set_pv"]
    assert flat["logging.level"] == "INFO"
    assert "control_system" not in flat


def test_flatten_dotted_keeps_an_empty_mapping_as_a_leaf():
    """Otherwise an emptied subtree vanishes and reads as an untouched one."""
    assert flatten_dotted({"approval": {}}) == {"approval": {}}


def test_protected_view_keeps_only_protected_keys():
    """The view is keyed by segment path, not by a dotted string — see
    test_protected_view_sees_a_server_whose_name_contains_a_dot for why."""
    view = protected_view("config.yml", _config_doc())
    assert ("logging", "level") not in view
    assert view[("approval", "mode")] == "always"
    assert view[("control_system", "write_tools")] == ["set_pv"]


def test_protected_view_ignores_an_unprotected_change():
    """The diff primitive must not refuse a write it has no business refusing."""
    changed = _config_doc()
    changed["logging"]["level"] = "DEBUG"
    assert protected_view("config.yml", changed) == protected_view("config.yml", _config_doc())


def test_protected_view_catches_an_added_key():
    changed = _config_doc()
    changed["control_system"]["writes_enabled"] = True
    assert protected_view("config.yml", changed) != protected_view("config.yml", _config_doc())


def test_protected_view_catches_a_deleted_key():
    changed = _config_doc()
    del changed["approval"]["mode"]
    assert protected_view("config.yml", changed) != protected_view("config.yml", _config_doc())


def test_protected_view_catches_a_changed_value():
    changed = _config_doc()
    changed["approval"]["mode"] = "never"
    assert protected_view("config.yml", changed) != protected_view("config.yml", _config_doc())


def test_protected_view_catches_a_changed_list_element():
    """Lists are compared whole — a widened `write_tools` is a privilege change."""
    changed = _config_doc()
    changed["control_system"]["write_tools"] = ["set_pv", "caput"]
    assert protected_view("config.yml", changed) != protected_view("config.yml", _config_doc())


def test_protected_view_catches_a_subtree_reshaped_into_a_scalar():
    """dict → scalar is the reshape a key-by-key comparison misses.

    Replacing the whole ``control_system`` block with a scalar deletes every
    protected key beneath it while adding one the flattener has never seen.
    """
    changed = _config_doc()
    changed["control_system"] = "disabled"
    before = protected_view("config.yml", _config_doc())
    after = protected_view("config.yml", changed)
    assert after != before
    assert ("control_system", "limits_checking", "enabled") not in after
    assert after[("control_system",)] == "disabled"


def test_protected_view_of_the_mcp_file():
    doc = {"mcpServers": {"osprey": {"command": "uv", "args": ["run"], "env": {"K": "v"}}}}
    view = protected_view(".mcp.json", doc)
    assert view == {
        ("mcpServers", "osprey", "command"): "uv",
        ("mcpServers", "osprey", "args"): ["run"],
        ("mcpServers", "osprey", "env", "K"): "v",
    }


# ── Protected set: keys spelled the way an attacker would spell them ───


def test_flatten_key_paths_keeps_a_dotted_key_whole():
    """A raw key containing a ``.`` is one segment, not two.

    The document is the attacker's to shape: ``.mcp.json`` server names are
    arbitrary strings. Splitting a flattened key back apart would let a chosen
    name change how many segments a pattern sees.
    """
    flat = flatten_key_paths({"mcpServers": {"evil.srv": {"command": "sh"}}})
    assert flat == {("mcpServers", "evil.srv", "command"): "sh"}


def test_a_server_name_containing_a_dot_is_still_protected():
    """``mcpServers.*.command`` covers ``evil.srv``'s command line.

    ``*`` stands for exactly one segment, so a dotted *rendering* of this key
    (``mcpServers.evil.srv.command``) has four segments and matches nothing —
    which is why protection is asked on segment paths.
    """
    assert is_protected_key_path(".mcp.json", ("mcpServers", "evil.srv", "command")) is True
    assert is_protected_key_path(".mcp.json", ("mcpServers", "a.b.c", "env", "K")) is True
    # The dotted front door splits on `.`, so it cannot express this key. That is
    # its documented contract, not a second protected-set answer: callers holding
    # a parsed document use the key-path form.
    assert is_protected_key(".mcp.json", "mcpServers.evil.srv.command") is False


def test_protected_view_sees_a_server_whose_name_contains_a_dot():
    """The PUT diff must not go blind on a chosen name.

    Adding a server called ``evil.srv`` installs a launcher the agent runs. With
    a dotted view its command flattens to ``mcpServers.evil.srv.command``, which
    no pattern covers, so it drops out of the view and before/after compare
    equal — the malicious server lands unnoticed on a whole-file PUT.
    """
    before = {"mcpServers": {"osprey": {"command": "uv"}}}
    after = {
        "mcpServers": {"osprey": {"command": "uv"}, "evil.srv": {"command": "sh", "args": ["-c"]}}
    }
    view_before = protected_view(".mcp.json", before)
    view_after = protected_view(".mcp.json", after)
    assert view_after != view_before
    assert view_after[("mcpServers", "evil.srv", "command")] == "sh"
    assert view_after[("mcpServers", "evil.srv", "args")] == ["-c"]


def test_a_dotted_name_cannot_impersonate_a_nested_key():
    """Two documents that render identically when dotted stay distinguishable.

    ``{"evil.srv": {...}}`` and ``{"evil": {"srv": {...}}}`` flatten to the same
    dotted string; on a segment path they do not, so one can never be swapped
    for the other under a PUT that compares views.
    """
    flat_name = protected_view(".mcp.json", {"mcpServers": {"evil.srv": {"command": "sh"}}})
    nested = protected_view(".mcp.json", {"mcpServers": {"evil": {"srv": {"command": "sh"}}}})
    assert flat_name != nested
    assert set(flatten_dotted({"mcpServers": {"evil.srv": {"command": "sh"}}})) == set(
        flatten_dotted({"mcpServers": {"evil": {"srv": {"command": "sh"}}}})
    )  # the dotted rendering really does collide — hence the tuple keys


# ── Posture: is the rendered persona still setup_patch-capable? ────────


def _persona_config(deny: object, *, include_permissions: bool = True) -> dict:
    """A persona config.yml document carrying (or not) a `claude_code` deny list."""
    config: dict = {"control_system": {"writes_enabled": False}}
    if include_permissions:
        config["claude_code"] = {"permissions": {"deny": deny}}
    return config


def test_setup_patch_capable_is_false_when_the_tool_is_denied():
    config = _persona_config(["Bash", SETUP_PATCH_TOOL])
    assert is_setup_patch_capable(config) is False


def test_setup_patch_capable_is_true_when_another_tool_is_denied():
    """A non-empty deny that leaves the tool alone still leaves the capability."""
    assert is_setup_patch_capable(_persona_config(["Bash", "Edit"])) is True


def test_setup_patch_capable_is_true_with_no_deny_block():
    """The deny is what removes the capability — its absence cannot impose one."""
    config = {"claude_code": {"permissions": {"remove_deny": ["Bash"]}}}
    assert is_setup_patch_capable(config) is True


def test_setup_patch_capable_is_true_with_no_claude_code_block():
    assert is_setup_patch_capable(_persona_config(None, include_permissions=False)) is True
    assert is_setup_patch_capable({}) is True


def test_setup_patch_capable_is_true_for_an_empty_or_null_deny():
    assert is_setup_patch_capable(_persona_config([])) is True
    assert is_setup_patch_capable(_persona_config(None)) is True


def test_setup_patch_capable_is_false_under_a_wildcard_deny():
    """A deny that names the tool by pattern removes the capability just as hard."""
    assert is_setup_patch_capable(_persona_config(["mcp__osprey_workspace__*"])) is False


def test_setup_patch_capable_reads_a_rendered_settings_json_shape():
    """A settings.json document has no `claude_code:` wrapper — read its own block.

    Without this a rendered `.claude/settings.json` handed to the predicate would
    find no `claude_code` block and read as capable while the tool is denied.
    """
    assert is_setup_patch_capable({"permissions": {"deny": [SETUP_PATCH_TOOL]}}) is False
    assert is_setup_patch_capable({"permissions": {"deny": ["Bash"]}}) is True


def test_setup_patch_capable_tolerates_a_misshapen_document():
    """A malformed block is not a deny list, and nothing else is judgeable from it."""
    assert is_setup_patch_capable({"claude_code": "off"}) is True
    assert is_setup_patch_capable({"claude_code": {"permissions": "off"}}) is True
    assert is_setup_patch_capable(None) is True


def test_a_bare_string_deny_is_no_deny_at_all():
    """A deny spelled without its list is not "one entry" — the render says so.

    ``settings.json.j2`` iterates the value, so a bare-string deny renders one
    entry per CHARACTER and denies the tool nowhere. Reading it as one entry
    here would report ``False`` for a render that lets the tool through, which
    is a disagreement with the render in the direction the roster guard reads.
    The spelling never reaches a build:
    ``build_profile_load._reject_permission_list_shapes`` refuses it at parse
    time, which is what makes reading it as nothing the safe answer here.
    """
    assert is_setup_patch_capable(_persona_config(SETUP_PATCH_TOOL)) is True
    assert is_setup_patch_capable(_persona_config("Bash")) is True
    # And the render agrees: 34 single-character entries, none of them the tool.
    from osprey.cli.templates.claude_code import _rendered_deny_list

    rendered = _rendered_deny_list(
        {
            "deny_defaults": [],
            "facility_permissions": {"deny": SETUP_PATCH_TOOL},
            "killswitch_deny": [],
        }
    )
    assert SETUP_PATCH_TOOL not in rendered
    assert all(len(entry) == 1 for entry in rendered)


def _lifting_config(deny: object, remove_deny: object) -> dict:
    """A persona config.yml carrying an inherited deny AND a tier's lift of it.

    The shape a rendered persona ``config.yml`` really has: a delta cannot
    subtract from an inherited list, so both lists ship side by side and the
    render composes them.
    """
    return {
        "control_system": {"writes_enabled": False},
        "claude_code": {"permissions": {"deny": deny, "remove_deny": remove_deny}},
    }


def test_setup_patch_capable_is_true_when_remove_deny_lifts_the_tool():
    """The admin tier's shape. Reading the deny alone would call it denied and
    every gate that withholds a privilege on ``False`` would fail open on it."""
    config = _lifting_config(["Bash", SETUP_PATCH_TOOL], [SETUP_PATCH_TOOL])
    assert is_setup_patch_capable(config) is True


def test_setup_patch_capable_is_false_when_remove_deny_lifts_another_tool():
    """A lift is not a blanket amnesty — it subtracts the entries it names."""
    config = _lifting_config([SETUP_PATCH_TOOL], ["Bash"])
    assert is_setup_patch_capable(config) is False


def test_a_bare_string_remove_deny_lifts_nothing():
    """A lift spelled without its list lifts nothing here — but the render's
    ``d not in remove_deny`` against a string is SUBSTRING containment, which
    lifts by accident. Neither reading is defensible, so the spelling is refused
    at profile-parse time and this reads it as no lift: the deny stands.
    """
    assert is_setup_patch_capable(_lifting_config([SETUP_PATCH_TOOL], SETUP_PATCH_TOOL)) is False
    assert is_setup_patch_capable(_lifting_config([SETUP_PATCH_TOOL], "Bash")) is False
    # The substring hazard the refusal exists for: a longer string that merely
    # CONTAINS the tool name would lift the exact deny in the render.
    from osprey.cli.templates.claude_code import _rendered_deny_list

    assert (
        _rendered_deny_list(
            {
                "deny_defaults": [],
                "facility_permissions": {
                    "deny": [SETUP_PATCH_TOOL],
                    "remove_deny": f"{SETUP_PATCH_TOOL}_x",
                },
                "killswitch_deny": [],
            }
        )
        == []
    )


def test_a_misshapen_remove_deny_lifts_nothing():
    """An unreadable lift is no lift: the deny stands. The tolerance that reads
    a misshapen ``deny`` as empty must not run the other way here, or a
    malformed document would hand out the capability."""
    for lift in (7, {"deny": SETUP_PATCH_TOOL}, None, [None, 3]):
        assert is_setup_patch_capable(_lifting_config([SETUP_PATCH_TOOL], lift)) is False, lift


def test_an_exact_remove_deny_does_not_lift_a_wildcard_deny():
    """Mirrors the render, which subtracts by exact membership
    (``d not in remove_deny``) while the tool match is a glob. So a wildcard
    deny survives an exact lift in the rendered settings.json — and here."""
    config = _lifting_config(["mcp__osprey_workspace__*"], [SETUP_PATCH_TOOL])
    assert is_setup_patch_capable(config) is False
    # The exact entry beside it IS lifted; only the wildcard is left standing.
    both = _lifting_config(["mcp__osprey_workspace__*", SETUP_PATCH_TOOL], [SETUP_PATCH_TOOL])
    assert is_setup_patch_capable(both) is False


def test_a_settings_shaped_document_ignores_a_stray_remove_deny():
    """``settings.json`` is the composed output — its deny array is what the
    agent may not call. A ``remove_deny`` key there was already applied (or
    never belonged), and re-subtracting it would lift a deny that the running
    deployment enforces."""
    settings = {"permissions": {"deny": [SETUP_PATCH_TOOL], "remove_deny": [SETUP_PATCH_TOOL]}}
    assert is_setup_patch_capable(settings) is False


# ── The parity the predicate's every reading rule is stated as ────────


def _render_ctx(config: dict) -> dict:
    """The render context a persona ``config.yml`` produces, as the build builds it.

    ``ctx['facility_permissions']`` is literally ``claude_code.permissions``
    (``templates/claude_code.py``), and the floor is left empty on purpose: the
    predicate reads the PROFILE's deny and never the ``DENY_DEFAULTS`` floor,
    a split ``test_the_floor_does_not_deny_the_setup_tool`` pins separately.
    """
    return {
        "deny_defaults": [],
        "facility_permissions": config["claude_code"]["permissions"],
        "killswitch_deny": [],
    }


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_lifting_config([], []), id="nothing_denied"),
        pytest.param(_lifting_config([SETUP_PATCH_TOOL], []), id="denied"),
        pytest.param(_lifting_config(["Bash", "Edit"], []), id="other_tools_denied"),
        pytest.param(_lifting_config([SETUP_PATCH_TOOL], [SETUP_PATCH_TOOL]), id="lifted"),
        pytest.param(_lifting_config([SETUP_PATCH_TOOL], ["Bash"]), id="other_tool_lifted"),
        pytest.param(
            _lifting_config(["Bash", SETUP_PATCH_TOOL], ["Bash", SETUP_PATCH_TOOL]),
            id="both_lifted",
        ),
        pytest.param(_lifting_config(["Bash"], ["Bash", SETUP_PATCH_TOOL]), id="lift_without_deny"),
    ],
)
def test_the_predicate_matches_the_rendered_deny_list(config: dict):
    """For every list-shaped document, the predicate answers what the render renders.

    Parity — not a lean in a safe direction — is the criterion, because the two
    consumers pull opposite ways: the Dockerfile GRANTS the config.yml chown on
    ``True`` and the persona-roster guard REFUSES on ``True``. This holds the
    predicate against ``_rendered_deny_list``, the function whose
    ``d not in remove_deny`` the exact subtraction mirrors, so the duplicated
    composition cannot drift from the render without a red test.
    """
    from osprey.cli.templates.claude_code import _rendered_deny_list

    rendered = _rendered_deny_list(_render_ctx(config))
    assert is_setup_patch_capable(config) is (SETUP_PATCH_TOOL not in rendered)


def test_the_glob_match_is_the_other_mirror():
    """The one place the predicate is deliberately not exact membership.

    A wildcard deny is IN the rendered array as a pattern, so exact membership
    would call the render capable. The predicate matches with ``fnmatchcase``
    because that is how Claude Code itself resolves a deny pattern — the second
    of the two mirrors the docstring names — and the honest parity statement for
    a wildcard document uses the same matcher against the rendered array.
    """
    from fnmatch import fnmatchcase

    from osprey.cli.templates.claude_code import _rendered_deny_list

    config = _lifting_config(["mcp__osprey_workspace__*"], [])
    rendered = _rendered_deny_list(_render_ctx(config))
    assert SETUP_PATCH_TOOL not in rendered  # exact membership would say "capable"
    assert is_setup_patch_capable(config) is False
    assert is_setup_patch_capable(config) is not any(
        fnmatchcase(SETUP_PATCH_TOOL, entry) for entry in rendered
    )


def test_dangerously_allow_bash_is_protected():
    """The waiver decides whether a Bash-capable agent may hold a launch token, so
    the agent must not be able to set it through a config write."""
    assert is_protected_key("config.yml", "dangerously_allow_bash") is True


def test_dangerously_allow_bash_is_protected_below_itself_too():
    """The key is written with a trailing ``*`` so its subtree is protected as well.

    The value is a boolean, so there is nothing legitimate below it -- but the
    protected diff walks *leaves*, and a writer that planted a block where the
    boolean goes would move the only leaf to an unprotected child path. Spelled
    without the star the key would be one of the exact-depth patterns
    test_put_protected_families_are_descent_safe_or_known_inert exists to catch.
    """
    assert is_protected_key_path("config.yml", ("dangerously_allow_bash", "anything")) is True
