"""The ``hooks/`` profile convention channel.

Hooks are the one artifact class that is neither markdown nor a directory: they
are executable scripts whose *filename* is the name ``settings.json`` wires. The
tests below prove that giving them a channel buys the same uniform behavior
every other channel has — claim into the profile, shadow the framework render,
own the result, restore it by excluding it — using REAL framework hook names, so
a rename in the artifact catalog fails here rather than passing against a
synthetic fixture.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_persistence import (
    _apply_config_overrides,
    _apply_conventions,
    _register_convention_artifacts,
)
from osprey.cli.build_profile_merge import _apply_exclude, _deep_merge
from osprey.cli.scaffold_cmd import claim_into_profile
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.manifest import MANIFEST_FILENAME
from osprey.errors import BuildProfileError

#: A hook the framework really renders. Named once so the assertions below are
#: about the channel, not about a string repeated eight times.
FRAMEWORK_HOOK = "osprey_cf_feedback_capture.py"


def _write(path: Path, content: str = "# artifact\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _render_project(tmp_path: Path, name: str, *, force: bool = False) -> Path:
    """A freshly rendered project — what step 11 of a build lands conventions on."""
    return TemplateManager().create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
        force=force,
    )


def _point_manifest_at(project: Path, profile_file: Path) -> None:
    _write(
        project / MANIFEST_FILENAME,
        json.dumps(
            {
                "schema_version": "1.0",
                "build_args": {"profile_path_abs": str(profile_file)},
                "reproducible_command": "osprey build demo --preset hello-world",
            },
            indent=2,
        ),
    )


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    root = tmp_path / "my-profile"
    root.mkdir()
    _write(root / "profile.yml", "name: my-profile\n")
    return root


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    root = tmp_path / "my-project"
    root.mkdir()
    _write(root / "config.yml", yaml.safe_dump({"scaffold": {"user_owned": []}}))
    return root


def _user_owned(project_path: Path) -> list[str]:
    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    return config["scaffold"]["user_owned"]


def _build(
    tmp_path: Path,
    name: str,
    profile: Path,
    *,
    declarations: dict | None = None,
    persona_delta: dict | None = None,
    excluded: set[str] | None = None,
    rebuild: bool = False,
) -> Path:
    """The artifact-shaping steps of a real build, in the order ``build_cmd`` runs them.

    Step 8 applies the profile's ``config:`` overrides, step 11 lands the
    convention directories and registers what they own, step 16b re-renders the
    Claude Code artifacts from the ``config.yml`` those two produced. The
    re-render is what makes an assertion about ``settings.json`` mean anything
    here — conventions alone never touch that file, so a test that stopped after
    step 11 would be reading the pre-conventions render.

    ``persona_delta`` goes through the real ``_deep_merge``, so a test about
    persona overrides is a test about the merge semantics a persona actually
    gets — not about a dict the test assembled itself.
    """
    project = _render_project(tmp_path, name, force=rebuild)
    if persona_delta is not None:
        declarations = _deep_merge(declarations or {}, persona_delta)
    if declarations:
        _apply_config_overrides(project, declarations)
    applied = _apply_conventions(profile, project, excluded=excluded or set())
    _register_convention_artifacts(project, applied)
    TemplateManager().regenerate_claude_code(project)
    return project


def _settings(project: Path) -> dict:
    return json.loads((project / ".claude" / "settings.json").read_text(encoding="utf-8"))


def _hook_commands(settings: dict) -> list[str]:
    """Every hook command the rendered settings.json wires, across all events."""
    return [
        hook["command"]
        for entries in settings["hooks"].values()
        for entry in entries
        for hook in entry["hooks"]
    ]


# ── Claim: a framework hook moves into the profile's hooks/ ──────────


def test_framework_hook_claims_into_the_profile_hooks_dir(tmp_path: Path, profile_dir: Path):
    """The whole point of the channel: a rendered hook becomes profile material."""
    project = _render_project(tmp_path, "claim-hook")
    _point_manifest_at(project, profile_dir / "profile.yml")
    rendered = project / ".claude" / "hooks" / FRAMEWORK_HOOK
    assert rendered.is_file(), "the framework should render this hook"
    body = rendered.read_text(encoding="utf-8")

    outcome = claim_into_profile(project, f"hooks/{FRAMEWORK_HOOK}")

    assert outcome.profile_path == profile_dir / "hooks" / FRAMEWORK_HOOK
    assert outcome.profile_path.read_text(encoding="utf-8") == body
    # A claim MOVES: the project copy is gone until the next build puts it back.
    assert not rendered.exists()


def test_claimed_hook_shadows_the_framework_render_on_the_next_build(
    tmp_path: Path, profile_dir: Path, caplog: pytest.LogCaptureFixture
):
    """Claim, edit, rebuild — the profile's version is the one that lands."""
    first = _render_project(tmp_path, "claim-then-build")
    _point_manifest_at(first, profile_dir / "profile.yml")
    claim_into_profile(first, f"hooks/{FRAMEWORK_HOOK}")
    (profile_dir / "hooks" / FRAMEWORK_HOOK).write_text("# facility edit\n", encoding="utf-8")

    rebuilt = _render_project(tmp_path / "again", "claim-then-build")
    with caplog.at_level(logging.INFO):
        applied = _apply_conventions(profile_dir, rebuilt)
    registered = _register_convention_artifacts(rebuilt, applied)

    landed = rebuilt / ".claude" / "hooks" / FRAMEWORK_HOOK
    assert landed.read_text(encoding="utf-8") == "# facility edit\n"
    assert f"overrides framework hook 'hooks/{FRAMEWORK_HOOK}'" in caplog.text
    assert applied.owned_artifacts == [f"hooks/{FRAMEWORK_HOOK}"]
    assert registered == 1
    # Alongside whatever the render already registered for itself (rules/facility).
    assert f"hooks/{FRAMEWORK_HOOK}" in _user_owned(rebuilt)


# ── Ownership: naming, regen, prune ──────────────────────────────────


def test_hook_ownership_name_keeps_its_suffix(profile_dir: Path, project_path: Path):
    """`.md` is stripped; `.py` is not — the filename IS the hook's identity.

    ``settings.json`` wires each hook as ``.claude/hooks/<filename>``, so an
    ownership name without the suffix would name nothing the render can match.
    """
    _write(profile_dir / "hooks" / "facility_guard.py", "print('guard')\n")

    applied = _apply_conventions(profile_dir, project_path)

    assert applied.owned_artifacts == ["hooks/facility_guard.py"]
    assert claude_code.is_user_owned(
        ".claude/hooks/facility_guard.py", {"user_owned": ["hooks/facility_guard.py"]}
    )


def test_owned_hooks_survive_regen_and_prune(tmp_path: Path):
    """Both halves of the single-writer contract, on a real render.

    Regen must not rewrite the shadowed framework hook, and the prune pass —
    driven here by an empty allow-list, which unlinks everything unowned — must
    not unlink the custom one that no catalog artifact accounts for.
    """
    project = _render_project(tmp_path, "hook-regen")
    profile = tmp_path / "profile"
    _write(profile / "hooks" / FRAMEWORK_HOOK, "# facility edit\n")
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    applied = _apply_conventions(profile, project)
    _register_convention_artifacts(project, applied)

    hooks = project / ".claude" / "hooks"
    pruned_probe = hooks / "osprey_limits.py"
    assert pruned_probe.is_file(), "probe for the prune pass should exist pre-regen"

    manager = TemplateManager()
    config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project, config
    )
    claude_code.create_claude_code_integration(
        manager.template_root, manager.jinja_env, project, ctx, set()
    )

    assert not pruned_probe.exists(), "an empty manifest should prune unowned hooks"
    assert (hooks / FRAMEWORK_HOOK).read_text(encoding="utf-8") == "# facility edit\n"
    assert (hooks / "facility_guard.py").read_text(encoding="utf-8") == "print('guard')\n"


# ── The generated write-safety config is not the channel's to carry ──


def test_shipping_hook_config_is_refused_before_it_can_freeze_write_safety(
    tmp_path: Path, profile_dir: Path
):
    """The channel must not be a route to a frozen write gate.

    ``.claude/hooks/hook_config.json`` lands inside the hooks/ subtree but is
    generated from the enabled servers and ``control_system.write_tools``, and
    ``osprey_writes_check.py`` reads its ``write_tools`` to decide what counts
    as a hardware write. Carried through the channel it would be registered
    user-owned, so every later regen would skip it and an empty list would
    disarm the gate silently. Refused at the source, before anything is copied.
    """
    project = _render_project(tmp_path, "hook-config-freeze")
    rendered = project / ".claude" / "hooks" / "hook_config.json"
    armed = json.loads(rendered.read_text(encoding="utf-8"))["write_tools"]
    assert armed, "the render should arm the write gate with real tools"

    _write(profile_dir / "hooks" / "hook_config.json", '{"write_tools": []}\n')

    with pytest.raises(BuildProfileError) as excinfo:
        _apply_conventions(profile_dir, project)

    assert "hooks/hook_config.json" in str(excinfo.value)
    # Refused before copying: the project's armed config is untouched.
    assert json.loads(rendered.read_text(encoding="utf-8"))["write_tools"] == armed


# ── Declared wiring: claude_code.hooks ───────────────────────────────
#
# Shipping a hook copies it; DECLARING it is what makes Claude Code run it.
# The declaration is a config key, so it resolves through the same pipeline as
# `claude_code.servers` and `claude_code.permissions`: persona deltas can
# override it, and the resolved profile is the only source. Every assertion
# below reads the settings.json a real build lands — i.e. after step 16b's
# re-render, not the pre-conventions one.


def test_a_declared_hook_is_wired_into_the_settings_the_build_lands(tmp_path: Path):
    """The route the whole feature exists for, end to end."""
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    project = _build(
        tmp_path,
        "hook-wired",
        profile,
        declarations={
            "claude_code.hooks.PreToolUse": [
                {"hook": "facility_guard.py", "matcher": "mcp__controls__.*", "timeout": 7}
            ]
        },
    )

    entry = _settings(project)["hooks"]["PreToolUse"][-1]
    assert entry["matcher"] == "mcp__controls__.*"
    (hook,) = entry["hooks"]
    assert hook["timeout"] == 7
    assert hook["command"].endswith('"$CLAUDE_PROJECT_DIR/.claude/hooks/facility_guard.py"')
    # Rewritten to the project's interpreter like every framework hook command:
    # a literal `python3` is what goes dark on hosts that ship only a venv.
    assert not hook["command"].startswith("python3 ")
    # And no `|| true` — a facility gate that swallowed its own non-zero exit
    # would stop being a gate.
    assert "|| true" not in hook["command"]


def test_a_shipped_hook_without_a_declaration_is_carried_but_never_wired(tmp_path: Path):
    """Explicit declaration only — no auto-wiring by filename.

    The honest half of the README paragraph: an undeclared hook still lands,
    survives rebuilds, and runs never.
    """
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")
    _write(profile / "hooks" / "facility_notes.py", "print('notes')\n")

    project = _build(
        tmp_path,
        "hook-undeclared",
        profile,
        declarations={"claude_code.hooks.SessionStart": ["facility_guard.py"]},
    )

    assert (project / ".claude" / "hooks" / "facility_notes.py").is_file()
    commands = _hook_commands(_settings(project))
    assert not any("facility_notes.py" in c for c in commands)
    assert any("facility_guard.py" in c for c in commands)


def test_declared_wiring_is_additive_and_leaves_the_framework_wiring_alone(tmp_path: Path):
    """Constraint the write gate depends on: declarations ADD, they never displace.

    Compared against a build of the same profile with no declarations at all —
    drop the declared entries from the wired build and the two documents are
    identical, so nothing the framework emitted was replaced, reordered, or
    shadowed.
    """
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    plain = _settings(_build(tmp_path, "plain", profile))
    wired = _settings(
        _build(
            tmp_path,
            "wired",
            profile,
            declarations={
                "claude_code.hooks.PreToolUse": [{"hook": "facility_guard.py"}],
                "claude_code.hooks.PostToolUse": [{"hook": "facility_guard.py"}],
                "claude_code.hooks.SessionStart": ["facility_guard.py"],
                "claude_code.hooks.UserPromptSubmit": ["facility_guard.py"],
            },
        )
    )

    stripped = json.loads(json.dumps(wired))
    for event in ("PreToolUse", "PostToolUse", "SessionStart", "UserPromptSubmit"):
        assert "facility_guard.py" in json.dumps(stripped["hooks"][event][-1]), event
        stripped["hooks"][event] = stripped["hooks"][event][:-1]
    assert stripped == plain

    # Named explicitly so this stays a statement about the write gate and not
    # only about dict equality.
    assert "osprey_writes_check.py" in json.dumps(wired["hooks"]["PreToolUse"])


def test_a_regex_matcher_survives_the_render_as_valid_json(tmp_path: Path):
    """A matcher is a regex, and ``\\.`` is not a valid JSON escape.

    Interpolated between hand-written quotes it would yield a settings.json
    Claude Code cannot parse — taking every hook in the file down with it, the
    framework's write gate included.
    """
    matcher = r"mcp__controls__channel_write\.[a-z]+"
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    project = _build(
        tmp_path,
        "hook-regex",
        profile,
        declarations={
            "claude_code.hooks.PreToolUse": [{"hook": "facility_guard.py", "matcher": matcher}]
        },
    )

    assert _settings(project)["hooks"]["PreToolUse"][-1]["matcher"] == matcher


def test_an_event_the_framework_wires_nothing_on_becomes_its_own_key(tmp_path: Path):
    """Declaring on Stop adds the key; a build that declares nothing has none."""
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    plain = _settings(_build(tmp_path, "no-stop", profile))
    assert "Stop" not in plain["hooks"]

    wired = _settings(
        _build(
            tmp_path,
            "with-stop",
            profile,
            declarations={"claude_code.hooks.Stop": ["hooks/facility_guard.py"]},
        )
    )
    (entry,) = wired["hooks"]["Stop"]
    assert "facility_guard.py" in entry["hooks"][0]["command"]
    # No matcher: Stop takes none, and an empty one would match nothing.
    assert "matcher" not in entry


def test_declared_wiring_survives_a_force_rebuild_and_a_regen(tmp_path: Path):
    """Rebuilt from the resolved profile every time, never frozen in the project."""
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")
    declarations = {"claude_code.hooks.PreToolUse": [{"hook": "facility_guard.py"}]}

    project = _build(tmp_path, "hook-regen-wiring", profile, declarations=declarations)
    assert any("facility_guard.py" in c for c in _hook_commands(_settings(project)))

    # `osprey build --force`: the project directory is re-rendered from scratch,
    # so the wiring has to come back from the profile, not from what survived.
    _build(tmp_path, "hook-regen-wiring", profile, declarations=declarations, rebuild=True)
    assert any("facility_guard.py" in c for c in _hook_commands(_settings(project)))

    # `osprey claude regen`: same again, from config.yml alone.
    TemplateManager().regenerate_claude_code(project)
    assert any("facility_guard.py" in c for c in _hook_commands(_settings(project)))


# ── Declared wiring: how a persona unwires it ────────────────────────
#
# A persona's `config:` merges through `_deep_merge`, where lists are UNIONED
# with the profile's rather than replacing them. That makes `null` — not `[]` —
# the spelling that unwires an event, and these tests are what keeps the README
# from drifting back to the one that silently does nothing.

#: What the root profile declares in the three persona tests below: one hook on
#: an event the persona will drop, one on an event it must leave alone.
_TWO_EVENT_DECLARATION = {
    "claude_code.hooks.PreToolUse": [{"hook": "facility_guard.py"}],
    "claude_code.hooks.SessionStart": ["facility_banner.py"],
}


def _two_hook_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")
    _write(profile / "hooks" / "facility_banner.py", "print('banner')\n")
    return profile


def test_a_persona_excluding_a_hook_unwires_it_with_null_in_the_same_delta(tmp_path: Path):
    """The one combination a persona author has to get right, start to finish.

    Excluding a shipped hook without unwiring it is refused (the wiring would
    point at a file the resolved profile no longer ships), so the exclusion and
    the ``null`` have to travel together. That pair must build.
    """
    profile = _two_hook_profile(tmp_path)

    project = _build(
        tmp_path,
        "persona-unwire",
        profile,
        declarations=_TWO_EVENT_DECLARATION,
        persona_delta={"claude_code.hooks.PreToolUse": None},
        excluded={"hooks/facility_guard.py"},
    )

    settings = _settings(project)
    commands = _hook_commands(settings)
    assert not any("facility_guard.py" in c for c in commands)
    assert not (project / ".claude" / "hooks" / "facility_guard.py").exists()
    # The event the persona said nothing about is untouched…
    assert any("facility_banner.py" in c for c in commands)
    # …and so is the framework's gate.
    assert "osprey_writes_check.py" in json.dumps(settings["hooks"]["PreToolUse"])


def test_an_empty_list_does_not_unwire_because_persona_lists_merge_additively(tmp_path: Path):
    """`[]` is a no-op, and that is a property of ``_deep_merge``, not of hooks.

    Pinned because the README used to promise the opposite. A persona author
    who writes ``[]`` gets a hook that is still wired — silently — so if list
    merging ever changes to replacement, this failure is the notice to rewrite
    the paragraph rather than discover it in a control room.
    """
    profile = _two_hook_profile(tmp_path)

    project = _build(
        tmp_path,
        "persona-empty-list",
        profile,
        declarations=_TWO_EVENT_DECLARATION,
        persona_delta={"claude_code.hooks.PreToolUse": []},
    )

    assert any("facility_guard.py" in c for c in _hook_commands(_settings(project)))


def test_the_empty_list_mistake_is_refused_with_the_spelling_that_works(tmp_path: Path):
    """Excluding + ``[]`` is the trap; the refusal has to hand over the way out."""
    profile = _two_hook_profile(tmp_path)

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "persona-empty-list-excluded",
            profile,
            declarations=_TWO_EVENT_DECLARATION,
            persona_delta={"claude_code.hooks.PreToolUse": []},
            excluded={"hooks/facility_guard.py"},
        )

    message = str(excinfo.value)
    assert "claude_code.hooks.PreToolUse: null" in message
    assert "`null`, not `[]`" in message


def test_a_persona_can_unwire_every_event_at_once(tmp_path: Path):
    """`claude_code.hooks: {}` clears the whole block, framework wiring intact.

    The `{}` form works only because ``_deep_merge`` (build_profile_merge.py)
    emits base keys first and child-only keys last, so the whole-key override
    is applied after the per-event ones. If this test goes red on a merge
    change, that key-ordering guarantee is what broke.
    """
    profile = _two_hook_profile(tmp_path)

    project = _build(
        tmp_path,
        "persona-unwire-all",
        profile,
        declarations=_TWO_EVENT_DECLARATION,
        persona_delta={"claude_code.hooks": {}},
    )

    settings = _settings(project)
    commands = _hook_commands(settings)
    assert not any("facility_guard.py" in c for c in commands)
    assert not any("facility_banner.py" in c for c in commands)
    assert "osprey_writes_check.py" in json.dumps(settings["hooks"]["PreToolUse"])
    # Unwired, not unshipped — both files are still in the project.
    assert (project / ".claude" / "hooks" / "facility_guard.py").is_file()


# ── Declared wiring: what it refuses ─────────────────────────────────


def test_wiring_a_hook_the_profile_does_not_ship_is_refused(tmp_path: Path):
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "hook-missing",
            profile,
            declarations={"claude_code.hooks.PreToolUse": ["nowhere.py"]},
        )

    assert "nowhere.py" in str(excinfo.value)
    assert "does not ship" in str(excinfo.value)


def test_wiring_a_persona_excluded_hook_is_refused(tmp_path: Path):
    """The exclusion is what makes this the interesting case.

    A persona that drops the hook leaves the declaration behind pointing at a
    file the resolved profile no longer ships. Refused rather than dropped
    quietly: silently unwiring a facility's own guard is the failure mode the
    operator would never notice.
    """
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "hook-excluded",
            profile,
            declarations={"claude_code.hooks.PreToolUse": ["facility_guard.py"]},
            excluded={"hooks/facility_guard.py"},
        )

    assert "facility_guard.py" in str(excinfo.value)
    assert "excludes" in str(excinfo.value)


def test_wiring_the_generated_write_safety_config_is_refused(tmp_path: Path):
    """``hook_config.json`` is reserved to the framework render, wiring included."""
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "hook-reserved",
            profile,
            declarations={"claude_code.hooks.PreToolUse": ["hook_config.json"]},
        )

    assert ".claude/hooks/hook_config.json" in str(excinfo.value)
    assert "osprey_writes_check.py" in str(excinfo.value)


@pytest.mark.parametrize(
    "reference",
    [
        "../evil.py",
        "hooks/../../evil.py",
        "/etc/passwd",
        "~/evil.py",
        "rules/safety.md",
        "nested/dir/guard.py",
    ],
)
def test_wiring_a_path_outside_the_hooks_channel_is_refused(tmp_path: Path, reference: str):
    """Only ``guard.py`` and ``hooks/guard.py`` address a hook — nothing else."""
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "hook-outside",
            profile,
            declarations={"claude_code.hooks.PreToolUse": [reference]},
        )

    assert "hooks/ channel" in str(excinfo.value)


def test_wiring_a_built_in_hook_is_refused(tmp_path: Path):
    """The framework owns its own hooks' wiring — declaring one would double it.

    A claimed built-in hook is the realistic way to hit this: the profile really
    does ship the file, but ``settings.json`` already wires that filename from
    the ``hooks:`` selection.
    """
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "osprey_limits.py", "# facility edit\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "hook-builtin",
            profile,
            declarations={"claude_code.hooks.PreToolUse": ["osprey_limits.py"]},
        )

    assert "osprey_limits.py" in str(excinfo.value)
    assert "built-in" in str(excinfo.value)


def test_an_unknown_hook_event_is_refused_with_the_real_vocabulary(tmp_path: Path):
    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    with pytest.raises(BuildProfileError) as excinfo:
        _build(
            tmp_path,
            "hook-bad-event",
            profile,
            declarations={"claude_code.hooks.OnTuesday": ["facility_guard.py"]},
        )

    message = str(excinfo.value)
    assert "OnTuesday" in message
    for event in ("PreToolUse", "PostToolUse", "UserPromptSubmit", "SessionStart", "Stop"):
        assert event in message


# ── Persona exclusion restores the framework hook ────────────────────


def test_excluding_a_shadowed_hook_restores_the_framework_render(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A persona that wants the stock hook back excludes the profile's file."""
    project = _render_project(tmp_path, "hook-exclude")
    profile = tmp_path / "profile"
    _write(profile / "hooks" / FRAMEWORK_HOOK, "# facility edit\n")
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")
    framework_body = (project / ".claude" / "hooks" / FRAMEWORK_HOOK).read_text(encoding="utf-8")

    with caplog.at_level(logging.INFO):
        applied = _apply_conventions(profile, project, excluded={f"hooks/{FRAMEWORK_HOOK}"})

    landed = project / ".claude" / "hooks" / FRAMEWORK_HOOK
    assert landed.read_text(encoding="utf-8") == framework_body
    assert "overrides framework" not in caplog.text
    # Post-exclude: not applied, so not owned — the framework keeps it.
    assert applied.owned_artifacts == ["hooks/facility_guard.py"]


def test_exclude_spellings_split_the_library_selection_from_the_shipped_file():
    """The two hook namespaces are disjoint, and each spelling reaches one.

    ``cf-feedback-capture`` is the built-in library's readable short name — a
    bare entry unselects it. ``hooks/osprey_cf_feedback_capture.py`` is the
    filename a profile ships, so the qualified entry omits that file and leaves
    the still-selected built-in version to render.
    """
    merged = {"hooks": ["cf-feedback-capture", "limits"]}
    artifacts: set[str] = set()
    shadow_candidates: set[str] = set()

    _apply_exclude(
        merged,
        {"hooks": ["cf-feedback-capture", f"hooks/{FRAMEWORK_HOOK}"]},
        artifacts=artifacts,
        shadow_candidates=shadow_candidates,
    )

    assert merged["hooks"] == ["limits"]
    assert artifacts == {f"hooks/{FRAMEWORK_HOOK}"}
    assert shadow_candidates == {"hooks/cf-feedback-capture"}


def test_exclude_rejects_a_hook_entry_qualified_with_another_channel():
    """`hooks: [rules/safety]` is a mistake, not a hook literally named that."""
    from osprey.errors import BuildProfileError

    with pytest.raises(BuildProfileError, match="rules"):
        _apply_exclude({"hooks": []}, {"hooks": ["rules/safety"]})


def test_bare_exclusion_of_a_shadowed_hook_is_diagnosed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Short name in, filename on disk — only the library's own mapping joins them.

    Without that mapping the warning would stay silent on the one exclusion
    mistake nothing else can see: the built-in hook is unselected while the
    profile's file of the same hook keeps rendering.
    """
    from osprey.cli.build_profile_merge import _warn_shadowed_bare_exclusions

    profile = tmp_path / "profile"
    _write(profile / "hooks" / FRAMEWORK_HOOK, "# facility edit\n")

    with caplog.at_level(logging.WARNING):
        _warn_shadowed_bare_exclusions(profile, {"hooks/cf-feedback-capture"})

    assert "cf-feedback-capture" in caplog.text
    assert "no effect" in caplog.text


def test_bare_exclusion_without_a_shipped_file_is_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    from osprey.cli.build_profile_merge import _warn_shadowed_bare_exclusions

    profile = tmp_path / "profile"
    _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    with caplog.at_level(logging.WARNING):
        _warn_shadowed_bare_exclusions(profile, {"hooks/cf-feedback-capture"})

    assert caplog.text == ""


# ── Profile material hash ────────────────────────────────────────────


def test_editing_a_hook_moves_the_profile_hash(tmp_path: Path):
    """Hooks are folded like every other channel — the table drives it."""
    import hashlib

    from osprey.cli.build_profile_merge import _fold_profile_material

    profile = tmp_path / "profile"
    hook = _write(profile / "hooks" / "facility_guard.py", "print('guard')\n")

    def digest() -> str:
        h = hashlib.sha256()
        _fold_profile_material(h, {}, profile)
        return h.hexdigest()

    before = digest()
    hook.write_text("print('guard v2')\n", encoding="utf-8")
    assert digest() != before
