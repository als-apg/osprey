"""Pin: a profile that selects a hook must also select the helpers it imports.

Two helpers are imported by hooks and wired to no event: ``target-state`` (the
write gates' posture source) and ``safety-file`` (the atomic file replace every
hook that writes a file uses). ``HOOK_HELPER_IMPORTERS`` names, per helper, the
hooks that import it; every shipped preset and the exemplar golden are held to
that one table, and a facility profile that breaks it gets one build-time
warning per missing helper.

``writes-check`` and ``approval`` decide whether a write may proceed by reading
the control target's write posture, and they read it by importing the
``target-state`` helper from beside them in ``.claude/hooks/``. That helper is
wired to no event, so nothing but SELECTION puts it there: a profile whose
``hooks:`` list names a gate and not the helper builds cleanly, deploys
cleanly, and then refuses every write, because a gate with no posture source
falls back to the most restrictive answer.

Fail-closed is the right default and not the right outcome here — the
deployment was configured to allow those writes. So the pairing is pinned two
ways: every preset OSPREY ships is checked here against its RESOLVED hook list
(what a facility actually gets, after ``extends`` and ``exclude``), and a
facility profile that breaks the pairing gets a build-time warning naming the
line to add. The warning does not refuse the build; the selection lists are the
author's to compose.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile import _load_preset_raw, _resolve_extends, list_presets
from osprey.cli.build_profile_merge import (
    HOOK_HELPER_IMPORTERS,
    _warn_missing_hook_helpers,
    resolve_profile_document,
)

#: The helper the gates import, spelled as a profile selects it.
TARGET_STATE = "target-state"

#: The helper the file-writing hooks import, spelled as a profile selects it.
SAFETY_FILE = "safety-file"

#: The hooks that read write posture, spelled as a profile selects them.
WRITE_GATES = ("approval", "writes-check")

#: preset name -> the write gates its RESOLVED hook list carries, sorted.
#:
#: Resolved rather than raw, deliberately: what matters is what a facility ends
#: up running. ``control-assistant-logbook`` is the entry that proves the
#: difference — it inherits ``writes-check`` from ``control-assistant`` and
#: excludes it again, so its raw file names neither gate and its resolved list
#: still names one.
PINNED_PRESET_WRITE_GATES: dict[str, tuple[str, ...]] = {
    "ariel-standalone": ("approval",),
    "channel-finder-standalone": ("approval",),
    "control-assistant": ("approval", "writes-check"),
    "control-assistant-admin": ("approval", "writes-check"),
    "control-assistant-knowledge": ("approval",),
    "control-assistant-logbook": ("approval",),
    "control-assistant-readonly": ("approval", "writes-check"),
    "control-assistant-readwrite": ("approval", "writes-check"),
    "hello-world": ("approval", "writes-check"),
}

#: The golden a deployment is scaffolded against — a facility profile in every
#: respect except that it lives in the test tree, so it is held to the same rule.
EXEMPLAR_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "goldens"
    / "exemplar-profile"
    / "profile.yml"
)


def _resolved_hooks(preset: str) -> list[str]:
    """The hook selections a facility gets from ``preset``, extends applied."""
    raw, path = _load_preset_raw(preset)
    hooks = _resolve_extends(raw, path).get("hooks") or []
    return [entry for entry in hooks if isinstance(entry, str)]


def _write_profile(directory: Path, hooks: list[str]) -> tuple[dict, Path]:
    """A minimal profile mapping selecting ``hooks``, and where it lives."""
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / "profile.yml"
    raw = {"name": "posture-fixture", "hooks": hooks}
    profile_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return raw, profile_path


# ── The pin ──────────────────────────────────────────────────────────


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the comparison below would silently skip an unpinned preset,
    and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_WRITE_GATES)


def test_every_bundled_preset_selects_the_write_gates_it_is_pinned_to():
    """Which presets gate writes is a decision, so it is written down."""
    actual = {
        name: tuple(sorted(gate for gate in WRITE_GATES if gate in _resolved_hooks(name)))
        for name in list_presets()
    }
    assert actual == PINNED_PRESET_WRITE_GATES


def test_every_bundled_preset_that_gates_writes_also_selects_target_state():
    """The pairing itself, over what a facility actually receives."""
    offenders = sorted(
        name
        for name in list_presets()
        if any(gate in _resolved_hooks(name) for gate in WRITE_GATES)
        and TARGET_STATE not in _resolved_hooks(name)
    )
    assert not offenders, (
        f"{offenders} select a write gate without {TARGET_STATE!r}; those gates "
        "would resolve write posture to the most restrictive answer"
    )


def test_every_bundled_preset_selects_every_helper_its_hooks_import():
    """The whole helper table, over what a facility actually receives."""
    offenders = sorted(
        (name, helper)
        for name in list_presets()
        for helper, importers in HOOK_HELPER_IMPORTERS.items()
        if any(hook in _resolved_hooks(name) for hook in importers)
        and helper not in _resolved_hooks(name)
    )
    assert not offenders, f"(preset, missing helper) pairs: {offenders}"


def test_the_exemplar_golden_profile_selects_every_helper_its_hooks_import():
    hooks = yaml.safe_load(EXEMPLAR_PROFILE.read_text(encoding="utf-8"))["hooks"]

    missing = sorted(
        helper
        for helper, importers in HOOK_HELPER_IMPORTERS.items()
        if any(hook in hooks for hook in importers) and helper not in hooks
    )
    assert not missing


def test_the_exemplar_golden_profile_pairs_its_write_gates_with_target_state():
    """The profile every scaffolded deployment is modelled on obeys the rule."""
    hooks = yaml.safe_load(EXEMPLAR_PROFILE.read_text(encoding="utf-8"))["hooks"]

    assert any(gate in hooks for gate in WRITE_GATES)
    assert TARGET_STATE in hooks


# ── The build-time warning ───────────────────────────────────────────


def test_a_profile_selecting_a_write_gate_without_target_state_is_diagnosed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """The warning has to name the fix, because the build otherwise succeeds."""
    # Arrange
    raw, profile_path = _write_profile(tmp_path / "profile", ["writes-check", "hook-log"])

    # Act
    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        resolve_profile_document(raw, profile_path)

    # Assert
    assert TARGET_STATE in caplog.text
    assert "writes-check" in caplog.text
    assert "most restrictive" in caplog.text


def test_the_approval_hook_alone_also_triggers_the_diagnosis(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Both gates read posture, so either one on its own is the same mistake."""
    raw, profile_path = _write_profile(tmp_path / "profile", ["approval"])

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        resolve_profile_document(raw, profile_path)

    assert "approval" in caplog.text
    assert TARGET_STATE in caplog.text


@pytest.mark.parametrize(
    "hook",
    ["approval", "control-context", "panels-context", "workspace-delta", "cf-feedback-capture"],
)
def test_a_file_writing_hook_without_safety_file_is_diagnosed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, hook: str
):
    """A writer hook without the helper writes nothing, and the build says so."""
    raw, profile_path = _write_profile(tmp_path / "profile", [hook, TARGET_STATE])

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        resolve_profile_document(raw, profile_path)

    assert repr(SAFETY_FILE) in caplog.text
    assert hook in caplog.text
    assert "write nothing" in caplog.text


def test_each_missing_helper_is_warned_once(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """The approval hook imports both helpers; missing both is two warnings."""
    raw, profile_path = _write_profile(tmp_path / "profile", ["approval", "control-context"])

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        resolve_profile_document(raw, profile_path)

    warned = sorted(
        helper
        for record in caplog.records
        for helper in HOOK_HELPER_IMPORTERS
        if f"but not {helper!r}" in record.getMessage()
    )
    assert warned == [SAFETY_FILE, TARGET_STATE]
    assert len(caplog.records) == 2


def test_a_profile_pairing_the_gate_with_target_state_is_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    raw, profile_path = _write_profile(
        tmp_path / "profile", ["writes-check", "approval", TARGET_STATE, SAFETY_FILE]
    )

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        resolve_profile_document(raw, profile_path)

    assert caplog.text == ""


def test_a_profile_selecting_neither_gate_is_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Nothing imports the helper, so its absence costs nothing."""
    raw, profile_path = _write_profile(tmp_path / "profile", ["hook-log", "error-guidance"])

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        resolve_profile_document(raw, profile_path)

    assert caplog.text == ""


def test_a_profile_shipping_its_own_posture_helper_is_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A profile's own ``hooks/`` file renders whether or not it is selected.

    The import the gates make resolves against the rendered directory, not
    against the selection list, so the file being there is the whole condition.
    """
    profile = tmp_path / "profile"
    (profile / "hooks").mkdir(parents=True)
    (profile / "hooks" / "osprey_target_state.py").write_text("# facility edit\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        _warn_missing_hook_helpers(profile, {"hooks": ["writes-check"]})

    assert caplog.text == ""


def test_a_profile_with_no_hooks_key_is_quiet(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """No selection list, no gate to be missing a helper."""
    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_profile_merge"):
        _warn_missing_hook_helpers(tmp_path, {})

    assert caplog.text == ""


# ── The rendered hook files ──────────────────────────────────────────

#: The hook files every bundled preset renders.
_COMMON_HOOK_FILES = frozenset(
    {
        "hook_config.json",
        "osprey_approval.py",
        "osprey_config_drift.py",
        "osprey_control_context.py",
        "osprey_error_guidance.py",
        "osprey_focus_validate.py",
        "osprey_hook_log.py",
        "osprey_memory_guard.py",
        "osprey_panels_context.py",
        "osprey_safety_file.py",
        "osprey_target_state.py",
        "osprey_turn_state.py",
        "osprey_workspace_delta.py",
    }
)

#: The hook files the full control-assistant tier renders.
_CONTROL_ASSISTANT_HOOK_FILES = _COMMON_HOOK_FILES | {
    "osprey_cf_feedback_capture.py",
    "osprey_limits.py",
    "osprey_notebook_update.py",
    "osprey_writes_check.py",
}

#: preset name -> the exact file set its render puts in ``.claude/hooks/``.
PINNED_PRESET_HOOK_FILES: dict[str, frozenset[str]] = {
    "ariel-standalone": _COMMON_HOOK_FILES | {"osprey_notebook_update.py"},
    "channel-finder-standalone": _COMMON_HOOK_FILES
    | {"osprey_cf_feedback_capture.py", "osprey_notebook_update.py"},
    "control-assistant": _CONTROL_ASSISTANT_HOOK_FILES,
    "control-assistant-admin": _CONTROL_ASSISTANT_HOOK_FILES,
    "control-assistant-knowledge": _COMMON_HOOK_FILES
    | {"osprey_cf_feedback_capture.py", "osprey_notebook_update.py"},
    "control-assistant-logbook": _COMMON_HOOK_FILES | {"osprey_notebook_update.py"},
    "control-assistant-readonly": _CONTROL_ASSISTANT_HOOK_FILES,
    "control-assistant-readwrite": _CONTROL_ASSISTANT_HOOK_FILES,
    "hello-world": _COMMON_HOOK_FILES | {"osprey_limits.py", "osprey_writes_check.py"},
}


@pytest.mark.parametrize("preset", list_presets())
def test_every_bundled_preset_renders_its_pinned_hook_files(preset: str, tmp_path: Path):
    """An equality check over the render, so a hook file only moves on purpose."""
    from osprey.cli.build_profile_presets import _nearest_preset_value
    from osprey.cli.templates.manager import TemplateManager
    from osprey.cli.templates.manifest import load_template_manifest
    from tests._preset_data import bundle_data_root

    bundle = _nearest_preset_value(preset, "app_template")
    assert bundle is not None
    selection = load_template_manifest(preset)
    assert selection is not None

    project = TemplateManager().create_project(
        project_name="hook-files",
        output_dir=tmp_path,
        data_bundle=bundle,
        context={"channel_finder_mode": "hierarchical"},
        artifacts=selection["artifacts"],
        data_root=bundle_data_root(bundle),
    )

    hooks_dir = project / ".claude" / "hooks"
    rendered = {path.name for path in hooks_dir.iterdir() if path.is_file()}
    assert rendered == PINNED_PRESET_HOOK_FILES[preset]
