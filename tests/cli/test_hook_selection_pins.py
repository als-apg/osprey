"""Pin: which bundled presets gate writes, and which hook files each one renders.

A profile names only the hooks it wants. The helper files those hooks import or
read beside themselves in ``.claude/hooks/`` (the logger, the target-state
reader, the atomic file replace and ``hook_config.json``) ship as the closure of
that selection, so no preset names them. These pins hold each preset's resolved
write gates and the exact hook file set its render produces, so either one only
changes on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import _load_preset_raw, _resolve_extends, list_presets

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
