"""Pin: which bundled presets gate writes, and which hook files each one renders.

A profile names only the hooks it wants. The helper files those hooks import or
read beside themselves in ``.claude/hooks/`` (the logger, the target-state
reader, the atomic file replace and ``hook_config.json``) ship as the closure of
that selection, so no preset names them. These pins hold the resolved write
gates of every bundled preset, tier and app, and the exact hook file set each
one renders, so either only changes on purpose. A bundled tier or app is
checked as the render it makes inside the root preset that scaffolds it: a tier
is merged over that root, and an app is composed from it, gaining
``writes-check`` whatever its own file selects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import _load_preset_raw, _resolve_extends, list_presets
from osprey.cli.build_profile_merge import render_document
from osprey.cli.build_profile_presets import list_apps, list_tiers
from osprey.cli.profile_root import ProfileKind

#: The hooks that read write posture, spelled as a profile selects them.
WRITE_GATES = ("approval", "writes-check")

#: The root preset whose ``osprey init`` scaffolds the bundled tiers and apps.
LAYER_HOST = "control-assistant"

#: Layer directory -> the kind its files render as.
LAYER_KINDS = {"tiers": ProfileKind.TIER, "apps": ProfileKind.APP}

#: preset, tier or app name -> the write gates its RESOLVED hook list carries,
#: sorted.
#:
#: Resolved rather than raw, deliberately: what matters is what a facility ends
#: up running. A preset that ``extends`` another inherits its hook list, and a
#: tier inherits its root's, so a raw file can name neither gate while its
#: resolved list names both.
PINNED_PRESET_WRITE_GATES: dict[str, tuple[str, ...]] = {
    "control-assistant": ("approval", "writes-check"),
    "hello-world": ("approval", "writes-check"),
    "tiers/admin": ("approval", "writes-check"),
    "tiers/readonly": ("approval", "writes-check"),
    "tiers/readwrite": ("approval", "writes-check"),
    "apps/channel-finder": ("approval", "writes-check"),
    "apps/knowledge": ("approval", "writes-check"),
    "apps/logbook": ("approval", "writes-check"),
}


def _bundled_names() -> list[str]:
    """Every bundled root preset, then every tier and app as ``<kind>/<stem>``."""
    return (
        list_presets()
        + [f"tiers/{name}" for name in list_tiers()]
        + [f"apps/{name}" for name in list_apps()]
    )


def _resolved_document(preset: str) -> dict:
    """The profile document a facility gets from ``preset``.

    A root preset with ``extends`` applied; a tier or app as the render it
    makes inside :data:`LAYER_HOST`.
    """
    raw, path = _load_preset_raw(preset)
    directory, _, _stem = preset.partition("/")
    if directory in LAYER_KINDS:
        host_raw, host_path = _load_preset_raw(LAYER_HOST)
        host = _resolve_extends(host_raw, host_path)
        return render_document(LAYER_KINDS[directory], host, raw, f"{preset}.yml")
    return _resolve_extends(raw, path)


def _resolved_hooks(preset: str) -> list[str]:
    """The hook selections a facility gets from ``preset``."""
    hooks = _resolved_document(preset).get("hooks") or []
    return [entry for entry in hooks if isinstance(entry, str)]


# ── The pin ──────────────────────────────────────────────────────────


def test_bundled_preset_set_is_pinned():
    """A new preset, tier or app must be classified here before it ships.

    Without this the comparison below would silently skip an unpinned one, and
    the pin would degrade as presets, tiers and apps are added.
    """
    assert sorted(_bundled_names()) == sorted(PINNED_PRESET_WRITE_GATES)


def test_every_bundled_preset_selects_the_write_gates_it_is_pinned_to():
    """Which presets, tiers and apps gate writes is a decision, so it is written down."""
    actual = {
        name: tuple(sorted(gate for gate in WRITE_GATES if gate in _resolved_hooks(name)))
        for name in _bundled_names()
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
        "osprey_prompt_render.py",
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

#: preset, tier or app name -> the exact file set its render puts in
#: ``.claude/hooks/``.
PINNED_PRESET_HOOK_FILES: dict[str, frozenset[str]] = {
    "control-assistant": _CONTROL_ASSISTANT_HOOK_FILES,
    "hello-world": _COMMON_HOOK_FILES | {"osprey_limits.py", "osprey_writes_check.py"},
    "tiers/admin": _CONTROL_ASSISTANT_HOOK_FILES,
    "tiers/readonly": _CONTROL_ASSISTANT_HOOK_FILES,
    "tiers/readwrite": _CONTROL_ASSISTANT_HOOK_FILES,
    # An app carries every guard its host selects (``limits`` among them),
    # whatever its own list says.
    "apps/channel-finder": _CONTROL_ASSISTANT_HOOK_FILES,
    "apps/knowledge": _CONTROL_ASSISTANT_HOOK_FILES,
    "apps/logbook": _CONTROL_ASSISTANT_HOOK_FILES - {"osprey_cf_feedback_capture.py"},
}


def test_hook_file_pins_cover_every_bundled_preset():
    """A new preset, tier or app gets its rendered hook files pinned before it ships."""
    assert sorted(_bundled_names()) == sorted(PINNED_PRESET_HOOK_FILES)


@pytest.mark.parametrize("preset", _bundled_names())
def test_every_bundled_preset_renders_its_pinned_hook_files(preset: str, tmp_path: Path):
    """An equality check over the render, so a hook file only moves on purpose."""
    from osprey.cli.build_profile_presets import _nearest_preset_value
    from osprey.cli.templates.manager import TemplateManager
    from tests._preset_data import bundle_data_root

    artifact_keys = ("hooks", "rules", "skills", "agents", "output_styles", "web_panels")
    document = _resolved_document(preset)
    artifacts = {key: document[key] for key in artifact_keys if key in document}
    root = LAYER_HOST if preset.partition("/")[0] in LAYER_KINDS else preset
    bundle = _nearest_preset_value(root, "app_template")
    assert bundle is not None

    project = TemplateManager().create_project(
        project_name="hook-files",
        output_dir=tmp_path,
        data_bundle=bundle,
        context={"channel_finder_mode": "hierarchical"},
        artifacts=artifacts,
        data_root=bundle_data_root(bundle),
    )

    hooks_dir = project / ".claude" / "hooks"
    rendered = {path.name for path in hooks_dir.iterdir() if path.is_file()}
    assert rendered == PINNED_PRESET_HOOK_FILES[preset]
