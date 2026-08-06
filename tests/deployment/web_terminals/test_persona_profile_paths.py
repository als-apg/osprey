"""Persona auto-render against a profile FILE (D7a), not just a bundled preset.

A catalog ``build_profile`` used to be a bundled preset name and nothing else.
Once ``osprey profile new`` emits sibling ``personas/<name>.yml`` profiles, the
same field may instead name a profile file — so :mod:`persona_images` decides
which it is from the STRING alone (a ``/`` or a ``.yml``/``.yaml`` suffix),
passes a file positionally to ``osprey build``, and resolves a relative path
against the profile directory the parent project was actually built from
(``profile_path_abs`` in its manifest), falling back to the project root and
then the current directory.

Preset names keep the argv they always had — pinned here beside the new form so
the two cannot drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from osprey.deployment.web_terminals import persona_images

_USERS = [{"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"}]


def _config(tmp_path: Path, build_profile: str | None) -> dict:
    """Local-mode config whose single persona 'ops' renders to <tmp_path>/ops-app."""
    return {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {
                        "project": "ops-app",
                        "project_path": str(tmp_path / "ops-app"),
                        "build_profile": build_profile,
                    }
                },
            }
        }
    }


def _profile_tree(root: Path, persona: str = "ops") -> Path:
    """A materialized-profile-shaped directory: profile.yml + personas/<p>.yml."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: Host\n", encoding="utf-8")
    (root / "personas").mkdir(exist_ok=True)
    (root / "personas" / f"{persona}.yml").write_text("name: Persona\n", encoding="utf-8")
    return root


def _write_manifest(project_root: Path, build_args: dict) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / ".osprey-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "build_args": build_args}), encoding="utf-8"
    )


@pytest.fixture
def calls(monkeypatch) -> list[list[str]]:
    recorded: list[list[str]] = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: recorded.append(cmd))
    return recorded


def _expect(project_path: Path, profile_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "osprey",
        "build",
        "ops-app",
        *profile_args,
        "-o",
        str(project_path.parent),
        "--skip-deps",
    ]


# ---------------------------------------------------------------------------
# String-only classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,is_file",
    [
        ("control-assistant", False),
        ("control_assistant", False),
        ("multi-user-demo-readonly", False),
        ("personas/readonly.yml", True),
        ("readonly.yml", True),
        ("readonly.yaml", True),
        ("../shared/profile.yml", True),
        ("/abs/profile.yml", True),
        ("some/dir/profile", True),  # a path, even without the suffix
    ],
)
def test_file_reference_is_decided_from_the_string(value: str, is_file: bool) -> None:
    """No filesystem probe: a typo'd path must never be silently retried as a
    preset name, and a preset name must never be probed as a path."""
    assert persona_images._is_profile_file_reference(value) is is_file


# ---------------------------------------------------------------------------
# Bundled preset names -- regression: argv unchanged
# ---------------------------------------------------------------------------


def test_preset_name_keeps_the_preset_argv(tmp_path: Path, calls: list) -> None:
    persona_images.auto_render_missing_personas(_config(tmp_path, "control-assistant"), _USERS, {})

    assert calls == [_expect(tmp_path / "ops-app", ["--preset", "control-assistant"])]


def test_preset_name_is_unaffected_by_a_profile_anchored_manifest(
    tmp_path: Path, calls: list
) -> None:
    """A parent built from a profile still renders preset-named personas by
    preset — the anchor only ever applies to a file-valued build_profile."""
    parent = tmp_path / "parent"
    _write_manifest(
        parent, {"profile_path_abs": str(_profile_tree(tmp_path / "prof") / "profile.yml")}
    )

    persona_images.auto_render_missing_personas(
        _config(tmp_path, "control-assistant"), _USERS, {}, project_root=parent
    )

    assert calls == [_expect(tmp_path / "ops-app", ["--preset", "control-assistant"])]


# ---------------------------------------------------------------------------
# File-valued build_profile -- positional, anchored on the parent's profile dir
# ---------------------------------------------------------------------------


def test_relative_path_resolves_against_the_manifest_profile_directory(
    tmp_path: Path, calls: list, monkeypatch
) -> None:
    """The anchor that matters: ``personas/ops.yml`` is relative to the profile
    the parent was built from, which is usually NOT the project root and never
    the cwd."""
    prof = _profile_tree(tmp_path / "prof")
    parent = tmp_path / "parent"
    _write_manifest(parent, {"profile_path_abs": str(prof / "profile.yml")})
    monkeypatch.chdir(tmp_path)

    persona_images.auto_render_missing_personas(
        _config(tmp_path, "personas/ops.yml"), _USERS, {}, project_root=parent
    )

    assert calls == [_expect(tmp_path / "ops-app", [str(prof / "personas" / "ops.yml")])]


def test_project_root_is_the_second_anchor(tmp_path: Path, calls: list, monkeypatch) -> None:
    """No profile_path_abs in the manifest (preset-built parent, or a legacy
    manifest) -> the project root."""
    parent = _profile_tree(tmp_path / "parent")
    _write_manifest(parent, {"preset": "control-assistant"})
    monkeypatch.chdir(tmp_path)

    persona_images.auto_render_missing_personas(
        _config(tmp_path, "personas/ops.yml"), _USERS, {}, project_root=parent
    )

    assert calls == [_expect(tmp_path / "ops-app", [str(parent / "personas" / "ops.yml")])]


def test_cwd_is_the_last_anchor(tmp_path: Path, calls: list, monkeypatch) -> None:
    here = _profile_tree(tmp_path / "here")
    monkeypatch.chdir(here)

    persona_images.auto_render_missing_personas(_config(tmp_path, "personas/ops.yml"), _USERS, {})

    assert calls == [_expect(tmp_path / "ops-app", [str(here / "personas" / "ops.yml")])]


def test_absolute_path_is_used_as_given(tmp_path: Path, calls: list) -> None:
    prof = _profile_tree(tmp_path / "prof")
    absolute = prof / "personas" / "ops.yml"

    persona_images.auto_render_missing_personas(_config(tmp_path, str(absolute)), _USERS, {})

    assert calls == [_expect(tmp_path / "ops-app", [str(absolute)])]


def test_forwarded_parent_overrides_still_trail_a_file_valued_render(
    tmp_path: Path, calls: list
) -> None:
    """Model-selection retinting is what keeps the stack coherent; it must
    survive the switch to file-valued profiles."""
    prof = _profile_tree(tmp_path / "prof")
    parent = tmp_path / "parent"
    _write_manifest(
        parent,
        {
            "profile_path_abs": str(prof / "profile.yml"),
            "provider": "cborg",
            "explicit_overrides": ["provider"],
        },
    )

    persona_images.auto_render_missing_personas(
        _config(tmp_path, "personas/ops.yml"), _USERS, {}, project_root=parent
    )

    assert calls == [
        [
            *_expect(tmp_path / "ops-app", [str(prof / "personas" / "ops.yml")]),
            "--set",
            "provider=cborg",
        ]
    ]


# ---------------------------------------------------------------------------
# The three-path error
# ---------------------------------------------------------------------------


def test_unresolvable_path_names_all_three_tried_locations(
    tmp_path: Path, calls: list, monkeypatch
) -> None:
    prof = tmp_path / "prof"
    prof.mkdir()
    (prof / "profile.yml").write_text("name: Host\n", encoding="utf-8")
    parent = tmp_path / "parent"
    _write_manifest(parent, {"profile_path_abs": str(prof / "profile.yml")})
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(
            _config(tmp_path, "personas/ops.yml"), _USERS, {}, project_root=parent
        )

    message = str(excinfo.value)
    assert "personas/ops.yml" in message
    assert str(prof) in message  # manifest-recorded profile directory
    assert str(parent) in message  # project root
    assert str(cwd) in message  # current directory
    assert "ops" in message  # the persona whose entry is wrong
    assert calls == []


def test_missing_absolute_path_still_errors_legibly(tmp_path: Path, calls: list) -> None:
    absolute = tmp_path / "nowhere" / "ops.yml"

    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(_config(tmp_path, str(absolute)), _USERS, {})

    assert str(absolute) in str(excinfo.value)
    assert calls == []


def test_missing_build_profile_still_raises_its_own_error(tmp_path: Path, calls: list) -> None:
    """The empty/absent case is a different mistake and keeps its own message."""
    with pytest.raises(ValueError, match="build_profile"):
        persona_images.auto_render_missing_personas(_config(tmp_path, None), _USERS, {})

    assert calls == []
