"""Unit tests for per-persona local image builds.

Covers ``osprey.deployment.web_terminals.persona_images`` in isolation: the
local-mode per-persona image builder and the on-demand persona project
auto-render.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.deployment.web_terminals import persona_images

# ---------------------------------------------------------------------------
# build_persona_images -- local-mode per-persona image builder
# ---------------------------------------------------------------------------


def _make_persona_project(tmp_path, name, cli_version=None):
    """Create a minimal persona project dir with a Dockerfile + config.yml."""
    project_dir = tmp_path / name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    if cli_version is not None:
        (project_dir / "config.yml").write_text(
            f"claude_code:\n  cli_version: {cli_version!r}\n", encoding="utf-8"
        )
    else:
        (project_dir / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    return str(project_dir)


@pytest.fixture
def _no_dev_wheel_staging(monkeypatch):
    """Stub out the dev-wheel staging collaborator (its own coverage lives with
    _build_project_image's tests) so build_persona_images tests never touch a
    real wheel build. Reports SUCCESS (True) — the OSPREY_DEV build-arg is
    keyed on staging success, so simulating a successful staging keeps the
    dev-path assertions meaningful; the failure path has its own test."""
    monkeypatch.setattr(
        persona_images, "_copy_local_framework_for_override", lambda project_root: True
    )


def test_build_persona_images_noop_in_registry_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda *a, **k: calls.append(a))
    config = {"modules": {"web_terminals": {"image_source": "registry"}}}

    persona_images.build_persona_images(config, [{"persona": "ops"}], False, {})

    assert calls == []


def test_build_persona_images_local_without_catalog_raises(tmp_path):
    config = {"modules": {"web_terminals": {"image_source": "local"}}}

    with pytest.raises(ValueError, match="requires both"):
        persona_images.build_persona_images(config, [], False, {})


def test_build_persona_images_local_without_default_persona_raises(tmp_path):
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "personas": {"ops": {"project": "ops-app", "project_path": str(tmp_path)}},
            }
        }
    }

    with pytest.raises(ValueError, match="requires both"):
        persona_images.build_persona_images(config, [], False, {})


def test_build_persona_images_builds_each_referenced_persona_once(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    sci_path = _make_persona_project(tmp_path, "sci-app")

    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {"project": "ops-app", "project_path": ops_path},
                    "sci": {"project": "sci-app", "project_path": sci_path},
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "persona": "ops", "project": "ops-app"},
        {"name": "bob", "persona": "ops", "project": "ops-app"},  # shares ops -- must not rebuild
        {"name": "carol", "persona": "sci", "project": "sci-app"},
    ]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    assert len(calls) == 2  # one build per DISTINCT persona, not per user

    ops_cmd = next(c for c in calls if "ops-app-ops:local" in c)
    sci_cmd = next(c for c in calls if "sci-app-sci:local" in c)

    assert ops_cmd[0] == "docker"
    assert "-f" in ops_cmd
    assert os.path.join(ops_path, "Dockerfile") == ops_cmd[ops_cmd.index("-f") + 1]
    assert ops_path == ops_cmd[-1]  # context is project_path
    assert "--label" in ops_cmd
    assert "com.osprey.project=myfacility" in ops_cmd

    assert "com.osprey.project=myfacility" in sci_cmd
    assert sci_path == sci_cmd[-1]


def test_build_persona_images_never_builds_zero_migration_entries(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """An entry with persona=None (no persona system in effect) is skipped --
    it never contributes a build unit, even in local mode."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "legacy", "persona": None, "project": "myfacility-assistant"}]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    assert calls == []


def test_build_persona_images_includes_cli_version_from_persona_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version="2.1.99")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "CLAUDE_CLI_VERSION=2.1.99" in cmd


def test_build_persona_images_omits_cli_version_when_unset_in_persona_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """The persona's own config.yml has no claude_code.cli_version -- the
    build-arg must be omitted entirely (never falls back to the framework
    default the facility/dispatch-worker path uses)."""
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any(str(arg).startswith("CLAUDE_CLI_VERSION=") for arg in cmd)
    # The facility config's own claude_code.cli_version (if any) must never
    # leak into a persona build either -- there is none set here, but the
    # generic OSPREY_PIP_SPEC build-arg is still present.
    assert any(str(arg).startswith("OSPREY_PIP_SPEC=") for arg in cmd)


def test_build_persona_images_never_reads_facility_cli_version(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """A claude_code.cli_version set on the FACILITY config must never leak
    into a persona build -- only the persona's own project_path/config.yml is
    consulted."""
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "claude_code": {"cli_version": "9.9.9"},  # facility-level pin -- must be ignored
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any("9.9.9" in str(arg) for arg in cmd)


def test_build_persona_images_dev_mode_adds_osprey_dev_build_arg(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Under --dev the persona build argv carries OSPREY_DEV=1 (mirroring the
    dispatch-worker project-image dev path)."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, True, {})

    (cmd,) = calls
    assert "OSPREY_DEV=1" in cmd
    assert cmd[cmd.index("OSPREY_DEV=1") - 1] == "--build-arg"


def test_build_persona_images_dev_mode_omits_osprey_dev_when_staging_fails(monkeypatch, tmp_path):
    """--dev with a FAILED wheel staging must build WITHOUT OSPREY_DEV: the
    pin-relaxing arg would otherwise silently install the latest published
    release instead of the local code the flag promises (fail-closed)."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(
        persona_images, "_copy_local_framework_for_override", lambda project_root: False
    )
    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, True, {})

    (cmd,) = calls  # the image is still built -- just without the dev relaxation
    assert "OSPREY_DEV=1" not in cmd


def test_build_persona_images_non_dev_omits_osprey_dev_build_arg(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "OSPREY_DEV=1" not in cmd


def test_build_persona_images_dev_mode_stages_and_cleans_wheel(monkeypatch, tmp_path):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
        (Path(project_root) / "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
        return True

    monkeypatch.setattr(persona_images, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: None)

    persona_images.build_persona_images(config, resolved_users, True, {})

    # Staged artifacts (wheel AND its requirements manifest) must be cleaned
    # up after the build so neither can poison a later non-dev build.
    assert list(Path(ops_path).glob("*.whl")) == []
    assert not (Path(ops_path) / "osprey-local-requirements.txt").exists()


def test_build_persona_images_dev_mode_cleans_staged_artifacts_on_build_failure(
    monkeypatch, tmp_path
):
    """The persona cleanup runs in a finally: a failing image build must still
    remove the staged wheel + manifest from the persona's context."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
        (Path(project_root) / "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
        return True

    def _failing_build(cmd, **k):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(persona_images, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(persona_images.subprocess, "run", _failing_build)

    with pytest.raises(subprocess.CalledProcessError):
        persona_images.build_persona_images(config, resolved_users, True, {})

    assert list(Path(ops_path).glob("*.whl")) == []
    assert not (Path(ops_path) / "osprey-local-requirements.txt").exists()


def test_build_persona_images_no_referenced_personas_runs_no_build(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Local mode + catalog + default_persona configured, but resolved_users
    references no catalog entry (e.g. empty roster) -- no-op, no crash."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }

    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, [], False, {})

    assert calls == []


# ---------------------------------------------------------------------------
# auto_render_missing_personas -- render a referenced persona's project on
# demand when its project_path directory is absent, BEFORE build_persona_images
# builds its image. Renders network-free (--skip-deps), never overwrites a
# complete (user-owned) render, and hard-errors on a partial render or a
# missing build_profile.
# ---------------------------------------------------------------------------


def _auto_render_config(tmp_path, **persona_overrides):
    """A local-mode config whose single persona 'ops' renders to <tmp_path>/ops-app.

    Defaults to a usable build_profile so the render path is exercised; pass
    ``build_profile=None`` to drop it.
    """
    persona = {
        "project": "ops-app",
        "project_path": str(tmp_path / "ops-app"),
        "build_profile": "control-assistant",
    }
    persona.update(persona_overrides)
    return {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": persona},
            }
        }
    }


_AUTO_RENDER_USERS = [{"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"}]


def test_auto_render_renders_when_project_path_missing(monkeypatch, tmp_path):
    """No directory at project_path -> exactly one `osprey build` render, argv
    verbatim: <project> --preset <build_profile> -o <parent(project_path)>
    --skip-deps (rendered into the parent so it lands AT project_path). The CLI
    is re-entered via the RUNNING interpreter (`python -m osprey`), never a
    bare `osprey` that PATH could resolve to a different install."""
    config = _auto_render_config(tmp_path)  # <tmp_path>/ops-app does not exist
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == [
        [
            sys.executable,
            "-m",
            "osprey",
            "build",
            "ops-app",
            "--preset",
            "control-assistant",
            "-o",
            str(tmp_path),
            "--skip-deps",
        ]
    ]


def test_auto_render_partial_render_raises(monkeypatch, tmp_path):
    """project_path exists but is missing its Dockerfile -> a partial render;
    raise (naming the dir) rather than silently rebuild over it."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    # Dockerfile deliberately absent -> partial render.
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    with pytest.raises(ValueError, match="partial render") as excinfo:
        persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert str(project_path) in str(excinfo.value)
    assert "Dockerfile" in str(excinfo.value)
    assert calls == []  # never rendered over the partial tree


def test_auto_render_complete_render_is_noop(monkeypatch, tmp_path):
    """project_path exists with both config.yml and Dockerfile -> user-owned
    complete render; never overwrite it, run no `osprey build`."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    (project_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == []


def test_auto_render_missing_build_profile_raises(monkeypatch, tmp_path):
    """project_path absent (a render IS needed) but the catalog entry has no
    build_profile -> raise, since there's nothing to render from."""
    config = _auto_render_config(tmp_path, build_profile=None)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    with pytest.raises(ValueError, match="build_profile"):
        persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == []  # nothing rendered


def test_auto_render_renders_each_distinct_persona_once(monkeypatch, tmp_path):
    """Two users sharing a persona collapse to one render; a second, distinct
    persona renders separately -- one `osprey build` per DISTINCT persona."""
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {
                        "project": "ops-app",
                        "project_path": str(tmp_path / "ops-app"),
                        "build_profile": "control-assistant",
                    },
                    "sci": {
                        "project": "sci-app",
                        "project_path": str(tmp_path / "sci-app"),
                        "build_profile": "physicist",
                    },
                },
            }
        }
    }
    resolved_users = [
        {"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"},
        {"name": "bob", "index": 1, "persona": "ops", "project": "ops-app"},  # shares ops
        {"name": "carol", "index": 2, "persona": "sci", "project": "sci-app"},
    ]
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, resolved_users, {})

    assert len(calls) == 2
    assert any("ops-app" in c and "control-assistant" in c for c in calls)
    assert any("sci-app" in c and "physicist" in c for c in calls)


# ---------------------------------------------------------------------------
# auto_render_missing_personas -- forwarding the parent project's explicit
# --set model-selection overrides (recorded in its .osprey-manifest.json) into
# each persona render, so one `--set provider=` at parent build time retints
# the whole multi-user stack.
# ---------------------------------------------------------------------------


def _write_parent_manifest(project_root: Path, build_args: dict) -> None:
    import json

    (project_root / ".osprey-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "build_args": build_args}), encoding="utf-8"
    )


def test_auto_render_forwards_explicit_overrides_from_parent_manifest(monkeypatch, tmp_path):
    """Parent manifest marks provider+model as explicit --set overrides ->
    the persona render argv carries the same --set pairs; the non-explicit
    channel_finder_mode (a preset default) is NOT forwarded, so a persona
    preset keeps its own say over anything the user didn't override."""
    parent = tmp_path / "parent"
    parent.mkdir()
    _write_parent_manifest(
        parent,
        {
            "provider": "als-apg",
            "model": "anthropic/claude-opus",
            "channel_finder_mode": "hierarchical",
            "explicit_overrides": ["provider", "model"],
        },
    )
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {}, project_root=parent)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[-4:] == ["--set", "provider=als-apg", "--set", "model=anthropic/claude-opus"]
    assert "channel_finder_mode=hierarchical" not in " ".join(cmd)


def test_auto_render_without_project_root_forwards_nothing(monkeypatch, tmp_path):
    """No project_root (legacy caller) -> argv identical to the pre-forwarding
    form, no --set anywhere."""
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert len(calls) == 1
    assert "--set" not in calls[0]


def test_auto_render_legacy_manifest_without_explicit_marker_forwards_nothing(
    monkeypatch, tmp_path
):
    """A manifest from before explicit_overrides existed records resolved
    values only -> nothing is forwarded (resolved preset defaults must never
    clobber a persona preset's own configuration)."""
    parent = tmp_path / "parent"
    parent.mkdir()
    _write_parent_manifest(parent, {"provider": "anthropic", "model": "claude-haiku-4-5"})
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {}, project_root=parent)

    assert len(calls) == 1
    assert "--set" not in calls[0]


def test_auto_render_malformed_or_absent_manifest_is_ignored(monkeypatch, tmp_path):
    """An unreadable/absent parent manifest degrades to no forwarding -- the
    render itself must never fail over provenance metadata."""
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / ".osprey-manifest.json").write_text("{not json", encoding="utf-8")
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {}, project_root=parent)

    assert len(calls) == 1
    assert "--set" not in calls[0]


def test_auto_render_forwards_only_keys_with_recorded_values(monkeypatch, tmp_path):
    """A key listed in explicit_overrides but missing its recorded value (a
    hand-edited or truncated manifest) is skipped, never rendered as
    `--set key=None`."""
    parent = tmp_path / "parent"
    parent.mkdir()
    _write_parent_manifest(
        parent, {"provider": "als-apg", "explicit_overrides": ["provider", "model"]}
    )
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, _AUTO_RENDER_USERS, {}, project_root=parent)

    cmd = calls[0]
    assert cmd[-2:] == ["--set", "provider=als-apg"]
    assert "model" not in " ".join(cmd)
