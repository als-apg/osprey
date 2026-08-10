"""Unit tests for per-persona local image builds.

Covers ``osprey.deployment.web_terminals.persona_images`` in isolation: the
local-mode per-persona image builder and the on-demand persona project
auto-render.
"""

from __future__ import annotations

import json
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
# builds its image.
#
# Every render comes from a delta inside THIS deployment's own profile
# (<profile root>/personas/<persona>.yml, located through the deployed
# project's manifest). Renders network-free (--skip-deps), never overwrites a
# complete (user-owned) render, and hard-errors on a partial render, a missing
# build_profile, or a build_profile that is not such a delta.
# ---------------------------------------------------------------------------


def _profile_root(tmp_path: Path, *personas: str) -> Path:
    """A materialized profile: profile.yml plus one personas/<name>.yml delta each."""
    root = tmp_path / "facility-profile"
    (root / "personas").mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: Facility\n", encoding="utf-8")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-facility\n", encoding="utf-8")
    for name in personas:
        (root / "personas" / f"{name}.yml").write_text(f"name: {name}\n", encoding="utf-8")
    return root


def _deployed_project(tmp_path: Path, profile_path: Path | None) -> Path:
    """The deployed project root, whose manifest records the profile it was built from.

    ``profile_path=None`` writes a manifest that records no profile at all --
    the legacy/unlocatable case.
    """
    project = tmp_path / "host"
    project.mkdir(parents=True, exist_ok=True)
    build_args = {} if profile_path is None else {"profile_path_abs": str(profile_path)}
    (project / ".osprey-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "build_args": build_args}), encoding="utf-8"
    )
    return project


def _auto_render_config(tmp_path, **persona_overrides):
    """A local-mode config whose single persona 'ops' renders to <tmp_path>/ops-app.

    Defaults to the ``build_profile`` ``osprey profile new`` emits; pass
    ``build_profile=None`` to drop it, or another value to exercise a rejection.
    """
    persona = {
        "project": "ops-app",
        "project_path": str(tmp_path / "ops-app"),
        "build_profile": "personas/ops.yml",
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


@pytest.fixture
def calls(monkeypatch) -> list[list[str]]:
    recorded: list[list[str]] = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: recorded.append(cmd))
    return recorded


def test_auto_render_renders_from_the_deployments_own_persona_delta(tmp_path, calls):
    """No directory at project_path -> exactly one `osprey build` render, argv
    verbatim: <project> <profile root>/personas/<persona>.yml -o
    <parent(project_path)> --skip-deps (rendered into the parent so it lands AT
    project_path). The CLI is re-entered via the RUNNING interpreter (`python -m
    osprey`), never a bare `osprey` that PATH could resolve to a different
    install."""
    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
    )

    assert calls == [
        [
            sys.executable,
            "-m",
            "osprey",
            "build",
            "ops-app",
            str(root / "personas" / "ops.yml"),
            "-o",
            str(tmp_path),
            "--skip-deps",
        ]
    ]


def test_auto_render_profile_argument_anchors_env_at_the_parent_profile_root(tmp_path, calls):
    """The reason the delta path is what gets handed to `osprey build`: root
    discovery resolves it back to the PARENT profile root, which is the
    directory the build derives the persona project's `.env` from. A persona
    therefore ships the facility's secrets, not a set of its own."""
    from osprey.cli.profile_root import resolve_profile_root

    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
    )

    rendered_from = Path(calls[0][calls[0].index("build") + 2])
    assert resolve_profile_root(rendered_from) == (root, True)
    assert (root / ".env").is_file()  # ...and that root is where the secrets live


def test_auto_render_resolves_against_the_root_when_the_deployment_is_itself_a_persona(
    tmp_path, calls
):
    """A project built from a persona delta records THAT file in its manifest.
    Root discovery -- not `Path.parent` -- is what makes its own personas
    resolve against the profile root one level above `personas/`."""
    root = _profile_root(tmp_path, "ops", "host")
    project_root = _deployed_project(tmp_path, root / "personas" / "host.yml")

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
    )

    assert calls[0][calls[0].index("build") + 2] == str(root / "personas" / "ops.yml")


def test_auto_render_renders_each_distinct_persona_once(tmp_path, calls):
    """Two users sharing a persona collapse to one render; a second, distinct
    persona renders separately -- one `osprey build` per DISTINCT persona."""
    root = _profile_root(tmp_path, "ops", "sci")
    project_root = _deployed_project(tmp_path, root / "profile.yml")
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {
                        "project": "ops-app",
                        "project_path": str(tmp_path / "ops-app"),
                        "build_profile": "personas/ops.yml",
                    },
                    "sci": {
                        "project": "sci-app",
                        "project_path": str(tmp_path / "sci-app"),
                        "build_profile": "personas/sci.yml",
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

    persona_images.auto_render_missing_personas(
        config, resolved_users, {}, project_root=project_root
    )

    assert len(calls) == 2
    rendered_from = {call[call.index("build") + 2] for call in calls}
    assert rendered_from == {
        str(root / "personas" / "ops.yml"),
        str(root / "personas" / "sci.yml"),
    }


def test_auto_render_partial_render_raises(tmp_path, calls):
    """project_path exists but is missing its Dockerfile -> a partial render;
    raise (naming the dir) rather than silently rebuild over it."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    # Dockerfile deliberately absent -> partial render.
    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    with pytest.raises(ValueError, match="partial render") as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
        )

    assert str(project_path) in str(excinfo.value)
    assert "Dockerfile" in str(excinfo.value)
    assert calls == []  # never rendered over the partial tree


def test_auto_render_complete_render_is_noop_and_reads_no_profile(tmp_path, calls, monkeypatch):
    """project_path exists with both config.yml and Dockerfile -> user-owned
    complete render; never overwrite it, run no `osprey build`. The deployment's
    profile is not even LOOKED UP: a deploy with nothing to render must not fail
    over a profile it never has to read.

    The lookup is stubbed to raise rather than merely asserting no crash — a
    bare project root makes `_parent_profile_root` return None either way, so an
    eager implementation would pass an assertion on the outcome alone. This
    fails unless the call genuinely never happens."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    (project_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    bare_root = tmp_path / "host"
    bare_root.mkdir()

    def _must_not_be_called(project_root):
        raise AssertionError(
            f"resolved the deployment profile for {project_root} with nothing to render"
        )

    monkeypatch.setattr(persona_images, "_parent_profile_root", _must_not_be_called)

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=bare_root
    )

    assert calls == []


def test_auto_render_existing_render_with_unservable_model_raises(tmp_path, calls):
    """A complete persona render whose config names a model its provider cannot
    serve must fail the deploy here, with the path and a remedy — not boot a
    web-terminal container that crash-loops behind the reverse proxy (502).
    An existing render is user-owned and never re-rendered, so a model the
    profile has since moved away from would otherwise persist forever."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text(
        "project_name: ops-app\n"
        "claude_code:\n"
        "  provider: anthropic\n"
        "  default_model: anthropic/claude-opus\n",
        encoding="utf-8",
    )
    (project_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    with pytest.raises(ValueError, match="neither a model tier nor a model ID") as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
        )

    assert str(project_path) in str(excinfo.value)
    assert "remove" in str(excinfo.value).lower()  # remedy: fix or remove the render
    assert calls == []  # never rendered over the user-owned tree


# ---------------------------------------------------------------------------
# Rejections. A persona project is rendered from a delta in the deployment's
# OWN profile or not at all -- every other value is refused, and every refusal
# names the file the operator has to point at instead.
# ---------------------------------------------------------------------------


def _expected_delta(tmp_path) -> Path:
    return tmp_path / "facility-profile" / "personas" / "ops.yml"


def _reject(tmp_path, build_profile) -> str:
    """Run the auto-render with a rejected build_profile; return the message."""
    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")
    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path, build_profile=build_profile),
            _AUTO_RENDER_USERS,
            {},
            project_root=project_root,
        )
    return str(excinfo.value)


@pytest.mark.parametrize("value", ["control-assistant", "control_assistant"])
def test_preset_valued_build_profile_is_rejected(tmp_path, calls, value):
    """The old shape, in both spellings a preset name is written in. A bundled
    preset would render a persona sharing none of this deployment's data,
    secrets or conventions, so it is refused outright rather than quietly
    built."""
    message = _reject(tmp_path, value)

    assert value in message
    assert "preset" in message
    assert "personas/ops.yml" in message
    assert str(_expected_delta(tmp_path)) in message
    assert calls == []


def test_slash_path_without_a_suffix_is_a_path_not_a_preset(tmp_path, calls):
    """`personas/ops` has no `.yml` but does have a separator, so it reads as a
    path: it must reach the missing-FILE error naming where that file has to be,
    never the "names the bundled preset 'personas/ops'" rejection, whose remedy
    would send the operator looking for a preset that was never the problem."""
    message = _reject(tmp_path, "personas/ops")

    assert "preset" not in message
    assert str(tmp_path / "facility-profile" / "personas" / "ops") in message
    assert str(_expected_delta(tmp_path)) in message  # the remedy still names the .yml
    assert calls == []


# ---------------------------------------------------------------------------
# Symlinks. The shape rule is LEXICAL (it is shared with lint, which has no
# filesystem), and `.resolve()` FOLLOWS a symlink -- so shape alone cannot see
# a delta that has been linked out of the profile. What has to hold is the
# property the whole design rests on: the path handed to `osprey build` must
# anchor BACK at the profile root it was resolved from. Checked on what
# `resolve_profile_root` returns, not on the presence of a symlink, because a
# symlink is only the mechanism -- the danger is the wrong root.
# ---------------------------------------------------------------------------


def test_delta_symlinked_out_of_the_profile_is_rejected(tmp_path, calls):
    """`personas/ops.yml` is a symlink to a file outside the profile. It is
    lexically perfect and `is_file()` succeeds, but root discovery reads the
    resolved target as a STANDALONE profile -- so the persona would build with
    no data tree, no `.env` secrets and no conventions, silently. Silence is
    worse than any rejection, so this fails."""
    from osprey.cli.profile_root import resolve_profile_root

    root = _profile_root(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    target = outside / "ops.yml"
    target.write_text("name: ops\n", encoding="utf-8")
    (root / "personas" / "ops.yml").symlink_to(target)
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    # The reason it must be rejected: the target anchors somewhere else, as a
    # standalone profile rather than a delta over this deployment's.
    assert resolve_profile_root(target) == (outside.resolve(), False)

    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
        )

    message = str(excinfo.value)
    assert str(root) in message  # the profile it must have belonged to
    assert str(outside.resolve()) in message  # where it actually landed
    assert calls == []


def test_symlinked_personas_directory_is_rejected(tmp_path, calls):
    """The whole `personas/` directory is a symlink to a shared one that has its
    OWN `profile.yml`. Root discovery follows it and anchors the delta at THAT
    profile — so the persona would merge over a different facility's profile
    than the one being deployed, again silently.

    This variant defeats a same-directory containment check too (both sides
    resolve through the symlink to the shared directory and compare equal),
    which is why the check is on the resolved ROOT rather than on the parent
    directory."""
    from osprey.cli.profile_root import resolve_profile_root

    root = _profile_root(tmp_path)
    (root / "personas").rmdir()
    shared = tmp_path / "shared"
    (shared / "personas").mkdir(parents=True)
    (shared / "profile.yml").write_text("name: Somebody else\n", encoding="utf-8")
    (shared / "personas" / "ops.yml").write_text("name: ops\n", encoding="utf-8")
    (root / "personas").symlink_to(shared / "personas")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    # Resolves inside a `personas/` dir and IS read as a delta -- just over the
    # wrong profile. That is exactly what makes it dangerous rather than broken.
    assert resolve_profile_root(root / "personas" / "ops.yml") == (shared.resolve(), True)

    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
        )

    message = str(excinfo.value)
    assert str(shared.resolve()) in message  # the profile it would have merged over
    assert str(root) in message  # the profile it had to merge over
    assert calls == []


def test_a_delta_that_anchors_back_at_the_profile_root_is_accepted(tmp_path, calls):
    """The positive control for the two above: an ordinary delta round-trips
    through root discovery to the very root it was resolved from, and renders."""
    from osprey.cli.profile_root import resolve_profile_root

    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
    )

    rendered_from = Path(calls[0][calls[0].index("build") + 2])
    assert resolve_profile_root(rendered_from) == (root, True)


def test_absolute_build_profile_is_rejected(tmp_path, calls):
    """An absolute path can name any profile on the host -- including one this
    deployment does not own."""
    outside = tmp_path / "elsewhere" / "personas" / "ops.yml"
    outside.parent.mkdir(parents=True)
    outside.write_text("name: ops\n", encoding="utf-8")

    message = _reject(tmp_path, str(outside))

    assert str(outside) in message
    assert str(_expected_delta(tmp_path)) in message
    assert calls == []


@pytest.mark.parametrize(
    "value",
    [
        "../elsewhere/personas/ops.yml",  # climbs out of the profile
        "personas/../ops.yml",  # climbs back out through the right directory
        "profiles/ops.yml",  # a sibling directory of personas/
        "ops.yml",  # the profile root itself, not personas/
        "personas/nested/ops.yml",  # deeper than one level: never read as a delta
    ],
)
def test_build_profile_outside_the_personas_directory_is_rejected(tmp_path, calls, value):
    """Root discovery reads a file as a delta only when its parent directory IS
    the profile's `personas/`. Anything else would build a hollow project from
    the delta alone, or from a profile this deployment does not own."""
    message = _reject(tmp_path, value)

    assert value in message
    assert str(_expected_delta(tmp_path)) in message
    assert calls == []


def test_missing_delta_file_is_rejected_by_absolute_path(tmp_path, calls):
    """Right shape, no file: the error names the absolute path that has to
    exist, not just the catalog value."""
    message = _reject(tmp_path, "personas/ghost.yml")

    assert str(tmp_path / "facility-profile" / "personas" / "ghost.yml") in message
    assert calls == []


def test_missing_build_profile_raises_and_names_the_delta_it_wants(tmp_path, calls):
    """project_path absent (a render IS needed) but the catalog entry has no
    build_profile -> raise, naming the file that entry should point at."""
    message = _reject(tmp_path, None)

    assert "build_profile" in message
    assert "personas/ops.yml" in message
    assert str(_expected_delta(tmp_path)) in message
    assert calls == []


@pytest.mark.parametrize("recorded", [None, "gone"])
def test_unlocatable_deployment_profile_is_rejected(tmp_path, calls, recorded):
    """No profile recorded in the manifest, or one that has since been deleted:
    there is no anchor, and guessing one is how a persona gets rendered from
    somebody else's profile. Fail, naming the project whose manifest is at
    fault."""
    profile_path = None if recorded is None else tmp_path / "gone" / "profile.yml"
    project_root = _deployed_project(tmp_path, profile_path)

    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
        )

    message = str(excinfo.value)
    assert str(project_root) in message
    assert "osprey build" in message  # the remedy
    assert calls == []


def test_manifest_naming_a_rootless_persona_delta_is_rejected(tmp_path, calls):
    """The deployed project was built from a file under `personas/` whose
    `profile.yml` is gone -- root discovery cannot answer, so neither can this."""
    orphan = tmp_path / "orphan" / "personas" / "host.yml"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("name: host\n", encoding="utf-8")
    project_root = _deployed_project(tmp_path, orphan)

    with pytest.raises(ValueError) as excinfo:
        persona_images.auto_render_missing_personas(
            _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
        )

    assert str(project_root) in str(excinfo.value)
    assert calls == []


# ---------------------------------------------------------------------------
# Deletion assertions: the parent-build forwarding this module used to do is
# GONE. It read the parent's `.osprey-manifest.json` `build_args`, took the
# keys the user had explicitly passed as `--set` at parent build time
# (`explicit_overrides`), and appended the same `--set KEY=VALUE` pairs to every
# persona render, so one parent override retinted the whole stack.
#
# It cannot come back by accident: the profile is the source of truth now, an
# explicit override is written INTO the profile at build time, and a persona
# delta merges over that same profile -- so replaying the parent's build
# invocation would apply the change twice, and a stale manifest entry would
# apply a value the profile no longer holds.
# ---------------------------------------------------------------------------


def test_parent_set_override_forwarding_helper_is_gone():
    assert not hasattr(persona_images, "_parent_set_override_args")


def test_parent_manifest_overrides_are_never_replayed_into_a_render(tmp_path, calls):
    """A manifest still carrying the retired `explicit_overrides` field (a
    project built before this changed) contributes nothing to the argv."""
    root = _profile_root(tmp_path, "ops")
    project = tmp_path / "host"
    project.mkdir()
    (project / ".osprey-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "build_args": {
                    "profile_path_abs": str(root / "profile.yml"),
                    "provider": "als-apg",
                    "model": "anthropic/claude-opus",
                    "explicit_overrides": ["provider", "model"],
                },
            }
        ),
        encoding="utf-8",
    )

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project
    )

    (cmd,) = calls
    assert "--set" not in cmd
    assert not any("als-apg" in arg or "claude-opus" in arg for arg in cmd)


def test_a_render_never_takes_the_preset_path(tmp_path, calls):
    """`--preset` is what would make the subprocess materialize a profile of its
    own; the argv only ever carries an existing profile FILE, positionally."""
    root = _profile_root(tmp_path, "ops")
    project_root = _deployed_project(tmp_path, root / "profile.yml")

    persona_images.auto_render_missing_personas(
        _auto_render_config(tmp_path), _AUTO_RENDER_USERS, {}, project_root=project_root
    )

    (cmd,) = calls
    assert "--preset" not in cmd
    assert Path(cmd[cmd.index("build") + 2]).is_file()
