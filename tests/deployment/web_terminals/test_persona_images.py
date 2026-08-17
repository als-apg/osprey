"""Unit tests for per-persona local image builds.

Covers ``osprey.deployment.web_terminals.persona_images`` in isolation: the
local-mode per-persona image builder and the pre-render verification that
refuses a start when `osprey build` has not written a persona's project.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.build.claude_code_telemetry import ObservabilityCredentialError
from osprey.deployment.errors import CapturedProcessError
from osprey.deployment.web_terminals import persona_images

# ---------------------------------------------------------------------------
# build_persona_images -- local-mode per-persona image builder
# ---------------------------------------------------------------------------


def _make_persona_project(tmp_path, name, cli_version=None):
    """Both copies of a persona project, as one ``osprey build`` writes them.

    The flat HOST render at ``<name>/`` — what the catalog's ``project_path``
    names, and what the credential sweep and lint read — and beside it the
    container copy at ``.image/<name>/``, a deployment repo whose render sits in
    ``build/``. The image is built from the second: only it records the
    ``/app/<name>`` paths a container can resolve. Tests that assert on the
    build argv are asserting against that context, so the fixture has to produce
    both or it would be pinning a shape no build makes.
    """
    project_dir = tmp_path / name
    render = tmp_path / ".image" / name / "build"
    config = (
        f"claude_code:\n  cli_version: {cli_version!r}\n"
        if cli_version is not None
        else "project_name: whatever\n"
    )
    for directory in (project_dir, render):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        (directory / "config.yml").write_text(config, encoding="utf-8")
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
    monkeypatch.setattr(persona_images, "run_captured", lambda *a, **k: calls.append(a))
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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, resolved_users, False, {})

    assert len(calls) == 2  # one build per DISTINCT persona, not per user

    ops_cmd = next(c for c in calls if "ops-app-ops:local" in c)
    sci_cmd = next(c for c in calls if "sci-app-sci:local" in c)

    # The context is the persona's CONTAINER repo, not its flat host render:
    # the render records this machine's project_root, so an image built from it
    # would ship servers and an agent-data root naming the build host. Its
    # Dockerfile is one level down, inside that repo's build/.
    ops_context = str(persona_images._persona_image_context(ops_path))
    assert ops_cmd[0] == "docker"
    assert "-f" in ops_cmd
    assert os.path.join(ops_context, "build", "Dockerfile") == ops_cmd[ops_cmd.index("-f") + 1]
    assert ops_context == ops_cmd[-1]
    assert ops_path != ops_cmd[-1], "the host render must never be an image build context"
    assert "--label" in ops_cmd
    assert "com.osprey.project=myfacility" in ops_cmd

    assert "com.osprey.project=myfacility" in sci_cmd
    assert str(persona_images._persona_image_context(sci_path)) == sci_cmd[-1]


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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: None)

    persona_images.build_persona_images(config, resolved_users, True, {})

    # Staged artifacts (wheel AND its requirements manifest) must be cleaned
    # up after the build so neither can poison a later non-dev build. Staged
    # into — and cleaned out of — the image CONTEXT, which is the only place
    # the Dockerfile's `COPY .dockerignore *.wh[l]` can see a wheel from.
    context = persona_images._persona_image_context(ops_path)
    assert list(context.glob("*.whl")) == []
    assert not (context / "osprey-local-requirements.txt").exists()


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
        # A captured build reports failure as CapturedProcessError, which carries
        # the spool path holding the output the terminal never saw.
        raise CapturedProcessError(cmd, 1)

    monkeypatch.setattr(persona_images, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(persona_images, "run_captured", _failing_build)

    with pytest.raises(CapturedProcessError):
        persona_images.build_persona_images(config, resolved_users, True, {})

    context = persona_images._persona_image_context(ops_path)
    assert list(context.glob("*.whl")) == []
    assert not (context / "osprey-local-requirements.txt").exists()


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
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    persona_images.build_persona_images(config, [], False, {})

    assert calls == []


# ---------------------------------------------------------------------------
# verify_persona_renders -- confirm every referenced persona has the project
# `osprey build` rendered for it, BEFORE build_persona_images builds its image.
#
# A start renders nothing. `osprey build` writes one project per delta in
# `personas/`, into the same `build/` it renders the deployment into, and this
# function only reads: it accepts a complete render, refuses a partial one, and
# refuses an absent one by naming `osprey build`. The `calls` fixture below spies
# on subprocess.run for exactly that reason -- every test here asserts nothing
# was executed, which is the property that keeps the old start-time render from
# creeping back.
#
# The repo root IS the profile root: under the three-zone layout `profile.yml`,
# `personas/` and `build/` are siblings at the top of one directory, so there is
# no separate profile to locate and no manifest to read in order to find it.
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path, *personas: str) -> Path:
    """A deployment repo: profile.yml, .env, and one personas/<name>.yml each.

    Also the profile root, which is the whole point — a three-zone repo is both,
    so this fixture is deliberately not two directories.
    """
    root = tmp_path / "facility"
    (root / "personas").mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: Facility\n", encoding="utf-8")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-facility\n", encoding="utf-8")
    for name in personas:
        (root / "personas" / f"{name}.yml").write_text(f"name: {name}\n", encoding="utf-8")
    return root


def _render(repo: Path, name: str = "ops-app", *, complete: bool = True) -> Path:
    """A persona project of the shape a build leaves under ``build/``.

    Both copies when complete — the flat host render and the container repo at
    ``build/.image/<name>/`` the image is actually built from. Returns the flat
    one, which is what the catalog names.
    """
    project = repo / "build" / name
    context_render = repo / "build" / ".image" / name / "build"
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yml").write_text(f"project_name: {name}\n", encoding="utf-8")
    if complete:
        (project / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        context_render.mkdir(parents=True, exist_ok=True)
        (context_render / "config.yml").write_text(f"project_name: {name}\n", encoding="utf-8")
        (context_render / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    return project


def _persona_config(repo: Path, **persona_overrides):
    """A local-mode config whose single persona 'ops' renders to build/ops-app.

    Defaults to the ``build_profile`` ``osprey init`` emits; pass
    ``build_profile=None`` to drop it, or another value to exercise a rejection.
    """
    persona = {
        "project": "ops-app",
        "project_path": str(repo / "build" / "ops-app"),
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


_PERSONA_USERS = [{"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"}]


@pytest.fixture
def calls(monkeypatch) -> list[list[str]]:
    """Every subprocess this module would run. It must stay empty here."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(persona_images, "run_captured", lambda cmd, **k: recorded.append(cmd))
    return recorded


def test_a_complete_render_is_accepted(tmp_path, calls):
    """The whole of the happy path: the render is there, so the start proceeds."""
    repo = _repo(tmp_path, "ops")
    _render(repo)

    persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    assert calls == []


def test_an_absent_render_refuses_and_names_the_build(tmp_path, calls):
    """No directory at project_path -> a refusal, not a render.

    The remedy has to be the one command that writes the directory, and it has
    to name the path so an operator can see which of the several renders in
    ``build/`` is the missing one.
    """
    repo = _repo(tmp_path, "ops")

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert "ops" in message
    assert "osprey build" in message
    assert str(repo / "build" / "ops-app") in message
    assert calls == []
    # Nothing was written, either: a start that "helpfully" rendered would have
    # made the directory rather than raising about it.
    assert not (repo / "build" / "ops-app").exists()


def test_the_refusal_names_where_a_build_would_have_put_it(tmp_path, calls):
    """The second thing that can be wrong: the catalog names a path no build
    writes.

    A build derives the render's location from the repo's name and the delta's
    — it never reads the catalog — so a hand-edited ``project_path`` produces a
    start that refuses after every successful build. The refusal names the
    location a build actually uses, which is the only way to tell those two
    situations apart from the message.
    """
    repo = _repo(tmp_path, "ops")
    config = _persona_config(repo, project_path=str(repo / "build" / "somewhere-else"))

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(config, _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert str(repo / "build" / f"{repo.name}-ops") in message
    assert "project_path" in message
    assert calls == []


def test_every_distinct_persona_is_checked(tmp_path, calls):
    """Two users sharing a persona collapse to one unit, and a second, distinct
    persona is a unit of its own -- so a gap in EITHER is caught."""
    repo = _repo(tmp_path, "ops", "sci")
    _render(repo, "ops-app")
    # sci-app deliberately absent.
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {
                        "project": "ops-app",
                        "project_path": str(repo / "build" / "ops-app"),
                        "build_profile": "personas/ops.yml",
                    },
                    "sci": {
                        "project": "sci-app",
                        "project_path": str(repo / "build" / "sci-app"),
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

    with pytest.raises(ValueError, match="'sci'"):
        persona_images.verify_persona_renders(config, resolved_users, repo_root=repo)

    assert calls == []


def test_partial_render_raises(tmp_path, calls):
    """project_path exists but is missing its Dockerfile -> a partial render;
    raise (naming the dir and the missing file) rather than hand a half-written
    tree to an image build."""
    repo = _repo(tmp_path, "ops")
    project_path = _render(repo, complete=False)

    with pytest.raises(ValueError, match="partial render") as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    assert str(project_path) in str(excinfo.value)
    assert "Dockerfile" in str(excinfo.value)
    assert "osprey build" in str(excinfo.value)  # the remedy: a rebuild, not a patch
    assert calls == []


def test_existing_render_with_unservable_model_raises(tmp_path, calls):
    """A persona render whose config names a model its provider cannot serve
    must fail the deploy here, with the path and a remedy — not boot a
    web-terminal container that crash-loops behind the reverse proxy (502)."""
    repo = _repo(tmp_path, "ops")
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\n"
        "claude_code:\n"
        "  provider: anthropic\n"
        "  default_model: anthropic/claude-opus\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="neither a model tier nor a model ID") as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    assert str(project_path) in str(excinfo.value)
    assert "osprey build" in str(excinfo.value)  # remedy: fix the profile, rebuild
    assert calls == []


def test_a_persona_placeholder_resolves_from_the_repo_env(tmp_path, calls, monkeypatch):
    """The model check reads the persona's render but expands from ``<repo>/.env``.

    Personas are rendered into the disposable build zone and the deployment
    keeps its secrets and facility values at its root. Expanding against the
    render leaves ``${OPS_MODEL}`` a literal, and the check then refuses a
    persona this deployment can serve — a start blocked on a value that is
    right there in the file every container reads.
    """
    monkeypatch.delenv("OPS_MODEL", raising=False)
    repo = _repo(tmp_path, "ops")
    (repo / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-facility\nOPS_MODEL=sonnet\n", encoding="utf-8"
    )
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\nclaude_code:\n  provider: anthropic\n  default_model: ${OPS_MODEL}\n",
        encoding="utf-8",
    )

    persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    assert calls == []


def _telemetry_render(repo: Path, password_line: str) -> Path:
    """A complete render whose model is fine and whose telemetry block varies.

    The model is deliberately servable in every one of these: the point is what
    the refusal says when the ONLY thing wrong is an observability credential.
    """
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\n"
        "claude_code:\n"
        "  provider: anthropic\n"
        "  default_model: sonnet\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: openobserve\n" + password_line,
        encoding="utf-8",
    )
    return project_path


def test_unresolved_observability_credential_is_not_a_model_problem(tmp_path, calls, monkeypatch):
    """An observability credential nothing on the host resolves is reported as
    a credential, with the file it belongs in.

    Both failures come out of the same resolve and the credential one is a
    ValueError too, so the general frame would otherwise tell an operator whose
    store password is unset to go and fix their model — an edit to a profile
    that was never wrong, leaving the actual gap in place.
    """
    monkeypatch.delenv("OBSERVABILITY_PASSWORD", raising=False)
    repo = _repo(tmp_path, "ops")
    project_path = _telemetry_render(
        repo,
        "    openobserve:\n"
        "      user: operator@example.com\n"
        "      password: ${OBSERVABILITY_PASSWORD}\n",
    )

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert "observability credentials this deployment cannot resolve" in message
    assert str(project_path) in message
    # The variable to set, and the one file that will be read for it -- a name,
    # never a value, since an unresolved placeholder has none to print.
    assert "OBSERVABILITY_PASSWORD" in message
    assert f'echo "OBSERVABILITY_PASSWORD=<value>" >> {repo / ".env"}' in message
    assert "`osprey up` mints" in message
    # And emphatically not the other remedy.
    assert "Fix the model in this deployment's profile" not in message
    assert isinstance(excinfo.value.__cause__, ObservabilityCredentialError)
    assert calls == []


def test_a_blank_observability_credential_names_the_config_keys(tmp_path, calls):
    """Nothing declared at all: there is no variable name to hand back, so the
    remedy names the config keys instead of inventing one."""
    repo = _repo(tmp_path, "ops")
    _telemetry_render(repo, "    openobserve:\n      org: default\n")

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert "openobserve.user" in message and "openobserve.password" in message
    assert str(repo / ".env") in message
    assert "Fix the model in this deployment's profile" not in message
    assert calls == []


def test_an_unservable_model_still_gets_the_model_remedy(tmp_path, calls):
    """The converse, and the reason arm order matters: the credential arm sits
    ahead of the general one and must not swallow a genuine model failure."""
    repo = _repo(tmp_path, "ops")
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\n"
        "claude_code:\n"
        "  provider: anthropic\n"
        "  default_model: anthropic/claude-opus\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: openobserve\n"
        "    openobserve:\n"
        "      user: operator@example.com\n"
        "      password: not-a-placeholder\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert "model configuration its provider cannot serve" in message
    assert "Fix the model in this deployment's profile" in message
    assert "observability" not in message
    assert calls == []


# ---------------------------------------------------------------------------
# Rejections. A persona project is rendered from a delta in the deployment's
# OWN profile or not at all -- every other value is refused, and every refusal
# names the file the operator has to point at instead.
#
# These are reached only when the render is ABSENT, and that is deliberate: the
# refusal above would otherwise send an operator whose catalog can never name a
# delta through a build that succeeds and a start that refuses again.
# ---------------------------------------------------------------------------


def _expected_delta(tmp_path) -> Path:
    return tmp_path / "facility" / "personas" / "ops.yml"


def _reject(tmp_path, build_profile) -> str:
    """Verify a repo whose render is absent and whose build_profile is refused."""
    repo = _repo(tmp_path, "ops")
    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(
            _persona_config(repo, build_profile=build_profile),
            _PERSONA_USERS,
            repo_root=repo,
        )
    return str(excinfo.value)


@pytest.mark.parametrize("value", ["control-assistant", "control_assistant"])
def test_preset_valued_build_profile_is_rejected(tmp_path, calls, value):
    """The old shape, in both spellings a preset name is written in. A bundled
    preset names a persona sharing none of this deployment's data, secrets or
    conventions -- and no build renders one, so the generic "run osprey build"
    remedy would be a loop."""
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
    assert str(tmp_path / "facility" / "personas" / "ops") in message
    assert str(_expected_delta(tmp_path)) in message  # the remedy still names the .yml
    assert calls == []


# ---------------------------------------------------------------------------
# Symlinks. The shape rule is LEXICAL (it is shared with lint, which has no
# filesystem), and `.resolve()` FOLLOWS a symlink -- so shape alone cannot see
# a delta that has been linked out of the profile. What has to hold is the
# property the whole design rests on: a delta must anchor BACK at the profile
# root it was resolved from. Checked on what `resolve_profile_root` returns, not
# on the presence of a symlink, because a symlink is only the mechanism -- the
# danger is the wrong root. The build enforces the same property over the deltas
# it enumerates (`build_cmd._persona_deltas`); this is the catalog's half.
# ---------------------------------------------------------------------------


def test_delta_symlinked_out_of_the_profile_is_rejected(tmp_path, calls):
    """`personas/ops.yml` is a symlink to a file outside the profile. It is
    lexically perfect and `is_file()` succeeds, but root discovery reads the
    resolved target as a STANDALONE profile -- so the persona it named would
    have no data tree, no `.env` secrets and no conventions, silently."""
    from osprey.cli.profile_root import resolve_profile_root

    repo = _repo(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    target = outside / "ops.yml"
    target.write_text("name: ops\n", encoding="utf-8")
    (repo / "personas" / "ops.yml").symlink_to(target)

    # The reason it must be rejected: the target anchors somewhere else, as a
    # standalone profile rather than a delta over this deployment's.
    assert resolve_profile_root(target) == (outside.resolve(), False)

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert str(repo) in message  # the profile it must have belonged to
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

    repo = _repo(tmp_path)
    (repo / "personas").rmdir()
    shared = tmp_path / "shared"
    (shared / "personas").mkdir(parents=True)
    (shared / "profile.yml").write_text("name: Somebody else\n", encoding="utf-8")
    (shared / "personas" / "ops.yml").write_text("name: ops\n", encoding="utf-8")
    (repo / "personas").symlink_to(shared / "personas")

    # Resolves inside a `personas/` dir and IS read as a delta -- just over the
    # wrong profile. That is exactly what makes it dangerous rather than broken.
    assert resolve_profile_root(repo / "personas" / "ops.yml") == (shared.resolve(), True)

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert str(shared.resolve()) in message  # the profile it would have merged over
    assert str(repo) in message  # the profile it had to merge over
    assert calls == []


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
    the profile's `personas/`. Anything else names a hollow project built from
    the delta alone, or a profile this deployment does not own."""
    message = _reject(tmp_path, value)

    assert value in message
    assert str(_expected_delta(tmp_path)) in message
    assert calls == []


def test_missing_delta_file_is_rejected_by_absolute_path(tmp_path, calls):
    """Right shape, no file: the error names the absolute path that has to
    exist, not just the catalog value."""
    message = _reject(tmp_path, "personas/ghost.yml")

    assert str(tmp_path / "facility" / "personas" / "ghost.yml") in message
    assert calls == []


def test_missing_build_profile_raises_and_names_the_delta_it_wants(tmp_path, calls):
    """project_path absent (so the render is genuinely missing) and the catalog
    entry has no build_profile -> raise, naming the file that entry should point
    at rather than the generic rebuild."""
    message = _reject(tmp_path, None)

    assert "build_profile" in message
    assert "personas/ops.yml" in message
    assert str(_expected_delta(tmp_path)) in message
    assert calls == []


def test_a_delta_with_no_profile_beside_it_is_rejected(tmp_path, calls):
    """The source zone is gone or was never there: a `personas/` directory with
    no `profile.yml` above it holds files that cannot be read as deltas at all.

    Root discovery's own message already names the file that has to exist, which
    is a better sentence than a second one here could be -- so what this pins is
    that the message survives to the operator rather than being flattened into
    the generic rebuild advice."""
    repo = tmp_path / "facility"
    (repo / "personas").mkdir(parents=True)
    (repo / "personas" / "ops.yml").write_text("name: ops\n", encoding="utf-8")
    # profile.yml deliberately absent.

    with pytest.raises(ValueError) as excinfo:
        persona_images.verify_persona_renders(_persona_config(repo), _PERSONA_USERS, repo_root=repo)

    assert str(repo / "profile.yml") in str(excinfo.value)
    assert calls == []


# ---------------------------------------------------------------------------
# Absence assertions. Two things this module must never do, both of which would
# be silent regressions rather than failures if they appeared.
#
# It must not shell out to `osprey build` for a persona whose project is
# absent, which is what would make a start able to write into `build/`. And it
# must not read the parent's `.osprey-manifest.json` `build_args`, take the keys
# passed as `--set` at parent build time, and append the same pairs to every
# persona render, retinting the whole stack from one parent override.
#
# Neither can appear by accident: `osprey build` renders every persona from
# the profile, an explicit override is written INTO the profile, and a delta
# merges over that same profile -- so replaying a build invocation would apply
# the change twice, and a stale manifest entry would apply a value the profile
# does not hold.
# ---------------------------------------------------------------------------


def test_parent_set_override_forwarding_helper_is_gone():
    assert not hasattr(persona_images, "_parent_set_override_args")


def test_the_start_time_render_is_gone():
    """Both halves: an entry point that renders, and a manifest indirection
    whose only job would be finding the profile it rendered from. Under three
    zones the repo root IS the profile root, so that lookup is a tautology with
    failure modes of its own."""
    assert not hasattr(persona_images, "auto_render_missing_personas")
    assert not hasattr(persona_images, "_parent_profile_root")
