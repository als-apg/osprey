"""Unit tests for per-render local image builds.

Covers ``osprey.deployment.web_terminals.render_images`` in isolation: the
local-mode per-render image builder and the pre-render verification that
refuses a start when `osprey build` has not written a render's project.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from osprey.deployment.errors import CapturedProcessError
from osprey.deployment.web_terminals import render_images
from osprey.deployment.web_terminals.roster import RENDER_CATALOG_REQUIRED

# ---------------------------------------------------------------------------
# build_render_images -- local-mode per-render image builder
# ---------------------------------------------------------------------------


def _make_render_project(tmp_path, name, cli_version=None):
    """Both copies of a render project, as one ``osprey build`` writes them.

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
    _build_project_image's tests) so build_render_images tests never touch a
    real wheel build. Reports SUCCESS (True) — the OSPREY_DEV build-arg is
    keyed on staging success, so simulating a successful staging keeps the
    dev-path assertions meaningful; the failure path has its own test."""
    monkeypatch.setattr(
        render_images, "_copy_local_framework_for_override", lambda project_root: True
    )


def test_build_render_images_noop_in_registry_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda *a, **k: calls.append(a))
    config = {"modules": {"web_terminals": {"image_source": "registry"}}}

    render_images.build_render_images(config, [{"render": "ops"}], False, {})

    assert calls == []


def test_build_render_images_local_without_renders_raises():
    config = {"modules": {"web_terminals": {"image_source": "local"}}}

    with pytest.raises(ValueError, match=re.escape(RENDER_CATALOG_REQUIRED)):
        render_images.build_render_images(config, [], False, {})


def test_build_render_images_local_without_default_tier_raises(tmp_path):
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": str(tmp_path)}
                },
            }
        }
    }

    with pytest.raises(ValueError, match=re.escape(RENDER_CATALOG_REQUIRED)):
        render_images.build_render_images(config, [], False, {})


def test_build_render_images_builds_each_referenced_render_once(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_render_project(tmp_path, "ops-app")
    sci_path = _make_render_project(tmp_path, "sci-app")

    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                # The roster serves the renders; resolved_users only names their images.
                "users": [
                    {"name": "alice", "index": 0},
                    {"name": "bob", "index": 1},
                    {"name": "carol", "index": 2, "tier": "sci"},
                ],
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path},
                    "sci": {"kind": "tier", "project": "sci-app", "project_path": sci_path},
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"},
        {
            "name": "bob",
            "render": "ops",
            "project": "ops-app",
            "image": "ops-app:local",
        },  # shares ops -- must not rebuild
        {"name": "carol", "render": "sci", "project": "sci-app", "image": "sci-app:local"},
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    assert len(calls) == 2  # one build per DISTINCT render, not per user

    ops_cmd = next(c for c in calls if "ops-app:local" in c)
    sci_cmd = next(c for c in calls if "sci-app:local" in c)

    # The context is the render's CONTAINER repo, not its flat host render:
    # the render records this machine's project_root, so an image built from it
    # would ship servers and an agent-data root naming the build host. Its
    # Dockerfile is one level down, inside that repo's build/.
    ops_context = str(render_images._render_image_context(ops_path))
    assert ops_cmd[0] == "docker"
    assert "-f" in ops_cmd
    assert os.path.join(ops_context, "build", "Dockerfile") == ops_cmd[ops_cmd.index("-f") + 1]
    assert ops_context == ops_cmd[-1]
    assert ops_path != ops_cmd[-1], "the host render must never be an image build context"
    assert "--label" in ops_cmd
    assert "com.osprey.project=myfacility" in ops_cmd

    assert "com.osprey.project=myfacility" in sci_cmd
    assert str(render_images._render_image_context(sci_path)) == sci_cmd[-1]


def test_build_render_images_builds_a_shared_render_once(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Two renders resolving to one image (same catalog `project` and
    `project_path` — one render deliberately serving both) build that image
    once, not once per tier: same tag, same content, second build wasted."""
    shared_path = _make_render_project(tmp_path, "shared-app")

    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "shared-app", "project_path": shared_path},
                    "sci": {"kind": "tier", "project": "shared-app", "project_path": shared_path},
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "shared-app", "image": "shared-app:local"},
        {"name": "bob", "render": "sci", "project": "shared-app", "image": "shared-app:local"},
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    assert len(calls) == 1
    assert "shared-app:local" in calls[0]


def test_build_render_images_never_builds_unresolved_entries(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """The lenient unresolved entry (render and image None) is skipped -- it
    never contributes a build unit, even in local mode. The default tier is
    served whoever is on it, so it is the one build."""
    ops_path = _make_render_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [{"name": "legacy", "render": None, "project": None, "image": None}]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    assert [cmd[cmd.index("-t") + 1] for cmd in calls] == ["ops-app:local"]


def test_build_render_images_includes_cli_version_from_render_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_render_project(tmp_path, "ops-app", cli_version="2.1.99")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "CLAUDE_CLI_VERSION=2.1.99" in cmd


def test_build_render_images_omits_cli_version_when_unset_in_render_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """The render's own config.yml has no claude_code.cli_version -- the
    build-arg must be omitted entirely (never falls back to the framework
    default the facility/dispatch-worker path uses)."""
    ops_path = _make_render_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any(str(arg).startswith("CLAUDE_CLI_VERSION=") for arg in cmd)
    # The facility config's own claude_code.cli_version (if any) must never
    # leak into a render build either -- there is none set here, but the
    # generic OSPREY_PIP_SPEC build-arg is still present.
    assert any(str(arg).startswith("OSPREY_PIP_SPEC=") for arg in cmd)


def test_build_render_images_never_reads_facility_cli_version(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """A claude_code.cli_version set on the FACILITY config must never leak
    into a render build -- only the render's own project_path/config.yml is
    consulted."""
    ops_path = _make_render_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "claude_code": {"cli_version": "9.9.9"},  # facility-level pin -- must be ignored
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any("9.9.9" in str(arg) for arg in cmd)


def test_build_render_images_dev_mode_adds_osprey_dev_build_arg(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Under --dev the render build argv carries OSPREY_DEV=1 (mirroring the
    dispatch-worker project-image dev path)."""
    ops_path = _make_render_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, True, {})

    (cmd,) = calls
    assert "OSPREY_DEV=1" in cmd
    assert cmd[cmd.index("OSPREY_DEV=1") - 1] == "--build-arg"


def test_build_render_images_dev_mode_omits_osprey_dev_when_staging_fails(monkeypatch, tmp_path):
    """--dev with a FAILED wheel staging must build WITHOUT OSPREY_DEV: the
    pin-relaxing arg would otherwise silently install the latest published
    release instead of the local code the flag promises (fail-closed)."""
    ops_path = _make_render_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    monkeypatch.setattr(
        render_images, "_copy_local_framework_for_override", lambda project_root: False
    )
    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, True, {})

    (cmd,) = calls  # the image is still built -- just without the dev relaxation
    assert "OSPREY_DEV=1" not in cmd


def test_build_render_images_non_dev_omits_osprey_dev_build_arg(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_render_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "OSPREY_DEV=1" not in cmd


def test_build_render_images_dev_mode_stages_and_cleans_wheel(monkeypatch, tmp_path):
    ops_path = _make_render_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
        (Path(project_root) / "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
        return True

    monkeypatch.setattr(render_images, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: None)

    render_images.build_render_images(config, resolved_users, True, {})

    # Staged artifacts (wheel AND its requirements manifest) must be cleaned
    # up after the build so neither can poison a later non-dev build. Staged
    # into — and cleaned out of — the image CONTEXT, which is the only place
    # the Dockerfile's `COPY .dockerignore *.wh[l]` can see a wheel from.
    context = render_images._render_image_context(ops_path)
    assert list(context.glob("*.whl")) == []
    assert not (context / "osprey-local-requirements.txt").exists()


def test_build_render_images_dev_mode_cleans_staged_artifacts_on_build_failure(
    monkeypatch, tmp_path
):
    """The render cleanup runs in a finally: a failing image build must still
    remove the staged wheel + manifest from the render's context."""
    ops_path = _make_render_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path}
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "render": "ops", "project": "ops-app", "image": "ops-app:local"}
    ]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
        (Path(project_root) / "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
        return True

    def _failing_build(cmd, **k):
        # A captured build reports failure as CapturedProcessError, which carries
        # the spool path holding the output the terminal never saw.
        raise CapturedProcessError(cmd, 1)

    monkeypatch.setattr(render_images, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(render_images, "run_captured", _failing_build)

    with pytest.raises(CapturedProcessError):
        render_images.build_render_images(config, resolved_users, True, {})

    context = render_images._render_image_context(ops_path)
    assert list(context.glob("*.whl")) == []
    assert not (context / "osprey-local-requirements.txt").exists()


def test_build_render_images_empty_roster_builds_only_the_default_tier(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Local mode with an empty roster: nothing is built but the default tier,
    which the next person added inherits -- no crash, and no build for a
    catalog entry nobody serves."""
    ops_path = _make_render_project(tmp_path, "ops-app")
    sci_path = _make_render_project(tmp_path, "sci-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": ops_path},
                    "sci": {"kind": "tier", "project": "sci-app", "project_path": sci_path},
                },
            }
        },
    }

    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: calls.append(cmd))

    render_images.build_render_images(config, [], False, {})

    assert [cmd[cmd.index("-t") + 1] for cmd in calls] == ["ops-app:local"]


# ---------------------------------------------------------------------------
# verify_renders -- confirm every referenced render has the project
# `osprey build` rendered for it, BEFORE build_render_images builds its image.
#
# A start renders nothing. `osprey build` writes one project per delta in
# the retired layer directory, into the same `build/` it renders the deployment into, and this
# function only reads: it accepts a complete render, refuses a partial one, and
# refuses an absent one by naming `osprey build`. The `calls` fixture below spies
# on subprocess.run for exactly that reason -- every test here asserts nothing
# was executed, which is the property that keeps the old start-time render from
# creeping back.
#
# The repo root IS the profile root: under the three-zone layout `profile.yml`,
# the retired layer directory and `build/` are siblings at the top of one directory, so there is
# no separate profile to locate and no manifest to read in order to find it.
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path, *tiers: str) -> Path:
    """A deployment repo: profile.yml, .env, and one tiers/<name>.yml each.

    Also the profile root, which is the whole point — a three-zone repo is both,
    so this fixture is deliberately not two directories.
    """
    root = tmp_path / "facility"
    (root / "tiers").mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: Facility\n", encoding="utf-8")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-facility\n", encoding="utf-8")
    for name in tiers:
        (root / "tiers" / f"{name}.yml").write_text(f"name: {name}\n", encoding="utf-8")
    return root


def _render(repo: Path, name: str = "ops-app", *, complete: bool = True) -> Path:
    """A render project of the shape a build leaves under ``build/``.

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


def _render_config(repo: Path, **render_overrides):
    """A local-mode config whose single tier 'ops' renders to build/ops-app.

    The ``renders`` entry is the one ``osprey build`` projects for
    ``tiers/ops.yml``; pass a keyword to override one of its fields.
    """
    render = {
        "kind": "tier",
        "project": "ops-app",
        "project_path": str(repo / "build" / "ops-app"),
    }
    render.update(render_overrides)
    return {
        "deployed_services": [],
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "renders": {"ops": render},
            }
        },
    }


_RENDER_USERS = [
    {"name": "alice", "index": 0, "render": "ops", "project": "ops-app", "image": "ops-app:local"}
]


@pytest.fixture
def calls(monkeypatch) -> list[list[str]]:
    """Every subprocess this module would run. It must stay empty here."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **k: recorded.append(cmd))
    return recorded


def test_a_complete_render_is_accepted(tmp_path, calls):
    """The whole of the happy path: the render is there, so the start proceeds."""
    repo = _repo(tmp_path, "ops")
    _render(repo)

    render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    assert calls == []


def test_an_absent_render_refuses_and_names_the_build(tmp_path, calls):
    """No directory at project_path -> a refusal, not a render.

    The remedy has to be the one command that writes the directory, and it has
    to name the path so an operator can see which of the several renders in
    ``build/`` is the missing one.
    """
    repo = _repo(tmp_path, "ops")

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

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

    A build derives the render's location from the deployment's
    ``project_name`` and the delta's name, so a config whose ``project_path``
    names anything else was rendered by an older build. The refusal names the
    location a build actually uses, which is the only way to tell those two
    situations apart from the message.
    """
    repo = _repo(tmp_path, "ops")
    config = _render_config(repo, project_path=str(repo / "build" / "somewhere-else"))
    config["project_name"] = "demo"

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(config, _RENDER_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert str(repo / "build" / "demo-ops") in message
    assert "project_path" in message
    assert calls == []


def test_every_distinct_render_is_checked(tmp_path, calls):
    """Two users sharing a render collapse to one unit, and a second, distinct
    render is a unit of its own -- so a gap in EITHER is caught."""
    repo = _repo(tmp_path, "ops", "sci")
    _render(repo, "ops-app")
    # sci-app deliberately absent.
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_tier": "ops",
                "users": [
                    {"name": "alice", "index": 0},
                    {"name": "bob", "index": 1},
                    {"name": "carol", "index": 2, "tier": "sci"},
                ],
                "renders": {
                    "ops": {
                        "kind": "tier",
                        "project": "ops-app",
                        "project_path": str(repo / "build" / "ops-app"),
                    },
                    "sci": {
                        "kind": "tier",
                        "project": "sci-app",
                        "project_path": str(repo / "build" / "sci-app"),
                    },
                },
            }
        }
    }
    resolved_users = [
        {"name": "alice", "index": 0, "render": "ops", "project": "ops-app"},
        {"name": "bob", "index": 1, "render": "ops", "project": "ops-app"},  # shares ops
        {"name": "carol", "index": 2, "render": "sci", "project": "sci-app"},
    ]

    with pytest.raises(ValueError, match="'sci'"):
        render_images.verify_renders(config, resolved_users, repo_root=repo)

    assert calls == []


def test_partial_render_raises(tmp_path, calls):
    """project_path exists but is missing its Dockerfile -> a partial render;
    raise (naming the dir and the missing file) rather than hand a half-written
    tree to an image build."""
    repo = _repo(tmp_path, "ops")
    project_path = _render(repo, complete=False)

    with pytest.raises(ValueError, match="partial render") as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    assert str(project_path) in str(excinfo.value)
    assert "Dockerfile" in str(excinfo.value)
    assert "osprey build" in str(excinfo.value)  # the remedy: a rebuild, not a patch
    assert calls == []


def test_existing_render_with_free_form_model_passes(tmp_path, calls):
    """A render naming a model outside the provider's tier map deploys.

    The resolver passes such an ID through verbatim (the operator is trusted
    to name a model the provider serves), so the startup-parity check must not
    refuse it either — the terminal would boot with ANTHROPIC_MODEL set to
    exactly that ID."""
    repo = _repo(tmp_path, "ops")
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\n"
        "claude_code:\n"
        "  provider: anthropic\n"
        "  default_model: anthropic/claude-opus\n",
        encoding="utf-8",
    )

    render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)
    assert calls == []


def test_a_render_placeholder_resolves_from_the_repo_env(tmp_path, calls, monkeypatch):
    """The model check reads the render's render but expands from ``<repo>/.env``.

    Renders are rendered into the disposable build zone and the deployment
    keeps its secrets and facility values at its root. Expanding against the
    render leaves ``${OPS_MODEL}`` a literal, and the check then refuses a
    render this deployment can serve — a start blocked on a value that is
    right there in the file every container reads.
    """
    monkeypatch.delenv("OPS_MODEL", raising=False)
    repo = _repo(tmp_path, "ops")
    (repo / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-facility\nOPS_MODEL=claude-sonnet-5\n", encoding="utf-8"
    )
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\nclaude_code:\n  provider: anthropic\n  default_model: ${OPS_MODEL}\n",
        encoding="utf-8",
    )

    render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    assert calls == []


def test_an_unservable_model_still_gets_the_model_remedy(tmp_path, calls):
    """The converse, and the reason arm order matters: the credential arm sits
    ahead of the general one and must not swallow a genuine model failure.

    The one model failure left (free-form IDs now pass through) is a custom
    provider with no models map and no default model to fall back on."""
    repo = _repo(tmp_path, "ops")
    project_path = _render(repo)
    (project_path / "config.yml").write_text(
        "project_name: ops-app\n"
        "api:\n"
        "  providers:\n"
        "    my-gateway:\n"
        "      base_url: https://gw.example.org/v1\n"
        "claude_code:\n"
        "  provider: my-gateway\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert "model configuration its provider cannot serve" in message
    assert "Fix the model in this deployment's profile" in message
    assert "observability" not in message
    assert calls == []


# ---------------------------------------------------------------------------
# Symlinks. `.resolve()` FOLLOWS a symlink, so the tier file's path alone cannot
# show one that has been linked out of the profile. What has to hold is the
# property the whole design rests on: a tier file must anchor BACK at the
# profile root it was resolved from. Checked on what `resolve_profile_root`
# returns, not on the presence of a symlink, because a symlink is only the
# mechanism -- the danger is the wrong root. These are reached only when the
# render is ABSENT: the refusal would otherwise send an operator through a
# build that succeeds and a start that refuses again.
# ---------------------------------------------------------------------------


def test_delta_symlinked_out_of_the_profile_is_rejected(tmp_path, calls):
    """`tiers/ops.yml` is a symlink to a file outside the profile. It is
    lexically perfect and `is_file()` succeeds, but root discovery reads the
    resolved target as a STANDALONE profile -- so the tier it named would
    have no data tree, no `.env` secrets and no conventions, silently."""
    from osprey.cli.profile_root import ProfileKind, resolve_profile_root

    repo = _repo(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    target = outside / "ops.yml"
    target.write_text("name: ops\n", encoding="utf-8")
    (repo / "tiers" / "ops.yml").symlink_to(target)

    # The reason it must be rejected: the target anchors somewhere else, as a
    # standalone profile rather than a layer over this deployment's.
    assert resolve_profile_root(target) == (outside.resolve(), ProfileKind.ROOT)

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert str(repo) in message  # the profile it must have belonged to
    assert str(outside.resolve()) in message  # where it actually landed
    assert calls == []


def test_symlinked_tiers_directory_is_rejected(tmp_path, calls):
    """The whole `tiers/` directory is a symlink to a shared one that has its
    OWN `profile.yml`. Root discovery follows it and anchors the tier at THAT
    profile — so the tier would merge over a different facility's profile
    than the one being deployed, again silently.

    This variant defeats a same-directory containment check too (both sides
    resolve through the symlink to the shared directory and compare equal),
    which is why the check is on the resolved ROOT rather than on the parent
    directory."""
    from osprey.cli.profile_root import ProfileKind, resolve_profile_root

    repo = _repo(tmp_path)
    (repo / "tiers").rmdir()
    shared = tmp_path / "shared"
    (shared / "tiers").mkdir(parents=True)
    (shared / "profile.yml").write_text("name: Somebody else\n", encoding="utf-8")
    (shared / "tiers" / "ops.yml").write_text("name: ops\n", encoding="utf-8")
    (repo / "tiers").symlink_to(shared / "tiers")

    # Resolves inside a `tiers/` dir and IS read as a layer -- just over the
    # wrong profile. That is exactly what makes it dangerous rather than broken.
    assert resolve_profile_root(repo / "tiers" / "ops.yml") == (shared.resolve(), ProfileKind.TIER)

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    message = str(excinfo.value)
    assert str(shared.resolve()) in message  # the profile it would have merged over
    assert str(repo) in message  # the profile it had to merge over
    assert calls == []


def test_missing_tier_file_is_rejected_by_absolute_path(tmp_path, calls):
    """The config names a render whose tier file the repo no longer has: the
    error names the absolute path that has to exist, since no build can write
    the render without it."""
    repo = _repo(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    assert str(tmp_path / "facility" / "tiers" / "ops.yml") in str(excinfo.value)
    assert calls == []


def test_a_tier_with_no_profile_beside_it_is_rejected(tmp_path, calls):
    """The source zone is gone or was never there: a `tiers/` directory with
    no `profile.yml` above it holds files that cannot be read as layers at all.

    Root discovery's own message already names the file that has to exist, which
    is a better sentence than a second one here could be -- so what this pins is
    that the message survives to the operator rather than being flattened into
    the generic rebuild advice."""
    repo = tmp_path / "facility"
    (repo / "tiers").mkdir(parents=True)
    (repo / "tiers" / "ops.yml").write_text("name: ops\n", encoding="utf-8")
    # profile.yml deliberately absent.

    with pytest.raises(ValueError) as excinfo:
        render_images.verify_renders(_render_config(repo), _RENDER_USERS, repo_root=repo)

    assert str(repo / "profile.yml") in str(excinfo.value)
    assert calls == []


# ---------------------------------------------------------------------------
# Absence assertions. Two things this module must never do, both of which would
# be silent regressions rather than failures if they appeared.
#
# It must not shell out to `osprey build` for a render whose project is
# absent, which is what would make a start able to write into `build/`. And it
# must not read the parent's `.osprey-manifest.json` `build_args`, take the keys
# passed as `--set` at parent build time, and append the same pairs to every
# render, retinting the whole stack from one parent override.
#
# Neither can appear by accident: `osprey build` renders every render from
# the profile, an explicit override is written INTO the profile, and a delta
# merges over that same profile -- so replaying a build invocation would apply
# the change twice, and a stale manifest entry would apply a value the profile
# does not hold.
# ---------------------------------------------------------------------------


def test_parent_set_override_forwarding_helper_is_gone():
    assert not hasattr(render_images, "_parent_set_override_args")


def test_the_start_time_render_is_gone():
    """Both halves: an entry point that renders, and a manifest indirection
    whose only job would be finding the profile it rendered from. Under three
    zones the repo root IS the profile root, so that lookup is a tautology with
    failure modes of its own."""
    assert not hasattr(render_images, "auto_render_missing_renders")
    assert not hasattr(render_images, "_parent_profile_root")
