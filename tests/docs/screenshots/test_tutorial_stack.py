"""Command-assembly, ordering, teardown, and skip tests for the stack provider.

CI-safe by construction: no real container engine, no real ``osprey`` binary,
and no real agent are ever invoked. ``subprocess.run``/``Popen`` and
``wait_for_port`` are mocked, so these tests only prove that
:func:`docs.screenshots.capture._tutorial_stack` assembles the *exact*,
project-scoped lifecycle commands, in the right order, tears everything down on
failure, and degrades to :class:`ScreenshotSkip` when the CLI is absent.

Safety invariant asserted here: no assembled command ever contains a prune,
``-a``/``--all``, ``volume``, or ``system`` teardown — only the exact,
project-scoped forms (``build``/``deploy up -d``/``sim apply``/``deploy down``).
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docs.screenshots import capture, recipes
from docs.screenshots.capture import ScreenshotSkip, assert_hero_structural
from docs.screenshots.recipes import DocShot
from PIL import Image

_ARTIFACT_PORT = 54321

# Tokens that must NEVER appear in any assembled command — a destructive or
# system-wide container operation would violate the project-scoped safety rule.
_FORBIDDEN_TOKENS = ("prune", "-a", "--all", "volume", "system", "rebuild", "clean")


def _ok_result() -> SimpleNamespace:
    """A stand-in for a successful ``subprocess.run`` return value."""
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _cmd_of(call) -> list[str]:
    """The command list ``argv[0]`` from a recorded ``subprocess.run`` call."""
    return call.args[0] if call.args else call.kwargs["args"]


def _drive_stack(monkeypatch, tmp_path, *, run):
    """Enter/exit ``_tutorial_stack`` with all side-effecting seams mocked.

    ``run`` is installed as ``subprocess.run``; ``mkdtemp`` yields ``tmp_path``;
    ``wait_for_port``, ``shutil.rmtree``, and ``subprocess.Popen`` are stubbed.
    Returns the parent :class:`Mock` whose ``.mock_calls`` records global order.
    """
    parent = Mock()
    parent.attach_mock(run, "run")
    wait = Mock()
    parent.attach_mock(wait, "wait")
    rmtree = Mock()
    parent.attach_mock(rmtree, "rmtree")

    monkeypatch.setattr(capture.subprocess, "run", run)
    monkeypatch.setattr(capture, "wait_for_port", wait)
    monkeypatch.setattr(capture.tempfile, "mkdtemp", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(capture.shutil, "rmtree", rmtree)
    monkeypatch.setattr(
        capture.subprocess, "Popen", Mock(side_effect=AssertionError("Popen not expected"))
    )
    return parent


# ---------------------------------------------------------------------------
# 1. Command assembly (exact, project-scoped forms only)
# ---------------------------------------------------------------------------


def test_command_assembly_is_exact_and_project_scoped(monkeypatch, tmp_path) -> None:
    run = Mock(return_value=_ok_result())
    _drive_stack(monkeypatch, tmp_path, run=run)

    # Project renders at <build_root>/<name>; build_root is the mkdtemp dir (tmp_path).
    proj = str(tmp_path / capture._TUTORIAL_PROJECT_NAME)

    with capture._tutorial_stack(artifact_port=_ARTIFACT_PORT) as project_dir:
        assert str(project_dir) == proj

    cmds = [_cmd_of(c) for c in run.call_args_list]

    # build <name> --preset control-assistant -o <build_root> --skip-deps
    #       --set config.artifact_server.port=<port>
    build = next(c for c in cmds if "build" in c)
    assert capture._TUTORIAL_PROJECT_NAME in build
    assert "--preset" in build
    assert build[build.index("--preset") + 1] == "control-assistant"
    assert "--skip-deps" in build
    assert "-o" in build
    assert build[build.index("-o") + 1] == str(tmp_path)
    assert f"config.artifact_server.port={_ARTIFACT_PORT}" in build
    # The profile 'config' bucket key — a bare 'artifact_server.port' is dropped.
    assert "artifact_server.port" not in build

    # deploy up MUST be detached (-d); the non-detached form execvpe's away.
    deploy_up = next(c for c in cmds if "deploy" in c and "up" in c)
    assert "-d" in deploy_up
    up_call = next(c for c in run.call_args_list if "deploy" in _cmd_of(c) and "up" in _cmd_of(c))
    assert up_call.kwargs["cwd"] == proj

    # sim apply nominal --yes --now <ANCHOR>, cwd == project dir.
    seed_call = next(c for c in run.call_args_list if "sim" in _cmd_of(c))
    seed = _cmd_of(seed_call)
    assert seed[:3] == ["osprey", "sim", "apply"]
    assert "nominal" in seed
    assert "--yes" in seed
    assert "--now" in seed
    assert recipes.ANCHOR in seed
    assert seed[seed.index("--now") + 1] == recipes.ANCHOR
    assert seed_call.kwargs["cwd"] == proj

    # deploy down (teardown) is project-scoped to the temp dir.
    down_call = next(
        c for c in run.call_args_list if "deploy" in _cmd_of(c) and "down" in _cmd_of(c)
    )
    assert down_call.kwargs["cwd"] == proj


def test_no_command_contains_a_destructive_token(monkeypatch, tmp_path) -> None:
    run = Mock(return_value=_ok_result())
    _drive_stack(monkeypatch, tmp_path, run=run)

    with capture._tutorial_stack(artifact_port=_ARTIFACT_PORT):
        pass

    for call in run.call_args_list:
        cmd = _cmd_of(call)
        for token in _FORBIDDEN_TOKENS:
            assert token not in cmd, f"forbidden token {token!r} in command {cmd!r}"


# ---------------------------------------------------------------------------
# 2. Readiness ordering: wait_for_port(5432) BEFORE sim apply
# ---------------------------------------------------------------------------


def test_waits_for_postgres_before_seeding(monkeypatch, tmp_path) -> None:
    run = Mock(return_value=_ok_result())
    parent = _drive_stack(monkeypatch, tmp_path, run=run)

    with capture._tutorial_stack(artifact_port=_ARTIFACT_PORT):
        pass

    # Locate the global-order indices of the readiness wait and the seed call.
    names = list(parent.mock_calls)
    wait_idx = next(
        i for i, c in enumerate(names) if c[0] == "wait" and c.args and c.args[0] == 5432
    )
    seed_idx = next(i for i, c in enumerate(names) if c[0] == "run" and "sim" in _cmd_of(c))
    assert wait_idx < seed_idx, "Postgres readiness must be awaited before seeding"


# ---------------------------------------------------------------------------
# 3. Teardown always runs (even when a lifecycle step raises)
# ---------------------------------------------------------------------------


def test_teardown_runs_when_seed_fails(monkeypatch, tmp_path) -> None:
    def run_side_effect(cmd, *args, **kwargs):
        argv = cmd
        if "sim" in argv:
            return SimpleNamespace(returncode=1, stdout="", stderr="seed boom")
        return _ok_result()

    run = Mock(side_effect=run_side_effect)
    parent = _drive_stack(monkeypatch, tmp_path, run=run)

    with pytest.raises(ScreenshotSkip):
        with capture._tutorial_stack(artifact_port=_ARTIFACT_PORT):
            pytest.fail("body must not run when seeding fails")

    # deploy down (project-scoped, cwd=project dir) still ran despite the failure.
    down_call = next(
        c for c in run.call_args_list if "deploy" in _cmd_of(c) and "down" in _cmd_of(c)
    )
    assert down_call.kwargs["cwd"] == str(tmp_path / capture._TUTORIAL_PROJECT_NAME)
    # rmtree of the exact build root still ran.
    parent.rmtree.assert_called_once()
    assert parent.rmtree.call_args.args[0] == tmp_path


# ---------------------------------------------------------------------------
# 4. Graceful skip when the osprey binary is absent
# ---------------------------------------------------------------------------


def test_missing_binary_raises_screenshot_skip(monkeypatch, tmp_path) -> None:
    run = Mock(side_effect=FileNotFoundError("no osprey on PATH"))
    parent = _drive_stack(monkeypatch, tmp_path, run=run)

    with pytest.raises(ScreenshotSkip):
        with capture._tutorial_stack(artifact_port=_ARTIFACT_PORT):
            pytest.fail("body must not run without the CLI")

    # Even the failed-preflight path tears the temp dir down.
    parent.rmtree.assert_called_once()


def test_capture_tutorial_stack_skips_without_runtime(monkeypatch, tmp_path) -> None:
    run = Mock(side_effect=FileNotFoundError("no osprey on PATH"))
    _drive_stack(monkeypatch, tmp_path, run=run)

    shot = DocShot(name="hero", environment="tutorial_stack", kind="static")
    with pytest.raises(ScreenshotSkip):
        capture.capture_tutorial_stack(lambda: None, shot, agentic=False)


# ---------------------------------------------------------------------------
# 5. assert_hero_structural pure-helper checks
# ---------------------------------------------------------------------------


def _png_bytes(size: tuple[int, int], *, uniform: bool) -> bytes:
    """A real PNG of ``size``; ``uniform`` makes every pixel identical (blank)."""
    img = Image.new("RGB", size, (30, 30, 30))
    if not uniform:
        # A single contrasting pixel is enough to break uniformity.
        img.putpixel((0, 0), (240, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_assert_hero_structural_accepts_right_size_non_blank() -> None:
    viewport = (48, 32)
    assert_hero_structural(_png_bytes(viewport, uniform=False), viewport)


def test_assert_hero_structural_rejects_wrong_size() -> None:
    with pytest.raises(AssertionError):
        assert_hero_structural(_png_bytes((48, 32), uniform=False), (64, 40))


def test_assert_hero_structural_rejects_blank() -> None:
    viewport = (48, 32)
    with pytest.raises(AssertionError):
        assert_hero_structural(_png_bytes(viewport, uniform=True), viewport)


def test_assert_hero_structural_rejects_non_png() -> None:
    with pytest.raises(AssertionError):
        assert_hero_structural(b"not a png at all", (48, 32))
