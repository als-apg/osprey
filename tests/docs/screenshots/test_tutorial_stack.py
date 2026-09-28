"""Command-assembly, ordering, teardown, and skip tests for the stack provider.

CI-safe by construction: no real container engine, no real ``osprey`` binary,
and no real agent are ever invoked. ``subprocess.run``/``Popen`` and
``wait_for_port`` are mocked, so these tests only prove that
:func:`docs.screenshots.capture.tutorial_stack` assembles the *exact*,
project-scoped lifecycle commands, in the right order, tears everything down on
failure, and degrades to :class:`ScreenshotSkip` when the CLI is absent.

Safety invariant asserted here: no assembled command ever contains a prune,
``-a``/``--all``, ``volume``, or ``system`` teardown — only the exact,
project-scoped forms (``build``/``osprey up -d``/``sim apply``/``osprey reset``).
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

# Tokens that must NEVER appear in any assembled command — a destructive or
# system-wide container operation would violate the project-scoped safety rule.
_FORBIDDEN_TOKENS = ("prune", "-a", "--all", "volume", "system", "rebuild", "clean")


_OPUS_ID = "claude-opus-test"


def _ok_result() -> SimpleNamespace:
    """A stand-in for a successful ``subprocess.run`` return value."""
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _cmd_of(call) -> list[str]:
    """The command list ``argv[0]`` from a recorded ``subprocess.run`` call."""
    return call.args[0] if call.args else call.kwargs["args"]


def _drive_stack(monkeypatch, tmp_path, *, run):
    """Enter/exit ``tutorial_stack`` with all side-effecting seams mocked.

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
    monkeypatch.setattr(capture, "_opus_model_id", lambda project_dir: _OPUS_ID)
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

    with capture.tutorial_stack() as project_dir:
        assert str(project_dir) == proj

    cmds = [_cmd_of(c) for c in run.call_args_list]

    # Creation is TWO commands, because they are two things: `init` writes the
    # source zone (and bakes --set into the emitted profile.yml), `build`
    # renders it. --skip-deps belongs to the render, --preset/--set to init.
    #
    # init <build_root>/<name> --preset control-assistant --no-git
    init = next(c for c in cmds if "init" in c)
    assert proj in init, "init must name the deployment repo directory positionally"
    assert "--preset" in init
    assert init[init.index("--preset") + 1] == "control-assistant"
    # The build derives the artifact-server port from `deployment.port_base` and
    # refuses a profile that states it, so init pins no port.
    assert not any("artifact_server" in token for token in init)
    # The capture drives a single-user `osprey web` on the host, whose index-0
    # panel ports the preset's multi-user roster would otherwise own.
    assert init[init.index("--set") + 1] == "config.modules.web_terminals.enabled=false"
    # The model is pinned after init, by id, because only the written catalog
    # says which Opus id the chosen provider serves.
    assert not any(t.startswith("model=") for t in init)
    set_call = next(c for c in run.call_args_list if _cmd_of(c)[:2] == ["osprey", "set"])
    assert _cmd_of(set_call) == ["osprey", "set", f"model={_OPUS_ID}"]
    assert set_call.kwargs["cwd"] == proj
    order = [_cmd_of(c)[1] for c in run.call_args_list]
    assert order.index("init") < order.index("set") < order.index("build")

    # build (zero-argument), run FROM the repo it renders.
    build = next(c for c in cmds if "build" in c)
    assert "--skip-deps" in build
    # This checkout is not a released version, so the containers are built from
    # it: a dev render here, and the dev start below.
    assert "--dev" in build
    assert capture._TUTORIAL_PROJECT_NAME not in build, (
        "build is zero-argument; the repo comes from its cwd, not an argument"
    )
    build_call = next(c for c in run.call_args_list if "build" in _cmd_of(c))
    assert build_call.kwargs["cwd"] == proj

    # osprey up MUST be detached (-d); the non-detached form execvpe's away.
    up_call = next(c for c in run.call_args_list if _cmd_of(c)[:2] == ["osprey", "up"])
    assert "-d" in _cmd_of(up_call)
    assert "--dev" in _cmd_of(up_call)
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

    # Teardown is `osprey reset --yes`, repo-scoped to the temp dir: it removes
    # this deployment's volumes too, so the next run's fresh credentials never
    # meet a store initialized with the last run's.
    down_call = next(c for c in run.call_args_list if _cmd_of(c)[:2] == ["osprey", "reset"])
    assert _cmd_of(down_call) == ["osprey", "reset", "--yes"]
    assert down_call.kwargs["cwd"] == proj


def test_provider_override_is_baked_into_init(monkeypatch, tmp_path) -> None:
    run = Mock(return_value=_ok_result())
    _drive_stack(monkeypatch, tmp_path, run=run)

    monkeypatch.delenv(capture.PROVIDER_ENV, raising=False)
    with capture.tutorial_stack():
        pass
    init = next(_cmd_of(c) for c in run.call_args_list if "init" in _cmd_of(c))
    assert not any(t.startswith("provider=") for t in init), "no override: the preset's own"

    run.reset_mock()
    monkeypatch.setenv(capture.PROVIDER_ENV, "some-proxy")
    with capture.tutorial_stack():
        pass
    init = next(_cmd_of(c) for c in run.call_args_list if "init" in _cmd_of(c))
    assert "provider=some-proxy" in init
    assert init[init.index("provider=some-proxy") - 1] == "--set"


def test_no_command_contains_a_destructive_token(monkeypatch, tmp_path) -> None:
    run = Mock(return_value=_ok_result())
    _drive_stack(monkeypatch, tmp_path, run=run)

    with capture.tutorial_stack():
        pass

    for call in run.call_args_list:
        cmd = _cmd_of(call)
        for token in _FORBIDDEN_TOKENS:
            assert token not in cmd, f"forbidden token {token!r} in command {cmd!r}"


# ---------------------------------------------------------------------------
# 2. Readiness ordering: wait_for_port(postgres host port) BEFORE sim apply
# ---------------------------------------------------------------------------


def test_waits_for_postgres_before_seeding(monkeypatch, tmp_path) -> None:
    run = Mock(return_value=_ok_result())
    parent = _drive_stack(monkeypatch, tmp_path, run=run)

    with capture.tutorial_stack():
        pass

    # Locate the global-order indices of the readiness wait and the seed call.
    names = list(parent.mock_calls)
    wait_idx = next(
        i
        for i, c in enumerate(names)
        if c[0] == "wait" and c.args and c.args[0] == capture._POSTGRES_PORT
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
        with capture.tutorial_stack():
            pytest.fail("body must not run when seeding fails")

    # osprey reset (repo-scoped, cwd=repo dir) still ran despite the failure.
    down_call = next(c for c in run.call_args_list if _cmd_of(c)[:2] == ["osprey", "reset"])
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
        with capture.tutorial_stack():
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


# ---------------------------------------------------------------------------
# Rendered artifact-server port
# ---------------------------------------------------------------------------


def test_rendered_artifact_port_reads_the_built_config(tmp_path) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text(
        "artifact_server:\n  port: 10200\n  host: 127.0.0.1\n"
    )
    assert capture.rendered_artifact_port(tmp_path) == 10200


def test_rendered_artifact_port_skips_without_a_rendered_port(tmp_path) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text("artifact_server:\n  host: 127.0.0.1\n")
    with pytest.raises(ScreenshotSkip, match="artifact_server.port"):
        capture.rendered_artifact_port(tmp_path)
    with pytest.raises(ScreenshotSkip, match="artifact_server.port"):
        capture.rendered_artifact_port(tmp_path / "missing")


# ---------------------------------------------------------------------------
# The demo's model: the chosen provider's Opus id, read from the written catalog
# ---------------------------------------------------------------------------


def _write_repo(tmp_path, provider: str, models: list[str]):
    (tmp_path / "profile.yml").write_text(f"extends: control-assistant\nprovider: {provider}\n")
    (tmp_path / "providers.yml").write_text(
        "providers:\n"
        f"  {provider}:\n"
        "    base_url: http://127.0.0.1:1/v1\n"
        "    models:\n" + "".join(f"      - {m}\n" for m in models)
    )
    return tmp_path


def test_the_opus_id_is_the_one_the_provider_serves(tmp_path) -> None:
    repo = _write_repo(tmp_path, "gw", ["claude-sonnet-9", "claude-opus-9-1", "claude-haiku-9"])
    assert capture._opus_model_id(repo) == "claude-opus-9-1"


def test_a_provider_serving_no_opus_skips_the_capture(tmp_path) -> None:
    repo = _write_repo(tmp_path, "gw", ["gpt-sol", "gpt-luna"])
    with pytest.raises(ScreenshotSkip, match="gw"):
        capture._opus_model_id(repo)


def test_an_unreadable_catalog_skips_the_capture(tmp_path) -> None:
    (tmp_path / "profile.yml").write_text("provider: gw\n")
    with pytest.raises(ScreenshotSkip):
        capture._opus_model_id(tmp_path)
