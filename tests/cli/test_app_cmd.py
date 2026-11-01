"""``osprey app check``: what the build checks, the import boundary, and the JS halves."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.app_cmd import app
from tests.cli._lifecycle_build import invoke_build

FIXTURE_APP = Path(__file__).parents[1] / "apps" / "fixtures" / "js_check_app"
LOADER_IMPORT = "from osprey.apps.loader import load\n"


def _check(runner: CliRunner, repo: Path) -> Result:
    return runner.invoke(app, ["check", "--repo", str(repo)])


def _list(repo: Path, *names: str) -> None:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["apps"] = {name: {} for name in names}
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _write(app_dir: Path, relative: str, text: str) -> None:
    path = app_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _manifest(name: str) -> str:
    return (
        f"name: {name}\n"
        f"containers:\n  {name}-service:\n    runtime: osprey\n"
        "    command: [python, -m, server]\n    health: /health\n"
    )


def _app(repo: Path) -> Path:
    """One listed application, its manifest valid and nothing else in its directory."""
    app_dir = repo / "apps" / "knob"
    _write(app_dir, "app.yml", _manifest("knob"))
    _list(repo, "knob")
    return app_dir


# --- what the build checks ------------------------------------------------------


def test_with_no_apps_directory_there_is_nothing_to_check(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert len(result.output.splitlines()) == 1
    assert "Nothing to check" in result.output


def test_an_unlisted_directory_is_reported_and_left_unchecked(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    unlisted = lifecycle_repo / "apps" / "helper"
    unlisted.mkdir(parents=True)
    (unlisted / "app.yml").write_text("name: [\n", encoding="utf-8")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "apps/helper/ not checked: the apps: block of profile.yml does not name it"
    ]


def test_a_violation_reads_as_the_build_prints_it(
    runner: CliRunner, lifecycle_repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(lifecycle_repo / "apps" / "phoebus", "app.yml", _manifest("phoebus"))
    _write(lifecycle_repo / "apps" / "ariel", "app.yml", _manifest("ariel"))
    _list(lifecycle_repo, "phoebus", "ariel")

    result = _check(runner, lifecycle_repo)
    built = invoke_build(lifecycle_repo)

    assert result.exit_code == 1
    assert built.exit_code != 0
    lines = [line for line in result.output.splitlines() if line.endswith("]")]
    assert len(lines) == 2
    for line, app_name in zip(lines, ("ariel", "phoebus"), strict=True):
        text, _, rule = line.rpartition(" [")
        assert f"apps/{app_name}/app.yml: name:" in text
        assert rule == "reserved_name]"
        assert text in built.output + caplog.text


def test_a_requirement_only_an_unlisted_directory_holds_is_reported(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    _write(lifecycle_repo / "apps" / "helper", "pyproject.toml", '[project]\nname = "lab-helper"\n')
    profile = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    profile["dependencies"] = ["lab-helper"]
    (lifecycle_repo / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert "./apps/helper" in result.output
    assert "[shadowed_package]" in result.output


def test_a_profile_that_does_not_resolve_is_named(runner: CliRunner, lifecycle_repo: Path) -> None:
    (lifecycle_repo / "profile.yml").write_text("config: [\n", encoding="utf-8")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert "does not resolve" in result.output


def test_a_directory_outside_any_repo_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    result = _check(runner, tmp_path)

    assert result.exit_code == 1
    assert "No OSPREY deployment repo" in result.output


# --- the import boundary --------------------------------------------------------


def test_only_an_applications_packages_face_the_boundary(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "knob_app/__init__.py", "")
    _write(app_dir, "knob_app/server.py", "from osprey.apps.sdk import run\n")
    _write(app_dir, ".venv/site-packages/x.py", LOADER_IMPORT)
    _write(app_dir, "tests/helper.py", LOADER_IMPORT)

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output


def test_a_refused_import_in_a_package_names_its_module(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "knob_app/__init__.py", "")
    _write(app_dir, "knob_app/server.py", LOADER_IMPORT)

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert "apps/knob/knob_app/server.py:1: osprey.apps.loader.load:" in result.output
    assert "[import_boundary]" in result.output


def test_an_application_with_no_package_passes_the_boundary_saying_so(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    _app(lifecycle_repo)

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert "apps/knob/: no Python package" in result.output


# --- the JS halves, with a stubbed runner ---------------------------------------


class _Js:
    """Stands in for Node: records every command and answers with a set status."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []
        self.status: dict[str, int] = {}

    def run(self, argv: Any, cwd: Path, env: Any) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        tool = Path(argv[0]).name
        self.calls.append((argv, cwd, dict(env)))
        status = self.status.get(tool, 0)
        return subprocess.CompletedProcess(argv, status, f"{tool} said no\n" if status else "", "")

    @property
    def tools(self) -> list[str]:
        return [Path(argv[0]).name for argv, _cwd, _env in self.calls]

    def call(self, tool: str) -> tuple[list[str], Path, dict[str, str]]:
        return next(c for c in self.calls if Path(c[0][0]).name == tool)


@pytest.fixture
def js(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Js:
    stub = _Js()
    monkeypatch.setattr("osprey.cli.app_cmd._run", stub.run)
    monkeypatch.setattr("osprey.cli.app_cmd.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return stub


@pytest.mark.parametrize("node", [True, False], ids=["node present", "node absent"])
def test_an_application_with_no_js_runs_no_js_tool(
    runner: CliRunner, lifecycle_repo: Path, js: _Js, monkeypatch: pytest.MonkeyPatch, node: bool
) -> None:
    if not node:
        monkeypatch.setattr("osprey.cli.app_cmd.shutil.which", lambda _name: None)
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "panel/index.html", "<p>no script</p>\n")
    _write(app_dir, "tests/test_server.py", "")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert js.calls == []


def test_panel_js_with_no_node_fails_naming_node(
    runner: CliRunner, lifecycle_repo: Path, js: _Js, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("osprey.cli.app_cmd.shutil.which", lambda _name: None)
    _write(_app(lifecycle_repo), "panel/main.js", "")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert "Node required" in result.output
    assert "[node_required]" in result.output
    assert js.calls == []


def test_panel_js_alone_runs_eslint_alone_over_what_the_application_wrote(
    runner: CliRunner, lifecycle_repo: Path, js: _Js
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "panel/main.js", "")
    _write(app_dir, "panel/lib/util.mjs", "")
    _write(app_dir, "panel/lib/chart.min.js", "")
    _write(app_dir, "panel/vendor/plot.js", "")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert js.tools == ["npm", "eslint"]
    argv, cwd, _env = js.call("eslint")
    assert cwd == app_dir
    assert argv[-2:] == ["panel/lib/util.mjs", "panel/main.js"]


def test_a_panel_of_fetched_js_only_runs_no_js_tool(
    runner: CliRunner, lifecycle_repo: Path, js: _Js
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "panel/vendor/plot.js", "")
    _write(app_dir, "panel/fetched.js", "")
    fetched = f"vendor:\n- url: https://example.org/f.js\n  into: panel/fetched.js\n  sha256: {'a' * 64}\n"
    _write(app_dir, "app.yml", _manifest("knob") + fetched)

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert js.calls == []


def test_a_js_test_alone_runs_vitest_alone_from_the_applications_directory(
    runner: CliRunner, lifecycle_repo: Path, js: _Js
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "tests/deep/main.test.mjs", "")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert js.tools == ["npm", "vitest"]
    argv, cwd, env = js.call("vitest")
    assert cwd == app_dir
    assert argv[1] == "run"
    design_system = Path(env["OSPREY_DESIGN_SYSTEM_JS"])
    assert (design_system / "dom.js").is_file()


def test_a_js_file_in_tests_that_matches_no_glob_runs_no_vitest(
    runner: CliRunner, lifecycle_repo: Path, js: _Js
) -> None:
    _write(_app(lifecycle_repo), "tests/helper.js", "")

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert js.calls == []


@pytest.mark.parametrize("tool", ["eslint", "vitest"])
def test_a_js_tool_that_fails_is_reported_with_its_output(
    runner: CliRunner, lifecycle_repo: Path, js: _Js, tool: str
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir, "panel/main.js", "")
    _write(app_dir, "tests/main.test.js", "")
    js.status[tool] = 1

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert js.tools == ["npm", "eslint", "vitest"]
    assert f"apps/knob: {tool} exited 1:" in result.output
    assert f"{tool} said no" in result.output


def test_a_toolchain_that_cannot_be_fetched_is_named(
    runner: CliRunner, lifecycle_repo: Path, js: _Js, tmp_path: Path
) -> None:
    _write(_app(lifecycle_repo), "panel/main.js", "")
    js.status["npm"] = 1

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert "cannot be fetched" in result.output
    assert "[js_toolchain]" in result.output
    assert js.tools == ["npm"]
    assert list((tmp_path / "cache" / "osprey" / "js_check").iterdir()) == []


def test_a_cached_toolchain_runs_without_fetching(
    runner: CliRunner, lifecycle_repo: Path, js: _Js
) -> None:
    _write(_app(lifecycle_repo), "panel/main.js", "")
    assert _check(runner, lifecycle_repo).exit_code == 0
    js.calls.clear()
    js.status["npm"] = 1

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert js.tools == ["eslint"]


def test_a_cached_toolchain_is_installed_beside_its_configs(
    runner: CliRunner, lifecycle_repo: Path, js: _Js
) -> None:
    _write(_app(lifecycle_repo), "panel/main.js", "")

    assert _check(runner, lifecycle_repo).exit_code == 0

    _argv, staging, _env = js.calls[0]
    assert js.calls[0][0][:2] == ["npm", "ci"]
    config = Path(js.call("eslint")[0][js.call("eslint")[0].index("--config") + 1])
    assert config.parent != staging
    assert sorted(path.name for path in config.parent.iterdir()) == [
        "eslint.config.js",
        "package-lock.json",
        "package.json",
        "vitest.config.js",
        "vitest.setup.mjs",
    ]


# --- the JS halves, for real ----------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.skipif(
    os.environ.get("OSPREY_JS_CHECK") != "1", reason="fetches a JS toolchain: OSPREY_JS_CHECK=1"
)
def test_the_pinned_toolchain_runs_over_a_fixture_app(
    runner: CliRunner,
    lifecycle_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    app_dir = lifecycle_repo / "apps" / "js-check-app"
    shutil.copytree(FIXTURE_APP, app_dir)
    _list(lifecycle_repo, "js-check-app")
    before = _tree(app_dir)

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert _tree(app_dir) == before

    source = app_dir / "panel" / "readout.js"
    source.write_text(source.read_text(encoding="utf-8") + "var loose = 1;\n", encoding="utf-8")
    source_test = app_dir / "tests" / "readout.test.mjs"
    source_test.write_text(
        source_test.read_text(encoding="utf-8").replace(
            "'42');\n    expect(l", "'43');\n    expect(l"
        ),
        encoding="utf-8",
    )

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 1
    assert "[eslint]" in result.output
    assert "no-var" in result.output
    assert "[vitest]" in result.output
