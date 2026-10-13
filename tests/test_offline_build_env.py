"""What ``offline_build_env`` promises the tests that run a real build."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from tests._builds import install_osprey_into, uv_path

# Any request a subprocess makes goes to a port nothing listens on.
NO_ROUTE = dict.fromkeys(
    ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"),
    "http://127.0.0.1:9",
)
NO_ROUTE["NO_PROXY"] = NO_ROUTE["no_proxy"] = ""


@pytest.mark.slow
def test_a_build_installs_osprey_with_no_route_to_an_index(offline_build_env, tmp_path):
    if uv_path() is None:
        pytest.skip("uv is not installed; a build installs with pip, which has no cache-only mode")
    venv = tmp_path / "venv"
    installed = install_osprey_into(venv, {**os.environ, **offline_build_env, **NO_ROUTE})
    assert installed.returncode == 0, installed.stdout + installed.stderr
    probe = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import osprey"], capture_output=True, text=True
    )
    assert probe.returncode == 0, probe.stderr


TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent

#: Modules the pin reads and lets build with the network on, each with the reason.
BUILDS_WITH_THE_NETWORK = {
    "tests/fixtures/explicit_config/freeze.py": (
        "run by hand to regenerate fixtures; never collected"
    ),
}


def _takes_offline_build_env(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            argument.arg == "offline_build_env"
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        )
        for node in ast.walk(tree)
    )


def _is_word(node: ast.expr, word: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == word


def _is_launcher(node: ast.expr) -> bool:
    return _is_word(node, "osprey") or ast.unparse(node).endswith(("osprey_bin", "osprey_bin)"))


def _build_argv(node: ast.AST) -> ast.List | ast.Tuple | None:
    """The argument list of an out-of-process ``osprey build`` that ``node`` is, if any."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if (
            name == "_run_cli"
            and node.args
            and isinstance(node.args[0], ast.List)
            and node.args[0].elts
            and _is_word(node.args[0].elts[0], "build")
        ):
            return node.args[0]
    if isinstance(node, ast.List | ast.Tuple):
        words = node.elts
        if any(
            _is_word(word, "build") and _is_launcher(before)
            for before, word in zip(words, words[1:], strict=False)
        ):
            return node
    return None


def _unguarded_builds(source: str) -> list[int]:
    """The line of every out-of-process build in ``source`` that may reach an index.

    A build is guarded when its argument list skips the install
    (``--skip-deps``) or when the module takes the ``offline_build_env``
    fixture, which some of them hand to the subprocess through a helper.
    """
    # A module that spells no build word or no launcher holds no build, so
    # it is not parsed.
    names_build = '"build"' in source or "'build'" in source
    names_launcher = any(
        word in source for word in ("_run_cli", "osprey_bin", '"osprey"', "'osprey'")
    )
    if not (names_build and names_launcher):
        return []
    tree = ast.parse(source)
    if _takes_offline_build_env(tree):
        return []
    found = []
    for node in ast.walk(tree):
        argv = _build_argv(node)
        if argv is not None and not any(_is_word(word, "--skip-deps") for word in argv.elts):
            found.append(argv.lineno)
    return sorted(found)


def _test_modules() -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(TESTS_ROOT.rglob("*.py"))
    }


def test_every_real_build_under_tests_takes_the_offline_environment(
    sources: dict[str, str] | None = None,
) -> None:
    """A test that runs ``osprey build`` with the install either skips it or installs offline.

    Read from source, so a new test that forgets both is red rather than a
    build that reaches the package index on whichever host runs it.
    """
    modules = _test_modules() if sources is None else sources
    unguarded = [
        f"{name}:{line}"
        for name, source in modules.items()
        if name not in BUILDS_WITH_THE_NETWORK
        for line in _unguarded_builds(source)
    ]
    assert unguarded == [], (
        f"these builds install with the network on: {unguarded}. Pass `--skip-deps`, or take "
        f"the `offline_build_env` fixture and hand its environment to the build."
    )


UNGUARDED_RUN_CLI = 'def test_it(project):\n    _run_cli(["build"], project)\n'
UNGUARDED_SUBPROCESS = (
    'def test_it(osprey_bin, r):\n    subprocess.run([str(osprey_bin), "build", "--repo", r])\n'
)


@pytest.mark.parametrize(
    "source", [UNGUARDED_RUN_CLI, UNGUARDED_SUBPROCESS], ids=["run-cli", "subprocess"]
)
def test_every_real_build_under_tests_takes_the_offline_environment__mutation_builds_unguarded(
    source: str,
) -> None:
    with pytest.raises(AssertionError):
        test_every_real_build_under_tests_takes_the_offline_environment(
            {"tests/test_planted.py": source}
        )


@pytest.mark.parametrize(
    "source",
    [
        UNGUARDED_RUN_CLI.replace('["build"]', '["build", "--skip-deps"]'),
        UNGUARDED_SUBPROCESS.replace('"--repo"', '"--skip-deps", "--repo"'),
        UNGUARDED_RUN_CLI.replace("(project)", "(project, offline_build_env)"),
        UNGUARDED_SUBPROCESS.replace("(osprey_bin, r)", "(osprey_bin, r, offline_build_env)"),
    ],
    ids=["run-cli-skips", "subprocess-skips", "run-cli-offline", "subprocess-offline"],
)
def test_every_real_build_under_tests_takes_the_offline_environment__guarded_builds_pass(
    source: str,
) -> None:
    test_every_real_build_under_tests_takes_the_offline_environment(
        {"tests/test_planted.py": source}
    )


def test_every_real_build_under_tests_takes_the_offline_environment__other_builds_do_not_count() -> (
    None
):
    planted = (
        "def test_it(tmp_path):\n"
        '    keys = [".osprey-manifest.json", "build"]\n'
        '    subprocess.run(["docker", "build", "--tag", "probe", str(tmp_path)])\n'
    )
    test_every_real_build_under_tests_takes_the_offline_environment(
        {"tests/test_planted.py": planted}
    )
