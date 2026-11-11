"""Tests for the pytest plugin that fails ci.yml's skip-gated lanes on any skipped test.

The plugin is what turns a lane whose tests all skipped into a red, so each way
a session can skip is run through a real pytest, with xdist where the count has
to cross from the workers to the controller, and the passing case as the
negative control. Each inner session runs in its own directory with an empty
ini, so the repository's own pytest configuration plays no part.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "pytest_plugins"

_PASSING = "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n"
_ONE_SKIP = "import pytest\n\n\ndef test_ran():\n    pass\n\n\n@pytest.mark.skip\ndef test_skipped():\n    pass\n"
_ALL_SKIP = "import pytest\n\npytestmark = pytest.mark.skip\n\n\ndef test_a():\n    pass\n\n\ndef test_b():\n    pass\n"
_MODULE_SKIP = 'import pytest\n\npytest.skip("no runtime", allow_module_level=True)\n'
_XFAIL = "import pytest\n\n\n@pytest.mark.xfail\ndef test_known():\n    assert False\n"
_FAILING = "def test_red():\n    assert False\n"


def _session(
    tmp_path: Path, modules: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    """Run a pytest session over *modules* with the plugin loaded as ci.yml loads it."""
    for name, body in modules.items():
        (tmp_path / name).write_text(body)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    env = {**os.environ, "PYTHONPATH": str(_PLUGIN_DIR)}
    env.pop("PYTEST_ADDOPTS", None)
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no_skips",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
    ]
    return subprocess.run(
        [*argv, *args, *modules],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_session_with_no_skips_passes_and_prints_the_message(tmp_path: Path) -> None:
    done = _session(
        tmp_path, {"test_ok.py": _PASSING}, "--no-skips-message", "✅ every lane test ran"
    )
    assert done.returncode == 0, done.stdout
    assert "no-skips gate: 2 test(s), 0 skipped" in done.stdout
    assert "✅ every lane test ran" in done.stdout


@pytest.mark.parametrize("xdist", [False, True], ids=["serial", "xdist"])
@pytest.mark.parametrize(
    ("modules", "skipped"),
    [
        ({"test_ok.py": _PASSING, "test_skip.py": _ONE_SKIP}, 1),
        ({"test_all.py": _ALL_SKIP}, 2),
        ({"test_ok.py": _PASSING, "test_mod.py": _MODULE_SKIP}, 1),
        ({"test_ok.py": _PASSING, "test_xfail.py": _XFAIL}, 1),
    ],
    ids=["one-skip", "all-skip", "module-skip", "xfail"],
)
def test_any_skip_fails_the_session(
    tmp_path: Path, modules: dict[str, str], skipped: int, xdist: bool
) -> None:
    """Under xdist the skips happen in the workers and the controller still sees them."""
    done = _session(tmp_path, modules, *(["-n", "2"] if xdist else []))
    assert done.returncode == 1, done.stdout
    assert f"::error::{skipped} skipped test(s)" in done.stdout
    assert "every test ran; none skipped" not in done.stdout


def test_a_failing_session_keeps_its_own_exit_status(tmp_path: Path) -> None:
    done = _session(tmp_path, {"test_red.py": _FAILING, "test_skip.py": _ONE_SKIP})
    assert done.returncode == 1, done.stdout
    assert "::error::1 skipped test(s)" in done.stdout


@pytest.mark.parametrize("args", [(), ("-k", "nothing_matches")], ids=["no-files", "deselected"])
def test_an_empty_collection_stays_red_and_says_so(tmp_path: Path, args: tuple[str, ...]) -> None:
    """Zero tests satisfies a zero-skip check trivially; pytest's exit 5 keeps it red."""
    modules = {"test_ok.py": _PASSING} if args else {"test_empty.py": ""}
    done = _session(tmp_path, modules, *args)
    assert done.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, done.stdout
    assert "::error::the session collected no tests" in done.stdout


def test_partial_skips_warn_when_allowed(tmp_path: Path) -> None:
    done = _session(
        tmp_path, {"test_ok.py": _PASSING, "test_skip.py": _ONE_SKIP}, "--no-skips-allow-partial"
    )
    assert done.returncode == 0, done.stdout
    assert "::warning::1 of 4 test(s) skipped" in done.stdout


def test_a_session_that_skipped_everything_fails_even_when_partial_skips_are_allowed(
    tmp_path: Path,
) -> None:
    done = _session(tmp_path, {"test_all.py": _ALL_SKIP}, "--no-skips-allow-partial")
    assert done.returncode == 1, done.stdout
    assert "::error::every one of the 2 test(s) skipped" in done.stdout


def test_the_annotation_names_at_most_ten_skips(tmp_path: Path) -> None:
    many = "import pytest\n\n\n@pytest.mark.parametrize('i', range(12))\n@pytest.mark.skip\ndef test_s(i):\n    pass\n"
    done = _session(tmp_path, {"test_many.py": many})
    assert done.returncode == 1, done.stdout
    (error,) = [line for line in done.stdout.splitlines() if line.startswith("::error::")]
    assert error.endswith(" and 2 more"), error
    assert error.count("test_many.py::test_s[") == 10, error
