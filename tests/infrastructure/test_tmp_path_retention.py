"""The suite keeps a test's temporary directory only when the test failed.

So no run leaves its passing tests' files behind for a later pytest process to
delete when that process exits. The proof runs a nested pytest under the
repository's own pytest configuration.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

from tests._nested_pytest import run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Keys of the outer run that must not reach an inner pytest: its worker id,
#: its diagnostics directory, and any options it was started with.
_OUTER_RUN_KEYS = ("PYTEST_XDIST_", "PYTEST_ADDOPTS", "OSPREY_CI_DIAG_DIR")


def test_only_a_failing_tests_temporary_directory_outlives_it(tmp_path: Path) -> None:
    seen = tmp_path / "seen"
    seen.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    (work / "test_retention.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            seen = Path({str(seen)!r})


            def test_a_passing_test(tmp_path):
                (tmp_path / "made").write_text("made")
                (seen / "passed").write_text(str(tmp_path))


            def test_a_failing_test(tmp_path):
                (tmp_path / "made").write_text("made")
                (seen / "failed").write_text(str(tmp_path))
                raise AssertionError("fails on purpose")


            def test_the_directories_left_behind():
                assert not Path((seen / "passed").read_text()).exists()
                assert Path((seen / "failed").read_text(), "made").exists()
            """
        )
    )

    env = {k: v for k, v in os.environ.items() if not k.startswith(_OUTER_RUN_KEYS)}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(_REPO_ROOT / "pyproject.toml"),
            "--rootdir",
            str(work),
            str(work / "test_retention.py"),
            "-q",
            "--color=no",
        ],
        cwd=work,
        env=env,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "1 failed, 2 passed" in result.stdout, output
