"""Tests for the zero-skip gate ci.yml's lanes run on their junit reports.

The gate is what turns a lane whose tests all skipped into a red, so each way a
report can fail it is checked, with the passing case as its negative control.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._by_path import load_module

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "junit_no_skips.py"
junit_no_skips = load_module("junit_no_skips", _MODULE_PATH)


def _report(path: Path, *suites: tuple[int, int]) -> Path:
    """Write a pytest-shaped junit report with one ``testsuite`` per ``(tests, skipped)``."""
    body = "".join(f'<testsuite name="pytest" tests="{t}" skipped="{s}"/>' for t, s in suites)
    path.write_text(f'<?xml version="1.0" encoding="utf-8"?><testsuites>{body}</testsuites>')
    return path


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_MODULE_PATH), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_report_with_no_skips_passes_and_prints_the_message(tmp_path: Path) -> None:
    report = _report(tmp_path / "r.xml", (5, 0))
    done = _run(report, "--message", "✅ every lane test ran")
    assert done.returncode == 0, done.stdout
    assert done.stdout.splitlines() == [f"{report}: 5 test(s), 0 skipped", "✅ every lane test ran"]


def test_any_skip_fails(tmp_path: Path) -> None:
    report = _report(tmp_path / "r.xml", (5, 1))
    done = _run(report)
    assert done.returncode == 1
    assert f"::error::{report} reports 1 skipped test(s)" in done.stdout
    assert junit_no_skips.DEFAULT_MESSAGE not in done.stdout


def test_skips_are_summed_over_every_suite(tmp_path: Path) -> None:
    report = _report(tmp_path / "r.xml", (3, 0), (2, 1))
    assert junit_no_skips.counts(report) == (5, 1)
    assert _run(report).returncode == 1


def test_an_empty_collection_fails(tmp_path: Path) -> None:
    """Zero tests satisfies a zero-skip check trivially, so it fails on its own."""
    report = _report(tmp_path / "r.xml", (0, 0))
    done = _run(report)
    assert done.returncode == 1
    assert f"::error::{report} collected no tests" in done.stdout


@pytest.mark.parametrize("content", [None, "", "not xml", '<testsuite tests="x"/>'])
def test_a_missing_empty_or_unreadable_report_fails(tmp_path: Path, content: str | None) -> None:
    """The inline gate this replaced raised on all four, which also exited non-zero."""
    report = tmp_path / "r.xml"
    if content is not None:
        report.write_text(content)
    done = _run(report)
    assert done.returncode == 1
    assert f"::error::{report}" in done.stdout


def test_every_report_is_checked_before_failing(tmp_path: Path) -> None:
    bad = _report(tmp_path / "bad.xml", (2, 2))
    good = _report(tmp_path / "good.xml", (2, 0))
    missing = tmp_path / "missing.xml"
    done = _run(bad, missing, good)
    assert done.returncode == 1
    assert f"::error::{bad} reports 2 skipped test(s)" in done.stdout
    assert f"::error::{missing} was never written" in done.stdout
    assert f"{good}: 2 test(s), 0 skipped" in done.stdout


def test_one_failing_report_fails_the_lane(tmp_path: Path) -> None:
    good = _report(tmp_path / "good.xml", (2, 0))
    bad = _report(tmp_path / "bad.xml", (2, 1))
    assert _run(good, bad).returncode == 1
    assert _run(good, good).returncode == 0


def test_allow_partial_skips_warns_on_a_partial_skip(tmp_path: Path) -> None:
    report = _report(tmp_path / "r.xml", (4, 1))
    done = _run(report, "--allow-partial-skips")
    assert done.returncode == 0, done.stdout
    assert f"::warning::{report} skipped 1 of 4 test(s)" in done.stdout


def test_allow_partial_skips_fails_a_report_that_skipped_in_full(tmp_path: Path) -> None:
    """The lane-wide gate's whole point: a module that skipped every test fails."""
    report = _report(tmp_path / "r.xml", (4, 4))
    done = _run(report, "--allow-partial-skips")
    assert done.returncode == 1
    assert f"::error::{report} skipped every one of its 4 test(s)" in done.stdout


@pytest.mark.parametrize("content", [None, "", '<testsuites><testsuite tests="0"/></testsuites>'])
def test_allow_partial_skips_still_fails_a_missing_or_empty_report(
    tmp_path: Path, content: str | None
) -> None:
    report = tmp_path / "r.xml"
    if content is not None:
        report.write_text(content)
    assert _run(report, "--allow-partial-skips").returncode == 1


def test_no_report_is_a_usage_error() -> None:
    assert _run().returncode == 2
