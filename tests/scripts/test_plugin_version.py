"""Tests for the plugin CalVer bump script.

Two manifests have to carry the same version string, and the failure that matters is
the quiet one: a bump that lands in one file and not the other, or a rewrite that
reflows the JSON and turns every later diff into noise. So the manifests here are
literal text with the real key order, 2-space indent and trailing newline, and the
rewrite is asserted byte-for-byte against that text with only the version value
substituted. `--today` freezes the calendar so the month-rollover case is not a
test that passes for eleven months of the year.

No test reads or writes the repository's real manifests.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plugin_version.py"
_spec = importlib.util.spec_from_file_location("plugin_version", _MODULE_PATH)
assert _spec and _spec.loader
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

CLAUDE_MANIFEST = """\
{
  "name": "osprey",
  "version": "2026.9.0",
  "description": "Agent skills for developing and deploying OSPREY.",
  "author": {
    "name": "ALS Accelerator Physics Group",
    "url": "https://github.com/als-apg"
  },
  "license": "BSD-3-Clause",
  "keywords": [
    "osprey",
    "skills"
  ]
}
"""

CODEX_MANIFEST = """\
{
  "name": "osprey",
  "version": "2026.9.0",
  "description": "Agent skills for developing and deploying OSPREY.",
  "skills": "./skills/",
  "interface": {
    "displayName": "OSPREY",
    "shortDescription": "Agent skills for developing and deploying OSPREY."
  }
}
"""

TEXTS = dict(zip(pv.MANIFESTS, (CLAUDE_MANIFEST, CODEX_MANIFEST), strict=True))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A throwaway checkout holding just the two plugin manifests, both at 2026.9.0."""
    for path, text in TEXTS.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def versions(root: Path) -> list[str]:
    """The version string each manifest carries, in manifest order."""
    return [pv.read_versions(root)[path] for path in pv.MANIFESTS]


def run(root: Path, *args: str) -> int:
    """Invoke the script against *root* the way the command line would."""
    return pv.main(["--root", str(root), *args])


def skew(root: Path) -> None:
    """Leave the Codex manifest one month behind the Claude one."""
    (root / pv.MANIFESTS[1]).write_text(
        CODEX_MANIFEST.replace("2026.9.0", "2026.8.3"), encoding="utf-8"
    )


class TestShow:
    def test_it_prints_the_shared_version(self, root, capsys):
        assert run(root, "show") == 0
        assert capsys.readouterr().out == "2026.9.0\n"

    def test_skew_is_an_error_naming_both_files(self, root, capsys):
        skew(root)
        assert run(root, "show") == 1
        err = capsys.readouterr().err
        assert "2026.9.0" in err and "2026.8.3" in err
        assert str(pv.MANIFESTS[0]) in err and str(pv.MANIFESTS[1]) in err

    def test_a_missing_manifest_is_an_error_not_a_traceback(self, root, capsys):
        (root / pv.MANIFESTS[1]).unlink()
        assert run(root, "show") == 1
        assert "cannot read the plugin manifests" in capsys.readouterr().err


class TestBump:
    def test_the_same_month_advances_the_serial(self, root, capsys):
        assert run(root, "bump", "--today", "2026-09-14") == 0
        assert versions(root) == ["2026.9.1", "2026.9.1"]
        assert capsys.readouterr().out == "2026.9.0 -> 2026.9.1\n"

    def test_a_new_month_resets_the_serial_to_zero(self, root):
        assert run(root, "bump", "--today", "2026-10-01") == 0
        assert versions(root) == ["2026.10.0", "2026.10.0"]

    def test_a_new_year_resets_the_serial_to_zero(self, root):
        assert run(root, "bump", "--today", "2027-01-20") == 0
        assert versions(root) == ["2027.1.0", "2027.1.0"]

    def test_the_month_is_never_zero_padded(self, root):
        """`2027.3.0`, not `2027.03.0` — the CI gate compares the string, not a tuple."""
        assert run(root, "bump", "--today", "2027-03-09") == 0
        assert versions(root) == ["2027.3.0", "2027.3.0"]

    def test_repeated_bumps_keep_counting_within_the_month(self, root):
        for _ in range(3):
            assert run(root, "bump", "--today", "2026-09-14") == 0
        assert versions(root) == ["2026.9.3", "2026.9.3"]


class TestSet:
    def test_it_writes_the_given_version_to_both_manifests(self, root, capsys):
        assert run(root, "bump", "--set", "2027.4.7") == 0
        assert versions(root) == ["2027.4.7", "2027.4.7"]
        assert capsys.readouterr().out == "2026.9.0 -> 2027.4.7\n"

    @pytest.mark.parametrize(
        "value",
        ["2026.9", "26.9.0", "v2026.9.0", "2026.9.0-rc1", "2026.9.0.1", "", "next"],
    )
    def test_a_non_calver_value_is_rejected_with_exit_two(self, root, capsys, value):
        assert run(root, "bump", "--set", value) == 2
        assert "not CalVer" in capsys.readouterr().err
        assert versions(root) == ["2026.9.0", "2026.9.0"]

    def test_a_padded_month_is_tolerated_and_the_next_bump_normalizes_it(self, root):
        """Documented tolerance: `--set` allows `2026.09.4`, generation never emits it."""
        assert run(root, "bump", "--set", "2026.09.4") == 0
        assert versions(root) == ["2026.09.4", "2026.09.4"]
        assert run(root, "bump", "--today", "2026-09-30") == 0
        assert versions(root) == ["2026.9.5", "2026.9.5"]

    def test_it_repairs_skew_that_a_plain_bump_refuses_to_guess_at(self, root, capsys):
        skew(root)
        assert run(root, "bump", "--today", "2026-09-14") == 1
        assert run(root, "bump", "--set", "2026.9.1") == 0
        assert versions(root) == ["2026.9.1", "2026.9.1"]
        assert capsys.readouterr().out.endswith("-> 2026.9.1\n")


class TestRewriteFidelity:
    def test_only_the_version_value_changes(self, root):
        assert run(root, "bump", "--today", "2026-09-14") == 0
        for path, text in TEXTS.items():
            expected = text.replace('"version": "2026.9.0"', '"version": "2026.9.1"')
            assert (root / path).read_text(encoding="utf-8") == expected

    def test_the_trailing_newline_survives(self, root):
        assert run(root, "bump", "--set", "2027.1.0") == 0
        for path in pv.MANIFESTS:
            body = (root / path).read_text(encoding="utf-8")
            assert body.endswith("}\n") and not body.endswith("\n\n")


class TestNextVersion:
    @pytest.mark.parametrize(
        ("current", "today", "expected"),
        [
            ("2026.9.0", "2026-09-01", "2026.9.1"),
            ("2026.9.11", "2026-09-30", "2026.9.12"),
            ("2026.9.4", "2026-10-01", "2026.10.0"),
            ("2026.12.4", "2027-01-01", "2027.1.0"),
            ("2026.10.0", "2026-09-30", "2026.9.0"),
        ],
    )
    def test_the_calendar_decides(self, current, today, expected):
        """The last row is a clock that went backwards: the date wins, not the file."""
        assert pv.next_version(current, date.fromisoformat(today)) == expected
