"""Tests for the plugin CalVer bump script.

Two manifests have to carry the same version string, and the failure that matters is
the quiet one: a bump that lands in one file and not the other, or a rewrite that
reflows the JSON and turns every later diff into noise. So the manifests here are
literal text with the real key order, 2-space indent and trailing newline, and the
rewrite is asserted byte-for-byte against that text with only the version value
substituted. `--today` freezes the calendar so the month-rollover case is not a
test that passes for eleven months of the year.

No test reads or writes the repository's real manifests.

The ``check`` gate is exercised against a real scratch git repository, because its
answers are git's own: the diff runs against the merge base, the versions are compared
against the base tip, and a table of canned answers would only restate those two halves.
No test reads the real checkout's history.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "plugin_version.py"
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


SKILL = Path("plugins/osprey/skills/x/SKILL.md")
MARKETPLACE = Path(".claude-plugin/marketplace.json")
TREE_FILES = {
    "plugins": SKILL,
    ".claude-plugin": MARKETPLACE,
    ".agents/plugins": Path(".agents/plugins/marketplace.json"),
}


def git(repo: Path, *args: str) -> str:
    """Run git in *repo*, failing the test on a non-zero exit; return stdout."""
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def write(repo: Path, path: Path, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def commit(repo: Path, message: str, files: dict[Path, str] | None = None) -> None:
    """Write *files* (if any) and commit everything on the current branch."""
    for path, text in (files or {}).items():
        write(repo, path, text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)


def make_repo(path: Path, *, manifests: bool = True) -> Path:
    """A scratch repository on ``main`` whose first commit holds the plugin tree."""
    git_init = ["git", "init", "-q", "-b", "main", str(path)]
    subprocess.run(git_init, check=True, capture_output=True)
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    git(path, "config", "maintenance.auto", "false")
    files = {SKILL: "skill\n", MARKETPLACE: "{}\n", Path("README.md"): "readme\n"}
    if manifests:
        files.update(TEXTS)
    commit(path, "initial", files)
    return path


@pytest.fixture(scope="module")
def repo_template(tmp_path_factory) -> Path:
    """One scratch repository built per module; every ``repo`` is a copy of it."""
    return make_repo(tmp_path_factory.mktemp("template"))


@pytest.fixture
def repo(root: Path, repo_template: Path) -> Path:
    """Both manifests at 2026.9.0, a skill and a marketplace file, committed on ``main``."""
    shutil.copytree(repo_template, root, dirs_exist_ok=True)
    return root


def on_work(repo: Path, files: dict[Path, str] | None = None, *, bump: bool = False) -> None:
    """Branch ``work`` off the current commit and commit *files* on it, bumped if asked."""
    git(repo, "checkout", "-q", "-b", "work")
    for path, text in (files or {}).items():
        write(repo, path, text)
    if bump:
        assert run(repo, "bump", "--today", "2026-09-15") == 0
    commit(repo, "work")


def check(repo: Path, capsys, base: str = "main") -> tuple[int, str, str]:
    """Run ``check --base`` against *repo*; return the code, stdout and stderr."""
    capsys.readouterr()
    code = run(repo, "check", "--base", base)
    out, err = capsys.readouterr()
    return code, out, err


class TestCheck:
    def test_the_sentences_are_the_lint_steps_own(self):
        assert pv.PLUGIN_TREE == ("plugins", ".claude-plugin", ".agents/plugins")
        assert pv.UNTOUCHED == "plugin tree untouched — no version bump required"
        assert pv.NEW_MANIFEST == "new plugin manifest — no bump required"
        assert pv.UNCHANGED == (
            "plugin tree changed without a version bump — run python scripts/plugin_version.py bump"
        )

    def test_an_untouched_tree_owes_no_bump(self, repo, capsys):
        on_work(repo, {Path("README.md"): "changed\n"})
        assert check(repo, capsys) == (0, pv.UNTOUCHED + "\n", "")

    def test_an_untouched_tree_is_answered_before_any_manifest_is_read(self, repo, capsys):
        on_work(repo, {Path("README.md"): "changed\n"})
        skew(repo)
        assert check(repo, capsys) == (0, pv.UNTOUCHED + "\n", "")

    def test_a_changed_tree_at_the_base_version_is_refused(self, repo, capsys):
        on_work(repo, {SKILL: "changed\n"})
        assert check(repo, capsys) == (1, pv.UNCHANGED + "\n", "")

    def test_a_changed_tree_with_a_bump_passes(self, repo, capsys):
        on_work(repo, {SKILL: "changed\n"}, bump=True)
        assert check(repo, capsys) == (0, "plugin version 2026.9.0 -> 2026.9.1\n", "")

    @pytest.mark.parametrize("tree", pv.PLUGIN_TREE)
    def test_every_plugin_tree_path_owes_a_bump(self, repo, capsys, tree):
        on_work(repo, {TREE_FILES[tree]: "changed\n"})
        assert check(repo, capsys)[:2] == (1, pv.UNCHANGED + "\n")

    def test_a_base_without_a_manifest_is_a_new_plugin(self, tmp_path, capsys):
        repo = make_repo(tmp_path, manifests=False)
        on_work(repo, {**TEXTS, SKILL: "changed\n"})
        assert check(repo, capsys) == (0, pv.NEW_MANIFEST + "\n", "")

    def test_a_version_the_base_tip_minted_after_the_branch_point_is_refused(self, repo, capsys):
        """The diff runs against the merge base; the version against the base tip."""
        on_work(repo, {SKILL: "changed\n"}, bump=True)
        git(repo, "checkout", "-q", "main")
        assert run(repo, "bump", "--today", "2026-09-15") == 0
        commit(repo, "main mints 2026.9.1")
        git(repo, "checkout", "-q", "work")
        assert check(repo, capsys) == (1, pv.UNCHANGED + "\n", "")

    def test_skewed_head_manifests_are_refused(self, repo, capsys):
        on_work(repo, {SKILL: "changed\n"})
        skew(repo)
        code, out, err = check(repo, capsys)
        assert (code, out) == (1, "")
        assert "version skew" in err

    def test_a_base_manifest_that_is_not_json_is_an_error(self, repo, capsys):
        write(repo, pv.MANIFESTS[0], "not json\n")
        commit(repo, "break the manifest")
        on_work(repo, {**TEXTS, SKILL: "changed\n"})
        code, out, err = check(repo, capsys)
        assert (code, out) == (1, "")
        assert str(pv.MANIFESTS[0]) in err and "at main" in err

    def test_an_unresolvable_base_never_runs_the_check(self, repo, capsys):
        on_work(repo, {SKILL: "changed\n"})
        code, out, err = check(repo, capsys, "no/such-ref")
        assert (code, out) == (2, "")
        assert "not found" in err

    def test_unrelated_histories_never_run_the_check(self, repo, capsys):
        git(repo, "checkout", "-q", "--orphan", "orphan")
        commit(repo, "unrelated root")
        git(repo, "checkout", "-q", "main")
        on_work(repo, {SKILL: "changed\n"})
        code, out, err = check(repo, capsys, "orphan")
        assert (code, out) == (2, "")
        assert "no common ancestor" in err

    @pytest.mark.parametrize("bump", [False, True])
    def test_a_sha_and_a_relative_ref_answer_as_the_branch_name(self, repo, capsys, bump):
        on_work(repo, {SKILL: "changed\n"}, bump=bump)
        by_name = check(repo, capsys, "main")
        assert check(repo, capsys, git(repo, "rev-parse", "main")) == by_name
        assert check(repo, capsys, "HEAD~1") == by_name

    def test_a_plain_branch_name_prefers_the_remote_copy(self, repo, capsys):
        """The changelog gate's resolution: ``main`` reads ``origin/main`` when both exist."""
        on_work(repo, {SKILL: "changed\n"}, bump=True)
        git(repo, "checkout", "-q", "main")
        assert run(repo, "bump", "--today", "2026-09-15") == 0
        commit(repo, "origin mints 2026.9.1")
        git(repo, "update-ref", "refs/remotes/origin/main", "main")
        git(repo, "reset", "-q", "--hard", "HEAD~1")
        git(repo, "checkout", "-q", "work")
        assert check(repo, capsys, "heads/main")[:2] == (
            0,
            "plugin version 2026.9.0 -> 2026.9.1\n",
        )
        assert check(repo, capsys, "main") == (1, pv.UNCHANGED + "\n", "")

    def test_base_is_required(self, repo, capsys):
        with pytest.raises(SystemExit) as exit_info:
            run(repo, "check")
        assert exit_info.value.code == 2
        assert "--base" in capsys.readouterr().err


def test_the_release_skill_names_the_refusal_the_gate_prints():
    """The skill's failure-mode row quotes the refusal, so the two cannot drift apart."""
    clause = pv.UNCHANGED.split(" — ")[0]
    skill = (_REPO_ROOT / "plugins/osprey/skills/release/SKILL.md").read_text(encoding="utf-8")
    assert clause in skill
