"""The panels hold their reads and their emitted version to the compat table.

``[tool.osprey-panels.compat]`` in the package's ``pyproject.toml`` records the
facility-document headers the panels read and the Screen-IR version they write.
Two things are held to it:

- the code: ``document.READS`` and ``screen_ir.SCREEN_IR_VERSION`` equal the
  current table's ``reads`` and ``emits``;
- the history: every ``reads`` or ``emits`` value a prior table listed is still
  in the current table or in its ``retired`` list. Dropping a read or changing
  the emitted version is therefore a recorded retirement, while adding a read is
  additive and needs only the table edit.

Prior tables come from git, never from a literal branch name: on a pull request
(``GITHUB_BASE_REF`` set) the table at the merge-base with
``origin/$GITHUB_BASE_REF``, otherwise the table at the first parent ``HEAD^1``,
and in both cases the table at every ``osprey-panels-v*`` tag reachable from
``HEAD``. A prior revision with no table contributes nothing. In CI
(``CI=true``) history that cannot be read fails the test; elsewhere (an unpacked
sdist, a fresh single-commit clone) only the code is checked.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from osprey_panels.document import READS
from osprey_panels.screen_ir import SCREEN_IR_VERSION
from osprey_panels_tests._harness._paths import PACKAGE_ROOT

#: The package's own tag stream; every reachable tag's table is a prior table.
TAG_GLOB = "osprey-panels-v*"


class HistoryMissing(Exception):
    """The git history the prior tables come from cannot be read."""


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def compat_table(text: str) -> Mapping[str, object] | None:
    """Return the ``[tool.osprey-panels.compat]`` table of a pyproject, or None."""
    table = tomllib.loads(text).get("tool", {}).get("osprey-panels", {}).get("compat")
    return table if isinstance(table, Mapping) else None


def prior_revisions(package_root: Path, env: Mapping[str, str]) -> list[str]:
    """Return the commits whose tables are prior tables, base first.

    Raises:
        HistoryMissing: the base revision cannot be resolved.
    """
    base_ref = env.get("GITHUB_BASE_REF", "")
    if base_ref:
        found = _git(package_root, "merge-base", f"origin/{base_ref}", "HEAD")
        if found.returncode != 0 or not found.stdout.strip():
            raise HistoryMissing(
                f"no merge-base of origin/{base_ref} and HEAD: fetch the base branch"
                f" and full history (fetch-depth: 0). {found.stderr.strip()}"
            )
    else:
        found = _git(package_root, "rev-parse", "--verify", "--quiet", "HEAD^1^{commit}")
        if found.returncode != 0 or not found.stdout.strip():
            raise HistoryMissing(
                "HEAD has no first parent: check out full history (fetch-depth: 0)."
            )
    revisions = [found.stdout.strip()]
    tags = _git(package_root, "tag", "--merged", "HEAD", "--list", TAG_GLOB)
    if tags.returncode != 0:
        raise HistoryMissing(f"cannot list {TAG_GLOB} tags: {tags.stderr.strip()}")
    revisions += [f"refs/tags/{tag}" for tag in tags.stdout.split()]
    return revisions


def prior_tables(package_root: Path, revisions: Iterable[str]) -> dict[str, Mapping[str, object]]:
    """Return each revision's compat table; a revision without one is left out."""
    tables = {}
    for revision in revisions:
        shown = _git(package_root, "show", f"{revision}:./pyproject.toml")
        if shown.returncode != 0:
            continue  # the package did not exist at that revision
        table = compat_table(shown.stdout)
        if table is not None:
            tables[revision] = table
    return tables


def _values(table: Mapping[str, object]) -> set[str]:
    reads = table.get("reads", [])
    emits = table.get("emits")
    return {*reads} | ({emits} if isinstance(emits, str) else set())  # type: ignore[misc]


def violations(
    current: Mapping[str, object] | None,
    reads: frozenset[str],
    emits: str,
    priors: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Return every way the code and the prior tables disagree with ``current``."""
    if current is None:
        return ["pyproject.toml has no [tool.osprey-panels.compat] table"]
    problems = []
    table_reads = current.get("reads")
    if not isinstance(table_reads, list) or not all(isinstance(r, str) for r in table_reads):
        problems.append(f"compat `reads` is not a list of strings: {table_reads!r}")
    elif set(table_reads) != set(reads):
        problems.append(
            f"document.READS {sorted(reads)} differs from compat `reads` {sorted(table_reads)}"
        )
    if current.get("emits") != emits:
        problems.append(
            f"screen_ir.SCREEN_IR_VERSION {emits!r} differs from compat `emits`"
            f" {current.get('emits')!r}"
        )
    retired = current.get("retired")
    if not isinstance(retired, list) or not all(isinstance(r, str) for r in retired):
        problems.append(f"compat `retired` is not a list of strings: {retired!r}")
        retired = []
    kept = _values(current) | set(retired)
    for revision, table in priors.items():
        for value in sorted(_values(table) - kept):
            problems.append(
                f"{value!r}, listed at {revision}, is in neither the current table nor"
                " `retired`: dropping a read or changing `emits` is a recorded retirement"
            )
    return problems


def check(
    package_root: Path, reads: frozenset[str], emits: str, env: Mapping[str, str]
) -> list[str]:
    """Hold ``reads`` and ``emits`` and the package's history to its compat table."""
    current = compat_table((package_root / "pyproject.toml").read_text(encoding="utf-8"))
    try:
        priors = prior_tables(package_root, prior_revisions(package_root, env))
    except HistoryMissing as missing:
        if env.get("CI") == "true":
            return [f"missing history: {missing}"]
        priors = {}
    return violations(current, reads, emits, priors)


def test_the_code_and_its_history_match_the_compat_table() -> None:
    assert check(PACKAGE_ROOT, READS, SCREEN_IR_VERSION, os.environ) == []


# --- scratch repositories -------------------------------------------------------


def _table_text(reads: list[str] | None, emits: str = "1.1", retired: list[str] = ()) -> str:
    text = '[project]\nname = "osprey-panels"\n'
    if reads is not None:
        text += (
            "\n[tool.osprey-panels.compat]\n"
            f"reads = {list(reads)!r}\nemits = {emits!r}\nretired = {list(retired)!r}\n"
        ).replace("'", '"')
    return text


class Scratch:
    """A scratch repository holding ``packages/osprey-panels/pyproject.toml``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.package = root / "packages" / "osprey-panels"
        self.package.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        done = subprocess.run(
            [
                "git",
                "-c",
                "user.name=scratch",
                "-c",
                "user.email=scratch@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                "-C",
                str(self.root),
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    def commit(self, text: str, message: str = "c") -> str:
        (self.package / "pyproject.toml").write_text(text, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")


READ_4 = "osprey.facility.facility/4"
READ_3 = "osprey.facility.facility/3"
CI = {"CI": "true"}


def test_a_pull_request_reads_the_merge_base_with_its_base_branch(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    base = repo.commit(_table_text([READ_3, READ_4]))
    repo.git("update-ref", "refs/remotes/origin/main", base)
    repo.git("checkout", "-q", "-b", "feature")
    repo.commit(_table_text([READ_4], retired=[READ_3]))
    repo.commit(_table_text([READ_4]))  # drops the retirement: HEAD^1 alone would pass
    env = {**CI, "GITHUB_BASE_REF": "main"}

    assert prior_revisions(repo.package, env) == [base]
    problems = check(repo.package, frozenset({READ_4}), "1.1", env)
    assert len(problems) == 1 and READ_3 in problems[0]
    assert check(repo.package, frozenset({READ_4}), "1.1", CI) == []  # HEAD^1 retired it


def test_a_push_reads_the_first_parent_and_every_reachable_tag(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    tagged = repo.commit(_table_text([READ_3], emits="1.0"))
    repo.git("tag", "osprey-panels-v2026.1.0")
    repo.git("checkout", "-q", "-b", "side")
    repo.commit(_table_text([READ_3], emits="0.9"))
    repo.git("tag", "osprey-panels-v2026.0.1")  # not reachable from HEAD below
    repo.git("tag", "v2026.1.0")  # the framework's stream, not the panels'
    repo.git("checkout", "-q", "main")
    parent = repo.commit(_table_text([READ_4], retired=[READ_3, "1.0"]))
    repo.commit(_table_text([READ_4], retired=[READ_3, "1.0"]))

    assert prior_revisions(repo.package, CI) == [parent, "refs/tags/osprey-panels-v2026.1.0"]
    assert repo.git("rev-parse", "refs/tags/osprey-panels-v2026.1.0") == tagged
    assert check(repo.package, frozenset({READ_4}), "1.1", CI) == []

    repo.commit(_table_text([READ_4], retired=[READ_3]))  # the tag's emits "1.0" unretired
    problems = check(repo.package, frozenset({READ_4}), "1.1", CI)
    assert len(problems) == 1 and "'1.0'" in problems[0]
    assert "osprey-panels-v2026.1.0" in problems[0]


def test_ci_without_a_first_parent_fails(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_4]))

    with pytest.raises(HistoryMissing):
        prior_revisions(repo.package, CI)
    problems = check(repo.package, frozenset({READ_4}), "1.1", CI)
    assert len(problems) == 1 and problems[0].startswith("missing history")
    assert check(repo.package, frozenset({READ_4}), "1.1", {}) == []  # outside CI


def test_ci_without_the_base_branch_fails(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_4]))
    repo.commit(_table_text([READ_4]))

    problems = check(repo.package, frozenset({READ_4}), "1.1", {**CI, "GITHUB_BASE_REF": "main"})
    assert len(problems) == 1 and problems[0].startswith("missing history")


def test_a_base_without_the_table_contributes_no_prior_values(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    base = repo.commit(_table_text(None))
    repo.commit(_table_text([READ_4]))

    assert prior_revisions(repo.package, CI) == [base]
    assert prior_tables(repo.package, [base]) == {}
    assert check(repo.package, frozenset({READ_4}), "1.1", CI) == []


def test_a_base_without_the_package_contributes_no_prior_values(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    (tmp_path / "README").write_text("x", encoding="utf-8")
    repo.git("add", "README")
    repo.git("commit", "-q", "-m", "before the package")
    repo.commit(_table_text([READ_4]))

    assert check(repo.package, frozenset({READ_4}), "1.1", CI) == []


def test_adding_a_read_is_additive(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_4]))
    repo.commit(_table_text([READ_4, "osprey.facility.facility/5"]))

    reads = frozenset({READ_4, "osprey.facility.facility/5"})
    assert check(repo.package, reads, "1.1", CI) == []


def test_a_recorded_retirement_passes(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_4], emits="1.1"))
    repo.commit(_table_text(["osprey.facility.facility/5"], emits="1.2", retired=[READ_4, "1.1"]))

    assert check(repo.package, frozenset({"osprey.facility.facility/5"}), "1.2", CI) == []


def test_deleting_the_whole_table_fails(tmp_path: Path) -> None:
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_4]))
    repo.commit(_table_text(None))

    problems = check(repo.package, frozenset({READ_4}), "1.1", CI)
    assert problems == ["pyproject.toml has no [tool.osprey-panels.compat] table"]


def test_deleting_the_table_from_the_real_pyproject_fails(tmp_path: Path) -> None:
    text = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    header = "[tool.osprey-panels.compat]\n"
    before, _, after = text.partition(header)
    assert after, "the real pyproject has no compat table"
    stripped = before + after[after.index("\n\n") + 2 :]
    assert compat_table(text) is not None and compat_table(stripped) is None
    repo = Scratch(tmp_path)
    repo.commit(text)
    repo.commit(stripped)

    problems = check(repo.package, READS, SCREEN_IR_VERSION, CI)
    assert problems == ["pyproject.toml has no [tool.osprey-panels.compat] table"]


@pytest.mark.parametrize(
    ("reads", "emits", "needle"),
    [
        (frozenset({READ_4, READ_3}), "1.1", "document.READS"),
        (frozenset({READ_4}), "1.2", "SCREEN_IR_VERSION"),
    ],
    ids=["reads-differ", "emits-differs"],
)
def test_code_that_disagrees_with_the_table_fails(
    tmp_path: Path, reads: frozenset[str], emits: str, needle: str
) -> None:
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_4]))
    repo.commit(_table_text([READ_4]))

    problems = check(repo.package, reads, emits, CI)
    assert len(problems) == 1 and needle in problems[0]


def test_the_history_check_catches_a_drop__mutation_reads_and_table_edited_together(
    tmp_path: Path,
) -> None:
    """``READS`` and the table both drop a value the base carries, with no ``retired``."""
    repo = Scratch(tmp_path)
    repo.commit(_table_text([READ_3, READ_4]))
    repo.commit(_table_text([READ_4]))  # the mutation: the table edited with READS

    problems = check(repo.package, frozenset({READ_4}), "1.1", CI)
    assert len(problems) == 1 and repr(READ_3) in problems[0]
