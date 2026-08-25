"""Suite for the docs split-accounting guard.

The guard exists so that no paragraph and no section heading quietly disappears
while the documentation is being cut apart and reassembled. A guard nobody has
watched fail proves nothing, so every check here has a negative control: a
synthetic two-file split that passes, the same split with a range removed, and
the same split with a heading that survives nowhere.

The last test is about the shipped table rather than the script: every
destination it names has to land inside the Target Tree.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "docs" / "check_split_accounting.py"
_TABLE = _REPO_ROOT / "scripts" / "docs" / "split_table.yml"
_TARGET_TREE_ROOTS = ("reference/", "architecture/", "contributing/", "how-to/")

# The synthetic page the fixture commits as the "old" file: an H1 (overlined)
# and two H2s, one of which the split moves to another page.
OLD_PAGE = """\
=====
Alpha
=====

Intro line about alpha.

Beta
====

Beta body line.

Gamma
=====

Gamma body line.
"""


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_split_accounting", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


class SyntheticSplit:
    """A throwaway git repo holding one old page and its new destinations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.docs_root = root / "docs" / "source"
        (self.docs_root / "old").mkdir(parents=True)
        (self.docs_root / "old" / "page.rst").write_text(OLD_PAGE, encoding="utf-8")
        for args in (["init", "-q"], ["add", "-A"]):
            subprocess.run(["git", "-C", str(root), *args], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "baseline",
            ],
            check=True,
        )
        # The tree as it looks after the split: Beta has moved to its own page.
        (self.docs_root / "old" / "page.rst").write_text(
            OLD_PAGE.replace("Beta\n====\n\nBeta body line.\n\n", ""), encoding="utf-8"
        )
        (self.docs_root / "reference").mkdir()
        (self.docs_root / "reference" / "beta.rst").write_text(
            "Beta\n====\n\nBeta body line.\n", encoding="utf-8"
        )


@pytest.fixture
def split(tmp_path: Path) -> SyntheticSplit:
    return SyntheticSplit(tmp_path)


def _table(**overrides) -> dict:
    table = {
        "baseline_ref": "HEAD",
        "sources": {
            "old/page.rst": {
                "ranges": [{"lines": "7-11", "dest": "reference/beta"}],
                "stays": ["1-6", "12-15"],
            }
        },
        "retired_headings": [],
    }
    table.update(overrides)
    return table


def _run(split: SyntheticSplit, table: dict, capsys) -> tuple[int, str]:
    path = split.root / "table.yml"
    path.write_text(yaml.safe_dump(table), encoding="utf-8")
    code = checker.main(["--table", str(path), "--docs-root", str(split.docs_root)])
    return code, capsys.readouterr().out


# ── the guard is green on a split that accounts for everything ───────────


def test_a_fully_accounted_split_passes(split, capsys):
    code, out = _run(split, _table(), capsys)
    assert code == 0, f"a split whose ranges and stays cover every line must pass:\n{out}"
    assert "OK" in out


# ── negative controls ────────────────────────────────────────────────────


def test_a_dropped_range_goes_red_and_names_the_file(split, capsys):
    table = _table()
    table["sources"]["old/page.rst"]["stays"] = ["1-6"]  # 12-15 assigned to nothing
    code, out = _run(split, table, capsys)
    assert code == 1, "lines belonging to no destination must fail the accounting"
    assert "old/page.rst" in out, f"the report must name the source file:\n{out}"
    assert "12" in out, f"the report must point at the unaccounted lines:\n{out}"


def test_a_vanished_heading_goes_red_until_it_is_retired(split, capsys):
    (split.docs_root / "reference" / "beta.rst").unlink()  # Beta now lives nowhere
    code, out = _run(split, _table(), capsys)
    assert code == 1, "an H2 that survives in no page and is not retired must fail"
    assert "Beta" in out, f"the report must name the vanished heading:\n{out}"

    code, out = _run(split, _table(retired_headings=["Beta"]), capsys)
    assert code == 0, f"a heading listed in retired_headings is a deliberate drop:\n{out}"


# ── the shipped table ────────────────────────────────────────────────────


def test_shipped_table_parses_and_targets_the_target_tree():
    table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))
    assert table["baseline_ref"], "the table must name the baseline commit it was measured against"
    assert table["sources"], "the table must list the pages being split"

    stray = []
    for source, spec in table["sources"].items():
        for item in spec.get("ranges") or []:
            dest = item.get("dest")
            if dest is None:
                assert item.get("retired"), (
                    f"{source}: a range with no destination must say why it is retired"
                )
                continue
            if not dest.startswith(_TARGET_TREE_ROOTS):
                stray.append(f"{source}: {item['lines']} -> {dest}")
    assert stray == [], (
        "every destination must sit under one of the Target Tree sections "
        f"({', '.join(_TARGET_TREE_ROOTS)}). Strays:\n" + "\n".join(stray)
    )


def test_only_restricts_the_check_to_the_named_sources():
    """`--only` must scope the run to the named table entries and reject unknown ones."""
    import yaml

    table_path = checker.REPO_ROOT / "scripts" / "docs" / "split_table.yml"
    table = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    first = next(iter(table["sources"]))
    problems, summary = checker.check(
        checker.REPO_ROOT / "scripts" / "docs" / "split_table.yml",
        checker.REPO_ROOT / "docs" / "source",
        None,
        [first],
    )
    assert "1 sources" in summary or first in problems
    with pytest.raises(SystemExit):
        checker.check(
            checker.REPO_ROOT / "scripts" / "docs" / "split_table.yml",
            checker.REPO_ROOT / "docs" / "source",
            None,
            ["how-to/does-not-exist.rst"],
        )
