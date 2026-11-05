"""Suite for the docs snippet-coverage report and its ratchet.

``scripts/docs/snippet_coverage.py`` counts every code block on every docs page
and, through the ledger ``scripts/docs/snippet_coverage.yml``, which of them a
test reads. Its ``--check`` is a ratchet: a page's untested count may shrink,
never grow. This module is the CI hook for it, in three parts:

* the scanner against a synthetic ``docs/source`` holding every opener form, so
  the counts are pinned by construction rather than by today's pages;
* the shipped ledger and baseline against the live tree: every claim names a
  real test and reads at least one block, no pending claim has landed, and no
  page is over (or under) its allowance;
* negative controls: a planted untested block, a stale allowance, a new page
  with an untested block and a landed pending test each fail.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from tests._by_path import load_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "docs" / "snippet_coverage.py"
_LEDGER = _REPO_ROOT / "scripts" / "docs" / "snippet_coverage.yml"


@lru_cache(maxsize=1)
def _coverage() -> ModuleType:
    """The script, loaded by path; registered because it defines dataclasses."""
    return load_module("snippet_coverage", _SCRIPT, register=True)


# -- the scanner, on a synthetic tree -------------------------------------------

#: One page with every form the scanner must recognise, and the lookalikes it
#: must not count: a prose ``.. note::``, a ``::`` with nothing under it.
_PAGE = """\
Guide
=====

Intro::

    literal line one
    literal line two

.. code-block:: bash
   :caption: Run it

   osprey build

.. code::

   plain text

.. sourcecode:: python

   print("hi")

.. note::

   A note that holds a block:

   .. code-block:: yaml

      key: value

   Prose after the nested block, back in the note.

.. tab-set::

   .. tab-item:: EPICS

      .. code-block:: console

         $ osprey up

.. literalinclude:: ../examples/demo.py
   :language: python

.. note:: Only prose here.

A paragraph that ends in colons with nothing under it::

Next paragraph.
"""


def _line(text: str, needle: str) -> int:
    return next(n for n, line in enumerate(text.splitlines(), start=1) if needle in line)


def _tree(root: Path) -> Path:
    """A repo root whose ``docs/source`` holds the synthetic page, an include and an empty page."""
    source = root / "docs" / "source"
    (source / "guide").mkdir(parents=True)
    (source / "examples").mkdir()
    (source / "guide" / "page.rst").write_text(_PAGE, encoding="utf-8")
    (source / "examples" / "demo.py").write_text('print("demo")\n', encoding="utf-8")
    (source / "empty.rst").write_text("Empty\n=====\n\nNo blocks.\n", encoding="utf-8")
    return root


def test_the_scanner_finds_every_opener_form(tmp_path):
    tree = _coverage().scan_tree(_tree(tmp_path) / "docs" / "source")

    found = [(b.page, b.line, b.language, b.include) for b in tree["guide/page.rst"]]

    page = "guide/page.rst"
    assert found == [
        (page, _line(_PAGE, "literal line one"), "literal", None),
        (page, _line(_PAGE, "osprey build"), "bash", None),
        (page, _line(_PAGE, "plain text"), "text", None),
        (page, _line(_PAGE, 'print("hi")'), "python", None),
        (page, _line(_PAGE, "key: value"), "yaml", None),
        (page, _line(_PAGE, "$ osprey up"), "console", None),
        (page, _line(_PAGE, "literalinclude::"), "literalinclude", "examples/demo.py"),
    ]
    assert tree["empty.rst"] == []
    assert Counter(b.language for blocks in tree.values() for b in blocks) == {
        "literal": 1,
        "bash": 1,
        "text": 1,
        "python": 1,
        "yaml": 1,
        "console": 1,
        "literalinclude": 1,
    }


def test_block_text_skips_options_and_stops_at_the_enclosing_indent(tmp_path):
    blocks = {
        b.language: b.text
        for b in _coverage().scan_tree(_tree(tmp_path) / "docs" / "source")["guide/page.rst"]
    }

    assert blocks["bash"] == "osprey build"
    assert blocks["yaml"] == "key: value"
    assert blocks["literal"] == "literal line one\nliteral line two"
    assert blocks["literalinclude"] == 'print("demo")\n'


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**/*.rst", "index.rst", True),
        ("**/*.rst", "how-to/a/b.rst", True),
        ("how-to/*.rst", "how-to/a/b.rst", False),
        ("how-to/*.rst", "how-to/b.rst", True),
        ("how-to/control-systems/examples/*.py", "how-to/control-systems/examples/x.py", True),
        ("reference/cli.rst", "reference/cli.rst.bak", False),
    ],
)
def test_globs_keep_a_single_star_inside_one_directory(pattern, path, expected):
    assert _coverage().glob_match(pattern, path) is expected


# -- the shipped ledger and baseline, on the live tree -------------------------


@lru_cache(maxsize=1)
def _live():
    module = _coverage()
    tree = module.scan_tree(_REPO_ROOT / "docs" / "source")
    live, pending = module.load_ledger()
    return module, tree, live, pending


def test_every_live_claim_names_a_real_test_and_reads_a_block():
    module, tree, live, _ = _live()

    assert live, "the ledger has no claims; the file moved or failed to parse"
    assert module.ledger_problems(live, tree, _REPO_ROOT) == []


def test_every_claimed_page_or_include_exists():
    """Live claims only: a pending claim may name a page or example its PR adds."""
    _, tree, live, _ = _live()
    source = _REPO_ROOT / "docs" / "source"
    files = [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()]
    module = _coverage()

    unmatched = [
        f"{claim.test}: {pattern}"
        for claim in live
        for pattern in claim.pages
        if not any(module.glob_match(pattern, path) for path in files)
    ]

    assert unmatched == [], "ledger page patterns that match no file under docs/source"


def test_every_claimed_language_is_one_the_scanner_produces():
    module, tree, live, pending = _live()
    vocabulary = {b.language for blocks in tree.values() for b in blocks}
    vocabulary |= {"text", "literal", "literalinclude"}

    unknown = sorted(
        f"{claim.test}: {language}"
        for claim in live + pending
        for language in claim.languages
        if language not in vocabulary
    )

    assert unknown == []
    assert {claim.kind for claim in live + pending} <= set(module.KINDS)


def test_no_pending_claim_has_landed():
    module, _, _, pending = _live()

    assert all(isinstance(claim.pr, int) for claim in pending), "every pending claim names its PR"
    assert module.pending_problems(pending, _REPO_ROOT) == []


def test_every_not_counted_test_exists():
    document = yaml.safe_load(_LEDGER.read_text(encoding="utf-8"))

    missing = [e["test"] for e in document["not_counted"] if not (_REPO_ROOT / e["test"]).is_file()]

    assert missing == [], "stale `not_counted:` entries: the test is gone"


def test_no_page_has_more_untested_blocks_than_its_baseline_allows():
    module, tree, live, pending = _live()
    rows = module.coverage(tree, live, pending)

    problems = module.ratchet_problems(rows, module.load_baseline())

    if problems:
        pytest.fail(
            "\n".join([*problems, "", module.FIX_GREW, "", module.FIX_STALE]), pytrace=False
        )


# -- negative controls -----------------------------------------------------------


def _repo(root: Path, *, allowed: dict[str, int], pending: list[dict] | None = None) -> list[str]:
    """A synthetic repo with one live claim, and the CLI flags that point the script at it."""
    _tree(root)
    (root / "tests" / "docs").mkdir(parents=True)
    (root / "tests" / "docs" / "test_demo.py").write_text("", encoding="utf-8")
    ledger = {
        "tested": [
            {
                "test": "tests/docs/test_demo.py",
                "pages": ["guide/*.rst"],
                "languages": ["bash"],
                "kind": "checks",
                "what": "demo",
            }
        ],
        "pending": pending or [],
    }
    (root / "ledger.yml").write_text(yaml.safe_dump(ledger), encoding="utf-8")
    (root / "baseline.json").write_text(json.dumps({"untested": allowed}), encoding="utf-8")
    return ["--root", str(root), "--ledger", str(root / "ledger.yml")] + [
        "--baseline",
        str(root / "baseline.json"),
    ]


def test_the_check_passes_at_the_allowance(tmp_path, capsys):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6})

    assert _coverage().main(["--check", *flags]) == 0
    assert "snippet coverage: ok" in capsys.readouterr().out


def test_a_planted_untested_block_fails_the_check(tmp_path, capsys):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6})
    page = tmp_path / "docs" / "source" / "guide" / "page.rst"
    page.write_text(page.read_text() + "\n.. code-block:: yaml\n\n   planted: true\n")

    assert _coverage().main(["--check", *flags]) == 1
    out = capsys.readouterr().out
    assert "GREW   guide/page.rst: 7 untested blocks, the baseline allows 6" in out
    assert "add or extend a test under tests/docs" in out


def test_a_new_page_with_an_untested_block_fails_the_check(tmp_path, capsys):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6})
    (tmp_path / "docs" / "source" / "new.rst").write_text("New\n===\n\nRun::\n\n   make\n")

    assert _coverage().main(["--check", *flags]) == 1
    assert "new.rst: 1 untested blocks, the page is not in the baseline" in capsys.readouterr().out


def test_an_allowance_larger_than_the_page_needs_fails_as_stale(tmp_path, capsys):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 7, "gone.rst": 2})

    assert _coverage().main(["--check", *flags]) == 1
    out = capsys.readouterr().out
    assert "STALE  guide/page.rst: the baseline allows 7 untested blocks, the page has 6." in out
    assert "STALE  gone.rst: the baseline allows 2 untested blocks, the page has 0" in out
    assert "--update-baseline" in out


def test_update_baseline_writes_what_the_check_then_accepts(tmp_path):
    flags = _repo(tmp_path, allowed={})
    module = _coverage()

    assert module.main(["--check", *flags]) == 1
    assert module.main(["--update-baseline", *flags]) == 0
    assert module.main(["--check", *flags]) == 0
    written = json.loads((tmp_path / "baseline.json").read_text())["untested"]
    assert written == {"guide/page.rst": 6}


def test_pending_claims_are_reported_but_not_counted(tmp_path, capsys):
    claim = {
        "test": "tests/docs/test_future.py",
        "pr": 9999,
        "pages": ["guide/page.rst"],
        "languages": ["yaml"],
        "kind": "runs",
        "what": "future",
    }
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6}, pending=[claim])
    module = _coverage()

    assert module.main(["--check", *flags]) == 0
    capsys.readouterr()
    module.main(["--json", *flags])
    (row,) = json.loads(capsys.readouterr().out)["pages"]
    assert (row["tested"], row["untested"], row["pending"]) == (1, 6, 1)
    # --with-pending counts a pending claim only once its test is on disk.
    module.main(["--json", "--with-pending", *flags])
    (row,) = json.loads(capsys.readouterr().out)["pages"]
    assert (row["tested"], row["untested"], row["pending"]) == (1, 6, 1)
    (tmp_path / "tests" / "docs" / "test_future.py").write_text("", encoding="utf-8")
    module.main(["--json", "--with-pending", *flags])
    (row,) = json.loads(capsys.readouterr().out)["pages"]
    assert (row["tested"], row["untested"], row["pending"]) == (2, 5, 0)


def test_a_pending_claim_whose_test_landed_must_be_promoted(tmp_path):
    module = _coverage()
    new = module.Claim("tests/docs/test_new.py", ("x.rst",), ("bash",), "checks", "", pr=1)
    widened = module.Claim(
        "tests/docs/test_demo.py", ("x.rst",), ("bash",), "checks", "", pr=2, marker="def wider"
    )
    (tmp_path / "tests" / "docs").mkdir(parents=True)
    demo = tmp_path / "tests" / "docs" / "test_demo.py"
    demo.write_text("def narrow(): ...\n")

    assert module.pending_problems([new, widened], tmp_path) == []

    (tmp_path / "tests" / "docs" / "test_new.py").write_text("")
    demo.write_text("def wider(): ...\n")
    problems = module.pending_problems([new, widened], tmp_path)
    assert [p.split(":")[0] for p in problems] == [
        "PROMOTE tests/docs/test_new.py (PR #1)",
        "PROMOTE tests/docs/test_demo.py (PR #2)",
    ]


def test_a_claim_that_reads_no_block_is_a_ledger_problem(tmp_path):
    module = _coverage()
    root = _tree(tmp_path)
    tree = module.scan_tree(root / "docs" / "source")
    claim = module.Claim("tests/docs/test_gone.py", ("guide/*.rst",), ("toml",), "checks", "")

    problems = module.ledger_problems([claim], tree, root)

    assert [p.split(":")[0] for p in problems] == [
        "LEDGER tests/docs/test_gone.py",
        "LEDGER tests/docs/test_gone.py",
    ]
    assert "no such test file" in problems[0]
    assert "matches no block" in problems[1]
