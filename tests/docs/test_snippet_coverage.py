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
  with an untested block and a landed pending test each fail;
* the three bins past "tested": a block only a ``parses`` claim reads is
  tested but parse-only, a ``not_testable:`` entry takes a block out of the
  ratchet, and each guard on those entries (stale, unknown reason, empty
  ``why``, a block a real claim reads) fails, as does a claim with both
  ``contains:`` and ``matches:``.
"""

from __future__ import annotations

import json
import re
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


def test_the_honesty_guard_holds_for_every_live_claim():
    """Each claimed page's stem (``docs/source`` for a ``**`` glob) is in the test's text."""
    module, _, live, _ = _live()

    unnamed = [
        f"{claim.test}: {module.page_stem(pattern)}"
        for claim in live
        for pattern in claim.pages
        if module.page_stem(pattern) not in (_REPO_ROOT / claim.test).read_text(encoding="utf-8")
    ]

    assert unnamed == []


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


def test_every_pending_claim_names_its_pr():
    """A PR number, or the campaign unit id (``W6-3``) until the pull request opens."""
    _, _, _, pending = _live()

    unnamed = [
        claim.test
        for claim in pending
        if not isinstance(claim.pr, int)
        and not (isinstance(claim.pr, str) and re.fullmatch(r"W\d+-\d+", claim.pr))
    ]

    assert unnamed == []


def test_every_not_testable_entry_names_a_block_no_claim_reads():
    module, tree, live, pending = _live()
    entries = module.load_not_testable()

    assert entries, "the ledger has no `not_testable:` entries; the section moved?"
    problems = module.ledger_problems(
        [], tree, _REPO_ROOT, waiting=live + pending, not_testable=entries
    )
    assert problems == []


def test_every_not_counted_test_exists():
    document = yaml.safe_load(_LEDGER.read_text(encoding="utf-8"))

    missing = [e["test"] for e in document["not_counted"] if not (_REPO_ROOT / e["test"]).is_file()]

    assert missing == [], "stale `not_counted:` entries: the test is gone"


def test_the_check_passes_on_this_tree():
    """What ``snippet_coverage.py --check`` judges, judged here: the CLI and CI agree.

    A pending claim whose test is on disk counts as tested, so an open PR that
    brings its own test and claim passes whatever order the PRs merge in.
    """
    module = _coverage()

    result = module.evaluate(_REPO_ROOT)

    if result.failures:
        pytest.fail(
            "\n".join([*result.failures, "", module.FIX_GREW, "", module.FIX_STALE]),
            pytrace=False,
        )


# -- negative controls -----------------------------------------------------------


def _repo(
    root: Path,
    *,
    allowed: dict[str, int],
    pending: list[dict] | None = None,
    tested: list[dict] | None = None,
    not_testable: list[dict] | None = None,
) -> list[str]:
    """A synthetic repo with one live claim, and the CLI flags that point the script at it."""
    _tree(root)
    (root / "tests" / "docs").mkdir(parents=True)
    (root / "tests" / "docs" / "test_demo.py").write_text("# reads guide/*.rst\n", encoding="utf-8")
    ledger = {
        "tested": [
            {
                "test": "tests/docs/test_demo.py",
                "pages": ["guide/*.rst"],
                "languages": ["bash"],
                "kind": "checks",
                "what": "demo",
            },
            *(tested or []),
        ],
        "pending": pending or [],
        "not_testable": not_testable or [],
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
    assert "new.rst: 1 untested block, the page is not in the baseline" in capsys.readouterr().out


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


def _pending_yaml_claim() -> dict:
    return {
        "test": "tests/docs/test_future.py",
        "pr": 9999,
        "pages": ["guide/page.rst"],
        "languages": ["yaml"],
        "kind": "runs",
        "what": "future",
    }


def _row(module, flags, capsys) -> tuple[int, int, int]:
    capsys.readouterr()
    module.main(["--json", *flags])
    (row,) = json.loads(capsys.readouterr().out)["pages"]
    return row["tested"], row["untested"], row["pending"]


def test_a_pending_claim_counts_once_its_test_is_on_disk(tmp_path, capsys):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6}, pending=[_pending_yaml_claim()])
    module = _coverage()

    assert _row(module, flags, capsys) == (1, 6, 1)
    assert module.main(["--check", *flags]) == 0

    (tmp_path / "tests" / "docs" / "test_future.py").write_text("# guide/page.rst\n")

    assert _row(module, flags, capsys) == (2, 5, 0)
    # The allowance is now above need, but the landed claim explains it: noted, not failed.
    assert module.main(["--check", *flags]) == 0
    out = capsys.readouterr().out
    assert "note: PROMOTE tests/docs/test_future.py (PR #9999)" in out
    assert "note: SHRINK guide/page.rst" in out


def test_a_pr_adding_a_block_and_its_test_passes_before_promotion(tmp_path):
    """The merge-order case: a new block, read by the PR's own still-pending claim."""
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6}, pending=[_pending_yaml_claim()])
    page = tmp_path / "docs" / "source" / "guide" / "page.rst"
    page.write_text(page.read_text() + "\n.. code-block:: yaml\n\n   added: true\n")
    module = _coverage()

    assert module.main(["--check", *flags]) == 1
    (tmp_path / "tests" / "docs" / "test_future.py").write_text("# guide/page.rst\n")
    assert module.main(["--check", *flags]) == 0


def test_a_landed_widened_claim_needs_its_marker(tmp_path):
    module = _coverage()
    new = module.Claim("tests/docs/test_new.py", ("x.rst",), ("bash",), "checks", "", pr=1)
    widened = module.Claim(
        "tests/docs/test_demo.py", ("x.rst",), ("bash",), "checks", "", pr=2, marker="def wider"
    )
    (tmp_path / "tests" / "docs").mkdir(parents=True)
    demo = tmp_path / "tests" / "docs" / "test_demo.py"
    demo.write_text("def narrow(): ...\n")

    assert [module.landed(c, tmp_path) for c in (new, widened)] == [False, False]
    assert module.pending_problems([new, widened], tmp_path) == []

    (tmp_path / "tests" / "docs" / "test_new.py").write_text("")
    demo.write_text("def wider(): ...\n")
    notes = module.pending_problems([new, widened], tmp_path)
    assert [module.landed(c, tmp_path) for c in (new, widened)] == [True, True]
    assert [n.split(":")[0] for n in notes] == [
        "PROMOTE tests/docs/test_new.py (PR #1)",
        "PROMOTE tests/docs/test_demo.py (PR #2)",
    ]


def test_a_claim_on_a_page_its_test_never_names_fails(tmp_path, capsys):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6})
    (tmp_path / "tests" / "docs" / "test_demo.py").write_text("# reads some other page\n")

    assert _coverage().main(["--check", *flags]) == 1
    assert "claims guide/*.rst, but the test never names 'guide'" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("pattern", "stem"),
    [
        ("**/*.rst", "docs/source"),
        ("reference/cli.rst", "cli"),
        ("how-to/control-systems/examples/*.py", "examples"),
        ("getting-started/hello-world-tutorial.sql", "hello-world-tutorial"),
    ],
)
def test_page_stem(pattern, stem):
    assert _coverage().page_stem(pattern) == stem


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


# -- parse-only, not testable, and `matches:` ------------------------------------------


def _claim(**entry) -> dict:
    return {"test": "tests/docs/test_demo.py", "pages": ["guide/*.rst"], "what": "demo", **entry}


def _skip(**entry) -> dict:
    return {"pages": ["guide/page.rst"], "reason": "llm", "why": "the model answers it", **entry}


def _page(module, flags, capsys) -> dict:
    capsys.readouterr()
    module.main(["--json", *flags])
    document = json.loads(capsys.readouterr().out)
    (row,) = document["pages"]
    return {"totals": document["totals"], **row}


def test_a_parses_claim_counts_as_tested_and_parse_only(tmp_path, capsys):
    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 4},
        tested=[_claim(languages=["python", "bash"], kind="parses")],
    )
    module = _coverage()

    row = _page(module, flags, capsys)

    # bash is also read by the `checks` claim, so only python is parse-only.
    assert (row["tested"], row["parse_only"], row["untested"]) == (2, 1, 5)
    assert row["kind"] == "parses"
    assert row["totals"]["parse_only"] == 1
    assert module.main(["--check", *flags]) == 1  # 4 allowed, 5 untested


def test_a_block_read_by_a_stronger_claim_is_not_parse_only(tmp_path, capsys):
    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 6},
        tested=[_claim(languages=["bash"], kind="parses")],
    )

    row = _page(_coverage(), flags, capsys)

    assert (row["tested"], row["parse_only"], row["kind"]) == (1, 0, "checks")


def test_a_not_testable_block_leaves_the_ratchet_and_shows_its_reason(tmp_path, capsys):
    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 5},
        tested=[_claim(languages=["console"], kind="parses")],
        not_testable=[_skip(languages=["console"], contains="$ osprey up")],
    )
    module = _coverage()

    row = _page(module, flags, capsys)
    assert (row["tested"], row["parse_only"], row["not_testable"], row["untested"]) == (1, 0, 1, 5)
    assert row["not_testable_blocks"] == [
        {
            "line": _line(_PAGE, "$ osprey up"),
            "language": "console",
            "reason": "llm",
            "why": "the model answers it",
        }
    ]
    assert row["totals"]["not_testable"] == 1
    assert module.main(["--check", *flags]) == 0

    capsys.readouterr()
    module.main([*flags])
    report = capsys.readouterr().out
    assert "1 tested (14%; 0 parse-only), 1 not testable, 5 untested" in report
    assert "| Page | Blocks | Tested | Parse-only | Not testable | Untested |" in report
    assert "| guide/page.rst | 7 | 1 |  | 1 | 5 |" in report

    module.main(["--page", "guide/page.rst", *flags])
    narrowed = capsys.readouterr().out
    assert f"- line {_line(_PAGE, '$ osprey up')} console: llm -- the model answers it" in narrowed


def test_matches_is_a_line_anchored_regex_and_contains_a_substring(tmp_path, capsys):
    module = _coverage()
    block = module.Block("guide/page.rst", 1, "bash", None, "cd demo\nosprey build --dev")
    claim = module.Claim("t.py", ("guide/*.rst",), ("bash",), "checks", "")

    assert module.Claim(**{**vars(claim), "regex": r"^osprey build"}).matches(block)
    assert not module.Claim(**{**vars(claim), "regex": r"\Aosprey build"}).matches(block)
    assert module.Claim(**{**vars(claim), "contains": "build --dev"}).matches(block)
    assert not module.Claim(**{**vars(claim), "contains": "^osprey"}).matches(block)

    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 6},
        tested=[_claim(languages=["literal"], kind="pins", matches=r"^literal line two$")],
    )
    assert _page(module, flags, capsys)["tested"] == 2


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"contains": "x", "matches": "x"}, "has both `contains:` and `matches:`"),
        ({"matches": "(unclosed"}, "is not a regular expression"),
        ({"kind": "guesses"}, "kind 'guesses' is not one of"),
    ],
)
def test_a_malformed_claim_is_a_ledger_problem(tmp_path, capsys, bad, message):
    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 6},
        tested=[_claim(languages=["bash"], **{"kind": "checks", **bad})],
    )

    assert _coverage().main(["--check", *flags]) == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (_skip(languages=["toml"], contains="x"), "matches no block on any page"),
        (_skip(languages=["text"], contains="plain", reason="hard"), "reason 'hard' is not one of"),
        (_skip(languages=["text"], contains="plain", why=" "), "`why:` is empty"),
        (_skip(languages=["text"]), "needs `contains:` or `matches:`"),
        (_skip(languages=["bash"], contains="osprey build"), "it is tested after all"),
    ],
)
def test_each_not_testable_guard_fails(tmp_path, capsys, entry, message):
    flags = _repo(tmp_path, allowed={"guide/page.rst": 6}, not_testable=[entry])

    assert _coverage().main(["--check", *flags]) == 1
    assert message in capsys.readouterr().out


def test_an_entry_on_a_block_a_waiting_pending_claim_reads_fails(tmp_path, capsys):
    """Whatever the merge order: the pending test will read the block, so it is not untestable."""
    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 5},
        pending=[_pending_yaml_claim()],
        not_testable=[_skip(languages=["yaml"], contains="key: value")],
    )

    assert _coverage().main(["--check", *flags]) == 1
    assert "read by tests/docs/test_future.py; it is tested after all" in capsys.readouterr().out


def test_an_entry_on_a_block_a_parses_claim_reads_passes(tmp_path, capsys):
    flags = _repo(
        tmp_path,
        allowed={"guide/page.rst": 5},
        tested=[_claim(languages=["python"], kind="parses")],
        not_testable=[_skip(languages=["python"], matches=r"^print\(")],
    )
    module = _coverage()

    assert module.main(["--check", *flags]) == 0
    row = _page(module, flags, capsys)
    assert (row["parse_only"], row["not_testable"]) == (0, 1)
