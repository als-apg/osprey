"""Suite for the docs directional-reference guard.

The guard exists because splitting a page turns "as described below" into a
promise the page can no longer keep. A guard nobody has watched fail proves
nothing, so the synthetic pages here pair one real offence with the lookalikes
that have to stay quiet — a "below" pointing into this very page, a numeric
comparison, an inline literal, a word inside a literal block, and a word one
paragraph away — plus the two blind spots review found: a literal block opened
inside a list item, and a Markdown page the guard cannot read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "docs" / "check_directional_refs.py"

OTHER_PAGE = """\
=====
Other
=====

Other body text.
"""

# One offence — the first paragraph — and four lookalikes.
GUIDE_PAGE = """\
=====
Guide
=====

.. _guide-details:

Details
=======

The full story is below --- see :doc:`other`.

Local pointer: the answer is below, in :ref:`guide-details`.

A value above 30 runs last, as :doc:`other` explains.

The ``above`` key is described in :doc:`other`.

Verbatim text does not count, says :doc:`other`::

   # the value above is only a default
"""

# A literal block opened inside a list item ends where the item's body ends,
# not where the bullet does: the sentence after it is still prose.
LIST_PAGE = """\
=====
Steps
=====

1. First do this::

      run --it

   The answer is below, see :doc:`other`.
"""

# A blank line is a wall: the word belongs to its own paragraph, not the link's.
PARAGRAPH_PAGE = """\
==========
Paragraphs
==========

The answer is below.

See :doc:`other` for the rest of the story.
"""


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_directional_refs", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _tree(root: Path, **pages: str) -> Path:
    """A throwaway ``docs/source`` holding *pages*, keyed by docname."""
    docs_root = root / "docs" / "source"
    docs_root.mkdir(parents=True, exist_ok=True)
    for docname, body in pages.items():
        (docs_root / f"{docname}.rst").write_text(body, encoding="utf-8")
    return docs_root


def _findings(docs_root: Path, capsys) -> tuple[int, list[str]]:
    code = checker.main(["--docs-root", str(docs_root)])
    out = capsys.readouterr().out
    return code, [line for line in out.splitlines() if line.strip() and "FAILED" not in line]


def _line_of(page: str, text: str) -> int:
    return page.splitlines().index(text) + 1


# ── the offence, and the four lookalikes that must stay quiet ────────────


def test_a_cross_page_link_next_to_below_is_reported(tmp_path, capsys):
    docs_root = _tree(tmp_path, guide=GUIDE_PAGE, other=OTHER_PAGE)
    code, findings = _findings(docs_root, capsys)

    assert code == 1, "a cross-page :doc: next to 'below' must be reported"
    offence = _line_of(GUIDE_PAGE, "The full story is below --- see :doc:`other`.")
    assert findings == [f"source/guide.rst:{offence}: below near :doc:`other` (target on other)"], (
        "exactly the one off-page 'below' must fire — a same-page :ref:, a numeric "
        "comparison, an inline literal and a literal block are not offences:\n"
        + "\n".join(findings)
    )


def test_a_same_page_target_is_not_an_offence(tmp_path, capsys):
    """Strip the cross-page paragraph and the page must go green."""
    docs_root = _tree(
        tmp_path,
        guide=GUIDE_PAGE.replace("The full story is below --- see :doc:`other`.\n\n", ""),
        other=OTHER_PAGE,
    )
    code, findings = _findings(docs_root, capsys)
    assert code == 0, "'below' next to a link into this very page is correct prose:\n" + "\n".join(
        findings
    )


def test_a_ref_no_label_explains_is_reported_as_unresolved(tmp_path, capsys):
    docs_root = _tree(
        tmp_path,
        guide=GUIDE_PAGE.replace(":ref:`guide-details`", ":ref:`no-such-label`"),
        other=OTHER_PAGE,
    )
    code, findings = _findings(docs_root, capsys)
    assert code == 1, "a label the tree does not define must not pass silently"
    assert any("unresolved :ref:`no-such-label`" in line for line in findings), findings


# ── the two blind spots review found ─────────────────────────────────────


def test_a_literal_block_in_a_list_item_does_not_hide_the_rest_of_the_item(tmp_path, capsys):
    """The mask must end with the block, not with the whole list item."""
    docs_root = _tree(tmp_path, steps=LIST_PAGE, other=OTHER_PAGE)
    code, findings = _findings(docs_root, capsys)

    assert code == 1, (
        "prose that follows a literal block inside a list item is still prose — "
        "measuring the block from the bullet swallows it silently"
    )
    offence = _line_of(LIST_PAGE, "   The answer is below, see :doc:`other`.")
    assert findings == [f"source/steps.rst:{offence}: below near :doc:`other` (target on other)"], (
        "\n".join(findings)
    )


def test_a_markdown_page_stops_the_run_instead_of_being_skipped(tmp_path):
    """MyST is enabled, so a Markdown page is legal — and unreadable here."""
    docs_root = _tree(tmp_path, guide=GUIDE_PAGE, other=OTHER_PAGE)
    (docs_root / "notes.md").write_text("# Notes\n\n(anchor)=\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        checker.check(docs_root)
    message = str(raised.value)
    assert "notes.md" in message and "Markdown" in message, message


# ── the window is the paragraph, not a fixed span ────────────────────────


def test_a_directional_word_does_not_bleed_across_a_paragraph_break(tmp_path, capsys):
    docs_root = _tree(tmp_path, paragraphs=PARAGRAPH_PAGE, other=OTHER_PAGE)
    code, findings = _findings(docs_root, capsys)
    assert code == 0, (
        "a blank line separates the word from the link; reporting across it is noise:\n"
        + "\n".join(findings)
    )


# ── the shipped tree ─────────────────────────────────────────────────────


def test_the_shipped_docs_tree_is_clean(capsys):
    findings, summary = checker.check(_REPO_ROOT / "docs" / "source")
    assert findings == [], "directional wording next to an off-page link:\n" + "\n".join(findings)

    assert checker.main([]) == 0, "the default --docs-root must be the shipped tree"
    out = capsys.readouterr().out
    assert "roles checked" in out and "across 0 pages" not in out, (
        f"the default run must actually read pages:\n{out}\n{summary}"
    )
