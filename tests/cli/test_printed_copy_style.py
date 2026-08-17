"""The house style for text OSPREY prints at a person.

Two registers, one seam. A docstring may argue why the code is shaped the way
it is; a string that reaches somebody's terminal may not. The people running
OSPREY operate accelerators and beamlines, and most of them did not write this
framework -- so the words that reach them have to be words they already know.

This module enforces the mechanical half of that. The rest (does the sentence
lead with an absence, is the block worth printing at all) is review's job.

Scope is deliberately narrow: string literals passed to a printing call. Log
records are exempt -- they carry a level, and ``--verbose`` output is addressed
to whoever is debugging. So are docstrings, comments, and ``--help`` text.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"

#: Calls whose string arguments land on a terminal, by attribute or bare name.
#: ``emit`` and ``echo`` are the phase reporter's (``echo`` is also click's,
#: along with ``secho``), and ``print`` is rich's console and the builtin.
#:
#: ``report``, ``note``, ``section``, ``table``, ``warn`` and ``fail`` are
#: :mod:`osprey.cli.output`'s primitives, the renderer every printed line is
#: moving onto. Matching is by bare name, so ``warn`` also catches
#: ``warnings.warn`` -- which is the right reach anyway: a deprecation warning
#: is read by whoever runs the build, not by whoever wrote it. ``table`` takes a
#: renderable rather than prose, so it normally contributes nothing; it is here
#: so that a string handed to it is caught rather than silently exempt.
#:
#: The last three are wrappers rather than primitives, and they are here because
#: of the rule they state: a wrapper around a renderer primitive either joins
#: this set, or carries a comment at its own definition saying its copy is
#: unguarded and why. A wrapper that does neither reads as covered while being
#: invisible, which is the one outcome this file exists to prevent.
#:
#: ``_abort`` is ``deploy_cmd``'s refusal wrapper, which hands its three
#: arguments straight to ``fail`` and then stops the run. ``_report_fact`` and
#: ``report_fact`` are the promotion wrappers: each prints through ``report``
#: and keeps a log record at the level it already had, and between them they
#: carry the deploy and web-terminal facts an operator reads.
#:
#: Named exception, so the rule above is enforced rather than merely stated:
#: ``_emit`` at ``cli/config_cmd.py`` and ``interfaces/okf_panel/validation.py``
#: prints, and stays out. Matching is by bare name, and ``cli/deploy_scaffold.py``
#: has an ``_emit`` that writes a FILE -- registering the name would judge a
#: file's contents as prose. Both printing ``_emit``s carry the comment.
_PRINTERS = frozenset(
    {
        "echo",
        "secho",
        "emit",
        "print",
        "report",
        "note",
        "section",
        "table",
        "warn",
        "fail",
        "_abort",
        "_report_fact",
        "report_fact",
        "_warn_fact",
        "warn_fact",
    }
)

#: In-house vocabulary. Each of these has a plain equivalent that costs no
#: precision, and none of them is defined anywhere the reader will have been.
#: The replacement is in the second column so a failure names it.
#:
#: This is a DENYLIST and proves nothing about copy it does not match: text can
#: be impenetrable using none of these words. What it is aimed at is the way
#: the jargon actually comes back. These words are still all over the
#: docstrings and comments here -- "materializ" in 29 files, "source zone" in
#: 14, "verbatim" in 81 -- so whoever adds the next `click.echo` is writing it
#: surrounded by them, and will reach for them. Catching that is worth a list
#: someone has to extend by hand.
_JARGON: dict[str, str] = {
    "materialize": "write",
    "materialized": "written",
    "materializes": "writes",
    "materializing": "writing",
    "materialization": "writing the files",
    "rematerialize": "write again",
    "re-materialize": "write again",
    "source zone": "the files you edit",
    "by construction": "(state the fact plainly)",
    "verbatim": "exactly as written",
    "deliberately": "on purpose, or say nothing",
    "canonical": "the one that counts",
    "inert": "has no effect",
    "write-armed": "allowed to make changes",
}

# NOT here, on purpose: a rule against absolutes used as emphasis ("loses
# nothing, ever", "can never reach", "100% disposable"). Those sentences were
# real and they were worth deleting, but a regex for them can only list the
# ones already deleted, which is a snapshot of one afternoon rather than a rule
# -- it would pass forever without reading anything. Generalizing it to
# ever/never/always fails the other way: "deleting it is always safe" and
# "nothing ever overwrites this" are both plain statements of fact. Whether an
# absolute is a fact or a slogan needs a reader, so it stays in review.


def _printed_strings() -> list[tuple[Path, int, str]]:
    """Every string literal handed to a printing call under ``src/osprey``.

    f-strings contribute their literal segments only: an interpolated path or
    count is data, and judging it as prose would flag whatever happened to be
    in the variable.
    """
    found: list[tuple[Path, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - src always parses
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in _PRINTERS:
                continue
            for arg in node.args:
                text = _literal_text(arg)
                if text:
                    found.append((path, node.lineno, text))
    return found


def _literal_text(node: ast.expr) -> str:
    """The literal prose in ``node``, or empty for anything that is not prose."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return ""


@pytest.fixture(scope="module")
def printed() -> list[tuple[Path, int, str]]:
    strings = _printed_strings()
    assert strings, "found no printed strings at all -- the AST walk is broken"
    return strings


def test_printed_text_uses_no_in_house_jargon(printed: list[tuple[Path, int, str]]) -> None:
    """Words only an OSPREY maintainer knows do not reach an operator."""
    offences: list[str] = []
    for path, line, text in printed:
        lowered = text.lower()
        for term, plain in _JARGON.items():
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                rel = path.relative_to(SRC.parents[1])
                offences.append(
                    f"{rel}:{line}: {term!r} -> {plain}\n      {' '.join(text.split())}"
                )
    assert not offences, "In-house jargon in printed text:\n  " + "\n  ".join(offences)


def test_printed_text_carries_no_em_dash_asides(printed: list[tuple[Path, int, str]]) -> None:
    """An aside worth reading is worth its own sentence.

    The em-dash aside is this codebase's default sentence shape, and it is what
    makes the output read as generated rather than written: the clause between
    the dashes carries the content, so it cannot be skipped, and the sentence
    around it stays open across it. Split it or cut it.
    """
    offences = [
        f"{path.relative_to(SRC.parents[1])}:{line}: {' '.join(text.split())}"
        for path, line, text in printed
        if "—" in text
    ]
    assert not offences, "Em-dash in printed text (split the sentence):\n  " + "\n  ".join(offences)


def test_the_walk_still_reaches_the_output_people_read() -> None:
    """The two rules above are only worth what this walk covers.

    They are silent on a call this does not recognize, and silence is
    indistinguishable from clean. So pin the reach: a refactor that renames the
    reporter's ``emit``, moves a verb out of ``cli/``, or routes output through
    a wrapper this does not match should fail HERE, loudly, rather than quietly
    switching the lint off.

    Two reaches it cannot pin, both of them the same shape -- the walk reads the
    POSITIONAL ARGUMENTS of a printing call, and nothing else:

    * copy built elsewhere and handed over in a variable, however loudly it
      prints afterwards. ``format_conflict_report`` is the worked example, and
      it is pinned in its own test instead.
    * copy passed by KEYWORD, even where the callee is registered here. Live
      today at ``deploy_cmd``'s ``nothing_done="Nothing was deployed."``, which
      is concatenated into a printed cause this walk never reads.

    Widening the matcher to keywords is not obviously right (it would start
    reading ``style=`` and every other non-prose keyword), so both stay review's
    job. What matters is that they are written down: a blind spot nobody has
    named reads as coverage.

    Measured on rebaseline: 454 literals across 55 files. That is the reading of
    one run rather than a constant, since every line of copy added or deleted
    moves it. The floors sit about
    10% under that, which is room for ordinary copy churn and not much more --
    a block getting deleted should not turn this red, and the walk losing a
    whole module should. Re-derive both numbers from a real run when they next
    move, and note that the count moves whenever a WRAPPER joins ``_PRINTERS``,
    not only when copy changes. Do not raise them to whatever the day's count
    happens to be: a floor pinned to the ceiling fails on the next honest
    deletion.
    """
    strings = _printed_strings()
    files = {path for path, _, _ in strings}

    assert len(strings) > 405, (
        f"only {len(strings)} printed literals found -- has _PRINTERS drifted?"
    )
    assert len(files) > 49, f"only {len(files)} files matched -- has the walk stopped descending?"

    # The onboarding path specifically, since that is what these rules exist for.
    # `validate_cmd.py` rather than `profile_cmd.py`: the profile group's verdict
    # copy lives there now, printed once for both spellings of the verb.
    for expected in ("cli/init_cmd.py", "cli/validate_cmd.py", "deployment/status_display.py"):
        assert any(str(path).endswith(expected) for path in files), (
            f"{expected} contributes no printed strings -- it is no longer covered"
        )
