#!/usr/bin/env python3
r"""Changelog fragments — one small file per change instead of one shared section.

Two pull requests that both append a bullet to ``CHANGELOG.md``'s
``## [Unreleased]`` section collide on the same lines, whichever merges first.
So no pull request writes that section any more. Each writes one small file
under ``changelog.d/`` instead, and the release fold collects them.

This module is both halves of that arrangement: ``check`` is the gate CI and
``premerge_check.sh`` run, ``apply`` is the fold the release skill runs. It
imports nothing outside the standard library and shells out to git only from
``main``, through an injectable runner.

Fragment files
--------------
``changelog.d/<name>.<type>.md``, matching::

    ^(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)\.(?P<type>[a-z]+)\.md$

``<name>`` is the issue number when the change has one (``745.fixed.md``,
``745-gate.fixed.md``) and a short slug otherwise (``gate.fixed.md``). A name
whose leading digits are followed by ``-`` or by the end of the name IS an
issue reference: ``apply`` appends `` (#745)`` to the rendered bullet. That
rule has one sharp edge, and it is deliberate — ``2026-cleanup.changed.md``
renders ``(#2026)``, so a name must never begin with a date or any other
number that is not an issue. A slug name gets no reference appended and may
end its own text with a hand-written ``(#735, #737)``.

``<type>`` is one of:

===============  ==========================================================
``added``        new capability
``changed``      behaviour of something that already existed
``deprecated``   still works, on its way out
``removed``      gone
``fixed``        a bug
``security``     a vulnerability or a hardening change
``internal``     work users never see — satisfies the gate, renders nothing
===============  ==========================================================

The first six become ``### Added`` … ``### Security`` headings in
Keep-a-Changelog order. ``internal`` renders nothing at all: it exists so a
refactor that touches shipped code can pass the gate honestly instead of
inventing a user-facing sentence.

Body
----
The body is the bullet's text *without* the leading ``- ``: one or two
user-facing sentences, present tense, hand-wrapped. It is copied verbatim, so
blank lines, sub-bullets and fenced blocks written at column 0 all survive the
fold, and a bold opener (``**Breaking change:** …``) is fine. Rejected: an
empty body, a first line that already carries a list marker, a heading or a
fence, and — for a name that already supplies the issue reference — a trailing
``(#745)`` at the end of the opening paragraph, which would render twice.

Text is read as UTF-8 — a byte-order mark, which some Windows editors write
by default, is stripped rather than shipped as an invisible first character —
with ``\r\n`` and ``\r`` normalized to ``\n``, and leading and trailing blank
lines are stripped before anything else looks at it.
``README.md`` is skipped by name and is the only permanent resident of the
directory, which is flat.

Apply
-----
``apply`` validates every fragment, then folds them into ``## [Unreleased]``
in filename order: each bullet goes directly under its own ``### <Type>``
heading and that heading's single blank line, and headings that do not exist
yet are created at the top of the block in Keep-a-Changelog order. Existing
headings are never merged or reordered — the file is 6000 lines old and
rewriting it is not this tool's business. The whole text is built in memory
and written once; then the fragment files are deleted, ``internal`` ones
included, and the report says what to stage.

Usage
-----
    python scripts/changelog_fragments.py check [--base origin/main]
    python scripts/changelog_fragments.py apply

Exit codes
----------
0   nothing to report
1   something a contributor fixes: a malformed fragment, a missing one, a
    hand-written bullet in ``## [Unreleased]``, a deleted fragment
2   something the environment has to fix: an unresolvable base ref, no common
    ancestor, a git call that failed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

TYPES = ("added", "changed", "deprecated", "removed", "fixed", "security", "internal")
RENDERED_TYPES = TYPES[:6]
# Both derived from RENDERED_TYPES so none of the three can drift: a type added
# to TYPES ahead of `internal` gets its heading, and its place in the fold's
# order, for free; one that renders nothing never gets either.
HEADING_FOR = {type_: type_.capitalize() for type_ in RENDERED_TYPES}
HEADING_ORDER = tuple(HEADING_FOR.values())
GATED_PREFIXES = ("src/", "packages/")
FRAGMENT_DIR = "changelog.d"
KEEP = "README.md"
REPO_ROOT = Path(__file__).resolve().parents[1]

NAME_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)\.(?P<type>[a-z]+)\.md$")
REF_RE = re.compile(r"^(\d+)(?:-|$)")
FIRST_LINE_REJECT_RE = re.compile(r"^(?:[-*+]\s|#{1,6}\s|```|~~~)")
PARAGRAPH_END_RE = re.compile(r"^(?:[-*+]\s|```)")
TRAILING_REF_RE = re.compile(r"\(#\d+\)\s*$")
UNRELEASED_RE = re.compile(r"^## \[Unreleased\]\s*$")
SECTION_END_RE = re.compile(r"^## \[")
TYPE_HEADING_RE = re.compile(r"^### (Added|Changed|Deprecated|Removed|Fixed|Security)\s*$")
BULLET_RE = re.compile(r"^- ")
PLAIN_REF_RE = re.compile(r"^[A-Za-z0-9._-]+$")

#: One git invocation: argv in, a finished process out. ``main`` takes one so
#: the tests can answer git from a table instead of building a repository.
Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]

# Every rejection that has anything to do with the type names spells all seven
# out. A contributor who has just been told "no" should not have to open this
# file, or the README, to find out what "yes" looks like.
_VOCABULARY = f"use one of {', '.join(TYPES)} ({TYPES[-1]} satisfies the gate and renders nothing)"


class FragmentError(ValueError):
    """Something a contributor can fix in the fragment itself.

    Raised for every malformed name, encoding and body. ``validate_dir``
    collects these rather than stopping at the first, so one run of the gate
    reports every offender.
    """


@dataclass(frozen=True)
class Fragment:
    """One validated fragment file.

    ``lines`` is the body already normalized to LF and trimmed of leading and
    trailing blank lines, so everything downstream — the ref placement, the
    rendered bullet — can index it without re-deriving that.
    """

    path: Path
    name: str
    type: str
    ref: str | None
    lines: tuple[str, ...]


def parse_fragment_name(filename: str) -> tuple[str, str, str | None]:
    """Split ``<name>.<type>.md`` into ``(name, type, ref)``.

    ``ref`` is the leading digit run of ``<name>`` when that run is followed by
    ``-`` or ends the name, and ``None`` otherwise. It is a string because it
    goes straight back into text as ``(#745)``.

    Raises ``FragmentError`` on a name that does not match the grammar or that
    carries a type outside the vocabulary.
    """
    match = NAME_RE.match(filename)
    if match is None:
        raise FragmentError(
            f"{filename}: not a fragment name — use <name>.<type>.md, such as "
            f"745.fixed.md (the issue number) or gate.fixed.md (a slug); for <type>, "
            f"{_VOCABULARY}"
        )
    name, type_ = match.group("name"), match.group("type")
    if type_ not in TYPES:
        raise FragmentError(f'{filename}: unknown type "{type_}" — {_VOCABULARY}')
    ref_match = REF_RE.match(name)
    return name, type_, ref_match.group(1) if ref_match else None


def _split_lines(text: str) -> list[str]:
    """Split *text* into LF lines, keeping leading and trailing blanks.

    The one place line endings are normalized. The gate compares two versions
    of ``CHANGELOG.md`` line for line, and a blank line that one side has and
    the other does not is exactly the kind of edit it is looking at, so this
    trims nothing; ``normalize_lines`` is the trimming variant for fragments.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def normalize_lines(text: str) -> list[str]:
    """Split *text* into LF lines with leading and trailing blank lines dropped.

    Editors and platforms disagree about line endings and about whether a file
    ends with a newline; none of that is allowed to reach the rendered bullet,
    where it would show up as a stray blank line inside ``CHANGELOG.md``.
    """
    lines = _split_lines(text)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def opening_paragraph_end(lines: Sequence[str]) -> int:
    """Index of the last line of the opening paragraph.

    The opening paragraph is the run of lines that make up the bullet's own
    sentence, which is where the issue reference belongs — appending it to the
    body's last line would park ``(#745)`` after a sub-bullet or inside a
    fenced block. The paragraph ends at the first blank line or at the first
    column-0 list marker or fence; a hand-wrapped continuation line, including
    one that opens with ``**``, does not end it.

    An empty body has no opening paragraph and yields 0, so no caller ever
    receives ``-1`` and indexes from the wrong end.
    """
    for index in range(1, len(lines)):
        line = lines[index]
        if not line.strip() or PARAGRAPH_END_RE.match(line):
            return index - 1
    return max(len(lines) - 1, 0)


def load_fragment(path: Path) -> Fragment:
    """Read and validate one fragment file.

    Read as ``utf-8-sig``, so a byte-order mark is dropped instead of becoming
    the bullet's invisible first character; plain UTF-8 decodes unchanged.

    Raises ``FragmentError`` naming the file for: a name outside the grammar or
    vocabulary, a file that cannot be read, bytes that are not UTF-8, an empty
    body, a body that opens with a list marker/heading/fence, and a duplicated
    issue reference (only for a name that already supplies one — a slug
    fragment is free to end its own text with ``(#735, #737)``).
    """
    name, type_, ref = parse_fragment_name(path.name)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FragmentError(f"{path.name}: not valid UTF-8 — save the file as UTF-8 text") from exc
    except OSError as exc:
        # An unreadable fragment is the contributor's to fix like any other
        # malformed one, and ``validate_dir`` should list it with the rest
        # rather than the gate dying on it with a traceback.
        raise FragmentError(f"{path.name}: cannot read — {exc.strerror or exc}") from exc
    lines = normalize_lines(text)
    if not lines:
        raise FragmentError(f"{path.name}: empty — write one user-facing sentence")
    if FIRST_LINE_REJECT_RE.match(lines[0]):
        raise FragmentError(
            f"{path.name}: a fragment must open with prose — drop the leading marker "
            f"(the fold writes the '- ' itself)"
        )
    if ref is not None and TRAILING_REF_RE.search(lines[opening_paragraph_end(lines)]):
        raise FragmentError(
            f"{path.name}: the issue ref comes from the filename — drop it from the text "
            f"(the fold appends ' (#{ref})')"
        )
    return Fragment(path=path, name=name, type=type_, ref=ref, lines=tuple(lines))


def validate_dir(directory: Path) -> tuple[list[Fragment], list[str]]:
    """Load every fragment in *directory*, returning the good and the bad.

    Errors are collected, never raised, so one run reports every offender.
    ``README.md`` is skipped by name — it is the directory's own documentation
    and the only file that stays. Files that are not ``*.md`` are ignored
    (editor droppings, ``.DS_Store``); subdirectories are an error, because the
    directory is flat and a fragment hiding in one would silently never ship.
    A directory that does not exist is empty, not an error: a checkout from
    before this landed still has to pass the gate.
    """
    fragments: list[Fragment] = []
    errors: list[str] = []
    if not directory.is_dir():
        return fragments, errors
    for entry in sorted(directory.iterdir()):
        if entry.name == KEEP:
            continue
        if entry.is_dir():
            errors.append(f"{entry.name}/: {FRAGMENT_DIR}/ is flat — no subdirectories")
            continue
        if entry.suffix != ".md":
            continue
        try:
            fragments.append(load_fragment(entry))
        except FragmentError as exc:
            errors.append(str(exc))
    return fragments, errors


def unreleased_span(lines: Sequence[str]) -> tuple[int, int] | None:
    r"""Locate the ``## [Unreleased]`` section as ``(heading, end)`` indices.

    ``lines[heading]`` is the heading itself and ``end`` is exclusive, so the
    block both the gate and the fold work on is ``lines[heading + 1:end]``. The
    section ends at the next release heading — ``^## \[`` and nothing else, the
    same terminator ``release.yml`` uses — so a bracket-less ``## Notes``
    written inside the block is part of it, while ``## [2026.8.0]`` closes it
    even before the release date has been filled in. An ``[Unreleased]`` that is
    the last section runs to the end of the file.

    Returns ``None`` when the heading is absent; every caller has something
    different to say about that, so this one says nothing.
    """
    for heading in range(len(lines)):
        if UNRELEASED_RE.match(lines[heading]):
            for end in range(heading + 1, len(lines)):
                if SECTION_END_RE.match(lines[end]):
                    return heading, end
            return heading, len(lines)
    return None


def _unreleased_block(lines: Sequence[str]) -> list[str] | None:
    """The lines inside ``## [Unreleased]``, or ``None`` when the heading is absent."""
    span = unreleased_span(lines)
    if span is None:
        return None
    return list(lines[span[0] + 1 : span[1]])


def is_empty_block(block: Sequence[str]) -> bool:
    """True when *block* holds no non-blank line.

    That is the state a release rotation leaves behind — the section is emptied
    down to a single blank line — and it is what tells the gate that a pull
    request which touched ``## [Unreleased]`` or deleted fragments is the
    release, not a contributor writing a bullet by hand.
    """
    return not any(line.strip() for line in block)


def render_bullet(frag: Fragment) -> list[str]:
    """Render *frag* as the lines of one ``CHANGELOG.md`` bullet.

    The first line gets ``- `` and every line after it two spaces. That indent
    is what keeps a sub-bullet a sub-bullet and a fenced block fenced once the
    body sits inside a list item; apart from it the body is copied unchanged,
    so a fence survives byte for byte. A blank line stays blank rather than
    becoming two spaces, because trailing whitespace in ``CHANGELOG.md`` is
    somebody else's diff noise later.

    The issue reference, when the filename supplies one, is appended to the end
    of the opening paragraph *before* any prefixing — ``opening_paragraph_end``
    explains why that is not simply the body's last line.

    Raises ``FragmentError`` for an ``internal`` fragment. It renders nothing
    at all, so a caller asking for its bullet has skipped the check that keeps
    it out of the changelog, and rendering one anyway would ship exactly the
    sentence the type exists to avoid writing.
    """
    if frag.type == "internal":
        raise FragmentError(
            f"{frag.path.name}: an internal fragment renders nothing — "
            f"the fold skips it and deletes the file"
        )
    lines = list(frag.lines)
    if frag.ref is not None:
        end = opening_paragraph_end(lines)
        lines[end] = f"{lines[end]} (#{frag.ref})"
    return ["- " + lines[0]] + ["  " + line if line else "" for line in lines[1:]]


def _type_heading_index(lines: Sequence[str], heading: str) -> int | None:
    """Index of the first ``### <heading>`` line inside ``## [Unreleased]``.

    The search is confined to that block, so the identically named heading of
    an already-released section — the file has one ``### Fixed`` per release —
    is never touched. It stops at the first match, so a block that has somehow
    grown two ``### Fixed`` headings gets its bullets in the one a reader
    reaches first rather than in whichever came last.
    """
    span = unreleased_span(lines)
    if span is None:
        return None
    start, end = span
    for index in range(start + 1, end):
        match = TYPE_HEADING_RE.match(lines[index])
        if match is not None and match.group(1) == heading:
            return index
    return None


def _insert_after_heading(lines: list[str], index: int, block: Sequence[str]) -> None:
    """Insert *block* after ``lines[index]`` and that heading's single blank line.

    Headings in this file are followed by one blank line; where one is missing
    it is supplied rather than the bullets being welded onto the heading text.
    New bullets sit directly on top of whatever bullet was already there — the
    same shape a hand-written entry would have had. A blank line is added below
    them only when the block would otherwise run straight into a heading, which
    happens when the type heading being written to had no bullets at all; a
    bullet already under that heading needs no separator and must not get one,
    and a block that already ends blank — the created-headings block does —
    must not get a second.
    """
    at = index + 1
    if at < len(lines) and not lines[at].strip():
        at += 1
    else:
        lines.insert(at, "")
        at += 1
    lines[at:at] = block
    below = at + len(block)
    if below < len(lines) and lines[below].startswith("#") and block and block[-1].strip():
        lines.insert(below, "")


def apply_fragments(text: str, fragments: Sequence[Fragment]) -> tuple[str, list[str]]:
    """Fold *fragments* into the ``## [Unreleased]`` section of *text*.

    Bullets are grouped by their type's heading and ordered within a group by
    filename — a plain string sort, so ``115`` precedes ``745`` and a slug
    sorts after both. A heading that already exists receives its bullets in
    place, wherever in the block it sits; one that does not is created at the
    top of the block in Keep-a-Changelog order. Existing headings are never
    merged, moved or reordered: this runs once per release against a file whose
    history is 6000 lines long, and rewriting that is not its business.

    ``internal`` fragments render nothing. They still appear in the report,
    because the release still has to delete their files.

    Returns the new text — LF throughout, exactly one trailing newline — and
    one report line per fragment, in the order the fold used, with the
    ``internal`` ones last. An empty *fragments* hands *text* back untouched.

    Raises ``FragmentError`` when *text* has no ``## [Unreleased]`` heading,
    which means the rotation has already happened and the fold has nowhere to
    write; that check runs before the empty-*fragments* shortcut, because a
    changelog in that state is worth hearing about either way.
    """
    lines = _split_lines(text)
    if lines and lines[-1] == "":
        lines.pop()
    span = unreleased_span(lines)
    if span is None:
        raise FragmentError("no ## [Unreleased] heading — run apply before the release rotation")
    # Every insertion below lands after this line, so its index never moves and
    # the creation pass can use it without re-locating the section.
    unreleased_index = span[0]
    if not fragments:
        return text, []

    grouped: dict[str, list[Fragment]] = {}
    internal: list[Fragment] = []
    for frag in fragments:
        if frag.type == "internal":
            internal.append(frag)
        else:
            grouped.setdefault(HEADING_FOR[frag.type], []).append(frag)
    for bucket in grouped.values():
        bucket.sort(key=lambda frag: frag.path.name)
    internal.sort(key=lambda frag: frag.path.name)

    report = [
        f"{FRAGMENT_DIR}/{frag.path.name} -> ### {heading}"
        + (f" (#{frag.ref})" if frag.ref is not None else "")
        for heading in HEADING_ORDER
        for frag in grouped.get(heading, ())
    ]
    report += [f"{FRAGMENT_DIR}/{frag.path.name} -> internal, not rendered" for frag in internal]
    if not grouped:
        return text, report

    # Existing headings first: creating the new ones afterwards puts them at the
    # top of the block without having to track how far the earlier inserts moved
    # everything below them.
    pending: list[tuple[str, list[str]]] = []
    for heading in HEADING_ORDER:
        group = grouped.get(heading)
        if not group:
            continue
        bullets = [line for frag in group for line in render_bullet(frag)]
        index = _type_heading_index(lines, heading)
        if index is None:
            pending.append((heading, bullets))
        else:
            _insert_after_heading(lines, index, bullets)

    if pending:
        created: list[str] = []
        for heading, bullets in pending:
            created += [f"### {heading}", "", *bullets, ""]
        _insert_after_heading(lines, unreleased_index, created)

    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n", report


def needs_fragment(changed: Iterable[str]) -> bool:
    """True when any changed path lives under a gated directory.

    The prefixes carry their trailing slash, so the test is directory-scoped:
    ``src/osprey/cli.py`` is gated, ``srcs/x.py`` and ``docs/source/x.rst`` are
    not, and neither is a ``src/`` nested somewhere below the repo root.
    """
    return any(path.startswith(GATED_PREFIXES) for path in changed)


def _fragment_dir_paths(paths: Iterable[str]) -> list[str]:
    """Every path under ``changelog.d/`` that is not its README."""
    prefix = f"{FRAGMENT_DIR}/"
    keep = f"{prefix}{KEEP}"
    return [path for path in paths if path.startswith(prefix) and path != keep]


def _named_fragment_paths(paths: Iterable[str]) -> list[str]:
    """Fragment-directory paths whose filename is a valid fragment name.

    ``NAME_RE`` admits no ``/``, so a file smuggled into a subdirectory fails
    here too — which is what the flat-directory rule wants, since such a file
    would never be folded in.
    """
    prefix = f"{FRAGMENT_DIR}/"
    return [path for path in _fragment_dir_paths(paths) if NAME_RE.match(path[len(prefix) :])]


def _listed(paths: Sequence[str], limit: int = 5) -> list[str]:
    """*paths* as detail lines, truncated to *limit* with a count of the rest.

    A refactor can touch a hundred files; the point of the list is to show the
    contributor which change tripped the gate, and five is enough for that.
    """
    shown = list(paths[:limit])
    if len(paths) > limit:
        shown.append(f"+{len(paths) - limit} more")
    return shown


def _bullet_count(block: Sequence[str]) -> int:
    """Number of top-level bullets in *block*."""
    return sum(1 for line in block if BULLET_RE.match(line))


def _release_count(lines: Sequence[str]) -> int:
    """Number of ``## [`` headings in *lines*, ``[Unreleased]`` included.

    This is how the gate recognizes the release pull request, and it has to be
    a *transition*: once fragments are carrying the changelog, ``[Unreleased]``
    is empty on every ordinary pull request too, so an empty block on its own
    says nothing. Cutting a release is the one thing that adds a heading —
    compare the count on both sides and only the release moves.
    """
    return sum(1 for line in lines if SECTION_END_RE.match(line))


def gate_failures(
    changed: Sequence[str],
    added: Sequence[str],
    deleted: Sequence[str],
    head_text: str,
    base_text: str,
) -> tuple[list[str], list[str]]:
    """Apply the three pull-request rules to one diff.

    *changed*, *added* and *deleted* are repository-relative paths from
    ``git diff --name-status`` between the merge base and HEAD; *head_text* and
    *base_text* are the two versions of ``CHANGELOG.md``. Nothing here touches
    the filesystem or git, so the caller decides where the diff came from.

    The rules:

    1. A change under ``src/`` or ``packages/`` needs a fragment *added* by
       this pull request. One already in the tree does not count — the gate
       reads committed history, so neither does one that is merely written.
    2. ``## [Unreleased]`` must come out of the merge base unchanged. Three
       shapes are allowed through: the release rotation, which empties it; a
       pull request that changes ``CHANGELOG.md`` and nothing else without
       growing the bullet count, which is a correction; and a head whose block
       is byte-identical to the base's.
    3. Only that rotation deletes fragments.

    The rotation is a *transition*, not a state: an empty ``[Unreleased]`` is
    what every pull request sees once fragments are carrying the changelog, so
    it is recognized by the release heading the release pull request adds
    (``_release_count``). Reading the head alone would exempt every pull
    request from rule 3 for the whole cycle between releases.

    Returns ``(failures, ok_lines)``. Each failure is a headline followed by
    detail lines — the caller prints the headline after ``✗ `` and indents the
    rest — and every ok line already carries its own ``✓``. Both lists are in
    rule order, so a run that trips everything reads top to bottom.

    A *base_text* of ``""`` (no ``CHANGELOG.md`` at the merge base) is read as
    a base with no heading, so its block is empty and the comparison proceeds
    normally rather than the gate having a special opinion about it. That path
    is for direct callers only: ``check`` treats a failing
    ``git show <base>:CHANGELOG.md`` as an environment error and never reaches
    here with an empty *base_text*.
    """
    failures: list[str] = []
    ok_lines: list[str] = []
    prefixes = ", ".join(GATED_PREFIXES)

    gated = [path for path in changed if needs_fragment((path,))]
    if not gated:
        ok_lines.append(
            f"✓ changelog gate: no {' or '.join(GATED_PREFIXES)} changes — no fragment required"
        )
    else:
        fragments = sorted(_named_fragment_paths(added))
        if fragments:
            ok_lines.append(
                f"✓ changelog gate: {fragments[0]} added for "
                f"{len(gated)} changed file(s) under {prefixes}"
            )
        else:
            failures.append(
                "\n".join(
                    [
                        f"no changelog fragment for {len(gated)} changed file(s) under {prefixes}",
                        *_listed(gated),
                        f"add {FRAGMENT_DIR}/<name>.<type>.md — <name> is the issue number when "
                        f"there is one, a short slug otherwise; for <type>, {_VOCABULARY}",
                        "the gate reads committed history — an unstaged fragment does not count",
                    ]
                )
            )

    base_lines = _split_lines(base_text)
    base_block = _unreleased_block(base_lines) or []
    head_lines = _split_lines(head_text)
    head_block = _unreleased_block(head_lines)
    rotation = False
    if head_block is None:
        failures.append(
            "CHANGELOG.md has no ## [Unreleased] heading — the heading must exist, "
            "because apply folds the fragments into it"
        )
    else:
        rotation = is_empty_block(head_block) and _release_count(head_lines) > _release_count(
            base_lines
        )
        if head_block == base_block:
            ok_lines.append("✓ [Unreleased]: untouched")
        elif rotation:
            ok_lines.append("✓ [Unreleased]: empty (rotation)")
        elif set(changed) == {"CHANGELOG.md"} and _bullet_count(head_block) <= _bullet_count(
            base_block
        ):
            # Editing or dropping a bullet that is already there is a correction,
            # and a correction has nowhere else to go. Counting bullets cannot
            # tell a reworded one from a swapped one, which is accepted: a pull
            # request that touches no other file cannot collide with a feature.
            ok_lines.append("✓ [Unreleased]: changelog-only correction")
        else:
            failures.append(
                "this PR adds to ## [Unreleased] by hand — move the entry into "
                f"{FRAGMENT_DIR}/<name>.<type>.md; corrections to existing bullets go in a "
                "CHANGELOG-only PR"
            )

    # A head with no heading is not a rotation, whatever the empty block would
    # otherwise suggest: the fold writes into that heading, so a diff that has
    # removed it has not folded anything.
    gone = _fragment_dir_paths(deleted)
    if gone and not rotation:
        failures.append(
            "\n".join(
                [
                    f"{len(gone)} fragment(s) deleted — only the release PR's apply removes "
                    f"fragments",
                    *_listed(gone),
                    "restore them; they are folded in and deleted when a release is cut",
                ]
            )
        )
    return failures, ok_lines


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run *argv* at the repository root and capture its text output.

    The default runner, and the only place in this module that starts a
    process. ``check`` is often invoked from a subdirectory — a pre-merge
    script, an editor task — so the working directory is pinned to the
    repository rather than inherited.
    """
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT)


def _emit(message: str, *, mark: str = "✗", stream: TextIO | None = None) -> None:
    """Print *message* as one marked headline plus indented detail lines.

    Every failure this tool reports is a headline a reader can scan and detail
    they only need once they have stopped scanning, so the two are formatted
    differently rather than run together.
    """
    headline, *detail = message.split("\n")
    target = stream if stream is not None else sys.stdout
    print(f"{mark} {headline}", file=target)
    for line in detail:
        print(f"  {line}", file=target)


def _environment_error(message: str) -> int:
    """Report *message* on stderr and hand back the environment exit code.

    Exit 2 is the code that says "not your fault": an unfetched base ref, a
    shallow clone, a git call that failed. It goes to stderr because it is not
    part of the gate's report — there is no report, the gate never ran.
    """
    _emit(message, stream=sys.stderr)
    return 2


def resolve_base(ref: str, run: Runner) -> str | None:
    """Resolve *ref* to a commit, preferring the remote for a plain branch name.

    A plain name — letters, digits, ``.``, ``_``, ``-`` and nothing else, and
    not ``HEAD`` — is almost always a branch, and in a fresh CI checkout the
    only copy of that branch is the remote one, so ``origin/<ref>`` is tried
    first and the local name second. Anything else (``HEAD~1``, a sha, a
    ``refs/`` path, a name that already says ``origin/``) is taken at its word
    first and only then looked for under ``origin/``.

    Returns the commit sha, or ``None`` when neither spelling resolves.
    """
    plain = PLAIN_REF_RE.match(ref) is not None and ref != "HEAD"
    candidates = [f"origin/{ref}", ref] if plain else [ref, f"origin/{ref}"]
    for candidate in candidates:
        result = run(["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"])
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def parse_name_status(stdout: str) -> list[tuple[str, str]]:
    """Parse ``git diff --name-status -z`` output into ``(status, path)`` pairs.

    ``-z`` is not a convenience here: a path with a space in it is ambiguous in
    the default output and this repository has such paths. The NUL-separated
    form alternates status and path, so the split list is read two at a time
    and a trailing empty element (every record is NUL-*terminated*) is ignored.
    """
    fields = [field for field in stdout.split("\0") if field != ""]
    return [(fields[index], fields[index + 1]) for index in range(0, len(fields) - 1, 2)]


def _check(directory: Path, changelog: Path, base: str, run: Runner) -> int:
    """Run the gate against the diff between *base*'s merge base and HEAD."""
    fragments, errors = validate_dir(directory)
    for error in errors:
        _emit(error)
    if not errors:
        print(f"✓ {FRAGMENT_DIR}: {len(fragments)} fragment(s) valid")

    def environment_error(message: str) -> int:
        """Exit 2, saying so without hiding the fragment errors already printed.

        The two codes answer different questions and only one can be returned,
        so the one that is not returned is spelled out instead of being left
        for the contributor to infer from output that scrolled past.
        """
        if errors:
            message += (
                f"\n{len(errors)} fragment error(s) reported above are separate "
                f"and still have to be fixed"
            )
        return _environment_error(message)

    base_sha = resolve_base(base, run)
    if base_sha is None:
        return environment_error(
            f"base ref {base} not found — git fetch origin <branch>\n"
            f"in CI, actions/checkout needs fetch-depth: 0"
        )

    merge_base = run(["git", "merge-base", base_sha, "HEAD"])
    if merge_base.returncode != 0 or not merge_base.stdout.strip():
        return environment_error(
            f"git merge-base {base_sha} HEAD found no common ancestor — shallow clone? "
            f"git fetch --unshallow\n{merge_base.stderr.strip()}"
        )
    point = merge_base.stdout.strip()

    diff_argv = ["git", "diff", "--name-status", "-z", "--no-renames", point, "HEAD"]
    diff = run(diff_argv)
    if diff.returncode != 0:
        return environment_error(f"{' '.join(diff_argv)} failed\n{diff.stderr.strip()}")
    pairs = parse_name_status(diff.stdout)
    # "changed" is every path the pull request touched, whatever git did to it:
    # deleting a module is as much a user-visible change as editing one.
    changed = [path for _, path in pairs]
    added = [path for status, path in pairs if status.startswith("A")]
    deleted = [path for status, path in pairs if status.startswith("D")]

    show_argv = ["git", "show", f"{point}:CHANGELOG.md"]
    show = run(show_argv)
    if show.returncode != 0:
        return environment_error(f"{' '.join(show_argv)} failed\n{show.stderr.strip()}")

    try:
        head_text = changelog.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return environment_error(f"cannot read {changelog}\n{exc}")

    failures, ok_lines = gate_failures(changed, added, deleted, head_text, show.stdout)
    for line in ok_lines:
        print(line)
    for failure in failures:
        _emit(failure)
    return 1 if errors or failures else 0


def _apply(directory: Path, changelog: Path) -> int:
    """Fold every fragment in *directory* into *changelog* and delete them."""
    fragments, errors = validate_dir(directory)
    if errors:
        for error in errors:
            _emit(error)
        return 1

    try:
        text = changelog.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _environment_error(f"cannot read {changelog}\n{exc}")

    # Run the fold even with nothing to fold: it is what notices that the
    # ## [Unreleased] heading is gone, and a release that has already rotated
    # is worth hearing about before anyone writes the next fragment.
    try:
        new_text, report = apply_fragments(text, fragments)
    except FragmentError as exc:
        _emit(str(exc))
        return 1

    if not fragments:
        print(f"{FRAGMENT_DIR}: no fragments; nothing to do")
        return 0

    try:
        changelog.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        return _environment_error(f"cannot write {changelog}\n{exc}")

    # The changelog is already written, so a fragment left on disk would be
    # folded in twice by the next run. Name the survivors rather than leaving
    # that to be discovered in the release diff.
    survivors: list[str] = []
    for fragment in fragments:
        try:
            fragment.path.unlink()
        except OSError as exc:
            survivors.append(f"{fragment.path}: {exc}")
    if survivors:
        return _environment_error(
            "\n".join(
                [
                    f"{changelog} is written, but {len(survivors)} fragment(s) could not "
                    f"be deleted — remove them by hand or the next apply folds them again",
                    *survivors,
                ]
            )
        )

    for line in report:
        print(line)
    print(f"stage with: git add -A {FRAGMENT_DIR}/ CHANGELOG.md")
    return 0


def main(argv: list[str] | None = None, run: Runner | None = None) -> int:
    """Entry point for ``check`` and ``apply``.

    *run* is the git runner, injected by the tests so that none of them has to
    build a repository. Exit codes: 0 clean, 1 something a contributor fixes,
    2 something the environment has to fix.
    """
    parser = argparse.ArgumentParser(
        prog="changelog_fragments.py",
        description="Check for a changelog fragment, or fold the fragments into CHANGELOG.md.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    check_parser = subcommands.add_parser("check", help="gate one pull request (CI, pre-merge)")
    check_parser.add_argument(
        "--base",
        default="origin/main",
        help="branch this pull request merges into (default: origin/main)",
    )
    check_parser.add_argument("--dir", help=f"fragment directory (default: {FRAGMENT_DIR}/)")
    check_parser.add_argument("--changelog", help="changelog to read (default: CHANGELOG.md)")

    apply_parser = subcommands.add_parser("apply", help="fold the fragments in (release)")
    apply_parser.add_argument("--changelog", help="changelog to write (default: CHANGELOG.md)")
    apply_parser.add_argument("--dir", help=f"fragment directory (default: {FRAGMENT_DIR}/)")

    args = parser.parse_args(argv)
    directory = Path(args.dir) if args.dir else REPO_ROOT / FRAGMENT_DIR
    changelog = Path(args.changelog) if args.changelog else REPO_ROOT / "CHANGELOG.md"
    if args.command == "apply":
        return _apply(directory, changelog)
    return _check(directory, changelog, args.base, run or _run)


if __name__ == "__main__":
    sys.exit(main())
