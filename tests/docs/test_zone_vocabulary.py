"""Vocabulary gate: the deployment repo has FOUR zones, not three.

SOURCE, SECRETS, OUTPUT and STATE — the redesign that gave OUTPUT its own zone
retired every "three-zone" description of the layout. "four zones" is now the
only bare zone count that may appear in shipped text; the sole surviving use of
the old count is the exact phrase "three ignored zones" (three of the four
zones are git-ignored, one — SOURCE — is not), which no shipped line currently
needs but which the pattern below deliberately does not flag.

The sweep scans whitespace-collapsed text rather than raw lines, because a
wrapped docstring or comment can carry "three" and "zone(s)" on different
physical lines with words in between — `` and the three\\n# generated or
secret zones below`` reads as one "three ... zones" phrase despite the line
break, and a line-anchored grep would miss it.

The sweep covers ``src/osprey/`` and ``docs/source/``, matching the shape of
the other ``tests/docs/`` removal gates (see ``test_deploy_ops_retired.py`` and
``test_env_production_retired.py``). ``tests/`` itself is out of scope: plenty
of test code still narrates fixtures and helpers as "three-zone" (variable and
function names, docstrings) and rewriting all of it is a separate concern from
guarding the shipped vocabulary an operator or facility reader actually sees.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: "three" followed by up to four intervening tokens (or none, for the adjacent
#: "three-zone" / "three zones" spellings) and then "zone"/"zones". Tokens are
#: any non-whitespace run — not just letters — so a wrapped comment's leading
#: ``#`` does not break the match: the real wrapped case, "the source zone is
#: tracked, and the three\n# generated or secret zones below", collapses to
#: "...the three # generated or secret zones..." and must still be caught. The
#: window is wide enough for that without being so wide it also catches
#: unrelated "three ... zone" pairs landing in the same paragraph by
#: coincidence (see ALLOWLIST).
_ZONE_COUNT_PATTERN = re.compile(r"\bthree\b(?:[\s-]+\S+){0,4}[\s-]+zones?\b", re.IGNORECASE)

#: The one phrasing the rule still permits. Not currently used anywhere shipped
#: — every wrapped site the rename touched dropped the count instead — but the
#: sweep encodes the exception rather than assuming it will never be reached.
_PERMITTED_PHRASE = "three ignored zones"

#: Confirms the positive half of the rule, not just the absence of the old one.
#: A sweep that only asserts "three" is gone would pass vacuously if every zone
#: mention were deleted rather than renamed; this pattern requires "four zones"
#: to still be live somewhere in the shipped text.
_FOUR_ZONES_PATTERN = re.compile(r"\bfour[\s-]+zones?\b", re.IGNORECASE)

ROOTS = ("src/osprey", "docs/source")
SCAN_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Coincidental "three ... zone" pairs that are not about the zone count at
#: all — a nearby word, not a rewrite site. Keyed by (repo-relative path,
#: matched snippet) so an edit to the surrounding prose re-argues the
#: exemption rather than inheriting it silently.
_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "src/osprey/cli/profile_conventions.py",
        "three chances for the zone",
    ): "'three independent string literals' — a count of literals, not zones",
}


def _shipped_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        base = _REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _collapsed(text: str) -> str:
    """Whitespace-collapsed text, so a phrase split across lines reads as one."""
    return re.sub(r"\s+", " ", text)


def _stale_count_hits(root_dir: Path | None = None) -> list[tuple[str, str]]:
    """Every ``(repo-relative path, matched snippet)`` still naming a bare "three" zone count."""
    found: list[tuple[str, str]] = []
    base_root = root_dir if root_dir is not None else _REPO_ROOT
    for root in ROOTS:
        base = base_root / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover - defensive, no such file today
                continue
            collapsed = _collapsed(content)
            relative = str(path.relative_to(base_root))
            for match in _ZONE_COUNT_PATTERN.finditer(collapsed):
                snippet = match.group(0)
                if snippet.lower() == _PERMITTED_PHRASE:
                    continue
                found.append((relative, snippet))
    return found


def test_no_shipped_text_still_counts_three_zones() -> None:
    offenders = [hit for hit in _stale_count_hits() if hit not in _ALLOWLIST]
    assert offenders == [], (
        "The deployment repo has FOUR zones (SOURCE, SECRETS, OUTPUT, STATE). "
        "Text still describing a three-zone layout remains:\n"
        + "\n".join(f"{path}: {snippet!r}" for path, snippet in offenders)
    )


def test_every_exemption_still_has_something_to_explain() -> None:
    """A coincidental match that stops occurring should leave the allowlist.

    Snippet-keyed exemptions make this sharp: an edit to the allowlisted
    sentence — even a small one — changes the matched snippet and stops it
    matching, so the exemption is re-argued rather than inherited.
    """
    seen = set(_stale_count_hits())
    stale = sorted(set(_ALLOWLIST) - seen)
    assert stale == [], f"allowlist entries with no remaining occurrence: {stale}"


def test_four_zones_is_actually_live() -> None:
    """The guard against a vacuous pass: the new vocabulary must be present.

    A sweep that only forbids "three" would also pass if every zone count were
    deleted outright rather than renamed to "four" — that is a regression this
    project does not want to call a pass.
    """
    hits = 0
    for path in _shipped_sources():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - defensive, no such file today
            continue
        hits += len(_FOUR_ZONES_PATTERN.findall(_collapsed(content)))
    assert hits > 0, "no shipped text says 'four zones' — the vocabulary rename regressed"


@pytest.mark.parametrize(
    "path_parts,text",
    (
        (("docs", "source", "regress.rst"), "The repo has a three-zone layout on disk.\n"),
        (("src", "osprey", "regress.py"), '"""A three-zone repo, materialized fresh."""\n'),
        (
            ("src", "osprey", "regress_wrapped.py"),
            "# the source zone is tracked, and the three\n# generated or secret zones never are\n",
        ),
    ),
)
def test_the_sweep_would_catch_a_regression(
    path_parts: tuple[str, ...], text: str, tmp_path: Path
) -> None:
    """A sweep that finds nothing looks identical to one whose pattern is broken.

    Covers both an adjacent reintroduction (``three-zone``) and the wrapped
    shape the rename actually had to fix, so the newline-collapsing half of
    the sweep is exercised too, not just the simple case.
    """
    fake_root = tmp_path / "repo"
    target = fake_root
    for part in path_parts[:-1]:
        target = target / part
    target.mkdir(parents=True)
    (target / path_parts[-1]).write_text(text, encoding="utf-8")

    offenders = [hit for hit in _stale_count_hits(fake_root) if hit not in _ALLOWLIST]
    assert offenders, "the sweep should have flagged the reintroduced three-zone count"


def test_the_permitted_phrase_is_not_flagged() -> None:
    """The one phrasing the mechanical rule still allows must not trip the sweep."""
    collapsed = _collapsed("the deployment keeps three ignored zones and one tracked zone")
    matches = [m.group(0) for m in _ZONE_COUNT_PATTERN.finditer(collapsed)]
    assert all(m.lower() == _PERMITTED_PHRASE for m in matches), matches
