"""Every host knob the environment-variable page names must be a real one.

``docs/source/reference/configuration/environment-variables.rst`` is a
hand-kept inventory. Nothing produces it, and nothing until now compared it to
what the framework actually reads — so it drifts in both directions, quietly:

* a build argument gains a sibling in :data:`SITE_IMAGE_AXES` and the page's
  row keeps listing the old three, leaving a deployer with no way to learn the
  fourth exists;
* a variable is renamed or retired in the source and the page goes on
  advertising the dead spelling, which a reader exports and then watches do
  nothing.

Two sweeps close that, each against an artefact that exists for its own
reasons. The site build-arg axes are compared to their producer, the mapping
the compose generator resolves them from. Every other uppercase name on the
page is checked for an occurrence anywhere under the shipped tree — a weaker
bar deliberately, because a host knob is read through ``os.environ`` in one
module and there is no ledger of them to compare against; what it catches is
the name nothing in OSPREY reads at all.

Scope is narrow on purpose. Only inline literals whose whole content is an
uppercase environment-variable name are checked, so prose, config keys, file
paths, CLI verbs and ``${VAR}`` placeholders in the surrounding text pass by
untouched.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pytest

from osprey.deployment.compose_generator import SITE_IMAGE_AXES

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this sweep reads, relative to the repo root.
_PAGE = "docs/source/reference/configuration/environment-variables.rst"

#: The shipped trees a documented name has to occur in. A wheel carries
#: ``src/osprey`` and the ``packages`` distributions; anything outside them is
#: test or docs scaffolding, which cannot vouch for a name a deployer sets.
_SOURCE_ROOTS = ("src/osprey", "packages")

#: An RST inline literal: ``like this``. Content may not span lines or contain
#: a backtick, which is exactly what Sphinx itself accepts.
_LITERAL_PATTERN = re.compile(r"``([^`\n]+)``")

#: A literal that is entirely an environment-variable name. Anchored at both
#: ends so ``PIP_*`` (a family, not a name), ``${VAR}`` (a placeholder) and
#: ``OSPREY_OFFLINE=1`` (an assignment) are not mistaken for one.
_ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]+$")

#: Documented names deliberately absent from the shipped tree, each with the
#: reason it is allowed to be. Keyed by the name so an edit to the surrounding
#: prose does not silently inherit the exemption. Empty by design: an entry
#: here is a knob promised to a reader that no shipped module reads, so every
#: one needs an argument.
_EXEMPTIONS: dict[str, str] = {}


def _documented_names(root_dir: Path | None = None) -> list[tuple[int, str]]:
    """Every ``(line number, name)`` the page writes as an uppercase literal."""
    path = (root_dir if root_dir is not None else _REPO_ROOT) / _PAGE
    if not path.is_file():
        return []
    found: list[tuple[int, str]] = []
    content = path.read_text(encoding="utf-8", errors="ignore")
    for number, line in enumerate(content.splitlines(), start=1):
        for match in _LITERAL_PATTERN.finditer(line):
            literal = match.group(1).strip()
            if _ENV_NAME_PATTERN.match(literal):
                found.append((number, literal))
    return found


@lru_cache(maxsize=1)
def _shipped_text() -> str:
    """Every shipped source file, concatenated once for substring lookup.

    ``__pycache__`` is skipped the way the identity guard skips it: a compiled
    copy of a module that was edited yesterday would vouch for a name that no
    longer exists in the source.
    """
    chunks: list[str] = []
    for root in _SOURCE_ROOTS:
        base = _REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:  # pragma: no cover - defensive
                continue
    return "\n".join(chunks)


def test_every_site_build_arg_is_documented() -> None:
    """The rule: a build argument the generator resolves must be on the page."""
    documented = {name for _, name in _documented_names()}
    missing = sorted(set(SITE_IMAGE_AXES) - documented)
    assert missing == [], (
        "Every ARG name in SITE_IMAGE_AXES must appear on "
        f"{_PAGE} as an inline literal — a build argument a deployer cannot "
        f"find is one they cannot set. Undocumented: {missing}"
    )


def test_every_documented_variable_is_read_somewhere_in_the_source() -> None:
    """The reverse rule: the page may only name knobs the shipped tree reads."""
    shipped = _shipped_text()
    offenders = [
        (number, name)
        for number, name in _documented_names()
        if name not in _EXEMPTIONS and name not in shipped
    ]
    detail = [f"{_PAGE}:{number}: {name}" for number, name in offenders]
    assert offenders == [], (
        "Every environment variable the host-knob page names must occur "
        f"somewhere under {' or '.join(_SOURCE_ROOTS)} — a documented name no "
        "shipped module reads is a knob the reader exports and watches do "
        "nothing. Unread names remain:\n" + "\n".join(detail)
    )


def test_the_page_yields_literals() -> None:
    """Guard against the page silently moving or losing its table.

    Both sweeps above pass vacuously on an empty parse, so an absent page or a
    reworked table would read as clean. Pinning the parse says plainly which
    artefact broke.
    """
    documented = _documented_names()
    assert documented, f"{_PAGE} yielded no environment-variable literals at all"


def test_every_exemption_still_has_something_to_explain() -> None:
    """An exemption whose name is no longer documented should leave the list.

    Exemptions weaken a promise made to a reader, so they are not allowed to
    outlive the row that needed them.
    """
    documented = {name for _, name in _documented_names()}
    stale = sorted(set(_EXEMPTIONS) - documented)
    assert stale == [], f"exemption entries with no remaining occurrence: {stale}"


@pytest.mark.parametrize(
    "literal",
    (
        "PIP_*",
        "${VAR}",
        "images.registry",
        "osprey vendor fetch",
        ".env",
        "OSPREY_OFFLINE=1",
        "A",
    ),
)
def test_non_variable_literals_are_left_alone(literal: str, tmp_path: Path) -> None:
    """The page is full of literals that are not variable names.

    Config keys, commands, file names, families, placeholders and assignments
    all appear in inline literals here. None of them is a bare variable name
    and none may be checked against the source, or the sweep becomes noise the
    next author learns to ignore.
    """
    page = tmp_path / _PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"Knobs\n=====\n\nSee ``{literal}`` here.\n", encoding="utf-8")

    assert _documented_names(tmp_path) == [], f"{literal!r} was mistaken for a variable name"


def test_the_sweep_reads_the_real_page(tmp_path: Path) -> None:
    """A parser that finds nothing looks identical to a clean page.

    The positive control: a fake page naming a variable is parsed at its real
    line number, so the sweeps above are known to be looking at something.
    """
    page = tmp_path / _PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("Knobs\n=====\n\nSet ``OSPREY_MADE_UP_KNOB`` here.\n", encoding="utf-8")

    assert _documented_names(tmp_path) == [(4, "OSPREY_MADE_UP_KNOB")]
