"""The docs describe the write switch in its three positions: off, asked, allowed.

The switch on the control-target chip has three positions per login and
machine, the tier key is three-valued (``never`` / ``asked`` / ``allowed``) and
the switch's position, not a per-tool approval policy, decides whether a
governed machine write asks. A page that still says a switch is turned *on*,
that the shipped ``readwrite`` tier spells ``writes: allowed``, or that spells
``approval.tools.<tool>`` for a governed tool teaches a reader the old model.

Scanned: every ``.rst`` page and every diagram (``_diagrams/*.html``) under
``docs/source``. The changelog is not under ``docs/source`` and stays as it was
written. The allow-list below is closed: each entry must still match, so a
quoted phrase that leaves the code leaves the list too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
DIAGRAMS = DOCS / "_diagrams"

#: Tools the switch's position decides for by default. A per-tool policy for
#: any of them is retired; spelling one in a docs example teaches a dead key.
GOVERNED_TOOLS = (
    "channel_write",
    "execute_file",
    "execute",
    "pyaml_measure",
    "queue_add",
    "queue_start",
)
_GOVERNED = "|".join(GOVERNED_TOOLS)

#: Two-position switch vocabulary: a switch turned on, writes on, a switch that
#: starts off, one login's on or off. Each pattern is scoped to the write
#: switch so an unrelated feature that is "turned on" stays legal.
ON_OFF_PATTERNS = (
    re.compile(r"\bturn(?:s|ed|ing)?\s+(?:\w+'s\s+)?writes\s+(?:back\s+)?on\b", re.I),
    re.compile(r"\bswitch(?:es|ed|ing)?\s+writes\s+on\b", re.I),
    re.compile(r"[·*`]\s*writes\s+on\b", re.I),
    re.compile(r"'s\s+writes\s+on\b", re.I),
    re.compile(r"\bwith\s+writes\s+on\b", re.I),
    re.compile(
        r"\bturn(?:s|ed|ing)?\s+(?:a|one|the|that|this|each|its|every)\s+"
        r"(?:real\s+)?(?:machine|simulator|switch)(?:'s\s+(?:writes|switch))?\s+(?:back\s+)?on\b",
        re.I,
    ),
    re.compile(r"\bturn(?:s|ed|ing)?\s+(?:its|their|the|a|that)\s+switch\s+on\b", re.I),
    re.compile(r"\bswitch(?:es)?\s+(?:that\s+)?starts?\s+off\b", re.I),
    re.compile(r"\bswitch(?:es)?\s+on\b(?!\s+(?:the|a|an|this|that|its|their|every|each)\b)", re.I),
    re.compile(r"\bwrite\s+switch(?:es)?\s+(?:is|are)\s+(?:on|off)\b", re.I),
    re.compile(r"'s\s+(?:recorded\s+)?\"?on\"?\s+(?:or\s+off\s+)?(?:stays|never|inert)", re.I),
    re.compile(r"\bwrites?\s+(?:switch\s+)?on\s+or\s+off\b", re.I),
    re.compile(r"\bswitch\s+(?:is\s+)?on/off\b", re.I),
)

#: ``writes: allowed`` near ``readwrite``: the shipped readwrite tier spells
#: ``asked``. A window of lines around each spelling is checked.
_ALLOWED_SPELLING = re.compile(r"writes:\s*allowed\b")
_READWRITE = re.compile(r"\breadwrite\b")
_READWRITE_WINDOW = (6, 2)

#: A retired per-tool approval policy for a governed tool, dotted or as YAML.
RETIRED_POLICY_PATTERNS = (
    re.compile(rf"\bapproval\.tools\.(?:{_GOVERNED})\b"),
    re.compile(rf"^\s*(?:{_GOVERNED}):\s*(?:always|selective|skip|ask|never)\b"),
)

#: Closed allow-list: (page relative to docs/source, exact text). The text is
#: blanked out of the page before it is scanned, and must be found there.
#: Each entry quotes a runtime phrase verbatim or describes the retirement.
ALLOW = (
    # The connectors package's switch_off refusal phrase, quoted verbatim.
    ("how-to/web-terminal/operate.rst", "turn it on from the chip"),
    ("reference/contracts/connectors.rst", "turn it on from the chip"),
    ("architecture/python-executor.rst", "turn it on from the chip"),
    # The notebook kernel's refusal phrase, quoted verbatim.
    ("how-to/web-terminal/notebooks.rst", "Turn writes on from the chip"),
    # OSPREY_PREBUILT_IMAGES, an environment switch unrelated to writes.
    ("how-to/deploy-project/compose-templates.rst", "turn the switch on"),
)


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.rst"))


def _diagrams() -> list[Path]:
    return sorted(DIAGRAMS.glob("*.html"))


def _rel(path: Path) -> str:
    return path.relative_to(DOCS).as_posix()


def _scrubbed(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for page, phrase in ALLOW:
        if page == _rel(path):
            text = text.replace(phrase, " " * len(phrase))
    return text


def _offences_in(label: str, text: str) -> list[str]:
    """Every old-model spelling in *text*, each named with *label*."""
    flat = re.sub(r"\s+", " ", text)  # prose patterns read across wrapped lines
    found = [
        f"{label}: on/off wording {m.group(0)!r}" for p in ON_OFF_PATTERNS for m in p.finditer(flat)
    ]
    lines = text.splitlines()
    before, after = _READWRITE_WINDOW
    for index, line in enumerate(lines):
        if _ALLOWED_SPELLING.search(line):
            window = "\n".join(lines[max(0, index - before) : index + after + 1])
            if _READWRITE.search(window):
                found.append(f"{label}:{index + 1}: readwrite spelled `writes: allowed`")
        for pattern in RETIRED_POLICY_PATTERNS:
            match = pattern.search(line)
            if match:
                found.append(
                    f"{label}:{index + 1}: retired governed policy {match.group(0).strip()!r}"
                )
    return found


def _offences(path: Path) -> list[str]:
    return _offences_in(_rel(path), _scrubbed(path))


def test_no_rst_page_uses_the_two_position_switch_words():
    offences = [o for page in _pages() for o in _offences(page)]
    assert offences == [], "\n".join(offences)


@pytest.mark.parametrize("diagram", _diagrams(), ids=lambda p: p.name)
def test_no_diagram_uses_the_two_position_switch_words(diagram: Path):
    offences = _offences(diagram)
    assert offences == [], "\n".join(offences)


@pytest.mark.parametrize("page,phrase", ALLOW, ids=lambda v: str(v)[:40])
def test_every_allow_list_entry_still_matches(page: str, phrase: str):
    path = DOCS / page
    assert path.is_file(), f"allow-list names a missing page: {page}"
    assert phrase in path.read_text(encoding="utf-8"), f"stale allow-list entry: {page}: {phrase!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Turn writes on for the simulator.",
        "● Rehearsal · writes on ▾",
        "With the simulator's writes on, try again.",
        "Your write switch starts off on each machine.",
        "the person turns one machine on from the chip",
        "Alice writes once she turns its switch on.",
        "one login's on or off never changes the other's",
        "each write: owner's switch on? within limits?",
    ],
)
def test_the_on_off_patterns_catch_the_old_wording(text: str):
    flat = re.sub(r"\s+", " ", text)
    assert any(p.search(flat) for p in ON_OFF_PATTERNS), text


@pytest.mark.parametrize(
    "text",
    [
        "Writes asked: every write prompts first.",
        "Set the real machine to off and work on the simulator.",
        "Expansion turned on or off rebuilds the mirror.",
        "the attribute writes on DeviceProxy",
        "The control-assistant preset turns it on; delete the lines to opt out.",
        "the switch is off by default and is turned on with an env var",
    ],
)
def test_the_on_off_patterns_leave_other_wording_alone(text: str):
    assert not any(p.search(text) for p in ON_OFF_PATTERNS), text


def test_a_readwrite_tier_spelling_allowed_is_caught():
    text = "``tiers/readwrite.yml``:\n\n.. code-block:: yaml\n\n   control_system:\n     writes: allowed\n"
    assert _offences_in("probe", text)


def test_an_admin_tier_spelling_allowed_is_not_caught():
    text = (
        "``tiers/admin.yml``:\n\n.. code-block:: yaml\n\n   control_system:\n     writes: allowed\n"
    )
    assert _offences_in("probe", text) == []


@pytest.mark.parametrize("line", ["approval.tools.channel_write: always", "    execute: selective"])
def test_a_retired_governed_policy_is_caught(line: str):
    assert any(p.search(line) for p in RETIRED_POLICY_PATTERNS)


@pytest.mark.parametrize("line", ["approval.tools.manual_fire", "    entry_create: always"])
def test_a_kept_policy_is_not_caught(line: str):
    assert not any(p.search(line) for p in RETIRED_POLICY_PATTERNS)
