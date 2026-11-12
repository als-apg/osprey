"""A preset's comments point at sections that are where the comment says.

``osprey init`` copies a preset into the deployment repo as ``profile.yml``,
comments included, and those comments are the operator's guide to the file.
"`the `dispatch:` section above`" is followed by scrolling up.
"""

from __future__ import annotations

import re

import pytest

from tests.docs.test_environment_variable_page import _REPO_ROOT

_PRESETS = _REPO_ROOT / "src" / "osprey" / "profiles" / "presets"

#: "the `a:`, `b:` and `c:` sections above" — the keys, then the direction.
_POINTER = re.compile(
    r"((?:`[a-z_]+:`(?:,| and| or)?\s+)+)(?:field|block|section)s?\s+(above|below)\b"
)


def _comment_paragraphs(lines: list[str]) -> list[tuple[int, str]]:
    """Each run of comment lines as one string, with the line it starts on."""
    paragraphs: list[tuple[int, list[str]]] = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        if paragraphs and paragraphs[-1][0] + len(paragraphs[-1][1]) == number:
            paragraphs[-1][1].append(stripped.lstrip("# "))
        else:
            paragraphs.append((number, [stripped.lstrip("# ")]))
    return [(start, " ".join(body)) for start, body in paragraphs]


@pytest.mark.parametrize("preset", sorted(_PRESETS.glob("*.yml")), ids=lambda path: path.name)
def test_a_section_named_above_or_below_is_there(preset) -> None:
    lines = preset.read_text(encoding="utf-8").splitlines()
    sections = {
        match.group(1): number
        for number, line in enumerate(lines, 1)
        if (match := re.match(r"([a-z_]+):", line))
    }
    offences = []
    for start, paragraph in _comment_paragraphs(lines):
        for keys, direction in _POINTER.findall(paragraph):
            for key in re.findall(r"`([a-z_]+):`", keys):
                where = sections.get(key)
                if where is None:
                    offences.append(f"{preset.name}:{start}: no top-level `{key}:` section")
                elif (where < start) != (direction == "above"):
                    offences.append(
                        f"{preset.name}:{start}: `{key}:` (line {where}) is not {direction}"
                    )
    assert not offences, "\n".join(offences)
