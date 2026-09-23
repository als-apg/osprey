"""The docs name models by id, never under a haiku / sonnet / opus key.

``haiku``, ``sonnet`` and ``opus`` are Claude Code's own alias names. A YAML
example that spells one as a key is a tier map, which no catalog entry or
profile accepts, so a reader who copies it gets a refused build. The one place
the words stand as keys is Claude Code's alias map — a gateway's
``claude_code_aliases:`` block, or ``claude_code.aliases.<name>`` — and those
are the only parents this allows.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"

_ALIAS_KEY = re.compile(r"^(?P<indent>\s*)(haiku|sonnet|opus):")
_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[\w.-]+):")
_ALLOWED_PARENTS = ("claude_code_aliases", "claude_code.aliases", "aliases")


def _parent_key(lines: list[str], index: int, indent: int) -> str | None:
    """The nearest key above ``lines[index]`` that is indented less than it."""
    for line in reversed(lines[:index]):
        match = _KEY.match(line)
        if match and len(match.group("indent")) < indent:
            return match.group("key")
    return None


def test_no_rst_spells_a_model_tier_as_a_yaml_key():
    offenders = []
    for path in sorted(DOCS.rglob("*.rst")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = _ALIAS_KEY.match(line)
            if match is None:
                continue
            parent = _parent_key(lines, index, len(match.group("indent")))
            if parent not in _ALLOWED_PARENTS:
                offenders.append(f"{path.relative_to(DOCS)}:{index + 1}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)
