"""Every ``hookSpecificOutput`` a shipped hook builds names its event.

Claude Code validates the envelope a hook prints against the schema for the
event named in ``hookSpecificOutput.hookEventName``. Omit that field and the
CLI DISCARDS the output — the additional context never reaches the turn, the
permission decision never reaches the gate — and reports a non-blocking hook
error into the session transcript rather than into the hook's stderr. The hook
still exits 0, still prints well-formed JSON, and still looks, from every angle
a test usually takes, like it worked.

That is why this guard is static rather than behavioural: a hook whose block
silently never lands has no failing assertion anywhere to find it. The check
reads each shipped hook's source and requires the key on every
``hookSpecificOutput`` mapping it constructs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.hooks.test_hook_docstring_frontmatter import HOOKS_DIR

pytestmark = pytest.mark.unit

#: Every hook script plus the helpers, so a mapping built in a shared module is
#: covered too. Discovered rather than listed: a new hook joins this guard by
#: existing.
HOOK_SOURCES = sorted(path.name for path in Path(HOOKS_DIR).glob("osprey_*.py"))


def _hook_specific_outputs(tree: ast.AST) -> list[ast.Dict]:
    """Every mapping literal assigned to a ``hookSpecificOutput`` key."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "hookSpecificOutput"
                and isinstance(value, ast.Dict)
            ):
                found.append(value)
    return found


@pytest.mark.parametrize("hook_name", HOOK_SOURCES)
def test_every_hook_specific_output_carries_its_event_name(hook_name: str) -> None:
    source = (Path(HOOKS_DIR) / hook_name).read_text(encoding="utf-8")
    outputs = _hook_specific_outputs(ast.parse(source))

    for mapping in outputs:
        keys = {
            key.value
            for key in mapping.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert "hookEventName" in keys, (
            f"{hook_name} builds a hookSpecificOutput mapping without "
            f"'hookEventName' (keys: {sorted(keys)}) — Claude Code drops such an "
            "envelope and the hook's output never reaches the session"
        )


def test_the_guard_reads_at_least_one_envelope() -> None:
    """A parser that found nothing would pass every case above vacuously."""
    total = sum(
        len(_hook_specific_outputs(ast.parse((Path(HOOKS_DIR) / name).read_text(encoding="utf-8"))))
        for name in HOOK_SOURCES
    )
    assert total >= 10, f"only {total} hookSpecificOutput mappings found across {HOOK_SOURCES}"
