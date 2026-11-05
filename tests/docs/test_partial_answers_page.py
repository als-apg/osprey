"""The ``missing`` map example on the partial-answers contract page is one the tools state.

The page's JSON block loads, its one key is :data:`MISSING_KEY`, and its map is
accepted both by :func:`answer_map` for a tool that declares ``missing`` and by
the :class:`NothingAnsweredError` that carries it when nothing answered.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from osprey.mcp_server.errors import MAP_KEYS, MISSING_KEY, NothingAnsweredError, answer_map

PAGE = Path(__file__).parents[2] / "docs/source/reference/contracts/partial-answers.rst"


def _json_blocks(text: str) -> list[str]:
    blocks = re.findall(r"\.\. code-block:: json\n\n((?:   .*\n|\n)+)", text)
    return [block.strip() for block in blocks]


def _example() -> dict:
    blocks = _json_blocks(PAGE.read_text(encoding="utf-8"))
    assert len(blocks) == 1
    return json.loads(blocks[0])


def test_the_example_is_the_missing_map():
    assert list(_example()) == [MISSING_KEY]


@pytest.mark.parametrize("tool", [tool for tool, key in MAP_KEYS.items() if key == MISSING_KEY])
def test_a_missing_tool_states_the_example_map(tool: str):
    missing = _example()[MISSING_KEY]
    assert answer_map(tool, missing=missing) == _example()
    assert NothingAnsweredError(None, missing=missing).missing == missing


def test_each_example_text_is_a_source_text_or_null():
    """A channel's text is ``"<ExceptionType>: <message>"``; an item with none is ``null``."""
    for text in _example()[MISSING_KEY].values():
        assert text is None or re.fullmatch(r"[A-Za-z_]\w*: .+", text)


def test_a_block_that_is_not_json_fails_the_load():
    """Negative control: the block reader hands a malformed block to the loader as is."""
    (block,) = _json_blocks('.. code-block:: json\n\n   {"missing": {"a": }}\n')
    with pytest.raises(json.JSONDecodeError):
        json.loads(block)
