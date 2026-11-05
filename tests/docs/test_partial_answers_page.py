"""The ``missing`` map example on the partial-answers contract page is one the tools state.

The page's JSON block loads and its one key is :data:`MISSING_KEY`.
:func:`answer_map` and :class:`NothingAnsweredError` take any map, so passing
the example through them checks its shape only. The channel entry's text is
checked against what ``channel_read`` itself writes for a read that raised.
"""

from __future__ import annotations

import builtins
import json
import re
from pathlib import Path

import pytest

from osprey.mcp_server.control_system.tools.channel_read import _gather_readings
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


async def test_the_channel_entry_is_the_text_channel_read_writes():
    """A read raising the example's error leaves the example's text under its address."""
    texts = {key: text for key, text in _example()[MISSING_KEY].items() if text is not None}
    ((address, text),) = texts.items()
    kind, message = text.split(": ", 1)
    error = getattr(builtins, kind)(message)

    class Connector:
        async def read_channel(self, channel: str) -> object:
            if channel == address:
                raise error
            return object()

    _, failures = await _gather_readings(Connector(), [address, "SR:OTHER"])
    assert failures == {address: text}


def test_each_example_text_is_a_source_text_or_null():
    """Shape only: a channel's text is ``"<ExceptionType>: <message>"``, or ``null``."""
    for text in _example()[MISSING_KEY].values():
        assert text is None or re.fullmatch(r"[A-Za-z_]\w*: .+", text)


def test_a_block_that_is_not_json_fails_the_load():
    """Negative control: the block reader hands a malformed block to the loader as is."""
    (block,) = _json_blocks('.. code-block:: json\n\n   {"missing": {"a": }}\n')
    with pytest.raises(json.JSONDecodeError):
        json.loads(block)
