"""The closed session-key grammar every surface asks.

:func:`~osprey.interfaces.web_terminal.session_key.is_posture_key` is the one
answer the posture, chat, hand-off and agent-turn routes, the terminal
websocket and the workspace file routes give to "is this a session key". It
accepts a canonical lowercase UUID and nothing else, answers ``False`` rather
than raising for any non-string, and requires the whole string to be the key.
"""

from __future__ import annotations

import uuid

import pytest

from osprey.interfaces.web_terminal.session_key import is_posture_key

_KEY = "cccccccc-1111-2222-3333-444444444444"


class TestIsPostureKey:
    def test_a_canonical_lowercase_uuid_is_a_key(self) -> None:
        assert is_posture_key(str(uuid.uuid4())) is True
        assert is_posture_key(_KEY) is True

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            " ",
            "not-a-uuid",
            "-" * 36,
            "0" * 36,
            "0" * 32,
            "AAAAAAAA-1111-2222-3333-444444444444",
            "{" + _KEY + "}",
            "urn:uuid:" + _KEY,
            _KEY + "\n",
            " " + _KEY,
            _KEY + " ",
            "operator-deadbeef",
            "../../../etc",
            123,
            1.5,
            True,
            ["x"],
            {"a": 1},
        ],
        ids=[
            "none",
            "empty",
            "space",
            "not-a-uuid",
            "dashes-36",
            "zeros-36",
            "bare-hex-32",
            "uppercase",
            "braced",
            "urn",
            "trailing-newline",
            "leading-space",
            "trailing-space",
            "operator-key",
            "traversal",
            "int",
            "float",
            "bool",
            "list",
            "dict",
        ],
    )
    def test_anything_else_is_not_a_key(self, value: object) -> None:
        assert is_posture_key(value) is False
