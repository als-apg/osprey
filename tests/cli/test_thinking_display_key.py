"""The ``claude_code.thinking.display`` key at build: one reader, one refusal."""

from pathlib import Path

import pytest

from osprey.cli.templates.claude_code import _thinking_display, config_derived_context
from osprey_connectors.errors import BuildProfileError


def _config(thinking: object) -> dict:
    return {"claude_code": {"thinking": thinking}}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "summarized"),
        ({"claude_code": None}, "summarized"),
        ({"claude_code": {}}, "summarized"),
        (_config(None), "summarized"),
        (_config({}), "summarized"),
        (_config({"display": "summarized"}), "summarized"),
        (_config({"display": "omitted"}), "omitted"),
    ],
    ids=[
        "no-claude-code-block",
        "null-claude-code-block",
        "empty-claude-code-block",
        "null-thinking-block",
        "empty-thinking-block",
        "summarized",
        "omitted",
    ],
)
def test_thinking_display_value(config: dict, expected: str) -> None:
    assert _thinking_display(config) == expected


@pytest.mark.parametrize(
    "thinking",
    [
        {"display": "full"},
        {"display": "Summarized"},
        {"display": ""},
        {"display": True},
        {"display": 1},
        "summarized",
        ["summarized"],
    ],
    ids=["unknown-word", "wrong-case", "empty", "bool", "int", "scalar-block", "list-block"],
)
def test_a_bad_value_is_refused_at_build(thinking: object) -> None:
    with pytest.raises(BuildProfileError, match=r"claude_code\.thinking\.display") as info:
        _thinking_display(_config(thinking))
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("config", "shown"),
    [
        ({}, True),
        (_config({"display": "summarized"}), True),
        (_config({"display": "omitted"}), False),
    ],
    ids=["default", "summarized", "omitted"],
)
def test_context_carries_show_thinking_summaries(config: dict, shown: bool, tmp_path: Path) -> None:
    ctx = config_derived_context(config, tmp_path)
    assert ctx["show_thinking_summaries"] is shown


def test_context_refuses_a_bad_value(tmp_path: Path) -> None:
    with pytest.raises(BuildProfileError, match=r"claude_code\.thinking\.display"):
        config_derived_context(_config({"display": "full"}), tmp_path)
