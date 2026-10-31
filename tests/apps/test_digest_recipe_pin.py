"""The application SDK and the approval hook compute one input digest.

The hook binds an approval to the input it showed with
``sha256(canonical_json({k: input[k] for k in over}))``
(``osprey_prompt_render.input_digest``); the SDK recomputes it over the
arguments the tool server receives. The hook ships stdlib-only beside the
hooks and cannot be imported by the SDK, so the recipe exists twice. These
tests pin the two equal.
"""

from __future__ import annotations

import pytest

from osprey.apps.sdk import _approval
from tests.hooks.conftest import import_hook

VECTORS = [
    ({"value": 4.2}, ["value"]),
    ({"mode": "abs", "setpoints": [1, 2.5, -3]}, ["mode", "setpoints"]),
    ({"b": {"y": 1, "x": [True, None]}, "a": "z"}, ["b", "a"]),
    ({"name": "Ünïcødé ‮"}, ["name"]),
    ({"a": 1, "b": 2}, []),
    ({"a": 1, "b": 2}, ["b"]),
]


@pytest.fixture(scope="module")
def render():
    return import_hook("osprey_prompt_render")


@pytest.mark.parametrize(("tool_input", "over"), VECTORS)
def test_sdk_digest_equals_hook_digest(render, tool_input, over):
    assert _approval.input_digest(tool_input, over) == render.input_digest(tool_input, over)


@pytest.mark.parametrize("value", [{"x": 1, "y": [2, {"z": 3}]}, [1.5, "s"], "ä", None])
def test_sdk_canonical_json_equals_hook_canonical_json(render, value):
    assert _approval.canonical_json(value) == render.canonical_json(value)


def test_both_refuse_nan(render):
    with pytest.raises(ValueError):
        render.input_digest({"v": float("nan")}, ["v"])
    with pytest.raises(ValueError):
        _approval.input_digest({"v": float("nan")}, ["v"])


def test_both_raise_on_a_missing_key(render):
    with pytest.raises(KeyError):
        render.input_digest({"a": 1}, ["a", "b"])
    with pytest.raises(KeyError):
        _approval.input_digest({"a": 1}, ["a", "b"])
