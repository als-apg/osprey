"""The application SDK hands out the one input-digest recipe the approval binds to.

An application recomputes a digest only through ``osprey.apps.sdk.canonical_json``
and ``osprey.apps.sdk.input_digest``, the functions ``check_approved_call`` itself
recomputes with; ``tests/apps/test_digest_recipe_pin.py`` pins them to the hook's.
"""

from __future__ import annotations

import hashlib

import osprey.apps.sdk as sdk
from osprey.apps.sdk import _approval


def test_the_sdk_exports_the_recipe_the_check_uses():
    assert sdk.canonical_json is _approval.canonical_json
    assert sdk.input_digest is _approval.input_digest


def test_a_digest_is_sha256_over_the_canonical_selection():
    tool_input = {"setpoints": [1, 2.5], "mode": "abs", "note": "ignored"}
    expected = hashlib.sha256(b'{"mode":"abs","setpoints":[1,2.5]}').hexdigest()
    assert sdk.canonical_json({"setpoints": [1, 2.5], "mode": "abs"}) == (
        '{"mode":"abs","setpoints":[1,2.5]}'
    )
    assert sdk.input_digest(tool_input, ["setpoints", "mode"]) == expected
