"""One definition decides what an environment-variable name is.

``osprey_connectors.connection.ENV_NAME_RE`` is it. The checkers that read it (a connection
block's ``auth.*_env``, the telemetry collector's ``auth.token_env``, the executor's
``child_env_passthrough``, a build profile's ``env:`` names and ``bind_env``) therefore cannot
disagree about an edge case.
"""

import pytest

from osprey_connectors.connection import ENV_NAME_RE


@pytest.mark.parametrize("name", ["A", "_", "_PRIVATE", "OTLP_TOKEN", "no_proxy", "mixedCase_1"])
def test_a_variable_name_matches(name):
    assert ENV_NAME_RE.match(name)
    assert ENV_NAME_RE.fullmatch(name)


@pytest.mark.parametrize(
    "value",
    ["", "1A", "A-B", "A B", "A=B", "${A}", "A\n", "\nA", "A\nB", " A", "A "],
    ids=[
        "empty",
        "leading-digit",
        "dash",
        "space",
        "equals",
        "reference",
        "trailing-newline",
        "leading-newline",
        "embedded-newline",
        "leading-space",
        "trailing-space",
    ],
)
def test_anything_else_does_not_match_either_way(value):
    """``match`` and ``fullmatch`` agree, so no caller can pick the looser one."""
    assert ENV_NAME_RE.match(value) is None
    assert ENV_NAME_RE.fullmatch(value) is None
