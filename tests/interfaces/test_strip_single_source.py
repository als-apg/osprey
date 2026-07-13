"""Single-source guard for the ``CLAUDE_CODE_*`` env-strip logic.

Prevents the "two diverging copies" regression: the strip comprehension must
live ONLY in ``osprey.interfaces.web_terminal.env_utils.strip_claude_code_env``.
Both ``operator_session.py`` (the SDK path) and ``pty_manager.py`` (the PTY
path) must call that shared helper rather than re-implementing the inline
comprehension. This test fails fast if anyone reintroduces a local copy.
"""

from __future__ import annotations

import inspect

from osprey.interfaces.web_terminal import env_utils, operator_session, pty_manager

# The tuple literal that drives the strip. It must appear exactly once across
# the package — inside the shared helper — and never inline in a consumer.
_STRIP_LITERAL = '("CLAUDECODE", "CLAUDE_CODE_")'


def _source(module) -> str:
    return inspect.getsource(module)


def test_strip_literal_defined_exactly_once():
    """The strip comprehension literal lives only in the shared helper."""
    counts = {
        "env_utils": _source(env_utils).count(_STRIP_LITERAL),
        "operator_session": _source(operator_session).count(_STRIP_LITERAL),
        "pty_manager": _source(pty_manager).count(_STRIP_LITERAL),
    }
    assert counts["env_utils"] == 1, (
        f"expected the strip literal exactly once in env_utils, got {counts['env_utils']}"
    )
    total = sum(counts.values())
    assert total == 1, (
        f"strip literal {_STRIP_LITERAL!r} must be defined exactly once across the "
        f"web_terminal package; found {counts}"
    )


def test_consumers_do_not_reimplement_strip():
    """Neither consumer re-implements the inline strip comprehension."""
    assert _STRIP_LITERAL not in _source(operator_session), (
        "operator_session.py re-implements the CLAUDE_CODE_ strip; call "
        "strip_claude_code_env() from env_utils instead."
    )
    assert _STRIP_LITERAL not in _source(pty_manager), (
        "pty_manager.py re-implements the CLAUDE_CODE_ strip; call "
        "strip_claude_code_env() from env_utils instead."
    )


def test_both_consumers_call_shared_helper():
    """Both consumer modules invoke the shared helper."""
    assert "strip_claude_code_env(" in _source(operator_session), (
        "operator_session.py does not call the shared strip_claude_code_env() helper"
    )
    assert "strip_claude_code_env(" in _source(pty_manager), (
        "pty_manager.py does not call the shared strip_claude_code_env() helper"
    )


def test_helper_defined_exactly_once():
    """The helper function is defined in exactly one place."""
    defs = sum(
        _source(mod).count("def strip_claude_code_env")
        for mod in (env_utils, operator_session, pty_manager)
    )
    assert defs == 1, f"strip_claude_code_env must be defined exactly once; found {defs}"
