"""The launch pin as a three-valued term: ``none``, ``pinned`` or ``unreadable``.

:func:`~osprey_connectors.posture_store._launch_pin_term` reads the executor's
launch stamp and says, for one target, whether the run was pinned read-only at
launch. The term fails closed: a stamp that is present but cannot be read is
``unreadable`` for every target, and the verdict refuses it as
``launch_pinned`` exactly as it refuses ``pinned``.
"""

from __future__ import annotations

import pytest

from osprey_connectors import posture_store
from osprey_connectors.types import CONTROL_TARGETS

ENV = posture_store.LAUNCH_POSTURE_ENV_VAR


@pytest.fixture
def stamp(monkeypatch):
    """Set the launch stamp to *value*, or remove it when *value* is ``None``."""

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, value)

    return _set


def _terms() -> dict[str | None, str]:
    return {t: posture_store._launch_pin_term(t) for t in [*CONTROL_TARGETS, None]}


def test_an_absent_stamp_is_none_for_every_target(stamp):
    stamp(None)
    assert set(_terms().values()) == {"none"}


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_a_blank_stamp_is_none_for_every_target(stamp, value):
    stamp(value)
    assert set(_terms().values()) == {"none"}


def test_a_target_sandbox_stamp_pins_that_target_only(stamp):
    stamp("live=sandbox")
    terms = _terms()
    assert terms["live"] == "pinned"
    assert terms["va"] == "none"
    assert terms["standin"] == "none"


def test_a_target_sandbox_stamp_pins_a_caller_that_names_no_target(stamp):
    # The most restrictive narrowing wins when the caller cannot say which
    # machine it is about, as in the record clause.
    stamp("live=sandbox")
    assert posture_store._launch_pin_term(None) == "pinned"


def test_the_stamp_written_by_launch_posture_stamp_reads_back_pinned(stamp):
    stamp(posture_store.launch_posture_stamp("va", posture_store.POSTURE_SANDBOX))
    assert posture_store._launch_pin_term("va") == "pinned"
    assert posture_store._launch_pin_term("live") == "none"


def test_an_all_targets_sandbox_stamp_pins_every_target(stamp):
    stamp("*=sandbox")
    assert set(_terms().values()) == {"pinned"}


def test_an_unnamed_target_stamp_from_the_writer_pins_every_target(stamp):
    stamp(posture_store.launch_posture_stamp(None, posture_store.POSTURE_SANDBOX))
    assert set(_terms().values()) == {"pinned"}


def test_a_writes_stamp_is_none_for_every_target(stamp):
    stamp("va=writes")
    assert set(_terms().values()) == {"none"}


def test_an_all_targets_writes_stamp_is_none_for_every_target(stamp):
    stamp("*=writes")
    assert set(_terms().values()) == {"none"}


def test_surrounding_whitespace_is_tolerated(stamp):
    stamp("  live = sandbox \n")
    assert posture_store._launch_pin_term("live") == "pinned"
    assert posture_store._launch_pin_term("va") == "none"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("mars=sandbox", id="unknown-target"),
        pytest.param("mars=writes", id="unknown-target-writes"),
        pytest.param("sandbox", id="no-separator"),
        pytest.param("live", id="no-separator-target"),
        pytest.param("=sandbox", id="empty-target"),
        pytest.param("live=bananas", id="garbage-posture"),
        pytest.param("live=", id="empty-posture"),
        pytest.param("live=SANDBOX", id="case-mismatched-posture"),
        pytest.param("live=sandbox=writes", id="two-separators"),
    ],
)
def test_an_unreadable_stamp_is_unreadable_for_every_target(stamp, value):
    stamp(value)
    assert set(_terms().values()) == {"unreadable"}


def test_an_unknown_target_is_unreadable_even_for_a_named_known_target(stamp):
    # Regression guard: a tolerant reading would keep an unknown target as a
    # pin of that name only, which would leave every real target "none".
    stamp("mars=sandbox")
    assert posture_store._launch_pin_term("live") == "unreadable"


def test_the_term_is_read_on_every_call(stamp):
    stamp("live=sandbox")
    assert posture_store._launch_pin_term("live") == "pinned"
    stamp(None)
    assert posture_store._launch_pin_term("live") == "none"


def test_launch_pin_term_is_private():
    assert "_launch_pin_term" not in posture_store.__all__
    assert not hasattr(posture_store, "launch_pin_term")
