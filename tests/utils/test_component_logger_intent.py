"""Tests for the ``osprey_intent`` marker ``ComponentLogger`` puts on every record.

The contract under test: every public intent method stamps its own name onto the
emitted record as ``record.osprey_intent``, levels are unchanged, and a
caller-supplied ``extra`` mapping survives the merge untouched. The attribute is
metadata for handlers (the CLI altitude gate) — it is deliberately invisible to
rendering, which ``test_noncli_handler_parity.py`` pins separately.
"""

from __future__ import annotations

import logging

import pytest

from osprey.utils.logger import ComponentLogger

#: Every public intent method, with the record level it must keep emitting.
#: Both halves are pinned here: a level remap would be as much a regression as a
#: missing marker.
INTENT_LEVELS: dict[str, int] = {
    "status": logging.INFO,
    "key_info": logging.INFO,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "success": logging.INFO,
    "timing": logging.INFO,
    "approval": logging.INFO,
    "resume": logging.INFO,
    "critical": logging.CRITICAL,
    "exception": logging.ERROR,
}


class _RecordCollector(logging.Handler):
    """Plain stdlib handler — no Rich — that keeps the records it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured(request):
    """A ``ComponentLogger`` on an isolated base logger plus its record sink."""
    base = logging.getLogger(f"osprey_intent_test.{request.node.name}")
    base.setLevel(logging.DEBUG)
    base.propagate = False  # keep records off the root handlers other tests own
    collector = _RecordCollector()
    base.handlers[:] = [collector]
    try:
        yield ComponentLogger(base, "test_component"), collector
    finally:
        base.handlers[:] = []


class TestIntentAttribute:
    @pytest.mark.parametrize("method", sorted(INTENT_LEVELS))
    def test_method_stamps_its_own_name(self, captured, method):
        logger, collector = captured

        getattr(logger, method)("hello")

        (record,) = collector.records
        assert record.osprey_intent == method

    @pytest.mark.parametrize(("method", "level"), sorted(INTENT_LEVELS.items()))
    def test_level_is_not_remapped(self, captured, method, level):
        logger, collector = captured

        getattr(logger, method)("hello")

        (record,) = collector.records
        assert record.levelno == level

    def test_attribute_reaches_a_plain_handler(self, captured):
        """A non-Rich handler sees the marker as an ordinary record attribute."""
        logger, collector = captured

        logger.key_info("important")

        (record,) = collector.records
        assert getattr(record, "osprey_intent", None) == "key_info"
        assert record.getMessage() == "important"

    def test_intents_are_distinguishable_at_the_same_level(self, captured):
        """``info`` and ``key_info`` share INFO — only the marker separates them."""
        logger, collector = captured

        logger.info("plain")
        logger.key_info("important")

        assert [r.levelno for r in collector.records] == [logging.INFO, logging.INFO]
        assert [r.osprey_intent for r in collector.records] == ["info", "key_info"]

    def test_percent_style_args_still_format(self, captured):
        logger, collector = captured

        logger.info("value is %s", 42)

        (record,) = collector.records
        assert record.getMessage() == "value is 42"
        assert record.osprey_intent == "info"


class TestCallerExtra:
    def test_caller_keys_survive_alongside_the_marker(self, captured):
        logger, collector = captured

        logger.warning("careful", extra={"request_id": "abc", "attempt": 2})

        (record,) = collector.records
        assert record.request_id == "abc"
        assert record.attempt == 2
        assert record.osprey_intent == "warning"

    def test_caller_dict_is_not_mutated(self, captured):
        logger, _ = captured
        caller_extra = {"request_id": "abc"}

        logger.info("hello", extra=caller_extra)

        assert caller_extra == {"request_id": "abc"}

    def test_marker_wins_over_a_caller_supplied_intent(self, captured):
        """The marker is Osprey's, not a caller's — the emitting method decides it."""
        logger, collector = captured

        logger.error("boom", extra={"osprey_intent": "spoofed"})

        (record,) = collector.records
        assert record.osprey_intent == "error"


class TestPreservedBehaviour:
    def test_event_system_kwargs_are_still_stripped(self, captured):
        """Legacy callers pass these; they must not reach ``Logger.log``."""
        logger, collector = captured

        logger.error(
            "boom",
            error="ValueError",
            error_type="ValueError",
            recoverable=True,
            stack_trace="...",
            warning="w",
        )

        (record,) = collector.records
        assert record.osprey_intent == "error"
        assert not hasattr(record, "error_type")

    def test_exception_still_attaches_traceback(self, captured):
        logger, collector = captured

        try:
            raise ValueError("nope")
        except ValueError:
            logger.exception("failed")

        (record,) = collector.records
        assert record.exc_info is not None
        assert record.osprey_intent == "exception"

    def test_error_honours_exc_info_flag(self, captured):
        logger, collector = captured

        try:
            raise ValueError("nope")
        except ValueError:
            logger.error("failed", exc_info=True)

        (record,) = collector.records
        assert record.exc_info is not None
        assert record.osprey_intent == "error"
