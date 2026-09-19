"""Unit tests for the shared ``X-Osprey-Owner`` header guard.

The guard is the single place two independent readers — the Bluesky bridge's
add path and the event dispatcher's ``manual_fire``/retry route — decide
whether a header value names a human. These tests pin both halves of that
contract: which shapes are accepted verbatim, and that every refusal is silent
about the value it refused while still saying *why* it refused.
"""

import logging

import pytest

from osprey.utils.owner_header import MAX_OWNER_LENGTH, OWNER_HEADER, owner_from_header


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


class TestAcceptedValues:
    """Values that name a roster account survive unchanged."""

    @pytest.mark.parametrize("value", ["alice", "op1", "alice-b", "alice.b", "alice_b", "A"])
    def test_roster_shaped_names_are_returned_verbatim(self, value, caplog):
        with caplog.at_level(logging.DEBUG):
            assert owner_from_header(value) == value
        assert _warnings(caplog) == []

    def test_name_at_the_length_ceiling_is_accepted(self, caplog):
        value = "a" * MAX_OWNER_LENGTH
        with caplog.at_level(logging.DEBUG):
            assert owner_from_header(value) == value
        assert _warnings(caplog) == []


class TestAbsentHeader:
    """A missing header is the ordinary owner-less case, not a refusal."""

    def test_none_is_owner_less_and_silent(self, caplog):
        with caplog.at_level(logging.DEBUG):
            assert owner_from_header(None) is None
        assert _warnings(caplog) == []


class TestRefusedValues:
    """Every malformed shape lands on ``None`` plus exactly one warning."""

    @pytest.mark.parametrize(
        ("value", "shape"),
        [
            ("", "empty"),
            ("   ", "empty"),
            ("${OSPREY_TERMINAL_USER}", "placeholder"),
            ("bob/../x", "path"),
            ("..", "path"),
            (".", "path"),
            ("bob\\x", "path"),
            ("a" * (MAX_OWNER_LENGTH + 1), "long"),
            ("älice", "charset"),
            ("alice bob", "charset"),
            ("alice;rm", "charset"),
        ],
    )
    def test_malformed_value_is_owner_less_with_one_warning(self, value, shape, caplog):
        with caplog.at_level(logging.DEBUG):
            assert owner_from_header(value) is None
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert OWNER_HEADER in message
        assert shape in message.lower()

    @pytest.mark.parametrize(
        "value",
        [
            "${OSPREY_TERMINAL_USER}",
            "bob/../x",
            "a" * (MAX_OWNER_LENGTH + 1),
            "älice",
            "alice;rm -rf /",
        ],
    )
    def test_warning_never_repeats_the_rejected_value(self, value, caplog):
        with caplog.at_level(logging.DEBUG):
            owner_from_header(value)
        message = _warnings(caplog)[0].getMessage()
        assert value not in message
        # Nor any distinctive fragment of it: the point is that a forged header
        # cannot write attacker-chosen text into an operator's log.
        for fragment in ("OSPREY_TERMINAL_USER", "bob", "rm -rf", "älice"):
            assert fragment not in message


class TestModuleContract:
    """The constants two callers import rather than restate."""

    def test_header_name_is_the_wire_spelling(self):
        assert OWNER_HEADER == "X-Osprey-Owner"

    def test_length_ceiling_matches_the_roster_username_bound(self):
        assert MAX_OWNER_LENGTH == 64

    def test_module_is_a_stdlib_only_leaf(self):
        import osprey.utils.owner_header as module

        source = module.__file__
        assert source is not None
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "import osprey" not in text
        assert "from osprey" not in text
