"""Tests for the chat's tool-result preview (``tool_preview.tool_result_preview``).

A tool row shows a preview of its result, on the live stream and in a reloaded
transcript alike. These pin what a reader relies on: an MCP envelope is peeled
off, a JSON preview stays valid JSON however long the result, every cut says it
made one, the budget holds, and the background-agent launch mark survives.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.agent_runner import ToolResultEvent
from osprey.interfaces.web_terminal.operator_session import _ASYNC_LAUNCH_MARK, _event_to_wire
from osprey.interfaces.web_terminal.routes.session import _chat_turns, _transcript_nodes
from osprey.interfaces.web_terminal.tool_preview import (
    MAX_DEPTH,
    MAX_ITEMS,
    MAX_PARSE_DEPTH,
    PLAIN_TEXT_CHARS,
    STRING_CHARS,
    STRUCTURED_CHARS,
    tool_result_preview,
)


def envelope(value: object) -> str:
    """The MCP envelope the SDK wraps a tool's JSON result in."""
    return json.dumps({"result": json.dumps(value)})


class TestEnvelope:
    """The envelope around a tool's text is peeled off before the preview is cut."""

    @pytest.mark.parametrize(
        ("text", "preview"),
        [
            pytest.param(
                envelope({"status": "success", "n": 3}), '{"status":"success","n":3}', id="json"
            ),
            pytest.param(
                json.dumps({"result": envelope({"status": "ok"})}),
                '{"status":"ok"}',
                id="nested-envelopes",
            ),
            pytest.param('{"result": "Wrote 3 rows"}', "Wrote 3 rows", id="plain-text"),
            pytest.param('{"result": 42}', "42", id="number"),
        ],
    )
    def test_the_envelope_is_peeled_off_to_what_it_holds(self, text, preview):
        assert tool_result_preview(text) == (preview, False)

    def test_an_envelope_beside_another_key_is_kept_whole(self):
        assert tool_result_preview('{"result": "ok", "warnings": []}') == (
            '{"result":"ok","warnings":[]}',
            False,
        )


class TestJsonCuts:
    """A JSON result keeps its shape: long values are cut inside it, and each cut is marked."""

    def test_a_short_result_is_compact_json_and_not_shortened(self):
        assert tool_result_preview(envelope({"a": 1, "b": [1, 2]})) == (
            '{"a":1,"b":[1,2]}',
            False,
        )

    @pytest.mark.parametrize("length", [1000, 100_000])
    def test_a_long_string_keeps_its_head_and_counts_the_rest(self, length):
        preview, shortened = tool_result_preview(envelope({"text": "x" * length}))
        rest = length - STRING_CHARS
        assert json.loads(preview) == {"text": "x" * STRING_CHARS + f"… +{rest} chars"}
        assert shortened is True

    def test_a_long_list_keeps_its_first_items_and_counts_the_rest(self):
        preview, shortened = tool_result_preview(envelope({"items": list(range(50))}))
        assert json.loads(preview) == {"items": [*range(MAX_ITEMS), "… +30 more"]}
        assert shortened is True

    def test_a_long_object_keeps_its_first_keys_and_counts_the_rest(self):
        preview, shortened = tool_result_preview(envelope({f"k{i}": i for i in range(50)}))
        parsed = json.loads(preview)
        assert list(parsed) == [f"k{i}" for i in range(MAX_ITEMS)] + ["…"]
        assert parsed["…"] == "+30 more"
        assert shortened is True

    def test_nesting_past_the_depth_cap_is_replaced_by_a_marker(self):
        value: object = {"leaf": 1}
        for _ in range(MAX_DEPTH + 3):
            value = {"next": value}
        preview, shortened = tool_result_preview(envelope(value))
        node = json.loads(preview)
        for _ in range(MAX_DEPTH):
            node = node["next"]
        assert node["next"] == "{…}"
        assert shortened is True

    def test_a_list_past_the_depth_cap_is_marked_as_a_list(self):
        value: object = [1]
        for _ in range(MAX_DEPTH + 2):
            value = [value]
        preview, shortened = tool_result_preview(envelope(value))
        node = json.loads(preview)
        for _ in range(MAX_DEPTH):
            node = node[0]
        assert node[0] == "[…]"
        assert shortened is True

    def test_nesting_at_the_parse_depth_is_read_as_json_and_capped(self):
        text = "[" * MAX_PARSE_DEPTH + "]" * MAX_PARSE_DEPTH
        expected = "[" * (MAX_DEPTH + 1) + '"[…]"' + "]" * (MAX_DEPTH + 1)
        assert tool_result_preview(text) == (expected, True)

    def test_brackets_inside_a_string_are_not_nesting(self):
        text = json.dumps({"text": "[" * (MAX_PARSE_DEPTH + 5) + '\\"{' * 3})
        preview, shortened = tool_result_preview(text)
        assert json.loads(preview) == json.loads(text)
        assert shortened is False


class TestBudget:
    """However large the result, the preview stays within the budget and stays valid JSON."""

    def test_a_large_result_fits_the_budget_and_stays_valid_json(self):
        rows = [{"name": f"pv{i}", "note": "n" * 500} for i in range(500)]
        preview, shortened = tool_result_preview(envelope({"rows": rows}))
        assert len(preview) <= STRUCTURED_CHARS
        assert len(json.loads(preview)["rows"]) <= MAX_ITEMS + 1
        assert shortened is True

    def test_the_cuts_tighten_until_the_preview_fits(self):
        items = ["y" * 290 for _ in range(20)]
        preview, shortened = tool_result_preview(envelope({"items": items}))
        parsed = json.loads(preview)
        # Ten kept at the tighter 150-character cut, and one marker for the rest.
        assert len(parsed["items"]) == 11
        assert parsed["items"][0] == "y" * 150 + "… +140 chars"
        assert parsed["items"][-1] == "… +10 more"
        assert len(preview) <= STRUCTURED_CHARS
        assert shortened is True

    def test_a_structure_whose_keys_alone_exceed_the_budget_is_cut_as_text(self):
        preview, shortened = tool_result_preview(envelope({"k" * 3000: 1, "j" * 3000: 2}))
        assert len(preview) == STRUCTURED_CHARS
        assert shortened is True


class TestPlainText:
    """Text that is not JSON is cut at its head, and only past the plain-text limit."""

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("Wrote 3 rows", id="short"),
            pytest.param("42.0", id="a-number"),
            pytest.param('"quoted"', id="a-quoted-word"),
            pytest.param("", id="empty"),
            pytest.param('{"result": "{\\"status\\": ', id="invalid-json"),
            pytest.param("[1, 2", id="an-unclosed-list"),
            pytest.param("b" * PLAIN_TEXT_CHARS, id="exactly-the-plain-limit"),
            pytest.param(
                "[" * (MAX_PARSE_DEPTH + 1) + "]" * (MAX_PARSE_DEPTH + 1),
                id="nested-one-past-the-parse-depth",
            ),
        ],
    )
    def test_text_within_the_plain_limit_is_returned_as_it_is(self, text):
        assert tool_result_preview(text) == (text, False)

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("a" * 2000, id="plain"),
            pytest.param("{" + "x" * 2000, id="invalid-json"),
            pytest.param("[" * 5000 + "]" * 5000, id="nested-too-deep-to-parse"),
        ],
    )
    def test_text_past_the_plain_limit_is_cut_at_its_head(self, text):
        assert tool_result_preview(text) == (text[:PLAIN_TEXT_CHARS], True)


class TestSystemReminders:
    """The CLI's notes to the model are not part of what a tool returned."""

    _NOTE = (
        "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>"
    )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(_NOTE, "", id="only-a-note"),
            pytest.param(f"line one\n{_NOTE}\n", "line one", id="text-then-a-note"),
            pytest.param(
                f"<system-reminder>a\nmultiline\nnote</system-reminder>{_NOTE}rest",
                "rest",
                id="several-notes-across-lines",
            ),
        ],
    )
    def test_notes_are_left_out_of_the_preview(self, text, expected):
        assert tool_result_preview(text) == (expected, False)

    def test_a_note_after_a_json_result_does_not_keep_it_from_reading_as_json(self):
        preview, _ = tool_result_preview(f'{{"status": "ok", "count": 3}}\n{self._NOTE}')
        assert json.loads(preview) == {"status": "ok", "count": 3}


class TestNonFiniteNumbers:
    """JSON cannot send NaN or infinity, so they reach the page as text."""

    def test_nan_and_infinity_become_strings_so_the_preview_is_valid_json(self):
        preview, shortened = tool_result_preview("[NaN, Infinity, -Infinity, 1e400]")
        assert json.loads(preview) == ["NaN", "Infinity", "-Infinity", "1e400"]
        assert shortened is False


class TestAsyncLaunchMark:
    """The background agent's launch result keeps its mark in every form a preview takes."""

    @pytest.mark.parametrize(
        ("text", "cut"),
        [
            pytest.param(
                f"{_ASYNC_LAUNCH_MARK} successfully.\nagentId: a1", False, id="plain-text"
            ),
            pytest.param(
                json.dumps({"result": f"{_ASYNC_LAUNCH_MARK} successfully."}), False, id="envelope"
            ),
            pytest.param(
                envelope({"status": f"{_ASYNC_LAUNCH_MARK} successfully.", "report": "r" * 5000}),
                True,
                id="long-structured-result",
            ),
        ],
    )
    def test_the_mark_survives(self, text, cut):
        preview, shortened = tool_result_preview(text)
        assert _ASYNC_LAUNCH_MARK in preview
        assert shortened is cut


def _write_transcript(path: Path, tool_content: object) -> None:
    """A transcript in which one prompt runs one tool, whose result is ``tool_content``."""
    stamp = datetime(2026, 2, 19, 12, 0, 0, tzinfo=UTC).isoformat()
    entries = [
        {"type": "user", "timestamp": stamp, "message": {"role": "user", "content": "run it"}},
        {
            "type": "assistant",
            "timestamp": stamp,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "mcp__osprey__execute",
                        "input": {"code": "print(1)"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "timestamp": stamp,
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


class TestReplayMatchesLive:
    """A reloaded page shows the preview the live stream sent for the same result.

    Both paths cut through :func:`tool_result_preview`, so a long result reaches
    the stream already cut, and a reload shows the same cut.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Wrote 3 rows",
            "p" * 2000,
            envelope({"rows": [{"name": f"pv{i}", "note": "n" * 400} for i in range(30)]}),
        ],
        ids=["plain", "long-plain", "long-json"],
    )
    @pytest.mark.parametrize("as_blocks", [False, True], ids=["string", "text-blocks"])
    def test_a_reloaded_result_previews_as_the_live_one_did(self, tmp_path, text, as_blocks):
        content: object = [{"type": "text", "text": text}] if as_blocks else text
        live = _event_to_wire(
            ToolResultEvent(
                tool_use_id="t1", content=content, is_error=False, parent_tool_use_id=None
            )
        )
        transcript = tmp_path / "session.jsonl"
        _write_transcript(transcript, content)

        turns = _chat_turns(_transcript_nodes(transcript))
        (replayed,) = [
            event
            for turn in turns
            for event in turn.get("events", [])
            if event["type"] == "tool_result"
        ]
        assert live is not None
        assert (live["preview"], live["truncated"]) == tool_result_preview(text)
        assert replayed["preview"] == live["preview"]
        assert replayed["truncated"] == live["truncated"]

    def test_an_image_result_is_a_placeholder_on_both_paths(self, tmp_path):
        """Neither path ships the image's bytes; both show ``[image]`` beside any text."""
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR" * 300},
            },
            {"type": "text", "text": "the plot"},
        ]
        live = _event_to_wire(
            ToolResultEvent(
                tool_use_id="t1", content=content, is_error=False, parent_tool_use_id=None
            )
        )
        transcript = tmp_path / "session.jsonl"
        _write_transcript(transcript, content)

        (replayed,) = [
            event
            for turn in _chat_turns(_transcript_nodes(transcript))
            for event in turn.get("events", [])
            if event["type"] == "tool_result"
        ]
        assert live is not None
        assert live["preview"] == "[image]\nthe plot"
        assert replayed["preview"] == live["preview"]
        assert "iVBOR" not in json.dumps(live)
