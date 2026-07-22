"""Prompt-assembly: routing input-file seam items into the SDK user message.

``run_dispatch`` pops the per-run input-file seam once, inlines ``image/*``
inputs as base64 image content blocks in the user message, and appends a
mechanism-only descriptor line for every input file. These tests pin:

  * the assembled ``content`` shape (plain string with no inputs; a
    ``[*image_blocks, {text}]`` list once an image is inlined; a
    prompt+descriptor string for non-inlining inputs),
  * the image content block shape (Anthropic native base64 block, raw b64 in
    ``source.data``, correct ``media_type``),
  * both descriptor line formats (``data_read("<entry_id>")`` for a stored file;
    the ``[shown_inline]`` marker for an inlined image),
  * the hygiene invariant: ``content_b64`` never enters the prompt text or any
    log record (the sdk_runner base64-redacting logging filter).

Hermetic: fake seam data, no real SDK — ``query`` is monkeypatched with a fake
async generator that drains the streamed prompt to capture the user message.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from osprey.mcp_server.dispatch_worker import sdk_runner

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"solid-red-square-body" * 8
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


@pytest.fixture(autouse=True)
def _stub_osprey_helpers(monkeypatch):
    """Stub the deferred OSPREY helper imports so run_dispatch can be driven."""
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.operator_session.build_clean_env",
        lambda **kw: {},
    )
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.sdk_context.build_system_prompt",
        lambda *a, **k: "system",
    )
    monkeypatch.setattr(
        "osprey.utils.config.get_facility_timezone",
        lambda *a, **k: "UTC",
    )


def _result_message(cost_usd: float = 0.1, num_turns: int = 1) -> ResultMessage:
    rm = MagicMock(spec=ResultMessage)
    rm.cost_usd = cost_usd
    rm.num_turns = num_turns
    return rm


async def _capture_user_message(monkeypatch, prompt: str, seam: list[dict], run_id: str = "run-1"):
    """Run run_dispatch with a seam preset; return the streamed user message dict."""
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(dispatch_api, "_run_input_seam", {run_id: seam})
    captured: dict = {}

    async def fake_query(prompt, options):  # noqa: A002 - matches SDK signature
        messages = []
        async for m in prompt:
            messages.append(m)
        captured["messages"] = messages
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield _result_message()

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    await sdk_runner.run_dispatch(prompt, ["Read"], event_queue=asyncio.Queue(), run_id=run_id)
    return captured["messages"][0]


# ---------------------------------------------------------------------------
# _assemble_user_content — pure routing
# ---------------------------------------------------------------------------


def test_prompt_stream_no_input_files_is_plain_string():
    """No inputs → content is the raw prompt string (unchanged legacy shape)."""
    assert sdk_runner._assemble_user_content("just do it", []) == "just do it"


def test_content_block_ingest_false_image_inlined_from_content_b64():
    """An ingest=False image inlines directly from the seam's content_b64."""
    content = sdk_runner._assemble_user_content(
        "look",
        [{"filename": "p.png", "mime": "image/png", "entry_id": None, "content_b64": PNG_B64}],
    )
    assert isinstance(content, list)
    block = content[0]
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
    }
    assert content[-1]["type"] == "text"


def test_content_block_ingest_true_image_inlined_from_store(monkeypatch):
    """An ingest=True image (entry_id, no content_b64) inlines from stored bytes."""
    monkeypatch.setattr(sdk_runner, "_load_entry_b64", lambda eid: PNG_B64)
    content = sdk_runner._assemble_user_content(
        "look",
        [{"filename": "p.png", "mime": "image/png", "entry_id": "art-1", "content_b64": None}],
    )
    assert isinstance(content, list)
    assert content[0]["source"]["data"] == PNG_B64
    # An inlined image is described as shown inline, not as a data_read target.
    assert "shown_inline" in content[-1]["text"]


def test_content_block_non_image_not_inlined():
    """A text-mime input is referenced by descriptor only — never an image block."""
    content = sdk_runner._assemble_user_content(
        "read it",
        [{"filename": "d.csv", "mime": "text/csv", "entry_id": "art-9", "content_b64": None}],
    )
    # No image block ⇒ content stays a string (prompt + descriptor).
    assert isinstance(content, str)
    assert 'data_read("art-9")' in content


def test_prompt_stream_image_missing_from_store_falls_back_to_descriptor(monkeypatch):
    """If stored bytes are gone, the image is not inlined but still described."""
    monkeypatch.setattr(sdk_runner, "_load_entry_b64", lambda eid: None)
    content = sdk_runner._assemble_user_content(
        "look",
        [{"filename": "p.png", "mime": "image/png", "entry_id": "art-1", "content_b64": None}],
    )
    # No inlinable bytes ⇒ no image block; falls back to a data_read descriptor.
    assert isinstance(content, str)
    assert 'data_read("art-1")' in content


# ---------------------------------------------------------------------------
# Descriptor block
# ---------------------------------------------------------------------------


def test_descriptor_line_for_stored_file_uses_data_read():
    line = sdk_runner._descriptor_line(
        {"filename": "d.csv", "mime": "text/csv", "entry_id": "art-9"}, inlined=False
    )
    assert line == '- d.csv (text/csv) — read with data_read("art-9")'


def test_descriptor_line_for_inlined_image_has_shown_inline_marker():
    line = sdk_runner._descriptor_line(
        {"filename": "p.png", "mime": "image/png", "entry_id": None}, inlined=True
    )
    assert "image shown inline" in line  # human-readable
    assert "[shown_inline]" in line  # machine-readable marker


def test_descriptor_block_one_line_per_input_file():
    content = sdk_runner._assemble_user_content(
        "go",
        [
            {"filename": "a.png", "mime": "image/png", "entry_id": None, "content_b64": PNG_B64},
            {"filename": "b.csv", "mime": "text/csv", "entry_id": "art-2", "content_b64": None},
            {
                "filename": "c.json",
                "mime": "application/json",
                "entry_id": "art-3",
                "content_b64": None,
            },
        ],
    )
    text = content[-1]["text"]
    descriptor_lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert len(descriptor_lines) == 3
    assert sdk_runner._INPUT_FILES_HEADER in text
    # Order preserved, each with the mechanism appropriate to its routing.
    assert "[shown_inline]" in descriptor_lines[0]
    assert 'data_read("art-2")' in descriptor_lines[1]
    assert 'data_read("art-3")' in descriptor_lines[2]


# ---------------------------------------------------------------------------
# Hygiene: content_b64 never leaks
# ---------------------------------------------------------------------------


def test_content_block_b64_never_in_prompt_text():
    """The inlined b64 rides source.data only — never the text block."""
    content = sdk_runner._assemble_user_content(
        "look",
        [{"filename": "p.png", "mime": "image/png", "entry_id": None, "content_b64": PNG_B64}],
    )
    assert content[0]["source"]["data"] == PNG_B64  # present in the image block
    assert PNG_B64 not in content[-1]["text"]  # absent from the prompt text


def test_content_block_b64_redacted_from_logs(caplog):
    """The logger's redacting filter scrubs long base64 runs from any record."""
    with caplog.at_level(logging.DEBUG, logger=sdk_runner.logger.name):
        sdk_runner.logger.info("would-be leak: %s", PNG_B64)
    assert PNG_B64 not in caplog.text
    assert "[redacted-b64]" in caplog.text


def test_content_block_b64_absent_from_logs_during_run(monkeypatch, caplog):
    """A full run that inlines an image logs nothing containing the b64 payload."""
    with caplog.at_level(logging.DEBUG, logger=sdk_runner.logger.name):
        msg = asyncio.run(
            _capture_user_message(
                monkeypatch,
                "look",
                [
                    {
                        "filename": "p.png",
                        "mime": "image/png",
                        "entry_id": None,
                        "content_b64": PNG_B64,
                    }
                ],
            )
        )
    # The image reached the message content ...
    assert msg["message"]["content"][0]["source"]["data"] == PNG_B64
    # ... but never a log record.
    assert PNG_B64 not in caplog.text


# ---------------------------------------------------------------------------
# End-to-end through run_dispatch (pop-once seam + streamed message)
# ---------------------------------------------------------------------------


def test_prompt_stream_inlines_image_and_appends_descriptor(monkeypatch):
    """run_dispatch streams one user message with the image block + descriptor."""
    msg = asyncio.run(
        _capture_user_message(
            monkeypatch,
            "hello",
            [
                {
                    "filename": "plot.png",
                    "mime": "image/png",
                    "entry_id": None,
                    "content_b64": PNG_B64,
                },
                {
                    "filename": "data.csv",
                    "mime": "text/csv",
                    "entry_id": "art-9",
                    "content_b64": None,
                },
            ],
        )
    )
    assert msg["type"] == "user"
    content = msg["message"]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    text = content[-1]["text"]
    assert text.startswith("hello")
    assert "[shown_inline]" in text  # the image
    assert 'data_read("art-9")' in text  # the csv
    assert PNG_B64 not in text


def test_prompt_stream_pops_seam_once(monkeypatch):
    """The seam is consumed pop-once: a second take returns nothing."""
    from osprey.mcp_server.dispatch_worker import dispatch_api

    asyncio.run(
        _capture_user_message(
            monkeypatch,
            "hello",
            [{"filename": "p.png", "mime": "image/png", "entry_id": None, "content_b64": PNG_B64}],
        )
    )
    assert dispatch_api.take_input_seam("run-1") == []


def test_prompt_stream_no_run_id_leaves_content_plain(monkeypatch):
    """A run without a run_id has no seam — content is the untouched prompt."""
    captured: dict = {}

    async def fake_query(prompt, options):  # noqa: A002
        async for m in prompt:
            captured.setdefault("messages", []).append(m)
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield _result_message()

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    asyncio.run(sdk_runner.run_dispatch("plain prompt", ["Read"], event_queue=asyncio.Queue()))
    assert captured["messages"][0]["message"]["content"] == "plain prompt"
