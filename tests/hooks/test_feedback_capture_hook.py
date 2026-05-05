"""Tests for the feedback capture hook (osprey_cf_feedback_capture.py).

Verifies that the hook correctly captures channel finder results into
pending_reviews.json, including handling the {"result": "..."} wrapper
format that Claude Code's PostToolUse delivers.
"""

import json

import pytest

# -- Helpers ------------------------------------------------------------------


def _cf_response(channels, total=None):
    """Build a channel finder response dict."""
    if total is None:
        total = len(channels)
    return {"channels": channels, "total": total}


def _hook_extra(tmp_path):
    """Standard hook_input_extra with cwd set for store path resolution."""
    return {
        "cwd": str(tmp_path),
        "session_id": "test-session",
        "transcript_path": "/tmp/transcript.jsonl",
    }


def _pending_items(tmp_path):
    """Read items from pending_reviews.json, return dict or empty."""
    store = tmp_path / "data" / "feedback" / "pending_reviews.json"
    if not store.exists():
        return {}
    data = json.loads(store.read_text())
    return data.get("items", {})


def _setup_subagent_transcript(tmp_path, main_name, subagent_content):
    """Create main transcript + subagent transcript in Claude Code layout.

    Layout::

        tmp_path/<main_name>.jsonl                       (main session)
        tmp_path/<main_name>/subagents/cf-agent.jsonl    (sub-agent)

    Returns the path to the main transcript (what hook receives as
    transcript_path).
    """
    main_transcript = tmp_path / f"{main_name}.jsonl"
    main_transcript.write_text(
        json.dumps({"type": "user", "message": {"content": "Original operator question"}}) + "\n"
    )

    subagent_dir = tmp_path / main_name / "subagents"
    subagent_dir.mkdir(parents=True, exist_ok=True)
    sub_transcript = subagent_dir / "channel-finder-abc123.jsonl"
    sub_transcript.write_text(subagent_content)

    return str(main_transcript)


# -- Tests --------------------------------------------------------------------


@pytest.mark.unit
class TestFeedbackCaptureResultUnwrapping:
    """The hook must unwrap the {"result": "..."} wrapper from PostToolUse."""

    def test_result_wrapper_with_json_string(self, tmp_path, hook_runner):
        """When tool_response is {"result": '{"channels":...,"total":3}'},
        the hook should unwrap and capture."""
        inner = _cf_response([{"name": f"BPM:{i}"} for i in range(3)])
        wrapped = {"result": json.dumps(inner)}

        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "BPM channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=wrapped,
            hook_input_extra=_hook_extra(tmp_path),
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1, f"Expected captured item, got: {items}"
        item = next(iter(items.values()))
        assert item["channel_count"] == 3

    def test_result_wrapper_with_parsed_dict(self, tmp_path, hook_runner):
        """When tool_response is {"result": {already-parsed dict}},
        the hook should unwrap and capture."""
        inner = _cf_response([{"name": f"QUAD:{i}"} for i in range(5)])
        wrapped = {"result": inner}

        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "QUAD channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=wrapped,
            hook_input_extra=_hook_extra(tmp_path),
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        assert item["channel_count"] == 5

    def test_direct_response_still_works(self, tmp_path, hook_runner):
        """Direct (non-wrapped) response should still be captured."""
        resp = _cf_response([{"name": "CORR:H1"}])

        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__query_channels",
            {"query": "correctors", "facility": "test"},
            cwd=tmp_path,
            tool_response=resp,
            hook_input_extra=_hook_extra(tmp_path),
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        assert item["channel_count"] == 1

    def test_json_string_response_still_works(self, tmp_path, hook_runner):
        """JSON string response (not wrapped) should still be captured."""
        resp = json.dumps(_cf_response([{"name": "RF:1"}, {"name": "RF:2"}]))

        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "RF channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=resp,
            hook_input_extra=_hook_extra(tmp_path),
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        assert item["channel_count"] == 2


@pytest.mark.unit
class TestFeedbackCaptureAgentTask:
    """The hook should extract the delegation prompt from the sub-agent transcript.

    transcript_path points to the MAIN session transcript.  The sub-agent
    transcript lives at ``<session_stem>/subagents/<agent>.jsonl``.
    """

    def test_reads_subagent_transcript_not_main(self, tmp_path, hook_runner):
        """The hook must read from the sub-agent dir, not the main transcript."""
        main_path = _setup_subagent_transcript(
            tmp_path,
            "session-001",
            json.dumps({"type": "user", "message": {"content": "Find BPM channels"}})
            + "\n"
            + json.dumps({"type": "assistant", "message": {"content": "Searching..."}})
            + "\n",
        )

        resp = _cf_response([{"name": "BPM:1"}])
        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "BPM channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=resp,
            hook_input_extra={
                "cwd": str(tmp_path),
                "session_id": "test-session",
                "transcript_path": main_path,
            },
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        # Must be the sub-agent delegation prompt, NOT "Original operator question"
        assert item["agent_task"] == "Find BPM channels"

    def test_empty_when_no_transcript(self, tmp_path, hook_runner):
        """When transcript doesn't exist, agent_task should be empty string."""
        resp = _cf_response([{"name": "BPM:1"}])
        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "BPM channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=resp,
            hook_input_extra={
                "cwd": str(tmp_path),
                "session_id": "test-session",
                "transcript_path": "/nonexistent/transcript.jsonl",
            },
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        assert item["agent_task"] == ""

    def test_empty_when_no_subagent_dir(self, tmp_path, hook_runner):
        """When subagents directory doesn't exist, agent_task should be empty."""
        main_transcript = tmp_path / "session-002.jsonl"
        main_transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "Operator question"}}) + "\n"
        )

        resp = _cf_response([{"name": "BPM:1"}])
        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "BPM channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=resp,
            hook_input_extra={
                "cwd": str(tmp_path),
                "session_id": "test-session",
                "transcript_path": str(main_transcript),
            },
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        assert item["agent_task"] == ""

    def test_skips_non_string_content(self, tmp_path, hook_runner):
        """When first user message has list content, skip to next user message."""
        main_path = _setup_subagent_transcript(
            tmp_path,
            "session-003",
            json.dumps({"type": "user", "message": {"content": [{"type": "image"}]}})
            + "\n"
            + json.dumps({"type": "user", "message": {"content": "Find corrector magnets"}})
            + "\n",
        )

        resp = _cf_response([{"name": "CH:1"}])
        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "channels", "facility": "test"},
            cwd=tmp_path,
            tool_response=resp,
            hook_input_extra={
                "cwd": str(tmp_path),
                "session_id": "test-session",
                "transcript_path": main_path,
            },
        )

        items = _pending_items(tmp_path)
        assert len(items) >= 1
        item = next(iter(items.values()))
        assert item["agent_task"] == "Find corrector magnets"


@pytest.mark.unit
class TestFeedbackCaptureNoTotal:
    """The hook should skip responses with no total or total <= 0."""

    def test_result_wrapper_no_channels(self, tmp_path, hook_runner):
        """Wrapped response with total=0 should NOT capture."""
        inner = _cf_response([], total=0)
        wrapped = {"result": json.dumps(inner)}

        hook_runner(
            "osprey_cf_feedback_capture.py",
            "mcp__channel-finder__build_channels",
            {"query": "nothing", "facility": "test"},
            cwd=tmp_path,
            tool_response=wrapped,
            hook_input_extra=_hook_extra(tmp_path),
        )

        items = _pending_items(tmp_path)
        assert len(items) == 0, f"Expected no items for total=0, got: {items}"
