"""The feedback capture hook follows Claude Code's saved-output notice.

An MCP answer larger than Claude Code's tool-output limit never reaches a
PostToolUse hook inline. The hook receives a notice in its place, and the
answer itself sits in a file under the session's ``tool-results`` directory —
the directory beside the session transcript. These tests pin that the hook
captures such an answer exactly as it captures the same answer inline, that
it reads only that directory, and that a notice it cannot follow captures
nothing and says why in the hook log.
"""

import json

import pytest

from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

HOOK = "osprey_cf_feedback_capture.py"
BUILD_CHANNELS = "mcp__channel-finder__build_channels"
TOOL_INPUT = {"query": "all BPM positions", "facility": "als"}


def _build_channels_answer():
    """A whole-ring build_channels answer, large enough to pass the output cap."""
    channels = [
        f"SR:DIAG:BPM:{bpm:02d}:{field}:{suffix}"
        for bpm in range(1, 73)
        for field in ("POSITION", "GOLDEN", "OFFSET", "STATUS")
        for suffix in ("X", "Y", "VALID", "CONNECTED")
    ]
    half = len(channels) // 2
    return {
        "channels": channels,
        "total": len(channels),
        "valid": channels[:half],
        "invalid": channels[half:],
        "valid_count": half,
        "invalid_count": len(channels) - half,
    }


def _inline_response(answer):
    """The PostToolUse ``tool_response`` Claude Code sends for an inline MCP answer."""
    return json.dumps({"result": json.dumps(answer)}, separators=(",", ":"))


def _notice(saved_path, characters):
    """The ``tool_response`` Claude Code sends in place of an oversized answer."""
    return (
        f"Error: result ({characters:,} characters) exceeds maximum allowed tokens. "
        f"Output has been saved to {saved_path}.\n"
        "Format: JSON with schema: {result: string}\n"
        "Use jq to make structured queries (find a value, filter by field).\n"
        "REQUIREMENTS FOR SUMMARIZATION/ANALYSIS/REVIEW:\n"
        f"- You MUST read the content from the file at {saved_path} in sequential "
        "chunks until 100% of the content has been read.\n"
    )


def _session(root):
    """A session transcript and its tool-results directory, laid out as Claude Code does."""
    session_id = "41801b6d-5c43-4620-b410-81401aa7df58"
    sessions = root / "projects" / "-repo"
    sessions.mkdir(parents=True)
    transcript = sessions / f"{session_id}.jsonl"
    transcript.write_text("")
    tool_results = sessions / session_id / "tool-results"
    tool_results.mkdir(parents=True)
    return transcript, tool_results


def _pending_items(repo):
    store = repo / DEFAULT_AGENT_DATA_BASE_DIR / "feedback" / "pending_reviews.json"
    if not store.exists():
        return {}
    return json.loads(store.read_text()).get("items", {})


def _hook_log(repo):
    log = repo / ".claude" / "hooks" / "hook_debug.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _run(hook_runner, repo, tool_response, transcript):
    repo.mkdir(parents=True, exist_ok=True)
    (repo / ".claude" / "hooks").mkdir(parents=True, exist_ok=True)
    hook_runner(
        HOOK,
        BUILD_CHANNELS,
        TOOL_INPUT,
        cwd=repo,
        tool_response=tool_response,
        hook_input_extra={
            "cwd": str(repo),
            "session_id": "41801b6d-5c43-4620-b410-81401aa7df58",
            "transcript_path": str(transcript),
        },
    )


def _comparable(item):
    return {k: v for k, v in item.items() if k not in ("id", "captured_at")}


@pytest.fixture(autouse=True)
def _hook_debug(monkeypatch):
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")


class TestSavedOutputIsCaptured:
    def test_saved_answer_is_captured_as_the_inline_answer_would_be(self, tmp_path, hook_runner):
        answer = _build_channels_answer()
        inline = _inline_response(answer)
        transcript, tool_results = _session(tmp_path / "config")
        saved = tool_results / "mcp-channel-finder-build_channels-1790977128862.txt"
        saved.write_text(inline)

        _run(hook_runner, tmp_path / "inline", inline, transcript)
        _run(hook_runner, tmp_path / "oversize", _notice(saved, len(inline)), transcript)

        inline_items = list(_pending_items(tmp_path / "inline").values())
        oversize_items = list(_pending_items(tmp_path / "oversize").values())
        assert len(inline_items) == 1
        assert len(oversize_items) == 1
        assert _comparable(oversize_items[0]) == _comparable(inline_items[0])
        assert oversize_items[0]["channel_count"] == answer["total"]
        assert json.loads(oversize_items[0]["tool_response"]) == {"result": json.dumps(answer)}

    def test_plain_text_saved_answer_is_captured(self, tmp_path, hook_runner):
        answer = _build_channels_answer()
        transcript, tool_results = _session(tmp_path / "config")
        saved = tool_results / "mcp-channel-finder-build_channels-1.txt"
        saved.write_text(json.dumps(answer))

        _run(hook_runner, tmp_path / "repo", _notice(saved, 90_000), transcript)

        items = list(_pending_items(tmp_path / "repo").values())
        assert len(items) == 1
        assert items[0]["channel_count"] == answer["total"]


class TestUnfollowableNoticeCapturesNothing:
    def test_missing_saved_file_is_logged_and_captures_nothing(self, tmp_path, hook_runner):
        transcript, tool_results = _session(tmp_path / "config")
        saved = tool_results / "mcp-channel-finder-build_channels-404.txt"

        _run(hook_runner, tmp_path / "repo", _notice(saved, 90_000), transcript)

        assert _pending_items(tmp_path / "repo") == {}
        statuses = [r["status"] for r in _hook_log(tmp_path / "repo")]
        assert statuses == ["saved-output-unreadable"]

    def test_unreadable_saved_file_is_logged_and_captures_nothing(self, tmp_path, hook_runner):
        transcript, tool_results = _session(tmp_path / "config")
        saved = tool_results / "mcp-channel-finder-build_channels-2.txt"
        saved.mkdir()  # a path the hook cannot read as a file

        _run(hook_runner, tmp_path / "repo", _notice(saved, 90_000), transcript)

        assert _pending_items(tmp_path / "repo") == {}
        statuses = [r["status"] for r in _hook_log(tmp_path / "repo")]
        assert statuses == ["saved-output-unreadable"]

    @pytest.mark.parametrize(
        "where",
        ["outside-session", "dot-dot-escape", "relative", "symlink-out", "nested"],
    )
    def test_path_outside_the_sessions_tool_results_is_never_read(
        self, tmp_path, hook_runner, where
    ):
        """A notice is followed only into the session's own tool-results directory.

        The answer text is not the harness, so a notice-shaped string naming any
        other file — elsewhere on disk, escaping with ``..``, relative, behind a
        symlink, or in a subdirectory — reads nothing.
        """
        transcript, tool_results = _session(tmp_path / "config")
        planted = tmp_path / "planted.txt"
        planted.write_text(_inline_response(_build_channels_answer()))
        paths = {
            "outside-session": planted,
            "dot-dot-escape": tool_results / ".." / ".." / ".." / ".." / "planted.txt",
            "relative": "planted.txt",
            "symlink-out": tool_results / "link.txt",
            "nested": tool_results / "sub" / "planted.txt",
        }
        (tool_results / "link.txt").symlink_to(planted)
        (tool_results / "sub").mkdir()
        (tool_results / "sub" / "planted.txt").write_text(planted.read_text())

        _run(hook_runner, tmp_path / "repo", _notice(paths[where], 90_000), transcript)

        assert _pending_items(tmp_path / "repo") == {}
        statuses = [r["status"] for r in _hook_log(tmp_path / "repo")]
        assert statuses == ["saved-output-untrusted"]

    def test_notice_without_a_transcript_reads_nothing(self, tmp_path, hook_runner):
        transcript, tool_results = _session(tmp_path / "config")
        saved = tool_results / "mcp-channel-finder-build_channels-3.txt"
        saved.write_text(_inline_response(_build_channels_answer()))

        _run(hook_runner, tmp_path / "repo", _notice(saved, 90_000), "")

        assert _pending_items(tmp_path / "repo") == {}
        statuses = [r["status"] for r in _hook_log(tmp_path / "repo")]
        assert statuses == ["saved-output-untrusted"]


class TestInlineAnswersAreUnchanged:
    def test_inline_answer_is_captured_without_any_file(self, tmp_path, hook_runner):
        answer = _build_channels_answer()
        transcript, _ = _session(tmp_path / "config")

        _run(hook_runner, tmp_path / "repo", _inline_response(answer), transcript)

        items = list(_pending_items(tmp_path / "repo").values())
        assert len(items) == 1
        assert items[0]["tool_response"] == _inline_response(answer)
        assert items[0]["channel_count"] == answer["total"]
        assert [r["status"] for r in _hook_log(tmp_path / "repo")] == ["captured"]

    def test_non_json_text_that_is_not_a_notice_is_still_a_parse_error(self, tmp_path, hook_runner):
        transcript, _ = _session(tmp_path / "config")

        _run(hook_runner, tmp_path / "repo", "Error: channel database unavailable", transcript)

        assert _pending_items(tmp_path / "repo") == {}
        assert [r["status"] for r in _hook_log(tmp_path / "repo")] == ["parse-error"]
