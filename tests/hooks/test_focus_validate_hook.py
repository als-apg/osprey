"""Tests for the osprey_focus_validate hook.

This UserPromptSubmit hook reads ``_agent_data/focus_state.txt``, drops
lines that reference deleted artifact IDs (by cross-checking
``_agent_data/artifacts/artifacts.json``), and prints the cleaned content
to stdout. It is defensive — never blocks a prompt — so all error paths
fail open with exit code 0.

The hook has a contract no other hook in the payload shares: stdin is
ignored entirely, and stdout is *raw text* that Claude Code injects into
the prompt rather than the JSON envelope every other hook emits. The
behavioural tests below pin both halves of that.
"""

import json

import pytest

HOOK_NAME = "osprey_focus_validate.py"


def _run(hook_runner_raw, monkeypatch, project_dir, stdin=None):
    """Run the hook against ``project_dir`` and return (rc, stdout, stderr).

    ``CLAUDE_PROJECT_DIR`` is set with ``monkeypatch`` so the curated
    subprocess environment forwards it and the test process reverts it.
    """
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))
    return hook_runner_raw(
        HOOK_NAME,
        "UserPromptSubmit",
        {},
        stdin_override=stdin,
    )


def _seed(project_dir, focus, entries):
    agent_data = project_dir / "_agent_data"
    (agent_data / "artifacts").mkdir(parents=True, exist_ok=True)
    (agent_data / "focus_state.txt").write_text(focus)
    if entries is not None:
        (agent_data / "artifacts" / "artifacts.json").write_text(json.dumps({"entries": entries}))


@pytest.mark.unit
def test_focus_validator_drops_stale_ids(hook_runner_raw, monkeypatch, tmp_path):
    """Lines referencing missing artifact IDs are dropped; valid lines kept."""
    focus = (
        "[Gallery Focus]\n"
        '  artifact: "Live Plot" (id=valid123)\n'
        '  pinned:   "Deleted Plot" (id=stale456)\n'
        '  pinned:   "Other Live" (id=valid789)\n'
    )
    _seed(tmp_path, focus, entries=[{"id": "valid123"}, {"id": "valid789"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert "id=valid123" in stdout
    assert "id=valid789" in stdout
    assert "id=stale456" not in stdout
    assert stdout.startswith("[Gallery Focus]")


@pytest.mark.unit
def test_focus_validator_drops_all_returns_empty(hook_runner_raw, monkeypatch, tmp_path):
    """When every line is stale, output is empty (header is also dropped)."""
    focus = '[Gallery Focus]\n  artifact: "Gone" (id=stale1)\n  pinned:   "Also Gone" (id=stale2)\n'
    _seed(tmp_path, focus, entries=[])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


@pytest.mark.unit
def test_focus_validator_fails_open_on_missing_index(hook_runner_raw, monkeypatch, tmp_path):
    """If artifacts.json is missing, hook prints original contents and exits 0."""
    focus = '[Gallery Focus]\n  artifact: "Anything" (id=whatever)\n'
    _seed(tmp_path, focus, entries=None)  # No artifacts.json

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == focus


@pytest.mark.unit
def test_focus_validator_fails_open_on_malformed_index(hook_runner_raw, monkeypatch, tmp_path):
    """Malformed artifacts.json is treated as 'unknown' → fail open."""
    focus = '[Gallery Focus]\n  artifact: "Anything" (id=whatever)\n'
    _seed(tmp_path, focus, entries=None)
    (tmp_path / "_agent_data" / "artifacts" / "artifacts.json").write_text("not json {{{")

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == focus


@pytest.mark.unit
def test_focus_validator_empty_focus_file(hook_runner_raw, monkeypatch, tmp_path):
    """Empty focus_state.txt yields empty output (no spurious header)."""
    _seed(tmp_path, "", entries=[{"id": "anything"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


@pytest.mark.unit
def test_focus_validator_missing_focus_file(hook_runner_raw, monkeypatch, tmp_path):
    """No focus_state.txt at all yields empty output, exit 0."""
    (tmp_path / "_agent_data" / "artifacts").mkdir(parents=True)
    (tmp_path / "_agent_data" / "artifacts" / "artifacts.json").write_text(
        json.dumps({"entries": []})
    )

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


@pytest.mark.unit
def test_focus_validator_keeps_lines_without_ids(hook_runner_raw, monkeypatch, tmp_path):
    """Lines carrying no ``(id=...)`` pass through untouched."""
    focus = (
        "[Gallery Focus]\n"
        "  (no artifacts pinned yet)\n"
        '  artifact: "Live" (id=live1)\n'
        '  pinned:   "Gone" (id=dead1)\n'
        "  note: gallery refreshed manually\n"
    )
    _seed(tmp_path, focus, entries=[{"id": "live1"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == (
        "[Gallery Focus]\n"
        "  (no artifacts pinned yet)\n"
        '  artifact: "Live" (id=live1)\n'
        "  note: gallery refreshed manually\n"
    )


@pytest.mark.unit
def test_focus_validator_ignores_stdin(hook_runner_raw, monkeypatch, tmp_path):
    """Output depends only on the files — stdin is never read.

    The hook is wired to UserPromptSubmit, where Claude Code hands it a JSON
    payload on stdin. It reads none of it: a well-formed payload naming a
    different project, valid JSON of the wrong type, unparseable bytes and a
    closed-empty stdin all produce byte-identical output.
    """
    focus = '[Gallery Focus]\n  artifact: "Live" (id=live1)\n  pinned: "Gone" (id=dead1)\n'
    _seed(tmp_path, focus, entries=[{"id": "live1"}])

    decoy = tmp_path / "decoy"
    _seed(decoy, '[Gallery Focus]\n  artifact: "Decoy" (id=live1)\n', entries=[{"id": "live1"}])

    expected = '[Gallery Focus]\n  artifact: "Live" (id=live1)\n'
    stdins = [
        "",
        json.dumps({"prompt": "hello", "cwd": str(decoy), "session_id": "abc"}),
        "not json at all {{{ \x00 binary-ish",
        "[]",
    ]
    for stdin in stdins:
        rc, stdout, stderr = _run(hook_runner_raw, monkeypatch, tmp_path, stdin=stdin)
        assert rc == 0, f"non-zero exit for stdin {stdin!r}"
        assert stdout == expected, f"stdout changed for stdin {stdin!r}"
        assert "Traceback" not in stderr, f"hook crashed for stdin {stdin!r}"


@pytest.mark.unit
def test_focus_validator_emits_raw_text_not_json(hook_runner_raw, monkeypatch, tmp_path):
    """stdout is the prompt text itself, not a JSON hook envelope."""
    focus = '[Gallery Focus]\n  artifact: "Live Plot" (id=live1)\n'
    _seed(tmp_path, focus, entries=[{"id": "live1"}])

    rc, stdout, stderr = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == focus
    assert stderr == ""
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)


@pytest.mark.unit
def test_clean_filters_lines_and_preserves_shape():
    """``_clean`` keeps valid and id-less lines, and preserves file shape."""
    from osprey.templates.claude_code.claude.hooks.osprey_focus_validate import _clean

    assert _clean("", {"a"}) == ""

    headed = '[Gallery Focus]\n  artifact: "A" (id=a)\n  pinned: "B" (id=b)\n'
    assert _clean(headed, {"a"}) == '[Gallery Focus]\n  artifact: "A" (id=a)\n'
    assert _clean(headed, set()) == ""

    # No header: body is returned as-is, and the trailing newline tracks the input.
    assert _clean("  x (id=a)\n  y (id=b)\n", {"a"}) == "  x (id=a)\n"
    assert _clean("  x (id=a)", {"a"}) == "  x (id=a)"

    # A line with no ``(id=...)`` is never dropped, even with no valid ids.
    assert _clean("plain line\n", set()) == "plain line\n"


@pytest.mark.unit
def test_load_valid_ids_returns_none_on_unusable_index(tmp_path):
    """``_load_valid_ids`` distinguishes 'no ids' from 'cannot tell' (``None``)."""
    from osprey.templates.claude_code.claude.hooks.osprey_focus_validate import _load_valid_ids

    index = tmp_path / "artifacts.json"

    assert _load_valid_ids(index) is None  # Missing file.

    index.write_text("not json {{{")
    assert _load_valid_ids(index) is None  # Unparseable.

    index.write_text(json.dumps(["a", "b"]))
    assert _load_valid_ids(index) is None  # Top level is not a mapping.

    index.write_text(json.dumps({"entries": {"id": "a"}}))
    assert _load_valid_ids(index) is None  # ``entries`` is not a list.

    index.write_text(json.dumps({"entries": []}))
    assert _load_valid_ids(index) == set()  # Empty index means "everything is stale".

    index.write_text(json.dumps({"entries": [{"id": "a"}, {"name": "no id"}, "junk", {"id": "b"}]}))
    assert _load_valid_ids(index) == {"a", "b"}  # Malformed entries are skipped, not fatal.
