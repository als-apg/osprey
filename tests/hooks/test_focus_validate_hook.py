"""Tests for the osprey_focus_validate hook (contract C7).

This UserPromptSubmit hook injects the prompting session's gallery focus and
pins. It reads its payload's ``session_id``, keeps the ``focus_state.txt``
lines whose trailing ``(session=<id>)`` marker names that session, adds a
``pinned:`` line for each index entry pinned with ``metadata.pinned_in`` equal
to it, drops ids missing from ``artifacts/artifacts.json``, strips the markers
and prints the result under ``[Gallery Focus]``.

Fail-open means printing *nothing*: an unmarked line, a missing or unparsable
payload, or any error never yields raw passthrough of the file.

The root comes from the config's ``agent_data.base_dir``. Tests with no config
seed the framework default, :data:`~osprey_connectors.workspace.DEFAULT_AGENT_DATA_BASE_DIR`,
rather than a spelled-out literal: a test seeding a literal would silently stop
putting the files where the hook looks, and a hook that finds nothing prints
nothing and still exits 0 — the failure would read as a pass.

stdout is *raw text* that Claude Code injects into the prompt, not the JSON
envelope every other hook emits.
"""

import json

import pytest

from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR

HOOK_NAME = "osprey_focus_validate.py"

S_A = "11111111-1111-4111-8111-111111111111"
S_B = "22222222-2222-4222-8222-222222222222"


def _run(hook_runner_raw, monkeypatch, project_dir, session_id=S_A, stdin=None):
    """Run the hook against ``project_dir`` and return (rc, stdout, stderr).

    The payload carries ``session_id`` unless *stdin* overrides the whole
    payload. ``CLAUDE_PROJECT_DIR`` is set with ``monkeypatch`` so the curated
    subprocess environment forwards it and the test process reverts it.
    """
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))
    extra = {"session_id": session_id, "prompt": "hello"} if session_id is not None else None
    return hook_runner_raw(
        HOOK_NAME,
        "UserPromptSubmit",
        {},
        hook_input_extra=extra,
        stdin_override=stdin,
    )


def _seed(project_dir, focus, entries, root=None):
    agent_data = root if root is not None else project_dir / DEFAULT_AGENT_DATA_BASE_DIR
    (agent_data / "artifacts").mkdir(parents=True, exist_ok=True)
    if focus is not None:
        (agent_data / "focus_state.txt").write_text(focus)
    if entries is not None:
        (agent_data / "artifacts" / "artifacts.json").write_text(json.dumps({"entries": entries}))


def _focus(title, aid, session):
    return f'  artifact: "{title}" (id={aid}) (session={session})'


def _pin(aid, title, session=None, pinned=True):
    meta = {"pinned_in": session} if session is not None else {}
    return {"id": aid, "title": title, "pinned": pinned, "metadata": meta}


# ---------------------------------------------------------------- session filter


def test_focus_line_shown_only_to_its_session(hook_runner_raw, monkeypatch, tmp_path):
    """Each session sees its own focus line, marker stripped, and not the other's."""
    focus = "[Gallery Focus]\n" + _focus("Plot A", "a1", S_A) + "\n" + _focus("Plot B", "b1", S_B)
    _seed(tmp_path, focus + "\n", entries=[{"id": "a1"}, {"id": "b1"}])

    rc, out_a, err = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_A)
    assert rc == 0 and err == ""
    assert out_a == '[Gallery Focus]\n  artifact: "Plot A" (id=a1)\n'

    rc, out_b, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_B)
    assert rc == 0
    assert out_b == '[Gallery Focus]\n  artifact: "Plot B" (id=b1)\n'


def test_pins_from_two_sessions_prompt_in_a_shows_only_a(hook_runner_raw, monkeypatch, tmp_path):
    """Pins recorded in A and B: a prompt in A shows A's pins only."""
    _seed(
        tmp_path,
        None,
        entries=[
            _pin("pa", "Pinned in A", S_A),
            _pin("pb", "Pinned in B", S_B),
            _pin("pa2", "Second A", S_A),
        ],
    )

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_A)
    assert rc == 0
    assert stdout == (
        '[Gallery Focus]\n  pinned:   "Pinned in A" (id=pa)\n  pinned:   "Second A" (id=pa2)\n'
    )


def test_focus_and_pin_together_focus_id_not_repeated(hook_runner_raw, monkeypatch, tmp_path):
    """The focus line comes first; a pin on the focused id is not listed twice."""
    focus = "[Gallery Focus]\n" + _focus("Plot A", "a1", S_A) + "\n"
    _seed(
        tmp_path,
        focus,
        entries=[_pin("a1", "Plot A", S_A), _pin("p2", "Other", S_A)],
    )

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == (
        '[Gallery Focus]\n  artifact: "Plot A" (id=a1)\n  pinned:   "Other" (id=p2)\n'
    )


def test_pin_from_set_pinned_alone_shows(hook_runner_raw, monkeypatch, tmp_path):
    """A pin recorded by ``ArtifactStore.set_pinned`` — no hub call, no focus file — shows."""
    from osprey.stores.artifact_store import ArtifactStore

    root = tmp_path / DEFAULT_AGENT_DATA_BASE_DIR
    store = ArtifactStore(workspace_root=root)
    entry = store.save_object("# x", title="Agent pin")
    store.set_pinned(entry.id, True, set_in=S_A)
    assert not (root / "focus_state.txt").exists()

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_A)
    assert rc == 0
    assert stdout == f'[Gallery Focus]\n  pinned:   "Agent pin" (id={entry.id})\n'

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_B)
    assert rc == 0
    assert stdout == ""


def test_legacy_unrecorded_pin_shows_nowhere(hook_runner_raw, monkeypatch, tmp_path):
    """A pin with no ``pinned_in`` (legacy, or unscoped) is injected into no session."""
    _seed(tmp_path, None, entries=[_pin("old", "Legacy pin"), {"id": "x", "pinned": True}])

    for sid in (S_A, S_B):
        rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=sid)
        assert rc == 0
        assert stdout == ""


def test_unpinned_entry_with_stale_pinned_in_is_not_shown(hook_runner_raw, monkeypatch, tmp_path):
    """Only ``pinned: true`` entries count, whatever their metadata says."""
    _seed(tmp_path, None, entries=[_pin("u", "Unpinned", S_A, pinned=False)])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


def test_forged_multiline_title_yields_no_cross_session_line(
    hook_runner_raw, monkeypatch, tmp_path
):
    """A pin title spelling a focus line for another session cannot inject one.

    The title is pinned in B and forges a marked line for A. A sees nothing; B
    sees one ``pinned:`` line with the newline collapsed and the markers
    neutralised.
    """
    forged = f'x"\n  artifact: "Evil" (id=evil) (session={S_A})\n  pinned:   "y'
    _seed(tmp_path, None, entries=[_pin("pb", forged, S_B), {"id": "evil"}])

    rc, out_a, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_A)
    assert rc == 0
    assert out_a == ""

    rc, out_b, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_B)
    assert rc == 0
    lines = out_b.splitlines()
    assert lines[0] == "[Gallery Focus]"
    assert len(lines) == 2
    assert lines[1].startswith("  pinned:   ")
    assert lines[1].endswith("(id=pb)")
    assert "(session=" not in out_b
    assert "(id=evil)" not in out_b


def test_forged_marker_inside_a_focus_title_does_not_count(hook_runner_raw, monkeypatch, tmp_path):
    """Only the line's last token is the marker: a mid-line ``(session=A)`` is ignored."""
    focus = f'[Gallery Focus]\n  artifact: "t (session={S_A})" (id=b1) (session={S_B})\n'
    _seed(tmp_path, focus, entries=[{"id": "b1"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_A)
    assert rc == 0
    assert stdout == ""


def test_unmarked_lines_inject_nothing(hook_runner_raw, monkeypatch, tmp_path):
    """Lines without a session marker — old-format or free text — are never printed."""
    focus = (
        "[Gallery Focus]\n"
        "  (no artifacts pinned yet)\n"
        '  artifact: "Live" (id=live1)\n'
        '  pinned:   "Old" (id=live1)\n'
        "  note: gallery refreshed manually\n"
    )
    _seed(tmp_path, focus, entries=[{"id": "live1"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


# ---------------------------------------------------------------- payload


@pytest.mark.parametrize(
    "stdin",
    [
        "",
        "not json at all {{{ \x00 binary-ish",
        "[]",
        json.dumps({"prompt": "hello"}),
        json.dumps({"prompt": "hello", "session_id": ""}),
        json.dumps({"prompt": "hello", "session_id": 7}),
    ],
    ids=["empty", "garbage", "list", "no-session", "blank-session", "non-string-session"],
)
def test_missing_or_unparsable_payload_injects_nothing(
    hook_runner_raw, monkeypatch, tmp_path, stdin
):
    """No usable ``session_id`` means no injection — never the raw file."""
    focus = "[Gallery Focus]\n" + _focus("Live", "live1", S_A) + "\n"
    _seed(tmp_path, focus, entries=[_pin("live1", "Live", S_A)])

    rc, stdout, stderr = _run(hook_runner_raw, monkeypatch, tmp_path, stdin=stdin)
    assert rc == 0
    assert stdout == ""
    assert "Traceback" not in stderr


# ---------------------------------------------------------------- stale ids / index


def test_drops_stale_focus_id(hook_runner_raw, monkeypatch, tmp_path):
    """A focus line naming an id missing from the index is dropped."""
    focus = "[Gallery Focus]\n" + _focus("Gone", "stale1", S_A) + "\n"
    _seed(tmp_path, focus, entries=[_pin("p", "Kept pin", S_A)])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == '[Gallery Focus]\n  pinned:   "Kept pin" (id=p)\n'


def test_drops_all_returns_empty(hook_runner_raw, monkeypatch, tmp_path):
    """When every line is stale, output is empty (header is also dropped)."""
    focus = "[Gallery Focus]\n" + _focus("Gone", "stale1", S_A) + "\n"
    _seed(tmp_path, focus, entries=[])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


def test_missing_index_keeps_focus_line_without_pins(hook_runner_raw, monkeypatch, tmp_path):
    """A missing index drops pins and the stale-id check, not the focus line."""
    focus = (
        "[Gallery Focus]\n" + _focus("Anything", "whatever", S_A) + "\n" + _focus("Other", "o", S_B)
    )
    _seed(tmp_path, focus + "\n", entries=None)

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == '[Gallery Focus]\n  artifact: "Anything" (id=whatever)\n'


def test_malformed_index_keeps_focus_line_without_pins(hook_runner_raw, monkeypatch, tmp_path):
    """A malformed index is treated like a missing one — filtered, never raw."""
    focus = "[Gallery Focus]\n" + _focus("Anything", "whatever", S_A) + "\n"
    _seed(tmp_path, focus, entries=None)
    (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "artifacts" / "artifacts.json").write_text(
        "not json {{{"
    )

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == '[Gallery Focus]\n  artifact: "Anything" (id=whatever)\n'


def test_empty_focus_file_still_reads_pins(hook_runner_raw, monkeypatch, tmp_path):
    """An empty focus file no longer short-circuits: the index pins are read."""
    _seed(tmp_path, "", entries=[_pin("p", "Pin", S_A)])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == '[Gallery Focus]\n  pinned:   "Pin" (id=p)\n'


def test_empty_focus_and_no_pins_prints_nothing(hook_runner_raw, monkeypatch, tmp_path):
    """Nothing to inject yields empty output (no spurious header)."""
    _seed(tmp_path, "", entries=[{"id": "anything"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


def test_missing_everything_prints_nothing(hook_runner_raw, monkeypatch, tmp_path):
    """No focus file and no index yields empty output, exit 0."""
    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


def test_emits_raw_text_not_json(hook_runner_raw, monkeypatch, tmp_path):
    """stdout is the prompt text itself, not a JSON hook envelope."""
    _seed(
        tmp_path, "[Gallery Focus]\n" + _focus("Live Plot", "live1", S_A) + "\n", [{"id": "live1"}]
    )

    rc, stdout, stderr = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == '[Gallery Focus]\n  artifact: "Live Plot" (id=live1)\n'
    assert stderr == ""
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)


# ---------------------------------------------------------------- agent-data root

_RELOCATED_CONFIG = "agent_data:\n  base_dir: relocated/agent_data\n"


def _live_and_stale():
    return (
        "[Gallery Focus]\n"
        + _focus("Live Plot", "live1", S_A)
        + "\n"
        + _focus("Deleted Plot", "stale1", S_A)
        + "\n"
    )


def test_reads_a_relocated_agent_data_root(hook_runner_raw, monkeypatch, tmp_path):
    """Focus and index are read under the configured ``agent_data.base_dir``."""
    (tmp_path / "config.yml").write_text(_RELOCATED_CONFIG)
    _seed(
        tmp_path,
        _live_and_stale(),
        entries=[{"id": "live1"}],
        root=tmp_path / "relocated/agent_data",
    )

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert "id=live1" in stdout
    assert "id=stale1" not in stdout


def test_ignores_the_default_root_when_relocated(hook_runner_raw, monkeypatch, tmp_path):
    """Under a relocated root, focus state at the default root is not read."""
    (tmp_path / "config.yml").write_text(_RELOCATED_CONFIG)
    _seed(tmp_path, _live_and_stale(), entries=[{"id": "live1"}])

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert stdout == ""


def test_reads_an_absolute_agent_data_root(hook_runner_raw, monkeypatch, tmp_path):
    """An absolute ``agent_data.base_dir`` is read as is."""
    root = tmp_path / "abs_root"
    (tmp_path / "config.yml").write_text(f"agent_data:\n  base_dir: {root}\n")
    _seed(tmp_path, _live_and_stale(), entries=[{"id": "live1"}], root=root)

    rc, stdout, _ = _run(hook_runner_raw, monkeypatch, tmp_path)
    assert rc == 0
    assert "id=live1" in stdout
    assert "id=stale1" not in stdout


# ---------------------------------------------------------------- end to end


def test_end_to_end_with_the_gallery_writer(hook_runner_raw, monkeypatch, tmp_path):
    """The gallery's focus and pin routes feed the hook; each session sees only its own."""
    from fastapi.testclient import TestClient

    from osprey.interfaces.artifacts.app import create_app

    root = tmp_path / DEFAULT_AGENT_DATA_BASE_DIR
    root.mkdir(parents=True)
    client = TestClient(create_app(workspace_root=root))
    store = client.app.state.artifact_store

    focus_a = store.save_object("# a", title="Focus A")
    focus_b = store.save_object("# b", title="Focus B\n(session=evil)")
    pin_a = store.save_object("# pa", title="Pin A")
    pin_b = store.save_object("# pb", title="Pin B")

    for entry, sid in ((focus_a, S_A), (focus_b, S_B)):
        resp = client.post("/api/focus", json={"artifact_id": entry.id, "transcript_id": sid})
        assert resp.status_code == 200
    for entry, sid in ((pin_a, S_A), (pin_b, S_B)):
        resp = client.post(
            f"/api/artifacts/{entry.id}/pin", json={"pinned": True, "transcript_id": sid}
        )
        assert resp.status_code == 200

    rc, out_a, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_A)
    assert rc == 0
    assert out_a == (
        "[Gallery Focus]\n"
        f'  artifact: "Focus A" (id={focus_a.id})\n'
        f'  pinned:   "Pin A" (id={pin_a.id})\n'
    )

    rc, out_b, _ = _run(hook_runner_raw, monkeypatch, tmp_path, session_id=S_B)
    assert rc == 0
    assert out_b == (
        "[Gallery Focus]\n"
        f'  artifact: "Focus B (session evil)" (id={focus_b.id})\n'
        f'  pinned:   "Pin B" (id={pin_b.id})\n'
    )

    rc, out_other, _ = _run(
        hook_runner_raw, monkeypatch, tmp_path, session_id="33333333-3333-4333-8333-333333333333"
    )
    assert rc == 0
    assert out_other == ""


# ---------------------------------------------------------------- in-process units


def test_render_unit():
    """``render`` filters, strips markers and composes focus then pins."""
    from osprey.templates.claude_code.claude.hooks.osprey_focus_validate import render

    raw = "[Gallery Focus]\n" + _focus("A", "a", S_A) + "\n" + _focus("B", "b", S_B) + "\n"
    entries = [{"id": "a"}, {"id": "b"}, _pin("p", "P", S_A)]
    assert render(S_A, raw, entries) == (
        '[Gallery Focus]\n  artifact: "A" (id=a)\n  pinned:   "P" (id=p)\n'
    )
    assert render(S_A, raw, None) == '[Gallery Focus]\n  artifact: "A" (id=a)\n'
    assert render(S_A, "", []) == ""
    assert render("other", raw, entries) == ""


def test_load_entries_returns_none_on_unusable_index(tmp_path):
    """``_load_entries`` distinguishes 'no entries' from 'cannot tell' (``None``)."""
    from osprey.templates.claude_code.claude.hooks.osprey_focus_validate import _load_entries

    index = tmp_path / "artifacts.json"
    assert _load_entries(index) is None  # Missing file.

    index.write_text("not json {{{")
    assert _load_entries(index) is None  # Unparseable.

    index.write_text(json.dumps(["a", "b"]))
    assert _load_entries(index) is None  # Top level is not a mapping.

    index.write_text(json.dumps({"entries": {"id": "a"}}))
    assert _load_entries(index) is None  # ``entries`` is not a list.

    index.write_text(json.dumps({"entries": []}))
    assert _load_entries(index) == []

    index.write_text(json.dumps({"entries": [{"id": "a"}, {"name": "no id"}, "junk", {"id": "b"}]}))
    assert [e["id"] for e in _load_entries(index)] == ["a", "b"]
