"""``ArtifactEntry.deliver``: a writer states at save time that an entry is for the person.

A dispatch run lists exactly the entries that say so, and a chat bridge posts
what the list names (``tests/dispatch/test_run_artifacts.py``). Pinned here: the
store records the statement and defaults to "not stated"; an entry from before
the field loads as not stated; and every place in ``src/osprey`` that saves to
the store is classified, so a new writer cannot start posting to chats (or
silently stop) without this table changing.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from osprey.stores.artifact_store import ArtifactStore

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"

SAVE_METHODS = frozenset(
    {
        "save_file",
        "save_object",
        "save_from_path",
        "save_data",
        "save_channel_reading",
        "save_or_touch_by_sha256",
    }
)

#: Every store save in ``src/osprey`` outside the store itself, in source order:
#: ``True`` where the writer states the entry is for the person.
WRITERS: dict[str, list[bool]] = {
    # the agent's deliberate output
    "mcp_server/workspace/tools/_viz_common.py": [True],  # the plot tools
    "mcp_server/workspace/tools/artifact_register.py": [True, True],  # a file, literal text
    "mcp_server/workspace/tools/artifact_export.py": [True],
    "services/ariel_search/attachments/artifacts.py": [True],  # a logbook picture
    # a document: the PDF, then its source
    "mcp_server/workspace/tools/create_document.py": [True, False],
    # a script run: refusal head, figure, save_artifact() file, notebook, result record
    "mcp_server/python_executor/tools/_response_builder.py": [False, True, True, False, False],
    "templates/claude_code/claude/hooks/osprey_approval.py": [False],  # approval preview
    # the filed answer: it reaches the person as the reply's text
    "mcp_server/workspace/tools/submit_response.py": [False],
    # a tool's own data record
    "mcp_server/control_system/tools/archiver_read.py": [False],
    "mcp_server/control_system/tools/channel_read.py": [False, False, False],
    "mcp_server/phoebus/tools/bridge_tools.py": [False],
    "mcp_server/workspace/tools/screen_capture.py": [False],
    "mcp_server/python_executor/tools/pyaml_measure.py": [False],
    # what a person gave, or a seeded session: never a run's output
    "mcp_server/dispatch_worker/dispatch_api.py": [False],
    "interfaces/web_terminal/routes/chat_attachments.py": [False],
    "interfaces/web_terminal/scenario_sessions.py": [False],
}


def _states_deliver(call: ast.Call) -> bool:
    """Whether a save call passes ``deliver=True`` (in ``save_kwargs`` for the sha256 save)."""
    for keyword in call.keywords:
        if keyword.arg == "deliver":
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        if keyword.arg == "save_kwargs" and isinstance(keyword.value, ast.Dict):
            for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "deliver":
                    return isinstance(value, ast.Constant) and value.value is True
    return False


def _store_saves() -> dict[str, list[bool]]:
    found: dict[str, list[bool]] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "stores" / "artifact_store.py":
            continue
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in SAVE_METHODS
        ]
        if calls:
            calls.sort(key=lambda call: (call.lineno, call.col_offset))
            found[path.relative_to(SRC).as_posix()] = [_states_deliver(call) for call in calls]
    return found


def test_every_writer_is_classified():
    assert _store_saves() == WRITERS


def _file_kwargs() -> dict:
    return {
        "file_content": b"x",
        "filename": "a.txt",
        "artifact_type": "text",
        "title": "a",
        "mime_type": "text/plain",
    }


class TestTheStoreRecordsTheStatement:
    @pytest.fixture
    def store(self, tmp_path) -> ArtifactStore:
        return ArtifactStore(workspace_root=tmp_path)

    def test_a_save_that_states_nothing_is_not_for_the_person(self, store, tmp_path):
        source = tmp_path / "f.txt"
        source.write_text("x")
        entries = [
            store.save_file(**_file_kwargs()),
            store.save_object("text", title="o"),
            store.save_from_path(source, title="p"),
            store.save_data(tool="t", data={}, title="d"),
            store.save_channel_reading(b"png", b"npy", title="c", metadata={"channel": "A:B"}),
        ]
        assert [entry.deliver for entry in entries] == [False] * 5

    def test_a_save_that_states_it_is_recorded_and_survives_a_reload(self, store, tmp_path):
        source = tmp_path / "f.txt"
        source.write_text("x")
        entries = [
            store.save_file(**_file_kwargs(), deliver=True),
            store.save_object("text", title="o", deliver=True),
            store.save_from_path(source, title="p", deliver=True),
            store.save_data(tool="t", data={}, title="d", deliver=True),
            store.save_channel_reading(
                b"png", b"npy", title="c", metadata={"channel": "A:B"}, deliver=True
            ),
        ]
        assert all(entry.deliver is True for entry in entries)

        reloaded = ArtifactStore(workspace_root=tmp_path)
        assert all(reloaded.get_entry(entry.id).deliver is True for entry in entries)

    def test_an_entry_from_before_the_field_is_not_for_the_person(self, store, tmp_path):
        entry = store.save_file(**_file_kwargs(), deliver=True)
        index = next(tmp_path.rglob("artifacts.json"))
        raw = json.loads(index.read_text())
        rows = raw["entries"] if isinstance(raw, dict) else raw
        for row in rows:
            del row["deliver"]
        index.write_text(json.dumps(raw))

        assert ArtifactStore(workspace_root=tmp_path).get_entry(entry.id).deliver is False
