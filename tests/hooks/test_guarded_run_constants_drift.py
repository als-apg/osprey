"""The approval hook's guarded-run names and journal parser against the runtime's.

The approval hook runs outside the osprey venv, so it restates in stdlib terms
where a control target's guarded-run lock and journal live, what the lock's
holder record holds, how a journal is parsed and what the approved call's input
carries. The runtime reads the same files and the same input; a hook that
spelled one of them differently would list a journal the runtime never restores,
or bind a digest the runtime never compares. Every name is read off the runtime
here, never re-spelled, and the two parsers are run on the same bytes.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import osprey.runtime.guarded_run as guarded_run
import osprey.templates.claude_code.claude.hooks.osprey_approval as hook
from osprey.runtime.journal import parse_pending_journal
from osprey_connectors.workspace import STATE_DIR_NAME

TOOLS_DIR = Path(guarded_run.__file__).parents[1] / "mcp_server" / "python_executor" / "tools"


def test_the_directory_names_are_the_runtimes() -> None:
    assert hook._STATE_DIR_NAME == STATE_DIR_NAME
    assert hook._GUARDED_RUN_DIR == guarded_run.GUARDED_RUN_DIR
    assert hook._GUARDED_RUN_LOCK_FILE == guarded_run.LOCK_FILE_NAME
    assert hook._GUARDED_RUN_JOURNAL_FILE == guarded_run.JOURNAL_FILE_NAME


def test_the_no_journal_digest_is_the_runtimes() -> None:
    assert hook._APPROVED_NO_JOURNAL == guarded_run.APPROVED_NO_JOURNAL


def test_the_holder_keys_are_what_the_runtime_writes(tmp_path) -> None:
    fd = os.open(tmp_path / guarded_run.LOCK_FILE_NAME, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        guarded_run._write_holder(fd)
        written = json.loads(os.pread(fd, 4096, 0))
    finally:
        os.close(fd)

    assert set(written) == {hook._HOLDER_PID_KEY, hook._HOLDER_STARTED_KEY}
    assert hook._holder_text(json.dumps(written).encode("utf-8")) == (
        str(written["pid"]),
        written["started"],
    )


@pytest.mark.parametrize("module", ["python_execute.py", "python_execute_file.py"])
def test_the_binding_keys_are_the_tools_arguments(module) -> None:
    tree = ast.parse((TOOLS_DIR / module).read_text(encoding="utf-8"))
    arguments = {
        arg.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        for arg in node.args.args
    }

    assert {hook._APPROVED_JOURNAL_KEY, hook._APPROVED_TARGET_KEY} <= arguments


def _journal(*lines: object, tail: bytes = b"") -> bytes:
    return b"".join(json.dumps(line).encode("utf-8") + b"\n" for line in lines) + tail


HEADER = {
    "header": {
        "target": "va",
        "generation": 3,
        "identity": "operator",
        "pid": 4242,
        "started": "2026-10-01T08:00:00Z",
    }
}

#: Journals and what both parsers make of them; ``ValueError`` is the
#: "unreadable" verdict.
JOURNALS = {
    "records-then-malformed-last-line": _journal(
        HEADER,
        {"address": "SR:QF:SP", "value": 1.5},
        {"address": "SR:QD:SP", "value": [1, 2]},
        {"address": "SR:QF:SP", "value": 9.0},
        tail=b"{not json\n",
    ),
    "torn-last-line": _journal(HEADER, {"address": "A", "value": 1}, tail=b'{"address": "B"'),
    "non-record-last-line": _journal(HEADER, {"address": "A", "value": 1}, {"other": 1}),
    "empty": b"",
    "header-only": _journal(HEADER),
    "no-header": _journal({"address": "A", "value": 1}),
    "untyped-header": _journal(
        {"header": {"target": 1, "generation": True, "pid": "x", "started": 5}},
        {"address": "A", "value": None},
    ),
    "malformed-middle-line": _journal(HEADER, tail=b"{not json\n" + _journal({"address": "A"})),
    "non-record-middle-line": _journal(HEADER, {"other": 1}, {"address": "A", "value": 1}),
}


def _runtime_verdict(raw: bytes, path: Path):
    try:
        pending = parse_pending_journal(raw, path)
    except ValueError as exc:
        return ("unreadable", str(exc))
    if pending is None:
        return None
    return {
        "target": pending.target,
        "generation": pending.generation,
        "identity": pending.identity,
        "pid": pending.pid,
        "started": pending.started,
        "values": list(pending.values.items()),
    }


def _hook_verdict(raw: bytes, path: Path):
    try:
        pending = hook._parse_pending_journal(raw, path)
    except ValueError as exc:
        return ("unreadable", str(exc))
    if pending is None:
        return None
    return {**pending, "values": list(pending["values"].items())}


@pytest.mark.parametrize("name", sorted(JOURNALS))
def test_the_hook_parses_a_journal_as_the_runtime_does(name, tmp_path) -> None:
    path = tmp_path / guarded_run.JOURNAL_FILE_NAME

    assert _hook_verdict(JOURNALS[name], path) == _runtime_verdict(JOURNALS[name], path)


def test_the_malformed_trailing_line_fixture_holds_records() -> None:
    verdict = _runtime_verdict(JOURNALS["records-then-malformed-last-line"], Path("j"))

    assert verdict["values"] == [("SR:QF:SP", 1.5), ("SR:QD:SP", [1, 2])]
