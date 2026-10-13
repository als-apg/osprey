"""The approval hook's guarded-run names and journal parser against the runtime's.

The approval hook runs outside the osprey venv, so it restates in stdlib terms
where a control target's guarded-run lock and journal live, what the lock's
holder record holds, how the two files are opened, how a journal is parsed and
what the approved call's input carries. The runtime reads the same files and
the same input; a hook that spelled one of them differently would list a
journal the runtime never restores, or bind a digest the runtime never
compares. Every name is read off the runtime here, never re-spelled, and the
two parsers and the two openers are run on the same bytes and the same planted
names.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import osprey.runtime.guarded_run as guarded_run
import osprey.runtime.journal as journal
import osprey.templates.claude_code.claude.hooks.osprey_approval as hook
from osprey.runtime.journal import GuardedRunDirError, parse_pending_journal
from osprey_connectors.workspace import STATE_DIR_NAME
from tests.runtime._guarded_run_fixtures import (
    PLANTED_REFUSAL,
    PLANTED_REFUSED,
    link_directory,
    plant_kind,
    returns_within,
)

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


#: The two ways the lock and the journal are opened: the journal's read and the lock's try.
OPENS = {"read": (os.O_RDONLY, False), "lock": (os.O_RDWR, True)}


def _refusal(opener, error, name, flags, create, dir_fd, where) -> str:
    """The text *opener* refuses the planted *name* with."""
    with pytest.raises(error) as refused:
        returns_within(
            lambda: opener(name, flags, dir_fd=dir_fd, where=where, create=create),
            release=where,
        )
    return str(refused.value)


@pytest.mark.parametrize("opening", sorted(OPENS))
@pytest.mark.parametrize("kind", PLANTED_REFUSED)
def test_the_hook_opens_a_planted_name_as_the_runtime_does(kind, opening, tmp_path) -> None:
    flags, create = OPENS[opening]
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep\n")
    folder = tmp_path / "folder"
    folder.mkdir()
    name = guarded_run.JOURNAL_FILE_NAME
    plant_kind(folder / name, kind, victim)
    dir_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        runtime = _refusal(
            journal._open_regular, GuardedRunDirError, name, flags, create, dir_fd, folder / name
        )
        hooked = _refusal(
            hook._guarded_open_regular,
            hook._GuardedRunNotRegular,
            name,
            flags,
            create,
            dir_fd,
            str(folder / name),
        )
    finally:
        os.close(dir_fd)

    assert hooked == runtime == f"{folder / name} {PLANTED_REFUSAL[kind]}"
    assert victim.read_bytes() == b"keep\n"


@pytest.mark.parametrize("opening", sorted(OPENS))
@pytest.mark.parametrize("present", [True, False], ids=["present", "missing"])
def test_the_hook_opens_a_regular_file_as_the_runtime_does(opening, present, tmp_path) -> None:
    flags, create = OPENS[opening]
    name = guarded_run.LOCK_FILE_NAME
    opened = []
    for side, opener, where in (
        ("runtime", journal._open_regular, tmp_path / "runtime" / name),
        ("hook", hook._guarded_open_regular, str(tmp_path / "hook" / name)),
    ):
        folder = tmp_path / side
        folder.mkdir()
        if present:
            (folder / name).write_bytes(b"held\n")
            os.chmod(folder / name, 0o600)
        dir_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                fd, created = opener(name, flags, dir_fd=dir_fd, where=where, create=create)
            except FileNotFoundError:
                opened.append("missing")
                continue
            try:
                on_disk = (folder / name).stat()
                assert os.fstat(fd).st_ino == on_disk.st_ino
                opened.append((created, on_disk.st_mode, os.get_blocking(fd), os.pread(fd, 16, 0)))
            finally:
                os.close(fd)
        finally:
            os.close(dir_fd)

    assert opened[0] == opened[1]
    if present:
        assert opened[0] == (False, 0o100600, True, b"held\n")
    elif create:
        assert opened[0][0] is True and opened[0][1] & 0o777 == 0o664
    else:
        assert opened[0] == "missing"


def test_the_hook_refuses_a_linked_directory_as_the_runtime_does(tmp_path) -> None:
    target = tmp_path / STATE_DIR_NAME / guarded_run.GUARDED_RUN_DIR / "va"
    target.mkdir(parents=True)
    link_directory(target)

    with pytest.raises(GuardedRunDirError) as runtime:
        guarded_run.read_run_journal(target)
    with pytest.raises(hook._GuardedRunNotRegular) as hooked:
        hook._guarded_open_dir(str(tmp_path), "va")

    assert str(hooked.value) == str(runtime.value)
    assert str(runtime.value) == f"{target} is a link; a guarded-run directory is never a link"


def test_the_hook_refuses_a_file_at_a_directory_level_as_the_runtime_does(tmp_path) -> None:
    level = tmp_path / STATE_DIR_NAME / guarded_run.GUARDED_RUN_DIR
    level.parent.mkdir()
    level.write_bytes(b"")

    with pytest.raises(GuardedRunDirError) as runtime:
        guarded_run.read_run_journal(level / "va")
    with pytest.raises(hook._GuardedRunNotRegular) as hooked:
        hook._guarded_open_dir(str(tmp_path), "va")

    assert str(hooked.value) == str(runtime.value) == f"{level} is not a directory"
