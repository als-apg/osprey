"""The approval hook's run-lock names and lock opener against the runtime's.

The approval hook runs outside the osprey venv, so it restates in stdlib terms
where a control target's run lock lives, what the lock's holder record holds,
how the lock file and its directory are opened and what the approved call's
input carries. The runtime takes the same lock; a hook that spelled one of them
differently would probe a lock no run holds, or bind a target the executor
never compares. Every name is read off the runtime here, never re-spelled, and
the two openers are run on the same planted names.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import osprey.runtime.guarded_run as guarded_run
import osprey.templates.claude_code.claude.hooks.osprey_approval as hook
from osprey.runtime.guarded_run import GuardedRunDirError
from osprey_connectors.workspace import STATE_DIR_NAME
from tests._guarded_run import (
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

    assert hook._APPROVED_TARGET_KEY in arguments


#: How the lock file is opened: for its non-blocking try, created when missing.
LOCK_OPEN = (os.O_RDWR, True)


def _refusal(opener, error, name, flags, create, dir_fd, where) -> str:
    """The text *opener* refuses the planted *name* with."""
    with pytest.raises(error) as refused:
        returns_within(
            lambda: opener(name, flags, dir_fd=dir_fd, where=where, create=create),
            release=where,
        )
    return str(refused.value)


@pytest.mark.parametrize("kind", PLANTED_REFUSED)
def test_the_hook_opens_a_planted_name_as_the_runtime_does(kind, tmp_path) -> None:
    flags, create = LOCK_OPEN
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep\n")
    folder = tmp_path / "folder"
    folder.mkdir()
    name = guarded_run.LOCK_FILE_NAME
    plant_kind(folder / name, kind, victim)
    dir_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        runtime = _refusal(
            guarded_run._open_regular,
            GuardedRunDirError,
            name,
            flags,
            create,
            dir_fd,
            folder / name,
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


@pytest.mark.parametrize("present", [True, False], ids=["present", "missing"])
def test_the_hook_opens_a_regular_file_as_the_runtime_does(present, tmp_path) -> None:
    flags, create = LOCK_OPEN
    name = guarded_run.LOCK_FILE_NAME
    opened = []
    for side, opener, where in (
        ("runtime", guarded_run._open_regular, tmp_path / "runtime" / name),
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
    else:
        assert opened[0][0] is True and opened[0][1] & 0o777 == 0o664


def test_the_hook_refuses_a_linked_directory_as_the_runtime_does(tmp_path) -> None:
    target = tmp_path / STATE_DIR_NAME / guarded_run.GUARDED_RUN_DIR / "va"
    target.mkdir(parents=True)
    link_directory(target)

    with pytest.raises(GuardedRunDirError) as runtime:
        guarded_run._open_levels(target, create=False)
    with pytest.raises(hook._GuardedRunNotRegular) as hooked:
        hook._guarded_open_dir(str(tmp_path), "va")

    assert str(hooked.value) == str(runtime.value)
    assert str(runtime.value) == f"{target} is a link; a guarded-run directory is never a link"


def test_the_hook_refuses_a_file_at_a_directory_level_as_the_runtime_does(tmp_path) -> None:
    level = tmp_path / STATE_DIR_NAME / guarded_run.GUARDED_RUN_DIR
    level.parent.mkdir()
    level.write_bytes(b"")

    with pytest.raises(GuardedRunDirError) as runtime:
        guarded_run._open_levels(level / "va", create=False)
    with pytest.raises(hook._GuardedRunNotRegular) as hooked:
        hook._guarded_open_dir(str(tmp_path), "va")

    assert str(hooked.value) == str(runtime.value) == f"{level} is not a directory"
