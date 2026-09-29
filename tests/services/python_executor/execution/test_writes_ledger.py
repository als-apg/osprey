"""Tests for the write ledger a readwrite executor run keeps.

A readwrite run registers an ``osprey.runtime`` write observer in the child
that appends ``{"channel": <address>}`` to ``writes.jsonl`` in the execution
folder on every ``attempt``; the parent reads that file into
``ExecutionResult.written_channels`` after every run, a timeout included.
"""

import json

import pytest

from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_TIMEOUT,
    _execute_via_local,
    _read_written_channels,
)
from osprey.services.python_executor.execution.wrapper import (
    WRITES_LEDGER_FILENAME,
    ExecutionWrapper,
)

_REGISTRATION = "_register_write_observer(_write_ledger_observer)"

#: User code that drives the observer the way ``osprey.runtime`` does: one
#: ``attempt`` per channel, plus a post-call phase the ledger must ignore.
_NOTIFY_TWO = """
import osprey.runtime as _rt
_rt._notify_write("SR:MAG:1", "attempt")
_rt._notify_write("SR:MAG:1", "landed")
_rt._notify_write("SR:MAG:2", "attempt")
_rt._notify_write("SR:MAG:1", "attempt")
print("notified")
"""


def _folder(tmp_path):
    folder = tmp_path / "exec"
    folder.mkdir()
    (folder / "figures").mkdir()
    return folder


def test_writes_ledger_registration_emitted_for_readwrite(tmp_path):
    """A readwrite run's script registers the ledger observer on the execution folder."""
    script = ExecutionWrapper(execution_mode="readwrite").create_wrapper("pass", tmp_path)

    assert _REGISTRATION in script
    assert str(tmp_path / WRITES_LEDGER_FILENAME) in script
    section = ExecutionWrapper(execution_mode="readwrite")._get_write_ledger_observer(tmp_path)
    assert section in script
    assert "import osprey.runtime" in section
    assert section.rstrip().endswith("except ImportError:\n    pass")


def test_writes_ledger_registration_absent_for_readonly(tmp_path):
    """A readonly run cannot write through the runtime, so it records nothing."""
    script = ExecutionWrapper(execution_mode="readonly").create_wrapper("pass", tmp_path)

    assert _REGISTRATION not in script
    assert WRITES_LEDGER_FILENAME not in script


def test_writes_ledger_registration_absent_without_execution_folder():
    """Without an execution folder the parent has nowhere to read a ledger from."""
    script = ExecutionWrapper(execution_mode="readwrite").create_wrapper("pass", None)

    assert _REGISTRATION not in script


def test_writes_ledger_registration_precedes_user_code(tmp_path):
    """The observer is registered before the user code can write."""
    script = ExecutionWrapper(execution_mode="readwrite").create_wrapper(
        "print('USER_CODE_MARKER')", tmp_path
    )

    assert script.index(_REGISTRATION) < script.index("USER_CODE_MARKER")


def test_read_written_channels_missing_ledger(tmp_path):
    """No ledger means no attempted write."""
    assert _read_written_channels(tmp_path) == []


def test_read_written_channels_truncated_ledger(tmp_path):
    """A last line cut off mid-append is skipped; the complete lines survive."""
    (tmp_path / WRITES_LEDGER_FILENAME).write_text(
        json.dumps({"channel": "SR:MAG:1"}) + "\n" + '{"channel": "SR:MA',
        encoding="utf-8",
    )

    assert _read_written_channels(tmp_path) == ["SR:MAG:1"]


def test_read_written_channels_keeps_first_attempt_order_without_repeats(tmp_path):
    """Repeated attempts on one channel list it once, at its first attempt."""
    lines = [
        {"channel": "SR:MAG:2"},
        {"channel": "SR:MAG:1"},
        {"channel": "SR:MAG:2"},
        {"not_a_channel": "x"},
        ["SR:MAG:3"],
    ]
    (tmp_path / WRITES_LEDGER_FILENAME).write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )

    assert _read_written_channels(tmp_path) == ["SR:MAG:2", "SR:MAG:1"]


async def test_writes_ledger_fills_written_channels(tmp_path):
    """A readwrite run's attempts reach ``written_channels``; other phases do not."""
    folder = _folder(tmp_path)

    result = await _execute_via_local(
        code=_NOTIFY_TWO,
        execution_mode="readwrite",
        config={"timeout": 60, "python_env_path": None},
        execution_folder=folder,
    )

    assert result.success, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.written_channels == ["SR:MAG:1", "SR:MAG:2"]
    entries = [
        json.loads(line)
        for line in (folder / WRITES_LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert entries == [
        {"channel": "SR:MAG:1"},
        {"channel": "SR:MAG:2"},
        {"channel": "SR:MAG:1"},
    ]


async def test_writes_ledger_readonly_run_records_nothing(tmp_path):
    """A readonly run leaves no ledger and an empty ``written_channels``."""
    folder = _folder(tmp_path)

    result = await _execute_via_local(
        code="print('read only')",
        execution_mode="readonly",
        config={"timeout": 60, "python_env_path": None},
        execution_folder=folder,
    )

    assert result.success, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.written_channels == []
    assert not (folder / WRITES_LEDGER_FILENAME).exists()


@pytest.mark.timeout(120)
async def test_writes_ledger_survives_a_timeout(tmp_path):
    """A run killed at its timeout still reports the attempts it made."""
    folder = _folder(tmp_path)
    code = _NOTIFY_TWO + "\nimport time\ntime.sleep(60)\n"

    result = await _execute_via_local(
        code=code,
        execution_mode="readwrite",
        config={"timeout": 8, "python_env_path": None},
        execution_folder=folder,
    )

    assert result.failure_kind == FAILURE_KIND_TIMEOUT
    assert result.written_channels == ["SR:MAG:1", "SR:MAG:2"]
