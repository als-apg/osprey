"""The guarded-run contract page states what the run lock does.

Each table on ``reference/contracts/guarded-run.rst`` is held to the runtime:
the lock's place and modes to the constants and to a lock taken under a
temporary repo, the holder record's keys to the record a held lock writes, and
every refusal text to the message the runtime raises in that case.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    _FILE_MODE,
    ENV_CONTROL_IDENTITY,
    GUARDED_RUN_DIR,
    GUARDED_RUN_DIR_MODE,
    LOCK_FILE_NAME,
    ActionContext,
    OspreyRunBusy,
    lock,
)
from osprey.utils.workspace import STATE_DIR_NAME
from osprey_connectors.types import CONTROL_TARGETS
from tests._guarded_run import link_directory, lock_path, reset_run_state, write_config

PAGE = Path(__file__).parents[2] / "docs/source/reference/contracts/guarded-run.rst"

ALICE_LIVE = ActionContext(target="live", owner="alice", approved_target="live")


def _table(caption: str, text: str | None = None) -> list[list[str]]:
    """The body rows of the page's ``list-table`` captioned *caption*, each cell one line."""
    text = PAGE.read_text(encoding="utf-8") if text is None else text
    head = f".. list-table:: {caption}\n"
    assert text.count(head) == 1, f"the page has no single table {caption!r}"
    body = text.split(head, 1)[1]
    lines = []
    for line in body.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line)
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("   * - "):
            rows.append([line[len("   * - ") :]])
        elif line.startswith("     - "):
            rows[-1].append(line[len("     - ") :])
        elif line.startswith("       ") and rows:
            rows[-1][-1] += " " + line.strip()
    return [[cell.strip() for cell in row] for row in rows[1:]]


def _literals(cell: str) -> list[str]:
    return re.findall(r"``(.+?)``", cell)


def _template(text: str) -> re.Pattern[str]:
    """*text* as a pattern, each ``<…>`` placeholder matching any value."""
    parts = re.split(r"<[^<>]+>", text)
    return re.compile(".+?".join(re.escape(part) for part in parts))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline, entered and stamped ``live``, nothing held."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(ENV_CONTROL_IDENTITY, raising=False)
    reset_run_state(monkeypatch)
    return root


def _location() -> dict[str, str]:
    return {row[0]: row[1] for row in _table("Where the lock lives")}


def test_the_lock_path_and_targets_are_the_runtime_ones():
    where = _location()
    assert _literals(where["Lock file"]) == [
        f"{STATE_DIR_NAME}/{GUARDED_RUN_DIR}/<target>/{LOCK_FILE_NAME}"
    ]
    assert _literals(where["Targets"]) == list(CONTROL_TARGETS)
    assert _literals(where["Directory contents"]) == [LOCK_FILE_NAME]


@pytest.mark.usefixtures("repo")
def test_the_modes_are_the_runtime_ones():
    where = _location()
    assert _literals(where["Directory mode"]) == [f"0o{GUARDED_RUN_DIR_MODE:o}"]
    assert _literals(where["File mode"]) == [f"0o{_FILE_MODE:o}"]

    with lock("live") as directory:
        path = lock_path("live")
        assert directory == path.parent
        assert stat.S_IMODE(path.stat().st_mode) == _FILE_MODE
        for level in (path.parent, path.parent.parent):
            assert stat.S_IMODE(level.stat().st_mode) == GUARDED_RUN_DIR_MODE
        assert sorted(os.listdir(directory)) == [LOCK_FILE_NAME]


@pytest.mark.usefixtures("repo")
def test_the_record_keys_are_the_ones_a_held_lock_writes(monkeypatch: pytest.MonkeyPatch):
    rows = _table("Holder record")
    documented = [_literals(row[0])[0] for row in rows]
    nullable = {_literals(row[0])[0] for row in rows if row[2] != "Never"}

    with lock("live") as directory:
        nobody = json.loads((directory / LOCK_FILE_NAME).read_text(encoding="utf-8"))
    assert list(nobody) == documented
    assert {key for key, value in nobody.items() if value is None} == nullable
    assert (directory / LOCK_FILE_NAME).read_text(encoding="utf-8") == ""

    reset_run_state(monkeypatch)
    monkeypatch.setenv(ENV_CONTROL_IDENTITY, "osprey-ops")
    with lock("live", context=ALICE_LIVE) as directory:
        named = json.loads((directory / LOCK_FILE_NAME).read_text(encoding="utf-8"))
    assert named["owner"] == "alice" and named["identity"] == "osprey-ops"
    assert None not in named.values()


def test_an_unread_record_field_reads_the_documented_word():
    text = PAGE.read_text(encoding="utf-8")
    word = re.search(r"a key reads ``(\w+)`` when", text)
    assert word is not None
    assert str(OspreyRunBusy("d")).count(word.group(1)) == 5


def _second_run() -> None:
    with lock("live"), lock("live"):
        pass


def _another_target() -> None:
    with lock("live"), lock("va"):
        pass


def _read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    with lock("live"):
        pass


def _record_mismatch() -> None:
    with lock("live", context=ActionContext(target="va", owner=None, approved_target="live")):
        pass


def _join_mismatch() -> None:
    bob = ActionContext(target="live", owner="bob", approved_target="live")
    with lock("live", context=ALICE_LIVE), lock("live", join=True, context=bob):
        pass


def _planted(kind: str) -> Callable[[], None]:
    def plant() -> None:
        path = lock_path("live")
        path.parent.mkdir(parents=True)
        victim = path.with_name("victim")
        victim.write_text("", encoding="utf-8")
        if kind == "link":
            os.symlink(victim, path)
        elif kind == "hard link":
            os.link(victim, path)
        else:
            os.mkfifo(path)
        with lock("live"):
            pass

    return plant


def _linked_directory() -> None:
    path = lock_path("live")
    path.parent.mkdir(parents=True)
    link_directory(path.parent)
    with lock("live"):
        pass


#: Each case the Refusals table names, and how to meet it.
CASES: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "Second run on a held target": lambda _: _second_run(),
    "Another target while the process holds one": lambda _: _another_target(),
    "Read-only run": _read_only,
    "Record names another target": lambda _: _record_mismatch(),
    "Join with a different record": lambda _: _join_mismatch(),
    "Link at the lock file": lambda _: _planted("link")(),
    "Hard link at the lock file": lambda _: _planted("hard link")(),
    "FIFO or other non-regular file at the lock file": lambda _: _planted("fifo")(),
    "Link at ``guarded_run`` or the target's directory": lambda _: _linked_directory(),
}


def _refusals() -> dict[str, str]:
    return {row[0]: _literals(row[1])[0] for row in _table("Refusals")}


def test_the_refusals_table_names_every_case_once():
    assert list(_refusals()) == list(CASES)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.usefixtures("repo")
def test_each_refusal_text_is_the_runtime_message(case: str, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(Exception) as raised:
        CASES[case](monkeypatch)
    assert _template(_refusals()[case]).fullmatch(str(raised.value)), str(raised.value)


@pytest.mark.usefixtures("repo")
def test_a_changed_refusal_text_fails_the_match():
    """Negative control: one word off in the documented text no longer matches."""
    text = _refusals()["Another target while the process holds one"]
    with pytest.raises(RuntimeError) as raised:
        _another_target()
    assert _template(text).fullmatch(str(raised.value))
    assert not _template(text.replace("cannot take", "may not take")).fullmatch(str(raised.value))


def test_a_table_the_page_lacks_fails_the_parse():
    """Negative control: the table reader refuses a caption the page does not have."""
    with pytest.raises(AssertionError, match="no single table"):
        _table("Holder records")
