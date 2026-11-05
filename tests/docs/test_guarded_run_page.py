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

import osprey.runtime.guarded_run as guarded_run
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
from osprey_connectors.workspace import reset_config_cache
from tests._guarded_run import (
    PLANTED_REFUSED,
    link_directory,
    lock_path,
    plant_kind,
    reset_run_state,
    returns_within,
    write_config,
)

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
    return re.compile("(.+?)".join(re.escape(part) for part in parts))


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
    assert _literals(where["Directory mode"])[0] == f"0o{GUARDED_RUN_DIR_MODE:o}"
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


Trigger = Callable[[pytest.MonkeyPatch], None]


def _enter(target: str = "live") -> None:
    with lock(target):
        pass


def _second_run(_: pytest.MonkeyPatch) -> None:
    with lock("live"), lock("live"):
        pass


def _another_target(_: pytest.MonkeyPatch) -> None:
    with lock("live"), lock("va"):
        pass


def _read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    _enter()


def _record_mismatch(_: pytest.MonkeyPatch) -> None:
    with lock("live", context=ActionContext(target="va", owner=None, approved_target="live")):
        pass


def _join_mismatch(_: pytest.MonkeyPatch) -> None:
    bob = ActionContext(target="live", owner="bob", approved_target="live")
    with lock("live", context=ALICE_LIVE), lock("live", join=True, context=bob):
        pass


def _planted(kind: str) -> Trigger:
    """Plant *kind* at the lock's name, the way the runtime tests do, and take the lock."""

    def plant(_: pytest.MonkeyPatch) -> None:
        path = lock_path("live")
        path.parent.mkdir(parents=True)
        victim = path.with_name("victim")
        victim.write_text("", encoding="utf-8")
        plant_kind(path, kind, victim)
        returns_within(_enter, release=path if kind == "fifo" else None)

    return plant


def _linked(levels_up: int) -> Trigger:
    """Put a link at the target's directory (0) or at ``guarded_run`` (1), and take the lock."""

    def link(_: pytest.MonkeyPatch) -> None:
        level = lock_path("live").parents[levels_up]
        lock_path("live").parent.mkdir(parents=True)
        link_directory(level)
        _enter()

    return link


def _not_a_directory(_: pytest.MonkeyPatch) -> None:
    level = lock_path("live").parent
    level.parent.mkdir(parents=True)
    level.write_text("", encoding="utf-8")
    _enter()


def _replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every open of the lock file finds a file no name holds, as a rename over it leaves."""
    lock_path("live").parent.mkdir(parents=True)
    real_fstat = os.fstat

    def nameless(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        return os.stat_result((*st[:3], 0, *st[4:10]))

    monkeypatch.setattr(guarded_run.os, "fstat", nameless)
    _enter()


def _unknown_target(_: pytest.MonkeyPatch) -> None:
    _enter("nowhere")


def _no_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    elsewhere = Path.cwd().parent / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    reset_config_cache()
    _enter()


def _with_mode(level: Path, mode: int) -> None:
    if os.geteuid() == 0:
        pytest.skip("root writes through any mode")
    level.mkdir(parents=True)
    os.chmod(level, mode)
    try:
        _enter()
    finally:
        os.chmod(level, 0o755)


def _uncreatable(_: pytest.MonkeyPatch) -> None:
    _with_mode(Path.cwd() / STATE_DIR_NAME, 0o555)


def _unwritable(_: pytest.MonkeyPatch) -> None:
    _with_mode(lock_path("live").parent, 0o555)


def _lock_file() -> str:
    return str(lock_path("live"))


def _target_dir() -> str:
    return str(lock_path("live").parent)


#: Each trigger of a refusal: the Refusals row it meets, how, and the values the
#: row's placeholders take, in order (``None`` for one the test does not pin).
TRIGGERS: list[tuple[str, str, Trigger, Callable[[], list[str | None]]]] = [
    (
        "Second run on a held target",
        "second-run",
        _second_run,
        lambda: [_target_dir(), str(os.getpid()), None, "unknown", "live", "unknown"],
    ),
    (
        "Another target while the process holds one",
        "another-target",
        _another_target,
        lambda: ["live", "va"],
    ),
    ("Read-only run", "read-only", _read_only, list),
    (
        "Record names another target",
        "record-mismatch",
        _record_mismatch,
        lambda: ["live", "va", "live"],
    ),
    ("Join with a different record", "join-mismatch", _join_mismatch, lambda: ["live"]),
    *(
        (case, kind, _planted(kind), lambda: [_lock_file()])
        for case, kind in (
            ("Link at the lock file", "symlink"),
            ("Link at the lock file", "dangling_symlink"),
            ("FIFO or other non-regular file at the lock file", "fifo"),
            ("FIFO or other non-regular file at the lock file", "directory"),
        )
    ),
    (
        "Hard link at the lock file",
        "hard_link",
        _planted("hard_link"),
        lambda: [_lock_file(), "2"],
    ),
    (
        "Link at ``guarded_run`` or the target's directory",
        "linked-target-directory",
        _linked(0),
        lambda: [_target_dir()],
    ),
    (
        "Link at ``guarded_run`` or the target's directory",
        "linked-guarded-run",
        _linked(1),
        lambda: [str(lock_path("live").parents[1])],
    ),
    (
        "Something other than a directory at ``guarded_run`` or the target's directory",
        "not-a-directory",
        _not_a_directory,
        lambda: [_target_dir()],
    ),
    ("Lock file replaced while it was opened", "replaced", _replaced, lambda: [_lock_file()]),
    (
        "Target that is not a control target",
        "unknown-target",
        _unknown_target,
        lambda: ["'nowhere'"],
    ),
    ("No deployment repo root", "no-repo-root", _no_repo_root, lambda: [None]),
    (
        "Directory cannot be created",
        "uncreatable",
        _uncreatable,
        lambda: [_target_dir(), None],
    ),
    ("Directory not writable", "unwritable", _unwritable, lambda: [_target_dir()]),
]


def test_the_planted_kinds_are_the_runtime_tests_ones():
    planted = [name for _, name, trigger, _ in TRIGGERS if trigger.__name__ == "plant"]
    assert sorted(planted) == sorted(PLANTED_REFUSED)


def _refusals() -> dict[str, str]:
    return {row[0]: _literals(row[1])[0] for row in _table("Refusals")}


def test_the_refusals_table_and_the_triggers_name_the_same_cases():
    documented = list(_refusals())
    assert len(documented) == len(set(documented))
    assert set(documented) == {case for case, *_ in TRIGGERS}


@pytest.mark.parametrize(
    ("case", "trigger", "fills"),
    [pytest.param(case, trigger, fills, id=name) for case, name, trigger, fills in TRIGGERS],
)
@pytest.mark.usefixtures("repo")
def test_each_refusal_text_is_the_runtime_message(
    case: str,
    trigger: Trigger,
    fills: Callable[[], list[str | None]],
    monkeypatch: pytest.MonkeyPatch,
):
    expected = fills()
    with pytest.raises(Exception) as raised:
        trigger(monkeypatch)
    message = str(raised.value)
    match = _template(_refusals()[case]).fullmatch(message)
    assert match, message
    assert len(match.groups()) == len(expected)
    for got, want in zip(match.groups(), expected, strict=True):
        if want is not None:
            assert got == want, message


@pytest.mark.usefixtures("repo")
def test_a_changed_refusal_text_fails_the_match():
    """Negative control: one word off in the documented text no longer matches."""
    text = _refusals()["Another target while the process holds one"]
    with pytest.raises(RuntimeError) as raised:
        _another_target(pytest.MonkeyPatch())
    match = _template(text).fullmatch(str(raised.value))
    assert match and match.groups() == ("live", "va")
    assert not _template(text.replace("cannot take", "may not take")).fullmatch(str(raised.value))


def test_a_table_the_page_lacks_fails_the_parse():
    """Negative control: the table reader refuses a caption the page does not have."""
    with pytest.raises(AssertionError, match="no single table"):
        _table("Holder records")
