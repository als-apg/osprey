"""``osprey reset`` keeps the agent record on either backend; ``--purge-audit`` removes it.

The record is audit data, so a reset treats it as it treats ``var/audit``: a
plain reset keeps it and says so under WILL BE KEPT, and ``--purge-audit``
removes it and says so under WILL BE REMOVED. On the embedded backend the
record is ``var/agent_record/``; on the ClickHouse backend it is also the store's
data volume. ``var/spool/`` goes with it on both, so the verb does the same to
the record whichever backend keeps it.

``var/archive/`` is kept whole by a plain reset. ``--purge-audit`` removes the
record's export of each day from it (``<day>/agent_record/``), appends an
``agent_record_purged`` line to that day's manifest, and keeps everything else
there — an artifact copy filed under the same day included. ``var/state/`` (the
locks, leases and markers of each identity) survives every reset. And every
reset, plain or purging, dry or real, refuses while a host session holds the
host-local record, naming the sessions that hold it.

No test here reaches a container runtime: the runtime is the recording fake of
``tests/deployment/test_reset_scoping.py``. The leases are real lease files
naming this test process, so liveness is judged by the code that judges it in
production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from osprey.agent_record import host
from osprey.deployment import reset as reset_mod
from osprey.deployment.compose_generator import AGENT_RECORD_CLICKHOUSE_VOLUME
from osprey.deployment.reset import HostRecordRunningError, plan_reset, reset_for_reinit
from tests.deployment.test_reset_scoping import (
    PROJECT,
    FakeRuntime,
    make_probe,
    run_reset,
    unlabelled,
)

BACKENDS = ("embedded", "clickhouse")
RECORD_VOLUME = f"{PROJECT}_{AGENT_RECORD_CLICKHOUSE_VOLUME}"
OTHER_VOLUME = f"{PROJECT}_mongodb_data"


@pytest.fixture(autouse=True)
def _no_image_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OSPREY_IMAGE_REGISTRY", "OSPREY_IMAGE_TAG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_down(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    monkeypatch.setattr(reset_mod, "down_deployment", lambda root, **_: calls.append(Path(root)))
    return calls


def _repo(tmp_path: Path, backend: str) -> Path:
    """A built deployment repo holding a record, its spool and an audit log."""
    root = tmp_path / PROJECT
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text(
        f"project_name: {PROJECT}\n"
        "build_dir: ./build\n"
        "deployed_services: []\n"
        "services: {}\n"
        f"agent_record:\n  backend: {backend}\n",
        encoding="utf-8",
    )
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")
    for relpath, text in (
        ("var/audit/2026-10-24.jsonl", '{"ok":true}\n'),
        ("var/agent_record/record.duckdb", "record"),
        ("var/spool/terminal-alice/0001.jsonl", '{"entry":1}\n'),
        ("var/state/alice/revocations.json", "[]"),
        ("var/archive/artifacts/plot-1.png", "png"),
        ("var/archive/bluesky/run-7.jsonl", '{"run":7}\n'),
    ):
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _runtime(backend: str) -> FakeRuntime:
    volumes = {OTHER_VOLUME: unlabelled()}
    if backend == "clickhouse":
        volumes[RECORD_VOLUME] = unlabelled()
    return FakeRuntime(volumes=volumes)


def _sections(lines: list[str]) -> tuple[str, str]:
    text = "\n".join(lines)
    removed, kept = text.split("WILL BE REMOVED", 1)[1].split("WILL BE KEPT", 1)
    return removed, kept


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_plain_reset_keeps_the_record_and_says_so(tmp_path: Path, backend: str) -> None:
    repo = _repo(tmp_path, backend)
    fake = _runtime(backend)

    plan = plan_reset(repo, probe=make_probe(fake))
    removed, kept = _sections(plan.render())

    assert "var/agent_record/" in kept
    assert "var/spool/" in kept
    assert "var/agent_record" not in removed
    assert "var/spool" not in removed
    if backend == "clickhouse":
        assert RECORD_VOLUME in kept
        assert RECORD_VOLUME not in removed
        assert [v.name for v in plan.kept_volumes] == [RECORD_VOLUME]
    else:
        assert AGENT_RECORD_CLICKHOUSE_VOLUME not in kept
        assert plan.kept_volumes == []
    assert [v.name for v in plan.volumes] == [OTHER_VOLUME]

    run_reset(repo, fake)

    assert (repo / "var" / "agent_record" / "record.duckdb").is_file()
    assert (repo / "var" / "spool" / "terminal-alice" / "0001.jsonl").is_file()
    assert fake.removed_names() == [OTHER_VOLUME]
    if backend == "clickhouse":
        assert RECORD_VOLUME in fake.volumes


@pytest.mark.parametrize("backend", BACKENDS)
def test_purge_audit_removes_the_record_and_says_so(tmp_path: Path, backend: str) -> None:
    repo = _repo(tmp_path, backend)
    fake = _runtime(backend)

    plan = plan_reset(repo, probe=make_probe(fake), purge_audit=True)
    removed, kept = _sections(plan.render())

    assert "var/agent_record" in removed
    assert "var/spool" in removed
    assert "var/agent_record/" not in kept
    assert "var/spool/" not in kept
    assert plan.kept_volumes == []
    if backend == "clickhouse":
        assert RECORD_VOLUME in removed
        assert RECORD_VOLUME not in kept

    run_reset(repo, fake, purge_audit=True)

    # Recreated empty, as every state directory a reset wipes is.
    for relpath in ("var/agent_record", "var/spool"):
        assert (repo / relpath).is_dir()
        assert not any((repo / relpath).iterdir())
    assert not (repo / "var" / "audit" / "2026-10-24.jsonl").exists()
    expected = {OTHER_VOLUME, RECORD_VOLUME} if backend == "clickhouse" else {OTHER_VOLUME}
    assert set(fake.removed_names()) == expected
    assert fake.volumes == {}


def test_only_the_exact_record_volume_name_is_kept(
    tmp_path: Path,
) -> None:
    """Only this project's record volume is kept: the name is matched whole."""
    repo = _repo(tmp_path, "clickhouse")
    lookalike = f"{PROJECT}_{AGENT_RECORD_CLICKHOUSE_VOLUME}_old"
    fake = FakeRuntime(volumes={lookalike: unlabelled(), RECORD_VOLUME: unlabelled()})

    plan = plan_reset(repo, probe=make_probe(fake))

    assert [v.name for v in plan.kept_volumes] == [RECORD_VOLUME]
    assert [v.name for v in plan.volumes] == [lookalike]


# ---------------------------------------------------------------------------
# What no reset touches
# ---------------------------------------------------------------------------

#: Files the fixture repo holds under the two directories every reset keeps.
SURVIVORS = (
    "var/state/alice/revocations.json",
    "var/archive/artifacts/plot-1.png",
    "var/archive/bluesky/run-7.jsonl",
)


@pytest.mark.parametrize("purge_audit", [False, True], ids=["plain", "purge-audit"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_state_and_archive_survive_every_reset(
    tmp_path: Path, backend: str, purge_audit: bool
) -> None:
    repo = _repo(tmp_path, backend)
    fake = _runtime(backend)

    plan = plan_reset(repo, probe=make_probe(fake), purge_audit=purge_audit)
    removed, kept = _sections(plan.render())

    assert "var/state" not in removed
    assert "var/archive" not in removed
    assert "var/state/" in kept
    assert "var/archive/" in kept
    assert not any(
        path == repo / "var" / name or (repo / "var" / name) in path.parents
        for path in plan.paths
        for name in ("state", "archive")
    )

    run_reset(repo, fake, purge_audit=purge_audit)

    for relpath in SURVIVORS:
        assert (repo / relpath).is_file(), relpath


# ---------------------------------------------------------------------------
# The record's exports in the archive
# ---------------------------------------------------------------------------

#: The arrival day the fixture's record exported, with an artifact copy filed
#: under the same day directory, and an empty day marked with no file.
EXPORT_DAY = "2026-10-20"
EMPTY_DAY = "2026-10-21"
EXPORT = f"var/archive/{EXPORT_DAY}/agent_record/entries.parquet"
SAME_DAY_ARTIFACT = f"var/archive/{EXPORT_DAY}/artifacts/plot-2.png"


def _with_exports(repo: Path) -> Path:
    """Give *repo*'s archive two exported days, as the record's daily export leaves them."""
    archive = repo / "var" / "archive"
    for relpath, text in ((EXPORT, "PAR1"), (SAME_DAY_ARTIFACT, "png")):
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for day, rows in ((EXPORT_DAY, 3), (EMPTY_DAY, 0)):
        manifest = archive / day / "MANIFEST.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "agent_record_day", "day": day, "rows": rows}) + "\n")
    return repo


def _manifest_kinds(repo: Path, day: str) -> list[str]:
    manifest = repo / "var" / "archive" / day / "MANIFEST.jsonl"
    return [json.loads(line)["kind"] for line in manifest.read_text().splitlines() if line]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_plain_reset_keeps_the_records_exports(tmp_path: Path, backend: str) -> None:
    from osprey.services.archive.manifest import load_state

    repo = _with_exports(_repo(tmp_path, backend))
    fake = _runtime(backend)

    plan = plan_reset(repo, probe=make_probe(fake))
    removed, kept = _sections(plan.render())

    assert plan.archive_record_days == ()
    assert "var/archive" not in removed
    assert "var/archive/" in kept
    assert "--purge-audit` removes the record's exports" in kept

    run_reset(repo, fake)

    assert (repo / EXPORT).is_file()
    assert (repo / SAME_DAY_ARTIFACT).is_file()
    assert _manifest_kinds(repo, EXPORT_DAY) == ["agent_record_day"]
    assert load_state(repo / "var" / "archive").agent_record_days == {EXPORT_DAY, EMPTY_DAY}


@pytest.mark.parametrize("backend", BACKENDS)
def test_purge_audit_removes_the_records_exports_and_marks_each_day(
    tmp_path: Path, backend: str
) -> None:
    from osprey.services.archive.manifest import load_state

    repo = _with_exports(_repo(tmp_path, backend))
    fake = _runtime(backend)

    plan = plan_reset(repo, probe=make_probe(fake), purge_audit=True)
    removed, kept = _sections(plan.render())

    # The empty day has no file but is marked exported, so it is purged too.
    assert plan.archive_record_days == (EXPORT_DAY, EMPTY_DAY)
    assert "var/archive/*/agent_record/" in removed
    assert f"2 days ({EXPORT_DAY} to {EMPTY_DAY})" in removed
    assert "the artifact and Bluesky copies and every manifest" in kept
    assert "the record's archive of 2 days" in plan.confirmation_summary()
    # The archive root itself is never on the directory removal list.
    assert not any((repo / "var" / "archive") in (p, *p.parents) for p in plan.paths)

    run_reset(repo, fake, purge_audit=True)

    assert not (repo / EXPORT).exists()
    assert not (repo / EXPORT).parent.exists()
    assert (repo / SAME_DAY_ARTIFACT).is_file()
    for relpath in SURVIVORS:
        assert (repo / relpath).is_file(), relpath
    assert _manifest_kinds(repo, EXPORT_DAY) == ["agent_record_day", "agent_record_purged"]
    assert _manifest_kinds(repo, EMPTY_DAY) == ["agent_record_day", "agent_record_purged"]
    assert load_state(repo / "var" / "archive").agent_record_days == set()
    assert not (repo / "var" / "audit" / "2026-10-24.jsonl").exists()
    assert not (repo / "var" / "agent_record" / "record.duckdb").exists()
    assert not any((repo / "var" / "spool").iterdir())
    if backend == "clickhouse":
        assert RECORD_VOLUME in fake.removed_names()


def test_purge_audit_with_no_exports_marks_nothing(tmp_path: Path) -> None:
    """An archive the record never exported into gains no manifest at all."""
    repo = _repo(tmp_path, "embedded")

    plan = plan_reset(repo, probe=make_probe(_runtime("embedded")), purge_audit=True)
    removed, _kept = _sections(plan.render())

    assert plan.archive_record_days == ()
    assert "agent_record/  the agent record's archive" not in removed

    run_reset(repo, _runtime("embedded"), purge_audit=True)

    assert not list((repo / "var" / "archive").rglob("MANIFEST.jsonl"))
    for relpath in SURVIVORS:
        assert (repo / relpath).is_file(), relpath


def test_a_dry_run_purge_touches_no_export(tmp_path: Path) -> None:
    repo = _with_exports(_repo(tmp_path, "embedded"))
    before = _snapshot(repo)

    run_reset(repo, _runtime("embedded"), purge_audit=True, dry_run=True)

    assert _snapshot(repo) == before


# ---------------------------------------------------------------------------
# The refusal while a host session holds the host-local record
# ---------------------------------------------------------------------------


def _write_lease(repo: Path, *, start: str | None = None) -> Path:
    """A lease file for this test process, as a host launcher writes it."""
    pid = os.getpid()
    if start is None:
        start = host._process_start(pid)
        assert start, "ps cannot read this process's start time"
    lease_dir = repo / "var" / "state" / "operator" / host.LEASE_DIR_NAME
    lease_dir.mkdir(parents=True, exist_ok=True)
    path = lease_dir / f"{pid}-test.lease"
    path.write_text(json.dumps({"pid": pid, "start": start}), encoding="utf-8")
    return path


def _snapshot(repo: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(repo)): p.read_bytes() for p in sorted(repo.rglob("*")) if p.is_file()
    }


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
@pytest.mark.parametrize("purge_audit", [False, True], ids=["plain", "purge-audit"])
def test_a_live_lease_refuses_every_reset_and_names_the_session(
    tmp_path: Path, no_down: list[Path], purge_audit: bool, dry_run: bool
) -> None:
    repo = _repo(tmp_path, "embedded")
    _write_lease(repo)
    fake = _runtime("clickhouse")
    before = _snapshot(repo)

    with pytest.raises(HostRecordRunningError) as caught:
        run_reset(repo, fake, purge_audit=purge_audit, dry_run=dry_run)

    error = caught.value
    assert [holder.pid for holder in error.holders] == [os.getpid()]
    assert f"pid {os.getpid()}" in str(error)
    # The session is named by its command line, which for this process is the
    # interpreter running the tests.
    assert "python" in error.holders[0].command.lower()
    assert "Nothing was stopped, removed, or written." in str(error)
    assert no_down == []
    assert fake.removed_names() == []
    assert _snapshot(repo) == before


def test_init_reset_refuses_on_a_live_lease_too(tmp_path: Path, no_down: list[Path]) -> None:
    repo = _repo(tmp_path, "embedded")
    _write_lease(repo)
    fake = _runtime("embedded")

    with pytest.raises(HostRecordRunningError):
        reset_for_reinit(repo, probe=make_probe(fake))

    assert no_down == []
    assert fake.removed_names() == []


def test_a_running_record_process_with_no_lease_refuses(
    tmp_path: Path, no_down: list[Path]
) -> None:
    """The process can outlive its last lease briefly; a reset still waits for it."""
    repo = _repo(tmp_path, "embedded")
    state_dir = repo / "var" / "state" / "operator"
    state_dir.mkdir(parents=True)
    (state_dir / host.PROCESS_FILE_NAME).write_text(
        json.dumps({"pid": os.getpid(), "start": host._process_start(os.getpid())}),
        encoding="utf-8",
    )

    with pytest.raises(HostRecordRunningError) as caught:
        run_reset(repo, _runtime("embedded"))

    assert caught.value.holders == ()
    assert f"pid {os.getpid()}" in str(caught.value)
    assert no_down == []


def test_a_stale_lease_does_not_refuse(tmp_path: Path, no_down: list[Path]) -> None:
    """A lease whose process is gone (same pid, other start time) holds nothing."""
    repo = _repo(tmp_path, "embedded")
    lease = _write_lease(repo, start="Mon Jan  1 00:00:00 2001")
    fake = _runtime("embedded")

    run_reset(repo, fake)

    assert len(no_down) == 1
    assert fake.removed_names() == [OTHER_VOLUME]
    # The lease directory is under var/state, which a reset never touches.
    assert lease.is_file()


def test_a_lease_taken_after_the_plan_stops_the_execution(
    tmp_path: Path, no_down: list[Path]
) -> None:
    """The plan is checked again just before the first removal."""
    repo = _repo(tmp_path, "embedded")
    fake = _runtime("embedded")
    plan = plan_reset(repo, probe=make_probe(fake))
    _write_lease(repo)

    with pytest.raises(HostRecordRunningError):
        reset_mod.execute_reset(plan, probe=make_probe(fake))

    assert no_down == []
    assert fake.removed_names() == []
