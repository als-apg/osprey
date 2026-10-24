"""``osprey reset`` keeps the agent record on either backend; ``--purge-audit`` removes it.

The record is audit data, so a reset treats it as it treats ``var/audit``: a
plain reset keeps it and says so under WILL BE KEPT, and ``--purge-audit``
removes it and says so under WILL BE REMOVED. On the embedded backend the
record is ``var/agent_record/``; on the ClickHouse backend it is also the store's
data volume. ``var/spool/`` goes with it on both, so the verb does the same to
the record whichever backend keeps it.

No test here reaches a container runtime: the runtime is the recording fake of
``tests/deployment/test_reset_scoping.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment import reset as reset_mod
from osprey.deployment.compose_generator import AGENT_RECORD_CLICKHOUSE_VOLUME
from osprey.deployment.reset import plan_reset
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
    monkeypatch.setattr(reset_mod, "down_deployment", lambda root: calls.append(Path(root)))
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
