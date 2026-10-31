"""The archive root: group-shared with the record, provisioned by every deployment.

``var/archive`` is the lasting half of the audit trail and has two writers in
two containers (agent-record's export and the archive service), so it is set up
exactly like ``var/agent_record``: setgid, group rwx, no world access, the
record's group. What is pinned here:

* the provisioner's mode and group, for a new directory and for one that exists
  at any other mode;
* a real ``osprey build`` provisions it on both record backends (embedded and
  the containerized ClickHouse store), with or without the archive service;
* a file a container leaves there, at the mode its writers give files, is read
  and deleted by the host-side paths that run as the deploying account;
* agent-record resolves the root from its env (the container's bind) or from
  the project root (the host), and a root it cannot write is reported at start
  without failing start-up.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from osprey.agent_record import service
from osprey.deployment.compose_generator import (
    SHARED_CORPUS_DIR_MODE,
    ensure_agent_record_dirs,
    ensure_archive_dir,
)
from osprey.health.core.archive import archive as archive_health
from osprey.services.archive.manifest import MANIFEST_NAME, append_record, load_state

posix_only = pytest.mark.skipif(os.name != "posix", reason="setgid modes are POSIX")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


#: The mode both writers give the files they create: the root's, without the
#: setgid and execute bits.
CONTAINER_FILE_MODE = SHARED_CORPUS_DIR_MODE & ~stat.S_ISGID & ~0o111


# ── The provisioner ─────────────────────────────────────────────────────────


@posix_only
def test_created_group_shared_with_the_records_group(tmp_path):
    old = os.umask(0)  # a permissive umask must not widen it
    try:
        record = ensure_agent_record_dirs(tmp_path)[-1]
        target = ensure_archive_dir(tmp_path)
    finally:
        os.umask(old)

    assert target == tmp_path / "var" / "archive"
    assert _mode(target) == 0o2770 == SHARED_CORPUS_DIR_MODE
    assert _mode(target) & stat.S_IRWXO == 0, "the audit trail is open to other"
    assert target.stat().st_gid == record.stat().st_gid


@posix_only
@pytest.mark.parametrize("existing", [0o700, 0o750, 0o755, 0o775, 0o2775])
def test_an_existing_root_is_set_to_exactly_the_shared_mode(tmp_path, existing):
    """The directory is the framework's: a private one would lock the record's
    export out, and a wider one would let other read the audit trail."""
    target = tmp_path / "var" / "archive"
    target.mkdir(parents=True)
    os.chmod(target, existing)

    ensure_archive_dir(tmp_path)

    assert _mode(target) == SHARED_CORPUS_DIR_MODE


def test_a_failure_warns_and_does_not_raise(tmp_path, monkeypatch, caplog):
    def _refuse(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", _refuse)

    with caplog.at_level(logging.WARNING, logger="deployment.compose"):
        assert ensure_archive_dir(tmp_path) is None

    assert "Could not provision archive directory" in caplog.text


# ── A real build, on both record backends ───────────────────────────────────


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, *, archive: bool):
    from tests._builds import init_project, run_build

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = init_project(tmp_path, "hello-world", "demo")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    if backend != "embedded":
        text += f"\nagent_record:\n  backend: {backend}\n"
    if archive:
        text = text.replace(
            "\nservices: {}\n", "\nservices:\n  archive:\n    template: osprey.archive\n"
        )
        assert "osprey.archive" in text, "the profile's services block moved"
    profile.write_text(text, encoding="utf-8")

    result = run_build(repo)

    assert result.exit_code == 0, result.output
    return repo


def _compose_services(repo: Path) -> dict[str, dict]:
    services: dict[str, dict] = {}
    for compose in (repo / "build" / "services").rglob("docker-compose.yml"):
        doc = yaml.safe_load(compose.read_text(encoding="utf-8")) or {}
        services.update(doc.get("services") or {})
    return services


def _mount_sources(service_block: dict) -> list[str]:
    sources = []
    for mount in service_block.get("volumes") or []:
        if isinstance(mount, str):
            sources.append(mount.split(":")[0])
        elif isinstance(mount, dict):
            sources.append(str(mount.get("source", "")))
    return sources


@posix_only
@pytest.mark.parametrize("backend", ["embedded", "clickhouse"])
def test_every_build_provisions_the_root_beside_the_record(tmp_path, monkeypatch, backend):
    """No archive service deployed: the record's own export still needs it."""
    repo = _build(tmp_path, monkeypatch, backend, archive=False)

    archive_root = repo / "var" / "archive"
    record_dir = repo / "var" / "agent_record"
    assert archive_root.is_dir()
    assert _mode(archive_root) == SHARED_CORPUS_DIR_MODE
    assert archive_root.stat().st_gid == record_dir.stat().st_gid

    record = _compose_services(repo)["agent-record"]
    container_root = "/app/demo/var/archive"
    assert record["environment"][service.ARCHIVE_DIR_ENV] == container_root
    assert f"./var/archive:{container_root}" in record["volumes"]


@posix_only
@pytest.mark.parametrize("backend", ["embedded", "clickhouse"])
def test_only_the_record_and_the_archive_service_mount_the_root(tmp_path, monkeypatch, backend):
    repo = _build(tmp_path, monkeypatch, backend, archive=True)

    services = _compose_services(repo)
    assert {"agent-record", "archive"} <= set(services), sorted(services)
    mounting = sorted(
        name
        for name, block in services.items()
        if any(source.rstrip("/").endswith("var/archive") for source in _mount_sources(block))
    )
    assert mounting == ["agent-record", "archive"]


# ── What a container writes, the host reads and deletes ─────────────────────


@posix_only
def test_a_container_written_file_is_read_and_deleted_by_the_host(tmp_path):
    """The deploying account runs ``osprey health``, the export and
    ``osprey reset --purge-audit``. It reaches a container's files through the
    root's group, never through ownership, so what is pinned is the group
    grant: the root's group is one this account holds, the root and the day
    directories grant that group write with no sticky bit (so it deletes any
    member's entries), and a file at the writers' file mode grants it read.
    """
    ensure_agent_record_dirs(tmp_path)
    root = ensure_archive_dir(tmp_path)
    assert root is not None
    assert root.stat().st_gid in {os.getegid(), *os.getgroups()}

    # What a writer leaves: a day directory at the root's mode, a manifest and
    # a copy at the file mode.
    day = root / datetime.now(UTC).strftime("%Y-%m-%d")
    day.mkdir()
    os.chmod(day, _mode(root))
    copy = day / "agent_record" / "records.parquet"
    copy.parent.mkdir()
    os.chmod(copy.parent, _mode(root))
    copy.write_bytes(b"PAR1")
    os.chmod(copy, CONTAINER_FILE_MODE)
    now = datetime.now(UTC).isoformat()
    append_record(
        day / MANIFEST_NAME,
        {"kind": "pass", "started_at": now, "completed_at": now, "errors": []},
        file_mode=CONTAINER_FILE_MODE,
    )

    for directory in (root, day, copy.parent):
        mode = _mode(directory)
        assert mode & stat.S_IRWXG == stat.S_IRWXG, (directory, oct(mode))
        assert not mode & stat.S_ISVTX, (directory, oct(mode))
        assert not mode & stat.S_IRWXO, (directory, oct(mode))
    for written in (copy, day / MANIFEST_NAME):
        mode = _mode(written)
        assert mode == 0o660, (written, oct(mode))
        assert mode & stat.S_IRGRP, (written, oct(mode))

    # health: reads the manifests.
    assert load_state(root).last_pass is not None
    rows = archive_health({"deployed_services": ["archive"]}, cwd=tmp_path)()
    assert rows and "unreadable" not in rows[0].message.lower(), rows
    # export: reads the copies.
    assert copy.read_bytes() == b"PAR1"
    # --purge-audit: deletes them.
    shutil.rmtree(day)
    assert list(root.iterdir()) == []


# ── agent-record resolves the root ──────────────────────────────────────────


def _settings(tmp_path: Path, env: dict[str, str]) -> service._Settings:
    return service._settings({"project_root": str(tmp_path)}, env)


def test_the_root_defaults_to_the_projects_archive_on_the_host(tmp_path):
    assert _settings(tmp_path, {}).archive_root == tmp_path / "var" / "archive"


def test_the_root_is_the_bind_the_container_env_names(tmp_path):
    bound = "/app/demo/var/archive"

    settings = _settings(tmp_path, {service.ARCHIVE_DIR_ENV: bound})

    assert settings.archive_root == Path(bound)


def test_a_writable_root_is_not_reported(tmp_path, caplog):
    root = ensure_archive_dir(tmp_path)

    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        assert service._report_archive_root(root) is True

    assert not caplog.records


def test_a_missing_root_is_reported_by_path(tmp_path, caplog):
    root = tmp_path / "var" / "archive"

    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        assert service._report_archive_root(root) is False

    assert str(root) in caplog.text


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
def test_an_unwritable_root_is_reported_by_path(tmp_path, caplog):
    root = tmp_path / "var" / "archive"
    root.mkdir(parents=True)
    os.chmod(root, 0o500)
    try:
        with caplog.at_level(logging.WARNING, logger=service.logger.name):
            assert service._report_archive_root(root) is False
    finally:
        os.chmod(root, 0o700)

    assert str(root) in caplog.text
    assert "not writable" in caplog.text


async def test_start_up_goes_on_over_an_unwritable_root(tmp_path, caplog):
    """The report is a warning, never a refusal: the service still serves."""
    import base64
    import json

    import httpx

    from osprey.agent_record import credentials as creds
    from osprey.agent_record import identities as ids

    def b64(text: str) -> str:
        return base64.b64encode(text.encode()).decode()

    config = {"project_root": str(tmp_path)}
    verifier = "".join(
        creds.verifier_line(login, f"token-of-{login}-0123456789") + "\n"
        for login in ids.credential_table(config)
    )
    root = tmp_path / "missing" / "archive"
    app = service.create_app(
        config=config,
        env={
            creds.VERIFIER_ENV: b64(verifier),
            creds.SECRETS_ENV: b64(json.dumps(["s3cr3t-token-value-0123456789"])),
            service.SPOOL_ROOT_ENV: str(tmp_path / "spool"),
            service.RECORD_DIR_ENV: str(tmp_path / "record"),
            service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "transcripts"),
            service.ARCHIVE_DIR_ENV: str(root),
        },
    )

    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://r") as client:
                response = await client.get("/health")

    assert response.status_code == 200, response.text
    assert app.state.settings.archive_root == root
    assert str(root) in caplog.text
