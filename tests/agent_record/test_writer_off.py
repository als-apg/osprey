"""The record off switch: ``OSPREY_AGENT_RECORD_OFF=1`` makes ``write`` a no-op.

With the switch set, :func:`osprey.agent_record.writer.write` answers ``None``
without resolving a spool directory or touching disk: no segment, no
``.errors`` line, no directory. Any other value leaves ``write`` unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.agent_record import contract, writer


@pytest.fixture(autouse=True)
def _fresh_writer(monkeypatch):
    writer.seal_all()
    monkeypatch.delenv(contract.SPOOL_DIR_ENV, raising=False)
    monkeypatch.delenv(contract.RECORD_OFF_ENV, raising=False)
    yield
    writer.seal_all()


@pytest.fixture
def host_project(tmp_path, monkeypatch) -> Path:
    """A host-form project root at *tmp_path*, acting as ``carol``."""
    import osprey_connectors.workspace as workspace

    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    monkeypatch.setattr(workspace, "resolve_project_root", lambda config=None: tmp_path)
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "carol")
    return tmp_path


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")) if root.exists() else []


def test_the_switch_is_named_in_the_contract():
    assert contract.RECORD_OFF_ENV == "OSPREY_AGENT_RECORD_OFF"


def test_off_writes_nothing_under_the_spool_root(tmp_path, monkeypatch):
    spool = tmp_path / "spool" / "alice"
    monkeypatch.setenv(contract.SPOOL_DIR_ENV, str(spool))
    monkeypatch.setenv(contract.RECORD_OFF_ENV, "1")

    assert not writer.write("audit", {"event": "x"})

    assert not (tmp_path / "spool").exists()


def test_off_writes_nothing_under_the_project_var_spool(host_project, monkeypatch):
    monkeypatch.setenv(contract.RECORD_OFF_ENV, "1")

    assert not writer.write("feedback", {"text": "hi"})

    assert not (host_project / "var").exists()
    assert _files(host_project) == []


def test_off_never_resolves_a_spool_dir(monkeypatch):
    monkeypatch.setenv(contract.RECORD_OFF_ENV, "1")

    def boom():
        raise AssertionError("spool_dir() must not be called with the record off")

    monkeypatch.setattr(writer, "spool_dir", boom)
    assert writer.write("audit", {}) is None


def test_off_leaves_an_existing_spool_untouched(tmp_path, monkeypatch):
    spool = tmp_path / "spool" / "alice"
    monkeypatch.setenv(contract.SPOOL_DIR_ENV, str(spool))
    assert writer.write("audit", {"n": 1})
    writer.seal_all()
    before = {p: p.read_bytes() for p in _files(spool)}

    monkeypatch.setenv(contract.RECORD_OFF_ENV, "1")
    assert writer.write("audit", {"n": 2}) is None

    assert {p: p.read_bytes() for p in _files(spool)} == before


def test_off_is_read_per_call(tmp_path, monkeypatch):
    spool = tmp_path / "spool" / "alice"
    monkeypatch.setenv(contract.SPOOL_DIR_ENV, str(spool))
    monkeypatch.setenv(contract.RECORD_OFF_ENV, "1")
    assert writer.write("audit", {}) is None
    assert not spool.exists()

    monkeypatch.delenv(contract.RECORD_OFF_ENV)
    assert writer.write("audit", {})
    assert spool.exists()


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_without_the_switch_write_is_unchanged(host_project, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(contract.RECORD_OFF_ENV, value)

    entry_id = writer.write("feedback", {"text": "hi"})

    assert entry_id
    directory = host_project / "var" / "spool" / "carol"
    writer.seal_all()
    segments = [p for p in directory.iterdir() if p.name.endswith(contract.SEALED_SUFFIX)]
    assert len(segments) == 1
    assert contract.parse_spool_line(segments[0].read_bytes().splitlines()[0]).entry_id == entry_id
