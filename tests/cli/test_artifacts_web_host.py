"""``osprey artifacts web`` binds the address its config names unless told otherwise."""

import pytest
import yaml
from click.testing import CliRunner

import osprey.interfaces.artifacts
from osprey.cli.artifacts_cmd import artifacts
from osprey.registry.web import framework_web_port_default

CONFIGURED_HOST = "192.0.2.30"
CONFIGURED_PORT = 18500
CONFIGURED = {"host": CONFIGURED_HOST, "port": CONFIGURED_PORT}


def _write_config(tmp_path, monkeypatch, section: dict) -> None:
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"project_name": "demo", "artifact_server": section})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_ARTIFACT_SERVER_PORT", raising=False)
    monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)


@pytest.fixture
def bound(monkeypatch):
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        osprey.interfaces.artifacts,
        "run_server",
        lambda host, port: seen.append((host, port)),
    )
    return seen


def test_configured_host_and_port_are_bound(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, CONFIGURED)
    result = CliRunner().invoke(artifacts, ["web"])
    assert result.exit_code == 0, result.output
    assert bound == [(CONFIGURED_HOST, CONFIGURED_PORT)]
    assert f"http://{CONFIGURED_HOST}:{CONFIGURED_PORT}" in result.output


def test_explicit_host_keeps_configured_port(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, CONFIGURED)
    result = CliRunner().invoke(artifacts, ["web", "--host", "127.0.0.1"])
    assert result.exit_code == 0, result.output
    assert bound == [("127.0.0.1", CONFIGURED_PORT)]


def test_explicit_port_keeps_configured_host(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, CONFIGURED)
    result = CliRunner().invoke(artifacts, ["web", "--port", "18999"])
    assert result.exit_code == 0, result.output
    assert bound == [(CONFIGURED_HOST, 18999)]


def test_port_zero_is_bound_as_given(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, CONFIGURED)
    result = CliRunner().invoke(artifacts, ["web", "--port", "0"])
    assert result.exit_code == 0, result.output
    assert bound == [(CONFIGURED_HOST, 0)]


def test_both_flags_do_not_read_the_config(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, {"web": {"port": 1}})
    result = CliRunner().invoke(artifacts, ["web", "--host", "127.0.0.1", "--port", "18999"])
    assert result.exit_code == 0, result.output
    assert bound == [("127.0.0.1", 18999)]


def test_no_configured_host_binds_loopback(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, {})
    result = CliRunner().invoke(artifacts, ["web"])
    assert result.exit_code == 0, result.output
    assert bound == [("127.0.0.1", framework_web_port_default("artifact"))]
