"""``osprey ariel web`` binds the address its config names unless told otherwise."""

import pytest
import yaml
from click.testing import CliRunner

import osprey.interfaces.ariel
from osprey.cli.ariel import ariel_group, web_command
from osprey.registry.web import framework_web_port_default

CONFIGURED_HOST = "192.0.2.10"
CONFIGURED_PORT = 18300


def _write_config(tmp_path, monkeypatch, *, web: bool) -> None:
    ariel: dict = {"database": {"uri": "postgresql://unused"}}
    if web:
        ariel["web"] = {"host": CONFIGURED_HOST, "port": CONFIGURED_PORT}
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"project_name": "demo", "ariel": ariel}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_ARIEL_PORT", raising=False)


@pytest.fixture
def bound(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(osprey.interfaces.ariel, "run_web", lambda **kw: seen.update(kw))
    return seen


def test_host_option_has_no_frozen_default():
    host = next(p for p in web_command.params if p.name == "host")
    assert host.default is None


def test_configured_host_and_port_are_bound(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, web=True)
    result = CliRunner().invoke(ariel_group, ["web"])
    assert result.exit_code == 0, result.output
    assert (bound["host"], bound["port"]) == (CONFIGURED_HOST, CONFIGURED_PORT)
    assert f"http://{CONFIGURED_HOST}:{CONFIGURED_PORT}" in result.output


def test_explicit_host_wins(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, web=True)
    result = CliRunner().invoke(ariel_group, ["web", "--host", "127.0.0.1"])
    assert result.exit_code == 0, result.output
    assert (bound["host"], bound["port"]) == ("127.0.0.1", CONFIGURED_PORT)


def test_explicit_port_keeps_configured_host(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, web=True)
    result = CliRunner().invoke(ariel_group, ["web", "--port", "18999"])
    assert result.exit_code == 0, result.output
    assert (bound["host"], bound["port"]) == (CONFIGURED_HOST, 18999)


def test_no_configured_host_binds_loopback(tmp_path, monkeypatch, bound):
    _write_config(tmp_path, monkeypatch, web=False)
    result = CliRunner().invoke(ariel_group, ["web"])
    assert result.exit_code == 0, result.output
    assert (bound["host"], bound["port"]) == (
        "127.0.0.1",
        framework_web_port_default("ariel"),
    )
