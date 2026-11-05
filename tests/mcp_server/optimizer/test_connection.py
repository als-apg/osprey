"""Which optimizer the server dials: ``OPTIMIZER_URL``, else the rendered config's port."""

from __future__ import annotations

import pytest
import yaml

from osprey.mcp_server.optimizer import server_context
from osprey.optimizer_connection import (
    DEFAULT_OPTIMIZER_PORT,
    optimizer_url_from_config,
    resolve_optimizer_url,
)
from osprey.port_layout import default_port


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A cwd whose config.yml the resolver loads; no override in the environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPTIMIZER_URL", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    def write(config: dict) -> None:
        (tmp_path / "config.yml").write_text(yaml.safe_dump(config))

    return write


def test_the_env_override_wins_over_the_config(project, monkeypatch):
    project({"services": {"optimizer": {"port": 18182}}})
    monkeypatch.setenv("OPTIMIZER_URL", "http://opt.example:9/")
    assert resolve_optimizer_url() == "http://opt.example:9"


def test_the_published_port_is_dialed_on_loopback(project):
    project({"services": {"optimizer": {"port": 18182}}})
    assert resolve_optimizer_url() == "http://127.0.0.1:18182"


def test_an_empty_override_falls_back_to_the_config(project, monkeypatch):
    # The registry renders `${OPTIMIZER_URL:-}`: unset reaches the server empty.
    project({"services": {"optimizer": {"port": 18182}}})
    monkeypatch.setenv("OPTIMIZER_URL", "")
    assert resolve_optimizer_url() == "http://127.0.0.1:18182"


def test_no_port_is_the_layout_slot_at_this_configs_base():
    config = {"deployment": {"port_base": 20000}}
    expected = default_port("optimizer", base=20000)
    assert optimizer_url_from_config(config) == f"http://127.0.0.1:{expected}"


def test_no_config_is_the_layout_default():
    assert optimizer_url_from_config(None) == f"http://127.0.0.1:{DEFAULT_OPTIMIZER_PORT}"


@pytest.mark.usefixtures("project")
def test_the_server_resolves_once_per_process(monkeypatch):
    monkeypatch.setenv("OPTIMIZER_URL", "http://first:1")
    server_context.reset_server_context()
    try:
        assert server_context.base_url() == "http://first:1"
        monkeypatch.setenv("OPTIMIZER_URL", "http://second:2")
        assert server_context.base_url() == "http://first:1"
    finally:
        server_context.reset_server_context()
