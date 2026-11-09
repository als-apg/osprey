"""Which optimizer the server dials, beyond what ``tests/deployment/test_reach_dials.py`` pins.

The env override winning and the published port on loopback are pinned there,
through the reach dial that calls the same :func:`optimizer_url`.
"""

from __future__ import annotations

import pytest

from osprey.optimizer_connection import DEFAULT_OPTIMIZER_PORT, optimizer_url
from osprey.port_layout import default_port


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("OPTIMIZER_URL", raising=False)


def test_an_empty_override_falls_back_to_the_config(monkeypatch):
    # The registry renders `${OPTIMIZER_URL:-}`: unset reaches the server empty.
    monkeypatch.setenv("OPTIMIZER_URL", "")
    assert optimizer_url({"services": {"optimizer": {"port": 18182}}}) == "http://127.0.0.1:18182"


def test_the_override_loses_its_trailing_slash(monkeypatch):
    monkeypatch.setenv("OPTIMIZER_URL", "http://opt.example:9/")
    assert optimizer_url({}) == "http://opt.example:9"


@pytest.mark.parametrize(
    ("config", "port"),
    [
        (None, DEFAULT_OPTIMIZER_PORT),
        ({"deployment": {"port_base": 20000}}, default_port("optimizer", base=20000)),
    ],
)
def test_no_port_is_the_layout_slot_at_this_configs_base(config, port):
    assert optimizer_url(config) == f"http://127.0.0.1:{port}"
