"""Tests for the ``_inject_optimizer_web`` build step and the profile block behind it.

The injector mirrors ``_inject_bluesky_web`` (``test_inject_bluesky_web.py``):
it copies the compose template, writes ``services.optimizer_web`` and
registers it, and registers the ``web.panels.optimizer`` tab at the sidecar
root, an explicit override winning. The profile rows pin that the block parses
onto the layout slot and is refused without the optimizer it relays to.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_optimizer_web
from osprey.cli.build_profile import OptimizerWebConfig
from osprey.port_layout import default_port

_WEB_PORT = default_port("optimizer_web")


def _write_config(project_path: Path, *, extra: dict | None = None) -> None:
    config: dict = {"deployed_services": ["optimizer"], "services": {"optimizer": {}}}
    if extra:
        config.update(extra)
    YAML().dump(config, project_path / "config.yml")


def _read_config(project_path: Path) -> dict:
    return YAML().load(project_path / "config.yml")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


def test_the_template_dir_is_copied(project_path: Path) -> None:
    _write_config(project_path)

    _inject_optimizer_web(OptimizerWebConfig(), project_path=project_path)

    dest = project_path / "services" / "optimizer_web"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()


def test_the_service_block_is_written_and_registered_once(project_path: Path) -> None:
    _write_config(project_path)

    _inject_optimizer_web(OptimizerWebConfig(port=_WEB_PORT), project_path=project_path)
    _inject_optimizer_web(OptimizerWebConfig(port=_WEB_PORT), project_path=project_path)

    config = _read_config(project_path)
    assert dict(config["services"]["optimizer_web"]) == {
        "path": "./services/optimizer_web",
        "port": _WEB_PORT,
    }
    deployed = [str(s) for s in config["deployed_services"]]
    assert deployed == ["optimizer", "optimizer_web"]


def test_the_optimizer_panel_is_registered_at_the_sidecar_root(project_path: Path) -> None:
    _write_config(project_path)

    _inject_optimizer_web(OptimizerWebConfig(port=_WEB_PORT), project_path=project_path)

    panel = dict(_read_config(project_path)["web"]["panels"]["optimizer"])
    assert panel == {
        "url": f"${{OPTIMIZER_WEB_URL:-http://localhost:{_WEB_PORT}}}",
        "path": "/optimizer/",
        "label": "OPTIMIZER",
    }


def test_an_explicit_panel_override_wins(project_path: Path) -> None:
    _write_config(
        project_path,
        extra={"web": {"panels": {"optimizer": {"url": "http://elsewhere:9", "label": "TUNE"}}}},
    )

    _inject_optimizer_web(OptimizerWebConfig(port=_WEB_PORT), project_path=project_path)

    panel = dict(_read_config(project_path)["web"]["panels"]["optimizer"])
    assert panel["url"] == "http://elsewhere:9"
    assert panel["label"] == "TUNE"
    assert panel["path"] == "/optimizer/"


def _profile(raw: dict):
    from osprey.cli.build_profile_load import _parse_profile

    return _parse_profile({"name": "x", "data": "data", **raw})


def test_the_block_parses_onto_the_layout_slot(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    profile = _profile({"optimizer": {}, "optimizer_web": {}})

    assert profile.optimizer_web == OptimizerWebConfig(port=_WEB_PORT)
    profile.validate(tmp_path)


def test_the_block_is_refused_without_an_optimizer(tmp_path: Path) -> None:
    from osprey_connectors.errors import BuildProfileError

    (tmp_path / "data").mkdir()
    profile = _profile({"optimizer_web": {}})

    with pytest.raises(BuildProfileError, match="optimizer_web needs an optimizer"):
        profile.validate(tmp_path)


def test_an_unknown_key_in_the_block_is_refused() -> None:
    from osprey_connectors.errors import BuildProfileError

    with pytest.raises(BuildProfileError, match="prot"):
        _profile({"optimizer": {}, "optimizer_web": {"prot": 1}})
