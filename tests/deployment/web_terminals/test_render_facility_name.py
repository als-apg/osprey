"""The web-terminal render names the facility from its ``facility_name`` parameter.

The landing page's title and the sign-in page's ``OSPREY_WEB_APP_NAME`` both
show the facility's name. The render opens no file, so the name arrives as a
parameter: the deploy fills it from the build's facility identity through
:func:`resolve_render_inputs`, and a config's own ``facility`` block names
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.app import ENV_WEB_APP_NAME


def _config(**extra: object) -> dict:
    """A minimal multi-user config the render accepts."""
    return {
        "facility": {"prefix": "dls"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice"],
                "auth": {"method": "password", "allow_insecure_http": True},
            }
        },
        **extra,
    }


def _landing(artifacts: dict[str, str]) -> str:
    return artifacts["nginx/landing.html"]


def _sidecar_env(artifacts: dict[str, str]) -> dict[str, str]:
    overlay = yaml.safe_load(artifacts["docker-compose.web.yml"])
    lines = overlay["services"]["auth"]["environment"]
    return dict(line.split("=", 1) for line in lines)


def _write_identity(render_root: Path, identity: dict) -> None:
    render_root.mkdir(parents=True, exist_ok=True)
    document = {"schema": "osprey.facility.facility/1", "identity": identity}
    (render_root / "facility.json").write_text(json.dumps(document), encoding="utf-8")


def test_the_landing_title_is_the_facility_name_parameter():
    landing = _landing(render_web_terminals(_config(), facility_name="Demo Light Source"))

    assert "<title>Demo Light Source Web Terminals</title>" in landing


def test_the_sign_in_page_name_is_the_facility_name_parameter():
    env = _sidecar_env(render_web_terminals(_config(), facility_name="Demo Light Source"))

    assert env[ENV_WEB_APP_NAME] == "Demo Light Source"


def test_without_a_facility_name_the_landing_keeps_its_own_title():
    artifacts = render_web_terminals(_config())

    assert "<title>OSPREY Web Terminals</title>" in _landing(artifacts)
    assert ENV_WEB_APP_NAME not in _sidecar_env(artifacts)


@pytest.mark.parametrize(
    "extra",
    [
        {"facility": {"prefix": "dls", "name": "Config Light Source"}},
        {"facility_name": "Config Light Source"},
    ],
    ids=["facility.name", "top-level facility_name"],
)
def test_a_config_spelling_of_the_name_names_nothing(extra):
    artifacts = render_web_terminals(_config(**extra), facility_name="Identity Light Source")

    assert "Config Light Source" not in _landing(artifacts)
    assert "<title>Identity Light Source Web Terminals</title>" in _landing(artifacts)
    assert _sidecar_env(artifacts)[ENV_WEB_APP_NAME] == "Identity Light Source"


def test_the_render_reads_no_facility_file(tmp_path, monkeypatch):
    """A config with no build/ beside it renders the name it is handed, reading no identity."""
    monkeypatch.chdir(tmp_path)

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the render read the facility identity")

    monkeypatch.setattr("osprey.utils.facility.facility_identity", _refuse)
    monkeypatch.setattr("osprey.utils.facility._recorded_identity", _refuse)

    landing = _landing(render_web_terminals(_config(), facility_name="Demo Light Source"))

    assert not (tmp_path / "build").exists()
    assert "<title>Demo Light Source Web Terminals</title>" in landing


def test_the_deploy_hands_the_render_the_builds_identity_name(tmp_path):
    _write_identity(tmp_path / "build", {"code": "dls", "name": "Demo Light Source"})

    inputs = resolve_render_inputs(_config(project_name="demo-project"), tmp_path)

    assert inputs["facility_name"] == "Demo Light Source"
    landing = _landing(render_web_terminals(_config(), **inputs))
    assert "<title>Demo Light Source Web Terminals</title>" in landing


def test_a_build_identity_without_a_name_hands_the_project_name(tmp_path):
    _write_identity(tmp_path / "build", {"code": "dls"})

    inputs = resolve_render_inputs(_config(project_name="demo-project"), tmp_path)

    assert inputs["facility_name"] == "demo-project"


def test_a_repo_without_a_build_hands_the_project_name(tmp_path):
    inputs = resolve_render_inputs(_config(project_name="demo-project"), tmp_path)

    assert inputs["facility_name"] == "demo-project"


def test_a_repo_without_a_build_or_project_name_hands_no_name(tmp_path):
    inputs = resolve_render_inputs(_config(), tmp_path)

    assert inputs["facility_name"] == ""
