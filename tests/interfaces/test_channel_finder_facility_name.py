"""Every facility-name reader answers from one source.

`app.state.facility_name` labels the review UI (it is the fallback consumed by
``pending_review_api``), and the pipeline server contexts feed the per-pipeline
label. Both read the identity of the render their config sits in: the facility
file's name, else the project name, else each site's own default.

The web-terminal landing title reads the project config, under both spellings
of the key.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

# Distinct from any facility name under test, so a crossed wire is visible: this
# is the per-pipeline name the registry reports into `app.state.facility_names`
# (plural), which is a different attribute than the config-derived singular.
_REGISTRY_FACILITY = "registry-reported"


def _base_config() -> dict:
    return {
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {
                "in_context": {
                    "database": {"path": "/tmp/test_db.json", "type": "flat"},
                    # Pipeline-local override: lets the in-context context finish
                    # initializing without a provider/model map to resolve against.
                    "subagent_model": "demo-model",
                    "subagent_provider": "demo-provider",
                },
            },
        },
    }


def _launch(config: dict):
    """Run the app's lifespan against `config` and return the started app."""
    mock_reg = MagicMock()
    mock_reg.database = MagicMock()
    mock_reg.facility_name = _REGISTRY_FACILITY

    with (
        patch("osprey.utils.workspace.load_osprey_config", return_value=config),
        patch(
            "osprey.mcp_server.channel_finder_in_context.server_context.initialize_cf_ic_context",
            return_value=mock_reg,
        ),
    ):
        from osprey.interfaces.channel_finder.app import create_app

        app = create_app(project_cwd="/tmp/test-project")
        with TestClient(app):
            return app


def _write_facility_file(render_root, name: str) -> None:
    (render_root / "facility.json").write_text(
        json.dumps({"identity": {"code": "demo", "name": name}}), encoding="utf-8"
    )


# (config keys, the facility file's name or None for no file). Every reader
# reports the same name for a case; the sites differ only in their default.
_IDENTITY_CASES = [
    pytest.param({"project_name": "demo-project"}, "Demo Light Source", id="facility file"),
    pytest.param({"project_name": "demo-project"}, None, id="project name"),
    pytest.param({}, None, id="neither"),
    pytest.param({"facility": {"name": "Config Light Source"}}, None, id="config block unread"),
]


def _expected(config_keys: dict, file_name: str | None, default: str) -> str:
    return file_name or config_keys.get("project_name") or default


@pytest.mark.parametrize(("config_keys", "file_name"), _IDENTITY_CASES)
def test_app_state_facility_name_resolution(tmp_path, monkeypatch, config_keys, file_name):
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    if file_name is not None:
        _write_facility_file(tmp_path, file_name)

    app = _launch({**_base_config(), **config_keys})

    assert app.state.facility_name == _expected(config_keys, file_name, "")


def test_identity_name_is_not_the_registry_reported_name(tmp_path, monkeypatch):
    """`facility_name` (singular) is the identity's; `facility_names` the registries'."""
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    _write_facility_file(tmp_path, "Demo Light Source")

    app = _launch(_base_config())

    assert app.state.facility_name == "Demo Light Source"
    assert app.state.facility_names["in_context"] == _REGISTRY_FACILITY


# ---------------------------------------------------------------------------
# The pipeline server contexts
# ---------------------------------------------------------------------------

# (module, initializer, resetter) for the three pipeline server contexts. They
# each fed `app.state.facility_names` and the CF UI's per-pipeline label.
#
# The `graph` paradigm is deliberately absent: `facility_names` is read out of a
# loaded channel *database* registry, and a graph project has none — its store
# is seeded from the facility corpus TTL. A graph-mode app reports its facility
# name from the identity alone (the `facility_name` reader above).
_PIPELINE_CONTEXTS = [
    (
        "osprey.mcp_server.channel_finder_hierarchical.server_context",
        "initialize_cf_hier_context",
        "reset_cf_hier_context",
    ),
    (
        "osprey.mcp_server.channel_finder_middle_layer.server_context",
        "initialize_cf_ml_context",
        "reset_cf_ml_context",
    ),
    (
        "osprey.mcp_server.channel_finder_in_context.server_context",
        "initialize_cf_ic_context",
        "reset_cf_ic_context",
    ),
]

# The default each context keeps when the render names no identity.
_CONTEXT_DEFAULT = "control system"


def _write_config(tmp_path, config: dict, monkeypatch) -> None:
    """Point the pipeline config loader at a real config.yml on disk.

    The in-context pipeline loads its flat database eagerly and raises on a
    missing file, so a real (single-channel) database is written alongside and
    the config is repointed at it.
    """
    db_path = tmp_path / "channels.json"
    db_path.write_text(
        json.dumps([{"channel": "DEMO:CHAN:01", "description": "demo"}]), encoding="utf-8"
    )
    config["channel_finder"]["pipelines"]["in_context"]["database"]["path"] = str(db_path)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))


@pytest.mark.parametrize(("module_name", "init_name", "reset_name"), _PIPELINE_CONTEXTS)
@pytest.mark.parametrize(("config_keys", "file_name"), _IDENTITY_CASES)
def test_pipeline_context_facility_name(
    tmp_path, monkeypatch, module_name, init_name, reset_name, config_keys, file_name
):
    """Each pipeline server context reads the render's identity, keeping its own default."""
    _write_config(tmp_path, {**_base_config(), **config_keys}, monkeypatch)
    if file_name is not None:
        _write_facility_file(tmp_path, file_name)

    module = importlib.import_module(module_name)
    reset = getattr(module, reset_name)
    reset()
    try:
        registry = getattr(module, init_name)()
        assert registry.facility_name == _expected(config_keys, file_name, _CONTEXT_DEFAULT)
    finally:
        reset()


# ---------------------------------------------------------------------------
# Web-terminal landing title
# ---------------------------------------------------------------------------


def _web_terminals_config(facility_block: dict) -> dict:
    """Minimal multi-user config the landing render accepts."""
    return {
        **facility_block,
        "registry": {"url": "git.example.org:5050/physics/demo-profiles"},
        "deploy": {"host": "demo-deploy", "fqdn": "demo-deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice"],
            }
        },
    }


@pytest.mark.parametrize(
    ("facility_block", "expected"),
    [
        (
            {"facility": {"name": "Canonical Light Source", "prefix": "cls"}},
            "Canonical Light Source",
        ),
        (
            {"facility": {"prefix": "lls"}, "facility_name": "Legacy Light Source"},
            "Legacy Light Source",
        ),
        (
            {
                "facility": {"name": "Canonical Light Source", "prefix": "cls"},
                "facility_name": "Legacy Light Source",
            },
            "Canonical Light Source",
        ),
    ],
    ids=["facility.name", "legacy facility_name", "canonical wins over legacy"],
)
def test_landing_title_facility_name(facility_block, expected):
    """The landing page title resolves both spellings."""
    from osprey.deployment.web_terminals.render import render_web_terminals

    landing = render_web_terminals(_web_terminals_config(facility_block))["nginx/landing.html"]
    assert f"<title>{expected} Web Terminals</title>" in landing


def test_landing_title_without_any_facility_name_keeps_its_own_default():
    """Neither spelling set: the template's own OSPREY fallback, not a blank title."""
    from osprey.deployment.web_terminals.render import render_web_terminals

    landing = render_web_terminals(_web_terminals_config({"facility": {"prefix": "dls"}}))[
        "nginx/landing.html"
    ]
    assert "<title>OSPREY Web Terminals</title>" in landing
