"""Every facility-name reader resolves through the same order.

`app.state.facility_name` labels the review UI (it is the fallback consumed by
``pending_review_api``). It used to read the legacy top-level ``facility_name``
only, so a project that adopted canonical ``facility.name`` got an empty string.

The pipeline server contexts and the web-terminal landing title had the mirror
defect — canonical-only, no legacy fallback — so a bundle shipping top-level
``facility_name`` (as ariel_standalone and channel_finder_standalone do) fell to
the literal ``"control system"`` / an empty title. Both directions are pinned
here so "both spellings work everywhere" stays true.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

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


@pytest.mark.parametrize(
    ("facility_block", "expected"),
    [
        ({"facility": {"name": "Canonical Light Source"}}, "Canonical Light Source"),
        ({"facility_name": "Legacy Light Source"}, "Legacy Light Source"),
        (
            {
                "facility": {"name": "Canonical Light Source"},
                "facility_name": "Legacy Light Source",
            },
            "Canonical Light Source",
        ),
        ({}, ""),
    ],
    ids=[
        "facility.name",
        "legacy facility_name",
        "canonical wins over legacy",
        "neither -> empty",
    ],
)
def test_app_state_facility_name_resolution(facility_block, expected):
    config = {**_base_config(), **facility_block}
    app = _launch(config)
    assert app.state.facility_name == expected


def test_facility_block_without_a_name_falls_back_to_the_legacy_key():
    """A `facility:` block carrying only `prefix` must not shadow the legacy key."""
    config = {**_base_config(), "facility": {"prefix": "ca"}, "facility_name": "Legacy LS"}
    assert _launch(config).state.facility_name == "Legacy LS"


def test_config_derived_name_is_not_the_registry_reported_name():
    """`facility_name` (singular) comes from config; `facility_names` from registries."""
    config = {**_base_config(), "facility": {"name": "Canonical Light Source"}}
    app = _launch(config)
    assert app.state.facility_name == "Canonical Light Source"
    assert app.state.facility_names["in_context"] == _REGISTRY_FACILITY


# ---------------------------------------------------------------------------
# The formerly canonical-only readers (task 3.5)
# ---------------------------------------------------------------------------

# (module, initializer, resetter) for the three pipeline server contexts. They
# each fed `app.state.facility_names` and the CF UI's per-pipeline label.
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

# The default each site keeps when neither spelling carries a name.
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
@pytest.mark.parametrize(
    ("facility_block", "expected"),
    [
        ({"facility": {"name": "Canonical Light Source"}}, "Canonical Light Source"),
        ({"facility_name": "Legacy Light Source"}, "Legacy Light Source"),
        (
            {
                "facility": {"name": "Canonical Light Source"},
                "facility_name": "Legacy Light Source",
            },
            "Canonical Light Source",
        ),
        ({}, _CONTEXT_DEFAULT),
    ],
    ids=[
        "facility.name",
        "legacy facility_name",
        "canonical wins over legacy",
        "neither -> control system",
    ],
)
def test_pipeline_context_facility_name(
    tmp_path, monkeypatch, module_name, init_name, reset_name, facility_block, expected
):
    """Each pipeline server context resolves both spellings, keeping its own default."""
    _write_config(tmp_path, {**_base_config(), **facility_block}, monkeypatch)

    module = importlib.import_module(module_name)
    reset = getattr(module, reset_name)
    reset()
    try:
        registry = getattr(module, init_name)()
        assert registry.facility_name == expected
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
                "nginx_port": 9080,
                "web_base_port": 9091,
                "artifact_base_port": 9291,
                "ariel_base_port": 9391,
                "lattice_base_port": 9491,
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
