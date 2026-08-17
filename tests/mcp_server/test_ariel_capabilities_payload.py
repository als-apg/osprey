"""The ARIEL ``capabilities()`` payload advertises only live configuration.

``reasoning`` and ``default_max_results`` were retired with the ReAct/RAG
pipelines. Neither governed anything, so echoing them told the agent about
capabilities the service does not have.
"""

import json

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context, reset_ariel_context
from osprey.registry import get_registry
from osprey.utils.workspace import reset_config_cache
from tests.mcp_server.conftest import get_tool_fn


@pytest.fixture
def capabilities_payload(tmp_path, monkeypatch):
    """Return the ``capabilities()`` payload for a config that sets the dead keys."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        json.dumps(
            {
                "ariel": {
                    "database": {"uri": "postgresql://localhost/test"},
                    "search_modules": {"keyword": {"enabled": True}},
                    "reasoning": {"provider": "openai", "max_iterations": 5},
                    "default_max_results": 15,
                }
            }
        )
    )
    # ``search_modes`` is now derived from the framework registry, so it has to
    # be initialized for the payload to advertise anything.
    get_registry().initialize()
    initialize_ariel_context()
    try:
        from osprey.mcp_server.ariel.tools.capabilities import capabilities

        yield get_tool_fn(capabilities)
    finally:
        reset_ariel_context()
        reset_config_cache()


@pytest.mark.unit
async def test_capabilities_omits_retired_keys(capabilities_payload):
    """Neither retired key reaches the agent, even when the config declares it."""
    data = json.loads(await capabilities_payload())

    assert not data.get("error", False)
    assert "reasoning" not in data
    assert "default_max_results" not in data


@pytest.mark.unit
async def test_capabilities_still_reports_live_config(capabilities_payload):
    """The live half of the payload is unaffected by the removal."""
    data = json.loads(await capabilities_payload())

    assert data["enabled_search_modules"] == ["keyword"]
    assert data["search_modes"] == ["keyword"]
    assert data["embedding"]["provider"] == "ollama"
