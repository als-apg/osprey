"""``osprey ariel status`` module health: a provider's first import never runs in a check's window.

Registry init imports no provider adapter, so the first lookup of a provider in
the ``status`` process imports it -- seconds of CPU for a real adapter and
LiteLLM. Each module's ``health_check()`` is awaited under a timeout; the
providers are resolved while the modules are built, all before the first check
starts, so the timeout bounds only the check itself.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.ariel_search import cli_operations
from osprey.services.ariel_search.config import ARIELConfig
from tests.services.ariel_search.slow_provider import (
    CHAT_PROBE,
    TEXT_EMBEDDING,
    SlowProviders,
    models_listing,
)

#: The check window the tests use, and how long a slow provider takes to import.
WINDOW_S = 1.0
IMPORT_S = 1.5


@pytest.fixture
def slow(tmp_path, monkeypatch) -> SlowProviders:
    monkeypatch.setattr(cli_operations, "_STATUS_HEALTH_TIMEOUT_S", WINDOW_S)
    return SlowProviders(tmp_path, monkeypatch)


def _config(base_url: str, **modules: dict[str, Any]) -> ARIELConfig:
    blocks: dict[str, Any] = {
        "semantic_processor": {
            "enabled": True,
            "provider": "slow-chat",
            "model": {"model_id": "m", "base_url": base_url},
        },
        **modules,
    }
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://unused@127.0.0.1:1/unused"},
            "enhancement_modules": blocks,
        }
    )


async def test_a_reachable_route_whose_adapter_imports_slowly_reads_reachable(slow):
    slow.register("slow-chat", CHAT_PROBE, sleep_s=IMPORT_S)

    with models_listing("m") as base_url:
        health = await cli_operations._modules_health(
            _config(base_url), ["semantic_processor"], None
        )

    assert health["semantic_processor"]["reachable"] is True, health


async def test_one_modules_import_does_not_run_inside_another_modules_window(slow):
    # semantic_processor's own provider is already imported; text_embedding's
    # imports slowly. Built between the checks, that import would block the
    # event loop while semantic_processor's window runs.
    slow.register("slow-chat", CHAT_PROBE, sleep_s=0)
    slow.registry.get_provider("slow-chat")
    slow.register("slow-embed", TEXT_EMBEDDING, sleep_s=IMPORT_S)
    text_embedding = {
        "enabled": True,
        "provider": "slow-embed",
        "models": [{"name": "nomic-embed-text", "dimension": 768}],
    }

    with models_listing("m") as base_url:
        health = await cli_operations._modules_health(
            _config(base_url, text_embedding=text_embedding),
            ["semantic_processor", "text_embedding"],
            None,
        )

    assert health["semantic_processor"]["reachable"] is True, health
    assert health["text_embedding"]["reachable"] is True, health
