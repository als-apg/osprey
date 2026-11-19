"""ARIEL enhancement modules' own health checks: on demand only."""

from __future__ import annotations

import asyncio

import yaml

from osprey.health.core.ariel_modules import ariel_modules
from osprey.health.models import Status
from osprey.health.records import ON_DEMAND_CORE
from osprey.models.providers.health import HealthResult


def _rows(config, *, modules=None) -> dict:
    return {row.name: row for row in asyncio.run(ariel_modules(config, modules=modules)())}


def test_the_category_is_on_demand():
    # A module's check may send a billed completion or mkdir its mirror root,
    # so the poll never runs it; a click (re-probe) or --full does.
    assert "ariel_modules" in ON_DEMAND_CORE


class _Module:
    def __init__(self, name: str, verdict=None, exc: Exception | None = None, delay: float = 0.0):
        self.name = name
        self._verdict = verdict
        self._exc = exc
        self._delay = delay

    async def health_check(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc:
            raise self._exc
        return self._verdict


_ARIEL = {"ariel": {"enhancement_modules": {"text_embedding": {"enabled": True}}}}


def test_module_health_checks():
    modules = [
        _Module("text_embedding", HealthResult(True, "ollama answered", None)),
        _Module("image_caption", HealthResult(False, "model missing", "model")),
        _Module("qmd_export", HealthResult(None, "no health check", None)),
        _Module("legacy", (True, "ok")),
        _Module("broken", exc=RuntimeError("boom")),
    ]
    rows = _rows(_ARIEL, modules=lambda _cfg: modules)

    assert rows["ariel.enhancement.text_embedding"].status is Status.OK
    assert rows["ariel.enhancement.text_embedding"].value.startswith("reachable · ")
    assert rows["ariel.enhancement.image_caption"].status is Status.WARNING
    assert rows["ariel.enhancement.image_caption"].value == "model"
    assert rows["ariel.enhancement.qmd_export"].status is Status.SKIP
    assert rows["ariel.enhancement.qmd_export"].value == "no health check"
    assert rows["ariel.enhancement.legacy"].status is Status.OK
    assert rows["ariel.enhancement.broken"].status is Status.WARNING
    assert "boom" in rows["ariel.enhancement.broken"].details


def test_a_module_factory_failure_is_one_warning_row():
    def explode(_cfg):
        raise ValueError("bad ariel config")

    rows = _rows(_ARIEL, modules=explode)
    assert rows["ariel.enhancement"].status is Status.WARNING
    assert rows["ariel.enhancement"].value == "config error"


def test_no_ariel_rows_without_enabled_modules():
    off = {"ariel": {"enhancement_modules": {"text_embedding": {"enabled": False}}}}
    assert _rows(off, modules=lambda _cfg: [_Module("x")]) == {}


def test_a_reachable_route_whose_adapter_imports_slowly_reads_ok(tmp_path, monkeypatch):
    # Registry init imports no provider adapter, so the first lookup of the
    # semantic processor's provider imports it. That happens while the module
    # is built, before its health window opens; the window holds the check.
    from osprey.health.core import ariel_modules as category
    from tests.services.ariel_search.slow_provider import (
        CHAT_PROBE,
        SlowProviders,
        models_listing,
    )

    monkeypatch.setattr(category, "_TIMEOUT_S", 1.0)
    SlowProviders(tmp_path, monkeypatch).register("slow-chat", CHAT_PROBE, sleep_s=1.5)

    with models_listing("m") as base_url:
        config = {
            "ariel": {
                "database": {"uri": "postgresql://unused@127.0.0.1:1/unused"},
                "enhancement_modules": {
                    "semantic_processor": {
                        "enabled": True,
                        "provider": "slow-chat",
                        "model": {"model_id": "m", "base_url": base_url},
                    }
                },
            }
        }
        # The module factory initializes the deployment's registry, which
        # reads the config file.
        (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        rows = _rows(config)

    row = rows["ariel.enhancement.semantic_processor"]
    assert row.status is Status.OK, row
