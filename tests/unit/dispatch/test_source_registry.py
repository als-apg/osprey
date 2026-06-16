"""Unit tests for the trigger SourceRegistry.

Entry-point discovery is mocked here because the real entry points are
registered in a later migration task (T06).
"""

from __future__ import annotations

import pytest

from osprey.dispatch.source_registry import SourceRegistry
from osprey.dispatch.sources.cron import CronSource
from osprey.dispatch.sources.webhook import WebhookSource
from osprey.dispatch.trigger_config import TriggerConfig


class _FakeEntryPoint:
    """Minimal EntryPoint stand-in exposing .name and .load()."""

    def __init__(self, name: str, cls: type) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type:
        return self._cls


def _patch_entry_points(monkeypatch, eps):
    def fake_entry_points(*, group):
        assert group == "osprey.trigger_sources"
        return list(eps)

    monkeypatch.setattr("osprey.dispatch.source_registry.entry_points", fake_entry_points)


def _make_trigger(name: str, source: str) -> TriggerConfig:
    return TriggerConfig(
        name=name,
        source=source,
        action={"prompt": "x", "allowed_tools": []},
    )


def test_discover_populates_both_sources(monkeypatch):
    eps = [
        _FakeEntryPoint("webhook", WebhookSource),
        _FakeEntryPoint("cron", CronSource),
    ]
    _patch_entry_points(monkeypatch, eps)

    reg = SourceRegistry()
    reg.discover()

    assert reg._source_classes == {"webhook": WebhookSource, "cron": CronSource}


def test_discover_conflicting_class_raises(monkeypatch):
    class OtherWebhook:
        source_type = "webhook"

        def register_routes(self, mcp_app) -> None: ...

        async def start(self, triggers, fire_callback) -> None: ...

        async def stop(self) -> None: ...

    reg = SourceRegistry()
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("webhook", WebhookSource)])
    reg.discover()

    # Second discovery maps the same name to a different class -> ValueError.
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("webhook", OtherWebhook)])
    with pytest.raises(ValueError, match="Conflicting trigger source for 'webhook'"):
        reg.discover()


def test_discover_rejects_non_source_class(monkeypatch):
    """An entry point that doesn't implement register_routes/start/stop fails fast."""

    class NotASource:
        source_type = "bogus"  # has the attr but no lifecycle methods

    reg = SourceRegistry()
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bogus", NotASource)])
    with pytest.raises(TypeError, match="not a TriggerSource"):
        reg.discover()


def test_discover_idempotent(monkeypatch):
    eps = [_FakeEntryPoint("webhook", WebhookSource)]
    _patch_entry_points(monkeypatch, eps)

    reg = SourceRegistry()
    reg.discover()
    reg.discover()  # same name -> same class, must not raise or duplicate

    assert reg._source_classes == {"webhook": WebhookSource}


class _FakeSource:
    """Records register_routes/start/stop calls. Instantiated by setup()."""

    source_type = "fake"

    def __init__(self) -> None:
        self.started_with: list[TriggerConfig] | None = None
        self.start_count = 0
        self.stop_count = 0
        self.routes_registered = False

    def register_routes(self, mcp_app) -> None:
        self.routes_registered = True

    async def start(self, triggers, fire_callback) -> None:
        self.started_with = list(triggers)
        self.start_count += 1

    async def stop(self) -> None:
        self.stop_count += 1


class _OtherFakeSource(_FakeSource):
    source_type = "other"


@pytest.mark.asyncio
async def test_start_all_routes_triggers_per_source_and_skips_unregistered():
    reg = SourceRegistry()
    reg._source_classes = {"fake": _FakeSource, "other": _OtherFakeSource}

    fake_a = _make_trigger("a", "fake")
    fake_b = _make_trigger("b", "fake")
    other_c = _make_trigger("c", "other")
    orphan = _make_trigger("d", "unregistered")

    async def fire_callback(trigger, payload):  # pragma: no cover - not invoked here
        return "x"

    # Factory phase: instantiate + register routes; orphan skipped.
    reg.setup([fake_a, fake_b, other_c, orphan], mcp_app=object())

    # Exactly two instances set up (fake + other); orphan was skipped.
    assert len(reg._active) == 2
    assert all(inst.routes_registered for inst, _group in reg._active)

    fake_inst, fake_group = next(
        (inst, group)
        for inst, group in reg._active
        if isinstance(inst, _FakeSource) and not isinstance(inst, _OtherFakeSource)
    )
    other_inst, other_group = next(
        (inst, group) for inst, group in reg._active if isinstance(inst, _OtherFakeSource)
    )

    # Per-source grouping is recorded in _active.
    assert fake_group == [fake_a, fake_b]
    assert other_group == [other_c]

    # Lifespan phase: start_all() starts each source with its own triggers.
    await reg.start_all(fire_callback)

    assert fake_inst.started_with == [fake_a, fake_b]
    assert other_inst.started_with == [other_c]


@pytest.mark.asyncio
async def test_stop_all_stops_every_instance():
    reg = SourceRegistry()
    reg._source_classes = {"fake": _FakeSource, "other": _OtherFakeSource}

    async def fire_callback(trigger, payload):  # pragma: no cover
        return "x"

    reg.setup([_make_trigger("a", "fake"), _make_trigger("b", "other")], mcp_app=object())
    await reg.start_all(fire_callback)
    started = [inst for inst, _group in reg._active]
    assert len(started) == 2

    await reg.stop_all()

    assert all(inst.stop_count == 1 for inst in started)
    assert reg._active == []
