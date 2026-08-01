"""Tests for the LLM-naming and error-recovery branches of ``build_database``.

Covers the three arms that the plain (no-LLM) path never reaches: successful
LLM naming, degradation to addresses when the namer cannot be constructed, and
the per-family fallthrough that turns a template-creation failure into
standalone entries instead of aborting the whole build.

``build_database`` imports ``create_namer_from_config`` *inside* the function
body, so the name is never bound on ``build_database``'s module globals.
Patching must therefore target the owning module,
``osprey.services.channel_finder.tools.llm_channel_namer``. Each fake records
that it was called so an inert patch fails the test rather than passing
silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.tools.build_database import build_database

NAMER_FACTORY = "osprey.services.channel_finder.tools.llm_channel_namer.create_namer_from_config"


class _FakeNamer:
    """Stand-in for ``LLMChannelNamer`` — no network, no API key."""

    def __init__(self, names: list[str]) -> None:
        self.model_id = "fake/model-1"
        self.batch_size = 7
        self._names = names
        self.received: list[dict] = []

    def generate_names(self, channels: list[dict]) -> list[str]:
        self.received = list(channels)
        return list(self._names)


class _RecordingFactory:
    """Callable that records its invocation so an inert patch is detectable."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[object] = []

    def __call__(self, config_path=None):
        self.calls.append(config_path)
        if self._error is not None:
            raise self._error
        return self._result


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    csv_path = tmp_path / "channels.csv"
    header = "address,description,family_name,instances,sub_channel"
    csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return csv_path


def _standalone(db: dict) -> list[dict]:
    return [entry for entry in db["channels"] if not entry["template"]]


def _templates(db: dict) -> list[dict]:
    return [entry for entry in db["channels"] if entry["template"]]


def test_llm_naming_success_applies_generated_names(tmp_path, monkeypatch):
    """Generated names land on the standalone entries and in the metadata."""
    csv_path = _write_csv(
        tmp_path,
        [
            "SR:BEND:1:CURRENT,Main bend magnet current,,,",
            "SR:RF:CAVITY:VOLTAGE,RF cavity gap voltage,,,",
        ],
    )
    output_path = tmp_path / "out" / "db.json"
    namer = _FakeNamer(["BendMagnetCurrent", "RfCavityVoltage"])
    factory = _RecordingFactory(result=namer)
    monkeypatch.setattr(NAMER_FACTORY, factory)

    config_path = tmp_path / "config.yml"
    db = build_database(csv_path, output_path, use_llm=True, config_path=config_path)

    # The patch was genuinely exercised, with the caller's config path.
    assert factory.calls == [config_path]

    entries = _standalone(db)
    assert [e["channel"] for e in entries] == ["BendMagnetCurrent", "RfCavityVoltage"]
    assert [e["address"] for e in entries] == [
        "SR:BEND:1:CURRENT",
        "SR:RF:CAVITY:VOLTAGE",
    ]

    # The namer saw the address/description pairs, not raw CSV rows.
    assert namer.received == [
        {"short_name": "SR:BEND:1:CURRENT", "description": "Main bend magnet current"},
        {
            "short_name": "SR:RF:CAVITY:VOLTAGE",
            "description": "RF cavity gap voltage",
        },
    ]

    llm_meta = db["_metadata"]["llm_naming"]
    assert llm_meta["enabled"] is True
    assert llm_meta["model"] == "fake/model-1"
    assert "2 standalone channels" in llm_meta["purpose"]

    # The same content reaches disk.
    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == db


def test_llm_naming_failure_degrades_to_addresses(tmp_path, monkeypatch):
    """A namer that cannot be constructed degrades instead of propagating."""
    csv_path = _write_csv(
        tmp_path,
        [
            "SR:BEND:1:CURRENT,Main bend magnet current,,,",
            "SR:RF:CAVITY:VOLTAGE,RF cavity gap voltage,,,",
        ],
    )
    output_path = tmp_path / "db.json"
    factory = _RecordingFactory(error=RuntimeError("no provider configured"))
    monkeypatch.setattr(NAMER_FACTORY, factory)

    db = build_database(csv_path, output_path, use_llm=True)

    assert factory.calls == [None]

    entries = _standalone(db)
    assert [e["channel"] for e in entries] == [
        "SR:BEND:1:CURRENT",
        "SR:RF:CAVITY:VOLTAGE",
    ]
    assert [e["channel"] for e in entries] == [e["address"] for e in entries]

    # Degraded run still reports that LLM naming was requested, but no model.
    llm_meta = db["_metadata"]["llm_naming"]
    assert llm_meta["enabled"] is True
    assert llm_meta["model"] is None
    assert llm_meta["purpose"] is None

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == db


def test_template_failure_falls_through_to_standalone(tmp_path):
    """A family whose template cannot be built becomes standalone entries."""
    csv_path = _write_csv(
        tmp_path,
        [
            # instances="abc" makes create_template's int() raise ValueError.
            "BPM01:X,BPM horizontal position,BPM,abc,:X",
            "BPM01:Y,BPM vertical position,BPM,abc,:Y",
            # A well-formed family must still make it into the database.
            "COR01:SP,Corrector setpoint,COR,4,:SP",
            "COR01:RB,Corrector readback,COR,4,:RB",
            "SR:TEMP,Ring temperature,,,",
        ],
    )
    output_path = tmp_path / "db.json"

    db = build_database(csv_path, output_path)

    templates = _templates(db)
    assert [t["base_name"] for t in templates] == ["COR"]

    # The two BPM rows fell through to standalone, appended after the genuine
    # standalone row, and are named by address.
    entries = _standalone(db)
    assert [e["address"] for e in entries] == ["SR:TEMP", "BPM01:X", "BPM01:Y"]
    assert [e["channel"] for e in entries] == ["SR:TEMP", "BPM01:X", "BPM01:Y"]
    assert [e["description"] for e in entries] == [
        "Ring temperature",
        "BPM horizontal position",
        "BPM vertical position",
    ]

    stats = db["_metadata"]["stats"]
    assert stats == {
        "template_entries": 1,
        "standalone_entries": 3,
        "total_entries": 4,
    }

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == db


def test_patch_target_is_the_owning_module():
    """Guard: the factory is not bound on build_database's module globals.

    If it ever becomes a module-level import there, this test fails and the
    patch target above must be revisited.
    """
    from osprey.services.channel_finder.tools import build_database as mod

    assert not hasattr(mod, "create_namer_from_config")


@pytest.mark.parametrize("use_llm", [True, False])
def test_no_standalone_channels_skips_naming_entirely(tmp_path, monkeypatch, use_llm):
    """With no standalone rows the namer is never constructed."""
    csv_path = _write_csv(
        tmp_path,
        [
            "COR01:SP,Corrector setpoint,COR,2,:SP",
            "COR01:RB,Corrector readback,COR,2,:RB",
        ],
    )
    factory = _RecordingFactory(error=AssertionError("must not be called"))
    monkeypatch.setattr(NAMER_FACTORY, factory)

    db = build_database(csv_path, tmp_path / "db.json", use_llm=use_llm)

    assert factory.calls == []
    assert _standalone(db) == []
    assert [t["base_name"] for t in _templates(db)] == ["COR"]
