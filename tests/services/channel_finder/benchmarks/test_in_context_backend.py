"""Integration tests for InContextBackend — real subprocess + real LLM call."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver
from osprey.services.channel_finder.benchmarks.backends.in_context_backend import (
    InContextBackend,
)
from osprey.services.channel_finder.benchmarks.sdk import init_project


def _resolve_spec(project_dir: Path):
    """Resolve the project's ClaudeCodeModelSpec for backend construction."""
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8")) or {}
    spec = ClaudeCodeModelResolver.resolve(
        config.get("claude_code", {}),
        config.get("api", {}).get("providers", {}),
    )
    assert spec is not None, f"claude_code.provider not configured in {project_dir}"
    return spec

# ---------------------------------------------------------------------------
# Minimal test DB — 8 channels, compact enough to stay within any model's context
# ---------------------------------------------------------------------------

_TEST_CHANNELS = [
    {
        "channel": "StorageRing_Current",
        "address": "SR:BEAM:CURRENT",
        "description": "Storage ring beam current in mA",
    },
    {
        "channel": "StorageRing_Energy",
        "address": "SR:BEAM:ENERGY",
        "description": "Storage ring beam energy in GeV",
    },
    {
        "channel": "Linac_Gun_Voltage",
        "address": "LI:GUN:VOLTAGE",
        "description": "Linac electron gun cathode voltage",
    },
    {
        "channel": "Linac_Klystron_Power",
        "address": "LI:KLY:POWER",
        "description": "Linac klystron RF power output",
    },
    {
        "channel": "BL_12_Photon_Flux",
        "address": "BL:12:FLUX",
        "description": "Beamline 12 photon flux",
    },
    {
        "channel": "BL_12_Mirror_Pitch",
        "address": "BL:12:MIRROR:PITCH",
        "description": "Beamline 12 mirror pitch angle",
    },
    {
        "channel": "Vacuum_SR_Sector3",
        "address": "SR:VAC:SEC3:PRESSURE",
        "description": "Storage ring vacuum pressure sector 3",
    },
    {
        "channel": "RF_Cavity_Voltage",
        "address": "SR:RF:CAV:VOLTAGE",
        "description": "Storage ring RF cavity voltage",
    },
]

# Use CBORG (available in CI env) — anthropic-compatible proxy with CBORG_API_KEY.
# Falls back to direct anthropic if ANTHROPIC_API_KEY is set.
_CBORG_KEY = os.environ.get("CBORG_API_KEY", "")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY_o", "")

if _CBORG_KEY:
    _PROVIDER = "cborg"
    _PROVIDER_API_KEY = _CBORG_KEY
    _SUBAGENT_MODEL = "anthropic/claude-haiku"  # CBORG model name
    _PROVIDER_BASE_URL = "https://api.cborg.lbl.gov/v1"
else:
    _PROVIDER = "anthropic"
    _PROVIDER_API_KEY = _ANTHROPIC_KEY
    _SUBAGENT_MODEL = "anthropic/claude-haiku-4-5"
    _PROVIDER_BASE_URL = None

pytestmark = pytest.mark.skipif(
    not _PROVIDER_API_KEY,
    reason="No LLM provider API key available (CBORG_API_KEY or ANTHROPIC_API_KEY)",
)


def _make_test_project(tmp_path: Path, subagent_model: str = _SUBAGENT_MODEL) -> Path:
    """Scaffold an in_context project with the minimal test DB and subagent_model."""
    project_dir = init_project(
        tmp_path,
        "ic-test-proj",
        channel_finder_mode="in_context",
        provider=_PROVIDER,
        model="haiku",  # shorthand accepted by osprey init
    )

    # Write minimal flat DB
    db_path = project_dir / "test_channels.json"
    db_path.write_text(json.dumps(_TEST_CHANNELS), encoding="utf-8")

    # Patch config.yml: wire in the DB path, subagent_model, and resolved API key
    # so the subprocess reads a literal key (not an unresolved ${...} placeholder).
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # Inject literal API key so subprocess env-var interpolation succeeds
    config.setdefault("api", {}).setdefault("providers", {})
    config["api"]["providers"].setdefault(_PROVIDER, {})
    config["api"]["providers"][_PROVIDER]["api_key"] = _PROVIDER_API_KEY
    if _PROVIDER_BASE_URL:
        config["api"]["providers"][_PROVIDER]["base_url"] = _PROVIDER_BASE_URL

    # Wire claude_code provider/model for subagent_provider resolution
    config.setdefault("claude_code", {})
    config["claude_code"]["provider"] = _PROVIDER

    # Wire in_context database and subagent_model
    config.setdefault("channel_finder", {})
    config["channel_finder"].setdefault("pipelines", {})
    config["channel_finder"]["pipelines"].setdefault("in_context", {})
    ic = config["channel_finder"]["pipelines"]["in_context"]
    ic["subagent_model"] = subagent_model
    ic.setdefault("database", {})
    ic["database"]["path"] = str(db_path)
    ic["database"]["type"] = "flat"

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return project_dir


@pytest.mark.integration
async def test_in_context_backend_basic(tmp_path):
    """InContextBackend runs a real query end-to-end and returns a WorkflowOutput."""
    project_dir = _make_test_project(tmp_path)
    spec = _resolve_spec(project_dir)
    backend = InContextBackend(project_dir, spec, "haiku")

    output = await backend.run_query(
        "What is the PV address for the storage ring beam current?",
        "in_context",
    )

    # Structural assertions — always true regardless of LLM output variability
    assert output.num_turns == 1
    assert len(output.tool_traces) == 1
    assert output.tool_traces[0].name == "query_channels"

    # Content assertion — the model should identify the beam current channel
    response = output.response_text
    assert "SR:BEAM:CURRENT" in response or "StorageRing_Current" in response

    # Inner provider + model identifier are recorded in the trace; both
    # come from the resolved spec, not a free-form caller string.
    trace_input = output.tool_traces[0].input
    assert trace_input.get("_inner_provider") == spec.provider
    assert trace_input.get("_inner_model_id") == spec.tier_to_model["haiku"]


@pytest.mark.integration
async def test_in_context_backend_records_tier_wire_id(tmp_path):
    """Backend records the resolved wire id for whichever tier it was constructed with."""
    project_dir = _make_test_project(tmp_path)
    spec = _resolve_spec(project_dir)
    backend = InContextBackend(project_dir, spec, "haiku")

    out = await backend.run_query("What channels monitor RF power?", "in_context")

    # The trace's `_inner_model_id` is the bare wire id from the spec, not
    # the LiteLLM-style slug. Backends never invent their own labels.
    assert out.tool_traces[0].input["_inner_model_id"] == spec.tier_to_model["haiku"]
    assert out.num_turns == 1
