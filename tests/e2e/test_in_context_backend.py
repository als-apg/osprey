"""Integration tests for InContextBackend — real subprocess + real LLM call.

Lives under ``tests/e2e/`` because it makes a real credentialed LLM call (the
inner ``ask_channels`` model call happens inside the spawned MCP subprocess).
Keeping it here means the fast lane — ``pytest tests/ --ignore=tests/e2e`` — is
hermetic regardless of which provider keys a developer happens to have exported;
the placement guard in ``tests/test_api_marker_placement.py`` enforces this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from osprey.services.channel_finder.benchmarks.backends.in_context_backend import (
    InContextBackend,
)
from tests.e2e.sdk_helpers import init_project, render_dir

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

# Provider preference: als-apg first (AWS Bedrock proxy — IP-unrestricted, works
# in CI and off-VPN), then CBORG (LBLnet-gated, faster locally), then anthropic
# direct. Matches the CI auth choice in commit 5d0dcd72.
_ALS_APG_KEY = os.environ.get("ALS_APG_API_KEY", "")
_CBORG_KEY = os.environ.get("CBORG_API_KEY", "")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY_o", "")

if _ALS_APG_KEY:
    _PROVIDER = "als-apg"
    _PROVIDER_API_KEY = _ALS_APG_KEY
    _SUBAGENT_MODEL = "claude-haiku-4-5-20251001"  # bare wire id; gateway rejects prefixed slugs
    # Overridable so a run can be aimed at another gateway; same convention as
    # judge.py. The value is written into the generated project's config, which
    # is what the MCP subprocess reads — it does not inherit this process's env.
    _PROVIDER_BASE_URL = os.environ.get("ALS_APG_BASE_URL") or "https://llm.gianlucamartino.com"
    _BACKEND_MODEL = "als-apg/claude-haiku-4-5-20251001"
    _EXPECTED_WIRE = "claude-haiku-4-5-20251001"
elif _CBORG_KEY:
    _PROVIDER = "cborg"
    _PROVIDER_API_KEY = _CBORG_KEY
    _SUBAGENT_MODEL = "anthropic/claude-haiku"  # CBORG model name
    _PROVIDER_BASE_URL = "https://api.cborg.lbl.gov/v1"
    _BACKEND_MODEL = "cborg/claude-haiku-4-5"
    _EXPECTED_WIRE = "claude-haiku-4-5"
else:
    _PROVIDER = "anthropic"
    _PROVIDER_API_KEY = _ANTHROPIC_KEY
    _SUBAGENT_MODEL = "anthropic/claude-haiku-4-5"
    _PROVIDER_BASE_URL = None
    _BACKEND_MODEL = "anthropic/claude-haiku-4-5-20251001"
    _EXPECTED_WIRE = "claude-haiku-4-5-20251001"

pytestmark = pytest.mark.requires_api


def _make_test_project(tmp_path: Path, subagent_model: str = _SUBAGENT_MODEL) -> Path:
    """Build a deployment repo with the minimal test DB and subagent_model.

    Returns the RENDER directory (``<repo>/build``), which is what
    :class:`InContextBackend` takes: it points the spawned MCP subprocess at
    ``<dir>/config.yml``.
    """
    repo = init_project(
        tmp_path,
        "ic-test-proj",
        channel_finder_mode="in_context",
        provider=_PROVIDER,
        model="haiku",  # shorthand accepted by osprey init
    )
    render = render_dir(repo)

    # Write minimal flat DB. It lives at the repo root, outside the render that
    # every build re-creates; the config below records its absolute path, so the
    # zone it sits in does not affect resolution.
    db_path = repo / "test_channels.json"
    db_path.write_text(json.dumps(_TEST_CHANNELS), encoding="utf-8")

    # Patch the rendered config.yml: wire in the DB path, subagent_model, and
    # resolved API key so the subprocess reads a literal key (not an unresolved
    # ${...} placeholder).
    config_path = render / "config.yml"
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

    return render


@pytest.mark.integration
async def test_in_context_backend_basic(tmp_path):
    """InContextBackend runs a real query end-to-end and returns a WorkflowOutput."""
    render = _make_test_project(tmp_path)
    backend = InContextBackend(render, _BACKEND_MODEL)

    output = await backend.run_query(
        "What is the PV address for the storage ring beam current?",
        "in_context",
    )

    # Structural assertions — always true regardless of LLM output variability
    assert output.num_turns == 1
    assert len(output.tool_traces) == 1
    assert output.tool_traces[0].name == "ask_channels"

    # Content assertion — the model should identify the beam current channel
    response = output.response_text
    assert "SR:BEAM:CURRENT" in response or "StorageRing_Current" in response

    # Inner provider + wire id are recorded in the trace from the model
    # string the backend was constructed with.
    trace_input = output.tool_traces[0].input
    assert trace_input.get("_inner_provider") == _PROVIDER
    assert trace_input.get("_inner_model_id") == _EXPECTED_WIRE


@pytest.mark.integration
async def test_in_context_backend_records_wire_id(tmp_path):
    """Backend records the wire id half of its provider/wire_id model string."""
    render = _make_test_project(tmp_path)
    backend = InContextBackend(render, _BACKEND_MODEL)

    out = await backend.run_query("What channels monitor RF power?", "in_context")

    # The trace's `_inner_model_id` is the bare wire id (the part after the
    # slash), not the LiteLLM-style slug. Backends never invent their own labels.
    assert out.tool_traces[0].input["_inner_model_id"] == _EXPECTED_WIRE
    assert out.num_turns == 1
