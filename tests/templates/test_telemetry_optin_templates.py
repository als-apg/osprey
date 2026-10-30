"""The agent record is ON BY DEFAULT in every preset that deploys a stack.

The framework's stance: the underlying harness already records every prompt,
response, and API body to disk, so keeping one queryable record of it beside
the agent adds no new exposure. So every bundled preset that deploys a stack
resolves to a top-level ``agent_record`` block naming its backend and its
retention, and nothing a user has to switch on first.

The record is the one place that data lives, so a preset states no
``claude_code.telemetry`` block and none of the services the record replaced
(:data:`~osprey.agent_record.config.RETIRED_DEPLOYED_SERVICES`), neither
declared nor in ``deployed_services``: no preset opts out of the record or
keeps a second store beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import osprey.profiles
from osprey.agent_record.config import BACKENDS, RETIRED_DEPLOYED_SERVICES
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile

#: The bundled presets that deploy a stack, and so must keep a record.
_DEPLOYING_PRESETS = ("hello-world", "control-assistant")

#: Service names a second store beside the record would take.
_RETIRED_SERVICES = tuple(sorted(RETIRED_DEPLOYED_SERVICES))


def _resolved(preset: str):
    profile, _profile_dir = resolve_build_profile(None, preset)
    return profile


def _cfg(preset: str) -> dict[str, Any]:
    """The preset's resolved ``config:`` block, nested."""
    return _expand_dotted(_resolved(preset).config)


def _source(preset: str) -> str:
    return (Path(osprey.profiles.__file__).parent / "presets" / f"{preset}.yml").read_text(
        encoding="utf-8"
    )


def test_record_declared_everywhere():
    """Every deploying preset states its record's backend in its own document."""
    for preset in _DEPLOYING_PRESETS:
        record = _resolved(preset).agent_record
        assert record is not None, f"{preset}: agent_record missing"
        assert record.backend in BACKENDS, f"{preset}: unknown backend {record.backend!r}"
        assert "\nagent_record:\n  backend: " in _source(preset), (
            f"{preset}: the backend is not stated in the preset, so a default chose it"
        )


def test_retention_bound_declared_everywhere():
    """Growth is bounded by age since a named volume has no size cap."""
    for preset in _DEPLOYING_PRESETS:
        retention = _resolved(preset).agent_record.retention_days
        assert isinstance(retention, int) and retention > 0, (
            f"{preset}: agent_record.retention_days is {retention!r}"
        )


def test_no_retired_store_is_declared_or_deployed():
    """No preset declares or deploys a store beside the record."""
    for preset in _DEPLOYING_PRESETS:
        cfg = _cfg(preset)
        services = cfg.get("services") or {}
        deployed = cfg.get("deployed_services") or []
        for service in _RETIRED_SERVICES:
            assert service not in services, f"{preset}: declares services.{service}"
            assert service not in deployed, f"{preset}: deploys {service}"


def test_no_telemetry_block_feeds_a_second_store():
    """No preset ships a ``claude_code.telemetry`` block or the tool-call audit."""
    for preset in _DEPLOYING_PRESETS:
        cfg = _cfg(preset)
        assert "telemetry" not in (cfg.get("claude_code") or {}), (
            f"{preset}: claude_code.telemetry is set"
        )
        assert "tool_call" not in (cfg.get("audit") or {}), f"{preset}: audit.tool_call is set"


def test_no_preset_regresses_to_opt_in_wording():
    """Guard against the old 'declared but stays OFF until you opt in' posture."""
    for preset in _DEPLOYING_PRESETS:
        src = _source(preset)
        assert "stays OFF until you opt in" not in src, f"{preset}: opt-in wording resurfaced"
        # The endpoint auto-derives; hardcoding localhost silently drops the
        # in-container worker's entries, so no preset pins it.
        assert "endpoint: http://localhost" not in src, f"{preset}: hardcoded localhost endpoint"
