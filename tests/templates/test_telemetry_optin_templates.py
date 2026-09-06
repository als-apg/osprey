"""Local OpenObserve telemetry is ON BY DEFAULT in every preset that deploys a stack.

The framework's stance: the underlying harness already records every prompt,
response, and API body to disk, so shipping a co-deployed, loopback-bound
OpenObserve store adds no new exposure — it just makes that data queryable. So
every bundled preset whose deployment runs services declares the ``openobserve``
service in its ``config:`` block, lists it in ``deployed_services``, and ships a
live ``claude_code.telemetry`` block (``enabled: true``) with full content
capture.

The presets are the source of that posture: the framework template renders only
derived keys, and a deployment's ``services.*`` / ``claude_code.telemetry.*``
come from its profile's ``config:`` block, which the preset writes down. So this
is a regression guard against any preset drifting back to the old opt-in posture
(service declared but telemetry commented/off — a store deployed empty, or a
telemetry block emitting to a service that was never deployed).

``channel-finder-standalone`` is not in the matrix: it deploys no services at
all (``deployed_services: []``) and ships telemetry off, so there is no store
for a live block to point at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import osprey.profiles
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile

#: The bundled presets that deploy a stack, and so must deploy the store too.
_DEPLOYING_PRESETS = ("hello-world", "ariel-standalone", "control-assistant")

_CONTENT_GATES = (
    "log_user_prompts",
    "log_assistant_responses",
    "log_tool_details",
    "log_raw_api_bodies",
)


def _cfg(preset: str) -> dict[str, Any]:
    """The preset's resolved ``config:`` block, nested."""
    profile, _profile_dir = resolve_build_profile(None, preset)
    return _expand_dotted(profile.config)


def _source(preset: str) -> str:
    return (Path(osprey.profiles.__file__).parent / "presets" / f"{preset}.yml").read_text(
        encoding="utf-8"
    )


def test_openobserve_declared_and_deployed_everywhere():
    """Every deploying preset declares the service AND lists it in deployed_services."""
    for preset in _DEPLOYING_PRESETS:
        cfg = _cfg(preset)
        services = cfg.get("services") or {}
        assert "openobserve" in services, f"{preset}: openobserve service missing"
        deployed = cfg.get("deployed_services") or []
        assert "openobserve" in deployed, (
            f"{preset}: openobserve not in deployed_services={deployed!r}"
        )


def test_telemetry_live_openobserve_everywhere():
    """Every deploying preset ships a live telemetry block pointed at openobserve."""
    for preset in _DEPLOYING_PRESETS:
        telemetry = (_cfg(preset).get("claude_code") or {}).get("telemetry")
        assert telemetry is not None, f"{preset}: telemetry block missing"
        assert telemetry.get("enabled") is True, (
            f"{preset}: telemetry not enabled, got {telemetry.get('enabled')!r}"
        )
        assert telemetry.get("backend") == "openobserve", f"{preset}: backend != openobserve"


def test_full_content_capture_is_the_default_posture():
    """The local store is loopback-bound, so all content gates default ON."""
    for preset in _DEPLOYING_PRESETS:
        telemetry = (_cfg(preset).get("claude_code") or {}).get("telemetry") or {}
        for gate in _CONTENT_GATES:
            assert telemetry.get(gate) is True, (
                f"{preset}: content gate {gate} not on (got {telemetry.get(gate)!r})"
            )


def test_retention_bound_declared_everywhere():
    """Growth is bounded by age since a named volume has no size cap."""
    for preset in _DEPLOYING_PRESETS:
        oo = (_cfg(preset).get("services") or {}).get("openobserve") or {}
        assert oo.get("retention_days"), f"{preset}: openobserve.retention_days missing"


def test_no_preset_regresses_to_opt_in_wording():
    """Guard against the old 'declared but stays OFF until you opt in' posture."""
    for preset in _DEPLOYING_PRESETS:
        src = _source(preset)
        assert "stays OFF until you opt in" not in src, f"{preset}: opt-in wording resurfaced"
        # The endpoint auto-derives; hardcoding localhost silently drops the
        # in-container worker's telemetry, so no preset pins it.
        assert "endpoint: http://localhost" not in src, f"{preset}: hardcoded localhost endpoint"
