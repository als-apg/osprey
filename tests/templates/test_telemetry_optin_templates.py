"""Local OpenObserve telemetry is ON BY DEFAULT in every OSPREY config template.

The framework's stance: the underlying harness already records every prompt,
response, and API body to disk, so shipping a co-deployed, loopback-bound
OpenObserve store adds no new exposure — it just makes that data queryable. So
every template that a project can be scaffolded from declares the ``openobserve``
service, lists it in ``deployed_services``, and ships a live
``claude_code.telemetry`` block (``enabled: true``) with full content capture.

This is a regression guard against any template drifting back to the old opt-in
posture (service declared but telemetry commented/off — a store deployed empty,
or a telemetry block emitting to a service that was never deployed).
"""

from __future__ import annotations

import yaml

from osprey.cli.templates.manager import TemplateManager

# Context sufficient to render every app + project template (extra keys are
# harmless; missing keys render empty and the `{% if enable_* %}` pipeline
# blocks stay off).
_CTX = {
    "project_name": "demo",
    "facility_name": "Demo",
    "default_provider": "anthropic",
    "default_model": "haiku",
    "current_python_env": "/usr/bin/python3",
    "selected_web_panels": [],
    "builtin_panels": [],
    "channel_finder_mode": "in_context",
    "default_pipeline": "in_context",
    "enable_in_context": False,
    "enable_hierarchical": False,
    "enable_middle_layer": False,
    "system": {"timezone": "UTC"},
    "osprey_labels": {"project_name": "demo", "project_root": "/tmp/demo", "deployed_at": "now"},
}

_PROJECT = "project/config.yml.j2"
_HELLO = "apps/hello_world/config.yml.j2"
_ARIEL = "apps/ariel_standalone/config.yml.j2"
_CONTROL = "apps/control_assistant/config.yml.j2"
_ALL = [_PROJECT, _HELLO, _ARIEL, _CONTROL]

_CONTENT_GATES = (
    "log_user_prompts",
    "log_assistant_responses",
    "log_tool_details",
    "log_raw_api_bodies",
)


def _render(path: str) -> str:
    return TemplateManager().jinja_env.get_template(path).render(**_CTX)


def _source(path: str) -> str:
    return (TemplateManager().template_root / path).read_text()


def _cfg(path: str) -> dict:
    return yaml.safe_load(_render(path))


def test_openobserve_declared_and_deployed_everywhere():
    """Every template declares the service AND lists it in deployed_services."""
    for path in _ALL:
        cfg = _cfg(path)
        services = cfg.get("services") or {}
        assert "openobserve" in services, f"{path}: openobserve service missing"
        deployed = cfg.get("deployed_services") or []
        assert "openobserve" in deployed, (
            f"{path}: openobserve not in deployed_services={deployed!r}"
        )


def test_telemetry_live_openobserve_everywhere():
    """Every template ships a live telemetry block pointed at openobserve."""
    for path in _ALL:
        telemetry = (_cfg(path).get("claude_code") or {}).get("telemetry")
        assert telemetry is not None, f"{path}: telemetry block missing"
        assert telemetry.get("enabled") is True, (
            f"{path}: telemetry not enabled, got {telemetry.get('enabled')!r}"
        )
        assert telemetry.get("backend") == "openobserve", f"{path}: backend != openobserve"


def test_full_content_capture_is_the_default_posture():
    """The local store is loopback-bound, so all content gates default ON."""
    for path in _ALL:
        telemetry = (_cfg(path).get("claude_code") or {}).get("telemetry") or {}
        for gate in _CONTENT_GATES:
            assert telemetry.get(gate) is True, (
                f"{path}: content gate {gate} not on (got {telemetry.get(gate)!r})"
            )


def test_retention_bound_declared_everywhere():
    """Growth is bounded by age since a named volume has no size cap."""
    for path in _ALL:
        oo = (_cfg(path).get("services") or {}).get("openobserve") or {}
        assert oo.get("retention_days"), f"{path}: openobserve.retention_days missing"


def test_no_template_regresses_to_opt_in_wording():
    """Guard against the old 'declared but stays OFF until you opt in' posture."""
    for path in _ALL:
        src = _source(path)
        assert "stays OFF until you opt in" not in src, f"{path}: opt-in wording resurfaced"
        # The endpoint auto-derives; hardcoding localhost silently drops the
        # in-container worker's telemetry, so no template pins it.
        assert "endpoint: http://localhost" not in src, f"{path}: hardcoded localhost endpoint"
