"""``protocol: grpc`` toward the deployment's telemetry gateway fails loud.

The gateway's receiver answers OTLP over HTTP only. A gRPC exporter aimed at it
drops every metric, log and span without an error, which is precisely the
silent-drop mode the module already refuses elsewhere (unresolvable endpoint,
leaked ``${VAR}``, missing login). These tests pin the refusal and, equally
important, pin the paths that must keep working: both HTTP protocols toward the
gateway, and ``backend: generic`` over gRPC to a collector an operator named.
"""

from __future__ import annotations

import pytest

from osprey.build.claude_code_telemetry import (
    GATEWAY_PROTOCOLS,
    TelemetryConfigError,
    _build_telemetry_env,
    _resolve_telemetry_endpoint,
)

GATEWAY = "http://telemetry_gateway:4318"


def _gateway_cfg(**overrides: object) -> dict:
    """An enabled gateway telemetry block."""
    return {"enabled": True, "backend": "gateway", **overrides}


def _gateway_env(cfg: dict) -> dict[str, str]:
    return _build_telemetry_env(cfg, gateway_url=GATEWAY, telemetry_login="alice:s3cret")


@pytest.mark.parametrize("gateway_url", [GATEWAY, "http://127.0.0.1:10051"])
def test_grpc_toward_the_gateway_raises_from_every_network_shape(gateway_url):
    """The bridge address and the published host address are HTTP-only alike."""
    with pytest.raises(TelemetryConfigError):
        _resolve_telemetry_endpoint(_gateway_cfg(protocol="grpc"), gateway_url=gateway_url)


@pytest.mark.parametrize("spelling", ["grpc", "GRPC", " gRPC "])
def test_grpc_refusal_is_case_and_whitespace_insensitive(spelling):
    """``resolve_env_vars`` can hand through any spelling a ``${VAR}`` carried."""
    with pytest.raises(TelemetryConfigError):
        _gateway_env(_gateway_cfg(protocol=spelling))


def test_grpc_refusal_message_names_the_keys():
    """The operator must be able to act on the message without reading source."""
    with pytest.raises(TelemetryConfigError) as excinfo:
        _gateway_env(_gateway_cfg(protocol="grpc"))
    message = str(excinfo.value)
    assert "claude_code.telemetry.protocol" in message
    assert "claude_code.telemetry.endpoint" in message
    assert "grpc" in message
    assert "gateway" in message


@pytest.mark.parametrize("protocol", GATEWAY_PROTOCOLS)
def test_the_gateway_accepts_both_http_protocols(protocol):
    env = _gateway_env(_gateway_cfg(protocol=protocol))
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == protocol
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == GATEWAY


def test_the_gateway_defaults_to_http_protobuf():
    env = _gateway_env(_gateway_cfg())
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"


def test_generic_backend_grpc_still_works():
    """A collector the operator named speaks whatever protocol they chose."""
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "generic",
            "protocol": "grpc",
            "endpoint": "http://collector:4317",
        }
    )
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"


def test_presets_do_not_offer_grpc_unqualified():
    """Every preset offering the knob must name the endpoint requirement for grpc.

    The knob is declarative, so it lives in the presets an operator edits
    rather than in a config template. The restriction is stated in the prose
    above the key — a preset spells one key per line, so the explanation cannot
    ride at the end of it.
    """
    from osprey.cli.build_profile_presets import _presets_dir

    declaring = sorted(
        path
        for path in _presets_dir().glob("*.yml")
        if "claude_code.telemetry.protocol: http/protobuf" in path.read_text(encoding="utf-8")
    )
    assert declaring, "no preset offers the protocol knob — the check is vacuous"
    for path in declaring:
        rows = path.read_text(encoding="utf-8").splitlines()
        key = next(
            i
            for i, raw in enumerate(rows)
            if "claude_code.telemetry.protocol: http/protobuf" in raw
        )
        prose = []
        i = key - 1
        while i >= 0 and rows[i].strip().startswith("#"):
            prose.insert(0, rows[i])
            i -= 1
        assert "endpoint" in "\n".join(prose), (
            f"{path.name} offers grpc without naming the endpoint requirement"
        )
