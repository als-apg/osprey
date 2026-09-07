"""``protocol: grpc`` against an auto-derived openobserve endpoint fails loud.

The openobserve service publishes HTTP on 5080 only, and
``_resolve_telemetry_endpoint`` derives ``http://<host>:5080/api/<org>`` for that
backend regardless of ``protocol``. A gRPC exporter aimed at that URL drops every
metric and log without an error, which is precisely the silent-drop mode the
module already refuses elsewhere (unresolvable endpoint, leaked ``${VAR}``, blank
credentials). These tests pin the refusal and, equally important, pin the paths
that must keep working: ``backend: generic`` over gRPC, and an explicit
``endpoint`` that an operator aimed at a real gRPC collector.
"""

from __future__ import annotations

import pytest

from osprey.build.claude_code_telemetry import (
    TelemetryConfigError,
    _build_telemetry_env,
    _resolve_telemetry_endpoint,
)

_OO_CREDS = {"user": "root@example.com", "password": "secret"}


def _openobserve_cfg(**overrides: object) -> dict:
    """An enabled openobserve telemetry block with no explicit ``endpoint``."""
    cfg: dict = {
        "enabled": True,
        "backend": "openobserve",
        "openobserve": {**_OO_CREDS, "org": "als"},
    }
    cfg.update(overrides)
    return cfg


def test_grpc_with_derived_openobserve_endpoint_raises():
    with pytest.raises(TelemetryConfigError):
        _resolve_telemetry_endpoint(_openobserve_cfg(protocol="grpc"), in_container=False)


def test_grpc_with_derived_openobserve_endpoint_raises_in_container():
    """The in-container derivation (``openobserve:5080``) is HTTP-only too."""
    with pytest.raises(TelemetryConfigError):
        _resolve_telemetry_endpoint(_openobserve_cfg(protocol="grpc"), in_container=True)


def test_grpc_refusal_survives_host_override():
    """An explicit deploy host still yields the same HTTP-only 5080 URL."""
    with pytest.raises(TelemetryConfigError):
        _resolve_telemetry_endpoint(
            _openobserve_cfg(protocol="grpc"),
            in_container=True,
            openobserve_host="otel-store",
        )


@pytest.mark.parametrize("spelling", ["grpc", "GRPC", " gRPC "])
def test_grpc_refusal_is_case_and_whitespace_insensitive(spelling):
    """``resolve_env_vars`` can hand through any spelling a ``${VAR}`` carried."""
    with pytest.raises(TelemetryConfigError):
        _build_telemetry_env(_openobserve_cfg(protocol=spelling))


def test_grpc_refusal_message_names_the_keys():
    """The operator must be able to act on the message without reading source."""
    with pytest.raises(TelemetryConfigError) as excinfo:
        _build_telemetry_env(_openobserve_cfg(protocol="grpc"))
    message = str(excinfo.value)
    assert "claude_code.telemetry.protocol" in message
    assert "claude_code.telemetry.endpoint" in message
    assert "grpc" in message
    assert "openobserve" in message


def test_generic_backend_grpc_still_works():
    """The knob's one working configuration must be untouched."""
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


def test_openobserve_backend_grpc_with_explicit_endpoint_still_works():
    """Only the *derived* endpoint is refused; an explicit one is the operator's call."""
    env = _build_telemetry_env(_openobserve_cfg(protocol="grpc", endpoint="http://otel-store:4317"))
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://otel-store:4317"


def test_derived_openobserve_endpoint_unaffected_for_http():
    """The shipped default configuration resolves the http/protobuf endpoint."""
    env = _build_telemetry_env(_openobserve_cfg(protocol="http/protobuf"), in_container=False)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/als"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"


def test_derived_openobserve_endpoint_unaffected_when_protocol_omitted():
    env = _build_telemetry_env(_openobserve_cfg(), in_container=True)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/als"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"


def test_presets_do_not_offer_grpc_unqualified():
    """Every preset offering the knob must name the openobserve restriction.

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
