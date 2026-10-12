"""``otelcol-contrib validate`` on the rendered gateway configuration.

``test_telemetry_gateway.py`` pins what the renderer writes; only the pinned
Collector can say whether it accepts it. ``validate`` parses every component's
options and every OTTL statement and condition, and refuses an unknown feature
gate, so a path the pinned release does not have (``span.links``,
``log.event_name``, the ``otelcol`` context) fails here rather than at a
deployment's first ``osprey up``.

Each test first asserts the constructs it means to validate are present, so a
renderer that stopped emitting them cannot pass on an emptier file. Skipped
when docker is unavailable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_gateway as gw
from osprey.deployment.telemetry_credentials import INGEST_PASSWORD_ENV
from tests._container_support import docker_cli_unavailable_reason

_DOCKER_UNAVAILABLE = docker_cli_unavailable_reason()

pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(_DOCKER_UNAVAILABLE is not None, reason=_DOCKER_UNAVAILABLE or ""),
]

#: One ledger receiver, enough for the ledger pipeline and its exporter to exist.
_LEDGER_RECEIVERS = {
    "file_log/d0123456789ab": {
        "include": [f"{gw.AUDIT_CONTAINER_ROOT}/alice/*.jsonl"],
        "start_at": "beginning",
        "storage": gw.STORAGE,
    }
}


def _validate(config_text: str, tmp_path: Path) -> subprocess.CompletedProcess:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text, encoding="utf-8")
    htpasswd = tmp_path / "senders.htpasswd"
    htpasswd.write_text("alice:{SHA}Vb0r2l6fN8sYbXBBo8m3j9C4Ofs=\n", encoding="utf-8")
    context = gw.gateway_render_context({})
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            f"{INGEST_PASSWORD_ENV}=validate",
            "-v",
            f"{config_file}:{gw.CONFIG_CONTAINER_PATH}:ro",
            "-v",
            f"{htpasswd}:{gw.HTPASSWD_CONTAINER_PATH}:ro",
            tc.DEFAULT_GATEWAY_IMAGE,
            "validate",
            *context["command"],
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _assert_constructs(text: str) -> None:
    parsed = yaml.safe_load(text)
    statements = str(parsed["processors"]["transform/identity"])
    for construct in (
        'otelcol.client.auth.attributes["username"]',
        "set(span.links, nil)",
        "set(datapoint.exemplars, nil)",
        "log.event_name",
        "IsMatch(",
    ):
        assert construct in statements, construct
    assert parsed["processors"]["filter/clamp"]["logs"]["log_record"]
    assert parsed["exporters"][gw.OTLP_EXPORTER]["sending_queue"]["sizer"] == "bytes"


def test_the_default_render_validates(tmp_path: Path) -> None:
    text = gw.render_collector_config({})
    _assert_constructs(text)
    result = _validate(text, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_render_with_a_ledger_pipeline_validates(tmp_path: Path) -> None:
    text = gw.render_collector_config({}, ledger_receivers=_LEDGER_RECEIVERS)
    _assert_constructs(text)
    parsed = yaml.safe_load(text)
    assert parsed["exporters"][gw.LEDGER_EXPORTER]["sending_queue"]["block_on_overflow"] is True
    result = _validate(text, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_refuses_a_path_the_pinned_release_lacks(tmp_path: Path) -> None:
    """The check is live: a statement on a path that does not exist is refused."""
    text = gw.render_collector_config({}).replace("set(span.links, nil)", "set(span.no_such, nil)")
    result = _validate(text, tmp_path)
    assert result.returncode != 0
    assert "no_such" in result.stdout + result.stderr


def test_ledger_receivers_for_any_directory_name_validate(tmp_path: Path) -> None:
    """``ops team``, ``a+b`` and ``EXPR(x)`` are legal directory names to the writer;
    escaped in the glob and quoted in ``add``, their receivers load."""
    config = {"deployed_services": ["telemetry_store", "telemetry_gateway", "bluesky"]}
    names = ["alice", "ops team", "a+b", "EXPR(x)", "queueserver", "sidecar"]
    receivers = gw.ledger_receivers(config, None, names)
    text = gw.render_collector_config(config, ledger_receivers=receivers)
    parsed = yaml.safe_load(text)
    assert set(receivers) <= set(parsed["receivers"])
    assert parsed["service"]["pipelines"][gw.LEDGER_PIPELINE]["processors"] == list(
        gw.LEDGER_PROCESSORS
    )
    result = _validate(text, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_ledger_chain_reads_real_lines(tmp_path: Path) -> None:
    """The pinned Collector runs the receivers and processors over writer lines.

    Exporters are swapped for ``debug`` so no store is needed; everything
    between the file and the exporter is the rendered configuration.
    """
    import time

    from osprey.audit.envelope import AuditEnvelope
    from osprey.audit.writer import _encode

    config = {"deployed_services": ["telemetry_store", "telemetry_gateway", "bluesky"]}
    audit = tmp_path / "audit"
    for name in ("alice", "queueserver"):
        (audit / name).mkdir(parents=True)

    def envelope(**fields: object) -> bytes:
        base = {
            "ts": "2026-10-11T10:00:00Z",
            "surface": "approval",
            "actor": "alice",
            "posture": "writes",
            "posture_source": "app",
            "session": "s1",
            "subject": "sub",
            "decision": "approved",
            "reason": "r",
        }
        base.update(fields)
        return _encode(AuditEnvelope(**base).to_dict())

    cap = tc.gateway_max_log_size(config)
    (audit / "alice" / "a.jsonl").write_bytes(
        envelope(surface="executor", on_behalf_of="bob", tool_use_id="toolu_1", source="x")
        + envelope(ts="garbage", surface="channel-finder")
        + b"not json\n"
        + b'{"ts":"2026-10-11T10:00:02Z","surface":"approval","actor":"al\n'
        + envelope(detail="d" * 4000).replace(b'"approval"', b'"over"').rstrip(b"}\n")
        + b"y" * cap
        + b"}\n"
    )
    (audit / "queueserver" / "q.jsonl").write_bytes(
        envelope(surface="preflight", actor="queueserver", on_behalf_of="alice")
        + envelope(surface="queue", actor="queueserver", on_behalf_of="alice")
    )
    (audit / "alice" / "link.jsonl").symlink_to("../queueserver/q.jsonl")

    receivers = gw.ledger_receivers(config, None, ["alice", "queueserver"])
    for body in receivers.values():
        del body["storage"]
    rendered = gw.collector_config(config, ledger_receivers=receivers)
    rendered["receivers"] = receivers
    rendered["processors"] = {
        name: body for name, body in rendered["processors"].items() if name in gw.LEDGER_PROCESSORS
    }
    rendered["exporters"] = {"debug": {"verbosity": "detailed"}}
    rendered["extensions"] = {}
    rendered["service"]["extensions"] = []
    rendered["service"]["pipelines"] = {
        gw.LEDGER_PIPELINE: {
            "receivers": list(receivers),
            "processors": list(gw.LEDGER_PROCESSORS),
            "exporters": ["debug"],
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(rendered, sort_keys=False, width=1 << 16))
    name = f"osprey-ledger-chain-{tmp_path.name[-12:]}".lower()
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-v",
            f"{config_file}:{gw.CONFIG_CONTAINER_PATH}:ro",
            "-v",
            f"{audit}:{gw.AUDIT_CONTAINER_ROOT}:ro",
            tc.DEFAULT_GATEWAY_IMAGE,
            *gw.gateway_render_context({})["command"],
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    try:
        logs = ""
        for _ in range(60):
            time.sleep(1)
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=60
            ).stderr
            if logs.count("LogRecord #") >= 7:
                break
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)

    records = logs.split("LogRecord #")[1:]
    assert len(records) == 7, logs[-4000:]

    def record(marker: str) -> str:
        (found,) = [r for r in records if marker in r]
        return found

    executor = record('"surface":"executor"')
    assert "osprey.audit.tool_use_id: Str(toolu_1)" in executor
    assert "osprey.audit.on_behalf_of" not in executor  # alice is not a trusted writer
    garbage = record('"ts":"garbage"')
    assert "osprey.audit.time_unparsed: Bool(true)" in garbage
    assert "osprey.audit.surface: Str(channel-finder)" in garbage
    unparsed = record("Body: Str(not json)")
    assert "osprey.audit.unparsed: Bool(true)" in unparsed
    assert "osprey.audit.surface: Str(unparsed)" in unparsed
    torn = record('"actor":"al)')
    assert "osprey.audit.incomplete: Bool(true)" in torn
    over = record('"surface":"over"')
    assert "osprey.audit.truncated: Bool(true)" in over
    assert "osprey.audit.incomplete" not in over
    preflight = record('"surface":"preflight"')
    assert "osprey.audit.on_behalf_of: Str(alice)" in preflight
    assert "osprey.user: Str(queueserver)" in preflight
    queue = record('"surface":"queue"')
    assert "osprey.audit.on_behalf_of" not in queue
    for each in records:
        assert "ledger_" not in each
        # Read through a symlink: dropped by filter/ledger-path.
        assert "log.file.path: Str(/audit/alice/link.jsonl)" not in each
