"""The telemetry gateway: its Collector configuration and its compose fragment.

The Collector configuration is pinned byte for byte by the goldens in
``telemetry_gateway_golden/``; regenerate them by running this module:

    PYTHONPATH=src ./.venv/bin/python tests/deployment/test_telemetry_gateway.py

The tests below state what the goldens hold, so a regenerated golden that lost a
property fails here with the property named, not as a diff.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey
from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_gateway as gw
from osprey.deployment.compose_generator import _inject_project_metadata
from osprey.deployment.telemetry_credentials import (
    HTPASSWD_FILE,
    INGEST_FILE,
    INGEST_PASSWORD_ENV,
)

_GOLDEN_DIR = Path(__file__).parent / "telemetry_gateway_golden"

#: A ledger receiver as the ledger feed renders one, enough for the pipeline
#: shape to exist; the receiver's own options are the ledger feed's to pin.
LEDGER_RECEIVERS = {
    "file_log/d0123456789ab": {
        "include": [f"{gw.AUDIT_CONTAINER_ROOT}/alice/*.jsonl"],
        "start_at": "beginning",
        "storage": gw.STORAGE,
    }
}

#: A bluesky deployment's ledger feed as rendered: a person, the sidecar and the
#: queueserver, so the trust condition names both trusted writers.
FEED_CONFIG: dict[str, Any] = {
    "deployed_services": ["telemetry_store", "telemetry_gateway", "bluesky"],
}
FEED_RECEIVERS = gw.ledger_receivers(FEED_CONFIG, None, ["alice", "queueserver", "sidecar"])

#: The renders the goldens pin, by golden file name.
RENDERS: dict[str, dict[str, Any]] = {
    "collector_default.yaml": {"config": {}, "ledger_receivers": None},
    "collector_with_ledger.yaml": {"config": {}, "ledger_receivers": LEDGER_RECEIVERS},
    "collector_ledger_feed.yaml": {"config": FEED_CONFIG, "ledger_receivers": FEED_RECEIVERS},
}


def _render(name: str) -> str:
    spec = RENDERS[name]
    return gw.render_collector_config(spec["config"], ledger_receivers=spec["ledger_receivers"])


def _parsed(config: dict | None = None, **kwargs: Any) -> dict[str, Any]:
    return yaml.safe_load(gw.render_collector_config(config or {}, **kwargs))


# --- the goldens ------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RENDERS))
def test_the_render_matches_its_golden(name: str) -> None:
    golden = _GOLDEN_DIR / name
    assert golden.is_file(), f"no golden {golden}; run this module to write it"
    assert _render(name) == golden.read_text(encoding="utf-8")


def test_two_renders_are_byte_equal() -> None:
    config = {"services": {"telemetry_store": {"retention_days": 30}}}
    assert gw.render_collector_config(config) == gw.render_collector_config(config)


# --- the receiver -------------------------------------------------------------


def test_senders_post_otlp_http_behind_their_login() -> None:
    http = _parsed()["receivers"]["otlp"]["protocols"]
    assert set(http) == {"http"}, "OTLP/HTTP only: no gRPC listener"
    http = http["http"]
    assert http["endpoint"] == f"0.0.0.0:{tc.GATEWAY_OTLP_CONTAINER_PORT}"
    assert http["auth"] == {"authenticator": gw.BASICAUTH}
    assert http["compression_algorithms"] == ["", "gzip"]
    assert http["include_metadata"] is False
    assert http["max_request_body_size"] == tc.gateway_max_request_body_size({})


def test_the_request_body_cap_follows_the_content_length() -> None:
    config = {"claude_code": {"telemetry": {"content_max_length": 1_000_000}}}
    http = _parsed(config)["receivers"]["otlp"]["protocols"]["http"]
    assert http["max_request_body_size"] == 16 * 1_000_000 * 3


def test_the_login_file_is_the_mounted_htpasswd() -> None:
    parsed = _parsed()
    assert parsed["extensions"][gw.BASICAUTH] == {"htpasswd": {"file": gw.HTPASSWD_CONTAINER_PATH}}
    assert gw.BASICAUTH in parsed["service"]["extensions"]


# --- identity ----------------------------------------------------------------


def _statements(signal: str) -> dict[str, list[str]]:
    groups = _parsed()["processors"]["transform/identity"][f"{signal}_statements"]
    return {group["context"]: group["statements"] for group in groups}


_LOGIN = 'otelcol.client.auth.attributes["username"]'


@pytest.mark.parametrize(
    ("signal", "record"),
    [("log", "log"), ("trace", "span"), ("metric", "datapoint")],
)
def test_the_login_is_stamped_on_the_resource_and_every_record(signal: str, record: str) -> None:
    statements = _statements(signal)
    for context in ("resource", record):
        stamp = f'set({context}.attributes["osprey.user"], {_LOGIN})'
        scrub = f'delete_matching_keys({context}.attributes, "^osprey\\\\.")'
        assert stamp in statements[context], (context, statements[context])
        # The sender's own osprey.* keys go first, so nothing it claimed survives.
        assert statements[context].index(scrub) < statements[context].index(stamp)


@pytest.mark.parametrize(
    ("signal", "contexts"),
    [
        ("log", {"resource", "scope", "log"}),
        ("trace", {"resource", "scope", "span", "spanevent"}),
        ("metric", {"resource", "scope", "datapoint"}),
    ],
)
def test_every_reachable_level_loses_the_senders_osprey_keys(signal: str, contexts: set) -> None:
    statements = _statements(signal)
    assert set(statements) == contexts
    for context in contexts:
        assert f'delete_matching_keys({context}.attributes, "^osprey\\\\.")' in statements[context]


def test_span_links_and_exemplars_are_dropped() -> None:
    """Their attributes are sender claims no statement can reach."""
    assert "set(span.links, nil)" in _statements("trace")["span"]
    assert "set(datapoint.exemplars, nil)" in _statements("metric")["datapoint"]


def test_the_identity_never_reads_the_raw_credential() -> None:
    text = gw.render_collector_config({})
    assert '"raw"' not in text
    assert "metadata" not in text.replace("include_metadata", "")


@pytest.mark.parametrize("context", ["resource", "log", "span", "datapoint"])
@pytest.mark.parametrize("key", ["session.id", "terminal.session"])
def test_a_session_token_that_is_not_plain_is_deleted(context: str, key: str) -> None:
    signal = {"span": "trace", "datapoint": "metric"}.get(context, "log")
    guard = next(
        line
        for line in _statements(signal)[context]
        if line.startswith(f'delete_key({context}.attributes, "{key}")')
    )
    assert f'IsString({context}.attributes["{key}"])' in guard
    assert '"^[A-Za-z0-9._-]{1,128}$"' in guard


def test_a_log_keeps_its_event_name_and_gets_a_time() -> None:
    log = _statements("log")["log"]
    assert (
        'set(log.attributes["event.name"], log.event_name) '
        'where log.event_name != "" and log.attributes["event.name"] == nil'
    ) in log
    assert "set(log.time, log.observed_time) where log.time_unix_nano == 0" in log


# --- the clamp ---------------------------------------------------------------


def test_the_clamp_drops_what_lies_outside_retention() -> None:
    clamp = _parsed({"services": {"telemetry_store": {"retention_days": 30}}})["processors"][
        "filter/clamp"
    ]
    bounds = 'Now() - Duration("720h") or {f} > Now() + Duration("1h")'
    assert clamp["logs"]["log_record"] == ["log.time < " + bounds.format(f="log.time")]
    assert clamp["traces"]["span"] == ["span.start_time < " + bounds.format(f="span.start_time")]
    assert clamp["metrics"]["datapoint"] == [
        "datapoint.time < " + bounds.format(f="datapoint.time")
    ]


def test_the_clamp_defaults_to_the_default_retention() -> None:
    clamp = _parsed()["processors"]["filter/clamp"]
    assert f'Duration("{tc.DEFAULT_RETENTION_DAYS * 24}h")' in clamp["logs"]["log_record"][0]


# --- pipelines ---------------------------------------------------------------


def test_the_otlp_pipelines_start_with_the_memory_limiter() -> None:
    pipelines = _parsed()["service"]["pipelines"]
    assert set(pipelines) == {"logs/otlp", "traces", "metrics"}
    for pipeline in pipelines.values():
        assert pipeline["receivers"] == ["otlp"]
        assert pipeline["processors"] == ["memory_limiter", "transform/identity", "filter/clamp"]
        assert pipeline["exporters"] == [gw.OTLP_EXPORTER]


def test_the_ledger_pipeline_exists_only_with_ledger_receivers() -> None:
    parsed = _parsed()
    assert gw.LEDGER_PIPELINE not in parsed["service"]["pipelines"]
    assert gw.LEDGER_EXPORTER not in parsed["exporters"]


def test_the_ledger_pipeline_has_no_memory_limiter() -> None:
    """A refused ledger batch is held and retried without end: the limiter would
    never let go of it."""
    parsed = _parsed(ledger_receivers=LEDGER_RECEIVERS)
    ledger = parsed["service"]["pipelines"][gw.LEDGER_PIPELINE]
    assert ledger["receivers"] == list(LEDGER_RECEIVERS)
    assert "memory_limiter" not in ledger["processors"]
    assert ledger["exporters"] == [gw.LEDGER_EXPORTER]
    assert (
        parsed["receivers"]["file_log/d0123456789ab"] == LEDGER_RECEIVERS["file_log/d0123456789ab"]
    )
    for name, pipeline in parsed["service"]["pipelines"].items():
        if name != gw.LEDGER_PIPELINE:
            assert pipeline["receivers"] == ["otlp"]


# --- exporters and storage ----------------------------------------------------


@pytest.mark.parametrize(
    ("exporter", "table", "blocks"),
    [(gw.OTLP_EXPORTER, "otel_logs", False), (gw.LEDGER_EXPORTER, "audit_ledger", True)],
)
def test_the_exporters_insert_as_ingest_through_a_disk_queue(
    exporter: str, table: str, blocks: bool
) -> None:
    stated = _parsed(ledger_receivers=LEDGER_RECEIVERS)["exporters"][exporter]
    assert stated["endpoint"] == f"http://{tc.STORE_SERVICE}:{tc.STORE_HTTP_CONTAINER_PORT}"
    assert stated["username"] == "ingest"
    assert stated["password"] == "${env:" + INGEST_PASSWORD_ENV + "}"
    assert stated["database"] == "telemetry"
    assert stated["logs_table_name"] == table
    assert stated["create_schema"] is False
    assert stated["async_insert"] is False
    assert stated["ttl"] == 0
    assert stated["timeout"] == "30s"
    assert stated["retry_on_failure"] == {"max_elapsed_time": 0}
    queue = stated["sending_queue"]
    assert queue["enabled"] is True
    assert queue["storage"] == gw.STORAGE
    assert queue["sizer"] == "bytes"
    assert queue["queue_size"] == 1024**3
    assert queue["batch"] == {"sizer": "bytes", "max_size": 8 * 1024**2}
    assert queue.get("block_on_overflow", False) is blocks


def test_the_state_storage_is_durable_and_compacts() -> None:
    parsed = _parsed()
    storage = parsed["extensions"][gw.STORAGE]
    assert storage["directory"] == f"{gw.STATE_CONTAINER_DIR}/state"
    assert storage["create_directory"] is True
    assert storage["fsync"] is True
    assert storage["recreate"] is True
    assert storage["compaction"] == {
        "on_start": True,
        "on_rebound": True,
        "directory": f"{gw.STATE_CONTAINER_DIR}/compact",
        "cleanup_on_start": True,
        "rebound_needed_threshold_mib": 100,
        "rebound_trigger_threshold_mib": 10,
    }
    assert gw.STORAGE in parsed["service"]["extensions"]


def test_the_gateways_own_counters_are_served_for_scraping() -> None:
    readers = _parsed()["service"]["telemetry"]["metrics"]["readers"]
    assert readers == [
        {
            "pull": {
                "exporter": {
                    "prometheus": {"host": "0.0.0.0", "port": tc.GATEWAY_METRICS_CONTAINER_PORT}
                }
            }
        }
    ]


# --- the render context ----------------------------------------------------------


def test_the_memory_limit_is_the_formula_for_r() -> None:
    assert gw.gateway_render_context({})["mem_limit"] == tc.gateway_memory_floor({}, 0)
    context = gw.gateway_render_context({}, ledger_receivers=LEDGER_RECEIVERS)
    assert context["receivers"] == 1
    assert context["mem_limit"] == tc.gateway_memory_floor({}, 1)


def test_a_stated_memory_limit_wins() -> None:
    config = {"services": {"telemetry_gateway": {"memory": "2g"}}}
    assert gw.gateway_render_context(config)["mem_limit"] == 2 * 1024**3


def test_the_command_runs_the_synchronous_emitter() -> None:
    assert gw.gateway_render_context({})["command"] == [
        f"--config={gw.CONFIG_CONTAINER_PATH}",
        "--feature-gates=stanza.synchronousLogEmitter",
    ]


def test_a_stated_image_wins() -> None:
    assert gw.gateway_render_context({})["image"] == tc.DEFAULT_GATEWAY_IMAGE
    config = {"services": {"telemetry_gateway": {"image": "mirror.example/otelcol:1"}}}
    assert gw.gateway_render_context(config)["image"] == "mirror.example/otelcol:1"


# --- the compose fragment -----------------------------------------------------------


def _templates_root() -> Path:
    return Path(str(resources.files(osprey).joinpath("templates")))


def _render_template(name: str, config: dict) -> str:
    from jinja2 import Environment, FileSystemLoader

    environment = Environment(loader=FileSystemLoader(str(_templates_root())), autoescape=False)
    base: dict[str, Any] = {
        "project_name": "demo",
        "project_root": "/r/demo",
        "services": {
            path.parent.name: {}
            for path in _templates_root().glob("services/*/docker-compose*.yml.j2")
        },
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": ["telemetry_gateway", "telemetry_store"],
    }
    for key, value in config.items():
        base[key] = {**base[key], **value} if isinstance(value, dict) else value
    return environment.get_template(f"services/telemetry_gateway/{name}").render(
        _inject_project_metadata(base)
    )


def _gateway_service(config: dict | None = None) -> dict[str, Any]:
    rendered = _render_template("docker-compose.yml.j2", config or {})
    return yaml.safe_load(rendered)["services"]["telemetry_gateway"]


def test_the_container_holds_only_what_it_needs() -> None:
    service = _gateway_service()
    assert service["user"] == "0:0"
    assert service["cap_drop"] == ["ALL"]
    assert service["cap_add"] == ["DAC_READ_SEARCH"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["read_only"] is True
    assert service["cpus"] == 1
    assert service["stop_grace_period"] == "30s"
    assert service["container_name"] == "demo-telemetry-gateway"
    assert service["command"] == gw.gateway_render_context({})["command"]
    assert service["env_file"] == [INGEST_FILE]
    assert service["labels"] == {
        "osprey.telemetry.cred.digest": "${OSPREY_TELEMETRY_DIGEST_TELEMETRY_GATEWAY:-}"
    }


def test_every_mount_but_the_state_volume_is_read_only() -> None:
    volumes = _gateway_service()["volumes"]
    assert volumes == [
        f"./build/services/telemetry_gateway/config.yaml:{gw.CONFIG_CONTAINER_PATH}:ro",
        f"./{HTPASSWD_FILE}:{gw.HTPASSWD_CONTAINER_PATH}:ro",
        f"./var/audit:{gw.AUDIT_CONTAINER_ROOT}:ro",
        f"{gw.STATE_VOLUME}:{gw.STATE_CONTAINER_DIR}",
    ]


def test_senders_post_on_the_bind_address_and_counters_stay_on_loopback() -> None:
    service = _gateway_service(
        {
            "deployment": {"bind_address": "0.0.0.0"},
            "services": {"telemetry_gateway": {"port": 4400}},
        }
    )
    metrics = tc.metrics_port({})
    assert service["ports"] == ["0.0.0.0:4400:4318", f"127.0.0.1:{metrics}:8888"]


def test_the_memory_limit_is_rendered_from_the_context() -> None:
    assert _gateway_service()["mem_limit"] == tc.gateway_memory_floor({}, 0)
    stated = _gateway_service({"services": {"telemetry_gateway": {"memory": "1g"}}})
    assert stated["mem_limit"] == 1024**3


def test_the_config_file_beside_the_fragment_is_the_rendered_collector_config() -> None:
    rendered = _render_template("config.yaml.j2", {})
    assert rendered.rstrip("\n") == gw.render_collector_config({}).rstrip("\n")


def _regenerate() -> None:
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(RENDERS):
        (_GOLDEN_DIR / name).write_text(_render(name), encoding="utf-8")
        print(f"wrote {_GOLDEN_DIR / name}")


if __name__ == "__main__":
    _regenerate()
