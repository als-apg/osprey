"""What the application SDK's own names do, beyond re-exporting.

``serve_tools`` refuses a tool server that registers a tool the rendered config
does not declare and answers every application error under its ``<app>.<type>`` name, which the
error-guidance hook classifies; the composed chain case runs a synthetic tool
server that checks the approved call and then fires a command write.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastmcp import Client, FastMCP

import osprey.apps.sdk as sdk
from osprey.apps.errors import AppManifestError
from osprey.mcp_server import approval_record, startup
from osprey.mcp_server.approval_record import APPROVAL_MISSING, Refusal
from osprey_connectors import posture_store, refusals
from osprey_connectors.workspace import reset_config_cache
from tests.apps import test_sdk_check_approved_call as approved_call
from tests.apps.test_sdk_check_approved_call import APP, DIGEST, FULL_NAME, TOOL, VALUE, _config
from tests.hooks.conftest import import_hook

#: The rendered deployment the approved-call tests run in, a fixture.
deployment = approved_call.deployment

TOOL_USE_ID = "toolu_composed"

#: The rendered ``app_error_types`` of the application, as the build writes it.
APP_ERROR_TYPES = {
    f"{APP}.refused": {"class": "Safety", "text": "The operator declined."},
    f"{APP}.unreachable": {"class": "Connection", "text": "The panel is not running."},
}


def _declaring(config: dict[str, Any], *read: str) -> dict[str, Any]:
    """*config* with ``apps.<app>.tools.read`` seeded, as the render writes it."""
    config["apps"][APP]["tools"] = {"read": list(read or ("read_knob",))}
    return config


def _write_config(path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(path))
    reset_config_cache()


@pytest.fixture
def rendered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A rendered config declaring ``read_knob`` and ``set_knob`` for the application."""
    monkeypatch.setenv("OSPREY_APP_NAME", APP)
    _write_config(tmp_path / "config.yml", monkeypatch, _declaring(_config()))
    yield
    reset_config_cache()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch):
    """Run :func:`sdk.serve_tools` with the transport loop left out; yield what it served."""
    import osprey_connectors.logger

    monkeypatch.setattr(startup, "_server_label", startup._server_label)
    monkeypatch.setattr(startup, "prime_config_builder", lambda: None)
    monkeypatch.setattr(osprey_connectors.logger, "configure_logging", lambda *a, **k: None)
    servers: list[FastMCP] = []

    def serve(build) -> FastMCP:
        def create() -> FastMCP:
            server = build()
            server.run = lambda *a, **k: servers.append(server)
            return server

        sdk.serve_tools(create)
        assert servers, "serve_tools returned without running the server"
        return servers[-1]

    return serve


def _read_knob() -> float:
    return 1.0


def _server(*extra: str, read_knob=_read_knob) -> FastMCP:
    server = FastMCP("knob")
    server.tool(read_knob, name="read_knob")

    @server.tool
    def set_knob(value: float) -> str:
        raise sdk.ApprovalRefused(
            APPROVAL_MISSING, "not run: no approval record.", {"value": value}
        )

    for name in extra:
        server.tool(lambda: None, name=name)
    return server


async def _serve_async(served, build) -> FastMCP:
    """Serve from a worker thread: start-up runs its own event loop, as at ``__main__``."""
    return await asyncio.to_thread(served, build)


async def _envelope(server: FastMCP, tool: str, arguments: dict | None = None) -> dict:
    async with Client(server) as client:
        result = await client.call_tool(
            tool,
            arguments or {},
            raise_on_error=False,
            meta={"claudecode/toolUseId": TOOL_USE_ID},
        )
    assert result.is_error
    return json.loads(result.content[0].text)


def _classified(envelope: dict) -> str:
    guidance = import_hook("osprey_error_guidance")
    return guidance._classify_type(envelope["error_type"], APP_ERROR_TYPES)[0]


# --- serve_tools: start-up --------------------------------------------------


@pytest.mark.usefixtures("rendered")
def test_serve_tools_refuses_a_tool_the_rendered_config_does_not_declare(served):
    with pytest.raises(AppManifestError) as info:
        served(lambda: _server("reset_knob"))
    assert info.value.rule == "undeclared_tool"
    assert "reset_knob" in str(info.value)
    assert f"apps.{APP}.tools.read" in str(info.value)


@pytest.mark.usefixtures("rendered")
def test_serve_tools_serves_a_server_whose_tools_are_all_declared(served):
    assert isinstance(served(_server), FastMCP)


def test_serve_tools_reads_the_declared_tools_from_the_rendered_config(
    served, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OSPREY_APP_NAME", APP)
    config = tmp_path / "config.yml"
    _write_config(config, monkeypatch, _declaring(_config(), "read_knob", "reset_knob"))
    assert isinstance(served(lambda: _server("reset_knob")), FastMCP)
    _write_config(config, monkeypatch, _config())
    with pytest.raises(AppManifestError, match="read_knob"):
        served(_server)


@pytest.mark.usefixtures("rendered")
def test_serve_tools_refuses_a_server_no_application_names(served, monkeypatch):
    monkeypatch.delenv("OSPREY_APP_NAME")
    with pytest.raises(AppManifestError) as info:
        served(_server)
    assert info.value.rule == "unnamed_app"


# --- serve_tools: application error types -----------------------------------


@pytest.mark.usefixtures("rendered")
async def test_an_uncaught_approval_refusal_answers_app_refused_classified_safety(served):
    server = await _serve_async(served, _server)
    envelope = await _envelope(server, "set_knob", {"value": 1.0})
    assert envelope["error_type"] == f"{APP}.refused"
    assert envelope["error_message"] == "not run: no approval record."
    assert envelope["details"] == {"reason": APPROVAL_MISSING, "value": 1.0}
    assert _classified(envelope) == "Safety"


@pytest.mark.usefixtures("rendered")
async def test_make_error_unreachable_answers_app_unreachable_classified_connection(served):
    def read_knob() -> float:
        sdk.make_error("unreachable", "the panel backend did not answer")

    server = await _serve_async(served, lambda: _server(read_knob=read_knob))
    envelope = await _envelope(server, "read_knob")
    assert envelope["error_type"] == f"{APP}.unreachable"
    assert envelope["error_message"] == "the panel backend did not answer"
    assert _classified(envelope) == "Connection"


@pytest.mark.usefixtures("rendered")
async def test_any_other_error_type_passes_through_unchanged(served):
    def read_knob() -> float:
        sdk.make_error("limits_violation", "out of range")

    server = await _serve_async(served, lambda: _server(read_knob=read_knob))
    envelope = await _envelope(server, "read_knob")
    assert envelope["error_type"] == "limits_violation"


# --- The composed chain: approved call, then command write ------------------


def _chain_server(fired: list[tuple[list[str], Any]]):
    def build() -> FastMCP:
        server = FastMCP(APP)

        @server.tool
        def read_knob() -> float:
            return 1.0

        @server.tool
        async def set_knob(
            value: float, approved_target: str | None = None, approved_digest: str | None = None
        ) -> list[str]:
            arguments = {"value": value}
            arguments.update(
                {
                    name: field
                    for name, field in (
                        ("approved_target", approved_target),
                        ("approved_digest", approved_digest),
                    )
                    if field is not None
                }
            )
            sdk.check_approved_call(TOOL, arguments)
            return await sdk.command_write(None, ["SYN:KNOB:ARM"], value)

        return server

    async def recording_command_write(_connector, addresses, value) -> list[str]:
        fired.append((list(addresses), value))
        return list(addresses)

    return build, recording_command_write


async def test_with_a_stamp_the_command_write_is_reached(deployment, served, monkeypatch):
    fired: list[tuple[list[str], Any]] = []
    build, recorder = _chain_server(fired)
    monkeypatch.setattr(sdk, "command_write", recorder)
    deployment.write_config(_declaring(_config()))
    deployment.stamps.ask(
        FULL_NAME, tool_use_id=TOOL_USE_ID, approved_target="va", approved_digest=DIGEST
    )
    server = await _serve_async(served, build)
    async with Client(server) as client:
        result = await client.call_tool(
            TOOL,
            {**VALUE, "approved_target": "va", "approved_digest": DIGEST},
            meta={"claudecode/toolUseId": TOOL_USE_ID},
        )
    assert not result.is_error
    assert fired == [(["SYN:KNOB:ARM"], VALUE["value"])]


async def test_without_a_stamp_the_record_refuses_and_the_write_is_never_reached(
    deployment, served, monkeypatch
):
    fired: list[tuple[list[str], Any]] = []
    build, recorder = _chain_server(fired)
    monkeypatch.setattr(sdk, "command_write", recorder)
    deployment.write_config(_declaring(_config()))
    approved = {"approved_target": "va", "approved_digest": DIGEST}
    assert approval_record.require_approval_record(
        FULL_NAME, _config(), approved=approved, guarded_run=True
    ) == Refusal(APPROVAL_MISSING, approval_record._NO_RECORD)

    server = await _serve_async(served, build)
    envelope = await _envelope(server, TOOL, {**VALUE, **approved})
    assert envelope["error_type"] == f"{APP}.refused"
    assert envelope["details"]["reason"] == APPROVAL_MISSING
    assert fired == []


# --- Wrappers ---------------------------------------------------------------


def test_agent_panel_headers_is_the_panel_token_header(monkeypatch):
    from osprey.mcp_server import http

    monkeypatch.setenv("OSPREY_PANEL_TOKEN", "panel-token-under-test")
    http.reset_panel_token_latch()
    try:
        assert sdk.agent_panel_headers() == http._panel_auth_headers()
        assert sdk.agent_panel_headers()
    finally:
        http.reset_panel_token_latch()


def test_facility_file_sits_at_the_render_root(tmp_path: Path, monkeypatch):
    (tmp_path / "config.yml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    reset_config_cache()
    assert sdk.facility_file() == tmp_path / "facility.json"


def _facts_document() -> dict[str, Any]:
    from osprey.facility.views.facts import zero_source_facts

    document = zero_source_facts({"code": "SYN", "name": "Synthetic", "description": None})
    document["measurement_models"] = {
        "ring": {
            "path": "data/pyaml/ring/configuration.yaml",
            "sha256": "0" * 64,
            "files": {"kick.json": "1" * 64, "turns.json": "2" * 64, "other.json": "3" * 64},
            "kinds": {
                "fixture_kind": {
                    "view_files": ["kick.json", "turns.json"],
                    "tool_keys": ["kick", "turns"],
                }
            },
        }
    }
    document["kinds"] = {"fixture_kind": {"turns": 1024}}
    return document


def test_read_facts_finds_a_kinds_view_files_and_own_keys(tmp_path: Path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "facility_facts.json").write_text(json.dumps(_facts_document()))
    (tmp_path / "config.yml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    reset_config_cache()

    facts = sdk.read_facts()
    assert facts == sdk.read_facts(tmp_path)
    model = facts["measurement_models"]["ring"]
    view_files = {
        name: model["files"][name] for name in model["kinds"]["fixture_kind"]["view_files"]
    }
    assert view_files == {"kick.json": "1" * 64, "turns.json": "2" * 64}
    assert facts.get("kinds", {}).get("fixture_kind", {}) == {"turns": 1024}
    assert sdk.tool_keys(facts, "ring", "fixture_kind") == ["kick", "turns"]


def test_tool_keys_of_an_unknown_kind_is_a_key_error():
    with pytest.raises(KeyError):
        sdk.tool_keys(_facts_document(), "ring", "orm")


def test_read_facts_of_a_render_without_facts_is_the_zero_source_facts(tmp_path: Path):
    facts = sdk.read_facts(tmp_path, project_name="probe")
    assert facts["measurement_models"] == {}
    assert facts["identity"]["name"] == "probe"


def test_queue_is_active_reads_the_manager_state():
    assert sdk.queue_is_active({"manager_state": "executing_queue"}) is True
    assert sdk.queue_is_active({"manager_state": "idle"}) is False


def test_configure_app_is_the_interface_setup_itself():
    from osprey.interfaces._app_setup import configure_interface_app

    assert sdk.configure_app is configure_interface_app


def test_the_write_chain_names_are_the_connectors_own():
    assert sdk.write_verdict is posture_store.write_verdict
    assert sdk.WriteVerdict is posture_store.WriteVerdict
    assert sdk.bind_owner is posture_store.bind_owner
    assert sdk.refusal_phrase is refusals.refusal_phrase
