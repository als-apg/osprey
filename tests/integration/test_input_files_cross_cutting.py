"""Cross-cutting integration tests for dispatch input files.

These pin the guarantees that span the worker and the dispatcher together and so
belong to neither single unit's test module:

1. **Combined-cap arithmetic** — the caller-file cap, the follow-up re-injection
   cap, and a folded-history budget are sized so a worst-case request body still
   fits the 32 MB request-body limit with roughly a quarter to spare. A change to
   any one constant that eroded that headroom is caught here.
2. **No base64 payload leak, end to end** — a file's ``content_b64`` bytes ride
   only an image content block; they must never reach the assembled prompt text,
   the persisted run record, the dispatcher's recorded history, or any log sink.
3. **Version skew** — a request carrying ``input_files`` against a model that
   predates the field is silently dropped, not rejected. This is why a bridge
   must gate the feature on the ``/health`` capability rather than assume the
   worker consumed the batch.
4. **Fatal-4xx error_code propagation** — a worker 400 with a machine-readable
   ``detail`` travels webhook -> worker_client ``FatalDispatchError`` -> pool
   record -> the dispatcher's ``/dispatch/{id}`` poll body as a top-level error
   carrying the same ``error_code``.

Hermetic: FastAPI/Starlette ``TestClient`` and direct calls, fake ``query`` and
``dispatch_to_worker`` stand-ins, tmp-rooted artifact stores — no network, no
real SDK, no real worker.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from pydantic import BaseModel
from starlette.testclient import TestClient

from osprey.dispatch import server
from osprey.dispatch.sources.webhook import _MAX_WEBHOOK_BYTES, WebhookSource
from osprey.dispatch.trigger_config import TriggerConfig
from osprey.dispatch.worker_client import FatalDispatchError
from osprey.mcp_server.dispatch_worker import dispatch_api, sdk_runner
from osprey.mcp_server.dispatch_worker.dispatch_api import (
    MAX_REQUEST_BYTES,
    DispatchRequest,
    InputFile,
)
from osprey.mcp_server.dispatch_worker.input_files_policy import (
    MAX_IMAGE_BYTES,
    MAX_INGEST_FILES,
    MAX_TOTAL_DECODED_BYTES,
)

_MiB = 1024 * 1024


# ===========================================================================
# 1. Combined-cap arithmetic pin
# ===========================================================================

# Bridge-side inbound caps, mirrored here as documented literals because the
# bridge lives in a separate repository and cannot be imported. A follow-up
# request may carry, at most, a batch of newly-attached caller files PLUS a
# re-injection of a few recent prior images:
#   * new caller files: at most 5 files, 10 MiB decoded in total (inbound);
#   * re-injected prior images: at most 3 images, 8 MiB decoded in total.
# The worker's own ceiling (MAX_TOTAL_DECODED_BYTES) is what actually bounds the
# sum; these two caps are sized so their sum lands exactly on it.
_BRIDGE_NEW_FILES_MAX_DECODED = 10 * _MiB
_BRIDGE_REINJECT_MAX_DECODED = 8 * _MiB
_BRIDGE_NEW_FILES_MAX_COUNT = 5
_BRIDGE_REINJECT_MAX_COUNT = 3

# Prior-turn conversation history folded into the prompt on a follow-up dispatch.
_HISTORY_BUDGET_BYTES = 40 * 1024

# Generous allowance for the JSON envelope around the batch (field names, quotes,
# per-file filename/mime, the prompt text itself). The real overhead for <=8
# files is a few hundred bytes; 64 KiB is a deliberate over-estimate so the
# headroom this test asserts is a floor, not a knife's edge.
_ENVELOPE_OVERHEAD_BYTES = 64 * 1024


def _b64_len(decoded_bytes: int) -> int:
    """Length of the base64 encoding of ``decoded_bytes`` raw bytes."""
    return ((decoded_bytes + 2) // 3) * 4


def test_input_files_combined_caps_sum_to_worker_total_ceiling():
    """New-file and re-injection caps are sized to exactly saturate the worker
    ceiling — never to exceed it."""
    assert _BRIDGE_NEW_FILES_MAX_DECODED + _BRIDGE_REINJECT_MAX_DECODED == MAX_TOTAL_DECODED_BYTES
    # The two per-batch count caps stay within the worker's ingest-file cap when
    # the new files are all ingested (re-injected images ride ingest=False).
    assert _BRIDGE_NEW_FILES_MAX_COUNT <= MAX_INGEST_FILES
    # Each re-injected image is itself within the per-image cap.
    assert _BRIDGE_REINJECT_MAX_DECODED <= _BRIDGE_REINJECT_MAX_COUNT * MAX_IMAGE_BYTES


def test_input_files_worst_case_body_fits_request_limit_with_headroom():
    """The worst-case decoded batch, base64-encoded, plus folded history and JSON
    envelope, fits the 32 MB request-body limit with ~25% headroom."""
    # Worst-case decoded content is the worker ceiling (the two bridge caps sum to
    # it); base64 inflates it by 4/3.
    worst_body = (
        _b64_len(MAX_TOTAL_DECODED_BYTES) + _HISTORY_BUDGET_BYTES + _ENVELOPE_OVERHEAD_BYTES
    )
    assert worst_body <= MAX_REQUEST_BYTES

    headroom_fraction = (MAX_REQUEST_BYTES - worst_body) / MAX_REQUEST_BYTES
    # "~25% headroom" — pin it to a band so eroding it (bigger caps) or bloating
    # it (a needlessly huge limit) both trip the test.
    assert 0.20 <= headroom_fraction <= 0.30


def test_input_files_worker_and_dispatcher_body_limits_agree():
    """The worker and the dispatcher enforce the same 32 MB request-body limit, so
    a body sized against one is not surprised by the other."""
    assert MAX_REQUEST_BYTES == _MAX_WEBHOOK_BYTES == 32 * 1024 * 1024


# ===========================================================================
# 2. End-to-end: content_b64 never leaks (prompt / record / logs)
# ===========================================================================

# A distinctive image body whose base64 is a single >64-char run — long enough to
# be caught by the sdk_runner redaction filter if it ever reached a log record.
_LEAK_MARKER_BYTES = b"\x89PNG\r\n\x1a\n" + b"LEAK-MARKER-PAYLOAD-BODY" * 16
_LEAK_MARKER_B64 = base64.b64encode(_LEAK_MARKER_BYTES).decode("ascii")


def _project_root(tmp_path: Path, monkeypatch) -> Path:
    """Point OSPREY_PROJECT_DIR at a tmp project so the artifact store roots there."""
    project = tmp_path / "project"
    (project / "_agent_data" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    return project


@pytest.fixture
def _stub_sdk_helpers(monkeypatch):
    """Stub the deferred OSPREY helper imports so run_dispatch can be driven."""
    monkeypatch.setattr(
        "osprey.agent_runner.clean_env.build_clean_env",
        lambda **kw: {},
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt",
        lambda *a, **k: "system",
    )
    monkeypatch.setattr(
        "osprey.utils.config.get_facility_timezone",
        lambda *a, **k: "UTC",
    )


async def test_input_files_content_b64_absent_from_worker_record_and_logs(
    tmp_path, monkeypatch, caplog, _stub_sdk_helpers
):
    """A full worker run that ingests and re-inlines an image never leaks its
    content_b64 into the prompt text, the persisted run record, or any log."""
    _project_root(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {})

    # Capture exactly what would be persisted to disk (the terminal run record).
    persisted: dict = {}
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda rid, r: persisted.update(r))

    # Fake SDK generator: drain the streamed prompt to capture the user message,
    # then return a normal completion.
    captured: dict = {}

    async def fake_query(prompt, options):  # noqa: A002 - matches SDK signature
        messages = []
        async for m in prompt:
            messages.append(m)
        captured["user_message"] = messages[0]
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        # Spec'd mock passes the isinstance(message, ResultMessage) branch; set the
        # exact fields run_dispatch reads for a clean (non-error) completion.
        rm = MagicMock(spec=ResultMessage)
        rm.num_turns = 1
        rm.is_error = False
        rm.subtype = "success"
        rm.result = "ok"
        rm.api_error_status = None
        yield rm

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    req = DispatchRequest(
        prompt="describe this",
        allowed_tools=[],
        input_files=[
            InputFile(filename="plot.png", mime="image/png", content_b64=_LEAK_MARKER_B64),
        ],
    )
    with caplog.at_level(logging.DEBUG, logger="osprey.mcp_server.dispatch_worker"):
        await dispatch_api._run_dispatch_task("run-leak", req)

    # The image reached the model — its bytes ride source.data of an image block.
    content = captured["user_message"]["message"]["content"]
    assert isinstance(content, list)
    image_block = content[0]
    assert image_block["type"] == "image"
    assert image_block["source"]["data"] == _LEAK_MARKER_B64
    # ... but never the prompt text block accompanying it.
    text_block = content[-1]
    assert text_block["type"] == "text"
    assert _LEAK_MARKER_B64 not in text_block["text"]

    # The persisted terminal record — every field, serialized — carries no b64.
    result = dispatch_api._runs["run-leak"]
    assert result["status"] == "completed"
    assert _LEAK_MARKER_B64 not in json.dumps(result, default=str)
    assert persisted and _LEAK_MARKER_B64 not in json.dumps(persisted, default=str)
    # The input file is surfaced by descriptor only (filename/mime/entry_id).
    assert [d["filename"] for d in result["input_artifacts"]] == ["plot.png"]

    # No log record anywhere in the dispatch loggers carries the payload.
    assert _LEAK_MARKER_B64 not in caplog.text


@pytest.mark.asyncio
async def test_input_files_content_b64_absent_from_dispatcher_prompt_record_logs(caplog):
    """On the dispatcher, the popped batch is forwarded to the worker but its
    content_b64 reaches neither the folded prompt, the recorded history, nor a log."""
    from osprey.dispatch.registry import TriggerRegistry

    trigger = TriggerConfig(
        name="deploy", source="webhook", action={"prompt": "do it", "allowed_tools": []}
    )
    registry = TriggerRegistry()
    await registry.register(trigger)

    captured: dict = {}

    async def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"run_id": "r1", "status": "accepted"}

    payload = {
        "question": "hello",
        "input_files": [
            {
                "filename": "plot.png",
                "mime": "image/png",
                "content_b64": _LEAK_MARKER_B64,
                "ingest": True,
            }
        ],
    }
    with caplog.at_level(logging.DEBUG, logger="osprey.dispatch"):
        with patch.object(server, "dispatch_to_worker", _fake_dispatch):
            result = await server._dispatch_with_policy(
                trigger, payload, registry, "http://worker", "tok"
            )

    assert result == {"run_id": "r1", "status": "accepted"}
    # Forwarded to the worker as a structured batch (bytes intact there)...
    assert captured["input_files"][0]["content_b64"] == _LEAK_MARKER_B64
    # ... but absent from the prompt the worker was asked to fold/run.
    assert _LEAK_MARKER_B64 not in captured["prompt"]

    # The durable trigger history — serialized — never carries the payload bytes.
    history = await registry.get_history("deploy")
    assert history and _LEAK_MARKER_B64 not in json.dumps(history, default=str)

    # And no dispatcher log line leaks it (the input-file log is size-only).
    assert _LEAK_MARKER_B64 not in caplog.text


# ===========================================================================
# 3. Version skew: input_files against a model that predates the field
# ===========================================================================


class _LegacyDispatchRequest(BaseModel):
    """A stand-in for a worker request model that predates ``input_files``.

    Mirrors the pre-feature ``DispatchRequest`` field set exactly. Pydantic's
    default ``extra='ignore'`` means an ``input_files`` key in the incoming JSON
    is silently dropped rather than rejected — the exact skew a bridge must guard
    against by gating on the ``/health`` capability.
    """

    prompt: str
    allowed_tools: list[str]
    max_turns: int = 25
    surface_prompt: str | None = None
    surface_tools: list[str] | None = None


def test_input_files_skew_old_model_silently_drops_field():
    """A payload with input_files parsed by a pre-field model drops it silently."""
    wire_payload = {
        "prompt": "look",
        "allowed_tools": ["Read"],
        "input_files": [
            {"filename": "plot.png", "mime": "image/png", "content_b64": "QUJD", "ingest": True}
        ],
    }
    legacy = _LegacyDispatchRequest(**wire_payload)
    # Not rejected — constructed fine — but the batch is gone: the field does not
    # exist on the model and no error was raised.
    assert not hasattr(legacy, "input_files")
    assert "input_files" not in legacy.model_dump()
    assert legacy.prompt == "look"


def test_input_files_skew_current_model_captures_field():
    """Positive control: the current model DOES carry the batch, so the skew is
    strictly a property of the old model, not of the wire payload."""
    wire_payload = {
        "prompt": "look",
        "allowed_tools": ["Read"],
        "input_files": [
            {"filename": "plot.png", "mime": "image/png", "content_b64": "QUJD", "ingest": True}
        ],
    }
    current = DispatchRequest(**wire_payload)
    assert current.input_files is not None
    assert len(current.input_files) == 1
    assert current.input_files[0].filename == "plot.png"


# ===========================================================================
# 4. Fatal-4xx error_code propagation: webhook -> pool -> poll body
# ===========================================================================


class _FakeEntryPoint:
    def __init__(self, name: str, cls: type) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type:
        return self._cls


@pytest.fixture(autouse=True)
def _reset_mcp_routes():
    baseline = list(server.mcp._additional_http_routes)
    yield
    server.mcp._additional_http_routes = baseline


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "triggers.yml"
    path.write_text(
        "dispatcher:\n"
        "  dispatch_target: http://localhost:9999\n"
        "  max_concurrent_runs: 2\n"
        "  max_queue_depth: 10\n"
        "triggers:\n"
        "  - name: deploy\n"
        "    source: webhook\n"
        "    action:\n"
        "      prompt: do the thing\n"
        "      allowed_tools: []\n"
    )
    monkeypatch.setenv("TRIGGERS_YML", str(path))
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", "worker-secret")

    def fake_entry_points(*, group):
        assert group == "osprey.trigger_sources"
        return [_FakeEntryPoint("webhook", WebhookSource)]

    monkeypatch.setattr("osprey.dispatch.source_registry.entry_points", fake_entry_points)
    return server.create_server().http_app()


def _poll_until_terminal(client: TestClient, dispatch_id: str, tries: int = 50) -> dict:
    """Poll /dispatch/{id} until the pool result leaves 'pending' (or give up)."""
    body: dict = {}
    for _ in range(tries):
        resp = client.get(f"/dispatch/{dispatch_id}", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200
        body = resp.json()
        if body.get("status") != "pending":
            return body
    return body


def test_input_files_fatal_400_propagates_webhook_to_poll_body(app):
    """A worker 400 (input_files_cap_exceeded) surfaces on the dispatcher's poll
    body as a top-level error carrying the same error_code and trigger name."""

    async def _fake_dispatch(**kwargs):
        raise FatalDispatchError("HTTP 400 from worker", error_code="input_files_cap_exceeded")

    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        with TestClient(app) as client:
            resp = client.post(
                "/webhook/deploy",
                headers={"Authorization": "Bearer secret"},
                content=json.dumps(
                    {
                        "question": "hi",
                        "input_files": [
                            {
                                "filename": "big.png",
                                "mime": "image/png",
                                "content_b64": "QUJD",
                                "ingest": True,
                            }
                        ],
                    }
                ),
            )
            assert resp.status_code == 202
            dispatch_id = resp.json()["dispatch_id"]

            body = _poll_until_terminal(client, dispatch_id)

    # The fatal rejection is unwrapped to a TOP-LEVEL pool error, not a
    # completed run and not a silent drop.
    assert body["status"] == "error"
    assert body["error_code"] == "input_files_cap_exceeded"
    assert body["error"]  # human-readable message present (status-only, no body echo)
    assert body["trigger_name"] == "deploy"


def test_input_files_fatal_generic_4xx_propagates_null_error_code(app):
    """A non-whitelisted 4xx (e.g. denied tools) still surfaces as a top-level
    error, with error_code None rather than an echoed internal detail."""

    async def _fake_dispatch(**kwargs):
        raise FatalDispatchError("HTTP 403 from worker", error_code=None)

    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        with TestClient(app) as client:
            resp = client.post(
                "/webhook/deploy",
                headers={"Authorization": "Bearer secret"},
                content=json.dumps({"question": "hi"}),
            )
            assert resp.status_code == 202
            dispatch_id = resp.json()["dispatch_id"]
            body = _poll_until_terminal(client, dispatch_id)

    assert body["status"] == "error"
    assert body["error_code"] is None
    assert body["trigger_name"] == "deploy"
