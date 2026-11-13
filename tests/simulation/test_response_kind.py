"""The response ``kind`` carried from the callers to the engine.

The request resolver checks the kind before anything is read, the child payload
carries it, the child's solve passes it to an engine whose ``response_matrix``
takes it, calls one that takes none as before for the orbit kind and refuses
any other kind for it, and the ``model_response_matrix`` tool takes it as a
parameter whose default keeps every existing call unchanged. The render is the
SDK tests' two-cell ring.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

from osprey.mcp_server.python_executor import executor  # noqa: E402
from osprey.mcp_server.python_executor.tools import (  # noqa: E402
    model_response_matrix as tool,
)
from osprey.simulation import model_request  # noqa: E402
from osprey.simulation.response import (  # noqa: E402
    DEFAULT_KIND,
    RESPONSE_KINDS,
    RESULT_TAG,
    solve_and_print,
)
from tests.apps.test_sdk_model import RING, _ring_render  # noqa: E402
from tests.mcp_server.conftest import assert_raises_error  # noqa: E402

CORRECTOR = "COR_1:X:SP"


@pytest.fixture
def ring(tmp_path: Path) -> Path:
    return _ring_render(tmp_path / "render")


def _answer(rows: list[str], columns: list[str]) -> Any:
    return types.SimpleNamespace(
        rows=rows, columns=columns, matrix=[[1.0] * len(columns) for _ in rows], unsolved={}
    )


def _fake_engine(
    monkeypatch: pytest.MonkeyPatch, name: str, *, takes_kind: bool
) -> list[dict[str, Any]]:
    """Install an engine module whose ``response_matrix`` records each call."""
    calls: list[dict[str, Any]] = []
    module = types.ModuleType(name)

    if takes_kind:

        def response_matrix(_deck, _wiring, _settings, correctors, step, *, kind="orbit"):
            calls.append({"correctors": correctors, "step": step, "kind": kind})
            return _answer(["BPM_1:X"], correctors)

    else:

        def response_matrix(_deck, _wiring, _settings, correctors, step):  # type: ignore[misc]
            calls.append({"correctors": correctors, "step": step})
            return _answer(["BPM_1:X"], correctors)

    module.response_matrix = response_matrix  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return calls


def _printed(payload: dict[str, Any]) -> dict[str, Any]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        solve_and_print(payload)
    tag, body = out.getvalue().strip().split(" ", 1)
    assert tag == RESULT_TAG
    line: dict[str, Any] = json.loads(body)
    return line


def _payload(ring: Path, engine_name: str, **extra: Any) -> dict[str, Any]:
    return {
        "render_root": str(ring),
        "model": RING,
        "engine_module": engine_name,
        "correctors": [CORRECTOR],
        "steps": {CORRECTOR: 0.1},
        **extra,
    }


# ---------------------------------------------------------------------------
# The request and its payload
# ---------------------------------------------------------------------------


class TestRequest:
    def test_the_kinds_are_orbit_then_trajectory_and_orbit_is_the_default(self):
        assert RESPONSE_KINDS == ("orbit", "trajectory")
        assert DEFAULT_KIND == "orbit"

    def test_a_request_without_a_kind_is_the_orbit_kind_and_its_payload_says_so(self, ring):
        request = model_request.resolve_model_request(RING, None, render_root=ring)

        assert request.kind == "orbit"
        assert model_request.child_payload(request)["kind"] == "orbit"

    def test_the_trajectory_kind_is_carried_into_the_payload_and_the_script(self, ring):
        request = model_request.resolve_model_request(
            RING, None, render_root=ring, kind="trajectory"
        )

        assert request.kind == "trajectory"
        assert model_request.child_payload(request)["kind"] == "trajectory"
        assert '"kind": "trajectory"' in model_request.child_script(request)

    def test_the_kind_changes_nothing_else_in_the_request(self, ring):
        orbit = model_request.resolve_model_request(RING, None, render_root=ring)
        trajectory = model_request.resolve_model_request(
            RING, None, render_root=ring, kind="trajectory"
        )

        orbit_payload = model_request.child_payload(orbit)
        trajectory_payload = model_request.child_payload(trajectory)
        orbit_payload.pop("kind")
        trajectory_payload.pop("kind")
        assert orbit_payload == trajectory_payload

    @pytest.mark.parametrize("kind", ["tune", "", "ORBIT", None, 1, ["orbit"]])
    def test_an_unknown_kind_is_refused_naming_the_two(self, tmp_path, kind):
        # A render that does not exist: the kind is refused before anything is read.
        with pytest.raises(model_request.ModelRequestRefused) as refused:
            model_request.resolve_model_request(
                RING, None, render_root=tmp_path / "nowhere", kind=kind
            )

        assert "'orbit'" in refused.value.message
        assert "'trajectory'" in refused.value.message
        assert "kind" in refused.value.message
        assert "'orbit'" in refused.value.suggestion


# ---------------------------------------------------------------------------
# The child's solve
# ---------------------------------------------------------------------------


class TestChildSolve:
    @pytest.mark.parametrize("kind", RESPONSE_KINDS)
    def test_an_engine_that_takes_a_kind_is_passed_it(self, ring, monkeypatch, kind):
        calls = _fake_engine(monkeypatch, "kind_engine", takes_kind=True)

        line = _printed(_payload(ring, "kind_engine", kind=kind))

        assert calls == [{"correctors": [CORRECTOR], "step": {CORRECTOR: 0.1}, "kind": kind}]
        assert line["columns"] == [CORRECTOR]

    def test_a_payload_without_a_kind_asks_for_the_orbit_kind(self, ring, monkeypatch):
        calls = _fake_engine(monkeypatch, "kind_engine", takes_kind=True)

        _printed(_payload(ring, "kind_engine"))

        assert [call["kind"] for call in calls] == ["orbit"]

    def test_an_engine_without_a_kind_is_called_as_before_for_the_orbit_kind(
        self, ring, monkeypatch
    ):
        calls = _fake_engine(monkeypatch, "plain_engine", takes_kind=False)

        line = _printed(_payload(ring, "plain_engine", kind="orbit"))

        assert calls == [{"correctors": [CORRECTOR], "step": {CORRECTOR: 0.1}}]
        assert line["columns"] == [CORRECTOR]
        assert line["matrix"] == [[1.0]]

    def test_an_engine_without_a_kind_refuses_any_other_kind_naming_the_engine(
        self, ring, monkeypatch
    ):
        calls = _fake_engine(monkeypatch, "plain_engine", takes_kind=False)

        line = _printed(_payload(ring, "plain_engine", kind="trajectory"))

        assert calls == []
        assert line["cause"] == "request"
        assert "plain_engine" in line["refused"]
        assert "trajectory" in line["refused"]
        assert "orbit" in line["refused"]

    def test_an_engine_taking_keyword_arguments_is_passed_the_kind(self, ring, monkeypatch):
        seen: dict[str, Any] = {}
        module = types.ModuleType("keywords_engine")

        def response_matrix(_deck, _wiring, _settings, correctors, _step, **options):
            seen.update(options)
            return _answer(["BPM_1:X"], correctors)

        module.response_matrix = response_matrix  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)

        _printed(_payload(ring, "keywords_engine", kind="trajectory"))

        assert seen == {"kind": "trajectory"}


class TestThePyatEngine:
    def test_the_orbit_kind_named_or_not_solves_the_same_matrix(self, ring):
        request = model_request.resolve_model_request(RING, None, render_root=ring)
        named = model_request.resolve_model_request(RING, None, render_root=ring, kind="orbit")

        plain = _printed(model_request.child_payload(request))
        explicit = _printed(model_request.child_payload(named))

        assert plain["columns"]
        assert plain == explicit

    def test_the_trajectory_kind_reaches_the_engine_which_refuses_a_ring_without_turns(self, ring):
        request = model_request.resolve_model_request(
            RING, None, render_root=ring, kind="trajectory"
        )

        line = _printed(model_request.child_payload(request))

        assert line["cause"] == "request"
        assert "trajectory" in line["refused"]


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


def _solved(seen: dict[str, Any]) -> Any:
    line = {"rows": ["BPM_1:X"], "columns": [CORRECTOR], "matrix": [[2.0]], "missing": {}}

    async def execute_code(**arguments: Any) -> executor.ExecutionResult:
        seen["code"] = arguments["code"]
        return executor.ExecutionResult(
            success=True, stdout=f"{RESULT_TAG} {json.dumps(line)}\n", stderr=""
        )

    return execute_code


class TestTool:
    def test_the_kind_defaults_to_orbit(self):
        import inspect

        parameter = inspect.signature(tool.model_response_matrix).parameters["kind"]
        assert parameter.default == "orbit"

    def test_a_call_without_a_kind_runs_the_orbit_request(self, ring, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: ring)
        monkeypatch.setattr(executor, "execute_code", _solved(seen))

        result = json.loads(asyncio.run(tool.model_response_matrix(model=RING)))

        request = model_request.resolve_model_request(RING, None, render_root=ring)
        assert seen["code"] == model_request.child_script(request)
        assert '"kind": "orbit"' in seen["code"]
        assert result["matrix"] == [[2.0]]

    def test_the_trajectory_kind_reaches_the_child(self, ring, monkeypatch):
        seen: dict[str, Any] = {}
        monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: ring)
        monkeypatch.setattr(executor, "execute_code", _solved(seen))

        asyncio.run(tool.model_response_matrix(model=RING, kind="trajectory"))

        request = model_request.resolve_model_request(
            RING, None, render_root=ring, kind="trajectory"
        )
        assert seen["code"] == model_request.child_script(request)

    def test_an_unknown_kind_is_a_validation_error_before_the_child_starts(self, ring, monkeypatch):
        async def execute_code(**_: Any) -> executor.ExecutionResult:
            raise AssertionError("no child may start")

        monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: ring)
        monkeypatch.setattr(executor, "execute_code", execute_code)

        with assert_raises_error() as raised:
            asyncio.run(tool.model_response_matrix(model=RING, kind="tune"))

        envelope = raised["envelope"]
        assert envelope["error_type"] == "validation_error"
        assert "'tune'" in envelope["error_message"]
        assert "'trajectory'" in envelope["error_message"]

    def test_an_engine_refusal_of_the_kind_is_a_validation_error(self, ring, monkeypatch):
        refused = {"refused": "answers only the orbit response", "cause": "request"}

        async def execute_code(**_: Any) -> executor.ExecutionResult:
            return executor.ExecutionResult(
                success=True, stdout=f"{RESULT_TAG} {json.dumps(refused)}\n", stderr=""
            )

        monkeypatch.setattr("osprey.utils.config_paths.resolve_config_dir", lambda: ring)
        monkeypatch.setattr(executor, "execute_code", execute_code)

        with assert_raises_error() as raised:
            asyncio.run(tool.model_response_matrix(model=RING, kind="trajectory"))

        envelope = raised["envelope"]
        assert envelope["error_type"] == "validation_error"
        assert "answers only the orbit response" in envelope["error_message"]
