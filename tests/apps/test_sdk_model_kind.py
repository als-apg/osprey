"""The SDK's ``model_response_matrix`` takes the response ``kind``.

``kind`` is keyword-only with the orbit kind as its default, so every existing
call — positional or keyword — keeps today's request; any other known kind is
carried to the child, and an unknown one is refused before a child starts.
The child is stubbed; the render is the two-cell ring of ``test_sdk_model``.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("at")

from osprey.apps import sdk
from osprey.simulation import model_request
from osprey.simulation.response import RESULT_TAG
from tests.apps.test_sdk_model import RING, _ring_render

LINE = {"rows": ["BPM_1:X"], "columns": ["COR_1:X:SP"], "matrix": [[1.5]], "missing": {}}


@pytest.fixture
def ring(tmp_path: Path) -> Path:
    return _ring_render(tmp_path / "render")


def _stub(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any], stdout: str) -> None:
    def run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", run)


def test_kind_is_keyword_only_and_defaults_to_orbit():
    parameters = inspect.signature(sdk.model_response_matrix).parameters

    assert list(parameters)[:4] == ["model", "correctors", "render_root", "timeout_s"]
    assert parameters["kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["kind"].default == "orbit"


def test_a_positional_call_without_a_kind_runs_the_orbit_request(ring, monkeypatch):
    seen: dict[str, Any] = {}
    _stub(monkeypatch, seen, f"{RESULT_TAG} {json.dumps(LINE)}\n")

    result = sdk.model_response_matrix(RING, None, ring, 9)

    request = model_request.resolve_model_request(RING, None, render_root=ring)
    assert request.kind == "orbit"
    assert seen["argv"] == [sys.executable, "-c", model_request.child_script(request)]
    assert result.matrix == [[1.5]]


def test_the_trajectory_kind_reaches_the_child(ring, monkeypatch):
    seen: dict[str, Any] = {}
    _stub(monkeypatch, seen, f"{RESULT_TAG} {json.dumps(LINE)}\n")

    result = sdk.model_response_matrix(RING, render_root=ring, kind="trajectory")

    request = model_request.resolve_model_request(RING, None, render_root=ring, kind="trajectory")
    assert seen["argv"] == [sys.executable, "-c", model_request.child_script(request)]
    assert '"kind": "trajectory"' in seen["argv"][2]
    assert result.columns == ["COR_1:X:SP"]


def test_an_unknown_kind_is_refused_before_any_child(ring, monkeypatch):
    def run(*args, **kwargs):
        raise AssertionError("no child may start")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ValueError, match="'trajectory'") as refused:
        sdk.model_response_matrix(RING, render_root=ring, kind="tune")

    assert isinstance(refused.value, model_request.ModelRequestRefused)
    assert "'tune'" in str(refused.value)


def test_an_engine_refusal_of_the_kind_raises_value_error(ring, monkeypatch):
    refused = {"refused": "answers only the orbit response", "cause": "request"}
    _stub(monkeypatch, {}, f"{RESULT_TAG} {json.dumps(refused)}\n")

    with pytest.raises(ValueError, match="answers only the orbit response"):
        sdk.model_response_matrix(RING, render_root=ring, kind="trajectory")
