"""Unit tests for :mod:`osprey.services.python_executor.models`.

Focuses on behavioural contracts rather than trivial field readback:

* ``NotebookAttempt.to_dict`` / ``PythonExecutionSuccess.to_dict`` serialisation
  (enum-to-value, ``Path``-to-``str``, derived counts, renamed keys).
* ``PythonExecutionContext`` initialisation flag and attempt bookkeeping.
* ``PythonExecutionRequest`` Pydantic defaults, required-field validation, and
  JSON round-trip.
* ``PythonExecutionEngineResult`` ``__post_init__`` None-normalisation.
* ``PythonServiceResult`` frozen/immutable guarantee.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from osprey.services.python_executor.models import (
    NotebookAttempt,
    NotebookType,
    PythonExecutionContext,
    PythonExecutionEngineResult,
    PythonExecutionRequest,
    PythonExecutionSuccess,
    PythonServiceResult,
)


class TestNotebookAttempt:
    def test_to_dict_serialises_enum_and_path(self):
        attempt = NotebookAttempt(
            notebook_type=NotebookType.FINAL_SUCCESS,
            attempt_number=2,
            stage="execution",
            notebook_path=Path("/tmp/exec/nb.ipynb"),
            notebook_link="http://jupyter/nb.ipynb",
        )
        data = attempt.to_dict()
        assert data["notebook_type"] == "final_success"
        assert data["notebook_path"] == "/tmp/exec/nb.ipynb"
        assert isinstance(data["notebook_path"], str)
        assert data["attempt_number"] == 2
        # Unset optional fields serialise as None.
        assert data["error_context"] is None
        assert data["created_at"] is None


class TestPythonExecutionContext:
    def test_not_initialised_by_default(self):
        ctx = PythonExecutionContext()
        assert ctx.is_initialized is False

    def test_initialised_once_folder_set(self):
        ctx = PythonExecutionContext(folder_path=Path("/tmp/exec"))
        assert ctx.is_initialized is True

    def test_attempt_numbering_is_sequential(self):
        ctx = PythonExecutionContext()
        assert ctx.get_next_attempt_number() == 1
        ctx.add_notebook_attempt(
            NotebookAttempt(
                notebook_type=NotebookType.PRE_EXECUTION,
                attempt_number=1,
                stage="execution",
                notebook_path=Path("/tmp/nb.ipynb"),
                notebook_link="http://x",
            )
        )
        assert ctx.get_next_attempt_number() == 2
        assert len(ctx.notebook_attempts) == 1

    def test_default_attempts_list_not_shared_between_instances(self):
        a = PythonExecutionContext()
        b = PythonExecutionContext()
        a.add_notebook_attempt(
            NotebookAttempt(
                notebook_type=NotebookType.PRE_EXECUTION,
                attempt_number=1,
                stage="execution",
                notebook_path=Path("/tmp/nb.ipynb"),
                notebook_link="http://x",
            )
        )
        assert b.notebook_attempts == []


class TestPythonExecutionRequest:
    def _minimal(self, **overrides) -> PythonExecutionRequest:
        base = {
            "user_query": "q",
            "task_objective": "do the thing",
            "execution_folder_name": "folder",
        }
        base.update(overrides)
        return PythonExecutionRequest(**base)

    def test_defaults_applied(self):
        req = self._minimal()
        assert req.retries == 3
        assert req.expected_results == {}
        assert req.capability_prompts == []
        assert req.approved_code is None
        assert req.capability_context_data is None

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            PythonExecutionRequest(user_query="q", task_objective="t")  # no folder name

    def test_json_round_trip(self):
        req = self._minimal(
            retries=5,
            expected_results={"stats": "dict"},
            approved_code="print(1)",
        )
        restored = PythonExecutionRequest.model_validate_json(req.model_dump_json())
        assert restored == req
        assert restored.retries == 5
        assert restored.approved_code == "print(1)"

    def test_default_collections_not_shared(self):
        a = self._minimal()
        b = self._minimal()
        a.capability_prompts.append("x")
        assert b.capability_prompts == []


class TestPythonExecutionSuccess:
    def test_to_dict_renames_and_derives_counts(self):
        success = PythonExecutionSuccess(
            results={"mean": 42.0},
            stdout="done",
            execution_time=2.5,
            folder_path=Path("/tmp/exec"),
            notebook_path=Path("/tmp/exec/nb.ipynb"),
            notebook_link="http://jupyter/nb.ipynb",
            figure_paths=[Path("/tmp/exec/fig1.png"), Path("/tmp/exec/fig2.png")],
        )
        data = success.to_dict()
        # Keys are renamed for backward compatibility.
        assert data["execution_stdout"] == "done"
        assert data["execution_time_seconds"] == 2.5
        assert data["execution_folder"] == "/tmp/exec"
        assert data["notebook_path"] == "/tmp/exec/nb.ipynb"
        # Paths become strings; count is derived.
        assert data["figure_paths"] == ["/tmp/exec/fig1.png", "/tmp/exec/fig2.png"]
        assert data["figure_count"] == 2

    def test_to_dict_empty_figures(self):
        success = PythonExecutionSuccess(
            results={},
            stdout="",
            execution_time=0.0,
            folder_path=Path("/tmp/exec"),
            notebook_path=Path("/tmp/exec/nb.ipynb"),
            notebook_link="http://x",
        )
        data = success.to_dict()
        assert data["figure_paths"] == []
        assert data["figure_count"] == 0


class TestPythonExecutionEngineResult:
    def test_captured_figures_defaults_to_empty_list(self):
        result = PythonExecutionEngineResult(success=True, stdout="ok")
        assert result.captured_figures == []

    def test_post_init_normalises_none_figures(self):
        # __post_init__ replaces an explicit None with an empty list.
        result = PythonExecutionEngineResult(success=False, stdout="", captured_figures=None)
        assert result.captured_figures == []


class TestPythonServiceResult:
    def _result(self) -> PythonServiceResult:
        success = PythonExecutionSuccess(
            results={},
            stdout="",
            execution_time=0.0,
            folder_path=Path("/tmp/exec"),
            notebook_path=Path("/tmp/exec/nb.ipynb"),
            notebook_link="http://x",
        )
        return PythonServiceResult(execution_result=success, generated_code="print(1)")

    def test_defaults(self):
        result = self._result()
        assert result.generation_attempt == 1
        assert result.analysis_warnings == []

    def test_is_frozen(self):
        result = self._result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.generated_code = "mutated"  # type: ignore[misc]
