"""Tests for the core ``model_chat`` health category (task 4.8).

Pins every grading branch (ok / empty / provider-timeout / wait_for-expiry /
error), the zero-models short-circuit, unique-pair extraction, and the per-item
budget arithmetic (``60 × max(N,1)`` category budget → per-item = budget/N;
explicit override wins). ``run_sync`` is patched at module scope both to capture
the per-item timeout and to synthesize each completion outcome without real
threads or wall-clock waits; one test drives the real offload path to pin the
``get_chat_completion`` wiring.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.health.core import model_chat as model_chat_mod
from osprey.health.core.model_chat import model_chat
from osprey.health.models import CheckResult, Status

MODELS_3 = {
    "fast": {"provider": "anthropic", "model_id": "claude-x"},
    "smart": {"provider": "openai", "model_id": "gpt-y"},
    "cheap": {"provider": "google", "model_id": "gemini-z"},
}


def _cfg(models: Any, override: float | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"models": models}
    if override is not None:
        cfg["health"] = {"categories": {"model_chat": {"timeout_s": override}}}
    return cfg


async def _run(config: Any) -> dict[str, CheckResult]:
    results = await model_chat(config)()
    assert isinstance(results, list)
    return {r.name: r for r in results}


def _patch_run_sync(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: Any = "OK",
    exc: BaseException | None = None,
    recorder: list[tuple[tuple[Any, ...], float]] | None = None,
) -> None:
    """Replace the module-level ``run_sync`` with an async fake.

    Records ``(args, timeout_s)`` per call, then either raises ``exc`` or returns
    ``result`` — no daemon thread, no real completion.
    """

    async def fake_run_sync(fn: Any, *args: Any, timeout_s: float) -> Any:
        if recorder is not None:
            recorder.append((args, timeout_s))
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr(model_chat_mod, "run_sync", fake_run_sync)


# --- Zero models ------------------------------------------------------------


async def test_no_models_short_circuits_to_single_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch)
    for config in ({}, {"models": {}}, {"models": None}, None):
        by_name = await _run(config)
        assert list(by_name) == ["model_chat"]
        row = by_name["model_chat"]
        assert row.status is Status.SKIP
        assert row.category == "model_chat"
        assert "no models configured" in row.message


async def test_non_dict_model_entries_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch)
    models = {
        "good": {"provider": "anthropic", "model_id": "claude-x"},
        "junk": "not-a-dict",
        "partial": {"provider": "openai"},  # no model_id → excluded
    }
    by_name = await _run(_cfg(models))
    assert list(by_name) == ["model_chat_anthropic_claude-x"]


# --- Grading branches -------------------------------------------------------


async def test_non_empty_string_response_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch, result="OK")
    row = (await _run(_cfg(MODELS_3)))["model_chat_anthropic_claude-x"]
    assert row.status is Status.OK
    assert "anthropic/claude-x" in row.message
    assert "successful" in row.message


@pytest.mark.parametrize("empty", ["", "   ", 123, ["OK"], None])
async def test_empty_or_non_string_response_warns(
    monkeypatch: pytest.MonkeyPatch, empty: Any
) -> None:
    _patch_run_sync(monkeypatch, result=empty)
    row = (await _run(_cfg({"m": {"provider": "p", "model_id": "x"}})))["model_chat_p_x"]
    assert row.status is Status.WARNING
    assert "Empty response" in row.message


async def test_wait_for_expiry_maps_to_warning_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch, exc=TimeoutError())
    row = (await _run(_cfg({"m": {"provider": "p", "model_id": "x"}})))["model_chat_p_x"]
    assert row.status is Status.WARNING
    assert row.message == "p/x: Timeout"


@pytest.mark.parametrize("message", ["Connection timed out", "Request TIMEOUT after 60s"])
async def test_provider_raised_timeout_maps_to_warning(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    _patch_run_sync(monkeypatch, exc=RuntimeError(message))
    row = (await _run(_cfg({"m": {"provider": "p", "model_id": "x"}})))["model_chat_p_x"]
    assert row.status is Status.WARNING
    assert row.message == "p/x: Timeout"


async def test_other_failure_is_error_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch, exc=ValueError("bad api key"))
    row = (await _run(_cfg({"m": {"provider": "p", "model_id": "x"}})))["model_chat_p_x"]
    assert row.status is Status.ERROR
    assert "p/x: bad api key" == row.message


# --- Unique extraction, ordering, multiplicity ------------------------------


async def test_duplicate_pairs_dedupe_and_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_sync(monkeypatch, result="OK")
    models = {
        "a": {"provider": "openai", "model_id": "gpt-y"},
        "b": {"provider": "anthropic", "model_id": "claude-x"},
        "dup": {"provider": "openai", "model_id": "gpt-y"},  # same pair as "a"
    }
    by_name = await _run(_cfg(models))
    # Deduped to two rows, emitted in sorted (provider, model_id) order.
    assert list(by_name) == [
        "model_chat_anthropic_claude-x",
        "model_chat_openai_gpt-y",
    ]


# --- Per-item budget arithmetic ---------------------------------------------


async def test_default_per_item_budget_is_60s(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: list[tuple[tuple[Any, ...], float]] = []
    _patch_run_sync(monkeypatch, result="OK", recorder=recorder)
    await _run(_cfg(MODELS_3))
    assert len(recorder) == 3
    assert all(timeout_s == 60.0 for _args, timeout_s in recorder)


async def test_override_divides_budget_across_items(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: list[tuple[tuple[Any, ...], float]] = []
    _patch_run_sync(monkeypatch, result="OK", recorder=recorder)
    # 3 models with an explicit category timeout_s of 60 → per-item = 20.
    await _run(_cfg(MODELS_3, override=60))
    assert len(recorder) == 3
    assert all(timeout_s == 20.0 for _args, timeout_s in recorder)


async def test_run_sync_receives_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: list[tuple[tuple[Any, ...], float]] = []
    _patch_run_sync(monkeypatch, result="OK", recorder=recorder)
    await _run(_cfg({"m": {"provider": "anthropic", "model_id": "claude-x"}}))
    (args, _timeout_s) = recorder[0]
    assert args == ("anthropic", "claude-x")


# --- Real offload path: get_chat_completion wiring --------------------------


async def test_real_offload_invokes_get_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the real ``run_sync`` daemon-thread path and pin the completion call."""
    import osprey.models.completion as completion_mod

    captured: dict[str, Any] = {}

    def fake_get_chat_completion(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "OK"

    monkeypatch.setattr(completion_mod, "get_chat_completion", fake_get_chat_completion)

    row = (await _run(_cfg({"m": {"provider": "anthropic", "model_id": "claude-x"}})))[
        "model_chat_anthropic_claude-x"
    ]
    assert row.status is Status.OK
    assert captured["message"] == "Reply with exactly: OK"
    assert captured["provider"] == "anthropic"
    assert captured["model_id"] == "claude-x"
    assert captured["max_tokens"] == 50
