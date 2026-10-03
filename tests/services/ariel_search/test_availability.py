"""Tests for ``enhancement.availability`` and the picture-module pass of ``image_driver``.

The fakes here stand in for the repository and a picture module; the catch-up
tests in ``test_catchup.py`` reuse them.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.availability import (
    ModuleConfigError,
    ModuleUnavailable,
    preflight,
    unavailable_reason,
)
from osprey.services.ariel_search.enhancement.base import (
    BaseEnhancementModule,
    ImageEntryOutcome,
    as_health_result,
)
from osprey.services.ariel_search.enhancement.image_driver import drive_image_module

pytestmark = pytest.mark.asyncio

MARKER = "vision-model"


@pytest.fixture(autouse=True)
def _reset_availability():
    availability.reset_availability()
    _offload.reset_offload_state()
    yield
    availability.reset_availability()
    _offload.reset_offload_state()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    async def fetchone(self) -> Any:
        return self._row


class _Conn:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    async def execute(self, sql: str, params: Any = None) -> _Cursor:
        self._pool.statements.append((sql, params))
        relation = (params or {}).get("t")
        return _Cursor((relation if relation in self._pool.relations else None,))


class _ConnCtx:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    async def __aenter__(self) -> _Conn:
        return _Conn(self._pool)

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Pool:
    conninfo = "postgresql://fake/ariel"

    def __init__(self, relations: set[str] | None = None) -> None:
        self.relations = set(relations or ())
        self.statements: list[tuple[str, Any]] = []

    def connection(self) -> _ConnCtx:
        return _ConnCtx(self)


class FakeRepo:
    """Repository double: picture entries to do, picture-less ones to batch-mark."""

    def __init__(
        self,
        todo: list[str] | None = None,
        markable: list[str] | None = None,
        *,
        has_copy_state: bool = True,
        relations: set[str] | None = None,
    ) -> None:
        self.todo = list(todo or [])
        self.markable = sorted(markable or [])
        self.has_copy_state = has_copy_state
        self.pool = _Pool(relations)
        self.complete: set[str] = set()
        self.failed: dict[str, int] = {}
        self.mark_calls: list[str] = []
        self.todo_calls = 0
        self.mark_error: Exception | None = None

    async def schema_facts(self) -> SchemaFacts:
        return SchemaFacts(has_v2_fts=True, has_copy_state=self.has_copy_state)

    async def mark_image_module_complete_batch(
        self, module_name: str, marker: str, *, after: str = "", limit: int = 1000
    ) -> list[str]:
        del module_name, marker  # the faked signature
        self.mark_calls.append(after)
        if self.mark_error is not None:
            raise self.mark_error
        ids = [i for i in self.markable if i > after and i not in self.complete][:limit]
        self.complete.update(ids)
        return ids

    async def get_incomplete_entries(
        self,
        module_name: str | None = None,  # noqa: ARG002 - the faked signature
        status: str | None = None,  # noqa: ARG002 - the faked signature
        limit: int = 100,
        *,
        marker: str | None = None,  # noqa: ARG002 - the faked signature
    ) -> list[dict]:
        self.todo_calls += 1
        return [{"entry_id": i} for i in self.todo if i not in self.complete][:limit]

    async def mark_enhancement_failed(
        self, entry_id: str, module_name: str, error: str, *, marker: str | None = None
    ) -> int:
        del module_name, error, marker  # the faked signature
        self.failed[entry_id] = self.failed.get(entry_id, 0) + 1
        return self.failed[entry_id]


Step = Callable[["FakeModule", dict, Any], ImageEntryOutcome]


class FakeModule(BaseEnhancementModule):
    """A picture module whose health and per-entry behaviour the test scripts."""

    runs_inline = False

    def __init__(
        self,
        step: Step | None = None,
        *,
        name: str = "image_caption",
        health: Any = (True, "OK"),
        relations: list[str] | None = None,
        marker: str | None = MARKER,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._name = name
        self.step = step or (lambda mod, entry, gate: ImageEntryOutcome.done())
        self.health = health
        self.relations = relations or []
        self.marker = marker
        self.timeout_seconds = timeout_seconds
        self.run_calls: list[str] = []
        self.enhance_calls = 0
        self.health_calls = 0
        self.written: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def required_relations(self) -> list[str]:
        return list(self.relations)

    def completion_marker(self) -> str | None:
        return self.marker

    async def enhance(self, entry, conn) -> None:  # noqa: ARG002 - the enhancement module signature
        self.enhance_calls += 1
        raise NotImplementedError

    async def health_check(self):
        self.health_calls += 1
        health = self.health() if callable(self.health) else self.health
        if isinstance(health, BaseException):
            raise health
        return health

    async def run_entry(self, entry, repository, *, gate):
        self.run_calls.append(entry["entry_id"])
        outcome = self.step(self, entry, gate)
        if outcome.kind == "done":
            repository.complete.add(entry["entry_id"])
        return outcome


def one_picture(kind: str, signature: str = "HTTP 400 bad image") -> Step:
    """A step handling one picture per entry: ``ok`` stores it, ``det`` fails deterministically."""

    def _step(mod: FakeModule, entry: dict, gate: Any) -> ImageEntryOutcome:
        if not gate.may_start_picture():
            return ImageEntryOutcome.partial()
        if kind == "ok":
            mod.written.append((entry["entry_id"], "caption"))
            gate.succeeded()
            return ImageEntryOutcome.done()
        if gate.deterministic(signature):
            mod.written.append((entry["entry_id"], "error"))
            return ImageEntryOutcome.done()
        return ImageEntryOutcome.partial()

    return _step


def scripted(kinds: list[str], signature: str = "HTTP 400 bad image") -> Step:
    """A one-picture step whose kind is taken in order, one per call."""
    queue = list(kinds)

    def _step(mod: FakeModule, entry: dict, gate: Any) -> ImageEntryOutcome:
        return one_picture(queue.pop(0), signature)(mod, entry, gate)

    return _step


async def drive(module: FakeModule, repo: FakeRepo, **kw: Any):
    return await drive_image_module(
        module, repo, budget=kw.pop("budget", None), stop_event=kw.pop("stop_event", None), **kw
    )


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level and r.name.startswith("ariel")]


# ---------------------------------------------------------------------------
# unavailable_reason -- one row per branch
# ---------------------------------------------------------------------------


class _ConnAndValue(ConnectionError, ValueError):
    pass


def _http_error(status: int):
    import requests

    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=response)


async def test_reason_module_unavailable_carries_its_own_reason() -> None:
    assert unavailable_reason(ModuleUnavailable("model", "no such model")) == "model"


async def test_reason_module_config_error_is_config() -> None:
    err = ModuleConfigError("bad", key="ariel.enhancement_modules.image_caption.model")
    assert isinstance(err, ValueError)
    assert unavailable_reason(err) == "config"


async def test_reason_embedding_dimension_error_is_config() -> None:
    from osprey.models.providers.base import EmbeddingDimensionError

    assert unavailable_reason(EmbeddingDimensionError("1024 != 768")) == "config"


@pytest.mark.parametrize("name", ["UndefinedTable", "UndefinedColumn"])
async def test_reason_missing_relation_is_config(name: str) -> None:
    import psycopg.errors

    assert unavailable_reason(getattr(psycopg.errors, name)("missing")) == "config"


async def test_reason_wrapped_database_error_is_classified_by_its_cause() -> None:
    import psycopg.errors

    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    try:
        try:
            raise psycopg.errors.UndefinedTable("image_embeddings_x")
        except psycopg.errors.UndefinedTable as inner:
            raise DatabaseQueryError("query failed") from inner
    except DatabaseQueryError as outer:
        assert unavailable_reason(outer) == "config"


async def test_reason_query_canceled_is_unreachable() -> None:
    import psycopg.errors

    assert unavailable_reason(psycopg.errors.QueryCanceled("timeout")) == "unreachable"


async def test_reason_pool_timeout_is_unreachable() -> None:
    from psycopg_pool import PoolTimeout

    assert unavailable_reason(PoolTimeout("no connection")) == "unreachable"


async def test_reason_connection_and_value_error_is_unreachable() -> None:
    assert unavailable_reason(_ConnAndValue("refused")) == "unreachable"


async def test_reason_local_server_classes_are_unreachable() -> None:
    from osprey.models.providers._local_server import LocalServerUnreachable
    from osprey.models.providers.ollama import OllamaUnreachableError

    assert unavailable_reason(OllamaUnreachableError("down")) == "unreachable"
    assert unavailable_reason(LocalServerUnreachable("down")) == "unreachable"


@pytest.mark.parametrize(("status", "reason"), [(401, "auth"), (403, "auth"), (404, "model")])
async def test_reason_http_status(status: int, reason: str) -> None:
    assert unavailable_reason(_http_error(status)) == reason


@pytest.mark.parametrize(
    "exc", [ValueError("400 unknown model"), RuntimeError("x"), _http_error(429)]
)
async def test_reason_anything_else_is_none(exc: Exception) -> None:
    assert unavailable_reason(exc) is None


# ---------------------------------------------------------------------------
# as_health_result and preflight
# ---------------------------------------------------------------------------


async def test_as_health_result_adapts_a_pair_and_passes_a_result_through() -> None:
    assert as_health_result((0, "down")) == HealthResult(False, "down", None)
    verdict = HealthResult(False, "401", "auth")
    assert as_health_result(verdict) is verdict


async def test_preflight_schema_behind_issues_no_statement() -> None:
    repo = FakeRepo(has_copy_state=False)
    module = FakeModule(relations=["image_embeddings_x"])
    result = await preflight(module, repo)
    assert result == HealthResult(
        False, "attachment copy state not migrated: run osprey ariel migrate", "config"
    )
    assert repo.pool.statements == []
    assert module.health_calls == 0


async def test_preflight_missing_relation_is_config_and_skips_health() -> None:
    repo = FakeRepo()
    module = FakeModule(relations=["image_embeddings_x"])
    result = await preflight(module, repo)
    assert result.reachable is False and result.reason == "config"
    assert result.message.startswith("image_embeddings_x missing: pgvector unavailable")
    assert module.health_calls == 0


async def test_preflight_caches_a_present_relation_per_pool() -> None:
    repo = FakeRepo(relations={"image_embeddings_x"})
    module = FakeModule(relations=["image_embeddings_x"])
    assert (await preflight(module, repo)).reachable is True
    assert (await preflight(module, repo)).reachable is True
    assert len(repo.pool.statements) == 1


async def test_preflight_passes_a_pair_through_as_health_result() -> None:
    result = await preflight(FakeModule(health=(False, "server down")), FakeRepo())
    assert result == HealthResult(False, "server down", "unreachable")
    assert await preflight(FakeModule(), FakeRepo()) == HealthResult(True, "OK", None)


async def test_preflight_classifies_a_raising_health_check() -> None:
    result = await preflight(FakeModule(health=_http_error(401)), FakeRepo())
    assert result.reachable is False and result.reason == "auth"
    result = await preflight(FakeModule(health=RuntimeError("weird")), FakeRepo())
    assert result.reason == "unreachable"


async def test_preflight_returns_within_six_seconds_on_a_blocking_health_body() -> None:
    release = threading.Event()

    class _Blocking(FakeModule):
        async def health_check(self):
            await _offload.run_blocking(release.wait, 30.0, key=self.name)
            return (True, "OK")

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.1)
            ticks += 1

    ticker = asyncio.ensure_future(_ticker())
    start = time.monotonic()
    try:
        result = await preflight(_Blocking(), FakeRepo())
        elapsed = time.monotonic() - start
    finally:
        ticker.cancel()
        release.set()
    assert result.reachable is False and result.reason == "unreachable"
    assert elapsed < 6.0
    assert ticks >= 30  # the loop kept running while the health body blocked


# ---------------------------------------------------------------------------
# The tracker through the driver: one WARNING per transition, never ERROR
# ---------------------------------------------------------------------------


async def test_schema_behind_makes_no_batch_mark_or_todo_call_and_warns_once(caplog) -> None:
    repo = FakeRepo(todo=["e1"], has_copy_state=False)
    module = FakeModule()
    with caplog.at_level(logging.DEBUG, logger="ariel"):
        for _ in range(3):
            await drive(module, repo)
    assert repo.mark_calls == [] and repo.todo_calls == 0
    assert len(_records(caplog, logging.WARNING)) == 1
    assert "osprey ariel migrate" in _records(caplog, logging.WARNING)[0].getMessage()
    assert _records(caplog, logging.ERROR) == []


async def test_undefined_table_in_a_driver_statement_ends_the_pass_quietly(caplog) -> None:
    import psycopg.errors

    repo = FakeRepo(todo=["e1"])
    repo.mark_error = psycopg.errors.UndefinedTable("attachment_files")
    module = FakeModule()
    with caplog.at_level(logging.DEBUG, logger="ariel"):
        for _ in range(3):
            result = await drive(module, repo)
    assert result.ended == "config"
    assert module.run_calls == [] and repo.failed == {}
    assert len(_records(caplog, logging.WARNING)) == 1
    assert _records(caplog, logging.ERROR) == []


async def test_module_error_auth_with_healthy_check_warns_once(caplog) -> None:
    repo = FakeRepo(todo=["e1", "e2"])
    module = FakeModule(lambda mod, entry, gate: ImageEntryOutcome.module_error("auth"))
    with caplog.at_level(logging.DEBUG, logger="ariel"):
        for _ in range(3):
            await drive(module, repo)
    assert module.run_calls == ["e1", "e1", "e1"]
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1 and "auth" in warnings[0].getMessage()
    assert _records(caplog, logging.ERROR) == []


async def test_unreachable_module_warns_once_then_info_on_recovery(caplog) -> None:
    repo = FakeRepo(todo=["e1"])
    module = FakeModule(health=HealthResult(False, "connection refused", "unreachable"))
    with caplog.at_level(logging.DEBUG, logger="ariel"):
        for _ in range(3):
            result = await drive(module, repo)
            assert result.skipped == "unavailable"
        assert module.run_calls == [] and repo.failed == {}
        module.health = (True, "OK")
        await drive(module, repo)
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "image_caption" in warnings[0].getMessage()
    assert "unreachable" in warnings[0].getMessage()
    assert "ariel.enhancement_modules.image_caption" in warnings[0].getMessage()
    infos = [r for r in _records(caplog, logging.INFO) if "available again" in r.getMessage()]
    assert len(infos) == 1
    assert _records(caplog, logging.ERROR) == []


async def test_a_changed_reason_warns_again(caplog) -> None:
    repo = FakeRepo()
    module = FakeModule(health=HealthResult(False, "down", "unreachable"))
    with caplog.at_level(logging.WARNING, logger="ariel"):
        await drive(module, repo)
        module.health = HealthResult(False, "401", "auth")
        await drive(module, repo)
        await drive(module, repo)
    assert len(_records(caplog, logging.WARNING)) == 2


async def test_no_marker_is_config() -> None:
    result = await drive(FakeModule(marker=None), FakeRepo(todo=["e1"]))
    assert result.ended == "config"


async def test_busy_offload_key_skips_the_pass(caplog) -> None:
    release = threading.Event()
    task = asyncio.ensure_future(_offload.run_blocking(release.wait, 10.0, key="image_caption"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        module = FakeModule()
        with caplog.at_level(logging.INFO, logger="ariel"):
            result = await drive(module, FakeRepo(todo=["e1"]))
        assert result.skipped == "busy"
        assert module.health_calls == 0 and module.run_calls == []
        assert (
            len([r for r in _records(caplog, logging.INFO) if "still running" in r.getMessage()])
            == 1
        )
    finally:
        release.set()


# ---------------------------------------------------------------------------
# Outcome table and pass breakers
# ---------------------------------------------------------------------------


async def test_two_connection_resets_charge_only_the_first_entry() -> None:
    repo = FakeRepo(todo=["e1", "e2", "e3"])
    module = FakeModule(lambda mod, entry, gate: ImageEntryOutcome.unavailable("unreachable"))
    result = await drive(module, repo)
    assert module.run_calls == ["e1", "e2"]
    assert repo.failed == {"e1": 1}
    assert result.ended == "transient"


async def test_429_on_every_call_charges_one_entry_one_attempt_per_pass() -> None:
    repo = FakeRepo(todo=["e1", "e2", "e3"])
    module = FakeModule(lambda mod, entry, gate: ImageEntryOutcome.transient_error("HTTP 429"))
    await drive(module, repo)
    assert repo.failed == {"e1": 1}
    await drive(module, repo)
    assert repo.failed == {"e1": 2}


async def test_a_done_between_transients_resets_the_count() -> None:
    repo = FakeRepo(todo=["e1", "e2", "e3"])
    kinds = iter(["t", "ok", "t"])

    def _step(mod, entry, gate):  # noqa: ARG001 - the faked signature
        return (
            ImageEntryOutcome.done()
            if next(kinds) == "ok"
            else ImageEntryOutcome.transient_error("reset")
        )

    await drive(FakeModule(_step), repo)
    assert repo.failed == {"e1": 1, "e3": 1}


async def test_unavailable_while_health_turns_false_leaves_the_entry_untouched() -> None:
    repo = FakeRepo(todo=["e1", "e2"])
    health = iter([(True, "OK"), HealthResult(False, "gone", "unreachable")])
    module = FakeModule(
        lambda mod, entry, gate: ImageEntryOutcome.unavailable("unreachable"),
        health=lambda: next(health),
    )
    result = await drive(module, repo)
    assert module.run_calls == ["e1"]
    assert repo.failed == {}
    assert result.ended == "unreachable"


@pytest.mark.parametrize("reason", ["auth", "model", "config"])
async def test_authoritative_unavailable_ends_the_pass_uncharged(reason: str) -> None:
    repo = FakeRepo(todo=["e1", "e2"])
    module = FakeModule(lambda mod, entry, gate: ImageEntryOutcome.unavailable(reason))
    result = await drive(module, repo)
    assert module.run_calls == ["e1"] and repo.failed == {}
    assert result.ended == reason
    assert availability.current_reason("image_caption") == reason


async def test_same_deterministic_failure_before_any_success_writes_nothing_and_reports_model(
    caplog,
) -> None:
    repo = FakeRepo(todo=["e1", "e2", "e3", "e4"])
    module = FakeModule(one_picture("det"))
    with caplog.at_level(logging.WARNING, logger="ariel"):
        result = await drive(module, repo)
    assert module.run_calls == ["e1", "e2", "e3"]
    assert module.written == []
    assert result.ended == "model"
    assert availability.current_reason("image_caption") == "model"
    assert len(_records(caplog, logging.WARNING)) == 1


async def test_after_one_success_two_failures_are_written_and_the_third_ends_the_pass() -> None:
    repo = FakeRepo(todo=["e0", "e1", "e2", "e3", "e4"])
    module = FakeModule(scripted(["ok", "det", "det", "det", "det"]))
    result = await drive(module, repo)
    assert module.written == [("e0", "caption"), ("e1", "error"), ("e2", "error")]
    assert module.run_calls == ["e0", "e1", "e2", "e3"]
    assert result.ended == "HTTP 400 bad image"
    assert availability.current_reason("image_caption") == "HTTP 400 bad image"


async def test_one_deterministic_failure_among_successes_is_written() -> None:
    repo = FakeRepo(todo=["e0", "e1", "e2"])
    module = FakeModule(scripted(["ok", "det", "ok"]))
    result = await drive(module, repo)
    assert module.written == [("e0", "caption"), ("e1", "error"), ("e2", "caption")]
    assert result.ended is None


async def test_budget_is_checked_before_each_picture() -> None:
    repo = FakeRepo(todo=["e1", "e2"])
    module = FakeModule(one_picture("ok"))
    result = await drive(module, repo, budget=4.0)  # under the 5 s floor
    assert module.written == []
    assert result.ended == "budget"


async def test_stop_event_is_checked_before_each_entry() -> None:
    stop = asyncio.Event()

    def _step(mod, entry, gate):
        stop.set()
        return one_picture("ok")(mod, entry, gate)

    repo = FakeRepo(todo=["e1", "e2", "e3"])
    module = FakeModule(_step)
    result = await drive(module, repo, stop_event=stop)
    assert module.run_calls == ["e1"]
    assert result.ended == "stop"


async def test_the_driver_never_calls_enhance() -> None:
    module = FakeModule(one_picture("ok"))
    await drive(module, FakeRepo(todo=["e1", "e2"]))
    assert module.enhance_calls == 0
    assert module.run_calls == ["e1", "e2"]
