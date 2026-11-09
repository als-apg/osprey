"""Tests for the catch-up pass: ``run_catchup``, ``catchup_budget`` and the poll cadence.

The repository and picture-module fakes come from ``test_availability.py``;
here they run behind ``run_catchup`` with the factory, the service, the
registry and the advisory lock patched at their owning modules.
"""

from __future__ import annotations

import asyncio
import logging
import time
import types
from contextlib import asynccontextmanager
from typing import Any

import pytest

from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.enhancement import _offload, availability, image_driver
from osprey.services.ariel_search.enhancement.base import ImageEntryOutcome
from osprey.services.ariel_search.exceptions import ModuleConfigError
from tests.services.ariel_search._cli_ops_doubles import (
    _Enhancer,
    _patch_migrations,
    _patch_pool,
    _patch_service,
    _StubService,
)
from tests.services.ariel_search.test_availability import FakeModule, FakeRepo
from tests.services.ariel_search.test_cli_operations_pipeline import (
    _patch_scheduler,
    _poll_result,
)

pytestmark = pytest.mark.asyncio

IMAGE_MODULES = ("image_caption", "image_embedding")


@pytest.fixture(autouse=True)
def _reset_availability():
    availability.reset_availability()
    _offload.reset_offload_state()
    yield
    availability.reset_availability()
    _offload.reset_offload_state()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class CatchupRepo(FakeRepo):
    """``FakeRepo`` that also serves the text stage (reads without a marker)."""

    def __init__(
        self,
        *,
        text_todo: list[str] | None = None,
        text_status: dict[str, dict[str, Any]] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.text_todo = list(text_todo or [])
        #: ``enhancement_status`` per text entry id; absent ids carry none.
        self.text_status = dict(text_status or {})
        self.text_complete: list[tuple[str, str]] = []
        self.text_failed: list[tuple[str, str]] = []
        #: ``(module_name, marker)`` of every text read.
        self.text_reads: list[tuple[str | None, str | None]] = []
        self.stamped: list[dict[str, str]] = []

    async def stamp_baseline_markers(self, markers: dict[str, str]) -> None:
        self.stamped.append(dict(markers))

    async def get_incomplete_entries(
        self,
        module_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
        *,
        marker: str | None = None,
    ) -> list[dict]:
        if module_name not in IMAGE_MODULES:
            self.text_reads.append((module_name, marker))
            done = {entry_id for entry_id, name in self.text_complete if name == module_name}

            def owed(entry_id: str) -> bool:
                if entry_id in done:
                    return False
                if marker is not None:
                    return True
                # The marker-less read never returns a completed key, whatever its marker.
                state = self.text_status.get(entry_id, {}).get(module_name)
                return not (isinstance(state, dict) and state.get("status") == "complete")

            # ``text_todo`` is the order the real catch-up query returns.
            return [
                {
                    "entry_id": i,
                    "raw_text": f"text of {i}",
                    "attachments": [],
                    "enhancement_status": self.text_status.get(i, {}),
                }
                for i in self.text_todo
                if owed(i)
            ][:limit]
        return await super().get_incomplete_entries(module_name, status, limit, marker=marker)

    async def mark_enhancement_complete(self, entry_id: str, module_name: str, **_kw: Any) -> None:
        self.text_complete.append((entry_id, module_name))

    async def mark_enhancement_failed(
        self, entry_id: str, module_name: str, error: str, *, marker: str | None = None
    ) -> int:
        if marker is None:
            self.text_failed.append((entry_id, module_name))
            return 1
        return await super().mark_enhancement_failed(entry_id, module_name, error, marker=marker)


class Locks:
    """Fake ``try_advisory_lock``: records each acquire and release."""

    def __init__(self, busy: set[str] | None = None) -> None:
        self.busy = set(busy or ())
        self.held: set[str] = set()
        self.events: list[tuple[str, str]] = []

    @asynccontextmanager
    async def __call__(self, conninfo: str, key: str, *, wait: bool = False):  # noqa: ARG002 - the faked signature
        assert conninfo
        if key in self.busy:
            yield False
            return
        self.held.add(key)
        self.events.append(("acquire", key))
        try:
            yield True
        finally:
            self.held.discard(key)
            self.events.append(("release", key))


class Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def _config(poll_interval: float = 10.0, budget: Any = "unset", **modules: bool) -> dict:
    enabled = {"text_embedding": True, "image_caption": True, "image_embedding": False}
    enabled.update(modules)
    cfg: dict[str, Any] = {
        "database": {"uri": "postgresql://fake/ariel"},
        "enhancement_modules": {name: {"enabled": on} for name, on in enabled.items()},
        "ingestion": {"adapter": "generic_json", "poll_interval_seconds": poll_interval},
    }
    if budget != "unset":
        cfg["enhancement"] = {"catchup_budget_seconds": budget}
    return cfg


def _register_image_modules() -> None:
    """Add the two picture modules (``runs_inline=False``) to the mocked registry."""
    from osprey.registry import get_registry
    from osprey.registry.base import ArielEnhancementModuleRegistration

    registry = get_registry()
    table = dict(registry.get_ariel_enhancement_module.side_effect.__self__)
    for order, name in ((40, "image_caption"), (50, "image_embedding")):
        table[name] = (
            FakeModule,
            ArielEnhancementModuleRegistration(
                name=name,
                module_path=__name__,
                class_name="FakeModule",
                description=name,
                execution_order=order,
            ),
        )
    registry.list_ariel_enhancement_modules.return_value = sorted(
        table, key=lambda n: table[n][1].execution_order
    )
    registry.get_ariel_enhancement_module.side_effect = table.get


class Harness:
    """Everything ``run_catchup`` reaches, patched; ``images`` maps a name to a module or error."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, repo: CatchupRepo) -> None:
        import osprey.services.ariel_search.database.connection as connection_mod
        import osprey.services.ariel_search.enhancement as enh

        self.repo = repo
        self.text: list[Any] = []
        self.images: dict[str, Any] = {}
        self.builds: list[tuple[str, tuple[str, ...]]] = []
        self.locks = Locks()
        self.service = _StubService(repo)
        _patch_service(monkeypatch, self.service)
        _register_image_modules()
        monkeypatch.setattr(connection_mod, "try_advisory_lock", self.locks)

        def _factory(config: Any, *, stage: str = "inline", names: Any = None) -> list[Any]:  # noqa: ARG001 - the faked signature
            self.builds.append((stage, tuple(names or ())))
            if stage == "inline":
                return list(self.text)
            built = []
            for name in names or ():
                value = self.images.get(name)
                if isinstance(value, BaseException):
                    raise value
                if value is not None:
                    built.append(value)
            return built

        monkeypatch.setattr(enh, "create_enhancers_from_config", _factory)

    def use_clock(self, monkeypatch: pytest.MonkeyPatch, clock: Clock) -> None:
        """Route the catch-up's and the driver's monotonic clock to ``clock``."""
        fake_time = types.SimpleNamespace(monotonic=clock.monotonic)
        monkeypatch.setattr(ops, "time", fake_time)
        monkeypatch.setattr(image_driver, "time", fake_time)


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level and r.name.startswith("ariel")]


# ---------------------------------------------------------------------------
# catchup_budget
# ---------------------------------------------------------------------------


async def test_budget_null_is_the_rest_of_the_poll_interval() -> None:
    assert ops.catchup_budget(_config(poll_interval=600, budget=None), 100.0) == 500.0
    assert ops.catchup_budget(_config(poll_interval=600), 100.0) == 500.0
    assert ops.catchup_budget(_config(poll_interval=10), 25.0) == 0.0


async def test_budget_zero_is_zero() -> None:
    assert ops.catchup_budget(_config(budget=0), 3.0) == 0.0


async def test_budget_number_is_taken_as_written() -> None:
    assert ops.catchup_budget(_config(budget=45.5), 3.0) == 45.5


@pytest.mark.parametrize("bad", [-1, -0.5, True, "60", [60]])
async def test_budget_bad_value_names_the_key(bad: Any) -> None:
    with pytest.raises(ValueError, match=r"ariel\.enhancement\.catchup_budget_seconds"):
        ops.catchup_budget(_config(budget=bad), 0.0)


# ---------------------------------------------------------------------------
# run_catchup
# ---------------------------------------------------------------------------


async def test_text_first_then_pictures_with_no_fetch_and_no_enhance(
    monkeypatch, attachment_fetch
) -> None:
    repo = CatchupRepo(text_todo=["t1", "t2"], todo=["p1"])
    h = Harness(monkeypatch, repo)
    text = _Enhancer("text_embedding")
    caption = FakeModule()
    h.text = [text]
    h.images = {"image_caption": caption}

    result = await ops.run_catchup(_config(), budget_s=60.0, stop_event=asyncio.Event())

    assert text.seen == ["t1", "t2"]
    assert caption.run_calls == ["p1"]
    assert caption.enhance_calls == 0
    assert attachment_fetch is None or attachment_fetch.calls == []
    assert result.entries_processed == 3
    assert result.module_names == ["text_embedding", "image_caption"]
    assert ("catchup", ("image_caption",)) in h.builds
    assert h.locks.events == [
        ("acquire", "ariel_enhance:image_caption"),
        ("release", "ariel_enhance:image_caption"),
    ]


async def test_caption_backlog_still_leaves_image_embedding_its_share(monkeypatch) -> None:
    repo = CatchupRepo(todo=[f"p{i:03d}" for i in range(100)])
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)

    def _slow(mod, entry, gate):  # noqa: ARG001 - the faked signature
        if not gate.may_start_picture():
            return ImageEntryOutcome.partial()
        clock.now += 1.0
        gate.succeeded()
        return ImageEntryOutcome.done()

    caption = FakeModule(_slow, name="image_caption")
    embedding = FakeModule(_slow, name="image_embedding")
    h.images = {"image_caption": caption, "image_embedding": embedding}

    await ops.run_catchup(_config(image_embedding=True), budget_s=20.0, stop_event=asyncio.Event())

    assert 0 < len(caption.run_calls) < 100
    assert len(embedding.run_calls) > 0
    # Caption stopped at its share (10 s, none started with < 5 s left), so the
    # embedding got the rest: more than an equal share.
    assert len(caption.run_calls) <= 6
    assert len(embedding.run_calls) >= len(caption.run_calls)


async def test_stop_returns_within_one_picture(monkeypatch) -> None:
    repo = CatchupRepo(todo=[f"p{i:03d}" for i in range(50)])
    h = Harness(monkeypatch, repo)

    class _Slow(FakeModule):
        async def run_entry(self, entry, repository, *, gate):
            self.run_calls.append(entry["entry_id"])
            if not gate.may_start_picture():
                return ImageEntryOutcome.partial()
            await asyncio.sleep(0.05)
            gate.succeeded()
            repository.complete.add(entry["entry_id"])
            return ImageEntryOutcome.done()

    caption = _Slow()
    h.images = {"image_caption": caption}
    stop = asyncio.Event()

    async def _stop_soon() -> float:
        await asyncio.sleep(0.12)
        stop.set()
        return time.monotonic()

    stopper = asyncio.ensure_future(_stop_soon())
    await ops.run_catchup(_config(), budget_s=None, stop_event=stop)
    stopped_at = await stopper
    assert time.monotonic() - stopped_at < 0.05 + 0.1
    assert len(caption.run_calls) < 10


async def test_a_cancelled_pass_frees_its_lock(monkeypatch) -> None:
    repo = CatchupRepo(todo=["p1"])
    h = Harness(monkeypatch, repo)
    entered = asyncio.Event()

    class _Hang(FakeModule):
        async def run_entry(self, entry, repository, *, gate):  # noqa: ARG002 - the faked signature
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    h.images = {"image_caption": _Hang()}
    task = asyncio.ensure_future(ops.run_catchup(_config(), budget_s=None, stop_event=None))
    await asyncio.wait_for(entered.wait(), 2.0)
    assert h.locks.held == {"ariel_enhance:image_caption"}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.locks.held == set()
    assert h.locks.events[-1] == ("release", "ariel_enhance:image_caption")


async def test_lock_held_elsewhere_skips_the_module(monkeypatch, caplog) -> None:
    repo = CatchupRepo(todo=["p1"])
    h = Harness(monkeypatch, repo)
    h.locks.busy = {"ariel_enhance:image_caption"}
    caption = FakeModule()
    h.images = {"image_caption": caption}
    with caplog.at_level(logging.INFO, logger="ariel"):
        await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)
    assert caption.run_calls == [] and caption.health_calls == 0
    assert _records(caplog, logging.ERROR) == []


async def test_unhealthy_module_is_skipped_with_statuses_untouched(monkeypatch) -> None:
    repo = CatchupRepo(text_todo=["t1"], todo=["p1"], markable=["m1"])
    h = Harness(monkeypatch, repo)
    h.text = [_Enhancer("text_embedding")]
    caption = FakeModule(health=HealthResult(False, "model not available", "model"))
    h.images = {"image_caption": caption}

    result = await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)

    assert caption.run_calls == []
    assert repo.mark_calls == [] and repo.failed == {} and repo.complete == set()
    assert repo.text_complete == [("t1", "text_embedding")]
    assert result.entries_processed == 1


async def test_a_registered_module_returning_a_pair_still_works(monkeypatch) -> None:
    repo = CatchupRepo(todo=["p1"])
    h = Harness(monkeypatch, repo)
    caption = FakeModule(health=(False, "server down"))
    h.images = {"image_caption": caption}
    await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)
    assert caption.run_calls == []

    caption.health = (True, "OK")
    await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)
    assert caption.run_calls == ["p1"]


async def test_three_unreachable_passes_warn_once_and_recovery_logs_info(
    monkeypatch, caplog
) -> None:
    repo = CatchupRepo(todo=["p1"])
    h = Harness(monkeypatch, repo)
    caption = FakeModule(health=ConnectionRefusedError("connection refused"))
    h.images = {"image_caption": caption}
    with caplog.at_level(logging.DEBUG, logger="ariel"):
        for _ in range(3):
            await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)
        assert caption.run_calls == [] and repo.failed == {}
        caption.health = (True, "OK")
        await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "image_caption" in warnings[0].getMessage()
    assert "unreachable" in warnings[0].getMessage()
    assert _records(caplog, logging.ERROR) == []
    infos = [r for r in _records(caplog, logging.INFO) if "available again" in r.getMessage()]
    assert len(infos) == 1
    assert caption.run_calls == ["p1"]


async def test_misconfigured_picture_module_warns_once_and_text_still_runs(
    monkeypatch, caplog
) -> None:
    repo = CatchupRepo(text_todo=["t1"], todo=["p1"])
    h = Harness(monkeypatch, repo)
    text = _Enhancer("text_embedding")
    h.text = [text]
    h.images = {
        "image_caption": ModuleConfigError(
            "image_caption.model names no model",
            key="ariel.enhancement_modules.image_caption.model",
        )
    }
    with caplog.at_level(logging.DEBUG, logger="ariel"):
        for _ in range(3):
            result = await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)
            assert result.module_names == ["text_embedding"]
        caption = FakeModule()
        h.images = {"image_caption": caption}
        await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)

    assert text.seen == ["t1"]
    assert caption.run_calls == ["p1"]
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "image_caption" in message and "config" in message
    assert "ariel.enhancement_modules.image_caption.model" in message
    assert _records(caplog, logging.ERROR) == []
    infos = [r for r in _records(caplog, logging.INFO) if "available again" in r.getMessage()]
    assert len(infos) == 1


async def test_slow_text_module_finishes_every_entry_past_the_budget(monkeypatch) -> None:
    repo = CatchupRepo(text_todo=["t1", "t2", "t3"], todo=["p1"])
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)

    class _SlowText(_Enhancer):
        async def enhance(self, entry, conn):
            await super().enhance(entry, conn)
            clock.now += 140.0  # three entries: budget (20 s) + 400 s

    text = _SlowText("text_embedding")
    h.text = [text]
    caption = FakeModule()
    h.images = {"image_caption": caption}

    await ops.run_catchup(_config(), budget_s=20.0, stop_event=None)

    assert text.seen == ["t1", "t2", "t3"]
    assert repo.text_complete == [
        ("t1", "text_embedding"),
        ("t2", "text_embedding"),
        ("t3", "text_embedding"),
    ]
    assert repo.text_failed == []
    # The text stage used the whole budget, so no picture started.
    assert caption.run_calls == []


async def test_ten_thousand_pictureless_entries_are_batch_marked_and_one_is_walked(
    monkeypatch,
) -> None:
    markable = [f"e{i:05d}" for i in range(10_000)]
    repo = CatchupRepo(markable=markable, todo=["p1"])
    h = Harness(monkeypatch, repo)
    caption = FakeModule()
    h.images = {"image_caption": caption}

    await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)

    assert repo.complete == {*markable, "p1"}
    assert caption.run_calls == ["p1"]
    # Keyset cursor: each batch starts after the largest id of the one before.
    assert repo.mark_calls == ["", *(markable[i * 1000 + 999] for i in range(10))]


async def test_text_stage_checks_the_stop_event_per_entry(monkeypatch) -> None:
    repo = CatchupRepo(text_todo=["t1", "t2", "t3"])
    h = Harness(monkeypatch, repo)
    stop = asyncio.Event()

    class _StopAfterOne(_Enhancer):
        async def enhance(self, entry, conn):
            await super().enhance(entry, conn)
            stop.set()

    text = _StopAfterOne("text_embedding")
    h.text = [text]
    await ops.run_catchup(_config(), budget_s=60.0, stop_event=stop)
    assert text.seen == ["t1"]


# ---------------------------------------------------------------------------
# Marker-stale text entries and their budget share
# ---------------------------------------------------------------------------

_NEW_MODEL = "new-model"


def _semantic_config(budget: Any = "unset", **modules: bool) -> dict:
    """``_config`` with ``semantic_processor`` enabled under ``model_id = new-model``."""
    cfg = _config(budget=budget, **modules)
    cfg["enhancement_modules"]["semantic_processor"] = {
        "enabled": True,
        "provider": "cborg",
        "model": {"model_id": _NEW_MODEL},
    }
    return cfg


def _stale(*ids: str) -> dict[str, dict[str, Any]]:
    """Statuses completed under an older ``semantic_processor`` marker."""
    return {i: {"semantic_processor": {"status": "complete", "marker": "old-model"}} for i in ids}


class _SlowSemantic(_Enhancer):
    """``semantic_processor`` double: each call advances ``clock`` by ``step`` seconds."""

    def __init__(self, clock: Clock, step: float) -> None:
        super().__init__("semantic_processor")
        self.clock = clock
        self.step = step

    async def enhance(self, entry, conn):
        await super().enhance(entry, conn)
        self.clock.now += self.step


async def test_stale_semantic_processor_entries_are_rerun(monkeypatch) -> None:
    repo = CatchupRepo(text_todo=["s1", "s2"], text_status=_stale("s1", "s2"))
    h = Harness(monkeypatch, repo)
    semantic = _Enhancer("semantic_processor")
    h.text = [semantic]

    await ops.run_catchup(_semantic_config(), budget_s=60.0, stop_event=None)

    # The read carries the current marker, so the repository owes marker-stale entries.
    assert ("semantic_processor", _NEW_MODEL) in repo.text_reads
    assert semantic.seen == ["s1", "s2"]
    assert repo.text_complete == [("s1", "semantic_processor"), ("s2", "semantic_processor")]


async def test_enhance_reads_text_modules_under_their_current_marker(monkeypatch) -> None:
    repo = CatchupRepo(text_todo=["s1"], text_status=_stale("s1"))
    h = Harness(monkeypatch, repo)
    semantic = _Enhancer("semantic_processor")
    h.text = [semantic, _Enhancer("text_embedding")]

    await ops.run_enhance(_semantic_config(), module=None, force=False, limit=5)

    assert ("semantic_processor", _NEW_MODEL) in repo.text_reads
    # text_embedding has no completion marker: its read stays marker-less.
    assert ("text_embedding", None) in repo.text_reads
    assert semantic.seen == ["s1"]


async def test_slow_stale_reruns_stop_at_their_share_and_pictures_still_run(monkeypatch) -> None:
    ids = [f"s{i}" for i in range(1, 7)]
    repo = CatchupRepo(text_todo=ids, text_status=_stale(*ids), todo=["p1"])
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    semantic = _SlowSemantic(clock, 4.0)
    h.text = [semantic]
    caption = FakeModule()
    h.images = {"image_caption": caption}

    await ops.run_catchup(_semantic_config(), budget_s=20.0, stop_event=None)

    # A picture module is enabled, so stale re-runs get half of 20 s: they stop
    # once 10 s are spent (4 + 4 + 4), and the rest wait for the next pass.
    assert semantic.seen == ["s1", "s2", "s3"]
    assert [c for c in repo.text_complete if c[1] == "semantic_processor"] == [
        ("s1", "semantic_processor"),
        ("s2", "semantic_processor"),
        ("s3", "semantic_processor"),
    ]
    assert repo.text_failed == []
    # The picture stage still runs on what is left of the budget.
    assert caption.run_calls == ["p1"]


async def test_stale_reruns_take_the_whole_budget_without_a_picture_module(monkeypatch) -> None:
    ids = [f"s{i}" for i in range(1, 9)]
    repo = CatchupRepo(text_todo=ids, text_status=_stale(*ids))
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    semantic = _SlowSemantic(clock, 4.0)
    h.text = [semantic]

    await ops.run_catchup(_semantic_config(image_caption=False), budget_s=20.0, stop_event=None)

    # No picture module: the whole 20 s is the share (4 s x 5).
    assert semantic.seen == ["s1", "s2", "s3", "s4", "s5"]


@pytest.mark.parametrize("budget", [0.0, 0.001])
async def test_one_stale_entry_runs_at_a_budget_near_zero(monkeypatch, budget: float) -> None:
    repo = CatchupRepo(text_todo=["s1", "s2", "s3"], text_status=_stale("s1", "s2", "s3"))
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    semantic = _SlowSemantic(clock, 1.0)
    h.text = [semantic]

    await ops.run_catchup(_semantic_config(), budget_s=budget, stop_event=None)

    assert semantic.seen == ["s1"]
    assert repo.text_failed == []


async def test_never_run_text_entries_are_not_throttled(monkeypatch) -> None:
    ids = ["s1", "s2", "n1", "n2", "n3"]
    # n1 is pending under the current marker; n2 and n3 never ran.
    status = {
        **_stale("s1", "s2"),
        "n1": {"semantic_processor": {"status": "pending", "marker": _NEW_MODEL}},
    }
    repo = CatchupRepo(text_todo=ids, text_status=status)
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    semantic = _SlowSemantic(clock, 30.0)
    embedder = _Enhancer("text_embedding")
    h.text = [semantic, embedder]

    await ops.run_catchup(_semantic_config(), budget_s=0.0, stop_event=None)

    # One stale re-run (the floor); every owed entry that is not stale runs.
    assert semantic.seen == ["s1", "n1", "n2", "n3"]
    # A throttled stale entry still gets the modules it owes that are not stale.
    assert embedder.seen == ids
    assert ("s2", "semantic_processor") not in repo.text_complete
    assert ("s2", "semantic_processor") not in repo.text_failed


async def test_stale_backlog_beyond_the_read_limit_does_not_starve_owed_work(monkeypatch) -> None:
    """A stale backlog filling the 1000-entry read leaves never-run and failed work unthrottled."""
    stale_ids = [f"s{i:04d}" for i in range(1001)]
    # The real catch-up order: pending first (a stale marker counts as pending with no
    # attempts, newest first), then failed. The older never-run entry and the failed
    # retry under the current marker sort behind the whole stale window.
    ids = [*stale_ids, "old", "f1"]
    status = {
        **_stale(*stale_ids),
        "f1": {"semantic_processor": {"status": "failed", "attempts": 1, "marker": _NEW_MODEL}},
    }
    repo = CatchupRepo(text_todo=ids, text_status=status)
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    semantic = _SlowSemantic(clock, 30.0)
    h.text = [semantic]

    result = await ops.run_catchup(_semantic_config(), budget_s=0.0, stop_event=None)

    # One stale re-run (the floor), and both owed entries behind the window run.
    assert semantic.seen == ["s0000", "old", "f1"]
    # Entries whose only owed work was a deferred stale re-run are not counted.
    assert result.entries_processed == 3


async def test_stale_reruns_are_unthrottled_without_a_budget(monkeypatch) -> None:
    ids = [f"s{i}" for i in range(1, 5)]
    repo = CatchupRepo(text_todo=ids, text_status=_stale(*ids))
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    semantic = _SlowSemantic(clock, 1000.0)
    h.text = [semantic]

    await ops.run_catchup(_semantic_config(), budget_s=None, stop_event=None)

    assert semantic.seen == ids


async def test_text_catchup_keeps_the_thousand_entry_ceiling(monkeypatch) -> None:
    ids = [f"n{i:04d}" for i in range(1005)]
    repo = CatchupRepo(text_todo=ids)
    h = Harness(monkeypatch, repo)
    semantic = _Enhancer("semantic_processor")
    h.text = [semantic]

    await ops.run_catchup(_semantic_config(), budget_s=0.0, stop_event=None)

    assert len(semantic.seen) == 1000


# ---------------------------------------------------------------------------
# run_sync step 3
# ---------------------------------------------------------------------------


async def test_run_sync_catchup_is_bounded_by_the_poll_interval(monkeypatch, fake_pool) -> None:
    """``poll_interval_seconds=10``, a 1 s caption and 100 pictures: back in < 10 + timeout + 1."""
    repo = CatchupRepo(todo=[f"p{i:03d}" for i in range(100)])
    h = Harness(monkeypatch, repo)
    clock = Clock()
    h.use_clock(monkeypatch, clock)
    _patch_pool(monkeypatch, fake_pool)
    _patch_migrations(monkeypatch, applied=[])
    _patch_scheduler(monkeypatch, _poll_result(added=0))

    def _one_second(mod, entry, gate):  # noqa: ARG001 - the faked signature
        if not gate.may_start_picture():
            return ImageEntryOutcome.partial()
        clock.now += 1.0
        gate.succeeded()
        return ImageEntryOutcome.done()

    caption = FakeModule(_one_second, timeout_seconds=30.0)
    h.images = {"image_caption": caption}
    start = clock.now

    out = await ops.run_sync(_config(poll_interval=10))

    assert clock.now - start < 10 + caption.timeout_seconds + 1
    assert 0 < len(caption.run_calls) < 100
    assert out.entries_enhanced == len(caption.run_calls)


# ---------------------------------------------------------------------------
# The poll cadence of run_forever
# ---------------------------------------------------------------------------


def _scheduler_on_clock(monkeypatch, clock: Clock, poll, *, interval: float = 10.0):
    """An ``IngestionScheduler`` whose clock and sleep are ``clock``; records poll starts."""
    import osprey.services.ariel_search.ingestion.scheduler as sched_mod
    from tests.services.ariel_search.test_scheduler import _make_config

    config = _make_config(
        poll_interval=int(interval), backoff_multiplier=2.0, max_interval=3600, max_failures=10
    )
    scheduler = sched_mod.IngestionScheduler(config=config, repository=object())
    waits: list[float] = []

    async def _wait_for(awaitable, timeout):
        awaitable.close()
        waits.append(timeout)
        clock.now += timeout
        raise TimeoutError

    monkeypatch.setattr(sched_mod, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(
        sched_mod, "asyncio", types.SimpleNamespace(wait_for=_wait_for, sleep=asyncio.sleep)
    )
    starts: list[float] = []

    async def _poll_once(*_a, **_kw):
        starts.append(clock.now)
        result = await poll(len(starts))
        if len(starts) >= 4:
            scheduler._stop_event.set()
        return result

    scheduler.poll_once = _poll_once  # type: ignore[method-assign]
    return scheduler, starts, waits


async def test_polls_start_one_interval_apart(monkeypatch) -> None:
    clock = Clock()

    async def _poll(n: int):  # noqa: ARG001 - the faked signature
        clock.now += 3.0  # poll plus catch-up
        return _poll_result()

    scheduler, starts, waits = _scheduler_on_clock(monkeypatch, clock, _poll)
    await asyncio.wait_for(scheduler.run_forever(), 5.0)
    assert [b - a for a, b in zip(starts, starts[1:], strict=False)] == [10.0, 10.0, 10.0]
    assert waits == [7.0, 7.0, 7.0]


async def test_an_overrunning_poll_repolls_at_once(monkeypatch) -> None:
    clock = Clock()

    async def _poll(n: int):
        clock.now += 15.0 if n == 1 else 1.0
        return _poll_result()

    scheduler, starts, waits = _scheduler_on_clock(monkeypatch, clock, _poll)
    await asyncio.wait_for(scheduler.run_forever(), 5.0)
    assert starts[1] - starts[0] == 15.0
    assert waits[0] == 9.0  # the first wait follows the second poll


async def test_a_failed_poll_still_backs_off(monkeypatch) -> None:
    clock = Clock()

    async def _poll(n: int):
        clock.now += 1.0
        if n == 1:
            raise ConnectionError("source down")
        return _poll_result()

    scheduler, starts, waits = _scheduler_on_clock(monkeypatch, clock, _poll)
    await asyncio.wait_for(scheduler.run_forever(), 5.0)
    # One failure doubles the interval: 20 s from the failed poll's start.
    assert starts[1] - starts[0] == 20.0
    assert waits[0] == 19.0
    assert starts[2] - starts[1] == 10.0


# ---------------------------------------------------------------------------
# run_enhance -- the manual drain of a picture module
# ---------------------------------------------------------------------------


@pytest.fixture
def no_render_worker(monkeypatch):
    """Fail any spawn of the picture render worker."""
    from osprey.imaging import render

    async def _spawn():
        raise AssertionError("the render worker was spawned")

    monkeypatch.setattr(render, "_spawn", _spawn)
    yield
    assert render.worker_pid() is None


class _DriveSpy:
    """Records the keyword arguments of every ``drive_image_module`` call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[dict[str, Any]] = []
        real = image_driver.drive_image_module

        async def _spy(module, repository, **kw):
            self.calls.append({"module": module.name, **kw})
            return await real(module, repository, **kw)

        monkeypatch.setattr(image_driver, "drive_image_module", _spy)


@pytest.mark.usefixtures("no_render_worker")
async def test_enhance_module_drives_the_picture_module_without_a_budget(
    monkeypatch, attachment_fetch
) -> None:
    repo = CatchupRepo(text_todo=["t1"], todo=["p1", "p2"], markable=["m1"])
    h = Harness(monkeypatch, repo)
    h.text = [_Enhancer("text_embedding")]
    caption = FakeModule()
    h.images = {"image_caption": caption}
    spy = _DriveSpy(monkeypatch)

    result = await ops.run_enhance(_config(), module="image_caption", force=False, limit=100)

    assert caption.run_calls == ["p1", "p2"]
    assert caption.enhance_calls == 0
    assert caption.health_calls == 1
    assert repo.complete == {"m1", "p1", "p2"}
    # The text modules and the generic unconditional mark are never reached.
    assert repo.text_complete == [] and repo.text_failed == []
    assert [b for b in h.builds if b[0] == "inline"] == []
    assert [(c["module"], c["budget"], c["limit"]) for c in spy.calls] == [
        ("image_caption", None, 100)
    ]
    assert h.locks.events == [
        ("acquire", "ariel_enhance:image_caption"),
        ("release", "ariel_enhance:image_caption"),
    ]
    assert attachment_fetch is None or attachment_fetch.calls == []
    assert result.entries_processed == 2
    assert result.module_names == ["image_caption"]


async def test_enhance_module_limit_caps_the_entries_handed_to_run_entry(monkeypatch) -> None:
    repo = CatchupRepo(todo=[f"p{i}" for i in range(5)], markable=["m1", "m2", "m3"])
    h = Harness(monkeypatch, repo)
    embedding = FakeModule(name="image_embedding")
    h.images = {"image_embedding": embedding}

    result = await ops.run_enhance(
        _config(image_caption=False, image_embedding=True),
        module="image_embedding",
        force=False,
        limit=2,
    )

    assert embedding.run_calls == ["p0", "p1"]
    # The set-based batch mark is not counted against the limit.
    assert {"m1", "m2", "m3"} <= repo.complete
    assert result.entries_processed == 2


async def test_enhance_module_lock_held_elsewhere_prints_and_skips(monkeypatch) -> None:
    repo = CatchupRepo(todo=["p1"])
    h = Harness(monkeypatch, repo)
    h.locks.busy = {"ariel_enhance:image_caption"}
    caption = FakeModule()
    h.images = {"image_caption": caption}
    lines: list[str] = []

    await ops.run_enhance(
        _config(), module="image_caption", force=False, limit=10, progress=lines.append
    )

    assert (
        "image_caption: skipped, another pass is running it (osprey ariel watch runs the picture modules on every poll)"
        in lines
    )
    assert caption.run_calls == [] and caption.health_calls == 0


async def test_enhance_module_unreachable_prints_the_skip_reason_and_writes_nothing(
    monkeypatch,
) -> None:
    repo = CatchupRepo(todo=["p1"], markable=["m1"])
    h = Harness(monkeypatch, repo)
    caption = FakeModule(health=ConnectionRefusedError("connection refused"))
    h.images = {"image_caption": caption}
    lines: list[str] = []

    await ops.run_enhance(
        _config(), module="image_caption", force=False, limit=10, progress=lines.append
    )

    assert "image_caption: skipped, unavailable (unreachable)" in lines
    assert caption.run_calls == []
    assert repo.mark_calls == [] and repo.failed == {} and repo.complete == set()


async def test_enhance_module_force_is_refused_for_a_picture_module(monkeypatch) -> None:
    repo = CatchupRepo(todo=["p1"])
    h = Harness(monkeypatch, repo)
    caption = FakeModule()
    h.images = {"image_caption": caption}

    with pytest.raises(ValueError) as refused:
        await ops.run_enhance(_config(), module="image_caption", force=True, limit=10)

    assert str(refused.value) == ops.FORCE_REFUSAL
    assert caption.run_calls == [] and h.builds == []


@pytest.mark.usefixtures("no_render_worker")
async def test_bare_enhance_runs_text_then_every_picture_module_through_the_driver(
    monkeypatch, attachment_fetch
) -> None:
    repo = CatchupRepo(text_todo=["t1"], todo=["p1"])
    h = Harness(monkeypatch, repo)
    text = _Enhancer("text_embedding")
    h.text = [text]
    caption = FakeModule()
    embedding = FakeModule(name="image_embedding")
    h.images = {"image_caption": caption, "image_embedding": embedding}
    spy = _DriveSpy(monkeypatch)

    result = await ops.run_enhance(
        _config(image_embedding=True), module=None, force=False, limit=100
    )

    assert text.seen == ["t1"]
    assert repo.text_complete == [("t1", "text_embedding")]
    assert caption.run_calls == ["p1"]
    assert caption.enhance_calls == 0 and embedding.enhance_calls == 0
    assert [(c["module"], c["budget"], c["limit"]) for c in spy.calls] == [
        ("image_caption", None, 100),
        ("image_embedding", None, 100),
    ]
    assert result.module_names == ["text_embedding", "image_caption", "image_embedding"]
    assert attachment_fetch is None or attachment_fetch.calls == []


async def test_bare_enhance_force_rereads_text_and_names_the_skipped_picture_modules(
    monkeypatch,
) -> None:
    class _Repo(CatchupRepo):
        async def search_by_time_range(self, limit: int = 100, **_kw: Any) -> list[dict]:
            return [{"entry_id": "t0", "raw_text": "done before", "attachments": []}][:limit]

    repo = _Repo(text_todo=["t1"], todo=["p1"])
    h = Harness(monkeypatch, repo)
    text = _Enhancer("text_embedding")
    h.text = [text]
    caption = FakeModule()
    h.images = {"image_caption": caption}
    lines: list[str] = []

    await ops.run_enhance(_config(), module=None, force=True, limit=10, progress=lines.append)

    assert text.seen == ["t0"]
    assert (
        "--force re-runs the text modules only; image_caption run their normal (unforced) pass"
        in lines
    )
    # The picture pass itself is not forced: it walks only what is still owed.
    assert caption.run_calls == ["p1"]


async def test_catchup_passes_no_entry_limit(monkeypatch) -> None:
    repo = CatchupRepo(todo=["p1", "p2", "p3"])
    h = Harness(monkeypatch, repo)
    caption = FakeModule()
    h.images = {"image_caption": caption}
    spy = _DriveSpy(monkeypatch)

    await ops.run_catchup(_config(), budget_s=60.0, stop_event=None)

    assert [c["limit"] for c in spy.calls] == [None]
    assert caption.run_calls == ["p1", "p2", "p3"]


async def test_driver_limit_counts_run_entry_calls_only() -> None:
    repo = FakeRepo(todo=["p1", "p2", "p3"], markable=["m1"])
    module = FakeModule()

    result = await image_driver.drive_image_module(
        module, repo, budget=None, stop_event=None, limit=1
    )

    assert module.run_calls == ["p1"]
    assert result.entries_walked == 1 and result.marked_complete == 1
    assert result.ended == "limit"


# ---------------------------------------------------------------------------
# enhance --retry-failed
# ---------------------------------------------------------------------------


async def test_retry_failed_misconfigured_embedding_prints_the_skip_line(monkeypatch) -> None:
    from tests.services.ariel_search._cli_ops_doubles import _forbid_service

    _forbid_service(monkeypatch, "a misconfigured retry must not touch the database")
    lines: list[str] = []

    reset = await ops.retry_failed_entries(
        ops._ariel_config(_config(image_embedding=True)), "image_embedding", lines.append
    )

    assert reset == 0
    assert len(lines) == 1
    assert lines[0].startswith("image_embedding: skipped, unavailable (config: ")
    assert "model is required" in lines[0]


async def test_enhance_retry_failed_misconfigured_module_reports_no_traceback(
    monkeypatch,
) -> None:
    h = Harness(monkeypatch, CatchupRepo(todo=["p1"]))
    h.images = {
        "image_embedding": ModuleConfigError(
            "image_embedding.model is required", key="image_embedding.model"
        )
    }
    lines: list[str] = []

    result = await ops.run_enhance(
        _config(image_embedding=True),
        module="image_embedding",
        force=False,
        limit=10,
        progress=lines.append,
        retry_failed=True,
    )

    assert lines == [
        "image_embedding: skipped, unavailable (config: image_embedding.model is required)"
    ]
    assert result.entries_processed == 0
    assert h.locks.events == []


async def test_retry_failed_runs_under_the_module_lock_and_skips_when_held(
    monkeypatch,
) -> None:
    h = Harness(monkeypatch, CatchupRepo(todo=["p1"]))
    h.locks.busy.add("ariel_enhance:image_caption")
    lines: list[str] = []

    reset = await ops.retry_failed_entries(
        ops._ariel_config(_config()), "image_caption", lines.append
    )

    assert reset == 0
    assert lines == [
        "image_caption: skipped, another pass is running it (osprey ariel watch runs the picture modules on every poll)"
    ]
