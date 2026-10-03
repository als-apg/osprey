"""Tests for the enhancement factory's stage filter and the catch-up module interface.

A ``runs_inline=False`` module (a picture module) never runs, nor fails
``configure()``, in an inline caller; the catch-up drives it through
``BaseEnhancementModule.run_entry`` with a ``PictureGate``.

The registry is the autouse ``_mock_ariel_registry`` mock; each test extends it
through ``monkeypatch`` with :func:`_register`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.registry.base import ArielEnhancementModuleRegistration
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.enhancement.base import (
    BaseEnhancementModule,
    ImageEntryOutcome,
    PictureGate,
)
from osprey.services.ariel_search.enhancement.factory import create_enhancers_from_config
from tests.services.ariel_search._cli_ops_doubles import _patch_service, _StubService

_DB = {"uri": "postgresql://localhost:5432/test"}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Calls:
    """Shared call log, so a test sees calls on instances the factory built."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def names(self, event: str) -> list[str]:
        return [name for kind, name in self.events if kind == event]


CALLS = _Calls()


class _MisconfiguredImageModule(BaseEnhancementModule):
    """``image_caption`` stand-in: catch-up only, and every touch is recorded."""

    runs_inline = False

    def __init__(self) -> None:
        CALLS.events.append(("init", "image_caption"))

    @property
    def name(self) -> str:
        return "image_caption"

    def configure(self, config: dict[str, Any]) -> None:  # noqa: ARG002 - the enhancement module signature
        CALLS.events.append(("configure", "image_caption"))
        raise ValueError("image_caption is misconfigured")

    async def enhance(self, entry, conn) -> None:  # noqa: ARG002 - the enhancement module signature
        CALLS.events.append(("enhance", "image_caption"))
        raise NotImplementedError("image_caption runs in the catch-up")

    async def run_entry(self, entry, repository, *, gate):  # noqa: ARG002 - the enhancement module signature
        CALLS.events.append(("run_entry", "image_caption"))
        return ImageEntryOutcome.done()


class _GoodImageModule(BaseEnhancementModule):
    """A well-configured catch-up module, for the catch-up and ``all`` stages."""

    runs_inline = False

    def __init__(self) -> None:
        self.configured: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "image_embedding"

    def configure(self, config: dict[str, Any]) -> None:
        self.configured = config

    async def enhance(self, entry, conn) -> None:
        raise NotImplementedError("image_embedding runs in the catch-up")


class _TextEmbedding(BaseEnhancementModule):
    """Inline ``text_embedding`` stand-in recording each entry it enhances."""

    def __init__(self) -> None:
        CALLS.events.append(("init", "text_embedding"))

    @property
    def name(self) -> str:
        return "text_embedding"

    async def enhance(self, entry, conn) -> None:  # noqa: ARG002 - the enhancement module signature
        CALLS.events.append(("enhance", "text_embedding"))


@pytest.fixture(autouse=True)
def _fresh_calls():
    CALLS.events.clear()
    yield
    CALLS.events.clear()


def _register(monkeypatch: pytest.MonkeyPatch, modules: dict[str, tuple[type, int]]) -> None:
    """Add or replace modules in the mocked registry, keeping execution order."""
    from osprey.registry import get_registry

    registry = get_registry()
    table = dict(registry.get_ariel_enhancement_module.side_effect.__self__)
    for name, (cls, order) in modules.items():
        table[name] = (
            cls,
            ArielEnhancementModuleRegistration(
                name=name,
                module_path=cls.__module__,
                class_name=cls.__name__,
                description=name,
                execution_order=order,
            ),
        )
    ordered = sorted(table, key=lambda n: table[n][1].execution_order)
    monkeypatch.setattr(registry.list_ariel_enhancement_modules, "return_value", ordered)
    monkeypatch.setattr(registry.get_ariel_enhancement_module, "side_effect", table.get)


def _config(modules: dict[str, dict[str, Any]]) -> ARIELConfig:
    return ARIELConfig.from_dict({"database": dict(_DB), "enhancement_modules": modules})


_IMAGE_ON = {"enabled": True, "provider": "nowhere", "model": "missing-vision-model"}


# ---------------------------------------------------------------------------
# Stage filter
# ---------------------------------------------------------------------------


class TestStageFilter:
    def test_stage_inline_is_default_and_never_touches_an_image_module(self, monkeypatch):
        _register(
            monkeypatch,
            {
                "text_embedding": (_TextEmbedding, 20),
                "image_caption": (_MisconfiguredImageModule, 40),
            },
        )
        config = _config({"text_embedding": {"enabled": True}, "image_caption": _IMAGE_ON})

        default = create_enhancers_from_config(config)
        inline = create_enhancers_from_config(config, stage="inline")

        assert [e.name for e in default] == ["text_embedding"]
        assert [e.name for e in inline] == ["text_embedding"]
        assert CALLS.names("init").count("image_caption") == 0
        assert CALLS.names("configure") == []

    def test_stage_catchup_keeps_only_runs_inline_false(self, monkeypatch):
        _register(
            monkeypatch,
            {"text_embedding": (_TextEmbedding, 20), "image_embedding": (_GoodImageModule, 50)},
        )
        config = _config(
            {"text_embedding": {"enabled": True}, "image_embedding": {"enabled": True, "k": 1}}
        )

        enhancers = create_enhancers_from_config(config, stage="catchup")

        assert [e.name for e in enhancers] == ["image_embedding"]
        # The module receives its own config section (the config layer adds provider keys).
        assert enhancers[0].configured["k"] == 1
        assert enhancers[0].configured["enabled"] is True
        assert "text_embedding" not in CALLS.names("init")

    def test_stage_catchup_still_configures_and_so_raises_for_a_misconfigured_module(
        self, monkeypatch
    ):
        _register(monkeypatch, {"image_caption": (_MisconfiguredImageModule, 40)})
        config = _config({"image_caption": _IMAGE_ON})

        with pytest.raises(ValueError, match="misconfigured"):
            create_enhancers_from_config(config, stage="catchup")

    def test_stage_all_keeps_both_in_execution_order(self, monkeypatch):
        _register(
            monkeypatch,
            {"image_embedding": (_GoodImageModule, 5), "text_embedding": (_TextEmbedding, 20)},
        )
        config = _config(
            {"text_embedding": {"enabled": True}, "image_embedding": {"enabled": True}}
        )

        enhancers = create_enhancers_from_config(config, stage="all")

        assert [e.name for e in enhancers] == ["image_embedding", "text_embedding"]

    def test_stage_unknown_is_refused(self):
        with pytest.raises(ValueError, match="stage"):
            create_enhancers_from_config(_config({}), stage="poll")  # type: ignore[arg-type]

    def test_stage_names_restricts_the_walk_and_keeps_order(self, monkeypatch):
        _register(
            monkeypatch,
            {
                "text_embedding": (_TextEmbedding, 20),
                "image_caption": (_MisconfiguredImageModule, 40),
            },
        )
        config = _config(
            {
                "semantic_processor": {"enabled": True, "provider": "ollama"},
                "text_embedding": {"enabled": True},
                "image_caption": _IMAGE_ON,
            }
        )

        enhancers = create_enhancers_from_config(
            config, stage="all", names=iter(["text_embedding", "semantic_processor"])
        )

        assert [e.name for e in enhancers] == ["semantic_processor", "text_embedding"]
        assert "image_caption" not in CALLS.names("init")

    def test_stage_names_still_applies_the_enabled_check(self, monkeypatch):
        _register(monkeypatch, {"text_embedding": (_TextEmbedding, 20)})
        config = _config({"text_embedding": {"enabled": False}})

        assert create_enhancers_from_config(config, names=["text_embedding"]) == []

    def test_stage_names_unknown_name_yields_nothing(self):
        config = _config({"text_embedding": {"enabled": True}})

        assert create_enhancers_from_config(config, names=["no_such_module"]) == []


# ---------------------------------------------------------------------------
# run_entry interface
# ---------------------------------------------------------------------------


class _RecordingGate:
    """A PictureGate admitting ``budget`` pictures; ``deterministic`` answers ``allow``."""

    def __init__(self, budget: int, allow: bool = True) -> None:
        self.budget = budget
        self.allow = allow
        self.starts = 0
        self.successes = 0
        self.signatures: list[str] = []

    def may_start_picture(self) -> bool:
        self.starts += 1
        return self.starts <= self.budget

    def succeeded(self) -> None:
        self.successes += 1

    def deterministic(self, signature: str) -> bool:
        self.signatures.append(signature)
        return self.allow


class _PictureModule(BaseEnhancementModule):
    """A run_entry module walking ``pictures``: 'ok', 'bad' (deterministic) or 'down'."""

    runs_inline = False

    def __init__(self, pictures: list[str]) -> None:
        self.pictures = pictures
        self.stored: list[int] = []
        self.failed: list[int] = []
        self.marked = False

    @property
    def name(self) -> str:
        return "image_caption"

    async def enhance(self, entry, conn) -> None:
        raise NotImplementedError

    async def run_entry(self, entry, repository, *, gate: PictureGate) -> ImageEntryOutcome:  # noqa: ARG002 - the enhancement module signature
        left_out = False
        for index, picture in enumerate(self.pictures):
            if index in self.stored or index in self.failed:
                continue
            if not gate.may_start_picture():
                return ImageEntryOutcome.partial()
            if picture == "down":
                return ImageEntryOutcome.unavailable("unreachable")
            if picture == "bad":
                if gate.deterministic(f"bad:{index}"):
                    self.failed.append(index)
                else:
                    left_out = True
                continue
            self.stored.append(index)
            gate.succeeded()
        if left_out:
            return ImageEntryOutcome.partial()
        self.marked = True
        return ImageEntryOutcome.done()


class TestRunEntryInterface:
    async def test_run_entry_default_raises_for_an_inline_module(self):
        with pytest.raises(NotImplementedError, match="runs inline"):
            await _TextEmbedding().run_entry({}, MagicMock(), gate=_RecordingGate(1))

    async def test_run_entry_all_pictures_handled_is_done(self):
        module = _PictureModule(["ok", "ok"])
        gate = _RecordingGate(budget=5)

        outcome = await module.run_entry({"entry_id": "e1"}, MagicMock(), gate=gate)

        assert outcome == ImageEntryOutcome.done()
        assert gate.successes == 2
        assert module.marked

    async def test_run_entry_gate_closing_returns_partial_then_resumes(self):
        module = _PictureModule(["ok", "ok", "ok"])

        first = await module.run_entry({"entry_id": "e1"}, MagicMock(), gate=_RecordingGate(1))
        second = await module.run_entry({"entry_id": "e1"}, MagicMock(), gate=_RecordingGate(5))

        assert first.kind == "partial"
        assert module.stored == [0, 1, 2]
        assert second.kind == "done"

    async def test_run_entry_deterministic_refused_writes_nothing_and_is_partial(self):
        module = _PictureModule(["ok", "bad"])
        gate = _RecordingGate(budget=5, allow=False)

        outcome = await module.run_entry({"entry_id": "e1"}, MagicMock(), gate=gate)

        assert outcome.kind == "partial"
        assert gate.signatures == ["bad:1"]
        assert module.failed == []
        assert not module.marked

    async def test_run_entry_deterministic_allowed_is_written_and_done(self):
        module = _PictureModule(["bad", "ok"])
        gate = _RecordingGate(budget=5, allow=True)

        outcome = await module.run_entry({"entry_id": "e1"}, MagicMock(), gate=gate)

        assert outcome.kind == "done"
        assert module.failed == [0]
        assert gate.successes == 1

    async def test_run_entry_service_down_is_unavailable(self):
        module = _PictureModule(["down"])

        outcome = await module.run_entry({"entry_id": "e1"}, MagicMock(), gate=_RecordingGate(5))

        assert outcome == ImageEntryOutcome("unavailable", "unreachable")

    def test_run_entry_gate_satisfies_the_protocol_structurally(self):
        gate: PictureGate = _RecordingGate(1)
        assert gate.may_start_picture() is True


class TestImageEntryOutcome:
    @pytest.mark.parametrize(
        ("outcome", "kind", "reason"),
        [
            (ImageEntryOutcome.done(), "done", None),
            (ImageEntryOutcome.partial(), "partial", None),
            (ImageEntryOutcome.transient_error("timeout"), "transient_error", "timeout"),
            (ImageEntryOutcome.unavailable("unreachable"), "unavailable", "unreachable"),
            (ImageEntryOutcome.unavailable("auth"), "unavailable", "auth"),
            (ImageEntryOutcome.unavailable("model"), "unavailable", "model"),
            (ImageEntryOutcome.unavailable("config"), "unavailable", "config"),
            (ImageEntryOutcome.module_error("refused"), "module_error", "refused"),
        ],
    )
    def test_run_entry_outcome_rows(self, outcome, kind, reason):
        assert outcome.kind == kind
        assert outcome.reason == reason

    @pytest.mark.parametrize(
        ("kind", "reason"),
        [
            ("done", "why"),
            ("partial", "why"),
            ("transient_error", None),
            ("transient_error", ""),
            ("module_error", None),
            ("unavailable", None),
            ("unavailable", "flaky"),
            ("finished", None),
        ],
    )
    def test_run_entry_outcome_rejects_malformed(self, kind, reason):
        with pytest.raises(ValueError):
            ImageEntryOutcome(kind, reason)  # type: ignore[arg-type]

    def test_run_entry_outcome_is_frozen(self):
        outcome = ImageEntryOutcome.done()
        with pytest.raises(AttributeError):
            outcome.kind = "partial"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Inline callers: ingest_one and run_enhance
# ---------------------------------------------------------------------------


class TestInlineCallersSkipImageModules:
    async def test_image_module_in_ingest_enhancer_list_is_skipped(self):
        from osprey.services.ariel_search.ingestion.ingest import _run_enhancers

        repo = MagicMock()
        repo.pool.connection = MagicMock(return_value=AsyncMock())
        repo.mark_enhancement_complete = AsyncMock()
        repo.mark_enhancement_failed = AsyncMock()

        outcome = await _run_enhancers(
            {"entry_id": "e1"},
            repo,
            [_TextEmbedding(), _MisconfiguredImageModule()],
            attachments_recorded=False,
        )

        assert outcome.enhanced == 1
        assert outcome.enhancer_failed == 0
        assert CALLS.names("enhance") == ["text_embedding"]
        assert CALLS.names("run_entry") == []
        marked = [c.args[1] for c in repo.mark_enhancement_complete.await_args_list]
        assert marked == ["text_embedding"]
        repo.mark_enhancement_failed.assert_not_awaited()

    async def test_image_module_bare_enhance_misconfigured_completes_text_modules(
        self, monkeypatch, mock_repository
    ):
        from osprey.services.ariel_search import cli_operations as ops

        _register(
            monkeypatch,
            {
                "text_embedding": (_TextEmbedding, 20),
                "image_caption": (_MisconfiguredImageModule, 40),
            },
        )
        entry = {"entry_id": "e1", "raw_text": "beam dump", "enhancement_status": {}}
        mock_repository.get_incomplete_entries = AsyncMock(return_value=[entry])
        _patch_service(monkeypatch, _StubService(mock_repository))

        result = await ops.run_enhance(
            {
                "database": dict(_DB),
                "enhancement_modules": {
                    "text_embedding": {"enabled": True},
                    "image_caption": _IMAGE_ON,
                },
            },
            module=None,
            force=False,
            limit=10,
        )

        assert result.module_names == ["text_embedding"]
        assert result.succeeded == 1
        assert result.failed == 0
        assert CALLS.names("enhance") == ["text_embedding"]
        image_calls = [k for k, n in CALLS.events if n == "image_caption"]
        assert "enhance" not in image_calls
        assert "run_entry" not in image_calls
        statuses = [
            c.args[1]
            for c in mock_repository.mark_enhancement_complete.await_args_list
            + mock_repository.mark_enhancement_failed.await_args_list
        ]
        assert "image_caption" not in statuses
        asked = [
            c.kwargs.get("module_name")
            for c in mock_repository.get_incomplete_entries.await_args_list
        ]
        assert asked == ["text_embedding"]

    async def test_image_module_enhance_module_misconfigured_prints_skip_line(self, monkeypatch):
        from osprey.services.ariel_search import cli_operations as ops
        from tests.services.ariel_search._cli_ops_doubles import _forbid_service

        _register(monkeypatch, {"image_caption": (_MisconfiguredImageModule, 40)})
        _forbid_service(monkeypatch)
        lines: list[str] = []

        result = await ops.run_enhance(
            {"database": dict(_DB), "enhancement_modules": {"image_caption": _IMAGE_ON}},
            module="image_caption",
            force=False,
            limit=10,
            progress=lines.append,
        )

        assert lines == [
            "image_caption: skipped, unavailable (config: image_caption is misconfigured)"
        ]
        assert result.entries_processed == 0
        assert result.module_names == []
        assert [k for k, _n in CALLS.events if k in ("enhance", "run_entry")] == []
