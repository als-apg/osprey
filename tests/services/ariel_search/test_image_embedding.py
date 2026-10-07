"""Tests for the ``image_embedding`` catch-up module.

The repository double keeps entries, attachment rows and the image table in
memory and answers the few statements the module and the pre-pass check issue,
so the module's own guarded write, the driver's outcome table and its pass
breakers run unchanged. Every server-facing test runs the real llama-cpp
adapter against the ``llama_stub`` server on 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
import requests

from osprey.models.providers import _local_server
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import image_table_name
from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.base import ImageEntryOutcome
from osprey.services.ariel_search.enhancement.image_driver import drive_image_module
from osprey.services.ariel_search.enhancement.image_embedding import module as embed_mod
from osprey.services.ariel_search.enhancement.image_embedding.migration import (
    ImageEmbeddingMigration,
)
from osprey.services.ariel_search.enhancement.image_embedding.module import (
    DEFAULT_TIMEOUT_SECONDS,
    IMAGE_EMBEDDING_KEY,
    ImageEmbeddingModule,
)
from osprey.services.ariel_search.exceptions import ModuleConfigError
from tests.services.ariel_search.llama_stub import MODEL

DIMS = 1024
TABLE = image_table_name(MODEL, DIMS)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()
    monkeypatch.delenv("LLAMA_CPP_HOST", raising=False)
    monkeypatch.setattr(_local_server, "container_fallback_urls", lambda base_url, default_port: [])
    monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: True)
    yield
    availability.reset_availability()
    _offload.reset_offload_state()
    _local_server.reset_cache()


def _refused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def _config(url: str, **settings: Any) -> dict[str, Any]:
    return {
        "enabled": True,
        "provider": {"name": "llama-cpp", "base_url": url},
        "provider_key": f"{IMAGE_EMBEDDING_KEY}.provider",
        "model": MODEL,
        "dimensions": DIMS,
        **settings,
    }


def _module(url: str, **settings: Any) -> ImageEmbeddingModule:
    module = ImageEmbeddingModule()
    module.configure(_config(url, **settings))
    return module


# ---------------------------------------------------------------------------
# Repository double
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, rows: list[tuple], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    async def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[tuple]:
        return list(self._rows)


class _Conn:
    def __init__(self, repo: FakeRepo) -> None:
        self._repo = repo

    async def execute(self, sql: str, params: dict[str, Any]) -> _Cursor:
        repo = self._repo
        repo.statements.append(sql)
        if sql.startswith("SELECT to_regclass"):
            name = params["t"]
            return _Cursor([(name if name in repo.relations else None,)])
        if sql.startswith(f"SELECT attachment_id FROM {TABLE}"):
            return _Cursor([(a,) for a in params["ids"] if a in repo.vectors])
        if sql.startswith(f"INSERT INTO {TABLE}"):
            assert "WHERE EXISTS" in sql and "ON CONFLICT (attachment_id) DO UPDATE" in sql
            if not repo.viewable(params["id"]):
                return _Cursor([], 0)
            literal = params["vector"]
            repo.vectors[params["id"]] = {
                "embedding": None
                if literal is None
                else [float(v) for v in literal.strip("[]").split(",")],
                "skip_reason": params["skip"],
                "model_ref": params["model"],
            }
            return _Cursor([], 1)
        raise AssertionError(f"unexpected statement: {sql}")


class _ConnCtx:
    def __init__(self, repo: FakeRepo) -> None:
        self._repo = repo

    async def __aenter__(self) -> _Conn:
        return _Conn(self._repo)

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Pool:
    def __init__(self, repo: FakeRepo) -> None:
        self._repo = repo

    def connection(self) -> _ConnCtx:
        return _ConnCtx(self._repo)


class FakeRepo:
    """In-memory entries, attachment rows and image table behind the module's calls."""

    def __init__(self, *, table: bool = True) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, dict[str, Any]] = {}
        self.relations: set[str] = {TABLE} if table else set()
        self.statements: list[str] = []
        self.status_writes: list[tuple[str, str]] = []
        self.batch_marks = 0
        self.failed: dict[str, int] = {}
        self.on_mark: Callable[[str], None] | None = None
        self.pool = _Pool(self)

    def add_entry(self, entry_id: str, pictures: int) -> list[str]:
        items = [
            {"url": f"https://logbook.example/{entry_id}/{n}.png", "filename": f"p{n}.png"}
            for n in range(pictures)
        ]
        self.entries[entry_id] = {
            "entry_id": entry_id,
            "raw_text": "beam dump",
            "attachments": items,
            "enhancement_status": {},
        }
        ids = []
        for n, item in enumerate(items):
            attachment_id = attachment_id_for(entry_id, item)
            assert attachment_id is not None
            self.files[attachment_id] = {
                "attachment_id": attachment_id,
                "entry_id": entry_id,
                "filename": item["filename"],
                "mime_type": "image/png",
                "copy_status": "copied",
                "skip_reason": None,
                "rendition_sha256": f"sha-{entry_id}-{n}",
                "rendition_mime": "image/png",
                "rendition_bytes": f"\x89PNG {entry_id} {n}".encode(),
            }
            ids.append(attachment_id)
        return ids

    def viewable(self, attachment_id: str) -> bool:
        row = self.files.get(attachment_id)
        return (
            row is not None
            and row["copy_status"] == "copied"
            and row["skip_reason"] is None
            and row["mime_type"].startswith("image/")
            and row["rendition_sha256"] is not None
        )

    def rows_of(self, entry_id: str) -> dict[str, dict[str, Any]]:
        return {
            a: v
            for a, v in self.vectors.items()
            if self.files.get(a, {}).get("entry_id") == entry_id
        }

    # -- reads -------------------------------------------------------------

    async def schema_facts(self) -> SchemaFacts:
        return SchemaFacts(has_v2_fts=True, has_copy_state=True)

    async def get_attachment_rows(self, entry_ids: list[str]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in self.files.values():
            if row["entry_id"] in entry_ids:
                out.setdefault(row["entry_id"], []).append(
                    {k: v for k, v in row.items() if k != "rendition_bytes"}
                )
        return out

    async def get_rendition(self, attachment_id: str) -> dict[str, Any] | None:
        row = self.files.get(attachment_id)
        return dict(row) if row is not None else None

    async def get_incomplete_entries(
        self,
        module_name: str | None = None,
        status: str | None = None,  # noqa: ARG002 - the faked signature
        limit: int = 100,
        *,
        marker: str | None = None,
    ) -> list[dict]:
        found = []
        for entry_id, entry in sorted(self.entries.items()):
            done = entry["enhancement_status"].get(module_name or "") or {}
            if done.get("status") == "complete" and done.get("marker") == marker:
                continue
            if self.failed.get(entry_id, 0) >= 3:
                continue
            found.append(dict(entry))
        return found[:limit]

    # -- writes ------------------------------------------------------------

    async def mark_image_module_complete_batch(
        self,
        module_name: str,  # noqa: ARG002 - the faked signature
        marker: str,  # noqa: ARG002 - the faked signature
        *,
        after: str = "",  # noqa: ARG002 - the faked signature
        limit: int = 1000,  # noqa: ARG002 - the faked signature
    ) -> list[str]:
        self.batch_marks += 1
        return []

    async def mark_image_module_complete(
        self, entry_id: str, module_name: str, marker: str
    ) -> bool:
        if self.on_mark is not None:
            self.on_mark(entry_id)
        for attachment_id, row in self.files.items():
            if row["entry_id"] != entry_id:
                continue
            if row["copy_status"] == "pending":
                return False
            if self.viewable(attachment_id) and attachment_id not in self.vectors:
                return False
        self.entries[entry_id]["enhancement_status"][module_name] = {
            "status": "complete",
            "marker": marker,
        }
        self.status_writes.append((entry_id, "complete"))
        return True

    async def mark_enhancement_failed(
        self, entry_id: str, module_name: str, error: str, *, marker: str | None = None
    ) -> int:
        self.failed[entry_id] = self.failed.get(entry_id, 0) + 1
        self.entries[entry_id]["enhancement_status"][module_name] = {
            "status": "failed",
            "attempts": self.failed[entry_id],
            "marker": marker,
            "error": error,
        }
        self.status_writes.append((entry_id, "failed"))
        return self.failed[entry_id]


class FakeGate:
    """A recording :class:`PictureGate`."""

    def __init__(self, *, deterministic: bool = True) -> None:
        self.allow_deterministic = deterministic
        self.started = 0
        self.successes = 0
        self.signatures: list[str] = []

    def may_start_picture(self) -> bool:
        self.started += 1
        return True

    def succeeded(self) -> None:
        self.successes += 1

    def deterministic(self, signature: str) -> bool:
        self.signatures.append(signature)
        return self.allow_deterministic


def _warnings(caplog, needle: str) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name.startswith("ariel") and needle in r.message
    ]


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_default_timeout_is_sized_from_the_cpu_picture_measurement(self):
        # Seconds one 1024x768 picture takes on a CPU-only llama-server on native
        # amd64, uncapped (the capped command is faster, so this bounds both).
        seconds_per_picture = 3.77
        sized = max(120, math.ceil(math.ceil(10 * seconds_per_picture) / 10) * 10)
        assert DEFAULT_TIMEOUT_SECONDS == sized

    def test_defaults_and_marker(self):
        module = _module("http://127.0.0.1:8080")
        assert module.runs_inline is False
        assert module.name == "image_embedding"
        assert module.migration is ImageEmbeddingMigration
        assert module.timeout_seconds == DEFAULT_TIMEOUT_SECONDS == 120
        assert module.target is not None and module.target.dims == DIMS
        assert module.completion_marker() == TABLE
        assert module.required_relations() == [TABLE]

    def test_dimensions_default_to_1024(self):
        config = _config("http://127.0.0.1:8080")
        del config["dimensions"]
        module = ImageEmbeddingModule()
        module.configure(config)
        assert module.target is not None and module.target.dims == 1024

    @pytest.mark.parametrize("provider", [None, "", "  ", {"base_url": "http://x:8080"}])
    def test_no_provider_raises_naming_the_key(self, provider):
        config = _config("http://127.0.0.1:8080")
        config["provider"] = provider
        with pytest.raises(ModuleConfigError) as caught:
            ImageEmbeddingModule().configure(config)
        assert str(caught.value) == f"{IMAGE_EMBEDDING_KEY}.provider is required"
        assert caught.value.key == f"{IMAGE_EMBEDDING_KEY}.provider"

    @pytest.mark.parametrize("name", ["openai", "anthropic", "no-such-provider"])
    def test_provider_without_image_embeddings_raises(self, name):
        config = _config("http://127.0.0.1:8080")
        config["provider"] = name
        with pytest.raises(ModuleConfigError) as caught:
            ImageEmbeddingModule().configure(config)
        assert caught.value.key == f"{IMAGE_EMBEDDING_KEY}.provider"

    def test_v1_base_url_is_refused_by_the_resolver(self):
        with pytest.raises(ModuleConfigError) as caught:
            _module("http://127.0.0.1:8080/v1")
        assert caught.value.key == f"{IMAGE_EMBEDDING_KEY}.provider"

    @pytest.mark.parametrize(
        ("setting", "value", "key"),
        [
            ("model", None, "model"),
            ("model", " ", "model"),
            ("dimensions", 0, "dimensions"),
            ("dimensions", 2001, "dimensions"),
            ("dimensions", "1024", "dimensions"),
            ("timeout_seconds", 0, "timeout_seconds"),
            ("timeout_seconds", "fast", "timeout_seconds"),
            ("timeout_seconds", True, "timeout_seconds"),
        ],
    )
    def test_malformed_setting_raises_naming_its_key(self, setting, value, key):
        config = _config("http://127.0.0.1:8080")
        config[setting] = value
        with pytest.raises(ModuleConfigError) as caught:
            ImageEmbeddingModule().configure(config)
        assert caught.value.key == f"{IMAGE_EMBEDDING_KEY}.{key}"

    def test_configure_does_no_network_io_and_logs_nothing(self, monkeypatch, caplog):
        def _no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("configure() touched the network")

        monkeypatch.setattr(requests, "get", _no_network)
        monkeypatch.setattr(requests, "post", _no_network)
        monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: False)
        with caplog.at_level(logging.DEBUG):
            _module(_refused_url(), timeout_seconds=30)
        # The resolver's own DEBUG line about a missing config.yml is not the module's.
        assert [r for r in caplog.records if r.levelno >= logging.INFO] == []


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_healthy_server_is_reachable(self, llama_stub):
        stub = llama_stub()
        result = await _module(stub.url).health_check()
        assert result.reachable is True
        assert result.reason is None

    async def test_connection_refused_is_unreachable(self):
        result = await _module(_refused_url()).health_check()
        assert (result.reachable, result.reason) == (False, "unreachable")

    async def test_401_is_auth(self, llama_stub, monkeypatch):
        # The reachability walk probes the same route; as in the adapter's own
        # test, the walk is let through so the verdict comes from the listing.
        monkeypatch.setattr(_local_server, "probe", lambda url, path, timeout: True)
        stub = llama_stub()
        stub.status = 401
        result = await _module(stub.url).health_check()
        assert (result.reachable, result.reason) == (False, "auth")

    async def test_alias_mismatch_is_model(self, llama_stub):
        stub = llama_stub()
        stub.alias = "another-model"
        result = await _module(stub.url).health_check()
        assert (result.reachable, result.reason) == (False, "model")

    async def test_no_reader_when_hybrid_is_disabled(self, llama_stub, monkeypatch, caplog):
        stub = llama_stub()
        module = _module(stub.url)
        hybrid = {"on": False}
        monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: hybrid["on"])

        with caplog.at_level(logging.DEBUG, logger="ariel"):
            result = await module.health_check()
            assert module.health_reason() == "no_reader"
            assert module.health_reason() == "no_reader"
            await module.health_check()
        assert result.reachable is True
        warned = _warnings(caplog, "no_reader")
        assert len(warned) == 1
        assert "ariel.search_modules.hybrid.enabled" in warned[0].message

        hybrid["on"] = True
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="ariel"):
            assert module.health_reason() is None
            assert module.health_reason() is None
        recovered = [r for r in caplog.records if "available again" in r.message]
        assert len(recovered) == 1

    async def test_no_reader_is_the_status_reason_with_reachable_kept(
        self, llama_stub, monkeypatch
    ):
        from osprey.services.ariel_search.cli_operations import _module_health

        stub = llama_stub()
        monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: False)
        health = await _module_health(_ariel_config(stub.url), "image_embedding", FakeRepo())
        assert health["reachable"] is True
        assert health["reason"] == "no_reader"

    async def test_slow_resolver_probe_reports_unreachable_within_six_seconds(
        self, llama_stub, monkeypatch
    ):
        stub = llama_stub()
        module = _module(stub.url)

        def _slow_probe(_url: str, _path: str, _timeout: float) -> bool:
            time.sleep(10)
            return True

        monkeypatch.setattr(_local_server, "probe", _slow_probe)
        began = time.monotonic()
        result = await availability.preflight(module, FakeRepo())
        assert time.monotonic() - began < 6
        assert (result.reachable, result.reason) == (False, "unreachable")

    async def test_health_and_ten_posts_resolve_the_url_once(self, llama_stub):
        stub = llama_stub()
        module = _module(stub.url)
        repo = FakeRepo()
        for n in range(10):
            repo.add_entry(f"e{n}", 1)

        assert (await module.health_check()).reachable is True
        probes_after_health = stub.models_gets
        for n in range(10):
            outcome = await module.run_entry(repo.entries[f"e{n}"], repo, gate=FakeGate())
            assert outcome.kind == "done"

        assert len(stub.embeddings) == 10
        assert stub.models_gets == probes_after_health
        assert module._reachable_base_url == stub.url


def _ariel_config(url: str) -> ARIELConfig:
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "enhancement_modules": {
                "image_embedding": {
                    "enabled": True,
                    "provider": {"name": "llama-cpp", "base_url": url},
                    "model": MODEL,
                    "dimensions": DIMS,
                }
            },
        }
    )


# ---------------------------------------------------------------------------
# run_entry
# ---------------------------------------------------------------------------


class TestRunEntry:
    async def test_every_picture_gets_a_unit_vector_and_the_entry_completes(self, llama_stub):
        stub = llama_stub()
        repo = FakeRepo()
        ids = repo.add_entry("e1", 3)
        gate = FakeGate()

        outcome = await _module(stub.url).run_entry(repo.entries["e1"], repo, gate=gate)

        assert outcome == ImageEntryOutcome.done()
        assert set(repo.vectors) == set(ids)
        for row in repo.vectors.values():
            assert len(row["embedding"]) == DIMS
            assert math.isclose(math.sqrt(sum(v * v for v in row["embedding"])), 1.0, rel_tol=1e-4)
            assert row["skip_reason"] is None
            assert row["model_ref"] == MODEL
        assert gate.successes == 3
        assert repo.entries["e1"]["enhancement_status"]["image_embedding"]["marker"] == TABLE
        assert all(body["model"] == MODEL for body in stub.embeddings)

    async def test_the_call_takes_the_configured_dimensions_and_timeout(self, monkeypatch):
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        module = _module("http://127.0.0.1:8080", timeout_seconds=42)
        calls: list[dict[str, Any]] = []

        class _Recorder:
            def execute_image_embedding(self, inputs, **kwargs):
                calls.append({"inputs": inputs, **kwargs})
                return [[1.0] + [0.0] * (DIMS - 1)]

        assert module._resolved is not None
        module._resolved = dataclasses.replace(module._resolved, instance=_Recorder())
        monkeypatch.setattr(module, "_base_url", lambda refresh=False: "http://127.0.0.1:8080")

        await module.run_entry(repo.entries["e1"], repo, gate=FakeGate())

        (call,) = calls
        assert call["dimensions"] == DIMS
        assert call["timeout"] == 42
        assert call["model_id"] == MODEL
        (picture,) = repo.files.values()
        assert call["inputs"] == [(picture["rendition_bytes"], "image/png")]

    async def test_picture_deleted_during_the_call_is_not_written(self, llama_stub, monkeypatch):
        stub = llama_stub()
        repo = FakeRepo()
        first, second = repo.add_entry("e1", 2)
        module = _module(stub.url)
        original = module._call

        def _call_then_delete(rendition: dict[str, Any]) -> Any:
            vectors = original(rendition)
            if rendition["attachment_id"] == second:
                del repo.files[second]
            return vectors

        monkeypatch.setattr(module, "_call", _call_then_delete)
        gate = FakeGate()

        outcome = await module.run_entry(repo.entries["e1"], repo, gate=gate)

        assert set(repo.vectors) == {first}
        assert gate.successes == 1
        assert outcome == ImageEntryOutcome.done()

    async def test_already_embedded_picture_is_not_posted_again(self, llama_stub):
        stub = llama_stub()
        repo = FakeRepo()
        first, _second = repo.add_entry("e1", 2)
        repo.vectors[first] = {"embedding": None, "skip_reason": "degenerate_vector"}

        await _module(stub.url).run_entry(repo.entries["e1"], repo, gate=FakeGate())

        assert len(stub.embeddings) == 1

    @pytest.mark.parametrize(
        ("switch", "reason"),
        [("refused", "unreachable"), (401, "auth"), (404, "model"), ("short", "config")],
    )
    async def test_call_failure_gives_the_availability_reason(self, llama_stub, switch, reason):
        stub = llama_stub()
        url = stub.url
        if switch == "refused":
            url = _refused_url()
        elif switch == "short":
            stub.short = 16
        else:
            stub.status = switch
        repo = FakeRepo()
        repo.add_entry("e1", 1)

        outcome = await _module(url).run_entry(repo.entries["e1"], repo, gate=FakeGate())

        assert outcome == ImageEntryOutcome.unavailable(reason)
        assert repo.vectors == {}

    async def test_400_refused_by_the_gate_writes_nothing(self, llama_stub):
        stub = llama_stub()
        stub.embed_status = 400
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        gate = FakeGate(deterministic=False)

        outcome = await _module(stub.url).run_entry(repo.entries["e1"], repo, gate=gate)

        assert outcome == ImageEntryOutcome.partial()
        assert repo.vectors == {}
        assert gate.signatures == ["HTTPError:400"]

    async def test_400_allowed_by_the_gate_is_a_skip_row(self, llama_stub):
        stub = llama_stub()
        stub.embed_status = 400
        repo = FakeRepo()
        (picture,) = repo.add_entry("e1", 1)

        outcome = await _module(stub.url).run_entry(repo.entries["e1"], repo, gate=FakeGate())

        assert outcome == ImageEntryOutcome.done()
        assert repo.vectors[picture]["embedding"] is None
        assert repo.vectors[picture]["skip_reason"] == "HTTPError:400"

    async def test_timeout_is_transient(self, llama_stub):
        stub = llama_stub()
        stub.hang = True
        repo = FakeRepo()
        repo.add_entry("e1", 1)

        outcome = await _module(stub.url, timeout_seconds=0.3).run_entry(
            repo.entries["e1"], repo, gate=FakeGate()
        )

        assert outcome.kind == "transient_error"
        assert repo.vectors == {}

    async def test_unconfigured_module_is_a_module_error(self):
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        outcome = await ImageEmbeddingModule().run_entry(repo.entries["e1"], repo, gate=FakeGate())
        assert outcome.kind == "module_error"


# ---------------------------------------------------------------------------
# Through the catch-up driver
# ---------------------------------------------------------------------------


class TestThroughTheDriver:
    async def test_unreachable_server_skips_with_statuses_untouched(self):
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        module = _module(_refused_url())

        result = await drive_image_module(module, repo, budget=None, stop_event=None)

        assert result.skipped == "unavailable"
        assert result.ended == "unreachable"
        assert repo.status_writes == []
        assert repo.batch_marks == 0
        assert repo.entries["e1"]["enhancement_status"] == {}
        assert repo.vectors == {}

    async def test_zero_vector_on_picture_two_of_three_is_one_skip_row(self, llama_stub):
        stub = llama_stub()
        stub.zero_calls = {2}
        repo = FakeRepo()
        first, second, third = repo.add_entry("e1", 3)
        (next_picture,) = repo.add_entry("e2", 1)

        result = await drive_image_module(_module(stub.url), repo, budget=None, stop_event=None)

        assert repo.vectors[second] == {
            "embedding": None,
            "skip_reason": "degenerate_vector",
            "model_ref": MODEL,
        }
        assert repo.vectors[first]["embedding"] is not None
        assert repo.vectors[third]["embedding"] is not None
        assert repo.entries["e1"]["enhancement_status"]["image_embedding"]["status"] == "complete"
        assert repo.vectors[next_picture]["embedding"] is not None
        assert result.entries_walked == 2
        assert result.ended is None

    async def test_server_refusing_after_picture_one_leaves_entry_two_uncharged(self, llama_stub):
        stub = llama_stub()
        stub.refuse_after = 1
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        repo.add_entry("e2", 1)

        result = await drive_image_module(_module(stub.url), repo, budget=None, stop_event=None)

        assert repo.entries["e1"]["enhancement_status"]["image_embedding"]["status"] == "complete"
        assert result.ended == "unreachable"
        assert result.charged == 0
        assert repo.failed == {}
        assert repo.entries["e2"]["enhancement_status"] == {}

    async def test_two_entries_timing_out_in_a_row_charge_only_the_first(self, llama_stub):
        stub = llama_stub()
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        repo.add_entry("e2", 1)
        repo.add_entry("e3", 1)
        module = _module(stub.url, timeout_seconds=0.3)
        assert (await module.health_check()).reachable is True
        stub.hang = True

        result = await drive_image_module(module, repo, budget=None, stop_event=None)

        assert result.ended == "transient"
        assert result.charged == 1
        assert repo.failed == {"e1": 1}
        assert repo.vectors == {}

    async def test_pgvector_less_store_skips_with_one_warning_and_status_config(
        self, llama_stub, caplog
    ):
        from osprey.services.ariel_search.cli_operations import _module_health

        stub = llama_stub()
        repo = FakeRepo(table=False)
        repo.add_entry("e1", 1)

        with caplog.at_level(logging.DEBUG, logger="ariel"):
            result = await drive_image_module(_module(stub.url), repo, budget=None, stop_event=None)
            await drive_image_module(_module(stub.url), repo, budget=None, stop_event=None)

        assert result.ended == "config"
        assert repo.status_writes == []
        assert repo.batch_marks == 0
        assert stub.embeddings == []
        warned = _warnings(caplog, "image_embedding")
        assert len(warned) == 1
        assert "osprey ariel migrate" in warned[0].message

        health = await _module_health(_ariel_config(stub.url), "image_embedding", repo)
        assert health["reachable"] is False
        assert health["reason"] == "config"

    async def test_server_without_picture_input_writes_nothing_and_reports_model(
        self, llama_stub, caplog
    ):
        from osprey.services.ariel_search.cli_operations import _module_health

        stub = llama_stub()
        stub.embed_status = 400
        repo = FakeRepo()
        for entry_id in ("e1", "e2", "e3"):
            repo.add_entry(entry_id, 1)

        with caplog.at_level(logging.DEBUG, logger="ariel"):
            for _ in range(3):
                result = await drive_image_module(
                    _module(stub.url), repo, budget=None, stop_event=None
                )
                assert result.ended == "model"

        assert repo.vectors == {}
        assert repo.failed == {}
        assert len(_warnings(caplog, "image_embedding")) == 1
        health = await _module_health(_ariel_config(stub.url), "image_embedding", repo)
        assert health["reason"] == "model"

    async def test_slow_call_keeps_the_loop_live_and_a_cancel_writes_nothing(self, monkeypatch):
        repo = FakeRepo()
        repo.add_entry("e1", 1)
        module = _module("http://127.0.0.1:8080")
        started = threading.Event()

        async def _healthy() -> Any:
            from osprey.models.providers.health import HealthResult

            return HealthResult(True, "ok", None)

        def _slow(_rendition: dict[str, Any]) -> Any:
            started.set()
            time.sleep(5)
            return [[1.0] + [0.0] * (DIMS - 1)]

        monkeypatch.setattr(module, "health_check", _healthy)
        monkeypatch.setattr(module, "_call", _slow)
        ticks = 0

        async def _ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        ticker = asyncio.create_task(_ticker())
        task = asyncio.create_task(drive_image_module(module, repo, budget=None, stop_event=None))
        while not started.is_set():
            await asyncio.sleep(0.01)
        ticks_at_start = ticks
        await asyncio.sleep(0.3)
        assert ticks > ticks_at_start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        ticker.cancel()

        assert repo.vectors == {}
        assert repo.status_writes == []
        assert _offload.offload_busy("image_embedding") is True


def test_cancelled_pass_lets_asyncio_run_return_within_one_second(monkeypatch):
    repo = FakeRepo()
    repo.add_entry("e1", 1)
    module = _module("http://127.0.0.1:8080")

    async def _healthy() -> Any:
        from osprey.models.providers.health import HealthResult

        return HealthResult(True, "ok", None)

    def _slow(_rendition: dict[str, Any]) -> Any:
        time.sleep(5)
        return [[1.0] + [0.0] * (DIMS - 1)]

    monkeypatch.setattr(module, "health_check", _healthy)
    monkeypatch.setattr(module, "_call", _slow)

    async def _main() -> None:
        task = asyncio.create_task(drive_image_module(module, repo, budget=None, stop_event=None))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    began = time.monotonic()
    asyncio.run(_main())
    assert time.monotonic() - began < 1.3
    assert repo.vectors == {}


# ---------------------------------------------------------------------------
# One image table for every caller
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimensions", [None, 768])
def test_every_caller_resolves_the_same_image_table(dimensions):
    """One config gives one table to migration, module, lane, purge and retry."""
    from types import SimpleNamespace

    from osprey.services.ariel_search import cli_operations as ops
    from osprey.services.ariel_search.database import migrations
    from osprey.services.ariel_search.search.image_lane import ImageLaneSettings

    raw: dict[str, Any] = {
        "database": {"uri": "postgresql://localhost:5432/test"},
        "search_modules": {"hybrid": {"enabled": True}},
        "enhancement_modules": {
            "image_embedding": {
                "enabled": True,
                "provider": {"name": "llama-cpp", "base_url": "http://127.0.0.1:1"},
                "model": MODEL,
            }
        },
    }
    if dimensions is not None:
        raw["enhancement_modules"]["image_embedding"]["dimensions"] = dimensions
    config = ARIELConfig.from_dict(raw)

    (target,) = migrations._image_embedding_args(SimpleNamespace(config=config))
    module = ImageEmbeddingModule()
    module.configure(config.get_enhancement_module_config("image_embedding"))
    assert module.target is not None
    lane = ImageLaneSettings.from_ariel_config(config)

    expected = image_table_name(MODEL, dimensions or 1024)
    assert target.table == expected
    assert module.target.table == expected
    assert module.completion_marker() == expected
    assert lane.table == expected
    assert ops.image_embedding_current_table(config) == expected
