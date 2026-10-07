"""``osprey ariel enhance`` on the picture modules, against a real PostgreSQL schema.

``run_enhance`` drains a picture module through the same driver as the
catch-up, with no budget: under its advisory lock, holding a pool connection
only around database work, never completing an entry that still has a picture
waiting to be copied. ``--retry-failed`` forgets a module's per-picture
failures so the next pass tries them again.

Every case runs on a fresh scratch database. ``image_embedding`` runs the real
llama-cpp adapter against the ``llama_stub`` server; ``image_caption`` runs the
real module with its vision call replaced by a fake.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from osprey.models.providers.health import HealthResult
from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.database.migrations import image_table_name
from osprey.services.ariel_search.database.repository import CopyRendition
from osprey.services.ariel_search.enhancement import _offload, availability
from osprey.services.ariel_search.enhancement.image_caption import module as caption_mod
from osprey.services.ariel_search.enhancement.image_embedding.module import ImageEmbeddingModule
from osprey.services.ariel_search.enhancement.vision_errors import EmptyReplyError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.ingestion.ingest import ingest_one
from tests.services.ariel_search.llama_stub import MODEL as LLAMA_MODEL

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(180)]

DIMS = 1024
TABLE = image_table_name(LLAMA_MODEL, DIMS)
EMBED = "image_embedding"
CAPTION = "image_caption"
CAPTION_MODEL = "vis-caption-1"
ORIGINS = frozenset({("https", "h.example", 443)})
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
RENDITION = CopyRendition(data=b"png", mime_type="image/png", width=1, height=1, sha256="cd" * 32)
TS = datetime(2026, 9, 1, tzinfo=UTC)
REPLY = "Klystron arc trace on the scope.\nVisible text: KLY-3"


# --- helpers -----------------------------------------------------------------------


class _Adapter(FacilityAdapter):
    """An http source whose pictures live on ``h.example``."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    def attachment_origins(self) -> frozenset:
        return ORIGINS

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():
            yield entry


def _config_dict(uri: str, *, embed_url: str | None = None, caption: bool = False) -> dict:
    """The raw ``ariel`` block: the picture modules asked for, on *uri*."""
    modules: dict[str, Any] = {}
    if embed_url is not None:
        modules[EMBED] = {
            "enabled": True,
            "provider": {"name": "llama-cpp", "base_url": embed_url},
            "model": LLAMA_MODEL,
            "dimensions": DIMS,
        }
    if caption:
        modules[CAPTION] = {
            "enabled": True,
            "provider": "openai",
            "model": {"model_id": CAPTION_MODEL},
            "timeout_seconds": 60,
        }
    return {
        "database": {"uri": uri},
        "attachments": {"copy_on_ingest": "images"},
        "ingestion": {
            "adapter": "generic_json",
            "source_url": "https://h.example/api",
            "watch": {"require_initial_ingest": False},
        },
        "search_modules": {"keyword": {"enabled": True}, "hybrid": {"enabled": True}},
        "enhancement_modules": modules,
    }


def _url(name: str) -> str:
    return f"https://h.example/files/{name}"


def _id(entry_id: str, name: str) -> str:
    aid = attachment_id_for(entry_id, {"url": _url(name)})
    assert aid is not None
    return aid


async def _seed(repo, cfg: dict, entry_id: str, copied: list[str], pending: tuple = ()) -> None:
    """Store an entry with one PNG per name; ``copied`` ones get a rendition."""
    config = ARIELConfig.from_dict(cfg)
    names = [*copied, *pending]
    entry = {
        "entry_id": entry_id,
        "source_system": "test",
        "timestamp": TS,
        "author": "tester",
        "raw_text": "beam lost at 14:02",
        "attachments": [{"url": _url(n), "type": "image/png", "filename": n} for n in names],
        "metadata": {},
        "enhancement_status": {},
    }
    await ingest_one(entry, _Adapter(config), repo, [], config, None)
    for name in copied:
        written = await repo.apply_copy_outcome(
            entry_id,
            _id(entry_id, name),
            copy_status="copied",
            data=PNG_MAGIC,
            mime_type="image/png",
            size_bytes=len(PNG_MAGIC),
            rendition=RENDITION,
        )
        assert written == "copied"


def _status(uri: str, entry_id: str) -> dict[str, Any]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT enhancement_status FROM enhanced_entries WHERE entry_id = %s", (entry_id,)
        ).fetchone()
    assert row is not None
    return row[0] or {}


def _vectors(uri: str) -> dict[str, bool]:
    """``attachment_id -> has a vector`` for every row of the image table."""
    with psycopg.connect(uri) as conn:
        rows = conn.execute(f"SELECT attachment_id, embedding IS NOT NULL FROM {TABLE}").fetchall()
    return {r[0]: r[1] for r in rows}


def _captions(uri: str, entry_id: str) -> dict[str, Any]:
    with psycopg.connect(uri) as conn:
        row = conn.execute(
            "SELECT attachment_captions FROM enhanced_entries WHERE entry_id = %s", (entry_id,)
        ).fetchone()
    assert row is not None
    return row[0] or {}


def _closed_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh availability and offload state; a configured caption provider."""
    availability.reset_availability()
    _offload.reset_offload_state()
    monkeypatch.setattr("osprey.models.config.get_provider_config", lambda name: {"api_key": "k"})
    monkeypatch.setattr(
        caption_mod, "probe_models_endpoint", lambda *a, **k: HealthResult(True, "served", None)
    )
    yield
    availability.reset_availability()
    _offload.reset_offload_state()


@pytest.fixture
def stub(llama_stub):
    return llama_stub()


@pytest.fixture
async def repo(scratch_database, stub):
    """An ``ARIELRepository`` on a fresh scratch database, the image table migrated."""
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    cfg = ARIELConfig.from_dict(_config_dict(scratch_database, embed_url=stub.url, caption=True))
    pool = await create_connection_pool(cfg.database)
    try:
        await run_migrations(pool, cfg)
        yield ARIELRepository(pool, cfg)
    finally:
        await pool.close()


# --- the drain of image_embedding ----------------------------------------------------


class TestEnhanceImageEmbedding:
    async def test_an_entry_with_a_pending_copy_stays_incomplete(
        self, repo, scratch_database, stub, attachment_fetch
    ):
        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "pend-1", ["a.png"], pending=("b.png",))

        result = await ops.run_enhance(cfg, module=EMBED, force=False, limit=100)

        assert result.module_names == [EMBED]
        assert _vectors(scratch_database) == {_id("pend-1", "a.png"): True}
        assert _status(scratch_database, "pend-1").get(EMBED, {}).get("status") != "complete"
        assert attachment_fetch is None or attachment_fetch.calls == []

    async def test_a_lock_held_elsewhere_prints_and_embeds_nothing(
        self, repo, scratch_database, stub
    ):
        from osprey.services.ariel_search.database.connection import try_advisory_lock

        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "lock-1", ["a.png"])
        lines: list[str] = []

        async with try_advisory_lock(scratch_database, f"ariel_enhance:{EMBED}") as held:
            assert held
            await ops.run_enhance(cfg, module=EMBED, force=False, limit=100, progress=lines.append)

        assert f"{EMBED}: running in another process" in lines
        assert _vectors(scratch_database) == {}
        assert stub.embeddings == []

    async def test_no_pool_connection_is_held_during_a_two_second_embed_call(
        self, repo, scratch_database, stub, monkeypatch
    ):
        import osprey.services.ariel_search as ariel_pkg

        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "slow-1", ["a.png"])
        pools: list[Any] = []
        real_service = ariel_pkg.create_ariel_service

        async def _service(config):
            service = await real_service(config)
            pools.append(service.pool)
            return service

        monkeypatch.setattr(ariel_pkg, "create_ariel_service", _service)
        used: list[int] = []
        real_call = ImageEmbeddingModule._call

        def _slow_call(self, rendition):
            time.sleep(1.0)
            stats = pools[-1].get_stats()
            used.append(stats.get("pool_size", 0) - stats.get("pool_available", 0))
            time.sleep(1.0)
            return real_call(self, rendition)

        monkeypatch.setattr(ImageEmbeddingModule, "_call", _slow_call)

        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100)

        assert used == [0]
        assert _vectors(scratch_database) == {_id("slow-1", "a.png"): True}

    async def test_bare_enhance_writes_vectors_and_no_not_implemented_failure(
        self, repo, scratch_database, stub
    ):
        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "bare-1", ["a.png", "b.png"])

        result = await ops.run_enhance(cfg, module=None, force=False, limit=100)

        assert EMBED in result.module_names
        assert _vectors(scratch_database) == {
            _id("bare-1", "a.png"): True,
            _id("bare-1", "b.png"): True,
        }
        status = _status(scratch_database, "bare-1")
        assert "NotImplementedError" not in str(status)
        assert status[EMBED]["status"] == "complete"
        assert status[EMBED]["marker"] == TABLE

    async def test_purge_then_migrate_then_enhance_rebuilds_the_vectors(
        self, repo, scratch_database, stub
    ):
        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "purge-1", ["a.png"])
        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100)
        assert _vectors(scratch_database) == {_id("purge-1", "a.png"): True}

        await ops.execute_purge(cfg, embeddings_only=True)
        await ops.run_migrate(cfg)
        assert _vectors(scratch_database) == {}
        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100)

        assert _vectors(scratch_database) == {_id("purge-1", "a.png"): True}

    @pytest.mark.parametrize("module", [EMBED, None])
    async def test_a_closed_port_prints_the_skip_reason_and_writes_no_key(
        self, repo, scratch_database, module
    ):
        cfg = _config_dict(scratch_database, embed_url=_closed_url())
        await _seed(repo, cfg, "closed-1", ["a.png"])
        lines: list[str] = []

        await ops.run_enhance(cfg, module=module, force=False, limit=100, progress=lines.append)

        assert f"{EMBED}: skipped, unavailable (unreachable)" in lines
        assert EMBED not in _status(scratch_database, "closed-1")
        assert _vectors(scratch_database) == {}

    async def test_retry_failed_forgets_skip_rows_and_the_next_pass_embeds(
        self, repo, scratch_database, stub
    ):
        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "skip-1", ["a.png"])
        aid = _id("skip-1", "a.png")
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute(
                f"INSERT INTO {TABLE} (attachment_id, embedding, skip_reason, model_ref)"
                " VALUES (%s, NULL, 'DegenerateVectorError', %s)",
                (aid, LLAMA_MODEL),
            )
        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100)
        assert _vectors(scratch_database) == {aid: False}
        assert _status(scratch_database, "skip-1")[EMBED]["status"] == "complete"
        assert stub.embeddings == []

        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100, retry_failed=True)

        assert _vectors(scratch_database) == {aid: True}
        assert _status(scratch_database, "skip-1")[EMBED]["status"] == "complete"

    async def test_retry_failed_resets_gave_up(self, repo, scratch_database, stub):
        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "gave-1", ["a.png"])
        for _ in range(3):
            await repo.mark_enhancement_failed("gave-1", EMBED, "TimeoutError", marker=TABLE)
        assert _status(scratch_database, "gave-1")[EMBED]["gave_up"] is True
        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100)
        assert _vectors(scratch_database) == {}

        await ops.run_enhance(cfg, module=EMBED, force=False, limit=100, retry_failed=True)

        assert _vectors(scratch_database) == {_id("gave-1", "a.png"): True}
        assert _status(scratch_database, "gave-1")[EMBED]["status"] == "complete"

    async def test_retry_failed_counts_an_entry_once(self, repo, scratch_database, stub):
        cfg = _config_dict(scratch_database, embed_url=stub.url)
        await _seed(repo, cfg, "both-1", ["a.png"])
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute(
                f"INSERT INTO {TABLE} (attachment_id, embedding, skip_reason, model_ref)"
                " VALUES (%s, NULL, 'DegenerateVectorError', %s)",
                (_id("both-1", "a.png"), LLAMA_MODEL),
            )
        for _ in range(3):
            await repo.mark_enhancement_failed("both-1", EMBED, "TimeoutError", marker=TABLE)
        lines: list[str] = []

        reset = await ops.retry_failed_entries(ops._ariel_config(cfg), EMBED, lines.append)

        assert reset == 1
        assert lines == [f"{EMBED}: 1 failed entries will be retried"]


# --- --retry-failed on image_caption -------------------------------------------------


class _Vision:
    """The vision call: raises an empty reply for the first ``fail`` calls, then answers."""

    def __init__(self, fail: int) -> None:
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, **_kwargs: Any) -> str:
        with self._lock:
            self.calls += 1
            number = self.calls
        if number <= self.fail:
            raise EmptyReplyError("the model answered nothing")
        return REPLY


class TestRetryFailedCaptions:
    async def test_a_deterministic_failure_is_retried_into_a_caption(
        self, repo, scratch_database, monkeypatch
    ):
        cfg = _config_dict(scratch_database, caption=True)
        await _seed(repo, cfg, "cap-1", ["a.png", "b.png"])
        aid, over = _id("cap-1", "a.png"), _id("cap-1", "b.png")
        # One picture already set aside over the cap: a decision, never retried.
        with psycopg.connect(scratch_database, autocommit=True) as conn:
            conn.execute(
                "UPDATE enhanced_entries SET attachment_captions = %s::jsonb WHERE entry_id = %s",
                (
                    psycopg.types.json.Jsonb({over: {CAPTION_MODEL: {"error": "over_image_cap"}}}),
                    "cap-1",
                ),
            )
        # A failure is stored only once the module has succeeded in this process.
        availability.note_success(CAPTION, CAPTION_MODEL)
        vision = _Vision(fail=1)
        monkeypatch.setattr(caption_mod, "_chat_completion", vision)

        await ops.run_enhance(cfg, module=CAPTION, force=False, limit=100)

        captions = _captions(scratch_database, "cap-1")
        assert captions[aid][CAPTION_MODEL] == {
            "error": "EmptyReplyError: the model answered nothing"
        }
        assert _status(scratch_database, "cap-1")[CAPTION]["status"] == "complete"
        assert vision.calls == 1

        await ops.run_enhance(cfg, module=CAPTION, force=False, limit=100, retry_failed=True)

        captions = _captions(scratch_database, "cap-1")
        assert captions[aid][CAPTION_MODEL]["caption"] == "Klystron arc trace on the scope."
        assert captions[over][CAPTION_MODEL] == {"error": "over_image_cap"}
        assert _status(scratch_database, "cap-1")[CAPTION]["status"] == "complete"
        assert vision.calls == 2
