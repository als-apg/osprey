"""The picture lane of hybrid search: the entries whose pictures lie nearest the query.

:func:`search_images` embeds the caller's query text with the configured
picture-embedding model and returns, per entry, its nearest stored picture.
It never raises: every failure (configuration, an unreachable or unhealthy
server, a missing table, a slow statement) is a *lane failure*. A lane failure
returns None, so the caller answers text-only with its diagnostic; it opens a
breaker that skips the lane for :data:`IMAGE_LANE_COOLDOWN_S`; and its reason is
recorded with the availability tracker under :data:`TRACKER_KEY`, which logs one
WARNING when a reason appears or changes, DEBUG while it repeats and one INFO on
the first success after it. The breaker logs nothing of its own.

Every blocking provider step (finding the reachable server, its health verdict,
the embedding call) runs inside one callable on the bounded search-call pool
under one deadline; the vector SQL runs under a pool-acquire timeout, a
statement timeout and the remaining lane budget.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.models.providers.base import TextInput
from osprey.services.ariel_search.attachments.fetch import redact_url
from osprey.services.ariel_search.database.migrations import image_embedding_target
from osprey.services.ariel_search.database.vector_literal import vector_literal
from osprey.services.ariel_search.enhancement import availability
from osprey.services.ariel_search.enhancement.provider_resolver import (
    provider_given,
    resolve_provider,
    resolve_reachable_base_url,
)
from osprey.services.ariel_search.exceptions import ModuleConfigError, ModuleUnavailable
from osprey.services.ariel_search.search._offload import run_search_call
from osprey.services.ariel_search.search.fusion import ImageHit

if TYPE_CHECKING:
    from osprey.models.providers.base import BaseProvider
    from osprey.services.ariel_search.config import ARIELConfig

__all__ = [
    "IMAGE_LANE_COOLDOWN_S",
    "IMAGE_LANE_SQL_TIMEOUT_S",
    "IMAGE_QUERY_TIMEOUT_S",
    "PICTURE_FACTOR",
    "STATEMENT_TIMEOUT_MS",
    "TRACKER_KEY",
    "VERDICT_TTL_S",
    "ImageLaneSettings",
    "last_unavailable_reason",
    "search_images",
]

#: The availability-tracker key lane failures are recorded under.
TRACKER_KEY = "image_lane"

#: The configuration block the lane reads.
IMAGE_EMBEDDING_KEY = "ariel.enhancement_modules.image_embedding"

#: Seconds the embedding call may take; the provider step as a whole gets one more.
IMAGE_QUERY_TIMEOUT_S = 5.0

#: Seconds a lane failure keeps the lane skipped.
IMAGE_LANE_COOLDOWN_S = 30.0

#: Pictures fetched per text candidate the caller fetches.
PICTURE_FACTOR = 3

#: Upper bound on pictures fetched, and on ``hnsw.ef_search`` (pgvector's maximum).
MAX_PICTURES = 1000

#: Lower bound on ``hnsw.ef_search`` (pgvector's default).
MIN_EF_SEARCH = 40

#: Seconds a healthy verdict of the server is reused before it is checked again.
VERDICT_TTL_S = 300.0

#: Seconds the lane waits for a pooled connection.
IMAGE_LANE_SQL_TIMEOUT_S = 2.0

#: Milliseconds the vector statement may run.
STATEMENT_TIMEOUT_MS = 2000

#: The monotonic clock the breaker and the verdict age are read from.
_now: Callable[[], float] = time.monotonic

_lock = threading.Lock()

# Monotonic time before which the lane is skipped; None while closed.
_breaker_until: float | None = None
# The reason the last lane attempt failed for; None when it succeeded or none ran.
_last_reason: str | None = None
# (reachable url, monotonic time of its healthy verdict); None until one is seen.
_verdict: tuple[str | None, float] | None = None
# Set by a lane failure: the next attempt re-checks the server with ``refresh``.
_verdict_stale = False
# The URL the last provider step resolved, for the failure log line.
_last_url: str | None = None
# Bumped by every failure and reset, so a call that outlived its deadline
# cannot write a verdict over a newer state.
_generation = 0
# (weak reference to the config, its settings) of the last config seen.
_settings_cache: tuple[weakref.ref[Any], ImageLaneSettings] | None = None


@dataclass(frozen=True)
class ImageLaneSettings:
    """What the lane needs from ``ariel.enhancement_modules.image_embedding``.

    Attributes:
        provider: The resolved adapter instance that takes every call.
        base_url: The configured base URL, or None for a provider that needs none.
        api_key: The configured API key, or None.
        model_id: The picture-embedding model.
        dims: The stored vector width.
        table: The image table of ``(model_id, dims)``.
    """

    provider: BaseProvider
    base_url: str | None
    api_key: str | None
    model_id: str
    dims: int
    table: str

    @classmethod
    def from_ariel_config(cls, config: ARIELConfig) -> ImageLaneSettings:
        """Build the settings from the ``image_embedding`` block. Does no I/O.

        Raises:
            ModuleConfigError: When the block is absent, names no provider or
                one that serves no image embeddings, or its model or
                dimensions are unusable; each names the key that fixes it.
        """
        module_cfg: Mapping[str, Any] | None = config.get_enhancement_module_config(
            "image_embedding"
        )
        if not module_cfg:
            raise ModuleConfigError(
                f"{IMAGE_EMBEDDING_KEY} is not configured", key=IMAGE_EMBEDDING_KEY
            )
        provider = module_cfg.get("provider")
        if not provider_given(provider):
            raise ModuleConfigError(
                f"{IMAGE_EMBEDDING_KEY}.provider is required",
                key=f"{IMAGE_EMBEDDING_KEY}.provider",
            )
        provider_key = module_cfg.get("provider_key") or f"{IMAGE_EMBEDDING_KEY}.provider"
        resolved = resolve_provider(
            provider, provider_key=provider_key, default=None, serves="image_embeddings"
        )
        target = image_embedding_target(module_cfg)
        return cls(
            provider=resolved.instance,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            model_id=target.model,
            dims=target.dims,
            table=target.table,
        )


def _settings_for(config: ARIELConfig) -> ImageLaneSettings:
    """The settings of ``config``, built once and reused while it is the same object."""
    global _settings_cache
    cached = _settings_cache
    if cached is not None and cached[0]() is config:
        return cached[1]
    settings = ImageLaneSettings.from_ariel_config(config)
    _settings_cache = (weakref.ref(config), settings)
    return settings


def last_unavailable_reason() -> str | None:
    """The reason the last lane attempt failed for, or None.

    Kept until the next lane attempt resolves it (a success clears it, a
    failure replaces it); a breaker cooldown ending does not clear it.
    """
    return _last_reason


def picture_count(fetch_limit: int) -> int:
    """Pictures fetched for ``fetch_limit`` text candidates."""
    return min(fetch_limit * PICTURE_FACTOR, MAX_PICTURES)


def _ef_search(k: int) -> str:
    """``hnsw.ef_search`` for ``k`` rows: at least ``k``, so the index returns all of them."""
    return str(min(MAX_PICTURES, max(k, MIN_EF_SEARCH)))


def _embed_query(settings: ImageLaneSettings, query: str, generation: int) -> list[float]:
    """The blocking provider step: resolve the server if stale, check it, embed ``query``.

    Raises:
        ModuleUnavailable: When the server's health verdict is not healthy.
        Exception: Whatever the resolver or the embedding call raised.
    """
    global _verdict, _verdict_stale, _last_url
    with _lock:
        verdict = _verdict
        stale = _verdict_stale
    now = _now()
    if verdict is None or stale or now - verdict[1] > VERDICT_TTL_S:
        url = settings.base_url
        if url is not None:
            url = resolve_reachable_base_url(
                type(settings.provider),
                url,
                deadline_s=IMAGE_QUERY_TIMEOUT_S / 2,
                refresh=verdict is not None or stale,
            )
        with _lock:
            _last_url = url
        health = settings.provider.check_embedding_health(
            settings.api_key, url, model_id=settings.model_id, timeout=IMAGE_QUERY_TIMEOUT_S / 2
        )
        if health.reachable is False:
            raise ModuleUnavailable(health.reason or "unreachable", health.message)
        with _lock:
            if generation == _generation:
                _verdict = (url, now)
                _verdict_stale = False
    else:
        url = verdict[0]
    (vector,) = settings.provider.execute_image_embedding(
        [TextInput(query)],
        settings.model_id,
        api_key=settings.api_key,
        base_url=url,
        dimensions=settings.dims,
        timeout=IMAGE_QUERY_TIMEOUT_S,
    )
    return vector


_NEAREST_SQL = """
SELECT DISTINCT ON (f.entry_id)
       f.entry_id, t.attachment_id, 1 - (t.embedding <=> %(q)s::vector) AS similarity
FROM (
    SELECT attachment_id, embedding
    FROM {table}
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(q)s::vector
    LIMIT %(k)s
) t
JOIN attachment_files f USING (attachment_id)
ORDER BY f.entry_id, similarity DESC
"""


def _row_values(row: Any) -> tuple[str, str, float]:
    """``(entry_id, attachment_id, similarity)`` of a tuple or dict row."""
    if isinstance(row, Mapping):
        return str(row["entry_id"]), str(row["attachment_id"]), float(row["similarity"])
    return str(row[0]), str(row[1]), float(row[2])


async def _nearest_pictures(
    pool: Any, settings: ImageLaneSettings, vector: list[float], k: int
) -> dict[str, ImageHit]:
    """Each entry's nearest picture among the ``k`` pictures nearest ``vector``."""
    params = {"q": vector_literal(vector), "k": k}
    async with pool.connection(timeout=IMAGE_LANE_SQL_TIMEOUT_S) as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('hnsw.ef_search', %(ef)s, true), "
                "set_config('statement_timeout', %(st)s, true)",
                {"ef": _ef_search(k), "st": str(STATEMENT_TIMEOUT_MS)},
            )
            cursor = await conn.execute(_NEAREST_SQL.format(table=settings.table), params)
            rows = await cursor.fetchall()
    hits: dict[str, ImageHit] = {}
    for row in rows:
        entry_id, attachment_id, similarity = _row_values(row)
        hits.setdefault(entry_id, ImageHit(attachment_id, similarity))
    return hits


async def search_images(
    query: str, repository: Any, config: ARIELConfig, *, fetch_limit: int
) -> dict[str, ImageHit] | None:
    """The entries whose pictures lie nearest ``query``, or None on a lane failure.

    Call only when picture search is in effect for the request.

    Args:
        query: The caller's original query text (never a vocabulary expansion).
        repository: Repository whose ``pool`` holds the image table.
        config: The ARIEL configuration; its ``image_embedding`` block is read
            once per config object.
        fetch_limit: Text candidates the caller fetches; the lane fetches
            :data:`PICTURE_FACTOR` times as many pictures, at most
            :data:`MAX_PICTURES`.

    Returns:
        ``{entry_id: ImageHit}`` (possibly empty) on success; None when the
        lane failed or its breaker is open.
    """
    if _breaker_until is not None and _now() < _breaker_until:
        return None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + IMAGE_QUERY_TIMEOUT_S + 1
    settings: ImageLaneSettings | None = None
    generation = _generation
    try:
        settings = _settings_for(config)
        vector = await run_search_call(
            _embed_query, settings, query, generation, timeout_s=IMAGE_QUERY_TIMEOUT_S + 1
        )
        sql_budget = min(
            deadline - loop.time(), IMAGE_LANE_SQL_TIMEOUT_S + STATEMENT_TIMEOUT_MS / 1000
        )
        hits = await asyncio.wait_for(
            _nearest_pictures(repository.pool, settings, vector, picture_count(fetch_limit)),
            max(sql_budget, 0.0),
        )
    except Exception as exc:
        _fail(exc, settings)
        return None
    _succeed()
    return hits


def _fail(exc: BaseException, settings: ImageLaneSettings | None) -> None:
    """Record a lane failure: open the breaker, mark the verdict stale, report the reason."""
    global _breaker_until, _last_reason, _verdict_stale, _generation
    reason = availability.unavailable_reason(exc) or "unreachable"
    with _lock:
        _breaker_until = _now() + IMAGE_LANE_COOLDOWN_S
        _last_reason = reason
        _verdict_stale = True
        _generation += 1
        url = _last_url
    if url is None and settings is not None:
        url = settings.base_url
    where = f" at {redact_url(url)}" if url else ""
    message = f"{type(exc).__name__}{where}"
    if isinstance(exc, (ModuleConfigError, ModuleUnavailable)):
        message = f"{message}: {exc}"
    availability.report_unavailable(
        TRACKER_KEY, reason, message, availability.fix_for("image_embedding", reason, exc)
    )


def _succeed() -> None:
    """Record a lane success: close the breaker, clear the reason."""
    global _breaker_until, _last_reason
    _breaker_until = None
    _last_reason = None
    availability.report_available(TRACKER_KEY)


def _reset_state() -> None:
    """Forget the breaker, the reason, the verdict and the settings cache. For tests."""
    global _breaker_until, _last_reason, _verdict, _verdict_stale, _last_url, _generation
    global _settings_cache
    with _lock:
        _breaker_until = None
        _last_reason = None
        _verdict = None
        _verdict_stale = False
        _last_url = None
        _generation += 1
        _settings_cache = None
