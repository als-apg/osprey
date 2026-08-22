"""Shared fixtures for the ARIEL search integration suite.

Everything here is opt-in: no autouse fixtures, so the sibling integration
modules that predate this file are untouched by its presence.

The pieces are:

* a **recording pool** wrapper that sits between :class:`ARIELRepository` and a
  real ``psycopg`` pool and logs every statement it issues (plus the value of
  ``statement_timeout`` observed *inside* the pattern transaction). That is the
  only way to assert, against a real database, that a pattern-free search opens
  no transaction and that the ``SET LOCAL`` envelope actually engaged.
* **vocabulary files** written to a tmp directory, so the tests exercise the
  real loader rather than a hand-built ``Vocabulary``.
* **seed fixtures** for the three row groups the vocabulary/pattern criteria
  need. Each seeds once per package -- the package-scoped
  :func:`seeded_prefixes` ledger memoizes what is already in the database --
  and every ``entry_id`` carries a distinctive prefix, so the ledger's
  finalizer deletes exactly these rows and no others. Teardown is a fixture
  finalizer rather than a trailing test, so it also runs when a test fails, a
  selection deselects part of the file, or the session is interrupted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

# ---------------------------------------------------------------------------
# Entry-id prefixes. Cleanup deletes exactly these.
# ---------------------------------------------------------------------------

#: Prose rows: canonical-only needles, decoys, the acronym-only row.
VOCAB_PREFIX = "vocab-prose-"

#: Control-system-name rows the glob/regex criteria match against.
GLOB_PREFIX = "vocab-glob-"

#: The bulky pattern fixture (2000 rows x ~2 KB).
BULK_PREFIX = "vocab-bulk-"

#: Rows in the bulky fixture, and the literal every one of them embeds.
BULK_ROW_COUNT = 2000
BULK_LITERAL = "SR01C___BPM3 trip aborted"

_BULK_FILLER = "orbit correction sequence completed without incident "


# ---------------------------------------------------------------------------
# Recording pool
# ---------------------------------------------------------------------------


def _scalar(row: Any) -> Any:
    """Return the single column of a result row, whatever the row factory."""
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


class _RecordingCursor:
    """Cursor proxy that records every statement it executes."""

    def __init__(self, cursor: Any, log: list[tuple[str, str, list[Any]]]) -> None:
        self._cursor = cursor
        self._log = log

    async def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        """Record and delegate."""
        self._log.append(("cursor.execute", str(query), list(params or [])))
        return await self._cursor.execute(query, params, **kwargs)

    # Python resolves the async protocols on the TYPE, so ``__getattr__`` never
    # sees them: a streaming ``async for row in cur`` would fail inside the
    # proxy rather than exercising the repository. Forward them explicitly.
    def __aiter__(self) -> Any:
        return self._cursor.__aiter__()

    async def __anext__(self) -> Any:
        return await self._cursor.__anext__()

    async def __aenter__(self) -> _RecordingCursor:
        await self._cursor.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._cursor.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _RecordingAsyncCM:
    """Async context manager proxy that wraps whatever the inner one yields."""

    def __init__(self, inner: Any, log: list[tuple[str, str, list[Any]]], wrapper: Any) -> None:
        self._inner = inner
        self._log = log
        self._wrapper = wrapper

    async def __aenter__(self) -> Any:
        return self._wrapper(await self._inner.__aenter__(), self._log)

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)


class _RecordingTransaction:
    """Transaction proxy that records the moment the block is ENTERED.

    Constructing ``conn.transaction()`` is not the same as opening one, and
    :attr:`RecordingPool.opened_transaction` documents the latter, so the log
    entry is appended from ``__aenter__``.
    """

    def __init__(self, inner: Any, log: list[tuple[str, str, list[Any]]]) -> None:
        self._inner = inner
        self._log = log

    async def __aenter__(self) -> Any:
        entered = await self._inner.__aenter__()
        self._log.append(("transaction", "BEGIN", []))
        return entered

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)


class _RecordingConnection:
    """Connection proxy that records statements, transactions and the timeout.

    ``SHOW statement_timeout`` is issued on the SAME connection immediately
    after the repository's ``set_config`` call, which is the only vantage point
    from which the ``SET LOCAL`` value is observable -- it is reverted the
    moment the transaction commits.
    """

    def __init__(self, conn: Any, log: list[tuple[str, str, list[Any]]]) -> None:
        self._conn = conn
        self._log = log
        # Which physical backend this checkout landed on. ``SET LOCAL`` is only
        # observable from the connection that ran it, so a test asserting the
        # value was reverted has to prove it looked at the same backend.
        self._log.append(("connection", str(conn.info.backend_pid), []))

    async def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        """Record, delegate, and probe ``statement_timeout`` after a set_config."""
        text = str(query)
        self._log.append(("conn.execute", text, list(params or [])))
        result = await self._conn.execute(query, params, **kwargs)
        if "set_config" in text:
            shown = await self._conn.execute("SHOW statement_timeout")
            row = await shown.fetchone()
            self._log.append(("statement_timeout_inside", str(_scalar(row)), []))
        return result

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        """Return a recording cursor context manager."""
        return _RecordingAsyncCM(self._conn.cursor(*args, **kwargs), self._log, _RecordingCursor)

    def transaction(self, *args: Any, **kwargs: Any) -> Any:
        """Return a transaction proxy that records when the block is entered."""
        return _RecordingTransaction(self._conn.transaction(*args, **kwargs), self._log)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class RecordingPool:
    """Pool proxy handing out :class:`_RecordingConnection` objects.

    Attributes:
        log: Ordered ``(kind, text, params)`` triples. ``kind`` is one of
            ``connection``, ``conn.execute``, ``cursor.execute``,
            ``transaction`` or ``statement_timeout_inside``.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.log: list[tuple[str, str, list[Any]]] = []

    def connection(self, *args: Any, **kwargs: Any) -> Any:
        """Return a recording connection context manager."""
        return _RecordingAsyncCM(
            self._pool.connection(*args, **kwargs), self.log, _RecordingConnection
        )

    def clear(self) -> None:
        """Forget every recorded statement."""
        self.log.clear()

    @property
    def statements(self) -> list[str]:
        """Every SQL string issued, in order."""
        return [text for kind, text, _ in self.log if kind.endswith("execute")]

    @property
    def pattern_clause_count(self) -> int:
        """Total number of ``~*`` predicates across every statement issued."""
        return sum(text.count("~*") for text in self.statements)

    @property
    def bound_params(self) -> list[Any]:
        """Every bound parameter, flattened in statement order."""
        return [
            value for kind, _, params in self.log if kind.endswith("execute") for value in params
        ]

    @property
    def opened_transaction(self) -> bool:
        """Whether ``conn.transaction()`` was entered at least once."""
        return any(kind == "transaction" for kind, _, _ in self.log)

    @property
    def backend_pids(self) -> list[int]:
        """Backend PID of every connection handed out, in checkout order."""
        return [int(text) for kind, text, _ in self.log if kind == "connection"]

    @property
    def timeouts_inside(self) -> list[str]:
        """``statement_timeout`` as observed inside each pattern transaction."""
        return [text for kind, text, _ in self.log if kind == "statement_timeout_inside"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


# ---------------------------------------------------------------------------
# Vocabulary files
# ---------------------------------------------------------------------------

_BASE_VOCABULARY = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms: [ts, t/s]
  - canonical: beam position monitor
    kind: acronym
    forms: [bpm, bpms]
  - canonical: input output
    kind: shorthand
    forms: [i/o]
"""

_AMBIGUOUS_EXTRA = """  - canonical: timing system
    kind: acronym
    forms: [ts]
"""


@pytest.fixture(scope="session")
def vocabulary_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A vocabulary binding ``ts``/``t/s``, ``bpm``/``bpms`` and ``i/o``."""
    path = tmp_path_factory.mktemp("ariel-vocab") / "vocabulary.yml"
    path.write_text(_BASE_VOCABULARY, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def ambiguous_vocabulary_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same vocabulary, with ``ts`` additionally bound to ``timing system``."""
    path = tmp_path_factory.mktemp("ariel-vocab-ambiguous") / "vocabulary.yml"
    path.write_text(_BASE_VOCABULARY + _AMBIGUOUS_EXTRA, encoding="utf-8")
    return path


@pytest.fixture
def vocabulary_config_factory(database_url: str):
    """Factory building an :class:`ARIELConfig` against the test database.

    Args:
        database_url: Session database URL from the package conftest.

    Returns:
        ``factory(vocabulary_path=None, **vocabulary_keys) -> ARIELConfig``.
        ``vocabulary_path`` of ``None`` leaves the block absent entirely;
        every other keyword lands in the ``ariel.vocabulary`` mapping, so a
        test can omit ``expand_by_default`` to exercise the unset path.
    """
    from osprey.services.ariel_search.config import ARIELConfig

    def _factory(
        vocabulary_path: Path | str | None = None,
        *,
        keyword_settings: dict[str, Any] | None = None,
        **vocabulary_keys: Any,
    ) -> ARIELConfig:
        keyword: dict[str, Any] = {"enabled": True}
        if keyword_settings is not None:
            keyword["settings"] = keyword_settings
        config_dict: dict[str, Any] = {
            "database": {"uri": database_url},
            "search_modules": {
                "keyword": keyword,
                "semantic": {"enabled": False},
            },
            "enhancement_modules": {
                "semantic_processor": {"enabled": False},
                "text_embedding": {"enabled": False},
            },
        }
        if vocabulary_path is not None:
            config_dict["vocabulary"] = {
                "enabled": True,
                "path": str(vocabulary_path),
                **vocabulary_keys,
            }
        elif vocabulary_keys:
            config_dict["vocabulary"] = {"enabled": False, **vocabulary_keys}
        return ARIELConfig.from_dict(config_dict)

    return _factory


# ---------------------------------------------------------------------------
# Seeded rows
# ---------------------------------------------------------------------------

#: The canonical-only needle: short, so the fuzzy path can reach it.
SHORT_NEEDLE_ID = f"{VOCAB_PREFIX}needle-short"
#: The needle that also mentions "undulator".
THREE_TERM_NEEDLE_ID = f"{VOCAB_PREFIX}needle-three"
#: Contains only the acronym, never the canonical -- the direction-gate probe.
ACRONYM_ONLY_ID = f"{VOCAB_PREFIX}acronym-only"
#: Contains the bare literal token "ab" for the literal-run rule.
LITERAL_AB_ID = f"{VOCAB_PREFIX}literal-ab"

_PROSE_ROWS: tuple[tuple[str, str], ...] = (
    (SHORT_NEEDLE_ID, "Had to troubleshoot the beam position monitor offset today."),
    (
        THREE_TERM_NEEDLE_ID,
        "Spent most of the shift trying to troubleshoot the beam position monitor "
        "readings near the undulator straight, then handed the work over.",
    ),
    (ACRONYM_ONLY_ID, "BPM readings drifted after the ramp and were rezeroed."),
    (LITERAL_AB_ID, "Channel ab reset by the operator on request."),
    (f"{VOCAB_PREFIX}decoy-undulator", "The undulator gap was scanned without incident."),
    (f"{VOCAB_PREFIX}decoy-vacuum", "Vacuum interlock tripped in the injector area."),
    (f"{VOCAB_PREFIX}decoy-timing", "The timing system reference was realigned overnight."),
)

#: Glob probe rows, keyed by the control-system name each one carries.
GLOB_BPM3_ID = f"{GLOB_PREFIX}bpm3"
GLOB_BARE_ID = f"{GLOB_PREFIX}bare"
GLOB_PREFIXED_ID = f"{GLOB_PREFIX}prefixed"
GLOB_SR02_ID = f"{GLOB_PREFIX}sr02"
GLOB_SR011_ID = f"{GLOB_PREFIX}sr011"
GLOB_BEAM_LOSS_ID = f"{GLOB_PREFIX}beam-loss"

_GLOB_ROWS: tuple[tuple[str, str], ...] = (
    (GLOB_BPM3_ID, "Signal from SR01C___BPM3 looked noisy after the ramp."),
    (GLOB_BARE_ID, "Recabled SR01C___BPM during the shutdown."),
    (GLOB_PREFIXED_ID, "Legacy alias XSR01C___BPM3 still exists in the old map."),
    (GLOB_SR02_ID, "Checked SR02C___BPM for drift."),
    (GLOB_SR011_ID, "Retired name SR011C___BPM3 appears in the archive only."),
    (GLOB_BEAM_LOSS_ID, "A beam loss event was recorded at sector five."),
)


async def _seed_rows(repository: Any, factory: Any, rows: tuple[tuple[str, str], ...]) -> None:
    """Upsert ``(entry_id, raw_text)`` pairs with a recent timestamp."""
    now = datetime.now(UTC)
    for entry_id, raw_text in rows:
        await repository.upsert_entry(
            factory(
                entry_id=entry_id,
                raw_text=raw_text,
                author="vocabulary_fixture",
                timestamp=now,
            )
        )


@pytest.fixture(scope="package")
def seeded_prefixes(database_url: str):
    """Ledger of the entry-id prefixes seeded so far, cleaned up at teardown.

    Doubles as the seed memo and the teardown owner. Every seed fixture below
    records its prefix here *before* inserting, so a run interrupted halfway
    through a seed still has its partial rows deleted, and the finalizer issues
    one ``DELETE ... LIKE '<prefix>%'`` per recorded prefix.

    It is a plain (synchronous) fixture on purpose: the pool fixtures it would
    otherwise reuse are function-scoped, and an async fixture of package scope
    would need an event loop of the same scope. Teardown therefore opens its own
    short-lived ``psycopg`` connection, which also means it works after the last
    test's pool has already been closed.

    Args:
        database_url: Session database URL from the package conftest.

    Yields:
        The mutable set of prefixes already present in the database.
    """
    import psycopg

    seeded: set[str] = set()
    yield seeded
    if not seeded:
        return
    with psycopg.connect(database_url, autocommit=True) as conn:
        for prefix in sorted(seeded):
            conn.execute("DELETE FROM enhanced_entries WHERE entry_id LIKE %s", (f"{prefix}%",))


@pytest.fixture
async def prose_repository(repository, seed_entry_factory, seeded_prefixes):
    """Repository seeded with the canonical-only needles and their decoys."""
    if VOCAB_PREFIX not in seeded_prefixes:
        seeded_prefixes.add(VOCAB_PREFIX)
        await _seed_rows(repository, seed_entry_factory, _PROSE_ROWS)
    return repository


@pytest.fixture
async def glob_repository(repository, seed_entry_factory, seeded_prefixes):
    """Repository seeded with the control-system-name rows."""
    if GLOB_PREFIX not in seeded_prefixes:
        seeded_prefixes.add(GLOB_PREFIX)
        await _seed_rows(repository, seed_entry_factory, _GLOB_ROWS)
    return repository


def bulk_raw_text(index: int) -> str:
    """Build one ~2 KB bulky row, embedding :data:`BULK_LITERAL`.

    Args:
        index: Row number, so each row is distinguishable.

    Returns:
        The ``raw_text`` for that row -- roughly 2 KB, carrying the literal the
        timeout pattern matches and the word ``troubleshoot`` so the expanded
        full-text leg of ``ts /.../`` is satisfied by every row.
    """
    head = f"shift log {index}: troubleshoot ticket raised, {BULK_LITERAL} at the front end. "
    return head + _BULK_FILLER * 37


@pytest.fixture
async def bulky_pattern_rows(migrated_pool, seeded_prefixes):
    """Seed :data:`BULK_ROW_COUNT` rows of ~2 KB in one batched insert.

    Built at most once per package (see :func:`seeded_prefixes`), so the tests
    that share it pay for the insert once.

    Timestamps are pushed far into the past so that the pattern-only ordering
    (``rank DESC, timestamp DESC``) never lets these rows crowd the small
    glob-probe set out of a ``LIMIT``ed result.

    Returns:
        The number of rows the fixture guarantees are present.
    """
    if BULK_PREFIX in seeded_prefixes:
        return BULK_ROW_COUNT
    seeded_prefixes.add(BULK_PREFIX)

    base = datetime(2001, 1, 1, tzinfo=UTC)
    payload = [
        (
            f"{BULK_PREFIX}{index:05d}",
            "bulk",
            base + timedelta(minutes=index),
            "bulk_fixture",
            bulk_raw_text(index),
        )
        for index in range(BULK_ROW_COUNT)
    ]
    async with migrated_pool.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO enhanced_entries
                (entry_id, source_system, timestamp, author, raw_text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (entry_id) DO NOTHING
            """,
            payload,
        )
    return BULK_ROW_COUNT
