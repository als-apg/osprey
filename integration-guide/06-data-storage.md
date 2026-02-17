# Recipe 6: Data Storage & Workspace Integration

## When You Need This

Your tool produces data (analysis results, plots, measurements) that should be accessible to other OSPREY components — Claude Code, the Artifact Gallery, or other MCP tools.

## Storage Architecture Overview

OSPREY uses two storage tiers:

1. **Workspace files** — JSON indices + data files in `osprey-workspace/` (file-backed, no database required)
2. **PostgreSQL** — structured data with queries, full-text search, embeddings (when you need it)

Most tools should start with workspace files and only add PostgreSQL if they need complex queries or cross-tool search.

## Tier 1: Workspace File Storage

### The `save_to_workspace()` Pattern

Every MCP tool that produces output should save it to the workspace:

```python
from osprey.mcp_server.common import save_to_workspace

filepath = save_to_workspace(
    category="txt_analysis",              # → osprey-workspace/txt_analysis/
    data={                                 # JSON-serializable dict
        "bpm_names": [...],
        "tunes": {"h": 0.234, "v": 0.317},
        "spectra": [...],
    },
    description="Turn-by-turn FFT analysis of horizontal plane",
    tool_name="txt_analyze",              # Prefix for filename
)
# Result: osprey-workspace/txt_analysis/txt_analyze_20260216_143022.json
```

The workspace directory structure:

```
osprey-workspace/
├── artifacts/           # Plots, tables, HTML (managed by ArtifactStore)
├── memory/              # Notes and pins (managed by MemoryStore)
├── data/                # Tool output data (managed by DataContext)
├── search_results/      # ARIEL search output
├── drafts/              # Entry drafts (TTL-based, 1 hour)
├── txt_analysis/        # YOUR CATEGORY (create as needed)
└── screenshots/         # Screen captures
```

### Singleton Store Pattern

If your tool needs more than simple file dumps — an index, listing, focused items — follow the **Singleton Store** pattern used by ArtifactStore, MemoryStore, and DataContext:

```python
"""Store for {your domain} data."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MyEntry:
    id: str
    title: str
    description: str
    data_file: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MyStore:
    """File-backed store with JSON index."""

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root / "my_data"
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "index.json"
        self._entries: list[MyEntry] = []
        self._listeners: list[callable] = []
        self._load_index()

    # --- Index Management ---

    def _load_index(self) -> None:
        if self._index_path.exists():
            raw = json.loads(self._index_path.read_text())
            self._entries = [MyEntry(**e) for e in raw]

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps([asdict(e) for e in self._entries], indent=2, default=str)
        )

    # --- CRUD ---

    def save(self, title: str, data: dict, description: str = "") -> MyEntry:
        entry_id = f"{len(self._entries):03d}"
        data_file = self._root / f"{entry_id}_{title.replace(' ', '_')}.json"
        data_file.write_text(json.dumps(data, indent=2, default=str))

        entry = MyEntry(
            id=entry_id,
            title=title,
            description=description,
            data_file=str(data_file),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._entries.append(entry)
        self._save_index()
        self._notify("new_entry", entry)
        return entry

    def list_entries(self) -> list[MyEntry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> MyEntry | None:
        return next((e for e in self._entries if e.id == entry_id), None)

    # --- Event Listeners (for SSE broadcasting) ---

    def add_listener(self, callback: callable) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        self._listeners.remove(callback)

    def _notify(self, event_type: str, entry: MyEntry) -> None:
        for listener in self._listeners:
            try:
                listener(event_type, asdict(entry))
            except Exception:
                logger.exception("Listener notification failed")
```

**Key rules:**
- Index is a flat JSON file (fast reads, atomic writes)
- Data files are separate (index stays small)
- Listeners enable SSE broadcasting to web interfaces
- `_load_index()` in `__init__` for crash recovery
- IDs are sequential integers (simple, sortable)

### Data Context Integration

If your tool's output should appear in OSPREY's data context (the unified index of all tool outputs), use the `DataContext` class:

```python
from osprey.mcp_server.workspace.data_context import DataContext

ctx = DataContext(workspace_root=Path("./osprey-workspace"))
entry = ctx.add_entry(
    tool="txt_analyze",
    data={"tunes": {...}, "spectra": [...]},
    description="TxT FFT analysis: H-tune=0.234, V-tune=0.317",
    summary={"h_tune": 0.234, "v_tune": 0.317, "num_bpms": 96},
    data_type="spectral_analysis",
)
```

The `summary` dict is what Claude sees inline (compact stats). The full `data` dict goes to a separate file. This keeps Claude's context window efficient.

## Tier 2: PostgreSQL Storage

### When to Use PostgreSQL

Use PostgreSQL when you need:
- Full-text search across entries
- Complex queries (time ranges + filters + sorting)
- Vector embeddings for semantic search
- Multi-process concurrent access
- Transactional guarantees

### Repository Pattern

Follow ARIEL's repository pattern — a class that wraps all SQL operations:

```python
"""Database repository for {your domain}."""

import logging
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


class MyRepository:
    """Async PostgreSQL operations for {your domain}."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def get_entry(self, entry_id: str) -> dict | None:
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT * FROM my_entries WHERE entry_id = %s",
                        (entry_id,),
                    )
                    return await cur.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get entry {entry_id}: {e}",
                query=f"SELECT entry_id={entry_id}",
            ) from e

    async def upsert_entry(self, entry: dict) -> None:
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO my_entries (entry_id, data, created_at)
                        VALUES (%(entry_id)s, %(data)s, %(created_at)s)
                        ON CONFLICT (entry_id) DO UPDATE SET
                            data = EXCLUDED.data,
                            updated_at = NOW()
                        """,
                        entry,
                    )
        except Exception as e:
            raise DatabaseQueryError(f"Failed to upsert: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict]:
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        """
                        SELECT *, ts_rank(search_vector, plainto_tsquery(%s)) AS score
                        FROM my_entries
                        WHERE search_vector @@ plainto_tsquery(%s)
                        ORDER BY score DESC
                        LIMIT %s OFFSET %s
                        """,
                        (query, query, limit, offset),
                    )
                    return await cur.fetchall()
        except Exception as e:
            raise DatabaseQueryError(f"Search failed: {e}") from e

    async def count_entries(self) -> int:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM my_entries")
                row = await cur.fetchone()
                return row[0] if row else 0
```

### Connection Pool

```python
from psycopg_pool import AsyncConnectionPool

async def create_connection_pool(uri: str) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        conninfo=uri,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
        open=False,
    )
    await pool.open()
    return pool
```

### Migration System

For schema management, follow ARIEL's migration pattern:

```python
"""Migration for {your domain} core schema."""

from .base import BaseMigration


class CoreMigration(BaseMigration):
    @property
    def name(self) -> str:
        return "my_tool_core_schema"

    @property
    def depends_on(self) -> list[str]:
        return []  # No dependencies for core

    async def up(self, conn) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS my_entries (
                entry_id TEXT PRIMARY KEY,
                data JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_my_entries_created
                ON my_entries (created_at DESC);
        """)

    async def is_applied(self, conn) -> bool:
        result = await conn.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'my_entries')"
        )
        row = await result.fetchone()
        return row[0] if row else False
```

## Concrete Reference

- `src/osprey/mcp_server/workspace/` — ArtifactStore, MemoryStore, DataContext (all file-backed)
- `src/osprey/services/ariel_search/database/repository.py` — Full PostgreSQL repository
- `src/osprey/services/ariel_search/database/connection.py` — AsyncConnectionPool setup
- `src/osprey/services/ariel_search/database/migrate.py` — Migration runner with topological sort

## Checklist

- [ ] Workspace output saved via `save_to_workspace()` for simple tool output
- [ ] Singleton store with JSON index if you need listing/retrieval
- [ ] Listener pattern if web interface needs real-time updates
- [ ] PostgreSQL repository if you need complex queries or full-text search
- [ ] Migration class if using PostgreSQL
- [ ] Connection pool with `min_size=1, max_size=10`
- [ ] All SQL exceptions wrapped in `DatabaseQueryError` with query description
- [ ] `dict_row` factory for psycopg cursors
