# Google Sheets Channel Database — Design Proposal

Connect the in-context channel finder pipeline to a Google Spreadsheet instead of a local JSON file, enabling collaborative editing and live sync.

## Current Architecture

The in-context pipeline database has three layers:

1. **`BaseDatabase`** (abstract, `core/base_database.py`) — defines `load_database()`, `get_channel()`, `get_all_channels()`
2. **`FlatChannelDatabase`** (`databases/flat.py`) — reads a JSON file, builds a lookup map, provides chunking/formatting
3. **`TemplateChannelDatabase`** (`databases/template.py`) — extends flat with template expansion (device family patterns)

CRUD operations (`ic_add_channel`, `ic_update_channel`, `ic_delete_channel` in `interfaces/channel_finder/database_crud.py`) bypass the database class entirely — they directly read/write the JSON file via `_load_json()` / `_atomic_write()`, then call `_reload_registry()` to refresh the in-memory cache.

The registry (`ChannelFinderICRegistry` in `mcp/in_context/registry.py`) picks `flat` or `template` based on `config.yml` and stores the database instance as a singleton.

## What a Google Sheets Backend Requires

### 1. New Dependency: `gspread`

`gspread` + `google-auth` handle OAuth2/service-account auth and Sheets API calls. The project already has `google-generativeai` so Google auth infra is partially present.

### 2. New Database Class: `GoogleSheetsChannelDatabase(BaseDatabase)`

New file: `src/osprey/services/channel_finder/databases/google_sheets.py`

| Method | Behavior |
|---|---|
| `load_database()` | Fetches all rows from the spreadsheet, populates `self.channels` and `self.channel_map` |
| `get_all_channels()` | Returns cached `self.channels` |
| `get_channel(name)` | Lookup from `self.channel_map` |
| `chunk_database()` / `format_chunk_for_prompt()` | Inherited from flat or reimplemented — operates on cached data |

The constructor takes a **spreadsheet ID** and **worksheet name** instead of `db_path`, plus credentials config. Expected spreadsheet layout: columns for `channel`, `address`, `description`.

**Caching strategy**: Fetch all rows on `load_database()`, cache in memory (same as flat does now). A refresh interval or explicit reload trigger keeps it in sync.

### 3. Rewrite CRUD to Support Sheets Writes

The current CRUD functions in `database_crud.py` directly manipulate JSON files. For Sheets, they need to:

- **Add**: Append a row to the worksheet
- **Delete**: Find the row by channel name, delete it
- **Update**: Find the row, update specific cells

Two approaches:

- **Option A** (recommended): Move CRUD into the database class itself. Each backend knows how to write to its own storage.
- **Option B**: Keep CRUD in `database_crud.py` but add a Sheets code path. Messier, but less refactoring.

Option A is better long-term and produces a cleaner abstraction boundary.

### 4. Config Changes

```yaml
channel_finder:
  pipelines:
    in_context:
      database:
        type: google_sheets
        spreadsheet_id: "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
        worksheet: "Sheet1"            # optional, defaults to first sheet
        credentials: "service_account.json"  # path to service account key file
        cache_ttl: 300                 # seconds before re-fetching (optional)
```

### 5. Registry Update

`ChannelFinderICRegistry.initialize()` needs a new branch:

```python
elif db_type == "google_sheets":
    from osprey.services.channel_finder.databases.google_sheets import (
        GoogleSheetsChannelDatabase,
    )
    self._database = GoogleSheetsChannelDatabase(
        spreadsheet_id=db_config["spreadsheet_id"],
        worksheet=db_config.get("worksheet"),
        credentials_path=db_config.get("credentials"),
        cache_ttl=db_config.get("cache_ttl", 300),
    )
```

### 6. Auth Considerations

Two auth modes to support:

- **Service account** (headless/production): A JSON key file, no browser needed. Best for deployed systems.
- **OAuth2 user flow** (development): Opens browser for consent, caches token locally. Best for local development.

`gspread` supports both natively.

## Effort Estimate

| Component | Effort |
|---|---|
| `google_sheets.py` database class | ~150 lines |
| CRUD refactor (move writes to database class) | ~100 lines changed |
| Registry + config additions | ~20 lines |
| Auth helper (credential loading) | ~40 lines |
| Config template updates | ~10 lines |
| Tests | ~150 lines |
| **Total** | **~500 lines, ~1 day** |

## Key Trade-offs

| Factor | JSON (current) | Google Sheets |
|---|---|---|
| **Latency** | ~1ms (local file) | ~200-500ms per API call |
| **Offline** | Works always | Requires internet |
| **Collaborative editing** | Single-user, file-level | Multi-user, cell-level |
| **Rate limits** | None | 300 req/min/project (Google API) |
| **Concurrent edits** | File locking | Google handles merge; in-memory cache stale until refresh |
| **Setup complexity** | Zero (just a file) | Google Cloud project + credentials |

### Mitigations

- **Latency**: Keep the in-memory cache, only go to network for writes + periodic refresh.
- **Rate limits**: Fine for interactive use; chunked reads during LLM pipeline runs could be a concern for very large sheets.
- **Offline**: Fall back to a local JSON cache when network is unavailable.
- **Cache staleness**: Configurable TTL (`cache_ttl`) with manual refresh via the web UI's refresh button.

## Files to Create/Modify

### New Files

- `src/osprey/services/channel_finder/databases/google_sheets.py` — new database class
- `tests/services/channel_finder/databases/test_google_sheets.py` — unit tests

### Modified Files

- `src/osprey/services/channel_finder/databases/__init__.py` — register new class
- `src/osprey/services/channel_finder/mcp/in_context/registry.py` — add `google_sheets` branch
- `src/osprey/interfaces/channel_finder/database_crud.py` — refactor CRUD to support non-file backends
- `src/osprey/interfaces/channel_finder/database_api.py` — update API endpoints if CRUD interface changes
- `pyproject.toml` — add `gspread` + `google-auth` dependencies (optional extra)
- Config templates — add `google_sheets` database type documentation
