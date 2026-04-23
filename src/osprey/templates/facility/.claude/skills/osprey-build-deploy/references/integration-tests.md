# Integration Tests Reference

The `integration_tests` MCP server provides a category-driven health check suite for the deployed facility assistant. It runs as a containerized HTTP service on port `${config.ports.integration_tests}` (Docker DNS name: `integration-tests`) and is **always available** — it is part of the core deploy and not gated on any optional module.

The check categories the server actually runs **are** module-gated, however. A category is exposed only if its underlying module is enabled in `facility-config.yml`. The runner discovers which categories to register at startup by inspecting the loaded config.

---

## Architecture

A single set of check functions lives under `mcp_servers/integration_tests/checks/<category>.py`. They are invoked through four interfaces, all calling the same code:

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  pytest          │  │  MCP tool        │  │  /checks HTTP    │  │  /dashboard HTML │
│  (CI / local)    │  │  health_check()  │  │  GET endpoint    │  │  status panel    │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │                     │
         │  (imports directly) │  (via runner.py)     │  (via runner.py)    │  (JS fetches /checks)
         │                     │                      │                     │
         └─────────────────────┼──────────────────────┼─────────────────────┘
                               │                      │
                ┌──────────────┴──────────┬───────────┴──────────┬───────────────┐
                │                         │                      │               │
            (always)                 (always)              (module-gated)   (CS-gated)
              config                 mcp_servers              olog            epics_*
              services               (configurable)           ariel
              physbase                                        wiki_search
              ...                                             accelpapers
                                                              matlab
```

| Interface | Purpose | Used by |
|-----------|---------|---------|
| **HTTP `/checks`** | One-shot REST poll, returns full JSON report | `verify.sh`, dashboards, ad-hoc operator |
| **MCP tools** | Assistant calls health checks during a conversation | The product Claude inside the web terminal |
| **pytest** | Local CI runs and lifecycle.validate phase | CI pipeline, developer pre-flight |
| **HTTP `/dashboard`** | Browser-rendered status panel with auto-refresh | Operators, web-terminal panel embed |

All four interfaces share the same check functions — there is no logic duplication.

### Category modules

Each category is a Python module under `mcp_servers/integration_tests/checks/<category>.py`. A module exposes:

- One or more `check_*() -> CheckResult` functions
- A `run_<category>_checks() -> list[CheckResult]` aggregator
- A registration entry in `runner.py`'s `CATEGORY_FUNCS` dict

Categories run **concurrently** (per-category `asyncio.to_thread`), with a per-check timeout (default 5 s) and a 60-second ceiling for the full suite. Per-category error isolation: a crash in one category never prevents the others from completing.

---

## Check Categories

The set of categories that actually register at runtime is determined by `facility-config.yml`. The table below lists every category the runner *can* expose, plus the gate condition.

### Always available (control-system-agnostic core)

| Category | Gate | What it does |
|----------|------|--------------|
| `config` | always | Verifies every variable named in the profile's `env.required` list is set and non-empty in the loaded environment |
| `mcp_servers` | always | (a) Each MCP server registered in `config.yml`'s `claude_code.servers`; (b) each HTTP server replies to `/health`; (c) `EXPECTED_TOOLS` consistency — see below |
| `services` | always | Generic connectivity probes (proxy reachable, LLM provider endpoint, archiver REST URL — driven by config, not hardcoded) |

### Module-gated

A category appears only if the corresponding module is enabled in `facility-config.yml`.

| Category | Required module | What it does |
|----------|-----------------|--------------|
| `physbase` | `modules.shared_disk` | Verify host bind-mount accessible, expected sub-tree exists, optional read-write round-trip |
| `wiki_search` | `modules.wiki_search` | Authenticated GET against the wiki API root; SKIP if token env var unset |
| `accelpapers` | `modules.custom_mcp_servers` includes one named `accelpapers` | Embedding endpoint reachable, search index reachable, sample query smoke-test |
| `matlab` | `modules.custom_mcp_servers` includes one named `matlab` | DB file exists / DB nonempty (dual-mode: subprocess or HTTP via `/health`) |
| `channel_finder` | `modules.custom_mcp_servers` includes one named `channel_finder` (or platform default) | Channel Finder service reachable; PV metadata lookup smoke-test |
| `olog` | `modules.olog` | RPC endpoint reachable, credentials configured, optional write-side post + readback (gated on `write_test_enabled`) |
| `ariel` | `modules.ariel` | Postgres reachable, schema present, entry count > 0, known-entry author non-empty |

### Control-system-gated

Available only when `control_system.type == "epics"`.

| Category | Additional gate | What it does |
|----------|-----------------|--------------|
| `epics_read` | always (when EPICS) | Validate `EPICS_CA_ADDR_LIST`; read a configured set of PVs (defined in profile, not hardcoded); read one PV through OSPREY's `ConnectorFactory` |
| `epics_write` | `modules.test_ioc.enabled` | Test IOC alive; numeric round-trip; string round-trip; heartbeat increment; DB-hash match against `${config.modules.test_ioc.db_path}` |

For non-EPICS control systems (DOOCS, TANGO, mock, custom), neither category registers and the runner does not list them as available.

---

## The `mcp_servers` Category — `EXPECTED_TOOLS` Consistency

The single most useful check in the suite. It catches the failure mode where a server is healthy (`/health` returns OK) but Claude can't actually use it — because tool decorators failed to register, the profile permission list is wrong, or the server's `tools/list` MCP handshake returns the wrong set.

How it works:

1. `EXPECTED_TOOLS` (a constant in `mcp_servers/integration_tests/checks/mcp_servers.py`) maps each MCP server to the list of tool names its profile YAML declares as permitted.
2. For every HTTP MCP server, the check performs a real MCP handshake: `initialize` → `notifications/initialized` → `tools/list`.
3. The returned tool name set is compared to `EXPECTED_TOOLS[<server>]`.
4. Mismatches surface as `ERROR`:

| Result | Meaning |
|--------|---------|
| `tools/list returned 0 tools` | Server up, all tool modules failed silent import |
| `N expected tools missing` | Some tool modules failed import; partial registration |
| `tool check failed` | Handshake itself failed (transport / FastMCP version) |

When you add or rename a tool, three lists must agree (see `SKILL.md` § Consistency Rule):
- `@mcp.tool()` decorators in the source
- `permissions.allow` in the profile YAML
- `EXPECTED_TOOLS` in this category's source

---

## How to Run

### HTTP `/checks` endpoint (most common)

```bash
# Full report (all registered categories)
curl http://${config.deploy.host}:${config.ports.integration_tests}/checks

# Single category
curl 'http://${config.deploy.host}:${config.ports.integration_tests}/checks?category=services'

# Multiple categories
curl 'http://${config.deploy.host}:${config.ports.integration_tests}/checks?categories=services,mcp_servers'
```

Response shape (`CheckReport.to_dict()`):

```json
{
  "summary": "8/10 checks passed (2 warnings)",
  "ok": 8, "warnings": 2, "errors": 0, "skipped": 0, "total": 10,
  "results": [
    {
      "name": "services.proxy",
      "category": "services",
      "status": "ok",
      "message": "...",
      "latency_ms": 32,
      "value": "200 OK"
    }
  ]
}
```

`value`, `latency_ms`, and `details` are present only when set.

### MCP tools (assistant interface)

When the `integration_tests` MCP server is configured in the running assistant, Claude has access to:

```python
health_check(categories=["services", "mcp_servers"])  # JSON string
health_check()                                          # all categories
```

Returns the same payload as the HTTP endpoint. Claude uses this when the user asks "is everything healthy?" or after a tool failure to localize the problem.

### pytest (CI / lifecycle.validate)

```bash
# Inside the built project directory:
pytest tests/integration/ -v --tb=short

# Specific category by marker:
pytest tests/integration/ -v -m services
pytest tests/integration/ -v -m "olog or services"

# Exclude write-side checks (default for the validate phase):
pytest tests/integration/ -v -m 'not epics_write'

# JUnit XML for CI:
pytest tests/integration/ -v --junitxml=check_results.xml
```

Exit codes: `0` = all passed, `1` = some failed, `2` = error (e.g. import failure).

The pytest wrappers are thin — each test calls a public `check_*()` function and asserts `status != ERROR`. `WARNING`, `OK`, and `SKIP` all pass; only `ERROR` fails the test.

OSPREY automatically prepends `_mcp_servers` to `PYTHONPATH` for lifecycle commands, so `import integration_tests` resolves without manual `PYTHONPATH=` prefixes.

The validate phase typically excludes write-side checks because the test IOC may not be running at build time:

```yaml
lifecycle:
  validate:
    - name: "integration checks"
      run: "pytest tests/integration/ -v --tb=short --junitxml=check_results.xml -m 'not epics_write'"
      timeout: 180
      stream: true
```

### HTTP `/dashboard` — visual status panel

```bash
# Live (auto-refresh every 60 s):
curl http://${config.deploy.host}:${config.ports.integration_tests}/dashboard

# Demo / sample data (no /checks fetch):
curl 'http://${config.deploy.host}:${config.ports.integration_tests}/dashboard?demo=1'
```

Self-contained HTML/CSS/JS (one file, served as `HTMLResponse`). Features:

- Category cards with LED indicators and expandable error detail
- Auto-refresh with a countdown timer and manual refresh button
- Status filter chips (OK / WARN / ERR / SKIP)
- Dark theme, iframe-friendly, `postMessage` listener for theme sync
- Designed to embed in the web-terminal panel system

If `modules.web_terminals` is enabled, the dashboard URL is wired into the web-terminal panel config so each user can see live health from inside their terminal:

| Profile | Panel URL |
|---------|-----------|
| Prod (intra-compose) | `http://integration-tests:${config.ports.integration_tests}/dashboard` |
| Client (host-routable) | `http://${config.deploy.fqdn}:${config.ports.integration_tests}/dashboard` |

### `verify.sh` — terminal formatter

Pretty-prints `/checks` results with colors and icons:

```bash
./scripts/verify.sh                      # all
./scripts/verify.sh services             # filter
./scripts/verify.sh epics_read olog      # multiple
```

Always exits 0 — advisory only. `scripts/deploy.sh` runs it automatically after containers come up.

---

## Status Meanings

| Status | Meaning | pytest behavior |
|--------|---------|-----------------|
| `OK` | Check passed | passes |
| `WARNING` | Degraded but not broken (empty list, default value, expected zero) | passes |
| `ERROR` | Check failed (service unreachable, assertion failed, exception) | fails |
| `SKIP` | Prerequisite missing (module disabled, credentials unset, gating check failed) | passes |

Common `WARNING` cases:
- A list-shaped resource is reachable but empty (e.g., archiver returned 0 points)
- An optional config value is unset and the check defaults to a sentinel
- A control-system-specific value is in a non-error sentinel state (beam off, shutter closed)

Common `SKIP` cases:
- `OLOG_PASSWORD` env var unset → `olog.post_entry` skips
- Test IOC down → `epics_write.numeric_roundtrip` skips (`epics_write.ioc_alive` is the gate)
- Wiki token unset → `wiki_search.api_reachable` skips

`SKIP` is **not** an error — it's the runner saying "this check has no opinion because its prerequisite isn't met." Investigate only if you expected the prerequisite to be present.

---

## Adding a New Check

1. **Pick the category** (or create a new module under `checks/<category>.py`).
2. **Write the check function:**

```python
import time
from integration_tests.models import CheckResult, Status

def check_my_new_thing() -> CheckResult:
    t0 = time.perf_counter()
    try:
        # ... do the probe ...
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return CheckResult(
            name="category.my_new_thing",
            category="category",
            status=Status.OK,
            message="thing reachable",
            latency_ms=latency_ms,
            value="<observed value>",
        )
    except Exception as e:
        return CheckResult(
            name="category.my_new_thing",
            category="category",
            status=Status.ERROR,
            message=f"{type(e).__name__}: {e}",
        )
```

Use `try/except` to isolate failures — never let a check crash. Return `Status.ERROR` with a useful message instead.

3. **Register it** in the module's `run_<category>_checks()` aggregator so the runner picks it up.
4. **Add a pytest wrapper** in `tests/integration/test_<category>.py`:

```python
def test_my_new_thing():
    result = check_my_new_thing()
    assert result.status != Status.ERROR, result.message
```

5. **If the check exists in a new category:**
   - Add a `run_<category>_checks()` entry-point function.
   - Add a lazy lambda in `runner.py`'s `CATEGORY_FUNCS` dict, gated on the relevant module.
   - Register the marker in `pytest.ini`.
   - Create `tests/integration/test_<category>.py` with `pytestmark = pytest.mark.<category>`.

### Concurrency notes

Each category runs in its own thread via `asyncio.to_thread`. Within a category, checks run sequentially (one Python file, one thread). For categories with many slow probes, prefer multiple categories over many sequential checks within one category.

The default per-check timeout is 5 s (configurable per category). The full suite is bounded at 60 s — anything longer is treated as a runner failure rather than a check result.

---

## Config-Driven URLs

Integration tests must **never hardcode infrastructure URLs**. The runner reads service URLs from the rendered `config.yml` via a `site_config.py` helper:

```python
from integration_tests.site_config import config

archiver_url = config.archiver_url        # from control_system.archiver_url
olog_url = config.olog.api_url            # from modules.olog.api_url
wiki_url = config.wiki.base_url           # from modules.wiki_search.base_url
```

This makes every category portable across facilities — when the facility renames a hostname or moves a service to a different port, only `facility-config.yml` (and the regenerated `config.yml`) need updating.

If a check needs a value that isn't yet in `facility-config.yml`, **extend the schema and the interview** rather than hardcoding. See `references/facility-config-schema.md`.

---

## Timeouts

Default timeouts (configurable per category in the runner):

| Scope | Default | Notes |
|-------|---------|-------|
| Per-check (HTTP probe) | 2 s | Most network probes |
| Per-check (DB / heavier) | 5 s | Postgres, large-result HTTP |
| Per-category total | 10 s | All checks within one category |
| Full suite | 60 s | All categories together |

If your check legitimately needs more time (e.g., a slow embedding model warming up), pass `timeout=` to the underlying client and bump the per-category timeout in the runner. Don't widen the suite-level timeout — that hides real outages.

---

## Common Failure Patterns

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| Every external HTTP check ERROR | Proxy misconfigured or unreachable | Check `${env.HTTP_PROXY}`, `${env.HTTPS_PROXY}`, `${env.NO_PROXY}` against `network.*` in config |
| One service category ERROR, others OK | That single service is down (or routing changed) | `${config.runtime.engine} ps` for the service container; `${config.runtime.engine} logs <name>` |
| All `mcp_servers.<name>.tools` ERROR with "0 tools" | Tool modules failing import inside the MCP server container | `${config.runtime.engine} logs ${config.facility.prefix}-mcp-<name>` — look for ImportError on startup |
| `mcp_servers.<name>.tools` ERROR "N expected tools missing" | Source ↔ permissions ↔ EXPECTED_TOOLS drift | Run the consistency check in `SKILL.md` § MCP Tool Sync Verification |
| `services.proxy` OK but every external probe SKIP/ERROR | `NO_PROXY` includes the wrong wildcard | Verify `network.no_proxy` covers internal services, not external ones |
| `config.required_env_vars` ERROR | Profile declares a required var that isn't in `.env` | Update `.env` from `.env.template` |
| `physbase.accessible` ERROR | Bind-mount missing on the deploy server | Check `modules.shared_disk.host_path` exists on the host |
| `epics_read.*` all ERROR | EPICS connectivity (broadcast / network) | Verify `EPICS_CA_ADDR_LIST`; check facility CA gateway |
| `ariel.db_reachable` ERROR | Postgres container down or DSN wrong | `${config.runtime.engine} logs ${config.facility.prefix}-postgres`; verify `modules.ariel.dsn` |

For deeper diagnosis, read `references/post-deploy-diagnosis.md`.

---

## Test IOC and Integration Tests

If `modules.test_ioc.enabled` is true, the test IOC runs as a host process on exotic CAS ports (`${config.modules.test_ioc.cas_server_port}` / `${config.modules.test_ioc.cas_beacon_port}`) — see `references/modules/test-ioc-safety.md`.

**The `epics_read` category does NOT see the test IOC.** It uses the production `EPICS_CA_ADDR_LIST` and the default CA server port (5064). The test IOC, by design, is on a different port and the production CA discovery never finds it. This isolation is the whole point.

To validate the test IOC, the **`epics_write`** category is dedicated to it — those checks set `EPICS_CA_SERVER_PORT=${config.modules.test_ioc.cas_server_port}` before connecting. They do not consult production CA at all.

The DB-hash check (`epics_write.db_hash`) compares the `<prefix>DB_HASH` PV against the MD5 of `${config.modules.test_ioc.db_path}` on disk. It catches the case where the IOC was started with a stale DB version.

If you want to validate the test IOC from outside the integration_tests suite, write a one-off script that overrides `EPICS_CA_SERVER_PORT` — never reuse production CA settings.

---

## Lifecycle Hook

The validate phase in the prod profile runs the suite (excluding `epics_write`) via pytest:

```yaml
lifecycle:
  validate:
    - name: "integration checks"
      run: "pytest tests/integration/ -v --tb=short --junitxml=check_results.xml -m 'not epics_write'"
      timeout: 180
      stream: true
```

`epics_write` is excluded from the validate phase because the test IOC isn't guaranteed to be running at build time. To include it, drop the `-m 'not epics_write'` filter.

---

## Summary of Endpoints

| Endpoint | Path | Purpose |
|----------|------|---------|
| Run all checks | `GET http://${config.deploy.host}:${config.ports.integration_tests}/checks` | JSON report |
| Run filtered checks | `GET http://${config.deploy.host}:${config.ports.integration_tests}/checks?category=<name>` | Single category |
| Visual dashboard | `GET http://${config.deploy.host}:${config.ports.integration_tests}/dashboard` | HTML status panel |
| Health (server itself) | `GET http://${config.deploy.host}:${config.ports.integration_tests}/health` | Lightweight liveness |
| MCP transport | `POST http://${config.deploy.host}:${config.ports.integration_tests}/mcp` | Streamable-HTTP MCP |

All HTTP endpoints honor the same `category=` / `categories=` query params.
