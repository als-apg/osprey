# Module: event_dispatcher

The event dispatcher converts external events (HTTP webhooks and, optionally, EPICS Channel Access PV transitions) into headless Claude agent runs. Enable it when the facility wants automated agent activations triggered by something other than a human typing in a chat window — CI smoke tests, control-system thresholds, scheduled jobs, dashboard "manual fire" buttons, or arbitrary callers.

The dispatcher itself is a small FastAPI/FastMCP service that owns the trigger registry and a bounded **dispatch pool**. Every incoming event is validated, matched to a trigger definition in `triggers.yml`, allocated to one of N **sidecar** containers, and tracked through to completion. Each sidecar wraps `claude_agent_sdk.query()` and is the thing that actually talks to the LLM and to MCP tools.

**Enabled when**: `modules.event_dispatcher.enabled: true` in `facility-config.yml`.
**EPICS-CA triggers also require**: `control_system.type == "epics"` AND `modules.event_dispatcher.epics_ca.enabled: true`.

---

## Architecture

```
   ┌──── webhook POST ──────────────┐                           caller polls / streams
   │  (CI, dashboard, cron, ...)    │                                    │
   │                                ▼                                    ▼
   │                     ┌─────────────────────┐    POST /dispatch    ┌─────────────────────┐
   │                     │  event-dispatcher   │ ───────────────────▶ │  dispatch-sidecar-1 │
   │                     │  :${config.modules. │                       │  :${config.modules.  │
   │  ┌── EPICS CA ─────▶│   event_dispatcher. │ ◀── {run_id} ────── │   event_dispatcher.  │
   │  │  monitor (PVs)   │   port}             │                       │   sidecar_port_base} │
   │  │  (only if epics_ │                     │ ───────────────────▶ │  dispatch-sidecar-2 │
   │  │   ca.enabled)    │   - trigger registry│                       │  :…+1               │
   │  │                  │   - dispatch_pool   │                       │  …                  │
   │  │                  │   - /dashboard      │                       │  dispatch-sidecar-N │
   │  └──── PV updates ──┤                     │                       │  :…+N-1             │
   │                     └─────────────────────┘                       │                     │
   │                              │                                    │  sdk_runner →       │
   │                              ▼                                    │  claude_agent_sdk    │
   │                        triggers.yml                               │  → MCP tools        │
   │                                                                   └─────────────────────┘
   │                                                                            │
   └────────────────────────────────────────────────────────────────────────────┘
                                                                                ▼
                                                                          SSE /stream
                                                                          GET /dispatch/{run_id}
```

**Two layers, two IDs:**

| ID | Returned by | Scope | Used to |
|----|-------------|-------|---------|
| `dispatch_id` | dispatcher's webhook endpoint (`POST /webhook/<trigger_name>`) | pool-level (one per accepted event) | poll `GET /dispatch/{dispatch_id}` on the **dispatcher** to learn which sidecar got the work and the `run_id` |
| `run_id` | sidecar (returned in the dispatcher's poll response once status is `completed`) | execution-level (one per agent run) | poll `GET /dispatch/{run_id}` or stream `GET /dispatch/{run_id}/stream` directly on the **sidecar** |

The two-layer design lets the dispatcher hand off a run, free the pool slot, and let the caller deal directly with the sidecar that's holding the agent.

---

## Configuration

Full schema in `references/facility-config-schema.md` § `modules.event_dispatcher`. The most-used fields:

```yaml
modules:
  event_dispatcher:
    enabled: true
    port: 8010                                # ${config.modules.event_dispatcher.port}
    token_env_var: "EVENT_DISPATCHER_TOKEN"   # bearer token for /webhook/<name>
    sidecar_count: 5                          # how many concurrent runs allowed
    sidecar_port_base: 9190                   # sidecars on base, base+1, ..., base+N-1
    sidecar_token_env_var: "DISPATCH_SIDECAR_TOKEN"
    triggers_file: "triggers.yml"
    epics_ca:                                  # optional, EPICS facilities only
      enabled: true
      ca_addr_list: "${config.control_system.ca_addr_list}"
```

**Sidecar count sizing**: if `modules.web_terminals` is also enabled, set `sidecar_count` equal to `len(modules.web_terminals.users)` so every web terminal can hold a long-running dispatch open without blocking other terminals. If web terminals are not enabled, the default of `5` is enough for most facilities — bump it only if you observe queueing in `GET /health`.

**Token rotation**: `EVENT_DISPATCHER_TOKEN` is what external callers present; `DISPATCH_SIDECAR_TOKEN` is the internal token sidecars require to accept dispatch requests from the dispatcher. Rotate both via `.env` and restart the dispatcher + every sidecar.

---

## What scaffolding adds when this module is enabled

### compose

- One service `event-dispatcher` (container name `${config.facility.prefix}-event-dispatcher`).
  - If `epics_ca.enabled: true`: `network_mode: host` because EPICS CA discovery uses subnet broadcasts that don't traverse bridge NAT reliably. Otherwise the service joins the bridge network like every other MCP server.
  - Mounts `./${config.modules.event_dispatcher.triggers_file}` read-only into the container at `/app/triggers.yml`.
  - Exposes `${config.modules.event_dispatcher.port}` (host-bound when on bridge network; bound directly when host-networked).
  - Loads `.env.production` for `EPICS_CA_ADDR_LIST`, `EVENT_DISPATCHER_TOKEN`, `DISPATCH_SIDECAR_TOKEN`, and any tokens trigger prompts need.
- N services `dispatch-sidecar-1 … dispatch-sidecar-N` (container names `${config.facility.prefix}-dispatch-sidecar-${index}`).
  - Each binds `${config.modules.event_dispatcher.sidecar_port_base} + (index - 1)`.
  - Each receives `DISPATCH_SIDECAR_TOKEN` and the LLM provider key (`${config.llm.api_key_env_var}`).

```
# FOR i in 0..${config.modules.event_dispatcher.sidecar_count - 1}
  dispatch-sidecar-${i}:
    image: ${config.registry.url}/dispatch-sidecar:latest
    container_name: ${config.facility.prefix}-dispatch-sidecar-${i}
    # IF modules.event_dispatcher.epics_ca.enabled
    network_mode: host                         # matches dispatcher; localhost traffic only
    # ELSE
    networks: [ ${config.facility.prefix}-net ]
    ports:
      - "0.0.0.0:${config.modules.event_dispatcher.sidecar_port_base + i}:9100"
    # END IF
    env_file: .env.production
    environment:
      - SIDECAR_PORT=${host-or-9100-depending-on-network-mode}
      - ${config.modules.event_dispatcher.sidecar_token_env_var}=${env.${...}}
      - ${config.llm.api_key_env_var}=${env.${config.llm.api_key_env_var}}
# END FOR
```

**Network coupling:** when `epics_ca.enabled`, BOTH the dispatcher and the sidecars run with `network_mode: host`. They exchange traffic over localhost on the deploy server. Bridge-networked sidecars can't reach a host-networked dispatcher via the compose DNS name (`event-dispatcher:${port}` doesn't resolve across that boundary), which is why the template couples the two.

### .gitlab-ci.yml

- One `build-event-dispatcher` job in the `docker-build` stage (Dockerfile at `docker/Dockerfile.event-dispatcher`).
- One `build-dispatch-sidecar` job in the same stage (Dockerfile at `docker/Dockerfile.dispatch-sidecar`). This is a SEPARATE image from the dispatcher — kept distinct so dispatcher rebuilds don't invalidate every sidecar's image layer.
- Both images appear in the `release` job's `needs:` and in the IMAGES retag loop. The release job is the only place `:latest` is published.

### scripts/deploy.sh

- No special pre-deploy steps for the dispatcher itself. The compose pull + up handles everything.
- If `epics_ca.enabled: true`: deploy.sh exports `EPICS_CA_ADDR_LIST` from `.env.production` before `compose up` so podman/docker passes it through to the host-networked container.

### scripts/verify.sh

- `dispatcher_health`: `curl -fsS http://localhost:${config.modules.event_dispatcher.port}/health` — surfaces pool size, queued count, recent errors.
- `sidecar_health`: loops over each sidecar port and curls `/health` with the sidecar token.

### .env.template

```
# Event dispatcher (modules.event_dispatcher)
${config.modules.event_dispatcher.token_env_var}=        # bearer token for external callers; pick a long random string
${config.modules.event_dispatcher.sidecar_token_env_var}=  # internal token; dispatcher → sidecar auth
```

If `epics_ca.enabled`, `EPICS_CA_ADDR_LIST` is already required by the control-system core — no new entry needed.

### Other files

- `triggers.yml` is created at the repo root if missing, with a single commented example. The user fills it in.

---

## Trigger schema (`triggers.yml`)

```yaml
triggers:
  - name: <unique-name>             # used in URL: /webhook/<name>
    source: webhook | epics_ca
    source_config:                  # source-specific config (see below)
      ...
    on_error:
      action: drop | retry | alert
      max_retries: 0                 # only meaningful for action=retry
      backoff_sec: 0.0
    action:
      prompt: |
        <multi-line prompt for the headless agent>
      allowed_tools:
        - <tool-id>                  # MCP tool ids the sidecar may call
        - ...
```

### `source: webhook`

`source_config` is empty (the trigger is fired by HTTP). The endpoint is:

```
POST http://${config.deploy.host}:${config.modules.event_dispatcher.port}/webhook/<trigger_name>
Authorization: Bearer ${env.EVENT_DISPATCHER_TOKEN}
Content-Type: application/json
{ ...optional JSON body, exposed to the prompt as $WEBHOOK_BODY... }
```

### `source: epics_ca` (only if `modules.event_dispatcher.epics_ca.enabled`)

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `pv` | string | required | PV name to monitor via pyepics |
| `threshold` | float | 0.0 | Edge-detection threshold value |
| `edge` | `rising` \| `falling` \| `both` | `rising` | Which crossing direction fires the trigger |
| `cool_down_sec` | float | 60.0 | Minimum seconds between fires (debounce) |

Watcher behavior:
- The first callback after the PV connects is suppressed (prevents firing on dispatcher restart when the PV is already past the threshold).
- Edges are computed across consecutive callbacks; no debounce beyond `cool_down_sec`.
- Watchers are started inside the dispatcher's lifespan (an `@asynccontextmanager` passed to `FastMCP(..., lifespan=...)`) so `asyncio.get_running_loop()` captures the serving loop. Don't move startup into `create_server()` — that runs synchronously before uvicorn and produces a "no running event loop" error.

---

## Operating the module

### Fire a webhook trigger manually (for testing)

```bash
curl -X POST \
  http://${config.deploy.host}:${config.modules.event_dispatcher.port}/webhook/<trigger_name> \
  -H "Authorization: Bearer ${env.EVENT_DISPATCHER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{ "dispatch_id": "8c7…", "status": "queued" }
```

### Poll dispatcher for status

```bash
curl http://${config.deploy.host}:${config.modules.event_dispatcher.port}/dispatch/<dispatch_id>
```

Response progresses through `queued` → `running` → `completed`. Once `completed`, the response includes `result.run_id` — use that to talk to the sidecar.

### Poll sidecar for the agent result

```bash
curl http://${config.deploy.host}:<sidecar_port>/dispatch/<run_id> \
  -H "Authorization: Bearer ${env.DISPATCH_SIDECAR_TOKEN}"
```

(`<sidecar_port>` is one of `${config.modules.event_dispatcher.sidecar_port_base} … +N-1`. The dispatcher's poll response tells you which one.)

### Stream the agent's events live

```bash
curl -N http://${config.deploy.host}:<sidecar_port>/dispatch/<run_id>/stream \
  -H "Authorization: Bearer ${env.DISPATCH_SIDECAR_TOKEN}"
```

The SSE stream emits one event per line, types include:
- `text` — assistant text deltas
- `tool_start` — agent invoked an MCP tool
- `tool_result` — tool returned
- `result` — final synthesized answer
- `done` / `error` — terminal event

### Dashboard

`GET http://${config.deploy.host}:${config.modules.event_dispatcher.port}/dashboard`

Renders a self-contained HTML page with:
- Trigger registry (read from `triggers.yml`)
- A "fire" button per trigger (POSTs to `/retry` with the trigger name; manual fires are routed to `/retry` and not `/webhook` so they bypass any production-bound rate-limiting)
- Recent activity feed (last N dispatches, status-coded)

The dashboard is intended to be embedded in the OSPREY web terminal panel system via iframe. It's safe to expose as long as the dispatcher itself is bearer-token-protected on `/webhook`.

### Add a new trigger

1. Edit `triggers.yml`, add a new entry under `triggers:`.
2. Restart the dispatcher so it re-reads the file:
   ```bash
   ssh ${config.deploy.host} "cd ${config.deploy.project_path} && \
     ${config.runtime.compose_command} restart event-dispatcher"
   ```
3. Verify it loaded:
   ```bash
   ssh ${config.deploy.host} "${config.runtime.engine} logs ${config.facility.prefix}-event-dispatcher 2>&1 | tail -50"
   ```
   Look for `Loaded N triggers` and (for EPICS triggers) `EPICS watcher started: <pv>`.

### Edit a trigger's prompt or tool list

Same as adding — edit `triggers.yml`, restart `event-dispatcher`. Sidecars don't need restarting (they receive the prompt from the dispatcher per dispatch).

### Remove a trigger

Delete the entry from `triggers.yml`, restart `event-dispatcher`. Any in-flight dispatches for that trigger continue to completion; new fires will return 404.

### Debug a misbehaving trigger

| Symptom | First check |
|---------|-------------|
| Webhook returns 401 | Check `Authorization: Bearer` value matches `${env.EVENT_DISPATCHER_TOKEN}` on the deploy server |
| Webhook returns 404 | Trigger name typo, or the trigger isn't loaded — check dispatcher logs for `Loaded N triggers` |
| Webhook returns 200 but agent never runs | Check sidecar pool: `curl http://${config.deploy.host}:${config.modules.event_dispatcher.port}/health` → look at `pool.queued` and `pool.running` |
| Dispatch sits at `running` forever | Sidecar is hung on a tool call — `${config.runtime.engine} logs ${config.facility.prefix}-dispatch-sidecar-<N>` |
| Dispatch ends in `error` | Stream the run with `/stream` to see exactly which event failed; the dispatcher's poll response also includes `result.error` |
| EPICS trigger never fires | Check `EPICS_CA_ADDR_LIST` includes the PV's host; check `cool_down_sec` isn't suppressing; check the first-callback suppression isn't masking the only edge you'll get |
| Agent transcript shows no MCP tool calls | The trigger's `allowed_tools` list is empty or wrong — sidecars enforce this list strictly |

### Inspect the agent transcript for a past dispatch

Sidecars persist run records; query the sidecar:

```bash
curl http://${config.deploy.host}:<sidecar_port>/dispatch/<run_id> \
  -H "Authorization: Bearer ${env.DISPATCH_SIDECAR_TOKEN}" | python3 -m json.tool
```

The `result.transcript` field has the full text + tool call sequence.

---

## Disabling

To remove the module from a running facility:

1. Set `modules.event_dispatcher.enabled: false` in `facility-config.yml` (and remove the sub-block contents to keep the file clean).
2. Re-run scaffolding — compose loses the dispatcher + sidecar services, CI loses the build jobs, deploy.sh loses the dispatcher-specific env exports.
3. Re-deploy: `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh --clean"` to ensure stopped containers are removed.
4. Optionally delete `triggers.yml` from the repo and the dispatcher token entries from `.env`.

In-flight dispatches are dropped on shutdown; if a long-running agent is mid-tool-call, it will be SIGTERM'd by compose. Re-enabling later is symmetric: re-edit config, re-scaffold, re-deploy. `triggers.yml` is preserved across enable/disable cycles unless the user deletes it explicitly.
