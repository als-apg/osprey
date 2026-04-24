# Module: olog

Electronic logbook integration. Enable this when the facility has an operations logbook (Phoebus OLOG, OLOG-RPC, custom REST) that the assistant should be able to read from — and optionally write to. Reading is almost always the right starting point: agents can summarize recent activity, correlate machine state with logbook entries, and cite the operator narrative when explaining what happened. Writing is more sensitive and is **off by default**.

This module is generic. OSPREY ships read/write adapters for the common patterns (Phoebus OLOG REST, OLOG-RPC); facility-specific quirks (custom auth, non-standard endpoints, ARIEL-style derived views) belong in a facility-local adapter that follows the same interface. See the OSPREY adapter directory referenced in `references/scaffolding.md` for the extension pattern; do not bake facility specifics into this module.

**Enabled when**: `modules.olog.enabled: true` in `facility-config.yml`.

---

## Configuration

Full schema in `references/facility-config-schema.md` § `modules.olog`. The fields:

```yaml
modules:
  olog:
    enabled: true
    api_url: "https://logbook.example.org/olog/"        # production endpoint
    test_url: "https://logbook.example.org/olog_test/"  # optional; required for write tests
    auth_method: "basic"                                 # basic | bearer | api_key
    username_env_var: "OLOG_USERNAME"                    # for basic
    password_env_var: "OLOG_PASSWORD"                    # for basic
    # token_env_var: "OLOG_TOKEN"                        # for bearer / api_key
    # api_key_header: "X-API-Key"                        # for api_key
    write_test_enabled: false                            # gate for the write-side check
```

### Auth methods

| Method | What the request looks like | Env vars needed |
|--------|------------------------------|-----------------|
| `basic` | `Authorization: Basic <base64(user:pass)>` | `username_env_var`, `password_env_var` |
| `bearer` | `Authorization: Bearer ${env.<token_env_var>}` | `token_env_var` |
| `api_key` | `<api_key_header>: ${env.<token_env_var>}` | `token_env_var`, `api_key_header` |

The choice depends on what the logbook server enforces. Phoebus OLOG defaults to basic; custom REST gateways often standardize on bearer.

### Why writes are off by default

Logbook entries are durable, signed-by-author records of operations history. A misbehaving agent that posts an entry to the production logbook leaves a permanent footprint that operations may have to manually annotate-as-test. Until the facility has explicit policy for agent-authored entries (and ideally a separate test logbook), `write_test_enabled` stays `false` and the write-side check is `SKIP`. Even with `write_test_enabled: true`, writes go to `test_url`, not `api_url` — production writes from this module are never automatic.

---

## What scaffolding adds when this module is enabled

### compose

Nothing. OLOG is reached over HTTPS from the integration-tests container and from any agent / MCP server that wants it; no new services are spawned.

If `modules.ariel.sync_source: olog`, the ARIEL ingestion pipeline reads from the `api_url` configured here — but that wiring lives in the ARIEL module, not OLOG.

### .gitlab-ci.yml

Nothing.

### scripts/deploy.sh

Nothing. The OLOG endpoint is external; deploy.sh doesn't touch it.

### scripts/verify.sh

- `olog_reachable`: `curl -fsS --max-time 10 ${config.modules.olog.api_url}` (with the configured auth headers). A `200` or `401` proves the endpoint is up; the integration tests distinguish between `reachable but unauthorized` (creds problem) and `unreachable` (network problem).
- `olog_credentials_present`: shell test that the configured env vars are non-empty. SKIP, not ERROR, if missing — operator may not have set creds yet.
- `olog_write_check` (only when `write_test_enabled: true` and `test_url` is set): POSTs a probe entry to `test_url` and reads it back; otherwise SKIP.

### .env.template

```
# OLOG (modules.olog)
${config.modules.olog.username_env_var}=
${config.modules.olog.password_env_var}=
# Set OLOG_WRITE_TEST_ENABLED=1 only on machines where automated test writes
# are explicitly permitted (test logbook only — never production).
OLOG_WRITE_TEST_ENABLED=0
```

(If `auth_method` is `bearer` or `api_key`, the template lists the relevant token env var instead of username/password.)

### Other files

None. If the facility writes a custom adapter, it lives in the facility profile repo (typically `mcp_servers/<adapter_name>/` or `overlays/`), and is wired into the assistant via the standard MCP server pattern — not via this module.

---

## Read paths

The integration test category `olog` exercises the read side:

- `olog.api_reachable` — GET `${api_url}` with auth, expect 200.
- `olog.recent_entry_count` — query the last N hours of entries, count them. WARNING if zero (logbook is up but no traffic), OK otherwise.

Agents and MCP tools that consume OLOG read it through whichever logbook client the facility wires in (the OLOG MCP server in OSPREY's standard set, or a facility-local adapter). They authenticate with the same env vars listed above.

## Write paths

The write-side check covers the round-trip:

- `olog.write_probe` — POST a probe entry to `${test_url}` with a known title, parse the response for an entry id, GET the entry back, assert author/title match. SKIP when `write_test_enabled` is false or `test_url` is empty.

Production writes (real operator-authored entries triggered by an agent action) use the same auth and endpoint patterns. If the facility's policy is "agents may write to OLOG with operator approval," that approval workflow is the agent's responsibility — this module just configures the credentials.

---

## Operating the module

### Create or rotate credentials

1. Get a logbook account from the facility's controls group (or generate a service account if the logbook supports them).
2. Edit `.env` on the deploy server:
   ```
   ${config.modules.olog.username_env_var}=newuser
   ${config.modules.olog.password_env_var}=newpass
   ```
3. Restart any container that holds the credentials in process memory. The integration-tests container reads `.env.production` on each `/checks` invocation, so a new check will use the new creds; long-running MCP servers that cached credentials at startup need a `${config.runtime.compose_command} restart <service>`.
4. Verify: `./scripts/verify.sh olog`.

### Test read access manually

```bash
# basic
curl -u "${env.OLOG_USERNAME}:${env.OLOG_PASSWORD}" \
  "${config.modules.olog.api_url}?op=retrieve&start=1&end=2"

# bearer
curl -H "Authorization: Bearer ${env.OLOG_TOKEN}" \
  "${config.modules.olog.api_url}/entries?limit=1"
```

A 200 with an XML/JSON body confirms the auth + endpoint. A 401 means the endpoint works but the credentials don't. A connection error means the endpoint is unreachable from the deploy server.

### Test write access manually (with permission)

Only against `test_url`, never `api_url`:

```bash
curl -X POST -u "${env.OLOG_USERNAME}:${env.OLOG_PASSWORD}" \
  -H "Content-Type: application/xml" \
  -d '<entry><title>probe</title><text>integration test</text></entry>' \
  "${config.modules.olog.test_url}"
```

The response should include the new entry's id. Read it back with `?op=retrieve&start=<id>&end=<id>`.

### Common failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| 401 on every call | Token expired, password changed, or wrong scope | Rotate via the steps above |
| 403 | Auth succeeded but the user isn't allowed in the requested logbook | Get the right group/role from controls |
| 404 on POST but GET works | `test_url` is wrong, or the endpoint expects a different POST path (`/entries` vs `/rpc.php`) | Re-check the logbook's API docs and update `test_url` |
| 200 on POST but the entry doesn't appear in the operator's view | You wrote to the test logbook, not production — by design | Confirm the entry is at `test_url`, not `api_url` |
| Verify check `olog_reachable` ERROR but browser works | Deploy server is behind a proxy that doesn't have the logbook in `no_proxy` | Add the logbook hostname to `${config.network.no_proxy}` |
| Read check OK, write check SKIP | `write_test_enabled` is false or `test_url` is unset — by design | Set both intentionally if a write test is wanted |

### When to write a facility-local adapter

If your logbook:
- Doesn't speak any of the standard OSPREY adapters' wire protocols, or
- Has a derived/normalized view (e.g., ARIEL-style enriched entries) the agent should query instead of the raw upstream, or
- Has an unusual auth flow (mTLS, session cookies, SSO) the bundled adapters don't handle,

then create a facility-local MCP server (under `mcp_servers/` in the facility profile repo) that wraps the logbook and exposes a clean tool surface to the assistant. This module still provides the credentials and endpoints; the adapter consumes them. **Writes always go to the upstream source via the adapter, never directly to a derived view.**

---

## Disabling

1. Set `modules.olog.enabled: false` in `facility-config.yml`.
2. Re-scaffold — `verify.sh` loses the OLOG check, `.env.template` loses the credential lines.
3. Remove the OLOG env vars from `.env` if they aren't used by any other tool.
4. If `modules.ariel.sync_source: olog` is set, ARIEL must be reconfigured first — disabling OLOG while ARIEL still depends on it leaves ARIEL with no ingestion source. The interview catches this.

No data is destroyed by disabling — the upstream logbook is untouched.
