# Module: web_terminals

A multi-user web terminal stack lets named operators reach the built assistant from a browser without installing Claude Code locally. Each user gets a dedicated container running `osprey web` with a private Claude Code config, private session memory, and a private CLAUDE.md. That same container also runs three companion services per user — an artifact gallery, ARIEL search, and a lattice dashboard — each bound to its own port, so nothing collides across users or across a single user's own services. An nginx reverse proxy in front offers a landing page and routes browser requests to the right per-user service. Enable this when the facility wants more than one or two people using the assistant interactively, when laptops can't reach internal services directly, or when persistent per-user state matters.

**Enabled when**: `modules.web_terminals.enabled: true` in `facility-config.yml`.

> **Perimeter-trust only — NOT authenticated, NOT TLS-terminated.** Anyone who can reach a configured port gets in as whoever owns that service; there is no login and nothing encrypts the traffic in front of it. Every per-user service binds `0.0.0.0` directly under `network_mode: host`, so nginx is a landing-page convenience, **not** a security chokepoint — it can be bypassed entirely by hitting a per-user port directly. Only deploy this module on a network you already trust (VPN-only segment, firewalled lab subnet, etc.). Authentication, TLS termination, and single-origin routing are Phase 2 work and are not present today.

---

## Architecture

```
                          ┌─────────────────────────────┐
   browser ───────────────│  nginx                      │
                          │  :${config.modules.         │
                          │   web_terminals.nginx_port} │
                          │  - landing page             │
                          │  - per-user reverse proxy   │
                          └────────────┬────────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
       ${prefix}-web-${user[0]}  ${prefix}-web-${user[1]}  ${prefix}-web-${user[N-1]}
        one container per user — same image, four independently-bound host ports
```

Each user's single container publishes four ports, one per service family:

| Family (per-user service) | Env var inside the container | Host port for user at index `i` |
|----------------------------|--------------------------------|-----------------------------------|
| web (terminal UI, `osprey web`) | `OSPREY_WEB_PORT` | `web_base_port + i` |
| artifact (build artifact gallery) | `OSPREY_ARTIFACT_SERVER_PORT` | `artifact_base_port + i` |
| ariel (ARIEL search) | `OSPREY_ARIEL_PORT` | `ariel_base_port + i` |
| lattice (lattice dashboard) | `OSPREY_LATTICE_DASHBOARD_PORT` | `lattice_base_port + i` |

Each container also has a private CLAUDE.md, a private named volume mounted at `/data/claude-config/`, and every MCP server / stdio subprocess the profile configures (confluence, etc.).

Each container is a complete OSPREY runtime. The Claude Code project (and its `.mcp.json`, `config.yml`, agents, skills, hooks) is **baked into the image at CI time** — there is no rsync to the deploy server. At runtime each container exec's `osprey web`, whose unconfigured default listen port is the fixed constant **8087**; this module overrides it per user via `OSPREY_WEB_PORT=<web_base_port + i>` (env-over-config), so the process binds directly to its published host port with no internal-port remapping under `network_mode: host`. The artifact/ariel/lattice companion servers auto-launch the same way inside the same container, each reading its own `OSPREY_*_PORT` env var (see `src/osprey/registry/web.py` for their unconfigured defaults, which are not 8087 and are not shared across families).

---

## Configuration

Full schema in `references/facility-config-schema.md` § `modules.web_terminals`. Most-used fields:

```yaml
modules:
  web_terminals:
    enabled: true
    nginx_port: 9080                    # ${config.modules.web_terminals.nginx_port}
    web_base_port: 9091                 # first per-user web-terminal port      → OSPREY_WEB_PORT
    artifact_base_port: 9291            # first per-user artifact-gallery port  → OSPREY_ARTIFACT_SERVER_PORT
    ariel_base_port: 9391               # first per-user ARIEL search port      → OSPREY_ARIEL_PORT
    lattice_base_port: 9491             # first per-user lattice-dashboard port → OSPREY_LATTICE_DASHBOARD_PORT
    users:                              # one container per user
      - alice
      - bob
      - carol
    landing:                            # grouped landing page served at nginx_port
      groups:
        - type: "users"                 # auto-populated: one card per entry in `users` above
        - type: "links"
          label: "Facility Tools"
          links:
            - label: "Elog"
              url: "https://elog.example.org"
```

**Port allocation rule**: each user's four services bind host port `<family>_base_port + index` (where `index` is the zero-based position in the `users:` list, and `family` is one of `web`, `artifact`, `ariel`, `lattice`). With the values above and three users, `alice` (index 0) gets `9091`/`9291`/`9391`/`9491`, `bob` (index 1) gets `9092`/`9292`/`9392`/`9492`, and so on. This arithmetic is implemented once, in `deployment/web_terminals/ports.py`'s `allocate_ports()` — the renderer and the lint both call it rather than reimplementing it. The interview enforces no collision across all four families, `${config.ports.*}`, and the event-dispatcher sidecar range (see validation rule 11 in `facility-config-schema.md`).

**The per-user list is durable**. Adding a user appends to the list (re-run the interview, or hand-edit then re-scaffold) without disturbing existing users' state. Removing a user from the list does **not** delete their named volume — the volume sticks around and can be reattached if the user comes back. To actually wipe a user's state, run `${config.runtime.engine} volume rm ${config.facility.prefix}-${user}-claude-config`.

---

## What scaffolding adds when this module is enabled

### compose

A compose overlay (`docker-compose.web.yml` by default; added to `${config.runtime.compose_files}` after the base) with:

- One `nginx` service (container name `${config.facility.prefix}-nginx`) listening on `0.0.0.0:${config.modules.web_terminals.nginx_port}`. Mounts `./nginx/nginx.conf` and `./nginx/landing.html` read-only.
- An anchor `&web-terminal` block holding image, restart policy, env_file, network — extended by every per-user service.
- One service per user, each publishing all four `*_base_port + index` host ports (per the Architecture table above) with `OSPREY_TERMINAL_USER=<user>` and the other three `OSPREY_*_PORT` env vars set alongside it — plus a paired `<user>-claude-config` / `<user>-agent-data` named volume per user, living on the container engine's local graphroot, **not** on any NFS mount. (Deliberate: rootless container UIDs don't survive NFS write paths reliably, but Claude Code writes to its config dir continuously during a session.)

**These artifacts are generated deterministically, not hand-rendered by the scaffolding skill.** Run:

```bash
osprey scaffold web-terminals render --config facility-config.yml -o <deploy dir>
```

This reads `modules.web_terminals` straight from `facility-config.yml` and writes the full compose overlay (nginx + one service per user + named volumes), the nginx routing fragment, and the static landing page into `<deploy dir>`. It lints the stanza first by default and aborts on error-severity findings (`osprey scaffold web-terminals lint --config facility-config.yml` runs the same check standalone; pass `--no-lint` to render anyway). **The skill must invoke this verb for the `web_terminals` module and must never emit its own web_terminals compose or nginx fragment** — the port/env-var mapping in Architecture/Configuration above is the contract this verb implements; hand-rendering it a second time in prose is exactly the drift this verb exists to prevent.

### .gitlab-ci.yml

- One `build-web-terminal` job in the `docker-build` stage (Dockerfile at `docker/Dockerfile.web-terminal`). The image is multi-stage:
  1. **Node stage** — installs Claude Code (the npm package).
  2. **Python stage** — installs OSPREY + facility profile dependencies into a venv.
  3. **Final stage** — copies the built project from `artifacts/${config.facility.prefix}-assistant/` (produced by an earlier `osprey build` CI job), then runs the path regen step (see below).

### scripts/deploy.sh

- For every user in `${config.modules.web_terminals.users}`:
  1. Skips the user if their container isn't up yet (a `${config.runtime.engine} container exists` check) — the seed step is a no-op on a partial/failed `compose up` rather than a hard failure.
  2. **Seeds CLAUDE.md** by streaming the concatenation of `docker/web-terminal-context/base.md` and the user's overlay into the user's `claude-config` volume at `/data/claude-config/CLAUDE.md`. The overlay is `docker/web-terminal-context/<user>/extra.md`; for facilities that haven't migrated off the old flat layout, the legacy `docker/web-terminal-context/<user>.md` is read as a fallback when `<user>/extra.md` is absent. This runs every deploy so per-user memory updates land without rebuilding the image.
  3. **Seeds `skills/`** by tar-piping `docker/web-terminal-context/<user>/skills/` into `/app/${config.facility.prefix}-assistant/.claude/skills/` inside the container — the **project-scope** skills directory (`<project_cwd>/.claude/skills/`, since the web terminal launches `claude` with `cwd` set to the project directory), **not** the user-scope `CLAUDE_CONFIG_DIR/skills/` (`/data/claude-config/skills/`). This is deliberate, and easy to get wrong by intuition: the web terminal launches Claude Code with `--setting-sources project`, and that flag gates skill *discovery* itself, not just `settings.json` loading — skills placed under `CLAUDE_CONFIG_DIR/skills/` are silently never picked up under `--setting-sources project`, while the same skills under the project's `.claude/skills/` are. Only the project-scope path is live; seed there. That directory is also where a running Claude Code session keeps live, user-installed skills (e.g. anything added interactively with `osprey skills install`), so this step never blanket-replaces it: every skill directory the overlay writes gets a `.deploy-managed` sentinel, and only sentinel-bearing directories are ever touched — refreshed on each deploy, removed if the overlay stops shipping them, and left alone if they lack the sentinel. (This `--setting-sources` gating applies only to `skills/`; the CLAUDE.md seeding in step 2 above is plain user-scope auto-discovery, unrelated to `--setting-sources`, and is unaffected.)
- `scripts/deploy.sh` does not itself HTTP-probe the per-user services after `compose up -d`; the post-deploy health check is `scripts/verify.sh` (below), run separately.

### scripts/verify.sh

- `nginx_health`: `curl -fsS http://localhost:${config.modules.web_terminals.nginx_port}/`
- Per-user probe (`probe_web_terminal`): not an HTTP health check — execs into the user's container as uid 1000 and confirms it can create and remove a probe file under both `/app/${config.facility.prefix}-assistant/_agent_data/` and `/data/claude-config/`. This catches volume-permission problems but does not individually probe the artifact/ariel/lattice companion ports.

### .env.template

No new entries are required by this module on its own — web terminals reuse the LLM provider key and any other secrets the assistant already needs. If `modules.event_dispatcher` is also enabled, the sidecar token env var is propagated into each terminal so it can call sidecar endpoints.

### Other files

- `nginx/nginx.conf` — generated; routes `/u/<user>/` to the matching upstream and serves the landing page at `/`.
- `nginx/landing.html` — rendered from `modules.web_terminals.landing.groups` (see Configuration above): a `type: "users"` group auto-populates one card per entry in `users:`, and any `type: "links"` groups render as static link lists (e.g. facility tools). There is no `landing_page_template` field anymore — the landing page is fully data-driven from `landing.groups`.
- `docker/web-terminal-context/base.md` — generated empty stub; intended to hold guidance every user sees.
- `docker/web-terminal-context/<user>/extra.md` — one empty stub per user; intended for per-user nicknames, working preferences, etc. The legacy flat `docker/web-terminal-context/<user>.md` is still read as a fallback when this file doesn't exist (see `scripts/deploy.sh` above).
- `docker/web-terminal-context/<user>/skills/` — optional per-user skill overlay, deploy-managed and sentinel-tracked (see `scripts/deploy.sh` above); never touches user-authored skills already present in the running container. Seeded to the container's **project-scope** `.claude/skills/`, not `CLAUDE_CONFIG_DIR` — see `scripts/deploy.sh` for why.

---

## Container path regeneration (build-time)

The CI job that builds the project produces it at a CI-runner path like `/builds/<group>/<project>/artifacts/${config.facility.prefix}-assistant/`. The Dockerfile then `COPY`s that into the image at `/app/${config.facility.prefix}-assistant/`. Without correction, `config.yml` and `.mcp.json` in the copied tree still reference the CI build paths — every Python MCP server fails because `command:` and `OSPREY_CONFIG=` point at directories that don't exist in the runtime image.

The Dockerfile's final stage fixes this with a post-COPY step:

1. Clears `execution.python_env_path` from `config.yml` so OSPREY falls back to `sys.executable` (i.e., the venv's interpreter).
2. Sets `project_root` to `/app/${config.facility.prefix}-assistant`.
3. Runs `osprey claude regen --project /app/${config.facility.prefix}-assistant` to re-render `.mcp.json` and the `claude/` settings using the new paths.

If MCP servers come up dead after a deploy and the dispatch logs show paths that look like CI runner paths, this regen step is the first place to look.

---

## REST chat API

Each web terminal exposes `POST /api/chat` for programmatic Claude access — useful for scripted smoke tests, dashboards, or integrations that want to ask the assistant a question without opening a browser.

```bash
# SSE (default): one event per token / tool call
curl -N -X POST http://${config.deploy.host}:<port>/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "what is the current beam current?"}'

# Buffered JSON (single response when the agent finishes)
curl -X POST 'http://${config.deploy.host}:<port>/api/chat?stream=false' \
  -H "Content-Type: application/json" \
  -d '{"prompt": "summarize the last hour"}'
```

`<port>` is `${config.modules.web_terminals.web_base_port} + <user_index>` — the REST API rides on the `web` family port, the same one the terminal UI uses, not the artifact/ariel/lattice ports. Scripts that don't care which user runs the prompt can pin to user[0]; scripts that want isolation per requester should pick deterministically.

---

## Operating the module

### Add a user

1. Append the new login to `modules.web_terminals.users` in `facility-config.yml` (re-run the interview, or hand-edit and skip ahead).
2. Re-scaffold: regenerates the compose overlay with the new service + named volumes, and adds a row to the nginx upstream block.
3. Create the per-user context stub: `mkdir -p docker/web-terminal-context/<new_user> && touch docker/web-terminal-context/<new_user>/extra.md`. (The legacy flat `docker/web-terminal-context/<new_user>.md` still works, but new facilities should use the directory form so a `skills/` overlay can be added later without restructuring.)
4. Commit and push. CI rebuilds nothing image-side (the user list isn't baked into the image), but the new container only appears after a deploy.
5. Deploy: `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh"`. The seeding step creates the user's CLAUDE.md on first run.

Existing users are unaffected — their named volumes are untouched.

### Remove a user

1. Remove the entry from `modules.web_terminals.users`.
2. Re-scaffold + commit + push + deploy. The user's container is stopped and removed.
3. Their named volumes (`<user>-claude-config`, `<user>-agent-data`) are **kept** unless explicitly deleted:
   ```bash
   ssh ${config.deploy.host} "${config.runtime.engine} volume rm \
     ${config.facility.prefix}-<user>-claude-config \
     ${config.facility.prefix}-<user>-agent-data"
   ```

### Inspect a user's session state

```bash
ssh ${config.deploy.host} "${config.runtime.engine} exec ${config.facility.prefix}-web-<user> \
  ls -la /data/claude-config/"
```

Look for `CLAUDE.md` (seeded), `.claude/` (Claude Code's session db, history, credentials), and any project-specific files the user added.

### Inspect the regenerated `.mcp.json` inside a container

```bash
ssh ${config.deploy.host} "${config.runtime.engine} exec ${config.facility.prefix}-web-<user> \
  cat /app/${config.facility.prefix}-assistant/.mcp.json | python3 -m json.tool"
```

Every `command:` should reference `/app/...`, never `/builds/...`. If the latter shows up, the path regen step in the Dockerfile didn't run — rebuild the web-terminal image.

### Common failure modes

| Symptom | First check |
|---------|-------------|
| User opens the landing page, clicks their button, gets a blank/500 page | Check `${config.runtime.engine} ps` for the user's container; if exited, check `${config.runtime.engine} logs ${config.facility.prefix}-web-<user>` |
| Container is up but `osprey web` immediately exits with "no project" | Path regen step skipped or failed during image build — rebuild the image |
| User's CLAUDE.md is empty | The seeding step in `deploy.sh` couldn't reach the named volume — verify the volume exists and is mounted |
| Claude session "forgets" between visits | Named volume was destroyed — `--nuke` deploy or someone manually `volume rm`'d it |
| All web terminals see the same memory | Containers are sharing a volume — check the per-user `volumes:` block in the compose overlay rendered correctly (no anchor reuse bug) |
| nginx serves landing page but reverse-proxy to user URLs returns 502 | Per-user service's env-var-overridden listen port doesn't match nginx's upstream for that family — check `OSPREY_WEB_PORT`/`OSPREY_ARTIFACT_SERVER_PORT`/`OSPREY_ARIEL_PORT`/`OSPREY_LATTICE_DASHBOARD_PORT` inside the container against the `*_base_port + index` nginx expects. (The `web` family's unconfigured default is `8087` — a service still listening on that default instead of its assigned host port is the usual cause under `network_mode: host`.) |

### Updating per-user CLAUDE.md and skills without redeploying images

The CLAUDE.md and `skills/` seeding runs in `deploy.sh`, not in the image build, so:

1. Edit `docker/web-terminal-context/base.md` and/or the user's `docker/web-terminal-context/<user>/extra.md` (or the legacy flat `<user>.md`); add or edit files under `docker/web-terminal-context/<user>/skills/<skill-name>/` for a per-user skill.
2. Commit + push (so the change reaches the deploy server's git checkout).
3. `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh"` — git pull picks up the new files, and the seed step copies them in. Containers stay up; the files just change under them.

The user must restart their Claude session in the browser to pick up the new CLAUDE.md or skills (Claude Code reads them on session start).

---

## Disabling

1. Set `modules.web_terminals.enabled: false` in `facility-config.yml`.
2. Re-scaffold — the compose overlay file is removed from `${config.runtime.compose_files}` and (optionally) deleted; the web-terminal CI job is removed.
3. `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh --clean"` to stop and remove the per-user containers and the nginx container.
4. Named volumes survive disable. To wipe them too, run `${config.runtime.engine} volume rm` per user (script the loop over the old user list).
5. If `modules.benchmarks` was enabled, it must also be disabled (benchmarks run inside a web-terminal container; the interview enforces this).
