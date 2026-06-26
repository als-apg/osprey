# Module: web_terminals

A multi-user web terminal stack lets named operators reach the built assistant from a browser without installing Claude Code locally. Each user gets a dedicated container running `osprey web` with a private Claude Code config, private session memory, and a private CLAUDE.md. An nginx reverse proxy in front offers a landing page and routes browser requests to the right per-user container. Enable this when the facility wants more than one or two people using the assistant interactively, when laptops can't reach internal services directly, or when persistent per-user state matters.

**Enabled when**: `modules.web_terminals.enabled: true` in `facility-config.yml`.

---

## Architecture

```
                          ┌──────────────────────────┐
   browser ───────────────│  nginx                    │
   :${config.modules.     │  :${config.modules.       │
    web_terminals.        │   web_terminals.          │
    nginx_port}           │   nginx_port}             │
                          │  - landing page           │
                          │  - per-user reverse proxy │
                          └────────────┬──────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
   ${prefix}-web-${user1}  ${prefix}-web-${user2}  ${prefix}-web-${userN}
   :${base_port + 0}        :${base_port + 1}        :${base_port + N-1}
   - osprey web             - osprey web             - osprey web
   - per-user CLAUDE.md     - per-user CLAUDE.md     - per-user CLAUDE.md
   - per-user named volume  - per-user named volume  - per-user named volume
   - all MCP servers, all   - all MCP servers, all   - all MCP servers, all
     stdio subprocesses       stdio subprocesses       stdio subprocesses
     (confluence, etc.)       (confluence, etc.)       (confluence, etc.)
```

Each container is a complete OSPREY runtime. The Claude Code project (and its `.mcp.json`, `config.yml`, agents, skills, hooks) is **baked into the image at CI time** — there is no rsync to the deploy server. At runtime each container exec's `osprey web`, which serves the chat UI on the container's internal port and mounts the per-user named volume at `/data/claude-config/`.

---

## Configuration

Full schema in `references/facility-config-schema.md` § `modules.web_terminals`. Most-used fields:

```yaml
modules:
  web_terminals:
    enabled: true
    nginx_port: 9080                    # ${config.modules.web_terminals.nginx_port}
    base_port: 9091                     # first per-user terminal port
    users:                              # one container per user
      - alice
      - bob
      - carol
    landing_page_template: "default"    # default | custom
```

**Port allocation rule**: each user container binds host port `${config.modules.web_terminals.base_port} + index` (where `index` is the zero-based position in the `users:` list). With `base_port: 9091` and three users, the bindings are `9091 → user[0]`, `9092 → user[1]`, `9093 → user[2]`. The interview enforces no collision with `${config.ports.*}`.

**The per-user list is durable**. Adding a user appends to the list (re-run the interview, or hand-edit then re-scaffold) without disturbing existing users' state. Removing a user from the list does **not** delete their named volume — the volume sticks around and can be reattached if the user comes back. To actually wipe a user's state, run `${config.runtime.engine} volume rm ${config.facility.prefix}-${user}-claude-config`.

---

## What scaffolding adds when this module is enabled

### compose

A compose overlay (`docker-compose.web.yml` by default; added to `${config.runtime.compose_files}` after the base) with:

- One `nginx` service (container name `${config.facility.prefix}-nginx`) listening on `0.0.0.0:${config.modules.web_terminals.nginx_port}`. Mounts `./nginx/nginx.conf` and `./nginx/landing.html` read-only.
- An anchor `&web-terminal` block holding image, restart policy, env_file, network — extended by every per-user service.
- One service per user via list expansion:

```
# FOR each in modules.web_terminals.users
  web-${each}:
    <<: *web-terminal
    container_name: ${config.facility.prefix}-web-${each}
    ports:
      - "0.0.0.0:${config.modules.web_terminals.base_port + each.index}:9087"
    environment:
      - CLAUDE_CONFIG_DIR=/data/claude-config
      - HOME=/data/claude-config
      - ${config.facility.prefix|upper}_TERMINAL_USER=${each}
      # ...plus any module-cross-references (DISPATCH_SIDECAR_TOKEN if event_dispatcher enabled)
    volumes:
      - ${each}-claude-config:/data/claude-config
      - ${each}-agent-data:/app/${config.facility.prefix}-assistant/_agent_data
# END FOR
```

- Named volumes (one pair per user):

```
# FOR each in modules.web_terminals.users
  ${each}-claude-config:
  ${each}-agent-data:
# END FOR
```

The named volumes live on the container engine's local graphroot — **not** on any NFS mount. This is deliberate: rootless container UIDs don't survive NFS write paths reliably, but Claude Code writes to its config dir continuously during a session.

### .gitlab-ci.yml

- One `build-web-terminal` job in the `docker-build` stage (Dockerfile at `docker/Dockerfile.web-terminal`). The image is multi-stage:
  1. **Node stage** — installs Claude Code (the npm package).
  2. **Python stage** — installs OSPREY + facility profile dependencies into a venv.
  3. **Final stage** — copies the built project from `artifacts/${config.facility.prefix}-assistant/` (produced by an earlier `osprey build` CI job), then runs the path regen step (see below).

### scripts/deploy.sh

- For every user in `${config.modules.web_terminals.users}`:
  1. Ensures the user's named volumes exist (the engine creates them on first `compose up`, but the seeding step below needs them present).
  2. **Seeds CLAUDE.md** by streaming the concatenation of `docker/web-terminal-context/base.md` and `docker/web-terminal-context/${user}.md` into the user's `claude-config` volume at `/data/claude-config/CLAUDE.md`. This runs every deploy so per-user memory updates land without rebuilding the image.
- After `compose up -d`, optional readiness probe: HEAD requests against `http://localhost:${config.modules.web_terminals.base_port + index}/` until a 200 comes back, with a 30s timeout per user.

### scripts/verify.sh

- `nginx_health`: `curl -fsS http://localhost:${config.modules.web_terminals.nginx_port}/`
- `web_terminal_health` (per user): `curl -fsS http://localhost:${config.modules.web_terminals.base_port + index}/health`

### .env.template

No new entries are required by this module on its own — web terminals reuse the LLM provider key and any other secrets the assistant already needs. If `modules.event_dispatcher` is also enabled, the sidecar token env var is propagated into each terminal so it can call sidecar endpoints.

### Other files

- `nginx/nginx.conf` — generated; routes `/u/<user>/` to the matching upstream and serves the landing page at `/`.
- `nginx/landing.html` — generated from `landing_page_template`. The default template lists all users with a button per user; the `custom` setting tells scaffolding to skip overwriting if the file already exists.
- `docker/web-terminal-context/base.md` — generated empty stub; intended to hold guidance every user sees.
- `docker/web-terminal-context/<user>.md` — one empty stub per user; intended for per-user nicknames, working preferences, etc.

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

`<port>` is `${config.modules.web_terminals.base_port} + <user_index>`. Scripts that don't care which user runs the prompt can pin to user[0]; scripts that want isolation per requester should pick deterministically.

---

## Operating the module

### Add a user

1. Append the new login to `modules.web_terminals.users` in `facility-config.yml` (re-run the interview, or hand-edit and skip ahead).
2. Re-scaffold: regenerates the compose overlay with the new service + named volumes, and adds a row to the nginx upstream block.
3. Create the per-user context stub: `touch docker/web-terminal-context/<new_user>.md`.
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
| nginx serves landing page but reverse-proxy to user URLs returns 502 | Per-user container internal port doesn't match nginx upstream — both should be `9087` (the port `osprey web` binds inside the container) |

### Updating per-user CLAUDE.md without redeploying images

The CLAUDE.md seeding runs in `deploy.sh`, not in the image build, so:

1. Edit `docker/web-terminal-context/base.md` or `docker/web-terminal-context/<user>.md`.
2. Commit + push (so the change reaches the deploy server's git checkout).
3. `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh"` — git pull picks up the new files, and the seed step copies them in. Containers stay up; the file just changes under them.

The user must restart their Claude session in the browser to pick up the new CLAUDE.md (Claude Code reads it on session start).

---

## Disabling

1. Set `modules.web_terminals.enabled: false` in `facility-config.yml`.
2. Re-scaffold — the compose overlay file is removed from `${config.runtime.compose_files}` and (optionally) deleted; the web-terminal CI job is removed.
3. `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh --clean"` to stop and remove the per-user containers and the nginx container.
4. Named volumes survive disable. To wipe them too, run `${config.runtime.engine} volume rm` per user (script the loop over the old user list).
5. If `modules.benchmarks` was enabled, it must also be disabled (benchmarks run inside a web-terminal container; the interview enforces this).
