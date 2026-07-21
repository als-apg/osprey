# Module: web_terminals

A multi-user web terminal stack lets named operators reach the built assistant from a browser without installing Claude Code locally. Each user gets a dedicated container running `osprey web` with a private Claude Code config, private session memory, and a private CLAUDE.md. That same container also runs the profile's companion panel services per user — artifact gallery, ARIEL search, channel finder, lattice dashboard, OKF knowledge — each bound to its own port, so nothing collides across users or across a single user's own services. An nginx reverse proxy in front offers a landing page and routes browser requests to the right per-user service. Enable this when the facility wants more than one or two people using the assistant interactively, when laptops can't reach internal services directly, or when persistent per-user state matters.

**Enabled when**: `modules.web_terminals.enabled: true` in `facility-config.yml`.

> **Loopback-bound apps behind a single chokepoint — still perimeter-trust only.** Every per-user service (the web family plus every companion-panel family) binds `127.0.0.1` inside its container, so nginx is the **only** process that can reach them from outside — hitting a per-user host port directly no longer reaches anything. That does **not** mean the stack is authenticated or encrypted: `modules.web_terminals.auth` and `.tls` exist as config-gated seams, but with their defaults (`auth.method: none`, `tls.enabled: false`) traffic reaching nginx is still unauthenticated, cleartext HTTP, open to anyone who reaches the deploy host. Enabling `auth.method` to anything other than `none` fails **closed** — every request gets `403` — rather than allow-all, since no real auth backend exists behind the seam yet. Only deploy this module on a network you already trust (VPN-only segment, firewalled lab subnet, etc.) until a real auth backend and real TLS certificates are configured.

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
        one container per user — same image by default, independently-bound host ports per family
```

Every user's container runs the same image and the same rendered project by default.
A facility can instead assign different users different images and different
projects — see "Personas" below — without changing anything else about this
architecture: each container still publishes the same port families, still gets its
own named volumes, and nginx still fronts all of them identically.

Each user's single container publishes one port per service family — the terminal
itself plus one family per companion panel in `registry/web.py`
`FRAMEWORK_WEB_SERVERS`. The family set is **derived from that registry** (a newly
registered panel gets its family, config knob, and compose env line automatically),
and every companion family carries a registry default base port so configs written
before a panel existed keep deploying unchanged:

| Family (per-user service) | Env var inside the container | Host port for user at index `i` |
|----------------------------|--------------------------------|-----------------------------------|
| web (terminal UI, `osprey web`) | `OSPREY_WEB_PORT` | `web_base_port + i` (required) |
| artifact (build artifact gallery) | `OSPREY_ARTIFACT_SERVER_PORT` | `artifact_base_port + i` (default 9291) |
| ariel (ARIEL search) | `OSPREY_ARIEL_PORT` | `ariel_base_port + i` (default 9391) |
| lattice (lattice dashboard) | `OSPREY_LATTICE_DASHBOARD_PORT` | `lattice_base_port + i` (default 9491) |
| channel_finder (channel-finder panel) | `OSPREY_CHANNEL_FINDER_PORT` | `channel_finder_base_port + i` (default 9591) |
| okf (OKF knowledge panel) | `OSPREY_FACILITY_KNOWLEDGE_PORT` | `okf_base_port + i` (default 9691) |

Each container also has a private CLAUDE.md, a private named volume mounted at `/data/claude-config/`, and every MCP server / stdio subprocess the profile configures (confluence, etc.).

Each container is a complete OSPREY runtime. The Claude Code project (and its `.mcp.json`, `config.yml`, agents, skills, hooks) is **baked into the image at CI time** (registry mode) or **built locally by `deploy up`** (local mode — see "Personas" below) — either way there is no rsync to the deploy server. At runtime each container exec's `osprey web`, whose unconfigured default listen port is the fixed constant **8087**; the compose overlay declares the published host port into the container as `OSPREY_TERMINAL_WEB_PORT=<web_base_port + i>`, and `osprey web`'s `resolve_web_port()` (`src/osprey/cli/web_cmd.py`) treats a *declared* `OSPREY_TERMINAL_WEB_PORT` as authoritative over `--port`, the `OSPREY_WEB_PORT` click envvar, and config — the same declared-wins pattern as `OSPREY_TERMINAL_BIND_HOST` below, so the process binds directly to its published host port with no internal-port remapping under `network_mode: host` even if a stale or hostile image `CMD` passes a mismatched `--port`. (`OSPREY_TERMINAL_WEB_PORT` is scoped to this one container and is never re-exported to child processes, unlike `OSPREY_WEB_PORT`, which is — a nested `osprey web --port X` launched manually inside the container still gets today's explicit-flag-wins behavior.) The companion panel servers (artifact, ariel, channel finder, lattice, OKF) auto-launch the same way inside the same container, each reading its own `OSPREY_*_PORT` env var (see `src/osprey/registry/web.py` for their unconfigured defaults, which are not 8087 and are not shared across families).

Bind address follows the same env-over-config precedent as port: the compose overlay sets `OSPREY_TERMINAL_BIND_HOST=127.0.0.1` on every per-user service, and `osprey web`'s `resolve_bind_host()` (`src/osprey/cli/web_cmd.py`) treats a *declared* `OSPREY_TERMINAL_BIND_HOST` as **authoritative over `--host`** — even a stale or hostile image `CMD` passing `--host 0.0.0.0` gets clamped back to loopback (logged as a stderr NOTICE). Every per-user app therefore only answers on `127.0.0.1`; nginx, which itself listens on every interface, is the sole off-host path in. A per-user image's `CMD` must never rely on `--host 0.0.0.0` to be reachable — reachability comes from nginx proxying to the loopback upstream, not from the app's own bind address. A deployment can still opt out of loopback binding for a given app by setting `OSPREY_TERMINAL_BIND_HOST=0.0.0.0` explicitly; this is loud and greppable, and the module's existing `0.0.0.0`-exposure warning still fires for it.

---

## Configuration

Full schema in `references/facility-config-schema.md` § `modules.web_terminals`. Most-used fields:

```yaml
modules:
  web_terminals:
    enabled: true
    nginx_port: 9080                    # ${config.modules.web_terminals.nginx_port}
    web_base_port: 9091                 # first per-user web-terminal port      → OSPREY_WEB_PORT (required)
    # Companion families are optional — omitted ones use registry defaults
    # (artifact 9291, ariel 9391, lattice 9491, channel_finder 9591, okf 9691).
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

**Port allocation rule**: each user's four services bind host port `<family>_base_port + index` (where `index` is the zero-based position in the `users:` list, and `family` is one of `web`, `artifact`, `ariel`, `lattice`). With the values above and three users, `alice` (index 0) gets `9091`/`9291`/`9391`/`9491`, `bob` (index 1) gets `9092`/`9292`/`9392`/`9492`, and so on. This arithmetic is implemented once, in `deployment/web_terminals/ports.py`'s `allocate_ports()` — the renderer and the lint both call it rather than reimplementing it. The interview enforces no collision across all four families, `${config.ports.*}`, and the event-dispatcher sidecar range (see validation rule 11 in `facility-config-schema.md`). When `modules.web_terminals.tls.enabled: true`, the lint also reserves port `443` (the TLS listener) against that same overlap set, so a facility can't independently assign `443` to something else in `${config.ports.*}`.

**The per-user list is durable**. Adding a user appends to the list (re-run the interview, or hand-edit then re-scaffold) without disturbing existing users' state. Removing a user from the list does **not** delete their named volume by default — the volume sticks around and can be reattached if the user comes back. To actually wipe a user's state, use `osprey deploy decommission <user> --purge` (see "Operating the module" below), or run `${config.runtime.engine} volume rm ${config.facility.prefix}-${user}-claude-config` by hand.

---

## Personas

By default, every web-terminal user shares one image and one rendered OSPREY
project — the architecture above. A **persona** lets a facility instead give a
user (or group of users) a *different* container image and a *different*
rendered project. Because a persona is its own rendered project, and per-tool
permissions are a property of a project's own `config.yml`, a persona gets
**real, enforced, per-tool permissions** for free, through the same
project-config pipeline every OSPREY project already uses — there is no new
permissions mechanism, and no merge step that could clobber a project's
pre-baked `settings.json`.

> **Naming note — this is a different "persona" than the one in code comments
> and docstrings.** A handful of comments/docstrings elsewhere in OSPREY (build
> profiles, manifest rendering, the build-artifacts catalog) already say
> "persona" to mean an alternate `CLAUDE.md` / system-prompt template selected
> at build time — e.g. a preset picking `CLAUDE.ariel.md.j2` instead of the
> default `CLAUDE.md.j2` for an ARIEL-focused build. That usage predates this
> module and is comments-only; there is no schema key for it. A web-terminal
> persona, as defined here, is the **deployment-level realization** of the same
> underlying idea — a distinct identity for an agent to run as — just carried
> one level further: instead of only swapping the system prompt within one
> shared project, it swaps the whole project (image, `config.yml`,
> permissions, MCP servers, everything). The two aren't in tension; a
> persona's own rendered project is free to also pick an alternate CLAUDE.md
> template via that older mechanism.

No `modules.web_terminals.personas` catalog at all (the common case) is exactly
today's behavior — one shared image, one shared project pinned to
`${config.facility.prefix}-assistant`, un-suffixed `web-terminal:latest`. Full
field reference: `references/facility-config-schema.md` § "Personas". Quick
shape:

```yaml
modules:
  web_terminals:
    image_source: "registry"              # registry (default) | local
    default_persona: "assistant"
    personas:
      assistant:
        project: "als-assistant"
        project_path: "../als-assistant"        # local-mode build context
        build_profile: "profiles/assistant.yml" # registry-mode CI input
      analysis:
        project: "als-analysis"
        project_path: "../als-analysis"
        build_profile: "profiles/analysis.yml"
        extra_mounts:                            # optional per-persona host mounts
          - "/opt/site-data:/app/site-data:ro"
        seed_base: false                   # optional (default true) — seed from this persona's extra.md alone, no base.md prepend
    users:
      - alice                              # no persona: → resolves to default_persona
      - name: "bob"
        index: 1                           # required for object-form entries
        persona: "analysis"                # explicit persona reference
```

### Persona-level extra mounts

A persona may declare `extra_mounts` — a list of compose volume strings appended
to the `volumes:` block of **every** user resolving to that persona, after the two
managed per-user mounts (the `<user>-claude-config` and `<user>-agent-data` named
volumes). Each entry is a plain compose volume string: a host-path bind
(`/opt/site-data:/app/site-data:ro`) or a named volume
(`shared-cache:/app/cache`), with 2 or 3 non-empty colon-separated parts. It is
persona-scoped, not per-user — mounts shared by a whole persona (a read-only
reference dataset, a shared cache) belong here, so they attach identically to
every user of that persona without repeating them on each roster entry. A
malformed entry (wrong colon count, or a non-list `extra_mounts` value) is a lint
ERROR. Omitting the key is the zero-migration default (no extra mounts). The
un-personaed roster (no `personas:` catalog at all) never gains an extra mount —
there is no catalog entry to read them from.

### Registry-mode operator flow (CI-built images, the default)

This is today's flow, extended: CI still builds and pushes one image per
persona, `deploy up` still only pulls, never builds.

1. Author each persona's project (`osprey build` against its own `profile.yml`,
   committed as that persona's `build_profile`).
2. Push. CI (`.gitlab-ci.yml`) runs one build job per **non-default** persona —
   the default persona keeps using the existing single `build-web-terminal`
   job and its existing un-suffixed `web-terminal:latest` tag, so introducing
   a catalog does not re-tag or re-migrate any existing user. Non-default
   personas push to `web-terminal-<persona>:latest`.
3. `osprey deploy up` on the deploy server pulls each referenced tag — the
   default persona's unsuffixed tag, and each non-default persona's suffixed
   tag — and reconciles the heterogeneous set of per-user containers. No
   build ever runs in this mode.

The image *tag* every ref carries (`web-terminal:<tag>`,
`web-terminal-<persona>:<tag>`) comes from `modules.web_terminals.image_tag`
(default `latest`). Any `${VAR}` in that field is expanded against the
environment **at render time** and baked into the compose file as a literal —
so a rendered artifact self-carries its pin and a pull-free re-`up` re-ups that
exact tag rather than whatever `latest` resolves to locally (there is no
compose-side `${...}` interpolation). This field is registry-mode only; local
mode always builds `:local` images.

### Local-mode operator flow (no CI/registry required)

Local mode is for facilities without a CI/registry pipeline: `deploy up`
builds every referenced persona's image itself, once per run, from that
persona's already-rendered project directory.

1. Render each persona project locally: `osprey build` (once, ahead of time)
   against each persona's project directory, producing the tree
   `personas.<name>.project_path` points at — `deploy up` never renders a
   project itself, only builds an image from what's already there.
2. Set `modules.web_terminals.image_source: local` and
   `default_persona: <name>`; both are required together in local mode (lint
   ERROR, and a plain `ValueError` on the `deploy`/build path itself, since
   `deploy up` doesn't run the lint pass).
3. `osprey deploy up` builds each referenced persona's image from its
   `project_path` (labeled `com.osprey.project=<project>`; the persona's own
   `config.yml` — never the facility config — pins `CLAUDE_CLI_VERSION`),
   tags it `<persona.project>-<persona>:local` — **every** persona, including
   the default, uses this suffixed local tag; only registry mode keeps the
   default persona's tag unsuffixed. `deploy up` runs no `pull` in this mode
   (a `compose pull` against a local-only tag hard-fails, so local mode
   guards it out entirely) and ensures a `.env.production` file exists before
   compose runs: if present it's used as-is; if absent but `.env` exists, a
   CI-variable-subset `.env.production` is generated (mode `0600`, excluding
   the registry token, the dispatcher sidecar token, and any external-project
   tokens); if neither exists, `deploy up` aborts with an actionable message
   rather than starting a stack with no environment file.
4. Two users assigned the same local persona reach healthy on distinct host
   ports the same way registry mode does — the port/env-var machinery in
   "Architecture" above doesn't change with `image_source`.

Facilities never mix the two modes in one deployment — `image_source` is a
single deployment-wide setting, not a per-persona choice.

---

## MCP topology

`modules.web_terminals.mcp.topology` is a schema key for choosing how the
framework's MCP servers are wired across a deployment's containers.
`per_container_stdio` — the default, and the only behavior this phase actually
implements — runs every framework MCP server (`controls`, `phoebus`, `python`,
`osprey_workspace`, `ariel`, `osprey_facility_knowledge`, `scan`,
`channel-finder`: eight servers today) as its own stdio subprocess inside each
per-user container, exactly as it works without this key at all.

`shared_http` — one process per framework server, shared across every
container over HTTP instead of one stdio subprocess per container per server —
is a **recognized but rejected** value in this schema revision: setting it is
a lint ERROR, and rendering it raises a `ValueError` scoped to the shared
framework MCP tier specifically (a facility's own `claude_code.servers` custom
`url` entry is a separate, already-supported path and is unaffected by this
restriction — it lints clean and renders its `{type: "http", url: ...}`
entry under the default topology exactly as it does today).

**Why fail-closed instead of wired**: most framework MCP servers hold
per-user state that a shared process can't safely multiplex across users.
Concretely, only **channel-finder** and **facility-knowledge** were found to
be genuinely shareable without corruption; the rest are not, because of
per-user, environment-derived callbacks (the `http.py` bridge pattern reads
its callback target from the process environment at import time, which is
inherently one value per process, not one per requesting user) and
process-global state (the Artifacts service's in-memory store is a single
global object, not partitioned per user). Sharing just those two servers
saves roughly two subprocess-per-user out of the eight-server total — not
enough to justify building and securing a whole shared-tier (a compose
service, a transport switch in `run_mcp_server()`, and a cross-user
isolation/auth story for that tier) for this phase. The schema key exists so
a future phase can wire a real shared tier as a config value flip instead of
a schema change; until then, `shared_http` is deliberately inert.

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

- One `build-web-terminal` job in the `docker-build` stage (Dockerfile at `docker/Dockerfile.web-terminal`), building the **default persona**'s image (or the single shared image, if no `personas` catalog is configured). The image is multi-stage:
  1. **Node stage** — installs Claude Code (the npm package).
  2. **Python stage** — installs OSPREY + facility profile dependencies into a venv.
  3. **Final stage** — copies the built project from the build-context path in `ARG OSPREY_PROJECT_SRC` (produced by an earlier `osprey build` CI job), then runs the path regen step (see below).
- In registry mode (`image_source: registry`, the default) with a `personas` catalog configured, one additional build job per **non-default** persona, each built from that persona's own `build_profile` and pushed to its own `web-terminal-<persona>:latest` tag, overriding both `OSPREY_PROJECT_SRC` and `OSPREY_PROJECT_NAME` (see "Container path regeneration" below) to point at that persona's own rendered project instead of the default one. Local mode (`image_source: local`) has no per-persona CI job at all — `deploy up` builds those images itself; see "Personas" above.

### `osprey deploy` (up / seed)

Per-user seeding is implemented in Python (`osprey.deployment.web_terminals.seeding`), not a shell script. `osprey deploy up` runs it automatically as the last step of its web-terminal reconcile, right after `compose up -d`; `osprey deploy seed [user]` runs the same logic standalone (one user, or the whole roster) without touching containers otherwise. For every user in `${config.modules.web_terminals.users}` (or just the named one, for `seed <user>`):

1. Skips the user if their container isn't up yet (a `${config.runtime.engine} container exists` check) — logged, not fatal, so the rest of the roster still seeds.
2. **Seeds CLAUDE.md** by streaming the concatenation of `docker/web-terminal-context/base.md` and the user's overlay into the user's `claude-config` volume at `/data/claude-config/CLAUDE.md`. The overlay is `docker/web-terminal-context/<user>/extra.md`; for facilities that haven't migrated off the old flat layout, the legacy `docker/web-terminal-context/<user>.md` is read as a fallback when `<user>/extra.md` is absent. This runs on every `osprey deploy up` (and every `osprey deploy seed`) so per-user memory updates land without rebuilding the image. A user whose resolved persona sets `seed_base: false` (see "Personas" above) is seeded from its `extra.md` **alone** — no `base.md` prepend — so a persona's shipped identity is never silently altered by a base-context change. `base.md` is required only while at least one seeded user keeps `seed_base: true` (the default); if every seeded user opts out, a missing `base.md` is not an error.
3. **Seeds `skills/`** by tar-piping `docker/web-terminal-context/<user>/skills/` into the user's **project-scope** skills directory inside the container — `/app/${config.facility.prefix}-assistant/.claude/skills/` for a user with no persona in effect (or resolved to a persona with `project == "${config.facility.prefix}-assistant"`), or `/app/<persona.project>/.claude/skills/` for a user resolved to a different persona (`<project_cwd>/.claude/skills/`, since the web terminal launches `claude` with `cwd` set to the resolved project directory — see "Personas" above for how that directory is derived) — **not** the user-scope `CLAUDE_CONFIG_DIR/skills/` (`/data/claude-config/skills/`). This is deliberate, and easy to get wrong by intuition: the web terminal launches Claude Code with `--setting-sources project`, and that flag gates skill *discovery* itself, not just `settings.json` loading — skills placed under `CLAUDE_CONFIG_DIR/skills/` are silently never picked up under `--setting-sources project`, while the same skills under the project's `.claude/skills/` are. Only the project-scope path is live; seed there. That directory is also where a running Claude Code session keeps live, user-installed skills (e.g. anything added interactively with `osprey skills install`), so this step never blanket-replaces it: every skill directory the overlay writes gets a `.deploy-managed` sentinel, and only sentinel-bearing directories are ever touched — refreshed on each deploy, removed if the overlay stops shipping them, and left alone if they lack the sentinel. (This `--setting-sources` gating applies only to `skills/`; the CLAUDE.md seeding in step 2 above is plain user-scope auto-discovery, unrelated to `--setting-sources`, and is unaffected.)

Neither `osprey deploy up` nor `osprey deploy seed` HTTP-probes the per-user services after seeding; the post-deploy health check is `scripts/verify.sh` (below) and `osprey deploy status`, run separately.

### scripts/verify.sh

- `nginx_health`: `curl -fsS http://localhost:${config.modules.web_terminals.nginx_port}/`
- Per-user probe (`probe_web_terminal`): not an HTTP health check — execs into the user's container as uid 1000 and confirms it can create and remove a probe file under both `/app/${config.facility.prefix}-assistant/_agent_data/` and `/data/claude-config/`. This catches volume-permission problems but does not individually probe the companion-panel ports.

### .env.template

No new entries are required by this module on its own — web terminals reuse the LLM provider key and any other secrets the assistant already needs. If `modules.event_dispatcher` is also enabled, the sidecar token env var is propagated into each terminal so it can call sidecar endpoints.

### Other files

- `nginx/nginx.conf` — generated; routes `/u/<user>/` to the matching upstream and serves the landing page at `/`.
- `nginx/landing.html` — rendered from `modules.web_terminals.landing.groups` (see Configuration above): a `type: "users"` group auto-populates one card per entry in `users:`, and any `type: "links"` groups render as static link lists (e.g. facility tools). There is no `landing_page_template` field anymore — the landing page is fully data-driven from `landing.groups`.
- `docker/web-terminal-context/base.md` — generated empty stub; intended to hold guidance every user sees.
- `docker/web-terminal-context/<user>/extra.md` — one empty stub per user; intended for per-user nicknames, working preferences, etc. The legacy flat `docker/web-terminal-context/<user>.md` is still read as a fallback when this file doesn't exist (see `osprey deploy` (up / seed) above).
- `docker/web-terminal-context/<user>/skills/` — optional per-user skill overlay, deploy-managed and sentinel-tracked (see `osprey deploy` (up / seed) above); never touches user-authored skills already present in the running container. Seeded to the container's **project-scope** `.claude/skills/`, not `CLAUDE_CONFIG_DIR` — see `osprey deploy` (up / seed) above for why.

---

## Container path regeneration (build-time)

`docker/Dockerfile.web-terminal` takes two build ARGs that parameterize which rendered
project it COPYs in and what that project is called inside the image:

- `ARG OSPREY_PROJECT_SRC=artifacts/${config.facility.prefix}-assistant` — the
  build-context path the Dockerfile `COPY`s from.
- `ARG OSPREY_PROJECT_NAME=${config.facility.prefix}-assistant` — the project name,
  used both for the in-image destination (`/app/${OSPREY_PROJECT_NAME}`) and the
  `osprey claude regen --project /app/${OSPREY_PROJECT_NAME}` step below.

Neither ARG is passed for the default `build-web-terminal` job, so its defaults keep
that build byte-identical to a pre-personas config. Each non-default persona's CI job
(see ".gitlab-ci.yml" above) overrides **both** ARGs together —
`OSPREY_PROJECT_SRC=artifacts/personas/<name>/<project>` and
`OSPREY_PROJECT_NAME=<project>` — to COPY that persona's own rendered project instead.
These are deliberately **not** named `OSPREY_PROJECT_DIR`: that name is already the
*runtime* env var the dispatch sidecar reads for its in-container project directory
(`docker-compose.yml.j2`, `sdk_runner.py`, `dispatch_api.py`) — reusing it here for a
build-context path would risk poisoning that runtime value if the Dockerfile ever
re-exported an ARG as-is.

The CI job that builds the project produces it at a CI-runner path like `/builds/<group>/<project>/${OSPREY_PROJECT_SRC}/`. The Dockerfile then `COPY`s that into the image at `/app/${OSPREY_PROJECT_NAME}/`. Without correction, `config.yml` and `.mcp.json` in the copied tree still reference the CI build paths — every Python MCP server fails because `command:` and `OSPREY_CONFIG=` point at directories that don't exist in the runtime image.

The Dockerfile's final stage fixes this with a post-COPY step:

1. Clears `execution.python_env_path` from `config.yml` so OSPREY falls back to `sys.executable` (i.e., the venv's interpreter).
2. Sets `project_root` to `/app/${OSPREY_PROJECT_NAME}`.
3. Runs `osprey claude regen --project /app/${OSPREY_PROJECT_NAME}` to re-render `.mcp.json` and the `claude/` settings using the new paths.

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

`<port>` is `${config.modules.web_terminals.web_base_port} + <user_index>` — the REST API rides on the `web` family port, the same one the terminal UI uses, not the companion-panel ports. Scripts that don't care which user runs the prompt can pin to user[0]; scripts that want isolation per requester should pick deterministically.

---

## Operating the module

### Add a user

1. Append the new login to `modules.web_terminals.users` in `facility-config.yml` (re-run the interview, or hand-edit and skip ahead).
2. Re-scaffold: regenerates the compose overlay with the new service + named volumes, and adds a row to the nginx upstream block.
3. Create the per-user context stub: `mkdir -p docker/web-terminal-context/<new_user> && touch docker/web-terminal-context/<new_user>/extra.md`. (The legacy flat `docker/web-terminal-context/<new_user>.md` still works, but new facilities should use the directory form so a `skills/` overlay can be added later without restructuring.)
4. Commit and push. CI rebuilds nothing image-side (the user list isn't baked into the image), but the new container only appears after a deploy.
5. Deploy: `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && osprey deploy up"`. The seeding step creates the user's CLAUDE.md on first run.

Existing users are unaffected — their named volumes are untouched.

### Remove a user

The one-command way, on the deploy server: `osprey deploy decommission <user>` (add `--archive` to tar the workspace first, or `--purge` to skip retention and delete the volumes outright). It edits the roster in `config.yml`, re-renders the web-terminal artifacts, and force-removes the user's container — all in one step, gated on a typed confirmation (or `--yes`) since the volume disposition is not reversible once purged. By default (no `--archive`/`--purge`) the two named volumes (`<user>-claude-config`, `<user>-agent-data`) are **retained**.

The equivalent by hand, if you're editing `facility-config.yml` directly instead of going through `decommission`:

1. Remove the entry from `modules.web_terminals.users`.
2. Re-scaffold + commit + push + `osprey deploy up`. The user's container is stopped and removed.
3. Their named volumes are **kept** unless explicitly deleted:
   ```bash
   ssh ${config.deploy.host} "${config.runtime.engine} volume rm \
     ${config.facility.prefix}-<user>-claude-config \
     ${config.facility.prefix}-<user>-agent-data"
   ```

To clean up users that were removed from the roster this second way (bypassing `decommission`) and left as orphans, run `osprey deploy prune [--dry-run] [--archive|--purge]` — it discovers containers/volumes not on the current roster and removes them the same way `decommission` would.

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
| User's CLAUDE.md is empty | The seeding step (`osprey deploy up`/`seed`) couldn't reach the named volume — verify the volume exists and is mounted |
| Claude session "forgets" between visits | Named volume was destroyed — `osprey deploy nuke`, `decommission --purge`/`prune --purge`, or someone manually `volume rm`'d it |
| All web terminals see the same memory | Containers are sharing a volume — check the per-user `volumes:` block in the compose overlay rendered correctly (no anchor reuse bug) |
| nginx serves landing page but reverse-proxy to user URLs returns 502 | Per-user service's env-var-overridden listen port doesn't match nginx's upstream for that family — check `OSPREY_WEB_PORT` and the companion `OSPREY_*_PORT` vars inside the container against the `*_base_port + index` nginx expects. (The `web` family's unconfigured default is `8087` — a service still listening on that default instead of its assigned host port is the usual cause under `network_mode: host`.) |

### Updating per-user CLAUDE.md and skills without redeploying images

The CLAUDE.md and `skills/` seeding runs in `osprey deploy` (the `up`/`seed` verbs), not in the image build, so:

1. Edit `docker/web-terminal-context/base.md` and/or the user's `docker/web-terminal-context/<user>/extra.md` (or the legacy flat `<user>.md`); add or edit files under `docker/web-terminal-context/<user>/skills/<skill-name>/` for a per-user skill.
2. Commit + push (so the change reaches the deploy server's git checkout).
3. `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && git pull && osprey deploy up"` (or `osprey deploy seed <user>` to reseed just that user without touching containers) — git pull picks up the new files, and the seed step copies them in. Containers stay up; the files just change under them.

The user must restart their Claude session in the browser to pick up the new CLAUDE.md or skills (Claude Code reads them on session start).

---

## Disabling

1. Set `modules.web_terminals.enabled: false` in `facility-config.yml`.
2. Re-scaffold — the compose overlay file is removed from `${config.runtime.compose_files}` and (optionally) deleted; the web-terminal CI job is removed.
3. `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && osprey deploy down && osprey deploy up"` to stop and remove the per-user containers and the nginx container.
4. Named volumes survive disable. To wipe them too, run `${config.runtime.engine} volume rm` per user (script the loop over the old user list).
5. If `modules.benchmarks` was enabled, it must also be disabled (benchmarks run inside a web-terminal container; the interview enforces this).
