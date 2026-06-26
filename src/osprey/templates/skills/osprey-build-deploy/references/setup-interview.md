# Setup Interview — Building `facility-config.yml`

This is the script the skill executes when `facility-config.yml` is missing, or when the user explicitly asks to re-run / update the deploy interview. Its sole output is `facility-config.yml` at the repo root, plus a `.env.template` that lists every secret env var the operator will need to fill in.

**Read `references/facility-config-schema.md` before starting.** Every question here maps to a field in that schema, and you must produce config that conforms to it.

## Operating principles

1. **Conversational, not interrogative.** This may be the operator's first contact with this skill. Explain *why* each question matters in one short sentence so they can answer with intent. Group related questions into a single `AskUserQuestion` call (up to 4 per call) so it feels like a conversation, not a form.

2. **Defaults that work.** For nearly every question, suggest a sensible default. The default ports below are what als-profiles uses and known to work; the operator can override if they conflict with something local.

3. **"I'm not sure" is always allowed.** If the user is uncertain, pick the safe/simple default and tell them. Note the assumption in the README the skill produces, so they can revisit later.

4. **Progressive saves.** After each phase, write what you have to `facility-config.yml`. If the interview is interrupted, the next invocation should resume from where it stopped (read the existing file, identify which phases are incomplete, ask only the missing questions).

5. **Re-entry is normal.** If `facility-config.yml` already exists when the user says "I want to re-run the interview" or "I want to enable the OLOG module", load the existing config and ask only the questions that are missing or that the user wants to revise. Never overwrite a value silently.

6. **Secrets stay in `.env`, not `facility-config.yml`.** When the user provides a secret (API key, deploy token, password), store the env var *name* in `facility-config.yml` and add the env var to `.env.template` with a placeholder. The operator fills `.env` later.

7. **Validate before writing.** Run the validation checklist from `facility-config-schema.md` § "Validation rules" before saving. If anything fails, show the user and offer a fix.

---

## Phase 0 — Welcome and context

If `facility-config.yml` does not exist:

> "I'm going to ask you some questions about your facility's deployment infrastructure. The answers go into `facility-config.yml` and drive everything: CI/CD pipeline, container deploys, MCP server ports, optional features like OLOG and webhook triggers. Most facilities take 10–15 minutes. You can say 'I'm not sure' to any question and I'll pick a sensible default. Ready?"

If `facility-config.yml` already exists, this is a re-entry:

> "I see you already have a `facility-config.yml`. Are we re-running the whole interview, updating a few specific things, or enabling a new module?"

Use `AskUserQuestion`:
- "Re-run everything (start from scratch, ask all questions)"
- "Update specific values (you tell me what to change)"
- "Enable a new opt-in module (OLOG, web terminals, event dispatcher, etc.)"
- "Just review what's there"

Branch accordingly. For "enable a module", skip directly to the relevant module block in Phase 4.

---

## Phase 1 — Facility identity and control system

Single `AskUserQuestion` call with three questions:

**Q1.1 Facility name** (text input via "Other"; suggest the user knows their own facility):
> "What's the full name of your facility? Example: 'Advanced Light Source', 'Diamond Light Source', 'European XFEL'."

**Q1.2 Facility prefix:**
> "Pick a short slug (2–6 lowercase characters, no spaces) that becomes part of generated filenames and container names. For ALS this is `als`; you'd see `als-prod.yml` and `als-mcp-matlab` everywhere. What should yours be?"

Validate: lowercase, alphanumeric + hyphens only, 2–6 chars. If invalid, explain why and ask again.

**Q1.3 Control system type:**
- EPICS (most US/Asian accelerators — APS, ALS, SLAC, NSLS-II, J-PARC, KEK, etc.)
- Mock / simulated (for development without real hardware) (Recommended for first deploy)
- DOOCS (DESY, European XFEL) — ROADMAP, no connector yet
- TANGO (ESRF, MAX IV, Soleil, Elettra) — ROADMAP, no connector yet
- Custom (you write your own connector) — no built-in support

> "Which control system do you connect to? OSPREY ships built-in support for **EPICS** and **Mock** today. DOOCS and TANGO are roadmap values — picking one writes the config but there is no working connector yet, so live control-system access won't work. If your site runs DOOCS/TANGO/Custom, pick `mock` for now so you can exercise the rest of the deploy pipeline; switch to the real value once a connector lands."

**Q1.4 Timezone** (in a follow-up question or the same group if room):
> "What IANA timezone should logs and schedules use? Default: `UTC`. ALS uses `America/Los_Angeles`."

**EPICS-only follow-up.** If `control_system.type == "epics"`:
> "Do you have an EPICS Channel Access broadcast address list (`EPICS_CA_ADDR_LIST`)? This is space-separated IPs/hostnames for CA discovery on your control network."
>
> Also: "Do you have an EPICS Archiver Appliance? If yes, what's its REST URL? (e.g., `http://arch-ml.als.lbl.gov:17668`). This is optional — leave blank if you don't have one yet."

Save what you have to `facility-config.yml`.

---

## Phase 2 — GitLab and container registry

This is the source-control + CI side. The user must already have a GitLab project created (or know they will create one).

> "Now let's talk about your GitLab setup. You should already have a project created (or be ready to create one) — that's where this repo gets pushed and where CI builds the container images."

Single `AskUserQuestion` call:

**Q2.1 GitLab host:**
> "What's your GitLab server hostname? (no `https://`, no path — just the hostname). Examples: `gitlab.com`, `git.als.lbl.gov`, `gitlab.desy.de`."

**Q2.2 Project path:**
> "What's the project path (the part after the host)? Format: `group/subgroup/project`. ALS uses `physics/production/als-profiles`."

**Q2.3 Project ID:**
> "What's the numeric project ID? Find it in GitLab at Settings → General → Project ID. It's a number like 951."

**Q2.4 Default branch:**
> "Which branch should CI watch and the release job run from? Default: `main`. Some facilities still use `master`."

Follow-up: ask for the **GitLab remote name** (the local `git remote` name pointing to GitLab; default `origin`, but ALS uses `gitlab` because `origin` is a GitHub mirror).

**Q2.5 Token env var name (NOT the value):**
> "What env var holds your GitLab Personal Access Token? Default: `${PREFIX}_GITLAB_TOKEN` (e.g., `ALS_GITLAB_TOKEN`). Required scopes: `api`, `read_registry`, plus `write_registry` if CI pushes images. The value goes in `.env`, not here."

Add `${TOKEN_ENV_VAR}=<paste your PAT here>` to `.env.template` with a comment.

**Q2.6 Container registry URL:**
> "Where do CI-built images go? For GitLab projects this is usually `${gitlab.host}:5050/${gitlab.project_path}`. Confirm or override."

Default: compute from the answers above and ask the user to confirm.

**Q2.7 External projects** (optional):
> "Do you depend on container images from other GitLab projects (e.g., a sibling team's service that gets pulled at deploy time)? If yes, list each — I'll need the project path, image name, and the env var name for its deploy token."

If yes, loop through each external project and gather details. If no, skip.

Save to `facility-config.yml`.

---

## Phase 3 — Deploy server, runtime, network

> "Now let's set up the deploy target — the server where containers actually run."

**Q3.1 Deploy server hostname:**
> "What's the SSH-resolvable hostname of your deploy server? You should be able to `ssh ${host}` and have it work (typically because it's in your `~/.ssh/config`). ALS uses `appsdev2`."

**Q3.2 Deploy server FQDN:**
> "What's the full DNS name developers' laptops use to reach this server? This is what client-mode profiles use to connect to remote MCP services. ALS uses `appsdev2.als.lbl.gov`."

**Q3.3 Deploy SSH user:**
> "What user runs deploys on the server? Often the same as your local user, but sometimes a service account. ALS uses `thellert`."

**Q3.4 Project path on server:**
> "Where on the server should this repo live? Default: `/home/${user}/projects/${facility.prefix}-profiles`."

Single `AskUserQuestion` call for runtime:

**Q3.5 Container engine:**
- podman (Recommended for rootless deploys; ALS uses this)
- docker

**Q3.6 Compose command:**
- podman-compose
- docker compose (newer Docker, built-in)
- docker-compose (older Docker, standalone binary)

> "Which compose command does your server use? Run `which podman-compose` or `which docker-compose` on the server to check."

**Q3.7 Compose file layout:**
> "Will you use multiple compose files (base + overrides)? Default: `[docker-compose.yml, docker-compose.host.yml]`. The host overlay is where EPICS host-networking goes."

Follow-up phase for network:

**Q3.8 Proxy:**
> "Does your facility require an HTTP proxy for outbound connections? (Most national labs do.) If yes, give me the URL — typically `http://<proxy-host>:3128` for Squid. Leave blank if no proxy."

If yes:

**Q3.9 NO_PROXY entries:**
> "What hosts should bypass the proxy? Defaults: `localhost, 127.0.0.1, host.containers.internal, host.docker.internal`. Add your internal hostnames (anything ending in `.facility.org`) and any internal services. ALS adds: `*.als.lbl.gov`, the archiver host, the Confluence wiki, the langfuse observability stack."

Save what you have. Validate the deploy server section — if SSH fails (`ssh -o BatchMode=yes -o ConnectTimeout=5 ${host} true`), warn but don't fail.

---

## Phase 4 — LLM provider and ports

**Q4.1 LLM provider:**
- CBORG (LBNL gateway — most LBL projects)
- Anthropic (direct API)
- OpenAI
- Google (Gemini)
- Ollama (self-hosted)
- AskSage
- vLLM (self-hosted)
- Argo (ANL gateway)
- ALS-APG (ALS-specific)
- Other

> "Which LLM provider does your assistant use? OSPREY ships providers for many gateways and direct APIs. If you're at LBL, CBORG is probably the right answer. If you're not sure, ask your IT team what AI access you have."

Follow-up:

**Q4.2 API key env var name:**
> "What env var holds your provider's API key? Default: `${PROVIDER_UPPER}_API_KEY` (e.g., `CBORG_API_KEY`, `ANTHROPIC_API_KEY`). Add the value to `.env` later."

**Q4.3 Default model:**
> "What's the default model id? Examples: `anthropic/claude-sonnet-4-20250514`, `gpt-4o`, `gemini-2.5-pro`, `cborg/anthropic/claude-sonnet`. You can override per-agent in profile YAMLs."

**Q4.4 Base URL** (only for self-hosted providers):
> "What's the API base URL? (Only needed for Ollama, vLLM, or custom endpoints.)"

Add `${API_KEY_ENV_VAR}=<paste your key here>` to `.env.template`.

**Q4.5 MCP server ports:**

> "Now let's allocate ports. The defaults below are what ALS uses; pick something that doesn't conflict with what's already running on your deploy server."

Show the user the proposed defaults as a single block and ask if they're OK or what to change:

```
Core MCP servers:
  matlab:                  8001
  accelpapers:             8002
  phoebus:                 8003
  integration_tests:       8004
  direct_channel_finder:   8005
```

If they want to change anything, ask which port and what to change it to. If they don't have a particular MCP server, drop it from the config (the interview will revisit MCP servers in Phase 5 when we ask about custom_mcp_servers).

---

## Phase 5 — Optional modules (the heart of the interview)

This is where the user picks which of the opt-in features they want. **Do not enable modules by default.** Each module is a substantial piece of infrastructure; only enable what the user explicitly opts into.

> "OSPREY supports a number of optional features beyond the basic deploy. I'll go through them one by one — say yes only if you actually want it, you can always enable more later by re-running this interview."

For each module below, ask "Do you want this?" first. If yes, ask the module-specific follow-up questions. If no, mark `enabled: false` in the config and move on. Use `multiSelect: true` for the high-level "which do you want?" if you're listing several at once, then drill into each yes.

### 5.1 Web terminals (multi-user)

> "Do you want multiple per-user web-terminal containers behind nginx, so several team members can each have their own browser-based Claude Code session against the assistant?"

If yes:
- Which users? (list of usernames; one container per user, named `${facility.prefix}-web-${user}`)
- Base port for the per-user containers (default: `9091`; consecutive ports get assigned)
- Nginx public-facing port (default: `9080`)
- Landing page: default template, or do you have a custom `nginx/landing.html`?

### 5.2 Event dispatcher (webhook + EPICS-CA → headless agents)

> "Do you want webhook-driven or (for EPICS facilities) control-system-driven automatic agent runs? For an accelerator this might be 'when beam current drops below X, run a diagnostic agent and email a report'; for a beamline or detector it might be 'when a scan finishes, run the reduction agent and post to the logbook.' This adds an event-dispatcher container plus N sidecar containers for parallel dispatch."

If yes:
- Dispatcher port (default: `8010`)
- Sidecar count (default: same as the number of web-terminal users, or 5 if web terminals disabled)
- Sidecar port base (default: `9190`; consecutive)
- Token env var name for the dispatcher (default: `EVENT_DISPATCHER_TOKEN`)
- Sidecar token env var name (default: `DISPATCH_SIDECAR_TOKEN`)
- If `control_system.type == "epics"`: enable EPICS-CA triggers? (uses CA broadcast list from earlier)

**Validate before saving:** the sidecar port range `[sidecar_port_base, sidecar_port_base + sidecar_count - 1]` MUST NOT overlap the web-terminal range `[web_terminals.base_port, base_port + len(users) - 1]`. If it does, reject the config and ask the user to pick a non-overlapping base. The default pairing (9091+ and 9190+) is safe up to 99 users.

Add both token vars to `.env.template`.

### 5.3 OLOG / electronic logbook

> "Does your facility have an electronic logbook with an HTTP API that you want the assistant to read or write to?"

If yes:
- Logbook type? (Phoebus OLOG, Elog, OLOG-RPC, custom)
- Production API URL
- Test/staging API URL (optional)
- Auth method: basic | bearer | api_key
- Username env var name (if basic)
- Password/token env var name
- Allow integration tests to write to the logbook? (default: `false`; only enable for non-prod logbooks)

### 5.4 ARIEL database (Postgres-backed semantic search)

> "ARIEL is a searchable database with vector embeddings — typically used to give Claude access to historical logbook entries with semantic similarity. Do you want this?"

If yes:
- Deployment: container (Postgres in compose) or external (existing DB)
- DSN (default: `postgresql://ariel:ariel@ariel-postgres:5432/ariel` for container — hostname matches the compose service key)
- Sync source (typically `olog` if you enabled the OLOG module)
- Embeddings provider: ollama (recommended if you'll enable Ollama), openai, cborg, ...

Note: requires either the OLOG module or some other source adapter to be useful.

### 5.5 Ollama (local embedding / inference server)

> "Do you have an Ollama server (or want to deploy one) for local embeddings or inference? Often paired with ARIEL."

If yes:
- Ollama URL (default: `http://${ollama-host}:11434`)
- Embedding model (default: `nomic-embed-text`)
- Chat model (optional, only if you want local LLM calls)

### 5.6 Wiki search (Confluence or similar)

> "Do you want the assistant to search a facility wiki — Confluence, MediaWiki, or similar?"

If yes:
- Wiki type
- Base URL
- API path (default: `/rest/api/` for Confluence)
- Auth method: bearer | basic
- Token/password env var name
- Restrict to specific spaces? (list)

### 5.7 Shared disk (NFS / bind-mount)

> "Do you have a shared filesystem on the deploy server (NFS mount, physics code repository, etc.) that some MCP servers need read access to?"

If yes:
- Host path on the deploy server
- Container path (where it appears inside containers that mount it)
- Mount mode: ro (read-only, recommended) or rw
- Which compose services should mount it? (list)

### 5.8 Custom MCP servers

> "Do you have facility-specific MCP servers (matlab, accelpapers-equivalent, custom analysis tools)? Each gets its own Dockerfile, port, and CI build job."

If yes, loop through each:
- Name (becomes container name suffix and port lookup key)
- Port (must match a `ports.${name}` entry from Phase 4)
- Dockerfile path (relative to repo root)
- Build context (default: `.`)
- Build artifacts (paths copied into the image at build time, e.g., `artifacts/mml.db`)
- depends_on (other compose services this one needs)

### 5.9 E2E benchmarks

> "Do you want a benchmark suite that runs realistic multi-step queries against the deployed assistant and judges the results with an LLM-judge?"

Requires `web_terminals` enabled (the suite runs inside a web terminal container). If web_terminals is disabled, warn the user and skip — or offer to enable web_terminals first.

If yes:
- Benchmark suite path (default: `data/benchmarks/e2e_workflow_benchmarks.json`)
- Container to exec into (default: `${facility.prefix}-web-${first_user}`)
- Project dir inside container (default: `/app/${facility.prefix}-assistant/`)
- Judge model (default: same as `llm.model`)

### 5.10 EPICS test IOC

> "Do you want the skill to manage an EPICS test IOC (start, stop, configure)? This is for safe development against simulated PVs without touching production EPICS."

**Only show this question if `control_system.type == "epics"`.** Otherwise skip silently.

If yes:
- Read `references/modules/test-ioc-safety.md` aloud (the key rules) and confirm the user understands
- CAS server port (default: `59064` — exotic, far from production CA defaults)
- CAS beacon port (default: `59065`)
- Test PV prefix (default: `OSPREY:TEST:`)
- DB file path (default: `ioc/test.db`)
- Startup script path (default: `/tmp/start-test-ioc.sh`)

Validate: ports must be outside 5064–5076 range. If user picks something inside, refuse and explain.

---

## Phase 6 — Validation and write

Before writing `facility-config.yml`, run the validation checklist from `references/facility-config-schema.md` § "Validation rules". For each failure:
- **Hard failures** (e.g., `prefix` invalid, `cas_server_port` in production range) → ask the user to fix.
- **Soft warnings** (e.g., SSH to deploy server fails, `localhost` missing from no_proxy) → list them and ask the user to confirm anyway.

Write the file with a header comment block:

```yaml
# facility-config.yml
# Generated by osprey-build-deploy on YYYY-MM-DD by user <username>
#
# This is the durable contract between the deploy skill and this repo.
# Every generated file (docker-compose.yml, .gitlab-ci.yml, scripts/deploy.sh,
# .env.template) is parameterized by these values.
#
# Re-run the deploy interview to update; never edit blindly.
# Secrets do NOT live here — they go in .env (gitignored).

schema_version: 1

facility:
  ...
```

Also write `.env.template` listing every secret env var collected during the interview, with placeholders and comments documenting where each goes:

```bash
# .env.template
# Copy this to .env and fill in real values.
# Both this file and the populated .env should be gitignored except .env.template itself.

# GitLab Personal Access Token — scopes: api, read_registry, write_registry
ALS_GITLAB_TOKEN=

# LLM provider key
CBORG_API_KEY=

# Event dispatcher webhook authentication
EVENT_DISPATCHER_TOKEN=

# ... etc
```

---

## Phase 7 — Next-step guidance

Tell the user what to do next:

> "✓ `facility-config.yml` written.
> ✓ `.env.template` written — copy it to `.env` and fill in the secrets.
>
> Next steps:
>
> 1. **Fill in `.env`** with your actual secrets (it's gitignored, so it never leaves your machine).
> 2. **Author your build profile**: run `/osprey-build-interview` to walk through creating a `profile.yml`. That's a separate skill — different concerns.
> 3. **Scaffold deploy infrastructure**: tell me 'scaffold the deploy infra' and I'll generate `docker-compose.yml`, `.gitlab-ci.yml`, `scripts/deploy.sh`, and `scripts/verify.sh` from the templates, parameterized by your config.
> 4. **Review the generated files** — they're plain text, fully editable. The skill regenerates them from `facility-config.yml` whenever you change values.
> 5. **First push and deploy**: `git push`, watch CI, trigger the release job, run `deploy.sh` on the server."

If you don't see them mention scaffolding within the same conversation, proactively offer it.

---

## Re-entry: enabling a module after the fact

When the user says "I want to enable OLOG" (or web terminals, or the event dispatcher, etc.):

1. Read the existing `facility-config.yml`.
2. Confirm the module is currently disabled.
3. Run only that module's question block from Phase 5.
4. Validate.
5. Save the updated config.
6. **Important**: tell the user that scaffolded files (compose, CI, scripts) need to be re-rendered to include the new module. Offer to do that immediately.
7. The user will need to add any new secret env vars to `.env`.
