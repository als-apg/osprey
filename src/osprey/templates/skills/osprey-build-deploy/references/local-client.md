# Local Client Build Reference

A "client" build produces a Claude Code project on a developer's laptop that talks to **remotely deployed MCP services** on `${config.deploy.host}` instead of running its own containers. Useful for developers who need the full assistant without standing up the whole stack locally.

This file documents the build + run workflow. It assumes the `${config.facility.prefix}-client.yml` profile already exists in the repo. Authoring or modifying that profile is the job of `/osprey-build-interview`, not this skill.

---

## When to use a client build

Use the client profile when:

- A developer wants to use the assistant against the real shared MCP services (matlab index, accelpapers, phoebus, etc.) but doesn't need to run those services locally.
- You're iterating on prompts, agent definitions, or overlays and want fast `osprey build` cycles without container rebuilds.
- The developer is on a laptop where running the full container stack is impractical (Mac, low-RAM machine, no GPU).

Don't use the client profile when:

- You're testing changes to a custom MCP server's source code (the client connects to the *deployed* version of that server, not your local edits).
- You need a fully isolated stack (use the prod profile + a local docker stack instead).
- The developer's machine is off the facility's control network (the deploy server is typically not internet-reachable).

---

## Prerequisites on the developer machine

| Requirement | Notes |
|-------------|-------|
| `pip install osprey-framework` | Install the OSPREY CLI. Use a fresh virtualenv if you want isolation; OSPREY tolerates global installs too. |
| Claude Code CLI installed | `npm install -g @anthropic-ai/claude-code` (or follow Anthropic's current install instructions). The built project is run via `claude` inside its directory. |
| `.env.local` in the repo | Holds the LLM provider key. At minimum: `${config.llm.api_key_env_var}=<your-key>`. Gitignored — never check this in. |
| Network access to `${config.deploy.fqdn}` | Typically requires being on the facility's control network, via VPN or on-site. Test with `curl -v http://${config.deploy.fqdn}:${config.ports.integration_tests}/health` before building. |
| Local container runtime — *only if* the client profile uses any containerized component | The standard client profile does NOT run local containers. If your facility's `${config.facility.prefix}-client.yml` adds local containers (rare), Docker Desktop (Mac/Windows) or Docker Engine (Linux) is sufficient — podman is not required for client builds. |

The developer does *not* need a deploy token, a registry login, or anything that touches `${config.gitlab.host}`. Client builds are read-only against the deploy server.

---

## The build command

Run from the root of the facility profile repo (where `${config.facility.prefix}-client.yml` lives):

```bash
osprey build ${config.facility.prefix}-client \
  $(pwd)/${config.facility.prefix}-client.yml \
  -o ~/projects \
  --force
```

What each piece does:

| Argument | Meaning |
|----------|---------|
| `${config.facility.prefix}-client` | Output project name. Becomes the directory name under `-o`. |
| `$(pwd)/${config.facility.prefix}-client.yml` | Absolute path to the client profile YAML. OSPREY needs absolute paths so overlay sources resolve correctly. |
| `-o ~/projects` | Output parent directory. The built project lands at `~/projects/${config.facility.prefix}-client/`. |
| `--force` | Wipe and rebuild if the output directory exists. Safe for client builds because the client project never holds user state — just generated config and a venv. |

The build:
1. Creates `~/projects/${config.facility.prefix}-client/`.
2. Renders the client profile's `config.yml` with MCP server URLs pointing at `${config.deploy.fqdn}`.
3. Creates a Python virtualenv at `~/projects/${config.facility.prefix}-client/.venv/` and installs the profile's declared dependencies.
4. Writes `.mcp.json` (consumed by Claude Code) with the remote HTTP transport URLs.
5. Copies overlays (rules, agents, skills) into `.claude/`.
6. Runs lifecycle hooks if any (most client profiles skip these).

Then run the assistant:
```bash
cd ~/projects/${config.facility.prefix}-client && claude
```

Claude Code reads `.mcp.json`, opens HTTP connections to each remote MCP server, and you're up.

---

## How `${config.facility.prefix}-client.yml` differs from `${config.facility.prefix}-prod.yml`

The two profiles share most overlay/agent/rule definitions (often via `extends:`). The differences are:

| Concern | `*-prod.yml` (deploy server) | `*-client.yml` (developer laptop) |
|---------|------------------------------|------------------------------------|
| MCP server URLs | `http://localhost:<port>/mcp` (or Docker DNS service names like `http://<service>:<port>/mcp` from inside containers) | `http://${config.deploy.fqdn}:<port>/mcp` |
| `env.file` | `.env.production` (server-side, has all secrets) | `.env.local` (developer-side, has just the LLM key) |
| `env.required` | Long list (every service credential) | Short list (`${config.llm.api_key_env_var}` and any client-specific key) |
| `container_runtime` | `${config.runtime.engine}` (whatever the server uses) | typically `docker` (most developer laptops) |
| `overlay:` extras | Server-side test infrastructure, soft-IOC management skills, etc. | Stripped down — no test IOC, no on-server admin skills |
| Stdio MCP servers (e.g., wiki search via uvx) | Often present (run inside the web-terminal container) | Often absent (developer probably doesn't have facility wiki credentials) |
| Database connections (when a module owns one) | `localhost:<port>` (container on the deploy server) | `${config.deploy.fqdn}:<port>` if reachable, else disabled — often fails gracefully |

The client profile is designed to **fail gracefully** when a remote service isn't reachable. Individual MCP servers that can't connect get marked unavailable; the rest still work. Don't treat a single connection failure as a build problem — check `claude --help` shows the available servers and only worry about ones you actually need.

---

## Network requirements

The client connects to `${config.deploy.fqdn}` over plain HTTP on the configured ports. Typical setup:

- **On-site / on-VPN**: works directly. The control network sees the deploy server.
- **Off-site without VPN**: doesn't work. Either get on the VPN, or use SSH port forwarding as a workaround:
  ```bash
  ssh -L ${config.ports.matlab}:localhost:${config.ports.matlab} \
      -L ${config.ports.accelpapers}:localhost:${config.ports.accelpapers} \
      -L ${config.ports.integration_tests}:localhost:${config.ports.integration_tests} \
      ${config.deploy.host}
  ```
  Then edit your built project's `.mcp.json` to point at `http://localhost:<port>/mcp` instead of `http://${config.deploy.fqdn}:<port>/mcp`. This is a band-aid — for routine off-site use, ask IT for VPN access.

Test connectivity before building:
```bash
curl -v http://${config.deploy.fqdn}:${config.ports.integration_tests}/health
```

If that fails (DNS, firewall, server down), the client build will succeed but the assistant won't be useful.

---

## Local container runtime

Most client profiles do not run local containers — every MCP server is HTTP-remote. So no container runtime is strictly required.

If a client profile *does* include local containers (uncommon — usually only if a developer needs a sandbox MCP server that the deploy server doesn't host), the runtime should be Docker:

- **macOS**: Docker Desktop (`brew install --cask docker`). The compose plugin ships with it.
- **Linux**: Docker Engine + compose plugin.
- **Windows**: Docker Desktop (WSL2 backend).

Podman on a developer laptop works but isn't the path of least resistance — most lifecycle hooks and overlays assume Docker syntax for any local stack.

Set `container_runtime: docker` in the client profile YAML; it's separate from `${config.runtime.engine}` (which describes the deploy server's runtime, not the client's).

---

## Re-building after profile or overlay changes

When you edit the profile or any overlay file in the repo:

```bash
osprey build ${config.facility.prefix}-client \
  $(pwd)/${config.facility.prefix}-client.yml \
  -o ~/projects \
  --force
```

`--force` wipes the previous build. Any state the client project accumulates (Claude Code session history, dropped credentials) is in `~/.claude/` (Claude Code's user state), not in the project — it survives rebuilds. Only the project's `.venv` and `.mcp.json` and overlays get regenerated.

Rebuild whenever:
- The client profile YAML changes.
- Any agent, skill, or rule under `overlays/` changes.
- The MCP server URLs change (e.g., the deploy server's FQDN is different).
- A new dependency is added to the profile.

You don't need to rebuild when:
- A remote MCP server's *implementation* changes (it's deployed centrally; the client just talks to whatever's there).
- You change `.env.local` (read at runtime by Claude Code, no rebuild needed).
- You change `~/.claude/` user-level settings.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `osprey: command not found` | OSPREY not installed in the active Python env | `pip install osprey-framework`, or activate the right venv |
| `osprey build` fails with `Profile file not found` | Path passed isn't absolute | Use `$(pwd)/${config.facility.prefix}-client.yml`, not a relative path |
| `osprey build` fails with `Required env var X not set` | `.env.local` missing or doesn't have the var | Create `~/projects/<repo>/.env.local`, add `X=<value>` (the project file, not `~/.env.local`) |
| `claude` starts but no MCP servers listed | `.mcp.json` missing or malformed | Re-run `osprey build` with `--force`; check that the rendered `.mcp.json` has entries |
| MCP server shows "connection failed" in Claude | Either wrong URL, server down, or network can't reach it | `curl -v http://${config.deploy.fqdn}:<port>/mcp` from the laptop. If that fails: VPN, firewall, or server issue. Verify with the deploy operator that the server is healthy. |
| Some MCP servers work, others don't | Selective firewall or service-specific outage | Check `${config.deploy.host}` for the failing service: `ssh ${config.deploy.host} "${config.runtime.compose_command} ps <service>"` |
| `claude` reports "API key invalid" | Wrong or missing `${config.llm.api_key_env_var}` in `.env.local` | Verify the key matches what `${config.llm.provider}` expects. Test the key directly with the provider's quick-start example before blaming Claude Code. |
| Builds succeed but the assistant ignores a new agent/rule you added | Overlay file path wrong, or `--force` not used | Verify the overlay's source path under `overlays/` matches what the profile references; rebuild with `--force` |
| Connection works but is slow | Proxy interception, or congested control network | Try without `HTTP_PROXY`/`HTTPS_PROXY` set on the laptop (these aren't usually needed for direct connections to internal services). For VPN slowness, ask IT. |
| `pip install osprey-framework` fails behind a corporate proxy | Proxy not configured for pip | `pip install --proxy=<your-proxy> osprey-framework`, or set `HTTPS_PROXY` in the shell |
| Claude Code can't find `claude` after `npm install -g` | npm global bin not in PATH | Add the npm global bin to PATH (`npm config get prefix` shows the directory; append `/bin`) |

---

## Out of scope for this skill

This file documents the build + run workflow once `${config.facility.prefix}-client.yml` already exists. It does NOT cover:

- **Authoring or modifying `${config.facility.prefix}-client.yml`** — that's `/osprey-build-interview`'s job. If the client profile needs new MCP server entries, agent definitions, env vars, or overlay paths, hand the user off to `/osprey-build-interview` and stop.
- **Running the deploy server's MCP services locally** — use the prod profile and a local container stack instead, or ask the deploy operator for shell access on the server.
- **Setting up VPN or network access** — IT problem; this skill assumes the laptop can reach `${config.deploy.fqdn}`.

If a developer's "client build doesn't work" problem turns out to be a profile gap (missing MCP server, wrong overlay), refer them to `/osprey-build-interview` to update the profile, then come back here for the build mechanics.
