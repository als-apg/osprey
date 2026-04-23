# Deploy Server Reference

Operational details for the server that runs the OSPREY assistant containers. Every concrete value (hostname, paths, ports, proxy URLs, container engine) comes from `facility-config.yml`. This file documents the *shape* of the on-server setup; the *values* live in config.

The deploy server's role is narrow: pull the CI-built images, run them via compose, hold the secrets. It never builds anything.

---

## Directory layout (sibling pattern)

```
${config.deploy.project_path}/                    ← the facility profile repo (git clone)
   ├── ${config.facility.prefix}-prod.yml
   ├── ${config.facility.prefix}-client.yml
   ├── facility-config.yml
   ├── docker-compose.yml                          ← scaffolded
   ├── docker-compose.host.yml                     ← scaffolded (if any)
   ├── scripts/deploy.sh                           ← scaffolded
   ├── scripts/verify.sh                           ← scaffolded
   ├── .env.template                               ← scaffolded; commit this
   └── .env.production                             ← operator-managed; gitignored

${config.deploy.project_path}/../osprey/           ← (optional) OSPREY framework checkout
${config.deploy.project_path}/../${config.facility.prefix}-assistant/   ← (built only if running osprey on-server, NOT the normal flow)
```

OSPREY, the profile repo, and any built-assistant outputs are siblings — never nested. The CI-native deploy doesn't usually need OSPREY on-server at all (the web-terminal image already has it baked in); keep an OSPREY checkout there only if an operator needs to do off-pipeline `osprey build` runs for debugging.

---

## Prerequisites

These must be present on the server before the first deploy. The interview asks IT to provision anything missing.

| Component | Where used | Notes |
|-----------|------------|-------|
| `${config.runtime.engine}` (podman or docker) | Container runtime | Rootless podman is the most-tested path; docker requires the user to be in the `docker` group |
| `${config.runtime.compose_command}` (e.g., `podman-compose`, `docker compose`) | Orchestration | Must understand all overlay files in `${config.runtime.compose_files}` |
| Python 3.12+ | Optional — only for off-pipeline OSPREY runs | Most facilities don't need a system Python on the deploy server |
| `git` | Pulling the profile repo and `${config.gitlab.default_branch}` | |
| `ssh` access | Operator workflow | Operator's laptop must be able to `ssh ${config.deploy.host}` (configure in `~/.ssh/config`) |
| EPICS base | If `control_system.type == "epics"` AND any host-network service binds to CA broadcast | Required for `caget`/`caput` debugging from the host shell. Container-internal EPICS use does NOT require host EPICS. |
| `uv` (optional) | Optional — only for off-pipeline OSPREY runs | `~/.local/bin/uv`, fastest Python project manager |

If the chosen `${config.runtime.engine}` isn't available, the deploy fails immediately at `scripts/deploy.sh`. There's no automatic fallback.

---

## Initial server setup

One-time per server. Run as `${config.deploy.user}`.

```bash
# 1. Clone the profile repo
git clone https://${config.gitlab.host}/${config.gitlab.project_path}.git ${config.deploy.project_path}
cd ${config.deploy.project_path}

# 2. Create .env.production from the scaffolded template (see next section)
cp .env.template .env.production
$EDITOR .env.production   # fill in real secrets

# 3. Verify the registry is reachable and credentials work
echo "${env.${config.gitlab.token_env_var}}" \
  | ${config.runtime.engine} login --username deploy --password-stdin "${config.registry.url%%/*}"

# 4. Run the deploy
./scripts/deploy.sh
```

Operator's laptop also needs an SSH config entry:
```
# ~/.ssh/config
Host ${config.deploy.host}
    HostName ${config.deploy.fqdn}
    User ${config.deploy.user}
    # IdentityFile ~/.ssh/id_<whatever>
```

After this, every subsequent deploy is just:
```bash
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && git pull && ./scripts/deploy.sh"
```

---

## `.env.production` — operator-managed secrets

`.env.template` (committed) lists every variable name and what it's for. `.env.production` (gitignored) holds the real values. The scaffolder regenerates `.env.template` whenever `facility-config.yml` changes — diff against `.env.production` to find new vars to fill in.

Required structure (every facility has at least these):

```bash
# --- Registry credentials ---
# Used by deploy.sh to login to ${config.registry.url}
${config.gitlab.token_env_var}=<paste-PAT-with-read_registry-scope>

# --- LLM provider ---
${config.llm.api_key_env_var}=<provider-api-key>

# --- Container registry URL (mirrors facility-config.yml; deploy.sh reads here) ---
REGISTRY=${config.registry.url}

# --- Timezone ---
TZ=${config.facility.timezone}
```

Module-conditional additions (only present when the module is enabled):

```bash
# IF MODULE network.http_proxy (set)
HTTP_PROXY=${config.network.http_proxy}
HTTPS_PROXY=${config.network.https_proxy}
http_proxy=${config.network.http_proxy}
https_proxy=${config.network.https_proxy}
NO_PROXY=${config.network.no_proxy | join(",")}
no_proxy=${config.network.no_proxy | join(",")}
# END IF

# IF MODULE olog.enabled
${config.modules.olog.username_env_var}=<olog-user>
${config.modules.olog.password_env_var}=<olog-pass>
# END IF

# IF MODULE wiki_search.enabled
${config.modules.wiki_search.token_env_var}=<wiki-token>
# END IF

# IF MODULE event_dispatcher.enabled
${config.modules.event_dispatcher.token_env_var}=<random-bearer>
${config.modules.event_dispatcher.sidecar_token_env_var}=<random-bearer>
# END IF

# IF MODULE registry.external_projects (any)
# FOR each in registry.external_projects
${each.token_env_var}=<deploy-token-for-${each.name}>
# END FOR
# END IF
```

**Both UPPERCASE and lowercase** proxy variants must be set. Different libraries respect different conventions: Python's `requests` honours lowercase, Node's `http` module honours uppercase, `curl` honours both. If you only set one, you'll get intermittent failures that look like "the proxy works for some tools but not others."

`NO_PROXY` must include every internal service hostname that containers need to reach without being intercepted by the proxy. At minimum:
- `localhost`, `127.0.0.1`, `host.containers.internal`, `host.docker.internal`
- Every service name in `docker-compose.yml` that other containers connect to (Docker DNS names — `typesense`, `matlab`, etc., as named in compose).
- `${config.deploy.fqdn}` if any container connects back to its own host.

Without those entries, the proxy intercepts inter-container HTTP and you get hangs or 502s.

---

## Sourcing `.env` correctly

Always use `set -a; . .env.production; set +a`. The `set -a` is non-negotiable: without it, variables sourced from `.env` are local to the calling shell and aren't passed to child processes. Python, Node, and any other tool you spawn will silently see `None` instead of the value.

The generated `scripts/deploy.sh` does this correctly. Anyone writing helper scripts on the server must do the same.

---

## Container runtime: rootless podman gotchas

If `${config.runtime.engine} == "podman"` and the deploy user is not root (the strongly recommended setup), you will hit one or more of the following:

### Subordinate UID/GID provisioning

Rootless podman needs `/etc/subuid` and `/etc/subgid` entries for `${config.deploy.user}`. Without them, podman falls back to a single-ID user namespace where container UID 1000 maps to the kernel overflow UID (a number near `4294967295`). NFS-mounted home directories with `sec=sys` refuse writes from that UID, so any bind mount of an NFS-backed path for a container-writable directory fails with `Permission denied`.

**Symptoms**:
- `mkdir: cannot create directory '/data/something': Permission denied` inside a container, even though the host directory exists and is owned by the deploy user.
- `chown` inside the container reports a giant numeric UID instead of the expected mapping.

**Fix paths** (in order of preference):
1. Get IT to add `${config.deploy.user}:100000:65536` to `/etc/subuid` and `/etc/subgid`. Then podman's user namespace works correctly with `userns: keep-id`.
2. Use **named volumes** (managed by podman, stored on local disk) instead of bind mounts for any container-writable path. Named volumes bypass NFS entirely. Read-only mounts are fine even from NFS.
3. Run podman as root (defeats most of the point — only do this if IT refuses both other options).

The skill's templates default to named volumes for all writable container paths. Bind mounts are reserved for read-only mounts (e.g., `${config.modules.shared_disk.host_path}` in read-only mode).

### NFS `.nfs*` stale handles

If the deploy server's home directories are NFS-mounted, you'll see `.nfs00000000XXXX` files block directory removal whenever a process has shared libraries open. This breaks `${config.runtime.compose_command} down --volumes` and any "remove the project directory and re-clone" recovery.

**Symptoms**:
- `rm: cannot remove '.../.nfs00000000XXXX': Device or resource busy`
- `[Errno 39] Directory not empty` during a force-rebuild
- `${config.runtime.engine} volume rm <name>` fails

**Fix**: kill every process that has files open in the affected directory before removing. The deploy script's `--nuke` mode does `compose down --volumes` before removal; if processes are stuck, use `lsof +D <path>` to identify them, kill, retry.

For named volumes specifically: `${config.runtime.engine} unshare ls /var/lib/containers/storage/volumes/<volume>/_data/` to inspect contents, and `${config.runtime.engine} unshare rm -rf` to force-clean if needed.

---

## Permissions: who owns what

| Path | Owner | Mode | Notes |
|------|-------|------|-------|
| `${config.deploy.project_path}` | `${config.deploy.user}` | 755 | Owned by the user that runs deploy.sh |
| `${config.deploy.project_path}/.env.production` | `${config.deploy.user}` | 600 | Restrict — contains every secret |
| Named volumes | Managed by podman/docker, owned by user namespace | — | Inspect with `${config.runtime.engine} unshare ls` |
| Bind-mounted shared-disk paths (`${config.modules.shared_disk.host_path}`) | Whatever the source filesystem says | usually `ro` | If `mount_mode: rw`, the container's effective UID must match the host owner (or NFS will refuse writes) |

If multiple operators deploy from the same server, they should either share a single `${config.deploy.user}` account (simplest, recommended) or each have their own checkout and their own named volumes (more isolated, more disk usage).

---

## Stale process detection (NFS holds files open)

When you `compose down` and then immediately try to remove the volume or rebuild the image, you'll sometimes see "device busy" errors. NFS keeps file handles open until every reference is released, which on a busy server can take seconds to minutes after the container exits.

**Fastest fix**:
1. `${config.runtime.compose_command} down` (waits for graceful shutdown).
2. `${config.runtime.engine} ps -a` — confirm no straggler containers.
3. `${config.runtime.engine} ps -a --filter status=exited --format '{{.ID}}' | xargs -r ${config.runtime.engine} rm` — clean up exited.
4. Wait 5–10 seconds for NFS to release handles.
5. Then proceed with the volume/image cleanup.

If a container refuses to die: `${config.runtime.engine} kill --signal=SIGKILL <id>` then `${config.runtime.engine} rm -f <id>`.

The `scripts/deploy.sh --nuke` mode does this sequence. Don't reinvent it.

---

## SSH port forwarding for developer access

`${config.deploy.host}` typically isn't directly reachable from a developer laptop — services bind to `localhost` or to the host's interface and aren't exposed publicly. SSH tunneling forwards the relevant ports.

**Pattern** (substitute from config):
```bash
# Single service
ssh -L <local_port>:localhost:${config.ports.<service>} ${config.deploy.host}

# Multiple services in one command
ssh -L ${config.ports.web_terminal_nginx}:localhost:${config.ports.web_terminal_nginx} \
    -L ${config.ports.web_terminal_base}:localhost:${config.ports.web_terminal_base} \
    -L ${config.ports.integration_tests}:localhost:${config.ports.integration_tests} \
    ${config.deploy.host}

# Tunnel-only (no remote shell)
ssh -N -L <ports> ${config.deploy.host}
```

After the tunnel is up: open `http://localhost:<local_port>` in the browser.

| Service (when enabled) | Remote port | Suggested local port |
|------------------------|-------------|----------------------|
| Nginx landing page | `${config.ports.web_terminal_nginx}` | same |
| Web terminal (per user) | `${config.ports.web_terminal_base + offset}` | same |
| Integration tests dashboard | `${config.ports.integration_tests}` | same |
| Event dispatcher | `${config.modules.event_dispatcher.port}` | same |
| Each custom MCP server | `${config.modules.custom_mcp_servers.servers[*].port}` | same |

Putting the same number on both sides keeps mental math simple — local port = remote port.

For long-lived tunnels, add `-o ServerAliveInterval=60 -o ServerAliveCountMax=3` to keep them from dying on idle, and consider `autossh` for auto-reconnect.

---

## Verifying a deploy

After `scripts/deploy.sh` exits, the deploy is technically done — but always confirm:

```bash
# 1. All containers are up
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ${config.runtime.compose_command} ps"

# 2. None are restarting in a loop
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ${config.runtime.compose_command} ps --format json" \
  | python3 -c "import json,sys;[print(c['Service'],c['State']) for c in json.load(sys.stdin)]"

# 3. The advisory verify.sh ran (it's part of deploy.sh; check its output)

# 4. Run the integration health check suite
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/verify.sh"
```

For deeper diagnosis after a sick deploy, see `references/post-deploy-diagnosis.md`.

---

## Migrating to a new deploy server

When the facility moves from `serverA` to `serverB` (e.g., hardware refresh, OS upgrade, decommissioning):

1. **Update `facility-config.yml`** with the new server's values:
   - `deploy.host`, `deploy.fqdn`, `deploy.user`, `deploy.project_path`
   - `runtime.engine` and `runtime.compose_command` if the new server has a different container stack
   - `network.http_proxy`/`network.no_proxy` if the new network differs
2. **Re-run scaffolding** so generated files (`scripts/deploy.sh`, `.env.template`, `docker-compose.yml`) reflect the new values.
3. **Bootstrap the new server**:
   - Install prerequisites (see "Prerequisites" above) — IT may need a ticket if Python/podman/EPICS are missing.
   - Create the deploy user, ensure SSH access from operator laptops.
   - Provision rootless podman subordinate UIDs if applicable.
4. **Copy `.env.production`** from the old server to the new (NOT through git — it's gitignored). Update any host-specific paths inside it (e.g., variables that point to local data directories that may live at a different absolute path on the new host).
5. **First deploy on the new server**:
   ```bash
   ssh ${config.deploy.host} "git clone https://${config.gitlab.host}/${config.gitlab.project_path}.git ${config.deploy.project_path}"
   scp .env.production ${config.deploy.host}:${config.deploy.project_path}/
   ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh"
   ```
6. **Cut over external traffic** — DNS, reverse proxies, anything that points to the old server.
7. **Decommission the old server** only after the new one passes the integration test suite for at least one full operational cycle.

If the new server has different prerequisites (e.g., docker instead of podman), expect to debug compose differences — the `compose_command` is interchangeable in most cases but rootless behaviour, volume drivers, and network names sometimes differ.

---

## Differences between server platforms

The skill is platform-agnostic but a few behaviours vary:

| Concern | RHEL/Rocky 8/9 | Ubuntu 22.04+ | macOS (rare for prod) |
|---------|----------------|---------------|----------------------|
| Default container stack | podman + podman-compose | docker + docker compose plugin | docker desktop |
| Rootless podman default | usually broken without IT setup | usually works | N/A |
| systemd integration | `systemctl --user` for podman generate | same | none |
| NFS home directories | common at HPC-style sites | rare | N/A |
| Proxy environment | often required for outbound | often required | typically not |

The skill doesn't try to detect these — `facility-config.yml` declares the engine and compose command, and the generated scripts use exactly what's declared. If the server changes platform, update `facility-config.yml` and re-scaffold.
