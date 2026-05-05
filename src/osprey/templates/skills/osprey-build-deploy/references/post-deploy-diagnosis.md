# Post-Deploy Diagnosis

A deploy completed without errors, but something is wrong with the running stack. This reference is the playbook for finding and fixing it without making things worse.

The cardinal mistake in post-deploy diagnosis is **changing config before reproducing the symptom**. Most "broken deploys" are diagnosed in five minutes by reading container status and logs; the rest take longer because someone restarted, re-pulled, or `--clean`'d before observing what was actually wrong, destroying the evidence.

---

## Diagnostic mindset

Walk down the stack from outside in. Stop at the first layer that shows the problem; don't dig further until you've explained what you see.

1. **Container status** — are all expected containers running? Any crashes / restarts?
2. **Container logs** — what does the failing service say at startup and on its last request?
3. **Inside-the-container config** — does the regenerated `config.yml` / `.mcp.json` reference the right paths?
4. **Network** — are containers on the same compose network? Can they resolve each other by name?
5. **Health endpoints** — does `/checks` reach every dependency? See `references/integration-tests.md`.

Only after the symptom is reproduced and localized: change config, restart, redeploy. Never the other way around.

---

## Useful commands

Substitute `${config.X.Y}` from `facility-config.yml`. The runtime engine is `${config.runtime.engine}` (typically `podman` or `docker`).

### List containers in this project

```bash
${config.runtime.engine} ps -a \
  --filter "label=io.${config.runtime.engine}.compose.project=${config.facility.prefix}-profiles"
```

What to look for:
- Every expected service is `Up`.
- Restart counts are 0 (high counts → something keeps crashing on startup).
- Exited containers with non-zero exit codes — `${config.runtime.engine} logs <name>` immediately.

### Per-container logs

```bash
# An MCP server
${config.runtime.engine} logs ${config.facility.prefix}-mcp-<name>

# A web terminal (only if modules.web_terminals.enabled)
${config.runtime.engine} logs ${config.facility.prefix}-web-<user>

# Tail and follow
${config.runtime.engine} logs -f --tail 200 ${config.facility.prefix}-mcp-<name>

# Logs since a timestamp
${config.runtime.engine} logs --since 10m ${config.facility.prefix}-mcp-<name>
```

The first 50 lines after `Started server process` are usually decisive — that's where import failures, missing env vars, and bind errors surface.

### Network state

```bash
${config.runtime.engine} network inspect \
  ${config.facility.prefix}-profiles_${config.facility.prefix}-net
```

Confirm:
- Every expected container appears under `Containers`.
- No duplicate networks (sometimes a stale network from a previous deploy leaves containers isolated).
- DNS aliases match service names from `docker-compose.yml`.

### Inspect regenerated MCP config inside a container

The web-terminal image bakes the project at one path during CI and copies it to a different path at runtime. A post-COPY regen step rewrites `.mcp.json` and `config.yml` to use container-internal paths. If that step fails or runs against the wrong path, all OSPREY-native MCP servers break.

```bash
# Check .mcp.json inside the container
${config.runtime.engine} exec ${config.facility.prefix}-web-<user> \
  cat /app/${config.facility.prefix}-assistant/.mcp.json | python3 -m json.tool

# Verify regenerated config paths
${config.runtime.engine} exec ${config.facility.prefix}-web-<user> \
  grep -E 'project_root|python_env_path' /app/${config.facility.prefix}-assistant/config.yml
```

Expected after regen:
- `project_root` = `/app/${config.facility.prefix}-assistant`
- `python_env_path` is empty or matches the container's `.venv`

If either still shows a CI build path (e.g., `/builds/...`), the regen step did not run — go straight to the Dockerfile.

### Health endpoint round-trip

```bash
# Full health report
curl http://${config.deploy.host}:${config.ports.integration_tests}/checks

# From inside a web terminal (validates intra-network DNS)
${config.runtime.engine} exec ${config.facility.prefix}-web-<user> \
  curl -s http://integration-tests:${config.ports.integration_tests}/checks
```

A discrepancy between host-routed (`${config.deploy.host}:<port>`) and intra-network (`integration-tests:<port>`) results means the network is broken even if both probes succeed individually.

### Process-level inspection inside a container

```bash
# What's actually running?
${config.runtime.engine} exec ${config.facility.prefix}-mcp-<name> ps -ef

# Open file descriptors / sockets
${config.runtime.engine} exec ${config.facility.prefix}-mcp-<name> ss -tlnp

# Env vars actually visible to the process
${config.runtime.engine} exec ${config.facility.prefix}-mcp-<name> env | sort
```

Useful when the logs say "started" but a probe times out — the process may be running but bound to the wrong interface.

---

## Common Failure Patterns

A short table covering the recurring failure modes. Match the **signature** column to what you're observing — don't try to apply more than one fix at a time.

| Signature | Root cause | Fix |
|-----------|-----------|-----|
| All Python MCP servers down inside web terminals; their `command` paths show `/builds/...` | Container path-regen step failed during image build | Inspect the regen layer in `docker/Dockerfile.web-terminal`. The `osprey claude regen --project /app/...` step must succeed; usually a missing volume or a permission issue. Rebuild the web-terminal image |
| Sidecars and MCP servers all healthy individually, but unreachable from web terminals (DNS resolution fails) | The compose network was recreated, leaving some containers attached to the previous network | `ssh ${config.deploy.host} "cd ${config.deploy.project_path} && ./scripts/deploy.sh --clean"` — `--clean` does `compose down` + `up`, which recreates the network and reattaches all services |
| One service down, all others fine | Single-container crash on startup or during a request | `${config.runtime.engine} logs ${config.facility.prefix}-<name>` — fix the underlying error before restarting. Restart loops without log inspection just hide the problem |
| `verify.sh` shows everything OK, but Claude says a tool is "not available" | MCP `tools/list` mismatch — server healthy, tools missing | Run the `mcp_servers.<name>.tools` check explicitly (see `references/integration-tests.md` § EXPECTED_TOOLS Consistency). Likely cause: a `@mcp.tool()` decorator has an import error that doesn't kill the server |
| `deploy.sh` fails at `${config.runtime.engine} login ${config.registry.url}` | Expired GitLab registry token | Rotate the value referenced by `${config.gitlab.token_env_var}` in `.env` (the `*_env_var` field in config tells you the name; the value lives in `.env`, never in config) |
| `deploy.sh` fails at `${config.runtime.engine} pull` for an external project image | Expired external deploy token (separate from the main GitLab token) | Update the relevant `${config.registry.external_projects[*].token_env_var}` value in `.env`. Each external project has its own token; main project token does not grant cross-project pulls |
| Container starts, exits in <5 seconds, restart loop | Missing required env var or invalid mount | Logs first — usually a clear `KeyError` or `FileNotFoundError`. If the env var is missing, check `.env` against `.env.template` (the template lists every var the deploy needs) |
| `services.proxy` ERROR on the deploy server but fine locally | Container's `${env.HTTP_PROXY}`/`${env.NO_PROXY}` not propagated | Verify compose `environment:` block lists the proxy vars. The `network.*` block in `facility-config.yml` is the source — regenerate compose if you changed it |
| Health check works on host, fails inside web terminal | Web-terminal container is on a different network than the MCP services | `${config.runtime.engine} network inspect ${config.facility.prefix}-profiles_${config.facility.prefix}-net` — every web-terminal container should appear. If not, `--clean` deploy |
| Newly deployed image still shows old behavior | Compose pulled a stale tag (cached locally) | `${config.runtime.engine} pull <image>` explicitly, or `deploy.sh --clean`. Avoid `--nuke` unless you're sure — it deletes named volumes |

---

## Deploy modes — when to use which

`scripts/deploy.sh` has three modes (see `SKILL.md` § Deploy Pipeline). They escalate in destructiveness:

| Mode | When to use | What dies |
|------|-------------|-----------|
| (no flag) | Default; nothing is obviously broken | Only changed containers restart |
| `--clean` | Network seems broken, sidecar DNS failing, weird routing issues | All containers stop and restart; network is recreated |
| `--nuke` | "Make it as fresh as possible" — unrecoverable state, tag confusion, registry image cache poisoning | All containers, images, volumes, *and the project network* are removed; full re-pull from registry |

**Reach for `--nuke` rarely.** It deletes named volumes, which means anything stored only in a container volume (a dev-only Postgres dump, a built MML index that takes 20 minutes to rebuild, etc.) is gone. Always check what's volume-mounted before nuking.

If `--clean` doesn't fix it, prefer **hand-debugging** over `--nuke`. The reason `--clean` didn't help is information; nuking erases that information.

---

## When to ask the running assistant to self-diagnose

If `modules.web_terminals` is enabled, the built assistant ships a `/diagnose` skill (`overlays/skills/diagnose/`). The product Claude can introspect its own MCP servers, validate config paths, and pattern-match failures from the *inside* — which gives a different vantage point than SSH'ing in from outside.

Use the in-container `/diagnose` when:
- The user is already in a web terminal session and reports a tool is broken.
- You suspect the issue is per-user (one terminal misbehaves, others don't).
- You want to see what Claude actually sees — paths, env vars, MCP `tools/list` results — not what the operator sees.

Use SSH + `${config.runtime.engine}` from outside when:
- The container itself is down (no web terminal to ask).
- You suspect a network or registry problem (cross-container concerns the assistant inside one container can't see).
- You're debugging the deploy script, the image build, or anything outside the running assistant's view.

The two views are complementary. If the operator and the in-container `/diagnose` disagree about what's healthy, the disagreement itself is a clue — usually a network or DNS issue that the assistant can't see across, but the operator can.

---

## Escalation

When the table above doesn't match and the logs aren't conclusive:

1. **Capture state** before changing anything:
   ```bash
   ${config.runtime.engine} ps -a > /tmp/diag-ps.txt
   ${config.runtime.engine} logs ${config.facility.prefix}-<failing> > /tmp/diag-logs.txt
   curl -s http://${config.deploy.host}:${config.ports.integration_tests}/checks > /tmp/diag-checks.json
   ```
   These three artifacts are usually enough for a second pair of eyes.

2. **Try `--clean`** (not `--nuke`) once. If the problem clears, document the failure mode — recurring `--clean` need is itself a bug worth fixing.

3. **Inspect the image directly:**
   ```bash
   ${config.runtime.engine} run --rm -it --entrypoint /bin/sh \
     ${config.registry.url}/<image>:latest
   ```
   Drops you into a fresh container with no compose context — useful when you suspect the image itself is wrong vs. the runtime config.

4. **Re-pull from registry explicitly** to rule out tag confusion:
   ```bash
   ${config.runtime.engine} pull ${config.registry.url}/<image>:latest
   ```

5. **Last resort:** `deploy.sh --nuke` and re-deploy. After this works, immediately investigate what was wedged — `--nuke` working when nothing else did is itself a bug.

---

## What never to do

- Don't `${config.runtime.engine} system prune -a` to "clean things up." It deletes images other projects on the same host depend on.
- Don't restart a container in a tight loop hoping it stabilizes. If it crashed once it will crash again until the cause is fixed.
- Don't edit `config.yml` or `.mcp.json` *inside* a container as a "quick fix." Those files are regenerated; your edit will be lost on the next deploy and the underlying template will still be wrong.
- Don't bypass `deploy.sh` with raw `compose up` commands. The script handles registry login, external project pulls, and verify steps — skipping it leaves a partially-deployed stack.
- Don't change `facility-config.yml` *and* re-deploy in the same step. Change the config, regenerate the affected files via scaffolding, review the diff, *then* deploy.
