# Module: custom_mcp_servers

The pattern for adding facility-specific MCP servers — anything beyond what OSPREY ships out of the box. Typical examples: a MATLAB Middle Layer search server, an in-house paper/document index, a hardware-specific telemetry adapter, a wiki integration that doesn't fit `modules.wiki_search`. Each entry under `modules.custom_mcp_servers.servers` is rendered into a compose service, a CI build job, and an image tag in the registry. The MCP server code itself lives in the facility profile repo (typically under `mcp_servers/<name>/`) and is the user's responsibility — this module is the wiring around it.

**Enabled when**: `modules.custom_mcp_servers.enabled: true` AND `servers:` is non-empty.

## Configuration

```yaml
modules:
  custom_mcp_servers:
    enabled: true
    servers:
      - name: "matlab"                                 # used in container name + image tag
        port: 8001                                     # also set in ports.matlab
        dockerfile: "docker/Dockerfile.matlab"         # path relative to repo root
        build_context: "."                             # build context for `docker build`
        artifacts:                                     # build-time artifacts copied into the image
          - "artifacts/mml.db"
        depends_on: []                                 # other compose services this depends on
      - name: "accelpapers"
        port: 8002
        dockerfile: "docker/Dockerfile.accelpapers"
        build_context: "."
        artifacts: []
        depends_on: ["typesense"]
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | string (lowercase, alnum + hyphens) | yes | Used in compose service name (`${config.facility.prefix}-mcp-${name}`), CI job name (`build-${name}`), image tag (`${name}:latest`). Must be unique across `servers`. |
| `port` | int | yes | Container HTTP port; must match `ports.${name}` in the same `facility-config.yml` so cross-references resolve. |
| `dockerfile` | path | yes | Path relative to repo root. The user authors this file; it should `COPY` source + artifacts and set `CMD`. |
| `build_context` | path | yes | Path relative to repo root. Usually `.` so the Dockerfile can reach both source and artifacts. |
| `artifacts` | list of paths | no | Files produced by the `osprey-build` CI stage that this image needs (e.g., a pre-built database). Each must be in the `artifacts/` output of that stage. |
| `depends_on` | list of strings | no | Compose service names (full names like `typesense`, not just `name` — these are the actual services in compose). |

## What scaffolding adds when this module is enabled

For **each** entry in `servers`:

- compose: a service block:
  ```yaml
  ${config.facility.prefix}-mcp-${server.name}:
    image: ${config.registry.url}/${server.name}:latest
    ports:
      - "${config.ports.${server.name}}:${server.port}"
    environment:
      - OSPREY_CONFIG=/app/config.yml
      - MCP_TRANSPORT=http
      - MCP_PORT=${server.port}
    depends_on: ${server.depends_on}
    networks:
      - ${config.facility.prefix}-net
  ```
  If `modules.shared_disk.enabled` is true and `${server.name}` appears in `modules.shared_disk.services_to_mount`, a `volumes:` entry is added too.
- .gitlab-ci.yml: one `build-${server.name}` job in the `docker-build` stage that builds `${server.dockerfile}` with context `${server.build_context}`, tags it `${config.registry.url}/${server.name}:$CI_COMMIT_SHORT_SHA` AND `:latest`, and pushes both. The job declares `needs:` on `osprey-build` if `${server.artifacts}` is non-empty.
- scripts/deploy.sh: nothing per-server — `compose pull` + `up -d` handles everything.
- scripts/verify.sh: an HTTP GET to `http://localhost:${config.ports.${server.name}}/health` for each server, expecting 200. Advisory only.
- .env.template: nothing per-server unless the server's Dockerfile references env vars (in which case the user adds them manually to the template after authoring).

## The MCP server is the user's code

This skill does NOT author the server itself. What the user must produce, per server:

1. **Source code** under `mcp_servers/${server.name}/` (or wherever they prefer). The server uses FastMCP (`from mcp.server.fastmcp import FastMCP`) and exposes one or more `@mcp.tool()` decorated functions.
2. **A `__main__.py`** that switches between stdio (for local OSPREY builds) and HTTP (for the container) based on `MCP_TRANSPORT`:
   ```python
   import os
   from .server import mcp
   if os.environ.get("MCP_TRANSPORT", "stdio") == "http":
       port = int(os.environ.get("MCP_PORT", "8000"))
       mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
   else:
       mcp.run()
   ```
3. **A Dockerfile** at `${server.dockerfile}` that installs deps, copies the server source + any artifacts, exposes the port, and runs `python -m ${server.name}` (or whatever module path).
4. **A profile YAML entry** (in the `mcp_servers:` block of `${config.facility.prefix}-prod.yml` and `-client.yml`) that declares the server with its URL and `permissions.allow` list.

The skill enforces only the wiring; the server itself is the user's domain.

## Standalone constraint: no OSPREY runtime imports

MCP servers must NOT `import osprey` at runtime. They get their config from the `OSPREY_CONFIG` env var, which points to a `config.yml` mounted (or COPYed) into the container at `/app/config.yml`. Keeping servers OSPREY-free means:

- The image rebuilds independently of OSPREY version churn — bumping OSPREY in CI doesn't force-rebuild every MCP container.
- The server is portable: it can be deployed without OSPREY in the picture at all (e.g., embedded in a different facility's stack that uses a different orchestrator).
- The dependency surface is small (just FastMCP + whatever the server needs for its actual job) → smaller image, faster cold start.

If a server needs facility-specific configuration, define it as a top-level key in `config.yml` (which is generated by the OSPREY build from the profile YAML's `services:` section) and have the server read it directly with `yaml.safe_load(open(os.environ["OSPREY_CONFIG"]))`.

## HTTP transport pattern

Each server runs as a `streamable-http` MCP endpoint at `http://${host}:${port}/mcp`. Inside the deploy network (Docker DNS), other containers reach it as `http://${config.facility.prefix}-mcp-${server.name}:${server.port}/mcp`. From outside (developer's laptop using `${config.facility.prefix}-client.yml`), they reach it as `http://${config.deploy.fqdn}:${config.ports.${server.name}}/mcp`.

The server should also expose `/health` returning `200 OK` with a small JSON body — the verify.sh check uses this, and it's good practice generally.

## CI build artifacts

If the server needs data files that are expensive to build at image-build time (e.g., a SQLite DB indexed from source data, or a vector index), produce them in the `osprey-build` CI stage and list them under `${server.artifacts}`. The CI pipeline:

1. `osprey-build` stage runs once, produces all artifacts in `artifacts/`.
2. Each `build-${server.name}` job declares `needs: [osprey-build]` and `dependencies: [osprey-build]`.
3. The Dockerfile `COPY artifacts/<file> /app/<file>` to bake them in.

This pattern keeps each server's image build fast (just `docker build` with the artifact already on disk) and avoids rebuilding the data per image.

## Adding a new server

1. **Author the server code** under `mcp_servers/<new_name>/` (this is outside the skill's scope; see existing servers as templates).
2. **Author the Dockerfile** at `docker/Dockerfile.<new_name>`. Follow the pattern of existing Dockerfiles in `docker/`.
3. **Re-run the deploy interview** (`references/setup-interview.md`) and at the `custom_mcp_servers` step, add a new entry. Or hand-edit `modules.custom_mcp_servers.servers` in `facility-config.yml`.
4. **Add the port allocation**: `${config.ports.<new_name>}` in the same file. Pick a port that doesn't collide.
5. **Re-scaffold** so the compose service and CI build job are generated.
6. **Add the profile YAML entry** in `${config.facility.prefix}-prod.yml` (and `-client.yml` if external clients should reach it):
   ```yaml
   mcp_servers:
     <new_name>:
       url: "http://${config.facility.prefix}-mcp-<new_name>:<port>/mcp"
       permissions:
         allow:
           - "<tool_name_1>"
           - "<tool_name_2>"
   ```
7. **Commit, push, deploy** via the standard pipeline.

## MCP tools sync invariant

Every `@mcp.tool()` decorator in the server must be matched by:

| List | Where | Purpose |
|------|-------|---------|
| `@mcp.tool()` decorators | `mcp_servers/<name>/tools/*.py` (or wherever) | What the server CAN expose |
| `permissions.allow` | profile YAML `mcp_servers.<name>.permissions.allow` | What Claude is ALLOWED to use |
| `EXPECTED_TOOLS` | `mcp_servers/integration_tests/checks/mcp_servers.py` | What the runtime check validates |

(The third list only applies if the integration_tests module is configured to validate MCP tool availability for this server.)

If the three lists drift apart, you'll see one of these symptoms:
- **Source has tool, permissions don't**: Claude sees the tool listed but is denied when it tries to use it. Fix: add to `permissions.allow`.
- **Permissions has tool, source doesn't**: Tool is allowed but doesn't exist; calls fail with "tool not found". Fix: remove from `permissions.allow` or add the `@mcp.tool()` to source.
- **EXPECTED_TOOLS has tool, source doesn't**: integration_tests reports "N expected tools missing" on the runtime check. Fix: remove from EXPECTED_TOOLS or add to source.

Verification snippet (run from repo root):

```bash
# Compare @mcp.tool() decorators in source against permissions.allow in the prod profile
python3 -c "
import yaml, subprocess, glob, re
source = {}
for srv_dir in glob.glob('mcp_servers/*/'):
    name = srv_dir.rstrip('/').split('/')[-1]
    tools = set()
    for f in glob.glob(f'{srv_dir}**/*.py', recursive=True):
        text = open(f).read()
        # find '@mcp.tool()' followed by 'def <name>'
        tools.update(re.findall(r'@mcp\.tool\(\)\s+(?:async\s+)?def\s+(\w+)', text))
    if tools: source[name] = tools

cfg = yaml.safe_load(open('${config.facility.prefix}-prod.yml'))
for n, s in cfg.get('mcp_servers', {}).items():
    p = set(s.get('permissions', {}).get('allow', []))
    src = source.get(n, set())
    if p and src:
        d = src - p
        if d: print(f'  {n}: in source but NOT in permissions: {d}')
        d = p - src
        if d: print(f'  {n}: in permissions but NOT in source: {d}')
"
```

## Operating the module

### Inspect a running server

```bash
ssh ${config.deploy.host} "${config.runtime.engine} logs ${config.facility.prefix}-mcp-${server.name} --tail 50"
ssh ${config.deploy.host} "curl -s http://localhost:${config.ports.${server.name}}/health"
```

### Force-rebuild one server's image

In CI: re-run the `build-${server.name}` job from the GitLab pipeline UI. The other build jobs are unaffected.

On the deploy server (NOT recommended — bypasses the audit trail):
```bash
ssh ${config.deploy.host} "cd ${config.deploy.project_path} && \
  ${config.runtime.engine} build -f ${server.dockerfile} -t ${config.registry.url}/${server.name}:latest ${server.build_context}"
```

### Remove a server

1. Delete the entry from `modules.custom_mcp_servers.servers`.
2. Delete the corresponding entry from `${config.ports}`.
3. Re-scaffold (compose service and CI job vanish).
4. Remove the entry from any profile YAMLs (`mcp_servers:` block).
5. `${config.runtime.compose_command} down ${config.facility.prefix}-mcp-${server.name}` on the deploy server.
6. Optionally, delete the source directory and Dockerfile from the repo.
7. Optionally, delete the registry image: `${config.runtime.engine} rmi ${config.registry.url}/${server.name}:latest`.

## Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `connection refused` from another container | Server crashed at startup | `${config.runtime.engine} logs <container>` — most common: import error in tool module |
| MCP tool listed but "tool not found" when called | Permissions ↔ source drift | Run the verification snippet above |
| `OSPREY_CONFIG` env var unset inside container | Compose env block wrong | Verify scaffolded compose service has `OSPREY_CONFIG=/app/config.yml`; re-scaffold if not |
| Image builds locally but CI build fails | Missing artifact dependency | Verify `${server.artifacts}` lists everything the Dockerfile `COPY`s from `artifacts/` |
| `/health` returns 200 but `/mcp` returns 404 | FastMCP not mounted at `/mcp` or wrong transport | Verify `__main__.py` uses `transport="streamable-http"` |
| Tools appear empty in MCP `tools/list` even though server is up | Tool modules failed to import (silent) | Check container startup logs for ImportError; tool decorators don't register if the module never executed |

## Cross-references

- If `modules.shared_disk.enabled` is also true and a server needs facility code from the shared disk, add `${server.name}`-equivalent (`${config.facility.prefix}-mcp-${server.name}`) to `modules.shared_disk.services_to_mount`.
- If `modules.integration_tests` validates MCP tool availability, every server here should be added to its `EXPECTED_TOOLS` map so the health check covers it.
- If `modules.wiki_search.type == "custom"`, the actual wiki search server is implemented as one of the entries here.

## Disabling

Set `modules.custom_mcp_servers.enabled: false` (or remove the `servers:` list entirely) and re-scaffold. Then:

- All `${config.facility.prefix}-mcp-<name>` services are removed from compose.
- All `build-<name>` CI jobs vanish.
- Existing registry images stay until manually deleted.
- Any profile YAML entries pointing at the removed servers will fail at Claude startup with "MCP server unreachable" — remove those entries too.
