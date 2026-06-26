# Module: shared_disk

Bind-mount a host-side directory (NFS or local) into selected containers. The canonical use case is the facility's physics code repository — MATLAB Middle Layer, in-house Python, lattice files, large reference datasets — that's too big or too live to bake into container images. Mount it read-only and let containers read what they need without committing the data to a registry.

**Enabled when**: `modules.shared_disk.enabled: true`

## Configuration

```yaml
modules:
  shared_disk:
    enabled: true
    host_path: "/mnt/physbase"               # absolute path on the deploy server
    container_path: "/physbase"              # absolute path inside containers that mount it
    mount_mode: "ro"                         # ro | rw — strongly prefer ro
    services_to_mount:                       # which compose services receive the bind mount
      - "matlab"
      - "integration_tests"
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `host_path` | absolute path | yes | Must exist and be readable by the user that runs the containers (`${config.deploy.user}`). NFS mounts and local paths both work. |
| `container_path` | absolute path | yes | Where the mount appears inside each container in `services_to_mount`. Use a stable path; consumer code references it. |
| `mount_mode` | enum | yes | `ro` (read-only) or `rw` (read-write). **Default to `ro`** — see "Read-only is the right default" below. |
| `services_to_mount` | list of strings | yes | Names of custom MCP servers (bare `matlab`, `integration_tests`) or full compose keys (`${prefix}-mcp-matlab`). The scaffolder resolves bare names to `${prefix}-mcp-<name>` automatically — see `references/scaffolding.md` § "Bare-name resolution inside FOR each". |

## What scaffolding adds when this module is enabled

- compose: in the host-overlay file (typically `docker-compose.host.yml` per `${config.runtime.compose_files}`), each service in `services_to_mount` gets an additional `volumes:` entry: `${config.modules.shared_disk.host_path}:${config.modules.shared_disk.container_path}:${config.modules.shared_disk.mount_mode}`.
- .gitlab-ci.yml: nothing — the data lives on the deploy server, not in CI.
- scripts/deploy.sh: a pre-flight check that verifies `${config.modules.shared_disk.host_path}` exists on the deploy server before running compose up. If missing, the deploy aborts with a clear error rather than starting containers that will fail to mount.
- scripts/verify.sh: a check that runs `${config.runtime.engine} exec` on one of the `services_to_mount` services and verifies `${config.modules.shared_disk.container_path}` is a populated directory inside the container. Advisory only.
- .env.template: nothing — the path is captured in `facility-config.yml`, not via env vars.

## Read-only is the right default

`mount_mode: "ro"` eliminates an entire class of bugs where a container accidentally writes to host data — overwriting lattice files, scribbling cache directories into the physics repo, or corrupting NFS state. Unless the facility has a documented reason for write access (e.g., a service legitimately produces output that other host-side tools consume), keep it read-only.

If you find yourself wanting to enable write access for "convenience," that's a signal to either:
- Move the writable artifact into a container-managed named volume (no host-data risk), or
- Add a dedicated writable bind to a sibling directory (e.g., `${config.modules.shared_disk.host_path}-output`) so the read-only invariant on the source data is preserved.

## Permissions: rootless podman + NFS pitfalls

Rootless podman uses user namespaces — UIDs inside the container are mapped to subordinate UIDs on the host. NFS servers configured with default `root_squash` and strict UID matching often refuse access to files owned by the host's `${config.deploy.user}` because the container-side UID maps to a sub-UID the NFS server doesn't recognize.

Symptoms:
- `Permission denied` on every read inside the container, even though `ls -l` on the host shows the directory is world-readable.
- The file IS visible inside the container (so the mount worked) but reads fail.

Workarounds, in order of preference:

1. **Configure the NFS export with `no_root_squash` and a wide UID range** that covers the user-namespace mapping. Coordinate with the storage team — this is the cleanest fix.
2. **Use `--userns=keep-id`** on the affected service (set in compose with `userns_mode: "keep-id"`). This maps the container's UID 0 to the host's `${config.deploy.user}` UID, avoiding the subordinate-UID range entirely. Side effect: any process in the container running as a different user (e.g., `nobody`) loses access too.
3. **Switch to a named volume populated by an init container**. An init container running with `--privileged` (or with explicit cap) copies the data from the NFS path into the volume on first boot; the application container then mounts the volume, not the NFS path directly. This isolates the application from the NFS UID issue but eats disk space and goes stale if the source data updates.

Option 1 is the long-term answer. Options 2 and 3 are workarounds for when the storage team won't change the export.

## Stale `.nfs` files

When a file on an NFS mount is opened by a process and then deleted (typically with `rm -f` or as part of a force-replace operation), the NFS client creates a placeholder `.nfsXXXXXX` file holding the inode open. The placeholder cannot be deleted while any process still has the file open, and `rm` on the placeholder fails with a confusing error.

Symptoms:
- `rm -rf <dir>` fails with `Device or resource busy` on a `.nfs*` file.
- A subsequent deploy that tries to clean up state fails for the same reason.
- `ls -la` shows `.nfsXXXXXX` files in directories that should be empty.

Diagnose:

```bash
# On the deploy server — find the placeholder
ssh ${config.deploy.host} "find ${config.modules.shared_disk.host_path} -name '.nfs*' -ls"

# Find the process holding it open
ssh ${config.deploy.host} "lsof | grep '.nfs<hash>'"
```

Fix: identify the holder (typically a stuck container, an old IOC process, or an SSH session with an open file in that directory), kill or restart it, then retry the `rm`. The placeholders disappear automatically once nothing has the inode open.

If the holder is one of the deploy's containers, `${config.runtime.compose_command} down` then `up -d` clears it. If the holder is unrelated, coordinate with whoever owns it.

## Operating the module

### Verify the mount is live

```bash
ssh ${config.deploy.host} "${config.runtime.engine} exec ${config.facility.prefix}-mcp-<service> \
  ls ${config.modules.shared_disk.container_path}"
```

Expected: the directory listing of `${config.modules.shared_disk.host_path}` as seen from inside the container. Empty output usually means the mount failed silently — check `${config.runtime.engine} inspect <container>` for the `Mounts` section.

### Add a service to `services_to_mount`

1. Edit `modules.shared_disk.services_to_mount` in `facility-config.yml`.
2. Re-scaffold (regenerates the compose overlay).
3. `${config.runtime.compose_command} up -d --force-recreate <service>` — recreating is needed because compose doesn't add volumes to a running container.

### Switch to read-write (don't, but if you must)

1. Confirm the storage team is OK with writes from the deploy server.
2. Change `mount_mode: "rw"` in `facility-config.yml`.
3. Re-scaffold + `up -d --force-recreate` for affected services.
4. Add a paragraph to the deploy README explaining what writes the service does and how to recover if a write goes wrong.

## Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Deploy aborts: "host_path does not exist" | NFS not mounted, or path wrong | Mount the NFS share on the deploy server (typically via `/etc/fstab`); confirm `host_path` matches |
| `Permission denied` inside container despite world-readable on host | Rootless podman + NFS UID mapping | See "Permissions" above |
| `rm -rf` on the host fails with `.nfsXXX busy` | Process holds file open | Find holder with `lsof`, kill or restart it, retry |
| Mount appears empty inside container | Bind mounted before host_path was populated, or wrong path | Verify `host_path` on host has content; recreate the container with `up -d --force-recreate` |
| Writes succeed but other hosts don't see them (rw mode) | NFS client-side caching | Increase NFS write coherence (`sync` mount option) or switch back to read-only |
| Mount works for one container, fails for another in the same compose | One service is missing from `services_to_mount` | Add it to the list and re-scaffold |

## Cross-references

- If `modules.custom_mcp_servers.enabled` is also true, custom MCP servers that need facility code (typical for matlab equivalents) should be listed in `services_to_mount`.
- If `modules.web_terminals.enabled` is also true, you can add `${config.facility.prefix}-web-${user}` entries to `services_to_mount` so web terminals can `Read`/`Glob`/`Grep` the shared disk directly via Claude's built-in file tools — no MCP server needed for read-only browsing.

## Disabling

Set `modules.shared_disk.enabled: false` (or remove the block) and re-scaffold. Then:

- The compose overlay no longer adds the bind mount.
- `${config.runtime.compose_command} up -d --force-recreate` for any service that previously had the mount, so it stops trying to read from the missing path.
- The host's NFS mount can stay or be unmounted independently of this module — it's not managed here.
- Any code inside the affected services that referenced `${config.modules.shared_disk.container_path}` will now fail on access; remove or guard those references.
