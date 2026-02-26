# Using a Local AgentsView Build with Osprey

Osprey discovers the `agentsview` binary via `shutil.which("agentsview")` (PATH lookup) and launches it as a subprocess on web terminal startup. To test a local development build of agentsview, you need to make your locally-built binary appear in PATH before the installed version.

## Prerequisites

- Go 1.21+ with CGO enabled
- Node.js + npm (for frontend build)
- The agentsview source repo (e.g. `/Users/thellert/LBL/ML/_tmp/agentsview`)

## Steps

### 1. Build the agentsview binary

```bash
cd /Users/thellert/LBL/ML/_tmp/agentsview
make build
```

This produces `bin/agentsview` with the embedded Svelte frontend.

### 2. Kill any running agentsview process

Osprey caches the subprocess, so you need to stop the old one first:

```bash
pkill agentsview
```

### 3. Start Osprey with the local binary in PATH

Prepend the agentsview `bin/` directory to PATH so it takes priority:

```bash
PATH="/Users/thellert/LBL/ML/_tmp/agentsview/bin:$PATH" make run
```

Or export it for the current shell session:

```bash
export PATH="/Users/thellert/LBL/ML/_tmp/agentsview/bin:$PATH"
make run
```

### 4. Verify

Open the web terminal and click the **SESSION ANALYTICS** tab. The iframe loads agentsview from `http://127.0.0.1:8096`. You should see your local changes (e.g. the sub-agent panel on sessions that have sub-agents).

To confirm which binary osprey is using:

```bash
which agentsview
# Should print: /Users/thellert/LBL/ML/_tmp/agentsview/bin/agentsview
```

## Alternative: Install to ~/.local/bin

If you want to replace the system-wide binary instead of overriding PATH per-session:

```bash
cd /Users/thellert/LBL/ML/_tmp/agentsview
make install
```

This copies the binary to `~/.local/bin/agentsview` (or `$GOPATH/bin`). Then restart osprey normally -- no PATH override needed.

## Iterating on changes

After making code changes in the agentsview repo:

```bash
# Rebuild
cd /Users/thellert/LBL/ML/_tmp/agentsview
make build          # or: make frontend && make build (if only frontend changed)

# Kill the old process so osprey relaunches on next request
pkill agentsview

# Refresh the SESSION ANALYTICS tab in the browser
```

Osprey's launcher will automatically restart agentsview on the next iframe load.
