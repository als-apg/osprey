# Installing Osprey Framework (next branch) v0.11.5

This guide covers installing the **next** branch of the Osprey Framework from source at the ALS.

## Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | 3.11+ | `python3 --version` |
| Git | any | `git --version` |
| Node.js | 18+ (for Claude Code CLI) | `node --version` |
| Container runtime | Docker 4.0+ or Podman 4.0+ (optional) | `docker --version` / `podman --version` |

## 1. Clone the Repository

Clone the `next` branch to your desired location:

```bash
git clone --branch next https://github.com/als-apg/osprey.git /home/oxygen/SHANG/next_osprey
```

If the target directory already exists and is non-empty, clone to a temporary location and copy:

```bash
git clone --branch next https://github.com/als-apg/osprey.git /tmp/osprey_next_clone
cp -a /tmp/osprey_next_clone/. /home/oxygen/SHANG/next_osprey/
rm -rf /tmp/osprey_next_clone
```

Verify the branch:

```bash
cd /home/oxygen/SHANG/next_osprey
git branch
# Should show: * next
```

## 2. Install uv (Recommended Package Manager)

```bash
pip install uv
```

## 3. Install Osprey

### Option A: Editable install into an existing environment

```bash
cd /home/oxygen/SHANG/next_osprey
uv pip install -e .
```

### Option B: Create an isolated virtual environment with uv sync

```bash
cd /home/oxygen/SHANG/next_osprey
uv sync
```

This creates a `.venv` directory automatically and installs all dependencies.

### Option C: With development and documentation extras

```bash
cd /home/oxygen/SHANG/next_osprey
uv sync --extra dev --extra docs
```

## 4. Install Additional Dependencies

### pyEpics (EPICS Channel Access)

Required for connecting to the EPICS control system:

```bash
uv pip install pyepics
```

### als-archiver-client (EPICS Archiver Appliance)

Required when using a real EPICS Archiver Appliance (i.e., `archiver.type: epics_archiver` in config.yml). Provides the `archivertools` Python module for retrieving historical time-series data:

```bash
uv pip install als-archiver-client
```

This is used by the archiver connector to query the APS archiver at `https://pvarchiver.aps.anl.gov`.

## 5. Verify Installation

```bash
osprey --version
# Expected: osprey, version 0.11.5

python3 -c "import osprey; print(osprey.__version__)"
# Expected: 0.11.5
```

## 6. Configure Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cd /home/oxygen/SHANG/next_osprey
cp env.example .env
```

Edit `.env` and set at least one API key:

```dotenv
# Required: at least one API key
ANTHROPIC_API_KEY=sk-ant-...        # Recommended
CBORG_API_KEY=...                   # LBNL institutional provider
OPENAI_API_KEY=...                  # OpenAI GPT models
GOOGLE_API_KEY=...                  # Google Gemini models

# File paths
PROJECT_ROOT=/home/oxygen/SHANG/next_osprey
LOCAL_PYTHON_VENV=/home/oxygen/SHANG/next_osprey/.venv
```

## 7. Install Claude Code CLI (Optional)

Claude Code is the AI orchestration layer used by Osprey:

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

## 8. Create a Project

```bash
# Interactive mode (recommended for new users)
osprey

# Or create directly
osprey init my-assistant
cd my-assistant
```

Available templates:
- `hello_world` -- Minimal tutorial with one MCP server and mock control system
- `control_assistant` (default) -- Control system integration with channel finder, archiver, and logbook
- `lattice_design` -- Accelerator lattice physics with pyAT integration

## 9. Deploy Services (Optional)

Services require Docker or Podman. They enhance agent capabilities but are not required for basic usage.

```bash
# Start services
osprey deploy up

# Or in background
osprey deploy up --detached

# Check status
osprey deploy status

# Stop services
osprey deploy down
```

## 10. Launch the Agent

```bash
# From within a project directory
claude

# Or use the managed launcher
osprey claude chat

# Or use the browser interface
osprey web
```

## Notes

### Playwright (Browser Binaries)

Playwright is installed as a Python dependency, but the actual browser binaries (Chromium) are **not** included. Osprey uses Playwright only for **artifact export** (converting HTML to PDF/PNG). It will **auto-install Chromium on first use**, so no action is needed upfront.

If the auto-install fails (e.g., no internet or permission issues), install manually:

```bash
playwright install chromium
```

Playwright is not involved in general usage (running agents, MCP servers, control system interaction).

### .loglogin File

On tcsh systems, the target directory may contain an auto-generated `.loglogin` file. `git clone` refuses to clone into any non-empty directory, so if this file is present you have two options:

1. **Delete it first**, then clone directly:
   ```bash
   rm /home/oxygen/SHANG/next_osprey/.loglogin
   git clone --branch next https://github.com/als-apg/osprey.git /home/oxygen/SHANG/next_osprey
   ```

2. **Clone to a temp location** and copy (works regardless):
   ```bash
   git clone --branch next https://github.com/als-apg/osprey.git /tmp/osprey_next_clone
   cp -a /tmp/osprey_next_clone/. /home/oxygen/SHANG/next_osprey/
   rm -rf /tmp/osprey_next_clone
   ```

Note that `.loglogin` may be recreated by tcsh on login, so option 2 is the safer approach.

## Troubleshooting

| Problem | Solution |
|---|---|
| `osprey` command not found | Ensure the install environment is activated, or re-run `uv pip install -e .` |
| Python version mismatch | Use `python3.11` or `python3.12` explicitly |
| `claude` command not found | Install with `npm install -g @anthropic-ai/claude-code` |
| MCP connection failed | Run from the project root where `.mcp.json` lives |
| Provider auth error | Check API keys in `.env` or export them: `export ANTHROPIC_API_KEY=...` |
| Container issues | Verify runtime is running: `docker ps` or `podman ps` |
| Dependency conflicts | Try a clean install: `uv sync` in a fresh clone |

## Useful Commands

```bash
osprey --help              # Show all CLI commands
osprey health              # System health check
osprey config              # Manage configuration
osprey channel-finder      # Interactive channel search
osprey artifacts           # Artifact gallery
osprey audit               # Safety auditor
```
