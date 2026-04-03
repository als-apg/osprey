# Osprey Framework

[![CI](https://github.com/als-apg/osprey/workflows/CI/badge.svg)](https://github.com/als-apg/osprey/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/osprey-framework/badge/?version=latest)](https://als-apg.github.io/osprey/)
[![codecov](https://codecov.io/gh/als-apg/osprey/branch/main/graph/badge.svg)](https://codecov.io/gh/als-apg/osprey)
[![PyPI version](https://badge.fury.io/py/osprey-framework.svg)](https://badge.fury.io/py/osprey-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Latest Release: v0.11.5** - Timezone Support & Open WebUI Fix

> **Early Access Release**
> This is an early access version of the Osprey Framework. While the core functionality is stable and ready for experimentation, documentation and APIs may still evolve. We welcome feedback and contributions!

A production-ready framework for deploying agentic AI in large-scale, safety-critical control system environments—particle accelerators, fusion experiments, beamlines, and complex scientific facilities.

**Research**
This work was presented as a contributed oral presentation at [ICALEPCS'25](https://indico.jacow.org/event/86/overview) and will be featured at the [Machine Learning and the Physical Sciences Workshop](https://ml4physicalsciences.github.io/2025/) at NeurIPS 2025.


## Quick Start

```bash
# Install the framework (using uv, recommended)
uv pip install osprey-framework

# Or using pip:
# pip install osprey-framework

# Recommended: Interactive setup (guides you through everything!)
osprey

# The interactive menu will:
# - Help you choose a template with descriptions
# - Guide you through AI provider and model selection
# - Automatically detect and configure API keys from your environment
# - Create a ready-to-use project with smart defaults

# Alternative: Direct command if you know what you want
osprey init my-assistant
cd my-assistant
# If API keys aren't in your environment, copy and edit .env:
# cp .env.example .env

# Start a Claude Code agent session
claude
```


## Documentation

**[Read the Full Documentation](https://als-apg.github.io/osprey)**

### Testing

```bash
# Run unit tests (fast, no API keys required)
pytest tests/ --ignore=tests/e2e -v

# Run e2e tests (slow, requires API keys)
pytest tests/e2e/ -v
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) and [tests/e2e/README.md](tests/e2e/README.md) for details.


## Key Features

- **Dual-Mode Orchestration** - Plan-first (complete upfront plans) and reactive (ReAct, step-by-step) execution with explicit dependencies and operator oversight
- **Control-System Safety** - Pattern detection, PV boundary checking, and mandatory approval for hardware writes
- **Protocol-Agnostic Integration** - Seamless connection to EPICS, LabVIEW, Tango, and mock environments
- **Scalable Capability Management** - Dynamic classification prevents prompt explosion as toolsets grow
- **Production-Proven** - Deployed at major facilities including LBNL's Advanced Light Source accelerator

---

## CLI Reference

All commands are accessed through the `osprey` command. Running `osprey` without arguments launches an interactive TUI menu.

```bash
osprey                    # Launch interactive menu
osprey --version          # Show framework version
osprey init PROJECT       # Create new project
osprey config             # Manage configuration
osprey build              # Build project from profile
osprey deploy COMMAND     # Manage services
osprey health             # Check system health
osprey migrate            # Run project migrations
osprey tasks              # Browse AI assistant tasks
osprey claude             # Manage Claude Code integration
osprey web                # Launch web terminal
osprey audit              # Audit project or profile safety
osprey eject              # Copy framework components for customization
osprey channel-finder     # Channel finder CLI
osprey ariel              # ARIEL logbook search service
osprey artifacts          # Artifact gallery
osprey prompts            # Prompt artifact overrides
```

### Global Options

- `--version` — Show framework version and exit.
- `--help` — Show help for any command (e.g., `osprey deploy --help`).

### osprey init

Create a new project from a template.

```bash
osprey init [OPTIONS] PROJECT_NAME
```

- `--template <name>` — `hello_world`, `control_assistant` (default), or `lattice_design`.
- `--registry-style <style>` — `extend` (default, recommended) or `standalone`.

```bash
osprey init my-agent
osprey init my-first-agent --template hello_world
```

### osprey config

Manage project configuration. Interactive menu if no subcommand is given.

```bash
osprey config show [--project PATH] [--format yaml|json]
osprey config export [--output PATH] [--format yaml|json]
osprey config set-control-system SYSTEM_TYPE [--project PATH]
osprey config set-epics-gateway [--facility als|aps|custom] [--address] [--port]
osprey config set-models [--provider PROVIDER] [--model MODEL] [--project PATH]
```

```bash
osprey config show
osprey config set-control-system epics
osprey config set-models --provider anthropic --model claude-sonnet-4
```

### osprey build

Build a facility-specific assistant from a profile.

```bash
osprey build PROJECT_NAME PROFILE [OPTIONS]
```

- `-o, --output-dir PATH` — Output directory (default: current directory).
- `-f, --force` — Overwrite existing project directory.
- `-s, --stream` — Stream build step output in real time.

```bash
osprey build als-test ~/profiles/als-dev.yml
osprey build my-assistant profile.yml --force -o /tmp
```

### osprey deploy

Manage containerized services. All subcommands accept `--project PATH`.

| Subcommand | Description |
|------------|-------------|
| `up [--detached] [--dev]` | Start services |
| `down` | Stop all running services |
| `restart` | Restart all services |
| `status` | Show status of deployed services |
| `clean` | Stop services and remove containers and volumes |
| `rebuild [--detached] [--dev]` | Rebuild containers from scratch |

```bash
osprey deploy up --detached
osprey deploy status
osprey deploy rebuild --dev
osprey deploy down
```

### osprey health

Run comprehensive system health check.

```bash
osprey health [--project PATH]
```

### osprey migrate

Run project migrations for newer framework versions.

```bash
osprey migrate
```

### osprey claude

Manage Claude Code integration — regenerate artifacts, launch chat, and check status.

| Subcommand | Description |
|------------|-------------|
| `chat [OPTIONS]` | Regenerate artifacts, launch companion servers, start Claude Code |
| `regen [OPTIONS]` | Re-render all Claude Code integration files from `config.yml` |
| `status [OPTIONS]` | Display provider config, model mappings, artifact sync status |

**chat options:** `-p, --project DIR`, `--resume SESSION_ID`, `--print`, `--effort [low|medium|high|max]`

**regen options:** `-p, --project DIR`, `--dry-run`

```bash
osprey claude chat
osprey claude chat --resume abc123
osprey claude regen --dry-run
osprey claude status
```

### osprey tasks

Browse and manage AI assistant tasks (structured development workflows).

```bash
osprey tasks                          # Launch interactive task browser
osprey tasks list                     # List all available tasks
osprey tasks copy TASK_NAME [--force] # Copy a task to .ai-tasks/
osprey tasks show TASK_NAME           # Print task instructions
osprey tasks path TASK_NAME           # Print path to task file
```

### osprey eject

Copy framework capabilities or services to your project for customization.

```bash
osprey eject list
osprey eject capability channel_finding
osprey eject service channel_finder --include-tests
```

### osprey channel-finder

Natural language channel search with REPL, queries, and benchmarking.

```bash
osprey channel-finder                                    # Launch interactive REPL
osprey channel-finder query "find beam position monitors" # Single query
osprey channel-finder benchmark [--queries] [--model]     # Run benchmarks
osprey channel-finder build-database [--csv PATH]         # Build from CSV
osprey channel-finder validate [--database PATH]          # Validate database
osprey channel-finder preview [--depth N] [--sections]    # Preview database
```

### osprey ariel

Manage the ARIEL logbook search service.

| Subcommand | Description |
|------------|-------------|
| `quickstart [--source PATH]` | Full setup: migrate and ingest demo data |
| `status [--json]` | Show service status |
| `migrate` | Create or update database tables |
| `ingest --source PATH [--adapter TYPE]` | Ingest logbook entries |
| `watch [--source] [--interval N]` | Poll for new entries |
| `enhance [--module NAME] [--force]` | Run enhancement modules |
| `search QUERY [--mode auto\|keyword\|semantic\|rag]` | Execute a search query |
| `web [--port N] [--host ADDR]` | Launch web interface |
| `purge [--yes]` | Delete all ARIEL data |

```bash
osprey ariel quickstart
osprey ariel search "RF cavity fault"
osprey ariel web --port 8080
```

### osprey web

Launch the Web Terminal interface.

```bash
osprey web                            # Default: http://127.0.0.1:8087
osprey web --port 9000 --host 0.0.0.0
osprey web --detach                   # Run in background
osprey web stop                       # Stop background server
```

### osprey audit

Audit a build profile or project directory for safety risks.

```bash
osprey audit my-project/
osprey audit profile.yml --build
osprey audit project/ --json
```

### osprey prompts

Manage prompt artifact overrides for customizing framework prompt templates.

### Environment Variables

```bash
OSPREY_PROJECT=/path/to/project   # Default project directory
ANTHROPIC_API_KEY=sk-...          # Or OPENAI_API_KEY, GOOGLE_API_KEY, etc.
```

`OSPREY_PROJECT` sets a default project directory for all commands. Priority:
`--project` flag > `OSPREY_PROJECT` > current directory.

---

## Citation

If you use the Osprey Framework in your research or projects, please cite our [paper](https://doi.org/10.1063/5.0306302):

```bibtex
@article{10.1063/5.0306302,
      author = {Hellert, Thorsten and Montenegro, João and Sulc, Antonin},
      title = {Osprey: Production-ready agentic AI for safety-critical control systems},
      journal = {APL Machine Learning},
      volume = {4},
      number = {1},
      pages = {016103},
      year = {2026},
      month = {02},
      doi = {10.1063/5.0306302},
      url = {https://doi.org/10.1063/5.0306302},
}
```

---

*For detailed installation instructions, tutorials, and API reference, please visit our [complete documentation](https://als-apg.github.io/osprey).*

---

**Copyright Notice**

Osprey Framework Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

---
