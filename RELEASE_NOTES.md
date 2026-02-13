# Osprey Framework - Latest Release (v0.11.0)

**Built-in Capabilities & ARIEL Logbook Search**

## What's New in v0.11.0

### Highlights

- **ARIEL logbook search** - New electronic logbook search capability with full-text and semantic search, web interface, CLI commands, and deployment support. The fastest way to try it is the [ARIEL Quick Start](https://als-apg.github.io/osprey/developer-guides/05_production-systems/07_logbook-search-service/index.html) -- three commands to a working search interface.
- **Native control capabilities** - 4 control capabilities (`channel_finding`, `channel_read`, `channel_write`, `archiver_retrieval`) migrated from Jinja2 templates to native Python modules in `src/osprey/capabilities/`
- **Channel Finder service as native package** - 48 service files moved from templates to `src/osprey/services/channel_finder/` with default prompt builders
- **`osprey eject` command** - Customization escape hatch to copy framework capabilities or services into a project for modification
- **Template simplification** - `control_assistant` template reduced from ~130 to ~40 files

### Getting Started with ARIEL

The easiest way to try ARIEL is to start fresh with the [Quick Start guide](https://als-apg.github.io/osprey/developer-guides/05_production-systems/07_logbook-search-service/index.html) -- you'll have a working logbook search in minutes.

### Migrating Existing Agents

This release includes a significant refactoring: capabilities and services that were previously generated from Jinja2 templates now ship as native Python modules. This is a one-time structural change -- with capabilities living in the framework itself, future upgrades should be straightforward version bumps rather than template re-generations. We expect this to be the last migration that requires manual effort.

If you have an existing Osprey agent, the built-in migration assistant will walk you through the upgrade:

```bash
osprey claude install migrate   # Install the migration skill
# Then ask Claude Code to migrate your project
```

See [AI-Assisted Development](https://als-apg.github.io/osprey/contributing/03_ai-assisted-development.html) for the full guide on using `osprey assist` tasks.

### Added
- **Capabilities**: Migrate control capabilities to native Python modules
  - Context classes inlined into capability files
  - `FrameworkRegistryProvider` registers native capabilities and context classes automatically
- **Services**: Migrate Channel Finder service to native package
  - Default prompt builders at `src/osprey/prompts/defaults/channel_finder/`
  - Facility-specific prompt overrides via framework prompts
- **CLI**: Add `osprey eject` command for customization escape hatch
  - Subcommands: `eject list`, `eject capability`, `eject service` with `--output` and `--include-tests` options
- **CLI**: Add `osprey channel-finder` command with interactive REPL, query, and benchmark modes
- **CLI**: Add `build-database`, `validate`, and `preview` subcommands to `osprey channel-finder`
  - Database tools migrated from Jinja2 templates to native `osprey.services.channel_finder.tools`
  - LLM channel namer available as library via `osprey.services.channel_finder.tools.llm_channel_namer`
- **Registry**: Add shadow warning system for backward compatibility
  - Detects when generated apps override native capabilities without explicit `override_capabilities` config
- **ARIEL**: Add electronic logbook search capability
  - Full-text and semantic search over facility logbooks (OLOG, custom sources)
  - Web interface with dashboard, search, and entry browsing (`osprey ariel web`)
  - CLI commands: `osprey ariel ingest`, `osprey ariel search`, `osprey ariel purge`
  - Deployment support: PostgreSQL and web service templates for `osprey deploy up`
  - Pluggable search modules and enhancement pipeline with registry-based discovery

### Changed
- **Templates**: Simplify `control_assistant` template (~130 to ~40 files)
  - `registry.py.j2` now uses `extend_framework_registry()` with prompt providers only
  - Capabilities, services, and database tools no longer generated from templates

---

## Installation

```bash
pip install --upgrade osprey-framework
```

Or install with all optional dependencies:

```bash
pip install --upgrade "osprey-framework[all]"
```

---

## What's Next?

Check out our [documentation](https://als-apg.github.io/osprey) for:
- [ARIEL Quick Start](https://als-apg.github.io/osprey/developer-guides/05_production-systems/07_logbook-search-service/index.html) -- get a working logbook search in minutes
- [Migration assistance](https://als-apg.github.io/osprey/contributing/03_ai-assisted-development.html) -- upgrade existing agents to v0.11
- Native capabilities and `osprey eject` guide
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
