# Osprey Framework - Latest Release (v0.10.9)

**Config-Driven Provider Loading & Capability Slash Commands**

## What's New in v0.10.9

### Highlights

- **Config-driven provider loading** - Registry skips unused provider imports, eliminating ~30s startup delay on air-gapped machines
- **Argo structured output** - Structured output support for Argo provider via direct httpx calls with JSON schema prompting
- **Capability slash commands** - Forward unregistered slash commands to capabilities for domain-specific actions

### Added
- **CLI**: Add `--channel-finder-mode` and `--code-generator` options to `osprey init`
  - Options are included in manifest's `reproducible_command` for full project recreation
- **Capabilities**: Add capability-specific slash commands
  - Unregistered slash commands (e.g., `/beam:diagnostic`, `/verbose`) are forwarded to capabilities
  - `slash_command()` helper and `BaseCapability.slash_command()` method for reading commands
  - Commands are execution-scoped (reset each conversation turn)

### Fixed
- **Registry**: Config-driven provider loading skips unused provider imports (#138)
  - Eliminates ~30s startup delay on air-gapped machines
- **Argo**: Structured output handler for Argo provider (JSON schema prompting via httpx)
- **Tests**: Fix e2e LLM provider tests broken by config-driven provider filtering

### Changed
- **Docs**: Update citation to published APL Machine Learning paper (doi:10.1063/5.0306302)

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
- Capability slash commands guide
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
