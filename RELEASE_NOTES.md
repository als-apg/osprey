# Osprey Framework - Latest Release (v0.10.6)

**Context Validation & Chat History**

## What's New in v0.10.6

### Highlights

- **Context key validation** - Orchestrator validates execution plans before running, catching invalid key references and ordering errors
- **Chat history in orchestrator** (#111) - Follow-up queries like "use the same time range" now resolve correctly
- **Task objective metadata** (#108) - Context entries track what they were created for, enabling intelligent reuse
- **Release workflow skill** - Claude Code skill for guided release process

### Added
- **CLI**: Add Claude Code skill for release workflow (`osprey claude install release-workflow`)
  - Custom SKILL.md wrapper with quick reference for version files and commands
  - Version consistency check command, pre-release testing steps, tag creation
- **Orchestration**: Context key validation in execution plans
  - Validates input key references match actual context keys
  - Detects ordering errors (step references key from later step)
  - New `InvalidContextKeyError` exception
- **Context**: Store task_objective metadata alongside capability context data (#108)
  - New helper methods: `get_context_metadata()`, `get_all_context_metadata()`
  - Orchestrator prompt displays task_objective for each available context

### Fixed
- **Graph**: Propagate chat history to orchestrator and respond nodes (#111)
  - Orchestrator now receives full conversation context when `task_depends_on_chat_history=True`
- **Deployment**: Fix Claude Code config path resolution in pipelines container

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
- TUI mode guide
- Artifact system API reference
- Complete tutorial series

## Contributors

Thank you to everyone who contributed to this release!

---

**Full Changelog**: https://github.com/als-apg/osprey/blob/main/CHANGELOG.md
