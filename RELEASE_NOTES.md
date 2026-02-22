# Osprey Framework - Latest Release (v0.11.3)

**uv/Hatchling Migration, AMSC Provider & Safety Fixes**

## What's New in v0.11.3

### Highlights

- **Reactive orchestrator** - New ReAct-style tool loop (`ReactiveOrchestratorNode`) for autonomous, LLM-driven decision-making as an alternative to rigid plan-then-execute. Includes tool registry, argument parsing, approval hooks, and automatic capability dependency expansion.
- **Unified typed event system** - 18 typed dataclass events across 7 categories replace dict-based logging. `EventEmitter` with LangGraph-first streaming, `consume_stream()` multi-mode helper, and complete elimination of raw Python logger usage across the framework.
- **Prompt builder refactoring** - All LLM prompts moved into composable `FrameworkPromptBuilder` subclasses. Applications can now customize any prompt via subclass overrides without forking capability code.
- **LLM token streaming** - Real-time token streaming across CLI, TUI, and Open WebUI with multi-mode architecture combining typed events and LLM tokens.
- **Web debug interface** - Browser-based real-time event visualization with WebSocket streaming, dark theme, component filtering, and search.

### Added
- **Infrastructure**: Reactive orchestrator with ReAct-style tool loop, dependency expansion, and pre-dispatch validation (#162)
- **Events**: Unified typed event system with 18 events, `EventEmitter`, `consume_stream()`, and LLM metadata tracking
- **Interfaces**: Web debug interface with FastAPI/WebSocket, dark theme, tooltips, and LLM streaming groups
- **Interfaces**: LLM token streaming across CLI (Rich table output), TUI (`StreamingChatMessage`, `CollapsibleCodeMessage`), and Open WebUI
- **TUI**: Info bar, consistent shortcuts, notebook preview, debug block widget, log viewer refinements
- **Models**: `chat_request()` method for native message-based completions
- **Channel Finder**: `--delimiter` option for CSV files (#161, @RemiLehe)

### Changed
- **Prompts**: Composable `FrameworkPromptBuilder` system with deprecation bridges and runtime context injection (#163)
- **Logging**: All Python logger calls replaced with unified `get_logger` system

### Fixed
- Capability approval state, rate-limit classification, E2E trace rebuilding, dependency management, and streaming fixes across all interfaces

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
