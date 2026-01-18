# Changelog

All notable changes to the Osprey Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Deployment**: Fix Claude Code config path resolution in pipelines container
  - Pipelines container has working directory `/app/` but files are mounted at `/pipelines/`
  - Config file was copied but relative path `claude_generator_config.yml` couldn't be found
  - Now reads `claude_config_path` from config, copies the file, and updates path to absolute `/pipelines/` for pipelines service

## [0.10.5] - 2026-01-16

### Added
- **Testing**: E2E test for LLM channel naming workflow (#103)

### Changed
- **Docs**: Update ALS Assistant reference to published paper (Phys. Rev. Res. **8**, L012017)
- **Models**: Decouple LiteLLM adapter from hardcoded provider checks
  - Providers now declare LiteLLM routing via class attributes (`litellm_prefix`, `is_openai_compatible`)
  - Structured output detection now uses LiteLLM's `supports_response_schema()` function
  - Custom providers can integrate without modifying the adapter layer
  - Maintains backward compatibility with fallback for existing code

### Fixed
- **CI**: Fix deploy-e2e test to actually test PR code by using `--dev` mode
  - Container was installing osprey from PyPI instead of the PR branch
  - Now builds and installs local wheel so the test validates actual changes
- **Channel Finder**: Fix `load_config` not defined error in LLM channel namer (#103)
  - Added `get_config_builder()` and `load_config()` as public API in `osprey.utils.config`
  - Exposed `load_config` in channel finder config utilities
  - Updated channel finder components to use public API instead of internal `_get_config`
- **Deployment**: Fix `--dev` mode failing when osprey is installed from PyPI (#86)
  - Detect site-packages installation and show clear warning about editable mode requirement
  - Add helpful error message when `build` package is missing
  - Add `build` to dev dependencies for wheel building support
- **Models**: Handle Python-style booleans in LLM JSON responses (#102)
  - Some LLM providers (including Argo) return `True`/`False` instead of `true`/`false`
  - `_clean_json_response()` now converts Python-style booleans to JSON-style
- **CLI**: Display full absolute paths for plot files in artifact output (#96)
  - Figure and notebook paths now resolved to absolute before artifact registration
  - Ensures users can directly access generated files from CLI output
- **Packaging**: Include TUI styles.tcss in package data (#97)
  - Textual CSS file was missing from PyPI releases since TUI was introduced in 0.10.0
  - Issue went unnoticed because editable installs (`pip install -e .`) symlink to source

## [0.10.4] - 2026-01-15

### Fixed
- **Dependencies**: Pin aiohttp>=3.10 for litellm compatibility (#87)
  - Fixes `AttributeError: module aiohttp has no attribute ConnectionTimeoutError`
  - `aiohttp.ConnectionTimeoutError` was added in aiohttp 3.10; litellm requires it but doesn't pin the version

## [0.10.3] - 2026-01-14

### Changed
- **CI**: Add E2E tests to GitHub Actions workflow
  - Runs on PRs only (not pushes) to control API costs
  - Skips fork PRs where secrets are unavailable
- **Dependencies**: Move TUI (textual) from optional to base dependencies
  - Removes `[tui]` extras group since textual is now always installed

## [0.10.2] - 2026-01-14

### Added
- **State**: Unified artifact system with `ArtifactType` enum and `register_artifact()` API
  - Single source of truth (`ui_artifacts`) for all artifact types: IMAGE, NOTEBOOK, COMMAND, HTML, FILE
  - Legacy methods (`register_figure`, `register_notebook`, `register_command`) delegate to new API
  - `populate_legacy_fields_from_artifacts()` helper for backward compatibility at finalization
- **TUI**: Artifact gallery and viewer widgets for interactive artifact browsing
  - ArtifactGallery with keyboard navigation (Ctrl+a focus, j/k navigate, Enter view, o open external)
  - ArtifactViewer modal with type-specific details and actions (copy path, open in system app)
  - Native image rendering via textual-image (Sixel for iTerm2/WezTerm, Kitty Graphics Protocol)
  - New/seen tracking with [NEW] badges for artifacts from current turn

### Changed
- **Tooling**: Consolidated formatting/linting to Ruff, removed Black and Isort (#80)
  - Ruff now handles both linting and formatting as a single tool
  - Updated scripts, docs, and templates to reference only Ruff
- **Capabilities**: Python capability uses unified `register_artifact()` API directly
  - Clean single-accumulation pattern for figures and notebooks
  - Legacy fields populated at finalization rather than registration
- **CLI**: Modernized artifact display to use unified `ui_artifacts` registry
  - Single `_extract_artifacts_for_cli()` replaces three legacy extraction methods
  - Supports all artifact types: IMAGE, NOTEBOOK, COMMAND, HTML, FILE
  - Grouped display with type-specific formatting and icons

### Fixed
- **Gateway**: `/chat` without arguments no longer triggers graph execution
  - Displays available capabilities table correctly, then returns immediately
  - New check for locally-handled commands with no remaining message
  - CLI handles state-only updates with no agent_state gracefully
- **Orchestrator**: Use descriptive context keys to prevent incorrect time range reuse (#90)
  - Similar time ranges (e.g., 12/5-12/10 vs 12/5-12/8) no longer incorrectly reuse old context
  - Context keys now encode actual dates (tr_MMDD_MMDD format) for proper comparison
- **Approval**: Fix KeyError when optional approval config keys are omitted (#79)
  - Logger now uses initialized config object instead of raw dict keys
- **Templates**: Include deployment infrastructure config for all templates (#85)
  - Fixes `osprey deploy up` failures for hello_world_weather template
  - Jupyter kernel templates now render correctly with execution.modes section
- **CLI**: Restrict `load_dotenv()` search to current directory only (#95)
  - Prevents python-dotenv from parsing shell config files in parent directories
  - Fixes warnings when users have `~/.env` as a Korn shell configuration file

## [0.10.1] - 2026-01-09

### Added
- **State**: Session state persistence for user preferences and mode tracking
  - New `session_state` field in AgentState with custom merge reducer
  - Enables direct chat mode and other session-level settings to persist across conversation turns
- **Infrastructure**: Direct chat mode routing and message handling
  - Router detects direct chat mode and routes directly to capability
  - Gateway preserves message history in direct chat mode
  - Validates capability supports direct_chat_enabled before routing
- **Capabilities**: Context management tools for ReAct agents
  - read_context, list_available_context, save_result_to_context
  - remove_context, clear_context_type, get_context_summary
  - Enables agents to manage accumulated context during direct chat
- **Capabilities**: StateManager capability for interactive state management
  - Natural language interface for context and agent settings
  - State inspection tools: session info, execution status, capability list, settings
  - State modification tools: clear session, modify agent settings
  - Registered as framework-level capability (/chat:state_manager)
- **CLI**: Direct chat mode for conversational interaction with capabilities
  - `/chat:<capability>` enters direct chat mode
  - `/chat` lists available direct-chat capabilities
  - `/exit` returns to normal mode (adds transition marker for context)
  - Dynamic prompt shows current mode (normal vs capability name)
  - Quieter logging during direct chat for cleaner experience
- **Generators**: Direct chat mode support in MCP capability generator
  - Generated capabilities have direct_chat_enabled=True by default
  - Adds context management tools when in direct chat mode
  - Handles both orchestrated and direct chat execution modes
  - Updated docstrings with direct chat usage examples
- **Models**: LangChain model factory for full LangGraph ReAct agent support
  - `get_langchain_model()` creates BaseChatModel instances from osprey config
  - Supports all 8 providers: anthropic, openai, google, ollama, cborg, vllm, stanford, argo
  - Native integration with `create_react_agent` and other LangGraph workflows
  - Automatic configuration loading from osprey's config system
- **Models**: New vLLM provider adapter for high-throughput local inference
  - Uses LiteLLM's OpenAI-compatible interface
  - Auto-detects served models via `/models` endpoint
  - Supports structured outputs with json_schema
- **Models**: Direct Ollama API for thinking models (bypasses LiteLLM bug #15463)
  - gpt-oss and other thinking models now work correctly
  - Automatic minimum token allocation (100) for thinking phase
- **Tests**: Consolidated E2E test suite for LLM providers (`tests/e2e/test_llm_providers.py`)
  - Provider × model × task matrix approach (anthropic, openai, google, cborg, ollama, vllm)
  - Tests basic completion, structured output (Pydantic), and ReAct agent workflows
  - Auto-skips unavailable providers/models based on environment
  - Graceful handling of API quota/rate limit errors (skips with warning instead of failing)
- **Documentation**: Direct chat mode user and developer documentation
  - CLI Reference: `/chat` and `/exit` commands, Direct Chat Mode section with examples
  - Gateway Architecture: Direct chat mode handling, message history preservation, GatewayResult fields
  - Classification and Routing: Router priority with direct chat bypass
  - Building First Capability: `direct_chat_enabled` attribute and tip box

### Changed
- **Capabilities**: Support direct chat execution mode in capability decorator
  - Creates synthetic execution step when no execution plan exists
  - Skips step progression in direct chat mode
  - Changed classifier missing log from warning to debug (expected for direct-chat-only capabilities)
- **Logging**: Reduced verbose third-party logging for cleaner CLI output
  - Added quiet_logging() context manager for temporary log suppression
  - Suppressed LiteLLM debug messages
- **Models**: Migrated all LLM provider implementations to LiteLLM unified interface (#23)
  - Replaced ~2,200 lines of custom provider code with ~700 lines using LiteLLM adapter
  - All 8 providers (anthropic, google, openai, ollama, cborg, stanford, argo, vllm) now use LiteLLM
  - Preserved extended thinking, structured outputs, and health check functionality
  - Access to 100+ providers through LiteLLM

### Removed
- **Models**: Removed unused `get_model()` function and `factory.py` module
  - The function was dead code (never called anywhere in the codebase)
  - All model access now goes through `get_chat_completion()`

### Fixed
- **Code Generation**: Fix `${VAR}` environment variable expansion in `claude_code_generator`
- **CBORG Provider**: Add missing `temperature` parameter to API calls
  - Fixes non-deterministic code generation behavior causing intermittent test failures
  - Both regular text completion and structured output paths now respect temperature setting
- **Code Generation**: Add simplicity guidance to prevent over-engineered solutions
  - LLM now prefers direct context usage over building complex systems to fetch data
- **Documentation**: Fixed workflow file references to use correct `@src/osprey/workflows/` path for copy-paste into Claude Code and Cursor
- **Gateway**: Mode switch handling for direct chat entry/exit
  - Use `update_state()` for mode switches instead of `ainvoke()` to avoid full graph execution
  - Correct field names (`planning_execution_plan`, `planning_current_step_index`)
  - New `is_state_only_update` flag signals callers to use proper update method
  - New `exit_interface` flag for `/exit` outside direct chat mode
- **Commands**: New `gateway_handled` flag ensures state-affecting commands route through gateway
  - /exit, /planning, /approval, /task, /caps, /chat marked as gateway_handled
  - Ensures consistent behavior across all interfaces (CLI, OpenWebUI, API)
- **CLI**: Proper routing for gateway_handled vs local commands
  - Local commands (/help, /clear) handled directly for instant response
  - State commands route through gateway for consistent state management
- **Router**: Suppress routing logs during state-only evaluations
  - Mode switches no longer produce confusing "routing to task extraction" logs
  - Uses `execution_start_time` to detect active vs state-only execution
- **Capabilities**: Context tool changes now persist to LangGraph state
  - State manager and MCP capabilities return `capability_context_data` in state updates
  - Fixes context save/remove operations having no effect in direct chat mode

## [0.10.0] - 2026-01-08

### Added
- **TUI**: New Textual-based Terminal User Interface (`osprey chat --tui`)
  - Full-screen terminal experience with real-time streaming of agent responses
  - Step-by-step visualization: Task Extraction → Classification → Orchestration → Execution
  - Welcome screen with ASCII banner and quick-start guidance
  - Theme support with 15+ built-in themes and interactive theme picker (Ctrl+T)
  - Command palette for quick access to all actions (Ctrl+P)
  - Slash commands support (`/exit`, `/caps:on`, `/caps:off`, etc.)
  - Query history navigation with up/down arrows
  - Content viewer for prompts and responses with multi-tab support and markdown rendering
  - Log viewer with live updates for debugging
  - Todo list visualization showing agent planning progress
  - Keyboard shortcuts for navigation (scroll, focus input, toggle help)
  - Double Ctrl+C to quit for safety
  - ~5,500 lines of new code across 17 files in `src/osprey/interfaces/tui/`
- **Logging**: Enhanced logging system with TUI data extraction support
  - New `_build_extra()` method embeds streaming event data into Python logs
  - Enables TUI to receive all data through a single logging source
  - Added `QueueLogHandler` for async log processing in TUI
- **CLI**: New `osprey tasks` command for browsing AI assistant tasks
  - `osprey tasks` - Interactive task browser (default)
  - `osprey tasks list` - List all available tasks
  - `osprey tasks show <task>` - Print task instructions to stdout
  - `osprey tasks copy <task>` - Copy task to project's `.ai-tasks/` directory
  - `osprey tasks path <task>` - Print path to task's instructions file
- **CLI**: New `osprey claude` command for Claude Code skill management
  - `osprey claude install <task>` - Install a task as a Claude Code skill
  - `osprey claude list` - List installed and available skills
- **Assist System**: General-purpose architecture for AI coding assistant integrations
  - Tool-agnostic task instructions in `src/osprey/assist/tasks/`
  - Tool-specific wrappers in `src/osprey/assist/integrations/`
  - Pre-commit task for validating code before commits
  - Migration task for upgrading downstream OSPREY projects
- **Tests**: Comprehensive tests for `tasks_cmd.py` and `claude_cmd.py`

### Changed
- **CLI**: `osprey chat` now supports `--tui` flag to launch the TUI interface
  - Default behavior unchanged (CLI interface)
  - TUI requires textual package: `pip install osprey-framework[tui]`
- **CLI**: Deprecated `osprey workflows` command (use `osprey tasks` instead)
  - Command still works for backward compatibility but shows deprecation warning
- **Code Generation**: Enhanced `claude_code_generator` with environment variable support
  - Config template now supports custom environment variables via `claude_generator_config.yml`
  - Added ARGO endpoint configuration to template
  - Fixed default URL to use correct localhost link
- **Documentation**: Updated workflow references to use new command structure
  - `osprey tasks list` for browsing tasks
  - `osprey claude install <task>` for installing Claude Code skills
- **Documentation**: Updated release-workflow instructions with accurate test counts
  - Unit tests: ~1850 tests (~1-2 min) instead of outdated ~370-380 tests (~5s)
  - E2E tests: ~32 tests (~10-12 min) instead of outdated ~5 tests (~2-3 min)

### Removed
- **Workflows**: Removed duplicate workflow files from `src/osprey/workflows/`
  - Content consolidated into `src/osprey/assist/tasks/{name}/instructions.md`
  - Only `README.md` deprecation notice remains in workflows directory

## [0.9.10] - 2025-01-03

### Fixed
- **Channel Finder**: Initialize `query_splitting` attribute in HierarchicalPipeline
  - Fixes `AttributeError: 'HierarchicalPipeline' object has no attribute 'query_splitting'`

### Added
- **Channel Finder**: Optional `query_splitting` parameter for hierarchical and middle_layer pipelines
  - Disable query splitting for facility-specific terminology that shouldn't be split
  - Enabled by default for backward compatibility

### Changed
- **Channel Finder Prompts**: Modularized prompt structure across all pipelines
  - Split `system.py` into `facility_description.py` (REQUIRED) and `matching_rules.py` (OPTIONAL)
  - Users now edit `facility_description.py` for facility-specific content
  - `system.py` auto-combines modules (no manual editing needed)
  - Query splitter prompts now accept `facility_name` parameter
- **Benchmark Dataset**: Renamed `in_context_main.json` to `in_context_benchmark.json` for consistency
- **Documentation**: Updated control assistant tutorials for modular prompt structure
  - Part 1: Updated directory structure with new prompt file layout
  - Part 2: Added cross-references to prompt customization section
  - Part 4: Expanded channel finder prompt customization with step-by-step guidance

### Added
- **Channel Finder**: Added explicit detection functionality to channel finder service
  - New `explicit_detection.py` prompt module for detecting explicit channel names, PV names, and IOC names
  - Updated `BasePipeline` with `build_result()` helper method for constructing pipeline results
  - Enhanced all pipeline implementations (hierarchical, in-context, middle layer) to use explicit detection
  - Added unit tests for explicit detection prompt and `build_result()` method
  - Updated e2e tests to verify explicit detection behavior
  - Configuration updates to include explicit detection in pipeline workflows
- **Tests**: `test_memory_capability.py`: 32 tests for memory operations, context, exceptions, and helper functions (37.7% → 62.4% coverage)
- **Tests**: `test_logging.py`: 27 tests for API call logging, caller info extraction, and file creation (29.1% → 55.7% coverage)
- **Tests**: `test_models.py` (generators): 21 tests for capability generation Pydantic models (0% → 100% coverage)
- **Tests**: `test_models_utilities.py` (python_executor): 39 tests for execution error handling, notebook tracking, and utility functions
- **Tests**: `test_models.py` (memory_storage): 13 tests for memory content formatting and validation (0% → 100% coverage)
- **Tests**: `test_storage_manager.py`: 22 tests for memory persistence, file operations, and entry management (24.1% → 72.4% coverage)
- **Tests**: `test_memory_provider.py`: 23 tests for memory data source integration and prompt formatting (32.2% → 94.9% coverage)
- **Tests**: `test_providers_argo.py`: 27 tests for ARGO provider adapter (18.6% → 54.8% coverage)
- **Tests**: `test_providers_ollama.py`: 31 tests for Ollama provider with fallback logic (24.2% → 96.0% coverage)
- **Tests**: `test_providers_anthropic.py`: 27 tests for Anthropic provider metadata, model creation, and health checks (23.5% → 50.0% coverage)
- **Tests**: `test_completion.py`: 28 tests for TypedDict conversion and proxy validation (30.9% → 58.0% coverage)
- **Tests**: `test_logging.py`: 19 tests for API call context and result sanitization (13.3% → 29.1% coverage)
- **Tests**: `test_respond_node.py`: 26 tests for response generation, context gathering, and mode determination (37.7% → 72.1% coverage, infrastructure module 54.7% → 58.4%)
- **Tests**: `test_task_extraction_node.py`: 25 tests for task extraction, data source integration, and error classification (33.0% → 62.1% coverage, infrastructure module 52.1% → 54.7%)
- **Tests**: `test_error_node.py`: 29 tests for error response generation and context handling (33.6% → 91.8% coverage, infrastructure module 45.2% → 52.1%)
- **Tests**: Expanded infrastructure and models tests - 40 new tests for error classification, retry policies, and helper functions (infrastructure module 37.2% → 45.2%, overall 45.8% → 46.4%)
- **Tests**: Added comprehensive tests for CLI and deployment modules (coverage expansion)
  - `test_preview_styles.py`: 23 tests for theme preview and color display functionality (0% → 88.1% coverage)
  - `test_main.py`: 23 tests for CLI entry point and lazy command loading (28.6% → 95.2% coverage)
  - `test_health_cmd.py`: 38 tests for health checks and environment diagnostics (0% → 69.6% coverage)
  - `test_loader.py`: 55 tests for YAML loading, imports, and parameter management (0% → 86.6% coverage)
  - `test_chat_cmd.py`: 15 tests for command execution and output formatting
  - `test_export_config_cmd.py`: 16 tests for deprecation warnings and format options
  - `test_deploy_cmd.py`: 23 tests for deployment actions (up/down/restart/status/build/clean/rebuild)
  - `test_registry_cmd.py`: 22 tests for registry display functions
  - `test_config_cmd.py`: 23 tests for config subcommands (show/export/set-control-system/set-epics-gateway/set-models)
  - `test_remove_cmd.py`: 16 tests for capability removal and backups
  - `test_generate_cmd.py`: 37 tests for code generation commands (capability/mcp-server/claude-config)
  - `test_orchestration_node.py`: 12 tests for execution planning validation and error handling
  - `test_classification_node.py`: 13 tests for capability classification structure and error handling
  - Fixed missing `Dict` import in `scripts/analyze_test_coverage.py`
  - Renamed `analyze_coverage.py` → `analyze_test_coverage.py` for clarity

### Fixed
- **CLI**: Fixed broken imports in `config_cmd.py`
  - Changed `update_control_system_type` → `set_control_system_type` (correct function name)
  - Changed `update_epics_gateway` → `set_epics_gateway_config` (correct function name)
  - Updated function calls to handle return values correctly (both functions return tuple of new_content, preview)

### Changed
- **Control Assistant**: Write access now enabled by default in control assistant template (`writes_enabled: true` for mock connector)
  - Simplifies tutorial experience - users can test write operations immediately with mock connector
  - Production deployments should carefully review hardware implications before enabling writes
- **License**: Added explicit "BSD 3-Clause License" header to LICENSE.txt for clarity

### Documentation
- Updated Hello World tutorial to reflect current weather capability implementation with natural language location handling
- Fixed version picker showing non-existent versioned directories causing 404 errors
  - Updated docs workflow to only list actually deployed versions (stable and latest/development)
  - Removed all individual version tag entries from versions.json until versioned directories are implemented
- Fixed double slash typos in image paths causing 404 errors on GitHub Pages for in-context and hierarchical channel finder CLI screenshots
- Added "Viewing Exported Workflows" section to AI-assisted development guide showing example output of exported workflow files
- Removed v0.9.2+ migration guide (no longer needed as framework has fully transitioned to instance method pattern)
  - Cleaned up all cross-references to migration guide across documentation
  - Streamlined architecture overview sections in main index and developer guides
  - Updated main index diagram from workflow to architecture overview
- Added academic reference (Hellert et al. 2025, arXiv:2512.18779) for semantic channel finding theoretical framework

## [0.9.9] - 2025-12-22

### Fixed
- **Testing**: Fixed middle layer benchmark test assertion to use `queries_evaluated` instead of `total_queries` field from benchmark results

### Changed
- **Workflows**: Moved AI workflow files from `docs/workflows/` to `src/osprey/workflows/` for package bundling
  - Workflows now distributed with installed package
  - Enables version-locked workflow documentation
- **Documentation**: Updated workflow references to use `@osprey-workflows/` path
  - Added workflow export instructions to AI-assisted development guide
  - Updated all @-mention examples across documentation

### Added
- **CLI**: New `osprey workflows` command to export AI workflow files
  - `osprey workflows export` - Export workflows to local directory (default: ./osprey-workflows/)
  - `osprey workflows list` - List all available workflow files
  - Interactive menu integration for easy access
- **Documentation - AI Workflows**: Channel Finder workflow guides for AI-assisted development
  - New workflow files: pipeline selection guide and database builder guide with AI prompts and code references
  - Workflow cards in AI-assisted development guide linking to pipeline selection and database building workflows
  - AI-assisted workflow dropdowns in tutorial "Build Your Database" sections for all three pipelines (in-context, hierarchical, middle layer)
  - AI-assisted pipeline selection dropdown before pipeline tab-set in tutorial
  - Enhanced workflows with guidance for AI assistants to read database format code and examples before giving advice
  - Code reference sections showing AI how to use source files for evidence-based recommendations
- **Documentation**: Comprehensive middle layer pipeline guide in Sphinx docs
  - Complete tutorial with architecture comparison and usage examples
  - CLI screenshots and integration examples
  - End-to-end benchmark tests validating complete integration
- **Channel Finder - Sample Data**: Middle layer database and benchmarks
  - 2,033-channel sample database covering 3 systems (SR, BR, BTS)
  - 20 device families with full metadata
  - 35-query benchmark dataset (20% coverage ratio - best of all pipelines)
  - Realistic accelerator physics context
- **Channel Finder - Tools**: Middle layer support across all CLI tools
  - Database preview tool with tree visualization for functional hierarchy
  - CLI query interface with middle_layer pipeline support
  - Benchmark runner with middle_layer dataset support
- **Templates - Channel Finder**: Middle layer configuration support
  - Conditional config generation for middle_layer pipeline
  - Dynamic AVAILABLE_PIPELINES list based on enabled pipelines
  - Database and benchmark paths auto-configured
- **Channel Finder - Middle Layer Testing**: Comprehensive tool and utility tests
  - 480 lines of tests covering all database query tools
  - Tests for prompt loader with middle_layer support
  - Tests for MML converter utility enhancements
- **Channel Finder - Middle Layer**: React agent prompts for functional navigation
  - Query splitter prompt for decomposing complex queries
  - System prompt with database exploration tools
- **Registry Manager**: Silent initialization mode for clean CLI output
  - Suppress INFO/DEBUG logging during initialization when `silent=True`
  - Useful for CLI tools that need clean output without verbose registry logs
- **Channel Finder: Middle Layer Pipeline**: Complete React agent-based channel finder pipeline for MATLAB Middle Layer (MML) databases with System→Family→Field hierarchy; includes MiddleLayerDatabase with O(1) validation and device/sector filtering, MiddleLayerPipeline with 5 database query tools (list_systems, list_families, inspect_fields, list_channel_names, get_common_names), MMLConverter utility for converting Python MML exports to JSON, optional _description fields at all levels for enhanced LLM guidance, comprehensive test suite (14 tests), sample database, and complete documentation

### Changed
- **CLI - Project Initialization**: Enhanced channel finder selection
  - Added middle_layer option to interactive menu
  - Changed default from "both" to "all" (now includes all three pipelines)
  - Updated descriptions for clarity: in_context (<200 channels), hierarchical (pattern-based), middle_layer (functional)
- **Channel Finder - Middle Layer Pipeline**: Migrated from Pydantic-AI to LangGraph
  - Now uses LangGraph's create_react_agent for improved agent behavior
  - Converted tools from Pydantic-AI format to LangChain StructuredTool
  - Enhanced structured output with ChannelSearchResult model
  - Better error handling and agent state management

### Fixed
- **Build Scripts**: Removed trailing whitespace from configuration and script files
- **Testing: Channel Finder test path correction**: Fixed incorrect database path in `test_multiple_direct_signals_fix.py` to point to correct example database location
- **Channel Finder: Multiple direct signal selection**: Fixed leaf node detection to properly handle multiple direct signals (e.g., "status and heartbeat") selected together at optional levels
- **Channel Finder: Optional levels LLM awareness**: Enhanced database descriptions and prompts to better distinguish direct signals from subdevice-specific signals
- **Channel Finder: Separator overrides**: Fixed `build_channels_from_selections()` to respect `_separator` metadata from tree nodes via new `_collect_separator_overrides()` method
- **Channel Finder: Separator overrides with expanded instances**: Fixed `_collect_separator_overrides()` navigation through expanded instance names (e.g., `CH-1`) by checking `_expansion` definitions to find container nodes
- **Channel Finder: Navigation through expanded instances**: Fixed `_navigate_to_node()` and `_extract_tree_options()` to properly handle expanded instances at optional levels - base containers with `_expansion` no longer appear as selectable options, and navigation through expanded instance names works correctly

### Removed
- **Documentation**: Obsolete markdown tutorials for middle layer
  - Content migrated to Sphinx documentation (control-assistant-part2-channel-finder.rst)

## [0.9.8] - 2025-12-19

### Added
- **Testing: Hello World Weather template coverage**: Added comprehensive unit test suite for hello_world_weather template including mock weather API validation, response formatting, and error handling scenarios
- **Hello World Weather: LLM-based location extraction**: Added structured output parser using LLM to extract locations from natural language queries, replacing simple string matching with intelligent parsing that handles nicknames, abbreviations, and defaults to "local" when no location is specified
- **Documentation Version Switcher**: PyData Sphinx Theme version switcher for GitHub Pages with multi-version documentation support; workflow dynamically generates `versions.json` from git tags and preserves historical versions in separate directories (e.g., `/v0.9.7/`, `/latest/`)
- **Developer Workflows System**: New `docs/workflows/` directory with 10 comprehensive workflow guides (pre-merge cleanup, commit organization, release process, testing strategy, AI code review, docstrings, comments, documentation updates) featuring YAML frontmatter metadata and AI assistant integration prompts
- **Custom Sphinx Extension**: `workflow_autodoc.py` extension with `.. workflow-summary::` and `.. workflow-list::` directives for auto-documenting workflow files from markdown with YAML frontmatter, including custom CSS styling
- **Testing: Workflow autodoc extension**: Comprehensive test suite for custom Sphinx extension including frontmatter parsing, directive rendering, and integration tests with actual workflow files
- **Contributing Guide**: Professional `CONTRIBUTING.md` with quick start guide, branch naming conventions, code standards summary, and links to comprehensive documentation
- **CI/CD Infrastructure**: Comprehensive GitHub Actions CI pipeline with parallel jobs for testing (Python 3.11 & 3.12, Ubuntu & macOS), linting (Ruff), type checking (mypy), documentation builds, and package validation
- **Pre-commit Hooks**: `.pre-commit-config.yaml` with Ruff linting/formatting, file quality checks (trailing whitespace, merge conflicts, large files), and optional mypy type checking
- **Dependabot Configuration**: Automated weekly dependency updates for Python packages and GitHub Actions with intelligent grouping (development, Sphinx, LangChain dependencies)
- **Release Automation**: `.github/workflows/release.yml` for automated PyPI publishing using trusted publishing (OIDC), version verification, and optional TestPyPI deployment
- **Pre-merge Check Script**: `scripts/premerge_check.sh` automated scanning for debug code, commented code, hardcoded secrets, missing CHANGELOG entries, incomplete docstrings, and unlinked TODOs
- **Code Coverage Reporting**: Codecov integration in CI pipeline with coverage reports uploaded for Python 3.11 Ubuntu runs
- **Status Badges**: README.md badges for CI status, documentation, code coverage, PyPI version, Python version support, and license

### Changed
- **Code Quality: Comprehensive Linting Cleanup**: Fixed multiple code quality issues across 47 files - B904 exception chaining (30 instances), E722 bare except clauses (5 instances), B007 unused loop variables (4 instances), formatting issues; removed B904 from ruff ignore list and added intentional per-file ignores for test files and example scripts; all changes verified with full test suite (968 unit + 15 e2e tests passing)
- **Code Formatting**: Applied automated Ruff formatting across codebase - modernized type hints to Python 3.10+ style (`Optional[T]` → `T | None`, `List[T]` → `list[T]`), normalized quotes, cleaned whitespace, and removed unused imports; no functional changes
- **Documentation Workflows**: Migrated workflow files from `docs/resources/other/` to `docs/workflows/` with updated references throughout; workflows now feature consistent YAML frontmatter for machine parsing and AI integration
- **Documentation Structure**: Reorganized contributing documentation from placeholder to comprehensive guide with 6 dedicated sections (Getting Started, Git & GitHub, Code Standards, Developer Workflows, AI-Assisted Development, Community Guidelines) using sphinx-design cards and grids
- **Contributing Guide**: Restructured `docs/source/contributing/index.rst` from placeholder to comprehensive 400+ line guide with learning paths, AI integration examples, workflow categories, and automation tools documentation
- **CI Pipeline**: Enhanced documentation job to create preview artifacts for pull requests with 7-day retention; added clear separation between CI checks (`.github/workflows/ci.yml`) and deployment (`.github/workflows/docs.yml`)
- **Development Dependencies**: Added `pytest-cov` to `[dev]` optional dependencies in `pyproject.toml` for code coverage reporting in CI pipeline
- **Hello World Weather: Mock API simplification**: Refactored mock weather API to accept any location string and generate random weather data, removing hardcoded city list and enabling flexible location support for tutorial demonstrations
- **Documentation: Citation update**: Updated paper citation to reflect new title "Osprey: Production-Ready Agentic AI for Safety-Critical Control Systems"
- **Documentation: Framework name cleanup**: Replaced all remaining references to "Alpha Berkeley Framework" with "Osprey Framework" across README, templates, documentation, and test files
- **Testing: E2E hello_world_weather tutorial test**: Enhanced test to exercise both weather AND Python capabilities with a multi-step query that validates configuration defaults, context passing, and code generation/execution workflows
- **Hello World Weather Template**: Enhanced mock weather API with improved error handling and response formatting; updated tutorial documentation for better clarity

### Fixed
- **Configuration: Execution defaults for Python code generation**: Added missing code generator configuration defaults to `ConfigBuilder._get_execution_defaults()`. Now includes `code_generator: "basic"` and corresponding generators configuration, preventing "Unknown provider: None" errors when using Python capabilities in projects with minimal configuration
- **Hello World Weather Template**: Fixed template conditional to include execution infrastructure configuration while excluding only EPICS-specific settings, ensuring Python code generation works out-of-the-box
- **Testing: CI workflow autodoc test collection**: Fixed `ModuleNotFoundError: No module named 'sphinx'` in CI by adding `pytest.importorskip` to `tests/documentation/test_workflow_autodoc.py`; Sphinx is only required for documentation builds and is not part of `[dev]` dependencies, so workflow autodoc tests now gracefully skip when Sphinx is unavailable

### Removed
- **Documentation: Local server launcher**: Removed `docs/launch_docs.py` script; users should use standard Sphinx commands (`make html` and `python -m http.server`) for local documentation builds and serving

## [0.9.7] - 2025-12-14

### Added
- **CLI: Model Configuration Command**: New `osprey config set-models` command to update all model configurations at once with interactive or direct mode
- **Channel Finder: API call context tracking**: Added context tracking to channel finder pipeline for better API call logging and debugging

### Changed
- **Documentation: Python version requirement consistency**: Updated all documentation and templates to consistently specify "Python 3.11+" instead of "Python 3.11", matching the pyproject.toml requirement of `>=3.11`
- **Channel Finder Service**: Improved configuration validation with clearer error messages when channel_finder model is not configured
- **Control Assistant Template: Use Osprey's completion module**: Removed duplicate `completion.py` implementation from channel finder service; now uses `osprey.models.completion` for consistency and maintainability

### Fixed
- **Channel Finder: Optional levels navigation**: Fixed bug where direct signals incorrectly appeared as subdevice options in optional hierarchy levels. The system now correctly distinguishes between container nodes (which belong at the current optional level) and leaf/terminal nodes (which belong to the next level). Also fixed `build_channels_from_selections()` to handle missing optional levels and apply automatic separator cleanup (removes `::` and trailing separators).
- **Hello World Weather Template**: Added service configuration (container runtime, deployed services) to prevent `'services/docker-compose.yml.j2' not found` error when following installation guide
- **Channel Write Capability**: Removed `verification_levels` field from approval `analysis_details` that incorrectly called `_get_verification_config()` method before connector initialization
- **Testing**: Added integration test for channel_write approval workflow to catch capability-approval interaction bugs
- **Testing: Channel Finder registration tests**: Updated test mocks to include `channel_finder` model configuration in the mocked `configurable` dict, fixing tests broken by stricter validation introduced in commit 5834de3
- **Testing: E2E workflow test**: Updated `test_hello_world_template_generates_correctly` to expect services directory and deployment configuration, matching current template structure
- **Testing: E2E benchmark tests**: Fixed registry initialization in `test_channel_finder_benchmarks.py` by calling `initialize_registry()` before creating `BenchmarkRunner` to prevent "Registry not initialized" errors
- **Code Quality**: Pre-merge cleanup - removed unused imports, applied black formatting to 13 files, and documented DEBUG and CONFIG_FILE environment variables in env.example

## [0.9.6] - 2025-12-06

### Added
- **Control Assistant Template: Custom Task Extraction Prompt**: Added control-system-specific task extraction prompt builder that replaces framework defaults with domain-specific examples
  - 14 control system examples covering channel references, temporal context, write operations, and visualization requests
  - Unit test suite verifying custom prompt usage without LLM invocation
  - Documentation in Part 4 tutorial explaining single-point-of-failure importance
- **Channel Finder: Enhanced Database Preview Tool**: Flexible display options for better hierarchy visibility
  - `--depth N` parameter to control tree depth display (default: 3, -1 for unlimited)
  - `--max-items N` parameter to limit items shown per level (default: 10, -1 for unlimited)
  - `--sections` parameter with modular output sections: tree, stats, breakdown, samples, all
  - `--path PATH` parameter to preview any database file directly without modifying config
  - `--focus PATH` parameter to zoom into specific hierarchy branches
  - New `stats` section showing unique value counts at each hierarchy level
  - New `breakdown` section showing channel count breakdown by path
  - New `samples` section showing random sample channel names
  - Backwards compatible `--full` flag support
  - Comprehensive unit tests covering all preview features and edge cases

### Changed
- **Channel Finder: Preview Tool Default Depth**: Default tree display depth increased from 2 to 3 levels for better visibility

### Fixed
- **MCP Server Template: Dynamic timestamps instead of hardcoded dates**: Fixed MCP server generation template to use current UTC timestamps instead of hardcoded November 15, 2025 dates. Prevents e2e test failures due to stale mock data and ensures demo servers return realistic "current" weather data.
- **Tests: Channel Finder unit test updates**: Updated channel finder test files for compatibility with hierarchical database changes (optional levels, custom separators)
- **Tests: Registry mock cleanup and fixture name collisions**: Fixed 7 registry isolation test failures caused by session-level registry mock pollution from capability tests, renamed conflicting test fixtures to prevent pytest naming collisions
- **Python Executor: Context File Creation for Pre-Approval Notebooks**: Fixed timing issue where `context.json` was not created until execution, causing warnings and test failures when approval was required. Context is now saved immediately when creating pre-approval, syntax error, and static analysis failure notebooks.
- **Code Quality: Pre-merge cleanup**: Removed unused imports and applied code formatting standards (black + isort) across entire codebase for consistency
- **Documentation: Fixed RST docstring formatting**: Corrected docstring syntax in `BaseInfrastructureNode.get_current_task()` to use proper RST code block notation (eliminates Sphinx warnings)

### Added
- **Hierarchical Channel Finder: Custom Separator Overrides**: Per-node control of channel name separators
  - New `_separator` metadata field overrides default separators from naming pattern
  - Solves EPICS naming conventions with mixed delimiters (e.g., `:` for subdevices, `_` for suffixes, `.` for legacy subsystems)
  - Backward compatible: nodes without `_separator` use pattern defaults
  - Documentation: New "Custom Separators" tab in Advanced Hierarchy Patterns section
- **Hierarchical Channel Finder: Automatic Leaf Detection**: Eliminates verbose `_is_leaf` markers for childless nodes
  - Nodes without children are automatically detected as leaves (no explicit marker needed)
  - `_is_leaf` now only required for nodes that have children but are also complete channels
  - Reduces verbosity in database definitions (e.g., RB/SP readback/setpoint nodes)
  - Backward compatible: explicit `_is_leaf` markers still work (take precedence)
  - Updated all examples and documentation to reflect cleaner syntax
  - Test coverage: 2 new tests for automatic leaf detection functionality
- **Channel Finder: Comprehensive Parameterized Test Suite**: Automated testing coverage for all example databases
  - New `test_all_example_databases.py` with 80 tests covering all 6 example databases
  - Parameterized tests automatically run on any new example database added
  - Core functionality tests: loading, navigation, channel generation, validation, statistics
  - Database-specific feature tests for unique characteristics (optional levels, legacy format, etc.)
  - Expected channel count validation for all databases (total: 30,908 channels)
  - Now testing previously uncovered databases: `hierarchical_legacy.json` and `optional_levels.json`
  - Suppresses expected deprecation warnings for intentional legacy format testing
- **Channel Finder: Pluggable Pipeline and Database System**: Registration pattern for custom implementations
  - `register_pipeline()` and `register_database()` methods for extending channel finder
  - Discovery API: `list_available_pipelines()` and `list_available_databases()`
  - Config-driven selection without modifying framework code
  - Examples for RAG pipeline and PostgreSQL database implementations
- **Hierarchical Channel Finder: Flexible Naming Configuration**: Navigation-only levels and decoupled naming
  - Naming pattern can reference subset of hierarchy levels (not all required in pattern)
  - New `_channel_part` field decouples tree keys from naming components
  - Enables semantic tree organization with PV names at leaf (JLab CEBAF pattern)
  - Enables friendly navigation with technical naming ("Magnets" → "MAG")
  - Backward compatible: existing databases work unchanged
  - Example database: `hierarchical_jlab_style.json` demonstrating both features
  - Test coverage: 18 new tests for flexible naming functionality

#### Configuration Management
- **EPICS Gateway Presets**: Built-in configurations for APS and ALS facilities
  - APS: pvgatemain1.aps4.anl.gov:5064 (read-only and write-access)
  - ALS: cagw-alsdmz.als.lbl.gov:5064 (read-only), :5084 (write-access)
  - Custom facility support with interactive configuration
- **Configuration Management API**: Programmatic control system and EPICS gateway configuration
  - `get_control_system_type()`, `set_control_system_type()` for runtime connector switching
  - `get_epics_gateway_config()`, `set_epics_gateway_config()` for gateway management
  - `validate_facility_config()` for preset validation
  - Comprehensive test coverage for all configuration operations
- **Unified Configuration Command**: `osprey config` command group following industry standards
  - `osprey config show` - Display current project configuration
  - `osprey config export` - Export framework default configuration
  - `osprey config set-control-system` - Switch between Mock/EPICS connectors
  - `osprey config set-epics-gateway` - Configure EPICS gateway (APS, ALS, custom)
  - Interactive menu integration for guided configuration workflows

#### Control System Operations
- **Runtime Utilities for Control System Operations**: Control-system-agnostic utilities for generated Python code
  - New `osprey.runtime` module with synchronous API (write_channel, read_channel, write_channels)
  - Automatic configuration from execution context for reproducible notebooks
  - Async operations handled internally for simple generated code
  - Works with any control system (EPICS, Mock, etc.) without code changes
  - Complete unit and integration test coverage
  - API reference documentation with usage examples
- **Connector Auto-Verification**: Connectors automatically determine verification level and tolerance from configuration
  - Per-channel verification config from limits database (highest priority)
  - Global verification config from config.yml (fallback)
  - Hardcoded safe defaults if no config available (test environments)
  - New `LimitsValidator.get_verification_config()` method for per-channel lookup
  - Automatic limits validation on all connector writes (no application-level checks needed)
  - Comprehensive test coverage including mock and EPICS connectors
- **Control System Prompt Builders**: Custom prompt builders teaching LLMs to use runtime utilities
  - New ControlSystemPythonPromptBuilder with osprey.runtime documentation
  - Automatic injection of domain-specific instructions into capability prompts
  - Enhanced classifier examples for control system operations
  - Graceful fallback if custom prompts unavailable
  - Comprehensive test coverage for prompt builder integration
  - Complete tutorial on framework prompt customization

#### Testing Infrastructure
- **E2E Test Infrastructure**: Improved test isolation and added warnings to prevent common test failures
  - Added pytest hook to warn users when running `pytest -m e2e` instead of `pytest tests/e2e/`
  - Enhanced registry cleanup in test fixtures to prevent state pollution between tests
  - Added module cleanup in channel finder benchmarks to prevent stale imports
  - Updated README.md, TESTING_GUIDE.md, and tests/e2e/README.md with correct test commands
  - Added critical warnings in pytest.ini about proper e2e test execution
- **Runtime Utilities E2E Tests**: Comprehensive end-to-end test suite validating complete workflows
  - LLM learning osprey.runtime API from prompts
  - Context snapshot preservation and configuration
  - Channel limits safety integration (validates runtime respects boundaries)
  - Positive and negative test cases for write operations
  - Calculation + write workflows (e.g., "set voltage to sqrt(4150)")
- **E2E Test Infrastructure**: Warnings and cleanup mechanisms to prevent state pollution from incorrect test invocation
- **Unit Tests**: Registry isolation and channel finder registration test coverage

#### Documentation
- **EPICS Integration and Configuration Guides**: Comprehensive documentation for production deployment
  - Getting Started: Mock-first workflow with clear migration path to EPICS
  - CLI Reference: Complete `osprey config` command documentation
  - Production Guide: EPICS gateway configuration with facility presets
  - Architecture Guide: Pattern detection security model and design principles
  - API Reference: Framework-standard pattern detection reference
- **Documentation Positioning**: Updated README and tutorials to emphasize production-ready control system focus
  - Highlight plan-first orchestration and control system safety
  - Emphasize protocol-agnostic integration (EPICS, LabVIEW, Tango)
  - Note production deployment at major facilities (LBNL Advanced Light Source)
  - Updated feature list for control system use cases
  - Added comprehensive tutorial section on how generated code interacts with control systems using osprey.runtime
- **Developer Documentation**: Commit organization workflow guide for managing complex Git changes

### Changed

- **Code Quality**: Pre-merge cleanup improvements across codebase
  - Code formatting: Applied Black and isort to all changed files for consistent style
  - Linting fixes: Resolved ruff warnings (unused imports, bare except, unused variables)
  - Logging improvements: Replaced debug print() statements with proper logger.debug() calls in runtime module
  - Type hints: Added return type hints to 6 public functions for better IDE support
  - Import cleanup: Removed duplicate import in memory capability

#### Configuration and Architecture
- **CLI Organization**: Deprecated `osprey export-config` in favor of `osprey config export`
  - Backward compatibility maintained with deprecation notice
  - All configuration operations now unified under `osprey config` namespace
- **Pattern Detection Architecture**: Refactored to framework-standard patterns with security enhancements
  - Control-system-agnostic patterns work across all connector types
  - Comprehensive security coverage detects circumvention attempts (epics.caput, tango.DeviceProxy, etc.)
  - Framework provides sensible defaults; users can override in config.yml
  - Separated approved API patterns (write_channel, read_channel) from direct library call detection
  - `control_system.type` config now only affects runtime connector, not pattern detection
- **Project Templates**: Simplified pattern detection configuration with framework defaults
  - Removed verbose per-control-system pattern definitions
  - Framework automatically provides comprehensive security patterns
  - Clear guidance on when to override patterns (advanced/custom workflows only)
  - Updated README with EPICS gateway configuration instructions
  - Mock-first approach: Projects start in Mock mode, switch to EPICS when ready
- **Dependencies**: Promoted Claude Agent SDK from optional to core dependency
  - Advanced code generation now available in all installations
  - No longer requires separate installation with [claude-agent] extra
  - Minimum framework version 0.9.6+ for Claude Code generator support

#### Control System Connectors and Safety
- **Control System Connector API**: Unified channel naming and comprehensive write verification
  - Method rename: read_pv → read_channel, write_pv → write_channel (deprecated methods emit DeprecationWarning)
  - Class rename: PVValue → ChannelValue, PVMetadata → ChannelMetadata (deprecated classes emit DeprecationWarning)
  - Three-tier write verification: none/callback/readback with configurable tolerance
  - Rich result objects: ChannelWriteResult and WriteVerification with detailed status
  - Mock connector verification simulation for development testing
  - All deprecated APIs will be removed in v0.10
- **Runtime Channel Limits Validation**: Comprehensive safety system for validating writes against configured boundaries
  - Synchronous validation engine with min/max/step/writable constraints
  - Failsafe design blocks all unlisted channels by default
  - Optional max_step checking with I/O overhead warnings
  - Configurable policy modes: strict (error) vs resilient (skip)
  - JSON-based limits database with embedded defaults support
  - New exception: ChannelLimitsViolationError with detailed violation context
- **Python Executor Limits Checking Integration**: Automatic runtime validation of all epics.caput() calls
  - Transparent monkeypatching of epics.caput() and PV.put() methods
  - Embedded validator configuration in wrapper for container isolation
  - Graceful degradation if pyepics unavailable
  - Clear operator feedback with safety status messages

#### Python Execution and Code Generation
- **Python Execution Infrastructure**: Integrated runtime utilities with execution wrapper and notebooks
  - Execution wrapper automatically configures runtime from context snapshots
  - Context manager preserves control system config for reproducible execution
  - Notebooks include runtime configuration cell for standalone execution
  - Proper cleanup in finally block ensures resource release
  - E2E test artifacts now include generated Python code files
  - Developer guide documentation with integration details
- **Channel Write Capability Template**: Simplified by removing limits config loading (now automatic in connector)
  - Capabilities focus on orchestration (parsing, approval)
  - Connectors handle safety (limits, verification)
  - Cleaner separation of concerns

#### Template Configuration and Capabilities
- **Template Configuration**: Updated minimal template and project config for control system safety features
  - Added control_system section with writes_enabled, limits_checking, write_verification
  - Updated integration guides for new connector API
  - Framework capabilities updated for connector method rename
  - Pattern detection updated with new read_channel/write_channel patterns
  - Registry and utility updates for new context types
- **Channel Value Retrieval Renamed to Channel Read**: Renamed `channel_value_retrieval` capability to `channel_read` throughout the entire codebase for consistency and clarity
  - **Capability Name**: `channel_value_retrieval` → `channel_read`
  - **Class Name**: `ChannelValueRetrievalCapability` → `ChannelReadCapability`
  - **File Name**: `channel_value_retrieval.py.j2` → `channel_read.py.j2`
  - **Description**: Updated from "Retrieve current values" to "Read current values"
  - **Documentation**: Updated all references in .rst files, README, and examples
  - **Symmetric Naming**: Now matches `channel_read` (read) / `channel_write` (write) pattern
  - **Registry**: Updated capability registration and context type references
  - **Config**: Updated logging colors and capability lists
- **Channel Write Approval Workflow**: Human-in-the-loop approval for direct control system writes
  - Structured interrupt with operation summary and safety concerns
  - Integration with existing approval_manager and evaluator system
  - Clear approval prompts with channel addresses and target values
  - Resume payload includes complete operation context
- **BaseCapability Helper Method**: get_step_inputs() for accessing orchestrator-provided input contexts
  - Simplifies access to step inputs list from within execute()
  - Handles None values gracefully with configurable defaults
  - Comprehensive tests for various edge cases

#### UI/UX and Documentation Structure
- **CLI Approval Display**: Enhanced approval message presentation with heavy-bordered panel, bold title, and helpful subtitle for improved visibility and user experience
- **Gateway Approval Detection**: Enhanced approval response detection with two-tier system - instant pattern matching for simple yes/no responses, with LLM-powered fallback for complex natural language
- **Documentation Structure**: Refactored Python execution service documentation for improved organization
  - Removed obsolete standalone 03_python-execution-service.rst file
  - Streamlined service-overview.rst (793 → 452 lines, 40% reduction)
  - Focused content on generator extensibility for developers
  - Updated all cross-references to use directory structure
  - Improved navigation and reduced redundancy
- **OpenWebUI**: Enhanced configuration for improved out-of-box experience
  - Auto-configure Ollama and Pipeline connections in docker-compose
  - Disable authentication for local development (WEBUI_AUTH=false)
  - Documentation: automatic vs manual configuration guidance
  - Documentation: Docker vs Podman container networking (host.docker.internal vs host.containers.internal)

#### Miscellaneous
- **Error Node**: Removed deprecated manual streaming code and progress tracking in favor of unified logger system with automatic streaming

### Fixed

- **Runtime Utilities**: Context file now created during pre-approval stage, ensuring configuration access for osprey.runtime

#### Test Infrastructure and Stability
- **E2E Test**: Fixed `test_runtime_utilities_basic_write` by ensuring `context.json` is created during pre-approval stage
  - Context file now created before pre-approval notebook generation in `_create_pre_approval_notebook()`
  - Executor node reuses existing context file instead of recreating it
  - Test properly disables approval workflows for automated e2e execution
- **Test Configuration Pattern Detection**: Removed pattern overrides from test fixtures to use framework defaults
  - Test configs now use complete default patterns from `pattern_detection.py`
  - Fixes approval workflow tests to correctly detect `write_channel`/`read_channel` operations
  - Ensures tests validate actual framework behavior rather than incomplete test-specific patterns
  - Fixed 3 failing tests in `TestApprovalWorkflow` integration test suite
- **E2E Test Stability**: Improved test isolation and removed flaky test
  - Added approval manager singleton cleanup to prevent state pollution between tests
  - Removed redundant `test_runtime_utilities_calculation_with_write` (flaky due to ambiguous LLM prompt)
  - Fixed runtime utilities tests to disable limits checking when testing LLM code generation
  - Corrected config field name from `limits_file` to `database_path`
  - Fixed `_disable_capabilities` helper to properly comment out multi-line capability registrations

#### Validation and Logging
- **Limits validator**: Properly exclude metadata fields (description, source) from unknown field warnings
- **Error Node Logging**: Removed duplicate start/completion logging that occurred when combining decorator's automatic logging with manual status messages

## [0.9.5] - 2025-12-01

### Added
- **CLI Commands**: New `osprey generate claude-config` command to generate Claude Code generator configuration files with sensible defaults and auto-detection of provider settings
- **Interactive Menu**: Added 'generate' command to project selection submenu, centralized menu choice management with `get_project_menu_choices()`, improved consistency between main and project selection flows
- **E2E Test Suites**: Added comprehensive end-to-end test coverage
  - **Claude Config Generation Tests** (`test_claude_config_generation.py`): Validates `osprey generate claude-config` command, tests configuration file structure, provider auto-detection, and profile customization
  - **Code Generator Workflow Tests** (`test_code_generator_workflows.py`): Tests complete code generation pipeline with basic and Claude Code generators. Validates example script guidance following, instruction adherence, and deterministic assertions for generated code content
  - **MCP Capability Generation Tests** (`test_mcp_capability_generation.py`): End-to-end MCP integration testing including server generation/launch, capability generation from live MCP server, registry integration, and query execution with LLM judge verification

### Changed
- **API Call Logging**: Enhanced with caller context tracking across all LLM-calling components. Logging metadata now includes capability/module/operation details for better debugging. Improved JSON serialization with Pydantic model support (mode='json') and better error visibility (warnings instead of silent failures)
- **Claude Code Generator Configuration**: Major simplification - profiles now directly specify phases to run instead of using planning_modes abstraction. Default profile changed from 'balanced' to 'fast'. Unified prompt building into single data-driven `_build_phase_prompt()` method. Reduced codebase by 564 lines through elimination of duplicate prompt builders and dead code
- **Registry Display**: Filtered infrastructure nodes table to exclude capability nodes (avoid duplication with Capabilities table), moved context classes to verbose-only mode, improved handling of tuple types in provides/requires fields
- **MCP Generator Error Handling**: Added pre-flight connectivity checks using httpx, much clearer error messages when server is not running, and actionable instructions in error messages
- **Test Infrastructure**: Added auto-reset registry fixtures in both unit and E2E test conftest files to ensure complete test isolation. Fixtures now reset registry, clear config caches, and clear CONFIG_FILE env var before/after each test to prevent state leakage. Removed manual registry reset calls from individual tests

### Removed
- **Claude Code Generator Profiles**: Removed 'balanced' profile (consolidated to 'fast' and 'robust' only)
- **Claude Code Generator Configuration**: Removed 'workflow_mode' setting (use direct 'phases' list specification), removed 'planning_modes' abstraction (profiles specify phases directly), removed dead code (_generate_direct, _generate_phased, _build_phase_options, 7 duplicate prompt builders)

### Fixed
- **Registry Import Timing**: Fixed module-level `get_registry()` calls that could cause initialization order issues. Moved registry access to runtime (function/method level) in python capability, time_range_parsing capability, generate_from_prompt, and hello_world_weather template
- **Python Executor Logging**: Replaced deprecated `get_streamer` with unified `get_logger` API in code generator node for consistent streaming support
- **MCP Generator Configuration**: Added proper model configuration validation with clear error messages when provider is not configured. Improved error handling with unused variable cleanup and better logging integration
- **Time Range Parsing Tests**: Added mock for `store_output_context` to bypass registry validation, allowing tests to run independently of registry state. Removed obsolete decorator integration tests that were duplicating coverage
- **Tutorial E2E Tests**: Relaxed over-strict plot count assertion (1+ PNG files instead of 2+) to accommodate both single-figure and multi-figure plotting approaches
- **Claude Code Generator Tests**: Refactored to skip low-level prompt building tests (implementation details now covered by E2E tests). Improved test maintainability by focusing on behavior rather than internal methods
- **E2E Test Documentation**: Complete rewrite of tests/e2e/README.md with clearer structure, better isolation guidance, and comprehensive examples. Added warnings about running E2E tests separately from unit tests
- **Documentation**: Updated all Claude Code generator documentation to reflect simplified configuration model. Restructured generator-claude.rst with improved UX using collapsible dropdowns and tabbed sections. Updated all examples to use 'fast' as default profile
- **Tests**: Updated Claude Code generator tests to check 'profile_phases' instead of removed 'workflow_mode', removed tests for removed features, added tests for new phase-based configuration model

### Added
- **Python Executor Service - Complete Modular Refactoring**
  - **Modular Subdirectory Structure**: Reorganized python_executor service into focused subdirectories
    - `analysis/` - Code analysis, pattern detection, and policy enforcement
    - `approval/` - Human approval workflows
    - `execution/` - Container management and code execution
    - `generation/` - Pluggable code generator system
    - Each subdirectory has proper `__init__.py` and dedicated README documentation

  - **Pluggable Code Generator System**: New extensible architecture for code generation
    - **Abstract Interface**: `CodeGenerator` protocol defining standard generator contract
    - **Generator Factory**: Dynamic registration and instantiation with `GeneratorFactory`
    - **Multiple Implementations**:
      * `BasicGenerator` - Simple template-based generation for straightforward tasks
      * `ClaudeCodeGenerator` - Advanced AI-powered generation with:
        - Full conversation history management
        - Result validation and error recovery
        - Streaming support with callbacks
        - Tool use integration
        - Configurable via `execution.code_generator` and `execution.generators` settings
      * `MockGenerator` - Deterministic generator for testing
    - **Registry Integration**: Generator lifecycle managed through framework registry system
    - **State Model Extensions**: `PythonExecutorState` enhanced to support generator configuration

  - **Generator Configuration**: Explicit, flexible configuration structure
    - New `execution.code_generator` setting specifies active generator
    - Generator-specific config in `execution.generators` with model references or inline config
    - Deprecation warnings for old `models.python_code_generator` approach (backward compatible)
    - Updated project templates with examples for all generator types

  - **Integration Enhancements**: Connected generator system to framework
    - Python capability updated to support generator configuration
    - Analysis node enhanced with generator-aware validation
    - Execution pipeline improved for generator output handling
    - Container engine with better error reporting

  - **Comprehensive Test Suite**: Extensive test coverage for new system
    - Unit tests for all generator implementations (BasicGenerator, ClaudeCodeGenerator, MockGenerator)
    - Integration tests for generator-service interaction
    - Pattern detection integration tests
    - Result validation test suites
    - State reducer tests
    - Shared test fixtures and utilities in `tests/services/python_executor/`

  - **CLI and Template Improvements**: Enhanced user experience
    - Generator selection and configuration in interactive menu
    - Template system with generator-specific configurations
    - Claude generator config template (`claude_generator_config.yml.j2`)
    - Example plotting scripts for common use cases (time series, multi-subplot, publication-quality)
    - Improved README templates with generator setup instructions

## [0.9.4] - 2025-11-28

### Added
- **Channel Finder E2E Benchmarks**
  - New benchmark test suite for hierarchical channel finder pipeline
  - Tests query processing across all hierarchy complexity levels
  - Performance metrics: navigation depth, branching factor, channel count
  - Validates correct channel finding across diverse hierarchy patterns
  - Example queries testing system understanding and multi-level navigation
- **Flexible Hierarchical Database Schema**
  - Clean, flexible schema for defining arbitrary control system hierarchies
  - Single `hierarchy` section combines level definitions and naming pattern with built-in validation
  - Support arbitrary mixing of tree navigation (semantic categories) and instance expansion (numbered/patterned devices) at any level
  - Enable multiple consecutive instance levels (e.g., SECTOR→DEVICE, FLOOR→ROOM), instance-first hierarchies, or any tree/instance pattern
  - Automatic validation ensures level names and naming patterns stay in sync (catches errors at load time, not runtime)
  - Each level specifies `name` and `type` (`tree` for semantic categories, `instances` for numbered expansions)
  - Removed redundant/confusing fields from schema (eliminated `_structure` documentation field, consolidated three separate config fields into one)
  - Comprehensive test suite with 33 unit tests including 6 new naming pattern validation tests (all passing)
  - Example databases demonstrating real-world use cases:
    - `hierarchical.json`: Accelerator control (1,048 channels) - SYSTEM[tree]→FAMILY[tree]→DEVICE[instances]→FIELD[tree]→SUBFIELD[tree]
    - `mixed_hierarchy.json`: Building management (1,720 channels) - SECTOR[instances]→BUILDING[tree]→FLOOR[instances]→ROOM[instances]→EQUIPMENT[tree]
    - `instance_first.json`: Manufacturing (85 channels) - LINE[instances]→STATION[tree]→PARAMETER[tree]
    - `consecutive_instances.json`: Accelerator naming (4,996 channels) - SYSTEM[tree]→FAMILY[tree]→SECTOR[instances]→DEVICE[instances]→PROPERTY[tree]
  - Backward compatibility: Legacy databases with implicit configuration automatically converted with deprecation warnings
  - Support hierarchies from 1 to 15+ levels with any combination of types
  - Updated documentation with clean schema examples and comprehensive guides
- **Hello World Weather E2E Test**
  - New end-to-end test validating complete Hello World tutorial workflow
  - Tests weather capability execution, mock API integration, and registry initialization
  - LLM judge evaluation ensures beginner-friendly experience
  - Validates template generation and framework setup for new users

### Changed
- **Test Infrastructure**
  - Fixed test isolation between unit tests and e2e tests using `reset_registry()`
  - Updated all e2e tests to use Claude Haiku (faster, more cost-effective)
  - Separated unit test and e2e test execution to prevent registry mock contamination
  - Updated channel finder tests to use new unified database schema (`"type"` instead of `"structure"`)
  - Documentation: Updated `RELEASE_WORKFLOW.md` with clear instructions for running unit tests (`pytest tests/ --ignore=tests/e2e`) and e2e tests (`pytest tests/e2e/`) separately

## [0.9.3] - 2025-11-27

### Added
- **LLM API Call Logging** - Comprehensive logging of all LLM API interactions for debugging and transparency
  - New `development.api_calls` configuration section with `save_all`, `latest_only`, and `include_stack_trace` options
  - Automatic capture of complete input/output pairs with rich metadata (caller function, module, class, line number, model config)
  - Context variable propagation through async/thread boundaries using Python's `contextvars` for accurate caller detection
  - Intelligent caller detection that skips thread pool and asyncio internals to find actual business logic
  - Integration with classifier and orchestrator nodes via `set_api_call_context()` helper function
  - Capability-aware logging: classifier logs include capability name in filename for parallel classification tasks
  - Files saved to `_agent_data/api_calls/` with descriptive naming: `{module}_{class}_{function}_{capability}_latest.txt`
  - Documentation added to prompt customization guide and configuration reference
  - Complements existing prompt debugging (`development.prompts`) for complete LLM interaction transparency
- **End-to-End Test Infrastructure** - Complete LLM-based testing system for workflow validation
  - New `tests/e2e/` directory with comprehensive e2e test framework
  - **LLM Judge System** (`judge.py`) - AI-powered test evaluation with structured scoring
    - Evaluates workflows against plain-text expectations for flexible validation
    - Provides confidence scores (0.0-1.0) and detailed reasoning
    - Identifies warnings and concerns even in passing tests
  - **E2E Project Factory** (`conftest.py`) - Automated test project creation and execution
    - Creates isolated test projects from templates in temporary directories
    - Full framework initialization with registry, graph, and gateway setup
    - Query execution with complete state management and artifact collection
    - Working directory management for correct `_agent_data/` placement
    - Root logger capture for comprehensive execution trace logging
  - **Tutorial Tests** (`test_tutorials.py`) - Validates complete user workflows
    - `test_bpm_timeseries_and_correlation_tutorial` - Full control assistant workflow (channel finding, archiver retrieval, plotting)
    - `test_simple_query_smoke_test` - Quick infrastructure validation
  - **CLI Options** - Flexible test execution and debugging
    - `--e2e-verbose` - Real-time progress updates during test execution
    - `--judge-verbose` - Detailed LLM judge reasoning and evaluation
    - `--judge-provider` and `--judge-model` - Configurable judge AI model
  - **Comprehensive Documentation** (`tests/e2e/README.md`) - Complete testing guide with examples
  - **Belt and Suspenders Validation** - LLM judge + hard assertions for reliable testing
- **CLI Provider/Model Configuration** - Added `--provider` and `--model` flags to `osprey init` command for configuring AI provider during project creation
- **Unified Logging with Automatic Streaming**
  - Added `BaseCapability.get_logger()` method providing single API for logging and streaming
  - Enhanced `ComponentLogger` with automatic LangGraph streaming support
  - New `status()` method for high-level progress updates
  - Streaming behavior configurable per method with `stream` parameter
  - Smart defaults: `status()`, `error()`, `success()`, `warning()` stream automatically to web UI
  - Detailed logging methods (`info()`, `debug()`) remain CLI-only by default
  - Lazy stream writer initialization with graceful degradation when LangGraph unavailable
  - Custom metadata support via `**kwargs` on all logging methods
  - Automatic step tracking integrated with existing TASK_PREPARATION_STEPS
  - All infrastructure nodes, capabilities, service nodes, and templates migrated to unified pattern
  - Comprehensive test coverage with 26 test cases in `tests/utils/test_logger.py`
  - Backward compatible: existing `get_logger()` and `get_streamer()` patterns continue to work

### Changed
- **Capability Base Class** - Moved exception handling for classifier/orchestrator guide creation to base class properties with warning logs
- **Capability Templates** - Cleaned up unused imports and logger usage in all capability templates (control_assistant, minimal)

## [0.9.2] - 2025-11-25

### 🎉 Major Features

- **Complete Documentation**: Comprehensive docs for new architecture
  - Main python_executor service documentation with architecture overview
  - Per-subdirectory READMEs (analysis, approval, execution, generation)
  - Detailed generator implementation guides:
    * BasicGenerator usage and customization
    * ClaudeCodeGenerator configuration and features
    * MockGenerator for testing
  - Updated developer guides with new modular architecture
  - API reference documentation updates

**Benefits of New Architecture**:
- **Extensibility**: Easy to add new code generators (e.g., Claude Code SDK, GPT-4, custom generators)
- **Testability**: MockGenerator enables deterministic testing without API calls
- **Maintainability**: Clear separation of concerns with modular subdirectories
- **Flexibility**: Swap generators without modifying core service logic
- **Zero Breaking Changes**: Existing configurations continue to work with deprecation warnings

### Fixed
- **Interactive Menu Registry Contamination** ([#29](https://github.com/als-apg/osprey/issues/29))
  - Fixed bug where creating multiple projects in the same interactive menu session caused capability contamination
  - Global registry singleton now properly reset when switching between projects
  - Added `reset_registry()` calls in `handle_chat_action()` before launching chat
  - Prevents second project from inheriting capabilities from first project
  - Added comprehensive test suite to verify registry isolation

#### Argo AI Provider (ANL Institutional Service)
- **New provider adapter** for Argonne National Laboratory's Argo proxy service
- **8 models supported**: Claude (Haiku 4.5, Sonnet 4.5, Sonnet 3.7, Opus 4.1), Gemini (2.5 Flash, 2.5 Pro), GPT-5, GPT-5 Mini
- **OpenAI-compatible interface** with automatic structured output support
- Uses `$USER` environment variable for ANL authentication
- File: `src/osprey/models/providers/argo.py`
- Added `ARGO_API_KEY` to all project templates

#### Infrastructure Node Instance Method Migration
- **All 7 infrastructure nodes** migrated from static method pattern to instance method pattern
- Aligns infrastructure nodes with capability node implementation
- **Decorator Enhancements**:
  - Automatic detection of static vs instance methods (backward compatible)
  - Runtime injection of `_state` for all infrastructure nodes
  - Selective `_step` injection only for in-execution nodes (clarify, respond)
  - Defensive None checks for step injection with warning logs
  - Validation for invalid method types (classmethod, property)
- **Migrated Nodes**:
  - Router: Minimal state usage, routing metadata
  - Task Extraction: Data source integration, state refs updated
  - Classification: Extensive state usage (100+ refs), bypass mode
  - Clarify: First `_step` injection, task_objective extraction
  - Respond: `_step` injection, response generation
  - Error: NO `_step` injection (uses `StateManager.get_current_step_index()`)
  - Orchestration: 200+ lines, nested functions via closure
- **Testing**: Added 15 unit tests for infrastructure pattern
  - Tests validate decorator injection logic (_state, _step)
  - Tests verify backward compatibility with static methods
  - All tests passing

#### Capability Instance Method Pattern Testing
- Added 12 comprehensive tests for migrated capabilities
- New test directory: `tests/capabilities/` with fixtures and integration tests
- Memory Capability Tests (4 tests): signature validation, state/step injection, decorator integration
- Python Capability Tests (3 tests): instance method pattern validation
- TimeRangeParsing Capability Tests (5 tests): full end-to-end integration
- All tests formatted with black and linted

#### Instance Method Pattern for Capabilities
- **New Recommended Pattern**: Capabilities can now use instance methods instead of static methods
  - Helper methods available via `self`: `get_required_contexts()`, `get_task_objective()`, `get_parameters()`, `store_output_context()`
  - Eliminates ~60% of boilerplate code in capability implementations
  - More intuitive and Pythonic API design
  - Full backward compatibility maintained - static methods still work
- **Automatic Context Extraction**: `get_required_contexts()` method with tuple unpacking support
  - Matches order of `requires` field for elegant unpacking: `data, time = self.get_required_contexts()`
  - Falls back to dict access when preferred: `contexts = self.get_required_contexts(); data = contexts["DATA"]`
  - Automatic extraction with cardinality validation
- **New Helper Methods** in `BaseCapability`:
  - `get_required_contexts()` - Extract required contexts with automatic validation
  - `get_task_objective()` - Get current task description
  - `get_parameters()` - Get step parameters
  - `store_output_context()` - Store single output context
  - `store_output_contexts()` - Store multiple output contexts
- **Runtime State Injection**: `@capability_node` decorator injects `_state` and `_step` at runtime
  - Available within `execute()` method context
  - Clean separation between class definition and runtime state
- **Migration Guide**: Comprehensive documentation for upgrading from static to instance pattern
  - Side-by-side code comparisons
  - Migration checklist
  - Common issues and solutions
  - Gradual migration strategy
  - Located at: `docs/source/developer-guides/migration-guide-instance-methods.rst`

### Added
- **Comprehensive Test Suite**: Added 15 tests for capability helper methods
  - Tests for `get_required_contexts()`, `get_task_objective()`, `get_parameters()`
  - Tests for `store_output_context()` and `store_output_contexts()`
  - Error case validation and edge condition handling
  - Located at: `tests/base/test_capability_helpers.py`
- **Prompt-Based Capability Generator**: Natural language capability generation
  - `--from-prompt` CLI option for natural language capability descriptions
  - LLM-powered capability implementation generation
  - Automatic domain inference and classification
- **Test Infrastructure**: Global test utilities for all Osprey tests
  - `create_test_state()` factory for AgentState objects with sensible defaults
  - `PromptTestHelpers` for structural prompt testing
  - Reusable pytest fixtures in `tests/conftest.py`
  - Reduces test boilerplate by 140+ lines per test file
- **Comprehensive Clarification Tests**: 21 tests for clarification prompt generation
  - 10 core functionality tests (prompt structure, content extraction)
  - 9 error handling tests (edge cases, malformed inputs, unicode)
  - 2 integration tests (full workflow validation)
- **Interactive Menu Enhancements**: Version number display in interactive menu banner
- **Stanford AI Playground Provider**: Added Stanford AI playground as a built-in API provider
- **Cardinality Constraints**: New optional cardinality validation in `requires` field
  - Capabilities declare requirements with cardinality: `requires = [("CONTEXT_TYPE", "single")]`
  - Framework automatically validates and raises clear errors if violated
  - Eliminates repetitive `isinstance(context, list)` checks in capability code
  - Options: `"single"` (exactly one), `"multiple"` (must be list), or plain string (any cardinality)
  - Works seamlessly with `get_required_contexts()` helper method
  - Added 9 comprehensive test cases for cardinality validation
  - Updated all framework capability templates to use new pattern

### Changed
- **Generator Architecture**: Refactored monolithic generator into modular design
  - Split MCP capability generator into `BaseGenerator`, `MCPCapabilityGenerator`, and `PromptCapabilityGenerator`
  - Added generator models for type safety and validation
  - Improved CLI with lazy imports for better performance
  - Better separation of concerns and extensibility
- **Clarification System**: Improved prompt structure and context extraction
  - Enhanced clarification prompt builder with better orchestrator integration
  - Improved `get_system_instructions()` method for cleaner prompt composition
  - Better task_objective prioritization in clarification queries
- **Mock Archiver Connector**: Improved BPM position data generation
  - BPM positions now use realistic ±100 µm equilibrium offsets with ±10 µm oscillations
  - Each BPM has unique, reproducible random characteristics based on PV name
  - Slow drift patterns simulate realistic beam position variations
  - Adjusted default noise level from 0.01 to 0.1 for more realistic data
- **Template Updates**: All capability templates now use instance method pattern
  - `hello_world_weather` template updated with new pattern and helper methods
  - `control_assistant` templates (archiver_retrieval, channel_finding, channel_value_retrieval) updated
  - `minimal` template updated to show recommended pattern
  - All templates include proper `requires` field with cardinality constraints

### Fixed
- **Interactive Menu Registry Contamination** ([#29](https://github.com/als-apg/osprey/issues/29))
  - Fixed bug where creating multiple projects in the same interactive menu session caused capability contamination
  - Global registry singleton now properly reset when switching between projects
  - Added `reset_registry()` calls in `handle_chat_action()` before launching chat
  - Prevents second project from inheriting capabilities from first project
  - Added comprehensive test suite to verify registry isolation
- **Stanford API Key Detection**: Added missing STANFORD_API_KEY to environment variable detection (Reported by Marty)
- **Weather Template**: Fixed context extraction example in hello world weather template (PR #26)
- **CRITICAL BUG FIX**: `ContextManager.extract_from_step()` now correctly handles multiple contexts of the same type
  - Previously, when multiple contexts of the same type were requested (e.g., two `CURRENT_WEATHER` contexts), only the last one was returned, causing silent data loss
  - Now returns a list when multiple contexts of the same type exist: `{"CURRENT_WEATHER": [ctx1, ctx2]}`
  - Single contexts still returned as objects for backward compatibility: `{"CURRENT_WEATHER": ctx_obj}`
  - Capabilities can check `isinstance(context, list)` to detect and handle multiple contexts
  - Added 17 comprehensive test cases covering all scenarios
- **Interactive Menu Registry Contamination** ([#29](https://github.com/als-apg/osprey/issues/29))
  - Fixed bug where creating multiple projects in the same interactive menu session caused capability contamination
  - Global registry singleton now properly reset when switching between projects
  - Added `reset_registry()` calls in `handle_chat_action()` before launching chat
  - Prevents second project from inheriting capabilities from first project
  - Added comprehensive test suite to verify registry isolation

### Breaking Changes
- **BREAKING CHANGE**: `BaseCapabilityContext.get_access_details()` signature simplified
  - **Old:** `get_access_details(self, key_name: Optional[str] = None)` with defensive fallback
  - **New:** `get_access_details(self, key: str)` - key parameter is required
  - **Reason:** Framework always provides the key; optional parameter was unnecessary defensive programming
  - **Impact:** Custom context classes must update signature
  - **Migration:** Remove `Optional[str] = None` and fallback logic; use `key` parameter directly
- **BREAKING CHANGE**: `BaseCapabilityContext.get_summary()` signature simplified
  - **Old:** `get_summary(self, key: str)` - required the storage key
  - **New:** `get_summary(self)` - no parameters needed
  - **Reason:** Summaries describe the context data, not storage details
  - **Impact:** Custom context classes must remove `key` parameter
  - **Migration:** Remove `key` parameter from method signature
- **BREAKING CHANGE**: `ContextManager.get_summaries()` now returns `list[dict]` instead of `dict[str, Any]`
  - Simplifies the API by eliminating flattened key format (e.g., `"CONTEXT_TYPE.key"`)
  - Each summary dict already contains a `"type"` field for identification
  - More natural format for UI/LLM consumption
  - Updated 4 consumer files: `respond_node.py`, `clarify_node.py`, `memory.py`, `response_generation.py`

### Documentation
- **Complete Documentation Overhaul**: Updated 20+ documentation files for new patterns
  - API Reference: Updated `BaseCapability` and `BaseCapabilityContext` documentation
  - Developer Guides: Updated all capability and context management guides
  - Quick Start: Updated building-your-first-capability guide with instance pattern
  - Tutorials: Updated hello-world-tutorial with new recommended pattern
  - Example Applications: Updated ALS Assistant and control assistant examples
- **New Migration Guide**: Comprehensive migration documentation
  - Side-by-side pattern comparisons (static vs instance)
  - Step-by-step migration checklist
  - Common migration issues with solutions
  - Gradual migration strategy
  - Testing patterns for migrated capabilities
- **Context Management**: Enhanced context management system documentation
  - Updated with list-handling examples for multiple contexts
  - Added "Handling Multiple Contexts" pattern to integration guide
  - Documented cardinality constraint usage patterns
  - Explained two-phase extraction algorithm

### Migration Notes
- **Early Access Phase**: This is an acceptable breaking change as the framework is in early access (0.9.x)
- **For Capability Developers (Cardinality)**: Replace `isinstance(context, list)` checks with cardinality constraints
  - Old: `constraints=["PV_ADDRESSES"]` + manual `isinstance` check
  - New: `constraints=[("PV_ADDRESSES", "single")]` - framework handles validation
- **For Capability Developers (Multi-Context)**: Add `isinstance(context, list)` validation after extracting contexts to ensure your capability behavior matches expectations (only if not using cardinality constraints)
- **For get_summaries() Consumers**: Update code to iterate over list instead of dict.items()
  - Old: `for key, summary in summaries.items()`
  - New: `for summary in summaries: context_type = summary.get('type')`

## [0.9.1] - 2025-11-16

### Added
- **MCP Capability Generator (Prototype)**: Auto-generate Osprey capabilities from MCP servers
  - `osprey generate capability` command for creating capabilities from MCP servers
  - `osprey generate mcp-server` command for creating demo MCP servers for testing
  - Automatic ReAct agent integration with LangGraph
  - LLM-powered classifier and orchestrator guide generation with examples
  - Interactive registry and config integration with user confirmation
  - Support for FastMCP server generation with weather demo preset
  - Complete end-to-end MCP integration tutorial
  - Dependencies: `langchain-mcp-adapters`, `langgraph`, provider-specific LangChain packages
- **Capability Removal Command**: Clean removal of generated capabilities
  - `osprey remove capability` command for safe capability cleanup
  - Removes registry entries, config models, and capability files
  - Automatic backup creation before modifications
  - Interactive confirmation with preview of changes

### Changed
- **Core Dependencies**: Added `matplotlib>=3.10.3` to core dependencies
  - Python capability visualization now works out of the box without requiring `[scientific]` extras
  - Ensures tutorial examples (plotting beam current, etc.) work immediately after installation
  - Moved from optional `scientific` extras to required dependencies for improved user experience

## [0.9.0] - 2025-11-16

### Added
- **Prompt Customization System**: Flexible inheritance for domain-specific prompt builders
  - Added `include_default_examples` parameter to `DefaultTaskExtractionPromptBuilder`
  - Applications can now choose to extend or replace framework examples
  - Exported `TaskExtractionExample` and `ExtractedTask` from `osprey.prompts.defaults` for custom builders
  - Weather template includes 8 domain-specific examples for conversational context handling
  - New `framework_prompts.py.j2` template demonstrating prompt customization patterns
- **Domain Adaptation Tutorial**: Comprehensive Step 5 in hello-world tutorial
  - Explains why domain-specific examples improve conversational AI
  - 8 weather-specific task extraction examples covering location carry-forward, temporal references, etc.
  - Shows complete implementation with code examples and explanations
  - Demonstrates multi-turn conversation context synthesis
- **Conceptual Tutorial**: New comprehensive tutorial introducing Osprey's core concepts and design patterns
  - Explains Osprey's foundation on LangGraph with link to upstream framework
  - Compares ReAct vs Planning agents with clear advantages/disadvantages
  - Introduces capabilities and contexts with architectural motivation (addressing context window limitations)
  - Walks through designing a weather assistant as practical example
  - Visual grid cards for capability design with color-coded headers
  - Extracted design pattern summary for general application
  - Step-by-step orchestration examples showing how capabilities chain together
  - Location: `docs/source/getting-started/conceptual-tutorial.rst`
- **Control System Connectors**: Two-layer pluggable abstraction for control systems and archivers
  - **MockConnector**: Development/R&D mode - works with any PV names, no hardware required
  - **EPICSConnector**: Production EPICS Channel Access with gateway support (requires `pyepics`)
  - **MockArchiverConnector**: Generates synthetic historical time series data
  - **EPICSArchiverConnector**: EPICS Archiver Appliance integration (requires `archivertools`)
  - **ConnectorFactory**: Centralized creation with automatic registration via registry system
  - **Pattern Detection**: Config-based regex patterns for detecting control system operations in generated code
  - **Plugin Architecture**: Custom connectors (LabVIEW, Tango, etc.) via `ConnectorRegistration`
  - Seamless switching between mock and production via config.yml `type` field
  - Comprehensive API reference and developer guide with LabVIEW example
- **Control Assistant Template**: Production-ready template for accelerator control applications
  - Complete multi-capability system with PV value retrieval, archiver integration, and Channel Finder
  - Dual-mode support (mock for R&D, production for control room)
  - 4-part tutorial series (setup, Channel Finder integration, production deployment, customization)
  - Python execution service with read/write container separation and approval workflows
  - Full documentation with screenshots and step-by-step guides
- **Pattern Detection Service**: Static code analysis for control system operations
  - Configurable regex patterns per control system type
  - Used by approval system to identify read vs write operations
  - Location: `osprey.services.python_executor.pattern_detection`
- **Registry System Enhancements**: Added `ConnectorRegistration` dataclass for connector management
  - Automatic connector registration during framework initialization
  - Lazy loading with unified component management
  - Support for control_system and archiver connector types
- **CLI Template Support**: Added control_assistant template to CLI initialization system
  - New template option in `osprey init` command
  - Interactive menu displays control assistant with description
  - Template validation and configuration support

### Changed
- **FrameworkPromptProviderRegistration API**: Simplified registration interface
  - Removed `application_name` parameter (no longer used by framework)
  - Removed `description` parameter (no longer used by framework)
  - Framework now uses `module_path` as the provider key
  - **Backward Compatible**: Old parameters still accepted with deprecation warnings until v0.10
  - Updated all documentation examples to reflect new simplified API

### Deprecated
- **FrameworkPromptProviderRegistration fields**: `application_name` and `description` parameters
  - Will be removed in v0.10
  - Deprecation warnings emitted when used
  - Migration: Simply remove these parameters from your `FrameworkPromptProviderRegistration` calls

### Removed
- **Migration Guides**: Removed version-specific migration documentation
  - Removed `docs/source/getting-started/migration-guide.rst` (v0.6→v0.8 and v0.7→v0.8 guides)
  - Removed `docs/resources/MIGRATION_GUIDE_v0.6_to_v0.8.md`
  - Removed `docs/resources/MIGRATION_GUIDE_v0.7_to_v0.8.md`
  - Superseded by conceptual tutorial which provides better onboarding for current version
  - Historical migration information still available in git history if needed
- **Wind Turbine Template**: Removed deprecated wind turbine application template
  - Replaced by Control Assistant template with better real-world applicability
  - Removed `src/osprey/templates/apps/wind_turbine/` directory and all associated files
  - Removed `docs/source/getting-started/build-your-first-agent.rst` (superseded by control assistant tutorials)

### Changed
- **Channel Finder Presentation Mode**: Renamed `presentation_mode` value from "compact" to "template"
  - Updated all config files, documentation, and database implementations
  - Method `_format_compact()` renamed to `_format_template()`
- **Hello World Tutorial**: Simplified and improved tutorial UX
  - Removed unnecessary container deployment steps (tutorial only needs `osprey chat`)
  - Added "Ready to Dive In?" admonition for users who want to run first, learn later
  - Added comprehensive API key dropdown matching Control Assistant tutorial format
  - Improved messaging to welcome institutional providers (CBorg, Stanford AI Playground) while recommending Claude Haiku 4.5
  - Simplified prerequisites to focus on essentials (Python, framework, API key)
  - Updated Step 7 from "Deploy and Test" to "Run Your Agent" with streamlined setup
- **Hello World Weather Template**: Simplified template to match minimal tutorial scope
  - Removed container runtime configuration (no containers needed for basic tutorial)
  - Removed safety controls (approval, execution_control) - not relevant for simple weather queries
  - Removed execution infrastructure (EPICS, Jupyter modes, python_executor) - production features only
  - Template system now conditionally generates config sections based on template type
  - Services directory no longer created for hello_world_weather template
  - Generated config.yml now contains only essential sections: project identity, models, API providers, logging
  - Updated template README with streamlined setup instructions and accurate time estimate
  - Test coverage ensures hello_world_weather stays minimal (no production features)
- **Environment Template**: Updated `env.example` with clearer API key guidance
  - Fixed typo: `ANTHROPIC_API_KEY_o` → `ANTHROPIC_API_KEY`
  - Reordered to prioritize Anthropic (recommended) while showing institutional alternatives
  - Added helpful comments about provider flexibility
- **Configuration System**: Enhanced to handle missing configuration sections gracefully
  - Added `_get_approval_config()` with sensible defaults for tutorial environments
  - Added `_get_execution_config()` with local Python execution defaults
  - Removed strict validation in approval_manager that required all sections to be present
  - Enables minimal templates (like hello_world_weather) to work without production-only config sections
  - Provides helpful warnings when using framework defaults instead of explicit configuration
- **Control Assistant Part 3 Documentation**: Improved classification phase explanation
  - Added concrete list of all 6 capabilities (3 framework + 3 application) with file locations
  - Clarified why classification matters: reduces orchestrator context for better latency and accuracy
  - Provided specific YES/NO classification examples for each capability
- **Documentation Build Instructions**: Updated installation.rst to use modern `pip install -e ".[docs]"` workflow
  - Replaced deprecated `pip install -r docs/requirements.txt` approach
  - Uses optional dependencies from pyproject.toml for cleaner package management
- **Dependencies**: Moved `pandas` and `numpy` from optional `scientific` dependencies to base requirements
  - Required by archiver connectors which return pandas DataFrames for time-series data
  - Needed for MongoDB connector support
  - Fixes initialization error when running tutorials from scratch without manual pandas installation
  - `scientific` extra now includes only scipy, matplotlib, seaborn, scikit-learn, and ipywidgets
- **Provider API Key Metadata**: Established providers as single source of truth for API key acquisition information
  - Added `api_key_url`, `api_key_instructions`, and `api_key_note` fields to `BaseProvider`
  - Updated all provider implementations (Anthropic, OpenAI, Google, CBorg, Ollama) with verified metadata
  - Refactored CLI interactive menu to dynamically read API key help from provider metadata
  - Eliminates hardcoded API key instructions in CLI code
  - New providers automatically inherit help system support
  - Follows consistent metadata pattern across framework
- **Configuration API Simplification**: Streamlined `get_model_config()` function signature
  - Removed unused `service` and `model_type` parameters
  - Function now accepts only `(model_name, config_path)` for cleaner API
  - Removed 50+ lines of legacy nested config format support
  - All internal framework calls updated to use new signature
  - Updated documentation examples across 6 files

### Fixed
- **Anthropic Provider Structured Outputs**: Fixed task extraction failures when using Claude Haiku
  - Added structured output support for all Anthropic models (Haiku, Sonnet, Opus)
  - Uses native `response_format` API for Sonnet 4.5 and Opus 4.1 models
  - Falls back to prompt-based JSON extraction for Haiku and older models
  - Resolves `'str' object has no attribute 'task'` error in task extraction
- **Google Provider Structured Outputs**: Added structured output support for Google Gemini models
  - Implements prompt-based JSON extraction with schema validation
  - Fixed health check to use adequate token budget (100 tokens) for models with thinking capabilities
  - Added proper error handling when model uses all tokens for thinking with no output
  - Updated available models list to only include working Gemini 2.5 models (pro, flash, flash-lite)
- **Test Suite**: Updated tests to reflect runtime helper daemon verification improvements (commit 6bf0a1d)
  - Fixed 13 runtime helper tests to expect both compose version and ps daemon checks
  - Fixed 6 connector factory tests with proper registration fixture
  - Removed 4 deprecated wind_turbine template tests
  - All 189 tests now passing
  - Removed Gemini 1.5 models that are not available in current API version
  - Ensures consistent behavior across all LLM providers
- **Jinja2 Template Syntax**: Fixed invalid `.get('KEY')` method calls in Jinja2 templates
  - Replaced `env.get('CBORG_API_KEY')` with `env.CBORG_API_KEY` in conditionals
  - Fixed `env.get('TZ', 'default')` to use proper Jinja2 filter syntax: `env.TZ | default('default')`
  - Affects `project/README.md.j2` and `project/env.j2` templates
  - Resolves "expected token 'end of print statement', got ':'" error during project creation
- **Hello World Tutorial**: Fixed project naming inconsistencies (`weather-demo` → `weather-agent` to match template output)
- **Container Path Resolution**: Fixed database and file paths in containerized deployments
  - Deployment system now automatically adjusts `src/` paths to `repo_src/` (or `/pipelines/repo_src/` for pipelines service) in container configs
  - Fixes channel finder database loading and other file-based resources in containers
  - Simplifies configuration by removing `PROJECT_ROOT` environment variable requirement for basic usage
  - `project_root` now hardcoded in `config.yml` during `framework init` for simpler tutorial experience
  - `PROJECT_ROOT` environment variable remains available for advanced multi-environment deployments
- **Dev Mode Pipeline Container**: Fixed namespace collision by switching from editable source install to wheel-based installation
  - Prevents osprey's `utils` module from shadowing OpenWebUI base image's `/app/utils/pipelines`
- **Container Runtime Detection**: Fixed auto-detection to verify daemon is running, enabling proper fallback from Docker to Podman when Docker Desktop isn't running
- **OrchestratorExample Formatting**: Fixed PlannedStep fields not appearing in orchestrator prompt examples
  - Changed from `getattr()` to `.get()` for TypedDict field access in `OrchestratorExample.format_for_prompt()`
  - Previously resulted in empty `PlannedStep()` blocks, now correctly displays all fields
- **Approval Detection**: Increased max_tokens for approval detection from 10 to 50
  - Critical fix for models that require more tokens to generate complete JSON structures
  - Previously caused "yes" responses to be rejected due to incomplete structured output
  - Ensures reliable approval parsing across all supported models
- **Control Assistant Tutorial Documentation**: Fixed project structure tree and file path inconsistencies
  - Part 1: Removed non-existent `mock_control_system/` and `mock_archiver/` directories (they're in framework, not project)
  - Part 1: Added missing files that are actually generated: `address_list.csv`, benchmark datasets, `llm_channel_namer.py`, `data/README.md`
  - Part 2: Fixed incorrect database output path (`data/processed/` → `data/channel_databases/`)
  - Part 2: Added `CSV_EXAMPLE.csv` reference and clarified distinction between format reference and real UCSB FEL data
  - Documentation now accurately reflects actual generated project structure
- **Environment Variable Substitution**: Added support for bash-style default value syntax `${VAR:-default}`
  - Previously only supported simple `${VAR}` and `$VAR` forms
  - Now properly resolves environment variables with fallback defaults
  - Enables flexible configuration for both local and remote deployments
- **Configuration Override**: Fixed `set_as_default` parameter to properly override existing default config
  - Previously ignored when a default config was already set
  - Now honors explicit caller intent when `set_as_default=True`
  - Fixes issues with CONFIG_FILE environment variable initialization
- **Documentation Build**: Fixed v0.8.5 documentation build failures
  - Resolved compatibility issues with Sphinx build system

### Breaking Changes
- **`get_model_config()` signature changed**: `(model_name, service, model_type, config_path)` → `(model_name, config_path)`
  - **Impact**: Low - Framework model calls are internal and already updated
  - **User Action Required**: Only if you have custom application code using application-specific models
  - **Migration**:
    ```python
    # Old (if you have this in your application code):
    get_model_config('my_app', 'custom_model')

    # New:
    get_model_config('custom_model')
    ```
  - **Note**: Most users will not need to change anything - framework models (orchestrator, classifier, response, etc.) are handled internally

## [0.8.5] - 2025-11-10

### Fixed
- **Python Executor Configuration**: Removed deprecated 'framework' config nesting from python_executor components
- **Subprocess Execution**: Added `CONFIG_FILE` environment variable support for proper registry/context loading in subprocesses
  - Critical fix for execution scenarios where CWD ≠ project root
  - Updated `execution_wrapper.py` to pass config_path to registry initialization
  - Fixed `LocalCodeExecutor` to correctly access python_env_path from flat config structure
- **Exception Handling**: Improved exception chaining with `from e` for better error traceability across multiple modules
- **Configuration Access**: Updated `utils/config.py` to remove legacy nested format references

### Changed
- **Code Quality**: Removed all trailing whitespace (W291, W293) across codebase
- **Formatting**: Applied automatic ruff formatting fixes for consistency
- **Logging**: Improved logging with reduced verbosity and structured formatting
- **CLI**: Extracted duplicate streaming logic into reusable helper method

## [0.8.4] - 2025-11-09

### Added
- **Registry Modes**: Introduced Standalone and Extend modes for application registries
  - **Extend Mode** (recommended): Applications extend framework defaults via `ExtendedRegistryConfig`
    - Framework components loaded automatically (memory, Python, time parsing, etc.)
    - Applications can add, exclude, or override framework components
    - Returned by `extend_framework_registry()` helper function
    - Reduces boilerplate and simplifies upgrades
  - **Standalone Mode** (advanced): Applications provide complete registry including all framework components
    - Framework registry is NOT loaded
    - Full control over all components
    - Used when `RegistryConfig` is returned directly (not via helper)
  - Mode detection is automatic based on registry type (`isinstance(config, ExtendedRegistryConfig)`)
- **New Class**: `ExtendedRegistryConfig` marker class for signaling Extend mode
  - Subclass of `RegistryConfig` with identical fields
  - Type-based detection enables automatic framework merging
  - Added to `__all__` exports in `osprey.registry`
- **New Helper Function**: `generate_explicit_registry_code()` for template generation
  - Generates complete registry Python code combining framework + app components
  - Used by CLI template system for creating explicit registries
  - Useful for applications that want full visibility of all components
  - Takes app metadata and component lists, returns formatted Python source code
- Comprehensive test suite for registry modes (500+ lines across 4 new test files)
  - `test_registry_modes.py`: Tests for Extend vs Standalone mode detection
  - `test_registry_loading.py`: Tests for registry loading mechanisms
  - `test_registry_helpers.py`: Tests for helper functions
  - `test_registry_validation.py`: Tests for registry validation

### Changed
- **Registry Helper**: `extend_framework_registry()` now returns `ExtendedRegistryConfig` instead of `RegistryConfig`
  - Backward compatible (ExtendedRegistryConfig is a subclass of RegistryConfig)
  - Type signature change enables automatic mode detection
  - Applications using type hints should update return type annotation
- Enhanced registry documentation with comprehensive coverage of both modes
  - Developer guide updated with mode selection guidance
  - API reference documentation expanded with ExtendedRegistryConfig details
  - Code examples updated to show ExtendedRegistryConfig return type

### Breaking Changes
- **RegistryManager Constructor**: Parameter changed from `registry_paths: List[str]` to `registry_path: Optional[str]`
  - **Impact**: Low - most applications use `initialize_registry()` which reads from config
  - **Migration**: Change `RegistryManager([path1, path2])` to `RegistryManager(path)` for single registry
  - **Rationale**: Simplified to single-application model matching actual usage patterns
  - Framework now supports one application registry per instance (loaded from `config.yml`)
- **Type Signature**: `extend_framework_registry()` return type changed to `ExtendedRegistryConfig`
  - **Impact**: Very low - backward compatible at runtime (subclass relationship)
  - **Migration**: Update type hints from `-> RegistryConfig` to `-> ExtendedRegistryConfig`
  - Only affects code using explicit type checking

### Removed
- Test file `test_path_based_discovery.py` (replaced with mode-specific tests)

### Developer Notes
- Registry system now uses type-based mode detection for cleaner separation of concerns
- Standalone mode enables minimal deployments and custom framework variations
- Extend mode remains the recommended default for >95% of applications
- See developer guide "Registry and Discovery" for complete mode selection guidance

## [0.8.3] - 2025-11-09

### Added
- **Docker Runtime Support**: Framework now supports both Docker and Podman container runtimes
  - New `runtime_helper.py` module for automatic runtime detection
  - Configuration setting `container_runtime` in `config.yml` (options: `auto`, `docker`, `podman`)
  - Environment variable `CONTAINER_RUNTIME` for per-command runtime override
  - Auto-detection prefers Docker first, falls back to Podman
  - Requires Docker Desktop 4.0+ or Podman 4.0+ (native compose support)
  - **User-friendly error messages**: Platform-specific guidance when Docker/Podman not running
    - macOS: "Open Docker Desktop from Applications" with menu bar icon hints
    - Linux: systemctl commands and docker group permissions
    - Windows: Start menu and system tray instructions
  - Comprehensive test suite for runtime detection and selection (33 tests)
- **Custom AI Provider Registration**: Applications can now register custom AI model providers through the registry system
  - Added `providers` parameter to `extend_framework_registry()` helper function
  - Added `exclude_providers` parameter to exclude framework providers
  - Added `override_providers` parameter to replace framework providers with custom implementations
  - Provider merging support in `RegistryManager._merge_application_with_override()`
  - Comprehensive test suite (16 tests) covering all provider registration scenarios
  - Support for institutional AI services (Azure, Stanford AI, national lab endpoints) and commercial providers

### Changed
- **Container Management**: All deployment commands now use runtime abstraction layer
  - Updated `container_manager.py`: 6 functions now use runtime helper
  - Updated `health_cmd.py`: Container health checks are runtime-agnostic
  - Updated `interactive_menu.py`: Mount checking uses configured runtime
  - All compose operations work seamlessly with both Docker and Podman
  - **Fixed JSON parsing**: `osprey deploy status` now handles both Docker (NDJSON) and Podman (JSON array) output formats
- **Dependencies**: Removed Python `podman` and `podman-compose` packages
  - Container runtimes must be installed via system package managers
  - Framework uses CLI tools (`docker`/`podman` commands), not Python SDKs
  - Added installation documentation for both runtimes
- Enhanced registry helper functions to support provider registration parameters
- Updated developer guide documentation with provider registration examples

### Breaking Changes
- **Installation**: Users must install Docker Desktop 4.0+ or Podman 4.0+ separately
  - Python packages no longer provide container runtime functionality
  - See installation guide for platform-specific instructions
- **Note**: Existing Podman users are unaffected - auto-detection will find Podman if Docker not installed

## [0.8.2] - 2025-11-05

### Added
- Registry display command (`osprey registry`) with themed output
- Rebuild and clean deployment actions in interactive menu with safety confirmations
- Strategic test suite: 22 tests covering logging filters and container status logic
- OSPREY_QUIET environment variable for subprocess noise reduction
- Helper functions for status table creation (reduced duplication)

### Changed
- **Complete CLI style migration**: All commands now use centralized Styles constants
- **Container manager logging**: Converted 53 print statements to ComponentLogger system
- **Status command rewrite**: Now uses direct `podman ps` for more reliable state checking
- Improved service name matching with underscore/hyphen variation handling
- Enhanced log suppression using quiet_logger for cleaner CLI output
- Theme-aware command completer using active theme colors
- Condensed verbose comments for better code readability

### Fixed
- Container status display now works independently of compose files
- Smart container-to-project matching with backward compatibility
- Proper separation of project vs non-project containers in status display
- CONFIG logger properly suppressed in interactive menu operations

### Improved
- Net change: -641 lines through cleanup and consolidation
- Better error handling in status command with timeout protection
- Enhanced maintainability through style consistency
- Clearer deployment operation confirmations for destructive actions

## [0.8.1] - 2024-11-04

### Fixed
- Post-release fixes and improvements from initial 0.8.0 testing
- Package distribution and metadata updates

### Changed
- Final production release of Osprey Framework rebrand
- Improved documentation and migration guides
- Enhanced CLI theme system consistency

## [0.8.0] - 2025-11-02

### 🦅 Major Changes - Rebranding to Osprey Framework

**BREAKING CHANGES:**
This release represents a complete rebranding of the project from "Alpha Berkeley Framework" to "Osprey Framework".

**Package & Installation Changes:**
- **Package name:** `alpha-berkeley-framework` → `osprey-framework`
  - Install with: `pip install osprey-framework` (note: hyphen in package name)
  - PyPI URL: https://pypi.org/project/osprey-framework/
- **Import paths:** `from framework.*` → `from osprey.*`
  - All Python imports updated throughout codebase
  - Example: `from osprey.state import AgentState`
- **CLI command:** `framework` → `osprey`
  - New primary command: `osprey init`, `osprey chat`, `osprey deploy`, etc.
  - Legacy `alpha-berkeley` commands maintained for backward compatibility
- **Repository:** `thellert/alpha_berkeley` → `als-apg/osprey`
  - New GitHub repository: https://github.com/als-apg/osprey
  - Old URLs automatically redirect via GitHub
- **Documentation:** https://als-apg.github.io/osprey

**Migration Guide:**

For existing users upgrading from Alpha Berkeley Framework:

1. **Uninstall old package:**
   ```bash
   pip uninstall alpha-berkeley-framework
   ```

2. **Install new package:**
   ```bash
   pip install osprey-framework
   ```

3. **Update imports in your code:**
   - Find and replace: `from framework.` → `from osprey.`
   - Find and replace: `import framework` → `import osprey`

4. **Update CLI commands:**
   - Replace `framework` with `osprey` in scripts and documentation
   - Example: `framework init` → `osprey init`

5. **Update project dependencies:**
   - In `requirements.txt`: `alpha-berkeley-framework` → `osprey-framework`
   - In `pyproject.toml`: `alpha-berkeley-framework` → `osprey-framework`

**Note:** GitHub automatically redirects old repository URLs. However, we recommend updating your git remotes for long-term stability:
```bash
git remote set-url origin https://github.com/als-apg/osprey.git
```

**Technical Details:**
- 134 Python files updated with new imports
- 67 documentation files updated with new branding
- All templates updated to generate osprey-based projects
- Package structure: `src/osprey/` (was `src/framework/`)
- Distribution files: `osprey_framework-0.8.0.whl` (underscore is automatic)

### Includes All Features from v0.7.7 and v0.7.8
- Interactive TUI menu system
- Multi-project support
- Enhanced documentation
- All bug fixes from previous releases

---

## [0.7.8] - 2025-11-01

### Fixed
- Fixed config system test failure by correcting global variable references (`_default_config` and `_default_configurable`)
- Enhanced `get_config_value()` function to fall back to raw config when path not found in processed configurable dict
- Updated template documentation to clarify "Example categories" instead of "Valid categories"

## [0.7.7] - 2025-11-01

### Added
- **Interactive Terminal UI (TUI)** - Comprehensive menu system for guided workflows
  - New `interactive_menu.py` (1,771 lines) - Main TUI implementation with context-aware menus
  - Context detection: Automatically adapts interface based on whether user is in a project directory
  - Interactive project initialization with template, provider, and model selection
  - Automatic API key detection from shell environment
  - Secure password-style input for API keys not found in environment
  - Beautiful Rich-formatted interface with colors and styled panels
  - Smart defaults based on detected environment variables
  - Seamless integration with existing Click commands
- **Multi-Project Support** - Work seamlessly across multiple framework projects
  - New `project_utils.py` (90 lines) - Unified project path resolution utilities
  - `--project` flag added to all CLI commands (init, chat, deploy, health, export-config)
  - `FRAMEWORK_PROJECT` environment variable support for persistent project selection
  - Three ways to specify project: current directory, --project flag, or env var
  - Explicit `config_path` parameter throughout configuration system
  - Per-path config caching for efficient multi-project workflows
  - Registry path resolution relative to config file location
  - Project isolation with no cross-project configuration contamination
- **Provider Descriptions** - User-friendly provider identification
  - Added `description` field to `BaseProvider` abstract class
  - All provider adapters updated with descriptions for TUI menus:
    - anthropic: "Anthropic (Claude models)"
    - openai: "OpenAI (GPT models)"
    - google: "Google (Gemini models)"
    - ollama: "Ollama (local models)"
    - cborg: "LBNL CBorg proxy (supports multiple models)"
- **Environment Variable Auto-Detection** - Intelligent project initialization
  - `_detect_environment_variables()` method in TemplateManager
  - Automatically detects API keys from system environment
  - Updates `.env.example` template with detected values
  - Displays detected environment variables during init command
  - Falls back to placeholder values if vars not found
- **Enhanced Container Status Display** - Professional formatted output
  - Rich table formatting for `framework deploy status` command
  - Colored status indicators with emoji (● Running / ● Stopped)
  - Health status display (healthy/unhealthy/starting) when available
  - Clear port mapping display (host→container format)
  - JSON-based parsing for structured container information
  - Helpful guidance when no services are running

### Changed
- **Configuration System** - Enhanced for multi-project scenarios
  - All config utility functions now accept optional `config_path` parameter:
    - `get_model_config()`, `get_provider_config()`, `get_framework_service_config()`
    - `get_config_value()`, `get_full_configuration()`
  - Implemented per-path config caching for performance
  - Added `set_as_default` parameter for explicit path handling
  - Maintains backward compatibility with singleton pattern
- **Registry Manager** - Enhanced path resolution
  - Added `config_path` parameter to `get_registry()` and `initialize_registry()`
  - Resolve relative registry paths against config file location
  - Pass config_path when initializing registry components
  - Better base path resolution for registry files
- **Data Source Manager** - Improved logging and status tracking
  - Enhanced logging to distinguish empty vs. failed data sources
  - Track sources that succeed but return no data
  - Better summary format: "Data sources checked: 3 (1 with data, 1 empty, 1 failed)"
  - Clearer UX for understanding data availability and debugging
- **Default Model Selection** - Better out-of-box experience
  - Changed default model from `gemini-2.0-flash-exp` to `claude-3-5-haiku-latest`
  - Better performance and reliability for common tasks
  - Lower latency and more consistent responses
- **Docker Compose Templates** - Cleaner initial configuration
  - Optional settings now commented out by default in generated templates
  - Simpler initial setup for new projects
  - Easy to uncomment and enable advanced features when needed
- **Template Manager** - Enhanced for TUI integration
  - Exported key functions for reuse by interactive menu
  - Better separation of concerns between CLI and TUI
  - Improved error handling and validation

### Enhanced
- **CLI Commands** - Unified project path resolution
  - All commands updated with `--project` flag support
  - Consistent path resolution using new `resolve_project_path()` utility
  - Better error messages when project path invalid
  - Project-aware command execution throughout
- **User Experience** - Professional terminal interface
  - Questionary library integration for interactive prompts
  - Custom styling matching framework theme
  - Rich console output with formatted panels and tables
  - Helpful guidance and next-step suggestions
  - Lower barrier to entry for new users
  - Faster workflows for common tasks

### Technical Details
- **TUI Architecture** - ~1,900 lines of new code
  - Context-aware menu system with adaptive interface
  - Integration with TemplateManager for project scaffolding
  - Integration with registry system for provider metadata
  - Direct function calls (not Click commands) for efficiency
  - Optional dependency on questionary (graceful fallback)
- **Multi-Project Infrastructure** - ~300 lines of enhanced code
  - Configuration system: per-path caching and explicit path support
  - Registry manager: config-aware initialization and path resolution
  - Data management: config path propagation
  - CLI commands: unified path resolution utility
- **Zero Breaking Changes** - Complete backward compatibility
  - All existing CLI commands work unchanged
  - TUI only activates when no arguments provided
  - Direct commands remain primary interface for power users
  - Configuration system maintains singleton pattern when no path specified

## [0.7.6] - 2025-10-30

### Added
- **Provider Registry System** - Centralized AI provider management integrated into framework registry
  - New `ProviderRegistration` dataclass for minimal provider metadata (module_path, class_name only)
  - Provider metadata introspected from class attributes (single source of truth)
  - Registry methods: `get_provider()`, `list_providers()`, `get_provider_registration()`
  - Providers added to component initialization order (early loading for use by capabilities)
- **Provider Adapter Architecture** - New base class and five framework provider implementations
  - `BaseProvider` abstract class defining provider interface (create_model, execute_completion, check_health)
  - `AnthropicProviderAdapter` with extended thinking support
  - `OpenAIProviderAdapter` with structured outputs and token parameter handling
  - `GoogleProviderAdapter` with extended thinking support
  - `OllamaProviderAdapter` with automatic localhost ↔ host.containers.internal fallback
  - `CBorgProviderAdapter` for LBNL institutional AI service
- **Log Filtering Utilities** - Dynamic log suppression system
  - New `framework.utils.log_filter` module with `LoggerFilter` class
  - Context managers: `suppress_logger()`, `suppress_logger_level()`, `quiet_logger()`
  - Filter by logger name, level, message patterns, or combinations
  - Thread-safe with pre-compiled regex patterns for performance
- **Testing Infrastructure** - pytest configuration and VCR support
  - New `pytest.ini` with test markers (unit, integration, requires_api, vcr, etc.)
  - `tests/cassettes/` directory with comprehensive README for VCR usage
  - Test markers for provider-specific tests (requires_openai, requires_anthropic, etc.)
  - Added pytest-vcr and vcrpy to dev dependencies
- **Custom Provider Registration** - Applications can register institutional/commercial providers
  - Full documentation with Azure OpenAI example in registry guide
  - Support for institutional AI services (Stanford AI Playground, national lab endpoints)
  - Support for commercial providers (Cohere, Mistral AI, Together AI, etc.)

### Changed
- **Model Factory Refactoring** - Simplified to use provider registry (~280 lines removed)
  - Replaced hardcoded provider requirements dict with registry lookups
  - Use `provider.create_model()` for all provider types
  - Removed `_create_openai_compatible_model()` helper function
  - Removed `_get_ollama_fallback_urls()` and `_test_ollama_connection()` (moved to OllamaProviderAdapter)
  - Validation uses provider class metadata instead of hardcoded dict
- **Completion Module Refactoring** - Streamlined to use provider registry (~290 lines removed)
  - Replaced all provider-specific if/elif blocks with `provider.execute_completion()`
  - Removed provider requirements validation dict
  - Removed `_get_ollama_fallback_urls()` helper
  - Added `temperature` parameter to completion function
- **Health Check Refactoring** - Updated to use provider registry (~290 lines removed)
  - Initialize registry before provider checks with loading spinner
  - Use `provider.check_health()` instead of provider-specific logic
  - Removed all provider-specific if/elif blocks
  - Removed `_test_provider_connectivity()` method
  - Added `quiet_logger` usage to suppress verbose registry initialization logs
  - Preserved charge-avoiding health checks for Anthropic and Google
- **Memory Storage Logging** - Improved logging consistency
  - Switched from root logger to framework logger (`get_logger("memory_storage")`)
  - Changed initialization message from INFO to DEBUG level
  - Reduced log verbosity for non-critical operations
- **Module Exports** - Cleaned up framework.models exports
  - Removed `_create_openai_compatible_model` from public API

### Fixed
- **Test Import Paths** - Updated config imports for relocated modules
  - Changed from `configs.config` to `framework.utils.config`
  - Added minimal config.yml in tests to prevent loading errors
  - Updated integration tests for new configuration module location

### Removed
- **Deprecated Code Cleanup**
  - Deleted `src/framework/interfaces/openwebui/` (deprecated interface implementation)
  - Deleted `docs/resources/other/EXECUTION_POLICY_SYSTEM.md` (outdated design document)

### Documentation
- **Provider Registry Documentation** - Comprehensive documentation for new system
  - Added `ProviderRegistration` to registry API reference
  - Custom provider registration guide with complete Azure OpenAI example
  - Updated component initialization order in all docs
  - Added provider access methods to RegistryManager documentation
  - Updated configuration docs with custom provider extensibility information
- **README and Examples** - Updated with custom provider examples
  - Common use cases: Azure OpenAI, institutional services, commercial providers
  - Integration with `get_model()` and `get_chat_completion()`
  - Health check system integration

### Technical Details
- **Code Reduction**: ~860 lines removed from factory.py, completion.py, health_cmd.py
- **New Code**: ~1,090 lines of well-structured provider adapter implementations
- **Net Result**: More maintainable, extensible architecture with single source of truth
- **Zero Breaking Changes**: All existing APIs remain unchanged

## [0.7.5] - 2025-10-28

### Added
- **Parallel Capability Classification** - Multiple capabilities now classified simultaneously using `asyncio.gather()`
  - New `CapabilityClassifier` class for individual capability processing with proper resource management
  - Semaphore-controlled concurrency to prevent API flooding while maintaining performance
  - Configurable `max_concurrent_classifications` setting (default: 5) in `execution_control.limits`
  - Enhanced error handling for individual classification failures
- **Improved Reclassification Logic** - New `_detect_reclassification_scenario()` function
  - Better detection of reclassification scenarios from error state
  - Cleaner error state cleanup during reclassification
  - Enhanced logging for reclassification process
- **New Configuration Function** - `get_classification_config()` for accessing classification settings
- **Documentation Build System** - Added `docs/config.yml` for documentation build compatibility

### Changed
- **Classification Architecture** - Refactored from sequential to parallel processing
  - `select_capabilities()` now uses parallel task execution with semaphore control
  - Removed old `_classify_capability()` function in favor of `CapabilityClassifier` class
  - Improved error handling and logging throughout classification process
- **Router Logic** - Simplified reclassification handling
  - Removed manual state setting in router, moved responsibility to classifier
  - Cleaner separation of concerns between router and classifier
- **State Management** - Enhanced agent control state with new classification limits
  - Added `max_concurrent_classifications` to `AgentControlState`
  - Updated state manager defaults and configuration builder

### Fixed
- **Documentation Build System** - Updated for pip-installable framework structure
  - Changed from `requirements.txt` to `pip install -e ".[docs]"` in Makefile and GitHub Actions
  - Added mock imports for documentation build compatibility
  - Fixed dropdown syntax and removed unused CSS rules
- **Installation Guide** - Added docs extras install option and fixed formatting
- **Command Help Text** - Fixed escaped newlines in command help strings

## [0.7.4] - 2025-10-27

### Fixed
- **Template Registry Class Names** - Fixed duplicate "RegistryProvider" suffix in generated registry class names
  - Class name generation now produces correct names like `WeatherTutorialRegistryProvider` instead of `WeatherTutorialRegistryProviderRegistryProvider`
  - Updated `_generate_class_name()` method to return PascalCase prefix only
  - Templates correctly append "RegistryProvider" suffix
  - Affects all three app templates: hello_world_weather, wind_turbine, minimal
- **Template Import Paths** - Updated documentation examples to use v0.7.0 import patterns
  - Changed from `applications.hello_world_weather.*` to `hello_world_weather.*`
  - Updated mock_weather_api.py documentation examples
  - Updated capabilities/__init__.py documentation and Sphinx references
  - Ensures generated projects follow correct v0.7.0 decoupled architecture
- **Requirements Template Rendering** - Fixed framework version substitution in generated requirements.txt
  - Moved requirements.txt from static files to rendered templates
  - Now properly replaces `{{ framework_version }}` placeholder with actual version
  - Ensures generated projects pin correct framework version in requirements.txt

## [0.7.3] - 2025-10-26

### Added
- **Development Mode Support** - New `--dev` flag for deploy CLI command
  - Local framework override capability for seamless development testing
  - Smart dependency installation in containers with dev mode detection
  - Automatic local framework installation when DEV_MODE is enabled

### Changed
- **Container Deployment** - Enhanced service templates and deployment workflow
  - Project templates now use PyPI framework distribution by default
  - Removed hardcoded framework paths from configuration templates
  - Improved container startup scripts with better logging and error handling
  - Changed container restart policy to 'no' for better development experience
- **Project Templates** - Automatic framework dependency management
  - Added framework dependency to generated `pyproject.toml` and `requirements.txt`
  - Created proper agent data directory structure for container mounts
  - Enhanced fallback mechanisms for missing requirements files

### Fixed
- **Container Manager** - Improved registry path resolution for different service types
- **Environment Handling** - Graceful .env file handling with fallback warnings
- **Mount Points** - Ensure container mount directories exist before deployment

## [0.7.2] - 2025-10-26

### Changed
- **Simplified Installation** - PostgreSQL dependencies moved to optional `[postgres]` extra
  - Basic framework now installs without PostgreSQL requirements
  - Uses in-memory checkpointing by default (perfect for development/testing)
  - Production users can install `alpha-berkeley-framework[postgres]` for persistent state
  - Resolves installation issues on systems without PostgreSQL packages

## [0.7.1] - 2025-10-26

### Added
- **Centralized Slash Command System** - Unified command registry for CLI and web interfaces
  - Command categorization (CLI, agent control, service commands)
  - Autocompletion and help system
  - Context-aware command execution

### Changed
- Enhanced CLI health command with command system integration
- Updated gateway architecture for command processing
- Improved state management for command execution context

## [0.7.0] - 2025-10-25

### 🎉 Major Architecture Release - Framework Decoupling

This is a **major breaking release** that fundamentally changes how applications are built and deployed. The framework is now pip-installable, enabling independent application development in separate repositories.

### Added

#### Unified CLI System
- **`framework` command** - Main CLI entry point with lazy loading for fast startup
- **`framework init`** - Create new projects from templates with project scaffolding
  - Templates: minimal, hello_world_weather, wind_turbine
  - Options: `--template`, `--registry-style`, `--output-dir`, `--force`
- **`framework deploy`** - Manage Docker services (up/down/restart/status/rebuild/clean)
  - Intelligent service management with validation
  - Service health checking
- **`framework chat`** - Interactive CLI conversation interface
- **`framework health`** - Comprehensive system diagnostics
  - Validates Python version, dependencies, configuration, registry files, containers
  - ~968 lines of diagnostic code
- **`framework export-config`** - View framework default configuration template
  - Supports YAML and JSON output
  - Helps understand configuration options

#### Template System
- **3 Production-Ready Templates** - Instant project generation
  - `minimal` - Bare-bones starter with TODO placeholders
  - `hello_world_weather` - Simple weather query example
  - `wind_turbine` - Complex multi-capability monitoring system
- **Project Scaffolding** - Complete self-contained projects
  - Application code (capabilities, registry, context classes)
  - Service configurations (Jupyter, OpenWebUI, Pipelines)
  - Self-contained configuration (~320 lines)
  - Environment template (.env.example)
  - Dependencies file (pyproject.toml)
  - Getting started documentation

#### Registry Helper Functions
- **`extend_framework_registry()`** - Simplify application registries by ~70%
  - Compact style: 5-10 lines instead of 80+ lines of boilerplate
  - Automatic framework component inclusion
  - Clean exclusion syntax: `exclude_capabilities=["python"]`
  - Optional override support for advanced customization
- **`get_framework_defaults()`** - Inspect framework components
- **Progressive disclosure** - Start simple, go explicit when needed

#### Path-Based Discovery
- **Explicit registry file paths** in `config.yml`
- **`registry_path`** configuration (top-level or nested)
- **`importlib.util` based loading** - Robust module loading
- **Temporary sys.path manipulation** - Like Django, Sphinx, Airflow
- **Strict validation** - Exactly one `RegistryConfigProvider` per file
- **Rich error messages** - Comprehensive resolution hints

#### Self-Contained Configuration
- **One `config.yml` per application** - Complete transparency
- **~320 lines** - All framework settings visible and editable
- **Framework defaults included** at project creation
- **`.env` file support** - Automatic loading with python-dotenv
- **Well-organized** - Clear section comments for easy navigation

#### Documentation
- **Migration Guide** - Comprehensive upgrade documentation (~730 lines)
  - Breaking changes overview
  - Step-by-step migration instructions (10 steps)
  - Production and tutorial migration paths
  - Common issues and solutions
  - Migration progress checklist
- **Updated Getting Started** - Fresh installation and migration paths
- **CLI Reference** - Complete command documentation
- **Registry Helper Documentation** - Helper function usage and examples

### Changed

#### Breaking Changes - Repository Structure
- **Framework** → Pip-installable package (`alpha-berkeley-framework`)
- **Applications** → Separate repositories (production) or templates (tutorials)
- **`interfaces/`** → `src/framework/interfaces/` (pip-installed)
- **`deployment/`** → `src/framework/deployment/` (pip-installed)
- **`src/configs/`** → `src/framework/utils/` (merged)

#### Breaking Changes - Import Paths
```python
# OLD ❌
from applications.my_app.capabilities import MyCapability

# NEW ✅
from my_app.capabilities import MyCapability
```

All `applications.*` imports must be updated to package names.

#### Breaking Changes - CLI Commands
```bash
# OLD ❌
python -m interfaces.CLI.direct_conversation
python -m deployment.container_manager deploy_up

# NEW ✅
framework chat
framework deploy up
```

#### Breaking Changes - Configuration
- **Per-application config** - Each app has own `config.yml`
- **No global framework config** - Self-contained configuration
- **`registry_path` required** - Explicit registry file location
- **All settings visible** - Complete transparency (~320 lines)

#### Breaking Changes - Discovery
- **Explicit path-based discovery** - No automatic `applications/` scanning
- **Registry must be importable** - Proper Python package structure required
- **Exactly one provider per file** - Strict enforcement

### Enhanced

#### Performance
- **Lazy Loading CLI** - Heavy dependencies loaded only when needed
- **Fast Help Display** - `framework --help` loads instantly
- **Immediate Code Changes** - No reinstall/rebuild required

#### Developer Experience
- **Template-Based Generation** - New projects in seconds
- **Registry Helpers** - 70% less boilerplate code
- **Health Diagnostics** - Comprehensive validation with one command
- **Self-Contained Config** - All settings in one place
- **Natural Imports** - Module paths match package structure

#### Backward Compatibility
- **Legacy entry points maintained** - `alpha-berkeley`, `alpha-berkeley-deploy` still work
- **Registry interface preserved** - `RegistryConfigProvider` unchanged
- **Core functionality maintained** - All framework features work as before

### Migration Guide

#### For Production Applications
1. Install framework: `pip install alpha-berkeley-framework`
2. Create new repository structure
3. Copy application code to new structure
4. Update import paths (find-and-replace `applications.` → ``)
5. Simplify registry with `extend_framework_registry()`
6. Create self-contained `config.yml`
7. Setup `.env` file with API keys
8. Validate with `framework health`
9. Test functionality with `framework chat`
10. Initialize git repository and push

#### For Tutorial Applications
Regenerate from templates:
```bash
framework init my-weather --template hello_world_weather
framework init my-turbine --template wind_turbine
```

#### Complete Instructions
See comprehensive migration guide:
https://als-apg.github.io/osprey/getting-started/migration-guide

### Implementation Stats
- **100+ tasks completed** across 6 implementation phases
- **CLI infrastructure** - 5 commands with lazy loading (~2000 lines)
- **Template system** - 3 app templates + project + services
- **Registry helpers** - `extend_framework_registry()` (~200 lines)
- **Migration guide** - Comprehensive documentation (~730 lines)
- **Health diagnostics** - System validation (~968 lines)

### Related Issues
- Implements [#8 - Decouple Applications from Framework Repository](https://github.com/thellert/alpha_berkeley/issues/8)

## [0.6.0] - 2025-10-14

### Added
- **Performance Optimization System**: Configurable bypass modes for task extraction and capability selection
- **Task Extraction Bypass**: Skip LLM-based task extraction and use full conversation context for downstream processing
- **Capability Selection Bypass**: Skip LLM-based classification and activate all registered capabilities
- **Runtime Slash Commands**: Added `/task:off`, `/task:on`, `/caps:off`, `/caps:on` for dynamic performance control
- **Configuration Support**: New `agent_control` section in config.yml with bypass settings and system-wide defaults
- **Comprehensive Documentation**: Added bypass mode documentation with use cases, tradeoffs, and real CLI examples

### Enhanced
- **Gateway**: Parse and apply new performance bypass slash commands with readable command formatting
- **Task Extraction Node**: Implement bypass logic that formats full chat history and data sources without LLM processing
- **Classification Node**: Implement bypass logic that activates all capabilities without LLM analysis
- **State Manager**: Add bypass configuration defaults to agent_control state
- **Documentation**: Cross-referenced gateway, task extraction, and classification docs with performance configuration section

### Fixed
- **Data Source Request Creation**: Fixed user_id extraction to properly use session info instead of non-existent state field

### Performance Benefits
- Reduced LLM call overhead in preprocessing pipeline (1-2 fewer LLM calls per request)
- Flexible performance tuning for R&D, debugging, and high-throughput scenarios
- Trade orchestration complexity for extraction/classification speed based on use case
- Configurable via both system defaults and runtime slash commands

## [0.5.1] - 2025-10-13

### Fixed
- **Task Extraction Data Integration**: Enhanced task extraction to properly format retrieved data content from external sources
- **LLM Context Quality**: Improved the quality of context provided to task extraction for better results
- **Data Source Formatting**: Added robust fallback handling for data source content formatting

## [0.5.0] - 2025-09-26

### Added
- **ALS Assistant Application**: Complete domain-specific application for Advanced Light Source operations
- **PV Finder Service**: Intelligent EPICS process variable discovery with MCP integration
- **Application Launcher Service**: Desktop integration with MCP protocol support
- **Comprehensive Knowledge Base**: ALS accelerator objects database, PV naming structures, and MATLAB codebase analysis
- **Observability Integration**: Langfuse support with Docker containerization
- **Data Analysis Capabilities**: 7 new capability modules for accelerator physics operations
- **Infrastructure Services**: MongoDB database service, container orchestration for specialized services

### Enhanced
- **Container Execution**: Improved WebSocket connectivity, proxy handling, and error recovery
- **UI State Management**: Renamed `ui_notebook_links` to `ui_captured_notebooks` for clarity
- **Documentation**: Complete RST documentation with architectural diagrams and setup guides
- **Benchmarking Suite**: Performance analysis tools and model comparison frameworks

### Technical Details
- Added 144 new files with 430,647 lines of code
- Integrated MCP (Model Context Protocol) for external service communication
- Enhanced Docker compose templates with Langfuse environment variables
- Added comprehensive test coverage for core ALS services
- Implemented specialized databases for accelerator operations (11k+ PVs, AO structures)
- Enhanced framework capabilities with domain-specific prompt engineering

This release represents the framework's first complete domain-specific application, demonstrating the capability-based architecture's effectiveness for specialized scientific computing environments.

## [0.4.5] - 2025-09-23

### Added
- **Centralized Launchable Commands System**: New infrastructure for registering and displaying executable commands (web apps, desktop tools) through both CLI and OpenWebUI interfaces
- **Enhanced UI Result Display**: Comprehensive display system for figures, commands, and notebooks with rich formatting and metadata
- **MCP Protocol Support**: Added `fastmcp` dependency for Model Context Protocol integrations

### Enhanced
- **CLI Interface**: Added comprehensive result display methods with formatted output for figures, commands, and notebooks
- **OpenWebUI Interface**: Refactored result extraction with improved command and notebook handling
- **Configuration Management**: Enhanced path resolution with host/container awareness and application-specific file paths
- **State Management**: New `ui_launchable_commands` registry and `StateManager.register_command()` method
- **Response Generation**: Updated prompts to handle command display with interface-aware formatting

### Improved
- **Documentation**: Reorganized static resources following Sphinx best practices
- **Service Configuration**: Streamlined deployed services configuration with better maintainability
- **Error Handling**: Enhanced logging and fallback mechanisms throughout UI components

### Technical Details
- Added `ui_launchable_commands` field to AgentState for centralized command registry
- Implemented command registration system for capability-agnostic command handling
- Enhanced `get_agent_dir()` with `host_path` parameter for container/host path control
- Updated response context with `commands_available` field for UI awareness
- Improved container environment detection and path resolution

## [0.4.4] - 2025-09-17

### Refactored
- **Example Formatting System**: Consolidated example formatting with unified `BaseExample.join()` static method
- **Code Deduplication**: Removed duplicate `format_examples_for_prompt()` methods from `OrchestratorExample` and `ClassifierExample` subclasses
- **Flexible Formatting Options**: Added configurable formatting with support for separators, numbering, randomization, and example limits
- **Bias Prevention**: Maintained randomization for classifier examples to prevent positional bias in few-shot learning
- **API Consistency**: Unified formatting interface reduces maintenance burden for future example types

### Technical Details
- Added `BaseExample.join()` with parameters: `separator`, `max_examples`, `randomize`, `add_numbering`
- Updated `classification_node.py` to use `join()` with randomization for bias prevention
- Updated prompt builders (`memory_extraction.py`, `orchestrator.py`) to use `join()` with numbering
- Maintains all existing formatting behavior while reducing code duplication by 23 lines

## [0.4.3] - 2025-09-13

### Enhanced
- **OpenWebUI Interface**: Added notebook link display functionality with comprehensive response integration
- **Response Generation**: Enhanced prompts with notebook awareness and interface-specific guidance for better user experience
- **Context Loading**: Improved logging and registry initialization for better debugging and error handling

### Improved
- **Wind Turbine Application**: Refactored response generation guidelines with streamlined structure and cleaner code organization
- **User Experience**: Better integration of text responses, figures, and clickable notebook links in OpenWebUI
- **Debugging**: Replaced print statements with proper logging throughout context loading system

### Technical Details
- Added notebook link extraction and display in OpenWebUI response pipeline
- Enhanced response prompts with conversational guidelines and notebook availability context
- Improved context loader with registry initialization for proper context reconstruction

## [0.4.2] - 2025-09-13

### Enhanced
- **Python Execution Integration**: Python capability now registers notebooks using centralized StateManager.register_notebook() with rich metadata
- **Notebook Link Generation**: Improved notebook URL generation in both local and container execution modes with FileManager integration
- **Notebook Structure**: Enhanced notebook cell organization with separate markdown headers and executable code blocks

### Technical Details
- Added notebook registration with execution time, context key, and code metrics to Python capability
- Standardized notebook naming to 'notebook.ipynb' across execution modes
- Improved notebook generation with cleaner separation of results documentation and executable code

## [0.4.1] - 2025-09-13

### Enhanced
- **Centralized Notebook Registry**: Added structured notebook registry system replacing simple link list with rich metadata support
- **StateManager Enhancements**: Added `register_notebook()` method for capability-agnostic notebook registration with timestamps and metadata
- **Response Context Tracking**: Enhanced ResponseContext to track notebook availability for improved user guidance

### Technical Details
- Replaced `ui_notebook_links` with structured `ui_captured_notebooks` registry in agent state
- Added notebook registration method supporting display names, metadata, and automatic timestamp generation
- Updated state reset logic to use new registry format for better notebook management

## [0.4.0] - 2025-09-12

### Major Features
- **Context Memory Optimization**: Added recursive data summarization with `recursively_summarize_data()` utility to prevent context window overflow
- **Configurable Python Executor**: Complete Python executor configuration system with `PythonExecutorConfig` class for centralized settings
- **Enhanced Figure Registration**: Added batch figure registration support with accumulation for improved performance
- **OpenWebUI Performance Optimizations**: Response chunking for large outputs (>50KB) and static URL serving for figures

### Fixed
- **Critical Infinite Loop Bug**: Fixed infinite reclassification loop when orchestrator hallucinated non-existent capabilities
- **Reclassification Limit Enforcement**: Router now properly enforces `max_reclassifications` limit for all reclassification paths
- **Dependency Issues**: Fixed OpenTelemetry version constraints to resolve compatibility issues
- **Error Handling**: Enhanced retry logic and error classification in infrastructure nodes

### Changed
- **Unified Error Handling**: Consolidated reclassification system to use single error-based path instead of dual state/error approaches
- **Context Method Naming**: Renamed `get_human_summary()` to `get_summary()` across all context classes with backwards compatibility
- **Infrastructure Node Improvements**: Infrastructure nodes now raise `ReclassificationRequiredError` exceptions instead of directly manipulating state
- **State Cleanup**: Removed obsolete `control_needs_reclassification` field from agent state

### Enhanced
- **Python Executor Improvements**: Configurable execution timeouts, better error handling, and improved figure collection in both local and container modes
- **Context Window Management**: Automatic truncation of large execution results and code outputs to manage LLM context limits
- **Deployment Configuration**: Updated for static file serving with proper environment variable support
- **Error Classification**: Better distinction between retriable LLM failures and configuration errors

### Technical Details
- Added `ReclassificationRequiredError` exception to framework error system
- Enhanced router error handling to enforce limits consistently across all reclassification triggers
- Updated orchestrator and classifier to use proper exception-based error handling
- Improved architecture with cleaner separation between error handling and state management
- Added python_executor configuration section with sensible defaults
- Implemented graceful fallback for legacy method names with deprecation warnings

## [0.3.1] - 2025-09-10

### Enhancements
- **Documentation Workflow Improvements**: Added manual trigger capability to GitHub Actions documentation workflow
- **Tag-based Documentation Rebuilds**: Documentation now automatically rebuilds when version tags are created or moved
- **Enhanced Build Controls**: Documentation workflow now supports both automatic (tag/push) and manual triggering

### Bug Fixes
- **Documentation Version Sync**: Fixed issue where moving git tags didn't trigger documentation rebuilds, ensuring docs always reflect current version
- **Gitignore Cleanup**: Added `.nfs*` pattern to gitignore and fixed malformed entries

### Technical Details
- Added `workflow_dispatch` trigger to `.github/workflows/docs.yml` for manual execution
- Added `tags: ['v*']` trigger for automatic rebuilds on version tag changes
- Updated deployment conditions to support manual and tag-based triggers
- Improved build artifact and deployment logic for consistent documentation updates

## [0.3.0] - 2025-09-09

### Features
- **Interface Context System**: Added runtime interface detection for multi-interface support (CLI, OpenWebUI)
- **Centralized Figure Registry**: Implemented capability-agnostic figure registration system with rich metadata
- **Enhanced Figure Display**: Added automatic base64 figure conversion for OpenWebUI with interface-aware rendering
- **Real-time Log Viewer**: Added `/logs` command to OpenWebUI for in-memory log viewing and debugging
- **Robust JSON Serialization**: Comprehensive serialization utilities for scientific objects (matplotlib, numpy, pandas)

### Framework Enhancements
- **Interface-Aware Response Generation**: Context-sensitive prompts and responses based on interface capabilities
- **Python Executor Improvements**: Enhanced error handling and metadata serialization with fallback mechanisms
- **State Management Updates**: Centralized figure registry with capability source tracking and timestamps
- **Configuration System**: Added `get_interface_context()` for runtime interface detection

### Technical Improvements
- **Serialization Utilities**: Added `make_json_serializable()` and `serialize_results_to_file()` for robust data handling
- **Path Resolution**: Capability-agnostic figure path resolution for different execution environments
- **Error Handling**: Enhanced Python executor with detailed error reporting and serialization failure recovery
- **UI Integration**: Seamless figure display with metadata and creation timestamps

## [0.2.2] - 2025-08-16

### Major Features
- **New RECLASSIFICATION Error Severity**: Added `RECLASSIFICATION` severity level to ErrorSeverity enum for improved task-capability matching
- **Enhanced Error Classification Workflow**: Capabilities can now request reclassification when receiving inappropriate tasks
- **Reclassification Routing Logic**: Router node now properly handles reclassification errors with configurable attempt limits

### Breaking Changes
- **ErrorClassification Metadata Migration**: Replaced custom error fields with unified metadata field in ErrorClassification
  - `format_for_llm()` now generically processes all metadata keys
  - Enhanced error context richness for better LLM understanding
  - All infrastructure nodes and capabilities updated to use metadata field
  - Maintains backward compatibility through systematic migration

### Framework Enhancements
- **Enhanced Classification Node**: Improved reclassification workflow with proper failure context handling
- **Router Node Improvements**: Added reclassification attempt tracking and routing logic
- **Execution Limits Configuration**: Added support for configurable reclassification limits
- **Error Node Enhancements**: Comprehensive error handling improvements with better metadata processing

### Documentation & Examples
- **Major Documentation Cleanup**: Removed outdated markdown files and enhanced RST documentation structure
- **Enhanced Hello World Weather Example**: Added comprehensive classifier examples and improved context access details
- **Error Handling Documentation**: Complete documentation updates for new reclassification workflow
- **API Reference Updates**: Enhanced error handling API documentation with examples and usage patterns
- **Developer Guide Improvements**: Updated infrastructure components documentation

### Infrastructure Improvements
- **Framework-wide Capability Updates**: All capabilities updated to use new ErrorClassification metadata approach
- **Enhanced Time Range Parsing**: Improved time range parsing capability with better error handling
- **Configuration System Updates**: Enhanced config system to support execution limits and reclassification controls

### Technical Details
- Enhanced error classification system enables better task-capability matching
- Unified metadata approach provides richer context for error analysis and recovery
- Reclassification workflow prevents infinite loops with configurable attempt limits
- Complete migration maintains backward compatibility across the entire framework

## [0.2.1] - 2025-08-11

### Critical Fixes
- **Containerized Python Execution**: Fixed critical bug where execution metadata wasn't being created in mounted volumes
- **Container Build Failures**: Removed obsolete python3-epics-simulation kernel mounts that caused build failures
- **Path Mapping**: Fixed hardcoded path patterns in container execution using config-driven approach
- **Timezone Consistency**: Standardized timezone across all services with centralized configuration

### Security & Stability
- **Repository Security**: Updated .gitignore to exclude development services and sensitive configurations
- **Network Security**: Renamed container network from als-agents-network to alpha-berkeley-network for consistency
- **Service Cleanup**: Removed mem0 service references and cleaned up leftover container code

### Developer Experience Improvements
- **Configuration System Refactoring**: Renamed `unified_config` module to `config` for improved developer experience
- **Professional Naming**: Replaced `UnifiedConfigBuilder` with `ConfigBuilder` to eliminate confusing terminology
- **Automatic Environment Detection**: Added container-aware Python environment detection for convenience
- **Graceful Ollama Fallback**: Implemented automatic URL fallback for development workflows
- **Documentation**: Updated all references across 43+ files to use consistent naming conventions

### Infrastructure Enhancements
- **Git-based Versioning**: Added automatic version detection from git tags in documentation
- **Path Resolution**: Replaced hardcoded paths with configuration-driven approach using `get_agent_dir()`
- **Container Integration**: Improved container execution reliability and error handling
- **Documentation Cleanup**: Enhanced error handling documentation and API references

### Technical Details
- Fixed 'Failed to read execution metadata from container' error through proper volume mounting
- Eliminated manual reconfiguration when switching between local and containerized execution
- Complete refactoring eliminates confusing "unified" terminology from LangGraph migration era
- Added proper timezone data (tzdata) package in Jupyter containers for accurate timestamps
- Maintains backward compatibility through systematic import updates across entire codebase

## [0.2.0] - 2025-01-31

### Added
- Enhanced execution plan editor with file-based persistence
- Comprehensive approval system with human-in-the-loop workflows
- Complete advanced wind turbine tutorial application
- Improved documentation with execution plan viewer
- Execution plan viewer JavaScript support for interactive documentation

### Changed
- Modernized docker-compose configurations
- Enhanced framework robustness and capabilities
- Improved documentation build system and content

### Fixed
- Repository hygiene improvements with better .gitignore
- Removed deprecated version fields from docker-compose files
- Cleaned up PID files from repository

## [0.1.1] - 2025-08-08

### Fixed
- Remove invalid retry_count parameter from ErrorClassification calls in infrastructure nodes
- Fix runtime error: `ErrorClassification.__init__() got an unexpected keyword argument 'retry_count'`
- Update documentation examples to reflect correct ErrorClassification API usage
- Complete migration from dual retry tracking to state-only retry tracking

## [0.1.0] - 2024-12-XX

### Added
- Core capability-based agent architecture
- LangGraph integration for structured orchestration
- Complete hello world weather agent tutorial
- Framework installation and setup documentation
- API reference documentation (actively being developed)
- Developer guides covering infrastructure components
- Container-based deployment system
- Basic CLI interface for direct conversation
- Memory storage and context management systems
- Human approval workflow integration
- Error handling and recovery infrastructure

### Documentation
- Getting started guide with installation instructions
- Complete hello world tutorial with working weather agent
- Early access documentation warnings across all sections
- API reference for core framework components
- Developer guides for infrastructure understanding

### Known Limitations
- Documentation is under active development
- Some advanced tutorials not yet included
- APIs may evolve before 1.0.0 release

---

*This is an early access release. We welcome feedback and contributions!*

