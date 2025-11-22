# Changelog

All notable changes to the Osprey Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2] - 2025-11-22

### 🎉 Major Features

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

### Fixed
- **Stanford API Key Detection**: Added missing STANFORD_API_KEY to environment variable detection (Reported by Marty)
- **Weather Template**: Fixed context extraction example in hello world weather template (PR #26)
- **CRITICAL BUG FIX**: `ContextManager.extract_from_step()` now correctly handles multiple contexts of the same type
  - Previously, when multiple contexts of the same type were requested (e.g., two `CURRENT_WEATHER` contexts), only the last one was returned, causing silent data loss
  - Now returns a list when multiple contexts of the same type exist: `{"CURRENT_WEATHER": [ctx1, ctx2]}`
  - Single contexts still returned as objects for backward compatibility: `{"CURRENT_WEATHER": ctx_obj}`
  - Capabilities can check `isinstance(context, list)` to detect and handle multiple contexts
  - Added 17 comprehensive test cases covering all scenarios

### Changed
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
- **Template Updates**: All capability templates now use instance method pattern
  - `hello_world_weather` template updated with new pattern and helper methods
  - `control_assistant` templates (archiver_retrieval, channel_finding, channel_value_retrieval) updated
  - `minimal` template updated to show recommended pattern
  - All templates include proper `requires` field with cardinality constraints

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