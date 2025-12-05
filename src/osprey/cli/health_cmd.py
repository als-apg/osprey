"""Health check command for Osprey Framework.

This module provides the 'osprey health' command which performs comprehensive
diagnostics on the osprey installation and application configuration. It checks
configuration validity, file system structure, container status, API providers,
and Python environment without actually running the osprey.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

from osprey.cli.styles import Messages, Styles, console
from osprey.deployment.runtime_helper import get_ps_command, get_runtime_command
from osprey.utils.log_filter import quiet_logger


class HealthCheckResult:
    """Result of a health check with status and details."""

    def __init__(self, name: str, status: str, message: str = "", details: str = ""):
        self.name = name
        self.status = status  # "ok", "warning", "error"
        self.message = message
        self.details = details

    def __repr__(self):
        return f"HealthCheckResult({self.name}, {self.status})"


class HealthChecker:
    """Comprehensive health checker for Osprey Framework."""

    def __init__(self, verbose: bool = False, full: bool = False, project_path: Path | None = None):
        self.verbose = verbose
        self.full = full
        self.results: list[HealthCheckResult] = []
        self.cwd = project_path if project_path else Path.cwd()
        self.config = {}  # Initialize empty config, will be populated in check_configuration()

        # Load .env file early so environment variables are available for all checks
        self._load_env_file()

    def add_result(self, name: str, status: str, message: str = "", details: str = ""):
        """Add a health check result."""
        self.results.append(HealthCheckResult(name, status, message, details))

    def _load_env_file(self):
        """Load .env file from current directory if it exists."""
        try:
            from dotenv import load_dotenv

            dotenv_path = self.cwd / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path, override=False)  # Don't override existing env vars
        except ImportError:
            # python-dotenv not available, skip loading
            pass

    def check_all(self) -> bool:
        """Run all health checks and return True if all passed."""
        console.print(f"\n{Messages.header('🏥 Osprey Framework - Health Check')}\n")

        # Initialize registry (needed for provider checks)
        try:
            from osprey.registry import initialize_registry

            # Show spinner during registry initialization
            spinner = Spinner(
                "dots", text="[dim]Initializing framework registry...[/dim]", style=Styles.INFO
            )
            with Live(spinner, console=console, transient=True):
                if not self.verbose:
                    with quiet_logger(["REGISTRY", "CONFIG"]):
                        initialize_registry()
                else:
                    initialize_registry()
        except Exception as e:
            if self.verbose:
                console.print(f"  [dim]Could not initialize registry: {e}[/dim]")

        # Phase 1: Core checks (always run)
        self.check_configuration()
        self.check_file_system()
        self.check_python_environment()

        # Phase 2: Container and provider checks (always run)
        self.check_containers()
        self.check_api_providers()

        # Phase 3: Full model testing (only in full mode)
        if self.full:
            self.check_model_chat_completions()

        # Display results
        self.display_results()

        # Return overall status
        errors = sum(1 for r in self.results if r.status == "error")
        return errors == 0

    def check_configuration(self):
        """Check configuration file validity and structure."""
        console.print("[bold]Configuration[/bold]")

        # Check if config.yml exists
        config_path = self.cwd / "config.yml"
        if not config_path.exists():
            self.add_result(
                "config_file_exists",
                "error",
                "config.yml not found in current directory",
                f"Looking in: {self.cwd}\n"
                "Please run this command from a project directory containing config.yml",
            )
            console.print(f"  {Messages.error('❌ config.yml not found')}")
            return

        self.add_result("config_file_exists", "ok", f"Found at {config_path}")
        console.print(f"  {Messages.success('config.yml found')}")

        # Try to load and parse YAML
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            if config is None:
                self.add_result("yaml_valid", "error", "Config file is empty")
                console.print(f"  {Messages.error('❌ Config file is empty')}")
                return

            if not isinstance(config, dict):
                self.add_result("yaml_valid", "error", "Config must be a dictionary")
                console.print(f"  {Messages.error('❌ Invalid YAML structure')}")
                return

            self.add_result("yaml_valid", "ok", "Valid YAML syntax")
            console.print(f"  {Messages.success('Valid YAML syntax')}")

            # Store config for use in other checks
            self.config = config

            # Check required sections
            self._check_config_structure(config)

            # Check environment variables
            self._check_environment_variables(config)

        except yaml.YAMLError as e:
            self.add_result("yaml_valid", "error", f"YAML parsing error: {e}")
            console.print(f"  {Messages.error(f'YAML parsing error: {e}')}")
        except Exception as e:
            self.add_result("yaml_valid", "error", f"Failed to read config: {e}")
            console.print(f"  {Messages.error(f'Failed to read config: {e}')}")

    def _check_config_structure(self, config: dict):
        """Check configuration structure and required sections."""

        # Check required framework models (8 total)
        required_models = [
            "orchestrator",
            "response",
            "classifier",
            "approval",
            "task_extraction",
            "memory",
            "python_code_generator",
            "time_parsing",
        ]

        models = config.get("models", {})
        missing_models = [m for m in required_models if m not in models]

        if missing_models:
            self.add_result(
                "required_models",
                "error",
                f"Missing required models: {', '.join(missing_models)}",
                "Framework requires 8 models: " + ", ".join(required_models),
            )
            missing_str = ", ".join(missing_models)
            console.print(f"  {Messages.error(f'Missing required models: {missing_str}')}")
        else:
            self.add_result(
                "required_models", "ok", f"All {len(required_models)} required models defined"
            )
            console.print(
                f"  {Messages.success(f'All {len(required_models)} required models defined')}"
            )

        # Check model configurations
        invalid_models = []
        for model_name, model_config in models.items():
            if not isinstance(model_config, dict):
                invalid_models.append(model_name)
                continue
            if "provider" not in model_config:
                invalid_models.append(f"{model_name} (missing provider)")
            if "model_id" not in model_config:
                invalid_models.append(f"{model_name} (missing model_id)")

        if invalid_models:
            self.add_result(
                "model_configs_valid",
                "warning",
                f"Invalid model configurations: {', '.join(invalid_models)}",
            )
            invalid_str = ", ".join(invalid_models)
            console.print(f"  {Messages.warning(f'Invalid model configs: {invalid_str}')}")
        else:
            self.add_result("model_configs_valid", "ok", "All model configurations valid")
            console.print(f"  {Messages.success('All model configurations valid')}")

        # Check deployed_services
        deployed_services = config.get("deployed_services", [])
        if not deployed_services:
            self.add_result("deployed_services", "warning", "No deployed services configured")
            console.print(f"  {Messages.warning('No deployed services configured')}")
        else:
            self.add_result(
                "deployed_services",
                "ok",
                f"{len(deployed_services)} services configured: {', '.join(deployed_services)}",
            )
            console.print(f"  {Messages.success(f'{len(deployed_services)} services configured')}")

        # Check if services defined in deployed_services exist in services section
        services = config.get("services", {})
        undefined_services = [s for s in deployed_services if s not in services]
        if undefined_services:
            self.add_result(
                "service_definitions",
                "error",
                f"Services not defined: {', '.join(undefined_services)}",
            )
            undefined_str = ", ".join(undefined_services)
            console.print(f"  {Messages.error(f'Undefined services: {undefined_str}')}")
        else:
            self.add_result("service_definitions", "ok", "All deployed services defined")
            if deployed_services:  # Only print if there are services
                console.print(f"  {Messages.success('All deployed services defined')}")

        # Check API providers
        api_providers = config.get("api", {}).get("providers", {})
        if not api_providers:
            self.add_result("api_providers", "warning", "No API providers configured")
            console.print(f"  {Messages.warning(' No API providers configured')}")
        else:
            self.add_result(
                "api_providers",
                "ok",
                f"{len(api_providers)} providers configured: {', '.join(api_providers.keys())}",
            )
            console.print(f"  {Messages.success(f'{len(api_providers)} API providers configured')}")

    def _check_environment_variables(self, config: dict):
        """Check if environment variables referenced in config are set."""
        import re

        # Find all ${VAR_NAME} patterns in config
        config_str = str(config)
        env_vars = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", config_str)
        env_vars = list(set(env_vars))  # Remove duplicates

        missing_vars = [var for var in env_vars if var not in os.environ]

        if missing_vars:
            self.add_result(
                "environment_variables",
                "warning",
                f"Missing environment variables: {', '.join(missing_vars)}",
                "These variables are referenced in config.yml but not set in environment",
            )
            missing_vars_str = ", ".join(missing_vars)
            console.print(f"  {Messages.warning(f' Missing env vars: {missing_vars_str}')}")
        else:
            if env_vars:
                self.add_result(
                    "environment_variables", "ok", f"All {len(env_vars)} environment variables set"
                )
                console.print(
                    f"  {Messages.success(f'All {len(env_vars)} environment variables set')}"
                )

    def _check_project_paths(self):
        """Check if project_root and agent data directory paths are valid and accessible."""

        try:
            # Get project_root from config (could be hardcoded or env var)
            project_root = self.config.get("project_root")
            if not project_root:
                self.add_result("project_paths", "warning", "No project_root configured")
                console.print(f"  {Messages.warning(' No project_root configured')}")
                return

            # Resolve project_root (handles ${PROJECT_ROOT} expansion)
            project_root_resolved = os.path.expandvars(str(project_root))
            project_root_path = Path(project_root_resolved)

            # Check if project_root exists
            if project_root_path.exists():
                self.add_result(
                    "project_root_path", "ok", f"Project root exists: {project_root_path}"
                )
                console.print(f"  {Messages.success(f'Project root exists: {project_root_path}')}")
            else:
                self.add_result(
                    "project_root_path",
                    "warning",
                    f"Project root does not exist: {project_root_path}",
                )
                console.print(
                    f"  {Messages.warning(f' Project root does not exist: {project_root_path}')}"
                )
                # Don't return - we can still check if it could be created

            # Check agent data directory
            file_paths = self.config.get("file_paths", {})
            agent_data_dir = file_paths.get("agent_data_dir", "_agent_data")
            agent_data_path = project_root_path / agent_data_dir

            if agent_data_path.exists():
                # Check if it's writable
                if os.access(agent_data_path, os.W_OK):
                    self.add_result(
                        "agent_data_dir", "ok", f"Agent data directory writable: {agent_data_path}"
                    )
                    console.print(f"  {Messages.success('Agent data directory writable')}")
                else:
                    self.add_result(
                        "agent_data_dir",
                        "warning",
                        f"Agent data directory not writable: {agent_data_path}",
                    )
                    console.print(f"  {Messages.warning(' Agent data directory not writable')}")
            else:
                # Check if parent directory exists and is writable (can we create it?)
                parent_dir = agent_data_path.parent
                if parent_dir.exists() and os.access(parent_dir, os.W_OK):
                    self.add_result(
                        "agent_data_dir",
                        "ok",
                        f"Agent data directory can be created: {agent_data_path}",
                    )
                    console.print(f"  {Messages.success('Agent data directory can be created')}")
                else:
                    self.add_result(
                        "agent_data_dir",
                        "warning",
                        f"Cannot create agent data directory: {agent_data_path}",
                    )
                    console.print(f"  {Messages.warning(' Cannot create agent data directory')}")

        except Exception as e:
            self.add_result("project_paths", "error", f"Error checking project paths: {e}")
            console.print(f"  {Messages.error(f'Error checking project paths: {e}')}")

    def check_file_system(self):
        """Check file system structure and permissions."""
        console.print("\n[bold]File System[/bold]")

        # Check project paths from config
        self._check_project_paths()

        # Check .env file
        env_file = self.cwd / ".env"
        if env_file.exists():
            self.add_result("env_file", "ok", ".env file found")
            console.print(f"  {Messages.success('.env file found')}")
        else:
            self.add_result("env_file", "warning", ".env file not found")
            console.print(f"  {Messages.warning(' .env file not found')}")

        # Check registry file (if specified in config)
        try:
            config_path = self.cwd / "config.yml"
            if config_path.exists():
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)

                registry_path_str = config.get("registry_path")
                if registry_path_str:
                    # Resolve environment variables in path
                    registry_path_str = os.path.expandvars(registry_path_str)
                    registry_path = self.cwd / registry_path_str

                    if registry_path.exists():
                        self.add_result(
                            "registry_file", "ok", f"Registry file found: {registry_path}"
                        )
                        console.print(f"  {Messages.success('Registry file found')}")
                    else:
                        self.add_result(
                            "registry_file", "error", f"Registry file not found: {registry_path}"
                        )
                        console.print(
                            f"  {Messages.error(f'Registry file not found: {registry_path}')}"
                        )
        except Exception:
            # Don't fail if we can't check registry
            pass

        # Check disk space
        try:
            stat = shutil.disk_usage(self.cwd)
            free_gb = stat.free / (1024**3)

            if free_gb < 1.0:
                self.add_result("disk_space", "warning", f"Low disk space: {free_gb:.1f} GB free")
                console.print(f"  {Messages.warning(f' Low disk space: {free_gb:.1f} GB free')}")
            else:
                self.add_result("disk_space", "ok", f"{free_gb:.1f} GB free")
                console.print(f"  {Messages.success(f'Disk space: {free_gb:.1f} GB free')}")
        except Exception as e:
            self.add_result("disk_space", "warning", f"Could not check disk space: {e}")

    def check_python_environment(self):
        """Check Python version and dependencies."""
        console.print("\n[bold]Python Environment[/bold]")

        # Check Python version
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major < 3 or (version.major == 3 and version.minor < 11):
            self.add_result(
                "python_version", "warning", f"Python {version_str} (recommended: 3.11+)"
            )
            console.print(f"  {Messages.warning(f' Python {version_str} (recommended: 3.11+)')}")
        else:
            self.add_result("python_version", "ok", f"Python {version_str}")
            console.print(f"  {Messages.success(f'Python {version_str}')}")

        # Check if we're in a virtual environment
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        if in_venv:
            self.add_result("virtual_environment", "ok", "Virtual environment active")
            console.print(f"  {Messages.success('Virtual environment active')}")
        else:
            self.add_result("virtual_environment", "warning", "Not in a virtual environment")
            console.print(f"  {Messages.warning(' Not in a virtual environment')}")

        # Check core dependencies
        core_deps = [
            "click",
            "rich",
            "yaml",
            "jinja2",
            "pydantic_ai",
            "langgraph",
            "langchain_core",
        ]

        missing_deps = []
        for dep in core_deps:
            # Handle special cases
            import_name = dep
            if dep == "yaml":
                import_name = "yaml"
            elif dep == "pydantic_ai":
                import_name = "pydantic_ai"
            elif dep == "langchain_core":
                import_name = "langchain_core"

            try:
                __import__(import_name)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            self.add_result(
                "core_dependencies", "error", f"Missing dependencies: {', '.join(missing_deps)}"
            )
            missing_deps_str = ", ".join(missing_deps)
            console.print(f"  {Messages.error(f'Missing dependencies: {missing_deps_str}')}")
        else:
            self.add_result(
                "core_dependencies", "ok", f"All {len(core_deps)} core dependencies installed"
            )
            console.print(
                f"  {Messages.success(f'Core dependencies installed ({len(core_deps)}/{len(core_deps)})')}"
            )

    def check_containers(self):
        """Check container runtime and deployed services."""
        console.print("\n[bold]Container Infrastructure[/bold]")

        # Check if a container runtime is available
        try:
            runtime_cmd = get_runtime_command()
            runtime = runtime_cmd[0]  # 'docker' or 'podman'

            result = subprocess.run(
                [runtime, "--version"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                self.add_result(f"{runtime}_available", "ok", version)
                console.print(f"  {Messages.success(f'{version}')}")

                # Check container status
                self._check_container_status()
            else:
                self.add_result(
                    f"{runtime}_available", "error", f"{runtime.capitalize()} command failed"
                )
                console.print(
                    f"  {Messages.error(f'❌ {runtime.capitalize()} not working properly')}"
                )
        except RuntimeError as e:
            # No runtime found
            self.add_result("container_runtime", "warning", str(e))
            console.print(
                f"  {Messages.warning(' No container runtime found (Docker or Podman required)')}"
            )
        except FileNotFoundError:
            self.add_result("container_runtime", "warning", "Container runtime not found in PATH")
            console.print(
                f"  {Messages.warning(' Container runtime not installed or not in PATH')}"
            )
        except Exception as e:
            self.add_result(
                "container_runtime", "warning", f"Could not check container runtime: {e}"
            )
            console.print(f"  {Messages.warning(f' Could not check container runtime: {e}')}")

    def _check_container_status(self):
        """Check status of deployed containers."""
        try:
            # Load config to get deployed services
            config_path = self.cwd / "config.yml"
            if not config_path.exists():
                return

            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            deployed_services = config.get("deployed_services", [])
            if not deployed_services:
                return

            # Get all containers
            result = subprocess.run(
                get_ps_command(config, all_containers=True),
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return

            containers = json.loads(result.stdout) if result.stdout.strip() else []

            # Check each expected service
            for service in deployed_services:
                # Extract service short name (handle dotted paths like "osprey.jupyter")
                service_short = str(service).split(".")[-1].lower()

                # Look for containers matching the service name
                # Use smart matching to handle underscore/hyphen variations
                matching = []
                for c in containers:
                    names = c.get("Names", [])
                    if isinstance(names, list):
                        names_str = " ".join(str(n) for n in names).lower()
                    else:
                        names_str = str(names).lower()

                    # Check for match (handles both underscore and hyphen)
                    if (
                        service_short in names_str
                        or service_short.replace("_", "-") in names_str
                        or service_short.replace("-", "_") in names_str
                    ):
                        matching.append(c)

                if matching:
                    container = matching[0]
                    state = container.get("State", "unknown")

                    if state == "running":
                        self.add_result(f"container_{service}", "ok", f"{service}: running")
                        console.print(f"  {Messages.success(f'{service}: running')}")
                    else:
                        self.add_result(f"container_{service}", "warning", f"{service}: {state}")
                        console.print(f"  {Messages.warning(f' {service}: {state}')}")
                else:
                    self.add_result(f"container_{service}", "warning", f"{service}: not found")
                    console.print(f"  {Messages.warning(f' {service}: not deployed')}")

        except Exception as e:
            # Don't fail the health check if container status can't be determined
            if self.verbose:
                console.print(f"  [dim]Could not check container status: {e}[/dim]")

    def check_api_providers(self):
        """Check API provider configurations and connectivity."""
        console.print("\n[bold]API Providers[/bold]")

        try:
            config_path = self.cwd / "config.yml"
            if not config_path.exists():
                return

            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            api_config = config.get("api", {}).get("providers", {})

            for provider_name, provider_config in api_config.items():
                self._check_provider(provider_name, provider_config)

        except Exception as e:
            if self.verbose:
                console.print(f"  [dim]Could not check API providers: {e}[/dim]")

    def _check_provider(self, provider_name: str, provider_config: dict):
        """Check a specific API provider using the provider registry."""
        from osprey.registry import get_registry

        try:
            # Suppress REGISTRY and CONFIG loggers unless in verbose mode
            if not self.verbose:
                with quiet_logger(["REGISTRY", "CONFIG"]):
                    registry = get_registry()
                    provider_class = registry.get_provider(provider_name)
            else:
                registry = get_registry()
                provider_class = registry.get_provider(provider_name)
        except Exception as e:
            if self.verbose:
                console.print(f"  [dim]Failed to get registry: {e}[/dim]")
            provider_class = None

        if not provider_class:
            self.add_result(
                f"provider_{provider_name}",
                "warning",
                f"{provider_name}: Unknown provider (not in registry)",
            )
            console.print(f"  {Messages.warning(f' {provider_name}: Unknown provider')}")
            return

        # Resolve API key from environment if needed
        api_key = provider_config.get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            var_name = api_key[2:-1]
            api_key = os.environ.get(var_name, "")

        # Get base URL from config
        base_url = provider_config.get("base_url")

        # Instantiate provider and check health
        provider = provider_class()
        success, message = provider.check_health(api_key, base_url, timeout=5.0)

        if success:
            self.add_result(f"provider_{provider_name}", "ok", f"{provider_name}: {message}")
            console.print(f"  {Messages.success(f'{provider_name}: {message}')}")
        else:
            self.add_result(f"provider_{provider_name}", "warning", f"{provider_name}: {message}")
            console.print(f"  {Messages.warning(f' {provider_name}: {message}')}")

    def _resolve_api_key(self, api_key: str) -> str:
        """Resolve API key if it's an environment variable reference."""
        if api_key.startswith("${") and api_key.endswith("}"):
            var_name = api_key[2:-1]
            return os.environ.get(var_name, "")
        return api_key

    def check_model_chat_completions(self):
        """Test actual chat completions with each unique model (full mode only)."""
        console.print("\n[bold]Model Chat Completions (Full Test)[/bold]")
        console.print("  [dim]Testing actual chat completion with each unique model...[/dim]")

        try:
            # Load config to get models
            config_path = self.cwd / "config.yml"
            if not config_path.exists():
                return

            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            models = config.get("models", {})
            if not models:
                console.print(f"  {Messages.warning(' No models configured')}")
                return

            # Extract unique (provider, model_id) pairs
            unique_models = set()
            for model_name, model_config in models.items():
                if not isinstance(model_config, dict):
                    continue

                provider = model_config.get("provider")
                model_id = model_config.get("model_id")

                if provider and model_id:
                    unique_models.add((provider, model_id))

            if not unique_models:
                console.print(f"  {Messages.warning(' No valid models found')}")
                return

            console.print(f"  [dim]Found {len(unique_models)} unique model(s) to test[/dim]\n")

            # Test each unique model
            for provider, model_id in sorted(unique_models):
                self._test_model_chat(provider, model_id)

        except Exception as e:
            if self.verbose:
                console.print(f"  [dim]Could not test model completions: {e}[/dim]")

    def _test_model_chat(self, provider: str, model_id: str):
        """Test a single model with a minimal chat completion."""
        model_label = f"{provider}/{model_id}"

        # Show that we're testing this model
        console.print(f"  [dim]Testing {model_label}...[/dim]", end=" ")

        try:
            # Use the simple get_chat_completion function
            from osprey.models.completion import get_chat_completion

            # Simple test prompt
            test_message = "Reply with exactly: OK"

            # Call get_chat_completion with a timeout
            # Suppress REGISTRY and CONFIG loggers unless in verbose mode
            try:
                if not self.verbose:
                    with quiet_logger(["REGISTRY", "CONFIG"]):
                        response = get_chat_completion(
                            message=test_message,
                            provider=provider,
                            model_id=model_id,
                            max_tokens=50,  # Keep it minimal
                        )
                else:
                    response = get_chat_completion(
                        message=test_message,
                        provider=provider,
                        model_id=model_id,
                        max_tokens=50,  # Keep it minimal
                    )

                # If we got here without exception and have a response, the model works
                if response and isinstance(response, str) and len(response.strip()) > 0:
                    self.add_result(
                        f"model_chat_{provider}_{model_id}",
                        "ok",
                        f"{model_label}: Chat completion successful",
                    )
                    console.print(Messages.success("Working"))
                else:
                    self.add_result(
                        f"model_chat_{provider}_{model_id}",
                        "warning",
                        f"{model_label}: Empty response",
                    )
                    console.print(Messages.warning(" Empty response"))

            except Exception as e:
                # Check if it's a timeout-like error
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    self.add_result(
                        f"model_chat_{provider}_{model_id}", "warning", f"{model_label}: Timeout"
                    )
                    console.print(Messages.warning(" Timeout"))
                else:
                    # Some other error
                    display_msg = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
                    self.add_result(
                        f"model_chat_{provider}_{model_id}", "error", f"{model_label}: {error_msg}"
                    )
                    console.print(Messages.error("❌ Failed"))
                    console.print(f"     [dim]{display_msg}[/dim]")
                return

        except KeyboardInterrupt:
            # Re-raise keyboard interrupt so user can stop
            raise

        except Exception as e:
            error_msg = str(e)
            # Truncate for display
            display_msg = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg

            self.add_result(
                f"model_chat_{provider}_{model_id}", "error", f"{model_label}: {error_msg}"
            )
            console.print(Messages.error("❌ Failed"))

            # Always show error details in full mode (not just verbose)
            console.print(f"     [dim]{display_msg}[/dim]")

    def display_results(self):
        """Display summary of health check results."""
        console.print()

        # Count results by status
        ok_count = sum(1 for r in self.results if r.status == "ok")
        warning_count = sum(1 for r in self.results if r.status == "warning")
        error_count = sum(1 for r in self.results if r.status == "error")
        total_count = len(self.results)

        # Build the content for the panel
        panel_content = []

        # Create summary line
        summary = f"Summary: {ok_count}/{total_count} checks passed"
        if warning_count > 0:
            summary += f" ({warning_count} warning{'s' if warning_count > 1 else ''})"
        if error_count > 0:
            summary += f" ({error_count} error{'s' if error_count > 1 else ''})"

        panel_content.append(summary)

        # Show detailed errors and warnings if verbose
        if self.verbose and (warning_count > 0 or error_count > 0):
            panel_content.append("")  # Empty line
            panel_content.append("Details:")
            for result in self.results:
                if result.status in ["warning", "error"]:
                    symbol = "⚠️ " if result.status == "warning" else "❌"
                    panel_content.append(f"  {symbol} {result.name}: {result.message}")
                    if result.details:
                        panel_content.append(f"     {result.details}")

        # Create and display the framed panel
        panel = Panel(
            "\n".join(panel_content),
            title="🏥 Osprey Health Check Results",
            border_style=Styles.BORDER_DIM,
            expand=False,
            padding=(1, 2),
        )
        console.print(panel)


@click.command()
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed information about warnings and errors"
)
@click.option(
    "--basic",
    "-b",
    is_flag=True,
    help="Skip model completion tests (only check configuration and connectivity)",
)
def health(project: str, verbose: bool, basic: bool):
    """Check the health of your Osprey installation and configuration.

    This command performs comprehensive diagnostics without actually running
    the osprey.

    Full Mode (DEFAULT):
    \b
      • Configuration file validity and structure
      • Required models and services configuration
      • File system structure and permissions
      • Python version and dependencies
      • Container runtime availability
      • Container status (running/stopped)
      • API provider endpoint connectivity (lightweight tests)
        - Ollama: Connection test
        - OpenAI/CBORG: GET /v1/models endpoint
        - Anthropic/Google: Key format validation
      • Actual chat completion tests with each unique model
      • Tests each unique (provider, model_id) pair only once
      • Sends minimal test prompt and verifies response
      • May incur small API costs (~$0.001-0.01 per model)

    Basic Mode (--basic):
    \b
      • All checks except model completion tests
      • Faster, no API costs
      • Only checks configuration and connectivity

    The health check must be run from a project directory containing
    config.yml (e.g., als_assistant, weather).

    Exit Codes:
    \b
      0 - All checks passed
      1 - Some warnings detected (non-critical)
      2 - Errors detected (critical issues)

    Examples:

    \b
      # Full health check with model completion tests (default)
      $ osprey health

      # Basic check without model tests
      $ osprey health --basic

      # Verbose output with detailed error messages
      $ osprey health --verbose

      # Basic mode with verbose output
      $ osprey health --basic --verbose

      # Check health of specific project
      $ osprey health --project ~/projects/my-agent

      # Use environment variable
      $ export OSPREY_PROJECT=~/projects/my-agent
      $ osprey health
    """
    from .project_utils import resolve_project_path

    try:
        # Resolve project directory
        project_path = resolve_project_path(project)

        # Full mode is default, basic flag disables it
        full = not basic

        checker = HealthChecker(verbose=verbose, full=full, project_path=project_path)
        success = checker.check_all()

        # Determine exit code
        error_count = sum(1 for r in checker.results if r.status == "error")
        warning_count = sum(1 for r in checker.results if r.status == "warning")

        if error_count > 0:
            console.print(f"\n{Messages.error('❌ Health check failed with errors')}")
            sys.exit(2)
        elif warning_count > 0:
            console.print(f"\n{Messages.warning('Health check completed with warnings')}")
            sys.exit(1)
        else:
            console.print(f"\n{Messages.success('All health checks passed!')}")
            sys.exit(0)

    except KeyboardInterrupt:
        console.print(f"\n\n{Messages.warning('Health check interrupted')}")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n{Messages.error(f'Health check failed: {e}')}")
        if verbose:
            console.print_exception()
        sys.exit(3)
