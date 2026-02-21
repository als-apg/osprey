"""Template management for project scaffolding.

This module provides the TemplateManager class which handles:
- Discovery of bundled templates in the osprey package
- Rendering Jinja2 templates with project-specific context
- Creating complete project structures from templates
- Copying service configurations to user projects
- Generating project manifests for migration support
- Prompt artifact user-ownership via PromptRegistry
"""

import hashlib
import json
import os
import re
import shutil
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from osprey.cli.prompt_registry import PromptRegistry
from osprey.cli.styles import console

# Manifest schema version for future compatibility
MANIFEST_SCHEMA_VERSION = "1.1.0"

# File used to store project manifest
MANIFEST_FILENAME = ".osprey-manifest.json"


class TemplateManager:
    """Manages project templates and scaffolding.

    This class handles all template-related operations for creating new
    projects from bundled templates. It uses Jinja2 for template rendering
    and provides methods for project structure creation.

    Attributes:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
    """

    def __init__(self):
        """Initialize template manager with osprey templates.

        Discovers the template directory from the installed osprey package
        using importlib, which works both in development and after pip install.
        """
        self.template_root = self._get_template_root()
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_root)),
            autoescape=select_autoescape(["html", "xml"]),
            keep_trailing_newline=True,
        )

    def _get_template_root(self) -> Path:
        """Get path to osprey templates directory.

        Returns:
            Path to the templates directory in the osprey package

        Raises:
            RuntimeError: If templates directory cannot be found
        """
        try:
            # Try to import osprey.templates to find its location
            import osprey.templates

            template_path = Path(osprey.templates.__file__).parent
            if template_path.exists():
                return template_path
        except (ImportError, AttributeError):
            pass  # Fall through to development fallback path below

        # Fallback for development: relative to this file
        fallback_path = Path(__file__).parent.parent / "templates"
        if fallback_path.exists():
            return fallback_path

        raise RuntimeError(
            "Could not locate osprey templates directory. Ensure osprey is properly installed."
        )

    def _detect_environment_variables(self) -> dict[str, str]:
        """Detect environment variables from the system for use in templates.

        This method checks for common environment variables that are typically
        needed in .env files (API keys, paths, etc.) and returns those that are
        currently set in the system.

        Returns:
            Dictionary of detected environment variables with their values.
            Only includes variables that are actually set (non-empty).

        Examples:
            >>> manager = TemplateManager()
            >>> env_vars = manager._detect_environment_variables()
            >>> env_vars.get('OPENAI_API_KEY')  # Returns key if set, None otherwise
        """
        # List of environment variables we want to detect and potentially use
        env_vars_to_check = [
            "CBORG_API_KEY",
            "AMSC_I2_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "ARGO_API_KEY",
            "STANFORD_API_KEY",
            "ALS_APG_API_KEY",
            "PROJECT_ROOT",
            "LOCAL_PYTHON_VENV",
            "CONFLUENCE_ACCESS_TOKEN",
            "TZ",
        ]

        detected = {}
        for var in env_vars_to_check:
            value = os.environ.get(var)
            if value:  # Only include if the variable is set and non-empty
                detected[var] = value

        return detected

    def list_app_templates(self) -> list[str]:
        """List available application templates.

        Returns:
            List of template names (directory names in templates/apps/)

        Examples:
            >>> manager = TemplateManager()
            >>> manager.list_app_templates()
            ['minimal', 'hello_world_weather', 'wind_turbine']
        """
        apps_dir = self.template_root / "apps"
        if not apps_dir.exists():
            return []

        return sorted(
            [d.name for d in apps_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        )

    def render_template(self, template_path: str, context: dict[str, Any], output_path: Path):
        """Render a single template file.

        Args:
            template_path: Relative path to template within templates directory
            context: Dictionary of variables for template rendering
            output_path: Path where rendered output should be written

        Raises:
            jinja2.TemplateNotFound: If template file doesn't exist
            IOError: If output file cannot be written
        """
        template = self.jinja_env.get_template(template_path)
        rendered = template.render(**context)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use UTF-8 encoding explicitly to support Unicode characters on Windows
        output_path.write_text(rendered, encoding="utf-8")

    def create_project(
        self,
        project_name: str,
        output_dir: Path,
        template_name: str = "minimal",
        registry_style: str = "extend",
        context: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Path:
        """Create complete project from template.

        This is the main entry point for project creation. It:
        1. Validates template exists
        2. Creates project directory structure
        3. Renders and copies project files
        4. Copies service configurations
        5. Creates application code from template

        Args:
            project_name: Name of the project (e.g., "my-assistant")
            output_dir: Parent directory where project will be created
            template_name: Application template to use (default: "minimal")
            registry_style: Registry style - "extend" (recommended) or "standalone" (advanced)
            context: Additional template context variables
            force: If True, skip existence check (used when caller already handled deletion)

        Returns:
            Path to created project directory

        Raises:
            ValueError: If template doesn't exist or project directory exists

        Examples:
            >>> manager = TemplateManager()
            >>> project_dir = manager.create_project(
            ...     "my-assistant",
            ...     Path("/projects"),
            ...     template_name="minimal",
            ...     registry_style="extend"
            ... )
            >>> print(project_dir)
            /projects/my-assistant
        """
        # 1. Validate template exists
        app_templates = self.list_app_templates()
        if template_name not in app_templates:
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available templates: {', '.join(app_templates)}"
            )

        # 2. Setup project directory
        project_dir = output_dir / project_name
        if not force and project_dir.exists():
            raise ValueError(
                f"Directory '{project_dir}' already exists. "
                "Please choose a different project name or location."
            )

        if not project_dir.exists():
            project_dir.mkdir(parents=True)

        # 3. Prepare template context
        package_name = project_name.replace("-", "_").lower()
        class_name = self._generate_class_name(package_name)

        # Detect current Python environment
        import sys

        current_python = sys.executable

        # Detect environment variables from the system
        detected_env_vars = self._detect_environment_variables()

        ctx = {
            "project_name": project_name,
            "package_name": package_name,
            "app_display_name": project_name,  # Used in templates for display/documentation
            "app_class_name": class_name,  # Used in templates for class names
            "registry_class_name": class_name,  # Backward compatibility
            "project_description": f"{project_name} - Osprey Agent Application",
            "framework_version": self._get_framework_version(),
            "project_root": str(project_dir.absolute()),
            "venv_path": "${LOCAL_PYTHON_VENV}",
            "current_python_env": current_python,  # Actual path to current Python
            "default_provider": "cborg",
            "default_model": "anthropic/claude-haiku",
            "template_name": template_name,  # Make template name available in config.yml
            # Add detected environment variables
            "env": detected_env_vars,
            **(context or {}),
        }

        # Derive channel finder configuration if control_assistant template
        if template_name == "control_assistant":
            channel_finder_mode = ctx.get("channel_finder_mode", "all")

            # Derive boolean flags for conditional templates
            enable_in_context = channel_finder_mode in ["in_context", "all"]
            enable_hierarchical = channel_finder_mode in ["hierarchical", "all"]
            enable_middle_layer = channel_finder_mode in ["middle_layer", "all"]

            # Determine default pipeline (for config.yml)
            if channel_finder_mode == "all":
                default_pipeline = "hierarchical"  # Default to most scalable option
            else:
                default_pipeline = channel_finder_mode

            # Determine which pipeline module to use for MCP server
            if channel_finder_mode == "all":
                channel_finder_pipeline = default_pipeline  # "hierarchical"
            else:
                channel_finder_pipeline = channel_finder_mode

            # Add channel finder context variables
            ctx.update(
                {
                    "channel_finder_mode": channel_finder_mode,
                    "enable_in_context": enable_in_context,
                    "enable_hierarchical": enable_hierarchical,
                    "enable_middle_layer": enable_middle_layer,
                    "default_pipeline": default_pipeline,
                    "channel_finder_pipeline": channel_finder_pipeline,
                    "facility_name": ctx.get("facility_name", project_name),
                }
            )

        claude_code_only = ctx.get("claude_code_only", False)

        # 4. Create project structure
        self._create_project_structure(project_dir, template_name, ctx)

        # 5. Copy services (conditional on mode)
        if claude_code_only:
            # Claude Code mode: only postgresql for control_assistant, nothing for others
            if template_name == "control_assistant":
                self._copy_services_selective(project_dir, ["postgresql"])
        else:
            self.copy_services(project_dir)

        # 6. Create application code (or just data files in claude_code_only mode)
        if claude_code_only:
            # Copy only data/ subdirectories from template (no src/ package)
            self._copy_template_data(project_dir, package_name, template_name, ctx)
        else:
            src_dir = project_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            self._create_application_code(
                src_dir, package_name, template_name, ctx, registry_style, project_dir
            )

        # 7. Create _agent_data directory structure
        self._create_agent_data_structure(project_dir, ctx)

        # 8. Create Claude Code integration files (if requested)
        if ctx.get("claude_code", True):  # Default ON
            # Load rendered config.yml so conditional sections (confluence, etc.)
            # are available to Claude Code templates (mcp.json.j2, CLAUDE.md.j2).
            config_file = project_dir / "config.yml"
            if config_file.exists():
                with open(config_file) as f:
                    rendered_config = yaml.safe_load(f) or {}
                if "confluence" in rendered_config:
                    ctx["confluence"] = rendered_config["confluence"]
                if "matlab" in rendered_config:
                    ctx["matlab"] = rendered_config["matlab"]
                if "deplot" in rendered_config:
                    ctx["deplot"] = rendered_config["deplot"]
                # Claude Code explicit overrides
                cc_config = rendered_config.get("claude_code", {})
                ctx.setdefault("disable_servers", cc_config.get("disable_servers", []))
                ctx.setdefault("disable_agents", cc_config.get("disable_agents", []))
                ctx.setdefault("extra_servers", cc_config.get("extra_servers", {}))
                # Model provider resolution for init-time rendering
                from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

                api_providers = rendered_config.get("api", {}).get("providers", {})
                try:
                    model_spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
                except ValueError:
                    model_spec = None
                ctx["claude_code_model_spec"] = model_spec
            # Ensure defaults exist even without rendered config
            ctx.setdefault("disable_servers", [])
            ctx.setdefault("disable_agents", [])
            ctx.setdefault("extra_servers", {})
            self._create_claude_code_integration(project_dir, ctx)

        return project_dir

    def _create_project_structure(self, project_dir: Path, template_name: str, ctx: dict):
        """Create base project files (config, README, pyproject.toml, etc.).

        Args:
            project_dir: Root directory of the project
            template_name: Name of the application template being used
            ctx: Template context variables
        """
        project_template_dir = self.template_root / "project"
        app_template_dir = self.template_root / "apps" / template_name

        claude_code_only = ctx.get("claude_code_only", False)

        # Render template files
        files_to_render = [
            ("config.yml.j2", "config.yml"),
            ("env.example.j2", ".env.example"),
            ("README.md.j2", "README.md"),
        ]
        # Skip pyproject.toml and requirements.txt in claude_code_only mode (no src/ package)
        if not claude_code_only:
            files_to_render.extend([
                ("pyproject.toml.j2", "pyproject.toml"),
                ("requirements.txt", "requirements.txt"),  # Render to replace framework_version
            ])

        # Copy static files
        static_files = [
            # requirements.txt moved to rendered templates to handle {{ framework_version }}
        ]

        for template_file, output_file in files_to_render:
            # Check if app template has its own version first (e.g., requirements.txt.j2)
            app_specific_template = app_template_dir / (
                template_file + ".j2" if not template_file.endswith(".j2") else template_file
            )
            default_template = project_template_dir / template_file

            if app_specific_template.exists():
                # Use app-specific template
                self.render_template(
                    f"apps/{template_name}/{app_specific_template.name}",
                    ctx,
                    project_dir / output_file,
                )
            elif default_template.exists():
                # Use default project template
                self.render_template(f"project/{template_file}", ctx, project_dir / output_file)

        # Create .env file only if API keys are detected
        detected_env_vars = ctx.get("env", {})
        api_keys = [
            "CBORG_API_KEY",
            "AMSC_I2_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "ARGO_API_KEY",
            "STANFORD_API_KEY",
            "ALS_APG_API_KEY",
        ]
        has_api_keys = any(key in detected_env_vars for key in api_keys)

        if has_api_keys:
            env_template = project_template_dir / "env.j2"
            if env_template.exists():
                self.render_template("project/env.j2", ctx, project_dir / ".env")
                # Set proper permissions (owner read/write only)
                import os

                os.chmod(project_dir / ".env", 0o600)

        # Copy static files
        for src_name, dst_name in static_files:
            src_file = project_template_dir / src_name
            if src_file.exists():
                shutil.copy(src_file, project_dir / dst_name)

        # Copy gitignore (renamed from 'gitignore' to '.gitignore')
        gitignore_source = project_template_dir / "gitignore"
        if gitignore_source.exists():
            shutil.copy(gitignore_source, project_dir / ".gitignore")

        # Render code generator config based on selected generator
        generator_configs = {
            "claude_code": "claude_generator_config.yml",
            "basic": "basic_generator_config.yml",
        }
        selected_generator = ctx.get("code_generator")
        if selected_generator in generator_configs:
            config_filename = generator_configs[selected_generator]
            config_template = app_template_dir / f"{config_filename}.j2"
            if config_template.exists():
                self.render_template(
                    f"apps/{template_name}/{config_filename}.j2",
                    ctx,
                    project_dir / config_filename,
                )

    def copy_services(self, project_dir: Path):
        """Copy service configurations to project (flattened structure).

        Services are copied with a flattened structure (not nested under osprey/).
        This makes the user's project structure cleaner.

        Args:
            project_dir: Root directory of the project
        """
        src_services = self.template_root / "services"
        dst_services = project_dir / "services"

        if not src_services.exists():
            return

        dst_services.mkdir(parents=True, exist_ok=True)

        # Copy each service directory individually (flattened)
        for item in src_services.iterdir():
            if item.is_dir():
                shutil.copytree(item, dst_services / item.name, dirs_exist_ok=True)
            elif item.is_file() and item.suffix in [".j2", ".yml", ".yaml"]:
                # Copy docker-compose template/config files
                shutil.copy(item, dst_services / item.name)

    def _copy_services_selective(self, project_dir: Path, service_names: list[str]):
        """Copy only specified service directories to project.

        Args:
            project_dir: Root directory of the project
            service_names: List of service directory names to copy (e.g., ["postgresql"])
        """
        src_services = self.template_root / "services"
        dst_services = project_dir / "services"

        if not src_services.exists():
            return

        dst_services.mkdir(parents=True, exist_ok=True)

        for name in service_names:
            src_dir = src_services / name
            if src_dir.is_dir():
                shutil.copytree(src_dir, dst_services / name, dirs_exist_ok=True)

        # Also copy docker-compose template if any services were copied
        if service_names:
            for item in src_services.iterdir():
                if item.is_file() and item.suffix in [".j2", ".yml", ".yaml"]:
                    shutil.copy(item, dst_services / item.name)

    def _copy_template_data(
        self,
        project_dir: Path,
        package_name: str,
        template_name: str,
        ctx: dict,
    ):
        """Copy data files from template to project root (no src/ package).

        In claude_code_only mode, data files (channel databases, channel_limits.json,
        logbook seeds, benchmark datasets) are placed at project_dir/data/ instead
        of inside a src/<package>/ directory.

        Args:
            project_dir: Root directory of the project
            package_name: Python package name (used to locate template data dirs)
            template_name: Name of the application template
            ctx: Template context variables
        """
        app_template_dir = self.template_root / "apps" / template_name

        # Look for data/ subdirectory in the template
        template_data_dir = app_template_dir / "data"
        if template_data_dir.exists() and template_data_dir.is_dir():
            dst_data = project_dir / "data"
            shutil.copytree(template_data_dir, dst_data, dirs_exist_ok=True)
            console.print(
                f"  [success]✓[/success] Copied template data files to [path]{dst_data}[/path]"
            )
            return

        # Fallback: scan for data/ directories inside template subdirectories
        # (some templates put data inside package-level dirs)
        for template_file in app_template_dir.rglob("*"):
            if not template_file.is_dir():
                continue
            if template_file.name == "data":
                # Copy to project root data/ (flatten from template structure)
                dst_data = project_dir / "data"
                if not dst_data.exists():
                    shutil.copytree(template_file, dst_data, dirs_exist_ok=True)
                else:
                    # Merge into existing data/
                    for item in template_file.iterdir():
                        dst_item = dst_data / item.name
                        if item.is_dir():
                            shutil.copytree(item, dst_item, dirs_exist_ok=True)
                        elif item.is_file():
                            dst_item.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dst_item)
                console.print(
                    f"  [success]✓[/success] Copied template data files to [path]{dst_data}[/path]"
                )
                return

    def _create_application_code(
        self,
        src_dir: Path,
        package_name: str,
        template_name: str,
        ctx: dict,
        registry_style: str = "extend",
        project_root: Path = None,
    ):
        """Create application code from template.

        Args:
            src_dir: src/ directory where package will be created
            package_name: Python package name (e.g., "my_assistant")
            template_name: Name of the application template
            ctx: Template context variables
            registry_style: Registry style - "extend" or "standalone"
            project_root: Actual project root (for placing scripts/ at root)

        Note:
            All templates support both extend and standalone styles. The extend style
            renders the template as-is. The standalone style uses generate_explicit_registry_code()
            to dynamically create a full registry with all framework + app components listed.
            This approach works generically for all templates without needing template-specific logic.

            Special handling: Files in scripts/ directory are placed at project root
            instead of inside the package to provide convenient CLI access.
        """
        app_template_dir = self.template_root / "apps" / template_name
        app_dir = src_dir / package_name
        app_dir.mkdir(parents=True)

        # Use src_dir's parent as project_root if not provided
        if project_root is None:
            project_root = src_dir.parent

        # Add registry_style to context for templates that might use it
        ctx["registry_style"] = registry_style

        # Project-level files that should only live at project root, not in src/
        # These are handled by _create_project_structure() and should be skipped here
        PROJECT_LEVEL_FILES = {
            "config.yml.j2",
            "config.yml",
            "README.md.j2",
            "README.md",
            "env.example.j2",
            "env.example",
            "env.j2",
            ".env",
            "requirements.txt.j2",
            "requirements.txt",
            "pyproject.toml.j2",
            "pyproject.toml",
            "claude_generator_config.yml.j2",
            "claude_generator_config.yml",
            "basic_generator_config.yml.j2",
            "basic_generator_config.yml",
        }

        # Process all files in the template
        for template_file in app_template_dir.rglob("*"):
            if not template_file.is_file():
                continue

            rel_path = template_file.relative_to(app_template_dir)

            # Skip project-level files at template root (they're handled by _create_project_structure)
            if len(rel_path.parts) == 1 and rel_path.name in PROJECT_LEVEL_FILES:
                continue

            # Special handling for scripts/ directory - place at project root
            if rel_path.parts[0] == "scripts":
                base_output_dir = project_root
                output_rel_path = rel_path
            else:
                base_output_dir = app_dir
                output_rel_path = rel_path

            # Determine output path
            if template_file.suffix == ".j2":
                # Template file - render it
                output_name = template_file.stem  # Remove .j2 extension
                output_path = base_output_dir / output_rel_path.parent / output_name

                # Special handling for standalone registry style
                if registry_style == "standalone" and output_name == "registry.py":
                    self._generate_explicit_registry(output_path, ctx, template_name)
                else:
                    # Convert Windows backslashes to forward slashes for Jinja2
                    # (harmless on Linux/macOS where paths already use forward slashes)
                    template_path_str = f"apps/{template_name}/{rel_path}".replace("\\", "/")
                    self.render_template(template_path_str, ctx, output_path)
            else:
                # Static file - copy directly
                output_path = base_output_dir / output_rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(template_file, output_path)

    def _get_framework_version(self) -> str:
        """Get current osprey version.

        Returns:
            Version string (e.g., "0.7.0")
        """
        try:
            from osprey import __version__

            return __version__
        except (ImportError, AttributeError):
            return "0.7.0"

    def _generate_class_name(self, package_name: str) -> str:
        """Generate a PascalCase class name prefix from package name.

        Args:
            package_name: Python package name (e.g., "my_assistant")

        Returns:
            PascalCase class name prefix (e.g., "MyAssistant")
            Note: The template adds "RegistryProvider" suffix

        Examples:
            >>> TemplateManager()._generate_class_name("my_assistant")
            'MyAssistant'
            >>> TemplateManager()._generate_class_name("weather_app")
            'WeatherApp'
        """
        # Convert snake_case to PascalCase
        words = package_name.split("_")
        class_name = "".join(word.capitalize() for word in words)
        return class_name

    def _generate_explicit_registry(self, output_path: Path, ctx: dict, template_name: str):
        """Generate explicit registry code using the generic code generation function.

        This method parses the template to extract app-specific components and uses
        the generate_explicit_registry_code() function to create the full explicit registry.

        Args:
            output_path: Where to write the generated registry.py
            ctx: Template context with app_class_name, app_display_name, package_name
            template_name: Name of the template being processed
        """
        from osprey.registry import (
            CapabilityRegistration,
            ContextClassRegistration,
            generate_explicit_registry_code,
        )

        # Read the compact template to extract app-specific components
        template_path = self.template_root / "apps" / template_name / "registry.py.j2"
        with open(template_path) as f:
            template_content = f.read()

        # Extract capabilities and context classes by parsing the template
        # This is a simple parser that looks for CapabilityRegistration and ContextClassRegistration calls
        capabilities = []
        context_classes = []

        # Parse CapabilityRegistration entries
        capability_pattern = r"CapabilityRegistration\((.*?)\)"
        for match in re.finditer(capability_pattern, template_content, re.DOTALL):
            reg_content = match.group(1)

            # Extract parameters (simple approach - could be more robust)
            name_match = re.search(r'name\s*=\s*"([^"]+)"', reg_content)
            module_path_match = re.search(r'module_path\s*=\s*"([^"]+)"', reg_content)
            class_name_match = re.search(r'class_name\s*=\s*"([^"]+)"', reg_content)
            description_match = re.search(r'description\s*=\s*"([^"]+)"', reg_content)
            provides_match = re.search(r"provides\s*=\s*\[([^\]]+)\]", reg_content)
            requires_match = re.search(r"requires\s*=\s*\[([^\]]*)\]", reg_content)

            if name_match and module_path_match and class_name_match:
                # Process provides list
                provides = []
                if provides_match:
                    provides_str = provides_match.group(1)
                    provides = [item.strip().strip("\"'") for item in provides_str.split(",")]

                # Process requires list
                requires = []
                if requires_match and requires_match.group(1).strip():
                    requires_str = requires_match.group(1)
                    requires = [item.strip().strip("\"'") for item in requires_str.split(",")]

                # Substitute template variables
                module_path = module_path_match.group(1).replace(
                    "{{ package_name }}", ctx["package_name"]
                )
                description = description_match.group(1) if description_match else ""

                capabilities.append(
                    CapabilityRegistration(
                        name=name_match.group(1),
                        module_path=module_path,
                        class_name=class_name_match.group(1),
                        description=description,
                        provides=provides,
                        requires=requires,
                    )
                )

        # Parse ContextClassRegistration entries
        context_pattern = r"ContextClassRegistration\((.*?)\)"
        for match in re.finditer(context_pattern, template_content, re.DOTALL):
            reg_content = match.group(1)

            context_type_match = re.search(r'context_type\s*=\s*"([^"]+)"', reg_content)
            module_path_match = re.search(r'module_path\s*=\s*"([^"]+)"', reg_content)
            class_name_match = re.search(r'class_name\s*=\s*"([^"]+)"', reg_content)

            if context_type_match and module_path_match and class_name_match:
                # Substitute template variables
                module_path = module_path_match.group(1).replace(
                    "{{ package_name }}", ctx["package_name"]
                )

                context_classes.append(
                    ContextClassRegistration(
                        context_type=context_type_match.group(1),
                        module_path=module_path,
                        class_name=class_name_match.group(1),
                    )
                )

        # Generate the explicit registry code
        registry_code = generate_explicit_registry_code(
            app_class_name=ctx["app_class_name"],
            app_display_name=ctx["app_display_name"],
            package_name=ctx["package_name"],
            capabilities=capabilities if capabilities else None,
            context_classes=context_classes if context_classes else None,
        )

        # Write to output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use UTF-8 encoding explicitly to support Unicode characters on Windows
        output_path.write_text(registry_code, encoding="utf-8")

    def _create_agent_data_structure(self, project_dir: Path, ctx: dict):
        """Create _agent_data directory structure for the project.

        This method creates the agent data directory and all standard subdirectories
        based on osprey's default configuration. This ensures that container
        deployments won't fail due to missing mount points.

        Args:
            project_dir: Root directory of the project
            ctx: Template context variables (used for conditional directory creation)
        """
        # Create main _agent_data directory
        agent_data_dir = project_dir / "_agent_data"
        agent_data_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories based on mode
        claude_code_only = ctx.get("claude_code_only", False)
        if claude_code_only:
            # Trimmed set for Claude Code-only projects
            subdirs = [
                "executed_scripts",
                "user_memory",
                "api_calls",
            ]
        else:
            subdirs = [
                "executed_scripts",
                "execution_plans",
                "user_memory",
                "registry_exports",
                "prompts",
                "checkpoints",
                "api_calls",
            ]

        # Conditionally add example_scripts for control_assistant with claude_code generator
        template_name = ctx.get("template_name", "")
        code_generator = ctx.get("code_generator", "")
        copy_example_scripts = (
            template_name == "control_assistant" and code_generator == "claude_code"
        )

        if copy_example_scripts:
            subdirs.append("example_scripts/plotting")

        for subdir in subdirs:
            subdir_path = agent_data_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)

        # Copy example script files if using claude_code generator
        if copy_example_scripts:
            template_examples_dir = (
                self.template_root
                / "apps"
                / "control_assistant"
                / "_agent_data"
                / "example_scripts"
            )
            if template_examples_dir.exists():
                # Copy plotting examples
                template_plotting = template_examples_dir / "plotting"
                project_plotting = agent_data_dir / "example_scripts" / "plotting"

                if template_plotting.exists():
                    # Copy all Python and README files
                    files_copied = 0
                    for file_path in template_plotting.iterdir():
                        if file_path.is_file() and (
                            file_path.suffix == ".py" or file_path.name == "README.md"
                        ):
                            shutil.copy2(file_path, project_plotting / file_path.name)
                            files_copied += 1

                    if files_copied > 0:
                        console.print(
                            f"  [success]✓[/success] Copied {files_copied} example script(s) to [path]_agent_data/example_scripts/plotting/[/path]"
                        )
                else:
                    console.print(
                        f"  [warning]⚠[/warning] Template example scripts not found at {template_plotting}",
                        style="yellow",
                    )

        console.print(
            f"  [success]✓[/success] Created agent data structure at [path]{agent_data_dir}[/path]"
        )

        # Create a README to explain the directory structure
        if claude_code_only:
            readme_content = """# Agent Data Directory

This directory contains runtime data for the Claude Code project:

- `executed_scripts/`: Python scripts executed via MCP tools
- `user_memory/`: User memory data
- `api_calls/`: Raw LLM API inputs/outputs (when API logging enabled)
"""
        else:
            # Base content for full LangGraph projects
            readme_content = """# Agent Data Directory

This directory contains runtime data generated by the Osprey Framework:

- `executed_scripts/`: Python scripts executed by the framework
- `execution_plans/`: Orchestrator execution plans (JSON format)
- `user_memory/`: User memory data and conversation history
- `registry_exports/`: Exported registry information
- `prompts/`: Generated prompts (when debug mode enabled)
- `checkpoints/`: LangGraph checkpoints for conversation state
- `api_calls/`: Raw LLM API inputs/outputs (when API logging enabled)
"""

        # Add example_scripts section if using Claude Code generator
        if template_name == "control_assistant" and code_generator == "claude_code":
            readme_content += """- `example_scripts/`: Example code for Claude Code generator to learn from

## Example Scripts

The `example_scripts/` directory contains example code that the Claude Code generator
can read and learn from when generating code. The framework has provided starter
examples organized by category:

- `example_scripts/plotting/`: Matplotlib visualization examples (included)
  - Basic time series plotting
  - Multi-subplot layouts
  - Publication-quality figures
  - Aligned multi-plot arrays

- `example_scripts/analysis/`: Data analysis patterns (add your own)
- `example_scripts/archiver/`: Archiver retrieval examples (add your own)

**Security Note:** Claude Code can ONLY read files in these example directories.
It cannot access your project configuration, secrets, or other sensitive files.
The directories listed in `claude_generator_config.yml` are the only accessible paths.

Add your own examples to help Claude generate better code for your specific use cases!

"""

        readme_content += """
This directory is excluded from git (see .gitignore) but is required for
proper framework operation, especially when using containerized services.
"""

        readme_path = agent_data_dir / "README.md"
        # Use UTF-8 encoding explicitly to support Unicode characters on Windows
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    def _build_claude_code_context(self, project_dir: Path, config: dict) -> dict:
        """Build template context for Claude Code artifact rendering.

        Reconstructs the template context needed by Claude Code templates
        (.mcp.json, CLAUDE.md, settings.json, agents) from the project's
        config.yml and manifest.

        Args:
            project_dir: Root directory of the project
            config: Parsed config.yml dictionary

        Returns:
            Template context dict suitable for Claude Code templates
        """
        import sys

        project_name = config.get("project_name", project_dir.name)
        package_name = project_name.replace("-", "_").lower()

        # Read template_name from manifest if available
        manifest_path = project_dir / MANIFEST_FILENAME
        template_name = "minimal"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                template_name = manifest.get("creation", {}).get("template", "minimal")
            except (json.JSONDecodeError, OSError):
                pass

        ctx = {
            "project_name": project_name,
            "package_name": package_name,
            "project_root": str(project_dir.absolute()),
            "current_python_env": sys.executable,
            "template_name": template_name,
            "facility_name": config.get("facility_name", project_name),
        }

        # Derive channel finder configuration
        channel_finder = config.get("channel_finder")
        if channel_finder and template_name == "control_assistant":
            pipeline_mode = channel_finder.get("pipeline_mode", "hierarchical")
            ctx["channel_finder_pipeline"] = pipeline_mode
            ctx["channel_finder_mode"] = pipeline_mode
            ctx["default_pipeline"] = pipeline_mode

        # Pass through optional config sections
        if "confluence" in config:
            ctx["confluence"] = config["confluence"]
        if "matlab" in config:
            ctx["matlab"] = config["matlab"]
        if "deplot" in config:
            ctx["deplot"] = config["deplot"]

        # Claude Code explicit overrides
        claude_code_config = config.get("claude_code", {})
        ctx["disable_servers"] = claude_code_config.get("disable_servers", [])
        ctx["disable_agents"] = claude_code_config.get("disable_agents", [])
        ctx["extra_servers"] = claude_code_config.get("extra_servers", {})
        # User-owned files: regen skips these, users edit in-place
        ctx["user_owned"] = config.get("prompts", {}).get("user_owned", [])

        # Model provider resolution for Claude Code
        from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

        api_providers = config.get("api", {}).get("providers", {})
        try:
            model_spec = ClaudeCodeModelResolver.resolve(claude_code_config, api_providers)
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
            model_spec = None
        ctx["claude_code_model_spec"] = model_spec

        return ctx

    # Known framework-managed files for checksum collection during regen.
    _REGEN_TRACKED_FILES = [
        "CLAUDE.md",
        ".mcp.json",
        ".claude/settings.json",
        ".claude/rules/safety.md",
        ".claude/rules/error-handling.md",
        ".claude/rules/artifacts.md",
        ".claude/rules/facility.md",
        ".claude/hooks/osprey_writes_check.py",
        ".claude/hooks/osprey_limits.py",
        ".claude/hooks/osprey_approval.py",
        ".claude/hooks/osprey_error_guidance.py",
        ".claude/hooks/osprey_notebook_update.py",
        ".claude/rules/code-generation.md",
        ".claude/commands/diagnose.md",
        ".claude/skills/session-report/SKILL.md",
        ".claude/skills/session-report/reference.md",
    ]

    def _compute_regen_summary(self, ctx: dict) -> dict:
        """Compute active/disabled server and agent lists from template context.

        Args:
            ctx: Template context dict with disable_servers, disable_agents, extra_servers,
                 and optional confluence/matlab/channel_finder_pipeline keys.

        Returns:
            Dict with active_servers, disabled_servers, active_agents, disabled_agents keys.
        """
        disable_servers = ctx.get("disable_servers", [])
        disable_agents = ctx.get("disable_agents", [])
        extra_servers = ctx.get("extra_servers", {})

        # Core servers (always available)
        core_servers = [
            "controls",
            "python",
            "workspace",
            "ariel",
            "accelpapers",
        ]

        # Conditional servers (from config sections)
        all_servers = list(core_servers)
        if ctx.get("confluence"):
            all_servers.append("confluence")
        if ctx.get("matlab"):
            all_servers.append("matlab")
        if ctx.get("channel_finder_pipeline"):
            all_servers.append("channel-finder")
        all_servers.extend(extra_servers.keys())

        # Core agents (always available)
        core_agents = ["logbook-search", "logbook-deep-research", "literature-search"]

        # Conditional agents (from config sections)
        all_agents = list(core_agents)
        if ctx.get("confluence"):
            all_agents.append("wiki-search")
        if ctx.get("matlab"):
            all_agents.append("matlab-search")
        if ctx.get("channel_finder_pipeline"):
            all_agents.append("channel-finder")
        if ctx.get("deplot"):
            all_agents.append("graph-analyst")

        active_servers = [s for s in all_servers if s not in disable_servers]
        active_agents = [a for a in all_agents if a not in disable_agents]
        actual_disabled_servers = [s for s in disable_servers if s in all_servers]
        actual_disabled_agents = [a for a in disable_agents if a in all_agents]

        return {
            "active_servers": active_servers,
            "disabled_servers": actual_disabled_servers,
            "extra_servers": list(extra_servers.keys()),
            "active_agents": active_agents,
            "disabled_agents": actual_disabled_agents,
        }

    def regenerate_claude_code(self, project_dir: Path, dry_run: bool = False) -> dict:
        """Regenerate Claude Code artifacts from current config.yml.

        Reads config.yml, reconstructs the template context, and re-renders
        all Claude Code .j2 templates. Backs up existing files before overwriting.

        Args:
            project_dir: Root directory of the project
            dry_run: If True, report what would change without writing files

        Returns:
            Dict with 'changed', 'unchanged', and 'backup_dir' keys

        Raises:
            FileNotFoundError: If config.yml doesn't exist in project_dir
        """
        config_file = project_dir / "config.yml"
        if not config_file.exists():
            raise FileNotFoundError(
                f"No config.yml found in {project_dir}. "
                "Are you in an OSPREY project directory?"
            )

        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        ctx = self._build_claude_code_context(project_dir, config)

        # Collect checksums of existing Claude Code files before regeneration.
        # Use known tracked files, then add agent files (always auto-managed).
        claude_code_files = list(self._REGEN_TRACKED_FILES)
        agents_dir = project_dir / ".claude" / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.iterdir():
                if agent_file.is_file() and agent_file.suffix == ".md":
                    rel = f".claude/agents/{agent_file.name}"
                    if rel not in claude_code_files:
                        claude_code_files.append(rel)

        old_checksums = {}
        for rel_path in claude_code_files:
            file_path = project_dir / rel_path
            if file_path.exists():
                old_checksums[rel_path] = self._sha256_file(file_path)

        if dry_run:
            # Render to temp dir and compare
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                # Create necessary subdirectories
                (tmp_dir / ".claude").mkdir(parents=True, exist_ok=True)
                self._create_claude_code_integration(tmp_dir, ctx)

                changed = []
                unchanged = []
                for rel_path in claude_code_files:
                    tmp_file = tmp_dir / rel_path
                    orig_file = project_dir / rel_path
                    if tmp_file.exists():
                        new_checksum = self._sha256_file(tmp_file)
                        old_checksum = old_checksums.get(rel_path)
                        if old_checksum != new_checksum:
                            changed.append(rel_path)
                        else:
                            unchanged.append(rel_path)
                    elif orig_file.exists():
                        changed.append(rel_path)  # File would be removed

                # Check for new files in tmp that aren't in old list
                for tmp_file in Path(tmp).rglob("*"):
                    if not tmp_file.is_file():
                        continue
                    rel = str(tmp_file.relative_to(tmp))
                    if rel not in claude_code_files and rel not in changed:
                        changed.append(rel)

                summary = self._compute_regen_summary(ctx)
                return {"changed": changed, "unchanged": unchanged, "backup_dir": None, **summary}

        # Create backup
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = project_dir / "osprey-workspace" / "backup" / f"claude-code-{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in claude_code_files:
            src = project_dir / rel_path
            if src.exists():
                dst = backup_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # Regenerate
        self._create_claude_code_integration(project_dir, ctx)

        # Compare checksums
        changed = []
        unchanged = []
        for rel_path in claude_code_files:
            file_path = project_dir / rel_path
            if file_path.exists():
                new_checksum = self._sha256_file(file_path)
                old_checksum = old_checksums.get(rel_path)
                if old_checksum != new_checksum:
                    changed.append(rel_path)
                else:
                    unchanged.append(rel_path)

        # Check for newly created files (e.g., new agents)
        new_agents_dir = project_dir / ".claude" / "agents"
        if new_agents_dir.exists():
            for agent_file in new_agents_dir.iterdir():
                if agent_file.is_file() and agent_file.suffix == ".md":
                    rel = f".claude/agents/{agent_file.name}"
                    if rel not in claude_code_files and rel not in changed:
                        changed.append(rel)

        # Check for user-owned drift (framework template changed since claiming)
        drift_warnings = self._check_user_owned_drift(project_dir, ctx)

        # Compute active/disabled summary
        summary = self._compute_regen_summary(ctx)

        return {
            "changed": changed,
            "unchanged": unchanged,
            "backup_dir": str(backup_dir),
            "drift_warnings": drift_warnings,
            **summary,
        }

    def _check_user_owned_drift(
        self, project_dir: Path, ctx: dict
    ) -> list[str]:
        """Check if framework templates changed since user claimed ownership.

        Compares the current rendered framework hash against the hash stored
        in the manifest at claim time.

        Returns:
            List of canonical names whose framework template has drifted.
        """
        manifest_path = project_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return []

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        user_owned_meta = manifest.get("user_owned", {})
        if not user_owned_meta:
            return []

        import tempfile

        registry = PromptRegistry.default()
        claude_code_dir = self.template_root / "claude_code"
        drift: list[str] = []

        for canonical_name, meta in user_owned_meta.items():
            stored_hash = meta.get("framework_hash")
            if not stored_hash:
                continue

            artifact = registry.get(canonical_name)
            if artifact is None:
                continue

            template_file = claude_code_dir / artifact.template_path
            if not template_file.exists():
                continue

            current_hash = None
            try:
                if template_file.suffix == ".j2":
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=template_file.stem, delete=False, encoding="utf-8"
                    ) as tmp:
                        template_rel = f"claude_code/{artifact.template_path}"
                        template = self.jinja_env.get_template(template_rel)
                        rendered = template.render(**ctx)
                        tmp.write(rendered)
                        tmp_path = Path(tmp.name)
                    current_hash = f"sha256:{self._sha256_file(tmp_path)}"
                    tmp_path.unlink(missing_ok=True)
                else:
                    current_hash = f"sha256:{self._sha256_file(template_file)}"
            except Exception:
                continue

            if current_hash and current_hash != stored_hash:
                drift.append(canonical_name)
                console.print(
                    f"  [warning]⚠[/warning] Framework updated {canonical_name} since you claimed it.\n"
                    f"    Run `osprey prompts diff {canonical_name}` to review changes.",
                    style="yellow",
                )

        return drift

    @staticmethod
    def _auto_register_user_owned(project_dir: Path, canonical_name: str):
        """Add a canonical name to ``prompts.user_owned`` in config.yml.

        Used during init to mark facility.md as user-owned so regen
        never overwrites user customizations.
        """
        config_path = project_dir / "config.yml"
        if not config_path.exists():
            return
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "prompts" not in data:
            data["prompts"] = {}
        user_owned = data["prompts"].get("user_owned", [])
        if canonical_name not in user_owned:
            user_owned.append(canonical_name)
        data["prompts"]["user_owned"] = user_owned
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _is_user_owned(self, rel_path: str, ctx: dict) -> bool:
        """Check if a file is user-owned (regen should skip it).

        User-owned files are listed in ``prompts.user_owned`` in config.yml.
        During init (empty list), nothing is user-owned so all files are written.
        Agent and skill files are never user-owned (always auto-managed).

        Args:
            rel_path: Relative path from project root (e.g. ".claude/rules/safety.md")
            ctx: Template context (must contain "user_owned" key)
        """
        if rel_path.startswith(".claude/agents/"):
            return False  # agents always auto-managed
        if rel_path.startswith(".claude/skills/"):
            return False  # skills always auto-managed
        user_owned = ctx.get("user_owned", [])
        if not user_owned:
            return False
        registry = PromptRegistry.default()
        art = registry.get_by_output(rel_path)
        return art is not None and art.canonical_name in user_owned

    @staticmethod
    def _output_path_to_canonical(output_path: str, registry: PromptRegistry) -> str | None:
        """Reverse-lookup: map an output file path to its canonical artifact name."""
        art = registry.get_by_output(output_path)
        return art.canonical_name if art else None

    def _create_claude_code_integration(self, project_dir: Path, ctx: dict):
        """Create Claude Code integration files for the project.

        Copies template files from templates/claude_code/ into the project,
        applying dotless-to-dotted naming convention (claude/ -> .claude/,
        mcp.json.j2 -> .mcp.json).

        User-owned files (listed in ``ctx["user_owned"]``) are skipped during
        regeneration, preserving user customizations.

        Args:
            project_dir: Root directory of the project
            ctx: Template context variables
        """
        claude_code_dir = self.template_root / "claude_code"

        if not claude_code_dir.exists():
            console.print(
                "  [warning]⚠[/warning] Claude Code templates not found — skipping",
                style="yellow",
            )
            return

        files_created = 0

        # 1. Render mcp.json.j2 -> .mcp.json
        mcp_template = claude_code_dir / "mcp.json.j2"
        if mcp_template.exists() and not self._is_user_owned(".mcp.json", ctx):
            self.render_template("claude_code/mcp.json.j2", ctx, project_dir / ".mcp.json")
            files_created += 1

        # 2. Render CLAUDE.md.j2 -> CLAUDE.md
        claude_md_j2 = claude_code_dir / "CLAUDE.md.j2"
        claude_md_static = claude_code_dir / "CLAUDE.md"
        if not self._is_user_owned("CLAUDE.md", ctx):
            if claude_md_j2.exists():
                self.render_template("claude_code/CLAUDE.md.j2", ctx, project_dir / "CLAUDE.md")
            elif claude_md_static.exists():
                shutil.copy2(claude_md_static, project_dir / "CLAUDE.md")
            files_created += 1

        # 2b. Create facility.md — user-owned artifact
        # During init, render the template in-place and auto-register as
        # user-owned so regen never overwrites user customizations.
        facility_md = project_dir / ".claude" / "rules" / "facility.md"
        facility_j2 = claude_code_dir / "claude" / "rules" / "facility.md.j2"
        if self._is_user_owned(".claude/rules/facility.md", ctx):
            pass  # Skip — user owns it
        elif not facility_md.exists() and facility_j2.exists():
            facility_md.parent.mkdir(parents=True, exist_ok=True)
            self.render_template(
                "claude_code/claude/rules/facility.md.j2", ctx, facility_md
            )
            # Auto-register as user-owned so regen preserves user edits
            self._auto_register_user_owned(project_dir, "rules/facility")
            files_created += 1

        # 3. Recursively copy/render claude/ -> .claude/ (dotless to dotted)
        #    Files with .j2 extension are rendered as Jinja2 templates.
        #    facility.md.j2 is handled above (create-only), so skip it here.
        claude_src = claude_code_dir / "claude"
        if claude_src.exists():
            for src_file in claude_src.rglob("*"):
                if not src_file.is_file():
                    continue
                rel_path = src_file.relative_to(claude_src)

                if src_file.suffix == ".j2":
                    output_rel = rel_path.with_suffix("")
                    dst_rel = f".claude/{output_rel}"

                    # Skip facility.md — handled above (create-only semantics)
                    if str(output_rel) == "rules/facility.md":
                        continue

                    # Skip user-owned files
                    if self._is_user_owned(dst_rel, ctx):
                        continue

                    # Render Jinja2 template, strip .j2 extension
                    dst_file = project_dir / ".claude" / output_rel
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    template_path = f"claude_code/claude/{rel_path}"
                    self.render_template(template_path, ctx, dst_file)
                else:
                    dst_rel = f".claude/{rel_path}"

                    # Skip user-owned files
                    if self._is_user_owned(dst_rel, ctx):
                        continue

                    dst_file = project_dir / ".claude" / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                files_created += 1

        # 4. Set hook scripts executable
        hooks_dir = project_dir / ".claude" / "hooks"
        if hooks_dir.exists():
            for hook in hooks_dir.iterdir():
                if hook.is_file() and hook.suffix == ".py":
                    hook.chmod(hook.stat().st_mode | 0o755)

        console.print(
            f"  [success]✓[/success] Created {files_created} Claude Code integration file(s)"
        )

    def generate_manifest(
        self,
        project_dir: Path,
        project_name: str,
        template_name: str,
        registry_style: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a project manifest for migration support.

        The manifest captures all information needed to recreate the project
        with the same OSPREY version and settings. This enables future migrations
        by providing a baseline for three-way diffs.

        Args:
            project_dir: Root directory of the created project
            project_name: Name of the project
            template_name: Template used to create the project
            registry_style: Registry style ("extend" or "standalone")
            context: Full context dict used during template rendering

        Returns:
            Dictionary containing the manifest data that was written to file

        Examples:
            >>> manager = TemplateManager()
            >>> manifest = manager.generate_manifest(
            ...     project_dir=Path("/projects/my-assistant"),
            ...     project_name="my-assistant",
            ...     template_name="control_assistant",
            ...     registry_style="extend",
            ...     context={"default_provider": "cborg", "default_model": "claude-haiku"}
            ... )
            >>> manifest["creation"]["template"]
            'control_assistant'
        """
        # Build init_args from context - extract the user-facing options
        init_args = self._extract_init_args(project_name, template_name, registry_style, context)

        # Build reproducible command string
        reproducible_command = self._build_reproducible_command(init_args)

        # Calculate file checksums for trackable files
        file_checksums = self._calculate_file_checksums(project_dir)

        # Get framework version
        framework_version = self._get_framework_version()

        # Build user_owned section from context
        user_owned_manifest = self._build_user_owned_manifest(project_dir, context)

        # Build manifest
        claude_code_only = context.get("claude_code_only", False)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "creation": {
                "osprey_version": framework_version,
                "timestamp": datetime.now(UTC).isoformat(),
                "template": template_name,
                "registry_style": registry_style,
                "claude_code_only": claude_code_only,
            },
            "init_args": init_args,
            "reproducible_command": reproducible_command,
            "file_checksums": file_checksums,
        }

        if user_owned_manifest:
            manifest["user_owned"] = user_owned_manifest

        # Write manifest to file
        manifest_path = project_dir / MANIFEST_FILENAME
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)

        return manifest

    def _extract_init_args(
        self,
        project_name: str,
        template_name: str,
        registry_style: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract init arguments from context for manifest storage.

        This extracts the user-facing init options from the full template context,
        filtering out derived values and internal state.

        Args:
            project_name: Name of the project
            template_name: Template used
            registry_style: Registry style
            context: Full template context

        Returns:
            Dictionary of init arguments that can be used to recreate the project
        """
        # Base arguments that are always present
        init_args = {
            "project_name": project_name,
            "template": template_name,
            "registry_style": registry_style,
        }

        # Track claude_code_only mode
        if context.get("claude_code_only"):
            init_args["claude_code_only"] = True

        # Optional arguments that may be in context
        optional_keys = [
            ("default_provider", "provider"),
            ("default_model", "model"),
            ("channel_finder_mode", "channel_finder_mode"),
            ("code_generator", "code_generator"),
            ("claude_code", "claude_code"),
        ]

        for context_key, arg_key in optional_keys:
            if context_key in context and context[context_key] is not None:
                # For boolean keys, include even when False; for strings, skip empty
                value = context[context_key]
                if isinstance(value, bool) or value:
                    init_args[arg_key] = value

        return init_args

    def _build_reproducible_command(self, init_args: dict[str, Any]) -> str:
        """Build a reproducible CLI command from init arguments.

        Args:
            init_args: Dictionary of init arguments

        Returns:
            CLI command string that can recreate the project
        """
        # Use init-legacy command for non-claude_code_only projects
        if init_args.get("claude_code_only"):
            cmd = "init"
        else:
            cmd = "init-legacy"
        parts = ["osprey", cmd, init_args["project_name"]]

        # Add template if not default
        if init_args.get("template") and init_args["template"] != "minimal":
            parts.extend(["--template", init_args["template"]])

        # Add registry style if not default (only for init-legacy)
        if cmd == "init-legacy":
            if init_args.get("registry_style") and init_args["registry_style"] != "extend":
                parts.extend(["--registry-style", init_args["registry_style"]])

        # Add provider if specified
        if init_args.get("provider"):
            parts.extend(["--provider", init_args["provider"]])

        # Add model if specified
        if init_args.get("model"):
            parts.extend(["--model", init_args["model"]])

        # Add channel_finder_mode if specified
        if init_args.get("channel_finder_mode"):
            parts.extend(["--channel-finder-mode", init_args["channel_finder_mode"]])

        # Add code_generator if specified
        if init_args.get("code_generator"):
            parts.extend(["--code-generator", init_args["code_generator"]])

        # Add --no-claude-code if claude_code is explicitly False
        if init_args.get("claude_code") is False:
            parts.append("--no-claude-code")

        return " ".join(parts)

    def _calculate_file_checksums(self, project_dir: Path) -> dict[str, str]:
        """Calculate SHA256 checksums for trackable project files.

        Trackable files are those that come from templates and may change
        between OSPREY versions. This excludes:
        - .env files (contain secrets)
        - _agent_data/ (runtime data)
        - data/ directories (user data)
        - __pycache__/ and .pyc files
        - .git/ directory

        Args:
            project_dir: Root directory of the project

        Returns:
            Dictionary mapping relative file paths to their SHA256 checksums
        """
        checksums = {}

        # Patterns to exclude
        exclude_patterns = {
            ".env",
            ".git",
            "__pycache__",
            ".pyc",
            "_agent_data",
            "data",
            ".osprey-manifest.json",  # Don't checksum ourselves
        }

        # Walk the project directory
        for file_path in project_dir.rglob("*"):
            if not file_path.is_file():
                continue

            # Get relative path
            rel_path = file_path.relative_to(project_dir)
            rel_path_str = str(rel_path)

            # Skip excluded patterns
            skip = False
            for pattern in exclude_patterns:
                if pattern in rel_path.parts or rel_path_str.startswith(pattern):
                    skip = True
                    break
            if skip:
                continue

            # Skip binary and large files
            if file_path.suffix in [".pyc", ".pyo", ".so", ".dll", ".dylib"]:
                continue

            # Calculate checksum
            try:
                checksum = self._sha256_file(file_path)
                checksums[rel_path_str] = f"sha256:{checksum}"
            except OSError:
                # Skip files that can't be read
                continue

        return checksums

    def _sha256_file(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex-encoded SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _build_user_owned_manifest(
        self, project_dir: Path, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Build user_owned section for the manifest.

        For each user-owned artifact, records the SHA-256 of the framework
        template as rendered at claim time. During regen, if the framework
        hash changes, a drift warning is shown.

        Args:
            project_dir: Root directory of the project
            context: Template context with ``user_owned`` key

        Returns:
            Dict mapping canonical names to user_owned metadata, or empty dict.
        """
        user_owned = context.get("user_owned", [])
        if not user_owned:
            return {}

        import tempfile

        registry = PromptRegistry.default()
        result: dict[str, Any] = {}
        claude_code_dir = self.template_root / "claude_code"

        for canonical_name in user_owned:
            artifact = registry.get(canonical_name)
            if artifact is None:
                continue

            framework_hash = None
            template_file = claude_code_dir / artifact.template_path
            if template_file.exists():
                try:
                    if template_file.suffix == ".j2":
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=template_file.stem, delete=False, encoding="utf-8"
                        ) as tmp:
                            template_rel = f"claude_code/{artifact.template_path}"
                            template = self.jinja_env.get_template(template_rel)
                            rendered = template.render(**context)
                            tmp.write(rendered)
                            tmp_path = Path(tmp.name)
                        framework_hash = f"sha256:{self._sha256_file(tmp_path)}"
                        tmp_path.unlink(missing_ok=True)
                    else:
                        framework_hash = f"sha256:{self._sha256_file(template_file)}"
                except Exception:
                    pass  # Best-effort

            entry: dict[str, Any] = {
                "claimed_at": datetime.now(UTC).isoformat(),
            }
            if framework_hash:
                entry["framework_hash"] = framework_hash

            result[canonical_name] = entry

        return result
