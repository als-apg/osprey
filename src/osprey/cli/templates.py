"""Template management for project scaffolding.

This module provides the TemplateManager class which handles:
- Discovery of bundled templates in the osprey package
- Rendering Jinja2 templates with project-specific context
- Creating complete project structures from templates
- Copying service configurations to user projects
"""

import os
import re
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from osprey.cli.styles import console


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
            autoescape=select_autoescape(['html', 'xml']),
            keep_trailing_newline=True
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
            pass

        # Fallback for development: relative to this file
        fallback_path = Path(__file__).parent.parent / "templates"
        if fallback_path.exists():
            return fallback_path

        raise RuntimeError(
            "Could not locate osprey templates directory. "
            "Ensure osprey is properly installed."
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
            'CBORG_API_KEY',
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'GOOGLE_API_KEY',
            'PROJECT_ROOT',
            'LOCAL_PYTHON_VENV',
            'TZ',
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

        return sorted([
            d.name for d in apps_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ])

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
        output_path.write_text(rendered)

    def create_project(
        self,
        project_name: str,
        output_dir: Path,
        template_name: str = "minimal",
        registry_style: str = "extend",
        context: dict[str, Any] | None = None,
        force: bool = False
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
            **(context or {})
        }

        # Derive channel finder configuration if control_assistant template
        if template_name == 'control_assistant':
            channel_finder_mode = ctx.get('channel_finder_mode', 'both')

            # Derive boolean flags for conditional templates
            enable_in_context = channel_finder_mode in ['in_context', 'both']
            enable_hierarchical = channel_finder_mode in ['hierarchical', 'both']

            # Determine default pipeline (for config.yml)
            if channel_finder_mode == 'both':
                default_pipeline = 'hierarchical'  # Default to more scalable option
            else:
                default_pipeline = channel_finder_mode

            # Add channel finder context variables
            ctx.update({
                'channel_finder_mode': channel_finder_mode,
                'enable_in_context': enable_in_context,
                'enable_hierarchical': enable_hierarchical,
                'default_pipeline': default_pipeline,
                'in_context_db_path': 'data/channel_databases/in_context.json',
                'hierarchical_db_path': 'data/channel_databases/hierarchical.json',
            })

        # 4. Create project structure
        self._create_project_structure(project_dir, template_name, ctx)

        # 5. Copy services (skip for hello_world_weather as it doesn't use containers)
        if template_name != 'hello_world_weather':
            self.copy_services(project_dir)

        # 6. Create src directory and application code
        src_dir = project_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        self._create_application_code(src_dir, package_name, template_name, ctx, registry_style, project_dir)

        # 7. Create _agent_data directory structure
        self._create_agent_data_structure(project_dir, ctx)

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

        # Render template files
        files_to_render = [
            ("config.yml.j2", "config.yml"),
            ("env.example.j2", ".env.example"),
            ("README.md.j2", "README.md"),
            ("pyproject.toml.j2", "pyproject.toml"),
            ("requirements.txt", "requirements.txt"),  # Render to replace framework_version
        ]

        # Copy static files
        static_files = [
            # requirements.txt moved to rendered templates to handle {{ framework_version }}
        ]

        for template_file, output_file in files_to_render:
            # Check if app template has its own version first (e.g., requirements.txt.j2)
            app_specific_template = app_template_dir / (template_file + ".j2" if not template_file.endswith(".j2") else template_file)
            default_template = project_template_dir / template_file

            if app_specific_template.exists():
                # Use app-specific template
                self.render_template(
                    f"apps/{template_name}/{app_specific_template.name}",
                    ctx,
                    project_dir / output_file
                )
            elif default_template.exists():
                # Use default project template
                self.render_template(
                    f"project/{template_file}",
                    ctx,
                    project_dir / output_file
                )

        # Create .env file only if API keys are detected
        detected_env_vars = ctx.get('env', {})
        api_keys = ['CBORG_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
        has_api_keys = any(key in detected_env_vars for key in api_keys)

        if has_api_keys:
            env_template = project_template_dir / "env.j2"
            if env_template.exists():
                self.render_template(
                    "project/env.j2",
                    ctx,
                    project_dir / ".env"
                )
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
            elif item.is_file() and item.suffix in ['.j2', '.yml', '.yaml']:
                # Copy docker-compose template/config files
                shutil.copy(item, dst_services / item.name)

    def _create_application_code(
        self,
        src_dir: Path,
        package_name: str,
        template_name: str,
        ctx: dict,
        registry_style: str = "extend",
        project_root: Path = None
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
        ctx['registry_style'] = registry_style

        # Project-level files that should only live at project root, not in src/
        # These are handled by _create_project_structure() and should be skipped here
        PROJECT_LEVEL_FILES = {
            'config.yml.j2', 'config.yml',
            'README.md.j2', 'README.md',
            'env.example.j2', 'env.example', 'env.j2', '.env',
            'requirements.txt.j2', 'requirements.txt',
            'pyproject.toml.j2', 'pyproject.toml'
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
                    self.render_template(
                        f"apps/{template_name}/{rel_path}",
                        ctx,
                        output_path
                    )
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
        words = package_name.split('_')
        class_name = ''.join(word.capitalize() for word in words)
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
        capability_pattern = r'CapabilityRegistration\((.*?)\)'
        for match in re.finditer(capability_pattern, template_content, re.DOTALL):
            reg_content = match.group(1)

            # Extract parameters (simple approach - could be more robust)
            name_match = re.search(r'name\s*=\s*"([^"]+)"', reg_content)
            module_path_match = re.search(r'module_path\s*=\s*"([^"]+)"', reg_content)
            class_name_match = re.search(r'class_name\s*=\s*"([^"]+)"', reg_content)
            description_match = re.search(r'description\s*=\s*"([^"]+)"', reg_content)
            provides_match = re.search(r'provides\s*=\s*\[([^\]]+)\]', reg_content)
            requires_match = re.search(r'requires\s*=\s*\[([^\]]*)\]', reg_content)

            if name_match and module_path_match and class_name_match:
                # Process provides list
                provides = []
                if provides_match:
                    provides_str = provides_match.group(1)
                    provides = [item.strip().strip('"\'') for item in provides_str.split(',')]

                # Process requires list
                requires = []
                if requires_match and requires_match.group(1).strip():
                    requires_str = requires_match.group(1)
                    requires = [item.strip().strip('"\'') for item in requires_str.split(',')]

                # Substitute template variables
                module_path = module_path_match.group(1).replace('{{ package_name }}', ctx['package_name'])
                description = description_match.group(1) if description_match else ""

                capabilities.append(CapabilityRegistration(
                    name=name_match.group(1),
                    module_path=module_path,
                    class_name=class_name_match.group(1),
                    description=description,
                    provides=provides,
                    requires=requires
                ))

        # Parse ContextClassRegistration entries
        context_pattern = r'ContextClassRegistration\((.*?)\)'
        for match in re.finditer(context_pattern, template_content, re.DOTALL):
            reg_content = match.group(1)

            context_type_match = re.search(r'context_type\s*=\s*"([^"]+)"', reg_content)
            module_path_match = re.search(r'module_path\s*=\s*"([^"]+)"', reg_content)
            class_name_match = re.search(r'class_name\s*=\s*"([^"]+)"', reg_content)

            if context_type_match and module_path_match and class_name_match:
                # Substitute template variables
                module_path = module_path_match.group(1).replace('{{ package_name }}', ctx['package_name'])

                context_classes.append(ContextClassRegistration(
                    context_type=context_type_match.group(1),
                    module_path=module_path,
                    class_name=class_name_match.group(1)
                ))

        # Generate the explicit registry code
        registry_code = generate_explicit_registry_code(
            app_class_name=ctx['app_class_name'],
            app_display_name=ctx['app_display_name'],
            package_name=ctx['package_name'],
            capabilities=capabilities if capabilities else None,
            context_classes=context_classes if context_classes else None
        )

        # Write to output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(registry_code)

    def _create_agent_data_structure(self, project_dir: Path, ctx: dict):
        """Create _agent_data directory structure for the project.

        This method creates the agent data directory and all standard subdirectories
        based on osprey's default configuration. This ensures that container
        deployments won't fail due to missing mount points.

        Args:
            project_dir: Root directory of the project
            ctx: Template context variables (unused but kept for consistency)
        """
        # Create main _agent_data directory
        agent_data_dir = project_dir / "_agent_data"
        agent_data_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories based on default framework configuration
        subdirs = [
            "executed_scripts",
            "execution_plans",
            "user_memory",
            "registry_exports",
            "prompts",
            "checkpoints"
        ]

        for subdir in subdirs:
            subdir_path = agent_data_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)

        console.print(f"  [success]✓[/success] Created agent data structure at [path]{agent_data_dir}[/path]")

        # Create a README to explain the directory structure
        readme_content = """# Agent Data Directory

This directory contains runtime data generated by the Osprey Framework:

- `executed_scripts/`: Python scripts executed by the framework
- `execution_plans/`: Orchestrator execution plans (JSON format)
- `user_memory/`: User memory data and conversation history
- `registry_exports/`: Exported registry information
- `prompts/`: Generated prompts (when debug mode enabled)
- `checkpoints/`: LangGraph checkpoints for conversation state

This directory is excluded from git (see .gitignore) but is required for
proper framework operation, especially when using containerized services.
"""

        readme_path = agent_data_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

