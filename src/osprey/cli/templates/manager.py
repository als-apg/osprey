"""TemplateManager facade: thin orchestrator delegating to submodules."""

import shutil
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from osprey.cli.styles import console
from osprey.cli.templates import claude_code, manifest, scaffolding
from osprey.cli.templates._rendering import render_template as _render_template
from osprey.utils.config import resolve_env_vars


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
        fallback_path = Path(__file__).parent.parent.parent / "templates"
        if fallback_path.exists():
            return fallback_path

        raise RuntimeError(
            "Could not locate osprey templates directory. Ensure osprey is properly installed."
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
        _render_template(self.jinja_env, template_path, context, output_path)

    def list_app_templates(self) -> list[str]:
        """List available application templates.

        Returns:
            List of template names (directory names in templates/apps/)
        """
        apps_dir = self.template_root / "apps"
        if not apps_dir.exists():
            return []

        return sorted(
            [d.name for d in apps_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        )

    def _generate_class_name(self, package_name: str) -> str:
        """Generate a PascalCase class name prefix from package name.

        Args:
            package_name: Python package name (e.g., "my_assistant")

        Returns:
            PascalCase class name prefix (e.g., "MyAssistant")
            Note: The template adds "RegistryProvider" suffix
        """
        # Convert snake_case to PascalCase
        words = package_name.split("_")
        class_name = "".join(word.capitalize() for word in words)
        return class_name

    def create_project(
        self,
        project_name: str,
        output_dir: Path,
        template_name: str = "control_assistant",
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
            template_name: Application template to use (default: "control_assistant")
            registry_style: Registry style - "extend" (recommended) or "standalone" (advanced)
            context: Additional template context variables
            force: If True, skip existence check (used when caller already handled deletion)

        Returns:
            Path to created project directory

        Raises:
            ValueError: If template doesn't exist or project directory exists
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
        detected_env_vars = scaffolding.detect_environment_variables()

        ctx = {
            "project_name": project_name,
            "package_name": package_name,
            "app_display_name": project_name,  # Used in templates for display/documentation
            "app_class_name": class_name,  # Used in templates for class names
            "registry_class_name": class_name,  # Backward compatibility
            "project_description": f"{project_name} - Osprey Agent Application",
            "framework_version": manifest.get_framework_version(),
            "project_root": str(project_dir.absolute()),
            "venv_path": "${LOCAL_PYTHON_VENV}",
            "current_python_env": current_python,  # Actual path to current Python
            "default_provider": "cborg",
            "default_model": "haiku",
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

        # 4. Create project structure
        scaffolding.create_project_structure(
            self.template_root, self.jinja_env, project_dir, template_name, ctx
        )

        # 5. Copy services (selective -- only postgresql for control_assistant)
        if template_name == "control_assistant":
            scaffolding.copy_services_selective(self.template_root, project_dir, ["postgresql"])

        # 6. Copy data files from template (no src/ package)
        scaffolding.copy_template_data(
            self.template_root, project_dir, package_name, template_name, ctx
        )

        # 6a. Copy machine_data/ for lattice templates
        if template_name == "lattice_design":
            machine_data_src = self.template_root / "apps" / "lattice_design" / "machine_data"
            if machine_data_src.exists():
                machine_data_dst = project_dir / "machine_data"
                shutil.copytree(machine_data_src, machine_data_dst, dirs_exist_ok=True)
                console.print(
                    f"  [success]✓[/success] Copied machine data to [path]{machine_data_dst}[/path]"
                )

        # 6b. Rebase demo logbook timestamps to current date
        scaffolding.rebase_logbook_timestamps(project_dir)

        # 7. Create _agent_data directory structure
        scaffolding.create_agent_data_structure(self.template_root, project_dir, ctx)

        # 8. Create Claude Code integration files
        # Load rendered config.yml so conditional sections (confluence, etc.)
        # are available to Claude Code templates (mcp.json.j2, CLAUDE.md.j2).
        config_file = project_dir / "config.yml"
        cc_cfg = {}
        if config_file.exists():
            with open(config_file) as f:
                rendered_config = yaml.safe_load(f) or {}
            rendered_config = resolve_env_vars(rendered_config)  # Match regen path
            if "confluence" in rendered_config:
                ctx["confluence"] = rendered_config["confluence"]
            if "matlab" in rendered_config:
                ctx["matlab"] = rendered_config["matlab"]
            if "deplot" in rendered_config:
                ctx["deplot"] = rendered_config["deplot"]
            # Claude Code explicit overrides
            cc_config = rendered_config.get("claude_code", {})
            cc_cfg = cc_config
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

            # System timezone for ARIEL tools
            system_config = rendered_config.get("system", {})
            ctx["system_timezone"] = system_config.get("timezone", "UTC")

            # Facility name fallback (already set for control_assistant at line 284,
            # but setdefault handles other templates)
            ctx.setdefault("facility_name", rendered_config.get("facility_name", project_name))

            # Override channel_finder_mode with the actual active pipeline from
            # rendered config. During phase 1, channel_finder_mode may be "all"
            # (meaning "render all pipeline configs"). But Claude Code agent templates
            # need the actual active pipeline mode, which is deterministic from
            # config.yml pipeline_mode.
            cf_config = rendered_config.get("channel_finder", {})
            if cf_config.get("pipeline_mode"):
                ctx["channel_finder_mode"] = cf_config["pipeline_mode"]

            # Embed hierarchy info for initial creation (mirrors _build_claude_code_context)
            if cf_config.get("pipeline_mode") == "hierarchical":
                try:
                    db_path = (
                        cf_config.get("pipelines", {})
                        .get("hierarchical", {})
                        .get("database", {})
                        .get("path", "")
                    )
                    if db_path:
                        from osprey.services.channel_finder.databases.hierarchical import (
                            HierarchicalChannelDatabase,
                        )

                        resolved = (project_dir / db_path).resolve()
                        db = HierarchicalChannelDatabase(str(resolved))
                        ctx["channel_finder_hierarchy"] = {
                            "hierarchy_levels": db.hierarchy_levels,
                            "hierarchy_config": db.hierarchy_config,
                            "naming_pattern": db.naming_pattern,
                        }
                except Exception:
                    import logging

                    logging.getLogger("osprey.cli.templates").warning(
                        "Could not load hierarchy info during project creation",
                        exc_info=True,
                    )
            ctx.setdefault("channel_finder_hierarchy", None)

            # Direct channel finder (separate from pipeline-based channel finder)
            direct_cf = cf_config.get("direct") if cf_config else None
            if direct_cf:
                ctx["direct_channel_finder"] = True
                try:
                    from osprey.services.channel_finder.utils.naming_summary import (
                        generate_naming_summary,
                    )

                    ctx["naming_patterns_summary"] = generate_naming_summary(rendered_config)
                except Exception:
                    import logging

                    logging.getLogger("osprey.cli.templates").warning(
                        "Could not generate naming summary during project creation",
                        exc_info=True,
                    )

        # Ensure defaults exist even without rendered config
        ctx.setdefault("disable_servers", [])
        ctx.setdefault("disable_agents", [])
        ctx.setdefault("extra_servers", {})

        # Textbooks root -- resolve relative to project directory
        _textbooks_dir = project_dir.parent / "data" / "textbooks"
        ctx["textbooks_root"] = str(_textbooks_dir) if _textbooks_dir.is_dir() else None
        # Tilde variant for permission matching (models abbreviate /Users/x to ~)
        import os as _os

        _home = _os.path.expanduser("~")
        if ctx["textbooks_root"] and ctx["textbooks_root"].startswith(_home):
            ctx["textbooks_root_tilde"] = "~" + ctx["textbooks_root"][len(_home) :]
        else:
            ctx["textbooks_root_tilde"] = None

        # Resolve servers and agents via the data-driven registry.
        # Merge init-time overrides (passed via context=) into cc_cfg
        # so the resolver sees them -- e.g. context={"disable_agents": [...]}.
        from osprey.cli.server_registry import resolve_agents, resolve_servers

        if ctx.get("disable_servers") and "disable_servers" not in cc_cfg:
            cc_cfg = {**cc_cfg, "disable_servers": ctx["disable_servers"]}
        if ctx.get("disable_agents") and "disable_agents" not in cc_cfg:
            cc_cfg = {**cc_cfg, "disable_agents": ctx["disable_agents"]}

        ctx["servers"] = resolve_servers(cc_cfg, ctx)
        ctx["agents"] = resolve_agents(cc_cfg, ctx, project_dir, ctx["servers"])
        # Update legacy keys from resolved data
        ctx["disable_servers"] = [s["name"] for s in ctx["servers"] if not s["enabled"]]
        ctx["disable_agents"] = [a["name"] for a in ctx["agents"] if not a["enabled"]]

        # Load template manifest and resolve allowed outputs
        manifest_data = manifest.load_template_manifest(self.template_root, template_name)
        allowed_outputs = (
            manifest.resolve_manifest_outputs(manifest_data) if manifest_data else None
        )

        # Filter agents to manifest (only generate agents the template declares)
        if allowed_outputs is not None:
            ctx["agents"] = [
                a for a in ctx["agents"] if f".claude/agents/{a['name']}.md" in allowed_outputs
            ]

        claude_code.create_claude_code_integration(
            self.template_root, self.jinja_env, project_dir, ctx, allowed_outputs
        )

        return project_dir

    def regenerate_claude_code(self, project_dir: Path, dry_run: bool = False) -> dict:
        """Regenerate Claude Code artifacts from current config.yml.

        Args:
            project_dir: Root directory of the project
            dry_run: If True, report what would change without writing files

        Returns:
            Dict with 'changed', 'unchanged', and 'backup_dir' keys
        """
        return claude_code.regenerate_claude_code(
            self.template_root, self.jinja_env, project_dir, dry_run
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

        Args:
            project_dir: Root directory of the created project
            project_name: Name of the project
            template_name: Template used to create the project
            registry_style: Registry style ("extend" or "standalone")
            context: Full context dict used during template rendering

        Returns:
            Dictionary containing the manifest data that was written to file
        """
        return manifest.generate_manifest(
            self.template_root,
            self.jinja_env,
            project_dir,
            project_name,
            template_name,
            registry_style,
            context,
        )

    def copy_services(self, project_dir: Path):
        """Copy service configurations to project (flattened structure).

        Args:
            project_dir: Root directory of the project
        """
        scaffolding.copy_services(self.template_root, project_dir)
