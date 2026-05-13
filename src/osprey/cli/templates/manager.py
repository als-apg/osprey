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
        data_bundle: str = "control_assistant",
        context: dict[str, Any] | None = None,
        force: bool = False,
        artifacts: dict[str, list[str]] | None = None,
        tier: int = 1,
    ) -> Path:
        """Create complete project from template.

        This is the main entry point for project creation. It:
        1. Validates data bundle exists
        2. Creates project directory structure
        3. Renders and copies project files
        4. Copies service configurations
        5. Creates application code from template

        Args:
            project_name: Name of the project (e.g., "my-assistant")
            output_dir: Parent directory where project will be created
            data_bundle: Data bundle (app template) to use (default: "control_assistant")
            context: Additional template context variables
            force: If True, skip existence check (used when caller already handled deletion)
            artifacts: Profile-driven artifact selection (hooks, rules, skills, agents, etc.)

        Returns:
            Path to created project directory

        Raises:
            ValueError: If data bundle doesn't exist or the project directory
                exists without ``force=True``.

        Note:
            ``default_provider`` is no longer defaulted here — callers must
            inject it via ``context``. ``osprey build`` enforces this at the
            CLI boundary (``click.UsageError``); internal callers that omit
            it produce an empty ``provider:`` in the rendered ``config.yml``,
            which the config loader rejects at project runtime. See
            plan-remove-implicit-synchronous-narwhal.
        """
        # 1. Validate data bundle exists
        bundle_dir = self.template_root / "apps" / data_bundle
        if not bundle_dir.is_dir():
            app_templates = self.list_app_templates()
            raise ValueError(
                f"Template '{data_bundle}' not found. "
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

        # Fall back to preset profile artifacts when the caller didn't pass any
        # (legacy code path). An explicit empty dict from `osprey build` means the
        # profile deliberately selects nothing, and must not be overridden.
        if artifacts is None:
            tmpl_manifest = manifest.load_template_manifest(self.template_root, data_bundle)
            if tmpl_manifest:
                artifacts = tmpl_manifest.get("artifacts", {})

        # Derive feature flags from artifact selections.
        selected_hooks = (artifacts or {}).get("hooks", [])
        selected_web_panels = (artifacts or {}).get("web_panels", [])

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
            "current_python_env": current_python,  # Default; overridden by caller context
            "template_name": data_bundle,  # Make bundle name available in config.yml
            "data_bundle": data_bundle,
            "selected_hooks": selected_hooks,
            "selected_web_panels": selected_web_panels,
            # Add detected environment variables
            "env": detected_env_vars,
            **(context or {}),
        }

        # Derive channel finder configuration:
        # - When called from build (artifacts provided): check if channel-finder agent is selected
        # - When artifacts is None (programmatic caller): check if bundle config template declares it
        _profile_agents = (artifacts or {}).get("agents", [])
        _bundle_has_channel_finder = (bundle_dir / "config.yml.j2").exists() and not artifacts
        if "channel-finder" in _profile_agents or _bundle_has_channel_finder:
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
            self.template_root, self.jinja_env, project_dir, data_bundle, ctx
        )

        # 5. Copy services: bundle-level services/ dir takes priority, then
        #    fall back to matching names from the top-level services/ dir
        bundle_services_dir = bundle_dir / "services"
        top_level_services_dir = self.template_root / "services"
        if bundle_services_dir.is_dir():
            service_names = [d.name for d in bundle_services_dir.iterdir() if d.is_dir()]
            if service_names:
                scaffolding.copy_services_selective(self.template_root, project_dir, service_names)
        elif top_level_services_dir.is_dir():
            # Copy top-level services whose names match subdirs declared in bundle config
            # (e.g., control_assistant's config.yml.j2 references postgresql)
            available = [d.name for d in top_level_services_dir.iterdir() if d.is_dir()]
            bundle_config = bundle_dir / "config.yml.j2"
            if bundle_config.exists():
                config_text = bundle_config.read_text(encoding="utf-8")
                to_copy = [name for name in available if name in config_text]
                if to_copy:
                    scaffolding.copy_services_selective(self.template_root, project_dir, to_copy)

        # 6. Copy data files from template (no src/ package)
        scaffolding.copy_template_data(
            self.template_root,
            project_dir,
            package_name,
            data_bundle,
            ctx,
            jinja_env=self.jinja_env,
        )

        # 6a. Copy machine_data/ if bundle provides it
        machine_data_src = bundle_dir / "machine_data"
        if machine_data_src.exists():
            machine_data_dst = project_dir / "machine_data"
            shutil.copytree(machine_data_src, machine_data_dst, dirs_exist_ok=True)
            console.print(
                f"  [success]✓[/success] Copied machine data to [path]{machine_data_dst}[/path]"
            )

        # 6b. Rebase demo logbook timestamps to current date
        scaffolding.rebase_logbook_timestamps(project_dir)

        # 6c. Flatten the preset's tier-routed channel DBs into the canonical
        # data/channel_databases/<paradigm>.json locations. Must run before the
        # Claude Code hierarchy probe below, which reads the flat path. No-op
        # for bundles without a tiers/ subtree (e.g. hello_world).
        scaffolding.materialize_tier_dbs(
            project_dir, tier, ctx.get("channel_finder_mode")
        )

        # 7. Create _agent_data directory structure
        scaffolding.create_agent_data_structure(self.template_root, project_dir, ctx)

        # 8. Create Claude Code integration files
        # Load rendered config.yml so conditional sections (confluence, etc.)
        # are available to Claude Code templates (mcp.json.j2, CLAUDE.md.j2).
        config_file = project_dir / "config.yml"
        cc_cfg = {}
        ctx.setdefault("facility_permissions", {})
        if config_file.exists():
            with open(config_file) as f:
                rendered_config = yaml.safe_load(f) or {}
            rendered_config = resolve_env_vars(rendered_config)  # Match regen path
            # Claude Code explicit overrides
            cc_config = rendered_config.get("claude_code", {})
            cc_cfg = cc_config
            ctx["facility_permissions"] = cc_config.get("permissions", {})
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
        from osprey.registry.mcp import resolve_agents, resolve_servers

        ctx["servers"] = resolve_servers(cc_cfg, ctx)
        ctx["agents"] = resolve_agents(cc_cfg, ctx, project_dir, ctx["servers"])
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}

        # Load template manifest and resolve allowed outputs
        manifest_data = manifest.load_template_manifest(self.template_root, data_bundle)
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

    def regenerate_claude_code(
        self,
        project_dir: Path,
        dry_run: bool = False,
        project_root_override: Path | str | None = None,
    ) -> dict:
        """Regenerate Claude Code artifacts from current config.yml.

        Args:
            project_dir: Root directory of the project
            dry_run: If True, report what would change without writing files
            project_root_override: If set, use this path as ``project_root``
                in the rendered context instead of ``project_dir``.

        Returns:
            Dict with 'changed', 'unchanged', and 'backup_dir' keys
        """
        return claude_code.regenerate_claude_code(
            self.template_root,
            self.jinja_env,
            project_dir,
            dry_run,
            project_root_override=project_root_override,
        )

    def generate_manifest(
        self,
        project_dir: Path,
        project_name: str,
        data_bundle: str | None = None,
        context: dict[str, Any] | None = None,
        artifacts: dict[str, list[str]] | None = None,
        preset_name: str | None = None,
        profile_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate a project manifest for migration support.

        Args:
            project_dir: Root directory of the created project.
            project_name: Name of the project.
            data_bundle: Underlying app bundle (default: "control_assistant").
            context: Full context dict used during template rendering.
            artifacts: Profile-driven artifact selection.
            preset_name: Hyphenated preset name (if --preset was used).
            profile_path: Path string to positional profile (if used).

        Returns:
            Dictionary containing the manifest data that was written to file.
        """
        if data_bundle is None:
            data_bundle = "control_assistant"
        if context is None:
            context = {}
        return manifest.generate_manifest(
            self.template_root,
            self.jinja_env,
            project_dir,
            project_name,
            data_bundle,
            context,
            artifacts=artifacts,
            preset_name=preset_name,
            profile_path=profile_path,
        )

    def copy_services(self, project_dir: Path):
        """Copy service configurations to project (flattened structure).

        Args:
            project_dir: Root directory of the project
        """
        scaffolding.copy_services(self.template_root, project_dir)
