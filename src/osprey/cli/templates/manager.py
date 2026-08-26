"""TemplateManager facade: thin orchestrator delegating to submodules."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateRuntimeError, select_autoescape

from osprey.build.build_tiers import (
    VALID_CHANNEL_FINDER_MODES,
    default_tier_for_mode,
    tier_mode_conflict,
)
from osprey.cli.templates import claude_code, manifest, scaffolding
from osprey.cli.templates._rendering import render_template as _render_template
from osprey.errors import BuildProfileError
from osprey.profiles.web_panels import BUILTIN_PANELS
from osprey.utils.config import resolve_env_vars
from osprey.utils.facility import resolve_facility_name
from osprey.utils.workspace import repo_root_for_config

logger = logging.getLogger("osprey.cli.templates")


def _enable_flags(channel_finder_mode: str) -> dict[str, bool]:
    """One ``enable_<paradigm>`` template flag per registered paradigm.

    Derived by iterating
    :data:`osprey.build.build_tiers.VALID_CHANNEL_FINDER_MODES`, so a paradigm
    added to the registry gets its render flag without an edit here. An
    unregistered mode leaves every flag off; callers reject it before this
    point.

    ``enable_graph`` is derived like the rest and read by nothing. The graph
    paradigm has no config block for a template to gate: ``pipeline_mode:
    graph`` plus the ``services.graphdb`` declaration is its whole
    configuration. Keeping the flag derived rather than special-casing it out
    holds this set to the registry, and
    ``tests/templates/test_control_assistant_config.py`` pins both halves —
    the flag is emitted, and no template under ``src/osprey/templates``
    mentions it.
    """
    return {f"enable_{mode}": channel_finder_mode == mode for mode in VALID_CHANNEL_FINDER_MODES}


def _fail(message: str) -> None:
    """Abort the render from inside a template.

    Registered as the ``fail`` global on the manager's Jinja environment so a
    template can refuse a case it has no branch for — ``{% else %}{{ fail(...)
    }}`` at the end of a chain — instead of quietly emitting nothing. Jinja
    otherwise has no way for a template to stop a render.
    """
    raise TemplateRuntimeError(message)


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
        self.jinja_env.globals["fail"] = _fail

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
        tier: int | None = None,
        data_root: Path | None = None,
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
            tier: Channel-database tier (1|3) to materialize. When ``None`` (the
                default), the paradigm-aware rule derives it from
                ``channel_finder_mode`` (in_context → 1, else → 3), matching
                ``BuildProfile.resolved_tier``. An explicit tier is honored but
                validated against the paradigm, so a tier/mode mismatch raises a
                legible rule error instead of an opaque FileNotFoundError.
            data_root: Facility data tree to copy instead of the bundle's
                ``apps/<data_bundle>/data/``, already resolved by
                ``BuildProfile.resolved_data_root``. A full replacement copied
                verbatim (no Jinja rendering); ``None`` keeps the bundle tree.

        Returns:
            Path to created project directory

        Raises:
            ValueError: If data bundle doesn't exist or the project directory
                exists without ``force=True``.

        Note:
            ``default_provider`` is not defaulted here — callers must inject
            it via ``context``. ``osprey build`` enforces this at the CLI
            boundary (``click.UsageError``); internal callers that omit it
            produce an empty ``provider:`` in the rendered ``config.yml``,
            which the config loader rejects at project runtime.
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

        # Fall back to preset profile artifacts when the caller didn't pass any.
        # An explicit empty dict from `osprey build` means the
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
            # Enable-able builtin panel registry (single source of truth in
            # osprey.profiles.web_panels). Templates derive their enable list
            # from this so it can't drift from the real registry. sorted() for
            # deterministic rendered output.
            "builtin_panels": sorted(BUILTIN_PANELS),
            # No `env` key: the render reads nothing from `os.environ`, and
            # writes no `.env` at all — the deployment's one secret store is the
            # repo-root `.env`, outside this tree.
            # Provider API-key env vars, derived from the provider registry
            # (single source of truth in osprey.models.provider_registry) so
            # env.example.j2 can't drift from the real provider list.
            # Ordered list of {"provider", "var"} dicts; key-less providers
            # (ollama, vllm, …) are excluded.
            "provider_api_keys": scaffolding.provider_api_key_entries(),
            # The subset of the above this profile actually uses. Empty here so
            # a caller with no profile still renders the whole list uncommented;
            # `osprey init` fills it in and the rest drop below a divider.
            "active_provider_vars": [],
            # Deploy-minted service credentials, derived from the map the
            # deploy path mints from, so .env.example documents every one of
            # them. Ordered list of {"var", "services", "note"} dicts.
            "service_token_vars": scaffolding.service_token_var_entries(),
            # The build profile's `env:` block, documented in .env.example.
            # Defaulted here so a caller that has no profile (programmatic
            # create_project) still renders the file.
            "env_required": [],
            "env_defaults": {},
            **(context or {}),
        }

        # Derive channel finder configuration when the channel-finder agent
        # is selected (either explicitly via build profile artifacts, or via
        # the preset-profile fallback above for programmatic callers).
        _profile_agents = (artifacts or {}).get("agents", [])
        if "channel-finder" in _profile_agents:
            channel_finder_mode = ctx.get("channel_finder_mode")
            if channel_finder_mode is None:
                raise BuildProfileError(
                    "channel_finder_mode is required when the channel-finder agent "
                    "is selected. Pin it in your profile "
                    "(e.g. `channel_finder_mode: hierarchical`) or pass "
                    "`--set channel_finder_mode=<paradigm>` to `osprey build`."
                )
            if channel_finder_mode not in VALID_CHANNEL_FINDER_MODES:
                raise BuildProfileError(
                    f"channel_finder_mode must be one of {VALID_CHANNEL_FINDER_MODES} "
                    f"(got {channel_finder_mode!r})"
                )
            from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE

            ctx.update(
                {
                    "channel_finder_mode": channel_finder_mode,
                    **_enable_flags(channel_finder_mode),
                    "default_pipeline": channel_finder_mode,
                    "channel_finder_pipeline": channel_finder_mode,
                    "channel_finder_tools": list(
                        CHANNEL_FINDER_TOOLS_BY_PIPELINE.get(channel_finder_mode, [])
                    ),
                }
            )

        # 4. Create project structure
        scaffolding.create_project_structure(
            self.template_root,
            self.jinja_env,
            project_dir,
            data_bundle,
            ctx,
        )

        # 5. Copy services: bundle-level services/ dir takes priority, then
        #    fall back to matching names from the top-level services/ dir.
        #    Skipped for an attached project (deploy_services: false), which
        #    scaffolds no services/ tree of its own — it connects to a stack
        #    another OSPREY project deployed on the same host.
        if ctx.get("deploy_services", True):
            bundle_services_dir = bundle_dir / "services"
            top_level_services_dir = self.template_root / "services"
            if bundle_services_dir.is_dir():
                service_names = [d.name for d in bundle_services_dir.iterdir() if d.is_dir()]
                if service_names:
                    scaffolding.copy_services_selective(
                        self.template_root, project_dir, service_names
                    )
            elif top_level_services_dir.is_dir():
                # Copy top-level services whose names match subdirs declared in bundle config
                # (e.g., control_assistant's config.yml.j2 references postgresql)
                available = [d.name for d in top_level_services_dir.iterdir() if d.is_dir()]
                bundle_config = bundle_dir / "config.yml.j2"
                if bundle_config.exists():
                    config_text = bundle_config.read_text(encoding="utf-8")
                    to_copy = [name for name in available if name in config_text]
                    if to_copy:
                        scaffolding.copy_services_selective(
                            self.template_root, project_dir, to_copy
                        )

        # 6. Copy data files from template (no src/ package), or from the
        # profile's own data tree when one was resolved. Either way this lands
        # before step 6b's tier materialization and the hierarchy probe in
        # step 7, both of which read the project's flat data/ paths.
        scaffolding.copy_template_data(
            self.template_root,
            project_dir,
            package_name,
            data_bundle,
            ctx,
            jinja_env=self.jinja_env,
            data_root=data_root,
        )

        # 6a. Copy machine_data/ if bundle provides it
        machine_data_src = bundle_dir / "machine_data"
        if machine_data_src.exists():
            machine_data_dst = project_dir / "machine_data"
            shutil.copytree(machine_data_src, machine_data_dst, dirs_exist_ok=True)
            logger.debug("Copied machine data to %s", machine_data_dst)

        # 6a'. Install the web-terminal context baseline. This base.md is the
        # framework FALLBACK: seeding hard-requires one for any project that
        # seeds a user, so every bundle gets a generic copy — and a profile
        # that ships its own `web-terminal-context/base.md` overrides it when
        # the convention copies apply after this render. Per-user
        # extra.md/skills stay user-authored under the same tree.
        context_src = self.template_root / "claude_code" / "web-terminal-context"
        context_dst = project_dir / "docker" / "web-terminal-context"
        shutil.copytree(context_src, context_dst, dirs_exist_ok=True)
        logger.debug("Installed web-terminal context to %s", context_dst)

        # 6b. Flatten the preset's tier-routed channel DBs into the canonical
        # data/channel_databases/<paradigm>.json locations. Must run before the
        # Claude Code hierarchy probe below, which reads the flat path. Only
        # relevant when channel-finder is selected — builds that skip the
        # channel-finder agent have no use for the materialized DB. No-op for
        # bundles without a tiers/ subtree (e.g. hello_world).
        channel_finder_mode = ctx.get("channel_finder_mode")
        if channel_finder_mode is not None:
            # Resolve the build-time tier with the same paradigm-aware rule the
            # build-profile validator applies, so programmatic callers that omit
            # `tier` (or pin a mismatched one) can't reach the materializer with a
            # tier/paradigm mismatch — that would surface as an opaque
            # FileNotFoundError instead of a legible rule error.
            if tier is None:
                effective_tier = default_tier_for_mode(channel_finder_mode)
            else:
                # Mirror BuildProfile.validate() at this boundary: range-check
                # first (so tier=2 gets the legible {1,3} error, not a later
                # FileNotFoundError), then the tier/paradigm conflict rule.
                if tier not in (1, 3):
                    raise BuildProfileError(f"tier must be 1 or 3 (got {tier!r})")
                conflict = tier_mode_conflict(tier, channel_finder_mode)
                if conflict:
                    raise BuildProfileError(conflict)
                effective_tier = tier
            scaffolding.materialize_tier_artifacts(project_dir, effective_tier, channel_finder_mode)
            scaffolding.prune_csv_build_artifacts(project_dir, channel_finder_mode)

        # 7. Create Claude Code integration files
        # Load rendered config.yml so conditional sections (confluence, etc.)
        # are available to Claude Code templates (mcp.json.j2, CLAUDE.md.j2).
        config_file = project_dir / "config.yml"
        cc_cfg = {}
        rendered_config: dict = {}
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
            from osprey.build.claude_code_resolver import ClaudeCodeModelResolver

            api_providers = rendered_config.get("api", {}).get("providers", {})
            try:
                model_spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
            except ValueError:
                model_spec = None
            ctx["claude_code_model_spec"] = model_spec

            # System timezone for ARIEL tools
            system_config = rendered_config.get("system", {})
            ctx["system_timezone"] = system_config.get("timezone", "UTC")

            # Facility identity: canonical `facility.name`, legacy top-level
            # `facility_name` as fallback (see utils.facility.resolve_facility_name).
            # setdefault so an explicit caller-supplied context value still wins.
            ctx.setdefault("facility_name", resolve_facility_name(rendered_config, project_name))

            cf_config = rendered_config.get("channel_finder", {})

            # Embed hierarchy info for initial creation, through the same
            # resolution every later re-render uses.
            if cf_config.get("pipeline_mode") == "hierarchical":
                hierarchy = claude_code.resolve_hierarchy_context(cf_config, project_dir)
                if hierarchy is not None:
                    ctx["channel_finder_hierarchy"] = hierarchy
            ctx.setdefault("channel_finder_hierarchy", None)

        # A bundle that renders no config.yml still needs a facility name for the
        # agent/CLAUDE.md prompts rendered below.
        ctx.setdefault("facility_name", project_name)

        # Everything the Claude Code templates read out of config.yml, through
        # the same helper the build's own render path uses. Without it this path
        # left `control_system_write_tools` (and the declared-hook wiring)
        # undefined, and the non-strict Jinja environment rendered that as
        # nothing: a hook_config.json whose write-kill-switch list was empty,
        # with no error to say so. setdefault, so an explicit caller-supplied
        # value still wins.
        for key, value in claude_code.config_derived_context(rendered_config, project_dir).items():
            ctx.setdefault(key, value)

        # ...except the deny floor, which is NOT the caller's to soften. Every
        # other key above is project configuration a caller may legitimately
        # pre-empt; this one is the security floor settings.json renders into
        # permissions.deny (Bash and Edit among them). While it was a literal
        # inside settings.json.j2 no caller could reach it at all, and hoisting
        # it into the context must not quietly hand over that authority: under
        # setdefault a caller passing deny_defaults=[] would render a project
        # that denies nothing. Facilities widen or narrow the floor through
        # config.yml's claude_code.permissions (deny / remove_deny), which the
        # template applies on top of this list -- that is the supported route,
        # and it is auditable in the profile. Assigned, not setdefault, so the
        # framework wins; the build's own path (build_claude_code_context) gets
        # the same precedence from its ctx.update.
        ctx["deny_defaults"] = list(claude_code.DENY_DEFAULTS)

        claude_code.apply_textbooks_root(ctx, project_dir)

        # Resolve servers and agents via the data-driven registry.
        from osprey.registry.mcp import mixed_read_write_tools, resolve_agents, resolve_servers

        ctx["servers"] = resolve_servers(cc_cfg, ctx)
        ctx["agents"] = resolve_agents(cc_cfg, ctx, project_dir, ctx["servers"])
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
        # The render's read/write-MIXED tools, exactly as build_claude_code_context
        # computes them for the regen path. Both paths render hook_config.json, and
        # only the regen path runs that function: without this line a freshly BUILT
        # project shipped an empty mixed list beside a populated write-tool list,
        # while a rebuild of the same config shipped the real one. The consequence
        # is fail-closed (the middleware clamps python execute as if pure-write),
        # which is why nothing surfaced it — the Jinja environment is not strict, so
        # the difference is one safety file quietly saying "nothing is exempt".
        ctx["mixed_read_write_tools"] = mixed_read_write_tools(ctx["servers"])

        # Resolve allowed outputs from THIS render's effective artifact selection —
        # `artifacts` above, already the caller's own selection where it supplied
        # one and the data bundle's otherwise. Re-loading the bundle manifest here
        # would discard the caller's: a persona render inherits its host's data
        # bundle, so the bundle manifest is the HOST's selection, and a persona
        # that drops an artifact by name would still have it allowed. Skills are
        # where that shows, being the one family copied on the strength of this set
        # alone (hooks and rules gate inside their own templates, agents are
        # filtered just below), so the leak was silent everywhere else.
        #
        # `is not None`, not truthiness: an empty selection is the deliberate
        # "this render selects nothing" of the fallback above, and must resolve to
        # the four config artifacts rather than fall back to a wider list.
        allowed_outputs = (
            manifest.resolve_manifest_outputs({"artifacts": artifacts})
            if artifacts is not None
            else None
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
        runtime_venv_dir: Path | str | None = None,
        runtime_interpreter: str | None = None,
    ) -> dict:
        """Regenerate Claude Code artifacts from current config.yml.

        Args:
            project_dir: Directory holding the ``config.yml`` and ``.claude/``
                being regenerated — the *render* (``<repo>/build``), not the
                repo root.
            dry_run: If True, report what would change without writing files
            project_root_override: The repo root the regenerated artifacts must
                name. Defaults to the repo *of the config being read*, resolved
                through :func:`osprey.utils.workspace.repo_root_for_config` —
                see the note below. Pass it explicitly only to render for a
                root other than the one this render sits in.
            runtime_venv_dir: Directory holding the ``.venv`` the regenerated
                artifacts will launch from, when the render is written somewhere
                other than where it will run. Defaults to *project_dir* whenever
                the repo root is derived — the render is then both, and the note
                below says why the two have to be stated together.
            runtime_interpreter: The interpreter the regenerated artifacts must
                launch with, for a render destined for a machine this one cannot
                probe — a container image. Overrides the filesystem-derived
                answer; see :func:`osprey.cli.templates.claude_code.build_claude_code_context`.

        Returns:
            Dict with 'changed' and 'unchanged' keys

        Note:
            The directory holding ``config.yml`` is not the project root. A
            rendered deployment lives one level down, at ``<repo>/build``, so
            taking the render for the root makes the registry's
            ``{project_root}/build/config.yml`` resolve to
            ``<repo>/build/build/config.yml`` — a file that does not exist —
            and every MCP server and hook that reads ``CONFIG_FILE`` fails on
            it. The default is derived through the same helper the *runtime*
            resolves a repo root with, so what a regen writes and what a
            running deployment computes cannot disagree.

            One rule covers both repo shapes, because they are the same shape:
            ``<repo>/build`` on a host and ``/app/<name>/build`` in a container
            (a container is a deployment repo in its own right). For a flat
            directory that holds its own ``config.yml`` the helper returns that
            directory, which is what this argument fell back to before.

            It follows that regen re-renders a deployment *where it lives*. It
            is deliberately not a relocation tool: rendering for a root other
            than the one on disk is a build concern, and those callers
            (``osprey build``, the container render, ``osprey status``) say so
            by passing the override.

            That is also why deriving the root states the venv's home in the
            same breath. Naming a ``project_root`` at all tells
            ``_derive_runtime_interpreter`` the render is destined for a machine
            this one cannot probe, so it stops looking for the project's own
            ``.venv`` — correct for a container render, wrong here, where the
            venv is ``<render>/.venv`` and is exactly what every MCP server must
            launch. The two answer different questions (*where the repo is*
            versus *what starts the processes*), so a caller that knows one
            still has to say the other.
        """
        if project_root_override is None:
            project_root_override = repo_root_for_config(
                Path(project_dir).absolute() / "config.yml"
            )
            if runtime_venv_dir is None:
                runtime_venv_dir = project_dir
        return claude_code.regenerate_claude_code(
            self.template_root,
            self.jinja_env,
            project_dir,
            dry_run,
            project_root_override=project_root_override,
            runtime_venv_dir=runtime_venv_dir,
            runtime_interpreter=runtime_interpreter,
        )

    def regen_if_drift(self, project_dir: Path) -> list[str]:
        """Regenerate Claude Code artifacts only if they have drifted from config.

        Runs a dry-run first and performs a real regeneration only when something
        would actually change, so a launch that has nothing to do rewrites
        nothing: no artifact is re-emitted, and — the part that is visible to the
        operator — ``settings.json``'s mtime is not disturbed on a no-op. That
        mtime is the SessionStart drift hook's signal (see the stamp below), so
        touching it needlessly is not merely wasted work, it is noise in the one
        channel that reports real drift. Meanwhile a stale ``settings.json`` /
        ``.mcp.json`` is still brought back in sync after a ``config.yml`` edit.

        Args:
            project_dir: Root directory of the project (contains ``config.yml``
                and ``.claude/``).

        Returns:
            The list of regenerated artifact paths (empty when already in sync,
            or when the project has no rendered ``.claude/`` to re-sync — this is
            a re-sync operation, never a from-scratch bootstrap).
            Exceptions from the underlying regeneration propagate to the caller,
            which decides whether to fail open (web) or surface the error (CLI).
        """
        # Resolve symlinks so the rendered project_root — which regenerate_claude_code
        # derives from this very path — matches what `osprey build` baked in (build
        # resolves the path). Without this, a project built under a symlinked path
        # (e.g. /tmp → /private/tmp on macOS, or a container bind mount) reports a
        # phantom .mcp.json diff and re-renders on first regen.
        project_dir = Path(project_dir).resolve()
        settings_path = project_dir / ".claude" / "settings.json"
        if not settings_path.exists():
            return []
        preview = self.regenerate_claude_code(project_dir, dry_run=True)
        if not preview.get("changed"):
            # Verified in sync — stamp settings.json so the SessionStart drift
            # hook's mtime signal clears. Without this, a config.yml edit that
            # changes no artifact (a comment, a runtime-read field) would warn
            # at every session start until the next full `osprey build`.
            config_path = project_dir / "config.yml"
            try:
                if config_path.stat().st_mtime > settings_path.stat().st_mtime:
                    os.utime(settings_path)
            except OSError:
                pass
            return []
        result = self.regenerate_claude_code(project_dir)
        return list(result.get("changed", []))

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
