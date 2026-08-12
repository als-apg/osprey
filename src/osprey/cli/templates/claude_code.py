"""Claude Code artifact rendering, regeneration, and user-ownership."""

import json
import logging
import os
import shutil
import sys
import warnings
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from osprey.cli.profile_conventions import ownership_name
from osprey.cli.styles import console
from osprey.cli.templates import manifest as manifest_mod
from osprey.cli.templates._rendering import render_template
from osprey.errors import BuildProfileError
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog
from osprey.utils.config import resolve_env_vars
from osprey.utils.facility import resolve_facility_name

logger = logging.getLogger("osprey.cli.templates")

# python's execute is the one documented read/write-mixed exception to the
# kill-switch's hard-deny default (see the docstring in
# build_claude_code_context below): it accepts both read-only and write-access
# kernels, so the kill switch pulls it OUT of `ask` instead of hard-denying it
# outright. Every other _WRITES_CHECK-gated tool is presumed pure-write and
# must be denied. Module-level so tests can assert against the same set the
# renderer actually uses instead of re-declaring it.
_MIXED_READ_WRITE_TEMPLATES = {"python"}


def apply_textbooks_root(ctx: dict, project_dir: Path) -> None:
    """Set the textbooks-root context keys, absolute and tilde-abbreviated.

    Both keys are written unconditionally — ``None`` when the project ships no
    textbooks tree — so a template can test them without a guard.

    The tilde variant exists for permission matching: a model asked to read a
    file under the home directory abbreviates the path to ``~/...``, so a rule
    written only against the absolute form would not match what it actually
    requests. ``None`` when the tree is outside the home directory, where the
    abbreviation never appears.

    Set the same way for both the initial project creation and every later
    re-render, so a permission rule granted at build time still matches after a
    regen.

    Args:
        ctx: Template context, mutated in place.
        project_dir: The project directory; the tree is its sibling
            ``data/textbooks``.
    """
    textbooks_dir = project_dir.parent / "data" / "textbooks"
    root = str(textbooks_dir) if textbooks_dir.is_dir() else None
    home = os.path.expanduser("~")
    ctx["textbooks_root"] = root
    ctx["textbooks_root_tilde"] = (
        "~" + root[len(home) :] if root and root.startswith(home) else None
    )


def apply_agent_data_root(ctx: dict, project_dir: Path) -> None:
    """Fill in ``agent_data_root`` from the project's config when unset.

    Several artifacts interpolate the agent-data root: settings.json grants the
    agent Read access to its own outputs through it, and the CLAUDE.md prose
    tells the agent where large results land. Both must name the directory
    ``agent_data.base_dir`` actually resolves to.

    A caller that already holds the config sets the key itself and this is a
    no-op. The initial project creation does not, and an unset Jinja variable
    renders as an empty string rather than raising — so the gap showed up as
    prose pointing at ``/`` and a permission rule matching nothing. Resolving
    it at the one funnel both render paths pass through is what keeps a first
    build and every later re-render agreeing.

    Args:
        ctx: Template context, mutated in place.
        project_dir: The project directory whose ``config.yml`` is read when
            the caller supplied no value.
    """
    from osprey.utils.workspace import agent_data_base_dir

    if ctx.get("agent_data_root"):
        return
    config_file = project_dir / "config.yml"
    config = None
    if config_file.is_file():
        config = yaml.safe_load(config_file.read_text()) or {}
    ctx["agent_data_root"] = agent_data_base_dir(config)


def resolve_hierarchy_context(channel_finder: dict, project_dir: Path) -> dict[str, object] | None:
    """Hierarchy info to embed at render time, or ``None`` if there is none.

    Embedding it means the agent needs no separate ``hierarchy_info()`` tool
    call. ``None`` covers every way that can legitimately not happen — no
    database path configured, or a database that will not open — and is the
    caller's cue to leave ``channel_finder_hierarchy`` as it found it. A failure
    is warned about, never raised: an unreadable database costs the agent a
    render-time shortcut, not the build.

    Both the initial project creation and every later Claude Code re-render read
    the hierarchy through here, so the two cannot embed different levels for the
    same database.

    The warning it emits names the database and repeats why the loader turned
    it down — which for a malformed document is that loader's own account of
    what the file should have looked like — and never a stack trace. The build
    carries on and exits 0, and a traceback under a successful build is an
    alarm that teaches its operator to scroll past the next one.

    Args:
        channel_finder: The resolved ``channel_finder`` config block. The caller
            has already established that its pipeline mode is hierarchical.
        project_dir: Project root the configured database path resolves against.
    """
    try:
        db_path = (
            channel_finder.get("pipelines", {})
            .get("hierarchical", {})
            .get("database", {})
            .get("path", "")
        )
    except AttributeError:
        return None
    if not db_path:
        return None

    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    database = (project_dir / db_path).resolve()
    try:
        db = HierarchicalChannelDatabase(str(database))
    except Exception as exc:
        logger.warning(
            "Channel database %s did not load (%s: %s) — rendering without its hierarchy, "
            "so the agent will read the levels through hierarchy_info() instead.",
            database,
            type(exc).__name__,
            exc,
        )
        return None
    return {
        "hierarchy_levels": db.hierarchy_levels,
        "hierarchy_config": db.hierarchy_config,
        "naming_pattern": db.naming_pattern,
    }


def _derive_runtime_interpreter(
    project_dir: Path,
    project_root_override: Path | str | None = None,
    *,
    runtime_venv_dir: Path | str | None = None,
) -> str:
    """Pick the interpreter OSPREY-runtime processes launch with.

    Every OSPREY-runtime launch site takes this one value: ``.mcp.json`` server
    commands, framework hook commands, and the registry's
    ``{current_python_env}`` substitution. All of them ``import osprey``, so the
    single hard requirement is that the result has ``osprey`` importable.

    It is *derived* from the filesystem, never read from config. A path recorded
    in a config file is a snapshot of one machine and goes stale the moment the
    project is moved, remounted, or rebuilt elsewhere — every MCP server then
    fails to launch. Ending that failure mode is the point of this role split.

    The project's own ``.venv`` wins when the project has one: ``osprey build``
    installs ``osprey`` into it, and a command pointing inside the build output
    keeps working when the framework tree that built it is moved or deleted.
    Without one, the generating interpreter is used — whatever process is
    rendering this is running OSPREY, so ``osprey`` is importable there by
    definition.

    The venv is not always beside the tree being written. A repo's build renders
    into a staging directory and swaps that tree into ``build/`` by rename, while
    the venv is created directly at ``build/.venv`` — its final path, because a
    virtual environment records its own absolute location in every console-script
    shebang — and joins the staged tree only as part of the swap. A caller
    rendering somewhere other than where the result will run passes
    *runtime_venv_dir* to name the venv's real home.

    ``project_root_override`` alone means the artifact is being rendered *for*
    another machine, typically a container's ``/app/<project>``. A local
    ``.venv`` is then neither the interpreter that will exist at run time nor
    probeable from here, so the generating interpreter is the only honest answer.
    It is a weaker signal than *runtime_venv_dir*: an override may equally be a
    local path one level up from the render, which is why a caller that knows
    where the venv lives says so and is believed.

    Args:
        project_dir: Project directory on the filesystem being rendered from.
        project_root_override: Runtime project root when it differs from
            *project_dir*, e.g. a container mount point.
        runtime_venv_dir: Directory holding the ``.venv`` the rendered artifacts
            will launch from, when that is not *project_dir*. Passing it asserts
            the render is for this machine.

    Returns:
        str: Absolute path to the interpreter runtime processes launch with.
    """
    if runtime_venv_dir is not None or project_root_override is None:
        probe_dir = Path(runtime_venv_dir if runtime_venv_dir is not None else project_dir)
        venv_python = probe_dir / ".venv" / "bin" / "python"
        if venv_python.is_file():
            return str(venv_python)
    return sys.executable


def config_derived_context(config: dict, project_dir: Path) -> dict[str, Any]:
    """The template-context keys read straight out of a project's ``config.yml``.

    Two paths render the Claude Code artifacts — the build's
    :func:`build_claude_code_context` and the first render inside
    :meth:`osprey.cli.templates.manager.TemplateManager.create_project` — and
    the Jinja environment is not strict, so a path that omits one of these keys
    does not fail: the template renders the undefined value as nothing. That is
    how ``hook_config.json`` came out of the create_project path with an EMPTY
    write-tool list — a kill-switch safety file that looks complete and covers
    no tool at all. One spelling here, consumed by both, so the two cannot fork.

    Args:
        config: Parsed ``config.yml`` mapping (``{}`` when the bundle renders
            none — every value below then falls back to its own empty default).
        project_dir: Root of the project being rendered; declared hooks are
            resolved against the files it ships.
    """
    from osprey.utils.workspace import agent_data_base_dir

    control_system = config.get("control_system", {}) or {}
    declared_hooks = _build_declared_hook_rules(config, project_dir)
    return {
        # User-owned files: regen skips these, users edit in-place
        "user_owned": (config.get("scaffold", {}) or {}).get("user_owned", []),
        # Declared wiring for the custom hooks the profile ships
        # (claude_code.hooks). Additive only — settings.json.j2 appends these
        # after the framework's own entries, which it renders unconditionally.
        "declared_hooks": declared_hooks,
        "declared_extra_events": [
            event
            for event in CLAUDE_CODE_HOOK_EVENTS
            if event in declared_hooks and event not in FRAMEWORK_WIRED_EVENTS
        ],
        # The agent-data root the rendered artifacts must agree with.
        # settings.json grants the agent Read access to its own artifacts
        # through it, so a literal in that template would silently stop covering
        # the directory the moment a project relocated `agent_data.base_dir` —
        # and the agent would be refused the files it had just written.
        "agent_data_root": agent_data_base_dir(config),
        # Write tools blocked by the writes kill switch (for hook_config.json)
        "control_system_write_tools": control_system.get("write_tools", []),
        # Control system type for protocol-aware safety rules
        "control_system_type": control_system.get("type", "mock"),
    }


def build_claude_code_context(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    config: dict,
    project_root_override: Path | str | None = None,
    runtime_venv_dir: Path | str | None = None,
    runtime_interpreter: str | None = None,
) -> dict:
    """Build template context for Claude Code artifact rendering.

    Reconstructs the template context needed by Claude Code templates
    (.mcp.json, CLAUDE.md, settings.json, agents) from the project's
    config.yml and manifest.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        config: Parsed config.yml dictionary
        project_root_override: Runtime project root when it differs from
            *project_dir*
        runtime_venv_dir: Directory holding the ``.venv`` the rendered artifacts
            will launch from, when that is not *project_dir* — see
            :func:`_derive_runtime_interpreter`
        runtime_interpreter: The interpreter the rendered artifacts must launch
            with, when the caller KNOWS it and this filesystem cannot be asked.
            The one such caller is a render destined for a container image: the
            interpreter that will exist at run time is the image's, which is not
            on this machine to probe. Overrides
            :func:`_derive_runtime_interpreter` outright rather than seeding it,
            because a path that is not here cannot be verified here.

    Returns:
        Template context dict suitable for Claude Code templates
    """
    project_name = config.get("project_name", project_dir.name)
    package_name = project_name.replace("-", "_").lower()

    # Read template_name and artifact selections from manifest if available
    manifest_path = project_dir / manifest_mod.MANIFEST_FILENAME
    template_name = "control_assistant"
    data_bundle = "control_assistant"
    claude_md_template = "CLAUDE.md.j2"
    artifacts: dict[str, list[str]] = {}
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            creation = manifest_data.get("creation", {})
            template_name = creation.get("template", "control_assistant")
            data_bundle = creation.get("data_bundle", template_name)
            claude_md_template = creation.get("claude_md_template", "CLAUDE.md.j2")
            artifacts = manifest_data.get("artifacts", {})
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to template manifest.yml artifact list when manifest has no artifacts
    # (projects created before artifact persistence was introduced)
    if not artifacts:
        tmpl_manifest = manifest_mod.load_template_manifest(template_root, template_name)
        if tmpl_manifest:
            artifacts = tmpl_manifest.get("artifacts", {})

    # Derive feature flags from artifact selections
    selected_hooks = artifacts.get("hooks", [])

    ctx = {
        "project_name": project_name,
        "package_name": package_name,
        "project_root": str(project_root_override)
        if project_root_override
        else str(project_dir.absolute()),
        # Derived from the filesystem, never read from config — see
        # _derive_runtime_interpreter. Agent-authored code is the other half of
        # the split and does not come through here: it re-resolves per call at
        # run time (resolve_agent_interpreter), where this value is frozen into
        # a generated artifact at render time.
        #
        # KNOWN LIMIT (pre-existing, unchanged by the three-zone work): a
        # profile that names its own interpreter under ``python_env:`` does not
        # reach here, so once a project venv exists the derivation wins and the
        # profile's choice has no effect on the rendered artifacts. The explicit
        # ``runtime_interpreter`` argument is the only override this honors, and
        # only its callers set it. Documented rather than repaired here because
        # the fix is a decision about which of the two is authoritative, not a
        # local correction.
        "current_python_env": runtime_interpreter
        or _derive_runtime_interpreter(
            project_dir, project_root_override, runtime_venv_dir=runtime_venv_dir
        ),
        "template_name": template_name,
        "data_bundle": data_bundle,
        "claude_md_template": claude_md_template,
        "facility_name": resolve_facility_name(config, project_name),
        "system_timezone": config.get("system", {}).get("timezone", "UTC"),
        "selected_hooks": selected_hooks,
    }

    # Derive channel finder configuration
    channel_finder = config.get("channel_finder")
    if channel_finder and "channel-finder" in artifacts.get("agents", []):
        pipeline_mode = channel_finder.get("pipeline_mode", "hierarchical")
        ctx["channel_finder_pipeline"] = pipeline_mode
        ctx["channel_finder_mode"] = pipeline_mode
        ctx["default_pipeline"] = pipeline_mode

        # Per-pipeline tool list — shared with the registry so the agent
        # frontmatter and the server's permissions.allow stay in lockstep.
        from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE

        ctx["channel_finder_tools"] = list(CHANNEL_FINDER_TOOLS_BY_PIPELINE.get(pipeline_mode, []))

        if pipeline_mode == "hierarchical":
            hierarchy = resolve_hierarchy_context(channel_finder, project_dir)
            if hierarchy is not None:
                ctx["channel_finder_hierarchy"] = hierarchy

    ctx.setdefault("channel_finder_hierarchy", None)

    # Claude Code server + agent resolution (data-driven registry)
    claude_code_config = config.get("claude_code", {})
    ctx["facility_permissions"] = claude_code_config.get("permissions", {})

    from osprey.registry.mcp import resolve_agents, resolve_servers

    ctx["servers"] = resolve_servers(claude_code_config, ctx)
    ctx["agents"] = resolve_agents(claude_code_config, ctx, project_dir, ctx["servers"])

    ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
    ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}

    # Approval-overlap guard: a facility permissions.remove_ask or
    # permissions.allow entry naming an approval-gated (ask) tool of an enabled
    # server auto-approves it at the permission layer — and the osprey_approval
    # hook keys its policy on the bare tool name, so a skip/allow there applies
    # to EVERY instance of a template (extends clones included; per-instance
    # gating is not possible). Warn loudly at build time.
    _facility_perms = ctx["facility_permissions"] or {}
    _overridden = set(_facility_perms.get("remove_ask") or []) | set(
        _facility_perms.get("allow") or []
    )
    if _overridden:
        for _srv in ctx["servers"]:
            if not _srv["enabled"]:
                continue
            _ask_full = [f"mcp__{_srv['name']}__{t}" for t in _srv["permissions_ask"]] + list(
                _srv["fixed_ask"]
            )
            for _tool in _ask_full:
                if _tool in _overridden:
                    logger.warning(
                        "facility permissions remove_ask/allow overrides the "
                        "approval-gated tool %s of server %r — it will be "
                        "auto-approved at the permission layer",
                        _tool,
                        _srv["name"],
                    )

    # Everything the templates read straight out of config.yml, in the one
    # spelling the create_project render path shares (see config_derived_context).
    ctx.update(config_derived_context(config, project_dir))

    apply_textbooks_root(ctx, project_dir)

    # Model provider resolution for Claude Code
    from osprey.build.claude_code_resolver import ClaudeCodeModelResolver

    api_providers = config.get("api", {}).get("providers", {})
    try:
        # Build time: telemetry credentials may legitimately be the
        # deployment's to supply (the runtime re-resolves them at agent-spawn),
        # so an unresolved ${VAR} omits the auth header instead of aborting.
        model_spec = ClaudeCodeModelResolver.resolve(
            claude_code_config, api_providers, defer_unresolved_telemetry_creds=True
        )
    except ValueError as exc:
        warnings.warn(str(exc), stacklevel=2)
        model_spec = None
    ctx["claude_code_model_spec"] = model_spec

    # Kill-switch hard-block: when control-system writes are disabled, render
    # pure-write tools into permissions.deny so Claude Code's permissions layer
    # blocks the call before can_use_tool ever fires. The osprey_writes_check
    # PreToolUse hook is defense-in-depth but cannot suppress the permissions.ask
    # → can_use_tool path: a static ask entry still drives the tool to the
    # approval prompt even when the hook returns deny (observed under
    # claude-agent-sdk 0.2.93). mcp__python__execute cannot be denied wholesale
    # (it has a legitimate readonly path), so instead it is pulled OUT of
    # permissions.ask — with no static ask entry, the writes_check hook's deny
    # on a readwrite execute stands alone and blocks the call before the prompt.
    #
    # Generalized over FRAMEWORK_SERVERS rather than hardcoded per server name:
    # any hooks_pre rule gated by _WRITES_CHECK is a hardware/state write and
    # defaults to a hard deny (covers controls' channel_write and the bluesky
    # queue's arming tools automatically, plus any future write server with no code
    # change here). python's execute is the one documented exception — it
    # accepts both read_only and write_access kernels, so it is pulled into
    # remove_ask instead (see docstring above); every other writes-check-gated
    # tool is presumed pure-write and denied outright.
    if not config.get("control_system", {}).get("writes_enabled", False):
        from osprey.registry.mcp import _WRITES_CHECK, FRAMEWORK_SERVERS

        facility_perms = dict(ctx["facility_permissions"])
        deny = list(facility_perms.get("deny", []))
        remove_ask = list(facility_perms.get("remove_ask", []))
        allow = list(facility_perms.get("allow", []))
        # Approval-gated tools an enabled agent hard-requires (e.g. the
        # pyat-specialist's only compute path, mcp__python__execute) must stay
        # reachable even with writes off, or the agent declares a tool absent
        # from permissions and build validation fails. They are still pulled
        # from `ask` below -- a static ask entry drives even a read-write
        # execute to the SDK's can_use_tool prompt, bypassing the kill switch
        # (see tests/cli/test_claude_regen.py::...covers_python_execute) -- and
        # re-granted via `allow` instead: no approval-prompt path to bypass,
        # while the PreToolUse writes-check hook still hard-denies write-access
        # kernels and permits only read-only ones. Net: read-only compute works,
        # control-system writes stay blocked.
        required_ask = {
            tool for a in ctx["agents"] if a["enabled"] for tool in a.get("requires_ask_tools", [])
        }
        # Cover extends clones too, not just the literal template names: the
        # runtime hook templates (osprey_writes_check.py, osprey_approval.py)
        # exact-match template tool names and are clone-unaware — they are the
        # belt layer only; this settings.json deny/remove_ask rendering is the
        # enforced layer for clones.
        for srv in ctx["servers"]:
            if not srv["enabled"]:
                continue
            template = srv.get("extends_of") or srv["name"]
            template_def = FRAMEWORK_SERVERS.get(template)
            if template_def is None:
                continue  # custom (non-framework) server — out of scope here
            old_prefix, new_prefix = f"mcp__{template}__", f"mcp__{srv['name']}__"
            for rule in template_def.hooks_pre:
                if _WRITES_CHECK not in rule.hooks:
                    continue
                matcher = rule.matcher
                if matcher.startswith(old_prefix):
                    matcher = new_prefix + matcher[len(old_prefix) :]
                if template in _MIXED_READ_WRITE_TEMPLATES:
                    if matcher not in remove_ask:
                        remove_ask.append(matcher)
                elif matcher not in deny:
                    deny.append(matcher)
        # Re-grant, via `allow`, any mixed read/write tool an enabled agent
        # hard-requires that the kill-switch just pulled from `ask`. `allow`
        # (not `ask`) keeps it off the approval-prompt path the writes-check
        # kill switch guards, so the read-only agent keeps its compute without
        # reopening a write-access bypass. A required *pure-write* tool would be
        # in `deny`, not `remove_ask`, and is deliberately NOT rescued here.
        for tool in sorted(required_ask):
            if tool in remove_ask and tool not in allow:
                allow.append(tool)
        facility_perms["deny"] = deny
        facility_perms["remove_ask"] = remove_ask
        facility_perms["allow"] = allow
        ctx["facility_permissions"] = facility_perms

    return ctx


def compute_regen_summary(ctx: dict) -> dict:
    """Compute active/disabled server and agent lists from template context.

    Args:
        ctx: Template context dict with ``servers`` and ``agents`` lists
             (populated by ``resolve_servers`` / ``resolve_agents``).

    Returns:
        Dict with active_servers, disabled_servers, extra_servers,
        active_agents, disabled_agents keys.
    """
    servers = ctx.get("servers", [])
    agents = ctx.get("agents", [])

    return {
        "active_servers": [s["name"] for s in servers if s["enabled"]],
        "disabled_servers": [s["name"] for s in servers if not s["enabled"]],
        "extra_servers": [s["name"] for s in servers if s.get("is_custom")],
        "active_agents": [a["name"] for a in agents if a["enabled"]],
        "disabled_agents": [a["name"] for a in agents if not a["enabled"]],
    }


def is_user_owned(rel_path: str, ctx: dict) -> bool:
    """Check if a file is user-owned (regen should skip it, prune never unlink it).

    User-owned entries are listed in ``scaffold.user_owned`` in config.yml —
    framework artifacts by catalog canonical name (``rules/facility``),
    convention-derived artifacts by their destination-derived name
    (``rules/facility-ops``, ``skills/orbit-check``, ``services/foo``). One
    rule for every artifact class: a directory-shaped entry owns every file
    beneath it, which is how a profile skill (owned as a whole directory)
    keeps its non-markdown files too. During init (empty list), nothing is
    user-owned so all files are written.

    The destination-derived names — this path's own, and every ancestor
    directory's — come from
    :func:`~osprey.cli.profile_conventions.ownership_name`, the same rule the
    build registers ownership under. Spelling the read differently from the
    write would not raise: it would quietly hand a user-owned artifact back to
    regen.

    Args:
        rel_path: Relative path from project root (e.g. ".claude/rules/safety.md")
        ctx: Template context (must contain "user_owned" key)
    """
    user_owned = ctx.get("user_owned", [])
    if not user_owned:
        return False
    registry = BuildArtifactCatalog.default()
    art = registry.get_by_output(rel_path)
    if art is not None and art.canonical_name in user_owned:
        return True
    if ownership_name(rel_path, is_directory=False) in user_owned:
        return True
    return any(
        ownership_name(str(parent), is_directory=True) in user_owned
        for parent in PurePosixPath(rel_path).parents
    )


def auto_register_user_owned(project_dir: Path, canonical_name: str):
    """Add a canonical name to ``scaffold.user_owned`` in config.yml.

    Used during init to mark facility.md as user-owned so regen
    never overwrites user customizations.  Uses ruamel.yaml round-trip
    mode to preserve comments and formatting.
    """
    from osprey.utils.config_writer import config_add_to_list

    config_path = project_dir / "config.yml"
    if not config_path.exists():
        return
    config_add_to_list(config_path, ["scaffold", "user_owned"], canonical_name)


def output_path_to_canonical(output_path: str, registry: BuildArtifactCatalog) -> str | None:
    """Reverse-lookup: map an output file path to its canonical artifact name."""
    art = registry.get_by_output(output_path)
    return art.canonical_name if art else None


def _build_framework_hook_rules(
    selected_hooks: list[str],
) -> tuple[list[dict], list[dict]]:
    """Build HookRule dicts for standalone framework hooks.

    Resolves each selected hook name to its file path, parses frontmatter,
    and builds wiring entries for hooks that declare ``wiring: standalone``.

    Returns:
        ``(pre_rules, post_rules)`` — same dict shape as server hook rules.
    """
    from osprey.cli.templates.artifact_library import parse_hook_frontmatter, resolve_artifact

    pre_rules: list[tuple[int, dict]] = []  # (safety_layer, rule)
    post_rules: list[tuple[int, dict]] = []

    for hook_name in selected_hooks:
        try:
            hook_path = resolve_artifact("hooks", hook_name)
        except ValueError:
            continue

        meta = parse_hook_frontmatter(hook_path)
        if meta is None:
            continue

        rule = {
            "matcher": meta["tools"],
            "hooks": [
                {
                    "type": "command",
                    # Invoke via ``python3``, not bare ``python``: stock macOS and
                    # many Linux distros ship only ``python3``, so a bare ``python``
                    # hook command silently fails to launch (Claude Code logs and
                    # continues) — the entire PreToolUse/PostToolUse hook layer
                    # (approval, writes kill-switch, feedback capture) goes dark.
                    # settings.json.j2 rewrites this ``python3`` token to the
                    # project's resolved venv interpreter (``current_python_env``).
                    "command": f'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/{hook_path.name}"',
                    "timeout": meta["timeout"],
                }
            ],
        }

        if meta["event"] == "PreToolUse":
            pre_rules.append((meta["safety_layer"], rule))
        elif meta["event"] == "PostToolUse":
            post_rules.append((meta["safety_layer"], rule))

    # Sort by safety_layer ascending (lower = outermost gate)
    pre_rules.sort(key=lambda x: x[0])
    post_rules.sort(key=lambda x: x[0])

    return [r for _, r in pre_rules], [r for _, r in post_rules]


#: Claude Code hook events a profile may declare wiring for. The four events
#: ``settings.json.j2`` renders unconditionally come first; the rest are keys the
#: template adds only when something declares them.
CLAUDE_CODE_HOOK_EVENTS: tuple[str, ...] = (
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    "SubagentStop",
    "Notification",
    "PreCompact",
)

#: Events the framework already wires, so declared entries are appended to an
#: array the template renders regardless. Everything else in
#: :data:`CLAUDE_CODE_HOOK_EVENTS` becomes a new key when declared.
FRAMEWORK_WIRED_EVENTS: frozenset[str] = frozenset(
    {"PreToolUse", "PostToolUse", "UserPromptSubmit", "SessionStart"}
)

#: Claude Code's own default hook timeout, in seconds. A declaration that says
#: nothing about timeouts gets Claude Code's behavior rather than a shorter
#: OSPREY-invented one that would kill a slow facility hook mid-flight.
_DECLARED_HOOK_TIMEOUT = 60

_DECLARED_HOOK_EXAMPLE = (
    "  config:\n"
    "    claude_code.hooks.PreToolUse:\n"
    "      - hook: facility_guard.py\n"
    '        matcher: "mcp__epics__.*"\n'
    "    claude_code.hooks.SessionStart:\n"
    "      - facility_banner.py"
)


@lru_cache(maxsize=1)
def _framework_hook_filenames() -> frozenset[str]:
    """Filenames of the built-in hook library, whose wiring the framework owns."""
    from osprey.cli.templates.artifact_library import list_artifacts, resolve_artifact

    return frozenset(resolve_artifact("hooks", name).name for name in list_artifacts("hooks"))


def _declared_hook_filename(ref: str, event: str) -> str:
    """Validate one declared hook reference and return the bare filename.

    A declaration names a file the profile ships through its ``hooks/`` channel,
    so the only two accepted spellings are the filename itself
    (``facility_guard.py``) and the channel-qualified form the exclusion and
    ownership vocabularies already use (``hooks/facility_guard.py``). Anything
    that could address a file elsewhere — an absolute path, a ``..`` traversal,
    another channel's prefix — is refused rather than normalized, because the
    wiring it would emit points at ``.claude/hooks/<name>`` either way and the
    mismatch would only surface as a hook that never fires.
    """
    text = str(ref).strip()
    if not text:
        raise BuildProfileError(
            f"claude_code.hooks.{event} contains an empty hook reference. Name the "
            f"file the profile ships in its hooks/ directory, e.g.:\n{_DECLARED_HOOK_EXAMPLE}"
        )

    def _refuse(reason: str) -> BuildProfileError:
        return BuildProfileError(
            f"claude_code.hooks.{event} declares {text!r}: {reason}. A declaration may "
            "only name a hook the profile ships through its hooks/ channel, as "
            "'facility_guard.py' or 'hooks/facility_guard.py'."
        )

    if text.startswith(("/", "~")) or PurePosixPath(text).is_absolute():
        raise _refuse("absolute paths are not accepted")
    if "\\" in text:
        raise _refuse("backslashes are not accepted")
    parts = PurePosixPath(text).parts
    if ".." in parts:
        raise _refuse("'..' traversals are not accepted")
    if len(parts) == 2 and parts[0] == "hooks":
        name = parts[1]
    elif len(parts) == 1:
        name = parts[0]
    else:
        raise _refuse("it is not a hook filename")
    if name in (".", ""):
        raise _refuse("it is not a hook filename")
    return name


def _build_declared_hook_rules(config: dict, project_dir: Path) -> dict[str, list[dict]]:
    """Resolve the profile's declared custom-hook wiring into settings.json rules.

    Shipping a hook through the ``hooks/`` channel copies it into
    ``.claude/hooks/``; it does not make Claude Code run it. ``claude_code.hooks``
    is the declaration that does — a config key, so it resolves through the same
    pipeline as ``claude_code.servers`` and ``claude_code.permissions``: persona
    deltas can override it and the resolved profile is the single source. It is
    read back from the built project's ``config.yml``, which is why every
    ``osprey build`` re-derives the wiring instead of finding it frozen in the
    render.

    The wiring is strictly additive. Nothing here can remove or alter a
    framework entry: the template appends these rules to arrays it has already
    filled, and a declaration naming a built-in hook file is refused outright so
    the framework stays the only writer of its own wiring. Claude Code runs
    every entry whose matcher fits, so a declared hook adds a gate and can never
    relax one — array position is a property of the document, not a precedence.

    Args:
        config: The built project's parsed ``config.yml``.
        project_dir: Project root, used to confirm the declared script is
            actually on disk where the emitted command will look for it.

    Returns:
        Event name → list of settings.json hook rules, in declaration order.

    Raises:
        BuildProfileError: On a malformed declaration, an unknown event name, an
            unsafe or reserved path, a hook the resolved profile does not ship
            (an excluded one included), or a built-in hook the framework wires.
    """
    declared = (config.get("claude_code") or {}).get("hooks")
    if not declared:
        return {}
    if not isinstance(declared, dict):
        raise BuildProfileError(
            "claude_code.hooks must map a Claude Code hook event to a list of hook "
            f"declarations, got {type(declared).__name__}. For example:\n"
            f"{_DECLARED_HOOK_EXAMPLE}"
        )

    shipped = {
        name[len("hooks/") :]
        for name in config.get("scaffold", {}).get("user_owned", [])
        if isinstance(name, str) and name.startswith("hooks/")
    }
    framework_hooks = _framework_hook_filenames()

    rules: dict[str, list[dict]] = {}
    for event, entries in declared.items():
        if event not in CLAUDE_CODE_HOOK_EVENTS:
            raise BuildProfileError(
                f"claude_code.hooks declares unknown hook event {event!r}. Valid "
                f"Claude Code events: {', '.join(CLAUDE_CODE_HOOK_EVENTS)}."
            )
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise BuildProfileError(
                f"claude_code.hooks.{event} must be a list of hook declarations, got "
                f"{type(entries).__name__}. For example:\n{_DECLARED_HOOK_EXAMPLE}"
            )

        for raw_entry in entries:
            entry = _declared_hook_entry(raw_entry, event)
            name = _declared_hook_filename(entry["hook"], event)
            _vet_declared_hook(name, event, project_dir, shipped, framework_hooks)
            rules.setdefault(event, []).append(_declared_hook_rule(entry, event, name))

    return rules


def _declared_hook_entry(entry: Any, event: str) -> dict:
    """Normalize and shape-check one ``claude_code.hooks.<event>`` entry.

    A declaration may be spelled as a bare hook filename or as a mapping
    carrying one; everything downstream reads the mapping, so the two spellings
    are collapsed here rather than at each reader.

    Raises:
        BuildProfileError: If the entry is neither spelling, carries a key the
            declaration does not accept, or omits ``hook``.
    """
    if isinstance(entry, str):
        entry = {"hook": entry}
    if not isinstance(entry, dict):
        raise BuildProfileError(
            f"claude_code.hooks.{event} entries must be a hook filename or a "
            f"mapping with a 'hook' key, got {type(entry).__name__}. For "
            f"example:\n{_DECLARED_HOOK_EXAMPLE}"
        )
    unknown = set(entry) - {"hook", "matcher", "timeout"}
    if unknown:
        raise BuildProfileError(
            f"claude_code.hooks.{event} entry has unknown key(s): "
            f"{', '.join(sorted(unknown))}. Accepted keys: hook, matcher, timeout."
        )
    if "hook" not in entry:
        raise BuildProfileError(
            f"claude_code.hooks.{event} entry is missing the required 'hook' "
            f"key. For example:\n{_DECLARED_HOOK_EXAMPLE}"
        )
    return entry


def _vet_declared_hook(
    name: str,
    event: str,
    project_dir: Path,
    shipped: set[str],
    framework_hooks: frozenset[str],
) -> None:
    """Refuse a hook filename a profile may not wire — all four ways it can fail.

    Together these are what keeps the declaration additive: a reserved path and
    a built-in hook both belong to the framework, and a name the profile does
    not ship (or that never reached the project) would wire a command at a path
    with nothing behind it.

    Args:
        name: The declared hook's filename, already resolved to a bare name.
        event: The Claude Code event it was declared under, for the message.
        project_dir: Project root — the emitted command's script must be there.
        shipped: Hook filenames the resolved profile ships.
        framework_hooks: Filenames the framework wires from ``hooks:`` itself.

    Raises:
        BuildProfileError: On a reserved destination, a built-in hook, a hook
            the resolved profile does not ship, or one absent from the project.
    """
    # The EXACT reservations only. `reserved_path_channel` additionally reserves
    # every convention destination prefix — including `.claude/hooks/` itself,
    # which is precisely where a declared hook is supposed to live.
    from osprey.cli.profile_conventions import RESERVED_PATH_CHANNELS as reserved

    destination = f".claude/hooks/{name}"
    owner = reserved.get(destination)
    if owner is not None:
        raise BuildProfileError(
            f"claude_code.hooks.{event} declares {name!r}, but {destination} "
            f"is owned by {owner}. It is not a hook a profile wires."
        )
    if name in framework_hooks:
        raise BuildProfileError(
            f"claude_code.hooks.{event} declares {name!r}, which is a built-in "
            "OSPREY hook. The framework wires its own hooks from the profile's "
            "`hooks:` selection — declaring one here would invoke it twice. "
            "Select or unselect it through `hooks:` instead."
        )
    if name not in shipped:
        raise BuildProfileError(
            f"claude_code.hooks.{event} declares {name!r}, which the resolved "
            "profile does not ship. Either add it to the profile's hooks/ "
            f"directory, or — if a persona excludes 'hooks/{name}' — unwire it "
            "in that same delta by adding this line to the persona's `config:`:"
            f"\n    claude_code.hooks.{event}: null\n"
            "Use `null`, not `[]`: persona lists merge additively with the "
            "profile's, so an empty list adds nothing and leaves the wiring in "
            "place. `claude_code.hooks: {}` unwires every event at once."
        )
    if not (project_dir / ".claude" / "hooks" / name).is_file():
        raise BuildProfileError(
            f"claude_code.hooks.{event} declares {name!r}, but "
            f"{destination} is not present in the project. Wiring a script "
            "that is not there would fail silently at session start."
        )


def _declared_hook_rule(entry: dict, event: str, name: str) -> dict:
    """The settings.json hook rule one vetted declaration renders to.

    Args:
        entry: The normalized declaration (:func:`_declared_hook_entry`).
        event: The event it was declared under.
        name: Its hook filename, already vetted by :func:`_vet_declared_hook`.

    Returns:
        One rule, ready to append to the event's array.

    Raises:
        BuildProfileError: On a non-positive or non-integer ``timeout``, or a
            non-string ``matcher``.
    """
    timeout = entry.get("timeout", _DECLARED_HOOK_TIMEOUT)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise BuildProfileError(
            f"claude_code.hooks.{event} entry for {name!r} has timeout="
            f"{timeout!r}; it must be a positive whole number of seconds."
        )
    matcher = entry.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        raise BuildProfileError(
            f"claude_code.hooks.{event} entry for {name!r} has a non-string "
            f"matcher ({type(matcher).__name__})."
        )
    if matcher is None and event in ("PreToolUse", "PostToolUse"):
        # These two events are always rendered with a matcher; "*" is the
        # match-everything spelling, so an undeclared matcher keeps the
        # entry's meaning ("on every tool call") explicit in the output.
        matcher = "*"

    return {
        "matcher": matcher or "",
        "hooks": [
            {
                "type": "command",
                # ``python3`` for the same reason the framework rules use it, and
                # rewritten to the project interpreter by the same
                # settings.json.j2 filter. Deliberately no ``|| true``: a
                # facility gate that swallows its own non-zero exit would stop
                # being a gate.
                "command": (f'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/{name}"'),
                "timeout": timeout,
            }
        ],
    }


def create_claude_code_integration(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    ctx: dict,
    allowed_outputs: set[str] | None = None,
):
    """Create Claude Code integration files for the project.

    Copies template files from templates/claude_code/ into the project,
    applying dotless-to-dotted naming convention (claude/ -> .claude/,
    mcp.json.j2 -> .mcp.json).

    User-owned files (listed in ``ctx["user_owned"]``) are skipped during
    regeneration, preserving user customizations.

    When ``allowed_outputs`` is provided (from a template manifest), only
    files whose output path is in the set are generated. Config artifacts
    (CLAUDE.md, .mcp.json, .claude/settings.json) should already be in the
    set. If ``allowed_outputs`` is None, all files are generated (backward compat).

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        ctx: Template context variables
        allowed_outputs: If set, only generate files whose output path is in this set.
            When None, all files are generated (no manifest filtering).
    """
    claude_code_dir = template_root / "claude_code"

    if not claude_code_dir.exists():
        console.print(
            "  [warning]⚠[/warning] Claude Code templates not found — skipping",
            style="yellow",
        )
        return

    # Both render paths funnel through here, so this is where the agent-data
    # root is guaranteed present regardless of which one called.
    apply_agent_data_root(ctx, project_dir)

    # Build framework hook rules from selected hooks' frontmatter
    fw_pre, fw_post = _build_framework_hook_rules(ctx.get("selected_hooks", []))
    ctx["framework_pre_hooks"] = fw_pre
    ctx["framework_post_hooks"] = fw_post

    files_created = 0

    # 1. Render mcp.json.j2 -> .mcp.json
    mcp_template = claude_code_dir / "mcp.json.j2"
    if mcp_template.exists() and not is_user_owned(".mcp.json", ctx):
        render_template(jinja_env, "claude_code/mcp.json.j2", ctx, project_dir / ".mcp.json")
        files_created += 1

    # 2. Render CLAUDE.md template -> CLAUDE.md
    # The template filename is selected by the build profile via the
    # `claude_md_template` field (default "CLAUDE.md.j2"). Presets that want
    # a different persona override it to e.g. "CLAUDE.ariel.md.j2".
    claude_md_template_name = ctx.get("claude_md_template", "CLAUDE.md.j2")
    claude_md_j2 = claude_code_dir / claude_md_template_name
    claude_md_static = claude_code_dir / "CLAUDE.md"
    if not is_user_owned("CLAUDE.md", ctx):
        if claude_md_j2.exists():
            render_template(
                jinja_env,
                f"claude_code/{claude_md_template_name}",
                ctx,
                project_dir / "CLAUDE.md",
            )
        elif claude_md_static.exists():
            shutil.copy2(claude_md_static, project_dir / "CLAUDE.md")
        files_created += 1

    # 2b. Create facility.md -- user-owned artifact
    # During init, render the template in-place and auto-register as
    # user-owned so regen never overwrites user customizations.
    facility_md = project_dir / ".claude" / "rules" / "facility.md"
    facility_j2 = claude_code_dir / "claude" / "rules" / "facility.md.j2"
    if allowed_outputs is not None and ".claude/rules/facility.md" not in allowed_outputs:
        pass  # Skip -- not in manifest
    elif is_user_owned(".claude/rules/facility.md", ctx):
        pass  # Skip -- user owns it
    elif not facility_md.exists() and facility_j2.exists():
        facility_md.parent.mkdir(parents=True, exist_ok=True)
        render_template(jinja_env, "claude_code/claude/rules/facility.md.j2", ctx, facility_md)
        # Auto-register as user-owned so regen preserves user edits
        auto_register_user_owned(project_dir, "rules/facility")
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

            # Skip files in _-prefixed directories (include-only fragments)
            if any(part.startswith("_") for part in rel_path.parts[:-1]):
                continue

            if src_file.suffix == ".j2":
                output_rel = rel_path.with_suffix("")
                dst_rel = f".claude/{output_rel}"

                # Skip facility.md -- handled above (create-only semantics)
                if str(output_rel) == "rules/facility.md":
                    continue

                # Skip user-owned files
                if is_user_owned(dst_rel, ctx):
                    continue

                dst_file = project_dir / ".claude" / output_rel

                # Not in manifest (when manifest is active): remove any stale
                # file left by an earlier render pass with a wider manifest
                # (e.g. an agent dropped from the profile's artifact
                # selection) instead of orphaning it on disk.
                if allowed_outputs is not None and dst_rel not in allowed_outputs:
                    if dst_file.exists():
                        dst_file.unlink()
                        if dst_file.parent != project_dir and not any(dst_file.parent.iterdir()):
                            dst_file.parent.rmdir()
                    continue

                # Render Jinja2 template, strip .j2 extension
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                template_path = f"claude_code/claude/{rel_path}"
                render_template(jinja_env, template_path, ctx, dst_file)

                # Clean up empty rendered files (template-conditional content)
                if dst_file.exists() and not dst_file.read_text(encoding="utf-8").strip():
                    dst_file.unlink()
                    # Remove empty parent dir (e.g., .claude/skills/some-skill/)
                    if dst_file.parent != project_dir and not any(dst_file.parent.iterdir()):
                        dst_file.parent.rmdir()
                    continue
            else:
                dst_rel = f".claude/{rel_path}"

                # Skip user-owned files
                if is_user_owned(dst_rel, ctx):
                    continue

                dst_file = project_dir / ".claude" / rel_path

                # Not in manifest: remove any stale file from an earlier,
                # wider render pass instead of orphaning it on disk.
                if allowed_outputs is not None and dst_rel not in allowed_outputs:
                    if dst_file.exists():
                        dst_file.unlink()
                    continue

                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
            files_created += 1

    # 4. Set hook scripts executable
    hooks_dir = project_dir / ".claude" / "hooks"
    if hooks_dir.exists():
        for hook in hooks_dir.iterdir():
            if hook.is_file() and hook.suffix == ".py":
                hook.chmod(hook.stat().st_mode | 0o755)

    console.print(f"  [success]✓[/success] Created {files_created} Claude Code integration file(s)")


def check_user_owned_drift(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    ctx: dict,
) -> list[str]:
    """Check if framework templates changed since user claimed ownership.

    Compares the current rendered framework hash against the hash stored
    in the manifest at claim time.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        ctx: Template context dict

    Returns:
        List of canonical names whose framework template has drifted.
    """
    manifest_path = project_dir / manifest_mod.MANIFEST_FILENAME
    if not manifest_path.exists():
        return []

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    user_owned_meta = manifest_data.get("user_owned", {})
    if not user_owned_meta:
        return []

    registry = BuildArtifactCatalog.default()
    claude_code_dir = template_root / "claude_code"
    drift: list[str] = []

    for canonical_name, meta in user_owned_meta.items():
        stored_hash = meta.get("framework_hash")
        if not stored_hash:
            continue

        artifact = registry.get(canonical_name)
        if artifact is None:
            continue

        # Computed exactly as the claim-time hash was — same function — so a
        # difference here is the framework template changing and nothing else.
        current_hash = manifest_mod.framework_template_hash(
            claude_code_dir, artifact.template_path, jinja_env, ctx
        )

        if current_hash and current_hash != stored_hash:
            drift.append(canonical_name)
            console.print(
                f"  [warning]⚠[/warning] Framework updated {canonical_name} since you claimed it.\n"
                f"    Run `osprey scaffold diff {canonical_name}` to review changes.",
                style="yellow",
            )

    return drift


def regenerate_claude_code(
    template_root: Path,
    jinja_env,
    project_dir: Path,
    dry_run: bool = False,
    project_root_override: Path | str | None = None,
    runtime_venv_dir: Path | str | None = None,
    runtime_interpreter: str | None = None,
) -> dict:
    """Regenerate Claude Code artifacts from current config.yml.

    Reads config.yml, reconstructs the template context, and re-renders
    all Claude Code .j2 templates, overwriting what is there. The snapshot of
    the outgoing artifacts is taken by the caller that owns the durable zone
    (:func:`osprey.cli.build_cmd._backup_outgoing_claude_artifacts`), not here.

    Args:
        template_root: Path to osprey's bundled templates directory
        jinja_env: Jinja2 environment for template rendering
        project_dir: Root directory of the project
        dry_run: If True, report what would change without writing files
        project_root_override: If set, use this path as ``project_root`` in
            the rendered context instead of ``project_dir``.  ``project_dir``
            is still used for all file I/O (reading config, writing output).
        runtime_venv_dir: Directory holding the ``.venv`` the regenerated
            artifacts will launch from, when the render is written somewhere
            other than where it will run — see ``_derive_runtime_interpreter``.
        runtime_interpreter: The interpreter the regenerated artifacts must
            launch with, for a render destined for a machine whose filesystem
            cannot be probed from here — see :func:`build_claude_code_context`.

    Returns:
        Dict with 'changed' and 'unchanged' keys (plus 'drift_warnings' and the
        active/disabled summary on a non-dry run)

    Raises:
        FileNotFoundError: If config.yml doesn't exist in project_dir
    """
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No config.yml found in {project_dir}. Are you in an OSPREY project directory?"
        )

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = resolve_env_vars(config)

    ctx = build_claude_code_context(
        template_root,
        jinja_env,
        project_dir,
        config,
        project_root_override=project_root_override,
        runtime_venv_dir=runtime_venv_dir,
        runtime_interpreter=runtime_interpreter,
    )

    # Resolve allowed_outputs from .osprey-manifest.json artifact list.
    # Fall back to loading the template's manifest.yml for legacy projects.
    template_name = ctx.get("template_name", "control_assistant")
    osprey_manifest_path = project_dir / manifest_mod.MANIFEST_FILENAME
    regen_manifest: dict | None = None
    stored_artifacts: dict | None = None
    if osprey_manifest_path.exists():
        try:
            osprey_manifest_data = json.loads(osprey_manifest_path.read_text(encoding="utf-8"))
            stored_artifacts = osprey_manifest_data.get("artifacts") or None
            if stored_artifacts:
                # Build an in-memory manifest dict in the same format as manifest.yml
                regen_manifest = {"artifacts": stored_artifacts}
        except (json.JSONDecodeError, OSError):
            pass
    if regen_manifest is None:
        regen_manifest = manifest_mod.load_template_manifest(template_root, template_name)

    allowed_outputs = (
        manifest_mod.resolve_manifest_outputs(regen_manifest) if regen_manifest else None
    )

    # Filter agents to allowed outputs
    if allowed_outputs is not None:
        ctx["agents"] = [
            a for a in ctx["agents"] if f".claude/agents/{a['name']}.md" in allowed_outputs
        ]

    # Collect checksums of existing Claude Code files before regeneration.
    # When stored_artifacts are present, derive tracked files from the manifest;
    # otherwise fall back to the template's static tracked-file list.
    if stored_artifacts and allowed_outputs is not None:
        claude_code_files = sorted(allowed_outputs)
    else:
        claude_code_files = manifest_mod.get_tracked_files(
            template_root, template_name, project_dir
        )
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
            old_checksums[rel_path] = manifest_mod.sha256_file(file_path)

    if dry_run:
        # Render to temp dir and compare
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Create necessary subdirectories
            (tmp_dir / ".claude").mkdir(parents=True, exist_ok=True)
            # Mirror a real regen's skip semantics: user-owned (ejected)
            # artifacts and the create-only facility.md are never rewritten
            # when they already exist, so seed the project's copies into the
            # comparison dir. Without this the fresh render leaves them missing
            # and every dry run reports a phantom "would be removed" diff.
            for rel_path in claude_code_files:
                src = project_dir / rel_path
                if not src.exists():
                    continue
                if rel_path == ".claude/rules/facility.md" or is_user_owned(rel_path, ctx):
                    dst = tmp_dir / rel_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            create_claude_code_integration(template_root, jinja_env, tmp_dir, ctx, allowed_outputs)

            changed = []
            unchanged = []
            for rel_path in claude_code_files:
                tmp_file = tmp_dir / rel_path
                orig_file = project_dir / rel_path
                if tmp_file.exists():
                    new_checksum = manifest_mod.sha256_file(tmp_file)
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

            summary = compute_regen_summary(ctx)
            return {"changed": changed, "unchanged": unchanged, **summary}

    # No backup is taken here. This used to snapshot the artifacts it is about
    # to overwrite into `<project>/_agent_data/backup/claude-code-<stamp>/` —
    # inside the very tree it regenerates, under the retired agent-data
    # spelling, so the next build discarded the snapshot along with the rest of
    # `build/`. The surviving snapshot is
    # `osprey.cli.build_cmd._backup_outgoing_claude_artifacts`, which writes the
    # same artifacts to the repo's own `var/agent_data/backup/` before the
    # atomic swap: the durable zone, which is the only place a snapshot taken to
    # protect against an overwrite can usefully live.

    # Regenerate
    create_claude_code_integration(template_root, jinja_env, project_dir, ctx, allowed_outputs)

    # Compare checksums
    changed = []
    unchanged = []
    for rel_path in claude_code_files:
        file_path = project_dir / rel_path
        if file_path.exists():
            new_checksum = manifest_mod.sha256_file(file_path)
            old_checksum = old_checksums.get(rel_path)
            if old_checksum != new_checksum:
                changed.append(rel_path)
            else:
                unchanged.append(rel_path)
        elif rel_path in old_checksums:
            changed.append(rel_path)  # Removed (e.g. dropped from manifest)

    # Check for newly created files (e.g., new agents)
    new_agents_dir = project_dir / ".claude" / "agents"
    if new_agents_dir.exists():
        for agent_file in new_agents_dir.iterdir():
            if agent_file.is_file() and agent_file.suffix == ".md":
                rel = f".claude/agents/{agent_file.name}"
                if rel not in claude_code_files and rel not in changed:
                    changed.append(rel)

    # Check for user-owned drift (framework template changed since claiming)
    drift_warnings = check_user_owned_drift(template_root, jinja_env, project_dir, ctx)

    # Compute active/disabled summary
    summary = compute_regen_summary(ctx)

    return {
        "changed": changed,
        "unchanged": unchanged,
        "drift_warnings": drift_warnings,
        **summary,
    }
