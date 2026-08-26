"""Claude Code artifact rendering, regeneration, and user-ownership."""

import json
import logging
import os
import re
import shutil
import sys
import warnings
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from osprey.bluesky_tool_names import QUEUE_CONTROL_TOOLS
from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.profile_conventions import SETUP_PATCH_TOOL, ownership_name
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
# must be denied — on a render where no target may write. Where the targets
# disagree nothing can be denied statically and the whole gated set is pulled
# from ask instead; membership here still decides which of them the
# required-tool rescue may re-grant. Module-level so tests can assert against
# the same set the renderer actually uses instead of re-declaring it.
_MIXED_READ_WRITE_TEMPLATES = {"python"}

#: Tools OSPREY denies outright in every generated ``.claude/settings.json``.
#:
#: This is the interactive permission layer's hard floor: entries land in
#: ``permissions.deny``, which Claude Code refuses without ever offering an
#: approval prompt. ``Bash`` and ``Edit`` are the two that matter most — they
#: are the unmediated shell-out and unmediated file-patch escape hatches around
#: every other control the profile installs.
#:
#: Three consumers share this one definition, and they must not fork:
#:
#: * ``settings.json.j2`` renders it, in this order, into ``permissions.deny``
#:   (minus anything a facility lists under ``permissions.remove_deny``). It
#:   arrives there as the ``deny_defaults`` context key, written by
#:   :func:`config_derived_context` so BOTH render paths carry it.
#: * The build lint checks that every write-capable built-in is either denied
#:   here or gated by a ``PreToolUse`` hook rule.
#: * ``tests/agent_runner/test_write_tools.py`` guards that the headless
#:   read-only floor is never more permissive than this interactive one.
#:
#: Order is load-bearing only in that it fixes the rendered array's order;
#: appending is always safe, reordering churns every built project's diff.
DENY_DEFAULTS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "WebFetch",
    "WebSearch",
    "mcp__plugin_playwright_playwright__*",
    "mcp__plugin_context7_context7__*",
)


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


def _graphdb_configured(config: dict) -> bool:
    """Whether this project declares a graph store to query.

    Truthy exactly when ``services.graphdb`` is a mapping — including the
    fully-defaulted ``graphdb: {}`` — and falsy when the key is absent or is the
    bare ``graphdb:`` that YAML parses as ``None``. That is
    :func:`~osprey.deployment.graphdb_service.resolve_graphdb_service_config`'s
    own reading of the block, borrowed rather than re-implemented so the render
    and the deploy cannot disagree about whether a store exists.

    A malformed block is *warned* about and treated as absent. The resolver
    raises on one, because a deploy that publishes a port nobody dials is worse
    than a refusal — but this caller is a render, and aborting it over a bad
    heap size would take out every unrelated artifact in the build. Dropping the
    graph server instead costs the agent its graph tools and says so loudly.

    Args:
        config: Parsed ``config.yml`` mapping.

    Returns:
        Whether the graph MCP server has a store to point at.
    """
    from osprey.deployment.graphdb_service import resolve_graphdb_service_config

    try:
        return resolve_graphdb_service_config(config) is not None
    except ValueError as exc:
        logger.warning(
            "services.graphdb is malformed (%s) — rendering as if no graph store were "
            "configured, so the graph MCP server and its tools are left out of this "
            "project. Fix the key named above in config.yml and re-render.",
            exc,
        )
        return False


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
        # Write tools the kill switch stops short of gating on the session's
        # control target: a queue operation binds to a plan lane, not to the
        # target this session points at, so the writes-check hook leaves it to
        # the tool's own lane gate. SHORT names — an `extends` clone renames
        # only the server prefix, and the hook compares the short name so the
        # clone keeps the same carve-out. Sourced from the registry rather than
        # spelled in the hook, so renaming a tool never touches that standalone
        # hook source.
        "lane_addressed_tools": list(QUEUE_CONTROL_TOOLS),
        # Control system type for protocol-aware safety rules
        "control_system_type": control_system.get("type", "mock"),
        # Whether this deployment renders the target switch, for the
        # switch-aware half of the control-system safety rule.
        "target_switch_enabled": _renders_the_target_switch(control_system),
        # Gate for the `graph` MCP server: a ServerDefinition condition is a
        # plain truthiness test on this key, so it must be merged into the
        # context BEFORE resolve_servers runs (both render paths do — see
        # build_claude_code_context). Without a configured store the server is
        # left out of the render entirely rather than shipped as a tool that
        # can only fail.
        "graphdb_configured": _graphdb_configured(config),
        # The interactive deny floor settings.json.j2 renders into
        # permissions.deny. Sourced from DENY_DEFAULTS so the template, the
        # build lint and the read-only-floor drift test cannot fork.
        "deny_defaults": list(DENY_DEFAULTS),
        # The writes-off kill switch's own deny entries, kept OUT of
        # `facility_permissions['deny']` so that `remove_deny` — which a profile
        # authors — can never subtract them. settings.json.j2 appends this list
        # last and unfiltered. Empty here and overwritten by
        # build_claude_code_context's kill-switch block when writes are off; the
        # create_project path never runs that block, so the empty default is
        # what it renders (and must render: writes state is not settled there).
        "killswitch_deny": [],
    }


def _renders_the_target_switch(control_system: dict) -> bool:
    """Whether the rendered config gives this deployment two targets to switch between.

    Delegates to :func:`osprey_connectors.types.switch_capable`, the same
    predicate the controls server uses at run time to decide whether its tools
    are served by a connector-host child. Restating it here — "an epics block
    and a virtual_accelerator block", say — would be a second opinion that gets
    a ``doocs`` deployment wrong and a ``mock``-with-an-epics-block deployment
    wrong in the other direction, and the failure would be a frozen rule
    promising the agent a switch the runtime refuses to perform.

    Deliberately not keyed on the ``control_system.target_switch`` tuning keys:
    those have defaults and can be present in a deployment with nowhere to
    switch to, while a connector block cannot be defaulted into existence.
    """
    from osprey_connectors.types import switch_capable

    return switch_capable(control_system)


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
        # KNOWN LIMIT (pre-existing, unchanged by the four-zone work): a
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

    # Everything the templates read straight out of config.yml, in the one
    # spelling the create_project render path shares (see config_derived_context).
    #
    # Merged HERE, before the registry resolves servers and agents, because a
    # `condition=` on a ServerDefinition is a plain truthiness test on a ctx key:
    # a key that lands after resolution reads as absent, and the server is
    # silently disabled with no warning. create_project merges the same helper
    # before its own resolve_servers call, so this is also what keeps the two
    # paths from forking on any server gated by a config-derived key.
    ctx.update(config_derived_context(config, project_dir))

    # Derive channel finder configuration
    channel_finder = config.get("channel_finder")
    if channel_finder and "channel-finder" in artifacts.get("agents", []):
        from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE
        from osprey.services.channel_finder.core.exceptions import PipelineModeError

        # No default paradigm. A project that ships the channel-finder agent was
        # built against one paradigm's store, and the agent's prompt and tools
        # differ per paradigm — guessing one here would render an agent that
        # does not match the data on disk, and the mismatch only shows up at
        # run time as a tool that is not there.
        pipeline_mode = channel_finder.get("pipeline_mode")
        if not pipeline_mode:
            raise PipelineModeError(
                "channel_finder.pipeline_mode is required when the channel-finder "
                "agent is selected. Set it in config.yml to one of: "
                f"{', '.join(VALID_CHANNEL_FINDER_MODES)}."
            )
        if pipeline_mode not in VALID_CHANNEL_FINDER_MODES:
            raise PipelineModeError(
                f"Unknown channel_finder.pipeline_mode: {pipeline_mode!r}. "
                f"Valid modes are: {', '.join(VALID_CHANNEL_FINDER_MODES)}."
            )
        ctx["channel_finder_pipeline"] = pipeline_mode
        ctx["channel_finder_mode"] = pipeline_mode
        ctx["default_pipeline"] = pipeline_mode

        # Per-pipeline tool list — shared with the registry so the agent
        # frontmatter and the server's permissions.allow stay in lockstep. The
        # mode is already known-good above, so a paradigm with no entry here is
        # one served by no channel-finder MCP pipeline of its own, and the empty
        # list is the right answer rather than a swallowed typo.
        ctx["channel_finder_tools"] = list(CHANNEL_FINDER_TOOLS_BY_PIPELINE.get(pipeline_mode, []))

        # Only the hierarchical paradigm embeds extra render context; the other
        # paradigms need none.
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

    # Build-time index of the Bluesky plan catalog, for the bundled plans skill.
    #
    # This lives here rather than in a caller's context because
    # regenerate_claude_code rebuilds the context from config.yml alone and runs
    # last: an index passed only through create_project's context would render
    # empty in the shipped artifact. Both paths come through this function, so
    # both carry the index.
    ctx["bluesky_plan_index"] = _build_bluesky_plan_index(config, project_dir, ctx)

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

    apply_textbooks_root(ctx, project_dir)

    # Model provider resolution for Claude Code
    from osprey.build.claude_code_resolver import ClaudeCodeModelResolver
    from osprey.build.claude_code_telemetry import openobserve_published_port

    api_providers = config.get("api", {}).get("providers", {})
    try:
        # Build time: telemetry credentials may legitimately be the
        # deployment's to supply (the runtime re-resolves them at agent-spawn),
        # so an unresolved ${VAR} omits the auth header instead of aborting.
        # The store's port is the one this config publishes — never the
        # builder's environment, which a rendered artifact must not bake in.
        model_spec = ClaudeCodeModelResolver.resolve(
            claude_code_config,
            api_providers,
            defer_unresolved_telemetry_creds=True,
            openobserve_port=openobserve_published_port(config),
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
    #
    # Write posture is per target, so the render is three-way rather than two:
    #
    # * NO target may write — the kill switch above, unchanged. One static deny
    #   is correct because the tool is illegal wherever the session points.
    # * EVERY target may write — nothing rendered, unchanged.
    # * The targets DISAGREE — the deny cannot be static, because the same tool
    #   is legal on one target and refused on the other and settings.json is
    #   rendered once, before any session picks a target. So the render steps
    #   aside: every writes-check-gated matcher is pulled from `ask` and none is
    #   denied, and the runtime carries it per call in three places —
    #   osprey_writes_check (stage 1 the session posture, stage 2 the active
    #   target's posture), the approval hook's defer, and, for the lane-addressed
    #   `queue_*` tools that stage 2 deliberately skips, the lane-bound posture
    #   re-read inside the bluesky queue tools themselves. Dropping the `ask` is
    #   the load-bearing half: a static ask entry left in settings.json would
    #   reopen the can_use_tool prompt none of those three can suppress (the SDK
    #   aggregates any-ask-wins), and an operator would be prompted to approve a
    #   write the target's posture forbids.
    #
    # The postures come from `session_posture`, which yields both targets only
    # on a deployment that renders the switch and otherwise yields the built
    # connector's own posture alone. Looping CONTROL_TARGETS here instead would
    # call a deployment mixed on the strength of a target no session can select
    # — a mock deployment carrying an armed epics block would lose its hard deny
    # over a machine it never reaches.
    from osprey_connectors.types import session_posture

    control_system = config.get("control_system", {}) or {}
    writes_by_target = session_posture(control_system)
    if not all(writes_by_target.values()):
        from osprey.registry.mcp import _WRITES_CHECK, FRAMEWORK_SERVERS

        mixed = any(writes_by_target.values())
        facility_perms = dict(ctx["facility_permissions"])
        # Kill-switch denies land in their OWN context key, never in
        # facility_permissions['deny']. `remove_deny` is profile-authored and
        # subtracts from profile-authored deny sources only; if these matchers
        # sat in the facility list, a profile could name one under remove_deny
        # and lift a writes-off deny. settings.json.j2 appends killswitch_deny
        # last and never filters it.
        killswitch_deny = list(ctx.get("killswitch_deny") or [])
        remove_ask = list(facility_perms.get("remove_ask", []))
        allow = list(facility_perms.get("allow", []))
        # Matchers the earlier (filtered) deny parts already render. Skipping
        # them keeps the rendered array duplicate-free and byte-identical to the
        # pre-split render for every profile that does not both deny and
        # remove_deny the same entry. A matcher the profile denies AND removes
        # is filtered out up there, so it is NOT in this set and the kill switch
        # re-adds it below — which is the whole point of the split.
        _remove_deny = list(facility_perms.get("remove_deny", []))
        already_denied = {
            d
            for d in list(ctx.get("deny_defaults") or []) + list(facility_perms.get("deny", []))
            if d not in _remove_deny
        }
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
        # The matchers pulled from `ask` because their template is documented
        # read/write-mixed — the only ones the rescue below may promote to
        # `allow`. Tracked separately from `remove_ask`, which on a mixed render
        # also holds the pure-write matchers (and which a profile can seed with
        # anything at all): a pure-write tool in `allow` is an unguarded write,
        # so the rescue must never be able to reach one.
        mixed_remove_ask: set[str] = set()
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
                    mixed_remove_ask.add(matcher)
                    if matcher not in remove_ask:
                        remove_ask.append(matcher)
                elif mixed:
                    if matcher not in remove_ask:
                        remove_ask.append(matcher)
                elif matcher not in killswitch_deny and matcher not in already_denied:
                    killswitch_deny.append(matcher)
        # Re-grant, via `allow`, any mixed read/write tool an enabled agent
        # hard-requires that the kill-switch just pulled from `ask`. `allow`
        # (not `ask`) keeps it off the approval-prompt path the writes-check
        # kill switch guards, so the read-only agent keeps its compute without
        # reopening a write-access bypass. Drawn from `mixed_remove_ask` and not
        # from `remove_ask`: a pure-write tool is denied on an all-off render and
        # hook-gated on a mixed one, and must reach `allow` from neither.
        for tool in sorted(required_ask):
            if tool in mixed_remove_ask and tool not in allow:
                allow.append(tool)
        facility_perms["remove_ask"] = remove_ask
        facility_perms["allow"] = allow
        ctx["facility_permissions"] = facility_perms
        ctx["killswitch_deny"] = killswitch_deny

    return ctx


def _build_bluesky_plan_index(config: dict, project_dir: Path, ctx: dict) -> dict | None:
    """Index the project's Bluesky plans for the bundled plans skill.

    Runs only when the ``bluesky`` MCP server is enabled: with the server off
    the agent has no plan tools, the skill renders empty, and there is nothing
    to describe. A returned dict therefore means the pass ran — a project whose
    plan directories resolved nothing still gets a dict with no rows, which the
    skill renders as an honest "no plans were visible at build time" line rather
    than as an empty table. ``None`` means the pass never ran.

    Never raises: an unreadable plan directory or a malformed plan file is a
    build warning, not a build failure.

    Args:
        config: The parsed ``config.yml`` mapping.
        project_dir: The built project's root, used to resolve a relative
            ``plan_dir``. Not the CWD — the build does not run from the project
            root — and not ``project_root_override``, which names a path on the
            runtime filesystem (a container's project root) that is not here to
            read.
        ctx: The context built so far, read for ``enabled_servers``.

    Returns:
        ``{"rows": [...], "overflow": int, "unreadable_dirs": [...]}``, or
        ``None`` when the pass did not run.
    """
    if "bluesky" not in ctx.get("enabled_servers", set()):
        return None

    from osprey.cli.templates.plan_index import build_plan_index

    try:
        index = build_plan_index(config, project_dir)
    except Exception as exc:  # pragma: no cover - defensive; builder is fail-soft
        # The builder documents itself as never raising for a bad input. If it
        # ever does, the build still finishes: the skill simply reports no
        # build-time listing and points the agent at ``list_plans``.
        logger.warning("bluesky plan index skipped: %s", exc)
        return None

    for warning in index.warnings:
        logger.warning("bluesky plan index: %s", warning)
    for directory in index.unreadable_dirs:
        logger.warning(
            "bluesky plan index: configured plan directory could not be read: %s", directory
        )

    return {
        "rows": [
            {
                "name": row.name,
                "description": row.description,
                "writes": row.writes,
                "provenance": row.provenance,
            }
            for row in index.rows
        ],
        "overflow": index.overflow,
        "unreadable_dirs": list(index.unreadable_dirs),
    }


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


#: Built-in Claude Code tools that can write — to the filesystem, or (``Bash``)
#: to anything the shell reaches. Every generated profile must gate each of
#: these — either by hard-denying it in ``permissions.deny`` or by matching it
#: with a ``PreToolUse`` hook rule — so a profile can never ship able to write
#: with no gate at all.
#:
#: ``Bash`` and ``Edit`` are here for the reason :data:`DENY_DEFAULTS` names
#: them first: they are the unmediated shell-out and unmediated file-patch
#: escape hatches around every other control the profile installs. Their only
#: gate in a shipped preset is that :data:`DENY_DEFAULTS` denies them — and
#: ``claude_code.permissions.remove_deny`` lets a facility take that away, which
#: before this entry did so with no lint and no warning. Listing them here is
#: what makes ``remove_deny: ["Bash"]`` a build failure unless something else
#: actually gates the tool.
#:
#: The memory-guard hook's ``Write|MultiEdit|NotebookEdit`` matcher is what
#: gates the other three in the shipped presets; see
#: :func:`_lint_write_tools_are_gated`.
_WRITE_CAPABLE_BUILTINS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
)


def _rendered_deny_list(ctx: dict) -> list[str]:
    """Reproduce the ``permissions.deny`` array ``settings.json.j2`` will render.

    Mirrors the template's own three-part construction exactly:

    1. the ``deny_defaults`` floor (the hoisted :data:`DENY_DEFAULTS` constant),
       minus anything a facility lists under ``permissions.remove_deny``;
    2. the profile-authored ``permissions.deny`` entries, minus ``remove_deny``
       as well — a profile may subtract what a profile added;
    3. the ``killswitch_deny`` entries, appended last and NEVER filtered.

    Part 3 is why the split exists: the writes-off kill switch writes its
    matchers into their own context key, so no ``remove_deny`` a profile authors
    can lift a deny the kill switch imposed.

    Reproduced in Python from the same context keys the template reads — never
    regex-parsed out of the rendered JSON — so the lint and the template share
    one source and cannot silently drift.

    A context that is missing ``deny_defaults`` yields an empty floor here, which
    is exactly the silent-empty-deny hazard :func:`_lint_write_tools_are_gated`
    guards against.

    Args:
        ctx: The render context, read for ``deny_defaults``,
            ``facility_permissions`` and ``killswitch_deny``.

    Returns:
        The deny entries, in render order.
    """
    facility = ctx.get("facility_permissions") or {}
    remove_deny = facility.get("remove_deny", []) or []
    deny = [d for d in (ctx.get("deny_defaults") or []) if d not in remove_deny]
    deny.extend(d for d in (facility.get("deny", []) or []) if d not in remove_deny)
    deny.extend(ctx.get("killswitch_deny") or [])
    return deny


#: Provenance tags for the ``PreToolUse`` matchers the lint reads.
#:
#: ``framework`` rules come from OSPREY's own wiring — a selected framework hook
#: (``fw_pre_rules``) or an enabled server's registry ``hooks_pre`` — so the
#: build knows which script runs and that it emits an explicit
#: ``permissionDecision``. ``profile`` rules are whatever the profile declared
#: under ``claude_code.hooks.PreToolUse``: vetted to exist and to be wired
#: correctly, but never inspected for what they decide.
_MATCHER_FRAMEWORK = "framework"
_MATCHER_PROFILE = "profile"


def _pretooluse_matchers(ctx: dict, fw_pre_rules: list[dict]) -> list[tuple[str, str]]:
    """Every ``PreToolUse`` matcher ``settings.json.j2`` will render, with provenance.

    Concatenates the same three sources the template does, in the same spirit:
    the framework hook rules (``fw_pre_rules``, e.g. the memory-guard hook), each
    enabled server's ``hooks_pre`` rules, and the profile's declared
    ``PreToolUse`` wiring. Every rule is a plain dict carrying a ``"matcher"``
    key, so access is uniform.

    Each matcher is paired with where it came from — :data:`_MATCHER_FRAMEWORK`
    for the first two sources, :data:`_MATCHER_PROFILE` for the third — because
    the lint treats the two differently: see
    :func:`_lint_write_tools_are_gated`.

    Args:
        ctx: The render context, read for ``servers`` and ``declared_hooks``.
        fw_pre_rules: The framework PreToolUse rules just computed by
            :func:`_build_framework_hook_rules`.

    Returns:
        ``(matcher, provenance)`` pairs, in render order.
    """
    matchers: list[tuple[str, str]] = [
        (rule.get("matcher", ""), _MATCHER_FRAMEWORK) for rule in (fw_pre_rules or [])
    ]
    for srv in ctx.get("servers", []) or []:
        if srv.get("enabled"):
            matchers.extend(
                (rule.get("matcher", ""), _MATCHER_FRAMEWORK)
                for rule in (srv.get("hooks_pre") or [])
            )
    declared = ctx.get("declared_hooks") or {}
    matchers.extend(
        (rule.get("matcher", ""), _MATCHER_PROFILE) for rule in (declared.get("PreToolUse") or [])
    )
    return matchers


#: Characters whose presence makes Claude Code read a hook matcher as a regular
#: expression rather than an exact tool name. Kept as a set so
#: :func:`_matcher_covers` can decide per matcher which mode applies, the same
#: way Claude Code does.
_REGEX_METACHARACTERS = frozenset(r".^$*+?{}[]()|\\")

#: Matcher spellings that gate every tool. Claude Code accepts the literal
#: ``"*"``, an empty string, and an omitted ``matcher`` key as "all tools";
#: ``".*"`` reaches the same place through the regex path.
_MATCH_ALL_MATCHERS = frozenset({"*", ".*", ""})


def _matcher_covers(matcher: str | None, tool: str) -> bool:
    """Whether one ``PreToolUse`` matcher gates ``tool``.

    Mirrors how Claude Code itself resolves a hook matcher, because a lint that
    read them more narrowly would refuse builds whose hooks genuinely do gate the
    tool. Recognised, in this order:

    * **Match-all** — the literal ``"*"``, the regex ``".*"``, an empty string,
      or ``None`` (an omitted ``matcher`` key). All four run the hook on every
      tool call, so all four cover every tool.
    * **Regex** — any matcher containing a regex metacharacter
      (:data:`_REGEX_METACHARACTERS`) is compiled and matched **unanchored**,
      the way Claude Code matches it. So ``"Write.*"``, ``"^(Write|Edit)$"`` and
      ``"Write|MultiEdit|NotebookEdit"`` (the memory-guard hook's own matcher)
      all cover ``Write``, and ``"Edit.*"`` also covers ``NotebookEdit`` —
      unanchored is what Claude Code does, so it is what this reports.
    * **Exact alternation** — otherwise, and as the fallback when a matcher
      containing metacharacters is not a compilable regex, the matcher is split
      on ``|`` and each alternative compared for equality.

    An author who wants a matcher this lint will read as covering a tool can
    therefore spell it as the bare tool name, as a ``|`` alternation, or as any
    regex that matches the name.

    Args:
        matcher: The matcher string (``None`` for an omitted ``matcher`` key).
        tool: The tool name to test for.

    Returns:
        ``True`` when ``tool`` is gated by this matcher.
    """
    if matcher is None:
        return True
    matcher = matcher.strip()
    if matcher in _MATCH_ALL_MATCHERS:
        return True
    if any(char in _REGEX_METACHARACTERS for char in matcher):
        try:
            if re.search(matcher, tool):
                return True
        except re.error:
            pass  # Not a compilable regex; fall through to exact alternation.
    return tool in [part.strip() for part in matcher.split("|")]


def _lint_write_tools_are_gated(ctx: dict, fw_pre_rules: list[dict]) -> None:
    """Refuse to build a profile that can write with no gate.

    Every write-capable built-in (:data:`_WRITE_CAPABLE_BUILTINS`) must be
    gated in one of two ways: hard-denied in the rendered ``permissions.deny``
    floor, OR matched by a ``PreToolUse`` hook rule. Either is accepted, and the
    distinction is load-bearing: Claude Code resolves permissions ``deny`` > ``ask``
    > ``allow``, so a hook ``allow`` can never override a ``deny``. Denying
    ``Write``/``MultiEdit``/``NotebookEdit`` outright would therefore permanently
    block legitimate memory writes (Write/MultiEdit to the Claude memory files)
    and artifact notebook edits (NotebookEdit to the agent-data tree). The
    memory-guard ``PreToolUse`` rule is what legitimately gates them instead —
    allowing the good paths and denying the rest — so the lint takes a matcher as
    sufficient. ``Bash`` and ``Edit`` have no such legitimate path and are gated
    by :data:`DENY_DEFAULTS`; the lint is what makes removing them from that
    floor via ``claude_code.permissions.remove_deny`` a build failure rather than
    a silent widening.

    An empty rendered deny floor is itself the hazard this guards against: a
    context missing ``deny_defaults`` renders an EMPTY ``permissions.deny`` array
    silently (a settings.json that looks whole but denies nothing). With an empty
    floor, a write-capable tool passes only if a ``PreToolUse`` rule still gates
    it; if none does, the build is refused rather than shipping an ungated writer.

    **What a matcher does and does not prove.** The lint checks that a covering
    rule EXISTS; it cannot check that the hook behind it ever refuses anything.
    A ``PreToolUse`` hook that exits 0 without emitting a ``permissionDecision``
    falls through to the normal permission flow — i.e. it allows — and one that
    exits non-zero (or fails to launch) is a non-blocking error that also lets
    the call proceed. The framework hooks OSPREY wires emit an explicit
    ``permissionDecision``, so a :data:`_MATCHER_FRAMEWORK` matcher is a real
    gate. A matcher the profile itself declares under
    ``claude_code.hooks.PreToolUse`` is not inspected at all, so it satisfies the
    lint while proving only that a script runs — which is why a tool gated by
    nothing but a profile-declared matcher is allowed through with a build-time
    warning rather than silently. (Refusing outright was considered and rejected:
    no framework hook matches ``Bash`` or ``Edit``, and ``deny`` outranks
    ``allow``, so refusing would make a facility-gated shell unbuildable by any
    means rather than merely loud.)

    Args:
        ctx: The render context, carrying the rendered deny floor
            (``deny_defaults`` + ``facility_permissions``) and the PreToolUse
            sources (``servers``, ``declared_hooks``).
        fw_pre_rules: The framework PreToolUse rules just computed for this
            render, before they are written into ``ctx``.

    It also carries one guard that is not about the built-ins at all: the
    container's setup-capability check reads the profile's deny and not this
    floor, so the two must not swap roles (see the check itself, below). It
    lives here because this is the build's one lint that already holds the
    rendered floor in its hand.

    Raises:
        BuildProfileError: Naming the first write-capable built-in that is
            neither hard-denied nor matched by any ``PreToolUse`` rule — or
            reporting that the setup tool has moved into :data:`DENY_DEFAULTS`,
            where the container's capability check cannot see it.
    """
    # The container's chown of `build/config.yml` is decided one step earlier,
    # from the PROFILE's own deny/remove_deny alone (build_cmd.
    # _profile_setup_patch_capable) — it never sees this floor. That parity
    # holds only while the setup tool is denied by profiles and not by
    # DENY_DEFAULTS, so moving the deny down into the floor would leave every
    # persona reading as capable and hand every image's config.yml to the
    # agent. Checked here rather than trusted: this is the one place that knows
    # the floor's contents at build time.
    if any(fnmatchcase(SETUP_PATCH_TOOL, entry) for entry in DENY_DEFAULTS):
        raise BuildProfileError(
            f"{SETUP_PATCH_TOOL!r} is denied by the DENY_DEFAULTS floor, which the "
            "container's setup-capability check does not read: every rendered "
            "Dockerfile would chown build/config.yml to the agent's user while "
            "the tool that edits it is in fact denied. Keep the deny in the "
            "presets (claude_code.permissions.deny), or teach "
            "build_cmd._profile_setup_patch_capable about the floor."
        )

    deny = _rendered_deny_list(ctx)
    matchers = _pretooluse_matchers(ctx, fw_pre_rules)
    for tool in _WRITE_CAPABLE_BUILTINS:
        if tool in deny:
            continue
        covering = [(m, src) for m, src in matchers if _matcher_covers(m, tool)]
        if covering:
            if all(src == _MATCHER_PROFILE for _, src in covering):
                logger.warning(
                    "%s is not in permissions.deny and is gated only by a "
                    "PreToolUse matcher this profile declares itself (%s). The "
                    "build checks that the rule exists, not that the hook "
                    "refuses anything — a hook that exits without a "
                    "permissionDecision allows the call. Verify that hook denies "
                    "what it is there to deny.",
                    tool,
                    ", ".join(repr(m) for m, _ in covering),
                )
            continue
        floor_note = (
            " The rendered permissions.deny floor is empty, so nothing is denied "
            "at the permission layer — check that deny_defaults reached the render "
            "context."
            if not deny
            else ""
        )
        raise BuildProfileError(
            f"The generated profile would ship able to run {tool!r} with no gate: "
            f"{tool!r} is neither in permissions.deny nor matched by any PreToolUse "
            "hook rule. A profile must not be able to write with no gate. "
            "Wire the memory-guard hook (its 'Write|MultiEdit|NotebookEdit' matcher "
            f"gates the three file-writing built-ins), or add {tool!r} to "
            "permissions.deny — if it is missing because "
            f"claude_code.permissions.remove_deny lists {tool!r}, drop that entry. "
            "A PreToolUse matcher satisfies this check by existing; the build "
            "cannot tell whether the hook behind it refuses anything, so a hook "
            "used as the gate must emit an explicit permissionDecision of 'deny' "
            f"(exiting 0 with no output allows the call).{floor_note}"
        )


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
            "  [warning]⚠[/warning] Claude Code templates not found. Skipping them.",
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

    # Build-time safety lint: refuse to render a profile in which any
    # write-capable built-in (Write/MultiEdit/NotebookEdit) is neither hard-denied
    # nor gated by a PreToolUse hook. Runs on both render paths (create + regen)
    # because both funnel through here, and before any file is written so a
    # failing profile never lands a half-rendered .claude/ tree.
    _lint_write_tools_are_gated(ctx, fw_pre)

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

    logger.debug("Created %s Claude Code integration file(s)", files_created)


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
    # Fall back to loading the template's manifest.yml for a project without one.
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

    # No backup is taken here. A snapshot written inside the tree this
    # regenerates would be discarded, along with the rest of `build/`, by the
    # next build. The snapshot that counts is
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
