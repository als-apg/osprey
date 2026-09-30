"""Claude Code tool names and deny entries that OSPREY enforces.

This is where the names of Claude Code's own tools, and the ``permissions.deny``
entries built from them, are defined. Each list is a separate policy and is
written out in full, never derived from another, so a rename in one list fails
a test instead of silently changing a second gate. The relations between them
are pinned by tests, not by construction.
"""

from __future__ import annotations

#: Tool-name namespaces of MCP servers the agent CLI takes from somewhere other
#: than the rendered ``.mcp.json``: Claude Code plugins
#: (``mcp__plugin_<plugin>_<server>__<tool>``) and claude.ai connectors
#: (``mcp__claude_ai_<name>__<tool>``).
#:
#: Every launch is already strict about MCP servers, so neither kind loads;
#: these entries are the second line, for a path that loses that flag. A deny
#: cannot be reopened by an allow rule. Every floor spells them out rather than
#: unpacking this tuple, and ``tests/agent_runner/test_write_tools.py`` pins
#: that none drops them.
FOREIGN_MCP_NAMESPACES: tuple[str, ...] = ("mcp__plugin_*", "mcp__claude_ai_*")

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
#:   :func:`osprey.cli.templates.claude_code.config_derived_context` so BOTH
#:   render paths carry it.
#: * The build lint
#:   :func:`osprey.cli.templates.claude_code._lint_write_tools_are_gated` checks
#:   that every write-capable built-in is either denied here or gated by a
#:   ``PreToolUse`` hook rule.
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
    "mcp__plugin_*",
    "mcp__claude_ai_*",
)

#: Built-in (non-MCP) Claude Code tools that can write to disk, execute shell
#: commands, or reach the network. Under ``permission_mode=bypassPermissions``
#: the settings.json allow/deny/ask layer is INERT, so for the headless,
#: read-only ``osprey query`` path these MUST be blocked via ``disallowed_tools``
#: or the read-only guarantee is hollow — e.g. ``Bash`` could ``caput`` a PV
#: (a hardware write) or ``rm`` files, entirely bypassing the MCP write guard.
#: This is a superset of the built-in (non-``mcp__``) entries in
#: :data:`DENY_DEFAULTS`; the ``mcp__`` entries reach the headless floor through
#: :data:`FOREIGN_MCP_NAMESPACES` in
#: :func:`~osprey.agent_runner.write_tools.read_only_disallowed_tools`.
#: test_write_tools.py guards that the headless floor never drifts below the
#: interactive deny policy. Every name is a tool the pinned CLI builds list,
#: checked by tests/agent_runner/test_tool_name_conformance.py.
READ_ONLY_DENIED_BUILTINS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    # Runs shell commands in the background.
    "Monitor",
    # A job may not fire or schedule jobs.
    "Workflow",
    "CronCreate",
    "ScheduleWakeup",
    # Sends messages to other sessions, out of this run.
    "SendMessage",
    # Creates a git worktree on disk.
    "EnterWorktree",
)

#: Server-side tool denylist — tools that must NEVER be used by headless dispatch.
#: Defense-in-depth: the event dispatcher already restricts tools via triggers.yml,
#: but the worker blocks dangerous tools regardless of what the trigger requests.
DISPATCH_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        "WebFetch",
        "WebSearch",
        "mcp__plugin_*",
        "mcp__claude_ai_*",
        # Arbitrary shell access from a headless, unattended run is never warranted —
        # the safety story is the per-trigger allowlist + MCP tools, not a raw shell.
        # ``Bash`` runs commands; ``TaskOutput`` reads a background command's output;
        # ``TaskStop`` stops one. Deny all three. Every name here is a tool the pinned
        # CLI builds list, which tests/agent_runner/test_tool_name_conformance.py checks.
        "Bash",
        "TaskOutput",
        "TaskStop",
        # Runs shell commands in the background.
        "Monitor",
        # A job may not fire or schedule jobs.
        "Workflow",
        "CronCreate",
        "ScheduleWakeup",
        # Sends messages to other sessions, out of this run.
        "SendMessage",
        # Creates a git worktree on disk.
        "EnterWorktree",
        # A job may not fire jobs. ``trigger_config`` already refuses the whole
        # ``mcp__event_dispatcher__`` prefix when the triggers file is loaded; this
        # entry is the run-time floor, which holds whatever a dispatch request asks
        # for and whether or not the dispatcher is wired into the render at all.
        "mcp__event_dispatcher__manual_fire",
    }
)

#: The ``permissions.deny`` entries every persona must ship before a deployment
#: may run OPEN (``modules.web_terminals.auth.method: none``). Each is a
#: host-network egress path an agent can take from *outside* the python
#: executor, which is where the open-mode socket guard sits: a shell, the two
#: web tools, and every Claude Code plugin's MCP server. The Playwright browser
#: server is a plugin, and ``mcp__plugin_*`` is the shipped entry that closes it.
#:
#: Every entry is spelled exactly as :data:`DENY_DEFAULTS` spells it — that
#: tuple is what ``settings.json.j2`` writes into the artifact this gate reads,
#: and the comparison is literal (see
#: :func:`~osprey.deployment.web_terminals.personas.settings_json_denies`).
#: A strict subset of it, deliberately: ``Edit`` writes files rather than
#: reaching the network, and ``mcp__claude_ai_*`` is left out because a
#: claude.ai connector runs in the provider's cloud, not on this host, so it is
#: no route back to the deployment's own terminals. Written out rather than
#: derived by filtering ``DENY_DEFAULTS``, so that a rename there fails a test
#: loudly instead of silently dropping an entry from this gate and weakening it
#: (``test_the_open_mode_egress_tools_are_spelled_as_the_template_ships_them``).
OPEN_MODE_EGRESS_TOOLS: tuple[str, ...] = (
    "Bash",
    "WebFetch",
    "WebSearch",
    "mcp__plugin_*",
)

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
#: ``claude_code.permissions.remove_deny`` lets a facility take that away.
#: Listing them here is what makes ``remove_deny: ["Bash"]`` a build failure
#: unless something else actually gates the tool.
#:
#: The memory-guard hook's ``Write|MultiEdit|NotebookEdit`` matcher is what
#: gates the other three in the shipped presets; see
#: :func:`~osprey.cli.templates.claude_code._lint_write_tools_are_gated`.
WRITE_CAPABLE_BUILTINS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
)

#: The exact ``permissions.deny`` entry that blocks the agent's shell wholesale.
#: A *scoped* deny (``Bash(rm:*)``) constrains one command family and leaves the
#: shell otherwise usable, so only this literal counts as "Bash is denied".
BASH_DENY_ENTRY: str = "Bash"
