"""The build's floor under a deployment's safety posture.

With the app templates gone, every declarative default a deployment runs on is
spelled in the file its operator edits. That is the point of the change — and
it is also the one way it can go wrong: a key the presets used to supply from
underneath, dropped from a ``profile.yml``, no longer becomes a template
default. It becomes silence, and each of these keys has a reader that answers
something when it is not told anything.

So the keys whose silence flips a posture are required outright, and a build
that cannot read one is refused naming it. Five keys, no more: the posture floor
is a *gate*, never a writer of defaults. It reads the render and writes
nothing, so a deployment that states its posture builds exactly the config it
built before this module existed.

Which five depends on what the deployment actually runs, because a key no
surface reads is not a posture at all:

* ``control_system.type`` and ``archiver.type`` — required when the ``controls``
  MCP server resolves enabled. That server is the one that reads channels,
  writes them and queries history; without a type, the connector and archiver
  factories fall back to a default backend, which is how a deployment ends up
  talking to a mock and reporting it as the machine. A standalone that switches
  the server off (the ARIEL and channel-finder presets do) is asked for
  neither: it has no control system to state a type for.
* ``approval.enabled`` and ``approval.default_policy`` — required when the
  ``approval`` hook is among the profile's selected hooks. The hook is the
  human gate in front of every write, and it reads both keys to decide whether
  to prompt and what to do about a tool nothing named. Selected, but unable to
  read its own posture, it is a gate whose behaviour nobody stated.
* ``hooks.debug`` — required unconditionally. It exists in every deployment
  (the hook chain is not optional), it is read at run time, and it is exactly
  the sort of key an operator wants to see the answer to in the file rather
  than infer from a reader's fallback.

The gate runs over the *rendered* config, beside
:func:`~osprey.cli.build_cmd._incomplete_limits_errors` and for the same
reason: what a deployment runs is the render, so a key an injector wrote, a
render inherited, or a preset's ``config:`` block spelled is read here the
same way an operator's own line is. Each refusal names the dotted key and the
file to add it to.

Beside the floor sits one check on who asks. The guarded tools — ``execute``,
``execute_file`` and ``pyaml_measure`` — each run a read-write call under the
target's run lock, and the executor refuses such a call before it is spawned
unless the approval hook recorded it whenever the tool asks.
:func:`effective_policy` and :func:`ask_capable` are the one reading that the
build and the executor share. A governed tool
(:func:`~osprey.registry.mcp.governed_write_tools`) is decided by the write
switch's position, so it counts as asking whatever ``approval.*`` says; any
other tool asks by its policy. The build guarantees that a render which asks
selects the hook that asks: a tool put to a human in a profile whose ``hooks:``
list lacks ``approval`` stops the build. So asks ⇒ hook selected, and
``config.yml`` alone answers whether a call of a guarded tool is asked.

:func:`check_code_review` is the same kind of read-only gate over the
``code_review`` section.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "APPROVAL_HOOK",
    "CODE_REVIEW_HOOKS",
    "CONTROLS_SERVER",
    "GUARDED_TOOLS",
    "REQUIRED_ALWAYS",
    "REQUIRED_WITH_APPROVAL_HOOK",
    "REQUIRED_WITH_CONTROLS",
    "STAGE1_TOOLS",
    "approval_enabled_fault",
    "ask_capable",
    "check_approval_enabled",
    "check_approval_hook_selected",
    "check_code_review",
    "effective_policy",
    "missing_posture_errors",
]

#: The file the refusal tells an operator to edit. The profile is the whole
#: declarative input; the render is generated and editing it is not a fix.
PROFILE_FILENAME = "profile.yml"

#: Registry name of the MCP server whose presence makes a control system and an
#: archiver something this deployment has.
CONTROLS_SERVER = "controls"

#: Artifact name of the approval hook, as a profile's ``hooks:`` list spells it.
APPROVAL_HOOK = "approval"

#: The hooks an enabled ``code_review`` section needs: the one that reviews a
#: notebook edit, and the one that gates where the edit may land.
CODE_REVIEW_HOOKS: tuple[str, ...] = ("code-review", "memory-guard")

#: The tools whose read-write call runs under the target's run lock and the
#: executor's approved-call check.
GUARDED_TOOLS: tuple[str, ...] = ("execute", "execute_file", "pyaml_measure")

#: The tools that run only on the approval prompt's record when their policy
#: asks (:func:`osprey.mcp_server.approval_record.require_approval_record`),
#: each with the registry name of the MCP server that serves it.
_STAGE1_SERVERS: Mapping[str, str] = {
    "channel_write": "controls",
    "queue_start": "bluesky",
    "queue_add": "bluesky",
    "phoebus_drive": "phoebus",
}
STAGE1_TOOLS: tuple[str, ...] = tuple(_STAGE1_SERVERS)

#: The approval policy that lets a call run with no prompt.
SKIP_POLICY = "skip"

#: What the approval hook does when no policy is stated at all.
HOOK_FALLBACK_POLICY = "always"

#: Required when :data:`CONTROLS_SERVER` resolves enabled, with why.
REQUIRED_WITH_CONTROLS: tuple[tuple[str, str], ...] = (
    (
        "control_system.type",
        "the controls server is enabled, and an unstated type leaves the connector "
        "factory to choose the backend this deployment talks to",
    ),
    (
        "archiver.type",
        "the controls server is enabled, and an unstated type leaves the archiver "
        "factory to choose the history this deployment reports",
    ),
)

#: Required when :data:`APPROVAL_HOOK` is among the selected hooks, with why.
REQUIRED_WITH_APPROVAL_HOOK: tuple[tuple[str, str], ...] = (
    (
        "approval.enabled",
        "the approval hook is selected, and this key is whether it prompts at all",
    ),
    (
        "approval.default_policy",
        "the approval hook is selected, and this is what it does about a tool no "
        "`approval.tools.<tool>` entry names",
    ),
)

#: Required of every deployment, with why. The surface exists in all of them.
REQUIRED_ALWAYS: tuple[tuple[str, str], ...] = (
    (
        "hooks.debug",
        "every deployment's hook chain either logs its decisions or does not, and a "
        "reader's fallback is not where an operator should have to find out which",
    ),
)


def _stated(rendered: Mapping[str, Any], dotted: str) -> bool:
    """Whether *rendered* answers *dotted* with something.

    Present-and-``None`` is not an answer: a bare ``hooks:\\n  debug:`` states
    the key and says nothing, which is the shape an operator gets from deleting
    a value and leaving its line, and every reader of these keys treats it as
    the silence it is. A blank or whitespace-only string (``type: ""``) is the
    same silence spelled with quotes instead of an absent value, so it is
    refused the same way.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.
        dotted: The dotted key to look up.

    Returns:
        ``True`` when every segment resolves through a mapping and the leaf is
        neither ``None`` nor a blank string.
    """
    node: Any = rendered
    for segment in dotted.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            return False
        node = node[segment]
    if node is None:
        return False
    return not (isinstance(node, str) and not node.strip())


def _enabled_servers(rendered: Mapping[str, Any]) -> frozenset[str]:
    """The MCP servers that resolve enabled for this render, by registry name.

    Resolved rather than read: ``claude_code.servers.<name>.enabled`` is an
    override on a registry default, so a render that says nothing about a
    server still runs it when its default is on.
    :func:`~osprey.registry.mcp.resolve_servers` is the one place that rule
    lives, and it is what writes ``.mcp.json``, so asking it here means the gate
    and the deployment agree about which servers run.

    The context is empty on purpose. It carries the conditions of the
    *conditional* servers (the channel finder's pipeline, the graph store);
    the servers this module asks about declare none, so nothing in a context
    can change their answer, and building a real one would mean re-deriving a
    render's context from the render.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.

    Returns:
        The names of the enabled servers.
    """
    from osprey.registry.mcp import resolve_servers

    claude_code = rendered.get("claude_code")
    servers = resolve_servers(dict(claude_code) if isinstance(claude_code, Mapping) else {}, {})
    return frozenset(str(server.get("name")) for server in servers if server.get("enabled"))


def _controls_enabled(rendered: Mapping[str, Any]) -> bool:
    """Whether the ``controls`` MCP server resolves enabled for this render.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.

    Returns:
        ``True`` when the resolved server list carries an enabled ``controls``.
    """
    return CONTROLS_SERVER in _enabled_servers(rendered)


def missing_posture_errors(rendered: Mapping[str, Any], selected_hooks: Iterable[str]) -> list[str]:
    """Every posture key a render's own surfaces read but nothing states, named.

    See the module docstring for the five keys and the surface that makes each
    of them required. A key is required only when the thing that reads it is
    part of *this* deployment, so a standalone with no controls server is never
    asked for a control-system type and a profile that does not select the
    approval hook is never asked for an approval policy.

    Args:
        rendered: The rendered ``config.yml``, read after the injectors — the
            config a deployment actually loads, so a key a preset's ``config:``
            block, a render's delta or an injector supplied counts as stated.
        selected_hooks: The resolved profile's ``hooks:`` list. Passed rather
            than re-derived: the build has already merged the render's
            selection into it, and a second derivation could disagree with the
            hooks the render installs.

    Returns:
        One line per unstated key, in the order the module lists them, each
        naming the dotted key, the file to add it to, and the surface that
        makes it required. Empty for a render that states its posture.
    """
    hooks = set(selected_hooks)
    required: list[tuple[str, str]] = []
    if _controls_enabled(rendered):
        required.extend(REQUIRED_WITH_CONTROLS)
    if APPROVAL_HOOK in hooks:
        required.extend(REQUIRED_WITH_APPROVAL_HOOK)
    required.extend(REQUIRED_ALWAYS)

    return [
        f"{key} is not stated; add `config: {key}: <value>` to {PROFILE_FILENAME} — {why}"
        for key, why in required
        if not _stated(rendered, key)
    ]


def _approval_section(rendered: Mapping[str, Any]) -> Mapping[str, Any]:
    """The render's ``approval`` mapping, empty when it states none."""
    approval = rendered.get("approval")
    return approval if isinstance(approval, Mapping) else {}


#: The remedy for an approval switch that is not true or false.
_ENABLED_FIX = f"set approval.enabled to true or false in {PROFILE_FILENAME}"


def _enabled_detail(value: Any) -> str:
    """What is wrong with an ``approval.enabled`` that states *value*."""
    return f"approval.enabled is {value!r}, not true or false"


def approval_enabled_fault(rendered: Mapping[str, Any]) -> str | None:
    """The sentence that refuses a run under a stated approval switch that is not a bool.

    The run-time readers of ``config.yml`` (the approval hook and the Python
    executor) refuse with it, and the build stops on the same detail and fix,
    so a hand edit after the build is named the way the build names it. A
    stated ``approval`` that is not a mapping (a bare ``approval:`` line) is a
    fault too: it states the section and says nothing a reader can trust.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.

    Returns:
        ``None`` when the switch is unstated, ``true`` or ``false``; else the
        sentence naming the value and the fix.
    """
    if "approval" in rendered and not isinstance(rendered["approval"], Mapping):
        return f"approval is {rendered['approval']!r}, not a mapping; {_ENABLED_FIX} and rebuild"
    approval = _approval_section(rendered)
    if "enabled" not in approval or isinstance(approval["enabled"], bool):
        return None
    return f"{_enabled_detail(approval['enabled'])}; {_ENABLED_FIX} and rebuild"


def _policy_key(rendered: Mapping[str, Any], tool: str) -> str:
    """The dotted key that supplies *tool*'s approval policy in *rendered*.

    ``approval.tools.<tool>`` when the render states it, else
    ``approval.default_policy`` — the same fallback the approval hook applies.
    """
    tools = _approval_section(rendered).get("tools")
    if isinstance(tools, Mapping) and tools.get(tool) is not None:
        return f"approval.tools.{tool}"
    return "approval.default_policy"


def effective_policy(rendered: Mapping[str, Any], tool: str) -> str:
    """The approval policy the hook applies to *tool* under *rendered*.

    ``approval.tools.<tool>`` when stated, else ``approval.default_policy``,
    else the hook's own fail-closed fallback.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.
        tool: The tool's short name, as ``approval.tools`` keys it.

    Returns:
        The policy name: ``always``, ``skip`` or ``selective``.
    """
    approval = _approval_section(rendered)
    tools = approval.get("tools")
    policy = tools.get(tool) if isinstance(tools, Mapping) else None
    if policy is None:
        policy = approval.get("default_policy")
    return str(policy) if policy is not None else HOOK_FALLBACK_POLICY


@lru_cache(maxsize=8)
def _resolved_governed(claude_code: str) -> tuple[str, ...]:
    """The governed matchers of a serialized ``claude_code`` section.

    Cached because :func:`ask_capable` runs on every write of a running server:
    the section does not change under it, and resolving the servers logs. The
    context is empty for the reason :func:`_enabled_servers` gives.

    Args:
        claude_code: The section as canonical JSON, the hashable cache key.

    Returns:
        The render's governed matchers, in the producer's order.
    """
    from osprey.registry.mcp import governed_write_tools, resolve_servers

    return tuple(governed_write_tools(resolve_servers(json.loads(claude_code), {})))


def _governed_matchers(rendered: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The governed matchers of *rendered*, or ``None`` when they cannot be resolved.

    Never raises: a server's run-time check must answer under a bad server spec.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.

    Returns:
        The matchers, or ``None`` for a section the registry refuses.
    """
    claude_code = rendered.get("claude_code")
    section = dict(claude_code) if isinstance(claude_code, Mapping) else {}
    try:
        return _resolved_governed(json.dumps(section, sort_keys=True, default=str))
    except Exception:
        return None


def _short_name(matcher: str) -> str:
    """The short name of an ``mcp__<server>__<tool>`` *matcher*."""
    parts = matcher.split("__", 2)
    return parts[2] if len(parts) == 3 else matcher


def ask_capable(rendered: Mapping[str, Any], tool: str) -> bool:
    """Whether a call of *tool* is put to a human before it runs.

    A governed tool is, whatever ``approval.*`` says: the switch's position
    decides for it, and the run-time check must keep requiring the record the
    hook leaves. So is any tool when the governed set cannot be resolved, which
    fails closed. Membership is by short name, so an ``extends`` clone's tool
    counts; a whole-server matcher names no tool and matches none here.

    Any other tool asks when ``approval.enabled`` is ``true`` or unstated (the
    hook reads an unstated switch as on) and its effective policy is not
    ``skip``. A switch that is not a bool, or an ``approval`` that is not a
    mapping, asks nobody: the approval hook and the executor refuse every call
    under it (:func:`approval_enabled_fault`). No hook term:
    :func:`check_approval_hook_selected` stops every build whose render asks
    without selecting the approval hook.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.
        tool: The tool's short name, as ``approval.tools`` keys it.

    Returns:
        ``True`` when a call of *tool* is put to a human.
    """
    governed = _governed_matchers(rendered)
    if governed is None or tool in map(_short_name, governed):
        return True
    if approval_enabled_fault(rendered) is not None:
        return False
    if _approval_section(rendered).get("enabled", True) is False:
        return False
    return effective_policy(rendered, tool) != SKIP_POLICY


def check_approval_enabled(rendered: Mapping[str, Any]) -> None:
    """Stop a render whose ``approval.enabled`` is stated and is not true or false.

    An empty line, a quoted ``"false"`` or a number is neither on nor off.
    With this stop the approval hook, :func:`ask_capable` and the executor see
    only a bool or an unstated key, which reads as asking; a value edited in
    after the build is refused at run time with the same detail and fix.

    Args:
        rendered: The rendered ``config.yml``, read after the injectors.

    Raises:
        FacilityBuildError: ``profile-invalid`` on ``approval.enabled``,
            naming the value it states.
    """
    approval = _approval_section(rendered)
    if "enabled" not in approval or isinstance(approval["enabled"], bool):
        return
    from osprey.facility.errors import FacilityBuildError

    raise FacilityBuildError(
        "profile-invalid",
        "approval.enabled",
        [PROFILE_FILENAME],
        _ENABLED_FIX,
        record_kind="path",
        detail=_enabled_detail(approval["enabled"]),
    )


def _no_hook_error(detail: str) -> Exception:
    """The build stop for a render that asks with no approval hook selected."""
    from osprey.facility.errors import FacilityBuildError

    return FacilityBuildError(
        "profile-invalid",
        "approval.enabled",
        [PROFILE_FILENAME],
        f"add `{APPROVAL_HOOK}` to `hooks:` in {PROFILE_FILENAME}",
        record_kind="path",
        detail=f"{detail} but the profile does not select the `{APPROVAL_HOOK}` hook, "
        "so nothing would ask",
    )


def check_approval_hook_selected(
    rendered: Mapping[str, Any], selected_hooks: Iterable[str]
) -> None:
    """Stop a render that puts a tool to a human with no hook to ask.

    Governed tools come first, because ``approval.*`` has no say over them: the
    first of :data:`GUARDED_TOOLS`, then of :data:`STAGE1_TOOLS`, then of the
    render's other governed tools (a facility's, a whole-server matcher) is
    named when :data:`APPROVAL_HOOK` is not among *selected_hooks*. Only an
    enabled server contributes one: a tool nobody can call asks nobody.

    After them, the first of the same two lists that is :func:`ask_capable` by
    its policy. A stage-1 tool counts only when the build enables the server
    that serves it.

    Args:
        rendered: The rendered ``config.yml``, read after the injectors.
        selected_hooks: The resolved profile's ``hooks:`` list.

    Raises:
        FacilityBuildError: ``profile-invalid`` on ``approval.enabled``, naming
            the tool and what puts it to a human.
    """
    if APPROVAL_HOOK in set(selected_hooks):
        return
    named = (*GUARDED_TOOLS, *STAGE1_TOOLS)
    governed = _governed_matchers(rendered) or ()
    shorts = {_short_name(matcher) for matcher in governed}
    offenders = [
        *(tool for tool in named if tool in shorts),
        *(matcher for matcher in governed if _short_name(matcher) not in named),
    ]
    if offenders:
        raise _no_hook_error(
            f"governed tool `{offenders[0]}` is put to a human by the write switch"
        )
    enabled: frozenset[str] | None = None
    for tool in named:
        if not ask_capable(rendered, tool):
            continue
        if tool in _STAGE1_SERVERS:
            if enabled is None:
                enabled = _enabled_servers(rendered)
            if _STAGE1_SERVERS[tool] not in enabled:
                continue
        raise _no_hook_error(
            f"guarded tool `{tool}` is put to a human under `{_policy_key(rendered, tool)}`"
        )


def check_code_review(
    rendered: Mapping[str, Any], selected_hooks: Iterable[str], repo_root: Path
) -> None:
    """Stop a render whose ``code_review`` section a session could not act on.

    A present section is validated whether or not it is enabled; a running
    session reads one that is off as off, so the build refuses a superset of what
    the session refuses. The hooks and the prompt are checked only when it is on.

    Args:
        rendered: The rendered ``config.yml``, read after the injectors.
        selected_hooks: The resolved profile's ``hooks:`` list.
        repo_root: Anchor of the prompt path and of the profile's source zone.

    Raises:
        FacilityBuildError: ``profile-invalid`` on the key at fault: a malformed
            section, a hook of :data:`CODE_REVIEW_HOOKS` not selected, or a prompt
            that is missing or lies where it is not protected from the agent.
    """
    if "code_review" not in rendered:
        return
    from osprey.facility.errors import FacilityBuildError
    from osprey.services.code_review.settings import (
        ReviewConfigError,
        parse_review_section,
        resolve_prompt,
    )

    try:
        settings = parse_review_section(rendered["code_review"])
    except ReviewConfigError as exc:
        raise FacilityBuildError(
            "profile-invalid",
            "code_review",
            [PROFILE_FILENAME],
            f"correct `code_review` under `config:` in {PROFILE_FILENAME}, or remove the section",
            record_kind="path",
            detail=str(exc),
        ) from exc
    if not settings.enabled:
        return

    selected = set(selected_hooks)
    missing = [f"`{hook}`" for hook in CODE_REVIEW_HOOKS if hook not in selected]
    if missing:
        hooks = " and ".join(missing)
        raise FacilityBuildError(
            "profile-invalid",
            "code_review.enabled",
            [PROFILE_FILENAME],
            f"add {hooks} to `hooks:` in {PROFILE_FILENAME}, or set "
            "`config: code_review.enabled: false`",
            record_kind="path",
            detail=f"code review is on but the profile does not select {hooks}",
        )

    try:
        resolve_prompt(settings, repo_root, rendered)
    except ReviewConfigError as exc:
        raise FacilityBuildError(
            "profile-invalid",
            "code_review.prompt",
            [PROFILE_FILENAME],
            f"point `code_review.prompt` in {PROFILE_FILENAME} at a readable file under "
            "the profile's source files, such as `data/`, outside the agent's data root",
            record_kind="path",
            detail=str(exc),
        ) from exc
