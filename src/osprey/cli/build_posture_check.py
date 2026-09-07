"""The build's floor under a deployment's safety posture.

With the app templates gone, every declarative default a deployment runs on is
spelled in the file its operator edits. That is the point of the change — and
it is also the one way it can go wrong: a key the presets used to supply from
underneath, dropped from a ``profile.yml``, no longer becomes a template
default. It becomes silence, and each of these keys has a reader that answers
something when it is not told anything.

So the keys whose silence flips a posture are required outright, and a build
that cannot read one is refused naming it. Six keys, no more: the posture floor
is a *gate*, never a writer of defaults. It reads the render and writes
nothing, so a deployment that states its posture builds exactly the config it
built before this module existed.

Which six depends on what the deployment actually runs, because a key no
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
* ``claude_code.telemetry.enabled`` and ``hooks.debug`` — required
  unconditionally. Both exist in every deployment (the agent session and the
  hook chain are not optional), both are read at run time, and both are exactly
  the sort of key an operator wants to see the answer to in the file rather
  than infer from a reader's fallback.

The gate runs over the *rendered* config, beside
:func:`~osprey.cli.build_cmd._incomplete_limits_errors` and for the same
reason: what a deployment runs is the render, so a key an injector wrote, a
persona inherited, or a preset's ``config:`` block spelled is read here the
same way an operator's own line is. Each refusal names the dotted key and the
file to add it to.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "APPROVAL_HOOK",
    "CONTROLS_SERVER",
    "REQUIRED_ALWAYS",
    "REQUIRED_WITH_APPROVAL_HOOK",
    "REQUIRED_WITH_CONTROLS",
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

#: Required of every deployment, with why. Both surfaces exist in all of them.
REQUIRED_ALWAYS: tuple[tuple[str, str], ...] = (
    (
        "claude_code.telemetry.enabled",
        "every deployment either emits agent telemetry or does not, and a reader's "
        "fallback is not where an operator should have to find out which",
    ),
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
    if isinstance(node, str) and not node.strip():
        return False
    return True


def _controls_enabled(rendered: Mapping[str, Any]) -> bool:
    """Whether the ``controls`` MCP server resolves enabled for this render.

    Resolved rather than read: ``claude_code.servers.controls.enabled`` is an
    override on a registry default, so a render that says nothing about the
    server still runs it. :func:`~osprey.registry.mcp.resolve_servers` is the
    one place that rule lives, and it is what writes ``.mcp.json``, so asking
    it here means the gate and the deployment agree about which servers run.

    The context is empty on purpose. It carries the conditions of the
    *conditional* servers (the channel finder's pipeline, the graph store);
    ``controls`` declares none, so nothing in a context can change its answer,
    and building a real one would mean re-deriving a render's context from the
    render.

    Args:
        rendered: A rendered ``config.yml``, as ``safe_load`` produced it.

    Returns:
        ``True`` when the resolved server list carries an enabled ``controls``.
    """
    from osprey.registry.mcp import resolve_servers

    claude_code = rendered.get("claude_code")
    servers = resolve_servers(dict(claude_code) if isinstance(claude_code, Mapping) else {}, {})
    return any(
        server.get("name") == CONTROLS_SERVER and server.get("enabled") for server in servers
    )


def missing_posture_errors(rendered: Mapping[str, Any], selected_hooks: Iterable[str]) -> list[str]:
    """Every posture key a render's own surfaces read but nothing states, named.

    See the module docstring for the six keys and the surface that makes each
    of them required. A key is required only when the thing that reads it is
    part of *this* deployment, so a standalone with no controls server is never
    asked for a control-system type and a profile that does not select the
    approval hook is never asked for an approval policy.

    Args:
        rendered: The rendered ``config.yml``, read after the injectors — the
            config a deployment actually loads, so a key a preset's ``config:``
            block, a persona's delta or an injector supplied counts as stated.
        selected_hooks: The resolved profile's ``hooks:`` list. Passed rather
            than re-derived: the build has already merged the persona's
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
