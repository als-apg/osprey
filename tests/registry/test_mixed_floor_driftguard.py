"""Drift guards: the audit middleware's degraded-render floors, and the one
word three layers spell for a posture refusal, stay pinned to their sources of
truth.

``osprey.mcp_server.audit_middleware`` ships two hardcoded floor lists, used
only when a render cannot be trusted (missing/unreadable/malformed
``hook_config.json``):

- ``_FALLBACK_MIXED_TOOLS`` -- the read/write-mixed exception. Documented as
  always equal to the registry's ``MIXED_READ_WRITE_TEMPLATES``-derived
  matcher list (``registry.mcp.framework_mixed_read_write_tools()``).
- ``_FALLBACK_WRITE_TOOLS`` -- the pre-existing write-tool floor, documented
  as the same literal as the client-side hooks' shared floor,
  ``osprey_hook_log.FALLBACK_WRITE_TOOLS`` (which ``osprey_writes_check.py``
  and ``osprey_approval.py`` both read through ``write_tools()``). The two
  layers (server-side middleware, client-side hooks) must not disagree about
  what a degraded deployment refuses.

``tests/mcp_server/test_audit_middleware.py`` already asserts
``_FALLBACK_MIXED_TOOLS == framework_mixed_read_write_tools()`` and
``_FALLBACK_WRITE_TOOLS == <the hooks' shared literal>`` directly. This module is
the independent second guard the PLAN calls for: it re-derives the mixed
floor's expected value from ``FRAMEWORK_SERVERS``/``_WRITES_CHECK`` by hand,
without going through ``framework_mixed_read_write_tools()`` at all, so a bug
in that helper's own derivation cannot make both guards go green for the same
wrong reason. It also re-pins the write floor against the hooks' shared module
so this file alone is a complete driftguard for both floors, not just the
mixed one.

The write floor used to be pinned ONLY against the hooks' identical copy, and
that is how it went stale: it named ``controls`` and ``python``'s write-gated
tools but not bluesky's ``queue_add``/``queue_start``, on the argument that
bluesky is opt-in (``default_enabled=False``). That argument has it backwards
-- the deployments that can call ``queue_add`` are exactly the ones that opted
in -- and, worse, two identical copies pinned to each other agree happily while
both drift away from the registry, so the next framework write tool would be
stranded the same way. ``registry.mcp.framework_write_tools()`` is the derived
third answer both copies are now checked against here.

The posture-refusal reason is the same shape of problem in the record
vocabulary: the middleware, the executor's in-tool session clamp and the hook
all refuse a control-system write under the sandbox posture, and an operator
asking "what did the sandbox posture refuse for this user?" greps one word.
They must not answer in two dialects. The hook is pinned by AST for both
values, since it ships as a copied-in template file run by a bare ``python3``
and imports nothing from ``osprey``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from osprey.mcp_server import audit_middleware as am
from osprey.mcp_server.python_executor.tools import _execution_gates as gates
from osprey.registry.mcp import (
    _WRITES_CHECK,
    FRAMEWORK_SERVERS,
    MIXED_READ_WRITE_TEMPLATES,
    framework_write_tools,
)

_HOOKS_DIR = Path(__file__).resolve().parents[2] / "src/osprey/templates/claude_code/claude/hooks"
#: The kill-switch hook: home of the posture-refusal reason word.
_HOOK_TEMPLATE = _HOOKS_DIR / "osprey_writes_check.py"
#: The hooks' shared helpers: home of the write-tool floor both gates read.
_HOOK_LOG_TEMPLATE = _HOOKS_DIR / "osprey_hook_log.py"


def _mixed_matchers_from_framework_servers() -> list[str]:
    """Fully-qualified ``_WRITES_CHECK`` matchers of every
    ``MIXED_READ_WRITE_TEMPLATES`` entry, built directly from
    ``FRAMEWORK_SERVERS``/``_WRITES_CHECK`` -- deliberately NOT by calling
    ``registry.mcp.framework_mixed_read_write_tools()``, so this guard does
    not share a single point of failure with the one in
    ``test_audit_middleware.py`` that does call it.

    Mirrors ``framework_mixed_read_write_tools()``'s own construction
    (sorted template names, hook order within a template, identity check on
    the shared ``_WRITES_CHECK`` singleton) without importing it.
    """
    matchers: list[str] = []
    for template in sorted(MIXED_READ_WRITE_TEMPLATES):
        template_def = FRAMEWORK_SERVERS[template]
        for rule in template_def.hooks_pre:
            if _WRITES_CHECK in rule.hooks:
                matchers.append(rule.matcher)
    return matchers


def _hook_literal(name: str, template: Path = _HOOK_TEMPLATE):
    """One module-level literal out of a hook template file, by AST.

    Never by import: the hooks ship as copied-in project template files run
    by a bare ``python3``, and importing one here would both pull in its
    ``sys.path`` games and hide the very drift this module exists to catch.
    """
    tree = ast.parse(template.read_text())
    return next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


def _write_floor_literal_from_hook() -> list[str]:
    """``osprey_hook_log.py``'s ``FALLBACK_WRITE_TOOLS`` literal, the floor
    ``osprey_writes_check.py`` and ``osprey_approval.py`` share."""
    return _hook_literal("FALLBACK_WRITE_TOOLS", _HOOK_LOG_TEMPLATE)


class TestMixedFloorDriftGuard:
    """Registry growth cannot silently strand the hook/middleware mixed floor."""

    def test_fallback_mixed_tools_matches_framework_servers_writes_check_matchers(self):
        expected = _mixed_matchers_from_framework_servers()
        assert expected, (
            "MIXED_READ_WRITE_TEMPLATES resolved to no _WRITES_CHECK matchers at "
            "all -- this guard would be vacuous"
        )
        assert am._FALLBACK_MIXED_TOOLS == expected

    def test_fallback_mixed_tools_names_are_all_fully_qualified(self):
        assert am._FALLBACK_MIXED_TOOLS
        for tool in am._FALLBACK_MIXED_TOOLS:
            assert tool.startswith("mcp__"), f"{tool!r} is not a fully-qualified tool name"


class TestWriteFloorDriftGuard:
    """The server-side (middleware) and client-side (hook) write floors must
    never disagree about what a degraded deployment refuses."""

    def test_fallback_write_tools_matches_the_hooks_own_floor(self):
        assert am._FALLBACK_WRITE_TOOLS == _write_floor_literal_from_hook()

    def test_the_write_floor_covers_every_framework_write_tool(self):
        """Registry growth cannot strand the floor.

        Equality rather than a superset: the floor is the framework's own
        write-gated set, and a floor *wider* than the registry would clamp a
        name no framework server serves — bounded over-refusal that nothing
        would ever explain to an operator. A deployment's own tools are not in
        it by design; those ride on the loaded ``write_tools``.
        """
        expected = framework_write_tools()
        assert expected, "the registry resolved no write-gated matchers -- this guard is vacuous"
        assert set(am._FALLBACK_WRITE_TOOLS) == set(expected)

    def test_the_write_floor_covers_the_bluesky_arming_pair(self):
        """The regression this guard was added for, named so it reads in a diff."""
        assert {"mcp__bluesky__queue_add", "mcp__bluesky__queue_start"} <= set(
            am._FALLBACK_WRITE_TOOLS
        )

    def test_fallback_mixed_tools_is_a_subset_of_fallback_write_tools(self):
        """The degraded clamp is computed as ``_FALLBACK_WRITE_TOOLS`` MINUS
        ``_FALLBACK_MIXED_TOOLS`` (see ``_FLOOR_CLAMP``); a mixed tool absent
        from the write-tool floor would silently no-op out of that
        subtraction instead of narrowing it."""
        assert set(am._FALLBACK_MIXED_TOOLS) <= set(am._FALLBACK_WRITE_TOOLS)


class TestThePostureRefusalIsOneWord:
    """Three layers refuse a control-system write under the sandbox posture.

    The MCP middleware (server-side, on every ``tools/call``), the python
    executor's in-tool session clamp (inner, on the ``executor`` surface) and
    the ``osprey_writes_check.py`` PreToolUse hook (client-side, fail-open).
    An operator answering "what did the sandbox posture refuse for this user?"
    runs one grep over ``var/audit/<identity>/*.jsonl``; two spellings means
    that grep silently returns whichever layer happened to fire.
    """

    def test_the_middleware_and_the_executor_agree(self):
        assert am.REASON_POSTURE == gates.REASON_POSTURE

    def test_the_hook_agrees_with_both(self):
        assert _hook_literal("_POSTURE_DENY_AUDIT_REASON") == am.REASON_POSTURE

    def test_the_word_is_posture(self):
        """Pinned to the literal too, so all three cannot drift together."""
        assert am.REASON_POSTURE == "posture"

    def test_the_posture_reason_is_not_the_kill_switch_reason(self):
        """A different action lifts each, so they stay different words."""
        assert _hook_literal("_WRITES_DISABLED_AUDIT_REASON") != am.REASON_POSTURE
