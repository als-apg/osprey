"""Tutorial-rot guard for the shipped ``tutorial_triggers.yml``.

The ``control-assistant`` preset ships a bundled set of demonstration triggers
that double as the event-dispatch tutorial *and* as the definitions the
real-token e2e tests fire (``tests/e2e/test_dispatch_tutorial.py`` /
``test_dispatch_deploy.py``). If someone edits the shipped file — renames a
trigger, drops the safety demo, changes an allowlist — the tutorial, the docs,
and those e2e tests silently drift apart.

This test loads the REAL shipped file (not an inline fixture) through the same
``load_triggers()`` parser the dispatcher uses at runtime and pins the exact
contract: which triggers exist, their tool allowlists, and the intentional
compose-service ``dispatch_target``. It costs nothing (no tokens, no network)
and runs in the blocking ``pytest tests/ --ignore=tests/e2e`` gate.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from osprey.dispatch.trigger_config import DispatcherConfig, TriggerConfig, load_triggers

# The package that holds the bundled trigger YAMLs (resolved the same way the
# CLI does in ``osprey.cli.build_profile._triggers_dir``).
_TRIGGERS_PACKAGE = "osprey.profiles.triggers"

# The exact contract the tutorial, docs, and e2e tests depend on. Keyed by
# trigger name → expected allowed_tools (order-independent).
_EXPECTED_TRIGGERS: dict[str, list[str]] = {
    "hello-dispatch": [],
    "triage-event": [],
    "save-report": [
        "Glob",
        "Read",
        "mcp__osprey_workspace__artifact_save",
        "mcp__osprey_workspace__create_document",
    ],
    "denied-tool-demo": ["WebFetch"],
}

# The dispatcher forwards to this compose service name inside the deploy
# network. It is intentionally NOT a localhost URL; the e2e subprocess test
# (no compose network) overrides it, while the full-stack deploy test relies on
# it resolving. See tutorial_triggers.yml.
_EXPECTED_DISPATCH_TARGET = "http://dispatch-worker-1:9190"


def _shipped_triggers_path() -> Path:
    return Path(str(importlib.resources.files(_TRIGGERS_PACKAGE))) / "tutorial_triggers.yml"


def _load() -> tuple[DispatcherConfig, list[TriggerConfig]]:
    return load_triggers(str(_shipped_triggers_path()))


def test_shipped_file_exists_and_parses() -> None:
    path = _shipped_triggers_path()
    assert path.is_file(), f"bundled tutorial_triggers.yml missing at {path}"
    dispatcher, triggers = _load()
    assert isinstance(dispatcher, DispatcherConfig)
    assert all(isinstance(t, TriggerConfig) for t in triggers)


def test_exactly_the_four_tutorial_triggers() -> None:
    """Pin the trigger set: exactly these four, no more, no fewer."""
    _, triggers = _load()
    names = [t.name for t in triggers]
    assert names == list(_EXPECTED_TRIGGERS), (
        f"shipped triggers drifted from the tutorial contract.\n"
        f"  expected (in order): {list(_EXPECTED_TRIGGERS)}\n"
        f"  found:               {names}"
    )


def test_every_trigger_is_a_webhook_with_a_prompt() -> None:
    _, triggers = _load()
    for t in triggers:
        assert t.source == "webhook", f"{t.name}: expected source 'webhook', got {t.source!r}"
        assert t.action.get("prompt", "").strip(), f"{t.name}: missing action.prompt"


def test_tool_allowlists_match_contract() -> None:
    """Each trigger exposes exactly its documented tool allowlist."""
    _, triggers = _load()
    by_name = {t.name: t for t in triggers}
    for name, expected_tools in _EXPECTED_TRIGGERS.items():
        got = by_name[name].action.get("allowed_tools", [])
        assert sorted(got) == sorted(expected_tools), (
            f"{name}: allowed_tools drifted — expected {expected_tools}, got {got}"
        )


def test_denied_tool_demo_requests_a_denylisted_tool() -> None:
    """The safety demo must actually request a server-denylisted tool (WebFetch)."""
    _, triggers = _load()
    denied = next(t for t in triggers if t.name == "denied-tool-demo")
    assert "WebFetch" in denied.action.get("allowed_tools", []), (
        "denied-tool-demo no longer requests WebFetch — it would no longer "
        "demonstrate the server-side denylist."
    )


def test_dispatch_target_is_the_compose_service_name() -> None:
    """The intentional in-network worker URL (overridden only by the e2e subprocess test)."""
    dispatcher, _ = _load()
    assert dispatcher.dispatch_target == _EXPECTED_DISPATCH_TARGET
