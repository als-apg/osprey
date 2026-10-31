"""The ARIEL companion follows the build's own statement of whether there is a logbook.

The config template renders an ``ariel:`` section only when the ARIEL server is
on. A deployment without one -- the hello-world preset -- has no store the panel
could reach, so the panel must stay dark rather than open a session that can
only be the degraded one, announced as pool errors on the operator's terminal
before ``osprey chat`` hands it to the agent.
"""

from __future__ import annotations

from osprey.infrastructure import server_launcher
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

#: What ``osprey build`` writes for the hello-world preset: the ARIEL server is
#: off, so no ``ariel:`` section is rendered at all.
HELLO_WORLD_SHAPED = {
    "artifact_server": {"host": "127.0.0.1", "port": 10200, "auto_launch": True},
    "deployed_services": [],
    "web": {},
}

#: What it writes for a deployment whose ARIEL server is on.
LOGBOOK_SHAPED = {
    **HELLO_WORLD_SHAPED,
    "ariel": {"enhancement_modules": {"semantic_processor": {"provider": "anthropic"}}},
    "services": {"postgresql": {"port_host": 10800}},
}


def _auto_launch(config: dict, monkeypatch) -> bool:
    monkeypatch.setattr("osprey.infrastructure.server_launcher.load_osprey_config", lambda: config)
    return server_launcher._make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["ariel"])()


def test_the_ariel_companion_requires_its_section():
    assert FRAMEWORK_WEB_SERVERS["ariel"].require_section is True


def test_a_deployment_with_no_logbook_gets_no_ariel_companion(monkeypatch):
    assert _auto_launch(HELLO_WORLD_SHAPED, monkeypatch) is False


def test_a_deployment_with_a_logbook_still_gets_one(monkeypatch):
    assert _auto_launch(LOGBOOK_SHAPED, monkeypatch) is True


def test_its_own_switch_still_turns_it_off(monkeypatch):
    switched_off = {
        **LOGBOOK_SHAPED,
        "ariel": {**LOGBOOK_SHAPED["ariel"], "web": {"auto_launch": False}},
    }
    assert _auto_launch(switched_off, monkeypatch) is False
