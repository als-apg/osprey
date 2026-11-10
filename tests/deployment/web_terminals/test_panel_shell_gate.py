"""The deploy-time shell gate on application panels.

A render whose rendered ``web.panels.<id>`` carries an application's
``launch_token_env`` or a write agent route arms that application; beside an open
shell its agent can reach the application's writes with no approval. The build
already refuses an ``allowed`` render with a lifted shell; this gate reads the
same conflict again from the shipped artifacts, and ``dangerously_allow_bash``
never waives it. The build cannot produce the conflicting pair, so every render
here is written by hand.
"""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals.artifacts import (
    BashLaunchTokenConflictError,
    bash_launch_token_offenders,
    check_bash_launch_token_conflict,
    dangerously_allowed_bash_renders,
    write_web_terminal_artifacts,
)
from osprey.deployment.web_terminals.provision import web_terminal_preflight_problems
from tests.deployment.web_terminals._panel_fixtures import _TOKEN, _panel_render, _roster

_WRITE_ROUTES = (("GET", "/state"), ("POST", "/set"))


def test_an_unarmed_render_with_the_panel_shown_and_a_shell_open_passes(tmp_path):
    config = _roster(tmp_path, {"ro": (_panel_render("never"), True)})

    result = check_bash_launch_token_conflict(config, tmp_path)

    assert result == {}
    assert result.by_token_env == {}
    assert bash_launch_token_offenders(config, tmp_path) == set()


def test_a_shared_render_carries_neither_and_passes(tmp_path):
    """A shared card's login gets the safe-method routes and no token."""
    config = _roster(tmp_path, {"shared": (_panel_render("allowed"), True)})

    result = check_bash_launch_token_conflict(config, tmp_path)

    assert result.by_token_env == {}


@pytest.mark.parametrize("writes", ["allowed", "asked"])
def test_a_render_holding_the_launch_token_beside_an_open_shell_refuses(tmp_path, writes):
    config = _roster(tmp_path, {"rw": (_panel_render(writes, token=True), True)})

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        check_bash_launch_token_conflict(config, tmp_path)

    assert excinfo.value.tiers == ["rw"]
    assert excinfo.value.applications == {"rw": {"knob": ("launch token",)}}
    message = str(excinfo.value)
    assert "'rw'" in message and "'knob'" in message and _TOKEN in message
    assert bash_launch_token_offenders(config, tmp_path) == {"rw"}


@pytest.mark.parametrize("writes", ["allowed", "asked"])
def test_a_render_holding_a_write_route_and_no_token_beside_an_open_shell_refuses(tmp_path, writes):
    config = _roster(tmp_path, {"rw": (_panel_render(writes, routes=_WRITE_ROUTES), True)})

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        check_bash_launch_token_conflict(config, tmp_path)

    assert excinfo.value.applications == {"rw": {"knob": ("write route",)}}
    assert "POST /set" in str(excinfo.value)


def test_the_refusal_stops_the_render_before_any_artifact_is_written(tmp_path):
    config = _roster(tmp_path, {"rw": (_panel_render("allowed", token=True), True)})

    with pytest.raises(BashLaunchTokenConflictError):
        write_web_terminal_artifacts(config, tmp_path)

    assert not (tmp_path / "build" / "docker-compose.web.yml").exists()


def test_the_waiver_never_reaches_an_application_render(tmp_path):
    """The key still waives the bluesky lane on the same roster; the application
    render is refused whatever it says, and the banner never names it."""
    bluesky = {
        "control_system": {"writes": "allowed"},
        "claude_code": {"servers": {"bluesky": {"enabled": True}}},
    }
    app = _panel_render("allowed", routes=_WRITE_ROUTES, token=True)
    lane_only = _roster(tmp_path / "lane", {"lane": (bluesky, True)})
    lane_only["dangerously_allow_bash"] = True
    both = _roster(tmp_path / "both", {"lane": (bluesky, True), "rw": (app, True)})
    both["dangerously_allow_bash"] = True

    assert check_bash_launch_token_conflict(lane_only, tmp_path / "lane") == {"bluesky": {"lane"}}
    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        check_bash_launch_token_conflict(both, tmp_path / "both")

    assert excinfo.value.tiers == ["rw"]
    assert excinfo.value.applications == {"rw": {"knob": ("launch token", "write route")}}
    assert bash_launch_token_offenders(both, tmp_path / "both") == {"rw"}
    assert dangerously_allowed_bash_renders(both, tmp_path / "both") == {"lane"}


def test_the_token_map_names_exactly_the_token_carrying_renders(tmp_path):
    config = _roster(
        tmp_path,
        {
            "rw": (_panel_render("allowed", routes=_WRITE_ROUTES, token=True), False),
            "admin": (_panel_render("allowed", token=True), False),
            "routes": (_panel_render("allowed", routes=_WRITE_ROUTES), False),
            "ro": (_panel_render("never"), True),
        },
    )

    result = check_bash_launch_token_conflict(config, tmp_path)

    assert result == {}
    assert result.by_token_env == {_TOKEN: {"rw", "admin"}}


def test_the_collected_preflight_report_names_the_application_under_the_waiver(tmp_path):
    config = _roster(tmp_path, {"rw": (_panel_render("allowed", routes=_WRITE_ROUTES), True)})
    config["dangerously_allow_bash"] = True

    problems = [
        problem for problem, _ in web_terminal_preflight_problems(config, repo_root=tmp_path)
    ]

    assert any("'knob'" in problem and "POST /set" in problem for problem in problems)
