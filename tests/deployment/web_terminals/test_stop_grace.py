"""A render whose project shows a panel that is slow to stop gets the time to stop.

A sidecar panel says in the registry how many seconds its stop waits between
the stop signal and the kill (``SidecarPanel.stop_grace_s``): its own ``stop``
hands that number to the process stop as the grace. The terminal stops its
sidecars before it stops itself, so the compose service of every user whose
project shows such a panel needs the slowest panel's grace plus the terminal's
own stop (:data:`WEB_TERMINAL_STOP_S`), and renders ``stop_grace_period`` set
to that sum whenever it exceeds compose's own default
(:data:`COMPOSE_DEFAULT_STOP_GRACE_S`). The notebooks panel's five seconds fit
inside the default, so a render that shows only it carries no line.

The tests register synthetic panels and never name a shipped one other than
``jupyter``, so they hold unchanged in a tree that gives a shipped panel a
longer grace. The fact travels the way every per-render fact does: read off
each render's rendered ``config.yml`` by :func:`render_stop_grace`, handed to
the render by :func:`resolve_render_inputs`, and applied per service; a roster
entry naming no tier runs ``default_tier`` and gets its period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.deployment.web_terminals.roster import (
    COMPOSE_DEFAULT_STOP_GRACE_S,
    WEB_TERMINAL_STOP_S,
    config_stop_grace,
    config_stop_period,
    render_stop_grace,
)
from osprey.profiles.sidecar_panels import SIDECAR_PANELS, SidecarPanel
from tests._renders import tier_render

#: Synthetic panel ids and their graces.
SLOW = "slowstop"
SLOWER = "slowerstop"
#: The compose periods those graces come to, once the terminal's own stop is added.
SLOW_PERIOD = 30 + WEB_TERMINAL_STOP_S
SLOWER_PERIOD = 45 + WEB_TERMINAL_STOP_S

_COMPOSE = "docker-compose.web.yml"


@pytest.fixture(autouse=True)
def slow_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register two sidecar panels needing 30 s and 45 s to stop, for every test."""
    monkeypatch.setitem(
        SIDECAR_PANELS, SLOW, SidecarPanel("Slow", "tests.synthetic:Slow", stop_grace_s=30)
    )
    monkeypatch.setitem(
        SIDECAR_PANELS, SLOWER, SidecarPanel("Slower", "tests.synthetic:Slower", stop_grace_s=45)
    )


def _shows(*panel_ids: str, enabled: bool = True) -> dict[str, Any]:
    """A project config that declares each of *panel_ids*."""
    return {"web": {"panels": {panel_id: {"enabled": enabled} for panel_id in panel_ids}}}


def _render_project(repo: Path, name: str, config: dict[str, Any]) -> str:
    """Write a render's rendered ``config.yml``; return its ``project_path``."""
    project = repo / "build" / name
    project.mkdir(parents=True)
    (project / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, **config}), encoding="utf-8"
    )
    return f"build/{name}"


def _deploy_config(repo: Path, readwrite: dict[str, Any], readonly: dict[str, Any]) -> dict:
    """A two-tier deployment: alice is ``readwrite``, bob ``readonly`` (the default)."""
    return {
        "project_name": "dls_controls",
        "facility": {"name": "Demo Light Source", "prefix": "dls"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "default_tier": "readonly",
                "renders": {
                    "readwrite": tier_render(
                        "readwrite",
                        "dls-readwrite",
                        project_path=_render_project(repo, "dls-readwrite", readwrite),
                    ),
                    "readonly": tier_render(
                        "readonly",
                        "dls-readonly",
                        project_path=_render_project(repo, "dls-readonly", readonly),
                    ),
                },
                "users": [
                    {"name": "alice", "index": 0, "tier": "readwrite"},
                    {"name": "bob", "index": 1, "tier": "readonly"},
                ],
            }
        },
    }


def _services(artifacts: dict[str, str]) -> dict[str, Any]:
    return yaml.safe_load(artifacts[_COMPOSE])["services"]


# ---------------------------------------------------------------------------
# The field and the predicates
# ---------------------------------------------------------------------------


def test_a_panel_waits_for_nothing_by_default() -> None:
    assert SidecarPanel("x", "a:b").stop_grace_s == 0
    assert SIDECAR_PANELS[SLOW].stop_grace_s == 30


def test_a_declared_panel_sets_the_grace() -> None:
    assert config_stop_grace(_shows(SLOW)) == 30


def test_the_largest_declared_grace_wins() -> None:
    assert config_stop_grace(_shows(SLOW, SLOWER, "jupyter")) == 45


def test_a_panel_switched_off_sets_no_grace() -> None:
    """The build switches off the block of a tab the profile does not select."""
    assert config_stop_grace(_shows(SLOW, SLOWER, enabled=False)) == 0


def test_no_declared_slow_panel_means_no_grace() -> None:
    assert config_stop_grace(_shows("events")) == 0
    assert config_stop_grace({}) == 0
    assert config_stop_grace(None) == 0


def test_the_period_is_the_grace_plus_the_terminals_own_stop() -> None:
    assert config_stop_period(_shows(SLOW)) == SLOW_PERIOD
    assert config_stop_period(_shows(SLOW, SLOWER)) == SLOWER_PERIOD


def test_a_period_within_composes_default_leaves_the_default() -> None:
    """The notebooks panel's grace plus the terminal's stop fits the default."""
    assert SIDECAR_PANELS["jupyter"].stop_grace_s + WEB_TERMINAL_STOP_S <= (
        COMPOSE_DEFAULT_STOP_GRACE_S
    )
    assert config_stop_period(_shows("jupyter")) == 0
    assert config_stop_period({}) == 0


# ---------------------------------------------------------------------------
# The per-render map
# ---------------------------------------------------------------------------


def test_render_stop_grace_reads_each_rendered_config(tmp_path: Path) -> None:
    config = _deploy_config(tmp_path, _shows(SLOW, SLOWER), _shows("jupyter"))

    assert render_stop_grace(config, tmp_path) == {"readwrite": SLOWER_PERIOD}


def test_an_unrendered_render_is_not_guessed_at(tmp_path: Path) -> None:
    config = _deploy_config(tmp_path, _shows(SLOW), _shows(SLOWER))
    (tmp_path / "build" / "dls-readonly" / "config.yml").unlink()

    assert render_stop_grace(config, tmp_path) == {"readwrite": SLOW_PERIOD}


# ---------------------------------------------------------------------------
# The render
# ---------------------------------------------------------------------------


def test_the_deploy_inputs_put_the_period_on_that_renders_service_only(tmp_path: Path) -> None:
    """Through the builder the deploy uses: one render's rendered config shows
    a slow panel, and exactly that render's service gets the period."""
    config = _deploy_config(tmp_path, _shows(SLOW), _shows("jupyter"))

    inputs = resolve_render_inputs(config, tmp_path)
    artifacts = render_web_terminals(config, **inputs)
    services = _services(artifacts)

    assert inputs["stop_grace_renders"] == {"readwrite": SLOW_PERIOD}
    assert services["web-alice"]["stop_grace_period"] == f"{SLOW_PERIOD}s"
    assert f"    stop_grace_period: {SLOW_PERIOD}s\n" in artifacts[_COMPOSE]
    assert [name for name, svc in services.items() if "stop_grace_period" in svc] == ["web-alice"]


def test_the_period_sits_in_the_service_block_after_init(tmp_path: Path) -> None:
    """The exact text, so the line cannot drift into another service's block."""
    config = _deploy_config(tmp_path, _shows(SLOW), _shows("jupyter"))

    compose = render_web_terminals(config, stop_grace_renders={"readwrite": 35})[_COMPOSE]

    assert compose.count("stop_grace_period:") == 1
    assert (
        "    container_name: dls_controls-web-alice\n"
        "    restart: unless-stopped\n"
        "    init: true\n"
        "    stop_grace_period: 35s\n"
        "    network_mode: host\n"
    ) in compose
    assert (
        "    container_name: dls_controls-web-bob\n"
        "    restart: unless-stopped\n"
        "    init: true\n"
        "    network_mode: host\n"
    ) in compose


def test_without_the_map_the_render_is_unchanged(tmp_path: Path) -> None:
    """A deployment with no slow render renders byte-for-byte as before the
    key existed: the line is the only thing the map adds."""
    config = _deploy_config(tmp_path, _shows(SLOW), _shows("jupyter"))

    plain = render_web_terminals(config)
    with_period = render_web_terminals(config, stop_grace_renders={"readwrite": 35})

    assert "stop_grace_period" not in plain[_COMPOSE]
    assert render_web_terminals(config, stop_grace_renders={}) == plain
    assert render_web_terminals(config, stop_grace_renders={"readwrite": 0}) == plain
    assert with_period[_COMPOSE].replace("    stop_grace_period: 35s\n", "") == plain[_COMPOSE]
    assert {k: v for k, v in with_period.items() if k != _COMPOSE} == {
        k: v for k, v in plain.items() if k != _COMPOSE
    }


def test_a_user_without_a_tier_runs_the_default_one(tmp_path: Path) -> None:
    """A roster entry naming no tier runs ``default_tier``, so that
    render's rendered config says which panels it shows."""
    config = _deploy_config(tmp_path, _shows(SLOW), _shows("jupyter"))
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["users"] = ["alice", "bob"]

    services = _services(render_web_terminals(config, **resolve_render_inputs(config, tmp_path)))
    assert all("stop_grace_period" not in service for service in services.values())

    web_terminals["default_tier"] = "readwrite"
    artifacts = render_web_terminals(config, **resolve_render_inputs(config, tmp_path))
    services = _services(artifacts)
    assert services["web-alice"]["stop_grace_period"] == f"{SLOW_PERIOD}s"
    assert services["web-bob"]["stop_grace_period"] == f"{SLOW_PERIOD}s"
    assert "stop_grace_period" not in services["nginx"]
    assert artifacts[_COMPOSE].count(f"    stop_grace_period: {SLOW_PERIOD}s\n") == 2


def test_the_scaffold_preview_carries_it_too(tmp_path: Path) -> None:
    """``osprey scaffold web-terminals render`` is the other caller of the
    render, and the compose file it writes gives the same period."""
    from click.testing import CliRunner

    from osprey.cli.scaffold_cmd import scaffold

    repo = tmp_path / "wt-repo"
    repo.mkdir()
    (repo / "profile.yml").write_text("name: demo\n", encoding="utf-8")
    config = _deploy_config(repo, _shows(SLOW), _shows("jupyter"))
    (repo / "build" / "config.yml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    out = tmp_path / "deploy"

    result = CliRunner().invoke(
        scaffold, ["web-terminals", "render", "--repo", str(repo), "--no-lint", "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    services = yaml.safe_load((out / _COMPOSE).read_text(encoding="utf-8"))["services"]
    assert [name for name, svc in services.items() if "stop_grace_period" in svc] == ["web-alice"]
    assert services["web-alice"]["stop_grace_period"] == f"{SLOW_PERIOD}s"
