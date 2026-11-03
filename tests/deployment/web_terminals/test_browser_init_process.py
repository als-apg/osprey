"""Every web terminal runs under an init process.

Every image carries a headless browser: the artifact converter's Chromium is
baked into the image unconditionally. A browser leaves child processes behind
when it exits or is killed, and the terminal's own server, as pid 1, reaps none
of them, so the compose service of every user renders ``init: true`` — not
only the users whose render shows a particular panel. The nginx front door runs
no browser and carries no such line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.render import render_web_terminals
from tests._renders import tier_render

_COMPOSE = "docker-compose.web.yml"


def _render_project(repo: Path, name: str) -> str:
    """Write a render's rendered ``config.yml``; return its ``project_path``."""
    project = repo / "build" / name
    project.mkdir(parents=True)
    (project / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, "web": {"panels": {"jupyter": {}}}}),
        encoding="utf-8",
    )
    return f"build/{name}"


def _deploy_config(repo: Path) -> dict[str, Any]:
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
                        project_path=_render_project(repo, "dls-readwrite"),
                    ),
                    "readonly": tier_render(
                        "readonly",
                        "dls-readonly",
                        project_path=_render_project(repo, "dls-readonly"),
                    ),
                },
                "users": [
                    {"name": "alice", "index": 0, "tier": "readwrite"},
                    {"name": "bob", "index": 1},
                ],
            }
        },
    }


def _services(artifacts: dict[str, str]) -> dict[str, Any]:
    return yaml.safe_load(artifacts[_COMPOSE])["services"]


def test_every_terminal_runs_under_an_init_process(tmp_path: Path) -> None:
    """Through the builder the deploy uses: every user's service, whatever its
    tier and whatever panels its render shows, and not the nginx front door."""
    config = _deploy_config(tmp_path)

    services = _services(render_web_terminals(config, **resolve_render_inputs(config, tmp_path)))

    assert services["web-alice"]["init"] is True
    assert services["web-bob"]["init"] is True
    assert "init" not in services["nginx"]


def test_init_sits_beside_the_restart_policy(tmp_path: Path) -> None:
    """The exact text, so the line cannot drift into another service's block."""
    compose = render_web_terminals(_deploy_config(tmp_path))[_COMPOSE]

    assert compose.count("    init: true\n") == 2
    for user in ("alice", "bob"):
        assert (
            f"    container_name: dls_controls-web-{user}\n"
            "    restart: unless-stopped\n"
            "    init: true\n"
        ) in compose


def test_the_scaffold_preview_carries_it_too(tmp_path: Path) -> None:
    """``osprey scaffold web-terminals render`` is the other caller of the
    render, and the compose file it writes says the same about pid 1."""
    from click.testing import CliRunner

    from osprey.cli.scaffold_cmd import scaffold

    repo = tmp_path / "wt-repo"
    repo.mkdir()
    (repo / "profile.yml").write_text("name: demo\n", encoding="utf-8")
    config = _deploy_config(repo)
    (repo / "build" / "config.yml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    out = tmp_path / "deploy"

    result = CliRunner().invoke(
        scaffold, ["web-terminals", "render", "--repo", str(repo), "--no-lint", "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    services = yaml.safe_load((out / _COMPOSE).read_text(encoding="utf-8"))["services"]
    assert sorted(name for name, service in services.items() if service.get("init")) == [
        "web-alice",
        "web-bob",
    ]
