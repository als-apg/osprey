"""The deploy titles each shared card by its login file's ``name:``.

The render opens no file, so a shared login's title arrives as the
``shared_titles`` input. :func:`resolve_render_inputs` is the deploy's one seam
between disk and render; it reads the titles off the repo's ``tiers/``
directory, so the landing card, the browser tab and the sign-in chooser all
read the login's own name and never fall back to the bare roster name.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.app import AuthSettings
from osprey.services.auth_sidecar.roster_env import ROSTER_LABEL_VAR_PREFIX, env_var_suffix

PROJECT = "dls-profiles"


def _config() -> dict:
    """A built config: one person on a tier, one shared app."""
    return {
        "project_name": PROJECT,
        "facility": {"prefix": "dls"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": [{"name": "alice", "index": 0, "tier": "assistant"}],
                "shared": ["logbook"],
                "default_tier": "assistant",
                "renders": {
                    "assistant": {
                        "kind": "tier",
                        "project": f"{PROJECT}-assistant",
                        "project_path": f"build/{PROJECT}-assistant",
                    },
                    "logbook": {
                        "kind": "tier",
                        "project": f"{PROJECT}-logbook",
                        "project_path": f"build/{PROJECT}-logbook",
                    },
                },
                "auth": {"method": "password", "allow_insecure_http": True},
            }
        },
    }


def _write_app(repo: Path, name: str, body: str) -> None:
    apps = repo / "tiers"
    apps.mkdir(parents=True, exist_ok=True)
    (apps / f"{name}.yml").write_text(body, encoding="utf-8")


def _sidecar_env(artifacts: dict[str, str]) -> dict[str, str]:
    overlay = yaml.safe_load(artifacts["docker-compose.web.yml"])
    lines = overlay["services"]["auth"]["environment"]
    return dict(line.split("=", 1) for line in lines)


def test_the_deploy_reads_each_app_title_off_its_app_file(tmp_path: Path) -> None:
    _write_app(tmp_path, "logbook", "name: Logbook Research\n")

    inputs = resolve_render_inputs(_config(), tmp_path)

    assert inputs["shared_titles"] == {"logbook": "Logbook Research"}


def test_the_deployed_app_card_tab_and_chooser_carry_the_title(tmp_path: Path) -> None:
    _write_app(tmp_path, "logbook", "name: Logbook Research\n")
    config = _config()

    artifacts = render_web_terminals(config, **resolve_render_inputs(config, tmp_path))

    assert "Logbook Research" in artifacts["nginx/landing.html"]
    env = _sidecar_env(artifacts)
    label_var = f"{ROSTER_LABEL_VAR_PREFIX}{env_var_suffix('logbook')}"
    assert env[label_var] == "Logbook Research"
    assert AuthSettings.from_env(env).label("logbook") == "Logbook Research"
    assert AuthSettings.from_env(env).label("alice") == "alice"


def test_an_app_without_a_title_falls_back_to_its_name(tmp_path: Path) -> None:
    _write_app(tmp_path, "logbook", "description: no title here\n")

    inputs = resolve_render_inputs(_config(), tmp_path)

    assert inputs["shared_titles"] == {}


def test_a_repo_without_logins_has_no_titles(tmp_path: Path) -> None:
    assert resolve_render_inputs(_config(), tmp_path)["shared_titles"] == {}
