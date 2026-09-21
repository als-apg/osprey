"""Unit tests for the bluesky-web sidecar's operator secret in the token-mint map.

Mirrors ``test_bluesky_token_mint.py``, but for the ``bluesky_web``
deployed-service entry in ``_SERVICE_TOKEN_VARS`` (container_lifecycle.py).
The sidecar is gated by WebAuthMiddleware and its container declares
``OSPREY_TERMINAL_BIND_HOST``, so web_auth refuses startup on an empty secret
rather than minting one only the container log would ever see — this mint is
what makes the deployed panel reachable at all.

The second half of the module covers the credentials the sidecar holds for
OTHER people: the roster grant (one ``OSPREY_TERMINAL_SECRET_<SUFFIX>`` per
entitled web-terminal user) and the ``OSPREY_TERMINAL_ROSTER_OWNERS`` map that
says whose each one is. Both are rendered by ``osprey build``, so those tests
render a real project rather than reading the template's text: a map that
agreed with the template and disagreed with the grant beside it would name the
wrong operator on every panel action, and only the rendered file shows the two
together.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import prepare_compose_files
from osprey.interfaces.web_auth import ROSTER_SECRET_ENV_PREFIX


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'bluesky_web' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["bluesky_web"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(container_lifecycle, "run_captured", _fake_run)
    return captured


@pytest.fixture
def _clean_secret_env(monkeypatch):
    monkeypatch.delenv("OSPREY_TERMINAL_SECRET", raising=False)
    monkeypatch.delenv("ARIEL_DSN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    path = tmp_path / ".env"
    return parse_dotenv_file(path) if path.is_file() else {}


@pytest.mark.usefixtures("captured_argv")
def test_bluesky_web_deploy_mints_the_operator_secret(_clean_secret_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    secret = env.get("OSPREY_TERMINAL_SECRET")
    assert secret, "deploy_up did not mint OSPREY_TERMINAL_SECRET for a bluesky_web deploy"
    assert len(secret) >= 32, f"minted secret is implausibly short: {len(secret)} chars"


@pytest.mark.usefixtures("captured_argv")
def test_bluesky_web_existing_secret_is_preserved(_clean_secret_env, tmp_path):
    (tmp_path / ".env").write_text("OSPREY_TERMINAL_SECRET=operator-chose-this\n", encoding="utf-8")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("OSPREY_TERMINAL_SECRET") == "operator-chose-this", (
        "an operator-supplied secret must never be overwritten by the mint"
    )


def test_service_token_vars_map_includes_bluesky_web():
    assert container_lifecycle._SERVICE_TOKEN_VARS.get("bluesky_web") == ("OSPREY_TERMINAL_SECRET",)


def test_sidecar_compose_template_matches_the_mint():
    """The template names exactly the minted var, fail-closed, container-marked.

    Pins the template half of the contract the mint map is the other half of:
    ``${OSPREY_TERMINAL_SECRET}`` must carry NO ``:-`` default (an unset secret
    must stay empty so web_auth refuses startup instead of booting on a
    guessable value), and ``OSPREY_TERMINAL_BIND_HOST`` must be declared so
    that refusal actually fires in the container shape.
    """
    template = (
        Path(container_lifecycle.__file__).parent.parent
        / "templates"
        / "services"
        / "bluesky_web"
        / "docker-compose.yml.j2"
    ).read_text(encoding="utf-8")

    assert "OSPREY_TERMINAL_SECRET: ${OSPREY_TERMINAL_SECRET}" in template, (
        "the sidecar compose template must pass the minted secret through verbatim, "
        "with no :- default"
    )
    assert "${OSPREY_TERMINAL_SECRET:-" not in template, (
        "the secret must not grow a :- default -- empty must mean refuse, not a shared known value"
    )
    assert "OSPREY_TERMINAL_BIND_HOST" in template, (
        "the container must declare the bind-host marker so web_auth refuses an "
        "empty secret at startup instead of minting an unreachable one"
    )


# ---------------------------------------------------------------------------
# The roster grant and the owner map beside it
# ---------------------------------------------------------------------------

#: The variable the sidecar's gate reads the suffix→username map out of.
_OWNERS_ENV = "OSPREY_TERMINAL_ROSTER_OWNERS"


def _render_sidecar_env(project: Path, roster: dict | None, monkeypatch) -> dict[str, str]:
    """Render a bluesky_web deployment and return the sidecar's environment.

    A real ``prepare_compose_files`` render off a written config, the way the
    build performs it, rather than a hand-built Jinja context: the grant and
    the map are resolved by two helpers on the render path, and a context
    assembled here could hold a pairing the build never produces.

    Args:
        project: Empty directory to write the project into.
        roster: The ``modules.web_terminals`` block, or ``None`` for a
            deployment with no web terminals at all.
        monkeypatch: The render resolves the packaged service templates
            relative to the working directory, so it runs in ``project``.

    Returns:
        The ``bluesky-web`` service's ``environment`` mapping, as rendered.
    """
    config: dict = {
        "project_name": "roster-owners-fixture",
        "build_dir": str(project / "build"),
        "deployed_services": ["bluesky_web"],
        # The sidecar's entry as its build injector writes it, plus the bridge
        # port its BLUESKY_BRIDGE_URL is built from. Only the sidecar is
        # deployed: the bridge is a port number here, not a rendered service.
        "services": {
            "bluesky_web": {"path": "./services/bluesky_web", "port": 10071},
            "bluesky": {"port": 10080},
        },
        "system": {"timezone": "UTC"},
        # What a persona-less roster user's entitlement is answered by: the
        # deployment's own project declares the BLUESKY tab.
        "web": {"panels": {"bluesky": {"url": "http://localhost:10071"}}},
    }
    if roster is not None:
        config["modules"] = {"web_terminals": roster}
    config_path = project / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    _copy_service_templates(project)
    monkeypatch.chdir(project)
    _, compose_files = prepare_compose_files(str(config_path))

    sidecar = [path for path in compose_files if "bluesky_web" in str(path)]
    assert len(sidecar) == 1, f"expected one rendered bluesky_web compose file, got {compose_files}"
    rendered = yaml.safe_load(Path(sidecar[0]).read_text(encoding="utf-8"))
    return rendered["services"]["bluesky-web"]["environment"]


def _write_persona(project: Path, name: str, *, declares_panel: bool) -> dict:
    """Render one persona project whose config decides its users' entitlement."""
    persona_dir = project / "build" / f"demo-{name}"
    persona_dir.mkdir(parents=True, exist_ok=True)
    panels = {"bluesky": {"url": "http://localhost:10071"}} if declares_panel else {}
    (persona_dir / "config.yml").write_text(
        yaml.safe_dump({"web": {"panels": panels}}), encoding="utf-8"
    )
    return {"project_path": f"build/demo-{name}"}


def test_the_rendered_sidecar_names_the_owner_of_every_granted_secret(tmp_path, monkeypatch):
    """Each granted secret variable appears in the owner map, keyed by its own suffix.

    The gate admits any of the roster secrets and has to say WHO it admitted.
    It sees variable names and values, never the roster, so the render is the
    only place the two can be joined — and joining them anywhere else is how a
    panel action gets attributed to the wrong account.
    """
    roster = {
        "personas": {
            "operator": _write_persona(tmp_path, "operator", declares_panel=True),
            "viewer": _write_persona(tmp_path, "viewer", declares_panel=False),
        },
        "users": [
            {"name": "alice-b", "index": 0, "persona": "operator"},
            {"name": "bob", "index": 1, "persona": "viewer"},
            # No persona: runs the deploy config, which shows the tab.
            {"name": "carol", "index": 2},
        ],
    }

    environment = _render_sidecar_env(tmp_path, roster, monkeypatch)

    granted = sorted(
        name
        for name in environment
        if name.startswith(ROSTER_SECRET_ENV_PREFIX) and name != _OWNERS_ENV
    )
    assert granted == ["OSPREY_TERMINAL_SECRET_ALICE_B", "OSPREY_TERMINAL_SECRET_CAROL"], (
        "the grant must cover exactly the users whose project declares the BLUESKY tab"
    )

    owners = dict(pair.split("=", 1) for pair in environment[_OWNERS_ENV].split(",") if pair)
    assert owners == {"ALICE_B": "alice-b", "CAROL": "carol"}, (
        "every granted secret needs its owner named, and nobody else's"
    )
    assert sorted(ROSTER_SECRET_ENV_PREFIX + suffix for suffix in owners) == granted, (
        "a map key that is not a granted variable's suffix names an operator for "
        "a credential the sidecar was never handed"
    )


def test_a_deployment_with_no_entitled_user_renders_no_owner_map(tmp_path, monkeypatch):
    """No grant, no map. An empty map would be a variable the gate must then
    treat as 'nobody is named', which is what an ABSENT variable already says —
    and two spellings of one state is how the gate grows a branch nothing
    tests."""
    roster = {
        "personas": {"viewer": _write_persona(tmp_path, "viewer", declares_panel=False)},
        "users": [{"name": "bob", "index": 0, "persona": "viewer"}],
    }

    environment = _render_sidecar_env(tmp_path, roster, monkeypatch)

    assert not [name for name in environment if name.startswith(ROSTER_SECRET_ENV_PREFIX)], (
        "a deployment whose users all lack the tab must be handed no roster secret"
    )
    assert _OWNERS_ENV not in environment


def test_the_owner_map_is_not_a_second_derivation_of_the_username(tmp_path, monkeypatch):
    """The map keys are sliced off the granted variable names.

    ``alice-b`` keys ``ALICE_B``: uppercased with ``-`` folded to ``_``. The
    fold is lossy, so a map that re-derived the key from the username could
    drift from the variable the grant emits the moment that derivation
    changes. Pinned on the dashed name because it is the one where the two
    spellings differ at all.
    """
    roster = {
        "personas": {"operator": _write_persona(tmp_path, "operator", declares_panel=True)},
        "users": [{"name": "alice-b", "index": 0, "persona": "operator"}],
    }

    environment = _render_sidecar_env(tmp_path, roster, monkeypatch)

    assert environment[_OWNERS_ENV] == "ALICE_B=alice-b"
    assert "OSPREY_TERMINAL_SECRET_ALICE_B" in environment
