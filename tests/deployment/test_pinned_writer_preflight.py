"""``env.pinned`` may not name a variable the deploy writes itself.

A pin says one thing: this variable's value comes from the repo's env chain and
nowhere else. The deploy has provisioners that write into ``.env`` on the way
past — the service token mint, the non-secret service defaults, the two bluesky
provisioners, and the credential ``--reuse-stores`` restores from a surviving
data volume — and a pin on any of those names is a statement the same command
contradicts a few lines later.

Refusing the *declaration*, before anything reads or writes the chain, is what
makes pinned-and-minted impossible rather than a race: without it, the override
and export refusals would eventually fire on a value ``osprey up`` put there
itself, and the operator would be told to remove a line no human wrote.

The census is checked whole, not narrowed to the services this deploy enables,
so the answer is a property of the profile and cannot change with the config.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.container_lifecycle import (
    _deploy_written_env_vars,
    _preflight_pinned_writers,
)


def _repo(tmp_path: Path, *pinned: str) -> Path:
    """A deployment root whose profile pins the given names."""
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    entries = "".join(f"    - {name}\n" for name in pinned)
    body = f"name: proj\nenv:\n  pinned:\n{entries}" if pinned else "name: proj\n"
    (root / "profile.yml").write_text(body, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
        "BLUESKY_LAUNCH_TOKEN",
        "BLUESKY_TILED_API_KEY",
        "ZO_ROOT_USER_PASSWORD",
        "ARIEL_DB_PASSWORD",
        "MONGO_ROOT_PASSWORD",
    ],
)
def test_every_minted_token_is_in_the_census(name):
    assert name in _deploy_written_env_vars()


def test_the_non_secret_service_defaults_are_in_the_census():
    """Not a secret, but still written by the deploy — the pin is as unsatisfiable."""
    assert "ZO_ROOT_USER_EMAIL" in _deploy_written_env_vars()


def test_the_control_socket_keypair_is_in_the_census():
    """Minted outside the token map, because the two halves have to pair."""
    census = _deploy_written_env_vars()
    assert {"BLUESKY_QSERVER_ZMQ_PRIVATE_KEY", "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY"} <= census


def test_the_bluesky_substrate_names_are_in_the_census():
    """Derived from the built project, then appended — a writer like any other."""
    census = _deploy_written_env_vars()
    assert {
        "BLUESKY_EPICS_SUBSTRATE",
        "BLUESKY_EPICS_SETPOINTS",
        "BLUESKY_EPICS_READBACKS",
    } <= census


def test_the_census_covers_every_registered_writer():
    """Read off the maps themselves, so a new entry cannot be added unpoliced."""
    census = _deploy_written_env_vars()
    for names in container_lifecycle._SERVICE_TOKEN_VARS.values():
        assert set(names) <= census
    for pairs in container_lifecycle._SERVICE_DEFAULT_VARS.values():
        assert {var for var, _default in pairs} <= census
    # The --reuse-stores adopter restores exactly these into `.env`.
    assert set(container_lifecycle._VOLUME_INITIALIZED_VARS) <= census


def test_a_name_no_writer_touches_is_not_in_the_census():
    assert "FACILITY_UPSTREAM_URL" not in _deploy_written_env_vars()


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_a_pin_on_a_minted_token_refuses(tmp_path, caplog):
    repo = _repo(tmp_path, "ARIEL_DB_PASSWORD")

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as excinfo:
        _preflight_pinned_writers(repo)

    assert "unpin ARIEL_DB_PASSWORD; it is machine-minted." in caplog.text
    assert "unpin ARIEL_DB_PASSWORD; it is machine-minted." in str(excinfo.value)


def test_the_remedy_names_every_offender(tmp_path, caplog):
    repo = _repo(tmp_path, "ARIEL_DB_PASSWORD", "ZO_ROOT_USER_EMAIL", "FACILITY_UPSTREAM_URL")

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as excinfo:
        _preflight_pinned_writers(repo)

    assert "unpin ARIEL_DB_PASSWORD; it is machine-minted." in str(excinfo.value)
    assert "unpin ZO_ROOT_USER_EMAIL; it is machine-minted." in str(excinfo.value)
    assert "FACILITY_UPSTREAM_URL" not in str(excinfo.value)


def test_a_pin_on_a_name_no_writer_touches_is_allowed(tmp_path):
    """The whole point of the feature must survive the guard on it."""
    assert _preflight_pinned_writers(_repo(tmp_path, "FACILITY_UPSTREAM_URL")) == []


def test_a_repo_that_declares_nothing_is_silent(tmp_path):
    assert _preflight_pinned_writers(_repo(tmp_path)) == []


def test_a_repo_with_no_profile_is_silent(tmp_path):
    """Same "nothing declared" answer the reader gives for every broken shape."""
    assert _preflight_pinned_writers(tmp_path) == []


def test_the_answer_does_not_depend_on_the_deployed_services(tmp_path, caplog):
    """A static declaration is checked statically.

    Narrowing the census to the services enabled today would hide the
    contradiction until the deploy that first turns one on — which is the deploy
    least able to absorb a surprise refusal.
    """
    repo = _repo(tmp_path, "MONGO_ROOT_PASSWORD")

    # No config is passed at all: nothing about this check can consult one.
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        _preflight_pinned_writers(repo)


# ---------------------------------------------------------------------------
# It is on the deploy path, before anything is provisioned
# ---------------------------------------------------------------------------


def test_the_start_sequence_refuses_before_the_mint(tmp_path, monkeypatch):
    """First of the pin checks, and ahead of every writer it is about."""
    repo = _repo(tmp_path, "ARIEL_DB_PASSWORD")
    services = repo / "build" / "services"
    services.mkdir(parents=True, exist_ok=True)
    (services / "docker-compose.0.yml").write_text(
        "services:\n  worker:\n    image: proj:local\n", encoding="utf-8"
    )
    (repo / ".env").write_text("A=x\n", encoding="utf-8")

    order: list[str] = []
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle,
        "_ensure_service_tokens",
        lambda config, expose, env_path: order.append("mint") or set(),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda config, dev, env, ctx=None: order.append("image"),
    )

    def _fake_run(cmd, *args, **kwargs):
        order.append("subprocess")
        return subprocess.CompletedProcess(list(cmd), 0, stdout="", stderr="")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="machine-minted"):
        container_lifecycle._start_stack(
            {"project_name": "proj", "deployed_services": ["postgresql"]},
            ["build/services/docker-compose.0.yml"],
            repo,
            detached=True,
            env_path=repo / ".env",
        )

    assert order == [], "the deploy provisioned something despite the refusal"
