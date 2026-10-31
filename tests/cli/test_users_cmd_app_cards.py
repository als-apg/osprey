"""The per-person roster verbs refuse a shared app card.

A name in ``modules.web_terminals.shared`` is an app card, not a person: it has
no password, and it leaves the deployment by being dropped from ``shared:`` in
``profile.yml``. ``osprey users remove`` and ``osprey users passwd`` — and the
engines behind them, :func:`~osprey.deployment.web_terminals.lifecycle.decommission_user`
and :func:`~osprey.deployment.web_terminals.lifecycle.rotate_user_password` —
refuse such a name with one fixed message before any side effect: no prompt, no
engine call, no file, no volume, no ``.env`` line.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.users_cmd import users
from osprey.deployment.web_terminals import lifecycle, seeding
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    TERMINAL_SECRET_VAR_PREFIX,
)
from osprey_connectors.dotenv import ENV_LOCAL_FILENAME
from tests._text import flat

REFUSAL = (
    "logbook is a shared app card; remove it from modules.web_terminals.shared "
    "in profile.yml and rebuild"
)

#: A rendered ``build/config.yml``: two people and two shared app cards, behind
#: a password wall so ``passwd`` would otherwise reach its own work.
RENDERED_CONFIG = textwrap.dedent(
    """
    project_name: demo-project
    facility:
      prefix: dls
    system:
      timezone: UTC
    modules:
      web_terminals:
        enabled: true
        image_source: local
        nginx_port: 8080
        auth:
          method: password
        users:
          - name: alice
            index: 0
          - name: bob
            index: 1
        shared:
          - logbook
          - knowledge
    """
)

PROFILE = textwrap.dedent(
    """
    preset: control-assistant
    config:
      modules.web_terminals:
        enabled: true
        users:
          - name: alice
            index: 0
          - name: bob
            index: 1
        shared:
          - logbook
          - knowledge
    """
)

LOGBOOK_SECRET = f"{TERMINAL_SECRET_VAR_PREFIX}LOGBOOK"
ENV_TEXT = f"{LOGBOOK_SECRET}=keep-me\nOTHER=1\n"
ENV_AUTH_TEXT = "OSPREY_AUTH_PW_HASH_ALICE=x\n"


@pytest.fixture
def repo_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text(PROFILE, encoding="utf-8")
    (root / "build" / "config.yml").write_text(RENDERED_CONFIG, encoding="utf-8")
    (root / ENV_LOCAL_FILENAME).write_text(ENV_TEXT, encoding="utf-8")
    (root / AUTH_ENV_FILENAME).write_text(ENV_AUTH_TEXT, encoding="utf-8")
    root = root.resolve()
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def engine_calls(monkeypatch) -> dict[str, list]:
    """Record every engine call, so a refusal can be shown to have made none."""
    calls: dict[str, list] = {
        "decommission_user": [],
        "rotate_user_password": [],
        "prune_users": [],
        "seed_web_terminals": [],
    }

    def _record(name):
        def _call(*args, **kwargs):
            calls[name].append((args, kwargs))

        return _call

    for name in ("decommission_user", "rotate_user_password", "prune_users"):
        monkeypatch.setattr(lifecycle, name, _record(name))
    monkeypatch.setattr(seeding, "seed_web_terminals", _record("seed_web_terminals"))
    return calls


@pytest.fixture
def no_runtime(monkeypatch) -> list:
    """Fail loudly if an engine reaches the container runtime."""
    reached: list = []

    def _runtime(*args, **kwargs):
        reached.append(args)
        raise AssertionError("the runtime must not be reached for an app card")

    monkeypatch.setattr(lifecycle, "_require_running_runtime", _runtime)
    return reached


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# The engines
# ---------------------------------------------------------------------------


def test_decommission_user_refuses_an_app_card_before_the_runtime(repo_root, no_runtime):
    # Arrange
    before = _snapshot(repo_root)

    # Act
    with pytest.raises(ValueError) as excinfo:
        lifecycle.decommission_user(
            repo_root / "build" / "config.yml", "logbook", purge=True, assume_yes=True
        )

    # Assert
    assert str(excinfo.value) == REFUSAL
    assert no_runtime == []
    assert _snapshot(repo_root) == before


def test_decommission_user_still_reports_an_unknown_name_as_off_roster(repo_root, monkeypatch):
    # Arrange: the runtime gate comes first for a non-app name, so let it pass.
    monkeypatch.setattr(lifecycle, "_require_running_runtime", lambda config: None)

    # Act
    with pytest.raises(ValueError) as excinfo:
        lifecycle.decommission_user(repo_root / "build" / "config.yml", "mallory")

    # Assert
    assert "shared app card" not in str(excinfo.value)
    assert "not present" in str(excinfo.value)


def test_rotate_user_password_refuses_an_app_card(repo_root, no_runtime):
    # Arrange
    before = _snapshot(repo_root)

    # Act
    with pytest.raises(ValueError) as excinfo:
        lifecycle.rotate_user_password(repo_root / "build" / "config.yml", "logbook", "s3cret!")

    # Assert
    assert str(excinfo.value) == REFUSAL
    assert no_runtime == []
    assert _snapshot(repo_root) == before


def test_rotate_user_password_refuses_an_app_card_even_without_a_login_wall(repo_root, no_runtime):
    # Arrange: auth off — the app is still not a person, and that is what to say.
    config = repo_root / "build" / "config.yml"
    config.write_text(RENDERED_CONFIG.replace("method: password", "method: none"), "utf-8")

    # Act
    with pytest.raises(ValueError) as excinfo:
        lifecycle.rotate_user_password(config, "logbook", "s3cret!")

    # Assert
    assert str(excinfo.value) == REFUSAL
    assert no_runtime == []


def test_refusal_names_the_app_it_was_given(repo_root, no_runtime):
    # Act
    with pytest.raises(ValueError) as excinfo:
        lifecycle.decommission_user(repo_root / "build" / "config.yml", "knowledge")

    # Assert
    assert str(excinfo.value) == REFUSAL.replace("logbook", "knowledge")
    assert no_runtime == []


# ---------------------------------------------------------------------------
# The CLI verbs
# ---------------------------------------------------------------------------


def test_users_remove_refuses_an_app_card_and_touches_nothing(repo_root, engine_calls):
    # Arrange
    before = _snapshot(repo_root)

    # Act
    result = CliRunner().invoke(users, ["remove", "logbook", "--purge", "--yes"])

    # Assert
    assert result.exit_code != 0
    assert REFUSAL in flat(result.output)
    assert engine_calls["decommission_user"] == []
    assert _snapshot(repo_root) == before


def test_users_remove_refuses_an_app_card_listed_only_in_the_profile(repo_root, engine_calls):
    # Arrange: the resume path (off the rendered roster, on the profile's) must
    # not see an app name as a half-finished person removal.
    rendered = repo_root / "build" / "config.yml"
    rendered.write_text(RENDERED_CONFIG.replace("      - logbook\n", ""), "utf-8")
    before = _snapshot(repo_root)

    # Act
    result = CliRunner().invoke(users, ["remove", "logbook"])

    # Assert
    assert result.exit_code != 0
    assert REFUSAL in flat(result.output)
    assert engine_calls["decommission_user"] == []
    assert _snapshot(repo_root) == before


def test_users_passwd_refuses_an_app_card_before_prompting(repo_root, engine_calls):
    # Arrange
    before = _snapshot(repo_root)

    # Act: no input supplied — a prompt would abort with a different message.
    result = CliRunner().invoke(users, ["passwd", "logbook"])

    # Assert
    assert result.exit_code != 0
    assert REFUSAL in flat(result.output)
    assert "New web-terminal password" not in result.output
    assert engine_calls["rotate_user_password"] == []
    assert _snapshot(repo_root) == before


@pytest.mark.usefixtures("repo_root")
def test_users_remove_still_reaches_the_engine_for_a_person(engine_calls):
    # Act
    result = CliRunner().invoke(users, ["remove", "bob"])

    # Assert
    assert result.exit_code == 0, result.output
    assert len(engine_calls["decommission_user"]) == 1
    assert "shared app card" not in result.output


@pytest.mark.usefixtures("repo_root")
def test_users_passwd_still_prompts_for_a_person(engine_calls):
    # Act
    result = CliRunner().invoke(users, ["passwd", "alice"], input="pw-123456\npw-123456\n")

    # Assert
    assert result.exit_code == 0, result.output
    assert len(engine_calls["rotate_user_password"]) == 1
