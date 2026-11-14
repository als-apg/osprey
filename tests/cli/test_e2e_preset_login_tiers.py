"""The login→tier maps the e2e stacks render name the identities their rows judge.

``test_dispatch_deploy`` O2 fires a job as ``FIRING_USER`` and expects its write
refused for that user's narrowing; if the dispatch worker's staged map does not
name the user, the owner rung answers ``unknown`` and the write is refused
``owner_unknown`` before the narrowing is read. ``test_single_user_narrowing_e2e``
judges the operator's own write the same way, with no roster at all — and there
the operator is the stamped ``deployment.host_identity`` the host terminal signs
in as, not the build account, whose name would leave the request unapprovable.

Each stack is rendered here from its own module's ``INIT_PRESET_ARGS`` through
one real in-process ``osprey init`` and ``osprey build``, so a preset or
set-list change that drops either identity from the map fails here, in seconds,
rather than an hour into a deploy.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import chdir, contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e import test_dispatch_deploy as dispatch_e2e
from tests.e2e import test_single_user_narrowing_e2e as single_user_e2e
from tests.fixtures.lifecycle_repo import preserved_environ

pytestmark = pytest.mark.slow

#: The pinned build account: roster-shaped, and no preset roster card's name.
BUILD_ACCOUNT = "builder"

#: The pinned host login the build stamps as ``deployment.host_identity`` (no
#: profile states one): roster-shaped, no card's or agent-record login's name, and
#: not the build account, so a single-user map that followed
#: ``OSPREY_AUDIT_IDENTITY`` would show.
HOST_LOGIN = "shiftlead"

#: The shared-container configs that carry the map, under ``build/``.
_STAGED = (
    Path("services/bluesky/config.yml"),
    Path("services/dispatch_worker/config.yml"),
)


@contextmanager
def _pinned_env() -> Iterator[None]:
    with preserved_environ():
        os.environ["OSPREY_AUDIT_IDENTITY"] = BUILD_ACCOUNT
        # ``getpass.getuser`` reads these first, in this order.
        for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
            os.environ[name] = HOST_LOGIN
        os.environ.pop("OSPREY_TERMINAL_USER", None)
        os.environ.pop("OSPREY_CONTROL_CONTEXT_TREE", None)
        yield


def _render(base: Path, project: str, init_args: list[str]) -> Path:
    """``osprey init <repo> *init_args`` then ``osprey build``, as the e2e fixture runs them."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command
    from osprey.cli.init_cmd import init as init_command

    repo = base / project
    with _pinned_env(), chdir(base):
        result = CliRunner().invoke(init_command, [str(repo), *init_args])
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(
            build_command,
            ["--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        )
        assert result.exit_code == 0, result.output
    return repo


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _staged_sections(repo: Path) -> dict[str, dict[str, Any]]:
    build = repo / "build"
    sections = {}
    for rel in _STAGED:
        path = build / rel
        assert path.is_file(), f"the build staged no {rel}; the e2e stack deploys that container"
        sections[str(rel)] = _load(path)["control_system"]
    return sections


@pytest.fixture(scope="module")
def dispatch_repo(tmp_path_factory) -> Path:
    return _render(
        tmp_path_factory.mktemp("e2e-dispatch"),
        dispatch_e2e.PROJECT_NAME,
        dispatch_e2e.INIT_PRESET_ARGS,
    )


@pytest.fixture(scope="module")
def single_user_repo(tmp_path_factory) -> Path:
    return _render(
        tmp_path_factory.mktemp("e2e-single-user"),
        single_user_e2e.PROJECT_NAME,
        single_user_e2e.INIT_PRESET_ARGS,
    )


def test_dispatch_deploy_maps_the_firing_user_at_an_arming_tier(dispatch_repo: Path) -> None:
    from osprey_connectors.posture_store import owner_tier_term

    user = dispatch_e2e.FIRING_USER
    for rel, section in _staged_sections(dispatch_repo).items():
        login_tiers = section.get("login_tiers")
        assert isinstance(login_tiers, dict), f"{rel} carries no control_system.login_tiers"
        assert login_tiers.get(user) == "asked", (
            f"{rel} maps {user!r} to {login_tiers.get(user)!r}: O2 needs the firing "
            f"user's own tier to arm writes, so its refusal is the narrowing, not "
            f"owner_unknown or the tier. Map: {login_tiers}"
        )
        with _pinned_env():
            assert owner_tier_term(section, user) == "asked"


def test_dispatch_worker_runs_the_tier_the_e2e_expects(dispatch_repo: Path) -> None:
    worker = _load(dispatch_repo / "build" / "services/dispatch_worker/config.yml")
    assert worker["control_system"]["writes"] == "asked", (
        f"the {dispatch_e2e.WORKER_TIER} dispatch worker's own render does not arm "
        "writes, so no write would ever reach the narrowing"
    )


def test_single_user_maps_the_host_identity_at_the_root_tier(single_user_repo: Path) -> None:
    from osprey_connectors.posture_store import owner_tier_term
    from osprey_connectors.types import tier_writes

    rendered = _load(single_user_repo / "build" / "config.yml")
    assert rendered["deployment"]["host_identity"] == HOST_LOGIN, (
        "the build did not stamp the host login as deployment.host_identity"
    )
    root = rendered["control_system"]
    assert tier_writes(root) == "asked"
    for rel, section in _staged_sections(single_user_repo).items():
        login_tiers = section.get("login_tiers")
        assert isinstance(login_tiers, dict), f"{rel} carries no control_system.login_tiers"
        assert login_tiers.get(HOST_LOGIN) == tier_writes(root), (
            f"{rel} does not map the host identity {HOST_LOGIN!r} at the root tier; "
            f"a single-user job the operator owns would be refused owner_unknown. "
            f"Map: {login_tiers}"
        )
        assert BUILD_ACCOUNT not in login_tiers, (
            f"{rel} maps the build account {BUILD_ACCOUNT!r} on a single-user deployment; "
            f"the operator signs in as {HOST_LOGIN!r}, so a request filed under the build "
            f"account is one no login can approve. Map: {login_tiers}"
        )
        assert "unknown" not in login_tiers
        with _pinned_env():
            assert owner_tier_term(section, HOST_LOGIN) == "asked"
