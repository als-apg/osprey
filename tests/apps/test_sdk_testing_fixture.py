"""An application's own tests build a synthetic deployment, render it and run the backend on it.

Everything here goes through :mod:`osprey.apps.sdk.testing` alone, the way a test
suite in a deployment repository would: the application is a copy ``osprey app
new`` made, the facility is the SDK's own small tree, and the backend is served
in process against the render one login made of it.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from fastapi.testclient import TestClient

from osprey.apps.sdk.testing import (
    Deployment,
    RenderedTree,
    plain_tree,
    synthetic_deployment,
)
from tests.facility._fixture_kind import (
    FACTS,
    KIND,
    PACKAGE_DIR,
    PYPROJECT,
    SOURCE,
    fake_wheel,
    fixture_tree,
    install,
    read_entry_points_afresh,
)

APP = "probe-app"
PACKAGE = "probe_app"
CHANNEL = "Q1:RB"


def _new_app(parent: Path, name: str) -> Path:
    """A fresh application, as ``osprey app new`` writes it into a repository of its own."""
    from osprey.cli.app_cmd import app

    source = parent / "source"
    source.mkdir()
    (source / "profile.yml").write_text("name: source\n", encoding="utf-8")
    made = CliRunner().invoke(app, ["new", name, "--repo", str(source)])
    assert made.exit_code == 0, made.output
    return source / "apps" / name


def _importable(app_dir: Path, package: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the application's source on the path for the test, nothing of it imported yet."""
    monkeypatch.syspath_prepend(str(app_dir))
    for module in [name for name in sys.modules if name.split(".")[0] == package]:
        monkeypatch.delitem(sys.modules, module)


@pytest.fixture
def app_source(tmp_path: Path) -> Path:
    return _new_app(tmp_path, APP)


@pytest.fixture
def backend(app_source: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The application's backend module, imported from its own source."""
    _importable(app_source, PACKAGE, monkeypatch)
    yield importlib.import_module(f"{PACKAGE}.server")


def _deployment(tmp_path: Path, app_source: Path, **kwargs: Any) -> Deployment:
    return synthetic_deployment(
        tmp_path / "site",
        apps={APP: app_source},
        profile={"apps": {APP: {"config": {"channel": CHANNEL}}}},
        **kwargs,
    )


@pytest.fixture
def rendered(tmp_path: Path, app_source: Path) -> tuple[Deployment, RenderedTree]:
    deployment = _deployment(tmp_path, app_source, logins={"operator": "asked", "viewer": "never"})
    return deployment, deployment.build()


def test_the_build_renders_one_config_per_login_with_its_write_ceiling(rendered) -> None:
    _deployment, tree = rendered

    assert set(tree.configs) == {"operator", "viewer"}
    writes = {login: tree.config(login)["control_system"]["writes"] for login in tree.configs}
    assert writes == {"operator": "asked", "viewer": "never"}


def test_the_render_holds_the_facility_and_the_applications_skill(rendered) -> None:
    _deployment, tree = rendered

    assert {c["id"] for c in tree.facility["channels"]} >= {"Q1:SP", "Q1:RB", "BPM1:X"}
    assert set(tree.skills("operator")) == {APP}
    assert tree.skills("operator")[APP].read_text(encoding="utf-8").startswith("---")


@pytest.mark.no_auth_seam
def test_the_backend_answers_its_state_route_on_a_logins_render(rendered, backend) -> None:
    deployment, _tree = rendered
    served = deployment.serve("operator")

    with served.environment():
        app = backend.create_app()
        with TestClient(app) as client:
            answer = client.get("/api/state", headers=served.headers())

    assert answer.status_code == 200, answer.text
    state = answer.json()
    assert (state["channel"], state["target"], state["error"]) == (CHANNEL, "va", None)
    assert isinstance(state["readback"], float)


def test_the_served_environment_is_the_containers_with_the_logins_render(rendered) -> None:
    deployment, tree = rendered
    served = deployment.serve("operator")

    assert served.env["OSPREY_APP_NAME"] == APP
    assert served.env["OSPREY_CONFIG"] == str(tree.configs["operator"])
    assert yaml.safe_load(Path(served.env["OSPREY_APP_CONFIG"]).read_text("utf-8")) == {
        "channel": CHANNEL
    }
    assert Path(served.env["OSPREY_CONTROL_CONTEXT_TREE"]) == served.control_tree
    assert served.control_tree.is_dir()
    assert served.launch_token and served.env[served.launch_token_env] == served.launch_token
    assert all("${" not in value for value in served.env.values())


@pytest.mark.no_auth_seam
def test_a_backend_without_a_credential_is_refused(rendered, backend) -> None:
    deployment, _tree = rendered
    served = deployment.serve("operator")

    with served.environment():
        with TestClient(backend.create_app()) as client:
            refused = client.get("/api/state")

    assert refused.status_code == 401


def test_the_environment_is_put_back_when_the_block_ends(rendered, monkeypatch) -> None:
    deployment, _tree = rendered
    monkeypatch.setenv("OSPREY_APP_NAME", "before")
    monkeypatch.delenv("OSPREY_CONTROL_CONTEXT_TREE", raising=False)
    served = deployment.serve("operator")

    with served.environment():
        assert os.environ["OSPREY_APP_NAME"] == APP

    assert os.environ["OSPREY_APP_NAME"] == "before"
    assert "OSPREY_CONTROL_CONTEXT_TREE" not in os.environ


def test_a_login_the_deployment_does_not_have_is_named(rendered) -> None:
    deployment, _tree = rendered

    with pytest.raises(KeyError, match="no login 'admin'; the logins are operator, viewer"):
        deployment.serve("admin")


def test_the_facility_is_the_tree_the_caller_hands_in(tmp_path: Path, app_source: Path) -> None:
    tree = plain_tree()
    tree["records/channels.yaml"].append({"id": "EXTRA:RB"})

    rendered = _deployment(tmp_path, app_source, facility=tree).build()

    assert "EXTRA:RB" in {c["id"] for c in rendered.facility["channels"]}


def test_a_failed_build_names_what_the_build_said(tmp_path: Path, app_source: Path) -> None:
    deployment = _deployment(tmp_path, app_source, logins={"operator": "sometimes"})

    with pytest.raises(AssertionError, match="osprey build failed"):
        deployment.build()


# --- a deployment's own measurement kind, through the same fixture -----------------------

# The kind's package, the tree that lists it and the wheel stand-in are the ones
# ``tests/facility/test_fixture_kind_deployment.py`` builds; only the route differs.


@pytest.fixture
def kind_installed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The kind's package on the path for the test, its entry points read afresh around it."""
    from osprey.deployment import wheel_build

    read_entry_points_afresh()
    install(monkeypatch, PACKAGE_DIR)
    monkeypatch.setattr(wheel_build, "_build_wheel", fake_wheel)
    yield
    read_entry_points_afresh()


@pytest.mark.usefixtures("kind_installed")
def test_a_deployments_kind_is_built_through_the_fixture(tmp_path: Path) -> None:
    deployment = synthetic_deployment(
        tmp_path / "site",
        apps={},
        facility=fixture_tree(),
        profile={"dependencies": [SOURCE]},
    )
    source = deployment.root / SOURCE
    shutil.copytree(
        PACKAGE_DIR / "fixture_kind_deployment",
        source / "fixture_kind_deployment",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (source / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")

    rendered = deployment.build()

    sr = rendered.facts["measurement_models"]["SR"]
    assert set(sr["kinds"]) == {"trm", KIND}
    assert sr["kinds"][KIND]["facts"] == FACTS
    assert KIND in rendered.facts["measurement_kinds"]
