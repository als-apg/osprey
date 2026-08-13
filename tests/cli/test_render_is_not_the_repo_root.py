"""A render is not a repo: what a deployment's own root is, and who reads it.

Two consumers derive a repo root from a directory that holds ``config.yml``,
and both were taking that directory for the root. They fail differently and are
pinned together here because it is one mistake: the rendered deployment lives at
``<repo>/build``, one level inside the repo, and everything durable — the
profile, the secrets, the state zone — is at the root above it.

1. ``TemplateManager.regen_if_drift`` declared ``project_root`` as the render,
   so every MCP server was handed a config path that names no file.
2. The SDK provider env read ``.env`` beside the config, where a deployment
   deliberately keeps none, so ``${VAR}`` expansion and provider auth resolved
   from ``os.environ`` alone.

The first one fires everywhere: ``regen_if_drift`` runs on every web-terminal
launch and after every config write from the terminal's config panel. Every MCP
server and hook receives its config as ``registry/mcp.py``'s
``{project_root}/build/config.yml``, so a ``project_root`` of ``<repo>/build``
declares ``<repo>/build/build/config.yml`` — a file that does not exist. The
servers come up pointed at nothing and ``get_config_value`` raises
``FileNotFoundError``. It fires on every host deployment, because a faithful
render always diffs against artifacts written with the wrong root.

The rule these tests pin is one rule, and it is the *runtime's* rule:
``project_root`` is ``workspace.repo_root_for_config`` of the config that was
read. It therefore inverts ``rendered_config_path`` — the path regen declares
always names the file regen read — and it holds unchanged in both repo shapes,
because they are the same shape: ``<repo>/build`` on a host and
``/app/<name>/build`` in a container, whose project tree has been a deployment
repo in its own right since the container render landed.

Both shapes are proven against a REAL ``osprey build`` here rather than a
stubbed tree, because the property is byte-level agreement between what a build
writes and what a regen writes; only the real renderer can show that.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build as build_command
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.workspace import (
    BUILD_DIR_NAME,
    IMAGE_DIR_NAME,
    RENDERED_CONFIG_RELPATH,
    rendered_config_path,
    repo_root_for_config,
)

#: The env keys every OSPREY MCP server reads its config path from. Both names
#: appear because the servers were wired at different times; the value is the
#: same string in every one of them.
CONFIG_ENV_KEYS = ("CONFIG_FILE", "OSPREY_CONFIG")


def _declared_config_paths(mcp_json: Path) -> set[str]:
    """Every config path declared to an MCP server in a rendered ``.mcp.json``."""
    servers = json.loads(mcp_json.read_text(encoding="utf-8")).get("mcpServers", {})
    return {
        value
        for server in servers.values()
        for key, value in (server.get("env") or {}).items()
        if key in CONFIG_ENV_KEYS
    }


def _server_commands(mcp_json: Path) -> set[str]:
    """Every interpreter an MCP server in a rendered ``.mcp.json`` launches with."""
    servers = json.loads(mcp_json.read_text(encoding="utf-8")).get("mcpServers", {})
    return {server["command"] for server in servers.values() if server.get("command")}


# ---------------------------------------------------------------------------
# The wiring, pinned without a build: which root the entry point derives.
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_override(monkeypatch):
    """Capture the ``project_root_override`` the manager hands the renderer.

    Stubs the rendering primitive so these pins cost no I/O and can name a root
    that does not exist on this machine — which is the point for the container
    case: ``/app/<name>`` is not mountable here, but the derivation is path
    arithmetic and answers for it exactly as it will inside the image.
    """
    seen: dict[str, object] = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return {"changed": [], "unchanged": []}

    monkeypatch.setattr(claude_code, "regenerate_claude_code", _spy)
    return seen


def test_the_root_of_a_render_is_the_repo_it_sits_in(tmp_path, captured_override):
    """A render at ``<repo>/build`` is rendered for ``<repo>``."""
    render = tmp_path / "als-exemplar" / BUILD_DIR_NAME
    render.mkdir(parents=True)

    TemplateManager().regenerate_claude_code(render)

    assert captured_override["project_root_override"] == tmp_path / "als-exemplar"


def test_the_container_render_resolves_to_the_container_repo(captured_override):
    """The same rule, applied at the path a container actually runs at.

    A container's project tree is a deployment repo at ``/app/<name>`` with its
    render at ``/app/<name>/build``, so the repo root the regen must name is
    the one the build baked into that render's ``config.yml``. No branch, no
    container-aware spelling — the arithmetic already answers.
    """
    container_root = Path("/app/als-exemplar")

    TemplateManager().regenerate_claude_code(container_root / BUILD_DIR_NAME)

    assert captured_override["project_root_override"] == container_root
    # ...which is the same statement as: what regen declares to an MCP server
    # is the file regen read.
    assert rendered_config_path(container_root) == container_root / RENDERED_CONFIG_RELPATH


def test_a_directory_holding_its_own_config_is_its_own_root(tmp_path, captured_override):
    """The default is unchanged for a tree whose ``config.yml`` is at its root."""
    flat = tmp_path / "flat-project"
    flat.mkdir()

    TemplateManager().regenerate_claude_code(flat)

    assert captured_override["project_root_override"] == flat


def test_an_explicit_override_is_never_second_guessed(tmp_path, captured_override):
    """A caller rendering for another machine says so, and is believed.

    ``osprey build`` (``--runtime-root``, the container render) and
    ``osprey status`` pass a root that is deliberately not the one on disk.
    Deriving over the top of them would silently relocate their render back.
    """
    render = tmp_path / "als-exemplar" / BUILD_DIR_NAME
    render.mkdir(parents=True)

    TemplateManager().regenerate_claude_code(render, project_root_override="/app/somewhere-else")

    assert captured_override["project_root_override"] == "/app/somewhere-else"


def test_naming_the_repo_root_does_not_cost_the_render_its_own_venv(tmp_path) -> None:
    """Two questions, two answers, and neither may be inferred from the other.

    Naming a ``project_root`` at all is how a caller says "this render is for a
    machine I cannot probe", which makes the interpreter derivation stop looking
    for a project ``.venv``. That is right for a container render and wrong for
    a regen, whose venv is ``<render>/.venv`` and is the whole reason
    ``osprey build`` installs OSPREY into the build output: the servers must
    keep launching after the framework tree that built them moves.

    So the repo root and the venv's home travel together here — and this is the
    pin that catches fixing one by breaking the other.
    """
    repo = tmp_path / "als-exemplar"
    render = TemplateManager().create_project(
        project_name=BUILD_DIR_NAME,
        output_dir=repo,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    venv_python = render / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    venv_python.chmod(0o755)

    TemplateManager().regenerate_claude_code(render)

    assert _server_commands(render / ".mcp.json") == {str(venv_python)}
    assert _declared_config_paths(render / ".mcp.json") == {str(render / "config.yml")}


# ---------------------------------------------------------------------------
# The property, over a real render: a build and a regen agree, byte for byte.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory) -> Path:
    """The exemplar, built for real (seconds, not milliseconds).

    ``seed_env=True`` so the render resolves ``${VAR}`` from a secrets zone the
    way a deployment does — a regen that read a different environment than the
    build did would diff for reasons that have nothing to do with paths.
    """
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(
        tmp_path_factory.mktemp("regenroot") / EXEMPLAR_DIRNAME, seed_env=True
    )
    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture(scope="module")
def rehomed_container_repo(built_repo: Path, tmp_path_factory) -> Path:
    """The build's own container repo, re-rooted somewhere this machine has.

    ``osprey build`` writes a second, container-destined copy of the deployment
    under ``build/.image/<name>/`` — a repo whose ``config.yml`` records
    ``/app/<name>`` and whose servers launch the image's interpreter. That is
    the tree an image ships, and the shape a regen meets when the web terminal
    starts inside the container.

    ``/app/<name>`` cannot be materialized on the host, so the copy is re-rooted
    onto a real directory: every occurrence of the recorded container root is
    rewritten to where the copy actually lives, and the image interpreter to the
    one running this test. That is precisely what a container is — the same repo
    rooted at its own absolute path, with its own python — so a regen over the
    result must still find nothing to change. Both values are read out of the
    build's own output rather than restated here.
    """
    from osprey.cli.build_cmd import _CONTAINER_INTERPRETER

    source = built_repo / BUILD_DIR_NAME / IMAGE_DIR_NAME / built_repo.name
    assert source.is_dir(), f"the build produced no container repo at {source}"

    container_root = yaml.safe_load((source / RENDERED_CONFIG_RELPATH).read_text(encoding="utf-8"))[
        "project_root"
    ]

    destination = tmp_path_factory.mktemp("rehomed") / built_repo.name
    shutil.copytree(source, destination)
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # a binary asset names neither
        if container_root not in text and _CONTAINER_INTERPRETER not in text:
            continue
        path.write_text(
            text.replace(container_root, str(destination)).replace(
                _CONTAINER_INTERPRETER, sys.executable
            ),
            encoding="utf-8",
        )
    return destination


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["host", "container"])
def test_a_regen_over_a_real_render_changes_nothing(
    shape, built_repo: Path, rehomed_container_repo: Path
) -> None:
    """The whole property in one line: a build and a regen render the same tree.

    Not only ``.mcp.json`` — nothing at all. Every artifact ``osprey build``
    leaves in the render is what ``regen_if_drift`` would write for the same
    repo, so a launch-time regen is a no-op and an operator's config edit is the
    only thing that ever moves an artifact.
    """
    repo = built_repo if shape == "host" else rehomed_container_repo
    render = repo / BUILD_DIR_NAME
    before = (render / ".mcp.json").read_text(encoding="utf-8")

    changed = TemplateManager().regen_if_drift(render)

    assert changed == [], f"regen rewrote a faithful {shape} render: {changed}"
    assert (render / ".mcp.json").read_text(encoding="utf-8") == before


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["host", "container"])
def test_every_config_path_a_regen_declares_is_the_config_it_read(
    shape, built_repo: Path, rehomed_container_repo: Path
) -> None:
    """The failure this guards is a path that exists nowhere.

    Deriving the root from the render itself produced
    ``<repo>/build/build/config.yml``: one string, declared to every server,
    naming no file. Asserting the paths *resolve* — and resolve to the config
    the regen actually read — is what makes this test fail on that bug rather
    than merely describing it.
    """
    repo = built_repo if shape == "host" else rehomed_container_repo
    render = repo / BUILD_DIR_NAME
    TemplateManager().regen_if_drift(render)

    declared = _declared_config_paths(render / ".mcp.json")
    assert declared, "a render with no declared config path would pass vacuously"
    assert declared == {str(render / "config.yml")}
    assert all(Path(path).is_file() for path in declared)


@pytest.mark.slow
def test_the_repo_root_regen_derives_is_the_one_the_build_recorded(built_repo: Path) -> None:
    """Regen and the running deployment resolve the same root, from one helper.

    ``config.yml``'s ``project_root`` is what the *runtime* anchors relative
    paths on (agent data, audit, simulation state). If regen derived a different
    root, ``.mcp.json`` and the config it names would disagree about where the
    deployment is — the artifacts would be internally consistent and still wrong.
    """
    render = built_repo / BUILD_DIR_NAME
    recorded = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))["project_root"]

    assert repo_root_for_config(render / "config.yml") == Path(recorded)


# ---------------------------------------------------------------------------
# The same distinction one layer over: which ``.env`` a provider resolves from.
# ---------------------------------------------------------------------------

#: A built-in proxy provider, used because its two auth vars differ — the raw
#: secret var and the var Claude Code authenticates with — so a test can tell
#: which value travelled rather than only that something did.
PROBE_PROVIDER = "als-apg"
PROBE_SECRET_ENV = "ALS_APG_API_KEY"


def _synthetic_deployment(root: Path, *, flat: bool = False) -> Path:
    """Write the smallest thing that is a deployment: a config and a secret.

    Returns the directory a caller passes as ``project_dir`` — the render for a
    repo, the root itself for the flat shape that predates it.
    """
    render = root if flat else root / BUILD_DIR_NAME
    render.mkdir(parents=True, exist_ok=True)
    (render / "config.yml").write_text(
        yaml.dump({"project_name": root.name, "claude_code": {"provider": PROBE_PROVIDER}}),
        encoding="utf-8",
    )
    (root / ".env").write_text(f"{PROBE_SECRET_ENV}=from-the-repos-dotenv\n", encoding="utf-8")
    return render


@pytest.fixture(autouse=True)
def _no_ambient_matrix_overrides(monkeypatch):
    """Keep the benchmark matrix's env overrides out of these resolutions."""
    monkeypatch.delenv("OSPREY_E2E_FORCE_MODEL", raising=False)
    monkeypatch.delenv("OSPREY_E2E_PROXY_BASE_URL", raising=False)


def test_the_sdk_provider_env_resolves_from_the_repos_secrets_zone(tmp_path, monkeypatch):
    """``.env`` is at the repo root, and it is the value that must win.

    ``osprey query`` and the SDK harness both hand ``provider_env_for_project``
    the render. Reading ``.env`` beside that config reads nothing — a build
    writes no secret into the zone it wipes — so the resolution silently fell
    back to whatever the ambient shell happened to export. Here the two differ,
    which is the only way to tell a superset from a substitute.
    """
    from osprey.agent_runner.primitives import provider_env_for_project

    render = _synthetic_deployment(tmp_path / "als-exemplar")
    monkeypatch.setenv(PROBE_SECRET_ENV, "from-a-stale-shell-export")

    env = provider_env_for_project(render)

    assert env[PROBE_SECRET_ENV] == "from-the-repos-dotenv"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "from-the-repos-dotenv"


def test_a_flat_project_still_resolves_from_its_own_env(tmp_path, monkeypatch):
    """The derivation is default-preserving where the two coincide."""
    from osprey.agent_runner.primitives import provider_env_for_project

    project = _synthetic_deployment(tmp_path / "flat-project", flat=True)
    monkeypatch.setenv(PROBE_SECRET_ENV, "from-a-stale-shell-export")

    assert provider_env_for_project(project)[PROBE_SECRET_ENV] == "from-the-repos-dotenv"


def test_the_web_launch_probe_resolves_the_provider_against_the_repo(tmp_path, monkeypatch):
    """``osprey web``'s pre-flight already holds both directories apart.

    The probe reports whether the provider the launch will use can authenticate.
    Resolving that provider without the repo's ``.env`` reports on a different
    provider than the one that starts — a pre-flight answering for the wrong
    thing is worse than none.
    """
    from osprey.build import claude_code_resolver
    from osprey.cli import web_cmd

    seen: dict[str, object] = {}

    def _spy(project_dir, **kwargs):
        seen["project_dir"] = project_dir
        seen.update(kwargs)
        return None

    monkeypatch.setattr(claude_code_resolver, "load_provider_spec", _spy)
    repo = tmp_path / "als-exemplar"
    render = _synthetic_deployment(repo)

    web_cmd._probe_auth_secret(render, repo)

    assert seen["project_dir"] == render
    assert seen["env_dir"] == repo


def test_the_web_terminal_lifespan_resolves_the_provider_against_the_repo(tmp_path, monkeypatch):
    """The server that actually starts resolves from the repo, not the render.

    ``osprey web`` chdirs into the render and serves ``build/config.yml``, so
    the lifespan's provider read had the render as both the config zone and the
    secrets zone. A deployment keeps neither its ``.env`` nor anything else
    durable there, so a facility that redirects its gateway through the repo's
    ``.env`` started the terminal on the built-in URL instead — the same class
    of miss the pre-flight above already avoids.

    The real resolver runs; only the injection that follows it is short-
    circuited, because ``inject_provider_env`` mutates the interpreter's own
    ``os.environ`` and this assertion is about what was resolved.
    """
    from fastapi.testclient import TestClient

    from osprey.build import claude_code_resolver
    from osprey.interfaces.web_terminal import app as web_app

    repo = tmp_path / "als-exemplar"
    render = _synthetic_deployment(repo)
    (repo / ".env").write_text(
        f"{PROBE_SECRET_ENV}=from-the-repos-dotenv\nALS_APG_BASE_URL=https://gw.test/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
    (render / "_agent_data").mkdir(exist_ok=True)

    resolved: list[object] = []
    real = claude_code_resolver.load_provider_spec

    def _record_then_stop(project_dir, **kwargs):
        resolved.append(real(project_dir, **kwargs))
        return None

    monkeypatch.setattr(claude_code_resolver, "load_provider_spec", _record_then_stop)
    monkeypatch.setattr(
        web_app, "_load_web_config", lambda *_a, **_k: {"watch_dir": str(render / "_agent_data")}
    )
    monkeypatch.chdir(render)

    app = web_app.create_app(
        config_path=str(render / "config.yml"),
        shell_command="echo",
        project_dir=str(render),
    )
    with TestClient(app):
        pass

    assert resolved, "the web terminal lifespan must resolve a provider spec"
    assert resolved[0].env_block["ANTHROPIC_BASE_URL"] == "https://gw.test"
