"""``osprey build`` renders every tier and every app beside the deployment.

One build of the control-assistant exemplar renders ``build/<project>-<name>``
for each login file under ``tiers/``, and an image context
``build/.image/<project>-<name>/`` for each of them beside the deployment's own.
A tier render is the deployment's agent at one access level, so it records the
deployment's preset; an app render is its own agent over the deployment's
services. The logbook app is checked end to end: its servers, its subagent, its
rules, its write posture, and the card the roster resolves for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from osprey.deployment.web_terminals.render_images import (
    render_build_profile_shape_problem,
    verify_renders,
)
from osprey.deployment.web_terminals.roster import resolve_cards
from osprey.utils.workspace import IMAGE_DIR_NAME
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests.cli._lifecycle_build import invoke_build
from tests.fixtures.lifecycle_repo import build_exemplar_repo, preserved_environ

PROJECT = "als-exemplar"
TIERS = ("readonly", "readwrite", "admin")
APPS = ("logbook", "knowledge", "channel-finder")
HOST_PRESET = "control-assistant"

#: Servers the logbook app must not run: the deployment's write and compute
#: surfaces, and the services only an operator's agent dials.
LOGBOOK_ABSENT_SERVERS = ("bluesky", "health", "controls", "python")


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append((record.levelno, record.getMessage()))


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The exemplar, built once from a working directory outside it."""
    outside = tmp_path_factory.mktemp("renders")
    repo = build_exemplar_repo(outside / "deployments" / PROJECT)
    records = _Records()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(records)
    root.setLevel(logging.DEBUG)
    try:
        with pytest.MonkeyPatch.context() as patch, preserved_environ():
            patch.chdir(outside)
            result = invoke_build(repo)
    finally:
        root.removeHandler(records)
        root.setLevel(previous_level)
    assert result.exit_code == 0, result.output
    return {"repo": repo, "output": result.output, "log": records.messages}


def _render(repo: Path, name: str) -> Path:
    return repo / BUILD_DIR_NAME / f"{PROJECT}-{name}"


def _config(render: Path) -> dict:
    return yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))


def _mcp_servers(render: Path) -> set[str]:
    return set(json.loads((render / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"])


def _values_of(node: object, key: str) -> list[object]:
    found: list[object] = []
    if isinstance(node, dict):
        for k, value in node.items():
            if k == key:
                found.append(value)
            found.extend(_values_of(value, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_values_of(item, key))
    return found


# ---------------------------------------------------------------------------
# One render per tier and app, and one image context for each
# ---------------------------------------------------------------------------


def test_every_tier_and_app_is_rendered(built):
    repo = built["repo"]
    rendered = {
        p.name for p in (repo / BUILD_DIR_NAME).iterdir() if p.name.startswith(f"{PROJECT}-")
    }

    assert rendered == {f"{PROJECT}-{name}" for name in (*TIERS, *APPS)}
    for name in (*TIERS, *APPS):
        render = _render(repo, name)
        assert (render / "config.yml").is_file(), name
        assert (render / "Dockerfile").is_file(), name
        assert (render / ".mcp.json").is_file(), name


def test_every_tier_and_app_has_an_image_context(built):
    repo = built["repo"]
    image_root = repo / BUILD_DIR_NAME / IMAGE_DIR_NAME
    contexts = {p.name for p in image_root.iterdir()}

    assert contexts == {PROJECT, *(f"{PROJECT}-{name}" for name in (*TIERS, *APPS))}
    for name in (*TIERS, *APPS):
        project = f"{PROJECT}-{name}"
        render = image_root / project / BUILD_DIR_NAME
        assert (render / "config.yml").is_file(), name
        assert (render / "Dockerfile").is_file(), name
        # The container copy is rendered against the path the image sees
        # itself at, so its servers read its own config.
        assert f"/app/{project}/" in (render / ".mcp.json").read_text(encoding="utf-8"), name


def test_the_build_reports_tier_and_app_render_counts(built):
    output = built["output"]

    assert f"{len(TIERS) + len(APPS)} render(s)" in output, output
    assert f"{len(TIERS) + len(APPS) + 1} image build context(s)" in output, output


# ---------------------------------------------------------------------------
# A tier is the deployment's agent: it records the deployment's preset
# ---------------------------------------------------------------------------


def test_no_render_falls_back_to_a_preset_this_osprey_does_not_ship(built):
    fallbacks = [m for _level, m in built["log"] if "does not ship" in m]

    assert fallbacks == []


@pytest.mark.parametrize("name", TIERS)
def test_a_tier_render_is_stamped_with_the_deployment_preset(built, name):
    repo = built["repo"]
    host = json.loads((repo / BUILD_DIR_NAME / ".osprey-manifest.json").read_text())
    tier = json.loads((_render(repo, name) / ".osprey-manifest.json").read_text())

    assert host["creation"]["template"] == HOST_PRESET
    assert tier["creation"]["template"] == HOST_PRESET
    # The render still names the file it came from.
    assert tier["build_args"]["profile_path"] == str(repo / "tiers" / f"{name}.yml")


#: The write ceiling each bundled tier pins over the exemplar's ``asked``.
_TIER_WRITES = {"readonly": "never", "readwrite": "asked", "admin": "allowed"}


@pytest.mark.parametrize("name", TIERS)
def test_tier_renders_keep_their_own_write_posture(built, name):
    config = _config(_render(built["repo"], name))

    assert (config.get("control_system") or {}).get("writes") == _TIER_WRITES[name], name
    # The retired leaves are gone from the render, so nothing else states a posture.
    assert _values_of(config, "writes_enabled") == [], name


# ---------------------------------------------------------------------------
# The logbook app, end to end
# ---------------------------------------------------------------------------


def test_logbook_mcp_servers_are_exactly_workspace_ariel_and_graph(built):
    render = _render(built["repo"], "logbook")

    assert _mcp_servers(render) == {"osprey_workspace", "ariel", "graph"}


def test_logbook_image_copy_names_the_same_servers(built):
    render = built["repo"] / BUILD_DIR_NAME / IMAGE_DIR_NAME / f"{PROJECT}-logbook" / BUILD_DIR_NAME

    assert _mcp_servers(render) == {"osprey_workspace", "ariel", "graph"}


def test_logbook_selects_the_graph_subagent_and_the_safety_rule(built):
    claude = _render(built["repo"], "logbook") / ".claude"

    agents = {p.stem for p in (claude / "agents").glob("*.md")}
    rules = {p.stem for p in (claude / "rules").glob("*.md")}

    assert "facility-knowledge-graph" in agents
    assert "safety" in rules


def test_logbook_writes_nothing(built):
    config = _config(_render(built["repo"], "logbook"))
    writes = _values_of(config, "writes")

    assert writes, "the logbook render states no write posture at all"
    assert (config.get("control_system") or {}).get("writes") == "never", writes
    assert all(value == "never" for value in writes), writes
    assert _values_of(config, "writes_enabled") == []


@pytest.mark.parametrize("server", LOGBOOK_ABSENT_SERVERS)
def test_logbook_runs_no_operator_server(built, server):
    config = _config(_render(built["repo"], "logbook"))
    servers = (config.get("claude_code") or {}).get("servers") or {}
    entry = servers.get(server)

    # Absent, or present only to say it is off.
    assert entry is None or entry.get("enabled") is False, entry


def test_the_logbook_card_serves_the_logbook_render(built):
    repo = built["repo"]
    web_terminals = _config(repo / BUILD_DIR_NAME)["modules"]["web_terminals"]

    entries = resolve_cards(web_terminals, {}, strict=True, project_name=PROJECT)
    logbook = [e for e in entries if e.get("render") == "logbook"]

    assert len(logbook) == 1, entries
    assert logbook[0]["image"] == f"{PROJECT}-logbook:local"
    assert logbook[0]["name"] == "logbook"


# ---------------------------------------------------------------------------
# The start path reads the renders the build wrote
# ---------------------------------------------------------------------------


def _resolved(repo: Path) -> tuple[dict, list[dict]]:
    config = _config(repo / BUILD_DIR_NAME)
    entries = resolve_cards(
        config["modules"]["web_terminals"], {}, strict=True, project_name=PROJECT
    )
    return config, entries


def _with_absolute_project_paths(config: dict, repo: Path) -> dict:
    renders = config["modules"]["web_terminals"]["renders"]
    for entry in renders.values():
        entry["project_path"] = str(repo / entry["project_path"])
    return config


def test_verify_accepts_every_tier_and_app_render(built, monkeypatch):
    repo = built["repo"]
    config, entries = _resolved(repo)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    verify_renders(_with_absolute_project_paths(config, repo), entries, repo_root=repo)


@pytest.mark.parametrize(("name", "layer_dir"), [("readonly", "tiers"), ("logbook", "tiers")])
def test_verify_names_the_layer_file_of_an_absent_render(built, tmp_path, name, layer_dir):
    """A render that is not on disk sends the operator to the build, naming
    the login file it is built from and the directory it lands in."""
    repo = built["repo"]
    config, entries = _resolved(repo)
    config = _with_absolute_project_paths(config, repo)
    config["modules"]["web_terminals"]["renders"][name]["project_path"] = str(
        tmp_path / "gone" / f"{PROJECT}-{name}"
    )
    unit = [e for e in entries if e.get("render") == name][:1]
    assert unit, entries

    with pytest.raises(ValueError) as excinfo:
        verify_renders(config, unit, repo_root=repo)
    message = str(excinfo.value)

    assert "osprey build" in message
    assert f"{layer_dir}/{name}.yml" in message
    assert str(repo / BUILD_DIR_NAME / f"{PROJECT}-{name}") in message


def test_verify_names_a_missing_layer_file(built, tmp_path):
    """The projection names a tier whose file is gone: no build can write it."""
    repo = built["repo"]
    config, entries = _resolved(repo)
    config = _with_absolute_project_paths(config, repo)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["renders"]["ghost"] = {
        "kind": "tier",
        "project": f"{PROJECT}-ghost",
        "project_path": str(tmp_path / f"{PROJECT}-ghost"),
    }
    # Served by the roster: a card pinned to it.
    web_terminals["users"] = [*web_terminals["users"], {"name": "ghosted", "tier": "ghost"}]
    unit = [{"render": "ghost", "project": f"{PROJECT}-ghost", "image": "x:local"}]

    with pytest.raises(ValueError) as excinfo:
        verify_renders(config, unit, repo_root=repo)
    message = str(excinfo.value)

    assert str(repo / "tiers" / "ghost.yml") in message


def test_shape_problem_speaks_of_login_files():
    problem = render_build_profile_shape_problem("control-assistant")

    assert problem is not None
    assert "preset" in problem
    assert render_build_profile_shape_problem("tiers/readonly.yml") is None
    assert render_build_profile_shape_problem("apps/logbook.yml") is not None
