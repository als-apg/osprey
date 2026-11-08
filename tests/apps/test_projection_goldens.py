"""Goldens of the example application as one real build projects it.

The ``knob-example`` application ships in the control-assistant preset. This
module builds that preset once and pins, as diff-readable YAML under
``goldens/``, every place the build writes the application into the deployment:

* ``deployment.yml`` — the host-level facts: the projected ``services`` row,
  the container's compose service and its ``app-config.yml``, the panel's
  roster-grant row, the reach contract of its service, and the launch-token
  variables each web-terminal container is handed.
* ``render-<name>.yml`` — one per render (``host`` is the deployment's own):
  the panel block, the tool server, the approval and tools blocks, the dispatch
  denial list, the settings hook matchers and permission entries, the hook
  config's application entries, the panel token's route table and the reach
  contract evaluated on that render.

A login's arming fact is read off what its panel block holds: a login whose
tier writes (``asked``, ``allowed``) carries every declared agent route and the
panel's ``launch_token_env``; a ``never`` login carries the safe-method routes
and no launch token. The tool server, its ``writes:`` list and the
``facility_write_tools`` entry are the same on every tier: the tier, not the
projection, decides what those writes may do.

Only slices that name the application are pinned, and nothing machine-specific
(an interpreter path, the build directory) survives into a golden.

To regenerate after an intended change, run ``uv run python -m
tests.apps.test_projection_goldens`` and read the diff before committing it.
"""

from __future__ import annotations

import difflib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from tests.cli._lifecycle_build import invoke_build, rendered_config

#: Every golden reads one real build of the flagship preset.
pytestmark = pytest.mark.slow

GOLDENS = Path(__file__).parent / "goldens"

APP = "knob-example"

#: The full-name prefix of every tool the application's server exposes.
TOOL_PREFIX = f"mcp__{APP}__"

#: The renders the preset serves, ``host`` first: the deployment's own render.
RENDERS = ("host", "readonly", "readwrite", "admin", "logbook", "knowledge", "channel-finder")

#: The repo's directory name, which is the project name every compose name carries.
PROJECT = "my-facility"


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build_preset(parent: Path) -> Path:
    """Init and build the control-assistant preset under *parent*; return the repo."""
    from tests._builds import init_project

    repo = init_project(parent, "control-assistant", PROJECT)
    result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    return repo


def _render_dir(repo: Path, render: str) -> Path:
    if render == "host":
        return repo / "build"
    entry = rendered_config(repo)["modules"]["web_terminals"]["renders"][render]
    return repo / entry["project_path"]


def _render_config(repo: Path, render: str) -> dict[str, Any]:
    if render == "host":
        return rendered_config(repo)
    config: dict[str, Any] = yaml.safe_load(
        (_render_dir(repo, render) / "config.yml").read_text("utf-8")
    )
    return config


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return document


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------


def _names_app(value: Any) -> bool:
    return APP in json.dumps(value)


def _hook_command(hook: dict[str, Any]) -> dict[str, Any]:
    """A settings hook with its interpreter path dropped: the script and its arguments."""
    command = str(hook.get("command", ""))
    _, _, tail = command.partition('"$CLAUDE_PROJECT_DIR/')
    return {"run": (tail or command).replace('"', ""), "timeout": hook.get("timeout")}


def _settings_slice(settings: dict[str, Any]) -> dict[str, Any]:
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event, groups in (settings.get("hooks") or {}).items():
        matched = [
            {
                "matcher": group["matcher"],
                "hooks": [_hook_command(h) for h in group.get("hooks") or []],
            }
            for group in groups
            if APP in str(group.get("matcher", ""))
        ]
        if matched:
            hooks[event] = matched
    permissions = {
        key: [entry for entry in entries if APP in entry]
        for key, entries in (settings.get("permissions") or {}).items()
        if isinstance(entries, list)
    }
    return {"hooks": hooks, "permissions": {k: v for k, v in permissions.items() if v}}


def _hook_config_slice(hook_config: dict[str, Any]) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in hook_config.items():
        if isinstance(value, dict):
            entries = {k: v for k, v in value.items() if APP in k or _names_app(v)}
        elif isinstance(value, list):
            entries = sorted({entry for entry in value if isinstance(entry, str) and APP in entry})
        else:
            continue
        if entries:
            sliced[key] = entries
    return sliced


def _routes(config: dict[str, Any]) -> list[list[str]]:
    """The panel token's routes below this application's panel, as the terminal derives them."""
    from osprey.interfaces.web_auth import panel_tier_routes
    from osprey.interfaces.web_terminal.app import _load_panel_config
    from osprey.interfaces.web_terminal.routes.proxy import application_panels
    from osprey.profiles.sidecar_panels import SIDECAR_PANELS, PanelRegistry

    with patch("osprey_connectors.workspace.load_osprey_config", return_value=config):
        _enabled, custom, _default = _load_panel_config()
    registry = PanelRegistry(SIDECAR_PANELS, application_panels(custom))
    prefix = f"/panel/{APP}/"
    return sorted([m, p] for m, p in panel_tier_routes(registry) if p.startswith(prefix))


def _reach_on(contract: Any, config: dict[str, Any]) -> dict[str, Any]:
    consumers = [
        {
            "name": consumer.name,
            "on": consumer.is_on(config),
            "resolves": consumer.resolves(config),
            "dial": list(dial) if (dial := consumer.dial(config)) else None,
        }
        for consumer in contract.consumers
    ]
    held = [grant.env for grant in contract.credentials if grant.gate is None or grant.gate(config)]
    return {"consumers": consumers, "credentials_held": held}


def _contract(host: dict[str, Any]) -> Any:
    from osprey.deployment.reach import reach_contracts

    return reach_contracts(host)[APP]


def render_slice(repo: Path, render: str) -> dict[str, Any]:
    """What one render carries of the application."""
    config = _render_config(repo, render)
    claude = _render_dir(repo, render) / ".claude"
    return {
        "control_system.writes": (config.get("control_system") or {}).get("writes"),
        "web.panels": {
            k: v for k, v in ((config.get("web") or {}).get("panels") or {}).items() if k == APP
        },
        "claude_code.servers": {
            k: v
            for k, v in ((config.get("claude_code") or {}).get("servers") or {}).items()
            if k == APP
        },
        "apps": {k: v for k, v in (config.get("apps") or {}).items() if k == APP},
        "dispatch.denied_app_tools": (config.get("dispatch") or {}).get("denied_app_tools"),
        "settings.json": _settings_slice(_json(claude / "settings.json")),
        "hook_config.json": _hook_config_slice(_json(claude / "hooks" / "hook_config.json")),
        "panel_token_routes": _routes(config),
        "reach": _reach_on(_contract(rendered_config(repo)), config),
    }


def deployment_slice(repo: Path) -> dict[str, Any]:
    """What the deployment carries of the application, across its renders."""
    from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
    from osprey.deployment.web_terminals.render import render_web_terminals
    from osprey.deployment.web_terminals.roster import panel_roster_grant_table

    host = rendered_config(repo)
    service_dir = repo / "build" / "services" / APP
    compose = yaml.safe_load((service_dir / "docker-compose.yml").read_text("utf-8"))
    app_config = yaml.safe_load((service_dir / "app-config.yml").read_text("utf-8"))

    grant = panel_roster_grant_table(host)[APP]
    contract = _contract(host)

    inputs = resolve_render_inputs(host, repo)
    web = yaml.safe_load(render_web_terminals(host, **inputs)["docker-compose.web.yml"])
    terminal_env: dict[str, list[str]] = {}
    for name, service in web["services"].items():
        if not name.startswith("web-"):
            continue
        environment = service.get("environment") or []
        if isinstance(environment, dict):
            environment = [f"{k}={v}" for k, v in environment.items()]
        terminal_env[name] = sorted(e for e in environment if "KNOB_EXAMPLE" in e)

    return {
        "services": {APP: host["services"][APP]},
        "deployed_services": [s for s in host.get("deployed_services") or [] if s == APP],
        "compose": compose["services"],
        "app-config.yml": app_config,
        "roster_grant": {
            "service": grant.service,
            "single_user_alias": grant.single_user_alias,
        },
        "reach_contract": {
            "service": contract.service,
            "consumers": [
                {"name": c.name, "switch_key": c.switch_key, "refuse": c.refuse}
                for c in contract.consumers
            ],
            "credentials": [grant.env for grant in contract.credentials],
            "no_client_reach": contract.no_client_reach,
            "derived_by": contract.derived_by,
        },
        "application_token_renders": {
            env: sorted(renders)
            for env, renders in inputs["application_token_renders"].items()
            if "KNOB_EXAMPLE" in env
        },
        "web_terminal_env": terminal_env,
    }


def goldens(repo: Path) -> dict[str, dict[str, Any]]:
    """Every golden file's name and the slice it pins."""
    files = {"deployment.yml": deployment_slice(repo)}
    for render in RENDERS:
        files[f"render-{render}.yml"] = render_slice(repo, render)
    return files


def _dump(document: dict[str, Any]) -> str:
    return yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=100)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def projected(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    repo = build_preset(tmp_path_factory.mktemp("projection-goldens"))
    return goldens(repo)


@pytest.mark.parametrize("name", ["deployment.yml", *(f"render-{r}.yml" for r in RENDERS)])
def test_projection_matches_its_golden(projected: dict[str, dict[str, Any]], name: str) -> None:
    path = GOLDENS / name
    assert path.is_file(), f"missing golden {path}; regenerate with this module's __main__"
    expected = path.read_text("utf-8")
    actual = _dump(projected[name])
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"goldens/{name}",
            tofile="build",
        )
    )
    assert actual == expected, f"the build's projection moved off goldens/{name}:\n{diff}"


@pytest.mark.parametrize(
    ("render", "armed"), [("readonly", False), ("readwrite", True), ("admin", True)]
)
def test_the_panel_block_carries_the_login_arming_fact(
    projected: dict[str, dict[str, Any]], render: str, armed: bool
) -> None:
    """A writing tier carries every route and the launch token; ``never`` the safe ones only."""
    block = projected[f"render-{render}.yml"]["web.panels"][APP]
    routes = [(route["method"], route["path"]) for route in block["agent_routes"]]
    if armed:
        assert routes == [("GET", "/api/state"), ("POST", "/api/write")]
        assert block["launch_token_env"] == "KNOB_EXAMPLE_LAUNCH_TOKEN"
    else:
        assert routes == [("GET", "/api/state")]
        assert "launch_token_env" not in block
    assert "arm_writes" not in block


@pytest.mark.parametrize("render", ["readonly", "readwrite", "admin"])
def test_the_write_tool_enters_the_gate_on_every_tier(
    projected: dict[str, dict[str, Any]], render: str
) -> None:
    sliced = projected[f"render-{render}.yml"]
    assert sliced["claude_code.servers"][APP]["writes"] == ["set_corrector"]
    assert sliced["hook_config.json"]["facility_write_tools"] == [f"{TOOL_PREFIX}set_corrector"]
    assert sliced["dispatch.denied_app_tools"] == [f"{TOOL_PREFIX}set_corrector"]


@pytest.mark.parametrize("render", ["logbook", "knowledge", "channel-finder"])
def test_a_shared_login_shows_no_example_panel(
    projected: dict[str, dict[str, Any]], render: str
) -> None:
    sliced = projected[f"render-{render}.yml"]
    assert sliced["web.panels"] == {} and sliced["claude_code.servers"] == {}


def _regenerate() -> None:
    """Overwrite the goldens from a fresh build of today's tree."""
    with tempfile.TemporaryDirectory() as parent:
        repo = build_preset(Path(parent))
        GOLDENS.mkdir(parents=True, exist_ok=True)
        for name, document in goldens(repo).items():
            path = GOLDENS / name
            path.write_text(_dump(document), encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
