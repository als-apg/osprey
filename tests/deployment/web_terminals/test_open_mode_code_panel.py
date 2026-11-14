"""The open posture's refusal of a panel that runs a person's code.

Under ``auth.method: none`` nginx vouches for every terminal it proxies, so
nothing inside the deployment may be able to reach nginx back. The egress gate
covers the agent's tools; a notebook kernel is the same reach with no tool in
front of it. These pin the rule on all three surfaces that speak it: the gate,
the collected preflight report and the lint rule, each off one predicate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.agent_runner.tool_names import DENY_DEFAULTS
from osprey.deployment.errors import DeploymentError
from osprey.deployment.web_terminals import provision
from osprey.deployment.web_terminals.artifacts import (
    OpenModeCodePanelError,
    OpenModeEgressError,
    check_open_mode_requirements,
    open_mode_code_panels_by_render,
)
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.roster import config_code_panels
from osprey.profiles.sidecar_panels import JUPYTER_PANEL_ID, SIDECAR_PANELS

_RULE = "web_terminals.open_mode_code_panel"


def _project(root: Path, name: str, *, panels: dict | None, deny: list[str] | None = None) -> str:
    """Write a rendered project showing *panels*; return its relative path."""
    project_dir = root / "profiles" / name
    (project_dir / ".claude").mkdir(parents=True)
    (project_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    document: dict = {"project_name": name}
    if panels is not None:
        document["web"] = {"panels": panels}
    (project_dir / "config.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    (project_dir / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"deny": list(DENY_DEFAULTS) if deny is None else deny}}),
        encoding="utf-8",
    )
    return f"profiles/{name}"


def _config(root: Path, renders: dict[str, dict | None], *, method: str = "none") -> dict:
    """A roster with one login per render, each render showing its panels."""
    return {
        "project_name": "demo-project",
        "modules": {
            "web_terminals": {
                "enabled": True,
                "auth": {"method": method},
                "default_tier": next(iter(renders)),
                "users": [
                    {"name": f"user{index}", "index": index, "tier": render}
                    for index, render in enumerate(renders)
                ],
                "renders": {
                    render: {
                        "kind": "tier",
                        "project": f"{render}-proj",
                        "project_path": _project(root, f"{render}-proj", panels=panels),
                    }
                    for render, panels in renders.items()
                },
            }
        },
    }


_NOTEBOOK = {JUPYTER_PANEL_ID: {"enabled": True}}


def test_the_notebook_panel_is_the_registry_entry_that_runs_code():
    """The rule is driven by the registry, so the registry is what is pinned: a
    panel added later without the flag is a decision somebody made, not an
    omission the gate inherits."""
    assert [panel_id for panel_id, panel in SIDECAR_PANELS.items() if panel.runs_code] == [
        JUPYTER_PANEL_ID
    ]


@pytest.mark.parametrize(
    ("panels", "expected"),
    [
        (None, ()),
        ({}, ()),
        ({"ariel": {"enabled": True}}, ()),
        ({JUPYTER_PANEL_ID: {"enabled": False}}, ()),
        ({JUPYTER_PANEL_ID: {"enabled": True}}, (JUPYTER_PANEL_ID,)),
        ({JUPYTER_PANEL_ID: {}}, (JUPYTER_PANEL_ID,)),
    ],
)
def test_a_project_counts_only_a_code_panel_it_shows(panels, expected):
    config = {} if panels is None else {"web": {"panels": panels}}

    assert config_code_panels(config) == expected


def test_an_open_deployment_showing_the_notebook_panel_is_refused(tmp_path):
    config = _config(tmp_path, {"physicist": _NOTEBOOK, "operator": {}})

    with pytest.raises(OpenModeCodePanelError) as excinfo:
        check_open_mode_requirements(config, tmp_path)

    assert excinfo.value.renders == ["physicist"]
    assert excinfo.value.panels_by_render == {"physicist": (JUPYTER_PANEL_ID,)}
    message = str(excinfo.value)
    assert "'physicist'" in message and "'operator'" not in message
    assert f"web.panels.{JUPYTER_PANEL_ID}" in message
    assert "'token', 'password' or 'oidc'" in message


def test_the_refusal_is_a_deployment_error_and_not_a_value_error():
    """A ``ValueError`` here would be caught by the callers that treat a bad
    config as a soft problem; the open gate's refusals are never that."""
    assert issubclass(OpenModeCodePanelError, DeploymentError)
    assert not issubclass(OpenModeCodePanelError, ValueError)


@pytest.mark.parametrize("method", ["token", "password", "oidc"])
def test_a_deployment_behind_a_login_wall_keeps_the_notebook_panel(tmp_path, method):
    config = _config(tmp_path, {"physicist": _NOTEBOOK}, method=method)

    check_open_mode_requirements(config, tmp_path)

    assert open_mode_code_panels_by_render(config, tmp_path) == {}


@pytest.mark.parametrize("panels", [{}, {JUPYTER_PANEL_ID: {"enabled": False}}])
def test_an_open_deployment_without_the_panel_passes(tmp_path, panels):
    """The negative control: the rule refuses the panel, not the posture."""
    config = _config(tmp_path, {"operator": panels})

    check_open_mode_requirements(config, tmp_path)


def test_a_render_no_login_runs_is_not_read(tmp_path):
    """The gate is about what the deployment starts. A catalog entry nobody is
    assigned to starts no terminal and no kernel."""
    config = _config(tmp_path, {"operator": {}, "physicist": _NOTEBOOK})
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["users"] = [user for user in web_terminals["users"] if user["tier"] == "operator"]

    check_open_mode_requirements(config, tmp_path)


def test_the_egress_refusal_is_spoken_first(tmp_path):
    """A render that lifts a shell and shows the panel hears about the shell:
    that refusal also covers the render nobody has rendered, and its remedy is
    the one that makes the config readable at all."""
    config = _config(tmp_path, {"physicist": _NOTEBOOK})
    settings = tmp_path / "profiles" / "physicist-proj" / ".claude" / "settings.json"
    settings.write_text(
        json.dumps({"permissions": {"deny": [e for e in DENY_DEFAULTS if e != "Bash"]}}),
        encoding="utf-8",
    )

    with pytest.raises(OpenModeEgressError):
        check_open_mode_requirements(config, tmp_path)


def test_preflight_refuses_before_any_credential_is_minted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reached: list[str] = []
    monkeypatch.setattr(
        provision, "ensure_env_production", lambda config, root: reached.append("env_production")
    )
    monkeypatch.setattr(
        provision, "_provision_auth_secrets", lambda wt, root: reached.append("auth_secrets")
    )

    with pytest.raises(OpenModeCodePanelError):
        provision.preflight_web_terminals(_config(tmp_path, {"physicist": _NOTEBOOK}))

    assert reached == []


def test_the_collected_report_carries_the_refusal(tmp_path):
    config = _config(tmp_path, {"physicist": _NOTEBOOK})

    blocking, _advisory = provision.web_terminal_preflight_report(config, repo_root=tmp_path)

    expected = str(OpenModeCodePanelError({"physicist": (JUPYTER_PANEL_ID,)}))
    assert [problem for problem, _remedy in blocking if problem == expected] == [expected]


def test_lint_reports_what_the_gate_refuses(tmp_path):
    """One predicate behind both: a lint that cleared what the start refuses
    would send the operator to the deploy to learn about their own edit."""
    config = _config(tmp_path, {"physicist": _NOTEBOOK, "operator": {}})

    findings = lint_web_terminals(config, project_root=tmp_path)

    errors = [f for f in findings if f.severity == "error" and f.code == _RULE]
    assert len(errors) == 1
    assert f"'physicist' shows '{JUPYTER_PANEL_ID}'" in errors[0].message
    assert "'operator'" not in errors[0].message
    with pytest.raises(OpenModeCodePanelError):
        check_open_mode_requirements(config, tmp_path)


def test_lint_is_silent_behind_a_login_wall(tmp_path):
    config = _config(tmp_path, {"physicist": _NOTEBOOK}, method="token")

    findings = lint_web_terminals(config, project_root=tmp_path)

    assert not [f for f in findings if f.code == _RULE]
