"""The three commands that name the features whose own model call has no API key.

``osprey build``, ``osprey validate`` and ``osprey web`` each read
:func:`osprey.models.inner_calls.missing_key_warnings` and print what it says.
These tests pin that each one prints the same sentences, that none of them
changes its exit status over a missing key, and that the view ``osprey
validate`` reads agrees with the config the framework template renders.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import web_cmd
from osprey.cli.build_cmd import model_call_view
from osprey.cli.build_environment import credential_environ, report_inner_call_keys
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate
from osprey.models.inner_calls import inner_model_calls
from tests._config_render_context import PROVIDER_CATALOG, render_config

NAMING = "session naming needs ANTHROPIC_API_KEY (not set): sessions keep their first-prompt title"
COMPOSE = (
    "logbook compose needs ANTHROPIC_API_KEY (not set): the compose panel cannot draft an entry"
)

CONFIG: dict[str, Any] = {
    "project_name": "demo",
    "claude_code": {"provider": "anthropic"},
    "api": {"providers": {"anthropic": {"api_key": "${ANTHROPIC_API_KEY}"}}},
}


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    """A developer's own key must not answer for the deployment under test."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo with a render on ``anthropic`` and no key anywhere."""
    root = tmp_path / "demo"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return root


# --- the environment a key is looked for in -------------------------------------------


def test_the_repo_env_file_supplies_the_key(repo: Path):
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-ant\nEMPTY_ONE=\n", encoding="utf-8")
    environ = credential_environ(repo / "build", profile_dir=repo)
    assert environ["ANTHROPIC_API_KEY"] == "sk-ant"
    assert "EMPTY_ONE" not in environ


def test_a_build_zone_env_file_outranks_the_repo_and_the_shell(repo: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    (repo / ".env").write_text("ANTHROPIC_API_KEY=from-the-repo\n", encoding="utf-8")
    assert credential_environ(repo / "build", profile_dir=repo)["ANTHROPIC_API_KEY"] == (
        "from-the-repo"
    )
    (repo / "build" / ".env").write_text("ANTHROPIC_API_KEY=from-the-build\n", encoding="utf-8")
    assert credential_environ(repo / "build", profile_dir=repo)["ANTHROPIC_API_KEY"] == (
        "from-the-build"
    )


# --- osprey build ---------------------------------------------------------------------


def test_build_warns_once_per_feature_and_names_the_env_file(repo: Path, caplog):
    with caplog.at_level(logging.WARNING):
        warnings = report_inner_call_keys(CONFIG, repo / "build", profile_dir=repo)

    assert warnings == [NAMING, COMPOSE]
    lines = [record.getMessage() for record in caplog.records]
    assert lines[:2] == [f"  ✗ {NAMING}", f"  ✗ {COMPOSE}"]
    assert str(repo / ".env") in lines[2]


def test_build_says_nothing_when_the_key_is_set(repo: Path, caplog):
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-ant\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert report_inner_call_keys(CONFIG, repo / "build", profile_dir=repo) == []
    assert caplog.records == []


# --- osprey web -----------------------------------------------------------------------


def test_web_names_the_features_as_warnings(repo: Path):
    render = repo / "build"
    assert web_cmd._probe_inner_call_keys(render, repo, render / "config.yml") == [NAMING, COMPOSE]


def test_web_reads_the_key_from_the_repo_env_file(repo: Path):
    (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-ant\n", encoding="utf-8")
    render = repo / "build"
    assert web_cmd._probe_inner_call_keys(render, repo, render / "config.yml") == []


def test_web_is_silent_on_a_config_it_cannot_read(repo: Path):
    render = repo / "build"
    (render / "config.yml").write_text("claude_code: [unclosed", encoding="utf-8")
    assert web_cmd._probe_inner_call_keys(render, repo, render / "config.yml") == []
    assert web_cmd._probe_inner_call_keys(render, repo, render / "absent.yml") == []


def test_web_preflight_keeps_the_agent_line_then_names_the_features(repo: Path, monkeypatch):
    """A missing key on a provider with a login launches: warnings, no failure."""
    monkeypatch.setattr(web_cmd, "_probe_companion_ports", lambda: [])
    render = repo / "build"

    failures, warnings = web_cmd._preflight(repo, render, render / "config.yml")

    assert failures == []
    assert "the agent falls back to subscription/OAuth login" in warnings[0]
    assert warnings[1:] == [NAMING, COMPOSE]


# --- osprey validate ------------------------------------------------------------------


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real materialization of the demo preset (provider ``anthropic``)."""
    root = tmp_path_factory.mktemp("inner-call-keys") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "demo", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def deployment(materialized: Path, tmp_path: Path) -> Path:
    target = tmp_path / "deployment"
    shutil.copytree(materialized, target)
    return target


def test_validate_lists_the_features_and_still_passes(deployment: Path):
    result = CliRunner().invoke(validate, ["--repo", str(deployment)])

    assert result.exit_code == 0, result.output
    assert f"⚠ {NAMING}" in result.output
    assert f"⚠ {COMPOSE}" in result.output
    assert "Profile is valid" in result.output


def test_validate_says_nothing_once_the_key_is_in_the_repo_env_file(deployment: Path):
    with (deployment / ".env").open("a", encoding="utf-8") as env_file:
        env_file.write("\nANTHROPIC_API_KEY=sk-ant\n")

    result = CliRunner().invoke(validate, ["--repo", str(deployment)])

    assert result.exit_code == 0, result.output
    assert "needs ANTHROPIC_API_KEY" not in result.output


# --- the view validate reads is the config the template renders ----------------------


@pytest.mark.parametrize("ariel_server_on", [False, True])
@pytest.mark.parametrize("pipeline", [None, "in_context", "hierarchical"])
def test_the_validate_view_lists_what_the_rendered_config_lists(pipeline, ariel_server_on):
    """``model_call_view`` restates template lines; the template is the authority."""
    context: dict[str, Any] = {"default_provider": "house", "ariel_server_on": ariel_server_on}
    if pipeline:
        context |= {"default_pipeline": pipeline, f"enable_{pipeline}": True}
    rendered = yaml.safe_load(render_config(**context))

    view = model_call_view(
        provider="house",
        pipeline=pipeline,
        ariel_server_on=ariel_server_on,
        provider_catalog=PROVIDER_CATALOG,
        config={},
    )

    for key in ("claude_code", "channel_finder", "logbook", "ariel"):
        rendered_block, view_block = rendered.get(key) or {}, view.get(key) or {}
        if key == "channel_finder":
            rendered_block = {"pipeline_mode": rendered_block.get("pipeline_mode")}
            view_block = {"pipeline_mode": view_block.get("pipeline_mode")}
        if key == "claude_code":
            rendered_block = {"provider": rendered_block.get("provider")}
        if key == "ariel":
            # The template also states the model; the list reads the provider.
            for block in (rendered_block, view_block):
                module = (block.get("enhancement_modules") or {}).get("semantic_processor") or {}
                module.pop("model", None)
        assert view_block == rendered_block, key
    assert view["api"]["providers"] == rendered["api"]["providers"]


def test_the_profiles_config_block_is_laid_over_the_derived_keys():
    view = model_call_view(
        provider="anthropic",
        pipeline="in_context",
        ariel_server_on=False,
        provider_catalog={"anthropic": {"api_key": "${ANTHROPIC_API_KEY}"}},
        config={
            "channel_finder.pipelines.in_context.subagent_provider": "cborg",
            "ariel.enhancement_modules.image_caption": {"enabled": True, "provider": "cborg"},
        },
    )
    listed = {call.feature: call.provider for call in inner_model_calls(view, environ={})}
    assert listed["in-context channel finder"] == "cborg"
    assert listed["ARIEL image captions"] == "cborg"
    assert listed["session naming"] == "anthropic"


# --- the page quotes the sentence the commands print ----------------------------------


def test_the_personal_login_page_quotes_a_warning_the_list_produces():
    page = Path(__file__).resolve().parents[2] / "docs/source/how-to/web-terminal/operate.rst"
    assert f"``{NAMING}``" in " ".join(page.read_text(encoding="utf-8").split())
