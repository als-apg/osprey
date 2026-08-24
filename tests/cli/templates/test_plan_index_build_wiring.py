"""Tests for the plan index reaching the Claude Code template context.

The index is built inside ``build_claude_code_context`` rather than by a
caller, because ``regenerate_claude_code`` rebuilds the context from
``config.yml`` alone and runs last in the build: an index handed only to
``create_project``'s render would be overwritten and ship empty. The
regeneration test below is the one that pins that — it captures the context the
regeneration path actually hands the renderer.

The rest pin the fail-soft contract: a malformed plan file and an unreadable
plan directory are build warnings that name the offender, never build failures,
and a build with no plan directory configured at all still succeeds.
"""

import logging

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager


def _plan_source(name, description, writes):
    """A plan file's source -- metadata only; the index never imports it."""
    return (
        '"""A plan fixture."""\n\n'
        "PLAN_METADATA = {\n"
        f"    'name': {name!r},\n"
        f"    'description': {description!r},\n"
        f"    'writes': {writes!r},\n"
        "}\n"
    )


def _make_project(tmp_path, name="plan-index-wiring", *, bluesky_enabled=True, config_edits=None):
    """Create a project and apply config edits, returning (manager, project_dir).

    ``config_edits`` receives the parsed ``config.yml`` mapping and mutates it
    in place; the result is written back, so a later regeneration reads exactly
    what a build would have left behind.
    """
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    if bluesky_enabled:
        config.setdefault("claude_code", {}).setdefault("servers", {}).setdefault("bluesky", {})[
            "enabled"
        ] = True
    if config_edits is not None:
        config_edits(config)
    config_path.write_text(yaml.dump(config))
    return manager, project_dir


def _build_ctx(manager, project_dir):
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )


def _write_plan(directory, stem, *, name, description="A fixture plan.", writes=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.py").write_text(_plan_source(name, description, writes))


# ---------------------------------------------------------------------------
# The context carries the index
# ---------------------------------------------------------------------------


def test_context_carries_plan_index_rows(tmp_path):
    ctx = _build_ctx(*_make_project(tmp_path))

    index = ctx["bluesky_plan_index"]
    assert index is not None, "an enabled bluesky server must get a build-time plan index"
    assert index["rows"], "the shipped plans_core layer always yields rows"
    row = index["rows"][0]
    assert set(row) == {"name", "description", "writes", "provenance"}
    assert isinstance(row["writes"], bool)
    assert row["provenance"] == "shipped"
    assert index["overflow"] == 0
    assert index["unreadable_dirs"] == []


def test_facility_plan_dir_rows_reach_the_context(tmp_path):
    plans = tmp_path / "facility_plans"
    _write_plan(plans, "align", name="facility_align", description="Align the thing.")

    def edits(config):
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = str(plans)

    ctx = _build_ctx(*_make_project(tmp_path, config_edits=edits))

    rows = {row["name"]: row for row in ctx["bluesky_plan_index"]["rows"]}
    assert rows["facility_align"]["provenance"] == "facility"
    assert rows["facility_align"]["description"] == "Align the thing."


def test_plan_index_absent_when_bluesky_server_disabled(tmp_path):
    ctx = _build_ctx(*_make_project(tmp_path, bluesky_enabled=False))

    assert ctx["bluesky_plan_index"] is None, (
        "with no bluesky tools the skill renders nothing; no index means the pass never ran"
    )


def test_empty_index_is_not_collapsed_into_absent(tmp_path):
    """A pass that resolved no plans still yields an index — with no rows.

    The skill has to tell "no plans were visible at build time" apart from "the
    pass never ran", so the empty case must not degrade to ``None``.
    """
    shipped = {
        row["name"] for row in _build_ctx(*_make_project(tmp_path))["bluesky_plan_index"]["rows"]
    }

    def edits(config):
        config.setdefault("bluesky", {})["excluded_plans"] = sorted(shipped)

    ctx = _build_ctx(*_make_project(tmp_path, name="plan-index-empty", config_edits=edits))

    index = ctx["bluesky_plan_index"]
    assert index is not None
    assert index["rows"] == []


# ---------------------------------------------------------------------------
# The regeneration path — the one that ships the artifact
# ---------------------------------------------------------------------------


def test_regenerate_path_carries_plan_index(tmp_path, monkeypatch):
    """``regenerate_claude_code`` rebuilds the context; the index must survive.

    Regeneration runs last in the build and rebuilds the context from
    ``config.yml`` alone, overwriting the earlier ``create_project`` render. So
    this captures the context regeneration actually hands the renderer (calling
    through to the real render, not stubbing it) and asserts the rows are in it.
    """
    plans = tmp_path / "facility_plans"
    _write_plan(plans, "sweep", name="facility_sweep", description="Sweep the thing.", writes=True)

    def edits(config):
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = str(plans)

    manager, project_dir = _make_project(tmp_path, config_edits=edits)

    captured = {}
    real_render = claude_code.create_claude_code_integration

    def capturing_render(template_root, jinja_env, target_dir, ctx, allowed_outputs=None):
        captured["ctx"] = ctx
        return real_render(template_root, jinja_env, target_dir, ctx, allowed_outputs)

    monkeypatch.setattr(claude_code, "create_claude_code_integration", capturing_render)

    claude_code.regenerate_claude_code(manager.template_root, manager.jinja_env, project_dir)

    index = captured["ctx"]["bluesky_plan_index"]
    assert index is not None, (
        "the regenerated context must carry the index, not just create_project's"
    )
    rows = {row["name"]: row for row in index["rows"]}
    assert rows["facility_sweep"]["writes"] is True
    assert rows["facility_sweep"]["provenance"] == "facility"


# ---------------------------------------------------------------------------
# Fail-soft: warnings, never a failed build
# ---------------------------------------------------------------------------


def test_malformed_plan_file_warns_and_names_the_file(tmp_path, caplog):
    plans = tmp_path / "facility_plans"
    _write_plan(plans, "good", name="facility_good")
    plans.joinpath("broken.py").write_text("PLAN_METADATA = {'name': 'x',\n")

    def edits(config):
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = str(plans)

    manager, project_dir = _make_project(tmp_path, config_edits=edits)

    with caplog.at_level(logging.WARNING, logger="osprey.cli.templates"):
        ctx = _build_ctx(manager, project_dir)

    messages = [record.getMessage() for record in caplog.records]
    assert any("broken.py" in message for message in messages), (
        f"the build warning must name the offending file; got {messages}"
    )
    names = {row["name"] for row in ctx["bluesky_plan_index"]["rows"]}
    assert "facility_good" in names, "one bad file must not cost the whole layer its rows"


def test_unreadable_plan_dir_reaches_the_context_and_warns(tmp_path, caplog):
    missing = tmp_path / "not_there"

    def edits(config):
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = str(missing)

    manager, project_dir = _make_project(tmp_path, config_edits=edits)

    with caplog.at_level(logging.WARNING, logger="osprey.cli.templates"):
        ctx = _build_ctx(manager, project_dir)

    index = ctx["bluesky_plan_index"]
    assert index["unreadable_dirs"] == [str(missing)]
    assert index["rows"], "the other layers' rows survive an unreadable directory"
    assert any(str(missing) in record.getMessage() for record in caplog.records)


def test_build_succeeds_with_no_plan_dir_configured(tmp_path):
    manager, project_dir = _make_project(tmp_path)
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    assert "plan_dir" not in config.get("services", {}).get("bluesky", {})

    ctx = _build_ctx(manager, project_dir)

    assert ctx["bluesky_plan_index"]["unreadable_dirs"] == []


def test_builder_failure_never_aborts_the_build(tmp_path, monkeypatch, caplog):
    """Even an unexpected raise from the builder is a warning, not a failure."""

    def boom(config, repo_root):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr("osprey.cli.templates.plan_index.build_plan_index", boom)

    manager, project_dir = _make_project(tmp_path)
    with caplog.at_level(logging.WARNING, logger="osprey.cli.templates"):
        ctx = _build_ctx(manager, project_dir)

    assert ctx["bluesky_plan_index"] is None
    assert any("builder exploded" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# repo_root is the project root, not the working directory
# ---------------------------------------------------------------------------


def test_relative_plan_dir_resolves_against_the_project_root(tmp_path, monkeypatch):
    """A relative ``plan_dir`` follows the compose bind rule, not the CWD.

    The build does not run from the project root, so resolving against the
    working directory would index the wrong directory (or nothing at all).
    """

    def edits(config):
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = "plans"

    manager, project_dir = _make_project(tmp_path, config_edits=edits)
    _write_plan(project_dir / "plans", "local", name="facility_local")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    ctx = _build_ctx(manager, project_dir)

    names = {row["name"] for row in ctx["bluesky_plan_index"]["rows"]}
    assert "facility_local" in names
    assert ctx["bluesky_plan_index"]["unreadable_dirs"] == []


@pytest.mark.parametrize("configured", [True, False])
def test_index_never_raises(tmp_path, configured):
    """Both a configured-but-broken and an unconfigured project build cleanly."""

    def edits(config):
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = str(
            tmp_path / "nope"
        )

    manager, project_dir = _make_project(
        tmp_path,
        name=f"plan-index-raise-{configured}",
        config_edits=edits if configured else None,
    )

    assert _build_ctx(manager, project_dir)["bluesky_plan_index"] is not None
