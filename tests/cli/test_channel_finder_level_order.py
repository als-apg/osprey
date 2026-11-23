"""The level order of the hierarchical channel finder, as its agent is told it.

``get_options`` refuses an options request whose selections leave out a
required level above the level asked for. The agent has to know that before it
calls: the rendered prompt and the tool's own description both state it, the
levels the prompt says may be left out are exactly those the check lets a
request leave out, and no example either of them carries breaks the rule.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.facility.views.channel_finder import ABSENT, hierarchical_document
from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey_facility.records import FacilityRecords

AGENT_TEMPLATE = "claude_code/claude/agents/channel-finder.md.j2"

#: A facility the build turns into an index whose every level is a tree level.
FACILITY: dict[str, Any] = {
    "places": [
        {"id": "M", "level": "machine"},
        {"id": "M/S1", "level": "sector"},
    ],
    "devices": [{"id": "M/Q1", "class": "Quadrupole", "place": "M/S1"}],
    "channels": [
        {"id": "M:Q1:RB", "on": {"device": "M/Q1"}, "signal": "current", "role": "readback"}
    ],
}

#: A hand-written index with a level marked optional and an instance level.
MIXED: dict[str, Any] = {
    "hierarchy": {
        "levels": [
            {"name": "system", "type": "tree"},
            {"name": "area", "type": "tree", "optional": True},
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        "naming_pattern": "{system}:{area}:{device}:{signal}",
    },
    "tree": {
        "SR": {
            "A1": {
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "D{:02d}", "_range": [1, 3]},
                    "X": {},
                }
            }
        }
    },
}

DATABASES = {
    "built": lambda: hierarchical_document(FacilityRecords(FACILITY)),
    "optional-and-instances": lambda: MIXED,
}


def _rendered_agent(tmp_path: Path, document: dict[str, Any]) -> str:
    """The channel-finder agent a render over *document* writes."""
    (tmp_path / "channels.json").write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / ".osprey-manifest.json").write_text(
        json.dumps({"artifacts": {"hooks": ["writes-check"], "agents": ["channel-finder"]}}),
        encoding="utf-8",
    )
    config = {
        "facility_name": "TestFacility",
        "channel_finder": {
            "pipeline_mode": "hierarchical",
            "pipelines": {"hierarchical": {"database": {"path": "channels.json"}}},
        },
    }
    manager = TemplateManager()
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, tmp_path, config
    )
    assert ctx["channel_finder_hierarchy"] is not None, "the render embedded no hierarchy"
    return manager.jinja_env.get_template(AGENT_TEMPLATE).render(**ctx)


def _section(text: str, heading: str) -> str:
    """The body under *heading*, up to the next heading of the same or higher rank."""
    start = text.index(heading) + len(heading)
    rank = heading.split(" ", 1)[0]
    end = re.search(rf"(?m)^#{{1,{len(rank)}}} ", text[start:])
    return text[start : start + end.start()] if end else text[start:]


def _levels_a_request_may_leave_out(database: HierarchicalChannelDatabase) -> set[str]:
    """Levels above the last that an options request for the last may omit, by asking the check."""
    *above, last = database.hierarchy_levels
    omittable = set()
    chosen: dict[str, str] = {}
    for level in above:
        chosen[level] = database.get_options_at_level(level, chosen)[0]["name"]
    for left_out in above:
        selections = {level: chosen[level] for level in above if level != left_out}
        try:
            database.get_options_at_level(last, selections)
        except ValueError as exc:
            assert repr(left_out) in str(exc)
            continue
        omittable.add(left_out)
    return omittable


@pytest.mark.parametrize("name", DATABASES)
def test_the_prompt_states_the_rule_the_options_check_holds(tmp_path, name):
    document = DATABASES[name]()
    (tmp_path / "db.json").write_text(json.dumps(document), encoding="utf-8")
    database = HierarchicalChannelDatabase(str(tmp_path / "db.json"))
    prompt = _rendered_agent(tmp_path, document)
    step = " ".join(_section(prompt, "### Step 2: Navigate level by level").split())

    assert "`selections` must name every level above the one you ask about" in step
    named = re.search(r"Only (.+?) may be left out\.", step)
    expected = _levels_a_request_may_leave_out(database)
    assert set(re.findall(r"`([^`]+)`", named.group(1)) if named else ()) == expected


@pytest.mark.parametrize("name", DATABASES)
def test_no_navigation_step_tells_the_agent_to_skip_a_level(tmp_path, name):
    prompt = _rendered_agent(tmp_path, DATABASES[name]())
    navigation = _section(prompt, "## Navigation Process")

    assert not re.search(r"(?i)\bskip", navigation), navigation


def _tool_doc(module: str) -> str:
    """The description an MCP client reads for the tool *module* defines."""
    tool = getattr(
        import_module(f"osprey.mcp_server.channel_finder_hierarchical.tools.{module}"), module
    )
    return inspect.getdoc(getattr(tool, "fn", tool)) or ""


def _demo_database(built_demo, tmp_path: Path) -> HierarchicalChannelDatabase:
    path = tmp_path / "demo.json"
    document = hierarchical_document(FacilityRecords(built_demo.facility))
    path.write_text(json.dumps(document), encoding="utf-8")
    return HierarchicalChannelDatabase(str(path))


def test_the_get_options_description_states_the_rule_and_its_example_keeps_it(built_demo, tmp_path):
    doc = " ".join(_tool_doc("get_options").split())
    database = _demo_database(built_demo, tmp_path)

    assert "must name every level above ``level``" in doc
    named = re.search(r"level name to get options for \(e\.g\., (.+?)\)", doc)
    assert named and ast.literal_eval(f"[{named.group(1)}]") == database.hierarchy_levels
    example = re.search(r"Example: (\{.+?\}) when querying the \"(\w+)\" level", doc)
    assert example
    assert database.get_options_at_level(example.group(2), ast.literal_eval(example.group(1)))


def test_the_build_channels_example_names_every_level_and_holds_a_channel(built_demo, tmp_path):
    doc = " ".join(_tool_doc("build_channels").split())
    database = _demo_database(built_demo, tmp_path)

    example = re.search(r"Example: (\{.+?\})", doc)
    assert example
    selections = ast.literal_eval(example.group(1))
    assert list(selections) == database.hierarchy_levels
    assert database.build_existing_channels(selections)["channels"]


def _absent_section(prompt: str, token: str) -> str:
    """The flattened body of the section on the *token* option."""
    return " ".join(_section(prompt, f"### `{token}` options").split())


def test_the_hierarchical_prompt_names_the_absent_option_from_the_code(tmp_path, monkeypatch):
    section = _absent_section(_rendered_agent(tmp_path, DATABASES["built"]()), ABSENT)
    assert f"An option `{ABSENT}` is a real node" in section

    monkeypatch.setattr("osprey.cli.templates.manager.ABSENT", "~")
    prompt = _rendered_agent(tmp_path, DATABASES["built"]())

    assert f"### `{ABSENT}` options" not in prompt
    assert "An option `~` is a real node" in _absent_section(prompt, "~")


def test_the_template_spells_no_absent_literal():
    source = (TemplateManager().template_root / AGENT_TEMPLATE).read_text(encoding="utf-8")

    assert f"`{ABSENT}`" not in source


def test_the_sentence_points_at_the_option_description(tmp_path):
    section = _absent_section(_rendered_agent(tmp_path, DATABASES["built"]()), ABSENT)

    assert "its description says which" in section


def test_the_absent_option_is_not_the_selection_for_an_unmentioned_level(tmp_path):
    section = _absent_section(_rendered_agent(tmp_path, DATABASES["built"]()), ABSENT)

    assert "Select it only when what the question asks for really has no value" in section
    assert "A level the question does not mention is not that case" in section
    assert "select every one that applies" in section
    assert "names none" not in section
