"""The ``pyaml-tools`` skill and the prompt's measurement section follow the pyAML view.

Both render from the build's agent facts: the skill lists each model's view at
its fixed path ``data/pyaml/<model>/configuration.yaml``, sends every
measurement through ``pyaml_measure``, keeps ``execute`` for reads on
``sr.design`` and ``sr.live``, and says a single-pass model has no design
simulator. Its kinds, each kind's step keys and which kinds take
``correctors`` and ``by`` are the per-model kind blocks of the facts, never a
list of their own. A render without a pyAML view renders neither.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.utils.workspace import TIERS_DIRNAME

SKILL = "claude_code/claude/skills/pyaml-tools/SKILL.md.j2"
PROMPT = "claude_code/CLAUDE.md.j2"
PRESETS = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"


def _kind(description: str, keys: list[str], *, correctors: bool = False, by: bool = False):
    return {
        "description": description,
        "tool_keys": keys,
        "corrector_roles": ["hcor", "vcor"] if correctors else [],
        "takes_by": by,
    }


#: The kind blocks each model's facts record carries, as the build writes them.
KINDS = {
    "SR": {
        "orm": _kind("orbit response", ["corrector_delta"], correctors=True),
        "trm": _kind("tune response", ["quad_delta"], by=True),
    },
    "LINE": {"line_scan": _kind("a registered kind", ["corrector_delta"], correctors=True)},
}


def _facts(**models: str | None) -> dict[str, Any]:
    return {
        "measurement_models": {
            name: {
                "path": f"data/pyaml/{name}/configuration.yaml",
                "sha256": "0" * 64,
                "kinds": KINDS.get(name, {}),
            }
            for name in models
        },
        "models": [
            {"name": name, "engine": "pyat", "served": True, "solve": solve}
            for name, solve in models.items()
        ],
    }


def _skill(view: bool, **models: str | None) -> str:
    template = TemplateManager().jinja_env.get_template(SKILL)
    return template.render(pyaml_view_present=view, facility_facts=_facts(**models))


def _prompt(view: bool, servers: set[str]) -> str:
    template = TemplateManager().jinja_env.get_template(PROMPT)
    return template.render(
        pyaml_view_present=view,
        enabled_servers=servers,
        agents=[],
        agent_data_root="var/agent_data",
    )


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


@pytest.fixture
def rendered() -> str:
    return _skill(True, SR="periodic", LINE="single_pass")


def test_the_skill_names_each_view_at_its_fixed_path(rendered: str):
    assert "- `SR`: `data/pyaml/SR/configuration.yaml`\n" in rendered
    assert (
        "- `LINE`: `data/pyaml/LINE/configuration.yaml` (single pass: no design simulator)\n"
        in (rendered)
    )


def test_the_skill_has_its_frontmatter(rendered: str):
    _, block, _ = rendered.split("---", 2)
    front = yaml.safe_load(block)
    assert front["name"] == "pyaml-tools"
    assert front["summary"]


def test_every_measurement_goes_through_pyaml_measure_and_execute_reads(rendered: str):
    text = _flat(rendered)
    assert "Every measurement goes through `mcp__python__pyaml_measure`." in text
    assert "Never step setpoints for a measurement in `mcp__python__execute`" in text
    assert "`sr.live` reads the machine; `sr.design` computes on the design lattice" in text
    assert "Reads only: a pyAML write in `execute` is refused" in text


def test_a_single_pass_model_reads_live_only(rendered: str):
    text = _flat(rendered)
    assert (
        "`LINE` is solved single pass: its view carries `simulators: []`, so it has no "
        "`sr.design`. Its reads are `sr.live` only."
    ) in text
    assert "`SR` is solved single pass" not in text


def test_the_group_knob_is_one_integrated_strength_per_member(rendered: str):
    text = _flat(rendered)
    assert "`by: group` moves every magnet of the group by the same integrated strength" in text
    assert "The group column is per unit integrated strength: the sum of its members'" in text
    assert "`tune excursion`, `chromaticity excursion`, `magnet step` or `max_step`" in text


def test_the_kinds_and_their_arguments_are_the_facts_kind_blocks(rendered: str):
    text = _flat(rendered)
    assert "`kind`: one of the kinds below." in text
    assert "- `line_scan`: a registered kind; `corrector_delta`." in text
    assert "- `orm`: orbit response; `corrector_delta`." in text
    assert "- `trm`: tune response; `quad_delta`." in text
    assert "for `line_scan` and `orm` only; omit for every corrector." in text
    assert "as one column; `trm` only; omit for magnet." in text
    for spelled in ("`dispersion`", "`chromaticity_monitor`"):
        assert spelled not in text


def test_a_render_whose_kinds_take_no_correctors_names_no_correctors():
    only_trm = (
        TemplateManager()
        .jinja_env.get_template(SKILL)
        .render(
            pyaml_view_present=True,
            facility_facts={
                "measurement_models": {
                    "SR": {"path": "p", "sha256": "0", "kinds": {"trm": KINDS["SR"]["trm"]}}
                },
                "models": [],
            },
        )
    )

    assert "`correctors`:" not in only_trm
    assert "`by`:" in only_trm


def test_step_sizes_point_at_the_authored_limits(rendered: str):
    assert "`max_step` in `data/facility/limits.yaml`" in rendered
    assert "channel_limits.json" not in rendered


def test_without_a_view_the_skill_renders_empty():
    assert _skill(False, SR="periodic").strip() == ""


def test_the_prompt_section_follows_the_view_and_the_python_server():
    section = "## Measurements"
    shown = _prompt(True, {"python"})
    assert section in shown
    assert "`mcp__python__pyaml_measure`" in shown
    assert "`mcp__python__model_response_matrix`" in shown
    assert section not in _prompt(False, {"python"})
    assert section not in _prompt(True, set())


def test_the_control_room_preset_installs_the_skill_and_sandboxless_renders_drop_it():
    base = yaml.safe_load((PRESETS / "demo.yml").read_text(encoding="utf-8"))
    assert "pyaml-tools" in base["skills"]
    # An app's lists are its whole selection, so a sandboxless app drops the
    # skill by not listing it.
    for app in ("channel-finder", "knowledge", "logbook"):
        preset = yaml.safe_load(
            (PRESETS / TIERS_DIRNAME / f"{app}.yml").read_text(encoding="utf-8")
        )
        assert "pyaml-tools" not in preset["skills"], app


def test_the_model_section_names_what_has_no_solve(rendered: str):
    section = rendered.split("## Read the model through `model_response_matrix`", 1)[1]
    section = _flat(section.split("\n---\n", 1)[0])
    assert "`missing`" in section
    assert "never read a missing column as zero response" in section
