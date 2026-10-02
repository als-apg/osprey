"""The file-backed paradigms' terminology tables are rendered from the build's facts.

Three partials — ``_terminology/{hierarchical,in_context,middle_layer}.md.j2``
— tell the channel-finder subagent what a facility's devices are called. A
framework prompt cannot know a facility's vocabulary, and a wrong device token
returns no rows and no error, so the rows have one source: the device classes
the build writes into ``data/facility_facts.json``, read into the render context
as ``facility_facts``. This file holds the three file-backed paradigms to that:

* every alias and every family of every class the facts carry reaches the
  table, read from the same file the render reads rather than spelled out;
* a build that holds no device class gets the one-line statement saying so, and
  no device token at all — never a quiet fallback to the demo machine's words;
* the ``.j2`` sources carry none of the tokens and name no config key, so the
  only route into a rendered prompt is the facts; and
* no partial under ``_terminology/`` names a protocol word, with the demo facts
  or with none.

The guards run against the terminology section alone. The rest of
``channel-finder.md.j2`` carries its own paradigm prose; slicing keeps this
file's failures about this file's subject.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.templates.claude_code import build_claude_code_context
from osprey.cli.templates.manager import TemplateManager
from osprey.facility.views.facts import FACTS_FILE, zero_source_facts
from tests._vocabulary import PROTOCOL_WORDS


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, facts: Path | None, **kwargs) -> Path:
    """``create_project`` plus the steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, writes
    the facility's facts and regenerates ``.claude/`` from the finished project.
    These fixtures render from a bundle rather than from a profile, so they
    overlay the preset ``osprey init`` pairs with that bundle and copy in the
    facts file *facts* (none: the project holds no facts file, which the render
    reads as a facility with no sources).
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", _bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    if facts is not None:
        (project / "data").mkdir(exist_ok=True)
        shutil.copyfile(facts, project / "data" / FACTS_FILE)
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` before the preset's block and the facts landed.
    manager.regenerate_claude_code(project)
    return project


#: The paradigms whose terminology table is a partial in the template tree. The
#: ``graph`` paradigm derives its vocabulary from the seeded store instead, and
#: is guarded by ``tests/cli/test_channel_finder_graph_tools.py``.
FILE_BACKED_MODES = ("hierarchical", "in_context", "middle_layer")

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/claude_code/claude/agents/_terminology"
)

#: Every partial under ``_terminology/``, as the Jinja environment names it.
PARTIALS = tuple(
    f"claude_code/claude/agents/_terminology/{path.name}"
    for path in sorted(_TEMPLATE_ROOT.glob("*.md.j2"))
)

#: Device tokens of the demo machine. None may appear as literal template text,
#: and none may reach a render whose build holds no device class.
FORBIDDEN_TOKENS = ("DCCT", "BCM", "BPM", "HCM", "VCM", "QF", "QD", "TC", "VGC", "CCG")

#: The statement a render whose build holds no device class carries instead of rows.
ZERO_CLASS_LINE = "This facility's build holds no device class."

#: What the table says about where its rows come from.
FROM_THE_BUILD = "read from this facility's build"

_SECTION_HEADING = "## Channel Database Terminology"

# The real-build cases read the session's one control-assistant build, so they
# share its worker and the build runs once per run.
_SHARES_THE_BUILD = pytest.mark.xdist_group("built_control_assistant")


def _demo_facts_path(built: Any) -> Path:
    """The facts file of the session's control-assistant build."""
    return built.build_dir / "data" / FACTS_FILE


def _device_classes(project_dir: Path) -> dict[str, dict[str, Any]]:
    """The device classes the render read, from the project's own facts file."""
    facts = json.loads((project_dir / "data" / FACTS_FILE).read_text(encoding="utf-8"))
    classes: dict[str, dict[str, Any]] = facts["device_classes"]
    assert classes, "the demo build's facts carry no device class"
    return classes


def _terminology_section(project_dir: Path) -> str:
    """The rendered terminology section, without the rest of the agent file."""
    text = (project_dir / ".claude" / "agents" / "channel-finder.md").read_text(encoding="utf-8")
    start = text.find(_SECTION_HEADING)
    assert start != -1, "the rendered channel finder has no terminology section"
    rest = text[start + len(_SECTION_HEADING) :]
    end = re.search(r"(?m)^## ", rest)
    assert end, "the terminology section is no longer followed by a heading; re-derive the slice"
    return _SECTION_HEADING + rest[: end.start()]


def _forbidden_hits(text: str) -> list[str]:
    """Which of the demo device tokens appear in *text*, word-bounded.

    Word boundaries keep ``TC`` from matching inside ``MATCH`` and ``QF`` from
    matching inside a longer family token, so a hit is a real device token
    rather than a substring of ordinary prose.
    """
    return [token for token in FORBIDDEN_TOKENS if re.search(rf"\b{token}\b", text)]


def _project(
    tmp_path: Path, name: str, mode: str, facts: Path | None
) -> tuple[TemplateManager, Path]:
    """A control-assistant project in *mode*, rendered the way the CLI renders one."""
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        facts,
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": mode, "deploy_services": True},
    )
    return manager, project_dir


# ---------------------------------------------------------------------------
# With device classes: the rows are the facts'
# ---------------------------------------------------------------------------


@_SHARES_THE_BUILD
@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_every_alias_and_family_reaches_the_table(tmp_path, built_control_assistant, mode):
    """The table's content is the build's facts, class by class.

    Derived rather than spelled: the expectation is read from the facts file the
    render reads, so a facility that gains a class gains a row here too and this
    test cannot drift from the table it guards.
    """
    _manager, project_dir = _project(
        tmp_path, f"cf-vocab-{mode}", mode, _demo_facts_path(built_control_assistant)
    )
    section = _terminology_section(project_dir)

    for name, entry in _device_classes(project_dir).items():
        assert f"(class {name})" in section or f"Class {name}:" in section, (
            f"{mode}: class {name} has no row"
        )
        for alias in entry["aliases"]:
            assert f'"{alias}"' in section, f"{mode}: alias {alias!r} of class {name} is missing"
        for family in entry["families"]:
            assert f"`{family}`" in section, f"{mode}: family {family} of class {name} is missing"


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_a_class_with_no_family_or_alias_still_has_a_row(tmp_path, mode):
    """A class no family holds is a row that says so, not a navigation target.

    A facility-added class can carry no device yet, and a class's devices can
    sit in no group; either way the row names the class and sends the agent to
    names and descriptions instead of to a family that does not exist.
    """
    facts = zero_source_facts({"code": "lab", "name": "lab", "description": None})
    facts["device_classes"] = {"Spare": {"count": 0, "aliases": [], "families": []}}
    facts_path = tmp_path / FACTS_FILE
    facts_path.write_text(json.dumps(facts), encoding="utf-8")

    _manager, project_dir = _project(tmp_path, f"cf-spare-{mode}", mode, facts_path)
    section = _terminology_section(project_dir)

    assert "| Spare | Class Spare: no family holds its devices;" in section
    assert ZERO_CLASS_LINE not in section


@_SHARES_THE_BUILD
@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_the_table_says_it_was_read_from_the_build(tmp_path, built_control_assistant, mode):
    """An operator reading the prompt is told where the vocabulary came from."""
    _manager, project_dir = _project(
        tmp_path, f"cf-provenance-{mode}", mode, _demo_facts_path(built_control_assistant)
    )
    section = _terminology_section(project_dir)

    assert FROM_THE_BUILD in section
    assert ZERO_CLASS_LINE not in section
    # The routing rows are the paradigm's own guidance, not vocabulary.
    assert '| "readback" / "monitor" |' in section
    assert '| "setpoint" / "control" |' in section


# ---------------------------------------------------------------------------
# With no device class: one honest line, and no borrowed vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_a_build_with_no_device_class_renders_the_zero_class_line(tmp_path, mode):
    """With no class in the facts the table says so, and names no device at all."""
    _manager, project_dir = _project(tmp_path, f"cf-silent-{mode}", mode, None)

    section = _terminology_section(project_dir)
    assert ZERO_CLASS_LINE in section
    assert _forbidden_hits(section) == [], (
        f"{mode}: a render whose build holds no device class still names device "
        "tokens — the demo machine's vocabulary has leaked into the prompt"
    )
    # The paradigm's own routing guidance is not vocabulary, and stays.
    assert '| "readback" / "monitor" |' in section


def test_no_device_class_leaves_no_device_token_anywhere_in_the_prompt(tmp_path):
    """The guard covers the whole agent file, not only its terminology table.

    A subagent reads the file top to bottom, so a token the table no longer
    claims is still a token the agent will try, and a family that does not
    exist here returns no rows and no error.
    """
    _manager, project_dir = _project(tmp_path, "cf-whole-file", "middle_layer", None)

    rendered = (project_dir / ".claude" / "agents" / "channel-finder.md").read_text(
        encoding="utf-8"
    )
    assert _forbidden_hits(rendered) == [], (
        "the rendered channel-finder prompt names device tokens the build's facts did not put there"
    )


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------


def test_the_agent_source_spells_no_device_token():
    """The agent template itself is a source of tokens exactly like the partials."""
    source = (_TEMPLATE_ROOT.parent / "channel-finder.md.j2").read_text(encoding="utf-8")
    assert _forbidden_hits(source) == [], (
        "channel-finder.md.j2 spells a device token itself. The vocabulary has "
        "one source — the build's facts, through `facility_facts`."
    )


@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_the_partial_source_spells_no_device_token_and_no_config_key(mode):
    """Tokens may only arrive through the facts, never as template text."""
    source = (_TEMPLATE_ROOT / f"{mode}.md.j2").read_text(encoding="utf-8")
    assert _forbidden_hits(source) == [], (
        f"_terminology/{mode}.md.j2 spells a device token itself. The vocabulary "
        "has one source — the build's facts, through `facility_facts`."
    )
    assert "`facility." not in source, f"_terminology/{mode}.md.j2 names a config key"


@_SHARES_THE_BUILD
@pytest.mark.parametrize("partial", PARTIALS)
@pytest.mark.parametrize("facts_of", ("demo", "zero classes"))
def test_no_partial_names_a_protocol_word(built_control_assistant, partial, facts_of):
    """The rendered partial says "channel" and "channel address", whatever the facts."""
    facts = zero_source_facts({"code": "lab", "name": "lab", "description": None})
    if facts_of == "demo":
        facts = json.loads(_demo_facts_path(built_control_assistant).read_text(encoding="utf-8"))

    rendered = TemplateManager().jinja_env.get_template(partial).render(facility_facts=facts)

    assert PROTOCOL_WORDS.search(rendered) is None, (
        f"{partial} with the {facts_of} facts names {PROTOCOL_WORDS.search(rendered).group(0)!r}"
    )


@_SHARES_THE_BUILD
@pytest.mark.parametrize("mode", FILE_BACKED_MODES)
def test_the_scaffold_render_equals_the_builds_agent_file(tmp_path, built_control_assistant, mode):
    """``osprey scaffold diff agents/channel-finder`` finds nothing to report.

    The scaffold command renders the agent template from
    ``build_claude_code_context`` and diffs it against the file on disk, so the
    two renders must agree byte for byte, terminology table included.
    """
    manager, project_dir = _project(
        tmp_path, f"cf-scaffold-{mode}", mode, _demo_facts_path(built_control_assistant)
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))

    ctx = build_claude_code_context(manager.template_root, manager.jinja_env, project_dir, config)
    rendered = manager.jinja_env.get_template("claude_code/claude/agents/channel-finder.md.j2")

    assert (
        rendered.render(**ctx).encode("utf-8")
        == (project_dir / ".claude" / "agents" / "channel-finder.md").read_bytes()
    )
