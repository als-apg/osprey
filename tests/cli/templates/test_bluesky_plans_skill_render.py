"""Render tests for the ``bluesky-plans`` routing skill template.

The skill is a Jinja template rendered at build time from
``ctx["bluesky_plan_index"]``, so almost everything worth pinning is a
rendering property: that it disappears when the Bluesky server is off, that a
build which resolved no plans produces an honest sentence rather than an empty
table, and that the ``None`` (pass never ran) and ``{"rows": []}`` (pass ran,
found nothing) inputs stay distinguishable.

Two tests here are not about rendering:

* ``test_description_shares_no_long_phrase_with_a_sibling`` guards the routing
  boundary. ``bluesky-plans`` is a broad family name sitting above two action
  skills that own the real procedures; a description sharing trigger wording
  with either would pull requests away from the skill that can act on them.
* ``test_every_tool_name_is_real`` fails on an invented tool. Prose naming a
  tool that does not exist sends the agent at a dead surface, which is exactly
  the failure a routing skill is in a position to cause.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from osprey.bluesky_tool_names import ALL_TOOLS
from osprey.cli.templates.manager import TemplateManager

TEMPLATE_REL = "claude_code/claude/skills/bluesky-plans/SKILL.md.j2"
REPO_ROOT = Path(__file__).parents[3]
SKILLS_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "claude_code" / "claude" / "skills"
#: Read as text rather than imported: importing the tool module registers MCP
#: tools, and this test only needs the ceiling constant's literal value.
READ_TOOLS_SOURCE = (
    REPO_ROOT / "src" / "osprey" / "mcp_server" / "bluesky" / "tools" / "read_tools.py"
)

#: Backticked terms in the skill that are deliberately not MCP tool names:
#: plan-file symbols, a config key, tool argument and response field names, and
#: the provenance tiers. Anything else snake_case in backticks must be a real
#: tool.
KNOWN_NON_TOOLS = {
    "bluesky.plan_module",
    "build_plan",
    "draft_revision",
    "facility",
    "max_chars",
    "params",
    "preset",
    "set_draft",
    "shipped",
    "truncated",
}

ROWS = [
    {
        "name": "setpoint_sweep",
        "description": "Step a setpoint across a range and record a readback.",
        "writes": True,
        "provenance": "shipped",
    },
    {
        "name": "channel_survey",
        "description": "Read a set of channels once and store the result.",
        "writes": False,
        "provenance": "facility",
    },
]


def render(index: dict | None, *, servers: set[str] | None = None) -> str:
    """Render the skill with the given plan index and enabled-server set."""
    template = TemplateManager().jinja_env.get_template(TEMPLATE_REL)
    return template.render(
        enabled_servers={"bluesky", "controls"} if servers is None else servers,
        bluesky_plan_index=index,
    )


def full_index(**overrides) -> dict:
    """A plan index dict in the shape the builder hands to the template."""
    return {"rows": ROWS, "overflow": 0, "unreadable_dirs": [], **overrides}


def table_rows(text: str) -> list[str]:
    """The rendered index table's data rows (header and rule excluded)."""
    return [
        line
        for line in text.splitlines()
        if line.startswith("|") and not line.startswith("|--") and "| Plan |" not in line
    ]


def unwrapped(text: str) -> str:
    """Rendered text with line wrapping collapsed, for whole-sentence assertions."""
    return re.sub(r"\s+", " ", text)


def frontmatter(text: str) -> dict:
    """Parse the rendered skill's YAML frontmatter."""
    _, block, _ = text.split("---", 2)
    return yaml.safe_load(block)


def sibling_description(name: str) -> str:
    """The ``description`` of another bundled skill, template or not."""
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.exists():
        path = SKILLS_DIR / name / "SKILL.md.j2"
    text = path.read_text(encoding="utf-8")
    # A Jinja-guarded skill opens with the guard, not the frontmatter fence.
    text = text[text.index("---") :]
    _, block, _ = text.split("---", 2)
    return yaml.safe_load(block)["description"]


def ngrams(description: str, n: int = 6) -> set[tuple[str, ...]]:
    """Every n-word window of a description, punctuation and case removed."""
    words = re.findall(r"[a-z0-9_-]+", description.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class TestEnablementGuard:
    def test_renders_empty_without_the_bluesky_server(self):
        """With no bluesky server the build deletes the file, which needs a blank render."""
        assert render(None, servers={"controls"}).strip() == ""

    def test_renders_when_the_bluesky_server_is_on(self):
        assert "bluesky-plans" in render(full_index())


class TestPlanIndexRendering:
    def test_index_none_names_list_plans_and_draws_no_table(self):
        """``None`` means the index pass never ran — never an empty table."""
        text = render(None)
        assert "`list_plans`" in text
        assert table_rows(text) == []

    def test_empty_rows_says_no_plans_were_visible_and_draws_no_table(self):
        """A pass that ran and resolved nothing is distinct from one that never ran."""
        text = render(full_index(rows=[]))
        assert "No plans were visible" in text
        assert "`list_plans`" in text
        assert table_rows(text) == []

    def test_one_row_per_plan_naming_what_it_does_and_whether_it_moves_anything(self):
        text = render(full_index())
        rows = table_rows(text)
        assert len(rows) == len(ROWS)
        sweep, survey = rows
        assert "setpoint_sweep" in sweep
        assert "Step a setpoint across a range and record a readback." in sweep
        assert "| yes |" in sweep
        assert "shipped" in sweep
        assert "channel_survey" in survey
        assert "| no |" in survey
        assert "facility" in survey

    def test_descriptions_are_not_truncated_again(self):
        """The builder already caps descriptions; the skill must render them whole."""
        long_description = "x" * 120
        text = render(full_index(rows=[{**ROWS[0], "description": long_description}]))
        assert long_description in text

    def test_overflow_adds_a_final_row_pointing_at_list_plans(self):
        text = render(full_index(overflow=7))
        rows = table_rows(text)
        assert len(rows) == len(ROWS) + 1
        assert "7 further plans" in rows[-1]
        assert "list_plans" in rows[-1]

    def test_no_overflow_row_when_nothing_overflowed(self):
        assert len(table_rows(render(full_index()))) == len(ROWS)

    def test_unreadable_plan_directory_is_stated_plainly(self):
        text = render(full_index(unreadable_dirs=["/opt/facility_plans"]))
        assert "could not be read" in text
        assert "no facility plans" in text

    def test_nothing_about_unreadable_directories_when_all_were_readable(self):
        assert "could not be read" not in render(full_index())


class TestHonestyAboutTheSnapshot:
    """Every caveat that keeps the table a choosing aid rather than an authority."""

    @pytest.mark.parametrize("index", [None, {"rows": [], "overflow": 0, "unreadable_dirs": []}])
    def test_list_plans_is_named_however_the_index_resolved(self, index):
        assert "`list_plans`" in render(index)

    def test_list_plans_is_required_before_set_draft(self):
        text = render(full_index())
        assert "`list_plans` is the authority" in text
        assert "`set_draft`" in text

    def test_says_the_listing_is_a_build_time_snapshot(self):
        assert "snapshot taken at build time" in render(full_index())

    def test_says_the_listing_is_metadata_level_and_a_plan_may_be_quarantined(self):
        text = render(full_index())
        assert "metadata-level" in text
        assert "quarantined" in text

    def test_names_the_absent_sources(self):
        text = render(full_index())
        assert "authored during this session" in text
        assert "`bluesky.plan_module`" in text

    def test_says_the_deployed_bridge_may_run_a_different_image(self):
        assert "pinned image" in render(full_index())


class TestRouting:
    def test_routes_reuse_to_operating_and_authoring_to_writing(self):
        text = render(full_index())
        reuse = text.index("`operating-bluesky-plans`")
        author = text.index("`writing-bluesky-plans`")
        assert reuse < author, "reuse is the preferred path and must be presented first"

    def test_names_get_plan_source_for_questions_about_a_parameter(self):
        assert "`get_plan_source(name)`" in render(full_index())

    def test_truncated_source_advice_stops_at_the_tool_s_ceiling(self):
        """``get_plan_source`` clamps ``max_chars``, so "ask for more" ends at 200000.

        Unqualified "ask again with a larger max_chars" is unfollowable at the
        ceiling: the same response comes back however much more is asked for.
        """
        source = READ_TOOLS_SOURCE.read_text(encoding="utf-8")
        assert "_SOURCE_MAX_MAX_CHARS = 200_000" in source, "the ceiling moved; update the skill"
        text = render(full_index())
        assert "200000 ceiling" in text
        assert "not retrievable" in text

    def test_stop_wording_matches_the_two_stop_tools(self):
        """The two stops are not interchangeable, and the skill must not blur them."""
        text = unwrapped(render(full_index()))
        assert "`queue_stop` halts the queue after the running item finishes" in text
        assert "`stop_run` aborts the plan already in motion" in text

    def test_description_shares_no_long_phrase_with_a_sibling(self):
        """A shared 6-word phrase would steal triggers from a skill that can act."""
        mine = ngrams(frontmatter(render(full_index()))["description"])
        for sibling in ("operating-bluesky-plans", "writing-bluesky-plans"):
            shared = mine & ngrams(sibling_description(sibling))
            assert not shared, f"{sibling} shares trigger wording: {shared}"


class TestReferencedNamesExist:
    def test_every_tool_name_is_real(self):
        text = render(full_index(overflow=7))
        # Table rows carry plan names, which are data, not tool names.
        prose = "\n".join(line for line in text.splitlines() if line not in table_rows(text))
        named = {
            term.rstrip("()").lower()
            for term in re.findall(r"`([a-z][a-z0-9_.]*)(?:\([^`]*\))?`", prose)
        }
        assert named - KNOWN_NON_TOOLS <= set(ALL_TOOLS), (
            f"unknown tool names: {sorted(named - KNOWN_NON_TOOLS - set(ALL_TOOLS))}"
        )

    def test_every_referenced_skill_exists(self):
        text = render(full_index())
        for skill in re.findall(r"`([a-z]+-bluesky-plans)`", text):
            assert (SKILLS_DIR / skill / "SKILL.md").is_file(), skill


class TestProse:
    def test_body_excluding_table_rows_fits_the_budget(self):
        text = render(full_index(overflow=7))
        body = [line for line in text.splitlines() if line not in table_rows(text)]
        assert len(body) <= 120, len(body)

    def test_carries_no_unfinished_markers(self):
        assert not re.search(r"\b(TODO|FIXME|XXX)\b", render(full_index()))
