"""Tests for the pyat-specialist AgentDefinition registration and wiring.

Verifies that:
- The agent appears in FRAMEWORK_AGENTS with the correct metadata and a
  ``python`` server dependency.
- resolve_agents() enables pyat-specialist by default when the python server is
  resolved and the render serves a deck-bearing model, disables it when the
  python server is disabled or no served model has a deck, and honors a config
  override.
- The agent template renders the exact tools/disallowedTools/maxTurns frontmatter
  and the distinctive body behaviors, and produces no output when disabled.
- The demo CLAUDE.md renders exactly one pyat-specialist roster
  bullet, the delegation prohibition, and the documented-vs-computed boundary in
  both the pyat-specialist and facility-knowledge blocks.

All template assertions pin what the artifacts actually contain (truthful pins).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.registry.mcp import FRAMEWORK_AGENTS, resolve_agents, resolve_servers
from osprey_connectors.simulation.view import VIEW_RELPATH
from osprey_connectors.workspace import BUILD_DIR_NAME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Where the shipped subagent files live, so the frontmatter pin reads the
#: same file the build renders rather than a copy of its text.
AGENT_TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
    / "claude"
    / "agents"
)


#: The simulator view's directory relative to the deployment root, as a render
#: names it.
_SIMULATOR_DIR = f"{BUILD_DIR_NAME}/{VIEW_RELPATH}"


def _base_ctx(**overrides):
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "served_decks": [{"model": "SR", "path": f"{_SIMULATOR_DIR}/decks/SR.json"}],
        "simulator_dir": _SIMULATOR_DIR,
    }
    ctx.update(overrides)
    return ctx


def _get_agent(agents: list[dict]) -> dict:
    matches = [a for a in agents if a["name"] == "pyat-specialist"]
    assert len(matches) == 1, "Expected exactly one pyat-specialist agent"
    return matches[0]


# ---------------------------------------------------------------------------
# FRAMEWORK_AGENTS catalog + tier
# ---------------------------------------------------------------------------


class TestPyatSpecialistAgentCatalog:
    """The AgentDefinition is correctly wired in FRAMEWORK_AGENTS."""

    def test_present_in_framework_agents(self):
        assert "pyat-specialist" in FRAMEWORK_AGENTS

    def test_name_field(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.name == "pyat-specialist"

    def test_description_is_non_empty(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.description

    def test_the_description_names_no_particular_machine(self):
        """This sentence is rendered into every render's prompt, everywhere.

        It is the framework speaking about a framework agent, so it can only
        say what the agent computes against — the lattice model this
        deployment configured. Naming the ring the shipped demo data happens
        to hold tells an operator at any other facility that the answer came
        from someone else's machine.
        """
        description = FRAMEWORK_AGENTS["pyat-specialist"].description
        assert "the deployment's configured lattice model" in description
        assert "ALS" not in description

    def test_the_shipped_agent_file_repeats_the_registry_description(self):
        """One sentence, two spellings — the frontmatter and the registry.

        The agent file's `description:` is what Claude Code itself reads when
        choosing a subagent; the registry's is what the roster in CLAUDE.md
        renders. Drift between them is invisible until an operator compares
        two prompts, so it is pinned instead.
        """
        template = (Path(AGENT_TEMPLATE_ROOT) / "pyat-specialist.md.j2").read_text(encoding="utf-8")
        assert FRAMEWORK_AGENTS["pyat-specialist"].description in template

    def test_server_dependency_is_python(self):
        """pyat-specialist computes via mcp__python__execute — it needs the python server."""
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.server_dependency == "python"

    def test_enabled_by_default(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.default_enabled is True

    def test_is_not_custom(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.is_custom is False


# ---------------------------------------------------------------------------
# resolve_agents() output
# ---------------------------------------------------------------------------


class TestPyatSpecialistAgentResolved:
    """resolve_agents() gates pyat-specialist on the python server dependency."""

    def test_appears_in_resolved_agents(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        names = [a["name"] for a in agents]
        assert "pyat-specialist" in names

    def test_enabled_by_default_when_python_resolved(self):
        """Default build: the python server is enabled, so pyat-specialist is enabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        assert "python" in {s["name"] for s in servers if s["enabled"]}
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["enabled"] is True

    def test_disabled_when_python_server_disabled(self):
        """Disabling the python server must disable the pyat-specialist agent."""
        ctx = _base_ctx()
        cfg = {"servers": {"python": {"enabled": False}}}
        servers = resolve_servers(cfg, ctx)
        assert "python" not in {s["name"] for s in servers if s["enabled"]}
        agents = resolve_agents(cfg, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["enabled"] is False

    def test_disabled_when_no_served_model_has_a_deck(self):
        """A render serving no deck-bearing model has nothing for the agent to load."""
        ctx = _base_ctx(served_decks=[])
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["enabled"] is False

    def test_a_hand_enable_without_a_served_deck_leaves_it_out(self, caplog):
        """Switched on by hand with no deck to load, the agent stays out and the
        build says why in one line."""
        ctx = _base_ctx(served_decks=[])
        servers = resolve_servers({}, ctx)
        with caplog.at_level("WARNING", logger="osprey.registry.mcp"):
            agents = resolve_agents(
                {"agents": {"pyat-specialist": {"enabled": True}}},
                ctx,
                resolved_servers=servers,
            )
        assert _get_agent(agents)["enabled"] is False
        lines = [r.getMessage() for r in caplog.records if "pyat-specialist" in r.getMessage()]
        assert lines == ["Agent 'pyat-specialist' is left out: it needs a served model with a deck"]

    def test_a_hand_enable_with_a_served_deck_says_nothing(self, caplog):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        with caplog.at_level("WARNING", logger="osprey.registry.mcp"):
            agents = resolve_agents(
                {"agents": {"pyat-specialist": {"enabled": True}}},
                ctx,
                resolved_servers=servers,
            )
        assert _get_agent(agents)["enabled"] is True
        assert not [r for r in caplog.records if "pyat-specialist" in r.getMessage()]

    def test_can_be_disabled_via_config(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"pyat-specialist": {"enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        agent = _get_agent(agents)
        assert agent["enabled"] is False

    def test_is_not_custom_in_resolved(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["is_custom"] is False

    def test_description_propagates(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["description"]


# ---------------------------------------------------------------------------
# Agent template rendering
# ---------------------------------------------------------------------------


class TestPyatSpecialistAgentTemplate:
    """The agent template renders the exact frontmatter and body behaviors."""

    def _full_ctx(self, enabled: bool = True, **overrides):
        ctx = _base_ctx(**overrides)
        claude_code_cfg = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_name", "Test Facility")
        ctx["servers"] = resolve_servers(claude_code_cfg, ctx)
        ctx["agents"] = []
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {"pyat-specialist"} if enabled else set()
        return ctx

    def _render(self, tm, ctx: dict) -> str:
        return tm.jinja_env.get_template("claude_code/claude/agents/pyat-specialist.md.j2").render(
            **ctx
        )

    def test_renders_when_enabled(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert rendered.strip()

    def test_renders_empty_when_disabled(self, template_manager):
        ctx = self._full_ctx(enabled=False)
        rendered = self._render(template_manager, ctx)
        assert not rendered.strip(), "Template must produce no output when agent is disabled"

    @pytest.mark.parametrize(
        "expected",
        [
            pytest.param("name: pyat-specialist", id="frontmatter-name"),
            pytest.param("maxTurns: 30", id="frontmatter-max-turns"),
            # Edge case: pyAT returns NaN without raising, so results must be guarded.
            pytest.param("NaN without raising", id="silent-nan"),
            # Edge case: one-turn map |trace| < 2 stability test.
            pytest.param("|trace| must be < 2 per plane", id="trace-stability-guard"),
            # Edge case: flag heavy DA/LMA/tracking runs before launching them.
            pytest.param("Flag heavy runs before launching them", id="flag-heavy-runs"),
        ],
    )
    def test_the_enabled_template_carries_its_pinned_text(self, template_manager, expected):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert expected in rendered

    def test_frontmatter_tools_exact(self, template_manager):
        """The tools: line pins exactly the four allowed tools, in order."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        expected = (
            "tools: mcp__python__execute, mcp__osprey_workspace__submit_response, "
            "mcp__osprey_workspace__artifact_read, Read"
        )
        assert expected in rendered

    def test_frontmatter_declares_no_results_contract(self, template_manager):
        """The agent's answer is its prose — there is no second data hand-in.

        ``results_category`` was a frontmatter declaration that made
        ``submit_response`` refuse a hand-in without a JSON copy of the numbers.
        The copy was re-typed by the model out of its own stdout, so it was no
        more authoritative than the prose; the run itself is preserved by the
        automatic notebook artifact.
        """
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "results_category" not in rendered
        assert "data={" not in rendered

    def test_frontmatter_disallowed_tools_exact(self, template_manager):
        """The disallowedTools: line pins exactly the eleven blocked tools, in order."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        expected = (
            "disallowedTools: Bash, Write, Edit, Glob, Grep, WebFetch, WebSearch, "
            "NotebookEdit, Task, Skill, Agent"
        )
        assert expected in rendered

    def test_no_readwrite_write_edit_in_tools(self, template_manager):
        """Compute-only agent: no Write/Edit/Bash in the tools: line."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        tools_line = next((ln for ln in rendered.splitlines() if ln.startswith("tools:")), "")
        for banned in ("Bash", "Write", "Edit"):
            assert banned not in tools_line

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(
                'execution_mode="readonly"',
                'Never use `execution_mode="readwrite"`.',
                id="execution-mode-readonly-pin",
            ),
            pytest.param("disable_6d", "lattice4d.disable_6d()", id="disable-6d"),
            # Edge case: enumerate elements by pyAT type, never by FamName pattern.
            pytest.param(
                "get_uint32_index",
                "Do **not** scan `FamName`",
                id="type-enumeration-over-name-pattern",
            ),
        ],
    )
    def test_the_enabled_template_body_carries_both_pinned_phrases(
        self, template_manager, first, second
    ):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert first in rendered
        assert second in rendered

    def test_body_edge_case_module_not_found(self, template_manager):
        """Edge case: ModuleNotFoundError → report the environment limitation and stop."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        normalized = " ".join(rendered.split())
        assert "ModuleNotFoundError" in rendered
        assert "Report that environment limitation plainly and stop" in normalized

    def test_body_hands_in_its_answer_as_prose(self, template_manager):
        """One submit_response call carrying the answer; no category (retired).

        The agent is told to report every quantity it was asked for in that
        prose (a table once there are several), and that a large array goes
        through save_artifact in the computing call instead of being retyped —
        the one case where a second artifact earns its place.
        """
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        normalized = " ".join(rendered.split())

        assert '`source_agent`: "pyat-specialist"' in rendered
        assert "data_type" not in rendered
        assert "Report every quantity you were asked for" in normalized
        assert "markdown table" in normalized
        # The large-array escape hatch names the injected helper, without a category.
        assert 'artifact_type="json"' in rendered
        assert "category=" not in rendered
        assert "np.asarray(x).tolist()" in rendered
        # No self-verification step: the agent does not re-read its own filing.
        assert "artifact_list" not in rendered


# ---------------------------------------------------------------------------
# Demo CLAUDE.md rendering
# ---------------------------------------------------------------------------


class TestPyatSpecialistClaudeMd:
    """CLAUDE.md renders the roster bullet, delegation prohibition, and boundary."""

    def _full_ctx(self, **overrides):
        ctx = _base_ctx(**overrides)
        claude_code_cfg = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_name", "Test Facility")
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers(claude_code_cfg, ctx)
        ctx["agents"] = resolve_agents(claude_code_cfg, ctx, resolved_servers=ctx["servers"])
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
        return ctx

    def _render(self, tm, ctx: dict) -> str:
        return tm.jinja_env.get_template("claude_code/CLAUDE.md.j2").render(**ctx)

    @pytest.mark.usefixtures("template_manager")
    def test_pyat_specialist_enabled_in_default_build(self):
        """Sanity: the default build enables pyat-specialist (python server on)."""
        ctx = self._full_ctx()
        assert "pyat-specialist" in ctx["enabled_agents"]

    def test_exactly_one_roster_bullet(self, template_manager):
        """The roster lists pyat-specialist exactly once — no duplicate from the fallthrough loop."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        matches = re.findall(r"^- \*\*pyat-specialist\*\*", rendered, re.MULTILINE)
        assert len(matches) == 1

    @pytest.mark.parametrize(
        "expected",
        [
            # The roster tells the parent NOT to compute lattice quantities itself.
            pytest.param(
                "do NOT compute lattice quantities (orbit, tunes, optics functions, "
                "response matrices) via `mcp__python__execute` yourself",
                id="delegation-prohibition",
            ),
            # The pyat-specialist block carries the documented-vs-computed boundary.
            pytest.param(
                "Anything that must be *computed* on the lattice model goes to **pyat-specialist**.",
                id="documented-vs-computed-in-pyat-block",
            ),
            # The facility-knowledge block carries the documented-vs-computed boundary too.
            pytest.param(
                "Anything that requires *computing* on the lattice model — orbit, tunes, "
                "optics functions, response matrices — goes to **pyat-specialist**, not here.",
                id="documented-vs-computed-in-facility-knowledge-block",
            ),
        ],
    )
    def test_the_roster_carries_its_pinned_text(self, template_manager, expected):
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        assert expected in rendered

    def test_documented_vs_computed_phrase_in_both_blocks(self, template_manager):
        """The 'Documented vs. computed:' lead-in appears in both agent blocks."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        assert rendered.count("Documented vs. computed:") == 2
