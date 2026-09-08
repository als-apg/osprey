"""Tests for template generation system.

Tests the TemplateManager class and template rendering,
including validation that generated projects use the new
registry helper pattern correctly.
"""

import json
from pathlib import Path

import pytest

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.templates import claude_code, manifest
from osprey.cli.templates.manager import TemplateManager
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE
from osprey.services.channel_finder.core.exceptions import PipelineModeError


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages, which is the content the render used to reach
    for on its own.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
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
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


class TestTemplateManager:
    """Test TemplateManager class."""

    def test_template_manager_initialization(self):
        """Test that TemplateManager initializes correctly."""
        manager = TemplateManager()

        assert manager.template_root is not None
        assert manager.template_root.exists()
        assert manager.jinja_env is not None

    def test_list_app_templates(self):
        """Test listing available application templates."""
        manager = TemplateManager()
        templates = manager.list_app_templates()

        assert "control_assistant" in templates
        assert len(templates) >= 1

    def test_create_project_control_assistant(self, tmp_path):
        """Test creating project with control_assistant template."""
        manager = TemplateManager()

        project_dir = _create_project(
            manager,
            project_name="test-project",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Verify structure (Claude Code mode — no src/ or pyproject.toml)
        assert project_dir.exists()
        assert (project_dir / "config.yml").exists()
        assert (project_dir / ".env.example").exists()
        assert (project_dir / "README.md").exists()
        # No runtime state and no secret store in the render: agent data lives
        # at <repo>/var/agent_data and secrets at <repo>/.env, both outside the
        # tree a rebuild wipes.
        assert not (project_dir / "_agent_data").exists()
        assert not (project_dir / ".env").exists()

        # Claude Code integration
        assert (project_dir / "CLAUDE.md").exists()
        assert (project_dir / ".mcp.json").exists()

    def test_create_project_in_context_derives_tier1(self, tmp_path):
        """Omitting ``tier`` with an in_context paradigm derives tier 1 and
        materializes the tier-1 DB (the paradigm-aware default, not a hardcoded
        1 — and provably tier 1, not tier 3)."""
        from pathlib import Path

        manager = TemplateManager()

        project_dir = _create_project(
            manager,
            project_name="test-project",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "in_context"},
        )

        assert project_dir.exists()
        assert (project_dir / "config.yml").exists()

        # The materialized flat DB must be byte-equal to the preset's TIER-1
        # in_context source. Tier 1 is a filtered subset of tier 3, so this
        # assertion fails if the derivation had (wrongly) resolved tier 3.
        preset_tier1 = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "osprey"
            / "templates"
            / "apps"
            / "control_assistant"
            / "data"
            / "channel_databases"
            / "tiers"
            / "tier1"
            / "in_context.json"
        )
        flat = project_dir / "data" / "channel_databases" / "in_context.json"
        assert flat.is_file()
        assert flat.read_bytes() == preset_tier1.read_bytes()

    def test_create_project_explicit_tier1_hierarchical_rejected(self, tmp_path):
        """An explicit ``tier=1`` paired with a non-in_context paradigm is
        rejected with the rule-naming error at the creation boundary, not left
        to surface as an opaque FileNotFoundError inside the materializer."""
        from osprey.errors import BuildProfileError

        manager = TemplateManager()

        with pytest.raises(
            BuildProfileError, match="tier 1 requires channel_finder_mode: in_context"
        ):
            _create_project(
                manager,
                project_name="test-project",
                output_dir=tmp_path,
                data_bundle="control_assistant",
                context={"channel_finder_mode": "hierarchical"},
                tier=1,
            )

    def test_create_project_explicit_tier2_rejected(self, tmp_path):
        """An out-of-range explicit ``tier`` is rejected with the {1,3} rule
        error at the creation boundary, mirroring BuildProfile.validate()."""
        from osprey.errors import BuildProfileError

        manager = TemplateManager()

        with pytest.raises(BuildProfileError, match="tier must be 1 or 3"):
            _create_project(
                manager,
                project_name="test-project",
                output_dir=tmp_path,
                data_bundle="control_assistant",
                context={"channel_finder_mode": "in_context"},
                tier=2,
            )

    def test_create_project_graph_derives_tier3(self, tmp_path, monkeypatch):
        """Omitting ``tier`` with the graph paradigm derives tier 3.

        Graph's store is a seeded graph service rather than a database file, so
        the derived tier reaches the materializer only to select the benchmark
        query set — no ``channel_databases/<paradigm>.json`` is flattened. The
        test stops the render at that boundary, which is the whole of the tier
        derivation; the rest of the render is exercised by the per-paradigm
        render tests.
        """
        from pathlib import Path

        from osprey.cli.templates import scaffolding

        class _StopAfterMaterialize(Exception):
            pass

        real_materialize = scaffolding.materialize_tier_artifacts
        seen: dict = {}

        def _record(project_dir, tier, channel_finder_mode):
            seen["tier"] = tier
            seen["project_dir"] = project_dir
            real_materialize(project_dir, tier, channel_finder_mode)
            raise _StopAfterMaterialize

        monkeypatch.setattr(scaffolding, "materialize_tier_artifacts", _record)

        manager = TemplateManager()
        with pytest.raises(_StopAfterMaterialize):
            _create_project(
                manager,
                project_name="test-project",
                output_dir=tmp_path,
                data_bundle="control_assistant",
                context={"channel_finder_mode": "graph"},
            )

        assert seen["tier"] == 3

        project_dir = seen["project_dir"]
        preset_data = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "osprey"
            / "templates"
            / "apps"
            / "control_assistant"
            / "data"
        )
        # The tier-3 query set landed. Tier 1 ships a different, smaller set,
        # so this byte-comparison fails if the derivation had resolved tier 1.
        queries = project_dir / "data" / "benchmarks" / "queries.json"
        expected = preset_data / "benchmarks" / "cross_paradigm" / "queries" / "tier3_queries.json"
        assert queries.read_bytes() == expected.read_bytes()

        # No paradigm database was materialized for graph.
        cdb = project_dir / "data" / "channel_databases"
        for paradigm in VALID_CHANNEL_FINDER_MODES:
            assert not (cdb / f"{paradigm}.json").exists()
        assert not (cdb / "tiers").exists()

    def test_create_project_explicit_tier_with_graph_rejected(self, tmp_path):
        """An explicit ``tier`` paired with graph is rejected at the creation
        boundary with the graph rule, not the in_context tier-1 rule."""
        from osprey.errors import BuildProfileError

        manager = TemplateManager()

        with pytest.raises(BuildProfileError, match="graph has no tiered artifacts; omit tier"):
            _create_project(
                manager,
                project_name="test-project",
                output_dir=tmp_path,
                data_bundle="control_assistant",
                context={"channel_finder_mode": "graph"},
                tier=3,
            )

    def test_duplicate_project_raises_error(self, tmp_path):
        """Test that creating duplicate project raises error."""
        manager = TemplateManager()

        # Create first project
        _create_project(
            manager,
            project_name="test-project",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Try to create again — directory-exists check fires before
        # channel-finder validation, so no context needed here.
        with pytest.raises(ValueError, match="already exists"):
            _create_project(
                manager,
                project_name="test-project",
                output_dir=tmp_path,
                data_bundle="control_assistant",
            )

    def test_invalid_template_raises_error(self, tmp_path):
        """Test that invalid template name raises error."""
        manager = TemplateManager()

        with pytest.raises(ValueError, match="not found"):
            _create_project(
                manager,
                project_name="test-project",
                output_dir=tmp_path,
                data_bundle="nonexistent_template",
            )


class TestBuildClaudeCodeContextHierarchy:
    """Tests for hierarchy embedding in build_claude_code_context."""

    def _make_manager_and_config(self, tmp_path, db_data):
        """Create a TemplateManager and config pointing at a hierarchy database."""
        import json as _json

        db_file = tmp_path / "channels.json"
        db_file.write_text(_json.dumps(db_data))

        # Manifest must declare control_assistant template for the
        # channel_finder block to activate.
        manifest = tmp_path / ".osprey-manifest.json"
        manifest.write_text(_json.dumps({"creation": {"template": "control_assistant"}}))

        config = {
            "facility_name": "TestFacility",
            "channel_finder": {
                "pipeline_mode": "hierarchical",
                "pipelines": {
                    "hierarchical": {
                        "database": {"path": "channels.json"},
                    },
                },
            },
        }
        return TemplateManager(), config

    @pytest.mark.unit
    def test_build_claude_code_context_embeds_hierarchy_info(self, tmp_path):
        """Hierarchy levels, config, and naming pattern are embedded in context."""
        manager, config = self._make_manager_and_config(
            tmp_path,
            {
                "hierarchy": {
                    "levels": [
                        {"name": "system", "type": "tree"},
                        {"name": "device", "type": "instances"},
                    ],
                    "naming_pattern": "{system}:{device}",
                },
                "tree": {
                    "SR": {
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "D{:02d}",
                                "_range": [1, 3],
                            }
                        }
                    }
                },
            },
        )
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, tmp_path, config
        )
        hier = ctx["channel_finder_hierarchy"]
        assert hier is not None
        assert hier["hierarchy_levels"] == ["system", "device"]
        assert hier["naming_pattern"] == "{system}:{device}"
        assert "system" in hier["hierarchy_config"]["levels"]

    @pytest.mark.unit
    def test_build_claude_code_context_hierarchy_missing_path(self, tmp_path):
        """Graceful fallback to None when database path is missing."""
        config = {
            "facility_name": "TestFacility",
            "channel_finder": {
                "pipeline_mode": "hierarchical",
                "pipelines": {
                    "hierarchical": {
                        "database": {},
                    },
                },
            },
        }
        manager = TemplateManager()
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, tmp_path, config
        )
        assert ctx["channel_finder_hierarchy"] is None

    @pytest.mark.unit
    def test_build_claude_code_context_hierarchy_non_hierarchical(self, tmp_path):
        """Non-hierarchical pipeline mode: channel_finder_hierarchy is None."""
        config = {
            "facility_name": "TestFacility",
            "channel_finder": {
                "pipeline_mode": "in_context",
            },
        }
        manager = TemplateManager()
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, tmp_path, config
        )
        assert ctx["channel_finder_hierarchy"] is None

    @pytest.mark.unit
    def test_create_project_embeds_hierarchy_info(self, tmp_path, monkeypatch):
        """create_project renders hierarchy info into the agent prompt."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        manager = TemplateManager()

        project_dir = _create_project(
            manager,
            project_name="test-hier-embed",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        agent_prompt = (project_dir / ".claude" / "agents" / "channel-finder.md").read_text()
        # Must contain embedded hierarchy info, NOT the fallback text
        assert "hierarchy_levels" in agent_prompt
        assert "Call `get_options()` at the first level to discover" not in agent_prompt


class TestBuildClaudeCodeContextPipelineMode:
    """The render path refuses a missing or unknown channel-finder paradigm.

    A project that ships the channel-finder agent is built against one
    paradigm's store. Guessing a default, or quietly rendering an empty tool
    list for a name nothing recognises, would produce an agent whose prompt and
    tools do not match the data on disk — so both cases raise here instead.
    """

    @staticmethod
    def _config(**channel_finder):
        return {"facility_name": "TestFacility", "channel_finder": dict(channel_finder)}

    @staticmethod
    def _project_selecting_the_agent(tmp_path):
        """A project directory whose manifest selects the channel-finder agent.

        The paradigm rule only fires for a project that ships that agent, and
        the selection is read from the project's own ``.osprey-manifest.json``.
        A bare directory selects nothing, so the rule would never be reached.
        """
        (tmp_path / ".osprey-manifest.json").write_text(
            json.dumps({"artifacts": {"agents": ["channel-finder"]}}), encoding="utf-8"
        )
        return tmp_path

    @pytest.mark.unit
    def test_missing_pipeline_mode_raises(self, tmp_path):
        """A channel_finder block with no pipeline_mode is an error, not a default."""
        manager = TemplateManager()
        with pytest.raises(PipelineModeError, match="pipeline_mode"):
            claude_code.build_claude_code_context(
                manager.template_root,
                manager.jinja_env,
                self._project_selecting_the_agent(tmp_path),
                self._config(pipelines={}),
            )

    @pytest.mark.unit
    def test_unknown_pipeline_mode_raises_and_names_the_mode(self, tmp_path):
        """An unrecognised paradigm raises and the message names it."""
        manager = TemplateManager()
        with pytest.raises(PipelineModeError, match="bogus"):
            claude_code.build_claude_code_context(
                manager.template_root,
                manager.jinja_env,
                self._project_selecting_the_agent(tmp_path),
                self._config(pipeline_mode="bogus"),
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", VALID_CHANNEL_FINDER_MODES)
    def test_known_modes_render_their_tool_list(self, tmp_path, mode):
        """Every registered paradigm renders the registry's tool list for it."""
        manager = TemplateManager()
        ctx = claude_code.build_claude_code_context(
            manager.template_root,
            manager.jinja_env,
            self._project_selecting_the_agent(tmp_path),
            self._config(pipeline_mode=mode),
        )
        assert ctx["channel_finder_pipeline"] == mode
        assert ctx["channel_finder_mode"] == mode
        assert ctx["default_pipeline"] == mode
        assert ctx["channel_finder_tools"] == CHANNEL_FINDER_TOOLS_BY_PIPELINE.get(mode, [])


class TestTemplateManifest:
    """Test template manifest loading, resolution, and filtering."""

    def test_control_assistant_example_profile_has_expected_artifacts(self):
        """Control assistant example profile declares all expected artifact categories."""
        import importlib.resources

        import yaml

        profile_text = (
            importlib.resources.files("osprey.profiles.presets")
            .joinpath("control-assistant.yml")
            .read_text(encoding="utf-8")
        )
        profile = yaml.safe_load(profile_text)

        assert "hooks" in profile
        assert "rules" in profile
        assert "skills" in profile
        assert "agents" in profile
        assert "output_styles" in profile
        assert "approval" in profile["hooks"]
        assert "channel-finder" in profile["agents"]

    def test_load_manifest_nonexistent_template(self):
        """Returns None for unknown template."""
        mf = manifest.load_template_manifest("nonexistent_template")
        assert mf is None

    def test_load_manifest_preset_profile_fallback_includes_web_panels(self):
        """Preset-profile fallback must surface web_panels in the artifacts dict.

        Direct ``TemplateManager.create_project()`` callers rely on this
        fallback: the bundle name resolves to the ``control-assistant`` preset,
        which is where the declaration lives. If web_panels is dropped,
        config.yml renders ``panels: {}`` and the web terminal shows only the
        universal panels — the exact bug reported when ARIEL / channel-finder
        panels went missing.
        """
        mf = manifest.load_template_manifest("control_assistant")
        assert mf is not None
        artifacts = mf.get("artifacts", {})
        # The preset profile declares these panels; they must round-trip through
        # the fallback so create_project() → Jinja → config.yml wires them up.
        assert "web_panels" in artifacts, (
            "web_panels stripped by preset-profile fallback — TemplateManager will "
            "render `panels: {}` and no built-in panels will appear."
        )
        assert set(artifacts["web_panels"]) >= {"ariel", "channel-finder"}

    def test_create_project_without_artifacts_enables_builtin_panels(self, tmp_path):
        """create_project() without explicit artifacts (legacy direct-call path)
        must render the builtin panels block from the preset profile."""
        import yaml as _yaml

        manager = TemplateManager()
        project_dir = _create_project(
            manager,
            project_name="init-panels-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        config = _yaml.safe_load((project_dir / "config.yml").read_text())
        panels = config["web"]["panels"]
        assert panels, "web.panels block is empty — no builtin panels were enabled"
        assert panels.get("ariel", {}).get("enabled") is True
        assert panels.get("channel-finder", {}).get("enabled") is True

    def test_resolve_manifest_outputs_includes_config_artifacts(self):
        """Resolved outputs always contain config artifacts."""
        mf = {"artifacts": {"hooks": ["approval"], "rules": ["safety"], "skills": [], "agents": []}}
        outputs = manifest.resolve_manifest_outputs(mf)

        assert "CLAUDE.md" in outputs
        assert ".mcp.json" in outputs
        assert ".claude/settings.json" in outputs

    def test_resolve_manifest_outputs_maps_hooks(self):
        """hooks: [approval] resolves to .claude/hooks/osprey_approval.py."""
        mf = {"artifacts": {"hooks": ["approval"]}}
        outputs = manifest.resolve_manifest_outputs(mf)

        assert ".claude/hooks/osprey_approval.py" in outputs

    def test_resolve_manifest_outputs_session_report_includes_reference(self):
        """skills: [session-report] resolves to both SKILL.md and reference.md."""
        mf = {"artifacts": {"skills": ["session-report"]}}
        outputs = manifest.resolve_manifest_outputs(mf)

        assert ".claude/skills/session-report/SKILL.md" in outputs
        assert ".claude/skills/session-report/reference.md" in outputs

    def test_control_assistant_has_all_hooks(self, tmp_path):
        """Pin the exact set of hook files the control-assistant preset renders.

        An equality check, not a subset one, so dropping a hook from the preset
        and adding an unlisted one both fail here — the rendered set only changes
        when someone updates this list on purpose.

        This does not read settings.json; it says nothing about whether the
        wiring matches. That parity is the two-direction invariant in
        ``tests/cli/test_claude_regen.py`` — every settings.json reference has a
        shipped file, and every shipped event hook is wired into settings.json.
        """
        manager = TemplateManager()
        project_dir = _create_project(
            manager,
            project_name="ctrl-hooks-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        expected_hooks = {
            "hook_config.json",
            "osprey_approval.py",
            "osprey_cf_feedback_capture.py",
            "osprey_config_drift.py",
            "osprey_control_context.py",
            "osprey_error_guidance.py",
            "osprey_focus_validate.py",
            "osprey_hook_log.py",
            "osprey_limits.py",
            "osprey_memory_guard.py",
            "osprey_notebook_update.py",
            "osprey_panels_context.py",
            "osprey_target_state.py",
            "osprey_turn_state.py",
            "osprey_workspace_delta.py",
            "osprey_writes_check.py",
        }
        hooks_dir = project_dir / ".claude" / "hooks"
        rendered = {p.name for p in hooks_dir.iterdir() if p.is_file()}
        assert rendered == expected_hooks

    def test_backward_compat_no_manifest(self, tmp_path):
        """If manifest doesn't exist, all files are generated (backward compat)."""
        # get_tracked_files falls back to REGEN_TRACKED_FILES when no manifest
        tracked = manifest.get_tracked_files("nonexistent_template")
        assert tracked == list(manifest.REGEN_TRACKED_FILES)

        # resolve_manifest_outputs with allowed_outputs=None means no filtering
        # Verify by checking that load_template_manifest returns None
        mf = manifest.load_template_manifest("nonexistent_template")
        assert mf is None


def test_get_framework_version_unknown_on_import_failure(monkeypatch):
    """C8: fallback returns 'unknown', not a stale hard-coded version."""
    import builtins

    import osprey.cli.templates.manifest as manifest_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("osprey", "osprey.version"):
            raise ImportError("simulated failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert manifest_mod.get_framework_version() == "unknown"
    assert manifest_mod.get_framework_release_version() == "unknown"


def test_registry_style_parameter_is_gone():
    """C1 regression guard: no entry point may carry the 'standalone' branch's
    registry-style parameter. Catches anyone re-adding it without a real CLI
    flag."""
    import inspect

    from osprey.cli.templates.manager import TemplateManager

    sig = inspect.signature(TemplateManager.create_project)
    assert "registry_style" not in sig.parameters, (
        "registry_style should be removed from create_project; resurrect only "
        "with a real CLI flag, not as a dead constant."
    )


class TestBuiltinPanelRegistryDrift:
    """Enable-able builtin panels must derive from the BUILTIN_PANELS registry,
    not a hardcoded template literal that drifts from it.

    Discovered wiring the native ``okf`` KNOWLEDGE panel into BELLA + ALS: both
    config templates hardcoded ``["ariel", "channel-finder"]``, so a
    profile listing a builtin the literal omitted (``okf`` or ``lattice``) in its
    ``web_panels`` got filtered out at build time → no ``web.panels.okf`` stanza
    → the runtime never enabled the tab and it silently never rendered. BELLA/ALS
    worked around it with an explicit ``web.panels.okf.enabled: true`` override.
    """

    #: The one template that renders the builtin-panel loop. The selection is
    #: derived from the profile's ``web_panels:`` field, so it belongs to the
    #: framework template rather than to any deployment's own ``config:``.
    PANEL_TEMPLATES = ["project/config.yml.j2"]

    @pytest.mark.parametrize("template_path", PANEL_TEMPLATES)
    def test_registry_injected_enables_builtin_absent_from_old_literal(self, template_path):
        """With the registry injected as ``builtin_panels``: a builtin the OLD
        literal omitted (``okf``) gets a stanza, a builtin the literal contained
        AND the profile lists (``channel-finder``) gets one, and a builtin the
        profile does NOT list (``ariel``) does not."""
        import yaml

        from osprey.profiles.web_panels import BUILTIN_PANELS

        manager = TemplateManager()
        template = manager.jinja_env.get_template(template_path)
        rendered = template.render(
            builtin_panels=sorted(BUILTIN_PANELS),
            selected_web_panels=["okf", "channel-finder"],
            port_base=DEFAULT_PORT_BASE,
            osprey_ports=layout_ports(DEFAULT_PORT_BASE),
        )
        panels = yaml.safe_load(rendered)["web"]["panels"]

        # okf: builtin NOT in the old hardcoded literal — the drift bug.
        assert panels.get("okf", {}).get("enabled") is True, (
            "okf builtin filtered out — enable list drifted from BUILTIN_PANELS"
        )
        # channel-finder: builtin that IS in the literal and IS listed — stanza present.
        assert panels.get("channel-finder", {}).get("enabled") is True
        # ariel: builtin but NOT listed in this profile's web_panels — no stanza.
        assert "ariel" not in panels

    @pytest.mark.parametrize("template_path", PANEL_TEMPLATES)
    def test_no_builtin_is_enabled_when_the_registry_is_absent(self, template_path):
        """The registry is the only source: without it, no builtin is enabled.

        The template carries no inline list of its own to fall back to, so a
        selection it cannot check against the registry enables nothing rather
        than being waved through. Proves ``okf`` is enabled above because
        ``BUILTIN_PANELS`` supplies it, not by accident of a literal that
        happens to name it.
        """
        import yaml

        manager = TemplateManager()
        template = manager.jinja_env.get_template(template_path)
        rendered = template.render(
            selected_web_panels=["okf", "channel-finder"],
            port_base=DEFAULT_PORT_BASE,
            osprey_ports=layout_ports(DEFAULT_PORT_BASE),
        )

        assert yaml.safe_load(rendered).get("web") is None

    def test_create_project_enables_okf_builtin_panel(self, tmp_path):
        """End-to-end: ``manager.py`` injects ``sorted(BUILTIN_PANELS)`` → template
        enables ``okf``. Fails against the hardcoded fallback literal (which omits
        okf) and passes with the registry-derived context. This removes the need
        for the ``web.panels.okf.enabled: true`` override BELLA/ALS carried."""
        import yaml

        manager = TemplateManager()
        project_dir = _create_project(
            manager,
            project_name="okf-panel-e2e",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            # The hooks are the build's own gates, not this test's subject:
            # memory-guard satisfies the write-tool lint, and the three write
            # gates are what the control-assistant preset's armed writes
            # require of any profile that selects hooks at all.
            artifacts={
                "hooks": ["memory-guard", "approval", "writes-check", "limits"],
                "web_panels": ["okf", "channel-finder"],
            },
        )
        panels = yaml.safe_load((project_dir / "config.yml").read_text())["web"]["panels"]

        assert panels.get("okf", {}).get("enabled") is True
        assert panels.get("channel-finder", {}).get("enabled") is True
        assert "ariel" not in panels


class TestControlAssistantMongoDBArchiver:
    """Where a control-assistant deployment learns to reach its archive.

    The preset selects ``mongodb_archiver`` and deploys the store to back it,
    but it deliberately spells none of the store's coordinates: those are
    derived once from its ``va_archiver:`` block, so the profile never states
    where the archive lives twice. That splits the contract in two, and both
    halves are pinned here — the preset documents the option and refuses to
    restate the coordinates, and the derivation supplies every key the
    connector refuses to default.
    """

    @staticmethod
    def _preset_text() -> str:
        from osprey.cli.build_profile_presets import _presets_dir

        return (_presets_dir() / "control-assistant.yml").read_text(encoding="utf-8")

    def test_preset_documents_the_mongodb_option(self):
        """The preset must name mongodb_archiver, and must not send the reader
        after an install that no longer exists.

        pymongo is a core dependency; the ``archiver-mongodb`` extra is gone, and
        pip accepts an unknown extra with a warning rather than an error — so a
        stale hint here would have a reader install nothing and hit the same
        failure again.
        """
        preset = self._preset_text()

        assert "archiver.type: mongodb_archiver" in preset
        assert "archiver-mongodb" not in preset

    def test_preset_states_none_of_the_derived_coordinates(self):
        """Spelling them under ``config:`` is refused, so the preset must not."""
        preset = self._preset_text()

        live = [
            line
            for line in preset.splitlines()
            if line.strip().startswith("archiver.mongodb_archiver.")
        ]
        assert not live, f"the preset restates derived coordinates: {live}"

    def test_the_derivation_supplies_every_required_key(self):
        """The block covers every key the connector refuses to default.

        The connector reads its coordinates under
        ``archiver.mongodb_archiver.*``, and nothing else writes them, so a key
        missing from this derivation is a deployment that cannot reach its own
        archive.
        """
        from osprey.cli.build_profile_archiver import (
            CONNECTION_CONFIG_PREFIX,
            VAArchiverConfig,
            va_archiver_config_overrides,
        )

        overrides = va_archiver_config_overrides(VAArchiverConfig())
        for key in ("host", "port", "name", "collection", "auth", "username", "password_env"):
            assert f"{CONNECTION_CONFIG_PREFIX}.{key}" in overrides, (
                f"required key {key!r} missing from the derived connection block"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
