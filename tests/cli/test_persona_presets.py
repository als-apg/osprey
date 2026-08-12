"""Tests for the control-assistant multi-user persona presets.

The ``control-assistant`` preset hosts its own multi-user web tier — nginx,
the landing page, and one terminal container per roster user — alongside the
full scan stack. Two persona presets extend it to carve out capability
postures that differ on exactly one axis, ``control_system.writes_enabled``:

* ``control-assistant-readwrite`` — write-capable tier; channel writes pass
  the ordinary safety chain (writes-check, limits, human approval).
* ``control-assistant-readonly`` — read-only tier; every write surface
  refuses.

The base's own ``web_terminals`` roster block is also exercised here. Shared
render helpers live at the top so new sections append without restructuring.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.deployment.web_terminals.lint import Finding, lint_web_terminals
from osprey.utils.config_writer import config_update_fields

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# The single config key the two persona tiers differ on — the reference
# monitor's master write switch (see osprey.connectors.control_system.base).
WRITES_KEY = "control_system.writes_enabled"

# The literal dotted key the hosting preset must carry: the whole web-terminals
# module subtree addressed as one leaf so config_writer sets only this leaf and
# never wholesale-replaces the rendered ``modules`` subtree.
WEB_TERMINALS_KEY = "modules.web_terminals"

# A valid Docker container/object name: must start with an alphanumeric, then
# alphanumerics plus `_`, `.`, `-` (see docker/docker daemon name validation).
DOCKER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def resolve_preset(name: str) -> BuildProfile:
    """Resolve a bundled preset by name, fully merging its ``extends`` chain.

    Args:
        name: Bundled preset name (hyphenated or underscored).

    Returns:
        The parsed and validated :class:`BuildProfile`.
    """
    profile, _profile_dir = resolve_build_profile(None, preset=name)
    return profile


def _render_config_overrides(tmp_path: Path, seed: dict, preset: str = "control-assistant") -> dict:
    """Render a hosting preset's ``config:`` overrides onto ``seed`` exactly as
    the build pipeline does — via :func:`config_update_fields`, the same
    dot-notation writer ``_apply_config_overrides`` calls — and reload the
    result as a plain dict.

    Args:
        tmp_path: Per-test temp directory (pytest fixture).
        seed: Pre-existing config contents to render the overrides onto.
        preset: Hosting preset whose overrides to render.

    Returns:
        The reloaded config after applying the preset's overrides.
    """
    config_path = tmp_path / "config.yml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(seed, fh)
    overrides = resolve_preset(preset).config
    config_update_fields(config_path, overrides)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _render_deployable_config(tmp_path: Path, preset: str = "control-assistant") -> dict:
    """The config a real build produces from ``preset`` — catalog rewrite included.

    :func:`_render_config_overrides` renders the preset's ``config:`` layer
    alone, where each persona's ``build_profile`` is still the PRESET NAME the
    preset author wrote. No build renders a project from that layer: every build
    materializes a profile first, and materialization emits one
    ``personas/<name>.yml`` delta per catalog entry and repoints the catalog at
    it (``_persona_catalog_layer``). A config linted before that rewrite is a
    shape the pipeline never emits.

    So this applies the same rewrite, through the same function the
    materializer calls and over the same catalog it derives, keeping the lint
    tests below a unit-cost check of the real output rather than a pin on an
    intermediate. ``repo_name`` is *preset*: a repo named after the preset is
    exactly what a real ``osprey init --preset control-assistant`` produces, so
    the rewritten ``project``/``project_path`` values match what materialization
    would actually emit. (The end-to-end proof that these agree lives in
    ``tests/cli/test_persona_profile_emission.py``, which drives ``osprey
    init`` → ``osprey build`` for every persona-bearing preset.)
    """
    from osprey.cli.build_profile_emit import persona_catalog
    from osprey.cli.profile_cmd import _persona_catalog_layer

    config_path = tmp_path / "config.yml"
    resolved = resolve_preset(preset)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"system": {}}, fh)
    config_update_fields(config_path, resolved.config)
    config_update_fields(
        config_path,
        _persona_catalog_layer(persona_catalog(resolved.config), repo_name=preset)["config"],
    )
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


# ---------------------------------------------------------------------------
# Hosting preset: multi-user web-terminal block
# ---------------------------------------------------------------------------


class TestControlAssistantWebTier:
    """The tutorial preset hosts its own web tier: block shape, port families
    clear of the tutorial's own service ports, and a lint-clean render."""

    def test_web_terminals_is_one_literal_dotted_key(self) -> None:
        """The block is carried as the single literal ``modules.web_terminals``
        key, never a nested ``modules:`` mapping under ``config:``.

        A nested mapping would deep-merge over the rendered ``modules`` subtree
        and drop sibling modules; the literal dotted key sets only its own leaf.
        """
        base = resolve_preset("control-assistant")
        assert WEB_TERMINALS_KEY in base.config
        assert "modules" not in base.config
        assert "facility" not in base.config
        assert "deploy" not in base.config

    def test_hosts_its_own_web_tier(self) -> None:
        """The preset carries the web-terminals block plus the web-tier
        companion keys (``facility.prefix`` for container names,
        ``deploy.fqdn`` for the landing URL) — the hosting posture the persona
        family below attaches to."""
        base = resolve_preset("control-assistant")
        assert WEB_TERMINALS_KEY in base.config
        assert "facility.prefix" in base.config
        assert "deploy.fqdn" in base.config

    def test_rendered_config_keeps_sibling_modules(self, tmp_path: Path) -> None:
        """Rendering the overrides adds ``modules.web_terminals`` without
        clobbering a pre-existing sibling ``modules.*`` key or unrelated config.

        This is the whole reason the block is a flat dotted key: config_writer
        writes each dotted key verbatim, so an existing ``modules.event_dispatcher``
        (and any other top-level key) must survive untouched.
        """
        seed = {
            "modules": {
                "event_dispatcher": {"enabled": True, "port": 8010},
            },
            "ports": {"matlab": 8001},
            "system": {"facility_name": "Seed Facility"},
        }
        rendered = _render_config_overrides(tmp_path, seed)

        modules = rendered["modules"]
        # The new block landed.
        assert modules["web_terminals"]["enabled"] is True
        # The sibling module and unrelated keys were not clobbered.
        assert modules["event_dispatcher"] == {"enabled": True, "port": 8010}
        assert rendered["ports"] == {"matlab": 8001}
        assert rendered["system"]["facility_name"] == "Seed Facility"

    def test_rendered_web_terminals_shape(self, tmp_path: Path) -> None:
        """The rendered ``modules.web_terminals`` subtree matches the two-persona
        tutorial shape: local image source, readonly default, a
        readonly/readwrite catalog whose ``project`` equals its ``project_path``
        basename, and a roster mapping alice→readonly and bob→readwrite, both
        explicit.

        Deliberately pins the preset's OWN ``config:`` layer, BEFORE the catalog
        rewrite every build performs — which is why the ``build_profile`` values
        asserted below are preset NAMES. That is the first half of a two-stage
        vocabulary, not stale data: a preset name is what materialization
        consumes to emit ``personas/<name>.yml``, and only the rewritten value
        ever reaches a rendered config. Do not "fix" these to the path spelling:
        the post-rewrite spelling is asserted by this class's own
        ``test_rendered_config_lints_without_errors`` (via
        :func:`_render_deployable_config`) and end-to-end by
        ``tests/cli/test_persona_profile_emission.py``."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        wt = rendered["modules"]["web_terminals"]

        assert wt["enabled"] is True
        assert wt["image_source"] == "local"
        assert wt["default_persona"] == "readonly"
        assert wt["nginx_port"] == 9080

        assert wt["users"][0] == {"name": "alice", "index": 0, "persona": "readonly"}
        assert wt["users"][1] == {"name": "bob", "index": 1, "persona": "readwrite"}

        personas = wt["personas"]
        assert set(personas) == {"readonly", "readwrite"}
        for name, profile in (
            ("readonly", "control-assistant-readonly"),
            ("readwrite", "control-assistant-readwrite"),
        ):
            entry = personas[name]
            # Name invariant: project == basename(project_path).
            assert entry["project"] == os.path.basename(entry["project_path"])
            assert entry["build_profile"] == profile

    def test_port_families_clear_tutorial_service_ports(self, tmp_path: Path) -> None:
        """Every per-user port family sits above the tutorial's own published
        service ports (VA 5064, dispatcher 8020, bluesky 8090/8091, panels
        8095) — the containers share the host network namespace, so a family
        landing on a service port would collide at deploy time."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        wt = rendered["modules"]["web_terminals"]
        service_ports = {5064, 8020, 8090, 8091, 8095}
        n_users = len(wt["users"])
        for family in (
            "nginx_port",
            "web_base_port",
            "artifact_base_port",
            "ariel_base_port",
            "lattice_base_port",
            "channel_finder_base_port",
        ):
            base_port = wt[family]
            spread = 1 if family == "nginx_port" else n_users
            for offset in range(spread):
                assert base_port + offset not in service_ports, (
                    f"{family} collides with a tutorial service port"
                )

    def test_rendered_config_lints_without_errors(self, tmp_path: Path) -> None:
        """``lint_web_terminals`` on the freshly-built tutorial config reports
        zero ERROR findings pre-deploy.

        The referenced persona projects do not exist yet at build time; the lint
        demotes those not-yet-rendered paths to WARNINGS (they carry a
        ``build_profile`` naming the delta ``osprey build`` renders them from),
        so the gate is clean before any project is rendered.

        Linted through :func:`_render_deployable_config`, i.e. after the catalog
        rewrite every build performs — the preset's own ``build_profile`` values
        are preset names, which no rendered config ever carries.
        """
        rendered = _render_deployable_config(tmp_path)
        assert _errors(lint_web_terminals(rendered)) == []

    def test_ships_companion_panels_multi_user(self) -> None:
        """Feature parity: multi-user must not shed single-user companion panels.

        The channel-finder panel was once dropped from a hosting preset to dodge
        a per-user port collision (its family was missing from the port-family
        derivation) — the fix is per-user ports, never removing the feature.
        """
        base = resolve_preset("control-assistant")
        assert "channel-finder" in base.web_panels
        assert "ariel" in base.web_panels

    def test_derived_web_container_names_are_valid_docker_names(self, tmp_path: Path) -> None:
        """The container names derived from the rendered prefix —
        ``<prefix>-nginx`` and ``<prefix>-web-<user>`` for the tutorial roster
        — all match the Docker name grammar (start alphanumeric, no leading
        dash). An empty prefix renders leading-dash names like ``-nginx``,
        which Docker rejects and ``osprey up`` fails the web stack on.
        """
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        prefix = rendered["facility"]["prefix"]
        assert isinstance(prefix, str) and prefix != ""
        for name in (f"{prefix}-nginx", f"{prefix}-web-alice", f"{prefix}-web-bob"):
            assert DOCKER_NAME_RE.match(name), f"invalid Docker container name: {name!r}"

    def test_rendered_config_satisfies_landing_url(self, tmp_path: Path) -> None:
        """The rendered config passes the exact check ``osprey up`` runs —
        ``_landing_url`` raises without ``deploy.fqdn``, aborting ``osprey up``
        for the otherwise zero-config tutorial."""
        from osprey.deployment.web_terminals.render import _landing_url

        rendered = _render_config_overrides(tmp_path, {"system": {}})
        fqdn = rendered["deploy"]["fqdn"]
        nginx_port = rendered["modules"]["web_terminals"]["nginx_port"]
        assert _landing_url(rendered, nginx_port) == f"http://{fqdn}:{nginx_port}"


# ---------------------------------------------------------------------------
# The persona pair: single-axis contract
# ---------------------------------------------------------------------------


class TestControlAssistantPersonas:
    """The readonly/readwrite pair: identical projects except for the one
    write-switch key. Any second difference that creeps in would turn the
    multi-user story ("one switch, every write surface") into a lie, so the
    invariant is asserted wholesale rather than key-by-key."""

    def test_readonly_extends_base_and_disables_writes(self) -> None:
        profile = resolve_preset("control-assistant-readonly")
        assert profile.name == "Control Assistant (Read-Only)"
        assert profile.data_bundle == "control_assistant"
        assert profile.config.get(WRITES_KEY) is False
        # Flat dotted key, never nested YAML under config:.
        assert "control_system" not in profile.config

    def test_readwrite_extends_base_and_enables_writes(self) -> None:
        profile = resolve_preset("control-assistant-readwrite")
        assert profile.name == "Control Assistant (Read-Write)"
        assert profile.data_bundle == "control_assistant"
        assert profile.config.get(WRITES_KEY) is True
        assert "control_system" not in profile.config

    def test_personas_retain_base_config_overrides(self) -> None:
        """Toggling the write switch must not drop the base's own config
        overrides — a representative base override survives both merges."""
        for name in ("control-assistant-readonly", "control-assistant-readwrite"):
            profile = resolve_preset(name)
            assert profile.config.get("control_system.type") == "virtual_accelerator"

    def test_personas_differ_only_on_writes_enabled(self) -> None:
        readonly = resolve_preset("control-assistant-readonly")
        readwrite = resolve_preset("control-assistant-readwrite")

        ro_cfg = dict(readonly.config)
        rw_cfg = dict(readwrite.config)
        assert ro_cfg.pop(WRITES_KEY) is False
        assert rw_cfg.pop(WRITES_KEY) is True
        # With the axis key removed, the rendered config overrides are identical.
        assert ro_cfg == rw_cfg

    def test_personas_share_every_artifact_list(self) -> None:
        """No tier is defined by artifact removal — both inherit the tutorial's
        full artifact set verbatim, scan skills and panels included (the
        boundary is enforcement, not absence)."""
        readonly = resolve_preset("control-assistant-readonly")
        readwrite = resolve_preset("control-assistant-readwrite")
        base = resolve_preset("control-assistant")
        for persona in (readonly, readwrite):
            assert persona.skills == base.skills
            assert persona.rules == base.rules
            assert persona.hooks == base.hooks
            assert persona.agents == base.agents
            assert persona.output_styles == base.output_styles
            assert persona.web_panels == base.web_panels

    def test_safety_chain_hooks_are_shipped(self) -> None:
        """The write-capable tier is supervised, not unguarded: the hooks that
        gate a write (writes-check, limits, approval) ship in the base and are
        inherited by both personas."""
        base = resolve_preset("control-assistant")
        for hook in ("writes-check", "limits", "approval"):
            assert hook in base.hooks

    def test_personas_are_attached(self) -> None:
        """Both personas set ``deploy_services: false`` — they build terminal
        images only, and the scan/VA/dispatch injector blocks inherited from
        the base are gated on this flag and skip cleanly. The hosting base
        keeps the default self-contained posture."""
        assert resolve_preset("control-assistant").deploy_services is True
        assert resolve_preset("control-assistant-readonly").deploy_services is False
        assert resolve_preset("control-assistant-readwrite").deploy_services is False

    def test_personas_do_not_host_a_second_web_tier(self) -> None:
        """Each persona pins ``modules.web_terminals.enabled: false`` so a
        persona-dir deploy never races the hosting project for the web ports."""
        for name in ("control-assistant-readonly", "control-assistant-readwrite"):
            profile = resolve_preset(name)
            assert profile.config.get("modules.web_terminals.enabled") is False


# ---------------------------------------------------------------------------
# Web-terminal context overlay (seeding's base.md requirement)
# ---------------------------------------------------------------------------


class TestWebTerminalContextShipped:
    """Every built project — not just the ``control_assistant`` bundle —
    carries the ``docker/web-terminal-context/base.md`` that seeding
    requires. base.md is framework-layer: any project that seeds a web
    terminal user needs it, so it ships from the framework template root
    rather than from one bundle. Without it, ``osprey up`` brings up the whole
    stack and then aborts at the seed step.

    The path is PROJECT-relative (``seeding._CONTEXT_RELPATH``), and in a
    deployment repo the rendered project is the ``build/`` zone — which is why
    the seeding test below places the render there rather than treating it as
    the repo root."""

    def test_built_project_ships_base_md(self, tmp_path: Path) -> None:
        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        project_dir = TemplateManager().create_project(
            project_name="ctx-ship-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        base_md = project_dir / seeding._CONTEXT_RELPATH / "base.md"
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""

    def test_base_md_is_package_data(self) -> None:
        """The template source is reachable through ``importlib.resources``, so
        it is packaged rather than only present in a source checkout — a build
        from an installed wheel has nothing to copy otherwise."""
        from importlib.resources import files

        base_md = files("osprey.templates").joinpath("claude_code/web-terminal-context/base.md")
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""

    def test_non_control_assistant_bundle_ships_base_md(self, tmp_path: Path) -> None:
        """A bundle that ships no persona content of its own still gets the
        framework baseline."""
        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        project_dir = TemplateManager().create_project(
            project_name="ctx-ship-hello",
            output_dir=tmp_path,
            data_bundle="hello_world",
        )
        base_md = project_dir / seeding._CONTEXT_RELPATH / "base.md"
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""

    def test_non_control_assistant_bundle_seeds_successfully(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A seeded roster on a non-control_assistant bundle reaches the
        CLAUDE.md write instead of aborting on a missing base.md — the
        pre-flight ``RuntimeError`` such a project hit before base.md became
        framework-layer.

        The render is placed where a deployment repo keeps it — as the repo's
        ``build/`` zone — because that is where seeding resolves the overlay
        tree. Seeding from the repo root is the deploy verbs' working
        directory, so this exercises the real pairing rather than a bare
        project directory no deployment has.
        """
        import shutil
        import subprocess

        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        rendered = TemplateManager().create_project(
            project_name="ctx-seed-hello",
            output_dir=tmp_path,
            data_bundle="hello_world",
        )
        repo = tmp_path / "repo"
        repo.mkdir()
        project_dir = repo / "build"
        shutil.move(str(rendered), str(project_dir))
        base_content = (project_dir / seeding._CONTEXT_RELPATH / "base.md").read_text(
            encoding="utf-8"
        )

        container = "dls-web-alice"
        seeded: list[bytes | None] = []

        def _fake_run(argv, capture_output=True, text=False, env=None, check=False, input=None):
            if argv[1] == "inspect":
                rc = 0 if argv[-1] == container else 1
                return subprocess.CompletedProcess(argv, returncode=rc, stdout="", stderr="")
            if argv[1] == "exec" and "id -u" in argv[-1]:
                return subprocess.CompletedProcess(
                    argv, returncode=0, stdout="1000:1000\n", stderr=""
                )
            if len(argv) >= 9 and "cat >" in argv[8]:
                seeded.append(input)
            return subprocess.CompletedProcess(argv, returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(seeding.subprocess, "run", _fake_run)
        monkeypatch.setattr(seeding, "get_runtime_command", lambda config=None: ["docker"])
        monkeypatch.setattr(seeding, "runtime_env", lambda config, base_env=None, **kw: {})
        monkeypatch.chdir(repo)

        seeding.seed_user_containers(
            {
                "project_name": "ctx-seed-hello",
                "facility": {"name": "Demo", "prefix": "dls", "timezone": "UTC"},
                "modules": {"web_terminals": {"enabled": True, "users": ["alice"]}},
            }
        )

        assert seeded == [base_content.encode("utf-8")]
