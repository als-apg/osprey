"""Tests for the control-assistant persona presets.

The ``control-assistant`` base preset ships the full control-room capability
set. Two persona presets extend it to carve out capability tiers:

* ``control-assistant-physicist`` — full-capability tier; opts into the scan
  MCP server for Bluesky-bridge scan plans (this file).
* ``control-assistant-operator`` — reduced tier (appended by a later task).

The base's own ``web_terminals`` block is also exercised here (appended by a
later task). Each persona owns its own test section below; shared render
helpers live at the top so new sections append without restructuring.
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

# Config key the bluesky MCP server reads to decide whether to register (see
# osprey.registry.mcp — the ``bluesky`` server is ``default_enabled=False``).
SCAN_ENABLED_KEY = "claude_code.servers.bluesky.enabled"


def resolve_preset(name: str) -> BuildProfile:
    """Resolve a bundled preset by name, fully merging its ``extends`` chain.

    Args:
        name: Bundled preset name (hyphenated or underscored).

    Returns:
        The parsed and validated :class:`BuildProfile`.
    """
    profile, _profile_dir = resolve_build_profile(None, preset=name)
    return profile


# ---------------------------------------------------------------------------
# Physicist persona (full-capability tier)
# ---------------------------------------------------------------------------


class TestPhysicistPersona:
    """The physicist preset extends the base and opts into scan tooling."""

    def test_extends_control_assistant_base(self) -> None:
        """The physicist preset inherits the base's identity fields."""
        profile = resolve_preset("control-assistant-physicist")
        # Child wins on name; base flows through for the rest.
        assert profile.name == "Control Assistant (Physicist)"
        assert profile.data_bundle == "control_assistant"
        assert profile.provider == "anthropic"

    def test_enables_scan_server(self) -> None:
        """The scan MCP server is opted in via the literal dotted config key."""
        profile = resolve_preset("control-assistant-physicist")
        assert profile.config.get(SCAN_ENABLED_KEY) is True

    def test_scan_key_is_flat_not_nested(self) -> None:
        """The override is a flat dotted key, never nested YAML under ``config:``.

        Nested YAML would deep-merge into and clobber the base's rendered config
        subtree, so the physicist must carry the literal ``key.path`` form.
        """
        profile = resolve_preset("control-assistant-physicist")
        assert SCAN_ENABLED_KEY in profile.config
        assert "claude_code" not in profile.config

    def test_retains_full_base_skill_set(self) -> None:
        """Extending the base must not drop any base skills, and in particular
        must keep ``writing-bluesky-plans`` — the scan-plan authoring skill the
        physicist's scan server backs.
        """
        base = resolve_preset("control-assistant")
        physicist = resolve_preset("control-assistant-physicist")
        # Every base skill is retained (extends unions string lists).
        assert set(base.skills).issubset(set(physicist.skills))
        assert "writing-bluesky-plans" in physicist.skills

    def test_retains_base_config_overrides(self) -> None:
        """Adding the scan key must not drop the base's own config overrides."""
        physicist = resolve_preset("control-assistant-physicist")
        # A representative base override survives the merge.
        assert physicist.config.get("control_system.type") == "mock"


# ---------------------------------------------------------------------------
# Operator persona (restricted control-room tier)
# ---------------------------------------------------------------------------


class TestOperatorPersona:
    """The operator preset extends the base and drops scan-plan authoring."""

    def test_extends_control_assistant_base(self) -> None:
        """The operator preset inherits the base's identity fields."""
        profile = resolve_preset("control-assistant-operator")
        # Child wins on name; base flows through for the rest.
        assert profile.name == "Control Assistant (Operator)"
        assert profile.data_bundle == "control_assistant"
        assert profile.provider == "anthropic"

    def test_excludes_bluesky_scan_skill(self) -> None:
        """The scan-plan authoring skill is subtracted from the inherited set."""
        base = resolve_preset("control-assistant")
        operator = resolve_preset("control-assistant-operator")
        # The base ships the skill; the operator tier must not.
        assert "writing-bluesky-plans" in base.skills
        assert "writing-bluesky-plans" not in operator.skills

    def test_retains_all_other_base_skills(self) -> None:
        """Excluding one skill must not disturb the rest of the inherited set."""
        base = resolve_preset("control-assistant")
        operator = resolve_preset("control-assistant-operator")
        # Only the scan skill is removed; every other base skill survives.
        expected = [s for s in base.skills if s != "writing-bluesky-plans"]
        assert operator.skills == expected

    def test_disables_scan_server(self) -> None:
        """The scan MCP server is explicitly denied via the literal dotted key."""
        profile = resolve_preset("control-assistant-operator")
        assert profile.config.get(SCAN_ENABLED_KEY) is False

    def test_scan_key_is_flat_not_nested(self) -> None:
        """The override is a flat dotted key, never nested YAML under ``config:``.

        Nested YAML would deep-merge into and clobber the base's rendered config
        subtree, so the operator must carry the literal ``key.path`` form.
        """
        profile = resolve_preset("control-assistant-operator")
        assert SCAN_ENABLED_KEY in profile.config
        assert "claude_code" not in profile.config

    def test_retains_base_config_overrides(self) -> None:
        """Denying the scan server must not drop the base's own config overrides."""
        operator = resolve_preset("control-assistant-operator")
        # A representative base override survives the merge.
        assert operator.config.get("control_system.type") == "mock"

    def test_leaves_other_artifact_lists_intact(self) -> None:
        """Only ``skills`` is touched; all other base artifact lists pass through.

        The ``exclude:`` mechanism subtracts from ``skills`` alone, so rules,
        hooks, agents, output styles, web panels, and dependencies must match the
        base preset entry-for-entry.
        """
        base = resolve_preset("control-assistant")
        operator = resolve_preset("control-assistant-operator")
        assert operator.rules == base.rules
        assert operator.hooks == base.hooks
        assert operator.agents == base.agents
        assert operator.output_styles == base.output_styles
        assert operator.web_panels == base.web_panels
        assert operator.dependencies == base.dependencies


# ---------------------------------------------------------------------------
# Persona attachment (deploy_services: false)
# ---------------------------------------------------------------------------


class TestPersonaAttachment:
    """Both personas are attached projects: they build per-user terminal images
    only and connect to the shared services stack the base project deploys, so
    they must set ``deploy_services: false`` while the base leaves it defaulted
    true. Their inherited service blocks still parse and validate — they just
    scaffold nothing at build time.
    """

    def test_base_is_self_contained(self) -> None:
        """The base preset deploys its own services stack (default posture)."""
        assert resolve_preset("control-assistant").deploy_services is True

    def test_operator_is_attached(self) -> None:
        assert resolve_preset("control-assistant-operator").deploy_services is False

    def test_physicist_is_attached(self) -> None:
        assert resolve_preset("control-assistant-physicist").deploy_services is False

    def test_attached_personas_still_validate(self) -> None:
        """Marking a persona attached does not disturb the rest of the profile —
        ``resolve_preset`` parses and validates it (raising on any error).
        The inherited bluesky/virtual_accelerator/bluesky_panels blocks survive.
        """
        for name in ("control-assistant-operator", "control-assistant-physicist"):
            profile = resolve_preset(name)
            assert profile.bluesky is not None
            assert profile.virtual_accelerator is not None
            assert profile.bluesky_panels is not None


# ---------------------------------------------------------------------------
# Base preset: multi-user web-terminal block
# ---------------------------------------------------------------------------

# The literal dotted key the base preset must carry: the whole web-terminals
# module subtree addressed as one leaf so config_writer sets only this leaf and
# never wholesale-replaces the rendered ``modules`` subtree.
WEB_TERMINALS_KEY = "modules.web_terminals"


def _render_config_overrides(tmp_path: Path, seed: dict) -> dict:
    """Render the base preset's ``config:`` overrides onto ``seed`` exactly as
    the build pipeline does — via :func:`config_update_fields`, the same
    dot-notation writer ``_apply_config_overrides`` calls — and reload the
    result as a plain dict.

    Args:
        tmp_path: Per-test temp directory (pytest fixture).
        seed: Pre-existing config contents to render the overrides onto.

    Returns:
        The reloaded config after applying the base preset's overrides.
    """
    config_path = tmp_path / "config.yml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(seed, fh)
    overrides = resolve_preset("control-assistant").config
    config_update_fields(config_path, overrides)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


class TestBaseWebTerminals:
    """The base preset ships the two-persona multi-user web-terminal stack."""

    def test_web_terminals_is_one_literal_dotted_key(self) -> None:
        """The block is carried as the single literal ``modules.web_terminals``
        key, never a nested ``modules:`` mapping under ``config:``.

        A nested mapping would deep-merge over the rendered ``modules`` subtree
        and drop sibling modules; the literal dotted key sets only its own leaf.
        """
        base = resolve_preset("control-assistant")
        assert WEB_TERMINALS_KEY in base.config
        assert "modules" not in base.config

    def test_ships_bluesky_bridge_on_8090(self) -> None:
        """The top-level ``bluesky:`` block ships the bridge on port 8090 — the
        port the physicist tier's scan MCP server defaults ``BLUESKY_BRIDGE_URL``
        to — so ``_inject_bluesky`` deploys the bridge service."""
        base = resolve_preset("control-assistant")
        assert base.bluesky is not None
        assert base.bluesky.port == 8090

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
        demo shape: local image source, operator default, an operator/physicist
        catalog whose ``project`` equals its ``project_path`` basename, and a
        roster mapping alice→operator (via default) and bob→physicist."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        wt = rendered["modules"]["web_terminals"]

        assert wt["enabled"] is True
        assert wt["image_source"] == "local"
        assert wt["default_persona"] == "operator"
        assert wt["nginx_port"] == 9080
        assert wt["web_base_port"] == 9091
        assert wt["artifact_base_port"] == 9291
        assert wt["ariel_base_port"] == 9391
        assert wt["lattice_base_port"] == 9491

        # Roster: alice is a bare string (inherits default_persona: operator);
        # bob is object-form with an explicit index and the physicist persona.
        assert wt["users"][0] == "alice"
        assert wt["users"][1] == {"name": "bob", "index": 1, "persona": "physicist"}

        personas = wt["personas"]
        assert set(personas) == {"operator", "physicist"}
        for name, profile in (
            ("operator", "control-assistant-operator"),
            ("physicist", "control-assistant-physicist"),
        ):
            entry = personas[name]
            # Name invariant: project == basename(project_path).
            assert entry["project"] == os.path.basename(entry["project_path"])
            assert entry["build_profile"] == profile

    def test_rendered_config_lints_without_errors(self, tmp_path: Path) -> None:
        """``lint_web_terminals`` on the freshly-rendered demo config reports
        zero ERROR findings pre-deploy.

        The referenced persona projects do not exist yet at build time; the lint
        demotes those missing-but-auto-renderable paths to informational findings
        (they carry a ``build_profile`` deploy up renders from), so the gate is
        clean before any project is rendered.
        """
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        assert _errors(lint_web_terminals(rendered)) == []


# ---------------------------------------------------------------------------
# Facility prefix (web container-name prefix)
# ---------------------------------------------------------------------------

# The key config_writer stores the base preset's facility-prefix override under,
# and the leaf it renders to in the nested config.
FACILITY_PREFIX_KEY = "facility.prefix"

# A valid Docker container/object name: must start with an alphanumeric, then
# alphanumerics plus `_`, `.`, `-` (see docker/docker daemon name validation).
DOCKER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


class TestFacilityPrefix:
    """The base preset sets a non-empty ``facility.prefix`` so the web stack's
    container names are valid Docker names.

    Web container names are ``<facility_prefix>-nginx`` and
    ``<facility_prefix>-web-<user>`` (see
    ``osprey.deployment.web_terminals.seeding`` / the compose template). An empty
    prefix renders leading-dash names like ``-nginx``, which Docker rejects and
    ``osprey deploy up`` fails the web stack on — the defect this guards against.
    """

    def test_base_config_sets_nonempty_facility_prefix(self) -> None:
        """The override is carried as the literal ``facility.prefix`` dotted key
        (never a nested ``facility:`` mapping) with a non-empty value."""
        base = resolve_preset("control-assistant")
        assert FACILITY_PREFIX_KEY in base.config
        assert "facility" not in base.config
        prefix = base.config[FACILITY_PREFIX_KEY]
        assert isinstance(prefix, str) and prefix != ""

    def test_rendered_config_has_nonempty_facility_prefix(self, tmp_path: Path) -> None:
        """Rendering the base overrides lands a non-empty ``facility.prefix`` leaf
        in the nested config the deploy pipeline reads."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        prefix = rendered["facility"]["prefix"]
        assert isinstance(prefix, str) and prefix != ""

    def test_derived_web_container_names_are_valid_docker_names(self, tmp_path: Path) -> None:
        """The container names derived from the rendered prefix — ``<prefix>-nginx``
        and ``<prefix>-web-<user>`` for each demo user — all match the Docker name
        grammar (start alphanumeric, no leading dash)."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        prefix = rendered["facility"]["prefix"]

        container_names = [
            f"{prefix}-nginx",
            f"{prefix}-web-alice",
            f"{prefix}-web-bob",
        ]
        for name in container_names:
            assert DOCKER_NAME_RE.match(name), f"invalid Docker container name: {name!r}"


# ---------------------------------------------------------------------------
# Deploy target (landing URL origin)
# ---------------------------------------------------------------------------


class TestDeployFqdn:
    """The base preset sets ``deploy.fqdn`` so the web-terminals render step
    can build the landing URL (``OSPREY_TERMINAL_LANDING_URL``) without manual
    config edits — ``_landing_url`` raises without it, aborting
    ``osprey deploy up`` for the otherwise zero-config demo."""

    def test_base_config_sets_deploy_fqdn(self) -> None:
        """Carried as the literal ``deploy.fqdn`` dotted key (never a nested
        ``deploy:`` mapping) with a non-empty string value."""
        base = resolve_preset("control-assistant")
        assert "deploy.fqdn" in base.config
        assert "deploy" not in base.config
        fqdn = base.config["deploy.fqdn"]
        assert isinstance(fqdn, str) and fqdn != ""

    def test_rendered_config_satisfies_landing_url(self, tmp_path: Path) -> None:
        """The rendered config passes the exact check ``deploy up`` runs."""
        from osprey.deployment.web_terminals.render import _landing_url

        rendered = _render_config_overrides(tmp_path, {"system": {}})
        fqdn = rendered["deploy"]["fqdn"]
        nginx_port = rendered["modules"]["web_terminals"]["nginx_port"]
        assert _landing_url(rendered, nginx_port) == f"http://{fqdn}:{nginx_port}"


# ---------------------------------------------------------------------------
# Web-terminal context overlay (seeding's base.md requirement)
# ---------------------------------------------------------------------------


class TestWebTerminalContextShipped:
    """A project built from the ``control_assistant`` bundle carries the
    ``docker/web-terminal-context/base.md`` that seeding requires — without
    it, ``osprey deploy up`` brings up the whole stack and then aborts at the
    seed step."""

    def test_built_project_ships_base_md(self, tmp_path: Path) -> None:
        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        project_dir = TemplateManager().create_project(
            project_name="ctx-ship-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        base_md = project_dir / seeding._CONTEXT_DIR / "base.md"
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""
