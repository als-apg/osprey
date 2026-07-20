"""Tests for the multi-user demo persona presets.

The ``multi-user-demo`` base preset is the hosting project for the multi-user
web-terminal demo: nginx, the landing page, and one terminal container per
roster user — deliberately scan-free (no Bluesky bridge, no Virtual
Accelerator, no panels sidecar). Two persona presets extend it to carve out
capability postures that differ on exactly one axis,
``control_system.writes_enabled``:

* ``multi-user-demo-readwrite`` — write-capable tier; channel writes pass the
  ordinary safety chain (writes-check, limits, human approval).
* ``multi-user-demo-readonly`` — read-only tier; every write surface refuses.

The base's own ``web_terminals`` roster block is also exercised here. Each
persona owns its own test section below; shared render helpers live at the top
so new sections append without restructuring.
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
# Read-write persona (write-capable tier)
# ---------------------------------------------------------------------------


class TestReadwritePersona:
    """The readwrite preset extends the base and arms the write path."""

    def test_extends_multi_user_demo_base(self) -> None:
        """The readwrite preset inherits the base's identity fields."""
        profile = resolve_preset("multi-user-demo-readwrite")
        # Child wins on name; base flows through for the rest.
        assert profile.name == "Multi-User Demo (Read-Write)"
        assert profile.data_bundle == "control_assistant"
        assert profile.provider == "anthropic"

    def test_enables_writes(self) -> None:
        """The write switch is armed via the literal dotted config key."""
        profile = resolve_preset("multi-user-demo-readwrite")
        assert profile.config.get(WRITES_KEY) is True

    def test_writes_key_is_flat_not_nested(self) -> None:
        """The override is a flat dotted key, never nested YAML under ``config:``.

        Nested YAML would deep-merge into and clobber the base's rendered config
        subtree, so the persona must carry the literal ``key.path`` form.
        """
        profile = resolve_preset("multi-user-demo-readwrite")
        assert WRITES_KEY in profile.config
        assert "control_system" not in profile.config

    def test_retains_base_config_overrides(self) -> None:
        """Arming writes must not drop the base's own config overrides."""
        profile = resolve_preset("multi-user-demo-readwrite")
        # A representative base override survives the merge.
        assert profile.config.get("control_system.type") == "mock"


# ---------------------------------------------------------------------------
# Read-only persona (observation tier)
# ---------------------------------------------------------------------------


class TestReadonlyPersona:
    """The readonly preset extends the base and pins the write switch off."""

    def test_extends_multi_user_demo_base(self) -> None:
        """The readonly preset inherits the base's identity fields."""
        profile = resolve_preset("multi-user-demo-readonly")
        # Child wins on name; base flows through for the rest.
        assert profile.name == "Multi-User Demo (Read-Only)"
        assert profile.data_bundle == "control_assistant"
        assert profile.provider == "anthropic"

    def test_disables_writes(self) -> None:
        """The write switch is pinned off via the literal dotted config key.

        The base already defaults it false; the persona pins it explicitly so
        the tier boundary cannot drift silently if the base's default changes.
        """
        profile = resolve_preset("multi-user-demo-readonly")
        assert profile.config.get(WRITES_KEY) is False

    def test_writes_key_is_flat_not_nested(self) -> None:
        """The override is a flat dotted key, never nested YAML under ``config:``."""
        profile = resolve_preset("multi-user-demo-readonly")
        assert WRITES_KEY in profile.config
        assert "control_system" not in profile.config

    def test_retains_base_config_overrides(self) -> None:
        """Pinning writes off must not drop the base's own config overrides."""
        profile = resolve_preset("multi-user-demo-readonly")
        # A representative base override survives the merge.
        assert profile.config.get("control_system.type") == "mock"


# ---------------------------------------------------------------------------
# The single-axis invariant
# ---------------------------------------------------------------------------


class TestSingleAxis:
    """The demo's whole point: the two tiers are identical projects except for
    the one write-switch key. Any second difference that creeps in would turn
    the demo's story ("one switch, every write surface") into a lie, so the
    invariant is asserted wholesale rather than key-by-key.
    """

    def test_personas_differ_only_on_writes_enabled(self) -> None:
        readonly = resolve_preset("multi-user-demo-readonly")
        readwrite = resolve_preset("multi-user-demo-readwrite")

        ro_cfg = dict(readonly.config)
        rw_cfg = dict(readwrite.config)
        assert ro_cfg.pop(WRITES_KEY) is False
        assert rw_cfg.pop(WRITES_KEY) is True
        # With the axis key removed, the rendered config overrides are identical.
        assert ro_cfg == rw_cfg

    def test_personas_share_every_artifact_list(self) -> None:
        """No tier is defined by artifact removal — both inherit the base's
        artifact set verbatim (the boundary is enforcement, not absence)."""
        readonly = resolve_preset("multi-user-demo-readonly")
        readwrite = resolve_preset("multi-user-demo-readwrite")
        base = resolve_preset("multi-user-demo")
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
        base = resolve_preset("multi-user-demo")
        for hook in ("writes-check", "limits", "approval"):
            assert hook in base.hooks


# ---------------------------------------------------------------------------
# Persona attachment (deploy_services: false) + scan-free posture
# ---------------------------------------------------------------------------


class TestPersonaAttachment:
    """Both personas are attached projects: they build per-user terminal images
    only and connect to the shared web tier the base project deploys, so they
    must set ``deploy_services: false`` while the base leaves it defaulted true.
    """

    def test_base_is_self_contained(self) -> None:
        """The base preset deploys its own web tier (default posture)."""
        assert resolve_preset("multi-user-demo").deploy_services is True

    def test_readonly_is_attached(self) -> None:
        assert resolve_preset("multi-user-demo-readonly").deploy_services is False

    def test_readwrite_is_attached(self) -> None:
        assert resolve_preset("multi-user-demo-readwrite").deploy_services is False

    def test_family_is_scan_free(self) -> None:
        """The demo family deliberately ships no scan stack: no Bluesky bridge,
        no Virtual Accelerator, no panels sidecar, no event dispatch. The full
        scan stack lives in ``control-assistant``; keeping it out of this family
        is what keeps the multi-user demo about multi-user provisioning.
        """
        for name in (
            "multi-user-demo",
            "multi-user-demo-readonly",
            "multi-user-demo-readwrite",
        ):
            profile = resolve_preset(name)
            assert profile.bluesky is None
            assert profile.virtual_accelerator is None
            assert profile.bluesky_panels is None
            assert profile.dispatch is None


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
    overrides = resolve_preset("multi-user-demo").config
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
        base = resolve_preset("multi-user-demo")
        assert WEB_TERMINALS_KEY in base.config
        assert "modules" not in base.config

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
        demo shape: local image source, readonly default, a readonly/readwrite
        catalog whose ``project`` equals its ``project_path`` basename, and a
        roster mapping alice→readonly (via default) and bob→readwrite."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        wt = rendered["modules"]["web_terminals"]

        assert wt["enabled"] is True
        assert wt["image_source"] == "local"
        assert wt["default_persona"] == "readonly"
        assert wt["nginx_port"] == 9080
        assert wt["web_base_port"] == 9091
        assert wt["artifact_base_port"] == 9291
        assert wt["ariel_base_port"] == 9391
        assert wt["lattice_base_port"] == 9491
        assert wt["channel_finder_base_port"] == 9591

        # Roster: alice is a bare string (inherits default_persona: readonly);
        # bob is object-form with an explicit index and the readwrite persona.
        assert wt["users"][0] == "alice"
        assert wt["users"][1] == {"name": "bob", "index": 1, "persona": "readwrite"}

        personas = wt["personas"]
        assert set(personas) == {"readonly", "readwrite"}
        for name, profile in (
            ("readonly", "multi-user-demo-readonly"),
            ("readwrite", "multi-user-demo-readwrite"),
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

    def test_demo_ships_companion_panels_multi_user(self) -> None:
        """Feature parity: multi-user must not shed single-user companion panels.

        The channel-finder panel was once dropped from this preset to dodge a
        per-user port collision (its family was missing from the port-family
        derivation) — the fix is per-user ports, never removing the feature.
        """
        base = resolve_preset("multi-user-demo")
        assert "channel-finder" in base.web_panels
        assert "ariel" in base.web_panels

    def test_control_assistant_no_longer_hosts_the_web_tier(self) -> None:
        """The multi-user roster moved out of ``control-assistant`` into this
        family — the scan tutorial preset must not carry a web-terminals block
        (or its web-tier-only companion keys) referencing deleted personas."""
        base = resolve_preset("control-assistant")
        assert WEB_TERMINALS_KEY not in base.config
        assert "facility.prefix" not in base.config
        assert "deploy.fqdn" not in base.config


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
        base = resolve_preset("multi-user-demo")
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
        base = resolve_preset("multi-user-demo")
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
