"""Tests for the control-assistant multi-user persona presets.

The ``control-assistant`` preset hosts its own multi-user web tier — nginx,
the landing page, and one terminal container per roster user — alongside the
full plan stack. Four persona presets extend it. Three of them are capability
TIERS over the same deployment, and one is a standalone service that happens to
live on the same landing page:

* ``control-assistant-readonly`` — read-only tier; simple chat-first surface;
  every write surface refuses; built without the EVENTS/BLUESKY panels.
* ``control-assistant-readwrite`` — write-capable tier; expert workspace;
  channel writes pass the ordinary safety chain (writes-check, limits, human
  approval); declares the EVENTS/BLUESKY panels.
* ``control-assistant-admin`` — deployment-editing tier; readwrite's write
  posture over the MACHINE (``writes_enabled``, ``ui_mode``) but NOT its
  operator panels, plus the privileges the base floors off: the
  ``setup_patch`` tool, the web Config panel, the scaffold gallery's editors,
  and the ``setup-mode`` skill that drives them.
* ``control-assistant-ariel`` — not a tier: the standalone logbook-research
  deployment, filed under its own landing-page heading.

The tier contract therefore has two halves, and both are asserted wholesale
below rather than key-by-key. The MACHINE axes separate readonly from
readwrite: enforcement (``control_system.writes_enabled``), surface
(``web.ui_mode``), and the write-oriented panel declarations (EVENTS +
BLUESKY, readwrite-only). The DEPLOYMENT axes separate admin from both:
``claude_code.permissions.remove_deny`` for the ``setup_patch`` tool,
``web.config_panel.enabled``, ``web.scaffold_gallery.write_enabled``, and the
``setup-mode`` skill. The base pins the restricted side of every deployment
axis, so a new persona inherits the floor and has to ask for a privilege by
name.

The base's own ``web_terminals`` roster block is also exercised here. Shared
render helpers live at the top so new sections append without restructuring.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.deployment.qmd_service import DEFAULT_PORT as QMD_DEFAULT_PORT
from osprey.deployment.qmd_service import resolve_qmd_service_config
from osprey.deployment.web_terminals.lint import Finding, lint_web_terminals
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.utils.config_writer import config_update_fields

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# The enforcement axis the control-system tiers differ on — the reference
# monitor's master write switch (see osprey.connectors.control_system.base).
WRITES_KEY = "control_system.writes_enabled"
UI_MODE_KEY = "web.ui_mode"
# The agent's deployment-editing tool: denied by the base for every tier, and
# subtracted back by the admin tier alone. A deny rather than an ask because
# string lists only ever UNION across `extends` — see the base preset's note.
SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"
DENY_KEY = "claude_code.permissions.deny"
REMOVE_DENY_KEY = "claude_code.permissions.remove_deny"
# The two browser-side halves of the same privilege: editing the running
# deployment's configuration, and writing to the shared prompt/skill gallery.
CONFIG_PANEL_KEY = "web.config_panel.enabled"
GALLERY_WRITE_KEY = "web.scaffold_gallery.write_enabled"
# The two browser-side floor keys, which the base pins false and the admin tier
# flips true. They differ from the tool axis in HOW they differ: every tier
# carries these keys (they are inherited), so the admin delta is a value, not a
# presence. `REMOVE_DENY_KEY` is the opposite — no other tier declares it at
# all — and `DENY_KEY` is neither: the base's deny reaches every tier verbatim,
# admin included, which is exactly why the admin tier has to subtract it.
ADMIN_LIFTED_FLOOR_KEYS = (CONFIG_PANEL_KEY, GALLERY_WRITE_KEY)
# The skill that drives those surfaces from the agent side.
ADMIN_ONLY_SKILL = "setup-mode"
# The write-oriented panels: declared only in the readwrite persona's
# `web_panels` list, so the readonly build genuinely lacks them (a persona
# delta can only add; `enabled: false` is inert for URL panels). Their URL,
# path and label are NOT preset config — the build projects them from the
# hosting deployment's render (osprey.deployment.reach), which is asserted on
# a real build in test_readwrite_is_told_its_panel_urls.
READWRITE_PANELS = ("events", "bluesky")

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


def _build_persona_stack(repo: Path) -> Path:
    """``osprey init`` the hosting preset, then ``osprey build`` the whole stack.

    One build renders the host project at ``build/`` and one project per catalog
    persona beside it at ``build/<repo>-<persona>``. Nothing here is stubbed: the
    config a persona render reads is the one the pipeline actually assembles, and
    a persona's ``config:`` overlay is applied by the build between the base
    template render and the Claude Code re-render — an ordering that decides
    whether a preset key can reach the agent surface at all, and one no
    config-only helper in this module can observe.

    Args:
        repo: Directory to materialize the deployment repo into.

    Returns:
        The repo root; persona renders live under ``<repo>/build/``.
    """
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    built = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return repo


def _graph_permission_entries() -> list[str]:
    """The four rendered permission strings, derived from the registry.

    ``ServerDefinition.permissions_allow`` holds BARE tool names — the settings
    template splices the ``mcp__<server>__`` prefix onto the whole list — so the
    qualification happens here rather than being restated where it could drift
    from what the render emits.
    """
    return sorted(f"mcp__graph__{tool}" for tool in FRAMEWORK_SERVERS["graph"].permissions_allow)


def _graph_entries(values) -> list[str]:
    return sorted(entry for entry in values if str(entry).startswith("mcp__graph__"))


@pytest.fixture(scope="module")
def built_persona_stack(tmp_path_factory) -> Path:
    """The hosting preset built once, personas included."""
    return _build_persona_stack(tmp_path_factory.mktemp("persona-stack") / "my-facility")


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
        """The rendered ``modules.web_terminals`` subtree matches the shipped
        tutorial shape: local image source, readonly default, a
        readonly/readwrite/admin/ariel catalog whose ``project`` equals its
        ``project_path`` basename, and a roster mapping alice→readwrite,
        bob→readonly, ariel→ariel and carol→admin, all explicit, each carrying
        the tab-title ``display_name`` that visibly marks which terminal is
        which.

        The ariel entry is not a person: it is the standalone ARIEL logbook
        deployment the stack ships beside the operator tiers, and its catalog
        entry carries the ``landing_group`` that files its card under its own
        landing-page heading.

        Carol is a person, and the roster's least-privileged-by-default
        property is what her entry pins: ``default_persona`` stays ``readonly``
        even though an admin login now exists, so an unrouted visitor lands on
        the read-only tier rather than the one that can rewrite the deployment.
        Her index is last on purpose — indices drive the per-user port families,
        so inserting her anywhere else would renumber alice's and bob's
        published ports on every already-deployed stack.

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
        # Password login ships ON, in the demo posture: plain HTTP is only
        # acceptable because the preset's fqdn is loopback.
        assert wt["auth"] == {"method": "password", "allow_insecure_http": True}

        assert wt["users"][0] == {
            "name": "alice",
            "index": 0,
            "persona": "readwrite",
            "display_name": "Control Room (Alice)",
        }
        assert wt["users"][1] == {
            "name": "bob",
            "index": 1,
            "persona": "readonly",
            "display_name": "Read-Only View (Bob)",
        }
        # The standalone research tier is public by design: `login: false`
        # keeps its card outside the login wall the preset ships enabled.
        assert wt["users"][2] == {
            "name": "ariel",
            "index": 2,
            "persona": "ariel",
            "display_name": "ARIEL Logbook Research",
            "login": False,
        }
        # The admin login carries NO `login: false`: the one account that can
        # rewrite the deployment must sit behind the login wall, unlike the
        # deliberately-public research card above it.
        assert wt["users"][3] == {
            "name": "carol",
            "index": 3,
            "persona": "admin",
            "display_name": "Deployment Admin (Carol)",
        }
        assert len(wt["users"]) == 4

        personas = wt["personas"]
        assert set(personas) == {"readonly", "readwrite", "admin", "ariel"}
        for name, profile in (
            ("readonly", "control-assistant-readonly"),
            ("readwrite", "control-assistant-readwrite"),
            ("admin", "control-assistant-admin"),
            ("ariel", "control-assistant-ariel"),
        ):
            entry = personas[name]
            # Name invariant: project == basename(project_path).
            assert entry["project"] == os.path.basename(entry["project_path"])
            assert entry["build_profile"] == profile

        # Only the standalone tier declares a landing section of its own; the
        # three operator tiers stay in the roster's default section.
        assert personas["ariel"]["landing_group"] == "Standalone deployments"
        for tier in ("readonly", "readwrite", "admin"):
            assert "landing_group" not in personas[tier]

        # And the roster's own section is titled for the people in it.
        assert wt["landing"]["groups"] == [{"type": "users", "label": "Users"}]

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

    def test_rendered_config_lints_with_nothing_wrong_but_the_missing_render(
        self, tmp_path: Path
    ) -> None:
        """``lint_web_terminals`` on the freshly-built tutorial config reports
        nothing about the config's SHAPE pre-deploy.

        The referenced persona projects do not exist yet at build time. For the
        personas nobody is exposed by, the lint demotes those not-yet-rendered
        paths to WARNINGS (they carry a ``build_profile`` naming the delta
        ``osprey build`` renders them from). For a persona a ``login: false``
        entry resolves to, the absent render is an ERROR
        (``persona_privileges_unknown``): its privileges cannot be read, so
        "holds nothing" would be a guess about the one terminal that is served
        to anyone — and `osprey up` now gates on this belt, where that guess
        would be fail-open on the deploy path itself.

        Both findings name the same remedy and it is the command that comes
        next anyway: ``osprey build``. What this test pins is that nothing ELSE
        is reported — every error here is about the render that has not happened
        yet, not about the preset.

        Linted through :func:`_render_deployable_config`, i.e. after the catalog
        rewrite every build performs — the preset's own ``build_profile`` values
        are preset names, which no rendered config ever carries.
        """
        rendered = _render_deployable_config(tmp_path)

        errors = _errors(lint_web_terminals(rendered))

        assert {finding.code for finding in errors} <= {"web_terminals.persona_privileges_unknown"}
        for finding in errors:
            assert "osprey build" in finding.message
        # And the exposed entry is the one it is about: the shipped stack serves
        # `ariel` without a login, which is why its unread render is refused.
        assert any("'ariel'" in finding.message for finding in errors)

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
    """The three tiers: identical projects except for the tier contract.

    Over the MACHINE, readonly and readwrite differ on enforcement
    (``writes_enabled``), surface (``ui_mode``) and the write-oriented panel
    declarations (readwrite-only). Over the DEPLOYMENT, admin differs from
    readwrite on the three keys the base floors off — ``remove_deny`` for
    ``setup_patch``, the Config panel, the gallery's write surfaces — plus the
    ``setup-mode`` skill that drives them.

    Any difference OUTSIDE those two lists would turn the multi-user story into
    a lie, so both invariants are asserted wholesale rather than key-by-key."""

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

    def test_admin_extends_base_and_lifts_the_deployment_floor(self) -> None:
        """The admin tier is readwrite plus deployment editing, not a fourth
        write posture: it pins the same ``writes_enabled: true`` and
        ``ui_mode: expert``, and adds the three keys that lift the base floor.

        ``remove_deny`` rather than a bare absence of the deny is the whole
        mechanism — string lists UNION across ``extends``, so the base's deny
        reaches this profile too and can only be taken back by subtraction. The
        deny is therefore expected to be present HERE as well; what differs is
        that it is cancelled."""
        profile = resolve_preset("control-assistant-admin")
        assert profile.name == "Control Assistant (Admin)"
        assert profile.data_bundle == "control_assistant"
        assert profile.config.get(WRITES_KEY) is True
        assert profile.config.get(UI_MODE_KEY) == "expert"
        assert "control_system" not in profile.config

        assert profile.config.get(DENY_KEY) == [SETUP_PATCH_TOOL]
        assert profile.config.get(REMOVE_DENY_KEY) == [SETUP_PATCH_TOOL]
        assert profile.config.get(CONFIG_PANEL_KEY) is True
        assert profile.config.get(GALLERY_WRITE_KEY) is True

    def test_base_floors_the_deployment_privileges(self) -> None:
        """The base ships the restricted side of every deployment axis.

        This is what makes a privilege something a profile has to ask for by
        name: a new persona that extends ``control-assistant`` and says nothing
        inherits no ``setup_patch`` tool, no Config panel and no gallery
        editors. Both control-system tiers are checked too — a floor that only
        held on the base while a sibling quietly re-enabled a surface would
        pass a base-only assertion and still be broken. The ``ariel`` sibling
        inherits the same floor; its render is pinned next door in
        test_preset_render.py's ``TestControlAssistantTierFloor``."""
        base = resolve_preset("control-assistant")
        assert base.config.get(DENY_KEY) == [SETUP_PATCH_TOOL]
        assert base.config.get(CONFIG_PANEL_KEY) is False
        assert base.config.get(GALLERY_WRITE_KEY) is False
        # The floor is a deny, never a base-level `remove_ask`: an inherited
        # `remove_ask` would strip the approval prompt from EVERY tier,
        # admin included, because string lists cannot be subtracted downward.
        assert "claude_code.permissions.remove_ask" not in base.config

        for name in ("control-assistant-readonly", "control-assistant-readwrite"):
            tier = resolve_preset(name)
            assert tier.config.get(DENY_KEY) == [SETUP_PATCH_TOOL]
            assert REMOVE_DENY_KEY not in tier.config
            assert tier.config.get(CONFIG_PANEL_KEY) is False
            assert tier.config.get(GALLERY_WRITE_KEY) is False
            assert ADMIN_ONLY_SKILL not in tier.skills

    def test_admin_differs_from_readwrite_only_on_the_deployment_axis(self) -> None:
        """Admin sits directly on top of readwrite: same machine posture, plus
        deployment editing. Asserted wholesale so a fourth difference cannot
        creep in unnoticed.

        Two deliberate asymmetries are subtracted before the comparison. The
        deployment axes are the point of the tier. The EVENTS/BLUESKY panel
        declarations are not: they belong to the OPERATOR tier, and the admin
        persona is an editing surface rather than a second control desk, so it
        is built without them exactly as readonly is."""
        readwrite = resolve_preset("control-assistant-readwrite")
        admin = resolve_preset("control-assistant-admin")

        rw_cfg = dict(readwrite.config)
        ad_cfg = dict(admin.config)
        # Shared machine posture: both tiers are write-armed expert desks.
        assert rw_cfg[WRITES_KEY] is True and ad_cfg[WRITES_KEY] is True
        assert rw_cfg[UI_MODE_KEY] == "expert" and ad_cfg[UI_MODE_KEY] == "expert"
        # Deployment axis 1 — the agent's tool. The base's deny reaches BOTH
        # tiers identically; what makes admin different is the subtraction.
        assert rw_cfg[DENY_KEY] == [SETUP_PATCH_TOOL]
        assert ad_cfg[DENY_KEY] == [SETUP_PATCH_TOOL]
        assert REMOVE_DENY_KEY not in rw_cfg, "readwrite persona must not lift the floor"
        assert ad_cfg.pop(REMOVE_DENY_KEY) == [SETUP_PATCH_TOOL]
        # Deployment axes 2 and 3 — the browser surfaces. Inherited by both
        # tiers, so these differ on VALUE rather than presence.
        for key in ADMIN_LIFTED_FLOOR_KEYS:
            assert rw_cfg.pop(key) is False, f"readwrite persona must not enable {key}"
            assert ad_cfg.pop(key) is True
        # The operator-only panels ride the readwrite persona's `web_panels`
        # list, not its config: their URLs are projected by the build, so
        # neither tier may spell one.
        for panel in READWRITE_PANELS:
            for cfg in (rw_cfg, ad_cfg):
                assert not any(key.startswith(f"web.panels.{panel}.") for key in cfg), (
                    f"no persona preset may pin web.panels.{panel}.* — the build projects it"
                )
        assert ad_cfg == rw_cfg

    def test_personas_retain_base_config_overrides(self) -> None:
        """Toggling the write switch must not drop the base's own config
        overrides — a representative base override survives both merges."""
        for name in (
            "control-assistant-readonly",
            "control-assistant-readwrite",
            "control-assistant-admin",
        ):
            profile = resolve_preset(name)
            assert profile.config.get("control_system.type") == "virtual_accelerator"

    def test_personas_differ_only_on_the_tier_contract(self) -> None:
        readonly = resolve_preset("control-assistant-readonly")
        readwrite = resolve_preset("control-assistant-readwrite")

        ro_cfg = dict(readonly.config)
        rw_cfg = dict(readwrite.config)
        # Axis 1 — enforcement: the write switch.
        assert ro_cfg.pop(WRITES_KEY) is False
        assert rw_cfg.pop(WRITES_KEY) is True
        # Axis 2 — surface: chat-first for the viewer, full dock for the operator.
        assert ro_cfg.pop(UI_MODE_KEY) == "simple"
        assert rw_cfg.pop(UI_MODE_KEY) == "expert"
        # Axis 3 — write-oriented panels — is not a config axis at all: the
        # tabs ride the readwrite persona's `web_panels` list (asserted in
        # test_personas_share_every_artifact_list_except_panels_and_the_admin_skill), and their
        # URLs are projected from the hosting deployment's render rather than
        # spelled in any preset.
        for panel in READWRITE_PANELS:
            for cfg in (ro_cfg, rw_cfg):
                assert not any(key.startswith(f"web.panels.{panel}.") for key in cfg), (
                    f"no persona preset may pin web.panels.{panel}.* — the build projects it"
                )
        # With the tier-contract keys removed, the personas are identical.
        assert ro_cfg == rw_cfg

    def test_personas_share_every_artifact_list_except_panels_and_the_admin_skill(
        self,
    ) -> None:
        """No tier is defined by *tool* removal — rules, hooks, agents and
        output styles are inherited verbatim by all three personas (the tier
        boundary is enforcement, not a stripped-down agent). Exactly two lists
        are allowed to differ, and both differ by ADDITION:

        * panels — the readwrite persona adds the write-oriented EVENTS/BLUESKY
          tabs (their URLs are projected by the build, not declared), and the
          readonly persona is built without them;
        * skills — the admin persona adds ``setup-mode``, the guided workflow
          that edits config.yml and .mcp.json, which is the agent-side half of
          the privilege its config keys turn on.

        A list that SHRANK on a tier would be the failure mode this guards: it
        would make a tier a different agent rather than the same agent under a
        different posture."""
        readonly = resolve_preset("control-assistant-readonly")
        readwrite = resolve_preset("control-assistant-readwrite")
        admin = resolve_preset("control-assistant-admin")
        base = resolve_preset("control-assistant")
        for persona in (readonly, readwrite, admin):
            assert persona.rules == base.rules
            assert persona.hooks == base.hooks
            assert persona.agents == base.agents
            assert persona.output_styles == base.output_styles
        assert readonly.skills == base.skills
        assert readwrite.skills == base.skills
        # UNION, not replacement: the inherited selection survives underneath.
        assert admin.skills == [*base.skills, ADMIN_ONLY_SKILL]
        assert ADMIN_ONLY_SKILL not in base.skills
        assert readonly.web_panels == base.web_panels
        assert admin.web_panels == base.web_panels
        assert set(readwrite.web_panels) == set(base.web_panels) | {"events", "bluesky"}

    def test_safety_chain_hooks_are_shipped(self) -> None:
        """The write-capable tier is supervised, not unguarded: the hooks that
        gate a write (writes-check, limits, approval) ship in the base and are
        inherited by both personas."""
        base = resolve_preset("control-assistant")
        for hook in ("writes-check", "limits", "approval"):
            assert hook in base.hooks

    def test_personas_are_attached(self) -> None:
        """Both personas set ``deploy_services: false`` — they build terminal
        images only, and the bluesky/VA/dispatch injector blocks inherited from
        the base are gated on this flag and skip cleanly. The hosting base
        keeps the default self-contained posture."""
        assert resolve_preset("control-assistant").deploy_services is True
        assert resolve_preset("control-assistant-readonly").deploy_services is False
        assert resolve_preset("control-assistant-readwrite").deploy_services is False
        assert resolve_preset("control-assistant-admin").deploy_services is False

    def test_personas_do_not_host_a_second_web_tier(self) -> None:
        """Each persona pins ``modules.web_terminals.enabled: false`` so a
        persona-dir deploy never races the hosting project for the web ports."""
        for name in (
            "control-assistant-readonly",
            "control-assistant-readwrite",
            "control-assistant-admin",
        ):
            profile = resolve_preset(name)
            assert profile.config.get("modules.web_terminals.enabled") is False

    def test_personas_pin_no_service_address(self) -> None:
        """No tier spells where the hosting deployment's services are.

        The personas render attached (``deploy_services: false``), so the app
        template gives them ``services: {}`` — and the build then tells each
        render every client-facing fact from the hosting deployment's own
        render (``osprey.deployment.reach``): the graph store's bolt port, the
        qmd sidecar's port, the Postgres, the telemetry store, the bridge, the
        EVENTS and BLUESKY tab URLs. A preset that pinned one would be a second
        copy of a number the host already states, free to drift when an
        operator moves the service — and the build refuses such a pin.
        """
        for name in (
            "control-assistant-readonly",
            "control-assistant-readwrite",
            "control-assistant-admin",
            "control-assistant-ariel",
        ):
            profile = resolve_preset(name)
            pinned = sorted(
                key
                for key in profile.config
                if str(key).startswith(("services.", "web.panels.events.", "web.panels.bluesky."))
            )
            assert pinned == [], f"{name} pins {pinned}"
            assert "services" not in profile.config

    @pytest.mark.parametrize("persona", ("readonly", "readwrite", "admin"))
    def test_operator_personas_render_the_graph_server(
        self, built_persona_stack: Path, persona: str
    ) -> None:
        """Every tier's terminal can query the hosting deployment's graph.

        Asserted on a real build rather than on the resolved profile, because the
        claim spans the whole pipeline: the store's port has to be projected
        from the hosting deployment's render (``osprey.deployment.reach``),
        land as a ``services.graphdb`` block in the rendered config through
        ``_apply_config_overrides``, be seen by ``config_derived_context`` as
        ``graphdb_configured``, and only then reach ``resolve_servers`` before
        the Claude Code artifacts are written. A projection applied after
        server resolution would leave every assertion below false.

        The whole main-agent surface is checked — permissions, a launchable
        ``.mcp.json`` entry and the PostToolUse prefix, with nothing behind
        ``ask`` because these tools read — together with the fact that the
        server stops there: it is a main-agent server, so the channel-finder
        subagent's frontmatter must not name one of its tools.
        """
        project = built_persona_stack / "build" / f"{built_persona_stack.name}-{persona}"
        assert project.is_dir(), f"{persona} was never rendered"

        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        assert (config.get("services") or {}).get("graphdb") == {"port_host": 7687}

        permissions = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]
        assert _graph_entries(permissions["allow"]) == _graph_permission_entries()
        assert _graph_entries(permissions.get("ask") or []) == []

        graph_server = json.loads((project / ".mcp.json").read_text())["mcpServers"]["graph"]
        assert graph_server["args"] == ["-m", "osprey.mcp_server.graph"]

        hook_cfg = json.loads((project / ".claude" / "hooks" / "hook_config.json").read_text())
        assert "mcp__graph__" in hook_cfg["server_prefixes"]
        assert "mcp__graph__" not in hook_cfg["approval_prefixes"]

        frontmatter = yaml.safe_load(
            (project / ".claude" / "agents" / "channel-finder.md")
            .read_text(encoding="utf-8")
            .split("---")[1]
        )
        tools = [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]
        assert _graph_entries(tools) == []

    @pytest.mark.parametrize("persona", ("readonly", "readwrite"))
    def test_projected_facts_land_inside_the_attached_renders_services_map(
        self, built_persona_stack: Path, persona: str
    ) -> None:
        """The projected keys are written *into* the attached render's empty
        ``services`` map, carry the host's values, and leave nothing else behind.

        An attached persona scaffolds no services of its own — the app template
        gives it ``services: {}`` — so after the projection that map must hold
        exactly the client-facing facts of the services this tier's consumers
        dial, each equal to the hosting render's, and nothing more: not the
        host's own service knobs (image, heap, retention), and no literal
        top-level ``"services.graphdb.port_host"`` string key beside the map,
        which would satisfy ``config.yml`` as YAML and be read by nobody.
        """
        project = built_persona_stack / "build" / f"{built_persona_stack.name}-{persona}"
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        host = yaml.safe_load(
            (built_persona_stack / "build" / "config.yml").read_text(encoding="utf-8")
        )

        assert config["services"] == {
            "qmd": {"port": host["services"]["qmd"]["port"]},
            "graphdb": {"port_host": host["services"]["graphdb"]["port_host"]},
            "postgresql": {
                "port_host": host["services"]["postgresql"]["port_host"],
                "username": host["services"]["postgresql"]["username"],
                "database_name": host["services"]["postgresql"]["database_name"],
            },
            "openobserve": {"port": host["services"]["openobserve"]["port"]},
            "bluesky": {"port": host["services"]["bluesky"]["port"]},
            "virtual_accelerator": {"port": host["services"]["virtual_accelerator"]["port"]},
        }
        assert config["services"]["qmd"]["port"] == QMD_DEFAULT_PORT
        assert [key for key in config if "." in str(key)] == []

    def test_readwrite_is_told_its_panel_urls(self, built_persona_stack: Path) -> None:
        """The EVENTS and BLUESKY tabs the readwrite tier declares are told
        exactly what the hosting render carries for them — where the dispatch
        and bluesky-web injectors derived them — not what any preset spells.
        Every projected leaf matches, absent ones included — the whole entry
        the dispatch and bluesky-web injectors derive (url, path, label, and
        the health endpoint where the service serves one) and nothing more."""
        project = built_persona_stack / "build" / f"{built_persona_stack.name}-readwrite"
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        host = yaml.safe_load(
            (built_persona_stack / "build" / "config.yml").read_text(encoding="utf-8")
        )
        for panel in ("events", "bluesky"):
            told = config["web"]["panels"][panel]
            hosts = host["web"]["panels"][panel]
            assert told["url"] == hosts["url"], panel
            for leaf in ("path", "label", "health_endpoint"):
                assert told.get(leaf) == hosts.get(leaf), (panel, leaf)
        readonly = yaml.safe_load(
            (
                built_persona_stack
                / "build"
                / f"{built_persona_stack.name}-readonly"
                / "config.yml"
            ).read_text(encoding="utf-8")
        )
        assert "events" not in readonly["web"]["panels"]
        assert "bluesky" not in readonly["web"]["panels"]

    @pytest.mark.parametrize("persona", ("readonly", "readwrite", "ariel"))
    def test_attached_personas_resolve_the_qmd_sidecar(
        self, built_persona_stack: Path, persona: str
    ) -> None:
        """Every persona's rendered config names the sidecar its hybrid search dials.

        Asserted on a real build and through the client's own resolver, because
        this is the seam the query-time failure lives on: the app template
        enables ``ariel.search_modules.hybrid`` for the persona, and the module
        resolves its endpoint from ``services.qmd`` of the *same* config. A
        persona that renders one without the other is a terminal whose logbook
        search fails on every query, and nothing before this test looked at
        both halves together.
        """
        project = built_persona_stack / "build" / f"{built_persona_stack.name}-{persona}"
        assert project.is_dir(), f"{persona} was never rendered"
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))

        hybrid = config["ariel"]["search_modules"]["hybrid"]
        assert hybrid["enabled"] is True, "the template no longer enables hybrid search"

        resolved = resolve_qmd_service_config(config)
        assert resolved is not None, f"{persona}: hybrid search is on but no services.qmd"
        assert resolved.port == QMD_DEFAULT_PORT
        assert resolved.base_url == f"http://127.0.0.1:{QMD_DEFAULT_PORT}"

    def test_ariel_persona_renders_no_graph_surface(self, built_persona_stack: Path) -> None:
        """The logbook tier is graph-less, and by veto rather than by omission.

        ``control-assistant-ariel`` switches off every control-surface tool
        server explicitly (``claude_code.servers.<name>.enabled: false``), the
        graph server among them. The line is load-bearing now: the build tells
        every attached render where the hosting deployment's services are, and
        a ``services.graphdb`` block is what makes the graph server render —
        so only a server switched off is told nothing about the store
        (``osprey.deployment.reach``, the graphdb contract's gate).
        """
        project = built_persona_stack / "build" / f"{built_persona_stack.name}-ariel"
        assert project.is_dir(), "the ariel persona was never rendered"

        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        assert (config.get("services") or {}).get("graphdb") is None
        preset = resolve_preset("control-assistant-ariel").config
        assert preset.get("claude_code.servers.graph.enabled") is False

        hits = sorted(
            str(path.relative_to(project))
            for path in (project / ".claude").rglob("*")
            if path.is_file()
            and "mcp__graph__" in path.read_text(encoding="utf-8", errors="ignore")
        )
        assert hits == [], f"the logbook tier configures no graph store but rendered {hits}"
        assert "graph" not in json.loads((project / ".mcp.json").read_text())["mcpServers"]


# ---------------------------------------------------------------------------
# Web-terminal context overlay (seeding's base.md requirement)
# ---------------------------------------------------------------------------


class TestWebTerminalContextShipped:
    """Every built project — not just the ``control_assistant`` bundle —
    carries the ``docker/web-terminal-context/base.md`` that seeding
    requires. The framework ships a generic FALLBACK from its template root:
    ``modules.web_terminals.enabled`` is a config key any profile can turn
    on, so any bundle may end up seeding a user, and without a baseline
    ``osprey up`` brings up the whole stack and then aborts at the seed step.
    A profile's own ``web-terminal-context/base.md`` overrides the fallback.

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
