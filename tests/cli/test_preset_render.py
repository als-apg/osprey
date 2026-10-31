"""Tests for the ``control-assistant`` preset's simulator default.

The preset's ``control_system.type`` defaults to ``virtual_accelerator``: a
fresh project starts on the sandbox simulator, the one machine in the
deployment where a write is harmless. The preset also declares
``virtual_accelerator.live_standin``, so a second copy of the simulator ships
and is deployed as the deployment's own third control target — the
hardware-shaped soft IOC an operator switches to on purpose to rehearse under
approval prompts and strict-limit refusals. Serving the simulator in process
is the documented fallback for environments with no containers to depend on —
it speaks no Channel Access, so plans are browse-only — reachable via
``osprey set config.control_system.connector.virtual_accelerator.serving=in_process``.

The module also pins the preset's TIER FLOOR: the privileges the base preset
takes away from every tier built on it (the ``setup_patch`` deployment-editing
tool, the web Config panel, and writes to the scaffold gallery), which a single
tier lifts back rather than each tier having to remember to restrict itself.

Covers six angles:

1. Profile resolution without a build: the base preset and the two tiers laid
   over it (``tiers/readonly``, ``tiers/readwrite``) all carry the
   ``virtual_accelerator`` baseline.
2. A full ``osprey build`` of the bare preset: the rendered ``config.yml``
   carries ``control_system.type: virtual_accelerator``.
3. The same rendered build's ``deployed_services`` list is unchanged by the
   flip — it is driven by the ``bluesky:``/``virtual_accelerator:``/
   ``bluesky_web:`` injector blocks, not by ``control_system.type``.
4. The tier floor, in-process on the base and on every tier and app that
   stays at it, and on the real render — where
   :func:`~osprey.cli.profile_conventions.is_setup_patch_capable` of the
   rendered ``config.yml`` is now ``False``. That real-render assertion is the
   one deferred when the predicate itself was introduced: until the preset
   carried the deny there was no shipped render to read it from.
5. ``seeded_logins`` on the init'd repo: the preset's ``env.defaults`` seed
   carol's demo password alongside alice's and bob's, so the admin tier's login
   is announced on the landing page the same way the other two are.
6. The ADMIN tier — ``tiers/admin``, the one layer that lifts the
   floor back: in-process (what it overrides, and that it is the only tier
   :func:`~osprey.cli.build_cmd._profile_setup_patch_capable` calls capable),
   on the base preset's roster and catalog, and on the real render, where the
   admin ``.claude/settings.json`` ships the setup tool in ``ask`` rather than
   ``deny`` with the approval hook still matching it — and where both rendered
   documents, the composed ``settings.json`` and the ``config.yml`` that
   carries the deny and the ``remove_deny`` together, read as capable.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import _profile_setup_patch_capable, build
from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.cli.build_profile_presets import _load_preset_raw, list_tiers
from osprey.cli.init_cmd import init
from osprey.cli.profile_conventions import SETUP_PATCH_TOOL, is_setup_patch_capable
from osprey.deployment.web_terminals.auth_credentials import seeded_logins
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.utils.workspace import APPS_DIRNAME, TIERS_DIRNAME
from tests.cli._lifecycle_build import rendered_config

CONTROL_SYSTEM_TYPE_KEY = "control_system.type"
DENY_KEY = "claude_code.permissions.deny"
REMOVE_DENY_KEY = "claude_code.permissions.remove_deny"
CONFIG_PANEL_KEY = "web.config_panel.enabled"
SCAFFOLD_WRITE_KEY = "web.scaffold_gallery.write_enabled"
UI_MODE_KEY = "web.ui_mode"
WEB_TERMINALS_KEY = "modules.web_terminals"
SEEDED_PASSWORD_CODE = "web_terminals.auth_seeded_password"

PRESET = "control-assistant"
ADMIN_LAYER = f"{TIERS_DIRNAME}/admin"

#: The deployment's own profile and every layer ``osprey init`` writes beside it
#: that stays AT the floor. The floor is asserted on all of them together
#: because a tier can only ADD config keys and an app keeps the deployment's
#: permissions — the point of putting the restrictions in the base is that no
#: layer can drop one, and that only holds while every layer is checked.
#:
#: ``tiers/admin`` is deliberately absent: it is the one tier that lifts the
#: floor, and what it lifts is pinned by :class:`TestControlAssistantAdminPreset`
#: instead. It still inherits the deny entry itself — a tier cannot subtract
#: from a list — which is asserted there beside the ``remove_deny`` that
#: neutralises it.
FLOOR_PRESETS = [
    PRESET,
    f"{TIERS_DIRNAME}/readonly",
    f"{TIERS_DIRNAME}/readwrite",
    f"{TIERS_DIRNAME}/logbook",
    f"{TIERS_DIRNAME}/knowledge",
    f"{TIERS_DIRNAME}/channel-finder",
]


@pytest.fixture(scope="module")
def initialized_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``osprey init`` of the bare ``control-assistant`` preset, not yet built.

    The repo holds ``profile.yml`` and the ``tiers/`` files init materializes
    beside it — the layers every render below is resolved from.
    """
    repo = tmp_path_factory.mktemp("preset-va-default") / "preset-va-default"
    created = CliRunner().invoke(
        init, [str(repo), "--preset", PRESET, "--no-git"], catch_exceptions=False
    )
    assert created.exit_code == 0, created.output
    return repo


@pytest.fixture(scope="module")
def resolve_preset(initialized_repo: Path):
    """Resolve the deployment's profile or one layer of it, by name.

    ``control-assistant`` names the repo's ``profile.yml``; ``tiers/<n>`` names
    the login file beside it, merged or composed over the profile exactly as
    the build does.
    """

    def resolve(name: str) -> BuildProfile:
        path = initialized_repo / ("profile" if name == PRESET else name)
        profile, _profile_dir = resolve_build_profile(path.with_suffix(".yml"), None)
        return profile

    return resolve


class TestControlAssistantPresetDefault:
    """The base preset and its extends children all default to the simulator."""

    def test_base_preset_defaults_to_the_simulator(self, resolve_preset) -> None:
        base = resolve_preset(PRESET)
        assert base.config.get(CONTROL_SYSTEM_TYPE_KEY) == "virtual_accelerator"

    @pytest.mark.parametrize(
        "preset_name", [f"{TIERS_DIRNAME}/readonly", f"{TIERS_DIRNAME}/readwrite"]
    )
    def test_extends_children_inherit_the_simulator_default(
        self, preset_name: str, resolve_preset
    ) -> None:
        """Neither tier overrides ``control_system.type`` — both inherit the
        deployment's baseline from the profile they are laid over."""
        profile = resolve_preset(preset_name)
        assert profile.config.get(CONTROL_SYSTEM_TYPE_KEY) == "virtual_accelerator"


@pytest.fixture(scope="module")
def rendered_preset_repo(initialized_repo: Path) -> Path:
    """``osprey build`` of the init'd bare ``control-assistant`` repo (no
    overrides), returning the REPO — the directory holding ``profile.yml`` and
    the ``.env`` init seeded from ``env.defaults``.

    Module-scoped: the build is the slow part and every test below only reads
    files out of the result, so they all share this one render.
    """
    result = CliRunner().invoke(
        build,
        ["--repo", str(initialized_repo), "--skip-deps", "--skip-lifecycle"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return initialized_repo


@pytest.fixture(scope="module")
def rendered_preset_project(rendered_preset_repo: Path) -> Path:
    """The rendered project inside the repo above — where ``config.yml`` lands."""
    project_dir = rendered_preset_repo / "build"
    assert (project_dir / "config.yml").exists()
    return project_dir


@pytest.fixture(scope="module")
def rendered_preset_config(rendered_preset_project: Path) -> dict:
    return yaml.safe_load((rendered_preset_project / "config.yml").read_text(encoding="utf-8"))


class TestControlAssistantRenderedConfig:
    """A real ``osprey build`` of the bare preset carries the flip through to
    the rendered ``config.yml`` — not just the in-process profile object."""

    def test_rendered_control_system_type_is_the_simulator(
        self, rendered_preset_config: dict
    ) -> None:
        assert rendered_preset_config["control_system"]["type"] == "virtual_accelerator"

    def test_rendered_deployed_services_unchanged_by_the_flip(
        self, rendered_preset_config: dict
    ) -> None:
        """``deployed_services`` is driven by the preset's injector blocks
        (``bluesky:``, ``virtual_accelerator:``, ``bluesky_web:``), all of
        which are unconditional on ``control_system.type`` — the connector-type
        flip must not drop or gate any of them. Mirrors
        ``TestControlAssistantTurnkeyPlanServices`` in test_build_profile.py,
        which asserted the same membership before this preset moved its
        baseline. ``live_standin`` joined the list when the preset shipped
        ``virtual_accelerator.live_standin: 5074`` active: the second simulator
        copy is deployed by default, so a fresh build stands its ``standin``
        target up rather than only naming it."""
        deployed = rendered_preset_config["deployed_services"]
        assert "bluesky" in deployed
        assert "virtual_accelerator" in deployed
        assert "bluesky_web" in deployed
        assert "live_standin" in deployed


class TestControlAssistantTierFloor:
    """The base preset takes three privileges away from every tier built on it.

    Resolved in-process (no build): what is asserted here is the merged
    ``config:`` mapping each preset hands the renderer, which is what the
    render is generated from.
    """

    @pytest.mark.parametrize("preset_name", FLOOR_PRESETS)
    def test_control_assistant_tier_floor_denies_setup_patch(
        self, preset_name: str, resolve_preset
    ) -> None:
        """The deployment-editing tool is denied at the base, so every render
        inherits the deny. The admin tier lifts it back with ``remove_deny``
        rather than the base leaving it on for everyone."""
        profile = resolve_preset(preset_name)
        assert SETUP_PATCH_TOOL in profile.config.get(DENY_KEY, [])

    @pytest.mark.parametrize("preset_name", FLOOR_PRESETS)
    def test_control_assistant_tier_floor_disables_config_panel(
        self, preset_name: str, resolve_preset
    ) -> None:
        profile = resolve_preset(preset_name)
        assert profile.config.get(CONFIG_PANEL_KEY) is False

    @pytest.mark.parametrize("preset_name", FLOOR_PRESETS)
    def test_control_assistant_tier_floor_disables_scaffold_writes(
        self, preset_name: str, resolve_preset
    ) -> None:
        """Reading the gallery stays on at every tier; only writing to it is
        floored, because the gallery's contents are shared deployment state."""
        profile = resolve_preset(preset_name)
        assert profile.config.get(SCAFFOLD_WRITE_KEY) is False

    def test_control_assistant_floor_uses_deny_not_remove_ask(self, resolve_preset) -> None:
        """The floor must never be spelled as a base-level ``remove_ask``.

        String lists UNION when a tier is laid over the profile — a tier adds
        to an inherited list and can never subtract from it — so a ``remove_ask``
        written in the base would reach every tier including admin and strip the approval
        prompt that keeps the tool supervised there. ``deny`` is the spelling a
        floor needs: a tier lifts it with ``remove_deny``."""
        base = resolve_preset(PRESET)
        assert "remove_ask" not in base.config.get("claude_code.permissions", {})
        assert base.config.get("claude_code.permissions.remove_ask") is None


class TestControlAssistantRenderedTierFloor:
    """The floor survives a real ``osprey build`` of the bare preset — the
    single-user render every deployment starts from."""

    def test_control_assistant_render_is_not_setup_patch_capable(
        self, rendered_preset_config: dict
    ) -> None:
        """The posture read every later gate asks, answered against a SHIPPED
        render rather than a hand-built dict. This assertion was deferred when
        ``is_setup_patch_capable`` was introduced: no preset denied the tool
        then, so there was no real render for it to report ``False`` on."""
        assert is_setup_patch_capable(rendered_preset_config) is False

    def test_control_assistant_render_denies_setup_patch(
        self, rendered_preset_config: dict
    ) -> None:
        deny = rendered_preset_config["claude_code"]["permissions"]["deny"]
        assert SETUP_PATCH_TOOL in deny

    def test_control_assistant_render_disables_config_panel(
        self, rendered_preset_config: dict
    ) -> None:
        """The template renders this key live with a ``true`` default; the
        preset override is applied OVER that render, so ``false`` is what a
        deployment actually ships with."""
        assert rendered_preset_config["web"]["config_panel"]["enabled"] is False

    def test_control_assistant_render_disables_scaffold_writes(
        self, rendered_preset_config: dict
    ) -> None:
        assert rendered_preset_config["web"]["scaffold_gallery"]["write_enabled"] is False


class TestControlAssistantSeededLogins:
    """``env.defaults`` seeds a demo password per roster login, and the landing
    page announces exactly the ones still at their shipped value."""

    def test_control_assistant_seeded_logins_include_carol(
        self, rendered_preset_repo: Path
    ) -> None:
        """Carol is the admin tier's login. Her password has to be seeded here,
        beside alice's and bob's, or she is the one user whose credential the
        deployment never tells anyone — the exact case ``seeded_logins`` exists
        to cover."""
        logins = seeded_logins(rendered_preset_repo, ["alice", "bob", "carol"])
        assert ("carol", "carol") in logins
        assert logins == [("alice", "alice"), ("bob", "bob"), ("carol", "carol")]

    def test_the_shipped_demo_starts_on_loopback(self, rendered_preset_repo: Path) -> None:
        """On this machine the demo logins are the demo, and nothing refuses them."""
        config = rendered_config(rendered_preset_repo)

        findings = lint_web_terminals(config, project_root=rendered_preset_repo)

        assert [f for f in findings if f.code == SEEDED_PASSWORD_CODE] == []

    def test_the_shipped_demo_is_refused_off_loopback(self, rendered_preset_repo: Path) -> None:
        """Pointed at a real host, every demo login with a published password is
        named, and the shared cards, which publish none, are not."""
        config = rendered_config(rendered_preset_repo)
        config = copy.deepcopy(config)
        config["deploy"]["fqdn"] = "ops.example.org"

        findings = lint_web_terminals(config, project_root=rendered_preset_repo)

        seeded = [f for f in findings if f.code == SEEDED_PASSWORD_CODE]
        assert [f.severity for f in seeded] == ["error"]
        message = seeded[0].message
        for name in ("'alice'", "'bob'", "'carol'"):
            assert name in message
        for card in ("logbook", "knowledge"):
            assert card not in message


class TestControlAssistantAdminPreset:
    """``tiers/admin`` — the one tier that lifts the base's floor.

    Resolved in-process. Every other tier inherits the three restrictions and
    keeps them; this one names all three and turns them back on, which is the
    whole of what makes it the admin tier.
    """

    def test_admin_preset_extends_the_base(self) -> None:
        """A tier is laid over the profile beside it because of where it lives,
        never because it says so: the bundled admin tier names no base and
        subtracts nothing, so the file ``osprey init`` writes from it is a pure
        addition over whichever ``profile.yml`` it sits next to."""
        raw, _path = _load_preset_raw(ADMIN_LAYER)
        assert "extends" not in raw
        assert "exclude" not in raw

    def test_admin_preset_lifts_the_setup_patch_deny(self, resolve_preset) -> None:
        """Both halves, in one place: the inherited deny is still there — a
        tier can add to a list and never subtract from it — and the
        ``remove_deny`` beside it is what takes the entry back out at render
        time. Reading either half alone tells the wrong story about this tier.
        """
        profile = resolve_preset(ADMIN_LAYER)
        assert SETUP_PATCH_TOOL in profile.config.get(DENY_KEY, [])
        assert SETUP_PATCH_TOOL in profile.config.get(REMOVE_DENY_KEY, [])

    def test_admin_is_the_only_setup_patch_capable_tier(self, resolve_preset) -> None:
        """The posture the container's render context is built from.

        ``_profile_setup_patch_capable`` composes deny minus ``remove_deny``,
        which is why the admin tier reads capable while carrying the same deny
        entry as the layers that are not. It decides whether the image chowns
        ``build/config.yml`` to the agent's user, so a layer landing on the
        wrong side of it is a filesystem privilege, not a cosmetic one.
        """
        assert _profile_setup_patch_capable(resolve_preset(ADMIN_LAYER)) is True
        for preset_name in FLOOR_PRESETS:
            assert _profile_setup_patch_capable(resolve_preset(preset_name)) is False, preset_name

    def test_admin_preset_enables_the_config_panel(self, resolve_preset) -> None:
        assert resolve_preset(ADMIN_LAYER).config.get(CONFIG_PANEL_KEY) is True

    def test_admin_preset_enables_scaffold_gallery_writes(self, resolve_preset) -> None:
        assert resolve_preset(ADMIN_LAYER).config.get(SCAFFOLD_WRITE_KEY) is True

    def test_admin_preset_adds_the_setup_mode_skill(self, resolve_preset) -> None:
        """The skill the base leaves out on purpose. Skill lists UNION when a
        tier is laid over the profile, so the admin tier keeps the base
        selection and adds this one rather than replacing it."""
        base = resolve_preset(PRESET)
        admin = resolve_preset(ADMIN_LAYER)
        assert "setup-mode" in admin.skills
        assert "setup-mode" not in base.skills
        assert set(base.skills) <= set(admin.skills)

    def test_admin_preset_is_an_attached_render(self, resolve_preset) -> None:
        """Attached like every tier: no services of its own, and no second web
        tier on the host's ports — the hosting deployment owns both."""
        admin = resolve_preset(ADMIN_LAYER)
        assert admin.deploy_services is False
        assert admin.config.get(f"{WEB_TERMINALS_KEY}.enabled") is False
        assert admin.config.get(UI_MODE_KEY) == "expert"

    def test_base_catalog_deploys_the_admin_render(self) -> None:
        """The admin tier exists because a bundled file ships it: ``osprey
        init`` writes one ``tiers/<name>.yml`` per bundled tier, and the build
        renders one project per file. The base preset keeps no render catalog,
        and the render's name and directory come from the deployment's
        project_name at build time, never from the shipped file."""
        assert "admin" in list_tiers()
        wt = resolve_build_profile(None, preset=PRESET)[0].config[WEB_TERMINALS_KEY]
        assert "renders" not in wt
        raw, _path = _load_preset_raw(ADMIN_LAYER)
        assert "project" not in raw
        assert "project_path" not in raw

    def test_admin_roster_user_carol_is_authenticated(self) -> None:
        """Carol is a person, not a shared app. The apps in ``shared:`` carry
        no password of their own; an admin card doing the same would hand the
        Config panel and the setup tool to every roster login, so the absence
        of an ``access`` key here is the assertion."""
        wt = resolve_build_profile(None, preset=PRESET)[0].config[WEB_TERMINALS_KEY]
        carol = next(u for u in wt["users"] if u["name"] == "carol")
        assert carol["tier"] == "admin"
        assert carol["index"] == 2
        assert "access" not in carol
        assert carol["display_name"]

    def test_admin_tier_is_not_the_default_tier(self) -> None:
        """A roster entry with no ``tier`` must not land on this tier."""
        wt = resolve_build_profile(None, preset=PRESET)[0].config[WEB_TERMINALS_KEY]
        assert wt["default_tier"] == "readonly"


@pytest.fixture(scope="module")
def rendered_admin_project(rendered_preset_repo: Path) -> Path:
    """The admin render's rendered project inside the shared build.

    No second build: one ``osprey build`` of the init'd repo renders the
    deployment's own project AND one per tier and app file, so the admin
    render is already sitting beside the base one that
    :func:`rendered_preset_config` reads.
    """
    project = rendered_preset_repo / "build" / f"{rendered_preset_repo.name}-admin"
    assert project.is_dir(), sorted(p.name for p in (rendered_preset_repo / "build").iterdir())
    return project


@pytest.fixture(scope="module")
def rendered_admin_settings(rendered_admin_project: Path) -> dict:
    return json.loads((rendered_admin_project / ".claude" / "settings.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def rendered_admin_config(rendered_admin_project: Path) -> dict:
    """The admin render's rendered ``config.yml`` — the deny AND the lift.

    The document the later render-roster guards read, and the counterpart to
    :func:`rendered_preset_config` one tier up.
    """
    return yaml.safe_load((rendered_admin_project / "config.yml").read_text("utf-8"))


class TestControlAssistantRenderedAdmin:
    """The admin tier through a real ``osprey init`` + ``osprey build``."""

    def test_init_emits_a_layer_delta_for_every_tier_including_admin(
        self, rendered_preset_repo: Path
    ) -> None:
        """Every bundled login, admin and the shared ones included. ``tiers/`` is
        what ``osprey build`` renders a project per, so a login with no file
        here exists on the landing page and in nothing else."""
        tiers = sorted(p.stem for p in (rendered_preset_repo / TIERS_DIRNAME).glob("*.yml"))
        assert tiers == ["admin", "channel-finder", "knowledge", "logbook", "readonly", "readwrite"]
        assert not (rendered_preset_repo / APPS_DIRNAME).exists()

    def test_admin_render_leaves_setup_patch_out_of_the_deny_list(
        self, rendered_admin_settings: dict
    ) -> None:
        """The lift, observed where it is ENFORCED. ``settings.json`` is the
        document the agent runs against, and the deny array in it is the
        composed one — the rendered ``config.yml`` keeps the inherited deny and
        the lift side by side. (:func:`is_setup_patch_capable` composes that
        pair itself, so it reads either document the same; what it cannot do is
        be the thing that stops a call, which is what is pinned here.)"""
        assert SETUP_PATCH_TOOL not in rendered_admin_settings["permissions"]["deny"]

    def test_admin_render_keeps_setup_patch_approval_gated(
        self, rendered_admin_settings: dict
    ) -> None:
        """Lifting the deny does not make the tool unsupervised: it lands in
        ``ask``, which is where the workspace server declares it, and the
        approval hook still matches it. The tier difference is whether the path
        exists, not whether it is watched."""
        assert SETUP_PATCH_TOOL in rendered_admin_settings["permissions"]["ask"]
        matchers = [
            entry.get("matcher") for entry in rendered_admin_settings["hooks"]["PreToolUse"]
        ]
        assert SETUP_PATCH_TOOL in matchers, matchers

    def test_admin_render_is_setup_patch_capable(
        self,
        rendered_admin_settings: dict,
        rendered_admin_config: dict,
        rendered_preset_config: dict,
    ) -> None:
        """The posture read every later gate asks, answered against a SHIPPED
        render — the assertion deferred when the predicate was introduced,
        because until this preset existed no render answered ``True``.

        Asked of BOTH rendered documents, which is the point. The
        ``settings.json`` (deny at the top level) is the composed output. The
        ``config.yml`` is not: a render cannot subtract from an inherited
        list, so it carries the base's deny and the admin ``remove_deny``
        together — and :func:`is_setup_patch_capable` composes them the way the
        settings render does, so the two documents agree. They have to: the
        roster guards are handed render ``config.yml`` documents, and a
        predicate that called the capable tier denied would let exactly the
        tier those guards exist to catch pass them. The base render is the
        control — it denies with nothing lifting it, and answers ``False``.
        """
        assert is_setup_patch_capable(rendered_admin_settings) is True
        assert is_setup_patch_capable(rendered_admin_config) is True
        assert is_setup_patch_capable(rendered_preset_config) is False

    def test_admin_rendered_config_carries_the_deny_and_the_lift_together(
        self, rendered_admin_config: dict
    ) -> None:
        """Why the predicate must compose, shown in the shipped document: the
        inherited deny is still there, and only the ``remove_deny`` beside it
        says the tier is capable."""
        permissions = rendered_admin_config["claude_code"]["permissions"]
        assert SETUP_PATCH_TOOL in permissions["deny"]
        assert SETUP_PATCH_TOOL in permissions["remove_deny"]

    def test_admin_render_enables_the_config_panel_and_gallery_writes(
        self, rendered_admin_project: Path
    ) -> None:
        """The two browser-side privileges, in the file the running deployment
        reads. Both are rendered live with permissive defaults, so what is
        pinned here is that the preset's override survived the render."""
        config = yaml.safe_load((rendered_admin_project / "config.yml").read_text("utf-8"))
        assert config["web"]["config_panel"]["enabled"] is True
        assert config["web"]["scaffold_gallery"]["write_enabled"] is True
        assert config["web"]["ui_mode"] == "expert"
