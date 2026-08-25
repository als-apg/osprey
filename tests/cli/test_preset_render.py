"""Tests for the ``control-assistant`` preset's Virtual Accelerator default.

The preset's ``control_system.type`` defaults to ``virtual_accelerator`` (the
VA soft-IOC ships and is deployed unconditionally as part of the turn-key
Bluesky stack, so a fresh tutorial project's plans drive it end to end
out of the box). ``mock`` is the documented fallback for environments with no
containers to depend on — its non-tracking readbacks make plans browse-only —
and is reachable via ``osprey set connector=mock``.

The module also pins the preset's TIER FLOOR: the privileges the base preset
takes away from every tier built on it (the ``setup_patch`` deployment-editing
tool, the web Config panel, and writes to the scaffold gallery), which a single
tier lifts back rather than each tier having to remember to restrict itself.

Covers six angles:

1. In-process profile resolution: the base preset and its two extends
   children (``control-assistant-readonly``, ``control-assistant-readwrite``)
   all carry the ``virtual_accelerator`` override (fast, no build).
2. A full ``osprey build`` of the bare preset: the rendered ``config.yml``
   carries ``control_system.type: virtual_accelerator``.
3. The same rendered build's ``deployed_services`` list is unchanged by the
   flip — it is driven by the ``bluesky:``/``virtual_accelerator:``/
   ``bluesky_web:`` injector blocks, not by ``control_system.type``.
4. The tier floor, in-process on the base and on the three persona deltas that
   stay at it (``readonly``, ``readwrite``, ``ariel``), and on the real render — where
   :func:`~osprey.cli.profile_conventions.is_setup_patch_capable` of the
   rendered ``config.yml`` is now ``False``. That real-render assertion is the
   one deferred when the predicate itself was introduced: until the preset
   carried the deny there was no shipped render to read it from.
5. ``seeded_logins`` on the init'd repo: the preset's ``env.defaults`` seed
   carol's demo password alongside alice's and bob's, so the admin tier's login
   is announced on the landing page the same way the other two are.
6. The ADMIN tier — ``control-assistant-admin``, the one preset that lifts the
   floor back: in-process (what it overrides, and that it is the only tier
   :func:`~osprey.cli.build_cmd._profile_setup_patch_capable` calls capable),
   on the base preset's roster and catalog, and on the real render, where the
   admin ``.claude/settings.json`` ships the setup tool in ``ask`` rather than
   ``deny`` with the approval hook still matching it — and where both rendered
   documents, the composed ``settings.json`` and the ``config.yml`` that
   carries the deny and the ``remove_deny`` together, read as capable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import _profile_setup_patch_capable, build
from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.cli.init_cmd import init
from osprey.cli.profile_conventions import SETUP_PATCH_TOOL, is_setup_patch_capable
from osprey.deployment.web_terminals.auth_credentials import seeded_logins

CONTROL_SYSTEM_TYPE_KEY = "control_system.type"
DENY_KEY = "claude_code.permissions.deny"
REMOVE_DENY_KEY = "claude_code.permissions.remove_deny"
CONFIG_PANEL_KEY = "web.config_panel.enabled"
SCAFFOLD_WRITE_KEY = "web.scaffold_gallery.write_enabled"
UI_MODE_KEY = "web.ui_mode"
WEB_TERMINALS_KEY = "modules.web_terminals"

ADMIN_PRESET = "control-assistant-admin"

#: Every preset built from the base that stays AT the floor, base included. The
#: floor is asserted on all of them together because a persona delta can only
#: ADD config keys — the point of putting the restrictions in the base is that
#: no delta can drop one, and that only holds while every delta is checked.
#:
#: ``control-assistant-admin`` is deliberately absent: it is the one tier that
#: lifts the floor, and what it lifts is pinned by
#: :class:`TestControlAssistantAdminPreset` instead. It still inherits the deny
#: entry itself — a delta cannot subtract from a list — which is asserted there
#: beside the ``remove_deny`` that neutralises it.
FLOOR_PRESETS = [
    "control-assistant",
    "control-assistant-readonly",
    "control-assistant-readwrite",
    "control-assistant-ariel",
]


def resolve_preset(name: str) -> BuildProfile:
    """Resolve a bundled preset by name, fully merging its ``extends`` chain."""
    profile, _profile_dir = resolve_build_profile(None, preset=name)
    return profile


class TestControlAssistantPresetDefault:
    """The base preset and its extends children all default to the VA."""

    def test_base_preset_defaults_to_virtual_accelerator(self) -> None:
        base = resolve_preset("control-assistant")
        assert base.config.get(CONTROL_SYSTEM_TYPE_KEY) == "virtual_accelerator"

    @pytest.mark.parametrize(
        "preset_name", ["control-assistant-readonly", "control-assistant-readwrite"]
    )
    def test_extends_children_inherit_virtual_accelerator_default(self, preset_name: str) -> None:
        """Neither persona overrides ``control_system.type`` — both inherit the
        base preset's new default through their ``extends:`` chain."""
        profile = resolve_preset(preset_name)
        assert profile.config.get(CONTROL_SYSTEM_TYPE_KEY) == "virtual_accelerator"


@pytest.fixture(scope="module")
def rendered_preset_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``osprey init`` + ``osprey build`` the bare ``control-assistant`` preset
    (no overrides) into a tmp repo, and return the REPO — the directory holding
    ``profile.yml`` and the ``.env`` init seeded from ``env.defaults``.

    Module-scoped: the build is the slow part and every test below only reads
    files out of the result, so they all share this one render.
    """
    tmp_path = tmp_path_factory.mktemp("preset-va-default")
    runner = CliRunner()
    repo = tmp_path / "preset-va-default"
    created = runner.invoke(
        init, [str(repo), "--preset", "control-assistant", "--no-git"], catch_exceptions=False
    )
    assert created.exit_code == 0, created.output
    result = runner.invoke(
        build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return repo


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

    def test_rendered_control_system_type_is_virtual_accelerator(
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
        which asserted the same membership before this preset defaulted to the
        VA."""
        deployed = rendered_preset_config["deployed_services"]
        assert "bluesky" in deployed
        assert "virtual_accelerator" in deployed
        assert "bluesky_web" in deployed


class TestControlAssistantTierFloor:
    """The base preset takes three privileges away from every tier built on it.

    Resolved in-process (no build): what is asserted here is the merged
    ``config:`` mapping each preset hands the renderer, which is what the
    render is generated from.
    """

    @pytest.mark.parametrize("preset_name", FLOOR_PRESETS)
    def test_control_assistant_tier_floor_denies_setup_patch(self, preset_name: str) -> None:
        """The deployment-editing tool is denied at the base, so every persona
        inherits the deny. The admin tier lifts it back with ``remove_deny``
        rather than the base leaving it on for everyone."""
        profile = resolve_preset(preset_name)
        assert SETUP_PATCH_TOOL in profile.config.get(DENY_KEY, [])

    @pytest.mark.parametrize("preset_name", FLOOR_PRESETS)
    def test_control_assistant_tier_floor_disables_config_panel(self, preset_name: str) -> None:
        profile = resolve_preset(preset_name)
        assert profile.config.get(CONFIG_PANEL_KEY) is False

    @pytest.mark.parametrize("preset_name", FLOOR_PRESETS)
    def test_control_assistant_tier_floor_disables_scaffold_writes(self, preset_name: str) -> None:
        """Reading the gallery stays on at every tier; only writing to it is
        floored, because the gallery's contents are shared deployment state."""
        profile = resolve_preset(preset_name)
        assert profile.config.get(SCAFFOLD_WRITE_KEY) is False

    def test_control_assistant_floor_uses_deny_not_remove_ask(self) -> None:
        """The floor must never be spelled as a base-level ``remove_ask``.

        String lists UNION across ``extends`` — a child adds to an inherited
        list and can never subtract from it — so a ``remove_ask`` written in the
        base would reach every tier including admin and strip the approval
        prompt that keeps the tool supervised there. ``deny`` is the spelling a
        floor needs: a tier lifts it with ``remove_deny``."""
        base = resolve_preset("control-assistant")
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


class TestControlAssistantAdminPreset:
    """``control-assistant-admin`` — the one tier that lifts the base's floor.

    Resolved in-process. Every other tier inherits the three restrictions and
    keeps them; this one names all three and turns them back on, which is the
    whole of what makes it the admin tier.
    """

    def test_admin_preset_extends_the_base(self) -> None:
        """A persona preset is emitted into ``personas/`` as a pure delta over
        the profile beside it, which only holds while its ``extends`` names the
        host preset DIRECTLY (``_off_chain_problem``). An admin preset reaching
        the base through an intermediate would emit a file with that
        intermediate's layer silently missing."""
        from osprey.cli.build_profile_presets import _load_preset_raw

        raw, _path = _load_preset_raw(ADMIN_PRESET)
        assert raw.get("extends") == "control-assistant"

    def test_admin_preset_lifts_the_setup_patch_deny(self) -> None:
        """Both halves, in one place: the inherited deny is still there — a
        delta can add to a list and never subtract from it — and the
        ``remove_deny`` beside it is what takes the entry back out at render
        time. Reading either half alone tells the wrong story about this tier.
        """
        profile = resolve_preset(ADMIN_PRESET)
        assert SETUP_PATCH_TOOL in profile.config.get(DENY_KEY, [])
        assert SETUP_PATCH_TOOL in profile.config.get(REMOVE_DENY_KEY, [])

    def test_admin_is_the_only_setup_patch_capable_tier(self) -> None:
        """The posture the container's render context is built from.

        ``_profile_setup_patch_capable`` composes deny minus ``remove_deny``,
        which is why the admin tier reads capable while carrying the same deny
        entry as the three tiers that are not. It decides whether the image
        chowns ``build/config.yml`` to the agent's user, so a tier landing on
        the wrong side of it is a filesystem privilege, not a cosmetic one.
        """
        assert _profile_setup_patch_capable(resolve_preset(ADMIN_PRESET)) is True
        for preset_name in FLOOR_PRESETS:
            assert _profile_setup_patch_capable(resolve_preset(preset_name)) is False, preset_name

    def test_admin_preset_enables_the_config_panel(self) -> None:
        assert resolve_preset(ADMIN_PRESET).config.get(CONFIG_PANEL_KEY) is True

    def test_admin_preset_enables_scaffold_gallery_writes(self) -> None:
        assert resolve_preset(ADMIN_PRESET).config.get(SCAFFOLD_WRITE_KEY) is True

    def test_admin_preset_adds_the_setup_mode_skill(self) -> None:
        """The skill the base leaves out on purpose. Skill lists UNION across
        ``extends``, so the admin tier keeps the base selection and adds this
        one rather than replacing it."""
        base = resolve_preset("control-assistant")
        admin = resolve_preset(ADMIN_PRESET)
        assert "setup-mode" in admin.skills
        assert "setup-mode" not in base.skills
        assert set(base.skills) <= set(admin.skills)

    def test_admin_preset_is_an_attached_render(self) -> None:
        """Same persona-sub-preset shape as its three siblings: no services of
        its own, and no second web tier on the host's ports — the hosting
        deployment owns both."""
        admin = resolve_preset(ADMIN_PRESET)
        assert admin.deploy_services is False
        # Spelled as its own dotted key, exactly as the siblings do: the base's
        # `modules.web_terminals` is ONE dotted key holding the whole roster,
        # and a delta that respelled it as a nested block would replace that
        # subtree and drop the roster with it.
        assert admin.config.get(f"{WEB_TERMINALS_KEY}.enabled") is False
        assert admin.config.get(UI_MODE_KEY) == "expert"

    def test_base_catalog_deploys_the_admin_persona(self) -> None:
        """The catalog entry is what makes the tier exist as a deployed login:
        ``osprey init`` emits one ``personas/<name>.yml`` per entry here."""
        catalog = resolve_preset("control-assistant").config[WEB_TERMINALS_KEY]["personas"]
        entry = catalog["admin"]
        assert entry["build_profile"] == ADMIN_PRESET
        assert entry["project"] == "control-assistant-admin"
        assert entry["project_path"] == "build/control-assistant-admin"
        # Name invariant the roster relies on: project == basename(project_path).
        assert entry["project"] == Path(entry["project_path"]).name

    def test_admin_roster_user_carol_is_authenticated(self) -> None:
        """Carol is a person, not a public service. The ARIEL card ships
        ``login: false`` deliberately; an admin card doing the same would hand
        the Config panel and the setup tool to anyone who opens the landing
        page, so the absence of that key here is the assertion."""
        wt = resolve_preset("control-assistant").config[WEB_TERMINALS_KEY]
        carol = next(u for u in wt["users"] if u["name"] == "carol")
        assert carol["persona"] == "admin"
        assert carol["index"] == 3
        assert "login" not in carol
        assert carol["display_name"]

    def test_admin_tier_is_not_the_default_persona(self) -> None:
        """A roster entry with no ``persona`` must not land on this tier."""
        wt = resolve_preset("control-assistant").config[WEB_TERMINALS_KEY]
        assert wt["default_persona"] == "readonly"


@pytest.fixture(scope="module")
def rendered_admin_project(rendered_preset_repo: Path) -> Path:
    """The admin persona's rendered project inside the shared build.

    No second build: one ``osprey build`` of the init'd repo renders the
    deployment's own project AND one per delta in ``personas/``, so the admin
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
    """The admin persona's rendered ``config.yml`` — the deny AND the lift.

    The document the later persona-roster guards read, and the counterpart to
    :func:`rendered_preset_config` one tier up.
    """
    return yaml.safe_load((rendered_admin_project / "config.yml").read_text("utf-8"))


class TestControlAssistantRenderedAdmin:
    """The admin tier through a real ``osprey init`` + ``osprey build``."""

    def test_init_emits_a_persona_delta_for_every_tier_including_admin(
        self, rendered_preset_repo: Path
    ) -> None:
        """FOUR deltas, not three. ``personas/`` is what ``osprey build``
        renders a project per, so a catalog entry with no file here is a tier
        that exists on the landing page and in nothing else."""
        deltas = sorted(p.stem for p in (rendered_preset_repo / "personas").glob("*.yml"))
        assert deltas == ["admin", "ariel", "readonly", "readwrite"]

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
        ``config.yml`` is not: a persona cannot subtract from an inherited
        list, so it carries the base's deny and the admin ``remove_deny``
        together — and :func:`is_setup_patch_capable` composes them the way the
        settings render does, so the two documents agree. They have to: the
        roster guards are handed persona ``config.yml`` documents, and a
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
