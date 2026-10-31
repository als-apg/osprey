"""Lint of the roster's derived shared-app cards and the renamed tier/app codes.

``modules.web_terminals.shared`` derives one card per app at port slot
``INDEX_MAX - position`` (see ``renders.roster_entries``). These tests pin what
lint does with those derived slots — the index, duplicate, port-overlap and
family-allocatable checks see them — and the errors only a ``shared:`` list can
raise.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.deployment.web_terminals import lint
from osprey.deployment.web_terminals.lint import (
    Finding,
    lint_profile_config,
    lint_web_terminals,
)
from osprey.port_layout import INDEX_MAX, default_port
from tests._renders import tier_render

_BASE = 20000
_WEB_BASE = default_port("web", 0, base=_BASE)


def _renders(**tiers: dict) -> dict:
    """The two shared apps every roster here carries, plus *tiers*."""
    return {
        "logbook": tier_render("logbook", "demo-logbook"),
        "knowledge": tier_render("knowledge", "demo-knowledge"),
        **tiers,
    }


def _config(**web_terminals: object) -> dict:
    """A clean rendered-altitude config whose roster is alice, bob and two apps."""
    subtree: dict = {
        "enabled": True,
        "nginx_port": default_port("nginx", base=_BASE),
        "web_base_port": _WEB_BASE,
        "artifact_base_port": default_port("artifact", 0, base=_BASE),
        "ariel_base_port": default_port("ariel", 0, base=_BASE),
        "lattice_base_port": default_port("lattice", 0, base=_BASE),
        "users": [{"name": "alice", "index": 0}, {"name": "bob", "index": 1}],
        "shared": ["logbook", "knowledge"],
        "renders": _renders(assistant=tier_render("assistant", "test-assistant")),
        "default_tier": "assistant",
    }
    subtree.update(web_terminals)
    return {
        "facility": {"prefix": "test"},
        "registry": {"url": "registry.example.org/demo"},
        "modules": {"web_terminals": subtree},
    }


def _codes(findings: list[Finding], severity: str = "error") -> list[str]:
    return [f.code for f in findings if f.severity == severity]


def _only(findings: list[Finding], code: str) -> list[Finding]:
    return [f for f in findings if f.code == code]


# --- the clean baseline -------------------------------------------------------


def test_clean_roster_with_shared_apps_has_no_errors() -> None:
    """People at 0..1 and two backed apps at 99..98 collide with nothing."""
    # Arrange
    config = _config()

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _codes(findings) == []


def test_rendered_config_without_renders_projection_does_not_judge_names() -> None:
    """With no projection to compare against, an app name is not called unknown."""
    # Arrange
    config = _config()
    del config["modules"]["web_terminals"]["renders"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert "web_terminals.unknown_shared_app" not in _codes(findings)


# --- shared: entries ----------------------------------------------------------


def test_unknown_shared_name_is_an_error_naming_scaffold_tiers() -> None:
    """A name with no login behind it has nothing to render, and the remedy says how."""
    # Arrange
    config = _config(shared=["logbook", "knowledge", "channel-finder"])

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.unknown_shared_app")

    # Assert
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "'channel-finder'" in findings[0].message
    assert "'logbook'" not in findings[0].message
    assert "`osprey scaffold tiers`" in findings[0].message


def test_profile_altitude_judges_shared_names_against_the_login_files(
    tmp_path: Path,
) -> None:
    """At profile time the shared logins are the ``tiers/<name>.yml`` files beside the profile."""
    # Arrange
    (tmp_path / "tiers").mkdir()
    (tmp_path / "tiers" / "logbook.yml").write_text("name: Logbook Research\nlists: own\n")
    profile_config = {
        "modules.web_terminals": {
            "enabled": True,
            "users": [{"name": "alice", "index": 0}],
            "shared": ["logbook", "knowledge"],
        }
    }

    # Act
    findings = _only(
        lint_profile_config(profile_config, profile_root=tmp_path),
        "web_terminals.unknown_shared_app",
    )

    # Assert
    assert len(findings) == 1
    assert "'knowledge'" in findings[0].message
    assert "'logbook'" not in findings[0].message
    assert "`osprey scaffold tiers`" in findings[0].message


def test_profile_altitude_with_no_tiers_directory_reports_every_name(tmp_path: Path) -> None:
    """No ``tiers/`` at all backs no login, so every shared name is unknown."""
    # Arrange
    profile_config = {
        "modules.web_terminals": {
            "enabled": True,
            "users": ["alice"],
            "shared": ["logbook"],
        }
    }

    # Act
    findings = lint_profile_config(profile_config, profile_root=tmp_path)

    # Assert
    assert "web_terminals.unknown_shared_app" in _codes(findings)


def test_duplicate_shared_name_is_an_error() -> None:
    """One app named twice would run two cards under one service name."""
    # Arrange
    config = _config(shared=["logbook", "knowledge", "logbook"])

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.duplicate_shared_app")

    # Assert
    assert len(findings) == 1
    assert "'logbook'" in findings[0].message
    assert "'knowledge'" not in findings[0].message


def test_non_string_shared_item_is_an_error() -> None:
    """An item that is not a name derives no card; lint is where that is said."""
    # Arrange
    config = _config(shared=["logbook", 3, ""])

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.invalid_shared_entry")

    # Assert
    assert [f.severity for f in findings] == ["error", "error"]
    assert "shared[1]" in findings[0].message
    assert "shared[2]" in findings[1].message


def test_shared_that_is_not_a_list_is_an_error() -> None:
    # Arrange
    config = _config(shared="logbook")

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert "web_terminals.invalid_shared_entry" in _codes(findings)


def test_person_named_like_an_app_is_an_error() -> None:
    """A person and an app card would share one service name, URL and secret."""
    # Arrange
    config = _config(users=[{"name": "alice", "index": 0}, {"name": "logbook", "index": 1}])

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.user_named_like_app")

    # Assert
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "'logbook'" in findings[0].message


# --- the app band at the top of the index range -------------------------------


def test_person_index_inside_the_app_band_is_an_error() -> None:
    """With two apps the slots 98..99 are theirs; a person at 98 sits in them."""
    # Arrange
    floor = INDEX_MAX - 2 + 1
    config = _config(users=[{"name": "alice", "index": 0}, {"name": "bob", "index": floor}])

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.user_index_in_app_band")

    # Assert
    assert len(findings) == 1
    assert "'bob'" in findings[0].message
    assert str(floor) in findings[0].message


def test_person_index_just_below_the_app_band_is_fine() -> None:
    # Arrange
    below = INDEX_MAX - 2
    config = _config(users=[{"name": "alice", "index": 0}, {"name": "bob", "index": below}])

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert _codes(findings) == []


def test_app_band_counts_a_non_string_item_that_keeps_its_slot() -> None:
    """``len(shared)`` is the band: an invalid item derives nothing but holds its slot."""
    # Arrange: three items, so the band is 97..99 even though only two derive.
    config = _config(
        shared=["logbook", 3, "knowledge"],
        users=[{"name": "alice", "index": INDEX_MAX - 2}],
    )

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert "web_terminals.user_index_in_app_band" in _codes(findings)


def test_person_on_an_app_slot_is_a_duplicate_index() -> None:
    """The duplicate-index check sees the derived slots, naming person and app."""
    # Arrange
    config = _config(users=[{"name": "alice", "index": INDEX_MAX}])

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.duplicate_index")

    # Assert
    assert len(findings) == 1
    assert "'alice'" in findings[0].message
    assert "'logbook'" in findings[0].message
    assert str(INDEX_MAX) in findings[0].message


def test_no_band_without_shared_apps() -> None:
    """A roster with no ``shared:`` reserves nothing at the top of the band."""
    # Arrange
    config = _config(users=[{"name": "alice", "index": INDEX_MAX}])
    del config["modules"]["web_terminals"]["shared"]

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert "web_terminals.user_index_in_app_band" not in _codes(findings)
    assert "web_terminals.duplicate_index" not in _codes(findings)


def test_service_port_on_an_app_slot_is_a_port_overlap() -> None:
    """The overlap set holds each app's slot, not only the people's."""
    # Arrange
    config = _config()
    app_port = _WEB_BASE + INDEX_MAX
    config["services"] = {"postgresql": {"port_host": app_port}}

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.port_overlap")

    # Assert
    assert len(findings) == 1
    assert str(app_port) in findings[0].message
    assert f"[index={INDEX_MAX}]" in findings[0].message
    assert "services.postgresql.port_host" in findings[0].message


def test_service_port_on_an_unused_slot_is_not_an_overlap() -> None:
    """Between the people and the apps nothing is bound."""
    # Arrange
    config = _config()
    config["services"] = {"postgresql": {"port_host": _WEB_BASE + 50}}

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert "web_terminals.port_overlap" not in _codes(findings)


def test_app_slots_are_allocatable_at_the_top_of_the_band(monkeypatch) -> None:
    """The family check allocates the highest slot, which is the top app's ``INDEX_MAX``."""
    # Arrange
    asked: list[int] = []
    real = lint.allocate_ports

    def recording(base_ports, *, index):
        asked.append(index)
        return real(base_ports, index=index)

    monkeypatch.setattr(lint, "allocate_ports", recording)
    config = _config(users=[])

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert asked == [INDEX_MAX]
    assert "web_terminals.incomplete_port_families" not in _codes(findings)


def test_derived_entries_are_never_people() -> None:
    """Person-only checks keep seeing people: an app-only roster is still empty of people."""
    # Arrange
    config = _config(users=[])

    # Act
    findings = lint_web_terminals(config)

    # Assert
    assert "web_terminals.empty_users" in _codes(findings, "warn")
    assert "web_terminals.duplicate_user" not in _codes(findings)


# --- the renamed codes --------------------------------------------------------

_RENAMED = {
    "conflicting_user_tier_and_role",
    "invalid_render_charset",
    "render_build_profile_not_a_delta",
    "render_bundle_path_divergence",
    "render_extra_mounts_not_list",
    "tier_inherits_armed_connector",
    "render_invalid_extra_mount",
    "render_invalid_seed_base",
    "render_mirror_path_divergence",
    "render_missing_build_profile",
    "render_missing_config_yml",
    "render_missing_dockerfile",
    "render_missing_project_path",
    "render_missing_project",
    "render_privileges_unknown",
    "render_project_collision",
    "render_project_mismatch",
    "render_project_path_name_mismatch",
    "render_project_path_not_dir",
    "render_project_path_not_rendered_yet",
    "render_shadows_project_image",
    "privileged_default_tier",
    "unknown_default_tier",
    "unknown_tier_reference",
}


def _emitted_codes() -> set[str]:
    source = Path(lint.__file__).read_text(encoding="utf-8")
    return set(re.findall(r'"web_terminals\.([a-z0-9_]+)"', source))


def test_every_renamed_lint_code_is_emitted() -> None:
    """The 24 renamed codes are what lint emits."""
    # Arrange
    emitted = _emitted_codes()

    # Assert
    assert len(_RENAMED) == 24
    assert _RENAMED <= emitted


def test_catalog_entry_codes_use_the_new_names() -> None:
    """A sample of catalog findings fires under the renamed codes."""
    # Arrange
    config = _config(
        default_tier="missing",
        renders=_renders(
            **{
                "Bad Name": tier_render("Bad Name", "x"),
                "seedy": tier_render("seedy", "seedy", seed_base="false", extra_mounts="nope"),
            }
        ),
        users=[{"name": "alice", "index": 0, "tier": "ghost"}],
    )

    # Act
    codes = set(_codes(lint_web_terminals(config)))

    # Assert
    assert {
        "web_terminals.unknown_default_tier",
        "web_terminals.unknown_tier_reference",
        "web_terminals.invalid_render_charset",
        "web_terminals.render_invalid_seed_base",
        "web_terminals.render_extra_mounts_not_list",
    } <= codes


def test_tiers_sharing_one_project_collide_even_on_one_project_path() -> None:
    """One render serving two names is no longer exempt: each tier and app is its own."""
    # Arrange
    config = _config(
        image_source="local",
        users=[
            {"name": "alice", "index": 0, "tier": "readonly"},
            {"name": "bob", "index": 1, "tier": "readwrite"},
        ],
        default_tier="readonly",
        renders=_renders(
            readonly=tier_render("readonly", "demo-ops", project_path="build/demo-ops"),
            readwrite=tier_render("readwrite", "demo-ops", project_path="build/demo-ops"),
        ),
    )

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.render_project_collision")

    # Assert
    assert len(findings) == 1
    assert "'readonly'" in findings[0].message
    assert "'readwrite'" in findings[0].message


def test_render_on_the_deployment_project_names_the_project_image() -> None:
    """The shadow rule keeps its logic and names the image it guards: the project's."""
    # Arrange
    config = _config(
        image_source="local",
        users=[{"name": "alice", "index": 0, "tier": "readonly"}],
        default_tier="readonly",
        renders=_renders(readonly=tier_render("readonly", "demo", project_path="build/demo")),
    )
    config["project_name"] = "demo"

    # Act
    findings = _only(lint_web_terminals(config), "web_terminals.render_shadows_project_image")

    # Assert
    assert len(findings) == 1
    assert "the project image" in findings[0].message
    assert "dispatch worker" not in findings[0].message
