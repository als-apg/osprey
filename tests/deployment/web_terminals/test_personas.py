"""Tests for roster normalization and persona identity resolution
(osprey.deployment.web_terminals.personas)."""

from __future__ import annotations

from typing import Any

import pytest

from osprey.deployment.web_terminals.personas import (
    EVENTS_PANEL_ID,
    config_archiver_password_env,
    config_declares_panel,
    config_needs_ariel_password,
    config_needs_dispatcher_token,
    config_needs_launch_token,
    entry_requires_login,
    env_var_suffix,
    env_var_suffix_collisions,
    freeze_user_indices,
    normalize_users,
    personas_needing_archiver_password,
    personas_needing_ariel_password,
    personas_needing_dispatcher_token,
    personas_needing_launch_token,
    personas_not_denying_bash,
    resolve_personas,
    settings_json_denies_bash,
)


def test_normalize_users_bare_strings_indexed_by_position() -> None:
    """A legacy bare-string roster gets its raw list position as its index."""
    # Arrange
    users_raw = ["alice", "bob"]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [
        {"name": "alice", "index": 0},
        {"name": "bob", "index": 1},
    ]


def test_normalize_users_explicit_entries_pass_through_unchanged() -> None:
    """Already-explicit object entries keep their own index, not their position."""
    # Arrange
    users_raw = [{"name": "alice", "index": 5}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 5}]


def test_normalize_users_mixed_bare_and_explicit_entries() -> None:
    """Bare entries keep raw position; object entries keep their explicit index."""
    # Arrange
    users_raw = ["alice", {"name": "bob", "index": 7}, "carol"]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [
        {"name": "alice", "index": 0},
        {"name": "bob", "index": 7},
        {"name": "carol", "index": 2},
    ]


def test_normalize_users_is_idempotent() -> None:
    """Normalizing an already-normalized list must be a no-op."""
    # Arrange
    users_raw = ["alice", {"name": "bob", "index": 7}, "carol"]

    # Act
    once = normalize_users(users_raw)
    twice = normalize_users(once)

    # Assert
    assert once == twice


def test_normalize_users_drops_malformed_entries() -> None:
    """Non-string entries and dicts missing a str name or int index are dropped."""
    # Arrange
    users_raw = [
        "alice",
        123,
        {"name": "bob"},  # missing index
        {"index": 2},  # missing name
        {"name": 4, "index": 1},  # name not a str
        {"name": "carol", "index": "1"},  # index not an int
        {"name": "dave", "index": True},  # bool is not a valid index (config typo)
        None,
        [],
    ]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0}]


def test_normalize_users_carries_string_display_name_through() -> None:
    """An object entry's string `display_name` is carried onto the normalized entry."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "display_name": "Operations"}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0, "display_name": "Operations"}]


def test_normalize_users_omits_display_name_key_when_absent() -> None:
    """An entry with no `display_name` keeps the plain two-key shape (no None key)."""
    # Act / Assert — bare string and object-without-display_name both stay two-key
    assert normalize_users(["alice"]) == [{"name": "alice", "index": 0}]
    assert normalize_users([{"name": "bob", "index": 1}]) == [{"name": "bob", "index": 1}]


def test_normalize_users_drops_non_string_display_name() -> None:
    """A non-string `display_name` (a config typo) is dropped defensively; the rest
    of a well-formed entry still normalizes."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "display_name": 123}]

    # Act
    result = normalize_users(users_raw)

    # Assert — entry survives (name/index valid), display_name omitted
    assert result == [{"name": "alice", "index": 0}]


def test_normalize_users_carries_string_theme_through() -> None:
    """An object entry's string `theme` is carried onto the normalized entry."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "theme": "desy-light"}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0, "theme": "desy-light"}]


def test_normalize_users_omits_theme_key_when_absent() -> None:
    """An entry with no `theme` keeps the plain two-key shape (no None key)."""
    # Act / Assert
    assert normalize_users(["alice"]) == [{"name": "alice", "index": 0}]
    assert normalize_users([{"name": "bob", "index": 1}]) == [{"name": "bob", "index": 1}]


def test_normalize_users_drops_non_string_theme() -> None:
    """A non-string `theme` (a config typo) is dropped defensively; the rest of a
    well-formed entry still normalizes."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "theme": ["desy"]}]

    # Act
    result = normalize_users(users_raw)

    # Assert — entry survives (name/index valid), theme omitted
    assert result == [{"name": "alice", "index": 0}]


def test_normalize_users_carries_display_name_and_theme_together() -> None:
    """The two optional per-user fields are independent — declaring both keeps
    both."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "display_name": "Operations", "theme": "desy"}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0, "display_name": "Operations", "theme": "desy"}]


def test_normalize_users_empty_list_returns_empty_list() -> None:
    """An empty users list normalizes to an empty list."""
    # Act / Assert
    assert normalize_users([]) == []


def test_normalize_users_non_list_input_returns_empty_list() -> None:
    """Anything that isn't a list (including None) normalizes to an empty list."""
    # Act / Assert
    assert normalize_users(None) == []
    assert normalize_users({"name": "alice", "index": 0}) == []
    assert normalize_users("alice") == []


def test_normalize_users_does_not_mutate_input_entries() -> None:
    """Normalization must return new dicts, never the original entry by reference."""
    # Arrange
    original_entry = {"name": "alice", "index": 5}
    users_raw = [original_entry]

    # Act
    result = normalize_users(users_raw)
    result[0]["index"] = 99

    # Assert
    assert original_entry == {"name": "alice", "index": 5}


# ---------------------------------------------------------------------------
# freeze_user_indices()
# ---------------------------------------------------------------------------


def test_freeze_user_indices_keeps_the_persona_normalize_users_drops() -> None:
    """The whole point: the roster that goes BACK to config.yml keeps `persona`.

    normalize_users projects it away, and a roster written from that projection
    re-resolves every entry onto `default_persona`.
    """
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "persona": "readonly"}]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0, "persona": "readonly"}]
    assert normalize_users(users_raw) == [{"name": "alice", "index": 0}]


def test_freeze_user_indices_freezes_bare_string_positions() -> None:
    """A bare string carries nothing but its name, so it gains only an index —
    the same one normalize_users assigns it."""
    # Arrange
    users_raw = ["alice", "bob"]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0}, {"name": "bob", "index": 1}]


def test_freeze_user_indices_carries_unknown_keys_through() -> None:
    """Keys this module does not read are still the facility's config."""
    # Arrange
    users_raw = [{"name": "alice", "index": 2, "shift": "swing"}]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 2, "shift": "swing"}]


def test_freeze_user_indices_survival_matches_normalize_users() -> None:
    """Who survives is normalize_users' contract, not a second copy of its rules."""
    # Arrange
    users_raw = [
        "alice",
        {"name": "bob", "index": 1, "persona": "readonly"},
        {"name": "carol", "index": True},  # bool index: a config typo, dropped
        {"index": 3},  # no name
        42,
    ]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert
    assert [entry["name"] for entry in result] == [
        entry["name"] for entry in normalize_users(users_raw)
    ]
    assert [entry["name"] for entry in result] == ["alice", "bob"]


def test_freeze_user_indices_index_comes_from_normalize_users() -> None:
    """A malformed `index` on an otherwise-valid entry cannot leak back into the
    file: the written index is always the normalized one."""
    # Arrange
    users_raw = [{"name": "alice", "index": 7, "persona": "readonly"}]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert
    assert result[0]["index"] == 7 == normalize_users(users_raw)[0]["index"]


def test_freeze_user_indices_is_idempotent() -> None:
    """Its own output re-freezes to itself, so a roster can be frozen twice."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "persona": "readonly"}, "bob"]

    # Act
    once = freeze_user_indices(users_raw)
    twice = freeze_user_indices(once)

    # Assert
    assert twice == once


def test_freeze_user_indices_duplicate_name_gives_both_the_last_entrys_extras() -> None:
    """A duplicate-name roster resolves extras by NAME, so both entries take the
    LAST authored entry's — including its persona.

    This is a guard, not an endorsement: the roster is already invalid (lint
    reports the duplicate) and the behaviour hands one listing another's
    persona. Pinned because it is the one path where matching by name is
    lossy, so a future change to the matching rule has to face it deliberately
    rather than discover it in a facility's config.
    """
    # Arrange
    users_raw = [
        {"name": "alice", "index": 0, "persona": "readonly"},
        {"name": "alice", "index": 4, "persona": "readwrite"},
    ]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert — each keeps its OWN frozen index...
    assert [entry["index"] for entry in result] == [0, 4]
    # ...but both carry the last authored entry's persona.
    assert [entry["persona"] for entry in result] == ["readwrite", "readwrite"]


def test_freeze_user_indices_carries_values_normalize_users_drops_as_malformed() -> None:
    """A malformed optional value goes back to the file as the author wrote it.

    normalize_users drops a non-string ``display_name`` because the render
    cannot use it. Removing one user must not quietly repair — or delete —
    another user's config line: the value round-trips and lint keeps reporting
    it.
    """
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "display_name": 123}]

    # Act
    result = freeze_user_indices(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0, "display_name": 123}]
    assert normalize_users(users_raw) == [{"name": "alice", "index": 0}]


def test_freeze_user_indices_does_not_mutate_input_entries() -> None:
    """Returns new dicts, never the authored entry by reference."""
    # Arrange
    original_entry = {"name": "alice", "index": 5, "persona": "readonly"}
    users_raw = [original_entry]

    # Act
    result = freeze_user_indices(users_raw)
    result[0]["persona"] = "readwrite"

    # Assert
    assert original_entry == {"name": "alice", "index": 5, "persona": "readonly"}


# ---------------------------------------------------------------------------
# resolve_personas()
# ---------------------------------------------------------------------------

_REGISTRY = {"url": "registry.example.org/osprey"}


def test_resolve_personas_no_catalog_resolves_to_todays_values() -> None:
    """Zero migration: no `personas` catalog at all resolves every entry to the
    exact pre-persona image/project-dir this module rendered before personas
    existed."""
    # Arrange
    web_terminals = {"users": ["alice", "bob"]}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "persona": None,
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
        },
        {
            "name": "bob",
            "index": 1,
            "persona": None,
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
        },
    ]


def test_resolve_personas_no_catalog_empty_registry_url_matches_template_concat() -> None:
    """An unset registry.url must reproduce the exact (leading-slash) string the
    compose template built by direct concatenation before this function existed."""
    # Arrange
    web_terminals = {"users": ["alice"]}

    # Act
    result = resolve_personas(web_terminals, {}, "als")

    # Assert
    assert result[0]["image"] == "/web-terminal:latest"


def test_resolve_personas_default_persona_keeps_unsuffixed_registry_image() -> None:
    """The default persona's registry-mode image stays un-suffixed even once a
    catalog is introduced; its container dir follows its own catalog project
    uniformly, like every other persona (here coinciding with the facility
    prefix path because the fixture's project is `als-assistant`)."""
    # Arrange
    web_terminals = {
        "users": ["alice"],
        "default_persona": "cli",
        "personas": {"cli": {"project": "als-assistant", "project_path": "profiles/cli"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "persona": "cli",
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
        }
    ]


def test_resolve_personas_default_persona_container_dir_follows_its_project() -> None:
    """The default persona's container dir is derived from its own catalog
    project, not the facility prefix — proven with a project name that does not
    coincide with the pre-persona `/app/<prefix>-assistant` path. The image stays
    un-suffixed, which is the only remaining default-persona special case."""
    # Arrange
    web_terminals = {
        "users": ["alice"],
        "default_persona": "cli",
        "personas": {"cli": {"project": "control-room", "project_path": "profiles/cli"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["project"] == "control-room"
    assert result[0]["container_project_dir"] == "/app/control-room"
    assert result[0]["image"] == "registry.example.org/osprey/web-terminal:latest"


def test_resolve_personas_non_default_persona_registry_mode_suffixes_image() -> None:
    """A non-default persona gets a `web-terminal-<persona>` registry tag and a
    container dir derived from its own catalog project, not the facility prefix."""
    # Arrange
    web_terminals = {
        "users": [{"name": "gmartino", "index": 0, "persona": "gui"}],
        "default_persona": "cli",
        "personas": {
            "cli": {"project": "als-assistant", "project_path": "profiles/cli"},
            "gui": {"project": "als-gui-assistant", "project_path": "profiles/gui"},
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result == [
        {
            "name": "gmartino",
            "index": 0,
            "persona": "gui",
            "image": "registry.example.org/osprey/web-terminal-gui:latest",
            "project": "als-gui-assistant",
            "container_project_dir": "/app/als-gui-assistant",
            "extra_mounts": [],
            "seed_base": True,
        }
    ]


def test_resolve_personas_local_mode_suffixes_every_persona_including_default() -> None:
    """Local mode builds `<persona.project>-<persona>:local` for every persona —
    unlike registry mode, the default persona is not special-cased on the image."""
    # Arrange
    web_terminals = {
        "users": ["alice", {"name": "gmartino", "index": 1, "persona": "gui"}],
        "default_persona": "cli",
        "image_source": "local",
        "personas": {
            "cli": {"project": "als-assistant", "project_path": "profiles/cli"},
            "gui": {"project": "als-gui-assistant", "project_path": "profiles/gui"},
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    images = {entry["name"]: entry["image"] for entry in result}
    assert images == {
        "alice": "als-assistant-cli:local",
        "gmartino": "als-gui-assistant-gui:local",
    }
    # Default persona's container dir follows its own catalog project, like every
    # other persona — no facility-prefix special case (here `als-assistant`).
    default_entry = next(entry for entry in result if entry["name"] == "alice")
    assert default_entry["container_project_dir"] == "/app/als-assistant"


def test_resolve_personas_entry_without_persona_key_inherits_default_persona() -> None:
    """A roster entry with no `persona:` key inherits `default_persona`, resolving
    through the catalog rather than falling onto the no-persona legacy path."""
    # Arrange
    web_terminals = {
        "users": ["alice"],
        "default_persona": "cli",
        "personas": {"cli": {"project": "als-assistant", "project_path": "profiles/cli"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["persona"] == "cli"


def test_resolve_personas_local_mode_without_catalog_strict_raises() -> None:
    """`image_source: local` with no catalog/default_persona configured at all
    must fail closed in strict mode (deploy/build/render/seed callers)."""
    # Arrange
    web_terminals = {"users": ["alice"], "image_source": "local"}

    # Act / Assert
    with pytest.raises(ValueError, match="local"):
        resolve_personas(web_terminals, _REGISTRY, "als", strict=True)


def test_resolve_personas_local_mode_without_catalog_lenient_degrades() -> None:
    """The lenient variant (lifecycle verbs) must never raise on the same
    misconfiguration — a bad/missing persona setup can't block decommission,
    prune, or nuke — and instead falls back to the zero-migration values."""
    # Arrange
    web_terminals = {"users": ["alice"], "image_source": "local"}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als", strict=False)

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "persona": None,
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
        }
    ]


def test_resolve_personas_unknown_persona_ref_strict_raises() -> None:
    """An explicit `persona:` referencing a name absent from the catalog raises
    in strict mode."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 0, "persona": "ghost"}],
        "personas": {"cli": {"project": "als-assistant", "project_path": "profiles/cli"}},
    }

    # Act / Assert
    with pytest.raises(ValueError, match="ghost"):
        resolve_personas(web_terminals, _REGISTRY, "als", strict=True)


def test_resolve_personas_unknown_persona_ref_lenient_degrades() -> None:
    """The lenient variant degrades an unknown persona ref to the zero-migration
    values instead of raising, but keeps the requested (bad) name visible."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 0, "persona": "ghost"}],
        "personas": {"cli": {"project": "als-assistant", "project_path": "profiles/cli"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als", strict=False)

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "persona": "ghost",
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
        }
    ]


def test_resolve_personas_preserves_normalize_users_index_freezing() -> None:
    """Explicit indices from an already-frozen roster carry through unchanged —
    resolve_personas must not recompute positions itself."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 5, "persona": "cli"}],
        "personas": {"cli": {"project": "als-assistant", "project_path": "profiles/cli"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["index"] == 5


def test_resolve_personas_exposes_display_name_when_set() -> None:
    """A roster entry's `display_name` is threaded onto the resolved svc dict as a
    `display_name` key (here on the no-persona zero-migration path)."""
    # Arrange
    web_terminals = {"users": [{"name": "alice", "index": 0, "display_name": "Operations"}]}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "persona": None,
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
            "display_name": "Operations",
        }
    ]


def test_resolve_personas_display_name_threads_through_persona_branch() -> None:
    """`display_name` is orthogonal to persona resolution — it rides through a
    fully-resolved non-default persona entry too, not only the zero-migration path."""
    # Arrange
    web_terminals = {
        "users": [
            {"name": "gmartino", "index": 0, "persona": "gui", "display_name": "Control GUI"}
        ],
        "default_persona": "cli",
        "personas": {
            "cli": {"project": "als-assistant", "project_path": "profiles/cli"},
            "gui": {"project": "als-gui-assistant", "project_path": "profiles/gui"},
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["persona"] == "gui"
    assert result[0]["image"] == "registry.example.org/osprey/web-terminal-gui:latest"
    assert result[0]["display_name"] == "Control GUI"


def test_resolve_personas_omits_display_name_key_when_unset_or_empty() -> None:
    """No `display_name` (or an empty-string one) omits the key entirely, so the
    resolved entry stays byte-identical to a pre-`display_name` resolution."""
    # Arrange
    web_terminals = {
        "users": [
            "alice",  # bare string — never carries one
            {"name": "bob", "index": 1},  # object form, no display_name
            {"name": "carol", "index": 2, "display_name": ""},  # empty is inert
        ]
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert — no entry carries a display_name key
    assert all("display_name" not in entry for entry in result)


def test_resolve_personas_exposes_theme_when_set() -> None:
    """A roster entry's `theme` is threaded onto the resolved svc dict as a
    `theme` key (here on the no-persona zero-migration path)."""
    # Arrange
    web_terminals = {"users": [{"name": "alice", "index": 0, "theme": "desy-light"}]}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "persona": None,
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
            "extra_mounts": [],
            "seed_base": True,
            "theme": "desy-light",
        }
    ]


def test_resolve_personas_theme_threads_through_persona_branch() -> None:
    """`theme` is orthogonal to persona resolution — it rides through a
    fully-resolved non-default persona entry too, not only the zero-migration path."""
    # Arrange
    web_terminals = {
        "users": [{"name": "gmartino", "index": 0, "persona": "gui", "theme": "desy"}],
        "default_persona": "cli",
        "personas": {
            "cli": {"project": "als-assistant", "project_path": "profiles/cli"},
            "gui": {"project": "als-gui-assistant", "project_path": "profiles/gui"},
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["persona"] == "gui"
    assert result[0]["theme"] == "desy"


def test_resolve_personas_omits_theme_key_when_unset_or_empty() -> None:
    """No `theme` (or an empty-string one) omits the key entirely, so the resolved
    entry stays byte-identical to a pre-`theme` resolution."""
    # Arrange
    web_terminals = {
        "users": [
            "alice",  # bare string — never carries one
            {"name": "bob", "index": 1},  # object form, no theme
            {"name": "carol", "index": 2, "theme": ""},  # empty is inert
        ]
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert — no entry carries a theme key
    assert all("theme" not in entry for entry in result)


def test_resolve_personas_empty_users_returns_empty_list() -> None:
    """An empty/missing roster resolves to an empty list regardless of catalog."""
    # Act / Assert
    assert resolve_personas({}, _REGISTRY, "als") == []
    assert resolve_personas({"users": []}, _REGISTRY, "als") == []


def test_resolve_personas_registry_cfg_missing_url_defaults_to_empty_string() -> None:
    """A registry section with no `url` key must not raise — it degrades to the
    same empty-prefix behavior as an entirely absent registry section."""
    # Act
    result = resolve_personas({"users": ["alice"]}, {}, "als")

    # Assert
    assert result[0]["image"] == "/web-terminal:latest"


# ---------------------------------------------------------------------------
# resolve_personas() — persona extra_mounts
# ---------------------------------------------------------------------------


def test_resolve_personas_reads_extra_mounts_from_catalog_entry() -> None:
    """A persona's `extra_mounts` list is carried onto every user of that persona."""
    # Arrange
    web_terminals = {
        "users": [{"name": "gmartino", "index": 0, "persona": "gui"}],
        "personas": {
            "gui": {
                "project": "als-gui",
                "extra_mounts": ["/opt/site-data:/app/site-data:ro", "cache-vol:/app/cache"],
            }
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["extra_mounts"] == [
        "/opt/site-data:/app/site-data:ro",
        "cache-vol:/app/cache",
    ]


def test_resolve_personas_default_persona_also_carries_extra_mounts() -> None:
    """The default persona is not special-cased for `extra_mounts` — its list is
    carried through like any other persona's."""
    # Arrange
    web_terminals = {
        "users": ["alice"],
        "default_persona": "cli",
        "personas": {"cli": {"project": "als-assistant", "extra_mounts": ["/data:/app/data:ro"]}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["extra_mounts"] == ["/data:/app/data:ro"]


def test_resolve_personas_no_persona_defaults_extra_mounts_to_empty_list() -> None:
    """The zero-migration path (no persona in effect) resolves `extra_mounts` to
    an empty list — there is no catalog entry to read host mounts from."""
    # Arrange
    web_terminals = {"users": ["alice"]}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["extra_mounts"] == []


def test_resolve_personas_catalog_entry_without_extra_mounts_defaults_to_empty_list() -> None:
    """A persona that sets no `extra_mounts` resolves to an empty list, not a
    missing key."""
    # Arrange
    web_terminals = {
        "users": [{"name": "gmartino", "index": 0, "persona": "gui"}],
        "personas": {"gui": {"project": "als-gui"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["extra_mounts"] == []


def test_resolve_personas_extra_mounts_defensive_reads() -> None:
    """A non-list `extra_mounts` drops to `[]`; a list with non-string/empty
    entries keeps only the well-formed strings (colon-part syntax is lint's job)."""
    # Arrange
    web_terminals = {
        "users": [
            {"name": "a", "index": 0, "persona": "bad"},
            {"name": "b", "index": 1, "persona": "mixed"},
        ],
        "personas": {
            "bad": {"project": "als-bad", "extra_mounts": "not-a-list"},
            "mixed": {
                "project": "als-mixed",
                "extra_mounts": ["/ok:/app/ok:ro", 42, "", None, "/two:/app/two"],
            },
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    by_name = {entry["name"]: entry["extra_mounts"] for entry in result}
    assert by_name["a"] == []
    assert by_name["b"] == ["/ok:/app/ok:ro", "/two:/app/two"]


def test_resolve_personas_lenient_degrade_extra_mounts_empty() -> None:
    """An unresolvable persona ref degrading to the zero-migration values carries
    an empty `extra_mounts` (no catalog entry to read from)."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 0, "persona": "ghost"}],
        "personas": {"cli": {"project": "als-assistant"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als", strict=False)

    # Assert
    assert result[0]["extra_mounts"] == []


# ---------------------------------------------------------------------------
# image_tag seam
# ---------------------------------------------------------------------------


def test_resolve_personas_image_tag_defaults_to_latest() -> None:
    """No `image_tag` key resolves to the pre-seam `:latest` registry tag."""
    # Arrange
    web_terminals = {"users": ["alice"]}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["image"] == "registry.example.org/osprey/web-terminal:latest"


def test_resolve_personas_image_tag_explicit_literal_is_emitted_verbatim() -> None:
    """A plain literal `image_tag` (no env reference) is baked into the image
    ref exactly as written, for both default and non-default persona images."""
    # Arrange
    web_terminals = {
        "users": ["alice", {"name": "gmartino", "index": 1, "persona": "gui"}],
        "default_persona": "cli",
        "image_tag": "v2026.7.8",
        "personas": {
            "cli": {"project": "als-assistant", "project_path": "profiles/cli"},
            "gui": {"project": "als-gui-assistant", "project_path": "profiles/gui"},
        },
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    images = {entry["name"]: entry["image"] for entry in result}
    assert images == {
        "alice": "registry.example.org/osprey/web-terminal:v2026.7.8",
        "gmartino": "registry.example.org/osprey/web-terminal-gui:v2026.7.8",
    }


def test_resolve_personas_image_tag_expands_env_var_at_render_time(monkeypatch) -> None:
    """A `${VAR}` reference in `image_tag` expands against the process
    environment to a literal tag — no `${...}` survives into the image ref."""
    # Arrange
    monkeypatch.setenv("IMAGE_TAG", "v9.9.9")
    web_terminals = {"users": ["alice"], "image_tag": "${IMAGE_TAG}"}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["image"] == "registry.example.org/osprey/web-terminal:v9.9.9"


def test_resolve_personas_image_tag_unset_env_var_expands_to_empty(monkeypatch) -> None:
    """An `image_tag` referencing an unset variable expands to the empty string
    (not a surviving `${...}`), yielding a tagless ref that lint warns on."""
    # Arrange
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    web_terminals = {"users": ["alice"], "image_tag": "${IMAGE_TAG}"}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["image"] == "registry.example.org/osprey/web-terminal:"


# ---------------------------------------------------------------------------
# resolve_personas() — seed_base opt-out
# ---------------------------------------------------------------------------


def test_resolve_personas_seed_base_defaults_true_for_catalog_entry() -> None:
    """A catalog entry with no `seed_base` key resolves to the default, True."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 0, "persona": "gui"}],
        "personas": {"gui": {"project": "als-gui"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["seed_base"] is True


def test_resolve_personas_seed_base_false_is_carried_through() -> None:
    """`seed_base: false` on a catalog entry resolves to False for its users."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 0, "persona": "gui"}],
        "personas": {"gui": {"project": "als-gui", "seed_base": False}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["seed_base"] is False


def test_resolve_personas_seed_base_non_bool_coerces_to_true() -> None:
    """A non-bool `seed_base` (a config typo lint reports separately) must not
    propagate — it defensively coerces to the safe default, True."""
    # Arrange
    web_terminals = {
        "users": [{"name": "alice", "index": 0, "persona": "gui"}],
        "personas": {"gui": {"project": "als-gui", "seed_base": "false"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["seed_base"] is True


def test_resolve_personas_no_persona_entry_is_seed_base_true() -> None:
    """The zero-migration path (no persona in effect) always keeps the base
    prepend — seed_base is only opt-out-able through a catalog entry."""
    # Act
    result = resolve_personas({"users": ["alice"]}, _REGISTRY, "als")

    # Assert
    assert result[0]["seed_base"] is True


# ---------------------------------------------------------------------------
# Roster OIDC identity mapping (normalize_users / resolve_personas passthrough)
# ---------------------------------------------------------------------------


def test_normalize_users_carries_string_oidc_subject_through() -> None:
    """An object entry's `oidc_subject` — the non-secret IdP claim value that
    identifies this roster user — is carried onto the normalized entry."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "oidc_subject": "alice@example.org"}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0, "oidc_subject": "alice@example.org"}]


def test_normalize_users_omits_oidc_subject_key_when_absent() -> None:
    """A roster declaring no OIDC mapping keeps the plain two-key shape, so a
    password-mode (or pre-auth) config normalizes exactly as it did before."""
    # Act / Assert
    assert normalize_users(["alice"]) == [{"name": "alice", "index": 0}]
    assert normalize_users([{"name": "bob", "index": 1}]) == [{"name": "bob", "index": 1}]


def test_normalize_users_drops_non_string_oidc_subject() -> None:
    """A non-string `oidc_subject` is dropped defensively; the rest of a
    well-formed entry still normalizes."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "oidc_subject": ["alice@example.org"]}]

    # Act
    result = normalize_users(users_raw)

    # Assert — entry survives (name/index valid), oidc_subject omitted
    assert result == [{"name": "alice", "index": 0}]


def test_normalize_users_drops_empty_oidc_subject() -> None:
    """An empty `oidc_subject` must never become a mapping: an identity whose
    claim is missing or empty would otherwise match this roster user. Dropping
    it leaves the user unmapped, which the OIDC callback answers with 403."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "oidc_subject": ""}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0}]


def test_normalize_users_oidc_subject_is_independent_of_the_cosmetic_fields() -> None:
    """The identity mapping and the cosmetic per-user fields don't interfere —
    declaring all three keeps all three."""
    # Arrange
    users_raw = [
        {
            "name": "alice",
            "index": 0,
            "display_name": "Operations",
            "theme": "desy",
            "oidc_subject": "alice@example.org",
        }
    ]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [
        {
            "name": "alice",
            "index": 0,
            "display_name": "Operations",
            "theme": "desy",
            "oidc_subject": "alice@example.org",
        }
    ]


def test_normalize_users_carries_login_false_through() -> None:
    """`login: false` — the one value that changes anything — rides onto the
    normalized entry."""
    # Act
    result = normalize_users([{"name": "ariel", "index": 2, "login": False}])

    # Assert
    assert result == [{"name": "ariel", "index": 2, "login": False}]


def test_normalize_users_drops_every_other_login_spelling() -> None:
    """`true`, absence, and every malformed spelling all normalize to "login
    required" — a typo can lock an entry down, never open it up."""
    # Act / Assert — explicit true and non-boolean spellings alike leave the
    # plain two-key shape, so downstream reads them all as gated
    for spelling in (True, "false", "no", 0, None, ["false"]):
        result = normalize_users([{"name": "ariel", "index": 2, "login": spelling}])
        assert result == [{"name": "ariel", "index": 2}], repr(spelling)


def test_entry_requires_login_reads_only_the_literal_false() -> None:
    """The shared predicate: one reading of the key for render, provisioning,
    and the `users passwd` refusal."""
    assert entry_requires_login({"name": "alice", "index": 0}) is True
    assert entry_requires_login({"name": "ariel", "index": 2, "login": False}) is False


def test_resolve_personas_threads_login_false_through_both_branches() -> None:
    """The exemption survives resolution on the zero-migration path and the
    persona-catalog path alike, and stays absent when never declared."""
    # Act
    zero_migration = resolve_personas(
        {"users": [{"name": "ariel", "index": 0, "login": False}]}, _REGISTRY, "als"
    )
    persona_branch = resolve_personas(
        {
            "users": [{"name": "ariel", "index": 0, "persona": "gui", "login": False}],
            "personas": {"gui": {"project": "als-gui"}},
        },
        _REGISTRY,
        "als",
    )
    undeclared = resolve_personas({"users": ["alice"]}, _REGISTRY, "als")

    # Assert
    assert zero_migration[0]["login"] is False
    assert persona_branch[0]["login"] is False
    assert "login" not in undeclared[0]


def test_resolve_personas_exposes_oidc_subject_when_set() -> None:
    """The mapping rides through to the resolved entry, so the sidecar's roster
    context reads it off the same object as every other per-user field."""
    # Arrange
    web_terminals = {"users": [{"name": "alice", "index": 0, "oidc_subject": "alice@example.org"}]}

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["oidc_subject"] == "alice@example.org"


def test_resolve_personas_oidc_subject_threads_through_persona_branch() -> None:
    """The passthrough is not confined to the zero-migration path — a user
    resolved through a catalog persona keeps its mapping too."""
    # Arrange
    web_terminals = {
        "users": [
            {"name": "alice", "index": 0, "persona": "gui", "oidc_subject": "alice@example.org"}
        ],
        "personas": {"gui": {"project": "als-gui"}},
    }

    # Act
    result = resolve_personas(web_terminals, _REGISTRY, "als")

    # Assert
    assert result[0]["persona"] == "gui"
    assert result[0]["oidc_subject"] == "alice@example.org"


def test_resolve_personas_omits_oidc_subject_key_when_unset() -> None:
    """A roster with no OIDC mapping resolves byte-identically to before the
    field existed — no `oidc_subject: None` key appears."""
    # Act
    result = resolve_personas({"users": ["alice"]}, _REGISTRY, "als")

    # Assert
    assert "oidc_subject" not in result[0]


# ---------------------------------------------------------------------------
# env_var_suffix() / env_var_suffix_collisions()
# ---------------------------------------------------------------------------


def test_env_var_suffix_uppercases_and_maps_dashes_to_underscores() -> None:
    """The one definition of how a username keys its per-user env vars."""
    # Act / Assert
    assert env_var_suffix("alice") == "ALICE"
    assert env_var_suffix("alice-b") == "ALICE_B"
    assert env_var_suffix("Alice-B-C") == "ALICE_B_C"


def test_env_var_suffix_leaves_already_conforming_names_untouched() -> None:
    """Idempotent on its own output — an already-uppercase, underscored name is
    returned unchanged, so re-keying an existing entry can't drift."""
    # Arrange
    once = env_var_suffix("alice-b")

    # Act / Assert
    assert env_var_suffix(once) == once


def test_env_var_suffix_is_total_and_does_not_validate_charset() -> None:
    """Charset enforcement belongs to the preflight raise and lint, not here —
    this helper maps whatever it is given rather than raising."""
    # Act / Assert
    assert env_var_suffix("") == ""
    assert env_var_suffix("alice.b") == "ALICE.B"


def test_env_var_suffix_collisions_reports_names_sharing_one_suffix() -> None:
    """`alice-b` and `alice_b` both key OSPREY_AUTH_PW_HASH_ALICE_B — without
    this check one user's password would open the other's terminal."""
    # Act
    result = env_var_suffix_collisions(["alice-b", "alice_b", "carol"])

    # Assert
    assert result == {"ALICE_B": ["alice-b", "alice_b"]}


def test_env_var_suffix_collisions_empty_for_an_unambiguous_roster() -> None:
    """A roster whose usernames map one-to-one reports nothing."""
    # Act / Assert
    assert env_var_suffix_collisions(["alice", "bob", "carol"]) == {}
    assert env_var_suffix_collisions([]) == {}


def test_env_var_suffix_collisions_ignores_case_only_and_repeated_names() -> None:
    """A verbatim-repeated name is one user listed twice (a duplicate-name error
    reported separately), not two users sharing a credential — while names
    differing only in case really do collide onto one suffix."""
    # Act / Assert
    assert env_var_suffix_collisions(["alice", "alice"]) == {}
    assert env_var_suffix_collisions(["alice", "Alice"]) == {"ALICE": ["Alice", "alice"]}


def test_env_var_suffix_collisions_output_is_sorted_for_stable_messages() -> None:
    """Suffix keys and the names under each are sorted, so a lint or preflight
    message built from this reads the same across runs."""
    # Act
    result = env_var_suffix_collisions(["zed_x", "b-1", "zed-x", "b_1"])

    # Assert
    assert list(result) == ["B_1", "ZED_X"]
    assert result == {"B_1": ["b-1", "b_1"], "ZED_X": ["zed-x", "zed_x"]}


def test_env_var_suffix_collisions_ignores_non_string_entries() -> None:
    """Drop-don't-raise, like the rest of this module: a malformed roster entry
    that slipped through can't crash the collision check."""
    # Act
    result = env_var_suffix_collisions(["alice-b", None, 7, "alice_b"])  # type: ignore[list-item]

    # Assert
    assert result == {"ALICE_B": ["alice-b", "alice_b"]}


def test_env_var_suffix_collisions_consumes_normalize_users_names() -> None:
    """The intended call shape: the names off a normalized roster, so callers
    holding entry dicts (lint, credential provisioning) share one check."""
    # Arrange
    users = normalize_users(["alice-b", {"name": "alice_b", "index": 4}, "bob"])

    # Act
    result = env_var_suffix_collisions(entry["name"] for entry in users)

    # Assert
    assert result == {"ALICE_B": ["alice-b", "alice_b"]}


# ---------------------------------------------------------------------------
# Panel declaration -> credential entitlement
# ---------------------------------------------------------------------------


def _write_persona_project(tmp_path, name: str, panels: dict) -> str:
    """Render a persona project carrying ``web.panels``; return its project_path."""
    import yaml

    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, "web": {"panels": panels}}),
        encoding="utf-8",
    )
    return name


def _catalog_config(catalog: dict, users: list[dict]) -> dict:
    return {"modules": {"web_terminals": {"personas": catalog, "users": users}}}


def test_config_declares_panel_reads_the_panel_block() -> None:
    """A declared panel counts; an absent one does not."""
    config = {"web": {"panels": {"events": {"label": "EVENTS", "url": "http://x"}}}}

    # Assert
    assert config_declares_panel(config, EVENTS_PANEL_ID) is True
    assert config_declares_panel(config, "bluesky") is False
    assert config_declares_panel({}, EVENTS_PANEL_ID) is False


def test_config_declares_panel_treats_disabled_as_undeclared() -> None:
    """A panel switched off needs no credential, so it does not count as declared."""
    config = {"web": {"panels": {"events": {"enabled": False}}}}

    # Assert
    assert config_declares_panel(config, EVENTS_PANEL_ID) is False


def test_config_needs_dispatcher_token_reads_the_events_panel_declaration() -> None:
    """A thin wrapper: it reads exactly what `config_declares_panel(config,
    EVENTS_PANEL_ID)` reads, with no separate config key of its own."""
    config = {"web": {"panels": {"events": {"label": "EVENTS"}}}}

    # Assert
    assert config_needs_dispatcher_token(config) is True
    assert config_needs_dispatcher_token({}) is False


def test_personas_needing_dispatcher_token_selects_only_the_declaring_persona(tmp_path) -> None:
    """The entitlement set is exactly the personas whose rendered project shows the
    panel -- the readonly tier is excluded by its own config, not by a separate key."""
    # Arrange
    catalog = {
        "readwrite": {
            "project": "rw",
            "project_path": _write_persona_project(tmp_path, "rw", {"events": {"label": "EVENTS"}}),
        },
        "readonly": {
            "project": "ro",
            "project_path": _write_persona_project(tmp_path, "ro", {"okf": {"enabled": True}}),
        },
    }
    config = _catalog_config(
        catalog,
        [
            {"name": "alice", "index": 0, "persona": "readwrite"},
            {"name": "bob", "index": 1, "persona": "readonly"},
        ],
    )

    # Act
    result = personas_needing_dispatcher_token(config, tmp_path)

    # Assert
    assert result == {"readwrite"}


def test_personas_needing_dispatcher_token_skips_unrendered_persona_projects(tmp_path) -> None:
    """A persona whose project isn't on disk contributes nothing: a credential is
    never granted on a guess."""
    # Arrange
    config = _catalog_config(
        {"ghost": {"project": "ghost", "project_path": "../never-rendered"}},
        [{"name": "alice", "index": 0, "persona": "ghost"}],
    )

    # Act / Assert
    assert personas_needing_dispatcher_token(config, tmp_path) == set()


# ---------------------------------------------------------------------------
# ARIEL config -> per-user database credential
#
# ARIEL's Postgres password is not panel-gated the way the dispatcher token is:
# two consumers inside a web terminal need it -- the ARIEL panel's own server
# and the `ariel` MCP server the agent calls -- and only one of them is visible
# in `web.panels`. The `ariel:` section is what BOTH read to resolve their DSN,
# so its presence is the entitlement.
# ---------------------------------------------------------------------------


def _write_persona_project_config(tmp_path, name: str, config: dict) -> str:
    """Render a persona project carrying an arbitrary config; return its project_path."""
    import yaml

    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, **config}), encoding="utf-8"
    )
    return name


def test_config_needs_ariel_password_reads_the_ariel_section() -> None:
    """A project carrying an `ariel:` section resolves a DSN and needs the password."""
    # Assert
    assert (
        config_needs_ariel_password({"ariel": {"search_modules": {"keyword": {"enabled": True}}}})
        is True
    )
    assert config_needs_ariel_password({}) is False


def test_config_needs_ariel_password_ignores_an_empty_section() -> None:
    """A key present but empty configures nothing, so it entitles nothing."""
    # Assert
    assert config_needs_ariel_password({"ariel": None}) is False
    assert config_needs_ariel_password({"ariel": {}}) is False


def test_personas_needing_ariel_password_selects_only_the_ariel_persona(tmp_path) -> None:
    """The entitlement set is exactly the personas whose rendered project configures
    ARIEL -- a persona with no logbook gets no logbook credential."""
    # Arrange
    catalog = {
        "readwrite": {
            "project": "rw",
            "project_path": _write_persona_project_config(
                tmp_path, "rw", {"ariel": {"search_modules": {"keyword": {"enabled": True}}}}
            ),
        },
        "readonly": {
            "project": "ro",
            "project_path": _write_persona_project_config(
                tmp_path, "ro", {"web": {"panels": {"okf": {"enabled": True}}}}
            ),
        },
    }
    config = _catalog_config(
        catalog,
        [
            {"name": "alice", "index": 0, "persona": "readwrite"},
            {"name": "bob", "index": 1, "persona": "readonly"},
        ],
    )

    # Act
    result = personas_needing_ariel_password(config, tmp_path)

    # Assert
    assert result == {"readwrite"}


def test_personas_needing_ariel_password_skips_unrendered_persona_projects(tmp_path) -> None:
    """A persona whose project isn't on disk contributes nothing: a credential is
    never granted on a guess."""
    # Arrange
    config = _catalog_config(
        {"ghost": {"project": "ghost", "project_path": "../never-rendered"}},
        [{"name": "alice", "index": 0, "persona": "ghost"}],
    )

    # Act / Assert
    assert personas_needing_ariel_password(config, tmp_path) == set()


# ---------------------------------------------------------------------------
# writes + bluesky server -> per-user queue launch token
#
# `BLUESKY_LAUNCH_TOKEN` arms a queue start, so it is gated on the intersection
# of the two things that make arming meaningful: writes actually enabled, and
# the bluesky MCP server (the token's consumer -- not the panel) actually run.
# The two reads are asymmetric on purpose: `writes_enabled` must be explicitly
# true, while the server's `enabled` override defaults to on when absent.
# ---------------------------------------------------------------------------


def _launch_config(writes_enabled: Any = "absent", bluesky_enabled: Any = "absent") -> dict:
    """A project config with either read spelled out, or omitted via the sentinel."""
    config: dict = {}
    if writes_enabled != "absent":
        config["control_system"] = {"writes_enabled": writes_enabled}
    if bluesky_enabled != "absent":
        config["claude_code"] = {"servers": {"bluesky": {"enabled": bluesky_enabled}}}
    return config


def test_config_needs_launch_token_requires_writes_and_the_bluesky_server() -> None:
    """Both halves of the capability present -- the readwrite tier."""
    # Assert
    assert config_needs_launch_token(_launch_config(True, True)) is True


def test_config_needs_launch_token_defaults_the_bluesky_server_to_enabled() -> None:
    """`claude_code.servers.bluesky.enabled` is an override over an already-enabled
    default, so its absence must not deny the token."""
    # Assert
    assert config_needs_launch_token(_launch_config(writes_enabled=True)) is True


def test_config_needs_launch_token_denies_when_the_bluesky_server_is_off() -> None:
    """Writes alone entitle nothing: with no bluesky server there is no consumer."""
    # Assert
    assert config_needs_launch_token(_launch_config(True, False)) is False


def test_config_needs_launch_token_denies_the_readonly_tier() -> None:
    """The read-only tier is `writes_enabled: false` and still runs the bluesky
    server for browsing -- the server alone must never arm a queue start."""
    # Assert
    assert config_needs_launch_token(_launch_config(False, True)) is False


def test_config_needs_launch_token_requires_writes_enabled_explicitly() -> None:
    """Anything short of a literal `True` -- absent, null, or a truthy non-bool --
    means writes were never granted."""
    # Assert
    assert config_needs_launch_token(_launch_config(bluesky_enabled=True)) is False
    assert config_needs_launch_token(_launch_config(None, True)) is False
    assert config_needs_launch_token(_launch_config("true", True)) is False
    assert config_needs_launch_token(_launch_config(1, True)) is False


def test_config_needs_launch_token_denies_an_empty_config() -> None:
    """A config configuring neither half entitles nothing, and non-dict sections
    are read defensively rather than raising."""
    # Assert
    assert config_needs_launch_token({}) is False
    assert config_needs_launch_token(None) is False
    assert config_needs_launch_token({"control_system": "on"}) is False


def test_personas_needing_launch_token_selects_only_the_readwrite_persona(tmp_path) -> None:
    """The entitlement set is exactly the personas whose rendered project both allows
    writes and runs the bluesky server -- the read-only tier runs the same server (for
    browsing) and must still be excluded, because it is `writes_enabled` that decides,
    not whether the server is present. This is the tier boundary the whole feature
    rests on: a read-only persona can never end up in this set."""
    # Arrange
    catalog = {
        "readwrite": {
            "project": "rw",
            "project_path": _write_persona_project_config(
                tmp_path,
                "rw",
                {
                    "control_system": {"writes_enabled": True},
                    "claude_code": {"servers": {"bluesky": {"enabled": True}}},
                },
            ),
        },
        "readonly": {
            "project": "ro",
            "project_path": _write_persona_project_config(
                tmp_path,
                "ro",
                {
                    "control_system": {"writes_enabled": False},
                    "claude_code": {"servers": {"bluesky": {"enabled": True}}},
                },
            ),
        },
    }
    config = _catalog_config(
        catalog,
        [
            {"name": "alice", "index": 0, "persona": "readwrite"},
            {"name": "bob", "index": 1, "persona": "readonly"},
        ],
    )

    # Act
    result = personas_needing_launch_token(config, tmp_path)

    # Assert
    assert result == {"readwrite"}


def test_personas_needing_launch_token_skips_unrendered_persona_projects(tmp_path) -> None:
    """A persona whose project isn't on disk contributes nothing: a credential is
    never granted on a guess."""
    # Arrange
    config = _catalog_config(
        {"ghost": {"project": "ghost", "project_path": "../never-rendered"}},
        [{"name": "alice", "index": 0, "persona": "ghost"}],
    )

    # Act / Assert
    assert personas_needing_launch_token(config, tmp_path) == set()


# ---------------------------------------------------------------------------
# Shipped-artifact reads: .claude/settings.json Bash deny
# ---------------------------------------------------------------------------


def _write_settings_json(tmp_path, name: str, body: Any) -> str:
    """Render a persona project carrying a ``.claude/settings.json``.

    ``body`` is written verbatim when it is a string (so a test can ship
    unparseable bytes) and JSON-encoded otherwise. Returns the project_path.
    """
    import json as _json

    claude_dir = tmp_path / name / ".claude"
    claude_dir.mkdir(parents=True)
    text = body if isinstance(body, str) else _json.dumps(body)
    (claude_dir / "settings.json").write_text(text, encoding="utf-8")
    return name


def test_settings_json_denies_bash_reads_the_shipped_deny_list(tmp_path) -> None:
    """The happy path: `Bash` listed in permissions.deny is a deny."""
    # Arrange
    _write_settings_json(
        tmp_path, "locked", {"permissions": {"allow": ["Read(/x/**)"], "deny": ["Bash", "Edit"]}}
    )

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "locked") is True


def test_settings_json_denies_bash_false_when_bash_absent_from_a_well_formed_deny(
    tmp_path,
) -> None:
    """A readable artifact that simply doesn't deny the shell -- the `remove_deny`
    case this whole guard exists to catch."""
    # Arrange
    _write_settings_json(
        tmp_path, "open", {"permissions": {"deny": ["Edit", "WebFetch"], "ask": ["Bash"]}}
    )

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "open") is False


def test_settings_json_denies_bash_ignores_a_scoped_bash_deny(tmp_path) -> None:
    """`Bash(rm:*)` constrains one command family and leaves the shell usable, so it
    is not a wholesale deny -- only the exact literal counts."""
    # Arrange
    _write_settings_json(tmp_path, "scoped", {"permissions": {"deny": ["Bash(rm:*)"]}})

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "scoped") is False


def test_settings_json_denies_bash_fails_closed_on_a_missing_file(tmp_path) -> None:
    """An unrendered project proves nothing about its permissions."""
    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "never-rendered") is False


def test_settings_json_denies_bash_fails_closed_on_invalid_json(tmp_path) -> None:
    """A truncated or hand-mangled artifact is unreadable, not safe."""
    # Arrange
    _write_settings_json(tmp_path, "broken", '{"permissions": {"deny": ["Bash"')

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "broken") is False


def test_settings_json_denies_bash_fails_closed_on_a_non_object_document(tmp_path) -> None:
    """Valid JSON that isn't an object carries no permissions at all."""
    # Arrange
    _write_settings_json(tmp_path, "listy", ["Bash"])

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "listy") is False


def test_settings_json_denies_bash_fails_closed_when_permissions_is_missing(tmp_path) -> None:
    """No permissions block means nothing has been denied."""
    # Arrange
    _write_settings_json(tmp_path, "bare", {"hooks": {}})

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "bare") is False


def test_settings_json_denies_bash_fails_closed_when_deny_is_missing(tmp_path) -> None:
    """A permissions block with an allow list but no deny list denies nothing."""
    # Arrange
    _write_settings_json(tmp_path, "allow-only", {"permissions": {"allow": ["Read(/x/**)"]}})

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "allow-only") is False


def test_settings_json_denies_bash_fails_closed_when_deny_is_not_a_list(tmp_path) -> None:
    """A malformed deny value is unreadable, so it proves no deny -- including the
    string "Bash", which must not be read as a one-element list."""
    # Arrange
    _write_settings_json(tmp_path, "stringly", {"permissions": {"deny": "Bash"}})
    _write_settings_json(tmp_path, "nully", {"permissions": {"deny": None}})

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "stringly") is False
    assert settings_json_denies_bash(tmp_path / "nully") is False


def test_settings_json_denies_bash_ignores_non_string_deny_entries(tmp_path) -> None:
    """Junk entries alongside a real deny neither raise nor suppress the deny."""
    # Arrange
    _write_settings_json(tmp_path, "mixed", {"permissions": {"deny": [None, 7, {}, "Bash"]}})

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "mixed") is True


def test_personas_not_denying_bash_reports_only_the_permissive_persona(tmp_path) -> None:
    """The roster form: the persona whose shipped settings dropped the Bash deny is
    named; the locked-down one is not."""
    # Arrange
    catalog = {
        "locked": {
            "project": "locked",
            "project_path": _write_settings_json(
                tmp_path, "locked", {"permissions": {"deny": ["Bash", "Edit"]}}
            ),
        },
        "shelly": {
            "project": "shelly",
            "project_path": _write_settings_json(
                tmp_path, "shelly", {"permissions": {"deny": ["Edit"]}}
            ),
        },
    }
    config = _catalog_config(
        catalog,
        [
            {"name": "alice", "index": 0, "persona": "locked"},
            {"name": "bob", "index": 1, "persona": "shelly"},
        ],
    )

    # Act
    result = personas_not_denying_bash(config, tmp_path)

    # Assert
    assert result == {"shelly"}


def test_personas_not_denying_bash_includes_unrendered_and_pathless_personas(tmp_path) -> None:
    """Fail-closed at roster level too: a persona whose artifact cannot be located
    is reported as permissive rather than silently skipped."""
    # Arrange
    config = _catalog_config(
        {
            "ghost": {"project": "ghost", "project_path": "../never-rendered"},
            "pathless": {"project": "pathless"},
        },
        [
            {"name": "alice", "index": 0, "persona": "ghost"},
            {"name": "bob", "index": 1, "persona": "pathless"},
        ],
    )

    # Act / Assert
    assert personas_not_denying_bash(config, tmp_path) == {"ghost", "pathless"}


def test_personas_not_denying_bash_covers_the_default_persona(tmp_path) -> None:
    """`default_persona` is deployed by every roster entry that names none, so it is
    walked alongside the explicitly referenced ones."""
    # Arrange
    catalog = {
        "fallback": {
            "project": "fallback",
            "project_path": _write_settings_json(
                tmp_path, "fallback", {"permissions": {"deny": ["Edit"]}}
            ),
        },
    }
    config = _catalog_config(catalog, [{"name": "alice", "index": 0}])
    config["modules"]["web_terminals"]["default_persona"] = "fallback"

    # Act / Assert
    assert personas_not_denying_bash(config, tmp_path) == {"fallback"}


def test_personas_not_denying_bash_ignores_unreferenced_catalog_entries(tmp_path) -> None:
    """A persona nobody deploys cannot leak a token, so it is not reported."""
    # Arrange
    catalog = {
        "unused": {
            "project": "unused",
            "project_path": _write_settings_json(tmp_path, "unused", {"permissions": {"deny": []}}),
        },
    }
    config = _catalog_config(catalog, [])

    # Act / Assert
    assert personas_not_denying_bash(config, tmp_path) == set()


def test_personas_not_denying_bash_reads_the_artifact_not_the_config(tmp_path) -> None:
    """`osprey up` does not rebuild, so a config edited after the last build does not
    change what ships. Both directions of that divergence are asserted here, because
    the function must follow the artifact whichever way the two disagree.

    `intent-permissive` has a config.yml that unblocks the shell while its rendered
    settings.json still denies it -- it must NOT be reported.

    `intent-safe` is the security-relevant inverse and the entire reason this function
    reads the artifact at all: its config.yml looks safe (it removes nothing), while the
    shipped settings.json is stale-permissive and omits the Bash deny. A config-reading
    implementation would clear this persona for a launch token while the running image
    still hands its agent a shell. It must be reported UNSAFE.
    """
    # Arrange
    import yaml

    def _write_config_yml(name: str, claude_code: dict) -> None:
        (tmp_path / name / "config.yml").write_text(
            yaml.safe_dump({"claude_code": claude_code}),
            encoding="utf-8",
        )

    # Intent says the shell was unblocked; the built artifact still blocks it.
    _write_settings_json(tmp_path, "intent-permissive", {"permissions": {"deny": ["Bash"]}})
    _write_config_yml("intent-permissive", {"permissions": {"remove_deny": ["Bash"]}})

    # Intent says the shell is blocked; the built artifact does not block it.
    _write_settings_json(tmp_path, "intent-safe", {"permissions": {"deny": ["Edit"]}})
    _write_config_yml("intent-safe", {"permissions": {"remove_deny": []}})

    config = _catalog_config(
        {
            "intent-permissive": {
                "project": "intent-permissive",
                "project_path": "intent-permissive",
            },
            "intent-safe": {"project": "intent-safe", "project_path": "intent-safe"},
        },
        [
            {"name": "alice", "index": 0, "persona": "intent-permissive"},
            {"name": "bob", "index": 1, "persona": "intent-safe"},
        ],
    )

    # Act
    result = personas_not_denying_bash(config, tmp_path)

    # Assert -- the verdict tracks the artifact, opposite to the intent, both ways
    assert result == {"intent-safe"}


def test_settings_json_denies_bash_reads_an_artifact_written_with_a_bom(tmp_path) -> None:
    """`json.load` does not strip a UTF-8 BOM, so a hand-edited artifact saved with one
    would fail to parse and a genuinely Bash-denying persona would read as permissive --
    an opaque deploy refusal. The file is opened as utf-8-sig so the deny is seen."""
    # Arrange
    _write_settings_json(tmp_path, "bommed", '﻿{"permissions": {"deny": ["Bash", "Edit"]}}')

    # Act / Assert
    assert settings_json_denies_bash(tmp_path / "bommed") is True


# ---------------------------------------------------------------------------
# Archiver connector -> per-user store password
#
# The archiver connector authenticates with the variable its own config block
# names (`archiver.<type>.password_env`); `osprey up` mints it into the deploy
# `.env` for a store the project deploys. `.env.users` excludes service tokens
# by design and cannot say "the personas whose archiver reads this", so the
# grant is per-user, and it carries the configured NAME because the block may
# point at a facility-run store under any variable.
# ---------------------------------------------------------------------------

_MONGO_ARCHIVER = {
    "type": "mongodb_archiver",
    "mongodb_archiver": {"host": "localhost", "password_env": "MONGO_ROOT_PASSWORD"},
}


def test_config_archiver_password_env_reads_the_selected_connector_block() -> None:
    """The variable named by the SELECTED connector's block is the entitlement."""
    assert config_archiver_password_env({"archiver": _MONGO_ARCHIVER}) == "MONGO_ROOT_PASSWORD"


def test_config_archiver_password_env_ignores_an_unselected_connector_block() -> None:
    """The shipped config carries a `mongodb_archiver:` block under `type: mock_archiver`;
    a block the selected type never reads entitles nothing."""
    archiver = {**_MONGO_ARCHIVER, "type": "mock_archiver"}

    assert config_archiver_password_env({"archiver": archiver}) is None


def test_config_archiver_password_env_follows_any_connector_that_names_one() -> None:
    """The key, not the connector name, is what is read: a future connector with a
    `password_env` is granted exactly like MongoDB's."""
    archiver = {"type": "facility_db", "facility_db": {"password_env": "FACILITY_DB_PW"}}

    assert config_archiver_password_env({"archiver": archiver}) == "FACILITY_DB_PW"


@pytest.mark.parametrize(
    "missing", [{}, {"archiver": None}, {"archiver": {"type": "mock_archiver"}}]
)
def test_config_archiver_password_env_is_none_without_a_named_variable(missing: dict) -> None:
    assert config_archiver_password_env(missing) is None


@pytest.mark.parametrize("blank", ["", "   ", None, 7])
def test_config_archiver_password_env_treats_a_blank_name_as_unset(blank: Any) -> None:
    archiver = {"type": "mongodb_archiver", "mongodb_archiver": {"password_env": blank}}

    assert config_archiver_password_env({"archiver": archiver}) is None


@pytest.mark.parametrize("bad", ["MONGO PASSWORD", "1PASS", "PW=x", "${PW}", "pw-name"])
def test_config_archiver_password_env_refuses_a_name_compose_cannot_carry(bad: str) -> None:
    """The name is emitted into a compose `environment:` line verbatim, so anything
    that is not a plain identifier is refused here rather than rendered broken."""
    archiver = {"type": "mongodb_archiver", "mongodb_archiver": {"password_env": bad}}

    with pytest.raises(ValueError, match="password_env"):
        config_archiver_password_env({"archiver": archiver})


def test_personas_needing_archiver_password_maps_each_persona_to_its_variable(tmp_path) -> None:
    """The grant is a persona -> variable-name map, so two personas reading two
    different stores each get their own line and a persona with no archiver gets none."""
    # Arrange
    catalog = {
        "readwrite": {
            "project": "rw",
            "project_path": _write_persona_project_config(
                tmp_path, "rw", {"archiver": _MONGO_ARCHIVER}
            ),
        },
        "facility": {
            "project": "fac",
            "project_path": _write_persona_project_config(
                tmp_path,
                "fac",
                {"archiver": {"type": "other_db", "other_db": {"password_env": "OTHER_DB_PW"}}},
            ),
        },
        "readonly": {
            "project": "ro",
            "project_path": _write_persona_project_config(
                tmp_path, "ro", {"archiver": {"type": "mock_archiver"}}
            ),
        },
    }
    config = _catalog_config(
        catalog,
        [
            {"name": "alice", "index": 0, "persona": "readwrite"},
            {"name": "bob", "index": 1, "persona": "readonly"},
            {"name": "carol", "index": 2, "persona": "facility"},
        ],
    )

    # Act
    result = personas_needing_archiver_password(config, tmp_path)

    # Assert
    assert result == {"readwrite": "MONGO_ROOT_PASSWORD", "facility": "OTHER_DB_PW"}


def test_personas_needing_archiver_password_skips_unrendered_persona_projects(tmp_path) -> None:
    """A persona whose project isn't on disk contributes nothing: a credential is
    never granted on a guess."""
    config = _catalog_config(
        {"ghost": {"project": "ghost", "project_path": "../never-rendered"}},
        [{"name": "alice", "index": 0, "persona": "ghost"}],
    )

    assert personas_needing_archiver_password(config, tmp_path) == {}
