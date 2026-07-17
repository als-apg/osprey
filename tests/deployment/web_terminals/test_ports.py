"""Tests for per-user host port allocation (osprey.deployment.web_terminals.ports)."""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals.ports import allocate_ports, normalize_users, resolve_personas

_BASE_PORTS = {"web": 8087, "artifact": 8086, "ariel": 8085, "lattice": 8097}


def test_allocate_ports_returns_all_four_families() -> None:
    """Every one of the four canonical families must be present in the result."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    result = allocate_ports(base_ports, index=1)

    # Assert
    assert set(result.keys()) == {"web", "artifact", "ariel", "lattice"}


def test_allocate_ports_adds_index_to_base_port() -> None:
    """Each family's allocated port equals its configured base port plus the index."""
    # Arrange
    base_ports = dict(_BASE_PORTS)
    index = 3

    # Act
    result = allocate_ports(base_ports, index)

    # Assert
    assert result == {
        "web": 8090,
        "artifact": 8089,
        "ariel": 8088,
        "lattice": 8100,
    }


def test_allocate_ports_index_zero_returns_base_ports_unchanged() -> None:
    """Index 0 (the first user) must map exactly onto the configured base ports."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    result = allocate_ports(base_ports, index=0)

    # Assert
    assert result == _BASE_PORTS


def test_allocate_ports_distinct_indices_do_not_collide() -> None:
    """Two different user indices must never allocate the same port for a family."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    user_a = allocate_ports(base_ports, index=0)
    user_b = allocate_ports(base_ports, index=1)

    # Assert
    for family in _BASE_PORTS:
        assert user_a[family] != user_b[family]


def test_allocate_ports_missing_family_raises_value_error() -> None:
    """A base_ports dict missing a required family must raise a clear ValueError."""
    # Arrange
    incomplete_base_ports = {"web": 8087, "artifact": 8086, "ariel": 8085}

    # Act / Assert
    with pytest.raises(ValueError, match="lattice"):
        allocate_ports(incomplete_base_ports, index=0)


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
        },
        {
            "name": "bob",
            "index": 1,
            "persona": None,
            "image": "registry.example.org/osprey/web-terminal:latest",
            "project": "als-assistant",
            "container_project_dir": "/app/als-assistant",
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
