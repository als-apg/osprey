"""The retired roster spellings are rewritten at the single profile parse point.

``persona`` / ``default_persona`` / ``authorization.roles.<r>.persona`` and the
``personas:`` catalog read as their ``tier`` successors in every config
spelling (dotted, partly nested, nested); each retired key logs one
deprecation line naming its file; a document carrying both spellings of one
key is refused naming both; and an old-spelled profile hashes like its rewrite.
"""

from __future__ import annotations

import copy
import importlib
import logging
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_deploy import parse_deploy_block
from osprey.cli.build_profile_document import (
    RETIRED_CATALOG_KEY,
    RETIRED_DEFAULT_TIER_KEY,
    RETIRED_TIER_KEY,
    _normalize_retired_roster_keys,
    _parse_profile_document,
    _read_profile_document,
)
from osprey.cli.build_profile_emit import emit_standalone_profile_yaml, emits_layer_profiles
from osprey.cli.build_profile_load import known_profile_keys
from osprey.cli.build_profile_merge import resolve_profile_document
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.build_profile_presets import _presets_dir
from osprey.cli.profile_card import format_profile_card
from osprey.cli.profile_root import ProfileKind
from osprey.deployment.web_terminals import lint as web_terminals_lint
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.render_images import (
    render_build_profile_shape_problem,
    verify_renders,
)
from osprey.deployment.web_terminals.roster import (
    UnresolvedRoleError,
    effective_tier,
    resolve_authorization_roles,
    resolve_cards,
)
from osprey.port_layout import default_port
from osprey.utils.config_writer import config_update_fields
from osprey.utils.workspace import TIERS_DIRNAME
from osprey_connectors.errors import BuildProfileError
from tests._renders import app_render, tier_render

_LOGGER_NAME = doc.__name__
_REMEDY = "run `osprey init --revalidate`"


@pytest.fixture(autouse=True)
def _fresh_dedupe():
    doc._WARNED_RETIRED_KEYS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()


def _parse(text: str, source: str = "profile.yml"):
    return _parse_profile_document(textwrap.dedent(text), source)


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]


OLD = """\
name: demo
config:
  modules.web_terminals:
    enabled: true
    default_persona: readonly
    users:
      - name: alice
        index: 0
        persona: readonly
      - name: bob
        index: 1
        persona: readwrite
      - carol
    authorization:
      roles:
        operator: {persona: readwrite}
        viewer: {persona: readonly}
    personas:
      readonly:
        build_profile: personas/readonly.yml
      readwrite:
        build_profile: personas/readwrite.yml
"""

NEW = """\
name: demo
config:
  modules.web_terminals:
    enabled: true
    default_tier: readonly
    users:
      - name: alice
        index: 0
        tier: readonly
      - name: bob
        index: 1
        tier: readwrite
      - carol
    authorization:
      roles:
        operator: {tier: readwrite}
        viewer: {tier: readonly}
"""


# ── rewrite ──────────────────────────────────────────────────────────────────


def test_old_spelling_parses_to_its_rewrite(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    assert _parse(OLD) == _parse(NEW)


def test_rewrite_preserves_key_order():
    raw = _parse(OLD)
    block = raw["config"]["modules.web_terminals"]
    assert list(block) == ["enabled", "default_tier", "users", "authorization"]
    assert list(block["users"][0]) == ["name", "index", "tier"]


def test_new_spelling_is_untouched_and_silent(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    parsed = _parse(NEW)
    assert parsed["config"]["modules.web_terminals"]["default_tier"] == "readonly"
    assert _warnings(caplog) == []


def test_document_without_config_is_untouched():
    raw = {"name": "x", "persona": "kept"}
    assert _normalize_retired_roster_keys(copy.deepcopy(raw), "f.yml") == raw


def test_top_level_persona_outside_config_is_not_a_roster_key():
    raw = {"name": "x", "config": {"persona": "unrelated", "web.persona": 1}}
    assert _normalize_retired_roster_keys(copy.deepcopy(raw), "f.yml") == raw


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            {"modules.web_terminals.default_persona": "ro"},
            {"modules.web_terminals.default_tier": "ro"},
        ),
        (
            {"modules": {"web_terminals": {"default_persona": "ro"}}},
            {"modules": {"web_terminals": {"default_tier": "ro"}}},
        ),
        (
            {"modules": {"web_terminals.default_persona": "ro"}},
            {"modules": {"web_terminals.default_tier": "ro"}},
        ),
        (
            {"modules.web_terminals.users": [{"name": "a", "persona": "ro"}]},
            {"modules.web_terminals.users": [{"name": "a", "tier": "ro"}]},
        ),
        (
            {"modules": {"web_terminals": {"users": [{"name": "a", "persona": "ro"}]}}},
            {"modules": {"web_terminals": {"users": [{"name": "a", "tier": "ro"}]}}},
        ),
        (
            {"modules.web_terminals.authorization.roles.op.persona": "rw"},
            {"modules.web_terminals.authorization.roles.op.tier": "rw"},
        ),
        (
            {"modules.web_terminals.authorization.roles": {"op": {"persona": "rw"}}},
            {"modules.web_terminals.authorization.roles": {"op": {"tier": "rw"}}},
        ),
        (
            {"modules.web_terminals.authorization": {"roles.op.persona": "rw"}},
            {"modules.web_terminals.authorization": {"roles.op.tier": "rw"}},
        ),
        (
            {"modules": {"web_terminals": {"authorization": {"roles": {"op": {"persona": "rw"}}}}}},
            {"modules": {"web_terminals": {"authorization": {"roles": {"op": {"tier": "rw"}}}}}},
        ),
        (
            {"modules.web_terminals.personas": {"ro": {"build_profile": "personas/ro.yml"}}},
            {},
        ),
        (
            {"modules.web_terminals.personas.ro.build_profile": "personas/ro.yml"},
            {},
        ),
        (
            {"modules": {"web_terminals": {"enabled": True, "personas": {"ro": None}}}},
            {"modules": {"web_terminals": {"enabled": True}}},
        ),
    ],
    ids=[
        "default-dotted",
        "default-nested",
        "default-partly-dotted",
        "users-dotted",
        "users-nested",
        "role-leaf-dotted",
        "role-block",
        "role-dotted-inside-block",
        "role-nested",
        "catalog-dotted",
        "catalog-leaf-dotted",
        "catalog-nested",
    ],
)
def test_every_config_spelling_is_rewritten(old, new):
    raw = {"name": "x", "config": old}
    assert _normalize_retired_roster_keys(raw, "f.yml") == {"name": "x", "config": new}


def test_wrong_shapes_are_left_for_the_schema():
    config = {
        "modules.web_terminals": {
            "users": "not-a-list",
            "authorization": {"roles": ["op"]},
        }
    }
    raw = {"config": copy.deepcopy(config)}
    assert _normalize_retired_roster_keys(raw, "f.yml") == {"config": config}


def test_user_entry_persona_in_bare_string_list_is_ignored():
    raw = {"config": {"modules.web_terminals.users": ["alice", "bob"]}}
    assert _normalize_retired_roster_keys(copy.deepcopy(raw), "f.yml") == raw


# ── deprecation lines ────────────────────────────────────────────────────────


def test_one_deprecation_line_per_key_naming_the_source(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    _parse(OLD, source="/repo/profile.yml")
    lines = _warnings(caplog)
    assert len(lines) == 4, lines
    for line in lines:
        assert "/repo/profile.yml" in line
        assert line.endswith(_REMEDY)
    joined = "\n".join(lines)
    assert "default_persona" in joined and "default_tier" in joined
    assert "users[].persona" in joined and "users[].tier" in joined
    assert "roles.<role>.persona" in joined and "roles.<role>.tier" in joined
    assert "modules.web_terminals.personas" in joined


def test_repeated_parse_of_one_source_warns_once(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    _parse(OLD, source="a.yml")
    _parse(OLD, source="a.yml")
    assert len(_warnings(caplog)) == 4
    _parse(OLD, source="b.yml")
    assert len(_warnings(caplog)) == 8


def test_unmappable_catalog_knobs_are_warned_with_their_destination(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    raw = _parse(
        """\
        config:
          modules.web_terminals:
            personas:
              readwrite:
                build_profile: personas/readwrite.yml
                extra_mounts: [/data]
                seed_base: /seed
                image: ghcr.io/x/y:1
              logbook:
                build_profile: control-assistant-logbook
                landing_group: Standalone deployments
        """,
        source="p.yml",
    )
    assert raw == {"config": {"modules.web_terminals": {}}}
    lines = _warnings(caplog)
    assert all(line.endswith(_REMEDY) and "p.yml" in line for line in lines)
    joined = "\n".join(lines)
    for knob in ("extra_mounts", "seed_base", "image"):
        assert f"modules.web_terminals.personas.readwrite.{knob}" in joined
    assert "tiers/readwrite.yml" in joined
    assert "modules.web_terminals.personas.logbook.landing_group" in joined
    assert "landing.groups" in joined
    # build_profile is derived from the file's location: no line of its own.
    assert "build_profile" not in joined
    # catalog line + three knobs + landing_group
    assert len(lines) == 5, lines


# ── mixed spellings ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "old", "new"),
    [
        (
            {"modules.web_terminals": {"default_persona": "ro", "default_tier": "ro"}},
            "default_persona",
            "default_tier",
        ),
        (
            {
                "modules.web_terminals.default_persona": "ro",
                "modules": {"web_terminals": {"default_tier": "ro"}},
            },
            "default_persona",
            "default_tier",
        ),
        (
            {
                "modules.web_terminals.users": [
                    {"name": "a", "persona": "ro"},
                    {"name": "b", "tier": "rw"},
                ]
            },
            "users[].persona",
            "users[].tier",
        ),
        (
            {"modules.web_terminals.users": [{"name": "a", "persona": "ro", "tier": "ro"}]},
            "users[].persona",
            "users[].tier",
        ),
        (
            {
                "modules.web_terminals.authorization.roles": {
                    "op": {"persona": "rw"},
                    "viewer": {"tier": "ro"},
                }
            },
            "roles.<role>.persona",
            "roles.<role>.tier",
        ),
        (
            {
                "modules.web_terminals.personas": {"ro": {"build_profile": "personas/ro.yml"}},
                "modules.web_terminals.renders": {"ro": {"kind": "tier"}},
            },
            "modules.web_terminals.personas",
            "modules.web_terminals.renders",
        ),
    ],
    ids=["default", "default-across-spellings", "users", "users-one-entry", "roles", "catalog"],
)
def test_document_with_both_spellings_is_refused_naming_both(config, old, new):
    with pytest.raises(BuildProfileError) as excinfo:
        _normalize_retired_roster_keys({"config": config}, "mixed.yml")
    message = str(excinfo.value)
    assert "mixed.yml" in message
    assert old in message
    assert new in message


def test_refusal_reaches_through_the_parse_point():
    with pytest.raises(BuildProfileError, match="default_persona"):
        _parse(
            """\
            config:
              modules.web_terminals.default_persona: ro
              modules.web_terminals.default_tier: ro
            """
        )


# ── content hash ─────────────────────────────────────────────────────────────


def test_old_spelled_profile_hashes_like_its_rewrite(tmp_path: Path):
    from osprey.cli.build_profile_merge import compute_profile_hash

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "profile.yml").write_text(OLD, encoding="utf-8")
    (new_dir / "profile.yml").write_text(NEW, encoding="utf-8")
    old_hash = compute_profile_hash(old_dir / "profile.yml")
    new_hash = compute_profile_hash(new_dir / "profile.yml")
    assert old_hash is not None
    assert old_hash == new_hash


def test_file_reads_go_through_the_rewrite(tmp_path: Path, caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    path = tmp_path / "profile.yml"
    path.write_text(OLD, encoding="utf-8")
    assert _read_profile_document(path) == _parse(NEW)
    assert any(str(path) in line for line in _warnings(caplog))


# ---------------------------------------------------------------------------
# Migration window
# ---------------------------------------------------------------------------


def test_the_rewrite_window_is_still_open():
    """The release that reaches the window's end must drop the rewrite.

    This test fails on that release: remove ``_normalize_retired_roster_keys``
    (and this test with it) so the retired spellings are refused outright.
    """
    from packaging.version import Version

    from osprey.cli.build_profile_load import RETIRED_ROSTER_KEYS_REFUSED_FROM
    from osprey.version import get_release_version

    assert Version(get_release_version()) < Version(RETIRED_ROSTER_KEYS_REFUSED_FROM)


def test_the_window_closes_one_minor_after_the_vocabulary_release():
    from packaging.version import Version

    from osprey.cli.build_profile_load import (
        _PROFILE_SCHEMA_MIN_OSPREY,
        RETIRED_ROSTER_KEYS_REFUSED_FROM,
    )

    floor = Version(_PROFILE_SCHEMA_MIN_OSPREY)
    end = Version(RETIRED_ROSTER_KEYS_REFUSED_FROM)
    assert end > floor
    assert (end.major, end.minor) == (floor.major, floor.minor + 1)
    assert end.micro == 0


def test_a_final_release_is_not_below_the_vocabulary_floor():
    """A final release tagged below the floor would refuse its own renders.

    Every render is stamped with its release; once that install is upgraded,
    ``check_drift`` refuses any stamp below the floor as "before tiers/apps".
    The release that carries the vocabulary must therefore be numbered at or
    above the floor — this guard fails the tag that is not.
    """
    from packaging.version import Version

    from osprey.cli.build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY
    from osprey.version import get_release_version, is_prerelease, is_release

    release = get_release_version()
    if not is_release() or is_prerelease(release):
        pytest.skip("only a tagged final release is bound by the floor")
    assert Version(release) >= Version(_PROFILE_SCHEMA_MIN_OSPREY)


# ---------------------------------------------------------------------------
# The retired words never leave the parse point
# ---------------------------------------------------------------------------
#
# Past the rewrite above, nothing the framework validates, emits, renders or
# reports spells a retired roster word: no kind's key set accepts one, the
# deploy block sends each to its real home, no verb or message is named for
# one, and no preset or rendered config carries one.


@pytest.mark.parametrize("kind", list(ProfileKind))
@pytest.mark.parametrize(
    "retired", [RETIRED_TIER_KEY, RETIRED_CATALOG_KEY, RETIRED_DEFAULT_TIER_KEY]
)
def test_no_profile_kind_accepts_a_retired_roster_key(kind: ProfileKind, retired: str) -> None:
    """Retired spellings are rewritten before validation, never accepted there."""
    assert retired not in known_profile_keys(kind)


def _deploy_refusal(**overrides: Any) -> str:
    """The deploy-block refusal for a minimal valid block plus *overrides*."""
    deploy = {
        "ci": "gitlab",
        "registry": {
            "url": "git.example.org:5050/physics/production/facility-profiles",
            "token_env_var": "FACILITY_REGISTRY_TOKEN",
        },
        **overrides,
    }
    with pytest.raises(BuildProfileError) as excinfo:
        parse_deploy_block({"name": "Facility", "deploy": deploy})
    return str(excinfo.value)


@pytest.mark.parametrize(
    ("key", "value", "expected_home"),
    [
        (RETIRED_CATALOG_KEY, {"analysis": {}}, "one file per tier in `tiers/`"),
        (RETIRED_DEFAULT_TIER_KEY, "assistant", "modules.web_terminals.default_tier"),
    ],
)
def test_the_deploy_block_sends_a_retired_roster_key_to_its_real_home(
    key: str, value: Any, expected_home: str
) -> None:
    message = _deploy_refusal(**{key: value})

    assert f"'{key}' is not a deploy key" in message
    assert expected_home in message


def test_no_scaffold_verb_is_named_for_the_retired_catalog() -> None:
    from osprey.cli.scaffold_cmd import scaffold

    assert RETIRED_CATALOG_KEY not in scaffold.commands
    with pytest.raises(ImportError):
        importlib.import_module(f"osprey.cli.scaffold_{RETIRED_CATALOG_KEY}")


@pytest.mark.parametrize("layer", ["tiers/readonly.yml", "apps/logbook.yml"])
def test_extends_in_a_layer_is_refused_without_the_retired_tier_word(
    tmp_path: Path, layer: str
) -> None:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "profile.yml").write_text("name: Facility\ndata: data\n", encoding="utf-8")
    path = repo / layer
    path.parent.mkdir(parents=True)
    path.write_text("extends: control-assistant\n", encoding="utf-8")

    with pytest.raises(BuildProfileError, match="'extends'") as excinfo:
        resolve_profile_document(yaml.safe_load(path.read_text(encoding="utf-8")), path, warn=False)
    assert RETIRED_TIER_KEY.capitalize() not in str(excinfo.value)


def test_an_emitted_preset_carries_no_retired_catalog() -> None:
    """The control-assistant roster, folded into one key by a dotted override."""
    text = emit_standalone_profile_yaml(
        "control-assistant",
        ("config.modules.web_terminals.enabled=false",),
        "Emitted",
        project_name="emitted",
    )
    subtree = yaml.safe_load(text)["config"]["modules.web_terminals"]

    assert subtree["users"]
    assert RETIRED_CATALOG_KEY not in subtree


def test_a_retired_catalog_in_a_profile_config_emits_no_render() -> None:
    # The retired catalog is dropped when a profile is read; a raw spelling of
    # it no longer triggers anything either.
    legacy = {
        "modules.web_terminals.enabled": True,
        f"modules.web_terminals.{RETIRED_CATALOG_KEY}": {
            "ops": {"build_profile": f"{TIERS_DIRNAME}/ops.yml"}
        },
    }

    assert not emits_layer_profiles(legacy)


def test_the_control_assistant_roster_renders_no_retired_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"system": {}}), encoding="utf-8")
    profile, _ = resolve_build_profile(None, preset="control-assistant")
    config_update_fields(config_path, profile.config)
    wt = yaml.safe_load(config_path.read_text(encoding="utf-8"))["modules"]["web_terminals"]

    assert wt["users"]
    for retired in (RETIRED_CATALOG_KEY, RETIRED_DEFAULT_TIER_KEY):
        assert retired not in wt, retired
    for user in wt["users"]:
        assert RETIRED_TIER_KEY not in user, user


def test_the_control_assistant_preset_never_says_the_retired_tier_word() -> None:
    """Every comment of the preset is copied into the operator's ``profile.yml``,
    so the word here would teach the retired vocabulary to every new deployment."""
    text = (_presets_dir() / "control-assistant.yml").read_text(encoding="utf-8")

    assert re.findall(rf"(?i){RETIRED_TIER_KEY}", text) == []


# ── render verification messages ──

_PROJECT = "demo"


def _render_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "profile.yml").write_text(
        f"name: Demo\nproject_name: {_PROJECT}\ndata: data\n", encoding="utf-8"
    )
    for layer in ("tiers/readonly.yml", "apps/logbook.yml"):
        (repo / layer).parent.mkdir(exist_ok=True)
        (repo / layer).write_text(f"name: {Path(layer).stem}\n", encoding="utf-8")
    return repo


def _verify_refusal(repo: Path, name: str, kind: str, project_path: Path) -> str:
    # The roster serves the render: a card on the tier, or the shared app.
    roster = (
        {"shared": [name]}
        if kind == "app"
        else {"users": [{"name": "a", "index": 0, "tier": name}]}
    )
    config = {
        "project_name": _PROJECT,
        "modules": {
            "web_terminals": {
                **roster,
                "renders": {
                    name: {
                        "kind": kind,
                        "project": f"{_PROJECT}-{name}",
                        "project_path": str(project_path),
                    }
                },
            }
        },
    }
    unit = [{"render": name, "project": f"{_PROJECT}-{name}", "image": "x:local"}]
    with pytest.raises(ValueError) as excinfo:
        verify_renders(config, unit, repo_root=repo)
    return str(excinfo.value)


@pytest.mark.parametrize(("name", "kind"), [("readonly", "tier"), ("logbook", "app")])
def test_an_absent_render_is_explained_without_the_retired_tier_word(
    tmp_path: Path, name: str, kind: str
) -> None:
    repo = _render_repo(tmp_path)
    message = _verify_refusal(repo, name, kind, tmp_path / "gone" / f"{_PROJECT}-{name}")

    assert "osprey build" in message
    assert f"{kind}s/{name}.yml" in message
    assert RETIRED_TIER_KEY not in message.lower()


def test_a_missing_layer_file_is_named_without_the_retired_catalog(tmp_path: Path) -> None:
    repo = _render_repo(tmp_path)
    message = _verify_refusal(repo, "ghost", "tier", tmp_path / f"{_PROJECT}-ghost")

    assert str(repo / "tiers" / "ghost.yml") in message
    assert f"modules.web_terminals.{RETIRED_CATALOG_KEY}" not in message


def test_the_render_shape_problem_never_says_the_retired_tier_word() -> None:
    problem = render_build_profile_shape_problem("control-assistant")

    assert problem is not None
    assert RETIRED_TIER_KEY not in problem.lower()


# ── lint ──


def _lint_config(**web_terminals: object) -> dict:
    """A rendered-altitude config whose roster is alice, bob and two apps."""
    subtree: dict = {
        "enabled": True,
        "nginx_port": default_port("nginx", base=20000),
        "web_base_port": default_port("web", 0, base=20000),
        "artifact_base_port": default_port("artifact", 0, base=20000),
        "ariel_base_port": default_port("ariel", 0, base=20000),
        "lattice_base_port": default_port("lattice", 0, base=20000),
        "users": [{"name": "alice", "index": 0}, {"name": "bob", "index": 1}],
        "shared": ["logbook", "knowledge"],
        "default_tier": "readonly",
        **web_terminals,
    }
    subtree["renders"] = {
        "logbook": app_render("logbook", "demo-logbook"),
        "knowledge": app_render("knowledge", "demo-knowledge"),
        **subtree.get("renders", {}),
    }
    return {
        "facility": {"prefix": "test"},
        "registry": {"url": "registry.example.org/demo"},
        "modules": {"web_terminals": subtree},
    }


def _lint_messages(config: dict, code: str) -> list[str]:
    return [f.message for f in lint_web_terminals(config) if f.code == code]


def test_no_lint_code_says_the_retired_tier_word() -> None:
    """The codes that spelled it were renamed outright: no alias survives."""
    source = Path(web_terminals_lint.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'"web_terminals\.([a-z0-9_]+)"', source))

    assert emitted
    assert sorted(code for code in emitted if RETIRED_TIER_KEY in code) == []
    assert not re.findall(rf'"web_terminals\.{RETIRED_TIER_KEY}_', source)


def test_a_project_collision_is_reported_without_the_retired_tier_word() -> None:
    config = _lint_config(
        image_source="local",
        users=[
            {"name": "alice", "index": 0, "tier": "readonly"},
            {"name": "bob", "index": 1, "tier": "readwrite"},
        ],
        renders={
            "readonly": tier_render("readonly", "demo-ops", project_path="build/demo-ops"),
            "readwrite": tier_render("readwrite", "demo-ops", project_path="build/demo-ops"),
        },
    )
    messages = _lint_messages(config, "web_terminals.render_project_collision")

    assert len(messages) == 1
    assert RETIRED_TIER_KEY not in messages[0]


def test_a_render_shadowing_the_project_image_is_reported_without_the_retired_tier_word() -> None:
    config = _lint_config(
        image_source="local",
        users=[{"name": "alice", "index": 0, "tier": "readonly"}],
        renders={"readonly": tier_render("readonly", "demo", project_path="build/demo")},
    )
    config["project_name"] = "demo"
    messages = _lint_messages(config, "web_terminals.render_shadows_project_image")

    assert len(messages) == 1
    assert RETIRED_TIER_KEY not in messages[0]


# ── Rendered readers know only the new keys ─────────────────────────────────
#
# A rendered config is written after the parse point, so it never carries a
# retired key. A hand-authored one that does is read as though the key were
# absent: a role or default spelled only the old way names no render, and the
# strict readers fail closed rather than guess.


def test_a_role_carrying_only_the_retired_key_names_no_render() -> None:
    stanza = {"authorization": {"roles": {"operator": {RETIRED_TIER_KEY: "console"}}}}
    with pytest.raises(ValueError, match="operator"):
        resolve_authorization_roles(stanza)
    with pytest.raises(UnresolvedRoleError):
        effective_tier({"name": "a", "role": "operator"}, {}, None, strict=True)


def test_a_user_pinned_only_by_the_retired_key_lands_on_the_default() -> None:
    assert effective_tier({"name": "a", RETIRED_TIER_KEY: "admin"}, {}, "readonly") == "readonly"


def test_the_profile_card_reads_no_default_from_the_retired_key() -> None:
    def ops_row(key: str) -> str:
        card = format_profile_card(
            BuildProfile(
                name="roster",
                config={
                    "modules.web_terminals.enabled": True,
                    "modules.web_terminals.users": [{"name": "ops", "index": 0}],
                    f"modules.web_terminals.{key}": "readonly",
                },
            ),
            {},
        )
        return next(line for line in card if "ops" in line)

    assert "readonly" in ops_row("default_tier")
    assert "readonly" not in ops_row(RETIRED_DEFAULT_TIER_KEY)


def test_resolve_cards_reads_no_retired_catalog_or_default() -> None:
    subtree = {
        "enabled": True,
        "image_source": "local",
        "users": [{"name": "alice", "index": 0}],
        RETIRED_DEFAULT_TIER_KEY: "readonly",
        RETIRED_CATALOG_KEY: {"readonly": tier_render("readonly", "demo-readonly")},
    }
    with pytest.raises(ValueError, match="default_tier"):
        resolve_cards(subtree, {})
