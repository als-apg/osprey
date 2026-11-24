"""Every profile key that can name a path has been decided.

The build protects the paths a profile names from three feeds the emitter
exports (:mod:`osprey.cli.build_protected_paths`). A profile key that can hold a
string and that no feed reads is a location agent code may write unseen, so
this module walks the profile schema — the :class:`BuildProfile` dataclasses,
the top-level keys the loader knows that are no field, and the per-render keys
a login file may state — and holds every string-capable leaf to exactly one
decision: a feed of the emitter, an exemption with its reason, or the reviewed
not-a-path remainder. A free-form mapping, whose members no schema names, is
listed as not walked with the reason its members need no walk.

The configuration ``config:`` block is decided key by key in the config-key
manifest (``path:``), checked by ``scripts/check_config_keys.py``; the build
warns about a path-shaped value under a configuration key no table decides,
and the bundled presets raise no such warning.
"""

from __future__ import annotations

import dataclasses
import shutil
import types
import typing
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import pytest
import yaml

from osprey.cli.build_profile_document import RENDER_KEYS
from osprey.cli.build_profile_load import ROOT_KIND_KEYS, TIER_KIND_KEYS
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.cli.build_protected_paths import (
    INPUT_ENUMERATION_KEYS,
    PROFILE_PATH_FIELDS,
    emit_protected_paths,
)
from osprey.utils.workspace import TIERS_DIRNAME

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

#: The root preset every bundled login is composed over, as a build composes
#: a shared login.
LOGIN_HOST_PRESET = "demo"

#: Profile keys the emitter protects, by feed.
FEEDS: dict[str, str] = {
    **dict.fromkeys(INPUT_ENUMERATION_KEYS, "input enumeration"),
    **{row.key: "profile path field" for row in PROFILE_PATH_FIELDS},
}

#: String-capable profile keys that can name a location but need no protection.
EXEMPT: dict[str, str] = {
    "osprey_install": "not-a-path ('local', 'pip' or a PEP 508 spec)",
    "environment.packages": "not-a-path (package specs)",
    "lifecycle.pre_build[].run": "not-a-path (a command)",
    "lifecycle.post_build[].run": "not-a-path (a command)",
    "lifecycle.validate[].run": "not-a-path (a command)",
    "lifecycle.post_build[].cwd": (
        "already-confined (anchored in the render; the schema refuses absolute and .. values)"
    ),
    "lifecycle.validate[].cwd": (
        "already-confined (anchored in the render; the schema refuses absolute and .. values)"
    ),
    "mcp_servers.<name>.command": "not-a-path (a command)",
    "mcp_servers.<name>.args": "not-a-path (command arguments)",
    "mcp_servers.<name>.env": "not-a-path (environment values)",
    "env.defaults": "not-a-path (environment values)",
    "dispatch.env": "not-a-path (environment variable names)",
    "dispatch.tier": "already-confined (names a file in tiers/, which is protected)",
    "claude_md_template": "not-a-path (a bundled template name)",
    "extra_mounts": (
        "a mount into the login's own container, with the mode the operator states (r-0005)"
    ),
    "seed_base": "not-a-path",
    "image": "not-a-path (an image reference)",
}

#: String-capable profile keys the key table does not name: not paths. Each
#: was read once against its parser when this table was written.
NOT_A_PATH: dict[str, str] = {
    "name": "a display label",
    "project_name": "the compose project name",
    "deploy.ci": "a CI provider name",
    "deploy.host.name": "a host name",
    "deploy.host.user": "a user name",
    "deploy.host.fqdn": "a host name",
    "deploy.registry.url": "a URL",
    "deploy.registry.token_env_var": "an environment variable name",
    "deploy.image_source": "a word",
    "deploy.external_projects[].name": "a project name",
    "deploy.external_projects[].url": "a URL",
    "deploy.external_projects[].image": "an image reference",
    "deploy.external_projects[].token_env_var": "an environment variable name",
    "provider": "a provider name",
    "model": "a model name",
    "channel_finder_mode": "a word",
    "mcp_servers.<name>.permissions": "tool names",
    "mcp_servers.<name>.url": "a URL",
    "mcp_servers.<name>.transport": "a word",
    "mcp_servers.<name>.hooks": "hook names",
    "mcp_servers.<name>.writes": "tool names",
    "mcp_servers.<name>.condition": "a condition expression",
    "mcp_servers.<name>.app": "an application name",
    "lifecycle.pre_build[].name": "a step name",
    "lifecycle.post_build[].name": "a step name",
    "lifecycle.validate[].name": "a step name",
    "env.required": "environment variable names",
    "env.pinned": "environment variable names",
    "environment.inherit_exclude": "package names",
    "requires_osprey_version": "a version specifier",
    "web_panels": "panel ids",
    "default_panel": "a panel id",
    "dispatch.workspace_mode": "a word",
    "dispatch.transport": "a word",
    "dispatch.channel_strip_prefix": "a channel-name prefix",
    "dispatch.network": "a word",
    "bluesky.excluded_plans": "plan names",
    "bluesky.external.zmq_control_addr": "a network address",
    "bluesky.external.zmq_public_key_env": "an environment variable name",
    "bluesky.external.tiled_uri": "a URL",
    "bluesky.external.tiled_api_key_env": "an environment variable name",
    "nextcloud_bridge.trigger": "a mention word",
    "gchat_bridge.trigger": "a mention word",
    "teams_bridge.trigger": "a mention word",
    "slack_bridge.trigger": "a mention word",
    "va_archiver.database": "a database name",
    "va_archiver.collection": "a collection name",
    "va_archiver.compression": "a word",
    "va_archiver.username": "a user name",
    "va_archiver.auth_database": "a database name",
    "va_archiver.password_env": "an environment variable name",
    "va_archiver.freshness_channel": "a channel name",
    "va_archiver.host": "a host name",
    "agent_record.backend": "a word",
    "agent_record.memory": "a memory size",
    "agent_record.clickhouse.image": "an image reference",
    "agent_record.clickhouse.memory": "a memory size",
    "apps.<name>.access": "login names",
    "lists": "'own' or 'add'",
    "provenance.preset": "a preset name",
    "provenance.preset_hash": "a digest",
    "provenance.providers_hash": "a digest",
    "connector": "a connector type",
    "exclude": "artifact and key names",
    "port_base": "a port number",
}

#: Free-form mappings: members no schema names, and why they need no walk.
NOT_WALKED: dict[str, str] = {
    "config": "decided key by key in the config-key manifest (`path:`)",
    "services.<name>.config": "read by the emitter as services.<name>.<key>, as the manifest decides",
    "panel_presets": "dock layouts naming panel ids",
    "artifact_server": "already-confined (extension viewers must sit under project/)",
    "agent_record.readers": "reader names and their grants",
    "apps.<name>.config": "the schema belongs to the app; OSPREY cannot know its paths (r-0005)",
    "apps.<name>.services.<name>.network": "a network mode",
    "apps.<name>.services.<name>.env": "environment values",
    "watchdogs": "each watchdog's directory is the input-enumeration entry watchdogs.<name>",
}

TABLES: dict[str, dict[str, str]] = {
    "the emitter's feeds": FEEDS,
    "EXEMPT": EXEMPT,
    "NOT_A_PATH": NOT_A_PATH,
}


@dataclasses.dataclass(frozen=True)
class Leaf:
    """One walked profile key: its spelling and what its type can hold."""

    key: str
    string_capable: bool
    free_form: bool = False


def _string_capable(hint: Any) -> bool:
    origin = get_origin(hint)
    if origin is Literal:
        return any(isinstance(arg, str) for arg in get_args(hint))
    return isinstance(hint, type) and issubclass(hint, str)


def _visit(hint: Any, key: str, out: list[Leaf]) -> None:
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        for arg in get_args(hint):
            if arg is not type(None):
                _visit(arg, key, out)
        return
    if dataclasses.is_dataclass(hint):
        walk_dataclass(hint, f"{key}.", out)
        return
    if origin in (list, tuple, set, frozenset):
        (item, *_rest) = get_args(hint) or (Any,)
        _visit(item, f"{key}[]", out)
        return
    if origin is dict or hint is dict:
        value = (get_args(hint) or (str, Any))[1]
        if value is Any or get_origin(value) is dict:
            out.append(Leaf(key, string_capable=True, free_form=True))
        else:
            _visit(value, f"{key}.<name>", out)
        return
    if hint is Any:
        out.append(Leaf(key, string_capable=True, free_form=True))
        return
    out.append(Leaf(key, string_capable=_string_capable(hint)))


def walk_dataclass(cls: type, prefix: str = "", out: list[Leaf] | None = None) -> list[Leaf]:
    """Every leaf below *cls*: ``[]`` marks a list member, ``<name>`` a mapping key."""
    out = [] if out is None else out
    hints = get_type_hints(cls)
    for item in dataclasses.fields(cls):
        _visit(hints[item.name], f"{prefix}{item.name}", out)
    return out


def walk_profile_schema() -> dict[str, Leaf]:
    """The profile schema's leaves by key; a leaf any of whose types holds a
    string is string-capable."""
    leaves: dict[str, Leaf] = {}
    for leaf in walk_dataclass(BuildProfile):
        seen = leaves.get(leaf.key)
        leaves[leaf.key] = Leaf(
            leaf.key,
            string_capable=leaf.string_capable or bool(seen and seen.string_capable),
            free_form=leaf.free_form or bool(seen and seen.free_form),
        )
    fields = {item.name for item in dataclasses.fields(BuildProfile)}
    for key in sorted((ROOT_KIND_KEYS | TIER_KIND_KEYS | RENDER_KEYS) - fields):
        # Spelled in a profile file but parsed into no field: the type is the
        # YAML author's, so the key is held to a decision like a string.
        leaves[key] = Leaf(key, string_capable=True)
    return leaves


def spellings(key: str) -> list[str]:
    """*key*, then the container spellings a table may name it by.

    ``hooks[]`` is decided as ``hooks`` and
    ``bluesky.external.parameter_schemas.<name>`` as
    ``bluesky.external.parameter_schemas``.
    """
    found = [key]
    while True:
        if key.endswith("[]"):
            key = key[:-2]
        elif key.endswith(".<name>"):
            key = key[: -len(".<name>")]
        else:
            return found
        found.append(key)


def decisions(key: str) -> list[tuple[str, str]]:
    """The tables deciding *key* at its most specific decided spelling."""
    for spelling in spellings(key):
        found = [(name, table[spelling]) for name, table in TABLES.items() if spelling in table]
        if found:
            return found
    return []


@pytest.fixture(scope="module")
def schema() -> dict[str, Leaf]:
    return walk_profile_schema()


def test_every_string_capable_profile_key_is_decided(schema: dict[str, Leaf]) -> None:
    undecided = sorted(
        leaf.key
        for leaf in schema.values()
        if leaf.string_capable and not leaf.free_form and not decisions(leaf.key)
    )
    assert not undecided, (
        "These profile keys can hold a string and nothing decides whether the build "
        "protects what they name:\n  "
        + "\n  ".join(undecided)
        + "\nDecide each one: if it can name a file or directory a build or a service "
        "reads, add a row to build_protected_paths.PROFILE_PATH_FIELDS; if it can "
        "name one that needs no protection, add it to EXEMPT here with the reason "
        "(not-a-path, already-confined, derived-not-statable, runtime-output, "
        "other-container-filesystem); if it never names a location, add it to "
        "NOT_A_PATH here with what it holds."
    )


def test_no_profile_key_is_decided_twice(schema: dict[str, Leaf]) -> None:
    twice = {
        leaf.key: [name for name, _reason in found]
        for leaf in schema.values()
        if len(found := decisions(leaf.key)) > 1
    }
    assert not twice, f"each key takes exactly one decision: {twice}"


def test_free_form_mappings_are_listed_as_not_walked(schema: dict[str, Leaf]) -> None:
    unlisted = sorted(
        leaf.key
        for leaf in schema.values()
        if leaf.free_form and not any(s in NOT_WALKED for s in spellings(leaf.key))
    )
    assert not unlisted, (
        f"{unlisted} take members no schema names; list each in NOT_WALKED with the "
        "reason its members need no walk, or give it a schema"
    )


def test_every_decision_names_a_profile_key(schema: dict[str, Leaf]) -> None:
    """A table row the schema no longer spells is a decision about nothing."""
    known = {spelling for key in schema for spelling in spellings(key)}
    known |= {f"{leaf.key}.<name>" for leaf in schema.values() if leaf.free_form}
    known |= {key.rsplit(".", 1)[0] for key in known if "." in key}
    stale = sorted(
        key for table in (*TABLES.values(), NOT_WALKED) for key in table if key not in known
    )
    assert not stale, f"these decided keys name no profile key: {stale}"


def test_a_feed_key_is_never_also_exempt() -> None:
    both = sorted(set(FEEDS) & (set(EXEMPT) | set(NOT_A_PATH)))
    assert not both, f"protected and exempt at once: {both}"


# ── the walk itself ──────────────────────────────────────────────────────


@dataclasses.dataclass
class _Inner:
    where: str
    count: int = 0


@dataclasses.dataclass
class _Toy:
    plain: str
    maybe: str | None = None
    either: int | str = 0
    number: int = 0
    word: Literal["a", "b"] = "a"
    flag: Literal[1, 2] = 1
    names: list[str] = dataclasses.field(default_factory=list)
    nested: _Inner | None = None
    items: list[_Inner] = dataclasses.field(default_factory=list)
    named: dict[str, _Inner] = dataclasses.field(default_factory=dict)
    values: dict[str, str] = dataclasses.field(default_factory=dict)
    free: dict[str, Any] = dataclasses.field(default_factory=dict)


def test_the_walk_reads_types_not_names() -> None:
    walked = {(leaf.key, leaf.string_capable, leaf.free_form) for leaf in walk_dataclass(_Toy)}
    assert walked == {
        ("plain", True, False),
        ("maybe", True, False),
        ("either", False, False),
        ("either", True, False),
        ("number", False, False),
        ("word", True, False),
        ("flag", False, False),
        ("names[]", True, False),
        ("nested.where", True, False),
        ("nested.count", False, False),
        ("items[].where", True, False),
        ("items[].count", False, False),
        ("named.<name>.where", True, False),
        ("named.<name>.count", False, False),
        ("values.<name>", True, False),
        ("free", True, True),
    }


def test_a_typed_leaf_that_becomes_string_capable_needs_a_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``deploy_services`` passes by type; widened to a string it is undecided."""
    schema = walk_profile_schema()
    assert not schema["deploy_services"].string_capable
    assert not decisions("deploy_services")

    hints = get_type_hints(BuildProfile)
    hints["deploy_services"] = bool | str
    real = typing.get_type_hints
    monkeypatch.setattr(
        f"{__name__}.get_type_hints",
        lambda cls: hints if cls is BuildProfile else real(cls),
    )
    widened = walk_profile_schema()
    assert widened["deploy_services"].string_capable


# ── the build-time net over the bundled presets ──────────────────────────


def _root_presets() -> list[str]:
    return sorted(path.stem for path in PRESET_DIR.glob("*.yml"))


def _login_presets() -> list[str]:
    return sorted(path.stem for path in (PRESET_DIR / TIERS_DIRNAME).glob("*.yml"))


def test_the_presets_are_found() -> None:
    assert LOGIN_HOST_PRESET in _root_presets()
    assert len(_root_presets()) >= 2
    assert _login_presets()


@pytest.mark.parametrize("preset", _root_presets())
def test_a_bundled_root_preset_raises_no_undecided_path(preset: str, tmp_path: Path) -> None:
    loaded = resolve_build_document(None, preset)
    found = emit_protected_paths(
        [loaded], deployment_root=tmp_path, agent_data_root=Path("var/agent_data")
    )
    assert found.undecided == (), (
        f"{preset} states a path-shaped value under a configuration key no table decides, "
        f"so `osprey build` warns: {found.undecided}. Decide the key in the config-key "
        "manifest (`path:`) and in build_protected_paths."
    )


@pytest.fixture(scope="module")
def composed_logins(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Each bundled login composed over its host preset, as a build composes a
    person's login (a shared login may not grant writes, so composing every
    file as shared would refuse the ones that do)."""
    repo = tmp_path_factory.mktemp("logins")
    (repo / "data").mkdir()
    logins = _login_presets()
    (repo / "profile.yml").write_text(
        yaml.safe_dump({"extends": LOGIN_HOST_PRESET, "data": "data"})
    )
    (repo / TIERS_DIRNAME).mkdir()
    root = resolve_build_document(repo / "profile.yml", None)
    composed: dict[str, Any] = {"<root>": (repo, root)}
    for name in logins:
        target = repo / TIERS_DIRNAME / f"{name}.yml"
        shutil.copyfile(PRESET_DIR / TIERS_DIRNAME / f"{name}.yml", target)
        composed[name] = (repo, resolve_build_document(target, None))
    return composed


@pytest.mark.parametrize("login", _login_presets())
def test_a_bundled_login_raises_no_undecided_path(
    login: str, composed_logins: dict[str, Any]
) -> None:
    repo, root = composed_logins["<root>"]
    _repo, loaded = composed_logins[login]
    found = emit_protected_paths(
        [root, loaded], deployment_root=repo, agent_data_root=Path("var/agent_data")
    )
    assert found.undecided == (), (
        f"the bundled login {login} states a path-shaped value under a configuration "
        f"key no table decides: {found.undecided}"
    )
