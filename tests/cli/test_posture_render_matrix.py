"""Every render path resolves the write posture PR-3's per-layer rule gives it.

A matrix over the documents a render is built from — the root, an ``extends``
base under it, a tier (live ``tiers/`` and the retired layer directory), an
app, and the ``--set`` edit — each written in every spelling of the write
posture: silent, the tier key nested or dotted, the retired flat leaf nested or
dotted, a retired per-type leaf, and a retired per-type leaf on a custom
connector type whose name holds a dot (``mypkg.Mod``, one key, never a path).

The oracle is restated here from PR-3's build-time rewrite; it never calls
the normalizer. Per document: the tier key's value when it spells one
(``never``, ``asked`` or ``allowed``, carried through verbatim); else, when it
spells any retired leaf, ``allowed`` if any of them (the flat leaf or a per-type
leaf, each type looked up whole) is a literal ``True`` and ``never`` otherwise,
so no retired spelling ever answers ``asked``; else the document is silent. A
root or tier render takes the nearest speaking document — the ``--set`` edit,
then the tier, then the root's ``extends`` chain from child to base — and is
``never`` when all are silent. An app render is ``never`` whatever any document
says, and an app spelling an arming word is refused.

Beside the posture it pins that no retired leaf survives into any render, that
a document or ``--set`` edit spelling both forms is refused, that a tier-key
value other than the three words is refused on every layer, and that the
settings.json an ``asked`` tier renders equals the ``allowed`` one's.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile_presets import _presets_dir
from osprey.cli.build_profile_resolve import resolve_build_document, resolve_build_profile
from osprey.utils.config_writer import config_update_fields
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import tier_writes
from tests.cli.test_killswitch_on_tier import _DUMB, _FACILITY, _settings

_ALLOWED = "allowed"
_ASKED = "asked"
_NEVER = "never"


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()


# ── the spellings ────────────────────────────────────────────────────────────
#
# Each spelling is a ``config:`` fragment. A per-type leaf sits beside a
# connection key in its own block, so removing the leaf never empties the block
# (an emptied block is a separate refusal, not this matrix's subject).

SPELLINGS: dict[str, dict[str, Any]] = {
    "absent": {},
    "new-nested-allowed": {"control_system": {"writes": "allowed"}},
    "new-nested-asked": {"control_system": {"writes": "asked"}},
    "new-nested-never": {"control_system": {"writes": "never"}},
    "new-dotted-allowed": {"control_system.writes": "allowed"},
    "new-dotted-asked": {"control_system.writes": "asked"},
    "new-dotted-never": {"control_system.writes": "never"},
    "flat-nested-true": {"control_system": {"writes_enabled": True}},
    "flat-nested-false": {"control_system": {"writes_enabled": False}},
    "flat-dotted-true": {"control_system.writes_enabled": True},
    "flat-dotted-false": {"control_system.writes_enabled": False},
    "per-type-true": {
        "control_system.connector.epics.timeout_s": 5,
        "control_system.connector.epics.writes_enabled": True,
    },
    "per-type-false": {
        "control_system.connector.epics.timeout_s": 5,
        "control_system.connector.epics.writes_enabled": False,
    },
    "custom-type-true": {
        "control_system.connector": {"mypkg.Mod": {"option": 1, "writes_enabled": True}}
    },
    "custom-type-false": {
        "control_system.connector": {"mypkg.Mod": {"option": 1, "writes_enabled": False}}
    },
    "flat-false-custom-type-true": {
        "control_system.writes_enabled": False,
        "control_system.connector": {"mypkg.Mod": {"option": 1, "writes_enabled": True}},
    },
    "flat-true-per-type-false": {
        "control_system.writes_enabled": True,
        "control_system.connector.epics.timeout_s": 5,
        "control_system.connector.epics.writes_enabled": False,
    },
}

#: Documents spelling both forms, each refused wherever it sits.
MIXED: dict[str, dict[str, Any]] = {
    "new-dotted-with-flat-dotted": {
        "control_system.writes": "never",
        "control_system.writes_enabled": True,
    },
    "new-nested-with-flat-nested": {
        "control_system": {"writes": "allowed", "writes_enabled": False}
    },
    "new-nested-asked-with-flat-nested": {
        "control_system": {"writes": "asked", "writes_enabled": False}
    },
    "new-nested-with-flat-dotted": {
        "control_system": {"writes": "never"},
        "control_system.writes_enabled": True,
    },
    "new-dotted-with-per-type": {
        "control_system.writes": "allowed",
        "control_system.connector.epics.timeout_s": 5,
        "control_system.connector.epics.writes_enabled": False,
    },
    "new-dotted-asked-with-per-type": {
        "control_system.writes": "asked",
        "control_system.connector.epics.timeout_s": 5,
        "control_system.connector.epics.writes_enabled": True,
    },
    "new-dotted-with-custom-type": {
        "control_system.writes": "never",
        "control_system.connector": {"mypkg.Mod": {"option": 1, "writes_enabled": True}},
    },
}

#: Tier-key values that are none of the three words, each refused wherever it sits.
INVALID_VALUES: dict[str, Any] = {
    "bool-true": True,
    "capitalized": "Allowed",
    "capitalized-asked": "Asked",
    "legacy-switch-word": "on",
    "yes": "yes",
    "null": None,
    "list": ["allowed"],
}

#: ``--set`` edits, in the tier key's spelling and the retired ones.
SET_EDITS: dict[str, tuple[str, ...]] = {
    "none": (),
    "new-allowed": ("config.control_system.writes=allowed",),
    "new-asked": ("config.control_system.writes=asked",),
    "new-never": ("config.control_system.writes=never",),
    "flat-true": ("config.control_system.writes_enabled=true",),
    "flat-false": ("config.control_system.writes_enabled=false",),
    "per-type-true": ("config.control_system.connector.epics.writes_enabled=true",),
    "per-type-false": ("config.control_system.connector.epics.writes_enabled=false",),
    "nested-flat-true": ("config.control_system={writes_enabled: true}",),
}

MIXED_SET_EDITS: dict[str, tuple[str, ...]] = {
    "new-and-flat": (
        "config.control_system.writes=allowed",
        "config.control_system.writes_enabled=false",
    ),
    "asked-and-flat": (
        "config.control_system.writes=asked",
        "config.control_system.writes_enabled=true",
    ),
    "new-and-per-type": (
        "config.control_system.writes=never",
        "config.control_system.connector.epics.writes_enabled=true",
    ),
    "one-mapping": ("config.control_system={writes: never, writes_enabled: true}",),
}

INVALID_SET_EDITS: dict[str, tuple[str, ...]] = {
    "yes": ("config.control_system.writes=yes",),
    "bool-true": ("config.control_system.writes=true",),
    "capitalized": ("config.control_system.writes=Allowed",),
    "capitalized-asked": ("config.control_system.writes=Asked",),
    "null": ("config.control_system.writes=null",),
}


# ── the oracle ───────────────────────────────────────────────────────────────

_CS = "control_system"
_CONNECTOR = (_CS, "connector")


def _posture_leaves(config: Any) -> dict[str, Any]:
    """Every write-posture leaf a ``config:`` mapping addresses, read the way a render reads it.

    Keys split on their dots, except a direct child of the connector table,
    which is one connector type whole. Returns ``{"new": ..., "flat": ...,
    "per_type": {type: value}}`` holding only what the mapping spells.
    """
    found: dict[str, Any] = {"per_type": {}}

    def walk(node: dict, prefix: tuple[str, ...]) -> None:
        for key, value in node.items():
            parts = (key,) if prefix == _CONNECTOR else tuple(str(key).split("."))
            path = prefix + parts
            if path[0] != _CS:
                continue
            if path == (_CS, "writes"):
                found["new"] = value
            elif path == (_CS, "writes_enabled"):
                found["flat"] = value
            elif len(path) == 4 and path[:2] == _CONNECTOR and path[3] == "writes_enabled":
                found["per_type"][path[2]] = value
            elif isinstance(value, dict) and len(path) < 4:
                walk(value, path)

    if isinstance(config, dict):
        walk(config, ())
    return found


def document_answer(config: Any) -> str | None:
    """One document's answer under PR-3: its tier key, else its retired answer, else silent."""
    leaves = _posture_leaves(config)
    if "new" in leaves:
        return leaves["new"]
    retired = ([leaves["flat"]] if "flat" in leaves else []) + list(leaves["per_type"].values())
    if not retired:
        return None
    return _ALLOWED if any(value is True for value in retired) else _NEVER


def render_answer(*documents: Any) -> str:
    """A root or tier render: the nearest speaking document, else ``never``.

    *documents* run from the farthest (the ``extends`` base) to the nearest (the
    ``--set`` edit).
    """
    for config in reversed(documents):
        answer = document_answer(config)
        if answer is not None:
            return answer
    return _NEVER


def _set_config(pairs: tuple[str, ...]) -> dict[str, Any]:
    """The ``config:`` mapping a ``--set`` edit spells, parsed as YAML scalars."""
    config: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        assert key.startswith("config.")
        config[key.removeprefix("config.")] = yaml.safe_load(value)
    return config


def test_the_oracle_answers_each_retired_spelling_as_the_build_rewrites_it():
    """Sanity of the restatement itself, on spellings whose nested form is already a section.

    The retired leaves are a build-time spelling only: the build rewrites an
    armed one to ``allowed``, while the runtime ``tier_writes`` reads no retired
    leaf at all (rows ``tier-absent-old-armed-flat`` and
    ``tier-absent-old-armed-connector``, ruling D2). That gap is why every
    render must carry the tier key.
    """
    flat = {"writes_enabled": True}
    assert document_answer(SPELLINGS["flat-nested-true"]) == _ALLOWED
    assert tier_writes(flat) == _NEVER
    custom = {"connector": {"mypkg.Mod": {"writes_enabled": True}}}
    assert document_answer({"control_system": custom}) == _ALLOWED
    assert tier_writes(custom) == _NEVER
    assert document_answer(SPELLINGS["flat-false-custom-type-true"]) == _ALLOWED
    assert document_answer(SPELLINGS["flat-true-per-type-false"]) == _ALLOWED
    assert document_answer(SPELLINGS["custom-type-false"]) == _NEVER
    assert document_answer(SPELLINGS["absent"]) is None
    assert render_answer({}, {}, {}) == _NEVER


def test_the_oracle_answers_asked_only_for_the_tier_key_spelling_it():
    """``asked`` is a word only the tier key states; a retired leaf cannot reach it."""
    assert document_answer(SPELLINGS["new-nested-asked"]) == _ASKED
    assert document_answer(SPELLINGS["new-dotted-asked"]) == _ASKED
    assert tier_writes({"writes": _ASKED}) == _ASKED
    assert render_answer(SPELLINGS["new-dotted-allowed"], SPELLINGS["new-dotted-asked"]) == _ASKED
    assert render_answer(SPELLINGS["new-dotted-asked"], SPELLINGS["new-dotted-allowed"]) == _ALLOWED
    retired = [name for name in SPELLINGS if not name.startswith("new-")]
    assert retired and all(document_answer(SPELLINGS[name]) != _ASKED for name in retired)


# ── the deployment and its render ────────────────────────────────────────────

_APP_BASE = {
    "name": "Logbook Research",
    "claude_md_template": "CLAUDE.logbook.md.j2",
    "default_panel": "ariel",
    "channel_finder_mode": None,
    "hooks": ["hook-log", "target-state"],
    "rules": ["safety"],
    "skills": ["session-report"],
    "agents": [],
    "output_styles": ["control-operator"],
    "web_panels": ["ariel"],
    "mcp_servers": {},
}
_APP_CONFIG = {"claude_code.servers.controls.enabled": False, "web.ui_mode": "expert"}


def _write_yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _deployment(
    tmp_path: Path,
    root: dict[str, Any],
    *,
    base: dict[str, Any] | None = None,
    layer: dict[str, Any] | None = None,
    layer_dir: str = "tiers",
) -> Path:
    """A deployment repo; returns the file the render is built from."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True, exist_ok=True)
    root_doc: dict[str, Any] = {"name": "Facility", "data": "data", "config": dict(root)}
    if base is not None:
        _write_yaml(repo / "base.yml", {"name": "Base", "config": dict(base)})
        root_doc = {"extends": "base.yml", **root_doc}
    root_path = _write_yaml(repo / "profile.yml", root_doc)
    if layer is None:
        return root_path
    if layer_dir == "apps":
        return _write_yaml(
            repo / "apps" / "logbook.yml", {**_APP_BASE, "config": {**_APP_CONFIG, **layer}}
        )
    return _write_yaml(repo / layer_dir / "ops.yml", {"name": "Ops", "config": dict(layer)})


def _render_section(path: Path, tmp_path: Path, set_pairs: tuple[str, ...] = ()) -> dict:
    """The ``control_system`` section the build writes for the render at *path*."""
    loaded = resolve_build_document(path, None, (), set_pairs)
    return _written_section(loaded.profile.config, tmp_path)


def _written_section(config: dict[str, Any], tmp_path: Path) -> dict:
    target = tmp_path / "rendered-config.yml"
    target.write_text("{}\n", encoding="utf-8")
    config_update_fields(target, config)
    rendered = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    section = rendered.get(_CS)
    return section if isinstance(section, dict) else {}


def _retired_leaves_in(node: Any, path: str = _CS) -> list[str]:
    if not isinstance(node, dict):
        return []
    found = []
    for key, value in node.items():
        here = f"{path}.{key}"
        if key == "writes_enabled":
            found.append(here)
        found.extend(_retired_leaves_in(value, here))
    return found


def _assert_render(section: dict, expected: str) -> None:
    assert section.get("writes", _NEVER) == expected, section
    assert tier_writes(section) == expected
    assert _retired_leaves_in(section) == []


def _ids(names):
    return [f"{a}|{b}" for a, b in names]


_PAIRS = list(itertools.product(SPELLINGS, SPELLINGS))
_NAMES = list(SPELLINGS)


# ── root, extends, tier ──────────────────────────────────────────────────────


@pytest.mark.parametrize("root", _NAMES)
def test_a_root_render_is_its_own_answer_else_never(root, tmp_path):
    path = _deployment(tmp_path, SPELLINGS[root])
    _assert_render(_render_section(path, tmp_path), render_answer(SPELLINGS[root]))


@pytest.mark.parametrize(("base", "root"), _PAIRS, ids=_ids(_PAIRS))
def test_an_extends_child_answers_before_its_base(base, root, tmp_path):
    path = _deployment(tmp_path, SPELLINGS[root], base=SPELLINGS[base])
    expected = render_answer(SPELLINGS[base], SPELLINGS[root])
    _assert_render(_render_section(path, tmp_path), expected)


@pytest.mark.parametrize("layer_dir", ["tiers", "personas"])  # retired-spelling
@pytest.mark.parametrize(("root", "tier"), _PAIRS, ids=_ids(_PAIRS))
def test_a_tier_render_is_the_tiers_answer_else_the_roots(root, tier, layer_dir, tmp_path):
    path = _deployment(tmp_path, SPELLINGS[root], layer=SPELLINGS[tier], layer_dir=layer_dir)
    expected = render_answer(SPELLINGS[root], SPELLINGS[tier])
    _assert_render(_render_section(path, tmp_path), expected)


@pytest.mark.parametrize(
    "tier", ["absent", "new-dotted-never", "flat-nested-true", "per-type-false"]
)
@pytest.mark.parametrize("base", ["absent", "flat-dotted-true", "new-nested-never"])
def test_a_tier_over_an_extended_root_takes_the_nearest_answer(base, tier, tmp_path):
    path = _deployment(tmp_path, SPELLINGS["absent"], base=SPELLINGS[base], layer=SPELLINGS[tier])
    expected = render_answer(SPELLINGS[base], SPELLINGS["absent"], SPELLINGS[tier])
    _assert_render(_render_section(path, tmp_path), expected)


# ── apps ─────────────────────────────────────────────────────────────────────


def _app_refused(spelling: str) -> bool:
    """An app spelling the tier key with an arming word is refused (apps are always ``never``)."""
    return document_answer(SPELLINGS[spelling]) in (_ASKED, _ALLOWED) and (
        "new" in _posture_leaves(SPELLINGS[spelling])
    )


@pytest.mark.parametrize(("root", "app"), _PAIRS, ids=_ids(_PAIRS))
def test_an_app_render_is_never_whatever_the_root_or_app_says(root, app, tmp_path):
    path = _deployment(tmp_path, SPELLINGS[root], layer=SPELLINGS[app], layer_dir="apps")
    if _app_refused(app):
        with pytest.raises(BuildProfileError):
            resolve_build_document(path, None)
        return
    _assert_render(_render_section(path, tmp_path), _NEVER)


@pytest.mark.parametrize("edit", list(SET_EDITS))
@pytest.mark.parametrize(
    "root", ["absent", "new-dotted-allowed", "new-dotted-asked", "flat-nested-true"]
)
def test_an_app_render_stays_never_under_a_set_edit(root, edit, tmp_path):
    path = _deployment(tmp_path, SPELLINGS[root], layer={}, layer_dir="apps")
    _assert_render(_render_section(path, tmp_path, SET_EDITS[edit]), _NEVER)


# ── the --set edit ───────────────────────────────────────────────────────────

_SET_TIERS = [None, "absent", "new-nested-never", "new-dotted-allowed", "new-nested-asked"]
_SET_TIERS += ["flat-dotted-true"]
_SET_TIERS += ["per-type-false", "custom-type-true"]


@pytest.mark.parametrize("edit", list(SET_EDITS))
@pytest.mark.parametrize("tier", _SET_TIERS, ids=lambda t: f"tier={t}")
@pytest.mark.parametrize("root", _NAMES)
def test_a_set_edit_overrides_the_tier_and_the_root(root, tier, edit, tmp_path):
    layer = None if tier is None else SPELLINGS[tier]
    path = _deployment(tmp_path, SPELLINGS[root], layer=layer)
    documents = [SPELLINGS[root], *([] if layer is None else [layer])]
    expected = render_answer(*documents, _set_config(SET_EDITS[edit]))
    _assert_render(_render_section(path, tmp_path, SET_EDITS[edit]), expected)


def _preset_answer(preset: str) -> str:
    """The posture a bundled preset's own authored documents give, read raw."""
    documents = []
    path = _presets_dir() / f"{preset}.yml"
    while True:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        documents.insert(0, raw.get("config") or {})
        parent = raw.get("extends")
        if parent is None:
            break
        candidate = _presets_dir() / f"{parent}.yml"
        path = candidate if candidate.exists() else (path.parent / parent)
    return render_answer(*documents)


@pytest.mark.parametrize("edit", list(SET_EDITS))
@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_a_set_edit_on_a_preset_build_overrides_the_preset(preset, edit, tmp_path):
    profile, _ = resolve_build_profile(None, preset, set_pairs=SET_EDITS[edit])
    preset_document = {"control_system.writes": _preset_answer(preset)}
    expected = render_answer(preset_document, _set_config(SET_EDITS[edit]))
    _assert_render(_written_section(profile.config, tmp_path), expected)


# ── refusals ─────────────────────────────────────────────────────────────────

_LAYERS = ["root", "base", "tiers", "personas", "apps"]  # retired-spelling


def _deployment_with(tmp_path: Path, where: str, config: dict[str, Any]) -> Path:
    if where == "root":
        return _deployment(tmp_path, config)
    if where == "base":
        return _deployment(tmp_path, {}, base=config)
    return _deployment(tmp_path, {}, layer=config, layer_dir=where)


@pytest.mark.parametrize("mixed", list(MIXED))
@pytest.mark.parametrize("where", _LAYERS)
def test_a_document_spelling_both_forms_is_refused(where, mixed, tmp_path):
    path = _deployment_with(tmp_path, where, MIXED[mixed])
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_document(path, None)
    assert "control_system.writes" in str(excinfo.value)


@pytest.mark.parametrize("edit", list(MIXED_SET_EDITS))
@pytest.mark.parametrize("where", ["root", "tiers"])
def test_a_set_edit_spelling_both_forms_is_refused(where, edit, tmp_path):
    path = _deployment_with(tmp_path, where, {})
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_document(path, None, (), MIXED_SET_EDITS[edit])
    assert "control_system.writes" in str(excinfo.value)
    with pytest.raises(BuildProfileError):
        resolve_build_profile(None, "hello-world", set_pairs=MIXED_SET_EDITS[edit])


@pytest.mark.parametrize("dotted", [True, False], ids=["dotted", "nested"])
@pytest.mark.parametrize("value", list(INVALID_VALUES))
@pytest.mark.parametrize("where", _LAYERS)
def test_an_invalid_tier_value_is_refused_on_every_layer(where, value, dotted, tmp_path):
    raw = INVALID_VALUES[value]
    config = {"control_system.writes": raw} if dotted else {"control_system": {"writes": raw}}
    path = _deployment_with(tmp_path, where, config)
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_document(path, None)
    message = str(excinfo.value)
    assert "allowed" in message
    assert "asked" in message
    assert "never" in message


@pytest.mark.parametrize("edit", list(INVALID_SET_EDITS))
@pytest.mark.parametrize("where", ["root", "tiers", "apps"])
def test_an_invalid_tier_value_in_a_set_edit_is_refused(where, edit, tmp_path):
    path = _deployment_with(tmp_path, where, {})
    with pytest.raises(BuildProfileError):
        resolve_build_document(path, None, (), INVALID_SET_EDITS[edit])
    with pytest.raises(BuildProfileError):
        resolve_build_profile(None, "hello-world", set_pairs=INVALID_SET_EDITS[edit])


# ── settings.json: the two arming words render alike ─────────────────────────
#
# The tier is a ceiling for the switch; what prompts is decided per write at run
# time. Nothing in the rendered permissions may differ between the two words, so
# an `asked` deployment is never one regen away from a different gate set. The
# whole document is compared, not the render context, whose `writes_tier` differs
# by design; only the project's own directory, which absolute grants spell, is
# blanked.


def _settings_of(tmp_path: Path, word: str, **kwargs: Any) -> tuple[dict, str]:
    """The settings.json a *word* tier renders, and the same text with its project blanked."""
    home = tmp_path / word
    home.mkdir()
    settings = _settings(home, writes=word, **kwargs)
    project = re.escape(str(home)) + r"/killswitch-tier-\d+"
    return settings, re.sub(project, "<project>", json.dumps(settings, sort_keys=True))


@pytest.mark.parametrize(
    ("servers", "write_tools"),
    [(None, None), (_FACILITY, None), ({"dumb": _DUMB}, ["mcp__dumb__poke"])],
    ids=["shipped-servers", "facility-servers", "uncovered-write-tool"],
)
def test_settings_json_for_asked_equals_the_allowed_render(servers, write_tools, tmp_path):
    asked, asked_text = _settings_of(tmp_path, _ASKED, servers=servers, write_tools=write_tools)
    _, allowed_text = _settings_of(tmp_path, _ALLOWED, servers=servers, write_tools=write_tools)
    assert asked_text == allowed_text
    # Not two equally wrong renders: neither carries the `never` deny.
    assert "mcp__controls__channel_write" not in asked["permissions"]["deny"]
    assert "mcp__controls__channel_write" not in asked["permissions"]["ask"]
