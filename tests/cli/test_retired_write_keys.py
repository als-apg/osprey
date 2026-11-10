"""The retired write-posture keys are rewritten to the tier key at the single parse point.

``control_system.writes_enabled`` and every
``control_system.connector.<type>.writes_enabled`` read, for one release
window, as ``control_system.writes: allowed | never``: ``allowed`` when any
retired leaf is a literal ``True``, else ``never`` (PR-3), and always ``never``
in an app. The oracle here is ``tier_writes`` of the section the *render*
writes from the raw, un-normalized document — the reading every consumer gave
the old keys — never the normalizer itself. A silent document is untouched; a
document spelling both the retired keys and the tier key is refused; the kind
of a document comes from where its file lives; and the leaves a parse removed
are recorded per resolved path (C-ORIGIN).
"""

from __future__ import annotations

import copy
import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_document import (
    _parse_profile_document,
    _read_profile_document,
    normalize_retired_write_keys,
    retired_write_answer,
    retired_write_origin,
)
from osprey.cli.build_profile_merge import _deep_merge, _resolve_extends
from osprey.cli.build_profile_presets import _presets_dir
from osprey.utils.config_writer import config_update_fields
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import tier_writes

_LOGGER_NAME = doc.__name__
_REMEDY = "run `osprey init --revalidate`"


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()


def _parse(text: str, source: str = "profile.yml", path: Path | None = None):
    return _parse_profile_document(textwrap.dedent(text), source, path=path)


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]


def _render(config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """What the build writes for *config*: each key applied in order, as the render does."""
    target = tmp_path / f"render-{len(list(tmp_path.glob('render-*')))}.yml"
    target.write_text("{}\n", encoding="utf-8")
    config_update_fields(target, config)
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def _section(config: dict[str, Any], tmp_path: Path) -> Any:
    return _render(config, tmp_path).get("control_system")


def _retired_leaves_in(section: Any) -> list[str]:
    if not isinstance(section, dict):
        return []
    found = ["writes_enabled"] if "writes_enabled" in section else []
    connector = section.get("connector")
    if isinstance(connector, dict):
        found += [
            f"connector.{name}.writes_enabled"
            for name, block in connector.items()
            if isinstance(block, dict) and "writes_enabled" in block
        ]
    return found


# ── the case table ───────────────────────────────────────────────────────────

# (id, config mapping, expected tier). The expected value is PR-3's rule read by
# hand off the row; the test also checks it against `tier_writes` of the render.
CASES: list[tuple[str, dict[str, Any], str]] = [
    ("flat-true-dotted", {"control_system.writes_enabled": True}, "allowed"),
    ("flat-true-nested", {"control_system": {"writes_enabled": True}}, "allowed"),
    ("flat-false-dotted", {"control_system.writes_enabled": False}, "never"),
    ("flat-false-nested", {"control_system": {"type": "epics", "writes_enabled": False}}, "never"),
    ("flat-quoted-true", {"control_system.writes_enabled": "true"}, "never"),
    ("flat-one", {"control_system.writes_enabled": 1}, "never"),
    ("flat-null", {"control_system.writes_enabled": None}, "never"),
    (
        "type-true-dotted",
        {"control_system.connector.virtual_accelerator.writes_enabled": True},
        "allowed",
    ),
    ("type-false-dotted", {"control_system.connector.epics.writes_enabled": False}, "never"),
    (
        "type-true-nested",
        {"control_system": {"connector": {"virtual_accelerator": {"writes_enabled": True}}}},
        "allowed",
    ),
    (
        "type-true-partly-nested",
        {"control_system.connector": {"virtual_accelerator": {"writes_enabled": True, "port": 1}}},
        "allowed",
    ),
    (
        "flat-false-type-true-dotted",
        {
            "control_system.writes_enabled": False,
            "control_system.connector.epics.writes_enabled": False,
            "control_system.connector.virtual_accelerator.writes_enabled": True,
        },
        "allowed",
    ),
    (
        "flat-true-type-false-nested",
        {
            "control_system": {
                "writes_enabled": True,
                "connector": {"epics": {"writes_enabled": False}},
            }
        },
        "allowed",
    ),
    (
        "flat-false-type-false-nested",
        {
            "control_system": {
                "writes_enabled": False,
                "connector": {"epics": {"writes_enabled": False}, "va": {"writes_enabled": None}},
            }
        },
        "never",
    ),
    (
        "mixed-spelling-flat-dotted-type-nested",
        {
            "control_system.writes_enabled": False,
            "control_system": {"connector": {"va": {"writes_enabled": True}}},
        },
        "allowed",
    ),
    (
        "mixed-spelling-flat-nested-type-dotted",
        {
            "control_system": {"type": "epics", "writes_enabled": False},
            "control_system.connector.epics.writes_enabled": True,
        },
        "allowed",
    ),
    (
        "mixed-spelling-cs-nested-connector-dotted",
        {"control_system": {"connector.va.writes_enabled": True, "writes_enabled": False}},
        "allowed",
    ),
    (
        "flat-spelled-twice-last-wins",
        {"control_system.writes_enabled": True, "control_system": {"writes_enabled": False}},
        "never",
    ),
    # A nested block after a dotted flat leaf: every spelling is read together.
    (
        "flat-dotted-before-nested-block",
        {"control_system.writes_enabled": True, "control_system": {"type": "epics"}},
        "allowed",
    ),
    # BE-5: a dotted custom connector type is ONE key of the connector table.
    (
        "dotted-type-mapping-armed-under-flat-false",
        {
            "control_system.writes_enabled": False,
            "control_system": {"connector": {"mypkg.Mod": {"writes_enabled": True}}},
        },
        "allowed",
    ),
    (
        "dotted-type-mapping-unarmed-under-flat-false",
        {
            "control_system.writes_enabled": False,
            "control_system": {"connector": {"mypkg.Mod": {"writes_enabled": False}}},
        },
        "never",
    ),
    (
        "dotted-type-in-connector-value-armed",
        {
            "control_system.writes_enabled": False,
            "control_system.connector": {"mypkg.Mod": {"writes_enabled": True}},
        },
        "allowed",
    ),
    # A dotted KEY running through the connector table is split all the way by
    # the render: `mypkg` is the type and `Mod.writes_enabled` arms nothing.
    (
        "dotted-key-through-dotted-type-is-no-type-leaf",
        {
            "control_system.writes_enabled": False,
            "control_system.connector.mypkg.Mod.writes_enabled": True,
        },
        "never",
    ),
]


def _expanded_section(config: dict[str, Any]) -> Any:
    """``control_system`` as every spelling of *config* addresses it (dotted keys folded in)."""
    return _expand_dotted(config).get("control_system")


def _has_dotted_connector_type(config: dict[str, Any]) -> bool:
    """Whether a row spells the dotted custom connector type ``mypkg.Mod``."""
    return "mypkg" in repr(config)


@pytest.mark.parametrize(("config", "expected"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_every_retired_document_reads_as_its_tier(config, expected, tmp_path):
    raw = {"config": copy.deepcopy(config)}
    # The oracle: PR-3's rule, read by hand off the row, and what the migration's
    # own retired-leaf reader makes of the expanded nested section (the run-time
    # `tier_writes` no longer reads the retired leaves: an unrebuilt render is
    # `never`). `_expand_dotted` splits a dotted connector type, which the
    # retired reader never does (BE-5), so those rows rest on the hand-read
    # value alone.
    if not _has_dotted_connector_type(config):
        assert doc._retired_tier(_expanded_section(config)) == expected
    assert retired_write_answer(config) == expected

    normalize_retired_write_keys(raw, "profile.yml")
    rendered = _section(raw["config"], tmp_path)
    assert rendered["writes"] == expected
    assert _retired_leaves_in(rendered) == []
    # The rewrite is what a new-style author writes: the render agrees with itself.
    assert tier_writes(rendered) == expected


def test_retired_write_answer_is_pure_and_none_for_a_silent_mapping():
    config = {"control_system.writes_enabled": True}
    before = copy.deepcopy(config)
    assert retired_write_answer(config) == "allowed"
    assert config == before
    assert retired_write_answer({"control_system": {"type": "epics"}}) is None
    assert retired_write_answer({}) is None
    assert retired_write_answer(None) is None
    # The tier key is not a retired leaf.
    assert retired_write_answer({"control_system.writes": "allowed"}) is None


#: The previous release's shipped tier documents, which still spell the retired
#: keys, and the bundled login that succeeds each one.
_PREVIOUS_TIER_LAYERS = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "previous_release_repo"
    / "personas"  # retired-spelling
)
_SUCCESSORS = {
    "admin": "tiers/admin.yml",
    "readwrite": "tiers/readwrite.yml",
    "readonly": "tiers/readonly.yml",
    "knowledge": "tiers/knowledge.yml",
    "logbook": "tiers/logbook.yml",
}
#: The successors that are shared logins, sealed ``never``.
_SHARED_SUCCESSORS = ("knowledge", "logbook")

#: The one successor that deliberately states a lower ceiling than its retired
#: predecessor reads as: the read-write tier ships `asked` (FR4), so its logins
#: keep today's prompts, while an un-revalidated `writes_enabled: true` reads as
#: the top word, `allowed`. Every other successor states the predecessor's word.
_LOWERED_SUCCESSORS = {"readwrite": "asked"}


def test_every_previous_release_document_reads_as_its_bundled_successor(tmp_path):
    """Every shipped document spelling the old keys reads as the old render did,
    and the bundled document that succeeds it states that same answer in the new key."""
    presets = _presets_dir()
    checked = 0
    for stem, successor in sorted(_SUCCESSORS.items()):
        path = _PREVIOUS_TIER_LAYERS / f"{stem}.yml"
        old = yaml.safe_load(path.read_text(encoding="utf-8"))
        answer = retired_write_answer(old["config"])
        assert answer is not None, path
        assert answer == doc._retired_tier(_expanded_section(old["config"])), path
        rendered = _section(_read_profile_document(path)["config"], tmp_path)
        assert rendered["writes"] == answer, path
        assert _retired_leaves_in(rendered) == [], path

        new_path = presets / successor
        new = yaml.safe_load(new_path.read_text(encoding="utf-8"))
        assert retired_write_answer(new["config"]) is None, new_path
        successor_answer = _LOWERED_SUCCESSORS.get(stem, answer)
        if stem in _LOWERED_SUCCESSORS:
            assert answer == "allowed", path
        assert new["config"]["control_system.writes"] == successor_answer, new_path
        if stem in _SHARED_SUCCESSORS:
            assert answer == "never", new_path
        new_rendered = _section(_read_profile_document(new_path)["config"], tmp_path)
        assert new_rendered["writes"] == successor_answer, new_path
        assert _retired_leaves_in(new_rendered) == [], new_path
        checked += 1
    assert checked == len(_SUCCESSORS)


def test_every_bundled_preset_agrees_with_tier_writes_of_its_raw_render(tmp_path):
    """No shipped document spells a retired leaf; each states the tier key, and
    its render agrees with ``tier_writes`` of the raw document."""
    checked = 0
    for path in sorted(_presets_dir().rglob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
            continue
        assert retired_write_answer(raw["config"]) is None, path
        assert raw["config"].get("control_system.writes") in ("allowed", "asked", "never"), path
        rendered = _section(_read_profile_document(path)["config"], tmp_path)
        if path.parent.name == "apps":
            assert rendered["writes"] == "never", path
        assert rendered["writes"] == tier_writes(_expanded_section(raw["config"])), path
        assert _retired_leaves_in(rendered) == [], path
        checked += 1
    assert checked > 0


# ── scope: silent documents ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "name: x\n",
        "config:\n  control_system.type: epics\n",
        "config:\n  control_system:\n    connector:\n      epics: {port: 1}\n",
        "config:\n  control_system.writes: never\n",
        "config: not-a-mapping\n",
    ],
)
def test_a_silent_document_is_untouched_and_quiet(text, caplog, tmp_path):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    path = tmp_path / "profile.yml"
    path.write_text(text, encoding="utf-8")
    assert _read_profile_document(path) == yaml.safe_load(text)
    assert _warnings(caplog) == []
    assert retired_write_origin(path) is None


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


_ALLOWED_ROOT = """\
    name: root
    config:
      control_system.type: epics
      control_system.writes_enabled: true
"""


@pytest.mark.parametrize(
    ("rel", "text"),
    [
        ("tiers/silent.yml", "config:\n  system.timezone: UTC\n"),
        ("overlay.yml", "config:\n  system.timezone: UTC\n"),
        ("tiers/empty.yml", "name: t\n"),
    ],
)
def test_a_silent_layer_under_an_allowed_root_stays_allowed(rel, text, tmp_path):
    root = _read_profile_document(_write(tmp_path / "profile.yml", _ALLOWED_ROOT))
    layer = _read_profile_document(_write(tmp_path / rel, text))
    merged = _deep_merge(root, layer)
    assert _section(merged["config"], tmp_path)["writes"] == "allowed"


def test_a_silent_extends_child_under_an_allowed_root_stays_allowed(tmp_path):
    _write(tmp_path / "base.yml", _ALLOWED_ROOT)
    child_path = _write(
        tmp_path / "child" / "profile.yml",
        "extends: ../base.yml\nconfig:\n  system.timezone: UTC\n",
    )
    child = _read_profile_document(child_path)
    resolved = _resolve_extends(child, child_path.resolve())
    assert _section(resolved["config"], tmp_path)["writes"] == "allowed"


# ── output spelling (BE-15): the tier key stands where the retired leaf stood ─


def test_the_tier_key_replaces_the_flat_leaf_in_place_and_spelling():
    raw = _parse(
        """\
        config:
          a: 1
          control_system.writes_enabled: true
          b: 2
        """
    )
    assert list(raw["config"].items()) == [("a", 1), ("control_system.writes", "allowed"), ("b", 2)]
    raw = _parse(
        """\
        config:
          control_system:
            type: epics
            writes_enabled: false
            port: 3
        """
    )
    assert list(raw["config"]["control_system"].items()) == [
        ("type", "epics"),
        ("writes", "never"),
        ("port", 3),
    ]


def test_per_type_only_writes_the_tier_key_in_the_spelling_of_its_control_system_block():
    nested = _parse(
        """\
        config:
          control_system:
            type: epics
            connector:
              va: {writes_enabled: true, port: 1}
        """
    )
    assert nested["config"] == {
        "control_system": {"type": "epics", "connector": {"va": {"port": 1}}, "writes": "allowed"}
    }
    dotted = _parse(
        """\
        config:
          a: 1
          control_system.connector.va.writes_enabled: true
          control_system.connector.epics.writes_enabled: false
          b: 2
        """
    )
    assert list(dotted["config"].items()) == [
        ("a", 1),
        ("control_system.writes", "allowed"),
        ("b", 2),
    ]


def test_a_connector_block_the_rewrite_empties_is_left_for_the_render_checks():
    raw = _parse("config:\n  control_system:\n    connector:\n      va: {writes_enabled: true}\n")
    assert raw["config"]["control_system"]["connector"] == {"va": {}}


_ROOT_SPELLINGS = {
    "nested": "config:\n  control_system:\n    type: epics\n    writes_enabled: true\n",
    "dotted": "config:\n  control_system.type: epics\n  control_system.writes_enabled: true\n",
    "nested-per-type": (
        "config:\n  control_system:\n    type: epics\n"
        "    connector:\n      va: {writes_enabled: true}\n"
    ),
    "dotted-per-type": (
        "config:\n  control_system.type: epics\n"
        "  control_system.connector.va.writes_enabled: true\n"
    ),
}
_TIER_SPELLINGS = {
    "nested": "config:\n  control_system:\n    writes: never\n",
    "dotted": "config:\n  control_system.writes: never\n",
}


@pytest.mark.parametrize("root_spelling", sorted(_ROOT_SPELLINGS))
@pytest.mark.parametrize("tier_spelling", sorted(_TIER_SPELLINGS))
def test_a_never_tier_overrides_a_rewritten_allowed_root_in_every_spelling(
    root_spelling, tier_spelling, tmp_path
):
    root = _read_profile_document(_write(tmp_path / "profile.yml", _ROOT_SPELLINGS[root_spelling]))
    assert _section(root["config"], tmp_path)["writes"] == "allowed"
    tier = _read_profile_document(
        _write(tmp_path / "tiers" / "ro.yml", _TIER_SPELLINGS[tier_spelling])
    )
    merged = _deep_merge(root, tier)
    assert _section(merged["config"], tmp_path)["writes"] == "never"


# ── shared logins ────────────────────────────────────────────────────────────


def _shared_login(tmp_path: Path, text: str) -> Path:
    """A repo whose ``shared:`` names ``tiers/logbook.yml``, a `lists: own` login."""
    _write(
        tmp_path / "profile.yml",
        "name: Facility\nconfig:\n  modules.web_terminals.shared: [logbook]\n",
    )
    return _write(tmp_path / "tiers" / "logbook.yml", "name: Logbook\nlists: own\n" + text)


def _render_shared(path: Path) -> dict[str, Any]:
    from osprey.cli.build_profile_merge import resolve_profile_document

    return resolve_profile_document(_read_profile_document(path), path, warn=False).raw


@pytest.mark.parametrize(
    "text",
    [
        "config:\n  control_system.writes_enabled: true\n",
        "config:\n  control_system.writes_enabled: false\n",
        "config:\n  control_system.connector.va.writes_enabled: true\n",
        "config:\n  control_system:\n    writes_enabled: true\n",
    ],
)
def test_a_shared_login_renders_never_whatever_its_retired_keys_say(text, tmp_path):
    rendered = _section(_render_shared(_shared_login(tmp_path, text))["config"], tmp_path)
    assert rendered["writes"] == "never"
    assert _retired_leaves_in(rendered) == []


def test_an_arming_retired_key_in_a_shared_login_warns_once_that_it_arms_nothing(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    path = _shared_login(
        tmp_path,
        "config:\n  control_system.writes_enabled: true\n"
        "  control_system.connector.epics.writes_enabled: false\n",
    )
    _render_shared(path)
    _render_shared(path)
    armed = [r.getMessage() for r in caplog.records if "arm nothing" in r.getMessage()]
    assert len(armed) == 1, armed
    assert "modules.web_terminals.shared" in armed[0]
    assert "control_system.writes: never" in armed[0]


def test_a_shared_login_authoring_allowed_is_refused_naming_shared(tmp_path):
    path = _shared_login(tmp_path, "config:\n  control_system.writes: allowed\n")
    with pytest.raises(BuildProfileError, match="modules.web_terminals.shared"):
        _render_shared(path)


# ── kind comes from the path (BE-14) ─────────────────────────────────────────


def test_a_root_under_an_apps_directory_is_still_a_root(tmp_path):
    path = _write(
        tmp_path / "apps" / "repo" / "profile.yml",
        "config:\n  control_system.writes_enabled: true\n",
    )
    assert _read_profile_document(path)["config"] == {"control_system.writes": "allowed"}


def test_the_source_label_never_decides_the_kind(tmp_path):
    # A display label that names an app does not make a root file an app...
    root = _write(tmp_path / "profile.yml", "config:\n  control_system.writes_enabled: true\n")
    assert _read_profile_document(root, "preset 'apps/logbook'")["config"] == {
        "control_system.writes": "allowed"
    }
    # ...and a text-only document is a root whatever it is called.
    text = _parse("config:\n  control_system.writes_enabled: true\n", "preset 'apps/logbook'")
    assert text["config"] == {"control_system.writes": "allowed"}


@pytest.mark.parametrize(
    ("dirname", "expected"),
    [
        ("tiers", "allowed"),
        ("personas", "allowed"),  # retired-spelling
    ],
)
def test_tiers_and_retired_layers_read_their_own_answer(dirname, expected, tmp_path):
    path = _write(
        tmp_path / dirname / "rw.yml",
        "config:\n  control_system.connector.va.writes_enabled: true\n",
    )
    assert _read_profile_document(path)["config"] == {"control_system.writes": expected}


# ── mixed spellings are refused ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "config",
    [
        "  control_system.writes_enabled: true\n  control_system.writes: allowed\n",
        "  control_system:\n    writes_enabled: false\n    writes: never\n",
        "  control_system.writes: never\n  control_system:\n    writes_enabled: true\n",
        "  control_system.connector.va.writes_enabled: true\n  control_system:\n    writes: never\n",
        "  control_system:\n    connector:\n      va: {writes_enabled: false}\n  control_system.writes: never\n",
    ],
)
def test_a_document_spelling_both_keys_is_refused_naming_the_source(config, tmp_path):
    path = _write(tmp_path / "profile.yml", "config:\n" + config)
    with pytest.raises(BuildProfileError) as excinfo:
        _read_profile_document(path, "my-profile.yml")
    message = str(excinfo.value)
    assert "my-profile.yml" in message
    assert "`control_system.writes`" in message
    assert "writes_enabled" in message
    assert retired_write_origin(path) is None


# ── warnings ─────────────────────────────────────────────────────────────────


def test_one_warning_per_retired_key_naming_the_tier_key_and_revalidate(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    _parse(
        """\
        config:
          control_system.writes_enabled: false
          control_system.connector.epics.writes_enabled: false
          control_system.connector.va.writes_enabled: true
        """,
        "rw.yml",
    )
    lines = _warnings(caplog)
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("rw.yml: ")
        assert "`control_system.writes: allowed`" in line
        assert line.endswith(_REMEDY)
    assert any("`control_system.connector.va.writes_enabled`" in line for line in lines)


def test_repeated_parse_of_one_source_warns_once(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    for _ in range(3):
        _parse("config:\n  control_system.writes_enabled: true\n", "p.yml")
    assert len(_warnings(caplog)) == 1
    _parse("config:\n  control_system.writes_enabled: true\n", "other.yml")
    assert len(_warnings(caplog)) == 2


# ── C-ORIGIN ─────────────────────────────────────────────────────────────────


def test_the_origin_records_the_removed_leaves_by_resolved_path(tmp_path):
    path = _write(
        tmp_path / "tiers" / "rw.yml",
        "config:\n  control_system.writes_enabled: false\n"
        "  control_system:\n    connector:\n      mypkg.Mod: {writes_enabled: true}\n",
    )
    _read_profile_document(path, "a display label")
    expected = {"writes_enabled": False, "connector": {"mypkg.Mod": {"writes_enabled": True}}}
    assert retired_write_origin(path) == expected
    # Keyed by the resolved path, never by the label it was read under.
    assert retired_write_origin(tmp_path / "tiers" / ".." / "tiers" / "rw.yml") == expected
    # Read-only to its consumers.
    retired_write_origin(path)["writes_enabled"] = True
    assert retired_write_origin(path) == expected


def test_a_text_only_parse_records_no_origin():
    _parse("config:\n  control_system.writes_enabled: true\n")
    assert doc._RETIRED_WRITE_ORIGINS == {}


def test_a_reparse_without_retired_leaves_deletes_the_origin(tmp_path):
    path = _write(tmp_path / "profile.yml", "config:\n  control_system.writes_enabled: true\n")
    _read_profile_document(path)
    assert retired_write_origin(path) == {"writes_enabled": True}
    path.write_text("config:\n  control_system.writes: allowed\n", encoding="utf-8")
    _read_profile_document(path)
    assert retired_write_origin(path) is None
    # A non-mapping rewrite clears it too.
    _read_profile_document(_write(path, "config:\n  control_system.writes_enabled: true\n"))
    path.write_text("- a list\n", encoding="utf-8")
    _read_profile_document(path)
    assert retired_write_origin(path) is None


def test_the_normalizer_module_stays_a_leaf():
    """No parse-time version compare: the sunset is a test tripwire (C-SUNSET)."""
    imports = [
        line
        for line in Path(doc.__file__).read_text(encoding="utf-8").splitlines()
        if line.startswith(("from ", "import ")) or line.lstrip().startswith(("from .", "import "))
    ]
    assert not [line for line in imports if "build_profile_load" in line]
