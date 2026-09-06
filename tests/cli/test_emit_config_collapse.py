"""Config-prefix collapse for emitted profiles.

An emitted profile is meant to be edited, so it must not carry two ``config``
keys where one addresses a subtree of the other: ``config_update_fields``
applies dotted keys in iteration order, so moving such a pair's lines around
changes what gets built. The emitter folds every such pair into one key, and
rejects the pair it cannot fold.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest
import yaml

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_emit import (
    _CHANNEL_FINDER_MODE_LIST,
    _collapse_config_prefixes,
    emit_standalone_profile_yaml,
)
from osprey.cli.build_profile_presets import _presets_dir
from osprey.errors import BuildProfileError
from osprey.port_layout import DEFAULT_PORT_BASE, default_port, layout_ports

#: A landing-page port for the fixtures below, spelled as the layout lookup a
#: real config carries so no retired number rides along in a filler value.
NGINX_PORT = default_port("nginx")


def _emit(preset: str, overrides: tuple[Path, ...] = ()) -> str:
    return emit_standalone_profile_yaml(preset, overrides, (), "Emitted")


def _prefix_pairs(config: dict) -> list[tuple[str, str]]:
    """Every (shorter, deeper) pair where the shorter key's dotted SEGMENTS are
    a strict prefix of the deeper key's."""
    paths = {key: tuple(key.split(".")) for key in config}
    return [
        (shorter, deeper)
        for shorter, short_path in paths.items()
        for deeper, deep_path in paths.items()
        if len(short_path) < len(deep_path) and deep_path[: len(short_path)] == short_path
    ]


# ---------------------------------------------------------------------------
# The helper's contract
# ---------------------------------------------------------------------------


def test_deeper_key_wins_on_a_conflicting_leaf() -> None:
    """The whole point of the collapse: the deeper key's value survives, which
    is what applying the two in file order does today."""
    collapsed = _collapse_config_prefixes(
        {"modules.web_terminals": {"enabled": True, "nginx_port": NGINX_PORT}},
    )
    assert collapsed == {"modules.web_terminals": {"enabled": True, "nginx_port": NGINX_PORT}}

    collapsed = _collapse_config_prefixes(
        {
            "modules.web_terminals": {"enabled": True, "nginx_port": NGINX_PORT},
            "modules.web_terminals.enabled": False,
        }
    )

    assert collapsed == {"modules.web_terminals": {"enabled": False, "nginx_port": NGINX_PORT}}


def test_deeper_key_wins_regardless_of_declaration_order() -> None:
    """Order-independence is the property being bought — the same two keys must
    collapse the same way whichever one a facility leaves on top."""
    ordered = _collapse_config_prefixes(
        {"web.panels": {"label": "OLD"}, "web.panels.label": "NEW"},
    )
    reversed_order = _collapse_config_prefixes(
        {"web.panels.label": "NEW", "web.panels": {"label": "OLD"}},
    )

    assert ordered == reversed_order == {"web.panels": {"label": "NEW"}}


def test_deeper_key_replaces_a_list_rather_than_extending_it() -> None:
    """``config_update_fields`` sets the addressed path verbatim, so a deeper
    key overriding a list replaces it. Union-merging (what profile inheritance
    does for artifact lists) would silently keep entries the facility removed."""
    collapsed = _collapse_config_prefixes(
        {
            "modules.web_terminals": {"users": ["alice", "bob"]},
            "modules.web_terminals.users": ["carol"],
        }
    )

    assert collapsed == {"modules.web_terminals": {"users": ["carol"]}}


def test_collapse_creates_missing_intermediate_levels() -> None:
    """A deeper key may address a path the parent mapping does not have yet —
    the writer would create it, so the collapse does too."""
    collapsed = _collapse_config_prefixes(
        {"modules.web_terminals": {"enabled": True}, "modules.web_terminals.limits.cpu": 2}
    )

    assert collapsed == {"modules.web_terminals": {"enabled": True, "limits": {"cpu": 2}}}


def test_a_chain_of_three_collapses_into_the_shallowest_key() -> None:
    """Collapsing pairwise would leave a middle key that is itself a prefix —
    the guarantee is about the emitted file, so the fold has to be transitive."""
    collapsed = _collapse_config_prefixes(
        {
            "modules.web_terminals": {"enabled": True},
            "modules.web_terminals.limits": {"cpu": 1, "mem": 512},
            "modules.web_terminals.limits.cpu": 4,
        }
    )

    assert collapsed == {
        "modules.web_terminals": {"enabled": True, "limits": {"cpu": 4, "mem": 512}}
    }
    assert not _prefix_pairs(collapsed)


def test_prefix_is_by_segment_not_by_string() -> None:
    """``modules.web`` addresses nothing inside ``modules.web_terminals``:
    a string-prefix test would fold two unrelated modules into one key and
    delete a real override."""
    config = {
        "modules.web": {"enabled": True},
        "modules.web_terminals.nginx_port": NGINX_PORT,
    }

    collapsed = _collapse_config_prefixes(config)

    assert collapsed == config


def test_unrelated_keys_pass_through_in_order() -> None:
    """No pair to fold means the block is returned as written — the emitter
    must not reorder or rewrite a config a facility already reads."""
    config = {"b.two": 2, "a.one": 1, "c": {"nested": True}}

    collapsed = _collapse_config_prefixes(config)

    assert collapsed == config
    assert list(collapsed) == ["b.two", "a.one", "c"]


def test_collapse_does_not_mutate_its_input() -> None:
    """The resolved dict is also what the parity path compares against, so the
    collapse has to hand back a new tree rather than edit the one it was given."""
    config = {"modules.web_terminals": {"enabled": True}, "modules.web_terminals.enabled": False}

    _collapse_config_prefixes(config)

    assert config["modules.web_terminals"] == {"enabled": True}
    assert config["modules.web_terminals.enabled"] is False


# ---------------------------------------------------------------------------
# The pair that cannot be folded
# ---------------------------------------------------------------------------


def test_scalar_parent_is_rejected_naming_both_keys() -> None:
    """A scalar cannot carry a nested override. Left alone the pair reaches the
    build, where ``config_update_fields`` raises a bare TypeError walking into
    the scalar — so the emitter refuses it with both keys named."""
    with pytest.raises(BuildProfileError) as excinfo:
        _collapse_config_prefixes(
            {"modules.web_terminals": True, "modules.web_terminals.enabled": False}
        )

    message = str(excinfo.value)
    assert "'modules.web_terminals'" in message
    assert "'modules.web_terminals.enabled'" in message
    assert "mapping" in message


def test_scalar_below_the_shallowest_key_is_rejected_naming_the_blocking_path() -> None:
    """The blocking scalar can sit inside the parent's mapping rather than being
    the parent itself; the error names the path that actually blocks."""
    with pytest.raises(BuildProfileError) as excinfo:
        _collapse_config_prefixes(
            {
                "modules.web_terminals": {"enabled": True},
                "modules.web_terminals.enabled.strict": False,
            }
        )

    message = str(excinfo.value)
    assert "'modules.web_terminals.enabled'" in message
    assert "'modules.web_terminals.enabled.strict'" in message


def test_emit_rejects_a_scalar_parent_introduced_by_an_override(tmp_path: Path) -> None:
    """The rejection is reachable from the CLI surface, not only from the helper.

    The pair has to come from ONE layer to survive as far as the collapse: a
    key the command line states outranks whatever an inherited layer spells
    beneath it, and the deeper key is dropped before there is a pair to fold.
    An overlay writing both spellings itself is what is left, and it is the
    real shape of the mistake — one file, two statements about one path.
    """
    override = tmp_path / "o.yml"
    override.write_text(
        "config:\n  modules.web_terminals: false\n  modules.web_terminals.enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildProfileError, match="modules.web_terminals.enabled"):
        _emit("control-assistant-readonly", (override,))


def test_an_override_scalar_replaces_the_subtree_it_names(tmp_path: Path) -> None:
    """The other half of the same rule: stated alone, the scalar just wins.

    ``config_update_fields`` applies the key verbatim and would replace the
    whole subtree at render time, so the emitted profile says that outright
    instead of carrying the inherited leaves it is about to overwrite.
    """
    override = tmp_path / "o.yml"
    override.write_text("config:\n  modules.web_terminals: false\n", encoding="utf-8")

    config = yaml.safe_load(_emit("control-assistant-readonly", (override,)))["config"]

    assert config["modules.web_terminals"] is False
    assert not [key for key in config if key.startswith("modules.web_terminals.")]


# ---------------------------------------------------------------------------
# Real presets
# ---------------------------------------------------------------------------


def test_readonly_preset_emits_one_web_terminals_key_holding_enabled_false() -> None:
    """The pair that motivates the feature: the tutorial base contributes the
    whole ``modules.web_terminals`` subtree with ``enabled: true`` and the
    persona layer a dotted ``enabled: false``. Emitted uncollapsed, a facility
    reordering those two lines would re-enable the persona's web tier."""
    config = yaml.safe_load(_emit("control-assistant-readonly"))["config"]

    web_terminal_keys = [
        key for key in config if key.split(".")[:2] == ["modules", "web_terminals"]
    ]
    assert web_terminal_keys == ["modules.web_terminals"]

    subtree = config["modules.web_terminals"]
    assert subtree["enabled"] is False
    # The rest of the inherited subtree survives the fold — it is what the
    # hosting project's roster is built from.
    assert subtree["default_persona"] == "readonly"
    assert set(subtree["personas"]) == {"readonly", "readwrite", "logbook", "knowledge", "admin"}


def test_the_collapsed_key_keeps_the_section_header_it_was_holding() -> None:
    """ruamel packs the comment block introducing the NEXT section onto the key
    before it, so deleting the folded key would take the following section's
    header with it."""
    text = _emit("control-assistant-readonly")

    assert "# ── Answering webhooks (optional)" in text
    assert text.index("# ── Answering webhooks (optional)") < text.index("\ndispatch:")


@pytest.mark.parametrize("preset", list_presets())
def test_no_emitted_config_key_prefixes_another(preset: str) -> None:
    """The acceptance guard, on every bundled preset."""
    config = yaml.safe_load(_emit(preset)).get("config") or {}

    assert not _prefix_pairs(config)


@pytest.mark.parametrize("preset", list_presets())
def test_collapsing_an_emitted_config_is_a_no_op(preset: str) -> None:
    """Idempotence — a second emission (or task 4.3 running the collapse before
    folding the web_terminals subtree) must find nothing left to do."""
    config = yaml.safe_load(_emit(preset)).get("config") or {}

    assert _collapse_config_prefixes(config) == config


# ---------------------------------------------------------------------------
# What an emitted comment may say
# ---------------------------------------------------------------------------


def _comment_lines(text: str) -> list[str]:
    """Every comment line of an emitted profile, whole-line and trailing alike."""
    lines: list[str] = []
    for line in text.splitlines():
        if "#" in line:
            lines.append(line[line.index("#") :].strip())
    return lines


@functools.cache
def _preset_comment_lines() -> frozenset[str]:
    """Comment lines the preset FILES contribute, verbatim.

    The emitter copies a preset's own comments onto the keys it emits, so a
    line already present in a preset file is that file's to answer for and the
    presets are checked on their own terms. What is left is what the EMITTER
    writes, which is what the two assertions below are about.
    """
    return frozenset(
        line.strip()
        for path in _presets_dir().glob("*.yml")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("#")
    )


def _emitter_comment_lines(preset: str) -> list[str]:
    from_presets = _preset_comment_lines()
    return [line for line in _comment_lines(_emit(preset)) if line not in from_presets]


@pytest.mark.parametrize("preset", list_presets())
def test_no_emitted_comment_spells_a_layout_port(preset: str) -> None:
    """A port number in a comment is wrong for every deployment that moved.

    Every framework port is a slot of the deployment's own block, so a number
    written into a comment describes the default base and nothing else — and a
    profile that sets `deployment.port_base` then carries a comment that lies
    about its own ports. The comments name the slot instead.
    """
    numbers = {str(port) for port in layout_ports(DEFAULT_PORT_BASE).values()}

    offenders = [
        line for line in _emitter_comment_lines(preset) if numbers & set(re.findall(r"\d+", line))
    ]

    assert not offenders


@pytest.mark.parametrize("preset", list_presets())
def test_no_emitted_comment_freezes_a_channel_finder_mode(preset: str) -> None:
    """Mode names appear only where the emitter renders them from the constant.

    The paradigms this release accepts are ``VALID_CHANNEL_FINDER_MODES``; a
    name typed into a comment keeps describing the release it was written in.
    The one place they are named is the enumeration the emitter builds from
    that constant, so that fragment is removed before the line is scanned.

    A mode reaches a comment in one of two shapes, and both are what this
    looks for: written as a VALUE (``# channel_finder_mode: hierarchical`` —
    the commented example that used to pin one), or spelled as an identifier no
    English sentence produces (``in_context``). Two of the four names are
    ordinary words — a knowledge *graph* and a *hierarchical* menu are prose,
    not settings — so a bare occurrence of those is not evidence of anything.
    """
    as_a_value = {f": {mode}" for mode in VALID_CHANNEL_FINDER_MODES}
    as_an_identifier = {mode for mode in VALID_CHANNEL_FINDER_MODES if "_" in mode}

    offenders = []
    for line in _emitter_comment_lines(preset):
        scrubbed = line.replace(_CHANNEL_FINDER_MODE_LIST, "")
        if any(spelling in scrubbed for spelling in as_a_value):
            offenders.append(line)
        elif as_an_identifier & set(re.findall(r"[A-Za-z_]+", scrubbed)):
            offenders.append(line)

    assert not offenders
