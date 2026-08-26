"""Unit tests for the profile emitters in ``osprey.cli.build_profile_emit``.

Covers what the emitted TEXT is, at the level of the emitter itself — the
``osprey init`` command that drives it, and everything it writes around
the text, is pinned by ``test_init_verb.py`` and
``test_persona_profile_emission.py``.

A persona file is a pure delta (FR-10): the persona preset's own layer, minus
the ``extends:`` key, retitled. Nothing is added and nothing else is subtracted,
because the file is merged over the ``profile.yml`` it sits beside and any extra
key here would be a second, divergent copy of a host setting.

A host ``profile.yml`` additionally carries ``provenance:`` (FR-6) — the source
preset and its content hash, as data a later build can read rather than as a
comment it would have to parse.
"""

from __future__ import annotations

import pytest
import yaml

from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.build_profile_emit import emit_persona_delta_yaml, emit_standalone_profile_yaml
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.build_profile_presets import _load_preset_raw
from osprey.errors import BuildProfileError

# A bundled persona preset and the host preset it extends.
PERSONA_PRESET = "control-assistant-readonly"
HOST_PRESET = "control-assistant"

PERSONA_PRESETS = (
    "control-assistant-readonly",
    "control-assistant-readwrite",
)


def _delta(preset: str = PERSONA_PRESET, name: str = "My Profile (readonly)") -> dict:
    text = emit_persona_delta_yaml(
        preset_name=preset,
        profile_name=name,
        profile_filename=f"my-profile/personas/{preset}.yml",
    )
    return yaml.safe_load(text)


@pytest.mark.parametrize("preset", PERSONA_PRESETS)
def test_delta_content_is_the_preset_layer_minus_extends(preset: str) -> None:
    """The one contract: emitted keys == the preset's own raw layer, with
    ``extends`` gone and ``name`` retitled. A key that appeared here without
    being in the preset would be a host setting frozen into the persona."""
    raw, _path = _load_preset_raw(preset)
    expected = {key: value for key, value in raw.items() if key != "extends"}
    expected["name"] = "Retitled"

    assert _delta(preset, "Retitled") == expected


@pytest.mark.parametrize("preset", PERSONA_PRESETS)
def test_delta_carries_no_extends(preset: str) -> None:
    """Position IS the inheritance — a written ``extends:`` inside
    ``personas/`` is rejected at build time, so emitting one would be a bug."""
    text = emit_persona_delta_yaml(
        preset_name=preset,
        profile_name="Persona",
        profile_filename="my-profile/personas/p.yml",
    )

    assert "extends" not in _delta(preset)
    assert not any(line.startswith("extends:") for line in text.splitlines()), text


def test_delta_does_not_re_anchor_profile_relative_paths() -> None:
    """``data:``, the trigger config, and the convention dirs all anchor at the
    host profile's directory, so a delta that named them itself would fight the
    merge rather than help it."""
    parsed = _delta()

    assert "data" not in parsed
    assert "dispatch" not in parsed


def test_delta_keeps_the_preset_comments() -> None:
    """The persona presets explain each override they make; that prose is the
    reason the delta reuses the preset's text instead of re-dumping the dict."""
    text = emit_persona_delta_yaml(
        preset_name=PERSONA_PRESET,
        profile_name="Persona",
        profile_filename="my-profile/personas/readonly.yml",
    )

    unwrapped = _unwrapped_comments(text)
    assert "it takes three keys rather than one" in unwrapped
    assert "Pared-down operator layout" in unwrapped


def _unwrapped_comments(text: str, *, header_only: bool = False) -> str:
    """Every comment line joined into one, so an assertion can be a sentence.

    The prose is hard-wrapped, and where a sentence breaks moves whenever it is
    edited. Asserting on the unwrapped text is what keeps these checks about
    what the emitted file SAYS rather than about where it happens to wrap.
    """
    block = text.split("\n\n", 1)[0] if header_only else text
    return " ".join(
        line.strip().lstrip("#").strip()
        for line in block.splitlines()
        if line.strip().startswith("#")
    )


def test_header_explains_the_implicit_merge_and_names_the_source() -> None:
    text = emit_persona_delta_yaml(
        preset_name=PERSONA_PRESET,
        profile_name="Persona",
        profile_filename="my-profile/personas/readonly.yml",
    )

    assert "settings for one web login" in text
    assert "Only the differences from profile.yml belong here" in _unwrapped_comments(text)
    assert "merges this file over that one" in _unwrapped_comments(text)
    assert "preset content hash: sha256:" in text
    assert "osprey validate my-profile/personas/readonly.yml" in text


def test_preset_without_extends_cannot_be_emitted_as_a_delta() -> None:
    """A preset that is not written against a base has no delta to emit —
    dropping its ``extends`` would silently discard the base it assumes."""
    with pytest.raises(BuildProfileError, match="declares no extends chain"):
        emit_persona_delta_yaml(
            preset_name=HOST_PRESET,
            profile_name="Persona",
            profile_filename="my-profile/personas/p.yml",
        )


# ---------------------------------------------------------------------------
# Provenance (FR-6): machine-readable, on every emitted host profile
# ---------------------------------------------------------------------------


def _standalone(preset: str) -> str:
    return emit_standalone_profile_yaml(preset, (), (), "Emitted")


@pytest.mark.parametrize("preset", list_presets())
def test_provenance_names_the_preset_and_its_hash(preset: str) -> None:
    """Every emitted profile records what it came from, for every preset — an
    emission with no provenance is one a later build cannot check for drift."""
    provenance = yaml.safe_load(_standalone(preset))["provenance"]

    assert provenance["preset"] == preset
    assert provenance["preset_hash"] == compute_preset_hash(preset)
    assert provenance["preset_hash"].startswith("sha256:")


def test_provenance_survives_a_resolution_round_trip(tmp_path) -> None:
    """The key is only useful if the loader keeps it: written as YAML, read back
    as a ``ProfileProvenance``, unchanged."""
    profile_file = tmp_path / "profile.yml"
    profile_file.write_text(_standalone(HOST_PRESET), encoding="utf-8")

    resolved, _dir = resolve_build_profile(profile_file, None)

    assert resolved.provenance is not None
    assert resolved.provenance.preset == HOST_PRESET
    assert resolved.provenance.preset_hash == compute_preset_hash(HOST_PRESET)


def test_provenance_agrees_with_the_header_prose() -> None:
    """The header says the same thing for people. Both come from the same two
    values here, so a reader and a build can never be told different stories."""
    text = _standalone(HOST_PRESET)
    provenance = yaml.safe_load(text)["provenance"]
    header = text.split("\nname:")[0]

    assert f"bundled `{provenance['preset']}` preset" in header
    assert f"preset content hash: {provenance['preset_hash']}" in header


@pytest.mark.parametrize("preset", PERSONA_PRESETS)
def test_persona_deltas_carry_no_provenance_key(preset: str) -> None:
    """A delta is the preset's layer and nothing else, so its provenance stays
    in the header prose — a key here would be merged into the host's resolved
    profile and overwrite the host's own record."""
    assert "provenance" not in _delta(preset)
