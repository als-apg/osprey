"""Unit tests for the profile emitters in ``osprey.cli.build_profile_emit``.

Covers what the emitted TEXT is, at the level of the emitter itself — the
``osprey init`` command that drives it, and everything it writes around
the text, is pinned by ``test_init_verb.py`` and
``test_layer_emission.py``.

A tier or app file is the bundled layer materialized explicitly: the bundled
file's own keys and comments, under a header saying what the file is, with a
``provenance:`` block naming what it was made from. Nothing else is added and
nothing is subtracted, and no ``extends:`` or other pointer into the package is
written, because the repo is the whole source of what a deployment runs.

A host ``profile.yml`` carries ``provenance:`` too (FR-6) — the source preset
and its content hash, as data a later build can read rather than as a comment
it would have to parse.
"""

from __future__ import annotations

import pytest
import yaml

from osprey.cli import build_profile_emit
from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.build_profile_emit import emit_layer_delta_yaml, emit_standalone_profile_yaml
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.build_profile_presets import _load_preset_raw, list_tiers
from osprey_connectors.errors import BuildProfileError

# A bundled tier and the host preset whose deployments carry it.
RENDER_PRESET = "tiers/readonly"
HOST_PRESET = "control-assistant"

#: Every bundled layer, kind-qualified the way the emitter names it.
RENDER_PRESETS = tuple(f"tiers/{name}" for name in list_tiers())


def _delta(preset: str = RENDER_PRESET, name: str = "My Profile") -> dict:
    kind, _sep, stem = preset.partition("/")
    text = emit_layer_delta_yaml(
        preset_name=preset,
        profile_name=name,
        profile_filename=f"my-profile/{kind}/{stem}.yml",
    )
    return yaml.safe_load(text)


def test_the_bundled_layers_are_listed() -> None:
    """Guards the parametrizations below against running over nothing."""
    assert "tiers/readonly" in RENDER_PRESETS
    assert "tiers/readwrite" in RENDER_PRESETS
    assert "tiers/logbook" in RENDER_PRESETS


@pytest.mark.parametrize("preset", RENDER_PRESETS)
def test_delta_content_is_the_preset_layer_minus_extends(preset: str) -> None:
    """The one contract: emitted keys == the bundled file's own raw layer plus
    its provenance, a tier's ``name`` retitled after the deployment and a
    ``lists: own`` login's kept. A key that appeared here without being in the
    bundled file would be a host setting frozen into the layer."""
    _kind, _sep, stem = preset.partition("/")
    raw, _path = _load_preset_raw(preset)
    expected = dict(raw)
    if raw.get("lists") != "own":
        expected["name"] = f"Retitled ({stem})"
    expected["provenance"] = {"preset": stem, "preset_hash": compute_preset_hash(preset)}

    assert _delta(preset, "Retitled") == expected


@pytest.mark.parametrize("preset", RENDER_PRESETS)
def test_delta_carries_no_extends(preset: str) -> None:
    """Position IS the inheritance — an ``extends:`` in a tier or app file is
    rejected at build time, so emitting one would be a bug."""
    kind, _sep, stem = preset.partition("/")
    text = emit_layer_delta_yaml(
        preset_name=preset,
        profile_name="Render",
        profile_filename=f"my-profile/{kind}/{stem}.yml",
    )

    assert "extends" not in _delta(preset)
    assert not any(line.startswith("extends:") for line in text.splitlines()), text


def test_delta_does_not_re_anchor_profile_relative_paths() -> None:
    """``data:``, the trigger config, and the convention dirs all anchor at the
    host profile's directory, so a tier that named them itself would fight the
    merge rather than help it."""
    parsed = _delta()

    assert "data" not in parsed
    assert "dispatch" not in parsed


def test_delta_keeps_the_preset_comments() -> None:
    """The bundled tiers explain each override they make; that prose is the
    reason the emitter reuses the bundled text instead of re-dumping the dict."""
    text = emit_layer_delta_yaml(
        preset_name=RENDER_PRESET,
        profile_name="Render",
        profile_filename="my-profile/tiers/readonly.yml",
    )

    unwrapped = _unwrapped_comments(text)
    assert "`never` wins over the deployment's own `allowed`" in unwrapped
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
    text = emit_layer_delta_yaml(
        preset_name=RENDER_PRESET,
        profile_name="Render",
        profile_filename="my-profile/tiers/readonly.yml",
    )
    header = _unwrapped_comments(text, header_only=True)

    assert "Render (readonly) — one access level of this deployment's agent" in header
    assert "A tier adds to profile.yml and never subtracts from it" in header
    assert "the build lays this file over that one" in header
    assert "Made from the bundled `readonly` login" in header
    assert f"content hash: {compute_preset_hash(RENDER_PRESET)}" in header
    assert "osprey validate my-profile/tiers/readonly.yml" in header


def test_an_own_list_login_header_says_it_is_a_whole_definition() -> None:
    """A ``lists: own`` login brings its own lists and borrows only the host's
    services, so its header says so instead of describing a delta."""
    text = emit_layer_delta_yaml(
        preset_name="tiers/knowledge",
        profile_name="Render",
        profile_filename="my-profile/tiers/knowledge.yml",
    )
    header = _unwrapped_comments(text, header_only=True)

    assert "a narrow login of this deployment (`lists: own`)" in header
    assert "nothing is inherited from profile.yml's lists or subtracted from them" in header
    assert "Services, data and provider are profile.yml's" in header
    assert "Made from the bundled `knowledge` login" in header
    assert "osprey validate my-profile/tiers/knowledge.yml" in header


def test_preset_without_extends_cannot_be_emitted_as_a_delta() -> None:
    """Only a bundled tier or app is a layer: a deployment preset is the
    ``profile.yml`` itself, and is refused by name rather than emitted as a
    file laid over another one."""
    with pytest.raises(BuildProfileError, match="names no login"):
        emit_layer_delta_yaml(
            preset_name=HOST_PRESET,
            profile_name="Render",
            profile_filename="my-profile/tiers/p.yml",
        )


def test_a_bundled_layer_declaring_extends_is_refused(monkeypatch) -> None:
    """A layer materialized with a parent it does not carry would be neither the
    bundled definition nor explicit, so a bundled file naming one is refused."""
    raw, path = _load_preset_raw(RENDER_PRESET)
    monkeypatch.setattr(
        build_profile_emit,
        "_load_preset_raw",
        lambda _name: ({**raw, "extends": HOST_PRESET}, path),
    )

    with pytest.raises(BuildProfileError, match="declares extends"):
        emit_layer_delta_yaml(
            preset_name=RENDER_PRESET,
            profile_name="Render",
            profile_filename="my-profile/tiers/readonly.yml",
        )


# ---------------------------------------------------------------------------
# Provenance (FR-6): machine-readable, on every emitted host profile
# ---------------------------------------------------------------------------


def _standalone(preset: str) -> str:
    return emit_standalone_profile_yaml(preset, (), "Emitted", project_name="emitted")


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
    # ``osprey init`` writes `data:` into the profile it materializes and lays
    # the tree down beside it; the emitter alone produces neither.
    (tmp_path / "data").mkdir()
    profile_file = tmp_path / "profile.yml"
    profile_file.write_text("data: data\n" + _standalone(HOST_PRESET), encoding="utf-8")

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


@pytest.mark.parametrize("preset", RENDER_PRESETS)
def test_layer_files_carry_their_provenance_block(preset: str) -> None:
    """A tier or app file records what it was made from as data, exactly as
    ``profile.yml`` does, so ``osprey validate`` can compare it against the
    installed layer and report drift."""
    _kind, _sep, stem = preset.partition("/")
    provenance = _delta(preset)["provenance"]

    assert provenance == {"preset": stem, "preset_hash": compute_preset_hash(preset)}
    assert provenance["preset_hash"].startswith("sha256:")
