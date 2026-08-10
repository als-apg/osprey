"""Partition guard for emitted profiles — "no silent config" (FR8).

Every ``BuildProfile`` field belongs to exactly one of three named sets, and
what each set promises is asserted against real emissions of every bundled
preset: EXPLICIT members always appear, COMMENTED members appear either active
or as a commented template, and build-mechanics keys are never synthesized.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile import _KNOWN_PROFILE_KEYS, BuildProfile, list_presets
from osprey.cli.build_profile_emit import (
    _BUILD_MECHANICS_KEYS,
    _COMMENTED_TEMPLATE_KEYS,
    _COMMENTED_TEMPLATE_ORDER,
    _COMMENTED_TEMPLATES,
    _EXPLICIT_DEFAULTS,
    _EXPLICIT_KEYS,
    _FIELD_TO_YAML,
    emit_standalone_profile_yaml,
)
from osprey.cli.build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY, CONNECTOR_PROFILE_KEY

_FIELDS = frozenset(f.name for f in dataclasses.fields(BuildProfile))


def _emit(preset: str, overrides: tuple[Path, ...] = ()) -> str:
    return emit_standalone_profile_yaml(preset, overrides, (), "Emitted", "profile.yml")


def _yaml_key(field: str) -> str:
    return _FIELD_TO_YAML.get(field, field)


def _active_and_commented(text: str) -> tuple[set[str], set[str]]:
    """Split an emitted profile into its active keys and its templated fields.

    "Templated" is decided by exact template text rather than by scanning for
    commented key-shaped lines: template prose legitimately mentions other keys
    (``the web_panels list above``), and a line-scanning heuristic reads those
    as offered keys. Active keys are YAML-spelled, templated ones field-spelled
    — the two coincide for every member today, and :func:`_yaml_key` bridges
    them where they would not.
    """
    active = set(yaml.safe_load(text) or {})
    templated = {field for field, template in _COMMENTED_TEMPLATES.items() if template in text}
    return active, templated


# ---------------------------------------------------------------------------
# (a) the partition is total and disjoint
# ---------------------------------------------------------------------------


def test_every_field_is_classified_exactly_once() -> None:
    """A new BuildProfile field fails here until someone classifies it."""
    union = _EXPLICIT_KEYS | _COMMENTED_TEMPLATE_KEYS | _BUILD_MECHANICS_KEYS

    assert union == _FIELDS, f"unclassified: {_FIELDS - union}; unknown: {union - _FIELDS}"
    assert not _EXPLICIT_KEYS & _COMMENTED_TEMPLATE_KEYS
    assert not _EXPLICIT_KEYS & _BUILD_MECHANICS_KEYS
    assert not _COMMENTED_TEMPLATE_KEYS & _BUILD_MECHANICS_KEYS


def test_every_commented_member_has_a_template() -> None:
    """ "Never absent" is only deliverable if each member has text to emit."""
    assert set(_COMMENTED_TEMPLATES) == _COMMENTED_TEMPLATE_KEYS
    assert set(_COMMENTED_TEMPLATE_ORDER) == _COMMENTED_TEMPLATE_KEYS
    assert len(_COMMENTED_TEMPLATE_ORDER) == len(_COMMENTED_TEMPLATE_KEYS)


def test_explicit_defaults_cover_every_synthesizable_member() -> None:
    """Only the three keys the emitter always writes may lack a default: the
    display `name`, the version stamp, and the `provenance` record. A default
    for any of them would be a value the emitter never falls back to."""
    assert set(_EXPLICIT_DEFAULTS) == _EXPLICIT_KEYS - {
        "name",
        "requires_osprey_version",
        "provenance",
    }


def test_explicit_defaults_match_the_loader() -> None:
    """A synthesized default must be what the loader would have used anyway,
    or emitting it would change the resolved profile."""
    loader_defaults = {}
    for field in dataclasses.fields(BuildProfile):
        if field.default is not dataclasses.MISSING:
            loader_defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            loader_defaults[field.name] = field.default_factory()  # type: ignore[misc]

    for field, value in _EXPLICIT_DEFAULTS.items():
        if field in ("env", "services"):
            # Parsed into dataclasses/dicts by the loader; `{}` is the raw-YAML
            # spelling of "the default block".
            assert value == {}
            continue
        assert value == loader_defaults[field], field


# ---------------------------------------------------------------------------
# (b) + (c) the sets keep their promises on real emissions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
def test_explicit_members_appear_in_every_emitted_profile(preset: str) -> None:
    """(b) Every EXPLICIT member is present, under its YAML spelling."""
    active, _commented = _active_and_commented(_emit(preset))

    missing = {_yaml_key(f) for f in _EXPLICIT_KEYS} - active
    assert not missing, f"{preset}: EXPLICIT keys missing from emission: {sorted(missing)}"


@pytest.mark.parametrize("preset", list_presets())
def test_commented_members_are_active_or_templated(preset: str) -> None:
    """(c) Every COMMENTED member appears — as a real key or a template."""
    text = _emit(preset)
    active, commented = _active_and_commented(text)

    for field in _COMMENTED_TEMPLATE_KEYS:
        key = _yaml_key(field)
        assert key in active or key in commented, (
            f"{preset}: {key} is neither active nor offered as a commented template"
        )


def test_commented_members_stay_covered_with_overrides_supplying_blocks(tmp_path: Path) -> None:
    """(c) again, with `-O` making two COMMENTED members active — the ones that
    would otherwise always take the template branch."""
    (tmp_path / "art").mkdir()
    override = tmp_path / "o.yml"
    override.write_text(
        "mcp_servers:\n"
        "  matlab:\n"
        "    command: /opt/matlab/bin/mcp-matlab\n"
        "artifact_server:\n"
        "  categories:\n"
        "    optics:\n"
        "      label: Optics\n"
        '      color: "#4C9AFF"\n',
        encoding="utf-8",
    )

    text = _emit("hello-world", (override,))
    active, commented = _active_and_commented(text)

    assert "mcp_servers" in active
    assert "artifact_server" in active
    # Active, so their templates must NOT also be appended — a second commented
    # `mcp_servers:` would be a duplicate key the moment a user uncommented it.
    # This is the branch where the two could collide, so the mutual exclusion is
    # asserted here as well as on plain emissions.
    assert not active & commented
    assert text.count("\n# mcp_servers:") == 0
    assert text.count("\n# artifact_server:") == 0
    for field in _COMMENTED_TEMPLATE_KEYS:
        key = _yaml_key(field)
        assert key in active or field in commented, key


@pytest.mark.parametrize("preset", list_presets())
def test_emission_is_byte_deterministic(preset: str) -> None:
    """Two emissions of the same preset are byte-identical.

    Facilities keep profile.yml under version control, so a set-iteration order
    or a timestamp leaking into the output would show up as a spurious diff on
    every re-emit.
    """
    assert _emit(preset) == _emit(preset)


@pytest.mark.parametrize("preset", list_presets())
def test_no_key_is_both_active_and_commented(preset: str) -> None:
    """A key offered as a commented template while already active would become a
    duplicate key the moment someone uncommented it."""
    active, commented = _active_and_commented(_emit(preset))

    assert not active & commented, f"{preset}: both active and templated: {active & commented}"


def test_data_emits_active_when_the_resolved_profile_carries_it(tmp_path: Path) -> None:
    """Forward-compat for `profile new` (3.4), which materializes a data tree and
    injects `data` into the resolved dict: the COMMENTED contract is
    active-when-carried, so nothing here may assume `data` renders commented."""
    override = tmp_path / "o.yml"
    override.write_text("data: data\n", encoding="utf-8")

    text = _emit("hello-world", (override,))
    active, _commented = _active_and_commented(text)

    assert "data" in active
    assert yaml.safe_load(text)["data"] == "data"
    # The template must not tag along behind the active key.
    assert "\n# data: data" not in text


@pytest.mark.parametrize("preset", list_presets())
def test_build_mechanics_are_never_synthesized(preset: str) -> None:
    """Build mechanics appear only when the preset actually set one."""
    from osprey.cli.build_profile_merge import _resolve_extends
    from osprey.cli.build_profile_presets import _load_preset_raw

    raw, anchor = _load_preset_raw(preset)
    carried = set(_resolve_extends(dict(raw), anchor)) & _BUILD_MECHANICS_KEYS

    active, _commented = _active_and_commented(_emit(preset))

    assert active & _BUILD_MECHANICS_KEYS == carried


@pytest.mark.parametrize("preset", list_presets())
def test_absent_build_mechanics_get_no_commented_template(preset: str) -> None:
    """The other half of the contract: absent build mechanics stay absent —
    no synthesis AND no commented template — which is what keeps the set
    meaningfully different from COMMENTED_TEMPLATE_KEYS."""
    from osprey.cli.build_profile_merge import _resolve_extends
    from osprey.cli.build_profile_presets import _load_preset_raw

    raw, anchor = _load_preset_raw(preset)
    carried = set(_resolve_extends(dict(raw), anchor)) & _BUILD_MECHANICS_KEYS

    text = _emit(preset)
    active, commented = _active_and_commented(text)

    for field in _BUILD_MECHANICS_KEYS - carried:
        assert field not in active, f"{preset}: {field} was synthesized"
        assert field not in commented, f"{preset}: {field} got a commented template"


def test_build_mechanics_carried_by_a_preset_survive_emission() -> None:
    """`claude_md_template` is a real preset choice on two bundled presets —
    dropping it would silently swap the deployment's CLAUDE.md persona."""
    parsed = yaml.safe_load(_emit("ariel-standalone"))

    assert parsed["claude_md_template"] == "CLAUDE.ariel.md.j2"


# ---------------------------------------------------------------------------
# (d) the loader's key set cannot drift from the partition
# ---------------------------------------------------------------------------


def test_known_profile_keys_equals_the_partition_plus_inheritance_keys() -> None:
    """FR8(d): the unknown-key hard error and the partition share one surface.

    Three key families sit outside the field partition and are named here so a
    fourth cannot be added without a reader deciding it belongs: the
    inheritance keys, consumed by ``extends`` resolution; the YAML-surface
    spellings of fields the loader renames; and the top-level shorthands
    (``connector``), folded into ``config:`` before parsing and therefore never
    emitted as keys of their own.
    """
    expected = (
        _FIELDS | {"extends", "exclude"} | set(_FIELD_TO_YAML.values()) | {CONNECTOR_PROFILE_KEY}
    )

    assert set(_KNOWN_PROFILE_KEYS) == expected


# ---------------------------------------------------------------------------
# Provenance header and the version stamp
# ---------------------------------------------------------------------------


def test_emitted_profile_carries_the_provenance_header() -> None:
    text = _emit("control-assistant")
    header = text.split("\nname:")[0]

    assert "source preset: control-assistant" in header
    assert "preset content hash: sha256:" in header
    assert "emitted by OSPREY" in header


def test_version_stamp_comes_from_the_pinned_constant() -> None:
    """Never the running __version__ — that would satisfy its own gate."""
    parsed = yaml.safe_load(_emit("hello-world"))

    assert parsed["requires_osprey_version"] == f">={_PROFILE_SCHEMA_MIN_OSPREY}"


@pytest.mark.parametrize("preset", list_presets())
def test_emitted_profile_keeps_the_app_template_comment_and_position(preset: str) -> None:
    """The rename must not cost the preset's comment or the key's place: a
    re-spell after the sync would find the key deleted and re-append it last."""
    lines = _emit(preset).splitlines()
    key_line = next(i for i, line in enumerate(lines) if line.startswith("app_template:"))
    preceding = "\n".join(lines[max(0, key_line - 4) : key_line])

    assert "#" in preceding, f"{preset}: app_template lost its comment"
    # Still in the file's opening section, not appended at the bottom.
    assert key_line < len(lines) / 2, f"{preset}: app_template drifted to the end of the file"


def test_live_preset_blocks_emit_active() -> None:
    """control-assistant really configures dispatch and bluesky — those must be
    active keys, not templates."""
    parsed = yaml.safe_load(_emit("control-assistant"))

    assert isinstance(parsed["dispatch"], dict)
    assert isinstance(parsed["bluesky"], dict)


def test_no_commented_template_is_offered_twice() -> None:
    """Each commented template appears exactly once, so uncommenting any of
    them can never produce a duplicate key."""
    text = _emit("hello-world")

    for field in _COMMENTED_TEMPLATE_KEYS:
        assert text.count(f"\n# {_yaml_key(field)}:") <= 1, field
