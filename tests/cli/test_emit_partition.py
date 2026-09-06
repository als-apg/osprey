"""Partition guard for emitted profiles — "no silent config" (FR8).

Every ``BuildProfile`` field belongs to exactly one of three named sets, and
what each set promises is asserted against real emissions of every bundled
preset: EXPLICIT members always appear, COMMENTED members appear either active
or as a commented template, and build-mechanics keys are never synthesized.

The second half of the file is about the ``config:`` block the presets now
carry. Those keys used to live in an app template the operator never opened, so
their documentation could say anything; written into the profile, each comment
is read beside the key it describes and has to still be true of the deployment
that reads it. Three properties are pinned: every key is documented, no comment
freezes a value the operator can move, and emission is byte-deterministic.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.build_profile import _KNOWN_PROFILE_KEYS, BuildProfile, list_presets
from osprey.cli.build_profile_emit import (
    _BUILD_MECHANICS_KEYS,
    _COMMENTED_TEMPLATE_KEYS,
    _COMMENTED_TEMPLATE_ORDER,
    _COMMENTED_TEMPLATES,
    _EXPLICIT_DEFAULTS,
    _EXPLICIT_KEYS,
    emit_standalone_profile_yaml,
)
from osprey.cli.build_profile_load import (
    _PROFILE_SCHEMA_MIN_OSPREY,
    CONNECTOR_PROFILE_KEY,
    PORT_BASE_PROFILE_KEY,
)
from osprey.cli.build_profile_presets import PRESET_DATA_BUNDLE_KEY
from osprey.port_layout import DEFAULT_PORT_BASE, LAYOUT

_FIELDS = frozenset(f.name for f in dataclasses.fields(BuildProfile))

#: Longest run of comment lines that still documents the key below it. Three is
#: the reader's rule of thumb: further up and the prose belongs to a section
#: heading or to the previous key, not to this one.
_COMMENT_REACH = 3

#: A key line inside the ``config:`` block: two spaces, then a dotted path.
_CONFIG_KEY_RE = re.compile(r"^ {2}([A-Za-z_][\w.]*):")


def _emit(preset: str, set_pairs: tuple[str, ...] = ()) -> str:
    return emit_standalone_profile_yaml(preset, set_pairs, "Emitted")


def _active_and_commented(text: str) -> tuple[set[str], set[str]]:
    """Split an emitted profile into its active keys and its templated fields.

    "Templated" is decided by exact template text rather than by scanning for
    commented key-shaped lines: template prose legitimately mentions other keys
    (``the web_panels list above``), and a line-scanning heuristic reads those
    as offered keys.
    """
    active = set(yaml.safe_load(text) or {})
    templated = {field for field, template in _COMMENTED_TEMPLATES.items() if template in text}
    return active, templated


def _config_block(text: str) -> list[str]:
    """The lines under ``config:``, exclusive of the ``config:`` line itself."""
    lines = text.splitlines()
    start = lines.index("config:")
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i] and not lines[i].startswith((" ", "#"))
        ),
        len(lines),
    )
    return lines[start + 1 : end]


def _comment_lines(text: str) -> list[str]:
    """Every whole-line comment in an emitted profile, stripped."""
    return [line.strip() for line in text.splitlines() if line.strip().startswith("#")]


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

    missing = set(_EXPLICIT_KEYS) - active
    assert not missing, f"{preset}: EXPLICIT keys missing from emission: {sorted(missing)}"


@pytest.mark.parametrize("preset", list_presets())
def test_commented_members_are_active_or_templated(preset: str) -> None:
    """(c) Every COMMENTED member appears — as a real key or a template."""
    text = _emit(preset)
    active, commented = _active_and_commented(text)

    for field in _COMMENTED_TEMPLATE_KEYS:
        assert field in active or field in commented, (
            f"{preset}: {field} is neither active nor offered as a commented template"
        )


def test_commented_members_stay_covered_with_set_supplying_blocks() -> None:
    """(c) again, with ``--set`` making two COMMENTED members active — the ones
    that would otherwise always take the template branch.

    A mapping given as a ``--set`` value is written whole at the key it names,
    so one pair per key states the block a facility would otherwise author by
    hand.
    """
    text = _emit(
        "hello-world",
        (
            'mcp_servers={"matlab": {"command": "/opt/matlab/bin/mcp-matlab"}}',
            'artifact_server={"categories": {"optics": {"label": "Optics", "color": "#4C9AFF"}}}',
        ),
    )
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
        assert field in active or field in commented, field


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
    """Forward-compat for `osprey init` (3.4), which materializes a data tree and
    injects `data` into the resolved dict: the COMMENTED contract is
    active-when-carried, so nothing here may assume `data` renders commented."""
    text = _emit("hello-world", ("data=data",))
    active, _commented = _active_and_commented(text)

    assert "data" in active
    assert yaml.safe_load(text)["data"] == "data"
    # The template must not tag along behind the active key.
    assert "\n# data: data" not in text


def test_hello_world_extension_surface_is_pinned() -> None:
    """The commented blocks a fresh `osprey init --preset hello-world` leaves
    behind ARE the onboarding curriculum: each one is a feature the tutorial
    invites the reader to turn on by uncommenting it. hello-world is minimal
    precisely so that surface stays wide, and nothing else in the suite would
    notice it narrowing — the partition tests above only require each COMMENTED
    member to be *either* active or templated, which a key going active
    satisfies just as well. So a preset gaining a key, or a template being
    dropped, would quietly delete a lesson. Pin the set exactly, by name, so
    both a shrink and an unexpected growth fail here and get looked at.

    The count is 12, not 13: a *bare* emission templates 13, but `osprey init`
    injects `data=data` into `set_pairs` (``profile_cmd``), which makes `data`
    an active key. What ships to the reader is the materialized profile, so
    that is what is pinned — do not "correct" this to the bare-emission 13.

    `mcp_servers` left this set deliberately, and no lesson left with it: the
    preset now carries a live `example_server` entry for the package seeded
    into the repo, so the key emits active instead of templated. The stdio
    teaching moved into that block's comments in the appendix's own words,
    and a commented `lattice:` sibling still shows the remote form, so the
    reader sees both without uncommenting anything. That copy is guarded
    against drift by ``tests/cli/test_example_server_launch.py``.
    """
    _, templated = _active_and_commented(_emit("hello-world", set_pairs=("data=data",)))

    assert templated == {
        "channel_finder_mode",
        "tier",
        "default_panel",
        "deploy",
        "artifact_server",
        "dispatch",
        "bluesky",
        "virtual_accelerator",
        "va_archiver",
        "bluesky_web",
        "nextcloud_bridge",
        "gchat_bridge",
    }


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
    """`claude_md_template` is a real preset choice on three bundled presets —
    dropping it would silently swap the deployment's CLAUDE.md persona."""
    parsed = yaml.safe_load(_emit("ariel-standalone"))

    assert parsed["claude_md_template"] == "CLAUDE.ariel.md.j2"


# ---------------------------------------------------------------------------
# (d) the loader's key set cannot drift from the partition
# ---------------------------------------------------------------------------


def test_known_profile_keys_equals_the_partition_plus_inheritance_keys() -> None:
    """FR8(d): the unknown-key hard error and the partition share one surface.

    Two key families sit outside the field partition and are named here so a
    third cannot be added without a reader deciding it belongs: the inheritance
    keys, consumed by ``extends`` resolution; and the top-level shorthands
    (``connector``, ``port_base``), folded into ``config:`` before parsing and
    therefore never emitted as keys of their own.

    ``app_template`` is in neither family and is deliberately absent from both
    sides: it names a packaged data tree, is consumed when a PRESET is read,
    and is refused in a profile.
    """
    expected = _FIELDS | {"extends", "exclude"} | {CONNECTOR_PROFILE_KEY, PORT_BASE_PROFILE_KEY}

    assert set(_KNOWN_PROFILE_KEYS) == expected
    assert PRESET_DATA_BUNDLE_KEY not in _KNOWN_PROFILE_KEYS


# ---------------------------------------------------------------------------
# Provenance header and the version stamp
# ---------------------------------------------------------------------------


def test_emitted_profile_carries_the_provenance_header() -> None:
    text = _emit("control-assistant")
    header = text.split("\nname:")[0]

    assert "bundled `control-assistant` preset" in header
    assert "preset content hash: sha256:" in header
    assert "emitted by OSPREY" in header


def test_version_stamp_comes_from_the_pinned_constant() -> None:
    """Never the running __version__ — that would satisfy its own gate."""
    parsed = yaml.safe_load(_emit("hello-world"))

    assert parsed["requires_osprey_version"] == f">={_PROFILE_SCHEMA_MIN_OSPREY}"


@pytest.mark.parametrize("preset", list_presets())
def test_the_preset_side_bundle_key_is_not_emitted(preset: str) -> None:
    """`app_template:` selects the packaged data tree and is not a profile key.

    An emitted profile that carried it would be refused by the loader that
    wrote it, so its absence is the contract — and the comment the preset put
    above it goes too, or the emitted file explains a key it does not contain.
    """
    text = _emit(preset)

    assert PRESET_DATA_BUNDLE_KEY not in (yaml.safe_load(text) or {})
    assert PRESET_DATA_BUNDLE_KEY not in text, (
        f"{preset}: the preset's app_template comment survived the key"
    )


@pytest.mark.parametrize("preset", list_presets())
def test_dropping_the_bundle_key_leaves_the_next_key_commented(preset: str) -> None:
    """Removing a key must not take the following key's comment with it.

    ruamel stores the block above a key on its PREDECESSOR, so a naive delete
    either orphans this key's own comment or eats the next key's. `provider:`
    follows `app_template:` in every bundled preset, and it is the one whose
    comment the drop passes through.
    """
    lines = _emit(preset).splitlines()
    key_line = next(i for i, line in enumerate(lines) if line.startswith("provider:"))
    preceding = "\n".join(lines[max(0, key_line - 4) : key_line])

    assert "#" in preceding, f"{preset}: provider lost its comment"


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
        assert text.count(f"\n# {field}:") <= 1, field


# ---------------------------------------------------------------------------
# The `config:` block: documented, unfrozen, reproducible
# ---------------------------------------------------------------------------


def test_every_config_key_carries_a_comment() -> None:
    """Nothing under ``config:`` arrives undocumented.

    This is the promise the app template could not keep. Its keys came with
    comments, but in a file the operator never opened; written into
    profile.yml, a key with no prose above it is a value someone has to guess
    the meaning of from its name — and there are ~300 of them.

    control-assistant is the preset asked because it is the widest: the other
    bundled presets configure subsets of the same surface.
    """
    undocumented = []
    block = _config_block(_emit("control-assistant"))
    for i, line in enumerate(block):
        match = _CONFIG_KEY_RE.match(line)
        if match is None:
            continue
        above = (entry.strip() for entry in block[max(0, i - _COMMENT_REACH) : i])
        if not any(entry.startswith("#") and entry.strip("# ") for entry in above):
            undocumented.append(match.group(1))

    assert undocumented == [], (
        f"{len(undocumented)} config key(s) have no comment within "
        f"{_COMMENT_REACH} lines above: {undocumented}"
    )


@pytest.mark.parametrize("preset", list_presets())
def test_no_comment_names_a_default_layout_port(preset: str) -> None:
    """A port an operator can move must not be spelled out in prose.

    Every host port a deployment publishes is ``deployment.port_base`` plus a
    fixed offset, so the numbers below are only this deployment's ports until
    someone moves the base. The app template's comments were rendered from
    ``{{ osprey_ports.* }}`` and so were correct at emission and wrong from the
    first time the base moved. A comment names the slot instead.
    """
    ports = sorted({DEFAULT_PORT_BASE + slot.offset for slot in LAYOUT})
    pattern = re.compile(rf"(?<!\d)({'|'.join(str(port) for port in ports)})(?!\d)")

    offenders = [line for line in _comment_lines(_emit(preset)) if pattern.search(line)]

    assert offenders == [], f"{preset}: comment(s) naming a default-layout port: {offenders}"


@pytest.mark.parametrize("mode", sorted(VALID_CHANNEL_FINDER_MODES))
@pytest.mark.parametrize("preset", ("control-assistant", "channel-finder-standalone"))
def test_no_comment_freezes_the_channel_finder_mode(preset: str, mode: str) -> None:
    """Prose about the channel finder must hold whatever paradigm is selected.

    Stated as invariance rather than as a word ban, because the words cannot be
    banned: "graph" is also the knowledge graph, the graph store and the graph
    data Neo4j pages. What would be wrong is a comment that changes with the
    field — the ``{{ channel_finder_mode }}`` interpolations the app template
    carried, which described the mode as a fact and went stale the moment
    ``osprey set channel_finder_mode=`` moved it.

    The two presets asked are the ones that spell the field. For a preset that
    leaves it to the commented template, setting the mode retires that template,
    which is a difference in the offered keys rather than in frozen prose.
    """
    baseline = _comment_lines(_emit(preset))
    switched = _comment_lines(_emit(preset, set_pairs=(f"channel_finder_mode={mode}",)))

    assert baseline == switched, (
        f"{preset}: the comments moved when channel_finder_mode became {mode!r}"
    )


@pytest.mark.parametrize("preset", list_presets())
def test_no_comment_freezes_the_port_base(preset: str) -> None:
    """The other half of the port rule, over a base the operator really moved.

    The literal check above only catches the numbers of the DEFAULT layout. A
    comment rendered from a profile's own ``deployment.port_base`` would pass
    it and still be a frozen port, so the same emission is asked at a moved
    base and its comments must not have moved with it.
    """
    baseline = _comment_lines(_emit(preset))
    moved = _comment_lines(_emit(preset, set_pairs=("config.deployment.port_base=21000",)))

    assert baseline == moved, f"{preset}: the comments moved with deployment.port_base"


@pytest.mark.parametrize("preset", list_presets())
def test_emission_is_byte_deterministic_across_two_runs(preset: str) -> None:
    """Re-emitting must produce the same bytes, config block included.

    ``test_emission_is_byte_deterministic`` above says the same thing and is
    kept: this one exists because the config block is where determinism became
    easy to lose. It is built by merging dotted keys and their comments out of
    a preset document, and a set iteration anywhere in that path would show up
    as a spurious diff on every re-emit of a file facilities keep in git.
    """
    first, second = _emit(preset), _emit(preset)

    assert first == second
    assert _config_block(first) == _config_block(second)
