"""Preset-vs-profile parity matrix, over every bundled preset.

``osprey profile new`` is only worth having if the profile it materializes is a
faithful stand-in for the preset it came from: a build from the profile has to
land the same project a ``--preset`` build lands. :mod:`tests.cli.test_profile_new`
proves that at the resolved-``BuildProfile`` level for every preset, and at the
project level for one rich preset. This module carries the project-level claim
across the whole preset list, and adds the two data cases that no single default
build reaches — the tier-1 pairing and the no-override negative.

Scope, so the two modules stay complementary rather than parallel:

* resolved-field parity, the persona ``build_profile`` rewrite at the resolved
  layer, profile-tree-vs-bundle byte parity, and the negative/atomicity matrix
  live in ``test_profile_new``;
* the emitter's partition guard, prefix collapse, and provenance header are
  pinned against :func:`emit_standalone_profile_yaml` in ``test_emit_partition``
  and ``test_emit_config_collapse``. Here they are re-asserted against the file
  ``profile new`` actually writes, which is not the emitter's raw output: the
  verb injects ``data:`` and layers the persona-catalog rewrite on top, and only
  the written file is what a facility edits and rebuilds from.

What legitimately differs between the two projects, and is therefore normalized
rather than asserted equal, is where each build was told to put its output: the
``project_root`` key, the absolute paths inside ``.mcp.json``, and the manifest's
source identity (which of preset name / profile path produced it). Everything
else is compared strictly, including the persona rewrite, which is asserted as
the exact value it is specified to take.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.build_profile_emit import (
    _COMMENTED_TEMPLATE_KEYS,
    _COMMENTED_TEMPLATES,
    _EXPLICIT_KEYS,
    _FIELD_TO_YAML,
    emits_persona_profiles,
)
from osprey.cli.profile_cmd import profile

# The presets that own their persona stack, derived from the live predicate so a
# new multi-user preset joins the sibling guards without an edit here. Pinned
# non-empty: an empty tuple would silently collect zero persona tests.
PERSONA_PRESETS: tuple[str, ...] = tuple(
    preset
    for preset in list_presets()
    if emits_persona_profiles(resolve_build_profile(None, preset=preset)[0].config)
)
assert PERSONA_PRESETS, "no bundled preset triggers persona emission — sibling guards would be void"


# ---------------------------------------------------------------------------
# The pair under test: one preset build and one profile build per preset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One preset's two projects, plus the profile the second was built from."""

    preset: str
    profile_dir: Path
    via_preset: Path
    via_profile: Path


def _build_from_preset(runner: CliRunner, preset: str, out_dir: Path, *extra: str) -> None:
    result = runner.invoke(
        build,
        [
            "proj",
            "--preset",
            preset,
            *extra,
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"preset build failed for {preset!r}: {result.output}"


def _build_from_profile(runner: CliRunner, profile_dir: Path, out_dir: Path) -> None:
    result = runner.invoke(
        build,
        [
            "proj",
            str(profile_dir / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"profile build failed: {result.output}"


def _materialize(runner: CliRunner, target: Path, preset: str, *extra: str) -> None:
    result = runner.invoke(profile, ["new", str(target), "--preset", preset, *extra])
    assert result.exit_code == 0, f"profile new failed for {preset!r}: {result.output}"


@pytest.fixture(scope="module", params=list_presets())
def pair(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Pair:
    """Build each preset both ways once, and assert many surfaces off the pair.

    Module-scoped so the matrix costs one preset build plus one profile build
    per preset rather than one of each per assertion.
    """
    preset = request.param
    runner = CliRunner()
    root = tmp_path_factory.mktemp("parity")
    preset_out = root / "via-preset"
    profile_out = root / "via-profile"
    preset_out.mkdir()
    profile_out.mkdir()
    profile_dir = root / "profile"

    _build_from_preset(runner, preset, preset_out)
    _materialize(runner, profile_dir, preset)
    _build_from_profile(runner, profile_dir, profile_out)

    return Pair(
        preset=preset,
        profile_dir=profile_dir,
        via_preset=preset_out / "proj",
        via_profile=profile_out / "proj",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tree_digest(root: Path) -> dict[str, str]:
    """Every path under ``root`` mapped to its content digest.

    Directories are recorded too, so a subtree present in one project and empty
    in the other is a difference rather than a silent match on file lists.
    """
    return {
        path.relative_to(root).as_posix() + ("/" if path.is_dir() else ""): (
            "<dir>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(root.rglob("*"))
    }


def _take_persona_build_profiles(config: dict) -> dict[str, str]:
    """Remove and return the built config's persona ``build_profile`` values.

    The one project-level field materialization rewrites (D7a). Lifted out so
    the rest of the catalog is still compared strictly, and so the rewrite can
    be asserted as an exact value rather than excluded.
    """
    modules = config.get("modules")
    web_terminals = modules.get("web_terminals") if isinstance(modules, dict) else None
    catalog = web_terminals.get("personas") if isinstance(web_terminals, dict) else None
    if not isinstance(catalog, dict):
        return {}
    return {
        name: entry.pop("build_profile")
        for name, entry in catalog.items()
        if isinstance(entry, dict) and "build_profile" in entry
    }


def _without_output_location(payload: object, *roots: Path) -> object:
    """Replace each project's own path with a placeholder.

    The two projects are built into different output directories on purpose —
    they have to coexist — so every absolute path that names a project's own
    root differs by construction. Both the given and the fully resolved spelling
    are replaced, because macOS temp dirs reach the rendered files through a
    symlink.
    """
    text = json.dumps(payload, sort_keys=True)
    for root in roots:
        for spelling in {str(root), str(root.resolve())}:
            text = text.replace(spelling, "<PROJECT>")
    return json.loads(text)


def _active_and_commented(text: str) -> tuple[set[str], set[str]]:
    """Split a profile into active keys and fields offered as templates.

    Mirrors ``test_emit_partition``'s split, including its reason for matching
    template text exactly rather than scanning for commented key-shaped lines:
    template prose legitimately names other keys.
    """
    active = set(yaml.safe_load(text) or {})
    templated = {field for field, template in _COMMENTED_TEMPLATES.items() if template in text}
    return active, templated


def _prefix_pairs(config: dict) -> list[tuple[str, str]]:
    """Every ``(shorter, deeper)`` pair whose dotted segments strictly nest."""
    paths = {key: tuple(key.split(".")) for key in config}
    return [
        (shorter, deeper)
        for shorter, short_path in paths.items()
        for deeper, deep_path in paths.items()
        if len(short_path) < len(deep_path) and deep_path[: len(short_path)] == short_path
    ]


# ---------------------------------------------------------------------------
# SC2: project surfaces, over every preset
# ---------------------------------------------------------------------------


def test_config_yml_matches_apart_from_the_output_location(pair: Pair) -> None:
    """The whole rendered config, not just the two sections a single-preset test
    can afford to check: deployed services, claude_code, modules, connectors,
    every key an injector writes.

    ``project_root`` is where the build was told to write, so it cannot match.
    The persona catalog's ``build_profile`` values are compared separately,
    against the rewrite the verb is specified to perform.
    """
    from_preset = yaml.safe_load((pair.via_preset / "config.yml").read_text())
    from_profile = yaml.safe_load((pair.via_profile / "config.yml").read_text())
    from_preset.pop("project_root", None)
    from_profile.pop("project_root", None)

    preset_personas = _take_persona_build_profiles(from_preset)
    profile_personas = _take_persona_build_profiles(from_profile)

    assert from_profile == from_preset
    # Repointed at the emitted siblings for a profile that owns the stack;
    # untouched for one that only inherits the catalog with the module off.
    rewrites = pair.preset in PERSONA_PRESETS
    assert profile_personas == {
        name: (f"personas/{name}.yml" if rewrites else value)
        for name, value in preset_personas.items()
    }


def test_services_tree_is_byte_identical(pair: Pair) -> None:
    """Not just the same file names — the same bytes. Compose files, Dockerfiles
    and entrypoints are rendered from the resolved profile, so a materialization
    gap in any service-driving field would change their content while leaving
    the tree shape intact.
    """
    assert _tree_digest(pair.via_profile / "services") == _tree_digest(pair.via_preset / "services")


def test_mcp_servers_match_apart_from_the_output_location(pair: Pair) -> None:
    """Server keys, and each server's full definition — command, args, env.

    The env carries each project's own ``CONFIG_FILE``/``OSPREY_CONFIG`` paths,
    which differ because the two projects are built side by side; everything
    else about the server wiring has to be identical.
    """
    from_preset = json.loads((pair.via_preset / ".mcp.json").read_text())
    from_profile = json.loads((pair.via_profile / ".mcp.json").read_text())

    assert sorted(from_profile["mcpServers"]) == sorted(from_preset["mcpServers"])
    assert _without_output_location(
        from_profile, pair.via_profile, pair.profile_dir
    ) == _without_output_location(from_preset, pair.via_preset)


def test_manifest_artifacts_match_and_only_the_source_identity_differs(pair: Pair) -> None:
    """Same artifact selection, and the only ``build_args`` that move are the
    ones that name where the build came from.

    ``source`` is the field the deploy side reads; ``preset``/``profile_path``
    are the two mutually exclusive spellings of the source itself. Pinning the
    difference set rather than just ``source`` keeps a future build_args field
    from drifting between the two paths unnoticed.
    """
    from_preset = json.loads((pair.via_preset / ".osprey-manifest.json").read_text())
    from_profile = json.loads((pair.via_profile / ".osprey-manifest.json").read_text())

    assert from_profile["artifacts"] == from_preset["artifacts"]

    preset_args, profile_args = from_preset["build_args"], from_profile["build_args"]
    differing = {
        key
        for key in set(preset_args) | set(profile_args)
        if preset_args.get(key) != profile_args.get(key)
    }
    assert differing == {"source", "preset", "profile_path", "profile_path_abs"}
    assert preset_args["source"] == "preset"
    assert profile_args["source"] == "profile"
    assert profile_args["profile_path_abs"] == str(pair.profile_dir / "profile.yml")


# ---------------------------------------------------------------------------
# SC3: the project data tree
# ---------------------------------------------------------------------------


def test_project_data_tree_is_byte_identical(pair: Pair) -> None:
    """The payload the agent actually reads at runtime.

    ``test_profile_new`` proves the profile's own tree matches the bundle; this
    is the other half — that routing the build through that copied tree lands
    the same project ``data/`` the packaged bundle does, staging pruned and
    tier DBs materialized exactly alike.
    """
    from_preset = _tree_digest(pair.via_preset / "data")
    from_profile = _tree_digest(pair.via_profile / "data")

    assert from_preset, f"{pair.preset}: preset build produced no data/ tree to compare"
    assert from_profile == from_preset


def test_tier_one_in_context_pair_builds_identical_data(tmp_path: Path) -> None:
    """The only tier-1-valid combination, materialized and built both ways.

    Tier 1 ships an ``in_context`` DB only, so ``--tier 1`` is valid solely
    alongside ``channel_finder_mode: in_context``; every other tier-1 pairing
    hard-fails validation by design. That makes this the one pairing that
    exercises ``tiers/tier1/in_context.json``, and the assertions below check it
    really did — a tier-3 default build lands ``hierarchical.json`` instead, so
    a silently-ignored ``--set tier=1`` would show up as the wrong DB rather
    than as a passing comparison of two identical defaults.
    """
    runner = CliRunner()
    preset_out = tmp_path / "via-preset"
    profile_out = tmp_path / "via-profile"
    preset_out.mkdir()
    profile_out.mkdir()
    profile_dir = tmp_path / "profile"

    _build_from_preset(
        runner,
        "control-assistant",
        preset_out,
        "--tier",
        "1",
        "--set",
        "channel_finder_mode=in_context",
    )
    _materialize(
        runner,
        profile_dir,
        "control-assistant",
        "--set",
        "tier=1",
        "--set",
        "channel_finder_mode=in_context",
    )

    # Both overrides are baked ACTIVE — a commented `tier:` would rebuild tier 3.
    written = yaml.safe_load((profile_dir / "profile.yml").read_text())
    assert written["tier"] == 1
    assert written["channel_finder_mode"] == "in_context"

    _build_from_profile(runner, profile_dir, profile_out)

    from_preset = _tree_digest(preset_out / "proj" / "data")
    from_profile = _tree_digest(profile_out / "proj" / "data")
    assert from_profile == from_preset
    # The tier-1 DB really is what landed, in both.
    assert "channel_databases/in_context.json" in from_preset
    assert "channel_databases/hierarchical.json" not in from_preset


@pytest.mark.parametrize("preset", list_presets())
def test_a_no_override_emission_pins_no_active_tier(preset: str, tmp_path: Path) -> None:
    """The negative that makes the default-case data parity hold.

    ``tier`` is derived from ``channel_finder_mode`` when unset. Baking a
    resolved ``tier:`` into a no-override profile would freeze today's default
    into the file, so a facility that later edits the mode would keep building
    the old tier's DB — and the profile build's ``data/`` would stop matching
    the preset build's. It has to stay a commented template.
    """
    target = tmp_path / "p"
    _materialize(CliRunner(), target, preset)

    active, commented = _active_and_commented((target / "profile.yml").read_text())
    assert "tier" not in active, f"{preset}: a resolved tier was baked into the profile"
    assert "tier" in commented, f"{preset}: tier is neither active nor offered as a template"


# ---------------------------------------------------------------------------
# SC6: the guards, applied to the file `profile new` writes
# ---------------------------------------------------------------------------


def test_materialized_profile_satisfies_the_partition_guard(pair: Pair) -> None:
    """FR8 on the written file, not on the emitter's raw output: the verb adds
    ``data:`` and the persona-catalog layer after emission, and it is the
    written file a facility edits."""
    text = (pair.profile_dir / "profile.yml").read_text()
    active, commented = _active_and_commented(text)

    missing = {_FIELD_TO_YAML.get(field, field) for field in _EXPLICIT_KEYS} - active
    assert not missing, f"{pair.preset}: EXPLICIT keys missing from the profile: {sorted(missing)}"
    for field in _COMMENTED_TEMPLATE_KEYS:
        key = _FIELD_TO_YAML.get(field, field)
        assert key in active or field in commented, f"{pair.preset}: {key} is neither"
    assert not active & commented, f"{pair.preset}: both active and templated: {active & commented}"
    # `data` is the key the verb injects; it must have landed active.
    assert active >= {"data"}


def test_materialized_profile_carries_the_provenance_header(pair: Pair) -> None:
    """Every preset's written profile records what it came from — the header is
    how a facility months later knows which preset and which content it was cut
    from."""
    header = (pair.profile_dir / "profile.yml").read_text().split("\nname:")[0]

    assert f"source preset: {pair.preset}" in header
    assert "preset content hash: sha256:" in header
    assert "emitted by OSPREY" in header


def test_no_config_key_in_the_materialized_profile_prefixes_another(pair: Pair) -> None:
    """The prefix collapse has to survive the persona-catalog layer.

    That layer addresses ``modules.web_terminals.personas.<name>.build_profile``,
    directly under a subtree the presets already configure — exactly the shape
    that produces an order-dependent ``config_update_fields`` application if it
    is left as a separate key.
    """
    config = yaml.safe_load((pair.profile_dir / "profile.yml").read_text()).get("config") or {}

    assert _prefix_pairs(config) == []


@pytest.mark.parametrize("preset", PERSONA_PRESETS)
def test_emitted_persona_profiles_carry_the_guards_too(preset: str, tmp_path: Path) -> None:
    """The siblings are DELTAS, so the EXPLICIT-key guarantee is the HOST's:
    every one of those fields is written once, there, and reaches each persona
    through the implicit merge. Defaults-synthesis deliberately does NOT run for
    a sibling — synthesizing a loader default into a delta would pin it against
    the host rather than inherit it. What a sibling owes for itself is the two
    guarantees that are about the file: prefix cleanliness and provenance."""
    runner = CliRunner()
    target = tmp_path / "profile"
    _materialize(runner, target, preset)

    host, _host_dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    for field in _EXPLICIT_KEYS:
        key = _FIELD_TO_YAML.get(field, field)
        assert getattr(host, field, None) is not None, (
            f"host profile: EXPLICIT key {key!r} does not survive resolution"
        )

    siblings = sorted((target / "personas").glob("*.yml"))
    assert siblings, f"{preset}: no persona profiles were emitted"
    for sibling in siblings:
        text = sibling.read_text()
        raw = yaml.safe_load(text)
        assert "extends" not in raw, f"{sibling.name}: a delta inherits by position, not by key"
        assert _prefix_pairs(raw.get("config") or {}) == []
        assert "emitted by OSPREY" in text.split("\nname:")[0]
