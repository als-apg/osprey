"""The partition of a rendered ``config.yml`` between its sources.

Every key the build renders comes from exactly one place, and the frozen
baseline renders under ``tests/fixtures/explicit_config/`` are the ledger this
is checked against. For each preset (and each ``channel_finder_mode`` the
preset admits), the keys of the frozen root render are partitioned into:

``C``
    The preset's own ``config:`` block, flattened to dotted leaves. Since the
    app templates were folded into the presets this is the whole of the
    literal, operator-editable configuration.
``D``
    :data:`~osprey.cli.derived_keys.DERIVED_KEYS`, prefix-claiming: the keys
    the framework template writes from the project layout, the port layout,
    the environment, or a profile field.
``P``
    ``api.providers``, rendered from ``providers.yml``.
``Ports``
    ``services.<name>.port|port_host|http_port_host`` for a service the preset
    spells. The one deliberate exception to "the preset states it": the number
    is the service's slot above ``deployment.port_base`` and
    :func:`~osprey.cli.build_profile_ports.layout_port_fill` supplies it at
    build time.
``Corpus``
    ``services.graphdb.ttl_path`` for a preset that spells a graph store block
    and names no corpus: :func:`~osprey.cli.build_injectors.graphdb_corpus_fill`
    supplies the graph view the build writes.
``W``
    Keys the build writes that no preset states: the MCP health address a
    render serving web terminals pins, the bundled record archive's compose
    directory, and the render projection — each tier's and app's ``kind``,
    project name and project directory, derived from the files in ``tiers/``
    and ``apps/``.
``B``
    Keys a profile SECTION other than ``config:`` stands for — the ``bluesky:``
    bridge and its panel, the ``va_archiver:`` store and recorder, the
    ``virtual_accelerator:`` soft IOC and its stand-in, the ``dispatch:``
    services and panel, the ``agent_record:`` block every deployment renders
    (defaulted when absent), an ``mcp_servers:`` entry, the ``web_panels:``
    selection, and the conventions the build registers as user-owned.

The sets are pairwise disjoint and their union is the render. A key in
none of them is a key nothing documents; a key in two is one fact with two
homes, which is exactly what the refusals in :mod:`osprey.cli.derived_keys`
and the ``va_archiver`` duplicate check exist to prevent.

The fixtures are frozen by ``freeze.py`` from the live tree, with
``api.providers`` masked to a sorted name list and ``project_root`` /
``execution.environment`` stripped (see ``meta.json``); both masked keys are
members of ``D``, so the partition is unaffected.
Each frozen render is read with its declared
:data:`~tests.cli.test_explicit_config_equivalence.CELL_DELTAS` applied, so the
partition is checked on the render the live tree builds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_injectors import graphdb_corpus_fill
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_health import MCP_URL_KEY
from osprey.cli.build_profile_merge import _attach, _resolve_extends, render_document
from osprey.cli.build_profile_ports import layout_port_fill
from osprey.cli.build_profile_presets import (
    _layer_presets_dir,
    _load_preset_raw,
    layer_preset_kind,
)
from osprey.cli.derived_keys import is_derived_key
from osprey.cli.profile_root import ProfileKind
from osprey.deployment.web_terminals.render_naming import (
    RENDER_KIND_APP,
    RENDER_KIND_TIER,
    RENDERS_KEY,
)
from osprey.port_layout import DEFAULT_PORT_BASE
from osprey.profiles.providers import load_provider_catalog
from tests.cli.test_explicit_config_equivalence import (
    _CATALOG_ADDITIONS,
    ABSENT,
    CELL_DELTAS,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "explicit_config"

#: The port leaves ``layout_port_fill`` owns. Everything else under a service
#: block is literal configuration the preset states.
_PORT_LEAF = re.compile(r"^services\.[^.]+\.(port|port_host|http_port_host)$")

#: The rendered path of the provider catalog. Masked in the fixtures to a
#: sorted list of names, so it is a single leaf there.
_PROVIDERS_KEY = "api.providers"

#: Keys whose rendered VALUE is legitimately not the preset's: the build
#: resolves ``auto`` to the runtime it used, and the injectors append every
#: service a profile section deploys.
_VALUE_REWRITTEN_BY_BUILD = ("container_runtime", "deployed_services")
#: The render projection's leaves, ``modules.web_terminals.renders.<name>.*``:
#: the build writes one entry per file in ``tiers/`` and ``apps/``
#: (:func:`~osprey.deployment.web_terminals.render_naming.render_projection`),
#: its ``kind`` from the file's directory and its ``project`` and
#: ``project_path`` from the deployment's ``project_name`` and the file's stem.
#: No profile may spell any of it
#: (:func:`~osprey.cli.build_profile_model.render_key_errors`).
_RENDER_PROJECTION_PATTERN = re.compile(
    rf"^{re.escape(RENDERS_KEY)}\.(?P<name>[^.]+)\.(kind|project|project_path)$"
)
#: Keys the build writes that no preset states: the MCP health address a
#: render serving web terminals pins (:mod:`osprey.cli.build_profile_health`),
#: the compose directory of the record archive the build bundles, and the
#: render projection (:data:`_RENDER_PROJECTION_PATTERN`).
_WRITTEN_BY_BUILD_PATTERN = re.compile(
    rf"^({re.escape(MCP_URL_KEY)}|services\.archive\.path)$|{_RENDER_PROJECTION_PATTERN.pattern}"
)


def _cells() -> list[tuple[str, str | None, str]]:
    """``(preset, mode, directory)`` for every frozen root render."""
    cells = json.loads((FIXTURE_ROOT / "cells.json").read_text(encoding="utf-8"))
    return [
        (cell["preset"], cell["mode"], cell["directory"])
        for cell in cells
        if cell.get("status") == "frozen"
    ]


def _leaves(node: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    """Every leaf of a nested mapping as ``(dotted path, value)``.

    An empty mapping is a leaf, as it is for the renderer: ``services: {}``
    addresses that key and nothing beneath it.
    """
    if isinstance(node, Mapping) and node:
        for key, value in node.items():
            yield from _leaves(value, (*prefix, str(key)))
    else:
        yield ".".join(prefix), node


def _render(directory: str, document: str = "root") -> dict[str, Any]:
    """The live render the frozen *document* stands for, as dotted leaves.

    The fixtures stay frozen at the baseline; the leaves a change moves are
    declared in :data:`~tests.cli.test_explicit_config_equivalence.CELL_DELTAS`,
    which the equivalence test proves exact against a live build. Applying them
    here partitions the render the tree now builds, not the one it used to.
    A pending delta is a prediction, not yet the render, so it is not applied.
    """
    leaves = dict(
        _leaves(yaml.safe_load((FIXTURE_ROOT / directory / f"{document}.yml").read_text()))
    )
    for delta in CELL_DELTAS.get(directory, ()):
        if delta.document != document or delta.pending is not None:
            continue
        if delta.added is not None:
            leaves[delta.path] = sorted([*leaves[delta.path], *delta.added])
        elif delta.live is ABSENT:
            leaves.pop(delta.path, None)
        else:
            leaves[delta.path] = delta.live
    return leaves


def _preset_document(preset: str) -> dict[str, Any]:
    """The preset as the build reads it: ``extends`` resolved, ``exclude`` applied."""
    raw, path = _load_preset_raw(preset)
    return _resolve_extends(dict(raw), path)


def _layer_document(preset: str, stem: str, kind: ProfileKind) -> dict[str, Any]:
    """The document a bundled tier or app renders from over *preset*, as the build composes it.

    The build's own composer (:func:`~osprey.cli.build_profile_merge.render_document`,
    then the attachment facts): a tier is merged over the deployment, an app is
    composed from the deployment's facts and its complete app file.
    """
    path = _layer_presets_dir(kind) / f"{stem}.yml"
    layer = _read_profile_document(path)
    assert isinstance(layer, dict), path
    return _attach(
        render_document(kind, _preset_document(preset), layer, f"{path.parent.name}/{path.name}")
    )


def _config_leaves(document: Mapping[str, Any]) -> dict[str, Any]:
    """The preset's ``config:`` block as dotted leaves, whichever spelling it used."""
    config = document.get("config") or {}
    return dict(_leaves(_expand_dotted(dict(config))))


def _block_derived_prefixes(document: Mapping[str, Any]) -> dict[str, str]:
    """Dotted prefix -> the profile section the build derives it from.

    Each entry names a section of the profile that is not ``config:`` and the
    rendered subtree the build writes for it. A key under one of these prefixes
    is that section's, so a preset must not spell it under ``config:``.
    """
    prefixes: dict[str, str] = {}
    if isinstance(document.get("bluesky"), Mapping):
        prefixes["services.bluesky"] = "bluesky:"
    if "bluesky_web" in document:
        prefixes["services.bluesky_web"] = "bluesky_web:"
        prefixes["web.panels.bluesky"] = "bluesky_web:"
    if isinstance(document.get("va_archiver"), Mapping):
        for prefix in (
            "va_archiver",
            "archiver.mongodb_archiver",
            "services.mongodb",
            "services.archiver_recorder",
            "health.categories.archiver",
        ):
            prefixes[prefix] = "va_archiver:"
    virtual_accelerator = document.get("virtual_accelerator")
    if isinstance(virtual_accelerator, Mapping):
        prefixes["services.virtual_accelerator"] = "virtual_accelerator:"
        if virtual_accelerator.get("live_standin"):
            prefixes["services.live_standin"] = "virtual_accelerator.live_standin"
            prefixes["control_system.connector.live_standin"] = "virtual_accelerator.live_standin"
    if isinstance(document.get("dispatch"), Mapping):
        for prefix in (
            "services.event_dispatcher",
            "services.dispatch_worker",
            "web.panels.events",
        ):
            prefixes[prefix] = "dispatch:"
    for name in document.get("mcp_servers") or {}:
        prefixes[f"claude_code.servers.{name}"] = "mcp_servers:"
    # Every deployment renders its agent record from the `agent_record:`
    # section, defaulted when the profile leaves it out.
    prefixes["agent_record"] = "agent_record:"
    # The build names the limits database it writes in every render whose
    # config states a limits block.
    if any(".limits_checking." in f".{key}" for key in _config_leaves(document)):
        prefixes["control_system.limits_checking.database_path"] = "data/facility/limits.yaml"
    # `osprey init` writes the facility rule into the repo's conventions, and
    # the build registers every convention copy as user-owned.
    prefixes["scaffold.user_owned"] = "the repo's convention files"
    return prefixes


def _under(key: str, prefixes: Mapping[str, str]) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)


def _panel_switches(document: Mapping[str, Any]) -> set[str]:
    """``web.panels.<id>.enabled`` for every panel the preset's ``web_panels:`` selects.

    Rendered by the framework template's panel loop from the field, and
    deliberately NOT a member of ``DERIVED_KEYS``: a ``config:`` spelling that
    agrees with the selection is accepted (agree-or-refuse), so it is its own
    term of the partition rather than a derived key.
    """
    return {f"web.panels.{panel}.enabled" for panel in document.get("web_panels") or []}


def _carried_block_switches(render: Mapping[str, Any], config: Mapping[str, Any]) -> set[str]:
    """``web.panels.<id>.enabled`` for every block the render carries a ``config:`` fact for.

    The build tells every ``web.panels`` block the render ends up with whether
    this render shows it (``build_profile_panels.panel_selection_overrides``),
    so a block a render inherits for a tab it does not select is written
    ``enabled: false`` beside the inherited fact. That switch belongs to the
    fact's block, not to the render's selection.
    """
    switches: set[str] = set()
    for key in render:
        if not (key.startswith("web.panels.") and key.endswith(".enabled")):
            continue
        block = key.removesuffix("enabled")
        if any(leaf.startswith(block) and leaf != key for leaf in config):
            switches.add(key)
    return switches


def _partition(render: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, set[str]]:
    """Split the render's keys into the eight named sets, in claim order.

    Claim order matters only where a key could be read two ways: a port leaf
    of a block-derived service (``services.bluesky.port``) is that block's,
    not the layout's, because the block writes the whole service. The test
    below checks the sets are disjoint regardless.
    """
    config = _config_leaves(document)
    spelled_services = {key.split(".")[1] for key in config if key.startswith("services.")}
    block_prefixes = _block_derived_prefixes(document)
    panel_switches = _panel_switches(document)
    corpus = set(graphdb_corpus_fill(dict(document.get("config") or {})))
    sets: dict[str, set[str]] = {
        "D": set(),
        "P": set(),
        "Panels": set(),
        "B": set(),
        "Ports": set(),
        "Corpus": set(),
        "W": set(),
        "C": set(),
    }
    for key in render:
        if is_derived_key(key):
            sets["D"].add(key)
        elif key == _PROVIDERS_KEY or key.startswith(_PROVIDERS_KEY + "."):
            sets["P"].add(key)
        elif key in panel_switches:
            sets["Panels"].add(key)
        elif _under(key, block_prefixes):
            sets["B"].add(key)
        elif _PORT_LEAF.match(key) and key.split(".")[1] in spelled_services:
            sets["Ports"].add(key)
        elif key in corpus:
            sets["Corpus"].add(key)
        elif _WRITTEN_BY_BUILD_PATTERN.match(key) and key not in config:
            sets["W"].add(key)
        elif key in config:
            sets["C"].add(key)
    return sets


@pytest.mark.parametrize(("preset", "_mode", "directory"), _cells())
def test_root_render_is_partitioned_between_its_sources(
    preset: str, _mode: str | None, directory: str
) -> None:
    """Every rendered key has exactly one source, and the union is the render."""
    render = _render(directory)
    document = _preset_document(preset)
    config = _config_leaves(document)
    sets = _partition(render, document)

    claimed = set().union(*sets.values())
    unclaimed = sorted(set(render) - claimed)
    assert not unclaimed, f"{directory}: rendered keys nothing accounts for: {unclaimed}"

    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left] & sets[right]
            assert not overlap, f"{directory}: {left} and {right} both claim {sorted(overlap)}"
    assert sum(len(members) for members in sets.values()) == len(render)

    # The panel-switch term is exactly the field's selection: every selected
    # panel renders its switch, and nothing else renders one there.
    assert sets["Panels"] == _panel_switches(document), directory
    assert not {key for key in config if key in sets["Panels"]}

    # Every key the preset states reaches the render.
    missing = set(config) - set(render)
    assert not missing, f"{directory}: preset keys absent from the render: {sorted(missing)}"

    # A preset must spell nothing the build renders on its own.
    for name in ("D", "P", "Panels", "B", "Ports", "Corpus"):
        assert not {key for key in config if key in sets[name]}, name
    assert not {key for key in config if is_derived_key(key)}
    assert not {key for key in config if _PORT_LEAF.match(key)}
    assert not {key for key in config if _under(key, _block_derived_prefixes(document))}


@pytest.mark.parametrize(("preset", "_mode", "directory"), _cells())
def test_preset_values_reach_the_render_unchanged(
    preset: str, _mode: str | None, directory: str
) -> None:
    """A stated value is the rendered value, except where the build resolves it."""
    render = _render(directory)
    config = _config_leaves(_preset_document(preset))
    for key, value in config.items():
        if key not in render:
            continue
        if key == "container_runtime":
            assert value == "auto"
            assert render[key] in {"docker", "podman"}
            continue
        if key == "deployed_services":
            # The injectors append what the profile's sections deploy; what the
            # preset lists comes first and unchanged.
            assert render[key][: len(value)] == value, key
            continue
        assert render[key] == value, f"{directory}: {key}: preset {value!r}, render {render[key]!r}"


@pytest.mark.parametrize(("preset", "_mode", "directory"), _cells())
def test_ports_are_the_layout_fill_and_nothing_else(
    preset: str, _mode: str | None, directory: str
) -> None:
    """The preset spells no port; the layout fill supplies exactly the render's."""
    render = _render(directory)
    document = _preset_document(preset)
    config = _config_leaves(document)
    ports = _partition(render, document)["Ports"]

    assert not {key for key in config if _PORT_LEAF.match(key)}
    filled = layout_port_fill(dict(document.get("config") or {}), DEFAULT_PORT_BASE)
    assert set(filled) == ports, f"{directory}: fill {sorted(filled)} vs render {sorted(ports)}"
    for key, port in filled.items():
        assert render[key] == port, f"{directory}: {key} renders {render[key]}, layout says {port}"


@pytest.mark.parametrize(("_preset", "_mode", "directory"), _cells())
def test_provider_catalog_covers_the_render(
    _preset: str, _mode: str | None, directory: str, tmp_path: Path
) -> None:
    """The render lists the packaged catalog, short only of the declared additions."""
    rendered_names = set(_render(directory)[_PROVIDERS_KEY])
    catalog_names = set(load_provider_catalog(tmp_path).entries)
    assert rendered_names | set(_CATALOG_ADDITIONS) == catalog_names, directory


def test_catalog_additions_are_catalog_entries_no_fixture_lists(tmp_path: Path) -> None:
    """Each declared catalog addition is packaged, and postdates every fixture.

    An addition no fixture lacks would be a stale declaration; one the catalog
    does not carry would be a typo the assertion above cannot name.
    """
    additions = set(_CATALOG_ADDITIONS)
    assert additions <= set(load_provider_catalog(tmp_path).entries)
    for _preset, _mode, directory in _cells():
        for path in sorted((FIXTURE_ROOT / directory).glob("*.yml")):
            # The frozen file itself: _render applies these very additions.
            listed = set(dict(_leaves(yaml.safe_load(path.read_text())))[_PROVIDERS_KEY])
            assert not listed & additions, f"{directory}/{path.stem}"


def test_no_cell_declares_a_leaf_twice() -> None:
    """No cell declares two deltas for the same ``(document, path)``.

    The equivalence check indexes a cell's deltas by that key, so a second delta
    for a leaf would silently replace the first.
    """
    for cell, deltas in CELL_DELTAS.items():
        keys = [delta.key for delta in deltas]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        assert not duplicates, f"{cell}: {duplicates}"


def _render_cells() -> list[tuple[str, str, str]]:
    """``(render stem, root preset, directory)`` for every frozen tier or app render."""
    out = []
    for preset, _mode, directory in _cells():
        for path in sorted((FIXTURE_ROOT / directory).glob("*.yml")):
            if path.stem != "root":
                out.append((path.stem, preset, directory))
    return out


#: The bundled layer kind each render-projection ``kind`` names.
_LAYER_KINDS = {RENDER_KIND_TIER: ProfileKind.TIER, RENDER_KIND_APP: ProfileKind.APP}


@pytest.mark.parametrize(("_preset", "_mode", "directory"), _cells())
def test_render_projection_names_every_frozen_render(
    _preset: str, _mode: str | None, directory: str
) -> None:
    """The root's render projection lists exactly the renders frozen beside it."""
    projected = {
        match["name"]
        for key in _render(directory)
        if (match := _RENDER_PROJECTION_PATTERN.match(key))
    }
    frozen = {stem for stem, _, cell in _render_cells() if cell == directory}
    assert projected == frozen, directory


@pytest.mark.parametrize(("stem", "preset", "directory"), _render_cells())
def test_render_stays_inside_the_partition(stem: str, preset: str, directory: str) -> None:
    """An attached render carries no key outside its own sources.

    A tier is the bundled ``tiers/<stem>.yml`` merged over the root preset; an
    app is composed from the root preset's facts and the bundled
    ``apps/<stem>.yml``. Either is built attached: it renders no services of
    its own and has the host's ports projected in (the Reach Contract), so the
    check is containment, not equality — every rendered key is the render's
    ``config:``, derived, a provider, a projected port, or a section's.
    """
    projected_kind = _render(directory)[f"{RENDERS_KEY}.{stem}.kind"]
    kind = layer_preset_kind(stem)
    assert kind is not None, f"{directory}/{stem}: no bundled tier or app {stem!r}"
    assert _LAYER_KINDS[projected_kind] is kind, (
        f"{directory}/{stem}: projected as {projected_kind!r}, bundled as {kind.value!r}"
    )
    render = _render(directory, stem)
    document = _layer_document(preset, stem, kind)
    config = _config_leaves(document)
    block_prefixes = _block_derived_prefixes(document)
    panel_switches = _panel_switches(document) | _carried_block_switches(render, config)
    root_document = _preset_document(preset)
    corpus = set(graphdb_corpus_fill(dict(root_document.get("config") or {})))
    unclaimed = [
        key
        for key in render
        if not (
            is_derived_key(key)
            or _WRITTEN_BY_BUILD_PATTERN.match(key)
            or key in corpus
            or key == _PROVIDERS_KEY
            or key in panel_switches
            or _PORT_LEAF.match(key)
            or _under(key, block_prefixes)
            or key in config
        )
    ]
    assert not unclaimed, f"{directory}/{stem}: {unclaimed}"
