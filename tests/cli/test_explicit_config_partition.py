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
``B``
    Keys a profile SECTION other than ``config:`` stands for — the ``bluesky:``
    bridge and its panel, the ``va_archiver:`` store and recorder, the
    ``virtual_accelerator:`` soft IOC and its stand-in, the ``dispatch:``
    services and panel, an ``mcp_servers:`` entry, the ``web_panels:``
    selection, and the conventions the build registers as user-owned.

The five sets are pairwise disjoint and their union is the render. A key in
none of them is a key nothing documents; a key in two is one fact with two
homes, which is exactly what the refusals in :mod:`osprey.cli.derived_keys`
and the ``va_archiver`` duplicate check exist to prevent.

The fixtures were frozen at the last commit before the conversion, with
``api.providers`` masked to a sorted name list and ``project_root`` /
``execution.environment`` stripped (see ``meta.json``); both masked keys are
members of ``D``, so the partition is unaffected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_merge import _resolve_extends
from osprey.cli.build_profile_ports import layout_port_fill
from osprey.cli.build_profile_presets import _load_preset_raw
from osprey.cli.derived_keys import is_derived_key
from osprey.port_layout import DEFAULT_PORT_BASE
from osprey.profiles.providers import load_provider_catalog

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "explicit_config"

#: The port leaves ``layout_port_fill`` owns. Everything else under a service
#: block is literal configuration the preset states.
_PORT_LEAF = re.compile(r"^services\.[^.]+\.(port|port_host|http_port_host)$")

#: The rendered path of the provider catalog. Masked in the fixtures to a
#: sorted list of names, so it is a single leaf there.
_PROVIDERS_KEY = "api.providers"

#: What the standalone presets gain under ``api.providers`` from the packaged
#: catalog: the four entries their own templates never carried (Requirement 1).
_STANDALONE_GAINS = frozenset({"amsc-i2", "stanford", "argo", "ds4"})

#: Keys whose rendered VALUE is legitimately not the preset's: the build
#: resolves ``auto`` to the runtime it used, the injectors append every service
#: a profile section deploys, and ``osprey init`` repoints the persona catalog
#: at the sibling profiles it writes.
_VALUE_REWRITTEN_BY_BUILD = ("container_runtime", "deployed_services")
#: The three persona-catalog leaves ``osprey init`` repoints at the repo's own
#: name and its ``personas/`` directory. Every other leaf under the catalog
#: (``landing_group``) is compared like any preset value.
_VALUE_REWRITTEN_BY_INIT_PATTERN = re.compile(
    r"^modules\.web_terminals\.personas\.[^.]+\.(project|project_path|build_profile)$"
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


#: Leaves the frozen renders carry that no preset states any more.
#:
#: The fixtures are a baseline, never re-frozen, so a key a preset has since
#: stopped rendering stays in them forever and would read here as a rendered
#: key with no source. These three named the OSPREY project's own mailbox,
#: tracker and documentation site; each preset documents them as a commented
#: example now, and the code defaults still apply, so the rendered document is
#: simply three leaves shorter. The same three are declared in
#: ``test_explicit_config_equivalence.CELL_DELTAS`` — that table is what pins
#: the divergence; this set only keeps the partition reading the render as it
#: is produced today.
_RETIRED_SINCE_THE_FREEZE = frozenset(
    {"web.docs_url", "web.feedback.email", "web.feedback.github_repo"}
)

#: The same, for a leaf one cell alone retired, keyed by fixture directory.
#:
#: ``hello-world`` gates the two ARIEL logbook tools while disabling the server
#: that serves them, so both rows were inert; dropping them leaves the frozen
#: render carrying an ``entry_create`` the preset no longer states. The other
#: presets that spell an ARIEL approval policy still render both rows, which is
#: why this retirement is per cell rather than global — subtracting the leaf
#: from every render would read as those presets stating a key that never
#: arrives. The same difference is declared in
#: ``test_explicit_config_equivalence.CELL_DELTAS``.
_RETIRED_PER_CELL: Mapping[str, frozenset[str]] = {
    "hello-world/unset": frozenset({"approval.tools.entry_create"}),
}


def _render(directory: str, document: str = "root") -> dict[str, Any]:
    retired = _RETIRED_SINCE_THE_FREEZE | _RETIRED_PER_CELL.get(directory, frozenset())
    frozen = _leaves(yaml.safe_load((FIXTURE_ROOT / directory / f"{document}.yml").read_text()))
    return {key: value for key, value in frozen if key not in retired}


def _preset_document(preset: str) -> dict[str, Any]:
    """The preset as the build reads it: ``extends`` resolved, ``exclude`` applied."""
    raw, path = _load_preset_raw(preset)
    return _resolve_extends(dict(raw), path)


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


def _partition(render: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, set[str]]:
    """Split the render's keys into the six named sets, in claim order.

    Claim order matters only where a key could be read two ways: a port leaf
    of a block-derived service (``services.bluesky.port``) is that block's,
    not the layout's, because the block writes the whole service. The test
    below checks the sets are disjoint regardless.
    """
    config = _config_leaves(document)
    spelled_services = {key.split(".")[1] for key in config if key.startswith("services.")}
    block_prefixes = _block_derived_prefixes(document)
    panel_switches = _panel_switches(document)
    sets: dict[str, set[str]] = {
        "D": set(),
        "P": set(),
        "Panels": set(),
        "B": set(),
        "Ports": set(),
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
        elif key in config:
            sets["C"].add(key)
    return sets


@pytest.mark.parametrize(("preset", "mode", "directory"), _cells())
def test_root_render_is_partitioned_between_its_sources(
    preset: str, mode: str | None, directory: str
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

    # Every key the preset states reaches the render, save the ones the
    # fixtures were frozen before: hello-world gains `hooks.debug: false`,
    # which the old template shipped commented out; every preset that spells an
    # ARIEL approval policy gains `approval.tools.entry_publish`, the logbook
    # write-through that was gated nowhere when the freeze ran; every preset
    # that names its approval policy tool by tool gains the three panel-rail
    # verbs, which decide what an operator can launch at all and were likewise
    # gated nowhere then; and every preset that carries a `channel_finder`
    # block gains `channel_finder.query_max_rows`, the middle-layer SQL row
    # cap, which was a number fixed in the tool.
    missing = set(config) - set(render)
    expected_gain = {"hooks.debug"} if preset == "hello-world" else set()
    if "approval.tools.entry_publish" in config:
        expected_gain = expected_gain | {"approval.tools.entry_publish"}
    for tool in ("add_panel_to_rail", "remove_panel_from_rail", "register_panel"):
        if f"approval.tools.{tool}" in config:
            expected_gain = expected_gain | {f"approval.tools.{tool}"}
    if "channel_finder.query_max_rows" in config:
        expected_gain = expected_gain | {"channel_finder.query_max_rows"}
    assert missing == expected_gain, (
        f"{directory}: preset keys absent from the render: {sorted(missing)}"
    )

    # A preset must spell nothing the build renders on its own.
    for name in ("D", "P", "Panels", "B", "Ports"):
        assert not {key for key in config if key in sets[name]}, name
    assert not {key for key in config if is_derived_key(key)}
    assert not {key for key in config if _PORT_LEAF.match(key)}
    assert not {key for key in config if _under(key, _block_derived_prefixes(document))}


@pytest.mark.parametrize(("preset", "mode", "directory"), _cells())
def test_preset_values_reach_the_render_unchanged(
    preset: str, mode: str | None, directory: str
) -> None:
    """A stated value is the rendered value, except where the build resolves it."""
    render = _render(directory)
    config = _config_leaves(_preset_document(preset))
    for key, value in config.items():
        if key not in render:
            continue
        if _VALUE_REWRITTEN_BY_INIT_PATTERN.match(key):
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


@pytest.mark.parametrize(("preset", "mode", "directory"), _cells())
def test_ports_are_the_layout_fill_and_nothing_else(
    preset: str, mode: str | None, directory: str
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


@pytest.mark.parametrize(("preset", "mode", "directory"), _cells())
def test_provider_catalog_covers_the_render(
    preset: str, mode: str | None, directory: str, tmp_path: Path
) -> None:
    """The packaged catalog carries every provider the old render had.

    The standalones gain the four entries their templates never listed and
    nothing else (Requirement 1); the other presets gain nothing.
    """
    rendered_names = set(_render(directory)[_PROVIDERS_KEY])
    catalog_names = set(load_provider_catalog(tmp_path).entries)
    assert rendered_names <= catalog_names
    gained = catalog_names - rendered_names
    if preset in {"ariel-standalone", "channel-finder-standalone"}:
        assert gained == _STANDALONE_GAINS
    else:
        assert not gained


def _persona_cells() -> list[tuple[str, str, str]]:
    """``(persona preset, root preset, directory)`` for every frozen persona render."""
    out = []
    for preset, _mode, directory in _cells():
        for path in sorted((FIXTURE_ROOT / directory).glob("*.yml")):
            if path.stem != "root":
                out.append((f"{preset}-{path.stem}", preset, directory))
    return out


@pytest.mark.parametrize(("persona", "preset", "directory"), _persona_cells())
def test_persona_render_stays_inside_the_partition(
    persona: str, preset: str, directory: str
) -> None:
    """An attached persona render carries no key outside its own preset's sources.

    A persona is a delta over the root preset, built attached: it renders no
    services of its own and has the host's ports projected in (the Reach
    Contract), so the check is containment, not equality — every rendered key
    is the persona's ``config:``, derived, a provider, a projected port, or a
    section's.
    """
    render = _render(directory, persona.removeprefix(f"{preset}-"))
    document = _preset_document(persona)
    config = _config_leaves(document)
    block_prefixes = _block_derived_prefixes(document)
    panel_switches = _panel_switches(document)
    unclaimed = [
        key
        for key in render
        if not (
            is_derived_key(key)
            or key == _PROVIDERS_KEY
            or key in panel_switches
            or _PORT_LEAF.match(key)
            or _under(key, block_prefixes)
            or key in config
        )
    ]
    assert not unclaimed, f"{directory}/{persona}: {unclaimed}"
