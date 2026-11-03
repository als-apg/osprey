"""What a render serves: the facility file's models it runs and the channels it derives.

``simulation.models`` in a render's ``config.yml`` names the models. Absent or
``null`` serves every model the facility file holds; ``[]`` and ``[texture]``
serve no physics. ``texture`` is always served and always listed last, after
the physics models sorted by name, so equal configs give equal lists.

The connector blocks whose channels the build serves, the virtual
accelerator's and the stand-in's, prove their target reachable by reading
their ``probe_channel``. The build derives it, and the archive's freshness
channel, from the channels the served simulator holds; address is the
tie-break inside that scope. A stated value is an override, and it must be in
the facility file, as must every name of a stated
``simulation.default_scenarios``.

Two other profile keys are facts of the facility tree, and the build states
them from it the same way: a tree that ships ``ariel/vocabulary.yml`` turns
ARIEL's shorthand expansion on at that file, and one that ships benchmark query
sources gets the dataset path the benchmarks view writes the query file to. A
stated value is the profile's own; one the tree cannot back is refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from osprey.facility import DEFAULT_ROLE, TEXTURE
from osprey.facility.errors import FacilityBuildError, quoted_slots
from osprey.facility.records import ChannelSelection, FacilityRecords
from osprey.facility.views.benchmarks import BENCHMARK_DATASET_KEY, QUERY_SOURCES
from osprey_connectors.types import LIVE_STANDIN, VIRTUAL_ACCELERATOR

if TYPE_CHECKING:
    from osprey.cli.build_profile_archiver import VAArchiverConfig

__all__ = [
    "BENCHMARK_DATASET_PATH",
    "CONFIG_SOURCE",
    "SERVED_PROBE_KEYS",
    "SIMULATION_MODELS_KEY",
    "VOCABULARY_ENABLED_KEY",
    "VOCABULARY_PATH_KEY",
    "VOCABULARY_SOURCE",
    "check_served_overrides",
    "facility_tree_fill",
    "resolve_served",
    "served_channel_fill",
]

#: The dotted config key that names the served models.
SIMULATION_MODELS_KEY = "simulation.models"

#: The file ``resolve_served`` reads the key from.
CONFIG_SOURCE = "config.yml"

#: The profile key that states the archive's freshness channel.
FRESHNESS_CHANNEL_KEY = "va_archiver.freshness_channel"

#: The dotted config key that names the scenarios a deployment starts in.
DEFAULT_SCENARIOS_KEY = "simulation.default_scenarios"

#: The file a profile key is stated in.
PROFILE_SOURCE = "profile.yml"

#: The one line of a build whose served models hold no channel to derive.
NO_READBACK_WARNING = (
    "facility: the served models hold no readback; no probe or freshness channel derived"
)

#: The profile key that turns ARIEL's shorthand expansion on.
VOCABULARY_ENABLED_KEY = "ariel.vocabulary.enabled"

#: The profile key that names ARIEL's vocabulary file.
VOCABULARY_PATH_KEY = "ariel.vocabulary.path"

#: The vocabulary a facility tree ships, relative to the render root.
VOCABULARY_SOURCE = "data/facility/ariel/vocabulary.yml"

#: Where a render carries the benchmark query file when the profile names no
#: other place, relative to the render root.
BENCHMARK_DATASET_PATH = "data/benchmarks/queries.json"

#: The probe keys of the connector blocks whose channels the build serves, in
#: the order they are checked: the stand-in copies the virtual accelerator's.
SERVED_PROBE_KEYS: tuple[str, ...] = tuple(
    f"control_system.connector.{block}.probe_channel"
    for block in (VIRTUAL_ACCELERATOR, LIVE_STANDIN)
)


def _invalid(
    detail: str,
    remedy: str,
    key: str = SIMULATION_MODELS_KEY,
    source: str = CONFIG_SOURCE,
) -> FacilityBuildError:
    return FacilityBuildError(
        "profile-invalid",
        key,
        [source],
        remedy,
        record_kind="path",
        detail=detail,
    )


def resolve_served(rendered_config: Mapping[str, Any], records: FacilityRecords) -> list[str]:
    """Compute the models one render serves.

    Args:
        rendered_config: The render's ``config.yml``, as a nested mapping.
        records: The build's facility file's records.

    Returns:
        The served model names: the physics models sorted by name, then
        ``texture``.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the key is not a list of
            names, or names a model the facility file does not hold.
    """
    physics = sorted(records.ids("model") - {TEXTURE})
    known = [*physics, TEXTURE]
    simulation = rendered_config.get("simulation")
    if simulation is None:
        return known
    if not isinstance(simulation, Mapping):
        raise _invalid(
            f"`simulation` is {simulation!r}, not a mapping",
            f"write `{SIMULATION_MODELS_KEY}` as a list of model names",
        )
    requested = simulation.get("models")
    if requested is None:
        return known
    if not isinstance(requested, list) or not all(isinstance(n, str) for n in requested):
        raise _invalid(
            f"`{SIMULATION_MODELS_KEY}` is {requested!r}, not a list of model names",
            f"write `{SIMULATION_MODELS_KEY}` as a list of model names",
        )
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise _invalid(
            f"{quoted_slots(unknown)} is not a model in the facility file; its models are "
            f"{quoted_slots(known)}",
            f"name only models the facility file holds in `{SIMULATION_MODELS_KEY}`",
        )
    return [*sorted(set(requested) - {TEXTURE}), TEXTURE]


def check_served_overrides(
    rendered_config: Mapping[str, Any], records: FacilityRecords, scenario_names: Iterable[str]
) -> None:
    """Hold a render's stated probe channels and start set to the facility file.

    A block that states no probe channel passes: the target switch reports a
    missing one itself. Absent, ``null`` and ``[]`` start sets pass, and so does
    one that is not a list of names, which its own stop refuses.

    Args:
        rendered_config: The render's ``config.yml``, as a nested mapping.
        records: The build's facility file's records.
        scenario_names: The names of the facility file's scenarios.

    Raises:
        FacilityBuildError: ``profile-invalid`` naming the first key of
            ``SERVED_PROBE_KEYS`` whose address is not a channel of the
            facility file, with the first readback by address as the remedy, or
            naming ``simulation.default_scenarios`` when it states a name the
            facility file does not hold, with the file's scenarios as the remedy.
    """
    for key in SERVED_PROBE_KEYS:
        address = _dotted(rendered_config, key)
        if address is None or records.record("channel", str(address)) is not None:
            continue
        readbacks = _readbacks(records.of_kind("channel"))
        remedy = (
            f"set `{key}` to a readback the facility file holds, such as `{readbacks[0]}`"
            if readbacks
            else f"remove `{key}` from the profile"
        )
        raise _invalid(f"`{address}` is not a channel of the facility file", remedy, key)
    stated = _dotted(rendered_config, DEFAULT_SCENARIOS_KEY)
    if not isinstance(stated, list):
        return
    known = sorted(scenario_names)
    unknown = sorted({str(name) for name in stated} - set(known))
    if unknown:
        remedy = f"set `{DEFAULT_SCENARIOS_KEY}` to `[]`"
        if known:
            remedy += f" or to scenarios of the facility file: {quoted_slots(known)}"
        raise _invalid(
            f"{quoted_slots(unknown)} is not a scenario of the facility file",
            remedy,
            DEFAULT_SCENARIOS_KEY,
        )


def served_channel_fill(
    records: FacilityRecords,
    rendered_config: Mapping[str, Any],
    profile_config: Mapping[str, Any],
    va_archiver: VAArchiverConfig | None,
    *,
    reported: set[str] | None = None,
) -> dict[str, Any]:
    """The dotted config keys the build derives from the channels the simulator serves.

    The probe is the first readback by address in the served scope, written only
    for a render that carries the simulator's connector block: a key under the
    block would make the simulator a target of a deployment that has none. The
    freshness check watches the same channel, and is written only for a profile
    with a ``va_archiver:`` block that states no channel and does not write the
    health category itself. Its contract is that the recorder is writing, which
    never reads whether the value moves. The start set is never derived.

    Args:
        records: The build's facility file's records.
        rendered_config: The render's ``config.yml`` before the profile's overlay.
        profile_config: The profile's ``config:`` block, laid over it.
        va_archiver: The profile's parsed ``va_archiver:`` block, or ``None``.
        reported: The notes this build has printed; a warning in it is not
            printed again and one printed is added. ``None`` prints it.

    Returns:
        The keys to write; empty when the served scope holds no readback.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the models do not resolve,
            or the profile's ``va_archiver.freshness_channel`` is not a channel
            of the facility file, with the first readback by address as the
            remedy.
    """
    from osprey.cli.build_profile_archiver import (
        _addresses_freshness_category,
        _expand_dotted,
        freshness_check_override,
    )
    from osprey.deployment.reach import dotted_get

    stated_channel = va_archiver.freshness_channel if va_archiver is not None else None
    if stated_channel and records.record("channel", stated_channel) is None:
        raise _stated_freshness_refusal(stated_channel, records)

    profile = _expand_dotted(profile_config)
    serves_simulator = _carries_the_simulator_block(rendered_config, profile)
    if not serves_simulator and va_archiver is None:
        return {}
    models = dotted_get(profile, SIMULATION_MODELS_KEY)
    if models is None:
        served = resolve_served(rendered_config, records)
    else:
        served = resolve_served({"simulation": {"models": models}}, records)

    readbacks = _readback_channels(_served_scope(records, served))
    if not readbacks:
        if reported is None or NO_READBACK_WARNING not in reported:
            click.echo(NO_READBACK_WARNING, err=True)
            if reported is not None:
                reported.add(NO_READBACK_WARNING)
        return {}

    fill: dict[str, Any] = {}
    if serves_simulator and dotted_get(profile, SERVED_PROBE_KEYS[0]) is None:
        fill[SERVED_PROBE_KEYS[0]] = readbacks[0]["id"]
    if (
        va_archiver is not None
        and not stated_channel
        and not _addresses_freshness_category(profile_config)
    ):
        fill.update(
            freshness_check_override(
                replace(va_archiver, freshness_channel=str(readbacks[0]["id"]))
            )
        )
    return fill


def facility_tree_fill(facility_dir: Path, profile_config: Mapping[str, Any]) -> dict[str, Any]:
    """The dotted config keys the build derives from what the facility tree ships.

    A tree with ``ariel/vocabulary.yml`` states ``ariel.vocabulary.enabled: true``
    and the file's path; one with benchmark query sources states
    ``channel_finder.benchmark.dataset_path``. A key the profile states is not
    written: a stated ``false`` turns the vocabulary off and a stated path is the
    location.

    Args:
        facility_dir: The build's ``data/facility`` directory.
        profile_config: The profile's ``config:`` block.

    Returns:
        The keys to write; empty when the tree ships neither.

    Raises:
        FacilityBuildError: ``profile-invalid`` when the profile turns the
            vocabulary on at no path of its own and the tree ships none, or
            states a benchmark dataset path and the tree ships no query sources.
    """
    from osprey.cli.build_profile_archiver import _expand_dotted
    from osprey.deployment.reach import dotted_get

    profile = _expand_dotted(profile_config)
    fill: dict[str, Any] = {}

    enabled = dotted_get(profile, VOCABULARY_ENABLED_KEY)
    stated_path = dotted_get(profile, VOCABULARY_PATH_KEY)
    shipped = (facility_dir / "ariel" / "vocabulary.yml").is_file()
    if enabled is True and stated_path is None and not shipped:
        raise _invalid(
            f"is true and the facility ships no {VOCABULARY_SOURCE}",
            f"add {VOCABULARY_SOURCE}, or remove `{VOCABULARY_ENABLED_KEY}`",
            VOCABULARY_ENABLED_KEY,
            PROFILE_SOURCE,
        )
    if shipped and enabled is not False:
        if enabled is None:
            fill[VOCABULARY_ENABLED_KEY] = True
        if stated_path is None:
            fill[VOCABULARY_PATH_KEY] = VOCABULARY_SOURCE

    stated_dataset = dotted_get(profile, BENCHMARK_DATASET_KEY)
    has_sources = (facility_dir / QUERY_SOURCES).is_dir()
    if stated_dataset is not None and not has_sources:
        raise _invalid(
            f"names a benchmark query file and the facility ships no data/facility/{QUERY_SOURCES}/",
            f"add the query sources to data/facility/{QUERY_SOURCES}/, "
            f"or remove `{BENCHMARK_DATASET_KEY}`",
            BENCHMARK_DATASET_KEY,
            PROFILE_SOURCE,
        )
    if has_sources and stated_dataset is None:
        fill[BENCHMARK_DATASET_KEY] = BENCHMARK_DATASET_PATH
    return fill


def _carries_the_simulator_block(
    rendered_config: Mapping[str, Any], profile: Mapping[str, Any]
) -> bool:
    """Whether the render configures a simulator: a key under its block makes it a target."""
    block = f"control_system.connector.{VIRTUAL_ACCELERATOR}"
    return bool(_dotted(profile, block) or _dotted(rendered_config, block))


def _stated_freshness_refusal(address: str, records: FacilityRecords) -> FacilityBuildError:
    readbacks = _readbacks(records.of_kind("channel"))
    remedy = (
        f"set it to a readback the facility file holds, such as `{readbacks[0]}`, or remove it"
        if readbacks
        else "remove it"
    )
    return _invalid(
        f"`{address}` is not a channel of the facility file",
        remedy,
        FRESHNESS_CHANNEL_KEY,
        PROFILE_SOURCE,
    )


def _served_scope(records: FacilityRecords, served: list[str]) -> list[Mapping[str, Any]]:
    """The channels the served simulator holds.

    Each served physics model's selection, plus every channel in no model's
    selection, served or not: a channel on a device the file assigns to an
    unserved model is that model's and is not served. With no physics model
    served, every channel.
    """
    channels = records.of_kind("channel")
    physics = [name for name in served if name != TEXTURE]
    if not physics:
        return list(channels)

    def selection(model: str) -> set[str]:
        selected = records.select_channels(ChannelSelection(model=model))
        return {str(channel["id"]) for channel in selected}

    owned = set().union(*(selection(name) for name in records.ids("model") - {TEXTURE}))
    in_scope = set().union(*(selection(name) for name in physics)) | {
        str(channel["id"]) for channel in channels if str(channel["id"]) not in owned
    }
    return [channel for channel in channels if str(channel["id"]) in in_scope]


def _readback_channels(channels: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The readbacks of ``channels``, by address: a role absent is the default role."""
    return sorted(
        (channel for channel in channels if channel.get("role", DEFAULT_ROLE) == "readback"),
        key=lambda channel: str(channel["id"]),
    )


def _readbacks(channels: Iterable[Mapping[str, Any]]) -> list[str]:
    """The addresses of the readbacks of ``channels``, in order."""
    return [str(channel["id"]) for channel in _readback_channels(channels)]


def _dotted(mapping: Mapping[str, Any], key: str) -> Any:
    """The value at a dotted ``key`` of a nested mapping, or ``None``."""
    node: Any = mapping
    for part in key.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node
