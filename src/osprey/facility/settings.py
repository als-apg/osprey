"""The profile settings whose value is a fact of the facility.

Each is derived from the facility file or tree when the profile leaves it
unstated, and a stated value is an override checked against the facility.
:data:`FACILITY_SETTINGS` is the list, one row per setting, and
:func:`fill_facility_settings` is its one run: the build calls it once per
render, at the derived-keys pass, and nothing else derives these keys or checks
a stated value for them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from osprey.facility import DEFAULT_ROLE
from osprey.facility.errors import FacilityBuildError, quoted_slots
from osprey.facility.records import FacilityRecords
from osprey.facility.served import SIMULATION_MODELS_KEY, _served_scope, resolve_served
from osprey.facility.views.benchmarks import BENCHMARK_DATASET_KEY, QUERY_SOURCES
from osprey.simulation import DEFAULT_SCENARIOS_KEY
from osprey.utils.config_paths import resolve_config_relative_path
from osprey_connectors.simulation import STILL_SCENARIO
from osprey_connectors.types import LIVE_STANDIN, VIRTUAL_ACCELERATOR
from osprey_connectors.workspace import BUILD_DIR_NAME

if TYPE_CHECKING:
    from osprey.cli.build_profile_archiver import VAArchiverConfig

    #: Every profile setting whose value is a fact of the facility, in the
    #: order their checks run; built on first read by the module's
    #: ``__getattr__``.
    FACILITY_SETTINGS: tuple[FacilitySetting, ...]

__all__ = [
    "BENCHMARK_DATASET_PATH",
    "FACILITY_SETTINGS",
    "FRESHNESS_CHANNEL_KEY",
    "NO_READBACK_WARNING",
    "PROFILE_SOURCE",
    "SERVED_PROBE_KEYS",
    "VOCABULARY_ENABLED_KEY",
    "VOCABULARY_PATH_KEY",
    "VOCABULARY_SOURCE",
    "FacilitySetting",
    "fill_facility_settings",
]

#: The file a profile key is stated in.
PROFILE_SOURCE = "profile.yml"

#: The profile key that states the archive's freshness channel.
FRESHNESS_CHANNEL_KEY = "va_archiver.freshness_channel"

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


class _Inputs:
    """What every row reads: the facility, the render and the expanded profile.

    The served readbacks are computed once per run and only for a render that
    serves the simulator or carries a ``va_archiver:`` block, so a build whose
    served models hold none says so once.
    """

    def __init__(
        self,
        records: FacilityRecords,
        facility_dir: Path,
        project_root: Path,
        rendered_config: Mapping[str, Any],
        profile_config: Mapping[str, Any],
        va_archiver: VAArchiverConfig | None,
        reported: set[str] | None,
    ) -> None:
        from osprey.cli.build_profile_archiver import _expand_dotted

        self.records = records
        self.facility_dir = facility_dir
        self.project_root = project_root
        self.rendered_config = rendered_config
        self.profile_config = profile_config
        self.profile: Mapping[str, Any] = _expand_dotted(profile_config)
        self.va_archiver = va_archiver
        self.reported = reported

    def stated(self, key: str) -> Any:
        """The value the profile states at a dotted ``key``, or ``None``."""
        from osprey.deployment.reach import dotted_get

        return dotted_get(self.profile, key)

    @cached_property
    def serves_simulator(self) -> bool:
        """Whether the render configures a simulator: a key under its block makes it a target."""
        from osprey.deployment.reach import dotted_get

        block = f"control_system.connector.{VIRTUAL_ACCELERATOR}"
        return bool(dotted_get(self.profile, block) or dotted_get(self.rendered_config, block))

    @cached_property
    def readbacks(self) -> list[Mapping[str, Any]]:
        """The served simulator's readbacks by address; empty, and said once, when it holds none."""
        if not self.serves_simulator and self.va_archiver is None:
            return []
        models = self.stated(SIMULATION_MODELS_KEY)
        if models is None:
            served = resolve_served(self.rendered_config, self.records)
        else:
            served = resolve_served({"simulation": {"models": models}}, self.records)
        readbacks = _readback_channels(_served_scope(self.records, served))
        if not readbacks and (self.reported is None or NO_READBACK_WARNING not in self.reported):
            click.echo(NO_READBACK_WARNING, err=True)
            if self.reported is not None:
                self.reported.add(NO_READBACK_WARNING)
        return readbacks

    @cached_property
    def vocabulary_shipped(self) -> bool:
        """Whether the facility tree ships ARIEL's vocabulary."""
        return (self.facility_dir / "ariel" / "vocabulary.yml").is_file()

    @cached_property
    def query_sources_shipped(self) -> bool:
        """Whether the facility tree ships benchmark query sources."""
        return (self.facility_dir / QUERY_SOURCES).is_dir()


@dataclass(frozen=True)
class FacilitySetting:
    """One profile setting whose value is a fact of the facility.

    Attributes:
        key: The rendered dotted key the row writes.
        stated_at: Where a profile states an override: ``key`` itself, or the
            profile key the rendered one is built from.
        derive: The value to write when the profile leaves the setting
            unstated, ``None`` to write nothing; ``None`` for a setting the
            build never derives.
        check: Refuses a stated value the facility contradicts; an unstated
            setting passes.
    """

    key: str
    stated_at: str
    derive: Callable[[_Inputs], Any] | None
    check: Callable[[_Inputs], None]


def _refusal(key: str, detail: str, remedy: str) -> FacilityBuildError:
    return FacilityBuildError(
        "profile-invalid", key, [PROFILE_SOURCE], remedy, record_kind="path", detail=detail
    )


def _readback_channels(channels: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The readbacks of ``channels``, by address: a role absent is the default role."""
    return sorted(
        (channel for channel in channels if channel.get("role", DEFAULT_ROLE) == "readback"),
        key=lambda channel: str(channel["id"]),
    )


def _unheld(inputs: _Inputs, address: Any) -> bool:
    """Whether ``address`` is stated and is not a channel of the facility file."""
    return address is not None and inputs.records.record("channel", str(address)) is None


def _first_readback(inputs: _Inputs) -> str | None:
    """The facility file's first readback by address, the remedy a refused channel names."""
    readbacks = _readback_channels(inputs.records.of_kind("channel"))
    return str(readbacks[0]["id"]) if readbacks else None


def _not_a_channel(key: str, address: Any, remedy: str) -> FacilityBuildError:
    return _refusal(key, f"`{address}` is not a channel of the facility file", remedy)


def _check_probe(inputs: _Inputs) -> None:
    for key in SERVED_PROBE_KEYS:
        address = inputs.stated(key)
        if not _unheld(inputs, address):
            continue
        readback = _first_readback(inputs)
        raise _not_a_channel(
            key,
            address,
            f"set `{key}` to a readback the facility file holds, such as `{readback}`"
            if readback
            else f"remove `{key}` from the profile",
        )


def _derive_probe(inputs: _Inputs) -> Any:
    if not inputs.serves_simulator:
        return None
    # Read before the stated value, so a build whose served models hold no
    # readback says so whether or not it states its own probe.
    readbacks = inputs.readbacks
    if not readbacks or inputs.stated(SERVED_PROBE_KEYS[0]) is not None:
        return None
    return readbacks[0]["id"]


def _stated_freshness_channel(inputs: _Inputs) -> str | None:
    return inputs.va_archiver.freshness_channel if inputs.va_archiver is not None else None


def _check_freshness(inputs: _Inputs) -> None:
    address = _stated_freshness_channel(inputs) or None
    if not _unheld(inputs, address):
        return
    readback = _first_readback(inputs)
    raise _not_a_channel(
        FRESHNESS_CHANNEL_KEY,
        address,
        f"set it to a readback the facility file holds, such as `{readback}`, or remove it"
        if readback
        else "remove it",
    )


def _derive_freshness(inputs: _Inputs) -> Any:
    from osprey.cli.build_profile_archiver import (
        FRESHNESS_CONFIG_KEY,
        _addresses_freshness_category,
        freshness_check_override,
    )

    if inputs.va_archiver is None:
        return None
    # Read before the stated value, as the probe's derive does.
    readbacks = inputs.readbacks
    if (
        not readbacks
        or _stated_freshness_channel(inputs)
        or _addresses_freshness_category(inputs.profile_config)
    ):
        return None
    derived = replace(inputs.va_archiver, freshness_channel=str(readbacks[0]["id"]))
    return freshness_check_override(derived)[FRESHNESS_CONFIG_KEY]


def _check_start_set(inputs: _Inputs) -> None:
    stated = inputs.stated(DEFAULT_SCENARIOS_KEY)
    if not isinstance(stated, list):
        return
    # The scenarios view adds the built-in still to every file that defines none.
    known = sorted(
        {str(s["name"]) for s in inputs.records.document.get("scenarios", [])} | {STILL_SCENARIO}
    )
    unknown = sorted({str(name) for name in stated} - set(known))
    if unknown:
        raise _refusal(
            DEFAULT_SCENARIOS_KEY,
            f"{quoted_slots(unknown)} is not a scenario the facility serves",
            f"set `{DEFAULT_SCENARIOS_KEY}` to `[]` or to scenarios it serves: "
            f"{quoted_slots(known)}",
        )


def _check_vocabulary_enabled(inputs: _Inputs) -> None:
    if (
        inputs.stated(VOCABULARY_ENABLED_KEY) is True
        and inputs.stated(VOCABULARY_PATH_KEY) is None
        and not inputs.vocabulary_shipped
    ):
        raise _refusal(
            VOCABULARY_ENABLED_KEY,
            f"is true and the facility ships no {VOCABULARY_SOURCE}",
            f"add {VOCABULARY_SOURCE}, or remove `{VOCABULARY_ENABLED_KEY}`",
        )


def _derive_vocabulary_enabled(inputs: _Inputs) -> Any:
    if inputs.vocabulary_shipped and inputs.stated(VOCABULARY_ENABLED_KEY) is None:
        return True
    return None


def _check_vocabulary_path(inputs: _Inputs) -> None:
    stated = inputs.stated(VOCABULARY_PATH_KEY)
    enabled = inputs.stated(VOCABULARY_ENABLED_KEY)
    if stated is None or not (enabled is True or (enabled is None and inputs.vocabulary_shipped)):
        return
    # Resolved as ARIEL resolves it from the host render's config.yml: a
    # relative path anchors on the project root, an absolute one is itself.
    if (
        not isinstance(stated, str)
        or not resolve_config_relative_path(stated, inputs.project_root / BUILD_DIR_NAME).is_file()
    ):
        raise _refusal(
            VOCABULARY_PATH_KEY,
            f"`{stated}` is not a file",
            f"set `{VOCABULARY_PATH_KEY}` to a vocabulary file, or remove it",
        )


def _derive_vocabulary_path(inputs: _Inputs) -> Any:
    if (
        inputs.vocabulary_shipped
        and inputs.stated(VOCABULARY_ENABLED_KEY) is not False
        and inputs.stated(VOCABULARY_PATH_KEY) is None
    ):
        return VOCABULARY_SOURCE
    return None


def _check_dataset(inputs: _Inputs) -> None:
    if inputs.stated(BENCHMARK_DATASET_KEY) is not None and not inputs.query_sources_shipped:
        raise _refusal(
            BENCHMARK_DATASET_KEY,
            f"names a benchmark query file and the facility ships no data/facility/{QUERY_SOURCES}/",
            f"add the query sources to data/facility/{QUERY_SOURCES}/, "
            f"or remove `{BENCHMARK_DATASET_KEY}`",
        )


def _derive_dataset(inputs: _Inputs) -> Any:
    if inputs.query_sources_shipped and inputs.stated(BENCHMARK_DATASET_KEY) is None:
        return BENCHMARK_DATASET_PATH
    return None


@cache
def _facility_settings() -> tuple[FacilitySetting, ...]:
    """Every profile setting whose value is a fact of the facility, in the order their checks run."""
    from osprey.cli.build_profile_archiver import FRESHNESS_CONFIG_KEY

    return (
        FacilitySetting(SERVED_PROBE_KEYS[0], SERVED_PROBE_KEYS[0], _derive_probe, _check_probe),
        FacilitySetting(
            FRESHNESS_CONFIG_KEY, FRESHNESS_CHANNEL_KEY, _derive_freshness, _check_freshness
        ),
        FacilitySetting(DEFAULT_SCENARIOS_KEY, DEFAULT_SCENARIOS_KEY, None, _check_start_set),
        FacilitySetting(
            VOCABULARY_ENABLED_KEY,
            VOCABULARY_ENABLED_KEY,
            _derive_vocabulary_enabled,
            _check_vocabulary_enabled,
        ),
        FacilitySetting(
            VOCABULARY_PATH_KEY,
            VOCABULARY_PATH_KEY,
            _derive_vocabulary_path,
            _check_vocabulary_path,
        ),
        FacilitySetting(
            BENCHMARK_DATASET_KEY, BENCHMARK_DATASET_KEY, _derive_dataset, _check_dataset
        ),
    )


def __getattr__(name: str) -> Any:
    # The freshness row's key is the archiver module's, and that module loads
    # the CLI package: the table is built on first read so that importing this
    # module does not.
    if name == "FACILITY_SETTINGS":
        return _facility_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def fill_facility_settings(
    records: FacilityRecords,
    facility_dir: Path,
    project_root: Path,
    rendered_config: Mapping[str, Any],
    profile_config: Mapping[str, Any],
    va_archiver: VAArchiverConfig | None,
    *,
    reported: set[str] | None = None,
) -> dict[str, Any]:
    """Check every stated facility setting, then derive every unstated one.

    Args:
        records: The build's facility file's records.
        facility_dir: The build's ``data/facility`` directory.
        project_root: The repository the build renders, which a stated
            relative path is resolved against.
        rendered_config: The render's ``config.yml`` before the profile's overlay.
        profile_config: The profile's ``config:`` block, laid over it.
        va_archiver: The profile's parsed ``va_archiver:`` block, or ``None``.
        reported: The notes this build has printed; a warning in it is not
            printed again and one printed is added. ``None`` prints it.

    Returns:
        The dotted keys to write, each row's derived value under its key.

    Raises:
        FacilityBuildError: ``profile-invalid`` naming the first setting whose
            stated value the facility contradicts, or ``simulation.models``
            when the served models do not resolve.
    """
    inputs = _Inputs(
        records, facility_dir, project_root, rendered_config, profile_config, va_archiver, reported
    )
    for row in _facility_settings():
        row.check(inputs)
    fill: dict[str, Any] = {}
    for row in _facility_settings():
        value = row.derive(inputs) if row.derive is not None else None
        if value is not None:
            fill[row.key] = value
    return fill
