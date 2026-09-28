"""The facility build's checks, run in fixed stages.

A stage runs only when every earlier stage is clean::

    S1 load        per-file parse, unknown keys, computed slots, PN_LOCAL,
                   model names equal but for case          (sources.py)
    S2 combine     layer merge and fixes.yaml              (combine.py)
    S3 schema      the combined file against the generated model
    S4 references  every id a record names exists; device classes and
                   signal roles are known
    S5 records     pair, value, limits and seed rules
    S6 compute     spans, places, wiring, engines          (later stages)
    S7 views       the views the profile asks for          (later stages)

``validate`` prints every error of the first failing stage, sorted by (table
row, record kind, id); ``build`` stops on the first of those lines. Each fault
yields one line of one kind: a slot naming a missing id is ``reference-missing``
only, because S5 never runs on a file whose references do not resolve, and a
non-float channel with motion is ``value-invalid`` only.

Every error is a ``FacilityBuildError``; the stages collect them and never
assemble a line by hand.
"""

from __future__ import annotations

import copy
import json
import math
import numbers
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import IO, Any

import click

from osprey.facility import fold_code
from osprey.facility.combine import FIXES_FILE, CombineResult, combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.sources import Sources, load_sources

__all__ = [
    "FACILITY_HEADER",
    "SET_VALUED_SLOTS",
    "STAGES",
    "Stage",
    "StageReport",
    "Validated",
    "check_records",
    "check_references",
    "check_schema",
    "known_classes",
    "ordered_slots",
    "report",
    "run_stages",
    "schema_document",
    "signal_roles",
    "sort_errors",
    "validate",
]

#: The stages, in the order they run.
STAGES: tuple[str, ...] = (
    "load",
    "combine",
    "schema",
    "references",
    "records",
    "compute",
    "views",
)

#: The header of the combined facility file.
FACILITY_HEADER = "osprey.facility.facility/1"

#: The multivalued slots that are sets: compared and written sorted by string form.
SET_VALUED_SLOTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Channel", "tags"),
        ("Channel", "former_addresses"),
        ("Channel", "endpoint_of"),
        ("Group", "members"),
        ("Device", "groups"),
        ("Measurement", "kinds"),
    }
)

#: The order errors print in: one entry per row of the error table.
_TABLE_ROWS: tuple[tuple[str, ...], ...] = (
    ("source-invalid",),
    ("layer-conflict", "layer-duplicate"),
    (
        "fix-missing",
        "fix-stale",
        "fix-duplicate",
        "fix-computed",
        "fix-authored",
        "fix-referenced",
    ),
    ("reference-missing",),
    ("class-unknown",),
    ("pair-invalid",),
    ("value-invalid",),
    ("seed-invalid",),
    ("limit-invalid",),
    ("place-conflict", "span-invalid", "wiring-conflict"),
    ("engine-missing", "engine-invalid"),
    ("model-conflict",),
)
_ROW: dict[str, int] = {kind: row for row, kinds in enumerate(_TABLE_ROWS) for kind in kinds}

#: The largest number of labels a bool or enum channel carries.
_MAX_OPTIONS = 16
#: The longest label, in ASCII characters.
_MAX_LABEL = 25

_NUMERIC_TYPES = ("float", "int")
_LIMIT_BOUNDS = ("min_value", "max_value", "max_step")
_MOTION = ("noise", "drift")
_STUCK = "stuck"


# --- stage results ---------------------------------------------------------------


@dataclass
class Validated:
    """What the stages have produced so far.

    Attributes:
        facility_dir: The ``data/facility`` directory.
        project_name: The project's name; the zero-source identity folds it.
        sources: What S1 read.
        combined: What S2 combined.
        document: The combined file with its header and identity, the shape
            S3 validated; set once S2 is clean.
    """

    facility_dir: Path
    project_name: str
    sources: Sources | None = None
    combined: CombineResult | None = None
    document: dict[str, Any] | None = None


#: A later stage: reads what the earlier stages produced, returns its errors.
Stage = Callable[[Validated], list[FacilityBuildError]]


@dataclass
class StageReport:
    """The outcome of running the stages.

    Attributes:
        failed: The first stage with an error, or ``None`` when every stage ran
            clean.
        errors: Every error of that stage, sorted by ``sort_errors``.
        validated: What the stages that ran produced.
    """

    failed: str | None
    errors: list[FacilityBuildError]
    validated: Validated = field(repr=False)

    @property
    def ok(self) -> bool:
        """True when every stage ran clean."""
        return self.failed is None

    def raise_first(self) -> None:
        """Raise the first error, as ``build`` stops on it.

        Raises:
            FacilityBuildError: The first sorted error, when any stage failed.
        """
        if self.errors:
            raise self.errors[0]


def sort_errors(errors: Iterable[FacilityBuildError]) -> list[FacilityBuildError]:
    """Sort errors by (table row, record kind, id), keeping ties in found order.

    Args:
        errors: The errors of one stage.

    Returns:
        A new sorted list.
    """
    return sorted(
        errors, key=lambda e: (_ROW.get(e.kind, len(_TABLE_ROWS)), e.record_kind, e.record_id)
    )


def run_stages(
    facility_dir: Path,
    *,
    project_name: str,
    later: Sequence[tuple[str, Stage]] = (),
) -> StageReport:
    """Run S1 to S5, then each later stage, stopping at the first that fails.

    Args:
        facility_dir: The ``data/facility`` directory; a missing one is zero
            sources.
        project_name: The project's name, folded into the identity code when
            there is no ``identity.yaml``.
        later: ``(stage name, check)`` for the stages after S5, in order.

    Returns:
        The first failing stage and all of its errors, or a clean report.
    """
    validated = Validated(facility_dir, project_name)
    loaded = load_sources(facility_dir)
    validated.sources = loaded.sources
    if loaded.errors:
        return _failed("load", loaded.errors, validated)
    combined = combine(loaded.sources)
    validated.combined = combined
    if combined.errors:
        return _failed("combine", combined.errors, validated)
    document = schema_document(combined.document, loaded.sources, project_name=project_name)
    validated.document = document
    checks: list[tuple[str, Stage]] = [
        ("schema", lambda v: check_schema(_need(v.document), _need(v.sources))),
        ("references", lambda v: check_references(_need(v.sources), _need(v.combined))),
        ("records", lambda v: check_records(_need(v.document))),
        *later,
    ]
    for name, check in checks:
        errors = check(validated)
        if errors:
            return _failed(name, errors, validated)
    return StageReport(None, [], validated)


def _need(value: Any) -> Any:
    if value is None:
        raise RuntimeError("a stage ran before the stage that produces its input")
    return value


def _failed(stage: str, errors: list[FacilityBuildError], validated: Validated) -> StageReport:
    return StageReport(stage, sort_errors(errors), validated)


def report(errors: Iterable[FacilityBuildError], file: IO[Any] | None = None) -> None:
    """Print each error's line, to stderr unless a stream is given.

    Args:
        errors: The errors, in print order.
        file: The stream to write to.
    """
    for error in errors:
        if file is None:
            click.echo(error.format_message(), err=True)
        else:
            click.echo(error.format_message(), file=file)


def validate(facility_dir: Path, *, project_name: str, file: IO[Any] | None = None) -> int:
    """Check a facility directory and print every error of the first failing stage.

    Args:
        facility_dir: The ``data/facility`` directory.
        project_name: The project's name.
        file: The stream the lines go to; stderr when omitted.

    Returns:
        The exit code: 0 when every stage is clean, else 1.
    """
    result = run_stages(facility_dir, project_name=project_name)
    report(result.errors, file)
    return 0 if result.ok else 1


# --- vocabulary --------------------------------------------------------------------


@cache
def _vocabulary() -> dict[str, Any]:
    table = resources.files("osprey.facility.schema._generated") / "vocabulary.json"
    data: dict[str, Any] = json.loads(table.read_text(encoding="utf-8"))
    return data


def signal_roles() -> frozenset[str]:
    """The signal roles of the vocabulary."""
    return frozenset(role["name"] for role in _vocabulary()["signal_roles"])


def known_classes(classes: Iterable[Mapping[str, Any]] = ()) -> frozenset[str]:
    """The vocabulary's device classes joined with the facility-added ones.

    Args:
        classes: ``classes.yaml``'s records.

    Returns:
        Every class name a device may carry.
    """
    names = {row["name"] for row in _vocabulary()["classes"]}
    names.update(str(row["class"]) for row in classes if isinstance(row.get("class"), str))
    return frozenset(names)


# --- ordered slots -----------------------------------------------------------------


@cache
def ordered_slots() -> frozenset[tuple[str, str]]:
    """The multivalued slots whose order carries meaning, as ``(class, slot)``.

    Read from the generated model: a slot is ordered when its field's
    ``linkml_meta`` sets ``list_elements_ordered``. Every other multivalued slot
    is one of ``SET_VALUED_SLOTS``.

    Returns:
        The ordered slots of every class of the facility file.
    """
    import pydantic

    from osprey.facility.schema import core

    ordered = set()
    for name, cls in sorted(vars(core).items()):
        if not (isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)):
            continue
        if cls.__module__ != core.__name__:
            continue
        for slot, info in cls.model_fields.items():
            extra = info.json_schema_extra
            meta = extra.get("linkml_meta", {}) if isinstance(extra, dict) else {}
            if isinstance(meta, dict) and meta.get("list_elements_ordered"):
                ordered.add((name, slot))
    return frozenset(ordered)


# --- S3: schema --------------------------------------------------------------------


def schema_document(
    document: Mapping[str, Any], sources: Sources, *, project_name: str
) -> dict[str, Any]:
    """The combined file with its header and identity filled.

    ``identity`` is ``identity.yaml``'s record, or ``{code, name}`` folded from
    the project name when there is none. Neither argument is modified.

    Args:
        document: S2's combined document.
        sources: What S1 read.
        project_name: The project's name.

    Returns:
        A new document that starts with ``schema`` and ``identity``.
    """
    body = copy.deepcopy(dict(document))
    body.pop("identity", None)
    if sources.identity is not None:
        identity = copy.deepcopy(sources.identity)
    else:
        identity = {"code": fold_code(project_name), "name": project_name}
    return {"schema": FACILITY_HEADER, "identity": identity, **body}


def check_schema(document: Mapping[str, Any], sources: Sources) -> list[FacilityBuildError]:
    """Validate the filled document against the generated model (stage S3).

    Each failing path is one ``source-invalid`` line naming the dotted path. The
    slots typed ``Any`` (scenario ``overrides`` and ``faults``) are checked for
    being mappings here too.

    Args:
        document: The document from ``schema_document``.
        sources: What S1 read, to name each path's source file.

    Returns:
        Every error, one per path.
    """
    import pydantic

    from osprey.facility.schema import Facility

    failures: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    try:
        Facility.model_validate(document)
    except pydantic.ValidationError as exc:
        for item in exc.errors():
            failures.setdefault(_path(item["loc"]), item)
    errors = [_schema_error(document, sources, loc, item) for loc, item in failures.items()]
    for index, scenario in enumerate(document.get("scenarios") or []):
        if not isinstance(scenario, dict):
            continue
        rel = _scenario_file(scenario)
        for slot, depth in (("overrides", 1), ("faults", 2)):
            value = scenario.get(slot)
            if value is not None and not _mapping_of_depth(value, depth):
                shape = "a mapping" if depth == 1 else "a mapping of model to a mapping"
                errors.append(
                    FacilityBuildError(
                        "source-invalid",
                        f"scenarios.{index}.{slot}",
                        [rel],
                        f"write `{slot}` in {rel} as {shape}",
                        record_kind="path",
                        detail=f"{rel} `{slot}` is not {shape}",
                    )
                )
    return errors


def _mapping_of_depth(value: Any, depth: int) -> bool:
    if not isinstance(value, dict):
        return False
    return depth == 1 or all(_mapping_of_depth(v, depth - 1) for v in value.values())


#: The tags pydantic appends to a location for each member of a union.
_UNION_TAGS = frozenset({"str", "float", "int", "bool", "SignalSentence", "LinearTerm"})


def _path(loc: Sequence[Any]) -> tuple[Any, ...]:
    """A location without the union-member tags, so one path is one line."""
    for position, part in enumerate(loc):
        if position >= 2 and part in _UNION_TAGS:
            return tuple(loc[:position])
    return tuple(loc)


def _schema_error(
    document: Mapping[str, Any], sources: Sources, loc: tuple[Any, ...], item: Mapping[str, Any]
) -> FacilityBuildError:
    dotted = ".".join(str(part) for part in loc)
    files = _path_files(document, sources, loc)
    where = ", ".join(files)
    slot = str(loc[-1]) if loc else "the file"
    message = str(item.get("msg", "is invalid"))
    message = message[:1].lower() + message[1:]
    if item.get("type") == "missing":
        remedy = f"add `{slot}` to {where}"
    elif item.get("type") == "extra_forbidden":
        remedy = f"remove `{slot}` from {where}"
    else:
        remedy = f"correct `{slot}` in {where}"
    return FacilityBuildError(
        "source-invalid",
        dotted,
        files,
        remedy,
        record_kind="path",
        detail=f"{where}: {message}",
    )


_TOP_SLOT_FILES = {"limits": "limits.yaml", "classes": "classes.yaml", "identity": "identity.yaml"}


def _path_files(document: Mapping[str, Any], sources: Sources, loc: tuple[Any, ...]) -> list[str]:
    """The source files the value at a location came from."""
    if not loc:
        return ["facility"]
    top = loc[0]
    if top in _TOP_SLOT_FILES:
        return [_TOP_SLOT_FILES[str(top)]]
    items = document.get(str(top))
    if len(loc) < 2 or not isinstance(items, list) or not isinstance(loc[1], int):
        return ["facility"]
    record = items[loc[1]] if loc[1] < len(items) else None
    if not isinstance(record, dict):
        return ["facility"]
    if top == "scenarios":
        return [_scenario_file(record)]
    rest = loc[2:]
    if (
        top == "models"
        and rest[:1] == ("measurement",)
        and record.get("name") in (sources.measurement)
    ):
        return [f"measurement/{record['name']}.yaml"]
    if top == "models" and rest[:1] == ("wiring",) and len(rest) >= 2:
        wiring = record.get("wiring") or []
        position = rest[1]
        if isinstance(position, int) and position < len(wiring):
            record, rest = wiring[position], rest[2:]
    if (
        top == "channels"
        and rest[:1] == ("simulation",)
        and str(record.get("id")) in (sources.seeds)
    ):
        return ["seeds.yaml"]
    return _files(record, str(rest[0]) if rest else None)


def _scenario_file(scenario: Mapping[str, Any]) -> str:
    return f"scenarios/{scenario.get('name')}.yaml"


def _files(record: Mapping[str, Any], slot: str | None) -> list[str]:
    """The files that state a slot of a combined record, fixes.yaml for a fix."""
    provenance = record.get("provenance")
    sources = provenance.get("sources", []) if isinstance(provenance, dict) else []
    stating = sorted(
        {
            str(source["file"])
            for source in sources
            if isinstance(source, dict) and (slot is None or slot in source.get("fields", ()))
        }
    )
    if stating:
        return stating
    fixes = provenance.get("fixes") if isinstance(provenance, dict) else None
    if fixes:
        return [FIXES_FILE]
    every = sorted({str(s["file"]) for s in sources if isinstance(s, dict)})
    return every or [FIXES_FILE]


# --- the combined file, indexed ------------------------------------------------------


class _Index:
    """The combined document's records by id, and who wires what."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = document
        self.places = _by_key(document.get("places"), "id")
        self.devices = _by_key(document.get("devices"), "id")
        self.channels = _by_key(document.get("channels"), "id")
        self.groups = _by_key(document.get("groups"), "id")
        self.models = _by_key(document.get("models"), "name")
        self.wiring: list[tuple[str, dict[str, Any]]] = []
        self.wired: dict[str, set[str]] = {}
        for name, model in sorted(self.models.items()):
            for entry in model.get("wiring") or []:
                if isinstance(entry, dict):
                    self.wiring.append((name, entry))
                    address = entry.get("address")
                    if isinstance(address, str):
                        self.wired.setdefault(address, set()).add(name)
        limits = document.get("limits")
        records = limits.get("records") if isinstance(limits, dict) else None
        self.limits: list[dict[str, Any]] = [r for r in records or [] if isinstance(r, dict)]
        self.former: dict[str, str] = {}
        for address, channel in sorted(self.channels.items()):
            for old in channel.get("former_addresses") or []:
                self.former.setdefault(str(old), address)


def _by_key(rows: Any, key: str) -> dict[str, dict[str, Any]]:
    return {
        str(row[key]): row
        for row in rows or []
        if isinstance(row, dict) and isinstance(row.get(key), str)
    }


# --- S4: references ------------------------------------------------------------------


def check_references(sources: Sources, combined: CombineResult) -> list[FacilityBuildError]:
    """Check that every id a record names exists (stage S4).

    Every referencing slot of the file is checked, plus the entries the combine
    attaches by address or model name without checking them (``seeds.yaml``,
    measurement files). A missing id is ``reference-missing``, naming the file
    and field, and the new address when the id is a channel's former address or
    the fix when a ``drop`` removed it. A device class or a channel signal the
    vocabulary does not know is ``class-unknown``.

    Args:
        sources: What S1 read.
        combined: What S2 combined, with its ``dropped`` records.

    Returns:
        Every error found.
    """
    return list(_References(sources, combined).run())


class _References:
    def __init__(self, sources: Sources, combined: CombineResult) -> None:
        self.sources = sources
        self.dropped = combined.dropped
        self.index = _Index(combined.document)

    def run(self) -> Iterator[FacilityBuildError]:
        yield from self._classes()
        yield from self._places()
        yield from self._devices()
        yield from self._channels()
        yield from self._groups()
        yield from self._wiring()
        yield from self._seeds()
        yield from self._limits()
        yield from self._scenarios()
        yield from self._measurement()

    def _exists(self, kind: str, rid: str) -> bool:
        table = {
            "place": self.index.places,
            "device": self.index.devices,
            "channel": self.index.channels,
            "group": self.index.groups,
            "model": self.index.models,
        }[kind]
        return rid in table

    def _missing(
        self,
        record_kind: str,
        record_id: str,
        files: Sequence[str],
        slot: str,
        kind: str,
        target: Any,
    ) -> Iterator[FacilityBuildError]:
        """Yield one ``reference-missing`` when ``target`` names no record."""
        target = str(target)
        if self._exists(kind, target):
            return
        where = f"{', '.join(files)} `{slot}` names {kind} {target}"
        remedy = f"add {kind} {target} or correct `{slot}`"
        drop = self.dropped.get((kind, target))
        moved = self.index.former.get(target) if kind == "channel" else None
        if drop is not None:
            detail = (
                f"{where}, dropped by fix {drop['op']} {drop['kind']} {drop['id']}: {drop['why']}"
            )
            remedy = "remove the entry or drop the fix"
            files = [*files, FIXES_FILE]
        elif moved is not None:
            detail = f"{where}, which does not exist; channel {moved} lists it in former_addresses"
            remedy = f"point `{slot}` at {moved}"
        else:
            detail = f"{where}, which does not exist"
        yield FacilityBuildError(
            "reference-missing", record_id, files, remedy, record_kind=record_kind, detail=detail
        )

    def _classes(self) -> Iterator[FacilityBuildError]:
        known = set(known_classes())
        for row in self.sources.classes:
            name = str(row.get("class"))
            parent = row.get("parent")
            if parent not in known:
                yield FacilityBuildError(
                    "class-unknown",
                    name,
                    ["classes.yaml"],
                    "name a vocabulary class or a class listed earlier in classes.yaml",
                    record_kind="class",
                    detail=f"classes.yaml `parent` {parent} is neither a vocabulary class "
                    "nor an earlier facility-added class",
                )
            known.add(name)

    def _places(self) -> Iterator[FacilityBuildError]:
        for pid, place in sorted(self.index.places.items()):
            if "/" in pid:
                parent = pid.rsplit("/", 1)[0]
                if parent not in self.index.places:
                    files = _files(place, None)
                    yield from self._missing("place", pid, files, "id", "place", parent)
            span = place.get("span")
            if isinstance(span, dict) and "model" in span:
                files = _files(place, "span")
                yield from self._missing("place", pid, files, "span.model", "model", span["model"])

    def _devices(self) -> Iterator[FacilityBuildError]:
        known = known_classes(self.sources.classes)
        for did, device in sorted(self.index.devices.items()):
            if "place" in device:
                files = _files(device, "place")
                yield from self._missing("device", did, files, "place", "place", device["place"])
            cls = device.get("class")
            if cls is not None and cls not in known:
                yield FacilityBuildError(
                    "class-unknown",
                    did,
                    _files(device, "class"),
                    "use a vocabulary class or add it to classes.yaml",
                    record_kind="device",
                    detail=f"class {cls} is in neither the vocabulary nor classes.yaml",
                )

    def _channels(self) -> Iterator[FacilityBuildError]:
        roles = signal_roles()
        for address, channel in sorted(self.index.channels.items()):
            on = channel.get("on")
            if isinstance(on, dict):
                for kind in ("device", "place"):
                    if kind in on:
                        files = _files(channel, "on")
                        yield from self._missing(
                            "channel", address, files, f"on.{kind}", kind, on[kind]
                        )
            if "pair" in channel:
                files = _files(channel, "pair")
                yield from self._missing(
                    "channel", address, files, "pair", "channel", channel["pair"]
                )
            for device in channel.get("endpoint_of") or []:
                files = _files(channel, "endpoint_of")
                yield from self._missing("channel", address, files, "endpoint_of", "device", device)
            simulation = channel.get("simulation")
            linear = simulation.get("linear") if isinstance(simulation, dict) else None
            if isinstance(linear, dict):
                files = self._seed_files(channel)
                for source in sorted(linear, key=str):
                    yield from self._missing(
                        "channel", address, files, "simulation.linear", "channel", source
                    )
            signal = channel.get("signal")
            if signal is not None and signal not in roles:
                yield FacilityBuildError(
                    "class-unknown",
                    address,
                    _files(channel, "signal"),
                    "use a vocabulary signal role or remove `signal`",
                    record_kind="channel",
                    detail=f"signal {signal} is not a vocabulary signal role",
                )

    def _seed_files(self, channel: Mapping[str, Any]) -> list[str]:
        if str(channel.get("id")) in self.sources.seeds:
            return ["seeds.yaml"]
        return _files(channel, "simulation")

    def _groups(self) -> Iterator[FacilityBuildError]:
        for gid, group in sorted(self.index.groups.items()):
            for member in group.get("members") or []:
                files = _files(group, "members")
                yield from self._missing("group", gid, files, "members", "device", member)

    def _wiring(self) -> Iterator[FacilityBuildError]:
        for _model, entry in self.index.wiring:
            wid = str(entry.get("id"))
            address = entry.get("address")
            yield from self._missing(
                "wiring", wid, _files(entry, "address"), "address", "channel", address
            )
            channel = self.index.channels.get(str(address), {})
            on = channel.get("on")
            own = on.get("device") if isinstance(on, dict) else None
            filled = "slices.device" in _provenance_defaults(entry)
            for piece in entry.get("slices") or []:
                if not isinstance(piece, dict) or "device" not in piece:
                    continue
                if filled and piece["device"] == own:
                    continue
                yield from self._missing(
                    "wiring",
                    wid,
                    _files(entry, "slices"),
                    "slices.device",
                    "device",
                    piece["device"],
                )

    def _seeds(self) -> Iterator[FacilityBuildError]:
        for address in sorted(self.sources.seeds):
            yield from self._missing("seed", address, ["seeds.yaml"], "address", "channel", address)

    def _limits(self) -> Iterator[FacilityBuildError]:
        for record in self.index.limits:
            address = str(record.get("address"))
            yield from self._missing(
                "limit", address, ["limits.yaml"], "records.address", "channel", address
            )

    def _scenarios(self) -> Iterator[FacilityBuildError]:
        for scenario in self.sources.scenarios:
            name = str(scenario["name"])
            files = [f"scenarios/{name}.yaml"]
            overrides = scenario.get("overrides")
            for address in sorted(overrides if isinstance(overrides, dict) else (), key=str):
                yield from self._missing("scenario", name, files, "overrides", "channel", address)
            for entry in scenario.get("archiver") or []:
                if isinstance(entry, dict) and "channel" in entry:
                    yield from self._missing(
                        "scenario", name, files, "archiver.channel", "channel", entry["channel"]
                    )
            faults = scenario.get("faults")
            if not isinstance(faults, dict):
                continue
            for model in sorted(faults, key=str):
                if not self._exists("model", str(model)):
                    yield from self._missing("scenario", name, files, "faults", "model", model)
                    continue
                targets = faults[model]
                for target in sorted(targets if isinstance(targets, dict) else (), key=str):
                    yield from self._missing(
                        "scenario", name, files, f"faults.{model}", "channel", target
                    )

    def _measurement(self) -> Iterator[FacilityBuildError]:
        for model, measurement in sorted(self.sources.measurement.items()):
            files = [f"measurement/{model}.yaml"]
            if not self._exists("model", model):
                yield from self._missing("measurement", model, files, "file name", "model", model)
                continue
            groups = measurement.get("groups")
            for role, group in sorted((groups if isinstance(groups, dict) else {}).items()):
                yield from self._missing(
                    "measurement", model, files, f"groups.{role}", "group", group
                )
            instruments = measurement.get("instruments")
            for role, address in sorted(
                (instruments if isinstance(instruments, dict) else {}).items()
            ):
                yield from self._missing(
                    "measurement", model, files, f"instruments.{role}", "channel", address
                )


def _provenance_defaults(record: Mapping[str, Any]) -> list[str]:
    provenance = record.get("provenance")
    defaults = provenance.get("defaults") if isinstance(provenance, dict) else None
    return list(defaults or [])


# --- S5: record rules ----------------------------------------------------------------


def check_records(document: Mapping[str, Any]) -> list[FacilityBuildError]:
    """Check the pair, value, limits and seed rules (stage S5).

    Runs only on a file whose references resolve. The stops are
    ``pair-invalid``, ``value-invalid``, ``limit-invalid`` and ``seed-invalid``
    (all but the nominal band, which needs the computed operating point).

    Args:
        document: The document S3 validated.

    Returns:
        Every error found.
    """
    return list(_Records(document).run())


class _Records:
    def __init__(self, document: Mapping[str, Any]) -> None:
        self.index = _Index(document)
        self.scenarios = [s for s in document.get("scenarios") or [] if isinstance(s, dict)]
        # Channels whose value_type, options or shape is wrong: their values
        # cannot be coerced, so no value rule runs on them.
        self.broken: set[str] = set()

    def run(self) -> Iterator[FacilityBuildError]:
        from osprey_connectors.simulation.values import coerce

        self.coerce = coerce
        for address, channel in sorted(self.index.channels.items()):
            yield from self._value_shape(address, channel)
        for address, channel in sorted(self.index.channels.items()):
            yield from self._seed(address, channel)
        yield from self._linear_cycles()
        yield from self._pairs()
        yield from self._wiring()
        yield from self._limits()
        yield from self._scenarios()

    # --- helpers -------------------------------------------------------------------

    def _error(
        self,
        kind: str,
        record_kind: str,
        record_id: str,
        files: Sequence[str],
        detail: str,
        remedy: str,
    ) -> FacilityBuildError:
        return FacilityBuildError(
            kind, record_id, files, remedy, record_kind=record_kind, detail=detail
        )

    def _type(self, address: str) -> str:
        return str(self.index.channels[address].get("value_type", "float"))

    def _coerce(self, address: str, value: Any) -> tuple[Any, str | None]:
        """The coerced value, or the reason coercion refused it."""
        channel = self.index.channels[address]
        try:
            return (
                self.coerce(
                    value,
                    self._type(address),
                    channel.get("options"),
                    channel.get("shape"),
                ),
                None,
            )
        except ValueError as exc:
            return None, str(exc)

    def _seed_files(self, channel: Mapping[str, Any]) -> list[str]:
        return _files(channel, "simulation")

    # --- value type, options, shape ------------------------------------------------

    def _value_shape(
        self, address: str, channel: Mapping[str, Any]
    ) -> Iterator[FacilityBuildError]:
        value_type = self._type(address)
        options = channel.get("options")
        shape = channel.get("shape")
        problem = None
        if value_type in ("bool", "enum"):
            problem = _options_problem(value_type, options)
        elif options is not None:
            problem = f"a {value_type} channel carries `options`"
        if problem is None:
            if value_type == "waveform":
                if not shape or any(n <= 0 for n in shape):
                    problem = "a waveform channel needs `shape`, a list of positive ints"
            elif shape is not None:
                problem = f"a {value_type} channel carries `shape`"
        if problem is not None:
            self.broken.add(address)
            yield self._error(
                "value-invalid",
                "channel",
                address,
                _files(channel, "options" if "options" in problem else "shape"),
                problem,
                "state `options` only on bool and enum channels and `shape` only on waveforms",
            )

    # --- seeds ---------------------------------------------------------------------

    def _seed(self, address: str, channel: Mapping[str, Any]) -> Iterator[FacilityBuildError]:
        seed = channel.get("simulation")
        if not isinstance(seed, dict):
            return
        files = self._seed_files(channel)
        value_type = self._type(address)
        role = channel.get("role")
        if value_type != "float":
            carried = [key for key in (*_MOTION, "clamp", "linear") if key in seed]
            if carried:
                yield self._error(
                    "value-invalid",
                    "channel",
                    address,
                    files,
                    f"a {value_type} channel carries {_names(carried)}; motion, `clamp` and "
                    "`linear` apply to float channels only",
                    f"remove {_names(carried)}",
                )
        else:
            drift = seed.get("drift")
            if isinstance(drift, dict) and not drift.get("period_s", 0) > 0:
                yield self._error(
                    "value-invalid",
                    "channel",
                    address,
                    files,
                    f"`drift.period_s` is {drift.get('period_s')}, not above 0",
                    "set `drift.period_s` above 0",
                )
            if "clamp" in seed:
                problem = _clamp_problem(seed["clamp"])
                if problem is not None:
                    yield self._error(
                        "value-invalid",
                        "channel",
                        address,
                        files,
                        problem,
                        "write `clamp` as [low, high] with low <= high; either side may be null",
                    )
            if "linear" in seed:
                yield from self._linear(address, seed, files)
            motion = [key for key in _MOTION if key in seed]
            if role == "setpoint" and motion:
                yield self._error(
                    "seed-invalid",
                    "channel",
                    address,
                    files,
                    f"a setpoint carries {_names(motion)}",
                    f"remove {_names(motion)}; a setpoint holds the value written to it",
                )
        if "nominal" in seed and "linear" not in seed:
            yield from self._nominal(address, seed["nominal"], files)

    def _nominal(
        self, address: str, nominal: Any, files: list[str]
    ) -> Iterator[FacilityBuildError]:
        wired = self.index.wired.get(address)
        if wired:
            yield self._error(
                "seed-invalid",
                "channel",
                address,
                files,
                f"a `nominal` on a channel model {', '.join(sorted(wired))} wires",
                "remove `nominal`; the operating point comes from the deck",
            )
            return
        yield from self._value("channel", address, files, "nominal", address, nominal)

    def _value(
        self,
        record_kind: str,
        record_id: str,
        files: Sequence[str],
        slot: str,
        address: str,
        value: Any,
    ) -> Iterator[FacilityBuildError]:
        """Coerce one seed nominal, override or fault value by its channel's type."""
        if address in self.broken:
            return
        value_type = self._type(address)
        if value_type == "int" and _non_integral(value):
            yield self._error(
                "seed-invalid",
                record_kind,
                record_id,
                files,
                f"`{slot}` of int channel {address} is {value}, not integral",
                f"write an integral `{slot}`",
            )
            return
        _coerced, refusal = self._coerce(address, value)
        if refusal is not None:
            yield self._error(
                "value-invalid",
                record_kind,
                record_id,
                files,
                f"`{slot}` of channel {address}: {refusal}",
                f"write a {value_type} value for `{slot}`",
            )

    def _linear(
        self, address: str, seed: Mapping[str, Any], files: list[str]
    ) -> Iterator[FacilityBuildError]:
        problems = []
        if "nominal" in seed:
            problems.append("carries `nominal` beside `linear`")
        for source in sorted(seed["linear"], key=str):
            source = str(source)
            if source in self.index.wired:
                problems.append(f"`linear` input {source} is wired")
            elif self._type(source) != "float":
                problems.append(f"`linear` input {source} is {self._type(source)}, not float")
        if problems:
            yield self._error(
                "value-invalid",
                "channel",
                address,
                files,
                "; ".join(problems),
                "take `linear` inputs from unwired float channels and drop `nominal`",
            )

    def _linear_cycles(self) -> Iterator[FacilityBuildError]:
        graph: dict[str, list[str]] = {}
        for address, channel in sorted(self.index.channels.items()):
            seed = channel.get("simulation")
            linear = seed.get("linear") if isinstance(seed, dict) else None
            if isinstance(linear, dict) and self._type(address) == "float":
                graph[address] = sorted(str(source) for source in linear)
        reported: set[frozenset[str]] = set()
        for start in sorted(graph):
            cycle = _cycle_from(graph, start)
            if cycle is None or frozenset(cycle) in reported or min(cycle) != start:
                continue
            reported.add(frozenset(cycle))
            yield self._error(
                "value-invalid",
                "channel",
                start,
                self._seed_files(self.index.channels[start]),
                f"`linear` inputs form a cycle: {' -> '.join([*cycle, start])}",
                "break the cycle",
            )

    # --- pairs ---------------------------------------------------------------------

    def _pairs(self) -> Iterator[FacilityBuildError]:
        readers: dict[str, list[str]] = {}
        for address, channel in sorted(self.index.channels.items()):
            role = channel.get("role", "readback")
            if role != "setpoint":
                if "pair" in channel:
                    yield self._error(
                        "pair-invalid",
                        "channel",
                        address,
                        _files(channel, "pair"),
                        f"a `pair` on a {role} channel",
                        "remove `pair`; only a setpoint names its readback",
                    )
                continue
            pair = str(channel.get("pair", address))
            if pair == address:
                continue
            readers.setdefault(pair, []).append(address)
            yield from self._pair(address, channel, pair)
        for readback, setpoints in sorted(readers.items()):
            if len(setpoints) > 1:
                yield self._error(
                    "pair-invalid",
                    "channel",
                    readback,
                    sorted({f for s in setpoints for f in _files(self.index.channels[s], "pair")}),
                    f"the pair of setpoints {', '.join(setpoints)}",
                    "pair each setpoint with its own readback",
                )

    def _pair(
        self, address: str, channel: Mapping[str, Any], pair: str
    ) -> Iterator[FacilityBuildError]:
        files = _files(channel, "pair")
        target = self.index.channels[pair]
        target_role = target.get("role", "readback")
        if target_role != "readback":
            yield self._error(
                "pair-invalid",
                "channel",
                address,
                files,
                f"`pair` names {target_role} channel {pair}",
                "name a readback, or remove `pair` to pair the setpoint with itself",
            )
            return
        differing = [
            slot
            for slot in ("value_type", "options", "shape")
            if channel.get(slot) != target.get(slot)
        ]
        if differing:
            yield self._error(
                "pair-invalid",
                "channel",
                address,
                files,
                f"setpoint and pair {pair} differ in {_names(differing)}",
                f"give {address} and {pair} the same {_names(differing)}",
            )
            return
        own = self.index.wired.get(address, set())
        theirs = self.index.wired.get(pair, set())
        if own and theirs and own != theirs and len(own) == 1 and len(theirs) == 1:
            yield self._error(
                "pair-invalid",
                "channel",
                address,
                files,
                f"model {', '.join(sorted(own))} wires the setpoint and model "
                f"{', '.join(sorted(theirs))} its pair {pair}",
                "wire the setpoint and its pair in the same model",
            )
            return
        yield from self._paired_seed(address, pair)

    def _paired_seed(self, setpoint: str, readback: str) -> Iterator[FacilityBuildError]:
        """At start a paired readback holds its setpoint's value."""
        seed = self.index.channels[readback].get("simulation")
        if not isinstance(seed, dict) or "nominal" not in seed:
            return
        if setpoint in self.index.wired or readback in self.index.wired:
            return
        if setpoint in self.broken or readback in self.broken:
            return
        mine, refused = self._coerce(readback, seed["nominal"])
        if refused is not None:
            return
        setpoint_seed = self.index.channels[setpoint].get("simulation")
        if isinstance(setpoint_seed, dict) and "nominal" in setpoint_seed:
            theirs, refused = self._coerce(setpoint, setpoint_seed["nominal"])
            if refused is not None:
                return
        else:
            from osprey_connectors.simulation.values import zero

            channel = self.index.channels[setpoint]
            theirs = zero(self._type(setpoint), channel.get("options"), channel.get("shape"))
        if mine != theirs:
            yield self._error(
                "seed-invalid",
                "channel",
                readback,
                self._seed_files(self.index.channels[readback]),
                f"`nominal` {seed['nominal']} differs from its setpoint {setpoint}'s {theirs}",
                f"remove `nominal` from {readback}; a paired readback starts at its setpoint's value",
            )

    # --- wiring --------------------------------------------------------------------

    def _wiring(self) -> Iterator[FacilityBuildError]:
        for _model, entry in self.index.wiring:
            wid = str(entry.get("id"))
            slices = entry.get("slices")
            if "element" in entry and slices is not None:
                yield self._error(
                    "pair-invalid",
                    "wiring",
                    wid,
                    _files(entry, "slices"),
                    "states both `element` and `slices`",
                    "keep one of `element` and `slices`",
                )
                continue
            bad = [
                piece.get("weight")
                for piece in slices or []
                if isinstance(piece, dict) and not _usable_weight(piece.get("weight", 1))
            ]
            if bad:
                yield self._error(
                    "pair-invalid",
                    "wiring",
                    wid,
                    _files(entry, "slices"),
                    f"slice weight {', '.join(str(w) for w in bad)} is zero or not finite",
                    "give every slice a finite, non-zero weight",
                )
                continue
            channel = self.index.channels.get(str(entry.get("address")), {})
            endpoints = channel.get("endpoint_of") or []
            if endpoints:
                named = {p.get("device") for p in slices or [] if isinstance(p, dict)}
                unnamed = [d for d in endpoints if d not in named]
                if unnamed:
                    yield self._error(
                        "pair-invalid",
                        "wiring",
                        wid,
                        _files(entry, "slices"),
                        f"`endpoint_of` device {', '.join(unnamed)} is named by no slice",
                        "name each `endpoint_of` device in a slice, or remove it from `endpoint_of`",
                    )

    # --- limits --------------------------------------------------------------------

    def _limits(self) -> Iterator[FacilityBuildError]:
        for record in self.index.limits:
            address = str(record.get("address"))
            channel = self.index.channels[address]
            value_type = self._type(address)
            files = ["limits.yaml"]
            if record.get("writable") is True and channel.get("role") != "setpoint":
                yield self._error(
                    "limit-invalid",
                    "limit",
                    address,
                    files,
                    f"`writable: true` on a {channel.get('role', 'readback')} channel",
                    "remove `writable`; only a setpoint is writable",
                )
            bounds = [slot for slot in _LIMIT_BOUNDS if record.get(slot) is not None]
            if bounds and value_type not in _NUMERIC_TYPES:
                yield self._error(
                    "value-invalid",
                    "limit",
                    address,
                    files,
                    f"{_names(bounds)} on a {value_type} channel",
                    f"remove {_names(bounds)}; a {value_type} channel carries `writable` and "
                    "`confirm` only",
                )
            elif value_type == "int":
                fractional = [slot for slot in bounds if _non_integral(record[slot])]
                if fractional:
                    shown = ", ".join(f"`{slot}` {record[slot]}" for slot in fractional)
                    yield self._error(
                        "limit-invalid",
                        "limit",
                        address,
                        files,
                        f"{shown} on an int channel is not integral",
                        "write integral bounds",
                    )

    # --- scenarios -----------------------------------------------------------------

    def _scenarios(self) -> Iterator[FacilityBuildError]:
        for scenario in self.scenarios:
            name = str(scenario.get("name"))
            files = [_scenario_file(scenario)]
            overrides = scenario.get("overrides") or {}
            for address, value in sorted(overrides.items(), key=lambda kv: str(kv[0])):
                yield from self._value(
                    "scenario", name, files, f"overrides.{address}", str(address), value
                )
            faults = scenario.get("faults") or {}
            for model, targets in sorted(faults.items(), key=lambda kv: str(kv[0])):
                for address, value in sorted(targets.items(), key=lambda kv: str(kv[0])):
                    yield from self._fault(name, files, str(model), str(address), value)

    def _fault(
        self, name: str, files: list[str], model: str, address: str, value: Any
    ) -> Iterator[FacilityBuildError]:
        slot = f"faults.{model}.{address}"
        if value == _STUCK:
            role = self.index.channels[address].get("role", "readback")
            if role != "setpoint":
                yield self._error(
                    "value-invalid",
                    "scenario",
                    name,
                    files,
                    f"`{slot}` is `stuck` on a {role} channel",
                    "fault a setpoint with `stuck`, or write a value",
                )
            return
        if isinstance(value, dict):
            return
        yield from self._value("scenario", name, files, slot, address, value)


def _options_problem(value_type: str, options: Any) -> str | None:
    if not options:
        return f"a {value_type} channel has no `options`"
    labels = [str(label) for label in options]
    if value_type == "bool" and len(labels) != 2:
        return f"a bool channel has {len(labels)} `options`, not 2"
    if value_type == "enum" and len(labels) < 2:
        return "an enum channel has fewer than 2 `options`"
    if len(labels) > _MAX_OPTIONS:
        return f"`options` has {len(labels)} labels, more than {_MAX_OPTIONS}"
    long = [label for label in labels if len(label) > _MAX_LABEL or not label.isascii()]
    if long:
        return f"`options` label {long[0]} is not ASCII of at most {_MAX_LABEL} characters"
    if len(set(labels)) != len(labels):
        return "`options` repeats a label"
    return None


def _clamp_problem(clamp: Any) -> str | None:
    if not isinstance(clamp, list) or len(clamp) != 2:
        return f"`clamp` {clamp} is not [low, high]"
    for side in clamp:
        if side is not None and not (_finite(side)):
            return f"`clamp` side {side} is not a finite number or null"
    low, high = clamp
    if low is not None and high is not None and low > high:
        return f"`clamp` low {low} is above high {high}"
    return None


def _finite(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _non_integral(value: Any) -> bool:
    """A finite number with a fractional part."""
    return _finite(value) and not float(value).is_integer()


def _usable_weight(weight: Any) -> bool:
    return _finite(weight) and weight != 0


def _cycle_from(graph: Mapping[str, list[str]], start: str) -> list[str] | None:
    """The first cycle through ``start`` in depth-first order, or ``None``."""
    path = [start]
    seen = {start}

    def walk(node: str) -> list[str] | None:
        for nxt in graph.get(node, ()):
            if nxt == start:
                return list(path)
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            found = walk(nxt)
            if found is not None:
                return found
            path.pop()
        return None

    return walk(start)


def _names(slots: Iterable[str]) -> str:
    return ", ".join(f"`{slot}`" for slot in slots)
