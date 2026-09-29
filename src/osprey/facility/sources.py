"""Stage S1: load every source file under ``data/facility/``, one file at a time.

The tree the loader reads::

    identity.yaml             {code, name?, description?}
    classes.yaml              [{class, parent, aliases?, description?}]
    models.yaml               [{name, engine, deck?, settings?, wiring: [...]}]
    limits.yaml               {records}
    seeds.yaml                {<address>: {nominal?, noise?, drift?, clamp?, linear?}}
    fixes.yaml                {schema: osprey.facility.fixes/1, fixes: [...]}
    records/<kind>s.yaml      [{id, ...}] for places, devices, channels, groups
    scenarios/<name>.yaml     {overrides?, faults?, archiver?, logbook?}
    measurement/<model>.yaml  {kinds, groups?, instruments?, <step/settle keys>}
    imported/<layer>/         places, devices, channels, groups and models files
                              written by one importer

``records/`` and ``models.yaml`` merge as the layer ``authored``; each directory
under ``imported/`` is the layer of that name. A layer writes only the fields
its source states, and never a slot the build computes.

Every problem this stage finds is collected rather than raised, so ``validate``
can print all of them; the result lists them in file order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from osprey.facility import PN_LOCAL
from osprey.facility.errors import FacilityBuildError

__all__ = [
    "AUTHORED",
    "COMPUTED_SLOTS",
    "RECORD_FILES",
    "LoadResult",
    "SourceRecord",
    "Sources",
    "load_sources",
    "slot_names",
]

#: The layer the hand-edited records merge as.
AUTHORED = "authored"

#: The record files a layer may write, and the kind each one holds.
RECORD_FILES: dict[str, str] = {
    "places.yaml": "place",
    "devices.yaml": "device",
    "channels.yaml": "channel",
    "groups.yaml": "group",
}

#: The slots the build alone writes, per record kind.
COMPUTED_SLOTS: dict[str, frozenset[str]] = {
    "place": frozenset({"provenance"}),
    "device": frozenset(
        {"model", "s", "length", "ordinalInPlace", "ordinalInModel", "groups", "provenance"}
    ),
    "channel": frozenset({"provenance"}),
    "group": frozenset({"provenance"}),
    "model": frozenset({"provenance"}),
    "wiring": frozenset({"direction", "unit", "default", "value_range", "provenance"}),
}

#: The schema class each record kind is shaped by.
_KIND_CLASS: dict[str, str] = {
    "place": "Place",
    "device": "Device",
    "channel": "Channel",
    "group": "Group",
    "model": "Model",
    "wiring": "Wiring",
}

_TOP_FILES = frozenset(
    {
        "identity.yaml",
        "classes.yaml",
        "models.yaml",
        "limits.yaml",
        "seeds.yaml",
        "fixes.yaml",
    }
)

_HEADER = re.compile(r"^osprey\.facility\.[a-z_]+/[0-9]+$")

_CORE_SCHEMA = Path(__file__).parent / "schema" / "core.yaml"


class _Loader(yaml.SafeLoader):
    """A safe loader that reads only ``true``/``false`` as booleans.

    YAML 1.1 reads a bare ``on``, ``off``, ``yes`` or ``no`` as a boolean, which
    would turn a channel's ``on:`` key into ``True``.
    """


_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def read_yaml(text: str) -> Any:
    """Parse one source file's text.

    Args:
        text: The file's contents.

    Returns:
        The parsed document; ``None`` for an empty file.
    """
    return yaml.load(text, Loader=_Loader)


@cache
def slot_names(class_name: str) -> frozenset[str]:
    """The slot names ``core.yaml`` declares for one class.

    Args:
        class_name: A class of ``core.yaml``, such as ``Channel``.

    Returns:
        The names of the class's attributes.
    """
    core = yaml.safe_load(_CORE_SCHEMA.read_text(encoding="utf-8"))
    return frozenset(core["classes"][class_name]["attributes"])


@dataclass(frozen=True)
class SourceRecord:
    """One record as one layer states it.

    Attributes:
        kind: ``place``, ``device``, ``channel``, ``group``, ``model`` or ``wiring``.
        id: The record's id; a model's name; a wiring record's ``<model>/<address>``.
        layer: The layer that wrote it.
        file: The file it came from, relative to ``data/facility/``.
        fields: The slots the layer states, without the id and, for a model,
            without its wiring.
    """

    kind: str
    id: str
    layer: str
    file: str
    fields: dict[str, Any]


@dataclass
class Sources:
    """Everything stage S1 read, one entry per source file.

    Attributes:
        identity: ``identity.yaml`` without its header, or ``None`` when absent.
        classes: ``classes.yaml``'s records, in file order.
        records: Every layer's records, ordered by layer, file and position.
        seeds: ``seeds.yaml``, address to seed record.
        limits: ``limits.yaml`` without its header, or ``None`` when absent or
            holding nothing but a header or comments (limits are opt-in).
        scenarios: One record per ``scenarios/<name>.yaml``, with its ``name``,
            sorted by name.
        measurement: ``measurement/<model>.yaml``, model name to its record.
        fixes: ``fixes.yaml``'s document as parsed, or ``None`` when absent.
    """

    identity: dict[str, Any] | None = None
    classes: list[dict[str, Any]] = field(default_factory=list)
    records: list[SourceRecord] = field(default_factory=list)
    seeds: dict[str, dict[str, Any]] = field(default_factory=dict)
    limits: dict[str, Any] | None = None
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    measurement: dict[str, dict[str, Any]] = field(default_factory=dict)
    fixes: Any = None


@dataclass
class LoadResult:
    """What stage S1 returns.

    Attributes:
        sources: The sources read; complete only when ``errors`` is empty.
        errors: Every stop found, in file order.
    """

    sources: Sources
    errors: list[FacilityBuildError]


def load_sources(facility_dir: Path) -> LoadResult:
    """Read every source file under a facility directory (stage S1).

    A missing directory is zero sources. The stops found are ``source-invalid``
    (a parse failure, an unknown key, a computed slot, an ``on`` naming both a
    device and a place, a name that is not ``PN_LOCAL``, model names equal but
    for case) and ``layer-duplicate`` (one layer stating an id twice).

    Args:
        facility_dir: The ``data/facility`` directory.

    Returns:
        The sources and every stop found.
    """
    return _Reader(facility_dir).read()


class _Reader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sources = Sources()
        self.errors: list[FacilityBuildError] = []

    def read(self) -> LoadResult:
        root = self.root
        if not root.is_dir():
            return LoadResult(self.sources, self.errors)
        for path in sorted(root.glob("*.yaml")):
            if path.name not in _TOP_FILES:
                self._path_error(path.name, "is not a facility source file", "remove or rename it")
        self._read_identity()
        self._read_classes()
        self._read_layer(AUTHORED, root / "records", root / "models.yaml")
        imported = root / "imported"
        if imported.is_dir():
            for layer_dir in sorted(p for p in imported.iterdir() if p.is_dir()):
                self._read_imported(layer_dir)
        self._read_seeds()
        self._read_limits()
        self._read_scenarios()
        self._read_measurement()
        fixes = self._parse(root / "fixes.yaml")
        if fixes is not _ABSENT:
            self.sources.fixes = fixes
        self._check_duplicates()
        self._check_model_names()
        return LoadResult(self.sources, self.errors)

    # --- reading ------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _parse(self, path: Path) -> Any:
        if not path.is_file():
            return _ABSENT
        try:
            return read_yaml(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            detail = " ".join(str(exc).split())
            self._path_error(self._rel(path), f"does not parse: {detail}", "fix the YAML")
            return _ABSENT

    def _mapping(self, path: Path) -> dict[str, Any] | None:
        data = self._parse(path)
        if data is _ABSENT:
            return None
        if data is None:
            return {}
        if not isinstance(data, dict):
            self._path_error(self._rel(path), "is not a mapping", "write it as a mapping")
            return None
        return self._strip_header(data, self._rel(path))

    def _list(self, path: Path) -> list[Any] | None:
        data = self._parse(path)
        if data is _ABSENT:
            return None
        if data is None:
            return []
        if not isinstance(data, list):
            self._path_error(self._rel(path), "is not a list", "write it as a list of records")
            return None
        return data

    def _strip_header(self, data: dict[str, Any], rel: str) -> dict[str, Any]:
        if "schema" not in data:
            return data
        header = data["schema"]
        if not isinstance(header, str) or not _HEADER.match(header):
            self._path_error(
                f"{rel}.schema",
                f"header {header!r} is not osprey.facility.<doc>/<version>",
                "fix the header",
            )
        return {k: v for k, v in data.items() if k != "schema"}

    def _read_identity(self) -> None:
        data = self._mapping(self.root / "identity.yaml")
        if data is None:
            return
        self._check_keys(data, "Identity", (), "path", "identity.yaml", "identity.yaml")
        code = data.get("code")
        if code is not None and not (isinstance(code, str) and PN_LOCAL.fullmatch(code)):
            self._path_error(
                "identity.yaml.code",
                f"code {code!r} does not match [A-Za-z_][A-Za-z0-9_]*",
                "rename the code",
            )
        self.sources.identity = data

    def _read_classes(self) -> None:
        rel = "classes.yaml"
        rows = self._list(self.root / rel)
        for index, row in enumerate(rows or []):
            if not isinstance(row, dict):
                self._path_error(f"{rel}.{index}", "is not a mapping", "write it as a mapping")
                continue
            self._check_keys(row, "FacilityClass", (), "path", f"{rel}.{index}", rel)
            self.sources.classes.append(row)

    def _read_imported(self, layer_dir: Path) -> None:
        layer = layer_dir.name
        if layer == AUTHORED:
            self._path_error(
                self._rel(layer_dir),
                f"uses the reserved layer name {AUTHORED}",
                "move the authored records to records/",
            )
            return
        for path in sorted(layer_dir.glob("*.yaml")):
            if path.name not in RECORD_FILES and path.name != "models.yaml":
                self._path_error(self._rel(path), "is not a layer file", "remove or rename it")
        self._read_layer(layer, layer_dir, layer_dir / "models.yaml")

    def _read_layer(self, layer: str, records_dir: Path, models_file: Path) -> None:
        if layer == AUTHORED and records_dir.is_dir():
            for path in sorted(records_dir.glob("*.yaml")):
                if path.name not in RECORD_FILES:
                    self._path_error(
                        self._rel(path), "is not a records file", "remove or rename it"
                    )
        for name, kind in sorted(RECORD_FILES.items()):
            path = records_dir / name
            rows = self._list(path)
            for index, row in enumerate(rows or []):
                self._read_record(layer, kind, self._rel(path), index, row)
        rows = self._list(models_file)
        if rows is not None:
            rel = self._rel(models_file)
            for index, row in enumerate(rows):
                self._read_model(layer, rel, index, row)

    def _record_id(self, row: Any, key: str, rel: str, index: int) -> str | None:
        if not isinstance(row, dict):
            self._path_error(f"{rel}.{index}", "is not a mapping", "write it as a mapping")
            return None
        rid = row.get(key)
        if not isinstance(rid, str) or not rid:
            self._path_error(f"{rel}.{index}.{key}", f"has no string `{key}`", f"add `{key}`")
            return None
        return rid

    def _read_record(self, layer: str, kind: str, rel: str, index: int, row: Any) -> None:
        rid = self._record_id(row, "id", rel, index)
        if rid is None:
            return
        self._check_keys(row, _KIND_CLASS[kind], COMPUTED_SLOTS[kind], kind, rid, rel)
        if kind == "channel":
            on = row.get("on")
            if isinstance(on, dict) and "device" in on and "place" in on:
                self._error(
                    "source-invalid",
                    kind,
                    rid,
                    rel,
                    "`on` names both a device and a place",
                    f"keep one of `device` and `place` in {rel}",
                )
        fields = {k: v for k, v in row.items() if k != "id"}
        self.sources.records.append(SourceRecord(kind, rid, layer, rel, fields))

    def _read_model(self, layer: str, rel: str, index: int, row: Any) -> None:
        name = self._record_id(row, "name", rel, index)
        if name is None:
            return
        self._check_keys(row, "Model", COMPUTED_SLOTS["model"], "model", name, rel)
        if not PN_LOCAL.fullmatch(name):
            self._error(
                "source-invalid",
                "model",
                name,
                rel,
                "the name does not match [A-Za-z_][A-Za-z0-9_]*",
                "rename the model",
            )
        fields = {k: v for k, v in row.items() if k not in ("name", "wiring")}
        self.sources.records.append(SourceRecord("model", name, layer, rel, fields))
        wiring = row.get("wiring")
        if wiring is None:
            return
        if not isinstance(wiring, list):
            self._path_error(f"{rel}.{index}.wiring", "is not a list", "write it as a list")
            return
        for position, entry in enumerate(wiring):
            where = f"{rel}.{index}.wiring.{position}"
            address = self._record_id(entry, "address", where.rsplit(".", 1)[0], position)
            if address is None:
                continue
            wid = f"{name}/{address}"
            stated = entry.get("id")
            if stated is not None and stated != wid:
                self._error(
                    "source-invalid",
                    "wiring",
                    wid,
                    rel,
                    f"`id` {stated!r} is not <model>/<address>",
                    f"remove `id` or write {wid}",
                )
            self._check_keys(entry, "Wiring", COMPUTED_SLOTS["wiring"], "wiring", wid, rel)
            fields = {k: v for k, v in entry.items() if k != "id"}
            self.sources.records.append(SourceRecord("wiring", wid, layer, rel, fields))

    def _read_seeds(self) -> None:
        rel = "seeds.yaml"
        data = self._mapping(self.root / rel)
        for address, seed in sorted((data or {}).items(), key=lambda item: str(item[0])):
            if not isinstance(seed, dict):
                self._path_error(f"{rel}.{address}", "is not a mapping", "write it as a mapping")
                continue
            self._check_keys(seed, "Seed", (), "channel", str(address), rel)
            self.sources.seeds[str(address)] = seed

    def _read_limits(self) -> None:
        rel = "limits.yaml"
        data = self._mapping(self.root / rel)
        if not data:
            return
        self._check_keys(data, "Limits", (), "path", rel, rel)
        records = data.get("records")
        for index, row in enumerate(records if isinstance(records, list) else []):
            if isinstance(row, dict):
                self._check_keys(row, "LimitRecord", (), "path", f"{rel}.records.{index}", rel)
        self.sources.limits = data

    def _read_scenarios(self) -> None:
        folder = self.root / "scenarios"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.yaml")):
            data = self._mapping(path)
            if data is None:
                continue
            rel = self._rel(path)
            name = data.get("name", path.stem)
            if name != path.stem:
                self._path_error(
                    f"{rel}.name", f"name {name!r} is not the file name", "remove `name`"
                )
            self._check_keys(data, "Scenario", (), "path", rel, rel)
            self.sources.scenarios.append({**data, "name": path.stem})

    def _read_measurement(self) -> None:
        folder = self.root / "measurement"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.yaml")):
            data = self._mapping(path)
            if data is None:
                continue
            self._check_keys(data, "Measurement", (), "model", path.stem, self._rel(path))
            self.sources.measurement[path.stem] = data

    # --- checks -------------------------------------------------------------

    def _check_keys(
        self,
        row: Mapping[str, Any],
        class_name: str,
        computed: frozenset[str] | tuple[()],
        record_kind: str,
        record_id: str,
        rel: str,
    ) -> None:
        allowed = slot_names(class_name)
        for key in row:
            if key in computed:
                self._error(
                    "source-invalid",
                    record_kind,
                    record_id,
                    rel,
                    f"`{key}` is computed by the build",
                    f"remove `{key}` from {rel}",
                )
            elif key not in allowed:
                self._error(
                    "source-invalid",
                    record_kind,
                    record_id,
                    rel,
                    f"unknown key `{key}`",
                    f"remove `{key}` from {rel}",
                )

    def _check_duplicates(self) -> None:
        first: dict[tuple[str, str, str], SourceRecord] = {}
        reported: set[tuple[str, str, str]] = set()
        for record in self.sources.records:
            key = (record.layer, record.kind, record.id)
            if key not in first:
                first[key] = record
                continue
            if key in reported:
                continue
            reported.add(key)
            self._error(
                "layer-duplicate",
                record.kind,
                record.id,
                (first[key].file, record.file),
                f"layer {record.layer} states it twice ({first[key].file}, {record.file})",
                f"keep one record per id in layer {record.layer}",
            )
        for address, _seed in sorted(self.sources.seeds.items()):
            key = (AUTHORED, "channel", address)
            if key in first and "simulation" in first[key].fields:
                self._error(
                    "layer-duplicate",
                    "channel",
                    address,
                    (first[key].file, "seeds.yaml"),
                    f"layer {AUTHORED} states `simulation` twice ({first[key].file}, seeds.yaml)",
                    "keep the seed in seeds.yaml only",
                )
        for model in sorted(self.sources.measurement):
            key = (AUTHORED, "model", model)
            rel = f"measurement/{model}.yaml"
            if key in first and "measurement" in first[key].fields:
                self._error(
                    "layer-duplicate",
                    "model",
                    model,
                    (first[key].file, rel),
                    f"layer {AUTHORED} states `measurement` twice ({first[key].file}, {rel})",
                    f"keep the measurement in {rel} only",
                )

    def _check_model_names(self) -> None:
        spellings: dict[str, dict[str, str]] = {}
        for record in self.sources.records:
            if record.kind == "model":
                spellings.setdefault(record.id.casefold(), {}).setdefault(record.id, record.file)
        for names in spellings.values():
            if len(names) < 2:
                continue
            ordered = sorted(names)
            self._error(
                "source-invalid",
                "model",
                ordered[0],
                [names[n] for n in ordered],
                f"model names {', '.join(ordered)} differ only in case",
                "use one spelling",
            )

    # --- errors -------------------------------------------------------------

    def _path_error(self, path: str, detail: str, remedy: str) -> None:
        source = path.split(".yaml")[0] + ".yaml" if ".yaml" in path else path
        self._error("source-invalid", "path", path, source, detail, remedy)

    def _error(
        self,
        kind: str,
        record_kind: str,
        record_id: str,
        sources: str | tuple[str, ...] | list[str],
        detail: str,
        remedy: str,
    ) -> None:
        files = [sources] if isinstance(sources, str) else list(sources)
        self.errors.append(
            FacilityBuildError(
                kind, record_id, files, remedy, record_kind=record_kind, detail=detail
            )
        )


class _Absent:
    """Marks a file that does not exist or did not parse."""


_ABSENT: Any = _Absent()
