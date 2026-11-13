"""Stage S1: load every source file under ``data/facility/``, one file at a time.

The tree the loader reads::

    identity.yaml             {code, name?, description?}
    classes.yaml              [{class, parent, aliases?, description?}]
    models.yaml               [{name, engine, deck?, settings?, wiring: [...]}]
    limits.yaml               {records}
    seeds.yaml                {<address>: {nominal?, noise?, drift?, clamp?, linear?}}
    fixes.yaml                {schema: osprey.facility.fixes/1, fixes: [...]}
    documents.yaml            [{id, title?, kind?, system?}]
    policy.yaml               {relations?: {precedence?, stated_over_inferred?}}
    records/<kind>s.yaml      [{id, ...}] for places, devices, channels, groups,
                              relations
    scenarios/<name>.yaml     {description?, overrides?, faults?, archiver?, logbook?,
                               drivers?, couple?, noise?, start?}
    measurement/<model>.yaml  {kinds, groups?, instruments?, <step/settle keys>}
    imported/<layer>/         places, devices, channels, groups, relations and
                              models files written by one importer, beside the
                              layer's own files (mapping.yaml), which the build
                              never reads; a relations.csv marks a relations layer

``records/`` and ``models.yaml`` merge as the layer ``authored``; each directory
under ``imported/`` is the layer of that name. A layer writes only the fields
its source states, and never a slot the build computes.

Every problem this stage finds is collected rather than raised, so ``validate``
can print all of them; the result lists them in file order.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from osprey.facility import PN_LOCAL
from osprey.facility.errors import FacilityBuildError
from osprey_connectors import yaml_loader

__all__ = [
    "AUTHORED",
    "COMPUTED_SLOTS",
    "KEYED_SETS",
    "KIND_CLASS",
    "LAYER_OWN_FILES",
    "RECORD_FILES",
    "RELATIONS_MARK",
    "SET_VALUED_SLOTS",
    "LoadResult",
    "SourceRecord",
    "Sources",
    "citation_dates",
    "group_families",
    "keyed_clashes",
    "keyed_order",
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
    "relations.yaml": "relation",
}

#: The files a layer keeps for itself under its directory: a layer's own
#: decisions file is read by that layer's importer, never by the build.
LAYER_OWN_FILES: frozenset[str] = frozenset({"mapping.yaml"})

#: The table the relations importer always copies into its layer; a layer
#: directory holding it is a relations layer.
RELATIONS_MARK = "relations.csv"

#: The slots the build alone writes, per record kind.
COMPUTED_SLOTS: dict[str, frozenset[str]] = {
    "place": frozenset({"provenance"}),
    "device": frozenset({"ordinalInPlace", "ordinalInModel", "groups", "provenance"}),
    "channel": frozenset({"provenance"}),
    "group": frozenset({"provenance"}),
    "relation": frozenset({"standing", "contests", "provenance"}),
    "model": frozenset({"provenance"}),
    "wiring": frozenset(
        {
            "direction",
            "unit",
            "default",
            "value_range",
            "engine_role",
            "plane",
            "refresh",
            "marks",
            "provenance",
        }
    ),
}

#: The multivalued slots that are sets: compared and written sorted by string form,
#: except the keyed sets of ``KEYED_SETS``.
SET_VALUED_SLOTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Place", "names"),
        ("Device", "names"),
        ("Channel", "names"),
        ("Channel", "tags"),
        ("Channel", "former_addresses"),
        ("Channel", "endpoint_of"),
        ("Group", "names"),
        ("Group", "members"),
        ("Device", "groups"),
        ("Wiring", "marks"),
        ("Measurement", "kinds"),
        ("MeasurementGroups", "bpm"),
        ("MeasurementGroups", "hcor"),
        ("MeasurementGroups", "vcor"),
        ("MeasurementGroups", "quad"),
        ("MeasurementGroups", "sext"),
        ("Place", "scheme_names"),
        ("Place", "citations"),
        ("Device", "scheme_names"),
        ("Device", "citations"),
        ("Relation", "contests"),
    }
)

#: The set slots whose entries are maps, and the fields that key an entry. An
#: absent (or null) key field is its own value. Identical entries are one entry,
#: the set is ordered by key, and one key with two contents is refused.
KEYED_SETS: dict[str, tuple[str, ...]] = {
    "scheme_names": ("name", "scheme"),
    "citations": ("document", "version", "locator"),
}


def _plain(value: Any) -> Any:
    """A value json can write with sorted keys: a map with a non-string key becomes
    a tagged list of its entries, each key spelled canonically."""
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _plain(v) for k, v in value.items()}
        return {"\u0000map": sorted([_canonical(k), _plain(v)] for k, v in value.items())}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _canonical(value: Any) -> str:
    """A value's text independent of map key order; ``1``, ``"1"`` and ``True`` differ.

    Total: a map with non-string keys, at any depth, has a text too.
    """
    return json.dumps(_plain(value), sort_keys=True, default=str)


def _is_entry(entry: Any) -> bool:
    """A keyed-set entry the set compares: a map with string keys only. Any other
    entry is left for the schema stage, which reports it."""
    return isinstance(entry, dict) and all(isinstance(k, str) for k in entry)


def _content(entry: Mapping[str, Any]) -> dict[str, Any]:
    """An entry without its null fields: a null field and an absent one are one content."""
    return {k: v for k, v in entry.items() if v is not None}


def _entry_key(slot: str, entry: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    """An entry's key, sortable: an absent field orders before any present value."""
    return tuple(
        (0, "") if entry.get(name) is None else (1, _canonical(entry[name]))
        for name in KEYED_SETS[slot]
    )


def keyed_order(slot: str, entries: list[Any]) -> list[Any]:
    """A keyed set in its compared and written form.

    Identical map entries collapse (a null field counting as absent, and
    written so), and the maps are ordered by key, then by content; an entry
    that is not a map with string keys is left for the schema stage and
    written after them, ordered by string form.
    """
    maps: dict[str, Any] = {}
    rest = []
    for entry in entries:
        if _is_entry(entry):
            content = _content(entry)
            maps.setdefault(_canonical(content), content)
        else:
            rest.append(entry)
    ordered = sorted(maps.items(), key=lambda item: (_entry_key(slot, item[1]), item[0]))
    return [entry for _, entry in ordered] + sorted(rest, key=str)


def keyed_clashes(slot: str, entries: Any) -> list[str]:
    """Each key of a keyed set that has two contents, spelled for an error line."""
    if slot not in KEYED_SETS or not isinstance(entries, list):
        return []
    contents: dict[tuple[tuple[int, str], ...], set[str]] = {}
    spelled: dict[tuple[tuple[int, str], ...], str] = {}
    for entry in entries:
        if not _is_entry(entry):
            continue
        key = _entry_key(slot, entry)
        contents.setdefault(key, set()).add(_canonical(_content(entry)))
        spelled[key] = ", ".join(
            f"no {name}" if entry.get(name) is None else f"{name} {entry[name]}"
            for name in KEYED_SETS[slot]
        )
    return [spelled[key] for key in sorted(contents) if len(contents[key]) > 1]


def _citation_text(citation: Any) -> Any:
    """A citation with its ``imported`` date as ISO text."""
    if isinstance(citation, dict) and isinstance(citation.get("imported"), datetime.date):
        return {**citation, "imported": citation["imported"].isoformat()}
    return citation


def citation_dates(kind: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """A record's fields with every ``Citation.imported`` date as its ISO text.

    YAML reads an unquoted date as a date; written as text, it is one content
    with the equal quoted string, and ``facility.json`` can hold it. Covers a
    place's or device's ``citations`` and ``scheme_names[].source``, and a
    relation's ``source``; any other shape is left as found.
    """
    out = dict(fields)
    if kind in ("place", "device"):
        citations = out.get("citations")
        if isinstance(citations, list):
            out["citations"] = [_citation_text(c) for c in citations]
        aliases = out.get("scheme_names")
        if isinstance(aliases, list):
            out["scheme_names"] = [
                {**a, "source": _citation_text(a["source"])}
                if isinstance(a, dict) and "source" in a
                else a
                for a in aliases
            ]
    elif kind == "relation" and "source" in out:
        out["source"] = _citation_text(out["source"])
    return out


#: The schema class each record kind is shaped by.
KIND_CLASS: dict[str, str] = {
    "place": "Place",
    "device": "Device",
    "channel": "Channel",
    "group": "Group",
    "model": "Model",
    "wiring": "Wiring",
    "relation": "Relation",
}

_TOP_FILES = frozenset(
    {
        "identity.yaml",
        "classes.yaml",
        "models.yaml",
        "limits.yaml",
        "seeds.yaml",
        "fixes.yaml",
        "documents.yaml",
        "policy.yaml",
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


_LOADERS: dict[type, type[Any]] = {yaml.SafeLoader: _Loader}


def _loader_on(base: type) -> type[Any]:
    """``_Loader``'s bool rule on the scanner of *base*: the one resolver table, shared."""
    loader = _LOADERS.get(base)
    if loader is None:
        loader = type("_ScannedLoader", (base,), {"__doc__": _Loader.__doc__})
        loader.yaml_implicit_resolvers = _Loader.yaml_implicit_resolvers
        _LOADERS[base] = loader
    return loader


def read_yaml(text: str) -> Any:
    """Parse one source file's text.

    The text is scanned on libyaml when it is installed; a text that does not
    parse is refused in the words of PyYAML's own parser, which quotes the line.

    Args:
        text: The file's contents.

    Returns:
        The parsed document; ``None`` for an empty file.

    Raises:
        yaml.YAMLError: On a text that does not parse.
    """
    loader = _loader_on(yaml_loader.safe_loader())
    try:
        return yaml.load(text, Loader=loader)
    except yaml.YAMLError:
        if loader is _Loader:
            raise
    return yaml.load(text, Loader=_Loader)


def group_families(value: Any) -> tuple[str, ...]:
    """The group ids a measurement-file group role names: each item of a list, else the value itself."""
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


@cache
def slot_names(class_name: str) -> frozenset[str]:
    """The slot names ``core.yaml`` declares for one class.

    Args:
        class_name: A class of ``core.yaml``, such as ``Channel``.

    Returns:
        The names of the class's attributes.
    """
    core = yaml_loader.safe_load(_CORE_SCHEMA.read_text(encoding="utf-8"))
    return frozenset(core["classes"][class_name]["attributes"])


@dataclass(frozen=True)
class SourceRecord:
    """One record as one layer states it.

    Attributes:
        kind: ``place``, ``device``, ``channel``, ``group``, ``relation``, ``model``
            or ``wiring``.
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
        documents: ``documents.yaml``'s records, in file order.
        policy: ``policy.yaml`` without its header, or ``None`` when absent or
            holding nothing but a header or comments. Key- and value-checked
            against the declared document kinds; never written into the facility file.
        relations_layers: The imported layers whose directory holds
            ``relations.csv``, the relations importer's mark.
    """

    identity: dict[str, Any] | None = None
    classes: list[dict[str, Any]] = field(default_factory=list)
    records: list[SourceRecord] = field(default_factory=list)
    seeds: dict[str, dict[str, Any]] = field(default_factory=dict)
    limits: dict[str, Any] | None = None
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    measurement: dict[str, dict[str, Any]] = field(default_factory=dict)
    fixes: Any = None
    documents: list[dict[str, Any]] = field(default_factory=list)
    policy: dict[str, Any] | None = None
    relations_layers: frozenset[str] = frozenset()


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
        self._read_documents()
        self._read_policy()
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
            if path.name in LAYER_OWN_FILES:
                continue
            if path.name not in RECORD_FILES and path.name != "models.yaml":
                self._path_error(self._rel(path), "is not a layer file", "remove or rename it")
        if (layer_dir / RELATIONS_MARK).is_file():
            self.sources.relations_layers |= {layer}
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
        row = citation_dates(kind, row)
        self._check_keys(row, KIND_CLASS[kind], COMPUTED_SLOTS[kind], kind, rid, rel)
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
        for slot in sorted(KEYED_SETS):
            if (KIND_CLASS[kind], slot) not in SET_VALUED_SLOTS:
                continue
            for key in keyed_clashes(slot, row.get(slot)):
                self._error(
                    "source-invalid",
                    kind,
                    rid,
                    rel,
                    f"`{slot}` states two contents for key {key}",
                    f"keep one entry per key in `{slot}` in {rel}",
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

    def _read_documents(self) -> None:
        rel = "documents.yaml"
        rows = self._list(self.root / rel)
        first: dict[str, int] = {}
        for index, row in enumerate(rows or []):
            rid = self._record_id(row, "id", rel, index)
            if rid is None:
                continue
            self._check_keys(row, "Document", (), "document", rid, rel)
            if rid in first:
                self._error(
                    "source-invalid",
                    "document",
                    rid,
                    rel,
                    f"{rel} states the id twice (entries {first[rid]} and {index})",
                    f"keep one document per id in {rel}",
                )
                continue
            first[rid] = index
            self.sources.documents.append(row)

    def _read_policy(self) -> None:
        from osprey.facility.validate import relation_types

        rel = "policy.yaml"
        data = self._mapping(self.root / rel)
        if not data:
            return
        self._check_policy_keys(data, slot_names("Policy"), rel)
        relations = data.get("relations")
        if relations is not None:
            where = f"{rel}.relations"
            if self._policy_mapping(relations, where):
                self._check_policy_keys(relations, slot_names("RelationPolicy"), where)
                precedence = relations.get("precedence")
                where = f"{where}.precedence"
                if precedence is not None and self._policy_mapping(precedence, where):
                    allowed = frozenset({"default", *relation_types()})
                    self._check_policy_keys(precedence, allowed, where)
                    for key, kinds in precedence.items():
                        if key in allowed:
                            self._check_precedence(key, kinds, f"{where}.{key}")
                ranked = relations.get("stated_over_inferred")
                if "stated_over_inferred" in relations and not isinstance(ranked, bool):
                    self._path_error(
                        f"{rel}.relations.stated_over_inferred",
                        f"is {ranked!r}, not true or false",
                        "write true or false",
                    )
        self.sources.policy = data

    def _check_precedence(self, key: str, kinds: Any, where: str) -> None:
        """Check one precedence list: distinct non-empty strings, each a declared document kind.

        ``default`` may instead be the string ``undecided`` (no ranking). Runs after
        ``documents.yaml`` is read, so the declared kinds are known.
        """
        if key == "default" and kinds == "undecided":
            return
        if not isinstance(kinds, list):
            accepted = "a list of document kinds"
            if key == "default":
                accepted += " or `undecided`"
            self._path_error(where, f"is not {accepted}", f"write {accepted}")
            return
        declared = {
            row["kind"] for row in self.sources.documents if isinstance(row.get("kind"), str)
        }
        seen: set[str] = set()
        for index, kind in enumerate(kinds):
            if not isinstance(kind, str) or not kind.strip():
                self._path_error(
                    where,
                    f"entry {index} is {kind!r}, not a document kind",
                    "list document kinds as non-empty strings",
                )
            elif kind in seen:
                self._path_error(where, f"lists `{kind}` twice", f"keep one `{kind}` in {where}")
            elif not self.sources.documents:
                self._path_error(
                    where,
                    f"lists `{kind}`, but no documents.yaml declares documents",
                    f"declare a document of kind `{kind}` in documents.yaml or remove `{kind}`",
                )
            elif kind not in declared:
                self._path_error(
                    where,
                    f"lists `{kind}`, which no document in documents.yaml has as its kind",
                    f"declare a document of kind `{kind}` in documents.yaml or remove `{kind}`",
                )
            if isinstance(kind, str):
                seen.add(kind)

    def _policy_mapping(self, value: Any, where: str) -> bool:
        if isinstance(value, dict):
            return True
        self._path_error(where, "is not a mapping", "write it as a mapping")
        return False

    def _check_policy_keys(
        self, data: Mapping[str, Any], allowed: frozenset[str], where: str
    ) -> None:
        for key in data:
            if key not in allowed:
                self._path_error(where, f"unknown key `{key}`", f"remove `{key}` from {where}")

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
