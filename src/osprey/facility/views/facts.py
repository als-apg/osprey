"""The agent facts view: what a render's agents are told about the facility.

Written to ``<render>/data/``::

    facility_facts.json   {schema: osprey.facility.facility_facts/3, identity,
                           place_levels, hierarchy_levels, places,
                           unplaced_devices, device_classes, vocabulary, models,
                           measurement_models, measurement_kinds,
                           channel_count, snapshot}
    facility_facts.md     the same facts as one page, rendered once from
                          ``_facility_facts.md.j2``

``identity`` is ``{code, name, description}``; a facility that authors no name
takes the project's. ``place_levels`` is the distinct ``level`` words of the
places, shallowest first. ``hierarchy_levels`` is the place levels of the
hierarchical channel-finder index: the level words of the places that hold a
channel or are an ancestor of one
(:func:`osprey.facility.views.channel_finder.hierarchy_levels`), named in
every channel_finder mode and never checked here. ``places`` has one entry
per place, sorted by id:
its ``level``, its ``names`` and the ``devices`` at or under it counted by
class. ``unplaced_devices`` counts by class the devices that sit in no place.
``device_classes`` has one entry per class a device carries and per
facility-added class: ``count`` devices, the ``aliases`` the vocabulary and
``classes.yaml`` give the class in their authored spelling, and ``groups``, the
sorted ids of the groups holding one of its devices. ``vocabulary`` counts the
channels by role and by signal, with each signal's description from the
document's signal-role table, the setpoints paired with a readback and the
channels that carry no signal.
``models`` lists every model with its ``engine``, whether the render serves it
and its engine's ``solve`` setting. ``measurement_models`` holds one record per
pyAML view the render carries, ``{path, sha256, files, kinds, names,
instruments, rf}``: the configuration's path under the render and its digest,
the digest of each other file of the view by its name in the view's directory,
and the model's kind block (:func:`osprey.facility.views.pyaml.kind_records`),
where each listed kind's ``facts_writer`` output for the model's sub-block is
the kind's ``facts`` (:func:`osprey_facility.kinds.written_facts`), present only
when the kind wrote some; the facts view is written after the pyAML view and
reads it from disk.
``measurement_kinds`` is every registered measurement kind's row
(:func:`osprey_facility.kinds.kind_table_row`), a model's ``kinds`` the subset
its measurement file allows; a render with no pyAML view has none. ``channel_count`` counts the channels, and
``snapshot`` is ``null``.

A render with no facts file is read as the zero-source facts: the identity of
its facility file or project name, no place, no class, ``texture`` alone and no
channel.

The schema's number rises whenever a key is added, removed or renamed or a
value changes shape; ``read_facts`` reads only the current number and takes
any other file for the zero-source facts, which every build rewrites.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs, view_bytes
from osprey.facility.views.channel_finder import hierarchy_levels
from osprey_facility import DEFAULT_ROLE, TEXTURE
from osprey_facility.records import FacilityRecords

__all__ = [
    "CONFIGURATION_FILE",
    "FACTS_FILE",
    "FACTS_PAGE",
    "FACTS_SCHEMA",
    "FACTS_TEMPLATE",
    "PYAML_DIR",
    "facts_document",
    "hook_measurement",
    "measurement_records",
    "read_facts",
    "render_facts_page",
    "write_facts_view",
    "zero_source_facts",
]

logger = logging.getLogger(__name__)

FACTS_FILE = "facility_facts.json"

#: The pyAML view's directory under the render's ``data/``, and each model's
#: configuration in it.
PYAML_DIR = "pyaml"
CONFIGURATION_FILE = "configuration.yaml"
FACTS_PAGE = "facility_facts.md"
FACTS_SCHEMA = "osprey.facility.facility_facts/3"

#: The class a classless device is counted under.
NO_CLASS = "-"

#: The page's template, relative to the packaged templates directory.
FACTS_TEMPLATE = "claude_code/_facility_facts.md.j2"

#: The digest of the facility file last parsed and the document it parsed to:
#: every render of one build carries byte-equal ``facility.json``, so it is
#: parsed once and the views it is handed to see one document, whose wiring they
#: memoize. Keyed by digest, so the memo does not also hold the file's bytes.
_parsed: tuple[str, dict[str, Any]] | None = None


def parsed_facility_file(path: Path) -> Any:
    """``facility.json`` at ``path``, as :func:`parsed_facility` gives it, checked.

    A file holding the bytes the build serialised gives the build's document
    without being read into memory (:func:`~osprey_facility.render.serialised_file_document`);
    any other file is read and parsed. Either way the document is handed back
    only once :func:`~osprey_facility.reading.check_header` has passed it, so
    the shared parse never gives a caller a document whose header it does not
    read.

    Args:
        path: The facility file.

    Returns:
        The parsed document, a JSON object whose ``schema`` header is in
        :data:`~osprey_facility.reading.READS`; shared, so callers read it and
        never change it.

    Raises:
        OSError: The file cannot be read; :exc:`FileNotFoundError` when it is
            not there.
        ValueError: The bytes are not UTF-8 JSON.
        FacilityFileError: The document is not a JSON object or carries a
            header outside :data:`~osprey_facility.reading.READS`; the message
            names ``path`` and the rebuild. A :exc:`ValueError`.
    """
    from osprey_facility.reading import check_header
    from osprey_facility.render import serialised_file_document

    document = serialised_file_document(path)
    if document is None:
        document = parsed_facility(path.read_bytes())
    check_header(document, str(path))
    return document


def parsed_facility(raw: bytes) -> Any:
    """``facility.json`` parsed from its bytes, once for each new bytes.

    The header is not checked here: :func:`parsed_facility_file`, the one
    caller, checks it on every document this returns.

    Args:
        raw: The file's bytes.

    Returns:
        The parsed document, shared with every caller handed the same bytes;
        callers read it and never change it. Bytes the build serialised from
        its own document give that document back
        (:func:`~osprey_facility.render.serialised_document`), unparsed.

    Raises:
        ValueError: The bytes are not UTF-8 JSON.
    """
    from osprey_facility.render import serialised_document

    built = serialised_document(raw)
    if built is not None:
        return built
    global _parsed
    digest = hashlib.sha256(raw).hexdigest()
    if _parsed is None or _parsed[0] != digest:
        _parsed = None  # the previous document goes before the next is parsed
        _parsed = (digest, json.loads(raw.decode("utf-8")))
    return _parsed[1]


def _identity(records: FacilityRecords, project_name: str | None) -> dict[str, Any]:
    from osprey.utils.facility import identity_record

    return dict(identity_record(records.document["identity"], project_name))


def _place_levels(records: FacilityRecords) -> list[str]:
    depth: dict[str, int] = {}
    for place in records.of_kind("place"):
        level = place.get("level")
        if level is None:
            continue
        here = str(place["id"]).count("/")
        depth[str(level)] = min(here, depth.get(str(level), here))
    return sorted(depth, key=lambda level: (depth[level], level))


def _class_counts(devices: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for device in devices:
        name = device.get("class")
        counts[NO_CLASS if name is None else str(name)] += 1
    return dict(sorted(counts.items()))


def _places(records: FacilityRecords) -> list[dict[str, Any]]:
    """Each place with its level, its names and the devices at or under it by class.

    One pass over the devices counts each under its own place and under every
    place id that place lies below, rather than asking every device of every
    place.
    """
    under: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for device in records.of_kind("device"):
        if device.get("place") is None:
            continue
        at = str(device["place"])
        name = device.get("class")
        counted = NO_CLASS if name is None else str(name)
        under[at][counted] += 1
        for cut, character in enumerate(at):
            if character == "/":
                under[at[:cut]][counted] += 1
    places = []
    for place in sorted(records.of_kind("place"), key=lambda place: str(place["id"])):
        here = str(place["id"])
        level = place.get("level")
        places.append(
            {
                "id": here,
                "level": None if level is None else str(level),
                "names": sorted({str(name) for name in place.get("names") or []}),
                "devices": dict(sorted(under[here].items())) if here in under else {},
            }
        )
    return places


def _unplaced_devices(records: FacilityRecords) -> dict[str, int]:
    return _class_counts([d for d in records.of_kind("device") if d.get("place") is None])


def _authored_aliases(
    vocabulary: Mapping[str, Any], added: list[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Each class's aliases as authored: stripped, the first spelling of a term kept."""
    spellings: dict[str, dict[str, str]] = defaultdict(dict)
    rows = [(str(row["name"]), row) for row in vocabulary.get("classes", [])]
    rows += [(str(row["class"]), row) for row in added]
    for name, row in rows:
        for alias in row.get("aliases") or []:
            spelling = str(alias).strip()
            if spelling:
                spellings[name].setdefault(spelling.casefold(), spelling)
    return {name: sorted(seen.values()) for name, seen in spellings.items()}


def _device_classes(records: FacilityRecords) -> dict[str, dict[str, Any]]:
    """Each device class with its count, authored aliases and groups."""
    from osprey_facility.validate import vocabulary

    added = records.document.get("classes") or []
    aliases = _authored_aliases(vocabulary(), added)

    count: dict[str, int] = {str(row["class"]): 0 for row in added}
    for device in records.of_kind("device"):
        if device.get("class") is None:
            continue
        name = str(device["class"])
        count[name] = count.get(name, 0) + 1

    groups: dict[str, set[str]] = defaultdict(set)
    for group in records.of_kind("group"):
        for member in group.get("members") or []:
            held = records.record("device", str(member))
            if held is not None and held.get("class") is not None:
                groups[str(held["class"])].add(str(group["id"]))

    return {
        name: {
            "count": count[name],
            "aliases": aliases.get(name, []),
            "groups": sorted(groups.get(name, ())),
        }
        for name in sorted(count)
    }


def _vocabulary(records: FacilityRecords) -> dict[str, Any]:
    """The roles and signals the channels use, counted."""
    rows = records.document.get("signals") or []
    described = {str(row["name"]): row.get("description") for row in rows}
    roles: dict[str, int] = defaultdict(int)
    signals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    paired = unsigned = 0
    for channel in records.of_kind("channel"):
        role = str(channel.get("role") or DEFAULT_ROLE)
        roles[role] += 1
        if role == "setpoint" and channel.get("pair") not in (None, channel["id"]):
            paired += 1
        signal = channel.get("signal")
        if signal is None:
            unsigned += 1
        else:
            signals[str(signal)][role] += 1
    return {
        "roles": dict(sorted(roles.items())),
        "paired_setpoints": paired,
        "unsigned_channels": unsigned,
        "signals": {
            name: {
                "count": sum(signals[name].values()),
                "roles": dict(sorted(signals[name].items())),
                "description": described.get(name),
            }
            for name in sorted(signals)
        },
    }


def _models(records: FacilityRecords, served: list[str]) -> list[dict[str, Any]]:
    models = []
    for model in records.of_kind("model"):
        engine = str(model["engine"])
        settings = (model.get("settings") or {}).get(engine) or {}
        models.append(
            {
                "name": str(model["name"]),
                "engine": engine,
                "served": model["name"] in served,
                "solve": settings.get("solve"),
            }
        )
    return models


def facts_document(
    records: FacilityRecords,
    served: list[str],
    project_name: str | None = None,
    measurement_models: Mapping[str, Any] | None = None,
    measurement_kinds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The agent facts of one render.

    Args:
        records: The facility file's records.
        served: The models the render serves.
        project_name: The project's name, the display name of a facility that
            authors none.
        measurement_models: The render's pyAML view records, as
            :func:`measurement_records` reads them; none when omitted.
        measurement_kinds: Every registered kind's row; none when omitted.

    Returns:
        The document ``facility_facts.json`` holds.
    """
    return {
        "schema": FACTS_SCHEMA,
        "identity": _identity(records, project_name),
        "place_levels": _place_levels(records),
        "hierarchy_levels": hierarchy_levels(records),
        "places": _places(records),
        "unplaced_devices": _unplaced_devices(records),
        "device_classes": _device_classes(records),
        "vocabulary": _vocabulary(records),
        "models": _models(records, served),
        "measurement_models": dict(measurement_models or {}),
        "measurement_kinds": dict(measurement_kinds or {}),
        "channel_count": len(records.of_kind("channel")),
        "snapshot": None,
    }


def zero_source_facts(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The facts of a facility with no sources.

    Args:
        identity: The facility's ``{code, name, description}``.

    Returns:
        The facts document: no place, no class, ``texture`` alone and no
        channel.
    """
    return {
        "schema": FACTS_SCHEMA,
        "identity": dict(identity),
        "place_levels": [],
        "hierarchy_levels": [],
        "places": [],
        "unplaced_devices": {},
        "device_classes": {},
        "vocabulary": {
            "roles": {},
            "paired_setpoints": 0,
            "unsigned_channels": 0,
            "signals": {},
        },
        "models": [{"name": TEXTURE, "engine": TEXTURE, "served": True, "solve": None}],
        "measurement_models": {},
        "measurement_kinds": {},
        "channel_count": 0,
        "snapshot": None,
    }


# Every key a facts document carries; a file missing one is not read as facts.
FACTS_KEYS = frozenset(zero_source_facts({}))


def read_facts(render_root: Path, project_name: str) -> dict[str, Any]:
    """Read a render's agent facts.

    Args:
        render_root: The directory that holds the rendered ``config.yml``.
        project_name: The project's name, the identity of a render with neither
            a facts file nor a facility file.

    Returns:
        The render's ``data/facility_facts.json``; when the render holds none,
        it cannot be parsed or it lacks a key, the zero-source facts of the
        identity the render's facility file records, or of the project when
        there is no facility file.

    Raises:
        FacilityFileError: The facts file is not used and the facility file
            the identity falls back to is there but cannot be read, is not a
            JSON object, carries a header this reader does not read, or names
            no identity code; the message names the rebuild
            (:func:`~osprey.utils.facility.facility_identity`).
    """
    path = Path(render_root) / "data" / FACTS_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        document = None
    except (OSError, ValueError):
        logger.warning("The facility facts %s could not be read", path, exc_info=True)
        document = None
    if (
        isinstance(document, dict)
        and document.get("schema") == FACTS_SCHEMA
        and document.keys() >= FACTS_KEYS
        and isinstance(document["identity"], dict)
        and "name" in document["identity"]
    ):
        return document
    if document is not None:
        logger.warning("The facility facts %s are not a facts document", path)

    from osprey.utils.facility import facility_identity
    from osprey_facility import fold_code

    identity = facility_identity(Path(render_root), project_name) or {
        "code": fold_code(project_name),
        "name": project_name,
        "description": None,
    }
    return zero_source_facts(identity)


def measurement_records(
    data_root: Path, records: FacilityRecords | None = None
) -> dict[str, dict[str, Any]]:
    """The record of each pyAML view written under a render's ``data/``.

    Args:
        data_root: The render's ``data/`` directory.
        records: The facility file's records; when given, each record carries
            the model's kind block too.

    Returns:
        ``{model: {path, sha256, files}}``, sorted by model: ``path`` the
        configuration under the render, ``sha256`` its digest, ``files`` each
        other file of the view by its name in the view's directory, with its
        digest; with ``records``, ``kinds``, ``names``, ``instruments`` and
        ``rf`` as :func:`osprey.facility.views.pyaml.kind_records` reads them,
        each kind with the ``facts`` its writer wrote for the model's sub-block.
    """
    found: dict[str, dict[str, Any]] = {}
    for configuration in sorted((data_root / PYAML_DIR).glob(f"*/{CONFIGURATION_FILE}")):
        view = configuration.parent
        raw = configuration.read_bytes()
        found[view.name] = {
            "path": f"{data_root.name}/{PYAML_DIR}/{view.name}/{CONFIGURATION_FILE}",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "files": {
                path.relative_to(view).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(view.rglob("*"))
                if path.is_file() and path != configuration
            },
        }
        if records is not None:
            from osprey.facility.views.pyaml import kind_records
            from osprey_connectors import yaml_loader

            found[view.name] |= kind_records(
                records,
                view.name,
                yaml_loader.safe_load(raw.decode("utf-8")) or {},
                list(found[view.name]["files"]),
            )
    if records is not None and found:
        from osprey_facility.kinds import registered, written_facts

        for kind, by_model in written_facts(records.document, registered()).items():
            for model, facts in by_model.items():
                if model in found:
                    found[model]["kinds"][kind]["facts"] = facts
    return found


#: The stepped role the block's ``rf`` key carries rather than ``groups``: the
#: approval hook reads the RF frequency's setpoints from ``rf``.
_RF_ROLE = "rf"


def hook_measurement(facts: Mapping[str, Any], render_root: Path) -> dict[str, dict[str, Any]]:
    """The approval hook's ``measurement`` block of one render, read from its facts alone.

    Every value is copied from the model kind blocks the build wrote into the
    facts (``measurement_models[<model>]``); no file is read and no kind is
    looked up, so a launch-time regeneration gives the build's block.

    Args:
        facts: The facts document.
        render_root: The render's root; unused, kept for the callers' shape.

    Returns:
        ``{model: {kinds: {kind: {role: name}}, groups: {role: [address]},
        rf: [address], by_kinds: [kind], corrector_roles: {kind: [role]},
        view_sha256}}`` for each pyAML view, sorted by model: ``kinds`` each
        kind the file allows that steps a role, with the view name of each
        stepped role (``rf`` the RF plant), the role itself where the view has
        none; ``groups`` each group role a kind uses with the addresses its
        array's members stand for, and every other stepped role but the RF,
        group or instrument, with its members' addresses; ``rf`` the
        setpoint each of the view's RF plants steps, empty when it has none;
        ``by_kinds`` the kinds that take ``by``; ``corrector_roles`` the roles
        whose setpoints ``correctors`` may name, for each kind that takes them.
    """
    del render_root
    block: dict[str, dict[str, Any]] = {}
    for model, record in sorted((facts.get("measurement_models") or {}).items()):
        kinds: Mapping[str, Any] = record.get("kinds") or {}
        names: Mapping[str, Any] = record.get("names") or {}
        instruments: Mapping[str, Any] = record.get("instruments") or {}
        arrays = [str(role) for role in names if role not in instruments]
        stepped: dict[str, dict[str, str]] = {}
        groups: dict[str, list[str]] = {}
        for kind, entry in sorted(kinds.items()):
            roles = [str(role) for role in entry.get("stepped_roles") or []]
            if roles:
                stepped[str(kind)] = {role: str(names.get(role, role)) for role in roles}
            for role, member in (entry.get("members") or {}).items():
                if role in arrays or (role in roles and role != _RF_ROLE):
                    groups[str(role)] = list(member.get("addresses") or [])
        ordered = [*(role for role in arrays if role in groups), *sorted(set(groups) - set(arrays))]
        block[str(model)] = {
            "kinds": stepped,
            "groups": {role: groups[role] for role in ordered},
            "rf": list(record.get("rf") or []),
            "by_kinds": sorted(str(k) for k, e in kinds.items() if e.get("takes_by")),
            "corrector_roles": {
                str(k): list(e["corrector_roles"])
                for k, e in sorted(kinds.items())
                if e.get("corrector_roles")
            },
            "view_sha256": record.get("sha256"),
        }
    return copy.deepcopy(block)


def render_facts_page(facts: Mapping[str, Any]) -> str:
    """Render the facts as the page ``facility_facts.md`` holds.

    Args:
        facts: The facts document.

    Returns:
        The page's text, its first line the ``schema`` header.
    """
    from importlib import resources

    from jinja2 import Environment, FileSystemLoader

    environment = Environment(
        loader=FileSystemLoader(str(resources.files("osprey.templates"))),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(FACTS_TEMPLATE).render(facility_facts=facts)


def write_facts_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the agent facts view into ``root``.

    Args:
        root: The render's ``data/`` directory.
        inputs: The render's view inputs.

    Returns:
        The files written, sorted.
    """
    project_name = inputs.rendered_config.get("project_name")
    from osprey_facility.kinds import kind_table_row, registered

    models = measurement_records(root, inputs.records)
    declared = registered() if models else {}
    table = {name: kind_table_row(declaration) for name, declaration in declared.items()}
    facts = facts_document(
        inputs.records,
        inputs.served,
        str(project_name) if project_name else None,
        models,
        table,
    )
    root.mkdir(parents=True, exist_ok=True)
    document = root / FACTS_FILE
    document.write_bytes(view_bytes(facts))
    page = root / FACTS_PAGE
    page.write_bytes(render_facts_page(facts).encode("utf-8"))
    return sorted([document, page])
