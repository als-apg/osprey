"""The agent facts view: what a render's agents are told about the facility.

Written to ``<render>/data/``::

    facility_facts.json   {schema: osprey.facility.facility_facts/2, identity,
                           place_levels, places, unplaced_devices,
                           device_classes, vocabulary, models,
                           measurement_models, channel_count, snapshot}
    facility_facts.md     the same facts as one page, rendered once from
                          ``_facility_facts.md.j2``

``identity`` is ``{code, name, description}``; a facility that authors no name
takes the project's. ``place_levels`` is the distinct ``level`` words of the
places, shallowest first. ``places`` has one entry per place, sorted by id:
its ``level``, its ``names`` and the ``devices`` at or under it counted by
class. ``unplaced_devices`` counts by class the devices that sit in no place.
``device_classes`` has one entry per class a device carries and per
facility-added class: ``count`` devices, the ``aliases`` the vocabulary and
``classes.yaml`` give the class in their authored spelling, and ``groups``, the
sorted ids of the groups holding one of its devices. ``vocabulary`` counts the
channels by role and by signal, with each signal's vocabulary description, the
setpoints paired with a readback and the channels that carry no signal.
``models`` lists every model with its ``engine``, whether the render serves it
and its engine's ``solve`` setting. ``measurement_models`` holds one record per
measurement view the render carries, ``channel_count`` counts the channels, and
``snapshot`` is ``null``.

A render with no facts file is read as the zero-source facts: the identity of
its facility file or project name, no place, no class, ``texture`` alone and no
channel.

The schema's number rises whenever a key is added, removed or renamed or a
value changes shape; ``read_facts`` reads only the current number and takes
any other file for the zero-source facts, which every build rewrites.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility import TEXTURE
from osprey.facility.views import ViewInputs, view_bytes

__all__ = [
    "FACTS_FILE",
    "FACTS_PAGE",
    "FACTS_SCHEMA",
    "FACTS_TEMPLATE",
    "facts_document",
    "hook_measurement",
    "read_facts",
    "render_facts_page",
    "write_facts_view",
    "zero_source_facts",
]

logger = logging.getLogger(__name__)

FACTS_FILE = "facility_facts.json"
FACTS_PAGE = "facility_facts.md"
FACTS_SCHEMA = "osprey.facility.facility_facts/2"

#: The class a classless device is counted under.
NO_CLASS = "-"

#: The role a channel that states none has.
DEFAULT_ROLE = "readback"

#: The page's template, relative to the packaged templates directory.
FACTS_TEMPLATE = "claude_code/_facility_facts.md.j2"


def _identity(doc: Mapping[str, Any], project_name: str | None) -> dict[str, Any]:
    from osprey.utils.facility import identity_record

    return dict(identity_record(doc["identity"], project_name))


def _place_levels(doc: Mapping[str, Any]) -> list[str]:
    depth: dict[str, int] = {}
    for place in doc.get("places", []):
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


def _places(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each place with its level, its names and the devices at or under it by class."""
    devices = doc.get("devices", [])
    places = []
    for place in sorted(doc.get("places", []), key=lambda place: str(place["id"])):
        here = str(place["id"])
        level = place.get("level")
        places.append(
            {
                "id": here,
                "level": None if level is None else str(level),
                "names": sorted({str(name) for name in place.get("names") or []}),
                "devices": _class_counts(
                    [
                        device
                        for device in devices
                        if device.get("place") is not None
                        and (
                            str(device["place"]) == here
                            or str(device["place"]).startswith(f"{here}/")
                        )
                    ]
                ),
            }
        )
    return places


def _unplaced_devices(doc: Mapping[str, Any]) -> dict[str, int]:
    return _class_counts([d for d in doc.get("devices", []) if d.get("place") is None])


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


def _device_classes(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each device class with its count, authored aliases and groups."""
    from osprey.facility.validate import vocabulary

    added = doc.get("classes") or []
    aliases = _authored_aliases(vocabulary(), added)

    class_of: dict[str, str] = {}
    count: dict[str, int] = {str(row["class"]): 0 for row in added}
    for device in doc.get("devices", []):
        if device.get("class") is None:
            continue
        name = str(device["class"])
        class_of[str(device["id"])] = name
        count[name] = count.get(name, 0) + 1

    groups: dict[str, set[str]] = defaultdict(set)
    for group in doc.get("groups", []):
        for member in group.get("members") or []:
            if str(member) in class_of:
                groups[class_of[str(member)]].add(str(group["id"]))

    return {
        name: {
            "count": count[name],
            "aliases": aliases.get(name, []),
            "groups": sorted(groups.get(name, ())),
        }
        for name in sorted(count)
    }


def _vocabulary(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The roles and signals the channels use, counted."""
    from osprey.facility.validate import vocabulary

    described = {str(row["name"]): row.get("description") for row in vocabulary()["signal_roles"]}
    roles: dict[str, int] = defaultdict(int)
    signals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    paired = unsigned = 0
    for channel in doc.get("channels", []):
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


def _models(doc: Mapping[str, Any], served: list[str]) -> list[dict[str, Any]]:
    models = []
    for model in doc.get("models", []):
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
    doc: Mapping[str, Any], served: list[str], project_name: str | None = None
) -> dict[str, Any]:
    """The agent facts of one render.

    Args:
        doc: The facility file.
        served: The models the render serves.
        project_name: The project's name, the display name of a facility that
            authors none.

    Returns:
        The document ``facility_facts.json`` holds.
    """
    return {
        "schema": FACTS_SCHEMA,
        "identity": _identity(doc, project_name),
        "place_levels": _place_levels(doc),
        "places": _places(doc),
        "unplaced_devices": _unplaced_devices(doc),
        "device_classes": _device_classes(doc),
        "vocabulary": _vocabulary(doc),
        "models": _models(doc, served),
        "measurement_models": {},
        "channel_count": len(doc.get("channels", [])),
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
        The render's ``data/facility_facts.json``, or the zero-source facts
        when the render holds none, it cannot be parsed or it lacks a key.
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
        and FACTS_KEYS <= document.keys()
        and isinstance(document["identity"], dict)
        and "name" in document["identity"]
    ):
        return document
    if document is not None:
        logger.warning("The facility facts %s are not a facts document", path)

    from osprey.facility import fold_code
    from osprey.utils.facility import facility_identity

    identity = facility_identity(Path(render_root), project_name) or {
        "code": fold_code(project_name),
        "name": project_name,
        "description": None,
    }
    return zero_source_facts(identity)


def hook_measurement(facts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The approval hook's ``measurement`` block of one render's facts.

    Args:
        facts: The facts document.

    Returns:
        ``{model: {view_sha256}}`` for each measurement view the render
        carries, sorted by model.
    """
    recorded = facts.get("measurement_models") or {}
    return {
        str(model): {"view_sha256": record.get("sha256")}
        for model, record in sorted(recorded.items())
    }


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
    facts = facts_document(inputs.doc, inputs.served, str(project_name) if project_name else None)
    root.mkdir(parents=True, exist_ok=True)
    document = root / FACTS_FILE
    document.write_bytes(view_bytes(facts))
    page = root / FACTS_PAGE
    page.write_bytes(render_facts_page(facts).encode("utf-8"))
    return sorted([document, page])
