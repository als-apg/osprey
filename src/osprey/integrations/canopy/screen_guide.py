"""Build-time guide to a project's CANOPY screen library, for the ``canopy-screens`` skill.

CANOPY's ``list_drafts`` names each library screen by its title and ``ref``, which
tells the agent neither what a screen is for nor which devices it shows. This module
reads the project's library once, at build time, and returns for every screen its
``ref``, title, the first sentence of its description, the screens it opens, and the
devices it draws or binds, by system and facility label ("SR: BPM 37", "SR: EPU7"). The
skill renders that listing, so the agent can answer "show me ION-PUMP 38" by opening
the screen that holds it. The guide is derived from the library the deployment
ships, so it cannot disagree with it.

A device belongs to a screen when a node names it as its ``tooltip``, or when the
screen binds one of its channels: a widget's ``pv``, a rule's input, a formula's
binding, a write action's target, or an embedded screen's macro. Without the
build's ``facility.json`` only the tooltips name devices.

Fail-soft: an unreadable or malformed screen is skipped with a warning, never a
build failure.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.integrations.canopy.library import SCREEN_SUFFIX, SCREENS_DIRNAME

logger = logging.getLogger(__name__)

#: The prefix CANOPY gives a library screen's ``ref``.
REF_PREFIX = "library/"
#: How many labels in a numbered run of one stem shorten to ``first–last``.
RUN_MIN = 3

_TRAILING_NUMBER = re.compile(r"^(?P<stem>.*?)(?P<number>\d+)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


@dataclass(frozen=True)
class GuideScreen:
    """One library screen: where it is, what it is, what it opens, what it shows."""

    ref: str
    title: str
    summary: str
    opens: tuple[str, ...]
    #: ``(system, labels)`` per system the screen's devices belong to (the device
    #: id's first segment, ``SR`` for ``SR/BPM37``), labels compressed.
    devices: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def is_sub_screen(self) -> bool:
        """A screen whose own name starts with ``_`` is opened from another, not listed."""
        return self.ref.rsplit("/", 1)[-1].startswith("_")


@dataclass(frozen=True)
class ScreenGuide:
    screens: tuple[GuideScreen, ...]
    warnings: tuple[str, ...] = field(default=())


def build_screen_guide(library: Path, facility: Mapping[str, Any] | None) -> ScreenGuide:
    """The guide to every screen under ``library/screens``, in ``ref`` order.

    Args:
        library: The project's screen library, ``data/canopy/library``.
        facility: The build's ``facility.json``, or ``None`` to name devices by
            tooltip only.
    """
    root = library / SCREENS_DIRNAME
    if not root.is_dir():
        return ScreenGuide(screens=())
    device_of, label_of = _facility_maps(facility)
    screens, warnings = [], []
    for path in sorted(root.rglob(f"*{SCREEN_SUFFIX}")):
        screen_id = path.relative_to(root).as_posix()[: -len(SCREEN_SUFFIX)]
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"{screen_id}: not read ({exc})")
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("root"), dict):
            warnings.append(f"{screen_id}: no screen root")
            continue
        screens.append(_entry(screen_id, doc, device_of, label_of))
    return ScreenGuide(screens=tuple(screens), warnings=tuple(warnings))


def load_facility(path: Path) -> Mapping[str, Any] | None:
    """The build's ``facility.json``, or ``None`` when it is absent or unreadable."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


#: The CANOPY tools whose default view the profile sets, by the key that sets it.
TOOL_QUERIES = (
    ("trace", "Data Browser (Trace)", "trace_default_query"),
    ("correlation", "Correlation", "correlation_default_query"),
    ("waveform", "Waveform", "waveform_default_query"),
)


def screen_guide_context(config: Mapping[str, Any], project_dir: Path) -> dict | None:
    """Index the project's CANOPY screens, tools and collections for the screens skill.

    The template context of the ``canopy-screens`` skill. Runs only when
    ``config.yml`` declares the CANOPY panel. Reads the built project's
    ``data/canopy/library``, its ``facility.json`` for device labels, its
    ``data/canopy/collections`` and the tools' default views from
    ``web.panels.canopy``. Never raises: an unreadable piece is a build warning
    and leaves its part of the guide empty.

    Returns:
        ``{"screens": [...], "tools": [...], "collections": [...]}``, or
        ``None`` when the panel is not declared.
    """
    from osprey.deployment.web_terminals.roster import config_declares_panel

    if not config_declares_panel(config, "canopy"):
        return None

    data = project_dir / "data" / "canopy"
    try:
        guide = build_screen_guide(data / "library", load_facility(project_dir / "facility.json"))
    except Exception as exc:  # pragma: no cover - defensive; the builder is fail-soft
        logger.warning("canopy screen guide skipped: %s", exc)
        return None
    for warning in guide.warnings:
        logger.warning("canopy screen guide: %s", warning)

    panel = ((config.get("web") or {}).get("panels") or {}).get("canopy") or {}
    tools = [
        {"tool": tool, "title": title, "query": panel[key]}
        for tool, title, key in TOOL_QUERIES
        if isinstance(panel.get(key), str) and panel[key]
    ]
    collections = []
    for path in sorted((data / "collections").glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("canopy screen guide: collection %s not read (%s)", path.name, exc)
            continue
        members = []
        for member in document.get("members") or []:
            if not isinstance(member, dict):
                continue
            if member.get("screen"):
                members.append(str(member["screen"]))
            elif member.get("tool"):
                label = f"{member.get('name') or member['tool']} ({member['tool']}"
                if isinstance(member.get("query"), str) and member["query"]:
                    label += f" on `{member['query']}`"
                members.append(label + ")")
        collections.append(
            {
                "name": str(document.get("name") or path.stem),
                "description": str(document.get("description") or ""),
                "members": members,
            }
        )
    return {
        "screens": [
            {
                "ref": screen.ref,
                "title": screen.title,
                "summary": screen.summary,
                "sub_screen": screen.is_sub_screen,
                "opens": list(screen.opens),
                "devices": [
                    {"system": system, "labels": list(labels)} for system, labels in screen.devices
                ],
            }
            for screen in guide.screens
        ],
        "tools": tools,
        "collections": collections,
    }


def compress(labels: Iterable[str]) -> list[str]:
    """``labels`` in natural order, each numbered run of one stem as ``first–last``.

    ``ION-PUMP 37`` … ``ION-PUMP 42`` read ``ION-PUMP 37–42``; a run shorter than
    :data:`RUN_MIN` stays listed. A run holds numbers written with the same number of
    digits, so ``BPM 01``…``BPM 72`` read ``BPM 01–72``.
    """
    runs: dict[tuple[str, int], list[int]] = {}
    out: list[tuple[str, int, str]] = []
    for label in set(labels):
        match = _TRAILING_NUMBER.match(label)
        if match is None:
            out.append((label.lower(), -1, label))
            continue
        digits = match["number"]
        width = len(digits)
        runs.setdefault((match["stem"], width), []).append(int(digits))
    for (stem, width), numbers in runs.items():
        for run in _runs(sorted(numbers)):
            if len(run) >= RUN_MIN:
                out.append((stem.lower(), run[0], f"{stem}{run[0]:0{width}d}–{run[-1]:0{width}d}"))
            else:
                out.extend((stem.lower(), n, f"{stem}{n:0{width}d}") for n in run)
    return [text for *_, text in sorted(out)]


def _runs(numbers: list[int]) -> Iterator[list[int]]:
    run: list[int] = []
    for n in numbers:
        if run and n != run[-1] + 1:
            yield run
            run = []
        run.append(n)
    if run:
        yield run


def _facility_maps(
    facility: Mapping[str, Any] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """``(channel → device, device → label)`` from a facility document."""
    if not facility:
        return {}, {}
    device_of = {
        c["id"]: c["on"]["device"]
        for c in facility.get("channels") or []
        if isinstance(c, dict) and isinstance(c.get("on"), dict) and c["on"].get("device")
    }
    label_of = {
        d["id"]: _spoken(d.get("label") or d["id"])
        for d in facility.get("devices") or []
        if isinstance(d, dict) and d.get("id")
    }
    return device_of, label_of


def _spoken(label: str) -> str:
    """A label without its parenthetical detail: ``EPU7 (50 mm)`` reads ``EPU7``."""
    return label.split(" (")[0].strip() or label


def _entry(
    screen_id: str, doc: dict, device_of: Mapping[str, str], label_of: Mapping[str, str]
) -> GuideScreen:
    held = doc.get("metadata")
    metadata = held if isinstance(held, dict) else {}
    title = str(metadata.get("title") or screen_id)
    description = " ".join(str(metadata.get("description") or "").split())
    summary = _SENTENCE_END.split(description, maxsplit=1)[0] if description else ""
    devices: set[str] = set()
    opens: set[str] = set()
    for name in _bound_names(doc):
        if name in device_of:
            devices.add(device_of[name])
        elif name in label_of:
            devices.add(name)
    for node in _nodes(doc["root"]):
        held_props = node.get("props")
        props = held_props if isinstance(held_props, dict) else {}
        tip = props.get("tooltip")
        if isinstance(tip, str) and (tip in label_of or (not label_of and "/" in tip)):
            devices.add(tip)
        for action in props.get("actions") or []:
            if isinstance(action, dict) and action.get("type") == "open_display":
                target = _resolve(screen_id, str(action.get("file") or ""))
                if target:
                    opens.add(REF_PREFIX + target)
    systems: dict[str, list[str]] = {}
    for device in devices:
        systems.setdefault(device.split("/", 1)[0], []).append(label_of.get(device, device))
    return GuideScreen(
        ref=REF_PREFIX + screen_id,
        title=title,
        summary=summary,
        opens=tuple(sorted(opens)),
        devices=tuple((system, tuple(compress(systems[system]))) for system in sorted(systems)),
    )


def _nodes(node: Any) -> Iterator[dict]:
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children") or []:
        yield from _nodes(child)


def _bound_names(doc: dict) -> Iterator[str]:
    """Every channel name the screen binds, without telling them apart from macros."""
    for formula in doc.get("formulas") or []:
        if isinstance(formula, dict):
            yield from (v for v in (formula.get("bindings") or {}).values() if isinstance(v, str))
    for node in _nodes(doc["root"]):
        held_props = node.get("props")
        props = held_props if isinstance(held_props, dict) else {}
        if isinstance(props.get("pv"), str):
            yield props["pv"]
        for value in (props.get("macros") or {}).values():
            if isinstance(value, str):
                yield value
        for action in props.get("actions") or []:
            if isinstance(action, dict) and isinstance(action.get("pvName"), str):
                yield action["pvName"]
        for rule in node.get("rules") or []:
            for pv in rule.get("pvs") or [] if isinstance(rule, dict) else []:
                if isinstance(pv, dict) and isinstance(pv.get("name"), str):
                    yield pv["name"]


def _resolve(declaring: str, file: str) -> str | None:
    """The screen id an ``open_display`` file names, relative to ``declaring``'s folder."""
    if not file.endswith(SCREEN_SUFFIX):
        return None
    parts = declaring.split("/")[:-1]
    for piece in file.split("/"):
        if piece == "..":
            if not parts:
                return None
            parts.pop()
        elif piece and piece != ".":
            parts.append(piece)
    return "/".join(parts)[: -len(SCREEN_SUFFIX)]
