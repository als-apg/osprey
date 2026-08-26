"""Read the queueserver worker's plan-device set from a YAML/JSON device file.

The worker runs in its own container and venv: it cannot import OSPREY's
virtual-accelerator manifest, the channel-finder database, or anything else
host-side to discover which channels are scan devices. So the device set
arrives as a file, mounted into the container and named by
``BLUESKY_DEVICES_FILE``, and is parsed here into the control-room-neutral
``SettableSpec``/``ReadableSpec`` shapes that ``devices/connector.py`` builds
from.

The document has exactly two top-level keys, both optional::

    settables:
      - name: SR:C01:QF:1          # device name; also the event-data column key
        setpoint: SR:C01:QF:1:SP
        readback: SR:C01:QF:1:RB   # optional; omitted => reads the setpoint PV
    readables:
      - name: SR:C01:BPM:1:X
        pv: SR:C01:BPM:1:X:RB

``yaml.safe_load`` parses it, so a ``.json`` file is read by the same code
path. Nothing here splits a value on any character, which is the point of the
format: EPICS PV names may contain commas (16 of ALS's BTS quadrupoles do),
and the env-var channel this replaced used commas as its entry separator.

Parsing is **fail-soft**. A malformed entry is skipped with a warning rather
than raised on — one typo must not cost every other device in a 13k-entry file
its connection — and an unreadable, empty or structurally wrong file yields an
empty device set, which the caller reports as "this worker will expose no
plans". The *build* is where a bad file is refused loudly:
``validate_device_document`` returns the human-readable problem list it uses.

Which channels are scan devices, and which readback pairs with which setpoint,
is a projection of the facility's namespace. That same namespace is described
by the channel-finder database and the knowledge graph; the three are to be
kept in step, and unifying them is a later piece of work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .specs import ReadableSpec, SettableSpec

logger = logging.getLogger("osprey.services.bluesky_bridge.devices._specs_from_file")

SETTABLES_KEY = "settables"
"""Top-level key holding the settable (setpoint/readback) device entries."""

READABLES_KEY = "readables"
"""Top-level key holding the read-only device entries."""

TOP_LEVEL_KEYS = (SETTABLES_KEY, READABLES_KEY)
"""The only keys the document may carry; anything else rejects the document."""

_SETTABLE_REQUIRED = ("name", "setpoint")
_SETTABLE_OPTIONAL = ("readback",)
_READABLE_REQUIRED = ("name", "pv")
_READABLE_OPTIONAL: tuple[str, ...] = ()


def _clean(value: Any) -> str | None:
    """Return ``value`` as a stripped non-empty string, or ``None`` if it is not
    one (wrong type, empty, or whitespace only).

    Names are stripped because a name with trailing whitespace becomes a
    subtly wrong event-data column key rather than an obvious error.
    """
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _load_document(path: Path) -> Any:
    """Parse the file at ``path``, returning ``None`` when it cannot be used.

    ``UnicodeDecodeError`` joins the caught errors because a binary file
    handed to the loader is a bad *file*, not a bug to crash the worker over.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("device file %s could not be read (%s); no devices built", path, exc)
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("device file %s is not valid YAML/JSON (%s); no devices built", path, exc)
        return None


def _entries(section: Any, key: str) -> list[Any]:
    """Return the entry list under ``key``, warning and yielding none when the
    section is present but is not a list."""
    if section is None:
        return []
    if not isinstance(section, list):
        logger.warning(
            "%r must be a list of device entries, got %s; skipping the section",
            key,
            type(section).__name__,
        )
        return []
    return section


def _parse_settables(section: Any) -> list[SettableSpec]:
    """Parse the ``settables`` section, skipping malformed entries with a warning.

    An entry needs a non-empty ``name`` and ``setpoint``. ``readback`` is
    optional: absent or explicitly null means the device reads its setpoint PV;
    present but not a non-empty string is a malformed entry, because silently
    reading the setpoint would hide a typo'd readback address.
    """
    specs: list[SettableSpec] = []
    for index, entry in enumerate(_entries(section, SETTABLES_KEY)):
        where = f"{SETTABLES_KEY}[{index}]"
        if not isinstance(entry, dict):
            logger.warning("skipping %s: entry is not a mapping (%r)", where, entry)
            continue

        name = _clean(entry.get("name"))
        setpoint_pv = _clean(entry.get("setpoint"))
        if name is None or setpoint_pv is None:
            logger.warning(
                "skipping malformed %s (%r): needs a non-empty 'name' and 'setpoint'",
                where,
                entry.get("name"),
            )
            continue

        readback_pv: str | None = None
        if entry.get("readback") is not None:
            readback_pv = _clean(entry["readback"])
            if readback_pv is None:
                logger.warning(
                    "skipping %s (%r): 'readback' is present but not a non-empty string",
                    where,
                    name,
                )
                continue

        specs.append(SettableSpec(name=name, setpoint_pv=setpoint_pv, readback_pv=readback_pv))
    return specs


def _parse_readables(section: Any) -> list[ReadableSpec]:
    """Parse the ``readables`` section, skipping malformed entries with a warning.

    An entry needs a non-empty ``name`` and ``pv``; see ``_parse_settables``.
    """
    specs: list[ReadableSpec] = []
    for index, entry in enumerate(_entries(section, READABLES_KEY)):
        where = f"{READABLES_KEY}[{index}]"
        if not isinstance(entry, dict):
            logger.warning("skipping %s: entry is not a mapping (%r)", where, entry)
            continue

        name = _clean(entry.get("name"))
        read_pv = _clean(entry.get("pv"))
        if name is None or read_pv is None:
            logger.warning(
                "skipping malformed %s (%r): needs a non-empty 'name' and 'pv'",
                where,
                entry.get("name"),
            )
            continue

        specs.append(ReadableSpec(name=name, read_pv=read_pv))
    return specs


def _drop_duplicate_names(
    setpoints: list[SettableSpec], readbacks: list[ReadableSpec]
) -> tuple[list[SettableSpec], list[ReadableSpec]]:
    """Drop any spec whose device name was already seen (setpoints first, then
    readbacks), warning on each collision.

    Device names become ophyd-async device names *and* event-data column keys;
    two devices sharing a name would make the scanned column ambiguous (see the
    bridge's device-column lookup), so a later entry that reuses an
    already-claimed name is dropped rather than silently shadowing the first.
    """
    seen: set[str] = set()

    def _keep(specs: list) -> list:
        kept = []
        for spec in specs:
            if spec.name in seen:
                logger.warning(
                    "skipping device %r: name already claimed by an earlier "
                    "setpoint/readback entry",
                    spec.name,
                )
                continue
            seen.add(spec.name)
            kept.append(spec)
        return kept

    return _keep(setpoints), _keep(readbacks)


def specs_from_file(path: str | Path) -> tuple[list[SettableSpec], list[ReadableSpec]]:
    """Read the device file at ``path`` into settable and readable specs.

    Returns ``([], [])`` — never raises — when the file is unreadable, empty,
    unparseable, not a mapping, or carries an unknown top-level key. The last
    two reject the *whole* document with a single warning rather than loading
    the half of it that parsed: a typo'd ``readable:`` key would otherwise ship
    a deployment that looks healthy while quietly exposing no readbacks.
    Individual malformed entries are skipped with their own warning, and any
    device name that collides with an earlier one is dropped
    (``_drop_duplicate_names``, settables resolving first).
    """
    doc = _load_document(Path(path))
    if doc is None:
        return [], []

    if not isinstance(doc, dict):
        logger.warning(
            "device file %s must hold a mapping with %r keys, got %s; no devices built",
            path,
            list(TOP_LEVEL_KEYS),
            type(doc).__name__,
        )
        return [], []

    unknown = [key for key in doc if key not in TOP_LEVEL_KEYS]
    if unknown:
        logger.warning(
            "device file %s has unknown top-level key(s) %r (expected only %r); no devices built",
            path,
            unknown,
            list(TOP_LEVEL_KEYS),
        )
        return [], []

    settables = _parse_settables(doc.get(SETTABLES_KEY))
    readables = _parse_readables(doc.get(READABLES_KEY))
    return _drop_duplicate_names(settables, readables)


def _validate_section(
    section: Any, key: str, required: tuple[str, ...], optional: tuple[str, ...]
) -> list[str]:
    """Return the problems in one device section (empty list ⇒ none)."""
    if section is None:
        return []
    if not isinstance(section, list):
        return [f"{key!r} must be a list of device entries, got {type(section).__name__}"]

    known = set(required) | set(optional)
    problems: list[str] = []
    for index, entry in enumerate(section):
        where = f"{key}[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: entry must be a mapping, got {type(entry).__name__}")
            continue
        for field in required:
            if _clean(entry.get(field)) is None:
                problems.append(f"{where}: {field!r} must be a non-empty string")
        for field in optional:
            if entry.get(field) is not None and _clean(entry[field]) is None:
                problems.append(f"{where}: {field!r} must be a non-empty string when present")
        unknown = [name for name in entry if name not in known]
        if unknown:
            problems.append(f"{where}: unknown key(s) {unknown!r}; expected only {sorted(known)!r}")
    return problems


def _duplicate_name_problems(doc: dict) -> list[str]:
    """Return one problem per device name already claimed by an earlier entry.

    The loader drops these (see ``_drop_duplicate_names``), so a file the build
    accepted with duplicates would ship fewer devices than it lists.
    """
    seen: set[str] = set()
    problems: list[str] = []
    for key in TOP_LEVEL_KEYS:
        section = doc.get(key)
        if not isinstance(section, list):
            continue
        for index, entry in enumerate(section):
            if not isinstance(entry, dict):
                continue
            name = _clean(entry.get("name"))
            if name is None:
                continue
            if name in seen:
                problems.append(
                    f"{key}[{index}]: device name {name!r} is already claimed by an earlier "
                    "entry and would be dropped"
                )
            else:
                seen.add(name)
    return problems


def validate_device_document(doc: Any) -> list[str]:
    """Return the problems in a parsed device document; empty list ⇒ valid.

    This is the strict counterpart to ``specs_from_file``: the loader degrades
    quietly so a running worker survives a bad file, while the build calls this
    and refuses to stage a file that reports anything. It reports every problem
    it finds rather than the first, each naming its section and index so a
    13k-entry file can be repaired without bisecting it, and it never raises —
    the caller may hand it any object ``yaml.safe_load`` could return.

    An empty (``None``) or partial document is *valid*: a device file may
    legitimately list only readables, or nothing at all. Unknown keys are not,
    at either level, because they are how a typo presents itself.
    """
    if doc is None:
        return []
    if not isinstance(doc, dict):
        return [
            f"top-level document must be a mapping with {list(TOP_LEVEL_KEYS)!r} keys, "
            f"got {type(doc).__name__}"
        ]

    problems: list[str] = []
    unknown = [key for key in doc if key not in TOP_LEVEL_KEYS]
    if unknown:
        problems.append(
            f"unknown top-level key(s) {unknown!r}; expected only {list(TOP_LEVEL_KEYS)!r}"
        )
    problems += _validate_section(
        doc.get(SETTABLES_KEY), SETTABLES_KEY, _SETTABLE_REQUIRED, _SETTABLE_OPTIONAL
    )
    problems += _validate_section(
        doc.get(READABLES_KEY), READABLES_KEY, _READABLE_REQUIRED, _READABLE_OPTIONAL
    )
    problems += _duplicate_name_problems(doc)
    return problems
