"""The per-user bar layout on disk.

One JSON document — ``{version, rev, header[], status[], header_visible,
status_visible}`` —
holding how one operator arranged their header and status bar. It is the
persistence half of the bar-items contract; the browser's normalizer
(``static/js/bar-layout.js``) owns the rendering half, and this module is
written to agree with it exactly.

**Where it lives.** The caller supplies the directory. In practice that is
``resolve_shared_data_root() / "bar_items"``: in a single-user deployment the
one operator's directory, and in a multi-user one the same in-container path
backed by a per-user named volume. Isolation comes from the mount, so the
identity never appears in the path.

**What each entry point promises.**

*Loading never fails.* :func:`load_layout` answers the deployment default for a
document that is absent, unreadable, unparseable, written against a schema
version this build cannot read, or structurally not a layout. A corrupt
preferences blob costs the operator their arrangement; it must never cost them
the terminal.

*Saving refuses exactly what the browser refuses.* :func:`save_layout` rejects
an unknown item type, a type placed in a bar that cannot hold it, more items
than a bar may carry, a malformed entry, and an option value outside its spec —
the same five classes ``bar-layout.js`` drops an entry for. It deliberately
does **not** check an item's runtime availability: the browser can ask whether
a bridge is reachable and the server cannot, and refusing here would make a
layout unsavable for as long as a service happened to be down.

*Revisions are the store's to assign.* Every accepted save lands one above
whatever is on disk, whatever revision the caller claimed. An optional
``expected_rev`` closes the read-modify-write window so a stale editor is
recognised rather than believed.

**What lives elsewhere.** HTTP semantics are the route's: it maps
:class:`BarLayoutInvalid` to 422, :class:`BarLayoutConflict` to 409, and an
``OSError`` escaping a write — an unwritable or missing mount — to 503. And the
item vocabulary is a *parameter*, not a fact of this module: ``app.py`` already
mirrors the browser catalog and is pinned against it by
``tests/interfaces/web_terminal/test_bar_items_ssr.py``, so a second copy here
would be one more place to drift. Passing :class:`BarVocabulary` in also keeps
the dependency pointing one way — ``app.py`` imports this module, never the
reverse.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json_store import read_json_object, write_json_atomic

logger = logging.getLogger(__name__)

__all__ = [
    "LAYOUT_FILENAME",
    "BarLayoutConflict",
    "BarLayoutError",
    "BarLayoutInvalid",
    "BarVocabulary",
    "layout_path",
    "load_layout",
    "reset_layout",
    "save_layout",
]

#: The document's name inside the store directory. One file, so a save is a
#: single ``os.replace`` and needs no ordering between writes.
LAYOUT_FILENAME = "layout.json"

#: One option's spec, mirroring ``BarOptionSpec`` in ``bar-catalog.js``:
#: ``{"kind": "number", "min": …, "max": …, "default": …}``,
#: ``{"kind": "boolean", "default": …}`` or
#: ``{"kind": "enum", "values": [...], "default": …}``.
BarOptionSpec = Mapping[str, Any]

#: One item type, mirroring a ``BAR_CATALOG`` entry as far as this module cares:
#: ``{"options": {name: BarOptionSpec}, "multi": bool}``. ``multi`` false (or
#: absent) means the type may be placed once per document. Any type may sit
#: in either bar, so the spec names no hosts.
BarItemSpec = Mapping[str, Any]


@dataclass(frozen=True)
class BarVocabulary:
    """The deployment facts a layout is validated against.

    Everything this module would otherwise have to know about bar items arrives
    here, so the store can be exercised against a fixture catalog and cannot
    become a second authority on item names, the schema version or the cap.

    Attributes:
        items: Item type -> its spec. A type absent from this mapping is an
            unknown type, exactly as it is for the browser's normalizer.
        version: The schema version this build reads and writes. A stored
            document carrying anything else is unreadable, not repairable.
        max_items_per_host: How many items one bar may hold. Per host rather
            than per document, so a legal header edit never fails because of
            what is in the status bar.
        hosts: The document's two item lists, in render order. Part of the
            schema rather than of the catalog, which is why it has a default.
    """

    items: Mapping[str, BarItemSpec]
    version: int
    max_items_per_host: int
    hosts: tuple[str, ...] = ("header", "status")


class BarLayoutError(Exception):
    """Base for every reason the store will not persist a layout."""


class BarLayoutInvalid(BarLayoutError):
    """A layout document the store refuses to write.

    Carries the machine-readable :attr:`reason` alongside the message so a
    route can answer 422 with a body naming the class of problem, and a log
    line can be grouped without parsing prose.

    Attributes:
        reason: One of ``malformed``, ``version``, ``unknown-type``,
            ``duplicate``, ``overflow`` or ``bad-option``. The four after
            ``version`` are spelled exactly as ``bar-layout.js`` spells its
            drop reasons, so client logs and server responses read alike.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


class BarLayoutConflict(BarLayoutError):
    """A save whose ``expected_rev`` no longer matches what is on disk.

    Attributes:
        expected: The revision the caller believed it was editing.
        actual: The revision actually stored.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"the stored bar layout is at revision {actual}, not {expected}; reload before saving"
        )
        self.expected = expected
        self.actual = actual


def layout_path(store_dir: Path) -> Path:
    """Return the layout document's path inside *store_dir*."""
    return store_dir / LAYOUT_FILENAME


def load_layout(
    store_dir: Path,
    *,
    vocabulary: BarVocabulary,
    default: Mapping[str, Any],
) -> dict[str, Any]:
    """Return this operator's layout, or the deployment default.

    Never raises and never writes: reading a store is not a reason to create
    its directory. The result is always a fresh, complete document, so a caller
    may hold it, mutate it, or hand it to a template without copying.

    Validation here is of the *envelope* only — the schema version, the
    revision, the two item lists and their entries' shape. It deliberately does
    not check item types against *vocabulary*, because an item this build
    cannot render is one entry the browser drops, not evidence the document is
    damaged. Discarding the whole arrangement over a retired item would destroy
    a layout that a rollback or a re-enabled panel would make whole again.

    Args:
        store_dir: The directory holding the layout document.
        vocabulary: Deployment facts; only the schema version and host names
            are consulted.
        default: This deployment's own layout, used whenever nothing readable
            is stored. If it is itself unreadable, an empty document is
            returned instead — rendering nothing rather than guessing an order
            the deployment never declared.

    Returns:
        A layout document. ``rev`` is ``0`` when nothing has been saved, which
        is also what a first conditional save should expect.
    """
    path = layout_path(store_dir)
    stored = read_json_object(path)
    if stored is not None:
        document = _read_envelope(stored, vocabulary)
        if document is not None:
            return document
        logger.warning(
            "bar layout at %s is not a readable version-%d document; "
            "falling back to the deployment default.",
            path,
            vocabulary.version,
        )

    fallback = _read_envelope(default, vocabulary)
    if fallback is None:
        logger.warning(
            "the configured default bar layout is not a readable version-%d document; "
            "rendering no items.",
            vocabulary.version,
        )
        return _empty_layout(vocabulary)
    fallback["rev"] = 0
    return fallback


def save_layout(
    store_dir: Path,
    layout: Mapping[str, Any],
    *,
    vocabulary: BarVocabulary,
    expected_rev: int | None = None,
) -> dict[str, Any]:
    """Validate *layout*, then write it at the next revision.

    The document is validated whole before anything touches the disk, so a
    refusal leaves whatever was stored exactly as it was. Options are completed
    against their specs on the way through: every option the type declares is
    written out at the stored or default value, and a key the type does not
    declare is dropped the way the browser drops it. A saved document therefore
    normalizes clean in the browser instead of arriving as an edit.

    Args:
        store_dir: The directory to write into; created if absent.
        layout: The document to persist. Its ``rev`` is ignored — the store
            assigns revisions.
        vocabulary: The item types, schema version and per-host cap to validate
            against.
        expected_rev: When given, the revision the caller believed it was
            editing. A mismatch raises rather than overwriting a save the
            caller never saw.

    Returns:
        The document as persisted, with its assigned ``rev``.

    Raises:
        BarLayoutInvalid: The document is not one this build can store.
        BarLayoutConflict: *expected_rev* does not match what is on disk.
        OSError: The store directory cannot be created or written.
    """
    document = _validate(layout, vocabulary)

    stored = load_layout(store_dir, vocabulary=vocabulary, default=_empty_layout(vocabulary))
    current_rev = int(stored["rev"])
    if expected_rev is not None and expected_rev != current_rev:
        raise BarLayoutConflict(expected_rev, current_rev)

    document["rev"] = current_rev + 1
    store_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(layout_path(store_dir), document)
    return _copy_layout(document)


def reset_layout(store_dir: Path) -> bool:
    """Delete this operator's layout, returning them to the deployment default.

    A delete rather than a write, because the deployment default is not a
    document a client can hold: an operator may have configured an arrangement
    none of the stock presets describes, and it may change under the user
    between one boot and the next. Removing the document is the only way to say
    "whatever this deployment renders".

    Args:
        store_dir: The directory holding the layout document.

    Returns:
        Whether a document was there to delete. A corrupt document counts — it
        is deleted like any other, which is what makes reset the way out of one.

    Raises:
        OSError: The document exists but cannot be removed.
    """
    path = layout_path(store_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


# ── envelope ───────────────────────────────────────────────────────────────


def _empty_layout(vocabulary: BarVocabulary) -> dict[str, Any]:
    """A valid, empty document: the last-resort fallback that renders nothing."""
    layout: dict[str, Any] = {"version": vocabulary.version, "rev": 0}
    for host in vocabulary.hosts:
        layout[host] = []
        layout[f"{host}_visible"] = True
    return layout


def _copy_layout(document: Mapping[str, Any]) -> dict[str, Any]:
    """A deep-enough copy that a caller mutating the result cannot reach the store."""
    copied: dict[str, Any] = dict(document)
    for key, value in document.items():
        if isinstance(value, list):
            copied[key] = [dict(entry) if isinstance(entry, Mapping) else entry for entry in value]
    return copied


def _read_envelope(raw: Any, vocabulary: BarVocabulary) -> dict[str, Any] | None:
    """A document's structure as this build reads it, or ``None`` if it cannot.

    Checks the schema version, the revision, both item lists and the two
    visibility flags; each entry only has to *look* like an item. Item types are the
    browser's to judge — see :func:`load_layout`.
    """
    if not isinstance(raw, Mapping):
        return None
    if raw.get("version") != vocabulary.version:
        return None

    rev = raw.get("rev", 0)
    # ``bool`` is an ``int`` in Python and ``True`` is not a revision.
    if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
        return None

    document: dict[str, Any] = {"version": vocabulary.version, "rev": rev}
    for host in vocabulary.hosts:
        entries = raw.get(host)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            return None
        items: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("type"), str):
                return None
            items.append(dict(entry))
        document[host] = items
        visible = raw.get(f"{host}_visible", True)
        if not isinstance(visible, bool):
            return None
        document[f"{host}_visible"] = visible
    return document


# ── validation ─────────────────────────────────────────────────────────────


def _validate(layout: Mapping[str, Any], vocabulary: BarVocabulary) -> dict[str, Any]:
    """The document as it will be stored, or raise :class:`BarLayoutInvalid`."""
    if not isinstance(layout, Mapping):
        raise BarLayoutInvalid("malformed", "a bar layout must be an object")
    if layout.get("version") != vocabulary.version:
        raise BarLayoutInvalid(
            "version",
            f"bar layout version {layout.get('version')!r} is not {vocabulary.version}",
        )

    document: dict[str, Any] = {"version": vocabulary.version, "rev": 0}
    # Single-node types are counted across the whole document, header first,
    # exactly as the browser's normalizer counts them.
    placed: set[str] = set()
    for host in vocabulary.hosts:
        entries = layout.get(host)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise BarLayoutInvalid("malformed", f"{host} must be a list of items")
        if len(entries) > vocabulary.max_items_per_host:
            raise BarLayoutInvalid(
                "overflow",
                f"{host} holds {len(entries)} items; the most one bar may hold is "
                f"{vocabulary.max_items_per_host}",
            )
        document[host] = [
            _validate_item(entry, host, index, vocabulary, placed)
            for index, entry in enumerate(entries)
        ]
        visible = layout.get(f"{host}_visible", True)
        if not isinstance(visible, bool):
            raise BarLayoutInvalid(
                "malformed", f"{host}_visible must be true or false, not {visible!r}"
            )
        document[f"{host}_visible"] = visible
    return document


def _validate_item(
    raw: Any, host: str, index: int, vocabulary: BarVocabulary, placed: set[str]
) -> dict[str, Any]:
    """One placed item, with its options completed, or raise.

    The order of the checks matters and matches the browser's: a type has to be
    known before its options mean anything. *placed* is the single-node types
    already seen in this document; a type is added to it here.
    """
    where = f"{host}[{index}]"
    if not isinstance(raw, Mapping) or not isinstance(raw.get("type"), str):
        raise BarLayoutInvalid("malformed", f"{where} is not an item with a type")
    item_type: str = raw["type"]

    spec = vocabulary.items.get(item_type)
    if spec is None:
        raise BarLayoutInvalid("unknown-type", f"{where} names unknown bar item {item_type!r}")

    if not spec.get("multi", False):
        if item_type in placed:
            raise BarLayoutInvalid(
                "duplicate", f"{where} places {item_type!r} a second time; it renders once"
            )
        placed.add(item_type)

    raw_options = raw.get("options", {})
    if raw_options is None:
        raw_options = {}
    if not isinstance(raw_options, Mapping):
        raise BarLayoutInvalid("malformed", f"{where} has options that are not an object")

    return {"type": item_type, "options": _validate_options(raw_options, spec, where)}


def _validate_options(raw: Mapping[str, Any], spec: BarItemSpec, where: str) -> dict[str, Any]:
    """Every option the type declares, at a value its spec allows.

    A missing option is completed with its default, which loses nothing and is
    how an option added in a later build reaches an older document. A value the
    spec does not allow is refused rather than repaired: the browser clamps or
    resets one and marks the layout read-only, so a value arriving here out of
    spec did not come from a client following the contract. A key the type does
    not declare is dropped, exactly as the browser drops it — refusing it would
    leave an older tab permanently unable to save.
    """
    options: dict[str, Any] = {}
    specs = spec.get("options", {})
    for name, option_spec in specs.items():
        if name not in raw:
            options[name] = option_spec.get("default")
            continue
        options[name] = _validate_option(raw[name], option_spec, f"{where}.{name}")
    return options


def _validate_option(value: Any, spec: BarOptionSpec, where: str) -> Any:
    """One option value as its spec allows it, or raise."""
    kind = spec.get("kind")
    if kind == "boolean":
        if not isinstance(value, bool):
            raise BarLayoutInvalid("bad-option", f"{where} must be true or false, not {value!r}")
        return value
    if kind == "number":
        # ``bool`` is an ``int``; a checkbox value is not a size.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BarLayoutInvalid("bad-option", f"{where} must be a number, not {value!r}")
        if not math.isfinite(value):
            raise BarLayoutInvalid("bad-option", f"{where} must be a finite number")
        low, high = spec.get("min"), spec.get("max")
        if (low is not None and value < low) or (high is not None and value > high):
            raise BarLayoutInvalid("bad-option", f"{where} must be between {low} and {high}")
        return value
    if kind == "enum":
        values = spec.get("values", ())
        if not isinstance(value, str) or value not in values:
            raise BarLayoutInvalid(
                "bad-option", f"{where} must be one of {', '.join(values)}, not {value!r}"
            )
        return value
    raise BarLayoutInvalid("bad-option", f"{where} has no readable option spec")
