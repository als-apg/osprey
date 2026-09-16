"""The family view: the one place the MML family grain is computed.

A normalised family body is read by the census, the direction voter, the
mapping skeleton and every emitter. They must agree on where the per-device
arrays live, how many devices the family has, which sub-dicts are fields, how
a 1-row channel list broadcasts across devices, and how many bindings the
family yields. ``FamilyView`` computes each of those once, from the body alone.

Rules:

* The family arrays are read per key from the family level, else from the
  ``setup`` sub-dict, else from ``_setup``; the family level wins when a key is
  present at both levels.
* ``device_rows`` is the ``DeviceList`` as Nx2 rows, a flat ``[sector,
  device]`` pair of numbers counting as one row; ``n_devices`` is its row
  count, else the longest channel list across fields.
* A field is a sub-dict of the family carrying a channel key; ``setup``,
  ``_setup`` and ``_``-prefixed keys are never fields.
* A 1-row channel list on a family of more than one device broadcasts to
  ``n_devices`` slots. 0-length and partial lists are kept as they are.
* In an ``ao`` document, a system or family is an entry whose key is a string
  not starting with ``_`` and whose value is a dict; every other entry
  (``_exports``, ``_import_order``, ``_description``) is bookkeeping.
  :func:`system_bodies` and :func:`family_views` are the one spelling of that
  rule.

The module is pure and depends on the standard library and ``CHANNEL_KEYS``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from osprey.services.channel_finder.databases.middle_layer import CHANNEL_KEYS

__all__ = ["FAMILY_ARRAYS", "FamilyView", "FieldView", "family_views", "system_bodies"]

#: Per-device family arrays, read from the family level or its setup block.
FAMILY_ARRAYS: tuple[str, ...] = (
    "DeviceList",
    "CommonNames",
    "ElementList",
    "Position",
    "Status",
    "DeviceType",
    "MemberOf",
)

#: Setup sub-dicts that carry the family arrays, in the order consulted.
_SETUP_KEYS: tuple[str, ...] = ("setup", "_setup")

#: Provenance of a description carried over from the export.
_IMPORTED = "imported"


def _text(value: Any) -> str | None:
    """Return ``value`` when it is a non-blank string, else ``None``."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _description(body: dict) -> str | None:
    """Return ``_description``, else ``Description``, when non-blank."""
    return _text(body.get("_description")) or _text(body.get("Description"))


def _as_slots(value: Any) -> list:
    """Return a channel-key value as a list of slots."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _is_binding(slot: Any) -> bool:
    """A slot binds when it is a non-blank string."""
    return isinstance(slot, str) and bool(slot.strip())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _device_rows(device_list: Any) -> list[list] | None:
    """Return ``DeviceList`` as Nx2 rows, or ``None`` when it states no rows."""
    if not isinstance(device_list, (list, tuple)) or not device_list:
        return None
    if all(isinstance(row, (list, tuple)) and len(row) == 2 for row in device_list):
        return [list(row) for row in device_list]
    if len(device_list) == 2 and all(_is_number(item) for item in device_list):
        return [list(device_list)]
    return None


class FieldView:
    """One channel-bearing field of a family.

    Attributes:
        name: The field's key in the family body.
        body: The field dict, as given.
        keys: The channel keys the field carries, in ``CHANNEL_KEYS`` order.
        broadcast: Whether any key's list is a 1-row list broadcast to
            ``n_devices``.
        description: ``_description`` or ``Description`` when non-blank.
        raw_slot_count: Slots across the field's keys, as exported.
        channel_count: Non-blank slots after broadcast expansion.
    """

    def __init__(self, name: str, body: dict, n_devices: int) -> None:
        self.name = name
        self.body = body
        self.n_devices = n_devices
        self.keys: tuple[str, ...] = tuple(key for key in CHANNEL_KEYS if key in body)
        self.description: str | None = _description(body)
        self.broadcast: bool = any(self._broadcasts(key) for key in self.keys)
        self.raw_slot_count: int = sum(len(self.raw_slots(key)) for key in self.keys)
        self.channel_count: int = sum(
            1 for key in self.keys for slot in self.slots(key) if _is_binding(slot)
        )

    def _broadcasts(self, key: str) -> bool:
        return self.n_devices > 1 and len(self.raw_slots(key)) == 1

    def raw_slots(self, key: str) -> list[str | None]:
        """Return the slots under ``key`` exactly as exported.

        Raises:
            KeyError: The field does not carry ``key``.
        """
        if key not in self.keys:
            raise KeyError(f"field {self.name!r} carries no {key!r}")
        return _as_slots(self.body[key])

    def slots(self, key: str) -> list[str | None]:
        """Return the slots under ``key``, a 1-row list broadcast to ``n_devices``.

        Raises:
            KeyError: The field does not carry ``key``.
        """
        raw = self.raw_slots(key)
        if self._broadcasts(key):
            return raw * self.n_devices
        return raw

    def __repr__(self) -> str:
        return f"FieldView({self.name!r}, keys={self.keys!r})"


class FamilyView:
    """The computed grain of one family body in one system.

    Attributes:
        system: The system (sub-machine) token the family sits under.
        raw_name: The family's key as exported.
        body: The normalised family body, as given; never modified.
        arrays: The family arrays found, keyed by array name.
        arrays_source: ``"setup"`` when any array came from ``setup`` or
            ``_setup``, else ``"family"``.
        device_rows: ``DeviceList`` as Nx2 rows, a flat pair as one row, or
            ``None`` when it states no rows. Rows are fresh lists.
        n_devices: The family's device count.
        n_devices_from_fallback: Whether ``n_devices`` came from the longest
            channel list rather than ``device_rows``.
        fields: The channel-bearing fields, in body order.
        raw_slot_count: Channel slots across all fields, as exported.
        channel_count: Non-blank channel slots after broadcast expansion; the
            binding count.
        description: ``(text, "imported")`` from ``_description`` or
            ``Description``, or ``None``.
        disabled_devices: 0-based positions where an aligned ``Status`` is 0.
    """

    def __init__(self, system: str, raw_name: str, body: dict) -> None:
        self.system = system
        self.raw_name = raw_name
        self.body = body
        self.arrays, self.arrays_source = self._read_arrays(body)

        field_bodies = {
            name: value
            for name, value in body.items()
            if isinstance(name, str)
            and not name.startswith("_")
            and name not in _SETUP_KEYS
            and isinstance(value, dict)
            and any(key in value for key in CHANNEL_KEYS)
        }

        self.device_rows: list[list] | None = _device_rows(self.arrays.get("DeviceList"))
        self.n_devices_from_fallback: bool = self.device_rows is None
        if self.device_rows is not None:
            self.n_devices: int = len(self.device_rows)
        else:
            self.n_devices = max(
                (
                    len(_as_slots(field[key]))
                    for field in field_bodies.values()
                    for key in CHANNEL_KEYS
                    if key in field
                ),
                default=0,
            )

        self.fields: dict[str, FieldView] = {
            name: FieldView(name, value, self.n_devices) for name, value in field_bodies.items()
        }
        self.raw_slot_count: int = sum(field.raw_slot_count for field in self.fields.values())
        self.channel_count: int = sum(field.channel_count for field in self.fields.values())

        text = _description(body)
        self.description: tuple[str, str] | None = None if text is None else (text, _IMPORTED)

        status = self.aligned("Status")
        self.disabled_devices: tuple[int, ...] = (
            ()
            if status is None
            else tuple(i for i, slot in enumerate(status) if _is_number(slot) and slot == 0)
        )

    @staticmethod
    def _read_arrays(body: dict) -> tuple[dict[str, Any], Literal["family", "setup"]]:
        setups = [body[key] for key in _SETUP_KEYS if isinstance(body.get(key), dict)]
        arrays: dict[str, Any] = {}
        source: Literal["family", "setup"] = "family"
        for name in FAMILY_ARRAYS:
            if name in body:
                arrays[name] = body[name]
                continue
            for setup in setups:
                if name in setup:
                    arrays[name] = setup[name]
                    source = "setup"
                    break
        return arrays, source

    def aligned(self, name: str) -> list | None:
        """Return the family array ``name`` when it has one slot per device.

        A list of exactly ``n_devices`` entries is aligned; a non-list value on
        a one-device family is aligned as a one-slot list.

        Returns:
            The aligned slots, or ``None`` when the array is absent or does not
            align.
        """
        if name not in self.arrays:
            return None
        value = self.arrays[name]
        if isinstance(value, (list, tuple)):
            return list(value) if len(value) == self.n_devices else None
        if self.n_devices == 1 and value is not None:
            return [value]
        return None

    def __repr__(self) -> str:
        return (
            f"FamilyView({self.system!r}, {self.raw_name!r}, "
            f"n_devices={self.n_devices}, fields={list(self.fields)!r})"
        )


def _is_entry(key: Any, value: Any) -> bool:
    """Whether an ``ao`` entry is a system or family rather than bookkeeping."""
    return isinstance(key, str) and not key.startswith("_") and isinstance(value, dict)


def system_bodies(ao: dict) -> Iterator[tuple[str, dict]]:
    """Yield ``(raw system token, system body)`` for every system of ``ao``.

    Args:
        ao: The merged export, keyed by raw system token plus bookkeeping keys.

    Yields:
        The systems in ``ao`` key order; bookkeeping entries are skipped.
    """
    for key, value in ao.items():
        if _is_entry(key, value):
            yield key, value


def family_views(system: str, body: dict) -> Iterator[FamilyView]:
    """Yield one :class:`FamilyView` per family of one ``ao`` system.

    Args:
        system: The raw system token the body sits under.
        body: The system body, keyed by raw family token plus bookkeeping keys.

    Yields:
        The families in ``body`` key order; bookkeeping entries are skipped.
    """
    for raw_name, family_body in body.items():
        if _is_entry(raw_name, family_body):
            yield FamilyView(system, raw_name, family_body)
