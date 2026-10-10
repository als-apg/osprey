"""The one mapping between OSPREY's addresses and ids and the names a pyAML view uses.

pyAML and pySC address everything by NAME: a magnet, a BPM, an array, the RF plant,
the tune monitor. OSPREY's tools take channel addresses and facility ids. Every name a
pyAML view holds is derived here, and every name is turned back into its address or
id here, so the view writer, the measurement tools and a reader of a pyAML
configuration cannot spell a name two ways.

* A magnet is named after its setpoint address, a BPM after its device id and an
  array after its group id, each through :func:`pyaml_name`.
* The RF plant is :data:`RF_PLANT_NAME`, pyAML's default plant, and stands for the
  one ``instruments.rf`` address of the view.
* The tune monitor is :data:`TUNE_MONITOR_NAME`, pyAML's default betatron tune
  monitor.

A :class:`ViewNames` holds the names of one view. It refuses to be built when two
addresses or ids would share a name, and every lookup of an address or id it does
not hold raises :class:`UnmappedName` naming it. This module imports nothing from
pyAML.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = [
    "CHROMATICITY_MONITOR_NAME",
    "NAME_RE",
    "RF_PLANT_NAME",
    "TUNE_MONITOR_NAME",
    "UnmappedName",
    "ViewNames",
    "pyaml_name",
]

#: A name pyAML can address an element by; matched with ``fullmatch``.
NAME_RE = re.compile(r"[A-Za-z0-9_.:\-]+")

#: The run of characters :func:`pyaml_name` replaces with one underscore.
_OUTSIDE_NAME = re.compile(r"[^A-Za-z0-9_.:\-]+")

#: pyAML's default RF plant, the one ``sr.live.rf`` reads.
RF_PLANT_NAME = "DEFAULT_RF_PLANT"

#: pyAML's default betatron tune monitor.
TUNE_MONITOR_NAME = "BETATRON_TUNE"

#: The chromaticity monitor the chromaticity tools read.
CHROMATICITY_MONITOR_NAME = "CHROMATICITY_MONITOR"


class UnmappedName(LookupError):
    """An address, id or name the pyAML view has no counterpart for.

    Attributes:
        key: The address, id or name looked up.
    """

    def __init__(self, what: str, key: str) -> None:
        self.key = key
        super().__init__(f"{key} is no {what} of the pyAML view")


def pyaml_name(identifier: str) -> str:
    """The pyAML name of an address or id.

    Each run of characters outside ``[A-Za-z0-9_.:-]`` becomes one ``_``, so
    ``SR/BPM`` is ``SR_BPM`` and ``SR:C02-MG{PS:QH1A}I:Sp1-SP`` is
    ``SR:C02-MG_PS:QH1A_I:Sp1-SP``. An identifier that is already a name is
    its own name.

    Args:
        identifier: The address or id.

    Returns:
        A name :data:`NAME_RE` matches.

    Raises:
        ValueError: ``identifier`` is empty or holds no name character.
    """
    name = _OUTSIDE_NAME.sub("_", identifier)
    if not identifier or not name.strip("_"):
        raise ValueError(f"{identifier!r} holds no character a pyAML name can carry")
    return name


def _table(kind: str, keys: Iterable[str]) -> dict[str, str]:
    """Each key to its :func:`pyaml_name`, refusing two keys sharing one name."""
    table: dict[str, str] = {}
    holders: dict[str, str] = {}
    for key in keys:
        if key in table:
            continue
        name = pyaml_name(key)
        other = holders.get(name)
        if other is not None:
            raise ValueError(f"{kind}s {other} and {key} would both be named {name} in pyAML")
        table[key] = name
        holders[name] = key
    return table


def _inverse(table: Mapping[str, str]) -> dict[str, str]:
    return {name: key for key, name in table.items()}


@dataclass(frozen=True)
class ViewNames:
    """The names of one pyAML view and the addresses and ids they stand for.

    Build one with :meth:`build`; the tables are keyed by address or id, each
    value the name pyAML knows it by.

    Attributes:
        magnets: Setpoint address to magnet name.
        bpms: BPM device id to BPM name.
        arrays: Group id to array name.
        rf: The ``instruments.rf`` address, or ``None`` when the view has no RF
            plant.
    """

    magnets: Mapping[str, str] = field(default_factory=dict)
    bpms: Mapping[str, str] = field(default_factory=dict)
    arrays: Mapping[str, str] = field(default_factory=dict)
    rf: str | None = None

    @classmethod
    def build(
        cls,
        *,
        magnets: Iterable[str] = (),
        bpms: Iterable[str] = (),
        groups: Iterable[str] = (),
        rf: str | None = None,
    ) -> ViewNames:
        """Name every magnet, BPM and array of one view.

        Args:
            magnets: Each magnet's setpoint address.
            bpms: Each BPM's device id.
            groups: Each array's group id.
            rf: The ``instruments.rf`` address, or ``None``.

        Returns:
            The view's names.

        Raises:
            ValueError: Two addresses or ids of one table, or a magnet and a BPM,
                would share a name, or a name is one of the fixed names; the
                message names both.
        """
        magnet_table = _table("magnet setpoint", magnets)
        bpm_table = _table("BPM device", bpms)
        array_table = _table("group", groups)
        fixed = {RF_PLANT_NAME, TUNE_MONITOR_NAME, CHROMATICITY_MONITOR_NAME}
        devices = {name: address for address, name in magnet_table.items()}
        for device, name in bpm_table.items():
            if name in devices:
                raise ValueError(
                    f"magnet setpoint {devices[name]} and BPM device {device} would both be "
                    f"named {name} in pyAML"
                )
            devices[name] = device
        for name in sorted(fixed & (set(devices) | set(array_table.values()))):
            held = devices.get(name) or _inverse(array_table)[name]
            raise ValueError(f"{held} would be named {name}, which pyAML reserves")
        return cls(magnets=magnet_table, bpms=bpm_table, arrays=array_table, rf=rf)

    # --- address or id to name --------------------------------------------------

    def magnet_name(self, address: str) -> str:
        """The magnet a setpoint address drives.

        Raises:
            UnmappedName: No magnet of the view is on ``address``.
        """
        try:
            return self.magnets[address]
        except KeyError:
            raise UnmappedName("magnet setpoint", address) from None

    def bpm_name(self, device: str) -> str:
        """The BPM a device id is.

        Raises:
            UnmappedName: No BPM of the view is that device.
        """
        try:
            return self.bpms[device]
        except KeyError:
            raise UnmappedName("BPM device", device) from None

    def array_name(self, group: str) -> str:
        """The array a group id is.

        Raises:
            UnmappedName: No array of the view is that group.
        """
        try:
            return self.arrays[group]
        except KeyError:
            raise UnmappedName("group", group) from None

    def rf_plant_name(self, address: str) -> str:
        """The RF plant the ``instruments.rf`` address drives.

        Raises:
            UnmappedName: ``address`` is not the view's RF address.
        """
        if self.rf is None or address != self.rf:
            raise UnmappedName("RF address", address)
        return RF_PLANT_NAME

    # --- name to address or id --------------------------------------------------

    def magnet_address(self, name: str) -> str:
        """The setpoint address a magnet name stands for.

        Raises:
            UnmappedName: No magnet of the view has that name.
        """
        try:
            return _inverse(self.magnets)[name]
        except KeyError:
            raise UnmappedName("magnet name", name) from None

    def bpm_device(self, name: str) -> str:
        """The device id a BPM name stands for.

        Raises:
            UnmappedName: No BPM of the view has that name.
        """
        try:
            return _inverse(self.bpms)[name]
        except KeyError:
            raise UnmappedName("BPM name", name) from None

    def array_group(self, name: str) -> str:
        """The group id an array name stands for.

        Raises:
            UnmappedName: No array of the view has that name.
        """
        try:
            return _inverse(self.arrays)[name]
        except KeyError:
            raise UnmappedName("array name", name) from None

    def rf_address(self, name: str) -> str:
        """The ``instruments.rf`` address the RF plant name stands for.

        Raises:
            UnmappedName: ``name`` is not the RF plant's, or the view has none.
        """
        if self.rf is None or name != RF_PLANT_NAME:
            raise UnmappedName("RF plant name", name)
        return self.rf
