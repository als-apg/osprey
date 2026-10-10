"""The pyAML control system that routes every device through the OSPREY runtime.

A configuration selects it with ``type: pyaml_cs_osprey.controlsystem``; pyAML
resolves that module path to :class:`OspreyControlSystem` through
:data:`PYAMLCLASS`. The control system takes only a name: which channels exist,
their limits and the connector behind them all come from the OSPREY project, not
from the pyAML configuration.
"""

from __future__ import annotations

from pyaml.common.exception import PyAMLException
from pyaml.control.controlsystem import ControlSystem
from pyaml.control.deviceaccess import DeviceAccess
from pyaml.validation.registry import register_schema
from pyaml.validation.validation_models import DynamicValidation
from pydantic import BaseModel

from pyaml_cs_osprey.catalog import parse_reference
from pyaml_cs_osprey.device import OspreyDevice
from pyaml_cs_osprey.devices import OspreyDeviceList

__all__ = ["PYAMLCLASS", "OspreyControlSystem"]

PYAMLCLASS = "OspreyControlSystem"


@register_schema
class OspreyControlSystem(ControlSystem, DynamicValidation):
    """A pyAML control system whose devices read and write through ``osprey.runtime``.

    Args:
        name: The control-system name; ``live`` binds it as ``Accelerator.live``.
    """

    def __init__(self, name: str) -> None:
        ControlSystem.__init__(self)
        self._name = name
        self._devices: dict[str, OspreyDevice] = {}

    def name(self) -> str:
        """Return the configured control-system name."""
        return self._name

    def get_device_access(self, ref: str | BaseModel | None) -> DeviceAccess | None:  # type: ignore[override]
        """Return the device for a channel reference, one device per reference text.

        Args:
            ref: A ``pyaml-cs-oa`` channel reference, or ``None`` for an absent device.

        Returns:
            The cached device for ``ref``, or ``None`` when ``ref`` is ``None``.

        A bare indexed reference (``ADDR@i``) loads as a read-only device that reads
        element ``i`` of ``ADDR``; its ``set()`` raises.

        Raises:
            PyAMLException: ``ref`` is malformed, or is a parenthesised indexed form
                (``(ADDR)@i`` or ``(RB, SP)@i``); the message names the reference.
        """
        if ref is None:
            return None
        reference = parse_reference(ref)
        device = self._devices.get(reference.text)
        if device is not None:
            return device
        if reference.index is not None and (
            reference.mode != "rw" or reference.readback is not None
        ):
            raise PyAMLException(
                f"indexed channel reference {ref!r} is not supported by the OSPREY "
                "control system: only the bare read-only form ADDR@i is accepted"
            )
        device = OspreyDevice(reference)
        self._devices[reference.text] = device
        return device

    def get_aggregator(self) -> OspreyDeviceList:
        """Return a new, empty device list; every call yields a distinct one."""
        return OspreyDeviceList()
