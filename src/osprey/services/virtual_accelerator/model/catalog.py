"""The LUME variable catalog for the pyat-coupled channel partition.

Every variable is keyed and named by its full six-level channel address
(``{ring}:{system}:{family}:{device}:{field}:{subfield}``) so there is no
address translation anywhere between the manifest, the IOC and the model.
The catalog is derived -- from the manifest, ``machine.json`` and
``channel_limits.json`` -- never hand-listed: the databases fix the model,
not vice versa.

The ``:RB`` setpoint echo is deliberately excluded. It is a serving-layer
concern (the IOC mirrors each ``:SP`` write back onto its ``:RB`` record),
not model state, so it is not a model variable.

**Declared ranges are metadata, not enforcement.** Inputs carry
``default_validation_config='none'``, so ``LUMEModel.set()`` neither
rejects nor clamps an out-of-band value -- and nothing in lume clamps
anywhere. A standalone consumer of this catalog must not read
``value_range`` as a guarantee. Real enforcement lives in three places:
DRVL/DRVH clamping on the EPICS record before the write hook runs,
``channel_limits.json`` at the control-assistant layer, and the
fail-closed orbit solve that rolls the ring back when a setpoint destroys
the closed orbit.

**Construction-time range validation, by contrast, is always on.**
``ScalarVariable`` validates ``default_value`` against ``value_range``
whenever both are given, raising ``pydantic.ValidationError`` (a
``ValueError``). So a ``channel_limits.json`` band that ever excludes the
corresponding ``machine.json`` nominal hard-fails catalog construction.
That coupling is accepted; the nominal-in-band unit test is the guard that
catches a bad band regeneration in CI rather than at VA boot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    build_manifest,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps `lume` out of import time
    from collections.abc import Callable

    from lume.variables import ScalarVariable

    VariableFactory = Callable[..., ScalarVariable]

_SETPOINT_SUBFIELD = "SP"


def _channel_limits_path() -> Path:
    """Locate ``channel_limits.json`` in the installed ``osprey.templates``
    package, the same convention ``manifest/paths.py`` uses for the other
    control-assistant data files."""
    import osprey.templates

    return (
        Path(osprey.templates.__file__).parent
        / "apps"
        / "control_assistant"
        / "data"
        / "channel_limits.json"
    )


def _load_limit_bands(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """Return one ``(min_value, max_value)`` band per writable ``:SP``
    address with numeric bounds, merging the file's ``defaults`` block under
    every entry. Mirrors the read ``entrypoint.py`` performs for the IOC's
    drive limits -- the same file, read independently so the model layer
    stays free of the IOC module."""
    raw = json.loads((path or _channel_limits_path()).read_text())
    defaults = raw.get("defaults", {})
    bands: dict[str, tuple[float, float]] = {}
    for address, entry in raw.items():
        if address.startswith("_") or address == "defaults" or not address.endswith(":SP"):
            continue
        merged = {**defaults, **entry}
        if not merged.get("writable", True):
            continue
        min_value = merged.get("min_value")
        max_value = merged.get("max_value")
        if min_value is None or max_value is None:
            continue
        bands[address] = (float(min_value), float(max_value))
    return bands


def _plain_scalar_variable(channel: dict, **scalar_kwargs) -> ScalarVariable:
    """The default variable factory: a plain ``ScalarVariable``, ignoring the
    channel it was derived from."""
    from lume.variables import ScalarVariable

    return ScalarVariable(**scalar_kwargs)


def build_variable_catalog(
    channels: list[dict] | None = None,
    *,
    input_factory: VariableFactory = _plain_scalar_variable,
    output_factory: VariableFactory = _plain_scalar_variable,
) -> dict[str, ScalarVariable]:
    """Build the model's variable catalog, keyed by full channel address.

    Inputs are the pyat-coupled magnet setpoints (``:SP``); outputs are the
    BPM position readings. Nominals and units come from ``machine.json``,
    input ranges from ``channel_limits.json``.

    Args:
        channels: the manifest's ``channels`` list. ``None`` (the default)
            builds the manifest here. A caller that already has one should
            pass it: ``build_manifest()`` costs ~70 ms, and a model that
            builds the manifest for its own device lookup would otherwise
            pay for it twice per construction -- which the 216-construction
            band test in ``tests/templates/test_channel_limits_va.py`` bills
            216 times over.
        input_factory: builds each writable ``:SP`` variable from
            ``(channel, **scalar_kwargs)``, where ``channel`` is the whole
            manifest channel dict and ``scalar_kwargs`` are the
            ``ScalarVariable`` fields derived here. Defaults to a plain
            ``ScalarVariable``; a caller binding variables to lattice
            elements passes a wrapper that derives the element, attribute
            and axis from ``channel`` -- keeping this the single derivation
            path for the catalog's fields.
        output_factory: the same, for the read-only BPM readings.

    Raises:
        pydantic.ValidationError: if a ``machine.json`` nominal falls
            outside its ``channel_limits.json`` band (a ``ValueError``; see
            the module docstring).
    """
    machine_channels = load_machine_json_channels()
    bands = _load_limit_bands()
    if channels is None:
        channels = build_manifest()["channels"]

    catalog: dict[str, ScalarVariable] = {}
    for channel in channels:
        if channel["partition"] != PARTITION_PYAT_COUPLED:
            continue
        subfield = channel["subfield"]
        if subfield == "RB":
            continue
        address = channel["address"]
        entry = machine_channels.get(address, {})
        read_only = subfield != _SETPOINT_SUBFIELD
        factory = output_factory if read_only else input_factory
        catalog[address] = factory(
            channel,
            name=address,
            read_only=read_only,
            default_validation_config="none",
            default_value=entry.get("value"),
            value_range=None if read_only else bands.get(address),
            unit=entry.get("units"),
        )
    return catalog
