"""The LUME variable catalog for the pyat-coupled channel partition.

Every variable is keyed and named by its channel address exactly as the
manifest carries it, so there is no address translation anywhere between the
manifest, the IOC and the model -- whatever a facility's addresses look like
(the bundled demo tree spells them as six colon-separated tokens,
``{ring}:{system}:{family}:{device}:{field}:{subfield}``). The catalog is
derived -- from the manifest, ``machine.json`` and ``channel_limits.json`` --
never hand-listed: the databases fix the model, not vice versa.

All three sources come from ONE served tree: the caller hands in the manifest
it resolved and a :class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`
over the data directory the service was given, and nothing here falls back to
a packaged tree. A fallback would serve the framework's own demo nominals and
bands behind a facility's addresses -- and the band a variable is weighed
against is the one thing a facility must recognise as its own.

The setpoint echo is deliberately excluded, selected by the manifest's
reserved ``READBACK_SUBFIELD`` rather than by anything in the address text. It
is a serving-layer concern (the IOC mirrors each setpoint write back onto its
readback record), not model state, so it is not a model variable.

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
corresponding ``machine.json`` nominal hard-fails catalog construction --
on the served files, which is what makes the refusal a statement about the
tree this process was handed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    READBACK_SUBFIELD,
    SETPOINT_SUBFIELD,
    setpoint_addresses,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps `lume` out of import time
    from collections.abc import Callable, Container, Mapping
    from pathlib import Path

    from lume.variables import ScalarVariable

    from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

    VariableFactory = Callable[..., ScalarVariable]


def _load_limit_bands(path: Path, *, setpoints: Container[str]) -> dict[str, tuple[float, float]]:
    """Return one ``(min_value, max_value)`` band per writable setpoint
    address with numeric bounds, merging the file's ``defaults`` block under
    every entry. Mirrors the read ``entrypoint.py`` performs for the IOC's
    drive limits -- the same file, read independently so the model layer
    stays free of the IOC module.

    ``setpoints`` is the manifest's own setpoint set -- the channels
    whose ``subfield`` says they are written. The limits file holds an entry
    per address, read-only ones included, so the setpoint half has to be
    named from outside it; asking the address text instead would silently
    drop every facility whose setpoints are not spelled ``...:SP``.

    An entry key this reader does not know is ignored rather than refused:
    the same file is the control assistant's write-safety database and
    carries per-entry bookkeeping of its own, which says nothing about a
    band.
    """
    raw = json.loads(path.read_text())
    defaults = raw.get("defaults", {})
    bands: dict[str, tuple[float, float]] = {}
    for address, entry in raw.items():
        if address.startswith("_") or address == "defaults":
            continue
        if address not in setpoints:
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
    paths: ManifestPaths,
    manifest: list[dict],
    action_variables: Mapping[str, VariableFactory],
) -> dict[str, ScalarVariable]:
    """Build the model's variable catalog, keyed by full channel address.

    Inputs are the pyat-coupled setpoints; outputs are the readings the
    model solves for. Nominals and units come from the served tree's
    ``machine.json``, input ranges from its ``channel_limits.json``.

    Args:
        paths: the served data tree. ``machine.json`` and
            ``channel_limits.json`` are read through it, so the model is
            built from the directory the service was given and from no
            other.
        manifest: the served manifest's ``channels`` list, exactly as
            ``loaders.load_manifest_file`` returns it. Passed in rather than
            generated here: the channel set a facility serves is the one its
            deployment resolved, and generating a second one would put this
            model on a different namespace than the IOC beside it.
        action_variables: one factory per channel address, each called as
            ``factory(channel, **scalar_kwargs)`` with the whole manifest
            channel dict and the ``ScalarVariable`` fields derived here. A
            caller binding variables to lattice elements passes the
            factories its bindings document yields -- keeping this the
            single derivation path for the catalog's fields, so a bound
            catalog differs from a declared one only in the binding. An
            address the mapping does not name becomes a plain
            ``ScalarVariable``; an empty mapping therefore builds the
            declared catalog, bound to nothing.

    Raises:
        FileNotFoundError: if the served tree carries no ``machine.json`` or
            no ``channel_limits.json``.
        pydantic.ValidationError: if a ``machine.json`` nominal falls
            outside its ``channel_limits.json`` band (a ``ValueError``; see
            the module docstring).
    """
    machine_channels = load_machine_json_channels(paths.machine_json)
    bands = _load_limit_bands(paths.channel_limits, setpoints=setpoint_addresses(manifest))

    catalog: dict[str, ScalarVariable] = {}
    for channel in manifest:
        if channel["partition"] != PARTITION_PYAT_COUPLED:
            continue
        subfield = channel["subfield"]
        if subfield == READBACK_SUBFIELD:
            continue
        address = channel["address"]
        entry = machine_channels.get(address, {})
        read_only = subfield != SETPOINT_SUBFIELD
        factory = action_variables.get(address, _plain_scalar_variable)
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
