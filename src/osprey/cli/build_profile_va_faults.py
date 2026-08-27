"""The ``virtual_accelerator.live_standin:`` refusals, and the fault-set grammar.

``live_standin: <port>`` stands a SECOND soft-IOC up and wires it in as the
deployment's ``live`` target, so an operator rehearses the whole go-live ritual
— the warnings, the approval prompts, the write refusals — against something
that cannot move a magnet. That makes it a port the deployment spends and a
target the deployment claims, and both can be claimed twice.

The rules live here rather than inside :meth:`BuildProfile.validate` for the
reason :func:`~osprey.cli.build_profile_archiver.va_archiver_errors` does: a
block's rules belong beside the block, but they are *reported* from validate's
single accumulator so a facility fixing a profile meets every problem it has in
one pass.

Two of them are refusals about going live rather than about ports:

* An ``epics`` baseline with a stand-in is refused outright. The stand-in IS the
  ``live`` target, so a deployment already pointed at the real machine has
  nothing left to stand in for, and the two would fight over one label.
* A stand-in on a build with no lattice behind it is refused, because the
  stand-in ships a deterministic readout perturbation and the IOC treats a
  perturbation without ``VA_LATTICE=builtin`` as fatal at boot
  (``services/virtual_accelerator/entrypoint.py``). Left alone that is a
  container in a crash loop, hours after the build reported success.

The perturbation grammar itself is parsed here too
(:func:`shipped_bpm_errors_field_errors`), mirroring the container-side splitting
without importing it: ``entrypoint.py`` runs inside the VA image and reads
``os.environ``, so it is not importable from a build.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.connectors.types import EPICS, VIRTUAL_ACCELERATOR

# The one nested-tree walker, borrowed rather than repeated for the same reason
# the path-tree builder below is.
from osprey.deployment.reach import dotted_get

# The one path-tree builder in this package, borrowed rather than repeated: a
# `config:` block addresses the same leaf through a dotted key or a nested
# mapping, and a second implementation of "which leaves does this reach" is a
# second answer free to disagree with the renderer's.
from .build_profile_archiver import _expand_dotted

#: Environment variable carrying the stand-in's shipped readout perturbation.
#: Named here so the build-time render check (which owns the default's value)
#: and this grammar check name the same variable.
STANDIN_BPM_ERRORS_ENV = "VA_STANDIN_BPM_ERRORS"

#: The only BPM fields the SHIPPED default is allowed to perturb.
#:
#: A stand-in exists so an operator can rehearse against a machine that reads
#: back plausibly wrong, and a static transverse offset is the one perturbation
#: that stays legible: the orbit is displaced, every downstream number follows,
#: and nothing about the readout chain is lying about its own gain. Gains,
#: polarities, roll and noise all change what a correction *means* rather than
#: what the machine is doing, which is a rehearsal that teaches the wrong
#: lesson. Facilities remain free to set ``VA_BPM_ERRORS`` themselves; this
#: bounds what OSPREY ships turned on.
STANDIN_BPM_ERROR_FIELDS = frozenset({"offset_x", "offset_y"})

#: Value ``VA_LATTICE`` must resolve to for the stand-in's perturbation to load.
#: Spelled rather than imported for the same reason ``build_cmd`` spells it when
#: it seeds the ``.env``: the constant lives in the container's entrypoint.
_LATTICE_BUILTIN = "builtin"

#: Key of the VA gateway table, in the nested spelling a rendered config reads.
#: Mirrors ``_VA_CONNECTOR_PATH`` in ``build_injectors``; the two ends of the
#: same block, one written by the injector and one refused here.
_VA_GATEWAYS_KEY = f"control_system.connector.{VIRTUAL_ACCELERATOR}.gateways"

#: Key of the deployment baseline's control-system type.
_CONTROL_SYSTEM_TYPE_KEY = "control_system.type"


def live_standin_errors(
    live_standin: int,
    va_port: int,
    claimed_ports: Mapping[str, int],
    config: Any,
    profile_dir: Path,
) -> list[str]:
    """Every reason a profile's ``live_standin`` port cannot be built.

    Args:
        live_standin: The port ``virtual_accelerator.live_standin`` names.
        va_port: The baseline soft-IOC's port, which the stand-in may not share.
        claimed_ports: Dotted key → port for every other port this profile
            spends, from :meth:`BuildProfile._claimed_ports`.
        config: The profile's resolved ``config:`` block.
        profile_dir: The profile root — also the deployment repo root, and so
            the directory whose env chain the containers are handed.

    Returns:
        The accumulated failures, empty when the stand-in validates.
    """
    errors: list[str] = []

    if not (1 <= live_standin <= 65535):
        errors.append(f"virtual_accelerator.live_standin must be in 1..65535 (got {live_standin})")
    elif live_standin == va_port:
        # The two soft-IOCs are two containers publishing on the host, so one
        # port cannot serve both — and the collision is worse than a refused
        # bind: the sandbox and the "live machine" would be the same endpoint.
        errors.append(
            f"virtual_accelerator.live_standin must differ from "
            f"virtual_accelerator.port (both {va_port})"
        )
    else:
        # Checked only for a usable port: an out-of-range value names no
        # endpoint, so "collides with" would be a second complaint about one
        # fault. Sorted so the report is stable whatever order the blocks
        # were read in.
        for key in sorted(claimed_ports):
            if claimed_ports[key] == live_standin:
                errors.append(
                    f"virtual_accelerator.live_standin ({live_standin}) "
                    f"collides with {key} ({claimed_ports[key]})"
                )
        errors.extend(_gateway_collision_errors(live_standin, config))

    errors.extend(_epics_baseline_errors(config))
    errors.extend(live_standin_lattice_errors(profile_dir))
    return errors


def _gateway_collision_errors(live_standin: int, config: Any) -> list[str]:
    """Refuse a hand-authored VA gateway sitting on the stand-in's port.

    The VA gateways are how a session dials the *simulation*; the stand-in is
    what ``live`` dials. A profile that points both at one endpoint has written
    a deployment where switching target changes the label and nothing else,
    which is the single thing the stand-in exists to make impossible.

    Read spelling-independently, because the renderer honors a dotted key and a
    nested mapping alike and either could be the one that lands.
    """
    node = dotted_get(_expand_dotted(config), _VA_GATEWAYS_KEY)
    if not isinstance(node, dict):
        return []

    errors: list[str] = []
    for role in sorted(node):
        row = node[role]
        if not isinstance(row, dict):
            continue
        port = row.get("port")
        if isinstance(port, int) and not isinstance(port, bool) and port == live_standin:
            dotted = f"{_VA_GATEWAYS_KEY}.{role}.port"
            errors.append(
                f"virtual_accelerator.live_standin ({live_standin}) collides with the "
                f"profile's `config:` {dotted} ({port}) — the virtual accelerator and "
                f"its live stand-in are two endpoints, never one"
            )
    return errors


def _epics_baseline_errors(config: Any) -> list[str]:
    """Refuse a stand-in on a deployment whose baseline is the real machine.

    The stand-in only means anything where ``live`` has nowhere else to point.
    An ``epics`` baseline has already named the machine, so the build would be
    overwriting a facility's real gateways with a loopback port — a deployment
    that says LIVE and dials a container.
    """
    if dotted_get(_expand_dotted(config), _CONTROL_SYSTEM_TYPE_KEY) != EPICS:
        return []
    return [
        f"control_system.type: {EPICS} with virtual_accelerator.live_standin — the "
        f"stand-in IS this deployment's live target, and a deployment already pointed "
        f"at the real machine has nothing to stand in for. Going live is three steps: "
        f"delete `virtual_accelerator.live_standin`, point "
        f"`control_system.connector.epics.gateways` at your facility, and replace "
        f"`control_system.target_switch.live_gateway_acknowledged` with your own live "
        f"gateway's hostname."
    ]


def live_standin_lattice_errors(project_root: Path, build_dir: Path | None = None) -> list[str]:
    """Reasons the stand-in would exit at boot for want of a lattice.

    The stand-in ships a readout perturbation, and the IOC refuses a
    perturbation it cannot apply: without ``VA_LATTICE=builtin`` there is no
    PyAT model to displace, and the entrypoint raises rather than serving a
    machine that ignores the faults it was configured with.

    Which half of that is knowable depends on where this is called from, so the
    caller says what it can see:

    * **From validation**, with no *build_dir*: the deployment's env chain only.
      A chain pinning ``VA_LATTICE`` to anything but ``builtin`` decides the
      question on its own — the build appends to that file and never overwrites
      it, so a value already on disk always wins.
    * **From the build**, once ``build/`` is the tree this render produced: also
      whether a channel manifest was generated. That is the same precondition
      :func:`~osprey.cli.build_cmd._wire_build_derived_env` gates its
      ``VA_LATTICE=builtin`` write on, so asking it here asks exactly what the
      build is about to answer. With no manifest nothing is written, and a
      chain that already names a facility ``VA_CHANNELS_FILE`` leaves the IOC
      on its file-backed default of ``none``.

    Args:
        project_root: The deployment repo root, whose env chain the containers
            are handed. Also the profile root at validation time.
        build_dir: The published output zone, when the caller has one.

    Returns:
        The accumulated failures, empty when the stand-in has a lattice.
    """
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
    from osprey.utils.dotenv import chain_files, parse_dotenv_file

    pinned: dict[str, tuple[Path, str]] = {}
    for path in chain_files(project_root):
        for key, value in parse_dotenv_file(path).items():
            if key in ("VA_LATTICE", "VA_CHANNELS_FILE"):
                # Later file wins, the same precedence merge_chain applies —
                # but the FILE is kept too, so the message names the line to go
                # and edit rather than a directory.
                pinned[key] = (path, value)

    lattice = pinned.get("VA_LATTICE")
    if lattice is not None and lattice[1].strip().lower() != _LATTICE_BUILTIN:
        return [
            f"virtual_accelerator.live_standin needs a lattice-backed virtual "
            f"accelerator, but {lattice[0]} pins VA_LATTICE={lattice[1]!r}. The build "
            f"appends to that file and never overwrites it, so the stand-in would boot "
            f"with no lattice behind the perturbation it ships and exit. Remove the "
            f"line, or delete `virtual_accelerator.live_standin`."
        ]

    channels = pinned.get("VA_CHANNELS_FILE")
    if lattice is None and channels is not None and build_dir is not None:
        manifest = build_dir / "data" / "simulation" / MANIFEST_FILENAME
        if not manifest.is_file():
            return [
                f"virtual_accelerator.live_standin needs a lattice-backed virtual "
                f"accelerator, but this build generated no channel manifest and "
                f"{channels[0]} pins VA_CHANNELS_FILE={channels[1]!r}. The IOC defaults "
                f"a file-backed channel source to VA_LATTICE=none, so the stand-in "
                f"would exit on the perturbation it ships. Restore the channel "
                f"databases the manifest is generated from, remove that line, or delete "
                f"`virtual_accelerator.live_standin`."
            ]
    return []


def shipped_bpm_errors_field_errors(spec: str) -> list[str]:
    """Fields a ``VA_BPM_ERRORS``-shaped spec perturbs that the shipped default may not.

    The grammar is ``DEVICE:field=value[,field=value...];DEVICE:...``, split
    exactly the way the container's ``_parse_bpm_errors`` splits it — ``;``
    between devices, ``:`` between a device and its fields, ``,`` between
    fields, ``=`` between a field and its value — so a spec this accepts is one
    the IOC will read the same way. Nothing here validates values or bounds:
    the IOC owns those, and repeating them would be a second set of limits free
    to drift from the ones that actually apply.

    An entry too malformed to name a field is left alone rather than reported
    twice — the IOC refuses it by name at boot, and this check is about *which*
    fields a default perturbs, not whether it parses.

    Args:
        spec: The env-var value to read.

    Returns:
        One failure per field outside :data:`STANDIN_BPM_ERROR_FIELDS`, in the
        order the spec spells them.
    """
    errors: list[str] = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        device, sep, fields_raw = entry.partition(":")
        if not sep or not device.strip() or not fields_raw.strip():
            continue
        for field_kv in fields_raw.split(","):
            field_kv = field_kv.strip()
            if not field_kv:
                continue
            field, _, _value = field_kv.partition("=")
            field = field.strip()
            if field in STANDIN_BPM_ERROR_FIELDS:
                continue
            errors.append(
                f"{STANDIN_BPM_ERRORS_ENV} entry {entry!r} perturbs {field!r}; the "
                f"shipped stand-in default is "
                f"{'/'.join(sorted(STANDIN_BPM_ERROR_FIELDS))} only"
            )
    return errors
