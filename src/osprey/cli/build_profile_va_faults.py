"""The ``virtual_accelerator.live_standin:`` refusals, and the fault-set grammar.

``live_standin: <port>`` stands a SECOND soft-IOC up and gives the deployment a
THIRD control target, ``standin``, configured from a block of its own
(``control_system.connector.live_standin``). ``live`` keeps meaning the
facility's authored ``epics`` block throughout — the stand-in never takes that
label — so a facility may stand the rehearsal up beside its real machine, and a
deployment may equally be baselined on the stand-in itself
(``control_system.type: live_standin``).

That makes the stand-in a port the deployment spends and a target the
deployment claims, and both can be claimed twice. The rules live here rather
than inside :meth:`BuildProfile.validate` for the reason
:func:`~osprey.cli.build_profile_archiver.va_archiver_errors` does: a block's
rules belong beside the block, but they are *reported* from validate's single
accumulator so a facility fixing a profile meets every problem it has in one
pass.

Three of them are about the third target rather than about ports:

* :func:`standin_baseline_errors` — a deployment baselined on ``live_standin``
  that builds no stand-in. The baseline names a machine this build does not
  stand up, so every session would dial a port nothing serves.
* :func:`standin_archive_errors` — the archive belongs to the machine it
  records. A stand-in plus a ``va_archiver`` recorder writes the deployment's
  OWN store, which is legal only where the baseline is a simulated machine or
  the stand-in itself; on a baseline naming the facility's own machine that
  store would be read as the real machine's past.
* :func:`live_standin_lattice_errors` — a readout perturbation with no lattice
  behind it. The IOC treats a perturbation without ``VA_LATTICE=builtin`` as
  fatal at boot (``services/virtual_accelerator/entrypoint.py``). Left alone
  that is a container in a crash loop, hours after the build reported success.

The perturbation grammar itself is parsed here too
(:func:`shipped_bpm_errors_field_errors`), mirroring the container-side splitting
without importing it: ``entrypoint.py`` runs inside the VA image and reads
``os.environ``, so it is not importable from a build.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.connectors.types import (
    _SIMULATED_TYPES,
    EPICS,
    LIVE_STANDIN,
    STANDIN_TYPES,
    VIRTUAL_ACCELERATOR,
    resolve_control_system_type,
)

# The one nested-tree walker, borrowed rather than repeated for the same reason
# the path-tree builder below is.
from osprey.deployment.reach import dotted_get

# The one path-tree builder in this package, borrowed rather than repeated: a
# `config:` block addresses the same leaf through a dotted key or a nested
# mapping, and a second implementation of "which leaves does this reach" is a
# second answer free to disagree with the renderer's.
from .build_profile_archiver import VAArchiverConfig, _expand_dotted
from .build_profile_schema import VAConfig

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

#: Key of the VA gateway table, in the nested spelling a rendered config reads.
#: Mirrors ``_VA_CONNECTOR_PATH`` in ``build_injectors``; the two ends of the
#: same block, one written by the injector and one refused here.
_VA_GATEWAYS_KEY = f"control_system.connector.{VIRTUAL_ACCELERATOR}.gateways"

#: Key of the deployment baseline's control-system section. The *type* inside
#: it is read through :func:`resolve_control_system_type` rather than by dotted
#: lookup, so an absent ``type:`` resolves the way every other reader resolves
#: it (to the mock) instead of to a second answer invented here.
_CONTROL_SYSTEM_KEY = "control_system"

#: Baseline types a deployment's own recorded store may legally belong to: the
#: machines a deployment stands up for itself. Spelled as the union the target
#: vocabulary already defines, so widening either half widens this rule with it.
_OWN_MACHINE_TYPES = _SIMULATED_TYPES + STANDIN_TYPES


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


def standin_baseline_errors(config: Any, va: VAConfig | None) -> list[str]:
    """Refuse a deployment baselined on a stand-in it does not build.

    ``control_system.type: live_standin`` is a legal baseline — the deployment
    that runs the soft IOC may also start every session on it — but only where
    the deployment actually stands one up. Without
    ``virtual_accelerator.live_standin`` the baseline names a machine no
    container serves, and the failure surfaces as a connector dialing a port
    nothing is listening on, one ``osprey up`` later.

    Args:
        config: The profile's resolved ``config:`` block.
        va: The parsed ``virtual_accelerator:`` block, or ``None`` when the
            profile declares none — which is itself a way to reach this fault.

    Returns:
        The single failure, or an empty list.
    """
    if _baseline_type(config) != LIVE_STANDIN:
        return []
    if va is not None and va.live_standin is not None:
        return []
    return [
        f"control_system.type: {LIVE_STANDIN} with no "
        f"virtual_accelerator.live_standin — the baseline names a machine this "
        f"deployment does not stand up, so every session would dial a port nothing "
        f"serves. Set `virtual_accelerator.live_standin` to the port the stand-in "
        f"should serve, or set `control_system.type` back to the connector that "
        f"reaches this machine (`{EPICS}` for a facility's own)."
    ]


def standin_archive_errors(
    config: Any, va: VAConfig | None, va_archiver: VAArchiverConfig | None
) -> list[str]:
    """Refuse a recorded archive that would be read as the real machine's past.

    **The archive belongs to the machine it records.** A ``va_archiver:`` block
    is a store this deployment writes for itself, filled by the recorder
    sampling whatever machine the deployment runs — with a stand-in built, that
    machine is the stand-in. Where the baseline names the facility's own
    control system, the same store is served to every session as the history of
    the real machine, and nothing in the readout says otherwise.

    Legal exactly where the baseline is a machine the deployment stands up for
    itself (:data:`_OWN_MACHINE_TYPES`): a simulated baseline, or the stand-in
    as its own baseline.

    Args:
        config: The profile's resolved ``config:`` block.
        va: The parsed ``virtual_accelerator:`` block, or ``None``.
        va_archiver: The parsed ``va_archiver:`` block, or ``None`` when the
            profile records no store of its own.

    Returns:
        The single failure, or an empty list.
    """
    if va is None or va.live_standin is None or va_archiver is None:
        return []
    baseline = _baseline_type(config)
    if baseline in _OWN_MACHINE_TYPES:
        return []
    return [
        f"virtual_accelerator.live_standin with a va_archiver block on a "
        f"control_system.type: {baseline} baseline — the archive belongs to the "
        f"machine it records. The recorder writes THIS deployment's store from the "
        f"stand-in, and on a baseline naming the facility's own machine that store "
        f"is served as the real machine's past. Either delete `va_archiver` and let "
        f"the deployment read the facility's own archiver, or baseline this "
        f"deployment on the machine being recorded (`control_system.type: "
        f"{LIVE_STANDIN}`, or `{VIRTUAL_ACCELERATOR}` for the simulation)."
    ]


def _baseline_type(config: Any) -> str:
    """The control-system type a profile's ``config:`` block selects.

    Read spelling-independently — the renderer honors a dotted key and a nested
    mapping alike — and resolved through the connector vocabulary's own
    resolver, so a profile that says nothing about its control system gets the
    same answer here as the factory gives it at runtime.
    """
    baseline: str = resolve_control_system_type(
        dotted_get(_expand_dotted(config), _CONTROL_SYSTEM_KEY)
    )
    return baseline


def effective_standin_bpm_errors(project_root: Path, build_dir: Path | None = None) -> str:
    """The readout perturbation the stand-in would actually boot with.

    The compose file renders the stand-in's ``VA_BPM_ERRORS`` from the
    deployment's ``VA_STANDIN_BPM_ERRORS``, so a chain that names the key
    answers this on its own — including with an EMPTY value, which is an empty
    perturbation rather than an absent one. Turning the shipped faults off is
    the documented way out of :func:`live_standin_lattice_errors`, and it can
    only be that if an empty value is honored rather than rounded back up.

    **The fallback is lattice-conditional**, and the condition is asked of
    :func:`~osprey.services.virtual_accelerator.manifest.standin_defaults.default_bpm_errors_for_lattice`
    rather than restated here: the shipped default exists for the builtin
    lattice only, since it names offsets on a PyAT model and there is nothing to
    displace anywhere else. That is the same function the render side writes the
    compose interpolation from, so validation and the rendered file cannot come
    to different answers about what the container receives.

    Args:
        project_root: The deployment repo root, whose env chain the containers
            are handed. Also the profile root at validation time.
        build_dir: The published output zone, when the caller has one — handed
            on to the lattice resolver, whose chain it extends.

    Returns:
        The perturbation spec, stripped; empty when the stand-in ships none.
    """
    from osprey.services.virtual_accelerator.manifest.standin_defaults import (
        default_bpm_errors_for_lattice,
    )
    from osprey.utils.dotenv import merge_chain, resolved_va_lattice

    chain: dict[str, str] = merge_chain(Path(project_root))
    if STANDIN_BPM_ERRORS_ENV in chain:
        return chain[STANDIN_BPM_ERRORS_ENV].strip()
    return default_bpm_errors_for_lattice(resolved_va_lattice(project_root, build_dir)).strip()


def live_standin_lattice_errors(project_root: Path, build_dir: Path | None = None) -> list[str]:
    """Reasons the stand-in would exit at boot for want of a lattice.

    The stand-in ships a readout perturbation, and the IOC refuses a
    perturbation it cannot apply: without ``VA_LATTICE=builtin`` there is no
    PyAT model to displace, and the entrypoint raises rather than serving a
    machine that ignores the faults it was configured with.

    Both halves are read the way the deployment will read them —
    :func:`effective_standin_bpm_errors` for the perturbation,
    :func:`~osprey.utils.dotenv.resolved_va_lattice` for the lattice — which
    narrows this to exactly one shape: a chain that ASKED for a fault set, on a
    lattice that cannot apply it. A deployment that never asked has nothing to
    refuse, because the shipped default is the builtin lattice's and the render
    gives a latticeless stand-in an empty set. So a facility may pin
    ``VA_LATTICE=none`` and still rehearse; only its own non-empty
    ``VA_STANDIN_BPM_ERRORS`` beside that pin is a build that cannot boot.

    Only the env chain is read here, at build time as at validation time. The
    other half of "what will VA_LATTICE be" — whether this render generated a
    channel manifest, which is what an UNPINNED chain resolves through — is
    knowable only once a render exists, and is asked on the deployment side
    (``compose_generator``) against the same resolver.

    Args:
        project_root: The deployment repo root, whose env chain the containers
            are handed. Also the profile root at validation time.
        build_dir: The published output zone, when the caller has one — handed
            straight to the lattice resolver, whose chain it extends.

    Returns:
        The accumulated failures, empty when the stand-in has a lattice or
        ships no perturbation to need one.
    """
    from osprey.services.virtual_accelerator.manifest.standin_defaults import LATTICE_BUILTIN
    from osprey.utils.dotenv import resolved_va_lattice

    if not effective_standin_bpm_errors(project_root, build_dir):
        return []

    lattice: str = resolved_va_lattice(project_root, build_dir)
    if lattice.strip().lower() == LATTICE_BUILTIN:
        return []
    return [
        f"virtual_accelerator.live_standin ships a readout perturbation, but this "
        f"deployment's env chain resolves VA_LATTICE={lattice!r}. There is no PyAT "
        f"model to displace, so the stand-in's IOC exits at boot rather than serving "
        f"a machine that ignores the faults it was configured with. Set "
        f"VA_LATTICE={LATTICE_BUILTIN}, or turn the perturbation off with "
        f"{STANDIN_BPM_ERRORS_ENV}= (empty)."
    ]


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
