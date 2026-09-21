"""The readout perturbation the stand-in instance ships with.

A deployment that sets ``virtual_accelerator.live_standin`` runs a SECOND
soft-IOC container as its own ``standin`` target. Both instances run one image
over one lattice and one machine description, so without a perturbation the two
would read identically -- and a stand-in that is indistinguishable from the
machine it stands in for proves nothing. This module resolves the perturbation
that makes them tell apart: the value the compose template renders as the
stand-in's ``VA_BPM_ERRORS`` default.

**The perturbation is the served tree's data.** Which devices are displaced
and by how much is a property of the machine being simulated, so it is stated
where that machine is described -- the :data:`STANDIN_BPM_ERRORS_KEY` entry of
the ``machine.json`` the deployment's own containers are handed -- and only
read here. Every deployment answers this from its own tree, the packaged demo
included; there is no framework-side fallback, so a tree that states no
perturbation ships none, and no deployment is ever handed another machine's
device names. Nothing in this package names a device: device names belong to a
facility, and one spelled in framework code would serve exactly that facility.

**Offsets only, and that is the design, not a simplification.** With the rest
of ``bpm_read``'s keyword set left at identity -- unit gain, unit calibration,
positive polarity, zero roll, zero noise -- a seeded reading is exactly
``x - offset``: a pure additive, deterministic transform of the unperturbed
value. That is what lets the archiver seed reproduce the same systematic
offsets by ADDING them to the values it synthesizes, rather than re-deriving
a reading through a second copy of the readout arithmetic. A gain or a roll
would make the transform depend on the unperturbed value in a way the seed
cannot reproduce additively, and a noise term would make it irreproducible at
all. A build-time check refuses any other field in the shipped default for
exactly this reason.

**What a tree's own entry has to satisfy** is pinned by the test suite rather
than restated here: every device it names is one that tree's manifest serves,
every axis it perturbs has a readback address there, and its magnitudes sit
well clear of the machine's own motion -- so a reader comparing the two targets
sees the offset, not the weather. How large a displacement may be is a question
for that machine alone: the container's parser carries a seeded offset through
as written, in the unit the monitor publishes.

**An operator can still override it at deploy time.** The template renders
``VA_BPM_ERRORS: "${VA_STANDIN_BPM_ERRORS-<the rendered default>}"``, so a
``VA_STANDIN_BPM_ERRORS`` in the deployment's env chain replaces that default
wholesale -- including an EMPTY one. ``-`` substitutes only for an UNSET
variable, so ``VA_STANDIN_BPM_ERRORS=`` is a deployment asking for an
unperturbed stand-in and gets one; that is the shortest way to run this
stand-in clean, and it is what validation points a facility at when its lattice
cannot carry these offsets.

**A stand-in with no lattice is handed the empty set** whatever its tree
states. These offsets displace a lattice's model, and a deployment whose env
chain serves no lattice has none to displace; the render hands that stand-in
nothing (``compose_generator._standin_perturbation``) and reports that it
serves its manifest unperturbed.

The container's interpolation is the authority on all of this, and the archiver
seed follows the same rule
(``container_lifecycle._standin_bpm_error_spec``), so the two cannot disagree
about which machine the stand-in is. The baseline instance's own
``VA_BPM_ERRORS`` is a separate variable and is untouched by either.
"""

from __future__ import annotations

import json
from pathlib import Path

from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths

#: The subdirectory a deployment keeps its served data tree in, under the
#: deployment repo root and under a published render alike. The build copies
#: the profile's tree here, so both roots carry one layout and
#: :class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`
#: resolves either.
DATA_DIR_NAME = "data"

#: The ``machine.json`` key a machine states its stand-in's perturbation under,
#: written in ``VA_BPM_ERRORS`` grammar
#: (``DEVICE:field=value[,field=value];DEVICE:...``).
STANDIN_BPM_ERRORS_KEY = "standin_bpm_errors"


def read_standin_bpm_errors(machine_json: Path) -> str:
    """The perturbation a machine description states for its stand-in.

    Args:
        machine_json: The ``machine.json`` describing the machine the stand-in
            stands in for.

    Returns:
        The spec as written, stripped. Empty where the machine states none --
        a machine whose stand-in is asked to read exactly as it does.

    Raises:
        ValueError: The key is present but is not text. The value is
            interpolated into a compose line verbatim, so anything else would
            reach the container as its own repr and fail at a boot nobody is
            watching.
    """
    machine = json.loads(machine_json.read_text(encoding="utf-8"))
    spec = machine.get(STANDIN_BPM_ERRORS_KEY, "")
    if not isinstance(spec, str):
        raise ValueError(
            f"{machine_json}: '{STANDIN_BPM_ERRORS_KEY}' must be a VA_BPM_ERRORS "
            f"string, not {type(spec).__name__}"
        )
    return spec.strip()


#: What the packaged demo tree's machine states for its own stand-in.
#:
#: The demo reaches this value the way every deployment reaches its own -- from
#: the ``machine.json`` in the tree its containers are handed, which a project
#: built from the demo preset carries a copy of. Nothing on the deployment path
#: reads this constant: it is here for the suites and the end-to-end lane that
#: serve exactly this tree and need to know what it says without re-reading it.
STANDIN_BPM_ERRORS_DEFAULT: str = read_standin_bpm_errors(PACKAGE_PATHS.machine_json)


def served_data_root(repo_root: Path, build_dir: Path | None = None) -> Path | None:
    """The data tree a deployment's containers will be handed, on this host.

    The two roots a deployment's answer can live in, in the order
    :func:`~osprey_connectors.dotenv.resolved_va_lattice` reads its chain from
    them: the deployment repo's own tree, then the published render, which wins
    because it is the tree the containers actually mount. A render that has not
    staged its data yet leaves the repo's tree as the honest answer, since that
    is what the build is about to copy.

    Args:
        repo_root: The deployment repo root.
        build_dir: The published output zone, when the caller has one.

    Returns:
        The data root to resolve a ``machine.json`` against, or ``None`` where
        neither root describes a machine -- a deployment whose stand-in has no
        machine to be a stand-in for.
    """
    resolved: Path | None = None
    for root in (repo_root, build_dir):
        if root is None:
            continue
        candidate = Path(root) / DATA_DIR_NAME
        if ManifestPaths(data_root=candidate).machine_json.is_file():
            resolved = candidate
    return resolved


def default_bpm_errors_for_lattice(served: bool, data_root: Path | None) -> str:
    """The perturbation a stand-in is handed by default, given its tree.

    One rule, one home, for the two sides that must agree about what the
    container receives: the render writes this into the stand-in's
    ``${VA_STANDIN_BPM_ERRORS-...}`` interpolation
    (``compose_generator._standin_perturbation``) and the build resolves the
    same value to decide whether the deployment has asked for a perturbation it
    cannot boot with
    (``build_profile_va_faults.effective_standin_bpm_errors``). Two spellings of
    it could disagree, and the disagreement would be a build that validated on
    one answer and rendered on another.

    The answer comes from the deployment's own tree and from nowhere else, so
    one bindings-driven path serves every facility: a tree that states no
    perturbation ships none rather than inheriting a machine it has never
    served. The read happens per call rather than once, because a build stages
    that tree while this process runs.

    A stand-in with no lattice is handed the EMPTY set whatever its tree says:
    these offsets displace a model, and a deployment serving no lattice has none
    for them to move. Such a stand-in serves its manifest unperturbed, which is
    honest; carrying faults nothing can apply is not.

    Args:
        served: Whether the deployment's env chain names a lattice for the
            stand-in to boot with -- ``VA_LATTICE`` resolved to anything but
            ``none``, per
            :func:`~osprey_connectors.dotenv.resolved_va_lattice`.
        data_root: The served tree, from :func:`served_data_root`. ``None``
            where the deployment describes no machine.

    Returns:
        The tree's own perturbation, or ``''`` where there is no lattice to
        carry it, no tree to read, or no entry in that tree.
    """
    if not served or data_root is None:
        return ""
    machine_json = ManifestPaths(data_root=Path(data_root)).machine_json
    if not machine_json.is_file():
        return ""
    return read_standin_bpm_errors(machine_json)


def parse_bpm_error_spec(spec: str) -> dict[str, dict[str, float]]:
    """``"D1:offset_x=1.5e-4;D2:offset_x=-2e-4"`` -> ``{fam: {field: value}}``.

    The host side's one copy of the ``VA_BPM_ERRORS`` split, for every consumer
    that needs the offsets as numbers rather than as an env string: this
    module's own shipped default, and the deploy-time archiver seed, which adds
    the same systematic offsets to the history it synthesizes so the stand-in's
    past and present agree. Spelled once here because two host-side splits of
    one grammar could disagree about what the container is serving, and the
    seed's whole job is to match it.

    The authority on the grammar is ``entrypoint._parse_bpm_errors``, which is
    what the container actually runs; it cannot be called here because it reads
    ``os.environ`` and lives in a module that imports the whole serving stack.
    So the split is spelled again -- ``;`` between devices, the LAST ``:``
    between a device and its fields (a device spelled as the address its
    reading is published on is colon-separated at every level and stays one
    token, while a field list carries no colon), ``,`` between fields, ``=``
    between a field and its value -- and pinned: the test suite parses the
    shipped default through the real entrypoint parser and asserts the two
    agree.

    Nothing here validates bounds or field names: the IOC owns those and
    refuses a bad spec by name at boot, and a second set of limits here would
    be free to drift from the ones that actually apply. An entry too malformed
    to yield a number is dropped rather than raised on -- a deploy must not die
    on a value the container is about to reject with a better message.

    Args:
        spec: The env-var value to split.

    Returns:
        One entry per device that named at least one readable field.
    """
    parsed: dict[str, dict[str, float]] = {}
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        device, separator, fields_raw = entry.rpartition(":")
        device = device.strip()
        if not separator or not device:
            continue
        fields: dict[str, float] = {}
        for field_kv in fields_raw.split(","):
            field, field_separator, raw_value = field_kv.strip().partition("=")
            if not field_separator:
                continue
            try:
                fields[field.strip()] = float(raw_value)
            except ValueError:
                continue
        if fields:
            parsed[device] = fields
    return parsed


def parse_standin_default() -> dict[str, dict[str, float]]:
    """:data:`STANDIN_BPM_ERRORS_DEFAULT` as ``{fam_name: {field: value}}``."""
    return parse_bpm_error_spec(STANDIN_BPM_ERRORS_DEFAULT)
