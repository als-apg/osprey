"""The BPM readout perturbation the stand-in instance ships with.

A deployment that sets ``virtual_accelerator.live_standin`` runs a SECOND
soft-IOC container as its ``live`` target. Both instances run one image over
one lattice and one machine description, so without a perturbation the two
would read identically -- and a stand-in that is indistinguishable from the
machine it stands in for proves nothing. This module holds the perturbation
that makes them tell apart, as the value the compose template renders as the
stand-in's ``VA_BPM_ERRORS`` default.

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

**Every device named here exists in the packaged manifest**
(``channel_manifest.json``), and every named axis has its
``SR:DIAG:BPM:<id>:POSITION:<axis>`` address there -- a fam_name the lattice
has no BPM for perturbs nothing (the physics bridge warns and carries on), and
an axis the manifest does not serve has no readback for the offset to show up
on. The magnitudes are 100-200 um: comfortably inside the parse bounds
(``|offset| <= 1e-2`` m, see ``entrypoint._BPM_ERROR_FIELD_BOUNDS``) and well
clear of the machine's own motion, whose BPM channels sit on a 0.0 m baseline
with a 30 um wander texture and 1 um noise. A reader comparing the two targets
sees the offset, not the weather.

**An operator can still override it at deploy time.** The template renders
``VA_BPM_ERRORS: "${VA_STANDIN_BPM_ERRORS:-<this value>}"``, so a non-empty
``VA_STANDIN_BPM_ERRORS`` in the project ``.env`` replaces this default
wholesale. An EMPTY one does not: ``:-`` treats empty exactly like unset, so
either way the stand-in boots on the shipped default. Running a stand-in
deliberately unperturbed therefore means setting a non-empty value that names
no offsets -- ``;`` is the shortest, and parses to an empty fault set -- or not
asking for the stand-in at all (drop ``virtual_accelerator.live_standin``). The
container's interpolation is the authority on this, and the archiver seed
follows the same rule, so the two cannot disagree about which machine the
stand-in is. The baseline instance's own ``VA_BPM_ERRORS`` is a separate
variable and is untouched by either.
"""

from __future__ import annotations

#: The shipped stand-in perturbation, in ``VA_BPM_ERRORS`` grammar
#: (``DEVICE:field=value[,field=value];DEVICE:...``; device keys are BPM
#: fam_names, ``BPM`` + the manifest's device id). Four devices spread around
#: the 72-BPM ring, mixed signs, some single-axis: enough that a comparison
#: between the two targets cannot come out zero by luck, few enough that the
#: rendered compose line stays readable. Deterministic by construction -- no
#: randomness, no host-dependent value.
STANDIN_BPM_ERRORS_DEFAULT: str = (
    "BPM03:offset_x=1.5e-4,offset_y=-1.0e-4;"
    "BPM21:offset_x=-2.0e-4;"
    "BPM45:offset_y=1.2e-4;"
    "BPM63:offset_x=1.0e-4,offset_y=1.8e-4"
)


def parse_bpm_error_spec(spec: str) -> dict[str, dict[str, float]]:
    """``"BPM03:offset_x=1.5e-4;BPM21:offset_x=-2e-4"`` -> ``{fam: {field: value}}``.

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
    So the split is spelled again -- ``;`` between devices, ``:`` between a
    device and its fields, ``,`` between fields, ``=`` between a field and its
    value -- and pinned: the test suite parses the shipped default through the
    real entrypoint parser and asserts the two agree.

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
        device, separator, fields_raw = entry.partition(":")
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
