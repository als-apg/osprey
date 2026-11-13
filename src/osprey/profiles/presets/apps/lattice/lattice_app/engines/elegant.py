"""The elegant engine: optics, chromaticity and the two apertures from elegant runs.

It runs ``elegant`` in the deck's own directory, so the files the deck's elements name by a
relative path are found, on a command file in a scratch directory that every output is
written to, and reads the SDDS output with ``sdds2stream``. Both programs are the facility's: they are not in the
OSPREY image. A facility with its own elegant routines registers them as an engine of
their own (``engines:`` in the application's config) rather than bending this one.

The model's ``engine_settings``:

``elegant``, ``sdds2stream``
    The two programs; found on ``PATH`` when omitted.
``use_beamline``
    The beamline to use; elegant's default (the deck's last) when omitted.
``p_central_mev``
    The central momentum, for a model with no energy selected.
``run_setup``, ``twiss_output``, ``run_control``, ``find_aperture``, ``momentum_aperture``,
``closed_orbit``
    Entries added to, or replacing, those this engine writes in that namelist, e.g.
    ``find_aperture: {n_splits: 3}``, or ``run_setup: {search_path: "/lattices/kickmaps"}``
    for input files kept away from the deck.
``setup``
    Raw command-file text placed after ``run_setup``, e.g. ``&load_parameters``.
``tune_turns``
    The turns one particle is tracked at each momentum offset of ``chromaticity`` (1024).
``parallel``
    The elegant processes ``chromaticity`` runs at once (4, or the machine's processors
    when fewer).

Run as ``python -m lattice_app.engines.elegant <job.json> <out.json>``.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lattice_app.engines import run_engine, tune_vs_dp

Job = Mapping[str, Any]

#: The electron's rest energy in MeV.
ELECTRON_MEV = 0.51099895
#: How much of a failed run's output the error carries.
OUTPUT_TAIL = 1500
#: The turns tracked at one momentum offset, and the amplitude the particle starts at
#: around the closed orbit, in metres: small enough to carry no amplitude detuning.
TUNE_TURNS = 1024
TUNE_AMPLITUDE_M = 1.0e-6


def namelist(name: str, entries: Mapping[str, Any]) -> str:
    """One elegant namelist; a string is quoted, ``None`` leaves the entry out."""
    lines = [f"&{name}"]
    for key, value in entries.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)
        lines.append(
            f'    {key} = "{value}",' if isinstance(value, str) else f"    {key} = {value},"
        )
    return "\n".join([*lines, "&end", ""])


def _settings(job: Job) -> Mapping[str, Any]:
    return job.get("engine_settings") or {}


def _merged(job: Job, name: str, entries: Mapping[str, Any]) -> str:
    return namelist(name, {**entries, **(_settings(job).get(name) or {})})


def run_setup(job: Job) -> str:
    """The ``run_setup`` namelist of every run, and the facility's ``setup`` text after it."""
    settings = _settings(job)
    momentum = settings.get("p_central_mev")
    if job.get("energy_gev"):
        energy_mev = float(job["energy_gev"]) * 1.0e3
        momentum = math.sqrt(energy_mev**2 - ELECTRON_MEV**2)
    if momentum is None and "p_central_mev" not in (settings.get("run_setup") or {}):
        raise ValueError(
            "the model names no energy: set energies_gev, or engine_settings.p_central_mev"
        )
    entries = {
        "lattice": str(job["deck"]),
        "use_beamline": settings.get("use_beamline"),
        "p_central_mev": momentum,
    }
    return _merged(job, "run_setup", entries) + str(settings.get("setup") or "")


def lma_places(elements: int, wanted: int) -> list[int]:
    """The elements, evenly spread over the beamline, the momentum aperture is searched at."""
    if elements < 1 or wanted < 1:
        return []
    if wanted == 1:
        return [0]
    return sorted({round(index * (elements - 1) / (wanted - 1)) for index in range(wanted)})


def command_file(job: Job, scratch: Path, places: Sequence[int] = (0,)) -> str:
    """The command file of the job's analysis, its output written into ``scratch``.

    ``places`` are the elements of ``lma``.
    """
    analysis, settings = job["analysis"], job.get("settings") or {}
    if analysis == "optics":
        twiss = {"filename": str(scratch / "out.twi"), "matched": 1, "radiation_integrals": 1}
        return run_setup(job) + _merged(job, "twiss_output", twiss)
    if analysis not in _READERS:
        raise ValueError(f"the elegant engine does not compute {analysis}")
    control = _merged(job, "run_control", {"n_passes": int(settings["nturns"])})
    if analysis == "da":
        reach = float(settings["amp_max_mm"]) * 1.0e-3
        aperture = {
            "output": str(scratch / "out.aper"),
            "mode": "n-lines",
            "n_lines": int(settings["n_angles"]),
            "xmax": reach,
            "ymax": reach,
            "nx": 2 ** min(int(settings["n_bisect"]), 6) + 1,
            "n_splits": max(int(settings["n_bisect"]) - 6, 0),
            "full_plane": 0,
        }
        return run_setup(job) + control + _merged(job, "find_aperture", aperture)
    limit = float(settings["dp_max_pct"]) / 100.0
    search = {
        "x_initial": 1.0e-5,
        "y_initial": 1.0e-5,
        "delta_negative_limit": -limit,
        "delta_positive_limit": limit,
        "delta_step_size": limit / 50.0,
        "splits": 2,
        "steps_back": 1,
        "fiducialize": 1,
        # A place where nothing is lost inside the limits reports the limit, as pyAT does;
        # without this elegant stops the whole run there.
        "soft_failure": 1,
    }
    # elegant searches at every element from the one it starts at, which a whole ring does
    # not finish in: one search per place, each a run of its own in the same command file.
    return "".join(
        run_setup(job)
        + control
        + _merged(
            job,
            "momentum_aperture",
            {
                "output": str(scratch / f"out{index}.mmap"),
                "skip_elements": place,
                "process_elements": 1,
            }
            | search,
        )
        for index, place in enumerate(places)
    )


def _run(command: Sequence[str], cwd: Path) -> str:
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"{command[0]} could not start: {exc.strerror or exc}") from exc
    if done.returncode != 0:
        tail = (done.stderr.strip() or done.stdout.strip())[-OUTPUT_TAIL:]
        raise RuntimeError(f"{command[0]} exited with status {done.returncode}: {tail}")
    return done.stdout


def read_columns(job: Job, file: Path, columns: Sequence[str]) -> dict[str, list[str]]:
    """The named columns of an SDDS file's first page, as text."""
    program = str(_settings(job).get("sdds2stream") or "sdds2stream")
    out = _run(
        [program, file.name, f"-columns={','.join(columns)}", "-page=1", "-delimiter=\t"],
        file.parent,
    )
    rows = [line.split("\t") for line in out.splitlines() if line.strip()]
    if any(len(row) != len(columns) for row in rows):
        raise RuntimeError(f"{file.name} did not read as {len(columns)} columns")
    return {column: [row[index].strip() for row in rows] for index, column in enumerate(columns)}


def read_parameters(job: Job, file: Path, parameters: Sequence[str]) -> dict[str, float]:
    """The named parameters of an SDDS file's first page."""
    program = str(_settings(job).get("sdds2stream") or "sdds2stream")
    out = _run([program, file.name, f"-parameters={','.join(parameters)}", "-page=1"], file.parent)
    values = out.split()
    if len(values) != len(parameters):
        raise RuntimeError(f"{file.name} did not read as {len(parameters)} parameters")
    return {name: float(value) for name, value in zip(parameters, values, strict=True)}


def _floats(values: Sequence[str], scale: float = 1.0) -> list[float]:
    return [float(value) * scale for value in values]


def _optics(job: Job, workdir: Path) -> dict[str, Any]:
    file = workdir / "out.twi"
    scalars = read_parameters(
        job,
        file,
        ["nux", "nuy", "dnux/dp", "dnuy/dp", "betaxMax", "betayMax", "etaxMax", "pCentral"],
    )
    names = ["s", "betax", "betay", "alphax", "alphay", "etax", "etay", "psix", "psiy"]
    columns = read_columns(job, file, [*names, "ElementName", "ElementType"])
    s = _floats(columns["s"])
    return {
        "scalars": {
            "tune_x": scalars["nux"],
            "tune_y": scalars["nuy"],
            "chromaticity_x": scalars["dnux/dp"],
            "chromaticity_y": scalars["dnuy/dp"],
            "circumference_m": s[-1] if s else None,
            "energy_gev": math.hypot(scalars["pCentral"], 1.0) * ELECTRON_MEV / 1.0e3,
            "beta_x_max": scalars["betaxMax"],
            "beta_y_max": scalars["betayMax"],
            "eta_x_max": scalars["etaxMax"],
        },
        "units": {
            "circumference_m": "m",
            "energy_gev": "GeV",
            "beta_x_max": "m",
            "beta_y_max": "m",
            "eta_x_max": "m",
        },
        "tables": {
            "twiss": {
                "columns": {
                    "s": s,
                    "beta_x": _floats(columns["betax"]),
                    "beta_y": _floats(columns["betay"]),
                    "alpha_x": _floats(columns["alphax"]),
                    "alpha_y": _floats(columns["alphay"]),
                    "eta_x": _floats(columns["etax"]),
                    "eta_y": _floats(columns["etay"]),
                    "mu_x": _floats(columns["psix"]),
                    "mu_y": _floats(columns["psiy"]),
                    "name": columns["ElementName"],
                    "kind": columns["ElementType"],
                },
                "units": {"s": "m", "beta_x": "m", "beta_y": "m", "eta_x": "m", "eta_y": "m"},
            }
        },
    }


def _da(job: Job, workdir: Path) -> dict[str, Any]:
    file = workdir / "out.aper"
    columns = read_columns(job, file, ["x", "y"])
    x, y = _floats(columns["x"], 1.0e3), _floats(columns["y"], 1.0e3)
    area = read_parameters(job, file, ["Area"])["Area"] * 1.0e6
    return {
        "scalars": {
            "area_mm2": area,
            "x_max_mm": max(x, default=None),
            "x_min_mm": min(x, default=None),
            "y_max_mm": max(y, default=None),
        },
        "units": {"area_mm2": "mm^2", "x_max_mm": "mm", "x_min_mm": "mm", "y_max_mm": "mm"},
        "tables": {
            "aperture": {
                "columns": {
                    "angle_deg": [
                        math.degrees(math.atan2(b, a)) for a, b in zip(x, y, strict=True)
                    ],
                    "x_mm": x,
                    "y_mm": y,
                },
                "units": {"angle_deg": "deg", "x_mm": "mm", "y_mm": "mm"},
            }
        },
    }


def _lma(job: Job, workdir: Path) -> dict[str, Any]:
    names = ["s", "deltaPositive", "deltaNegative"]
    columns: dict[str, list[str]] = {name: [] for name in names}
    for file in sorted(workdir.glob("out*.mmap"), key=lambda path: int(path.stem[3:])):
        for name, values in read_columns(job, file, names).items():
            columns[name] += values
    plus = _floats(columns["deltaPositive"], 100.0)
    minus = _floats(columns["deltaNegative"], 100.0)
    return {
        "scalars": {
            "dp_plus_min_pct": min(plus, default=None),
            "dp_minus_max_pct": max(minus, default=None),
        },
        "units": {"dp_plus_min_pct": "%", "dp_minus_max_pct": "%"},
        "tables": {
            "aperture": {
                "columns": {"s": _floats(columns["s"]), "dp_plus_pct": plus, "dp_minus_pct": minus},
                "units": {"s": "m", "dp_plus_pct": "%", "dp_minus_pct": "%"},
            }
        },
    }


_READERS = {"optics": _optics, "da": _da, "lma": _lma}


def compute(job: Job) -> dict[str, Any]:
    """Run elegant on the job's deck and read the analysis's result."""
    with tempfile.TemporaryDirectory(prefix="lattice-elegant-") as scratch:
        workdir = Path(scratch)
        # elegant runs where the deck lives, as it does by hand there; every file it writes
        # is named into the scratch directory, so nothing lands beside the deck.
        where = Path(job["deck"]).parent
        program = str(_settings(job).get("elegant") or "elegant")
        places: list[int] = []
        if job["analysis"] == "lma":
            # The beamline's length in elements is elegant's to say: the deck is not parsed here.
            count = run_setup(job) + namelist(
                "twiss_output", {"filename": str(workdir / "count.twi")}
            )
            (workdir / "count.ele").write_text(count, encoding="utf-8")
            _run([program, str(workdir / "count.ele")], where)
            elements = len(read_columns(job, workdir / "count.twi", ["s"])["s"]) - 1
            places = lma_places(elements, int((job.get("settings") or {})["n_refpts"]))
        (workdir / "run.ele").write_text(command_file(job, workdir, places), encoding="utf-8")
        _run([program, str(workdir / "run.ele")], where)
        return _READERS[job["analysis"]](job, workdir)


def tune_command_file(job: Job, scratch: Path, index: int, dp: float, turns: int) -> str:
    """The command file of one momentum offset of ``chromaticity``.

    One particle is tracked around the closed orbit at the offset ``dp``, which a
    zero-length ``MALIGN`` at the start of the ring gives it once. elegant's own
    off-momentum ``twiss_output`` is good to first order in the offset only: its matrix
    is truncated, and so are ``dnux/dp2`` and ``dnux/dp3``. The Twiss tune is still
    written, to say which side of the half integer the tracked tune is on.
    """
    watch = scratch / f"tune{index}.w1"
    start = {"name": "*", "skip": 0, "add_at_start": 1, "allow_no_matches": 1}
    kick = f"DPKICK: MALIGN,DP={dp!r},ON_PASS=0,FORCE_MODIFY_MATRIX=1"
    monitor = f'TUNEWATCH: WATCH,FILENAME=\\"{watch}\\",MODE=CENTROID'
    orbit = {"closed_orbit_accuracy": 1.0e-12, "iteration_fraction": 1}
    orbit["closed_orbit_iterations"] = 200
    twiss = {"filename": str(scratch / f"tune{index}.twi"), "matched": 1, "output_at_each_step": 1}
    beam = f"centroid[0] = {TUNE_AMPLITUDE_M},0,{TUNE_AMPLITUDE_M},0,0,0,"
    return (
        run_setup(job)
        + namelist("insert_elements", start).replace("&end", f'    element_def = "{kick}",\n&end')
        + namelist("insert_elements", start).replace(
            "&end", f'    element_def = "{monitor}",\n&end'
        )
        + namelist("run_control", {"n_passes": turns})
        + _merged(job, "closed_orbit", orbit)
        + _merged(job, "twiss_output", twiss)
        + namelist("bunched_beam", {"n_particles_per_bunch": 1}).replace(
            "&end", f"    {beam}\n&end"
        )
        + namelist("track", {"offset_by_orbit": 1})
    )


def fractional_tune(signal: Sequence[float]) -> float:
    """The tune of a turn-by-turn position, between 0 and 0.5.

    The strongest line of the Hann-windowed spectrum, refined between the bins of the
    transform by a search on the windowed sum itself.
    """
    import numpy as np

    values = np.asarray(signal, dtype=float)
    turns = np.arange(len(values))
    weighted = (values - values.mean()) * np.hanning(len(values))

    def strength(tune: float) -> float:
        return float(abs(np.sum(weighted * np.exp(-2j * np.pi * tune * turns))))

    peak = int(np.argmax(np.abs(np.fft.rfft(weighted))[1:])) + 1
    low, high = (peak - 1) / len(values), (peak + 1) / len(values)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        left, right = high - golden * (high - low), low + golden * (high - low)
        if strength(left) > strength(right):
            high = right
        else:
            low = left
    return min(max((low + high) / 2.0, 0.0), 0.5)


def _tracked_tunes(job: Job, workdir: Path, index: int, turns: int) -> tuple[float, float]:
    """The fractional tunes of one finished offset; NaN where it has no stable optics.

    elegant exits with status 0 whether or not it found a closed orbit and periodic
    optics, so the output is what says: a Twiss page with finite tunes, and a watch file
    in which the particle lived every turn.
    """
    gap = (math.nan, math.nan)
    watch = workdir / f"tune{index}.w1"
    try:
        twiss = read_parameters(job, workdir / f"tune{index}.twi", ["nux", "nuy"])
    except RuntimeError:
        return gap  # no closed orbit: the Twiss file has no page
    if not (math.isfinite(twiss["nux"]) and math.isfinite(twiss["nuy"])) or not watch.is_file():
        return gap
    columns = read_columns(job, watch, ["Cx", "Cy", "Particles"])
    if len(columns["Cx"]) < turns or any(float(alive) != 1 for alive in columns["Particles"]):
        return gap
    tunes = []
    for plane, column in (("nux", "Cx"), ("nuy", "Cy")):
        tune, expected = fractional_tune(_floats(columns[column])), twiss[plane] % 1.0
        # The spectrum of a position cannot tell a tune from its mirror about the half
        # integer; the Twiss tune, close enough at any offset, can.
        mirror = 1.0 - tune
        tunes.append(mirror if abs(mirror - expected) < abs(tune - expected) else tune)
    return tunes[0], tunes[1]


def chromaticity(job: Job) -> dict[str, Any]:
    """The tunes against momentum offset, one tracking run per offset, several at once."""
    settings, engine = job.get("settings") or {}, _settings(job)
    count = int(settings["n_steps"])
    low, high = float(settings["dp_min_pct"]), float(settings["dp_max_pct"])
    offsets = [low + (high - low) * step / max(count - 1, 1) for step in range(count)]
    turns = int(engine.get("tune_turns") or TUNE_TURNS)
    program = str(engine.get("elegant") or "elegant")
    where = Path(job["deck"]).parent
    with tempfile.TemporaryDirectory(prefix="lattice-elegant-") as scratch:
        workdir = Path(scratch)

        def solve(index: int) -> tuple[float, float]:
            commands = workdir / f"tune{index}.ele"
            text = tune_command_file(job, workdir, index, offsets[index] / 100.0, turns)
            commands.write_text(text, encoding="utf-8")
            _run([program, str(commands)], where)
            return _tracked_tunes(job, workdir, index, turns)

        workers = int(engine.get("parallel") or min(4, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
            tunes = list(pool.map(solve, range(count)))
    return tune_vs_dp(offsets, [tune[0] for tune in tunes], [tune[1] for tune in tunes])


ANALYSES = {**dict.fromkeys(_READERS, compute), "chromaticity": chromaticity}

if __name__ == "__main__":
    sys.exit(run_engine(ANALYSES))
