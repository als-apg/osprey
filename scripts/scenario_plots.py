#!/usr/bin/env python3
"""Write the pictures the control-assistant demo scenarios attach to their logbook entries.

Usage: ``python scripts/scenario_plots.py [SIMULATION_DIR]``

``SIMULATION_DIR`` defaults to the control-assistant template's
``data/simulation``. Every output goes to ``<scenario>/plots/`` inside it and is
named by an entry's ``attachments`` in that scenario's ``logbook.json``.

Every value is derived from the scenario's own definition, never typed in:

* ``bpm-polarity`` -- a shipped PNG: each sector BPM's reading against a kick
  of the corrector upstream of the reversed BPM, from the demo ring's closed
  orbit, with the scenario's BPM polarity applied to the readings and the
  machine file's BPM noise added. It has no time axis, so it is drawn here.
* ``rf-thermal`` -- a plot spec: the two cavity temperatures over a week,
  synthesized by the simulation engine from the scenario's archiver events.
* ``nominal`` -- a plot spec: the RMS of all ring BPM readings over a week,
  synthesized by the simulation engine from the machine file's BPM texture and
  noise.

A plot spec carries its series and a time axis in hours before the entry that
attaches it, resolved from that entry's ``when`` against the same anchor the
telemetry is synthesized against. The seeder draws it with
:func:`osprey.simulation.plots.render_plot_spec` at the entry's real timestamp,
so the picture shows the dates the entry was written on.

Every picture states only what an operator's strip-chart export would: device
and quantity names, units, and the data. Nothing on it points at a finding.

The output is deterministic: draws come from fixed seeds and keyed series, and
the PNG is re-encoded without text chunks, so a re-run on the same library
versions rewrites identical bytes.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATION_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"

#: The scenario start every time series is synthesized against. Any instant
#: gives the same event shapes relative to an entry; a fixed one keeps the keyed
#: noise and texture reproducible.
ANCHOR = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

#: Span of every time-series picture, ending at its entry's time.
WINDOW = timedelta(days=7)

#: Corrector kicks of the bump test, in microradians.
KICKS_URAD = np.linspace(-40.0, 40.0, 9)

#: Seed of the BPM read noise in the bump test.
BUMP_NOISE_SEED = 17

CAVITY_TEMPERATURES = ("SR:RF:CAVITY:01:TEMPERATURE:RB", "SR:RF:CAVITY:02:TEMPERATURE:RB")

BPM_POLARITY_PLOT = "plots/corrector_bump_test.png"
RF_THERMAL_SPEC = "plots/cavity_temperatures.json"
NOMINAL_SPEC = "plots/orbit_rms.json"

#: Decimals kept in a spec's numbers: far below the noise drawn, and it keeps
#: the bundle files small.
_HOURS_DECIMALS = 4
_VALUE_DECIMALS = 3


@dataclass(frozen=True)
class BumpTest:
    """One corrector's bump test over the BPMs of a sector."""

    sector: str
    corrector: str
    bpms: tuple[str, ...]
    model_slopes: tuple[float, ...]
    readings_um: tuple[tuple[float, ...], ...]
    reversed_bpms: tuple[str, ...]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sector_layout(ring) -> dict[str, list[tuple[str, int]]]:
    """Sector marker name -> ordered ``(element name, ring index)`` of its BPMs and HCMs."""
    layout: dict[str, list[tuple[str, int]]] = {}
    sector = ""
    for index, element in enumerate(ring):
        name = element.FamName
        if name.startswith("SECT"):
            sector = name
        elif name.startswith(("BPM", "HCM")):
            layout.setdefault(sector, []).append((name, index))
    return layout


def bump_test(simulation_dir: Path) -> BumpTest:
    """The bpm-polarity scenario's reversed BPM, seen in a one-corrector bump test.

    The sector is the one holding the reversed BPM, and the corrector is the
    horizontal corrector immediately upstream of it. Model slopes are the ring's
    closed-orbit response (micrometres per microradian); readings apply each
    BPM's scenario polarity to the model and add the machine file's BPM noise.
    """
    from osprey.simulation.lattice.ring import build_ring

    spec = _read_json(simulation_dir / "scenarios/bpm-polarity/scenario.json")
    polarity = {
        name: int(error.get("polarity", 1)) for name, error in spec["physics"]["bpm_errors"].items()
    }
    reversed_bpms = tuple(sorted(name for name, sign in polarity.items() if sign == -1))
    if not reversed_bpms:
        raise ValueError("bpm-polarity scenario reverses no BPM")
    target = reversed_bpms[0]

    ring = build_ring()
    layout = _sector_layout(ring)
    sector = next(s for s, members in layout.items() if any(n == target for n, _ in members))
    members = layout[sector]
    names = [n for n, _ in members]
    corrector = next(n for n in reversed(names[: names.index(target)]) if n.startswith("HCM"))
    corrector_index = dict(members)[corrector]
    bpms = [(n, i) for n, i in members if n.startswith("BPM")]

    kick = 1e-5
    _, base = ring.find_orbit(refpts=[i for _, i in bpms])
    kicked = ring.deepcopy()
    kicked[corrector_index].KickAngle = [kick, 0.0]
    _, orbit = kicked.find_orbit(refpts=[i for _, i in bpms])
    slopes = (orbit[:, 0] - base[:, 0]) / kick  # m/rad == um/urad

    channels = _read_json(simulation_dir / "machine.json")["channels"]
    rng = np.random.default_rng(BUMP_NOISE_SEED)
    readings = []
    for (name, _), slope in zip(bpms, slopes, strict=True):
        number = name.removeprefix("BPM")
        noise_um = float(channels[f"SR:DIAG:BPM:{number}:POSITION:X"].get("noise_abs", 0.0)) * 1e6
        sign = polarity.get(name, 1)
        values = sign * slope * KICKS_URAD + rng.normal(0.0, noise_um, KICKS_URAD.size)
        readings.append(tuple(float(v) for v in values))

    return BumpTest(
        sector=sector,
        corrector=corrector,
        bpms=tuple(n for n, _ in bpms),
        model_slopes=tuple(float(s) for s in slopes),
        readings_um=tuple(readings),
        reversed_bpms=reversed_bpms,
    )


def _synthesize(simulation_dir: Path, scenario: str, channels: list[str], times: list[datetime]):
    """Synthesize ``channels`` at ``times`` with ``scenario`` active.

    The engine runs on a copy of the simulation tree without its narratives:
    loading a logbook checks the plot specs it names, which are what this
    script is about to write, and telemetry never depends on a narrative.
    """
    from osprey.simulation.engine import SimulationEngine

    with tempfile.TemporaryDirectory() as scratch:
        telemetry = Path(scratch) / "simulation"
        shutil.copytree(
            simulation_dir, telemetry, ignore=shutil.ignore_patterns("logbook.json", "plots")
        )
        engine = SimulationEngine.from_file(
            telemetry / "machine.json", state_dir=Path(scratch) / "state"
        )
        engine.set_active_scenarios([scenario], anchor=ANCHOR)
        return {pv: np.asarray(engine.synthesize_series(pv, times)) for pv in channels}


def entry_time(simulation_dir: Path, scenario: str, spec: str) -> datetime:
    """When the entry attaching plot spec ``spec`` was written, resolved against :data:`ANCHOR`.

    The entry is found by the spec its ``attachments`` names, so a spec moved
    to another entry moves its window with it.
    """
    from osprey_connectors.relative_time import RelativeTimestamp, resolve_relative_timestamp

    entries = _read_json(simulation_dir / "scenarios" / scenario / "logbook.json")
    for entry in entries:
        if any(item.get("plot") == spec for item in entry.get("attachments", [])):
            when = entry["when"]
            relative = RelativeTimestamp(
                days_ago=int(when["days_ago"]), time=dtime.fromisoformat(when["time"])
            )
            return resolve_relative_timestamp(relative, ANCHOR)
    raise ValueError(f"no {scenario} logbook entry attaches {spec}")


def _week_before(end: datetime, step: timedelta) -> list[datetime]:
    count = int(WINDOW / step)
    return [end - WINDOW + step * i for i in range(count + 1)]


def event_instant(event: dict) -> datetime:
    """Where one anchored event of a scenario script sits, by the engine's own rule."""
    from osprey.simulation.series import anchored_instant

    return datetime.fromtimestamp(anchored_instant(event, ANCHOR.timestamp(), UTC), UTC)


def _plot_spec(
    filename: str,
    title: str,
    ylabel: str,
    end: datetime,
    times: list[datetime],
    series: dict[str, np.ndarray],
    ylim: tuple[float, float] | None = None,
) -> dict:
    """A plot spec of ``series`` on ``times``, its axis counted back from ``end``."""
    hours = [round((end - t).total_seconds() / 3600.0, _HOURS_DECIMALS) + 0.0 for t in times]
    spec: dict = {
        "filename": filename,
        "title": title,
        "ylabel": ylabel,
        "hours_before": hours,
        "series": [
            {"label": label, "values": [round(float(v), _VALUE_DECIMALS) for v in values]}
            for label, values in series.items()
        ],
    }
    if ylim is not None:
        spec["ylim"] = list(ylim)
    return spec


def cavity_temperatures_spec(simulation_dir: Path, end: datetime) -> dict:
    """Both cavity temperatures, every 10 minutes over the week up to ``end``, under rf-thermal."""
    times = _week_before(end, timedelta(minutes=10))
    synthesized = _synthesize(simulation_dir, "rf-thermal", list(CAVITY_TEMPERATURES), times)
    return _plot_spec(
        "cavity_temperatures.png",
        "CAVITY01 / CAVITY02 body temperature",
        "°C",
        end,
        times,
        {pv.split(":")[2] + pv.split(":")[3]: synthesized[pv] for pv in CAVITY_TEMPERATURES},
    )


def orbit_rms_spec(simulation_dir: Path, end: datetime) -> dict:
    """RMS over all ring BPMs, per plane, hourly across the week up to ``end`` (µm)."""
    times = _week_before(end, timedelta(hours=1))
    channels = _read_json(simulation_dir / "machine.json")["channels"]
    rms: dict[str, np.ndarray] = {}
    for plane in ("X", "Y"):
        pvs = sorted(
            pv
            for pv in channels
            if pv.startswith("SR:DIAG:BPM:") and pv.endswith(f":POSITION:{plane}")
        )
        series = _synthesize(simulation_dir, "nominal", pvs, times)
        stacked = np.vstack([series[pv] for pv in pvs]) * 1e6
        rms[plane] = np.sqrt(np.mean(stacked**2, axis=0))
    return _plot_spec("orbit_rms.png", "SR orbit RMS", "µm", end, times, rms, ylim=(0.0, 20.0))


def draw_bump_test(test: BumpTest, path: Path) -> None:
    """One panel per sector BPM, all drawn alike: model slope dashed, readings as dots."""
    from matplotlib import style

    from osprey.simulation.plots import DPI, FIGSIZE, figure_png

    with style.context("default"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=FIGSIZE, dpi=DPI)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3, sharex=True)
        for ax, bpm, slope, readings in zip(
            axes.flat, test.bpms, test.model_slopes, test.readings_um, strict=True
        ):
            ax.plot(KICKS_URAD, slope * KICKS_URAD, "--", color="0.45", lw=1, label="model")
            ax.plot(KICKS_URAD, readings, "o", ms=4, color="C0", label="measured")
            ax.set_title(bpm, fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        for ax in axes[1]:
            ax.set_xlabel(f"{test.corrector} kick (µrad)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("X (µm)", fontsize=8)
        axes.flat[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{test.corrector} bump test, horizontal", fontsize=10)
        fig.tight_layout()
        data = figure_png(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_spec(spec: dict, path: Path) -> None:
    """Write ``spec`` as JSON: one key per line, each array on one line."""
    from osprey.simulation.machine import parse_plot_spec

    parse_plot_spec(spec, str(path))
    lines = []
    for key, value in spec.items():
        if key == "series":
            items = ",\n".join(f"    {json.dumps(s, ensure_ascii=False)}" for s in value)
            lines.append(f'  "series": [\n{items}\n  ]')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")


def write_rf_thermal(simulation_dir: Path, path: Path) -> None:
    end = entry_time(simulation_dir, "rf-thermal", RF_THERMAL_SPEC)
    write_spec(cavity_temperatures_spec(simulation_dir, end), path)


def write_nominal(simulation_dir: Path, path: Path) -> None:
    end = entry_time(simulation_dir, "nominal", NOMINAL_SPEC)
    write_spec(orbit_rms_spec(simulation_dir, end), path)


def main(argv: list[str]) -> int:
    simulation_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_SIMULATION_DIR
    scenarios = simulation_dir / "scenarios"
    outputs = {
        scenarios / "bpm-polarity" / BPM_POLARITY_PLOT: lambda p: draw_bump_test(
            bump_test(simulation_dir), p
        ),
        scenarios / "rf-thermal" / RF_THERMAL_SPEC: lambda p: write_rf_thermal(simulation_dir, p),
        scenarios / "nominal" / NOMINAL_SPEC: lambda p: write_nominal(simulation_dir, p),
    }
    for path, write in outputs.items():
        write(path)
        print(f"wrote {path.relative_to(simulation_dir)} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
