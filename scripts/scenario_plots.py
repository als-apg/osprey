#!/usr/bin/env python3
"""Draw the pictures the control-assistant demo scenarios attach to their logbook entries.

Usage: ``python scripts/scenario_plots.py [SIMULATION_DIR]``

``SIMULATION_DIR`` defaults to the control-assistant template's
``data/simulation``. Each picture is written to ``<scenario>/plots/`` inside it
and is named by an entry's ``attachments`` in that scenario's ``logbook.json``.

Every value drawn is derived from the scenario's own definition, never typed in:

* ``bpm-polarity`` -- each sector BPM's reading against a kick of the corrector
  upstream of the reversed BPM, from the demo ring's closed orbit, with the
  scenario's BPM polarity applied to the readings and the machine file's BPM
  noise added.
* ``rf-thermal`` -- the two cavity temperatures over a week, synthesized by
  the simulation engine from the scenario's archiver events.
* ``nominal`` -- the RMS of all ring BPM readings over a week, synthesized by
  the simulation engine from the machine file's BPM texture and noise.

A time series ends at the time of the entry that attaches it, resolved from
that entry's ``when`` against the same anchor the telemetry is drawn against, so
the picture shows what its author could have seen when writing the entry.

The output is deterministic: draws come from fixed seeds and keyed series, the
time axis is relative, and the PNGs are re-encoded without text chunks, so a
re-run on the same library versions rewrites identical bytes.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATION_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"

#: The scenario start every time series is drawn against. Any instant gives the
#: same event shapes; a fixed one keeps the keyed noise and texture reproducible.
ANCHOR = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

#: Span of every time-series picture, ending at its entry's time.
WINDOW = timedelta(days=7)

#: Corrector kicks of the bump test, in microradians.
KICKS_URAD = np.linspace(-40.0, 40.0, 9)

#: Seed of the BPM read noise in the bump test.
BUMP_NOISE_SEED = 17

#: Pixel size of every picture (figure inches at 100 dpi).
FIGSIZE = (8.0, 4.5)
DPI = 100

CAVITY01_TEMPERATURE = "SR:RF:CAVITY:01:TEMPERATURE:RB"

BPM_POLARITY_PLOT = "plots/corrector_bump_test.png"
RF_THERMAL_PLOT = "plots/cavity_temperatures_week.png"
NOMINAL_PLOT = "plots/orbit_rms_week.png"


@dataclass(frozen=True)
class BumpTest:
    """One corrector's bump test over the BPMs of a sector."""

    sector: str
    corrector: str
    bpms: tuple[str, ...]
    model_slopes: tuple[float, ...]
    readings_um: tuple[tuple[float, ...], ...]
    reversed_bpms: tuple[str, ...]


@dataclass(frozen=True)
class Trend:
    """Named series on one shared time axis: hours before ``end``."""

    end: datetime
    hours: tuple[float, ...]
    series: dict[str, tuple[float, ...]]


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


def _engine(simulation_dir: Path, scenario: str, state_dir: Path):
    from osprey.simulation.engine import SimulationEngine

    engine = SimulationEngine.from_file(simulation_dir / "machine.json", state_dir=state_dir)
    engine.set_active_scenarios([scenario], anchor=ANCHOR)
    return engine


def _synthesize(simulation_dir: Path, scenario: str, channels: list[str], times: list[datetime]):
    with tempfile.TemporaryDirectory() as state_dir:
        engine = _engine(simulation_dir, scenario, Path(state_dir))
        return {pv: np.asarray(engine.synthesize_series(pv, times)) for pv in channels}


def entry_time(simulation_dir: Path, scenario: str, plot: str) -> datetime:
    """When the entry attaching ``plot`` was written, resolved against :data:`ANCHOR`.

    The entry is found by the picture its ``attachments`` names, so a picture
    moved to another entry moves its window with it.
    """
    from osprey_connectors.relative_time import RelativeTimestamp, resolve_relative_timestamp

    entries = _read_json(simulation_dir / "scenarios" / scenario / "logbook.json")
    for entry in entries:
        if any(item.get("path") == plot for item in entry.get("attachments", [])):
            when = entry["when"]
            spec = RelativeTimestamp(
                days_ago=int(when["days_ago"]), time=dtime.fromisoformat(when["time"])
            )
            return resolve_relative_timestamp(spec, ANCHOR)
    raise ValueError(f"no {scenario} logbook entry attaches {plot}")


def _week_before(end: datetime, step: timedelta) -> list[datetime]:
    count = int(WINDOW / step)
    return [end - WINDOW + step * i for i in range(count + 1)]


def _trend(end: datetime, times: list[datetime], series: dict[str, np.ndarray]) -> Trend:
    return Trend(
        end=end,
        hours=tuple((t - end).total_seconds() / 3600.0 for t in times),
        series={name: tuple(float(v) for v in values) for name, values in series.items()},
    )


def event_instant(event: dict) -> datetime:
    """Where one anchored event of a scenario script sits."""
    return ANCHOR + timedelta(seconds=float(event["at_offset"]))


def excursion_peaks(simulation_dir: Path, end: datetime) -> list[tuple[float, float]]:
    """``(hours before end, peak temperature)`` of each CAVITY01 excursion in the window.

    Read from the rf-thermal scenario's spike events on the channel: the
    machine file's baseline plus the spike amplitude, at the spike's instant.
    """
    baseline = float(
        _read_json(simulation_dir / "machine.json")["channels"][CAVITY01_TEMPERATURE]["value"]
    )
    spec = _read_json(simulation_dir / "scenarios/rf-thermal/scenario.json")
    events = next(a["events"] for a in spec["archiver"] if a["channel"] == CAVITY01_TEMPERATURE)
    peaks = []
    for event in events:
        if event["shape"] != "spike":
            continue
        at = event_instant(event)
        if end - WINDOW <= at <= end:
            peaks.append(
                ((at - end).total_seconds() / 3600.0, baseline + float(event["amplitude"]))
            )
    return peaks


def cavity_temperatures(simulation_dir: Path, end: datetime) -> Trend:
    """Both cavity temperatures over the week up to ``end``, under rf-thermal."""
    times = _week_before(end, timedelta(minutes=10))
    channels = [CAVITY01_TEMPERATURE, "SR:RF:CAVITY:02:TEMPERATURE:RB"]
    return _trend(end, times, _synthesize(simulation_dir, "rf-thermal", channels, times))


def orbit_rms_week(simulation_dir: Path, end: datetime) -> Trend:
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
    return _trend(end, times, rms)


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    return plt


def _save(fig, path: Path) -> None:
    """Write ``fig`` as a palette PNG with no text chunks."""
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, metadata={"Software": None})
    buffer.seek(0)
    with Image.open(buffer) as image:
        flat = image.convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = io.BytesIO()
    flat.save(out, format="PNG", optimize=True)
    path.write_bytes(out.getvalue())


def draw_bump_test(test: BumpTest, path: Path) -> None:
    plt = _figure()
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True)
    for ax, bpm, slope, readings in zip(
        axes.flat, test.bpms, test.model_slopes, test.readings_um, strict=True
    ):
        flipped = bpm in test.reversed_bpms
        ax.plot(KICKS_URAD, slope * KICKS_URAD, "--", color="0.45", lw=1, label="model")
        ax.plot(
            KICKS_URAD,
            readings,
            "o",
            ms=4,
            color="tab:red" if flipped else "tab:blue",
            label="measured",
        )
        title = f"{bpm}: moves opposite to model" if flipped else bpm
        ax.set_title(title, fontsize=9, color="tab:red" if flipped else "black")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel(f"{test.corrector} kick (µrad)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Horizontal reading (µm)", fontsize=8)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    sector = test.sector.removeprefix("SECT")
    fig.suptitle(
        f"Corrector bump test, sector {sector}: BPM readings vs {test.corrector} kick",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def draw_cavity_temperatures(trend: Trend, peaks: list[tuple[float, float]], path: Path) -> None:
    plt = _figure()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    days = np.asarray(trend.hours) / 24.0
    colors = {"01": "tab:red", "02": "tab:blue"}
    for pv, values in trend.series.items():
        number = pv.split(":")[3]
        ax.plot(days, values, color=colors.get(number, "black"), lw=1.0, label=f"CAVITY{number}")
    cavity1 = np.asarray(trend.series[CAVITY01_TEMPERATURE])
    for at, peak in peaks:
        ax.annotate(
            f"peak {peak:.1f} °C",
            (at / 24.0, peak),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Days before this entry")
    ax.set_ylabel("Cavity body temperature (°C)")
    ax.set_title("RF cavity body temperatures, the week before this entry")
    ax.set_ylim(24.0, float(cavity1.max()) + 2.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def draw_orbit_rms(trend: Trend, path: Path) -> None:
    plt = _figure()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    days = np.asarray(trend.hours) / 24.0
    for plane, color, label in (("X", "tab:blue", "horizontal"), ("Y", "tab:green", "vertical")):
        values = np.asarray(trend.series[plane])
        ax.plot(days, values, color=color, lw=1.2, label=f"{label} (max {values.max():.1f} µm)")
    ax.set_xlabel("Days before this entry")
    ax.set_ylabel("Orbit RMS over all 72 ring BPMs (µm)")
    ax.set_title("Slow orbit drift, the week before this entry (hourly)")
    ax.set_ylim(0.0, 20.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def draw_rf_thermal(simulation_dir: Path, path: Path) -> None:
    end = entry_time(simulation_dir, "rf-thermal", RF_THERMAL_PLOT)
    trend = cavity_temperatures(simulation_dir, end)
    draw_cavity_temperatures(trend, excursion_peaks(simulation_dir, end), path)


def draw_nominal(simulation_dir: Path, path: Path) -> None:
    end = entry_time(simulation_dir, "nominal", NOMINAL_PLOT)
    draw_orbit_rms(orbit_rms_week(simulation_dir, end), path)


def main(argv: list[str]) -> int:
    simulation_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_SIMULATION_DIR
    scenarios = simulation_dir / "scenarios"
    outputs = {
        scenarios / "bpm-polarity" / BPM_POLARITY_PLOT: lambda p: draw_bump_test(
            bump_test(simulation_dir), p
        ),
        scenarios / "rf-thermal" / RF_THERMAL_PLOT: lambda p: draw_rf_thermal(simulation_dir, p),
        scenarios / "nominal" / NOMINAL_PLOT: lambda p: draw_nominal(simulation_dir, p),
    }
    for path, draw in outputs.items():
        draw(path)
        print(f"wrote {path.relative_to(simulation_dir)} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
