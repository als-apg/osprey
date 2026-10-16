#!/usr/bin/env python3
"""Draw the picture the example facility's bpm-polarity scenario attaches to its logbook entry.

Usage: ``python scripts/bpm_polarity_picture.py [FACILITY_DIR] [--simulator VIEW_DIR]``

``FACILITY_DIR`` defaults to the example facility shipped in the templates.
Without ``--simulator`` the script renders that facility's simulator view
in-process; with it, it reads an existing one, such as the
``build/data/simulator`` an ``osprey build`` writes.

Inputs, all read through the simulator view's reader:

* ``scenarios.json`` -- the scenario's polarity faults;
* ``variables.json`` -- the faulted model's deck, settings and wiring;
* ``seeds.json`` -- each monitor's absolute noise;
* ``decks/<model>.json`` -- the lattice the model solves.

Output: ``<FACILITY_DIR>/scenarios/bpm-polarity/plots/corrector_bump_test.png``,
each monitor's reading against a kick of the corrector upstream of the reversed
monitor, over the monitors of the lattice segment holding it. Model slopes come
from the model's closed-orbit response through its wiring and calibration; the
readings apply the scenario's polarity and add each monitor's noise.

The picture states only what an operator's export of a bump test would: device
names, units and the data. Nothing on it points at the finding.

The output is deterministic: the noise comes from a fixed seed and the PNG is
encoded through :func:`osprey.simulation.plots.figure_png` without text chunks,
so a re-run on the same library versions rewrites identical bytes on one host
and the same picture on any (the PNG compressor and the last bit of an
antialiased edge differ between hosts).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from osprey_connectors.simulation.view import VIEW_RELPATH, Binding, SimulatorView

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FACILITY = REPO_ROOT / "src/osprey/templates/facilities/example"

SCENARIO = "bpm-polarity"
PICTURE = "plots/corrector_bump_test.png"

#: Corrector kicks of the bump test, in microradians.
KICKS_URAD = np.linspace(-40.0, 40.0, 9)

#: Seed of the monitor read noise in the bump test.
BUMP_NOISE_SEED = 17

#: The corrector step the response is measured with, in radians.
STEP_RAD = 1e-5


@dataclass(frozen=True)
class BumpTest:
    """One corrector's bump test over the monitors of a lattice segment.

    Names are deck element names.
    """

    corrector: str
    bpms: tuple[str, ...]
    model_slopes: tuple[float, ...]
    readings_um: tuple[tuple[float, ...], ...]
    reversed_bpms: tuple[str, ...]


def render_simulator_view(facility_dir: Path, root: Path) -> Path:
    """Write ``facility_dir``'s simulator view to ``root`` and return ``root``."""
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    doc = build_facility(facility_dir, project_name="example")
    config = {"control_system": {"type": "virtual_accelerator"}}
    write_simulator_view(
        root,
        ViewInputs(
            doc=doc,
            rendered_config=config,
            facility_dir=facility_dir,
            served=resolve_served(config, doc),
            reported=None,
        ),
    )
    return root


def _element_index(lattice: Any, name: str) -> int:
    """The lattice index of the one element named ``name``."""
    found = [i for i, element in enumerate(lattice) if element.FamName == name]
    if len(found) != 1:
        raise ValueError(f"the deck holds element {name} {len(found)} times, not once")
    return found[0]


def bump_test(view: SimulatorView, scenario: str = SCENARIO) -> BumpTest:
    """The scenario's reversed horizontal monitor, seen in a one-corrector bump test.

    The segment is the stretch of lattice between the plain markers around the
    reversed monitor, and the corrector is the horizontal kick corrector
    immediately upstream of it. Model slopes are the model's closed-orbit
    response (micrometres per microradian); readings apply each monitor's
    scenario polarity to the model and add the monitor's noise.
    """
    import at

    from osprey.simulation.engines.calibration import curve_from_record, to_physics
    from osprey.simulation.engines.pyat import response_matrix

    scenarios = view.scenarios()
    (entry,) = [s for s in scenarios if s["name"] == scenario]
    reversed_by_model = {
        model: {
            address: -1
            for address, fault in (faults.get("writes") or {}).items()
            if fault.get("polarity") == -1
        }
        for model, faults in (entry.get("faults") or {}).items()
    }
    reversed_by_model = {model: found for model, found in reversed_by_model.items() if found}
    if len(reversed_by_model) != 1:
        raise ValueError(
            f"{scenario} reverses monitors of {len(reversed_by_model)} models, not exactly one"
        )
    ((model_name, polarity),) = reversed_by_model.items()

    model = view.model(model_name)
    if model.deck is None:
        raise ValueError(f"{model_name} names no deck")
    deck = model.deck
    settings = model.settings
    wiring = [binding.record for binding in model.bindings]
    by_address = {binding.address: binding for binding in model.bindings}

    lattice = at.load_lattice(str(deck))
    horizontal = [
        a
        for a in polarity
        if by_address[a].role == "monitor"
        and by_address[a].plane == "x"
        and by_address[a].element is not None
    ]
    if not horizontal:
        raise ValueError(f"{scenario} reverses no horizontal monitor")
    horizontal.sort(key=lambda a: _element_index(lattice, by_address[a].element))
    reversed_bpms = tuple(by_address[a].element for a in horizontal)
    target = _element_index(lattice, reversed_bpms[0])

    # The plain markers bound the segment; a monitor is not one of them.
    markers = [i for i, element in enumerate(lattice) if type(element) is at.Marker]
    start = max((i for i in markers if i < target), default=-1)
    end = min((i for i in markers if i > target), default=len(lattice))

    monitors: list[tuple[int, Binding]] = []
    correctors: list[tuple[int, Binding]] = []
    for binding in model.bindings:
        if binding.element is None:
            continue
        is_monitor = binding.role == "monitor" and binding.plane == "x"
        is_corrector = binding.role == "setpoint" and binding.plane == "x"
        if not (is_monitor or is_corrector):
            continue
        index = _element_index(lattice, binding.element)
        if not start < index < end:
            continue
        if is_monitor:
            monitors.append((index, binding))
        elif index < target:
            correctors.append((index, binding))
    if not monitors:
        raise ValueError(f"no horizontal monitor is wired between {start} and {end}")
    if not correctors:
        raise ValueError(f"no kick corrector is wired upstream of {reversed_bpms[0]}")
    monitors.sort(key=lambda item: item[0])
    _, corrector = max(correctors, key=lambda item: item[0])

    curve = curve_from_record((corrector.record.get("calibration") or {}).get("curve"))
    rad_per_unit = to_physics(curve, 0.5) - to_physics(curve, -0.5)
    if rad_per_unit == 0:
        raise ValueError(f"{corrector.address} has a calibration with no slope")
    address = corrector.address
    result = response_matrix(deck, wiring, settings, [address], {address: STEP_RAD / rad_per_unit})
    if result.unsolved:
        raise ValueError(f"{address} has no solve: {result.unsolved[address]}")
    rows, matrix = result.rows, result.matrix

    rng = np.random.default_rng(BUMP_NOISE_SEED)
    slopes = []
    readings = []
    for _, record in monitors:
        monitor = record.address
        unit = view.channel(monitor).unit
        if unit != "m":
            raise ValueError(f"{monitor} reads in {unit!r}, not 'm'")
        slope = float(matrix[rows.index(monitor), 0] / rad_per_unit)  # m/rad == um/urad
        noise = (view.seed(monitor) or {}).get("noise") or {}
        if "relative" in noise:
            raise ValueError(f"{monitor} carries relative noise; the picture needs absolute")
        noise_um = float(noise.get("absolute", 0.0)) * 1e6
        sign = polarity.get(monitor, 1)
        values = sign * slope * KICKS_URAD + rng.normal(0.0, noise_um, KICKS_URAD.size)
        slopes.append(slope)
        readings.append(tuple(float(v) for v in values))

    return BumpTest(
        corrector=corrector.element,
        bpms=tuple(binding.element for _, binding in monitors),
        model_slopes=tuple(slopes),
        readings_um=tuple(readings),
        reversed_bpms=reversed_bpms,
    )


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("facility_dir", nargs="?", type=Path, default=EXAMPLE_FACILITY)
    parser.add_argument("--simulator", type=Path, default=None, metavar="VIEW_DIR")
    args = parser.parse_args(argv)
    facility_dir: Path = args.facility_dir
    path = facility_dir / "scenarios" / SCENARIO / PICTURE
    if args.simulator is not None:
        draw_bump_test(bump_test(SimulatorView.open(args.simulator)), path)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            view_dir = render_simulator_view(facility_dir, Path(tmp) / VIEW_RELPATH)
            draw_bump_test(bump_test(SimulatorView.open(view_dir)), path)
    print(f"wrote {path.relative_to(facility_dir)} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
