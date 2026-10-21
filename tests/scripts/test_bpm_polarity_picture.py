"""The bpm-polarity logbook picture says what the example facility's model says.

``scripts/bpm_polarity_picture.py`` draws the picture the example facility's
bpm-polarity scenario attaches to its logbook entry, from the simulator view of
that facility. These tests pin the data it carries to the scenario and the
model wiring it is derived from, pin that nothing on it names a finding, and pin
that a re-run writes the same bytes and the committed picture is the same
picture.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from osprey_connectors.simulation.view import VIEW_RELPATH, SimulatorView

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "bpm_polarity_picture", REPO_ROOT / "scripts" / "bpm_polarity_picture.py"
)
assert _SPEC is not None and _SPEC.loader is not None
plots = importlib.util.module_from_spec(_SPEC)
# import-time required because scripts/ is not a package: bpm_polarity_picture.py is
# loaded by path and registered in sys.modules before exec so its dataclasses
# can resolve cls.__module__.
sys.modules[_SPEC.name] = plots
_SPEC.loader.exec_module(plots)

EXAMPLE = plots.EXAMPLE_FACILITY
COMMITTED_PICTURE = EXAMPLE / "scenarios" / plots.SCENARIO / plots.PICTURE


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> SimulatorView:
    return SimulatorView.open(
        plots.render_simulator_view(
            plots.EXAMPLE_FACILITY, tmp_path_factory.mktemp("render") / VIEW_RELPATH
        )
    )


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scenario_reversed_horizontal_elements() -> set[str]:
    """The deck elements of the scenario's horizontal ``polarity: -1`` addresses."""
    scenario = _read_yaml(EXAMPLE / "scenarios" / f"{plots.SCENARIO}.yaml")
    models = {model["name"]: model for model in _read_yaml(EXAMPLE / "models.yaml")}
    elements = set()
    for model, addresses in scenario["faults"].items():
        wiring = {record["address"]: record for record in models[model]["wiring"]}
        for address, fault in addresses.items():
            record = wiring[address]
            if fault.get("polarity") == -1 and record.get("engine", {}).get("axis") == "x":
                elements.add(record["element"])
    return elements


def test_the_bump_test_flips_exactly_the_scenarios_reversed_monitors(view):
    test = plots.bump_test(view)
    assert set(test.reversed_bpms) == _scenario_reversed_horizontal_elements()
    assert set(test.reversed_bpms) <= set(test.bpms)
    kicks = plots.KICKS_URAD
    for bpm, model, readings in zip(test.bpms, test.model_slopes, test.readings_um, strict=True):
        measured = float(np.polyfit(kicks, readings, 1)[0])
        expected = -model if bpm in test.reversed_bpms else model
        assert measured == pytest.approx(expected, abs=0.1), bpm
        assert abs(model) > 0.5, f"{bpm} barely responds; the picture would show nothing"


def test_the_bump_test_draws_every_monitor_alike(view, monkeypatch, tmp_path):
    """Only the data tells the reversed BPM apart: same colour, title = BPM name."""
    from matplotlib.figure import Figure

    drawn: list[Figure] = []
    monkeypatch.setattr("osprey.simulation.plots.figure_png", lambda fig: drawn.append(fig) or b"")
    test = plots.bump_test(view)
    plots.draw_bump_test(test, tmp_path / "bump.png")

    (fig,) = drawn
    assert fig._suptitle.get_text() == f"{test.corrector} bump test, horizontal"
    axes = fig.axes
    assert [ax.get_title() for ax in axes] == list(test.bpms)
    styles = {
        tuple((line.get_color(), line.get_linestyle(), line.get_marker()) for line in ax.lines)
        for ax in axes
    }
    assert len(styles) == 1, "one BPM is drawn differently from the others"
    assert all(len(ax.texts) == 0 for ax in axes)


def test_the_scenario_attaches_the_picture_this_script_draws():
    scenario = _read_yaml(EXAMPLE / "scenarios" / f"{plots.SCENARIO}.yaml")
    (entry,) = [e for e in scenario["logbook"] if e["entry_id"] == "DEMO-031"]
    assert entry["attachments"] == [{"path": plots.PICTURE}]
    assert COMMITTED_PICTURE.is_file()


#: Largest RMS colour difference (on the 0-255 scale) between a rendered
#: picture and the committed one that still counts as the same picture. Two
#: hosts antialias an edge a few counts apart (measured 0.013 across CI
#: runners); any change to the data moves whole lines and measures tens.
_SAME_PICTURE_RMS = 1.0


def _colours(png: bytes) -> tuple[str, tuple[int, int], np.ndarray]:
    """A PNG's mode, size and RGB pixels: what it shows, not how it was encoded.

    The colours are read as RGB: a palette PNG stores indices into a palette
    whose order the encoder chooses, so two pictures of the same colours can
    carry different indices.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        return image.mode, image.size, np.asarray(image.convert("RGB"), dtype=float)


def _assert_same_picture(rendered: bytes, committed: bytes) -> None:
    mode, size, pixels = _colours(rendered)
    committed_mode, committed_size, committed_pixels = _colours(committed)
    assert (mode, size) == (committed_mode, committed_size)
    rms = float(np.sqrt(np.mean((pixels - committed_pixels) ** 2)))
    assert rms < _SAME_PICTURE_RMS, (
        f"rendered picture differs from the committed one: RMS {rms:.3f}"
    )


def _copy_example(tmp_path: Path) -> Path:
    copy = tmp_path / "example"
    shutil.copytree(EXAMPLE, copy)
    return copy


def test_a_rerun_writes_the_same_bytes_and_the_committed_picture_is_the_same_picture(tmp_path):
    copy = _copy_example(tmp_path)
    picture = copy / "scenarios" / plots.SCENARIO / plots.PICTURE
    assert plots.main([str(copy)]) == 0
    first = picture.read_bytes()
    assert plots.main([str(copy)]) == 0
    assert picture.read_bytes() == first
    # The facility holds what the script writes, as the same picture. The PNG
    # compressor's output and the last bit of an antialiased edge differ
    # between hosts, so the committed picture is compared decoded and within a
    # tolerance no data change stays under.
    _assert_same_picture(first, COMMITTED_PICTURE.read_bytes())


def test_the_script_reads_a_built_simulator_view_as_it_reads_its_own_render(view, tmp_path):
    copy = _copy_example(tmp_path)
    picture = copy / "scenarios" / plots.SCENARIO / plots.PICTURE
    assert plots.main([str(copy), "--simulator", str(view.path)]) == 0
    from_view = picture.read_bytes()
    assert plots.main([str(copy)]) == 0
    assert picture.read_bytes() == from_view
