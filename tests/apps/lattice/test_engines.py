"""The two engines the application carries: pyAT on a real ring, elegant on stand-in binaries."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import lattice_app
import numpy as np
import pytest
from lattice_app.catalog import ANALYSES, JOB_SCHEMA, normalize_result
from lattice_app.engines import elegant, run_engine, tune_vs_dp

DECK = Path(__file__).resolve().parents[2] / "fixtures/mml/spear3/spear3.storagering.lattice.mat"
#: Settings small enough for a test; the catalog's defaults are a physicist's.
QUICK = {
    "chromaticity": {"dp_min_pct": -1.0, "dp_max_pct": 1.0, "n_steps": 5},
    "da": {"nturns": 16, "n_angles": 3, "amp_max_mm": 20.0, "n_bisect": 4},
    "footprint": {"n_amp": 2, "x_max_mm": 2.0, "y_max_mm": 1.0, "n_half": 32},
    "lma": {"nturns": 16, "n_refpts": 2, "dp_max_pct": 3.0, "n_bisect": 4},
}


def _job(analysis: str, **over: Any) -> dict[str, Any]:
    return {
        "schema": JOB_SCHEMA,
        "analysis": analysis,
        "model": "ring",
        "deck": str(DECK),
        "energy_gev": None,
        "settings": {**ANALYSES[analysis].settings, **QUICK.get(analysis, {})},
        "engine": "pyat",
        "engine_settings": {},
        **over,
    }


def _pyat() -> Any:
    pytest.importorskip("at")
    from lattice_app.engines import pyat

    return pyat


def test_pyat_optics_are_the_rings() -> None:
    result = normalize_result(ANALYSES["optics"], _pyat().ANALYSES["optics"](_job("optics")))

    assert result["scalars"]["tune_x"] == pytest.approx(14.13, abs=0.01)
    assert result["scalars"]["tune_y"] == pytest.approx(6.22, abs=0.01)
    twiss = result["tables"]["twiss"]["columns"]
    assert len(twiss["s"]) == len(twiss["beta_x"]) > 100
    assert min(twiss["beta_x"]) > 0


def test_the_pass_method_and_the_steps_are_the_facilitys_to_set() -> None:
    pyat = _pyat()
    default = pyat.ANALYSES["optics"](_job("optics"))["scalars"]
    settings = {"bend_pass": "ExactSectorBendPass", "dp_step": 1.0e-5}
    exact = pyat.ANALYSES["optics"](_job("optics", engine_settings=settings))["scalars"]

    assert pyat.steps({"engine_settings": settings}) == {"DPStep": 1.0e-5}
    assert exact["tune_x"] == pytest.approx(default["tune_x"], abs=0.05)
    # The exact bend moves the chromaticity: the difference the facility asked to control.
    assert abs(exact["chromaticity_y"] - default["chromaticity_y"]) > 0.1


@pytest.mark.parametrize("analysis", ["chromaticity", "da", "footprint", "lma"])
def test_pyat_answers_every_analysis_in_the_panels_shape(analysis: str) -> None:
    result = normalize_result(ANALYSES[analysis], _pyat().ANALYSES[analysis](_job(analysis)))

    for table, columns in ANALYSES[analysis].tables.items():
        assert set(columns) <= set(result["tables"][table]["columns"])


def test_the_chromatic_orders_are_derivatives() -> None:
    offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]
    # nu = nu0 + xi1 d + xi2 d^2 / 2 + xi3 d^3 / 6, with d the fractional offset.
    curve = [
        0.2 + 3.0 * d + 40.0 * d**2 / 2 + 6000.0 * d**3 / 6 for d in (o / 100 for o in offsets)
    ]

    result = tune_vs_dp(offsets, curve, [0.3] * 5)

    assert result["scalars"]["chromaticity_x"] == pytest.approx(3.0)
    assert result["scalars"]["chromaticity2_x"] == pytest.approx(40.0)
    assert result["scalars"]["chromaticity3_x"] == pytest.approx(6000.0)
    assert result["scalars"]["chromaticity_y"] == pytest.approx(0.0, abs=1e-9)
    assert result["tables"]["tune_vs_dp"]["columns"]["tune_x"] == curve


def test_an_offset_with_no_stable_optics_is_a_gap_and_too_few_points_fit_nothing() -> None:
    nan = float("nan")

    gap = tune_vs_dp([-2.0, -1.0, 0.0, 1.0, 2.0], [0.1, 0.2, 0.3, 0.4, nan], [0.3] * 5)
    few = tune_vs_dp([-1.0, 0.0, 1.0], [0.1, 0.2, 0.3], [0.3] * 3)

    assert gap["scalars"]["chromaticity_x"] == pytest.approx(10.0)
    assert few["scalars"] == {}


def test_pyat_reports_the_second_order_chromaticity_as_a_derivative() -> None:
    result = _pyat().ANALYSES["chromaticity"](_job("chromaticity"))

    dp, tune = (result["tables"]["tune_vs_dp"]["columns"][name] for name in ("dp_pct", "tune_y"))
    assert dp == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    # The even part of the curve: nu(+d) + nu(-d) - 2 nu(0) = xi2 d^2, up to d^4.
    even = tune[4] + tune[0] - 2 * tune[2]
    assert abs(even) > 1.0e-5
    assert result["scalars"]["chromaticity2_y"] * 0.01**2 == pytest.approx(even, rel=0.05)


def test_the_engine_runs_as_the_command_the_backend_starts(tmp_path: Path) -> None:
    pytest.importorskip("at")
    (tmp_path / "job.json").write_text(json.dumps(_job("optics")))

    done = subprocess.run(
        [sys.executable, "-m", "lattice_app.engines.pyat", "job.json", "out.json"],
        cwd=tmp_path,
        # A deployment's image has the package installed; here it is on the test path.
        env={**os.environ, "PYTHONPATH": str(Path(lattice_app.__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert "twiss" in json.loads((tmp_path / "out.json").read_text())["tables"]


def test_an_engine_says_why_it_failed_and_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_: Any) -> Any:
        raise RuntimeError("unstable ring")

    (tmp_path / "job.json").write_text(json.dumps({"analysis": "optics"}))
    arguments = [str(tmp_path / "job.json"), str(tmp_path / "out.json")]

    assert run_engine({"optics": broken}, arguments) == 1
    assert "RuntimeError: unstable ring" in capsys.readouterr().err
    assert run_engine({}, arguments) == 2
    assert "does not compute optics" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def _elegant_job(analysis: str, **over: Any) -> dict[str, Any]:
    return _job(
        analysis, **{"engine": "elegant", "energy_gev": 6.0, "deck": "/decks/sr.lte", **over}
    )


def test_the_elegant_command_file_carries_the_analysis_settings() -> None:
    job = _elegant_job(
        "da", settings={"nturns": 128, "n_angles": 9, "amp_max_mm": 20, "n_bisect": 8}
    )

    text = elegant.command_file(job, Path("/scratch"))

    assert 'lattice = "/decks/sr.lte",' in text
    assert 'output = "/scratch/out.aper",' in text
    assert "p_central_mev = 5999.99997" in text
    assert "n_passes = 128," in text
    assert "n_lines = 9," in text
    assert "xmax = 0.02," in text
    assert 'mode = "n-lines",' in text


def test_the_facility_adds_to_and_replaces_namelist_entries() -> None:
    settings = {
        "use_beamline": "RING",
        "twiss_output": {"radiation_integrals": 0, "higher_order_chromaticity": 1},
        "setup": "&load_parameters filename=errors.param &end\n",
    }

    text = elegant.command_file(_elegant_job("optics", engine_settings=settings), Path("/s"))

    assert 'use_beamline = "RING",' in text
    assert "radiation_integrals = 0," in text
    assert "higher_order_chromaticity = 1," in text
    assert text.index("&load_parameters") < text.index("&twiss_output")


def test_elegant_needs_an_energy() -> None:
    with pytest.raises(ValueError, match="names no energy"):
        elegant.command_file(_elegant_job("optics", energy_gev=None), Path("/s"))
    assert "p_central_mev = 880" in elegant.command_file(
        _elegant_job("optics", energy_gev=None, engine_settings={"p_central_mev": 880}),
        Path("/s"),
    )
    with pytest.raises(ValueError, match="does not compute footprint"):
        elegant.command_file(_elegant_job("footprint"), Path("/s"))


def _program(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_elegant_runs_where_the_deck_lives_and_writes_only_to_scratch(tmp_path: Path) -> None:
    decks = tmp_path / "decks"
    decks.mkdir()
    deck = decks / "sr.lte"
    deck.write_text("ring: line=(q)\n")
    # A file an element of the deck names by a relative path.
    (decks / "kickmap.sdds").write_text("")
    seen = tmp_path / "seen.ele"
    # Stand-ins: the real programs are the facility's and are not on a test machine.
    fake_elegant = _program(
        tmp_path / "elegant",
        "import pathlib, re, sys\n"
        "commands = pathlib.Path(sys.argv[1]).read_text()\n"
        f"pathlib.Path({str(seen)!r}).write_text(commands)\n"
        "assert pathlib.Path('kickmap.sdds').exists()\n"
        "lattice = re.search(r'lattice = \"(.*)\"', commands).group(1)\n"
        "assert pathlib.Path(lattice).read_text().startswith('ring')\n"
        "for file in re.findall(r'(?:filename|output) = \"(.*)\"', commands):\n"
        "    assert pathlib.Path(file).is_absolute()\n"
        "    pathlib.Path(file).write_text('')\n",
    )
    # The beamline has eleven elements after its start, so two places are its first and last.
    fake_sdds2stream = _program(
        tmp_path / "sdds2stream",
        "import sys\n"
        "rows = {'count.twi': ['0.0'] * 12, 'out0.mmap': ['0.0\\t0.031\\t-0.028'],\n"
        "        'out1.mmap': ['5.5\\t0.025\\t-0.030']}[sys.argv[1]]\n"
        "assert sys.argv[2] in ('-columns=s', '-columns=s,deltaPositive,deltaNegative')\n"
        "print('\\n'.join(rows))\n",
    )
    job = _elegant_job(
        "lma",
        deck=str(deck),
        engine_settings={"elegant": fake_elegant, "sdds2stream": fake_sdds2stream},
    )

    result = normalize_result(ANALYSES["lma"], elegant.compute(job))

    commands = seen.read_text()
    assert f'lattice = "{deck}",' in commands
    assert commands.count("&momentum_aperture") == 2
    assert "skip_elements = 10," in commands and "soft_failure = 1," in commands
    assert result["tables"]["aperture"]["columns"] == {
        "s": [0.0, 5.5],
        "dp_plus_pct": pytest.approx([3.1, 2.5]),
        "dp_minus_pct": pytest.approx([-2.8, -3.0]),
    }
    assert result["scalars"]["dp_plus_min_pct"] == pytest.approx(2.5)
    assert sorted(path.name for path in decks.iterdir()) == ["kickmap.sdds", "sr.lte"]


def test_the_tune_of_a_turn_by_turn_signal_is_found_between_the_fft_bins() -> None:
    turns = np.arange(1024)
    for tune in (0.2327252, 0.0871, 0.4613):
        signal = 3.0e-7 + 1.0e-6 * np.cos(2 * np.pi * tune * turns + 0.4)
        assert elegant.fractional_tune(signal) == pytest.approx(tune, abs=1e-7)


# Stand-ins for a tracking run at one momentum offset: the tunes are a known curve, the
# vertical one above the half integer, and one offset has no stable optics.
_TUNE_ELEGANT = """
import json, pathlib, re, sys
commands = pathlib.Path(sys.argv[1]).read_text()
dp = float(re.search(r'MALIGN,DP=([-+.e0-9]+),ON_PASS=0,FORCE_MODIFY_MATRIX=1', commands).group(1))
turns = int(re.search(r'n_passes = (\\d+)', commands).group(1))
twiss = re.search(r'filename = "(.*)"', commands).group(1)
watch = re.search(r'FILENAME=\\\\"(.*?)\\\\"', commands).group(1)
assert 'skip = 0,' in commands and 'offset_by_orbit = 1,' in commands
stable = abs(dp - 0.01) > 1e-9
tunes = [0.23 - 0.3 * dp + 2.5 * dp**2, 0.63 - 0.8 * dp + 3.0 * dp**2]
pathlib.Path(twiss).write_text(json.dumps([3 + tunes[0], 3 + tunes[1]] if stable else None))
if stable:
    pathlib.Path(watch).write_text(json.dumps({"tunes": tunes, "turns": turns}))
"""
_TUNE_SDDS2STREAM = """
import json, math, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
if sys.argv[2] == '-parameters=nux,nuy':
    print('nan nan' if data is None else f'{data[0]!r} {data[1]!r}')
else:
    assert sys.argv[2] == '-columns=Cx,Cy,Particles'
    for turn in range(data['turns']):
        x, y = (1e-6 * math.cos(2 * math.pi * tune * turn) for tune in data['tunes'])
        print(f'{x!r}\\t{y!r}\\t1')
"""


def test_elegant_tracks_the_tune_at_each_momentum_offset(tmp_path: Path) -> None:
    deck = tmp_path / "sr.lte"
    deck.write_text("ring: line=(q)\n")
    job = _elegant_job(
        "chromaticity",
        deck=str(deck),
        settings={"dp_min_pct": -2.0, "dp_max_pct": 2.0, "n_steps": 9},
        engine_settings={
            "elegant": _program(tmp_path / "elegant", _TUNE_ELEGANT),
            "sdds2stream": _program(tmp_path / "sdds2stream", _TUNE_SDDS2STREAM),
            "tune_turns": 512,
        },
    )

    result = normalize_result(ANALYSES["chromaticity"], elegant.ANALYSES["chromaticity"](job))

    columns = result["tables"]["tune_vs_dp"]["columns"]
    assert columns["dp_pct"] == pytest.approx([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    # The offset without stable optics is a gap in the curve, not a failed analysis.
    assert columns["tune_x"][6] is None and columns["tune_y"][6] is None
    assert columns["tune_x"][0] == pytest.approx(0.23 + 0.006 + 0.001, abs=1e-6)
    # Above the half integer: the spectrum alone cannot tell 0.63 from 0.37.
    assert columns["tune_y"][4] == pytest.approx(0.63, abs=1e-6)
    assert result["scalars"]["chromaticity_x"] == pytest.approx(-0.3, abs=1e-3)
    assert result["scalars"]["chromaticity2_x"] == pytest.approx(5.0, abs=0.05)
    assert result["scalars"]["chromaticity_y"] == pytest.approx(-0.8, abs=1e-3)
    assert [path.name for path in tmp_path.iterdir() if path.suffix in (".twi", ".w1")] == []


_FODO_RING = """\
QFH: KQUAD, L=0.15, K1=1.35, N_SLICES=50
QD: KQUAD, L=0.30, K1=-1.30, N_SLICES=50
SF: KSEXT, L=0.2, K2=1.0, N_SLICES=20
SD: KSEXT, L=0.2, K2=-1.5, N_SLICES=20
B: CSBEND, L=1.5, ANGLE=0.19634954084936207, E1=0.098174770424681035, &
   E2=0.098174770424681035, N_SLICES=40
D1: DRIFT, L=0.25
D3: DRIFT, L=0.50
CELL: LINE=(QFH, D1, SF, D1, B, D3, QD, D1, SD, D1, B, D3, QFH)
RING: LINE=(16*CELL)
"""


def test_real_elegant_tracks_the_tunes_of_a_fodo_ring(tmp_path: Path) -> None:
    """Runs where elegant is installed: the facility's machine, not the OSPREY image."""
    if not (shutil.which("elegant") and shutil.which("sdds2stream")):
        pytest.skip("elegant and sdds2stream are not on PATH")
    deck = tmp_path / "ring.lte"
    deck.write_text(_FODO_RING)
    job = _elegant_job(
        "chromaticity",
        deck=str(deck),
        energy_gev=2.0,
        settings={"dp_min_pct": -2.0, "dp_max_pct": 2.0, "n_steps": 9},
    )

    curve = elegant.chromaticity(job)
    optics = elegant.compute({**job, "analysis": "optics"})

    columns = curve["tables"]["tune_vs_dp"]["columns"]
    # Tracked tunes of an independent run of this ring with the beam itself off momentum.
    assert columns["tune_x"][0] == pytest.approx(0.2406724, abs=5e-7)
    assert columns["tune_y"][8] == pytest.approx(0.3515137, abs=5e-7)
    assert columns["tune_x"][4] == pytest.approx(optics["scalars"]["tune_x"] % 1, abs=1e-7)
    for plane in ("x", "y"):
        assert curve["scalars"][f"chromaticity_{plane}"] == pytest.approx(
            optics["scalars"][f"chromaticity_{plane}"], abs=1e-4
        )
    assert curve["scalars"]["chromaticity2_x"] == pytest.approx(5.92, abs=0.05)
    assert [path.name for path in tmp_path.iterdir()] == ["ring.lte"]


def test_the_momentum_aperture_is_searched_at_places_spread_over_the_beamline() -> None:
    assert elegant.lma_places(802, 6) == [0, 160, 320, 481, 641, 801]
    assert elegant.lma_places(3, 15) == [0, 1, 2]
    assert elegant.lma_places(802, 1) == [0]
    assert elegant.lma_places(0, 6) == []


def test_pyat_does_not_guess_the_energy_of_an_elegant_deck(tmp_path: Path) -> None:
    deck = tmp_path / "sr.lte"
    deck.write_text("q: drift, l=1\nring: line=(q)\n")

    with pytest.raises(ValueError, match="names no energy"):
        _pyat().load_ring(_job("optics", deck=str(deck)))


def test_a_missing_elegant_is_named(tmp_path: Path) -> None:
    deck = tmp_path / "sr.lte"
    deck.write_text("x")
    job = _elegant_job(
        "optics", deck=str(deck), engine_settings={"elegant": str(tmp_path / "elegant")}
    )

    with pytest.raises(RuntimeError, match="elegant could not start"):
        elegant.compute(job)
