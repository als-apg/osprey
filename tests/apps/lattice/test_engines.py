"""The two engines the application carries: pyAT on a real ring, elegant on stand-in binaries."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import lattice_app
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
