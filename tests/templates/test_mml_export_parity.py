"""The shipped MML exporter run for real, against the committed facility exports.

A MATLAB runs here. For each sub-machine, the shipped ``mml_export.m`` is run
against a checkout of MML-prod in simulator mode and its two files are compared
with the export committed under ``tests/fixtures/mml/``. The sibling
``test_mml_export_template.py`` holds the script to its written contract with no
MATLAB; this file holds it to the real thing.

What is compared, and what is not
---------------------------------
The Accelerator Objects are compared family by family after
``normalize_family``, the leveller for the three places a MATLAB release
difference would otherwise read as a false failure: a dropped ``Handles``
field, a whitespace-only slot, and the spelling of a non-finite number.

The Accelerator Data is compared structurally: both documents carry the same
keys, strings and integers are equal, and floats agree to within a relative
tolerance of 1e-9 (absolute 1e-12), because a value that passes through a
lattice simulator is not bit-reproducible across releases.

Two keys are excluded everywhere, because they differ between any two runs by
design:

* ``_export.matlab`` --- the version string of the MATLAB that ran the export;
* ``_export.timestamp`` --- when the export ran.

The rest of the ``_export`` block --- ``exporter``, ``machine`` and
``submachine`` --- is compared and must match.

Running it
----------
Two variables name the checkouts, and the lane skips when either is unset:

* ``OSPREY_MML_PROD_ROOT`` --- an MML-prod checkout: the folder holding ``mml/``;
* ``OSPREY_AT_ROOT`` --- an Accelerator Toolbox checkout: the folder holding
  ``atmat/``.

``matlab`` must also be on PATH. The MML checkout's own AT (``simulators/at2.0``)
must have integrators compiled for that MATLAB (``atmexall`` in its ``atmat/``):
the machine's ``setpathmml`` puts that AT on the path and initialises the
Accelerator Objects, tracking through it for the momentum compaction, before
anything else can be put on the path. Each sub-machine costs one MATLAB session, so
the export of a sub-machine is produced once and shared by every case below.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from osprey.services.mml.loaders import LoadedInput
from osprey.services.mml.loaders.json_any import load_json
from osprey.services.mml.normalize import normalize_family
from osprey.services.mml.systems import resolve_system

REPO_ROOT = Path(__file__).resolve().parents[2]
MML_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "apps" / "control_assistant" / "data" / "mml"
EXPORTER = MML_DIR / "mml_export.m"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

MML_PROD_ROOT_ENV = "OSPREY_MML_PROD_ROOT"
AT_ROOT_ENV = "OSPREY_AT_ROOT"

MATLAB = shutil.which("matlab")

#: A full MML setup plus a storage-ring export is minutes of work, not seconds.
MATLAB_TIMEOUT_SECONDS = 1800

#: The ``_export`` keys that carry the run rather than the machine.
VOLATILE_EXPORT_KEYS = frozenset({"matlab", "timestamp"})

FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-12

pytestmark = [
    pytest.mark.requires_matlab,
    pytest.mark.slow,
    pytest.mark.skipif(
        MATLAB is None,
        reason="matlab not on PATH — a MATLAB installation with the AT toolbox is required",
    ),
]


@dataclass(frozen=True)
class Case:
    """One sub-machine: how MATLAB reaches it, and where its export is committed."""

    fixture: str
    machine: str
    submachine: str
    setup: str

    @property
    def key(self) -> str:
        return f"{self.machine}-{self.submachine}"

    @property
    def stem(self) -> str:
        """The file stem the exporter writes for this sub-machine."""
        return f"{_filename(self.machine)}.{_filename(self.submachine)}"


CASES = (
    Case("nsls2", "NSLS2", "StorageRing", "setpathnsls2('StorageRing')"),
    Case("nsls2", "NSLS2", "LTB", "setpathnsls2('LTB')"),
    Case("spear3", "SPEAR3", "StorageRing", "setpathspear3"),
)


def _filename(text: str) -> str:
    """The exporter's own file-name rule, as ``local_filename`` spells it."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).lower()


# ---------------------------------------------------------------------------
# What the run needs
# ---------------------------------------------------------------------------


def _named_directory(variable: str, what: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        pytest.skip(f"{variable} is not set — point it at {what}")
    path = Path(value).expanduser()
    if not path.is_dir():
        pytest.skip(f"{variable} points at {path}, which is not a directory")
    return path


def _checkouts() -> tuple[Path, Path]:
    """The MML function folder and the AT folder holding ``atpath.m``."""
    mml_root = _named_directory(MML_PROD_ROOT_ENV, "an MML-prod checkout")
    mml = mml_root / "mml"
    if not (mml / "setpathmml.m").is_file():
        pytest.skip(f"{MML_PROD_ROOT_ENV} points at {mml_root}, which holds no mml/setpathmml.m")

    at_root = _named_directory(AT_ROOT_ENV, "an Accelerator Toolbox checkout")
    at_dir = next((d for d in (at_root / "atmat", at_root) if (d / "atpath.m").is_file()), None)
    if at_dir is None:
        pytest.skip(f"{AT_ROOT_ENV} points at {at_root}, which holds no atmat/atpath.m")

    return mml, at_dir


@pytest.fixture(params=CASES, ids=[case.key for case in CASES])
def case(request: pytest.FixtureRequest) -> Case:
    return request.param


@pytest.fixture
def committed(case: Case) -> Path:
    """The committed AO file for this sub-machine, with its AD sibling beside it."""
    directory = FIXTURES / case.fixture
    if not directory.is_dir():
        pytest.skip(
            f"{directory.relative_to(REPO_ROOT)} is not committed — "
            f"the {case.fixture} export fixture is missing"
        )
    for suffix in (".ao.json", ".ad.json"):
        path = directory / f"{case.stem}{suffix}"
        if not path.is_file():
            pytest.skip(f"{path.relative_to(REPO_ROOT)} is not committed")
    return directory / f"{case.stem}.ao.json"


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Case], Path]:
    """Export a sub-machine once per module and hand back its AO path.

    The checkouts are resolved on the first export rather than up front, so a
    sub-machine whose fixture is not committed says so instead of asking for
    two environment variables it has no use for.
    """
    produced: dict[str, Path] = {}

    def run(case: Case) -> Path:
        if case.key not in produced:
            workdir = tmp_path_factory.mktemp(case.stem.replace(".", "_"))
            produced[case.key] = _run_export(case, workdir, *_checkouts())
        return produced[case.key]

    return run


# ---------------------------------------------------------------------------
# Running MATLAB
# ---------------------------------------------------------------------------


def _matlab_literal(text: str | Path) -> str:
    """*text* as a MATLAB character-vector literal."""
    return "'" + str(text).replace("'", "''") + "'"


def _program(case: Case, mml: Path, at_dir: Path, exporter_dir: Path, outdir: Path) -> str:
    """The one command ``matlab -batch`` is given.

    One line: ``matlab -batch`` runs only the first line of a statement that
    holds a newline (R2026a), and exits 0. The machine's own setpath points AT
    at whatever the MML checkout ships and initialises through it, so the named
    AT checkout is put on the path after it, not before.
    """
    return "; ".join(
        [
            f"addpath({_matlab_literal(mml)})",
            case.setup,
            f"setpathat({_matlab_literal(at_dir)})",
            "switch2sim",
            f"addpath({_matlab_literal(exporter_dir)})",
            f"mml_export({_matlab_literal(outdir)})",
        ]
    )


def _run_export(case: Case, workdir: Path, mml: Path, at_dir: Path) -> Path:
    """Export one sub-machine into *workdir* and return the AO file."""
    assert MATLAB is not None
    outdir = workdir / "export"
    outdir.mkdir()
    shutil.copy2(EXPORTER, workdir / EXPORTER.name)
    program = _program(case, mml, at_dir, workdir, outdir)

    completed = subprocess.run(
        [MATLAB, "-batch", program],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=MATLAB_TIMEOUT_SECONDS,
        check=False,
    )
    report = (
        f"exporting {case.key} failed (exit {completed.returncode})\n"
        f"--- program ---\n{program}\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}"
    )

    assert completed.returncode == 0, report
    # matlab -batch reports some failures on stdout and still exits 0, so the
    # files themselves are the proof that the export ran.
    for suffix in (".ao.json", ".ad.json"):
        assert (outdir / f"{case.stem}{suffix}").is_file(), report
    return outdir / f"{case.stem}.ao.json"


# ---------------------------------------------------------------------------
# Comparing two exports
# ---------------------------------------------------------------------------


def _families(ao: dict) -> dict[str, object]:
    return {
        name: normalize_family(body) if isinstance(body, dict) else body
        for name, body in ao.items()
    }


def _body(loaded: LoadedInput) -> dict:
    """An Accelerator Data document without its ``_export`` block."""
    assert loaded.ad is not None
    return {key: value for key, value in loaded.ad.items() if key != "_export"}


def _differences(actual: object, expected: object, where: str = "") -> list[str]:
    """Every place two decoded documents disagree, named by path."""
    place = where or "<root>"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{place}: expected a map, got {type(actual).__name__}"]
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            return [f"{place}: missing keys {missing}, unexpected keys {extra}"]
        found: list[str] = []
        for key in expected:
            found += _differences(actual[key], expected[key], f"{where}.{key}")
        return found
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{place}: expected a list, got {type(actual).__name__}"]
        if len(actual) != len(expected):
            return [f"{place}: expected {len(expected)} entries, got {len(actual)}"]
        found = []
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            found += _differences(left, right, f"{where}[{index}]")
        return found
    if _is_number(expected) and _is_number(actual) and not _exact(expected, actual):
        if math.isclose(actual, expected, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            return []
        return [f"{place}: expected {expected!r}, got {actual!r}"]
    if actual != expected or type(actual) is not type(expected):
        return [f"{place}: expected {expected!r}, got {actual!r}"]
    return []


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _exact(expected: object, actual: object) -> bool:
    """True when both numbers are integers, which must match to the digit."""
    return isinstance(expected, int) and isinstance(actual, int)


def _unbounded_ranges(document: object, where: str = "") -> dict[str, object]:
    """Every ``Range`` carrying an infinite bound, by path."""
    found: dict[str, object] = {}
    if isinstance(document, dict):
        for key, value in document.items():
            path = f"{where}.{key}"
            if key == "Range" and any(leaf in ("Inf", "-Inf") for leaf in _leaves(value)):
                found[path] = value
            found.update(_unbounded_ranges(value, path))
    elif isinstance(document, list):
        for index, entry in enumerate(document):
            found.update(_unbounded_ranges(entry, f"{where}[{index}]"))
    return found


def _leaves(value: object) -> Iterator[object]:
    if isinstance(value, list):
        for entry in value:
            yield from _leaves(entry)
    elif isinstance(value, dict):
        for entry in value.values():
            yield from _leaves(entry)
    else:
        yield value


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_the_exported_objects_match_the_committed_export(
    case: Case, committed: Path, export: Callable[[Case], Path]
) -> None:
    """Family for family, the fresh export is the committed one."""
    expected = load_json(committed)
    actual = load_json(export(case))

    assert sorted(actual.ao) == sorted(expected.ao)
    assert _families(actual.ao) == _families(expected.ao)


def test_the_exported_data_matches_the_committed_export(
    case: Case, committed: Path, export: Callable[[Case], Path]
) -> None:
    """The Accelerator Data, to the tolerance this module's docstring names."""
    expected = load_json(committed)
    actual = load_json(export(case))

    assert _differences(_body(actual), _body(expected)) == []


def test_the_export_block_names_the_machine_and_sub_machine(
    case: Case, committed: Path, export: Callable[[Case], Path]
) -> None:
    """Everything in ``_export`` but the MATLAB version and the timestamp."""
    expected = load_json(committed)
    actual = load_json(export(case))

    assert actual.export is not None
    assert set(actual.export) == set(expected.export)
    assert {k: v for k, v in actual.export.items() if k not in VOLATILE_EXPORT_KEYS} == {
        k: v for k, v in expected.export.items() if k not in VOLATILE_EXPORT_KEYS
    }
    assert actual.export["machine"] == case.machine
    assert actual.export["submachine"] == case.submachine
    assert resolve_system(actual, None) == case.submachine


def test_an_unbounded_range_round_trips_as_the_inf_strings(
    case: Case, committed: Path, export: Callable[[Case], Path]
) -> None:
    """An infinite limit arrives as the string ``Inf`` or ``-Inf``, never as null."""
    expected = _unbounded_ranges(load_json(committed).ao)
    if not expected:
        pytest.skip(f"the committed {case.key} export carries no Range with an infinite bound")

    actual = _unbounded_ranges(load_json(export(case)).ao)

    assert actual == expected
    for path, value in actual.items():
        for leaf in _leaves(value):
            assert leaf is not None, path
            assert not isinstance(leaf, float) or math.isfinite(leaf), path
