"""A run of this suite owns its temporary root; nothing else on the host touches it.

``tests/conftest.py``, through the harness's ``_run_temp.py``, gives every run a
fresh ``osprey-tests-*`` directory of its own as pytest's base temporary
directory, outside the shared ``pytest-of-<user>`` that any pytest on the host
numbers and prunes. Two runs
on one temporary root therefore never delete each other's trees, an xdist
run's workers live under their controller's root, and the run removes its root
itself once it passed.

The proofs run nested pytests of the probe below under the repository's own
conftest. The probe runs only when ``OSPREY_RUN_TEMP_PROBE`` names a directory:
it writes its base temporary directory to ``<dir>/root-<pid>``, then, when
``OSPREY_RUN_TEMP_PROBE_HOLD`` is ``1``, waits for ``<dir>/release``, and when
``OSPREY_RUN_TEMP_PROBE_FAIL`` is ``1``, fails.
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests._nested_pytest import STALL_SECONDS, NestedRun, outer_run_free_env, run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE_KEY = "OSPREY_RUN_TEMP_PROBE"
_HOLD_KEY = "OSPREY_RUN_TEMP_PROBE_HOLD"
_FAIL_KEY = "OSPREY_RUN_TEMP_PROBE_FAIL"

_PROBE_NODE = f"{Path(__file__).relative_to(_REPO_ROOT)}::test_probe_records_its_temporary_root"

#: How long the outer test waits for a held run to record its root.
_RECORD_WAIT_SECONDS = STALL_SECONDS


def test_probe_records_its_temporary_root(tmp_path_factory: pytest.TempPathFactory) -> None:
    probe = os.environ.get(_PROBE_KEY)
    if not probe:
        pytest.skip(f"runs only inside a nested run that sets {_PROBE_KEY}")
    probe_dir = Path(probe)
    (probe_dir / f"root-{os.getpid()}").write_text(str(tmp_path_factory.getbasetemp()))
    if os.environ.get(_HOLD_KEY) == "1":
        deadline = time.monotonic() + STALL_SECONDS
        while not (probe_dir / "release").exists():
            assert time.monotonic() < deadline, "the holding run was never released"
            time.sleep(0.1)
    fails_on_purpose = os.environ.get(_FAIL_KEY) == "1"
    assert not fails_on_purpose, "the probe fails on purpose"


def _run_probe(
    probe_dir: Path,
    *,
    temproot: Path,
    extra_env: dict[str, str] | None = None,
    extra_args: tuple[str, ...] = (),
    stall_seconds: float = STALL_SECONDS,
) -> NestedRun:
    probe_dir.mkdir(parents=True, exist_ok=True)
    env = outer_run_free_env(also_drop=(_PROBE_KEY,))
    env[_PROBE_KEY] = str(probe_dir)
    env.update(extra_env or {})
    return run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--color=no",
            *extra_args,
            _PROBE_NODE,
        ],
        cwd=_REPO_ROOT,
        env=env,
        stall_seconds=stall_seconds,
        temproot=temproot,
    )


def _recorded_roots(probe_dir: Path) -> list[Path]:
    return [Path(p.read_text()) for p in sorted(probe_dir.glob("root-*"))]


def _output(result: NestedRun) -> str:
    return result.stdout + result.stderr


def _assert_outside_the_numbered_root(root: Path) -> None:
    assert not any(part.startswith("pytest-of-") for part in root.parts), root
    assert root.name.startswith("osprey-tests-"), root


def test_two_runs_on_one_temproot_never_touch_each_other(tmp_path: Path) -> None:
    temproot = tmp_path / "temproot"
    temproot.mkdir()
    probe_a = tmp_path / "a"
    probe_b = tmp_path / "b"

    with ThreadPoolExecutor(max_workers=1) as pool:
        held = pool.submit(
            _run_probe,
            probe_a,
            temproot=temproot,
            extra_env={_HOLD_KEY: "1"},
            stall_seconds=2 * STALL_SECONDS,
        )
        try:
            deadline = time.monotonic() + _RECORD_WAIT_SECONDS
            while not _recorded_roots(probe_a):
                assert not held.done(), _output(held.result())
                assert time.monotonic() < deadline, "the held run never recorded its root"
                time.sleep(0.1)
            [root_a] = _recorded_roots(probe_a)

            run_b = _run_probe(probe_b, temproot=temproot)
            assert run_b.returncode == 0, _output(run_b)
            [root_b] = _recorded_roots(probe_b)

            assert root_b != root_a
            assert not root_b.exists(), "a green run removes its own root"
            assert root_a.is_dir(), "the other run's root was touched"
        finally:
            (probe_a / "release").write_text("")
        run_a = held.result()

    assert run_a.returncode == 0, _output(run_a)
    assert not root_a.exists()
    _assert_outside_the_numbered_root(root_a)
    _assert_outside_the_numbered_root(root_b)


def test_an_xdist_run_puts_its_workers_under_its_root(tmp_path: Path) -> None:
    temproot = tmp_path / "temproot"
    temproot.mkdir()
    probe = tmp_path / "probe"

    result = _run_probe(probe, temproot=temproot, extra_args=("-n", "2"))

    assert result.returncode == 0, _output(result)
    worker_roots = _recorded_roots(probe)
    assert worker_roots, _output(result)
    run_roots = {root.parent for root in worker_roots}
    assert len(run_roots) == 1, worker_roots
    [run_root] = run_roots
    for root in worker_roots:
        assert re.fullmatch(r"popen-gw\d+", root.name), root
    assert run_root.parent == temproot.resolve()
    _assert_outside_the_numbered_root(run_root)
    assert not run_root.exists(), "the controller removes its root, workers included"


def test_a_given_basetemp_is_honoured_and_kept(tmp_path: Path) -> None:
    temproot = tmp_path / "temproot"
    temproot.mkdir()
    given = tmp_path / "given"
    probe = tmp_path / "probe"

    result = _run_probe(probe, temproot=temproot, extra_args=("--basetemp", str(given)))

    assert result.returncode == 0, _output(result)
    assert _recorded_roots(probe) == [given.resolve()]
    assert given.is_dir()
    assert "temporary files kept at" not in _output(result)


def test_a_red_run_keeps_its_root_and_names_it(tmp_path: Path) -> None:
    temproot = tmp_path / "temproot"
    temproot.mkdir()
    probe = tmp_path / "probe"

    result = _run_probe(probe, temproot=temproot, extra_env={_FAIL_KEY: "1"})

    assert result.returncode == 1, _output(result)
    [root] = _recorded_roots(probe)
    assert root.is_dir()
    assert f"temporary files kept at {root}" in result.stdout, _output(result)
    _assert_outside_the_numbered_root(root)
