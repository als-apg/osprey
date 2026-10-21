"""Shared fixtures for lattice dashboard physics-worker tests.

Provides small, in-memory pyAT FODO rings so the compute workers exercise
the real pyAT code paths (get_optics, tracking, rotate) without loading a
full facility lattice.  The rings are deliberately tiny and stable so that
tracking-based workers (DA, LMA, footprint) run in well under a second.

Also provides ``FakeSlots``, a job-slot set whose jobs spawn no process, for
the tests of the workers' supervision.
"""

from __future__ import annotations

import asyncio
import itertools

import numpy as np
import pytest

from osprey_connectors.process import ChildExit, ExitCause

at = pytest.importorskip("at")


def _build_fodo(kf: float = 1.0, with_sextupole: bool = False) -> at.Lattice:
    """Construct a small, stable FODO ring.

    Two identical cells of focusing/defocusing quads, drifts and dipoles.
    kf=1.0 yields a stable working point (tune ~ [0.48, 0.10]).
    """
    d = at.Drift("DR", 0.5)
    qf = at.Quadrupole("QF", 0.2, kf)
    qd = at.Quadrupole("QD", 0.2, -kf)
    b = at.Dipole("BM", 0.5, np.pi / 8)
    if with_sextupole:
        sf = at.Sextupole("SF", 0.1, 5.0)
        cell = [qf, d, sf, b, d, qd, d, b, d]
    else:
        cell = [qf, d, b, d, qd, d, b, d]
    return at.Lattice(cell * 2, name="FODO", energy=2e9)


@pytest.fixture
def make_fodo():
    """Factory returning fresh FODO rings (avoids cross-test mutation)."""
    return _build_fodo


@pytest.fixture
def fodo_ring():
    """A fresh, stable FODO ring without sextupoles."""
    return _build_fodo()


@pytest.fixture
def fodo_ring_sext():
    """A fresh, stable FODO ring including a sextupole family."""
    return _build_fodo(with_sextupole=True)


class FakeJob:
    """A ``ChildJob`` stand-in that spawns nothing; a test ends it with ``finish``."""

    def __init__(self, slots: FakeSlots, name: str, job: int) -> None:
        self._slots = slots
        self.name = name
        self.job = job
        self.argv: list[str] | None = None
        self.started = False
        self._exit = None
        self._done: asyncio.Event | None = None

    def _event(self) -> asyncio.Event:
        if self._done is None:
            self._done = asyncio.Event()
        return self._done

    def finish(self, cause: ExitCause, returncode: int | None = 0, stderr_tail: str = "") -> None:
        """End the job with *cause*, as its child's reap would."""
        if self._exit is None:
            self._exit = ChildExit(self.job, cause, returncode, stderr_tail)
        self._event().set()

    async def start(self, argv, **_kwargs) -> None:
        for earlier in [j for j in self._slots.jobs if j.name == self.name and j.job < self.job]:
            await earlier.stop(ExitCause.CANCELLED)
        if not self._slots.is_current(self) or self._exit is not None:
            self.finish(ExitCause.CANCELLED, None)
            return
        self.argv = list(argv)
        self.started = True

    async def wait(self, deadline_s=None, grace_s=None):  # noqa: ARG002
        await self._event().wait()
        return self._exit

    async def stop(self, cause: ExitCause = ExitCause.CANCELLED, grace_s=None):  # noqa: ARG002
        self.finish(cause, None)
        return self._exit


class FakeSlots:
    """A ``JobSlots`` stand-in whose jobs are ``FakeJob`` instances."""

    def __init__(self) -> None:
        self.jobs: list[FakeJob] = []
        #: The names reserved, in launch order; a test may clear it.
        self.launched: list[str] = []
        self._current: dict[str, FakeJob] = {}
        self._ids = itertools.count(1)

    def reserve(self, name: str) -> FakeJob:
        job = FakeJob(self, name, next(self._ids))
        self.jobs.append(job)
        self.launched.append(name)
        self._current[name] = job
        return job

    def is_current(self, job: FakeJob) -> bool:
        return self._current.get(job.name) is job

    def current(self, name: str) -> FakeJob | None:
        return self._current.get(name)

    async def stop_all(self, cause: ExitCause = ExitCause.CANCELLED):
        self._current.clear()
        return [await job.stop(cause) for job in self.jobs if job._exit is None]


@pytest.fixture
def fake_slots(monkeypatch):
    """Every ComputeManager built in the test takes one shared ``FakeSlots``."""
    from osprey.interfaces.lattice_dashboard import compute as compute_mod

    slots = FakeSlots()
    monkeypatch.setattr(compute_mod, "JobSlots", lambda: slots)
    return slots
