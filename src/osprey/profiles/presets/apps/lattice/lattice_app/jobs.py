"""One computation per result, whoever asks: the shared cache and the workers that fill it.

A result's key is the hash of everything that decides its numbers: the deck's bytes, the
engine and its version and settings, the analysis and its settings, the energy. Every
terminal and every agent asking for the same key reads one cache file, and while it is
being computed they all wait on the one worker. A changed deck or setting is a new key,
so nothing cached is ever stale.

A worker is a separate process, ``<engine command> <job.json> <out.json>``, in its own
process group, so its deadline can end it and whatever it started.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lattice_app.catalog import ANALYSES, JOB_SCHEMA, RESULT_SCHEMA, normalize_result
from lattice_app.config import Model

#: How much of a failed worker's stderr the error carries.
STDERR_TAIL = 2000

#: Runs one worker: ``(command, job, deadline_s) -> the engine's output``.
Runner = Callable[[tuple[str, ...], Mapping[str, Any], float], Awaitable[Any]]


_DECK_HASHES: dict[tuple[str, int, int], str] = {}


class WorkerFailed(Exception):
    """A worker ended without a result; the message says why."""


def deck_sha256(deck: Path) -> str:
    """The hash of the deck's bytes.

    Raises:
        ValueError: The deck cannot be read.
    """
    try:
        stat = deck.stat()
        seen = (str(deck), stat.st_mtime_ns, stat.st_size)
        if seen not in _DECK_HASHES:  # every state poll asks; an unchanged file is read once
            _DECK_HASHES[seen] = hashlib.sha256(deck.read_bytes()).hexdigest()
        return _DECK_HASHES[seen]
    except OSError as exc:
        raise ValueError(f"the deck {deck} cannot be read: {exc.strerror or exc}") from exc


@dataclass(frozen=True)
class Request:
    """One result asked for: a model's analysis at an energy, with the key that names it."""

    key: str
    command: tuple[str, ...]
    job: Mapping[str, Any]
    deadline_s: float


def build_request(model: Model, analysis: str, energy_gev: float | None) -> Request:
    """The request for *analysis* of *model* at *energy_gev*.

    Raises:
        ValueError: The deck cannot be read.
    """
    setup = model.setups[analysis]
    job = {
        "schema": JOB_SCHEMA,
        "analysis": analysis,
        "model": model.name,
        "deck": str(model.deck),
        "deck_sha256": deck_sha256(model.deck),
        "energy_gev": energy_gev,
        "settings": dict(setup.settings),
        "engine": model.engine.name,
        "engine_settings": dict(model.engine_settings),
    }
    # The deck's path and the model's name decide no number: two models over one deck
    # with one setup share a result.
    identity = {
        key: value for key, value in job.items() if key not in ("deck", "model", "schema")
    } | {"engine_version": model.engine.version, "command": list(model.engine.command)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
    return Request(key, model.engine.command, job, setup.deadline_s)


def _end(process: asyncio.subprocess.Process) -> None:
    """End the worker's own process group: the worker and whatever it started."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


async def run_worker(command: tuple[str, ...], job: Mapping[str, Any], deadline_s: float) -> Any:
    """Run one worker to its output or its deadline.

    Raises:
        WorkerFailed: The worker could not start, failed, ran past *deadline_s*, or wrote
            no readable output.
    """
    workdir = Path(tempfile.mkdtemp(prefix="lattice-job-"))
    job_path, out_path = workdir / "job.json", workdir / "out.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    try:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                str(job_path),
                str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise WorkerFailed(f"the engine could not start: {exc}") from exc
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), deadline_s)
        except TimeoutError:
            _end(process)
            await process.wait()
            raise WorkerFailed(
                f"stopped at its deadline of {deadline_s:g} s; raise the analysis's "
                "deadline_s or reduce its settings"
            ) from None
        except asyncio.CancelledError:
            _end(process)
            raise
        if process.returncode != 0:
            tail = stderr.decode(errors="replace").strip()[-STDERR_TAIL:]
            raise WorkerFailed(tail or f"the engine exited with status {process.returncode}")
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkerFailed(f"the engine wrote no readable output: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@dataclass
class _Running:
    task: asyncio.Task[None]
    started: float | None = None


class Jobs:
    """The cache of results and the workers filling it."""

    def __init__(self, cache_dir: Path, max_workers: int, runner: Runner = run_worker) -> None:
        """
        Args:
            cache_dir: Where results are kept, one file per key.
            max_workers: Workers run at once; further requests queue.
            runner: Runs one worker.
        """
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._slots = asyncio.Semaphore(max_workers)
        self._runner = runner
        self._running: dict[str, _Running] = {}
        self._failed: dict[str, str] = {}
        self._summaries: dict[str, dict[str, Any]] = {}

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def result(self, key: str) -> dict[str, Any] | None:
        """The cached result document of *key*, or ``None``."""
        try:
            document: dict[str, Any] = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return document if document.get("schema") == RESULT_SCHEMA else None

    def status(self, key: str) -> dict[str, Any]:
        """Where *key* stands: ``ready``, ``running``, ``queued``, ``failed`` or ``absent``.

        A ready result's status carries its scalars, so a state read shows every ready
        number without opening a table.
        """
        if (running := self._running.get(key)) is not None:
            if running.started is None:
                return {"status": "queued", "key": key}
            elapsed = round(time.monotonic() - running.started, 1)
            return {"status": "running", "key": key, "elapsed_s": elapsed}
        if key in self._failed:
            return {"status": "failed", "key": key, "error": self._failed[key]}
        if key not in self._summaries:
            if (document := self.result(key)) is None:
                return {"status": "absent", "key": key}
            self._summaries[key] = {
                name: document.get(name)
                for name in ("computed_at", "elapsed_s", "scalars", "units")
            }
        return {"status": "ready", "key": key, **self._summaries[key]}

    def ensure(self, request: Request, *, retry: bool = False) -> None:
        """Start computing *request* unless it is cached, running, or failed.

        A failed result stays failed until asked for with *retry*, so twenty terminals
        polling a failing analysis do not restart it twenty times.
        """
        if request.key in self._running or self._path(request.key).is_file():
            return
        if request.key in self._failed:
            if not retry:
                return
            del self._failed[request.key]
        task = asyncio.get_running_loop().create_task(self._compute(request))
        self._running[request.key] = _Running(task)

    async def _compute(self, request: Request) -> None:
        key = request.key
        try:
            async with self._slots:
                self._running[key].started = started = time.monotonic()
                raw = await self._runner(request.command, request.job, request.deadline_s)
                analysis = ANALYSES[str(request.job["analysis"])]
                document = {
                    "schema": RESULT_SCHEMA,
                    "key": key,
                    "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_s": round(time.monotonic() - started, 1),
                    "job": dict(request.job),
                    **normalize_result(analysis, raw),
                }
            scratch = self._path(key).with_suffix(".tmp")
            scratch.write_text(json.dumps(document), encoding="utf-8")
            os.replace(scratch, self._path(key))  # whole or absent: no reader sees half
        except (WorkerFailed, ValueError, OSError) as exc:
            self._failed[key] = str(exc)
        finally:
            del self._running[key]

    async def wait(self, key: str, wait_s: float) -> None:
        """Wait up to *wait_s* for *key*'s worker; returns at once when none runs."""
        if (running := self._running.get(key)) is not None and wait_s > 0:
            await asyncio.wait({running.task}, timeout=wait_s)

    async def close(self) -> None:
        """End every worker still running."""
        tasks = [running.task for running in self._running.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
