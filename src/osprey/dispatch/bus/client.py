"""The dispatcher's worker calls, carried over the bus.

:class:`BusWorkerClient` offers the calls :mod:`osprey.dispatch.worker_client`
makes over HTTP, and raises the same
:class:`~osprey.dispatch.worker_client.WorkerRequestError` subclasses for the
same outcomes. So the ``on_error`` policy and every dispatcher route treat a
bus failure exactly as they treat an HTTP one:

- no worker connected, a worker that left or never answered →
  :class:`~osprey.dispatch.worker_client.WorkerUnreachableError` (retryable);
- ``ERR 401`` → :class:`~osprey.dispatch.worker_client.WorkerAuthRejectedError`;
- ``ERR 503`` from the run list →
  :class:`~osprey.dispatch.worker_client.WorkerRecordUnavailableError`;
- any other ``ERR 4xx`` →
  :class:`~osprey.dispatch.worker_client.WorkerRejectedRequestError`, carrying
  a whitelisted ``error_code`` (non-retryable);
- ``ERR 5xx`` → :class:`~osprey.dispatch.worker_client.WorkerUnavailableError`.

It also serves the calls the bridges make of a worker today (run status and
artifacts), which the dispatcher answers under its ``/worker`` facade.

Run-scoped calls go to the worker that owns the run. A run the broker has no
owner for — the dispatcher restarted, or the run is old enough to be forgotten
— is asked of every live worker at once, and the first answer that is not
``404`` wins.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.broker import StreamGoneError, WorkerBus, WorkerGoneError
from osprey.dispatch.worker_client import (
    KNOWN_REJECTION_CODES,
    RECORD_READ_TIMEOUT,
    WorkerAuthRejectedError,
    WorkerRecordUnavailableError,
    WorkerRejectedRequestError,
    WorkerRequestError,
    WorkerUnavailableError,
    WorkerUnreachableError,
    dispatch_payload,
)

#: Seconds a dispatch waits for a worker with room to connect.
DEFAULT_WORKER_WAIT_SEC = 30.0
#: Seconds a worker has to accept a dispatch, as over HTTP.
_DISPATCH_TIMEOUT_SEC = 30.0

#: The most rows one run list answers, as one worker's ``/dashboard/runs`` does.
_DASHBOARD_RUNS = 50


def _errors(answers: list[protocol.Message | BaseException]) -> list[Exception]:
    """The failures among gathered answers; a ``BaseException`` that is not an error re-raises."""
    errors: list[Exception] = []
    for answer in answers:
        if isinstance(answer, protocol.Message):
            continue
        if not isinstance(answer, Exception):
            raise answer
        errors.append(answer)
    return errors


def _translate(exc: Exception, what: str, *, record_503: bool = False) -> WorkerRequestError:
    """The worker-client exception that stands for a bus failure."""
    if isinstance(exc, WorkerGoneError):
        return WorkerUnreachableError(f"{what}: {exc}")
    if isinstance(exc, TimeoutError):
        return WorkerUnreachableError(f"{what}: timed out")
    if isinstance(exc, protocol.BusError):
        if exc.status == 401:
            return WorkerAuthRejectedError(f"{what}: unauthorized (401)")
        if record_503 and exc.status == 503:
            return WorkerRecordUnavailableError(exc.detail or "record unavailable")
        if 400 <= exc.status < 500:
            code = exc.detail if exc.detail in KNOWN_REJECTION_CODES else None
            return WorkerRejectedRequestError(f"{what}: {exc.status} from worker", error_code=code)
        return WorkerUnavailableError(f"{what}: {exc.status} from worker")
    raise exc


class BusWorkerClient:
    """The dispatcher's worker calls over a :class:`~osprey.dispatch.bus.broker.WorkerBus`.

    Args:
        bus: The started broker.
        worker_wait_sec: Seconds a dispatch waits for a worker with room.
    """

    def __init__(self, bus: WorkerBus, *, worker_wait_sec: float = DEFAULT_WORKER_WAIT_SEC):
        self._bus = bus
        self._wait = worker_wait_sec

    # -- the four calls the dispatcher makes ------------------------------------

    async def dispatch(
        self,
        *,
        prompt: str,
        allowed_tools: list[str],
        surface_prompt: str | None = None,
        surface_tools: list[str] | None = None,
        input_files: list[dict[str, Any]] | None = None,
        max_turns: int | None = None,
        owner: str | None = None,
        prior_answer_runs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a prompt to the least-loaded worker; see ``dispatch_to_worker``.

        Returns:
            The worker's answer, ``{"status": "accepted", "run_id": ...}``.
        """
        payload = dispatch_payload(
            prompt,
            allowed_tools,
            surface_prompt=surface_prompt,
            surface_tools=surface_tools,
            input_files=input_files,
            max_turns=max_turns,
            owner=owner,
            prior_answer_runs=prior_answer_runs,
        )
        try:
            _, answer = await self._bus.dispatch(
                payload, wait=self._wait, timeout=_DISPATCH_TIMEOUT_SEC
            )
        except Exception as exc:
            raise _translate(exc, "dispatch") from exc
        return answer

    async def runs(self) -> list[dict[str, Any]]:
        """Recent runs across every live worker, newest first.

        Raises:
            WorkerUnreachableError: No worker is live, or none answered.
            WorkerRecordUnavailableError: Every worker that answered could not
                read the agent record.
        """
        live = self._bus.live()
        if not live:
            raise WorkerUnreachableError("runs: no dispatch worker is connected")
        answers = await asyncio.gather(
            *(self._bus.request(w, protocol.RUNS, {}, timeout=RECORD_READ_TIMEOUT) for w in live),
            return_exceptions=True,
        )
        rows: dict[str, dict[str, Any]] = {}
        for answer in answers:
            if isinstance(answer, protocol.Message):
                for row in answer.body.get("body") or []:
                    if isinstance(row, dict) and isinstance(row.get("run_id"), str):
                        rows.setdefault(row["run_id"], row)
        errors = _errors(answers)
        if not rows and errors:
            raise _translate(errors[0], "runs", record_503=True) from errors[0]
        newest = sorted(rows.values(), key=lambda r: r.get("created_at") or 0, reverse=True)
        return newest[:_DASHBOARD_RUNS]

    async def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel ``run_id`` on the worker that holds it; see ``cancel_worker_run``."""
        reply, _ = await self._routed(
            run_id,
            protocol.CANCEL,
            "cancel",
            prefer=lambda body: isinstance(body, dict) and body.get("cancelled") is True,
        )
        return cast(dict[str, Any], reply)

    async def stream(self, run_id: str) -> AsyncIterator[bytes]:
        """The run's events as server-sent-event bytes; see ``proxy_worker_stream``.

        Every viewer gets the events buffered so far, then each as it arrives, with
        a keepalive comment after each second of silence.

        Raises:
            WorkerUnavailableError: The broker has no stream for the run.
        """
        try:
            events = self._bus.stream(run_id)
            async for event in events:
                if event is None:
                    yield b": keepalive\n\n"
                else:
                    yield f"data: {json.dumps(event)}\n\n".encode()
        except KeyError:
            raise WorkerUnavailableError(f"stream: run {run_id!r} not found") from None
        except StreamGoneError:
            raise WorkerUnavailableError(f"stream: run {run_id!r} no longer streams") from None

    # -- the calls the bridges make, served under the facade ---------------------

    async def status(self, run_id: str) -> dict[str, Any]:
        """The status body of ``run_id``, as a worker's ``GET /dispatch/{id}`` answers."""
        reply, _ = await self._routed(run_id, protocol.STATUS, "status")
        return cast(dict[str, Any], reply)

    async def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        """Descriptors of the artifacts ``run_id`` produced."""
        reply, _ = await self._routed(run_id, protocol.ARTIFACTS, "artifacts", prefer=bool)
        return reply if isinstance(reply, list) else []

    async def artifact(self, run_id: str, artifact_id: str) -> tuple[dict[str, Any], bytes]:
        """One artifact's descriptor (``mime``, ``filename``) and its bytes."""
        meta, blob = await self._routed(
            run_id, protocol.ARTIFACT, "artifact", extra={"artifact_id": artifact_id}
        )
        return (meta if isinstance(meta, dict) else {}), blob or b""

    def health(self) -> dict[str, Any]:
        """The workers' health as the bridges read a worker's ``/health``.

        ``status`` is ``ok`` while at least one worker is live, and
        ``capabilities`` is what every live worker advertises.
        """
        live = self._bus.live()
        return {
            "status": "ok" if live else "unavailable",
            "capabilities": self._bus.capabilities(),
            "workers": self._bus.describe(),
        }

    # -- routing ----------------------------------------------------------------

    async def _routed(
        self,
        run_id: str,
        kind: bytes,
        what: str,
        *,
        extra: dict[str, Any] | None = None,
        prefer: Callable[[Any], bool] | None = None,
    ) -> tuple[Any, bytes | None]:
        """Ask the run's owner, or every live worker when the owner is not known or not live.

        Of several answers, the first one ``prefer`` accepts wins, else the first.
        """
        body = {"run_id": run_id, **(extra or {})}
        live = self._bus.live()
        owner = self._bus.owner_of(run_id)
        targets = [owner] if owner in live else live
        if not targets:
            raise WorkerUnreachableError(f"{what}: no dispatch worker is connected")
        answers = await asyncio.gather(
            *(self._bus.request(w, kind, body, timeout=RECORD_READ_TIMEOUT) for w in targets),
            return_exceptions=True,
        )
        oks = [a for a in answers if isinstance(a, protocol.Message)]
        if oks:
            chosen = next((m for m in oks if prefer and prefer(m.body.get("body"))), oks[0])
            return chosen.body.get("body"), chosen.blob
        errors = _errors(answers)
        # A 404 from a worker that does not hold the run says nothing about the
        # run; any other failure is the one worth reporting.
        first = next(
            (e for e in errors if not (isinstance(e, protocol.BusError) and e.status == 404)),
            errors[0],
        )
        raise _translate(first, what) from first
