"""HTTP client for the OSPREY dispatch pipeline.

The full sequence:

  1. ``POST {dispatcher}/webhook/{trigger}`` (Bearer EVENT_DISPATCHER_TOKEN,
     question in the JSON body)  ->  202 ``{"dispatch_id": ...}``.
  2. Poll ``GET {dispatcher}/dispatch/{dispatch_id}`` until the dispatcher reports
     ``status == "completed"`` and hands back a ``run_id`` in ``result``. NOTE:
     the dispatcher's "completed" is only the 202 handshake — it means the run was
     *accepted* by the worker, NOT that the agent finished.
  3. Poll ``GET {worker}/dispatch/{run_id}`` (Bearer DISPATCH_WORKER_TOKEN) until
     the *worker* reports ``status in {completed, error}``. This is the real
     terminal state, carrying ``text_output`` / ``error``.

Returns
``{"status", "text_output", "run_id", "error", "artifacts", "input_artifacts",
"failure_class", "num_tool_calls", "error_code"}``.
``error_code`` is the dispatcher's structured pool-error code, surfaced onto the
result when the dispatcher rejected the request with one (e.g.
``input_files_cap_exceeded`` / ``input_files_invalid``); ``None`` otherwise. The
retry policy reads it to force such a rejection non-retryable.
``artifacts`` is the run-status body's artifact field, passed through verbatim:
a current worker returns a list of **descriptor dicts**
(``{"artifact_id", "filename", "source_mime", "delivered_mime", "convertible"}``);
an older worker (during the deploy window) returns bare id strings. Either shape
is normalized downstream by :func:`osprey.bridges.core.artifacts.artifact_ids`
(to ids) or :func:`osprey.bridges.core.artifacts.artifact_descriptors` (to whole
descriptors). ``delivered_mime`` is what the worker *predicted* it would serve,
not what a fetch returns — see that module on why delivery routes on the served
Content-Type instead.
``[]`` when there are no artifacts or the worker predates the field.
``input_artifacts`` is the run-status body's list of inbound files the worker
ingested for this run (``[{entry_id, filename, mime}]``), passed through
verbatim; ``[]`` when there were none or the worker predates the field. The
bridge records these as ``origin="input"`` history descriptors — they are never
republished, so unlike output ``artifacts`` they never carry a public URL.

``failure_class`` / ``num_tool_calls`` are the worker's run-classification
fields, passed through from the run-status body. They drive the retry policy
(is this run worth re-dispatching, and did the agent get far enough to matter).
Both are ``None`` when **UNKNOWN** — an older worker that predates the fields
omits them, and ``.get`` yields ``None``, never ``0``. Bridge-local error
results (transport/handshake/poll-budget failures synthesized without ever
reaching a worker terminal state) default ``failure_class="infrastructure"``
and ``num_tool_calls=None``. Every consumer reads these with ``.get`` so an
older worker skew (worker deployed before this field lands) degrades to
UNKNOWN rather than breaking.

Connection behavior is the deployment's to decide: the default client is built
with ``trust_env=cfg.trust_env`` (default ``False``), so by default it ignores
``HTTP(S)_PROXY`` and reaches the dispatcher and the worker directly — the same
rule :mod:`osprey.bridges.core.health_gate` and
:mod:`osprey.bridges.core.capabilities` follow. A site whose bridge sits behind
an egress proxy sets ``trust_env`` true once, in config. Tests inject their own
:class:`httpx.MockTransport` client.

Like every ``osprey.bridges.core`` module this carries ZERO osprey dispatch
imports.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from .config import TERMINAL_STATUSES, CoreConfig


class DispatchError(Exception):
    """Raised for unexpected HTTP failures talking to the pipeline.

    ``error_code`` carries the dispatcher's structured pool-error code when the
    dispatcher reported ``status="error"`` with one (e.g.
    ``input_files_cap_exceeded`` / ``input_files_invalid`` — a deterministic
    rejection of the request). It is ``None`` for every other failure (transport
    error, handshake glitch, poll-budget exhaustion). :meth:`DispatchClient.run`
    lifts it onto the error result so
    :func:`osprey.bridges.core.retry.is_retryable` can classify such a rejection
    as non-retryable.
    """

    def __init__(self, message: str, *, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


def _error_result(
    run_id: str | None,
    error: str,
    *,
    failure_class: str = "infrastructure",
    num_tool_calls: int | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    """The module's uniform terminal-result shape, error branch.

    Every path out of :meth:`DispatchClient.run`/:meth:`poll_worker` returns
    ``{"status", "text_output", "run_id", "error", "artifacts", "input_artifacts",
    "failure_class", "num_tool_calls", "error_code"}`` — build the error variant in
    ONE place so no
    branch can drift from the contract (the channel poster and the history
    recording both key off these exact fields).

    These are the *bridge-local* failures — a transport error, a failed
    dispatcher/worker handshake, or poll-budget exhaustion — where no worker
    terminal state was ever observed. They are ``failure_class="infrastructure"``
    (the retry policy treats them as worth re-dispatching) with an UNKNOWN
    (``None``) ``num_tool_calls`` since no agent progress is known. A caller may
    override ``failure_class`` for a bridge-local failure it wants classified
    differently, and ``error_code`` when the dispatcher surfaced a structured
    rejection code (the retry policy uses it to force non-retryable regardless of
    the defaulted ``failure_class``).
    """
    return {
        "status": "error",
        "text_output": "",
        "run_id": run_id,
        "error": error,
        "artifacts": [],
        "input_artifacts": [],
        "failure_class": failure_class,
        "num_tool_calls": num_tool_calls,
        "error_code": error_code,
    }


class DispatchClient:
    def __init__(
        self,
        cfg: CoreConfig,
        client: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self._client = client or httpx.Client(timeout=30.0, trust_env=cfg.trust_env)
        self._sleep = sleep
        self._monotonic = monotonic

    # --- step 1: fire the webhook -----------------------------------------
    def fire(self, question: str, extra: dict[str, Any] | None = None) -> str:
        """POST the question to the dispatcher webhook; return the dispatch_id."""
        payload: dict[str, Any] = {"question": question}
        if extra:
            payload.update(extra)
        url = f"{self.cfg.dispatcher_url}/webhook/{self.cfg.trigger}"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.cfg.event_dispatcher_token}"},
        )
        if resp.status_code not in (200, 202):
            raise DispatchError(f"webhook {url} returned {resp.status_code}: {resp.text[:300]}")
        dispatch_id: str | None = resp.json().get("dispatch_id")
        if not dispatch_id:
            raise DispatchError(f"webhook response missing dispatch_id: {resp.text[:300]}")
        return dispatch_id

    # --- step 2: dispatcher handshake -> run_id ---------------------------
    def wait_for_run_id(self, dispatch_id: str, deadline: float) -> str:
        """Poll the dispatcher until it hands back a run_id (or raise)."""
        url = f"{self.cfg.dispatcher_url}/dispatch/{dispatch_id}"
        headers = {"Authorization": f"Bearer {self.cfg.event_dispatcher_token}"}
        while True:
            resp = self._client.get(url, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                status = body.get("status")
                if status == "completed":
                    run_id: str | None = (body.get("result") or {}).get("run_id")
                    if not run_id:
                        raise DispatchError(f"dispatcher completed but no run_id: {body}")
                    return run_id
                if status == "error":
                    raise DispatchError(
                        f"dispatcher reported error: {body.get('error')}",
                        error_code=body.get("error_code"),
                    )
                # else pending -> keep polling
            elif resp.status_code != 404:
                raise DispatchError(
                    f"dispatcher {url} returned {resp.status_code}: {resp.text[:300]}"
                )
            if self._monotonic() >= deadline:
                raise DispatchError(f"timed out waiting for run_id (dispatch_id={dispatch_id})")
            self._sleep(self.cfg.poll_interval)

    # --- step 3: worker poll to terminal ----------------------------------
    def poll_worker(self, run_id: str, deadline: float | None = None) -> dict[str, Any]:
        """Poll the worker until the run reaches a terminal state.

        Returns ``{"status", "text_output", "run_id", "error", "artifacts",
        "input_artifacts", "failure_class", "num_tool_calls", "error_code"}``. On
        budget exhaustion returns a
        synthesized ``status="error"`` result rather than raising, so the caller
        always has something to post.
        """
        if deadline is None:
            deadline = self._monotonic() + self.cfg.poll_budget
        url = f"{self.cfg.worker_url}/dispatch/{run_id}"
        headers = {"Authorization": f"Bearer {self.cfg.dispatch_worker_token}"}
        while True:
            resp = self._client.get(url, headers=headers)
            if resp.status_code == 200:
                body = resp.json()
                status = body.get("status")
                if status in TERMINAL_STATUSES:
                    return {
                        "status": status,
                        "text_output": body.get("text_output", "") or "",
                        "run_id": run_id,
                        "error": body.get("error"),
                        # Missing key OR an explicit null -> [] so an older
                        # worker image (deployed independently) degrades
                        # gracefully instead of dropping the whole result. The
                        # worker does serialize explicit nulls (see "error").
                        # Whatever shape it carries (descriptor dicts on a
                        # current worker, bare id strings on an older one) is
                        # passed through verbatim and normalized downstream by
                        # artifacts.artifact_ids.
                        "artifacts": body.get("artifacts") or [],
                        # Inbound files the worker ingested for this run, passed
                        # through verbatim as ``[{entry_id, filename, mime}]``.
                        # Missing key OR explicit null -> [] so a worker predating
                        # the field (or a run with no inputs) degrades to "no
                        # input artifacts" rather than dropping the result. The
                        # bridge records these as ``origin="input"`` history
                        # descriptors; they are never republished, so never carry
                        # a public URL.
                        "input_artifacts": body.get("input_artifacts") or [],
                        # Worker run-classification, passed through as-is. An
                        # older worker omits them -> None meaning UNKNOWN (never
                        # coerce a missing count to 0). The retry policy reads
                        # both to decide whether re-dispatching is worthwhile.
                        "failure_class": body.get("failure_class"),
                        "num_tool_calls": body.get("num_tool_calls"),
                        # Structured pool-error code if the worker terminal body
                        # carries one; None keeps the contract uniform otherwise.
                        "error_code": body.get("error_code"),
                    }
                # pending / running -> keep polling
            elif resp.status_code != 404:
                # 404 can be a brief race (run just registered); anything else is fatal.
                raise DispatchError(f"worker {url} returned {resp.status_code}: {resp.text[:300]}")
            if self._monotonic() >= deadline:
                return _error_result(
                    run_id,
                    f"bridge timed out after {self.cfg.poll_budget:.0f}s polling run {run_id}",
                )
            self._sleep(self.cfg.poll_interval)

    # --- single non-polling status probe ----------------------------------
    def status(self, run_id: str) -> dict[str, Any] | None:
        """One-shot ``GET {worker}/dispatch/{run_id}`` — no polling, no deadline.

        Returns the raw worker run-status body (which may be **non-terminal**,
        e.g. ``{"status": "pending"}``) so a caller reconciling a persisted
        ``run_id`` on restart can decide for itself whether the run is still in
        flight or has reached a terminal state. Unlike :meth:`poll_worker` this
        does not synthesize the uniform result contract — it hands back exactly
        what the worker reported.

        ``None`` on 404: the worker has no such run. That is a *signal* (the run
        never registered, or its record has aged out), not an error — the caller
        distinguishes "gone" from a live status body. Any other non-200 raises
        :class:`DispatchError`, matching :meth:`poll_worker`.
        """
        url = f"{self.cfg.worker_url}/dispatch/{run_id}"
        headers = {"Authorization": f"Bearer {self.cfg.dispatch_worker_token}"}
        resp = self._client.get(url, headers=headers)
        if resp.status_code == 200:
            body: dict[str, Any] = resp.json()
            return body
        if resp.status_code == 404:
            return None
        raise DispatchError(f"worker {url} returned {resp.status_code}: {resp.text[:300]}")

    # --- full sequence -----------------------------------------------------
    def run(
        self,
        question: str,
        extra: dict[str, Any] | None = None,
        *,
        on_run_id: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Fire the webhook and poll all the way to a terminal worker result.

        Always returns ``{"status", "text_output", "run_id", "error", "artifacts",
        "input_artifacts", "failure_class", "num_tool_calls", "error_code"}``.
        Any failure in the sequence — a dispatcher/worker non-2xx (``DispatchError``)
        or a transient network error (``httpx.HTTPError``: ConnectError,
        ReadTimeout, ...) — is converted to an ``error`` result so the caller can
        always post *something* back to the user rather than escaping unhandled.

        ``on_run_id`` (if given) is invoked with the run_id the moment the
        dispatcher handshake yields one, BEFORE the long worker poll. This lets
        the caller persist a non-terminal ``run_id`` so a crash mid-poll can be
        reconciled on restart instead of silently dropping the question.
        """
        try:
            # The handshake gets its own budget; the worker poll below starts a
            # fresh budget so a slow handshake can never eat into the worker's
            # full poll window (which must outlast the worker's own run cap).
            handshake_deadline = self._monotonic() + self.cfg.poll_budget
            dispatch_id = self.fire(question, extra)
            run_id = self.wait_for_run_id(dispatch_id, handshake_deadline)
        except (DispatchError, httpx.HTTPError) as exc:
            # A dispatcher pool-error may carry a structured error_code (an
            # input_files rejection); lift it onto the result so the retry policy
            # can classify it. Ordinary transport/handshake failures have none.
            return _error_result(None, str(exc), error_code=getattr(exc, "error_code", None))
        if on_run_id is not None:
            on_run_id(run_id)
        try:
            return self.poll_worker(run_id)  # fresh budget (deadline=None)
        except (DispatchError, httpx.HTTPError) as exc:
            return _error_result(run_id, str(exc))
