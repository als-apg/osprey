"""Propagation of validated session-tier plans into the worker namespace.

A shipped/preset/facility plan reaches the queueserver worker on its own: the
worker's startup module (``qserver_startup.py``) scans the same layered plan
catalog the bridge serves and exposes every entry as a named plan function. A
*session* plan cannot travel that road. It is authored through
``POST /plans/session`` into a bridge-owned directory
(``session_dir.resolve_session_plan_dir()``) that the queueserver container
does not mount, and — unlike every higher tier — it is not operator-trusted:
it is trusted only for the exact bytes that passed
``plan_validation.validate_plan``.

This module is the one road it does travel, and the gate on that road:

**Upload happens only on a PASS.** ``POST /plans/session`` stays write-only —
authoring a file must never execute it. Only when ``POST /plans/validate``
records a passing validation for the file's *current* content hash does
:meth:`SessionPlanUploader.upload_validated` push a script into the worker
namespace. Editing the file changes its hash, which drops it back to "no
record" (see ``validation_record.py``), so an edited file is not re-uploaded
until it is re-validated — and the copy sitting in the namespace is the one
that passed, never the edit.

**The namespace is rebuilt on every environment cycle.** Queueserver builds
the worker namespace by executing the startup module, which knows nothing
about session plans, so every environment open/reload drops the whole
uploaded set. :meth:`SessionPlanUploader.sync_namespace` is the repair path:
it re-uploads every currently-validated session plan the namespace is missing.
:meth:`SessionPlanUploader.reload_environment` is the same thing around a
deliberate cycle, and refuses outright while the queue is active — a reload
mid-plan would abort work an operator is watching.

**Enqueue and start re-check, and refuse rather than repair.** The armed path
goes through :meth:`SessionPlanUploader.check_ready` (module-level shorthand:
:func:`check_session_plan_ready`), which admits a session plan only when all
three of these hold *right now*: the file's current content hash has a passing
record, that same hash is what this bridge uploaded into the current
namespace, and the manager still lists the plan as allowed. Any gap is a
machine-readable refusal (:class:`SessionPlanNotReadyError`, carrying one of
the ``REASON_*`` codes) telling the caller to re-validate, never a silent
re-upload: an item that reaches the RunEngine must correspond to bytes a
human-visible validation actually passed.

That triple is what makes a bridge restart fail closed. Validation records are
in-memory (``validation_record.py``) and so is the uploaded-hash bookkeeping
here, while queue items are Redis-durable — so a restarted bridge finds a
queued session-plan item with no record and no bookkeeping and refuses it,
even though the worker namespace still holds the plan from before the
restart. The inverse case is covered too: a namespace plan tagged
:data:`SESSION_PLAN_MODULE` whose file is gone from this bridge is refused
rather than treated as a catalog plan.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from pathlib import Path
from threading import Lock
from typing import Any

from .plan_fields import declared_channels
from .plan_validation import hash_plan_body
from .session_dir import resolve_session_plan_dir
from .validation_record import ValidationRecordStore, validation_records

logger = logging.getLogger("osprey.services.bluesky_bridge.session_upload")

# `__module__` stamped on every plan function this module installs, and the
# marker `check_ready` reads back out of the manager's `plans_allowed`
# description (queueserver publishes a plan's `__module__` as its `module`
# key). It is how the bridge tells "this namespace entry is a session plan I
# uploaded" from "this is a catalog plan the startup module built" — the
# distinction a restarted bridge needs to refuse a namespace leftover it can
# no longer vouch for. Deliberately not an importable module path: nothing
# imports it, it only has to be a stable, recognizable string.
SESSION_PLAN_MODULE = "osprey_session_plan"

# Refusal codes. Every one of these means "this session plan is not admissible
# right now"; the route layer puts the code on the wire and the detail explains
# the fix.
REASON_UNVALIDATED = "session_plan_unvalidated"
REASON_NOT_IN_NAMESPACE = "session_plan_not_in_namespace"
REASON_UPLOAD_FAILED = "session_plan_upload_failed"
REASON_ENVIRONMENT_BUSY = "environment_busy"

# Every refusal ends with the same instruction, because every refusal has the
# same fix: re-validating re-records the current bytes and re-uploads them.
_REVALIDATE_HINT = "Re-validate it (POST /plans/validate) before enqueuing it."

# Bounds on waiting for the manager to finish executing an uploaded script.
# The script only defines one function, so completion is near-immediate; the
# window exists so a wedged manager fails closed instead of hanging a request.
_UPLOAD_POLL_INTERVAL = 0.25
_UPLOAD_POLL_ATTEMPTS = 120


class SessionPlanError(Exception):
    """Base class for every refusal this module reports.

    ``reason`` is the stable, machine-readable code; the message is the human
    half. Mirrors ``queue_backend.py``'s error convention so the route layer
    handles both the same way.
    """

    reason = "session_plan_error"

    def __init__(self, message: str, *, plan: str | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.plan = plan
        self.detail = message
        if reason is not None:
            self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        """The refusal as the JSON object a route puts on the wire."""
        # The queue surface's canonical branch key is "code" — its routes add
        # `"code": reason` alongside these fields (see queue.py's _session_refusal).
        return {"reason": self.reason, "detail": self.detail, "plan": self.plan}


class SessionPlanNotReadyError(SessionPlanError):
    """A session plan may not be enqueued or started as things currently stand.

    Raised by :meth:`SessionPlanUploader.check_ready` with one of
    :data:`REASON_UNVALIDATED` or :data:`REASON_NOT_IN_NAMESPACE`.
    """

    reason = REASON_UNVALIDATED


class SessionPlanUploadError(SessionPlanError):
    """A script upload was rejected, failed in the worker, or never completed."""

    reason = REASON_UPLOAD_FAILED


class SessionPlanEnvironmentBusyError(SessionPlanError):
    """An environment reload was asked for while the queue is running a plan."""

    reason = REASON_ENVIRONMENT_BUSY


# ---------------------------------------------------------------------------
# Worker side: what an uploaded script actually does
# ---------------------------------------------------------------------------


def build_upload_script(name: str, source: str) -> str:
    """The script the bridge hands to ``script_upload`` for one session plan.

    Three lines, on purpose. Queueserver executes an uploaded script with the
    worker namespace as *both* globals and locals, so the script can hand
    ``globals()`` — the live namespace, devices and all — straight to
    :func:`install_session_plan` and let real, tested code do the work, rather
    than templating the wrapper logic into a string. The plan source travels
    as a ``repr`` literal, which is injection-proof by construction: no source
    text can end the literal early.

    The bridge and the queueserver container run the same image, so the
    ``osprey`` import on the worker side resolves to this same module at the
    same version.
    """
    return (
        "from osprey.services.bluesky_bridge import session_upload as _osprey_session_upload\n"
        f"_osprey_session_upload.install_session_plan(globals(), {name!r}, {source!r})\n"
        "del _osprey_session_upload\n"
    )


def install_session_plan(
    namespace: MutableMapping[str, Any], name: str, source: str
) -> Callable[..., Iterator[Any]]:
    """Define one session plan in the worker namespace. Runs in the RE worker.

    Executes ``source`` in a private dict rather than in ``namespace``: a plan
    file's module-level names (``build_plan``, ``PARAMS``, ``PLAN_METADATA``)
    are the same for every plan file, so executing two of them into the shared
    namespace would leave each one's wrapper looking at the other's schema.
    Only the wrapper function lands in ``namespace``, under ``name``.

    The wrapper is a generator function taking ``**kwargs`` only, matching
    ``qserver_startup._make_plan_function``: queueserver's namespace scan
    recognizes generator functions and nothing else, and the queue item's
    ``kwargs`` are the plan's ``PARAMS`` fields, so the plan's own pydantic
    model stays the single validation authority. As there, that validation
    therefore happens on first iteration — when the RunEngine starts the item
    — not at enqueue time.

    Devices are resolved from ``namespace`` with queueserver's own
    ``is_device`` predicate, so the wrapper closes over exactly the device set
    the manager itself considers available in this worker. What one run may
    reach is narrower still: ``build_plan`` receives only the channels the
    validated params declare movable or readable (:func:`_declared_devices`),
    and a device referenced through a parameter carrying no role declaration is
    absent from that mapping — a legible ``KeyError`` naming both the reason and
    the worker's devices, not a silent resolution.

    Args:
        namespace: The worker namespace (``globals()`` inside the script).
        name: The plan name, which is also the namespace key.
        source: The validated plan file's full content.

    Returns:
        The installed plan function.

    Raises:
        ValueError: ``source`` does not satisfy the plan-file contract — no
            callable ``build_plan``, or a ``PARAMS`` that is not a pydantic
            model. The manager surfaces this as a failed upload task.
    """
    from pydantic import BaseModel

    module_namespace: dict[str, Any] = {"__name__": f"{SESSION_PLAN_MODULE}.{name}"}
    # `dont_inherit=True` matters: `compile` otherwise applies THIS module's
    # `from __future__ import annotations` to the plan file, turning every
    # annotation in it into a string. A plan's `PARAMS` model is then left
    # incomplete for any annotation pydantic cannot resolve from builtins — the
    # role-typed channel fields among them, since the exec'd module is not in
    # `sys.modules` for pydantic to look their names up in — and the plan fails
    # at its first enqueue with "`PARAMS` is not fully defined". The plan file's
    # own `__future__` imports still apply; only this module's leak is cut.
    code = compile(source, f"<session plan {name}>", "exec", dont_inherit=True)
    exec(code, module_namespace)  # noqa: S102

    build_plan = module_namespace.get("build_plan")
    if not callable(build_plan):
        raise ValueError(f"session plan {name!r}: missing required build_plan callable")

    params_model = module_namespace.get("PARAMS")
    if params_model is None:

        class _EmptyParams(BaseModel):
            """Schema for a session plan that declares no parameters."""

        params_model = _EmptyParams
    elif not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
        raise ValueError(
            f"session plan {name!r}: PARAMS must be a pydantic BaseModel subclass, "
            f"got {params_model!r}"
        )

    devices = collect_devices(namespace)

    def plan_function(**kwargs: Any) -> Iterator[Any]:
        params = params_model.model_validate(kwargs)
        try:
            plan = build_plan(_declared_devices(params_model, params, devices), params)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "<unknown>"
            if missing in devices:
                raise KeyError(
                    f"session plan {name!r} referenced device {missing!r}, which its parameters "
                    f"do not declare as a movable or readable channel — a plan resolves only "
                    f"the channels it declares; available devices: {sorted(devices)}"
                ) from exc
            raise KeyError(
                f"session plan {name!r} referenced device {missing!r}, which this worker "
                f"did not build; available devices: {sorted(devices)}"
            ) from exc
        return (yield from plan)

    plan_function.__name__ = name
    plan_function.__qualname__ = name
    plan_function.__module__ = SESSION_PLAN_MODULE
    plan_function.__doc__ = _plan_docstring(name, module_namespace)
    # REAL annotation objects, not the PEP 563 strings this module's
    # `from __future__ import annotations` would otherwise leave behind. The
    # manager builds each plan's validation model straight off
    # `inspect.signature`, wrapping a VAR_KEYWORD annotation as
    # `typing.Dict[str, <annotation>]` for `pydantic.create_model`; a STRING
    # there is an unresolvable forward reference, and the manager then refuses
    # EVERY enqueue of this plan with "`Model` is not fully defined; you should
    # define `Any`" — authoring, validation and upload all succeed and only the
    # enqueue fails, which reads like a plan problem and is not.
    #
    # `qserver_startup._make_plan_function` carries the identical line for the
    # identical reason. These are the only two places a `**kwargs`-only plan
    # wrapper is handed to the manager, and
    # `test_no_plan_wrapper_ships_pep563_string_annotations` enumerates both.
    plan_function.__annotations__ = {"kwargs": Any, "return": Iterator[Any]}

    namespace[name] = plan_function
    return plan_function


def collect_devices(namespace: MutableMapping[str, Any]) -> dict[str, Any]:
    """The devices in a worker namespace, keyed by their namespace name.

    Uses the manager's own ``is_device`` so this can never disagree with the
    device list the manager publishes for the same namespace. Imported here
    rather than at module scope: ``bluesky_queueserver`` is the *manager's*
    package, present wherever this function runs (the RE worker) but not
    something the bridge's own import path should pull in.
    """
    from bluesky_queueserver.manager.profile_ops import is_device

    return {key: value for key, value in namespace.items() if is_device(value)}


def _declared_devices(schema: Any, params: Any, devices: dict[str, Any]) -> dict[str, Any]:
    """The devices one session plan may resolve: the channels its params declare.

    Same bound `qserver_startup` puts on a catalog plan, for the same reason: a
    plan's ``PARAMS`` model declares which channels it moves and which it reads,
    and that declaration is its whole claim on this worker. The mapping handed
    to ``build_plan`` holds those channels and nothing else, so a session plan
    cannot reach a device it never declared — and a plan file with no ``PARAMS``
    at all declares nothing and so resolves nothing.

    Names the worker did not build are left out rather than raising; the
    wrapper's own handler is what turns either miss into a legible failure.
    """
    return {name: devices[name] for name in declared_channels(schema, params) if name in devices}


def _plan_docstring(name: str, module_namespace: dict[str, Any]) -> str:
    """The wrapper's docstring: the plan file's own description when it has one.

    Queueserver parses this into the plan description panels and the agent
    read, so it is worth carrying across rather than inventing.
    """
    metadata = module_namespace.get("PLAN_METADATA")
    if isinstance(metadata, dict):
        description = metadata.get("description")
        if isinstance(description, str) and description.strip():
            return description
    return f"OSPREY session plan {name!r}."


# ---------------------------------------------------------------------------
# Bridge side: when an upload happens, and what the armed path re-checks
# ---------------------------------------------------------------------------


def _default_backend_getter() -> Any:
    """The bridge's process-wide ``QueueBackend``, resolved on each call.

    Late import, and resolved per call rather than captured: ``app.py``
    imports this module, and a test (or a redeploy of the backend via
    ``app.set_queue_backend``) must not leave this module holding a stale
    handle.
    """
    from .app import get_queue_backend

    return get_queue_backend()


class SessionPlanUploader:
    """Keeps the worker namespace in step with the validated session-plan set.

    Owns exactly one piece of state: which content hash this bridge uploaded
    under which plan name, for the *current* worker namespace. That is not a
    cache of anything the manager knows — the manager can say a plan named
    ``x`` exists, but only the bridge knows which bytes it put there — and it
    is deliberately dropped whenever the namespace is rebuilt.

    Args:
        backend_getter: Returns the ``QueueBackend`` to talk to. Defaults to
            the bridge's process-wide backend.
        records: The validation-record store to consult. Defaults to the
            bridge's process-wide store, which the validate route writes.
        session_dir_resolver: Returns the session-plan directory.
        poll_interval: Seconds between polls of an upload task's result.
        poll_attempts: How many such polls before the upload is declared failed.
    """

    def __init__(
        self,
        *,
        backend_getter: Callable[[], Any] = _default_backend_getter,
        records: ValidationRecordStore = validation_records,
        session_dir_resolver: Callable[[], Path] = resolve_session_plan_dir,
        poll_interval: float = _UPLOAD_POLL_INTERVAL,
        poll_attempts: int = _UPLOAD_POLL_ATTEMPTS,
    ) -> None:
        self._backend_getter = backend_getter
        self._records = records
        self._session_dir_resolver = session_dir_resolver
        self._poll_interval = poll_interval
        self._poll_attempts = poll_attempts
        self._uploaded: dict[str, str] = {}
        self._lock = Lock()

    # ------------------------------------------------------------- inspection

    def uploaded_hashes(self) -> dict[str, str]:
        """Plan name to the content hash uploaded into the current namespace."""
        with self._lock:
            return dict(self._uploaded)

    def forget_namespace(self) -> None:
        """Drop the uploaded-hash bookkeeping: the worker namespace is gone.

        Called whenever the environment is known to have closed. After this,
        every session plan reads as "not in the namespace" until it is
        re-uploaded, which is the fail-closed answer.
        """
        with self._lock:
            self._uploaded.clear()

    def validated_session_plans(self) -> dict[str, str]:
        """Every session plan file whose CURRENT content has a passing record.

        Re-read and re-hashed from disk on every call — a file edited since it
        was validated has a different hash, so it simply drops out of this set
        rather than lingering as a stale entry.
        """
        validated: dict[str, str] = {}
        for path in sorted(self._session_dir_resolver().glob("*.py")):
            name = path.stem
            content = self._read(path)
            if content is None:
                continue
            content_hash = hash_plan_body(content)
            if self._records.has_passing_record(content_hash):
                validated[name] = content_hash
        return validated

    # ----------------------------------------------------------------- upload

    async def upload_validated(self, name: str) -> str:
        """Upload one session plan whose current content has a passing record.

        The validate route's hook (via :func:`upload_after_validation`) on a
        PASS. Re-reads and re-hashes the file rather than trusting a hash
        passed in, so the bytes uploaded are provably the bytes on disk now.

        Args:
            name: The session plan's name, which is also its file stem.

        Returns:
            The content hash that was uploaded.

        Raises:
            SessionPlanNotReadyError: No such file, or its current content has
                no passing validation record.
            SessionPlanUploadError: The manager refused the script, the worker
                raised while executing it, or it never completed.
            QueueBackendError: The manager could not be reached.
        """
        path = self._plan_path(name)
        content = self._read(path)
        if content is None:
            raise SessionPlanNotReadyError(
                f"No session plan named {name!r} exists on this bridge.", plan=name
            )
        content_hash = hash_plan_body(content)
        if not self._records.has_passing_record(content_hash):
            raise SessionPlanNotReadyError(
                f"Session plan {name!r} has no passing validation record for its current "
                f"content. {_REVALIDATE_HINT}",
                plan=name,
            )
        await self._upload(name, content, content_hash)
        return content_hash

    async def sync_namespace(self) -> list[str]:
        """Re-upload every validated session plan the worker namespace lacks.

        The repair path for an environment cycle: the namespace is rebuilt
        from the startup module, which knows nothing about session plans, so
        every uploaded plan is gone and the whole validated set has to go back
        in. Idempotent — a plan the namespace already holds at the hash this
        bridge uploaded is skipped — so it is safe to call after any open,
        whether or not anything was actually lost.

        A closed environment is not an error: there is no namespace to sync,
        the bookkeeping is dropped, and nothing is uploaded.

        Returns:
            The names re-uploaded, in upload order.

        Raises:
            QueueBackendError: The manager could not be reached.
        """
        backend = self._backend_getter()
        from .queue_backend import is_environment_open

        if not is_environment_open(await backend.status(reload=True)):
            self.forget_namespace()
            return []

        allowed = await self._allowed_plans(backend)
        validated = self.validated_session_plans()
        with self._lock:
            self._uploaded = {
                name: value for name, value in self._uploaded.items() if name in validated
            }
            uploaded_now = dict(self._uploaded)

        restored: list[str] = []
        for name, content_hash in validated.items():
            if name in allowed and uploaded_now.get(name) == content_hash:
                continue
            content = self._read(self._plan_path(name))
            if content is None or hash_plan_body(content) != content_hash:
                # Edited out from under the scan a moment ago; the next
                # validation is what puts it back.
                continue
            await self._upload(name, content, content_hash)
            restored.append(name)

        if restored:
            logger.info(
                "session_upload: re-uploaded %d session plan(s) after an environment cycle: %s",
                len(restored),
                ", ".join(restored),
            )
        return restored

    async def reload_environment(self) -> list[str]:
        """Cycle the worker environment, then restore the validated session set.

        Refused while the queue is active: a reload tears down the RunEngine,
        which would abort a plan an operator is watching. The refusal is
        checked here *and* by ``QueueBackend.close_environment``; the check
        here exists so the caller gets this module's reason code rather than a
        generic rejection.

        Returns:
            The session plans re-uploaded into the new namespace.

        Raises:
            SessionPlanEnvironmentBusyError: A plan is running or about to.
            QueueBackendError: The manager could not be reached, or the
                environment would not come back up.
        """
        backend = self._backend_getter()
        from .queue_backend import is_queue_active

        status = await backend.status(reload=True)
        if is_queue_active(status):
            raise SessionPlanEnvironmentBusyError(
                "Refusing to reload the worker environment while the queue is active "
                f"(manager state {status.get('manager_state')!r}). Stop the queue first."
            )

        await backend.close_environment()
        self.forget_namespace()
        await backend.ensure_environment()
        return await self.sync_namespace()

    # ------------------------------------------------------------------ gate

    async def check_ready(self, name: str) -> None:
        """Admit or refuse ``name`` on the armed (enqueue / queue-start) path.

        A no-op for any name this bridge has no session plan file for *and*
        that the worker namespace does not hold as a session plan: catalog
        tiers (shipped/preset/facility) are operator-trusted by construction
        and carry no record gate.

        For a session plan, all three must hold right now:

        1. the file's current content has a passing validation record;
        2. that same hash is what this bridge uploaded into the current
           namespace;
        3. the manager still lists the plan as allowed.

        Refuses rather than repairing. A missing or mismatched record is
        exactly the state a bridge restart leaves behind for a Redis-durable
        queue item, and re-uploading on its behalf would mean executing bytes
        no live validation vouches for.

        Args:
            name: The plan name on the queue item.

        Raises:
            SessionPlanNotReadyError: Any of the three checks fails.
            QueueBackendError: The manager could not be reached.
        """
        allowed = await self._allowed_plans(self._backend_getter())
        self._check_one(name, allowed)

    async def check_many_ready(self, names: Iterable[str]) -> None:
        """Run :meth:`check_ready` over several plan names, in the order given.

        For the queue-start path, which arms every item already in the queue,
        not one name: the manager's allowed-plan list is fetched once and each
        name checked against it, so starting a long queue costs one round trip
        rather than one per item. Raises on the first name that is not
        admissible — a start is all-or-nothing.
        """
        allowed = await self._allowed_plans(self._backend_getter())
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            self._check_one(name, allowed)

    def _check_one(self, name: str, allowed: dict[str, Any]) -> None:
        """The gate itself, against an already-fetched allowed-plan list."""
        in_namespace_as_session = self._is_session_namespace_plan(allowed.get(name))

        content = self._read(self._plan_path(name))
        if content is None:
            if in_namespace_as_session:
                # The worker still holds a plan this bridge uploaded, but the
                # file behind it is gone (an ephemeral session directory
                # across a bridge restart, or a deleted plan). Nothing here can
                # vouch for what is in the namespace.
                raise SessionPlanNotReadyError(
                    f"The worker still holds a session plan named {name!r}, but this bridge "
                    f"has no session plan file for it, so it cannot vouch for what would "
                    f"run. Re-author it (POST /plans/session) and validate it.",
                    plan=name,
                    reason=REASON_NOT_IN_NAMESPACE,
                )
            return

        content_hash = hash_plan_body(content)
        if not self._records.has_passing_record(content_hash):
            raise SessionPlanNotReadyError(
                f"Session plan {name!r} has no passing validation record for its current "
                f"content — it was never validated, was edited since it passed, or its "
                f"record was lost when the bridge restarted. {_REVALIDATE_HINT}",
                plan=name,
            )

        if self.uploaded_hashes().get(name) != content_hash or not in_namespace_as_session:
            raise SessionPlanNotReadyError(
                f"Session plan {name!r} passed validation but its validated content is not "
                f"in the worker namespace — the environment was rebuilt, or its upload did "
                f"not land. {_REVALIDATE_HINT}",
                plan=name,
                reason=REASON_NOT_IN_NAMESPACE,
            )

    # --------------------------------------------------------------- plumbing

    def _plan_path(self, name: str) -> Path:
        return self._session_dir_resolver() / f"{name}.py"

    @staticmethod
    def _read(path: Path) -> str | None:
        """The file's text, or `None` if it is not there / not readable text."""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    @staticmethod
    def _is_session_namespace_plan(description: Any) -> bool:
        """Whether a ``plans_allowed`` entry is a plan this module installed."""
        return isinstance(description, dict) and description.get("module") == SESSION_PLAN_MODULE

    @staticmethod
    async def _allowed_plans(backend: Any) -> dict[str, Any]:
        """The manager's allowed-plan descriptions, keyed by plan name."""
        reply = await backend.plans_allowed()
        allowed = reply.get("plans_allowed")
        return allowed if isinstance(allowed, dict) else {}

    async def _upload(self, name: str, content: str, content_hash: str) -> None:
        """Run one plan's install script in the worker and wait for the outcome.

        The uploaded-hash bookkeeping is written only after the manager
        reports the task completed successfully, so a failed or timed-out
        upload leaves the plan reading as "not in the namespace" rather than
        as installed.
        """
        backend = self._backend_getter()
        reply = await backend.upload_script(build_upload_script(name, content))
        task_uid = reply.get("task_uid")
        if not task_uid:
            raise SessionPlanUploadError(
                f"The queue server accepted no upload task for session plan {name!r}: "
                f"{reply.get('msg') or reply!r}",
                plan=name,
            )

        result = await self._await_task(backend, task_uid, name)
        if not result.get("success"):
            detail = result.get("msg") or result.get("return_value") or "no detail reported"
            raise SessionPlanUploadError(
                f"The worker failed to define session plan {name!r}: {detail}", plan=name
            )

        with self._lock:
            self._uploaded[name] = content_hash
        logger.info(
            "session_upload: defined session plan %r in the worker (%s)", name, content_hash
        )

    async def _await_task(self, backend: Any, task_uid: str, name: str) -> dict[str, Any]:
        """Poll one manager task to completion, within a bounded window.

        A task that has expired out of the manager's result retention reads as
        ``not_found``; that is treated as a failure rather than a success,
        because a result this module never saw is a result it cannot vouch for.
        """
        for _ in range(self._poll_attempts):
            reply = await backend.task_result(task_uid)
            status = reply.get("status")
            if status == "completed":
                result = reply.get("result")
                return result if isinstance(result, dict) else {}
            if status == "not_found":
                raise SessionPlanUploadError(
                    f"The queue server lost the result of the upload task for session plan "
                    f"{name!r} before the bridge could read it.",
                    plan=name,
                )
            await asyncio.sleep(self._poll_interval)

        raise SessionPlanUploadError(
            f"The upload of session plan {name!r} did not complete within "
            f"{self._poll_interval * self._poll_attempts:.0f}s.",
            plan=name,
        )


# ---------------------------------------------------------------------------
# Process-wide uploader + the two hooks the rest of the bridge calls
# ---------------------------------------------------------------------------

_uploader: SessionPlanUploader | None = None


def get_session_uploader() -> SessionPlanUploader:
    """The bridge's single :class:`SessionPlanUploader`, built on first use.

    One per process, because the uploaded-hash bookkeeping describes one
    worker namespace.
    """
    global _uploader
    if _uploader is None:
        _uploader = SessionPlanUploader()
    return _uploader


def set_session_uploader(uploader: SessionPlanUploader | None) -> None:
    """Override the uploader :func:`get_session_uploader` returns.

    Passing `None` clears it, so the next call builds a fresh one.
    """
    global _uploader
    _uploader = uploader


async def check_session_plan_ready(name: str) -> None:
    """Armed-path gate for one plan name — the queue routes' hook.

    Shorthand for :meth:`SessionPlanUploader.check_ready` on the process-wide
    uploader. Call it before enqueuing an item and before starting the queue;
    it does nothing for catalog plans and raises
    :class:`SessionPlanNotReadyError` (``.to_dict()`` is the wire form) for a
    session plan that is not currently admissible.
    """
    await get_session_uploader().check_ready(name)


async def check_session_plans_ready(names: Iterable[str]) -> None:
    """Armed-path gate for a whole queue — the queue-start route's hook.

    Shorthand for :meth:`SessionPlanUploader.check_many_ready` on the
    process-wide uploader: pass the plan names of every item the start would
    drain (pending plus running). Raises
    :class:`SessionPlanNotReadyError` for the first one that is not currently
    admissible, so a queue holding one stale session plan does not start.
    """
    await get_session_uploader().check_many_ready(names)


async def upload_after_validation(name: str) -> dict[str, Any]:
    """Push a just-validated session plan to the worker. Never raises.

    The validate route's hook, called only when validation passed. It reports
    rather than raises because the validation result is the answer that route
    owes its caller: a deployment with no queue server at all (the browse-only
    mock case) validates plans perfectly well and simply has nowhere to upload
    them, and a manager that is momentarily unreachable must not turn a PASS
    into an error. An upload that did not happen is not a safety gap — the
    armed path re-checks namespace presence (:func:`check_session_plan_ready`)
    and refuses on its own.

    Returns:
        ``{"uploaded": bool, "reason": str | None, "detail": str | None}``.
    """
    try:
        await get_session_uploader().upload_validated(name)
    except SessionPlanError as exc:
        logger.warning("session_upload: could not upload session plan %r: %s", name, exc)
        return {"uploaded": False, "reason": exc.reason, "detail": exc.detail}
    except Exception as exc:  # queue-backend errors, and anything else the client raises
        reason = getattr(exc, "reason", "upload_unavailable")
        logger.warning(
            "session_upload: could not upload session plan %r to the queue server: %s", name, exc
        )
        return {"uploaded": False, "reason": reason, "detail": str(exc)}
    return {"uploaded": True, "reason": None, "detail": None}
