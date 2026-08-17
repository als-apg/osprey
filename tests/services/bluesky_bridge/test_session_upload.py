"""Contract tests for session-plan propagation (`session_upload.py`).

Two halves, matching the module's two halves.

The *worker* half is exercised for real: the script the bridge would hand to
``script_upload`` is executed here exactly as queueserver executes it (globals
and locals both being the namespace dict), against a stand-in namespace
holding stand-in devices. So these tests pin what actually lands in the worker
namespace — a generator function under the plan's own name, closed over the
worker's device set and its own schema.

The *bridge* half runs against a scripted `QueueBackend` stand-in — no
manager, no Redis, no container — pinning the policy this module owns: upload
only on a passing record for the current bytes, re-upload the whole validated
set after an environment cycle, and refuse (never repair) on the armed path.
The three states the plan calls out by name have a test each: edited after
upload, a bridge restart that left a Redis-durable item behind, and an
environment cycle that emptied the namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.session_upload import (
    REASON_NOT_IN_NAMESPACE,
    REASON_UNVALIDATED,
    SESSION_PLAN_MODULE,
    SessionPlanEnvironmentBusyError,
    SessionPlanNotReadyError,
    SessionPlanUploader,
    SessionPlanUploadError,
    build_upload_script,
    install_session_plan,
    set_session_uploader,
    upload_after_validation,
)
from osprey.services.bluesky_bridge.validation_record import ValidationRecordStore

PLAN_SOURCE = """PLAN_METADATA = {
    "name": "sample_plan",
    "description": "Step a motor and read it back \\N{DEGREE SIGN} 'quoted' \\\\ backslash.",
    "category": "accelerator",
    "required_devices": ["motor"],
    "writes": True,
}

from pydantic import BaseModel


class PARAMS(BaseModel):
    steps: int


def build_plan(devices, params):
    yield ("move", devices["motor"], params.steps)
"""

OTHER_PLAN_SOURCE = """PLAN_METADATA = {"name": "other_plan"}

from pydantic import BaseModel


class PARAMS(BaseModel):
    label: str


def build_plan(devices, params):
    yield ("label", params.label)
"""


class FakeDevice:
    """Stand-in the manager's own ``is_device`` predicate recognizes.

    ``bluesky_queueserver``'s ``is_device`` accepts anything with a
    ``children`` attribute (its ophyd-async escape hatch), which is what makes
    a stub like this usable in place of a connected ophyd-async device.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.children: dict[str, Any] = {}


def worker_namespace(**devices: Any) -> dict[str, Any]:
    """A stand-in worker namespace: an ``RE`` placeholder plus some devices."""
    namespace: dict[str, Any] = {"RE": object(), "__name__": "__main__"}
    namespace.update(devices)
    return namespace


def run_upload_script(namespace: dict[str, Any], name: str, source: str) -> None:
    """Execute an upload script the way queueserver's worker executes it.

    ``load_script_into_existing_nspace`` calls ``exec(script, nspace, nspace)``
    — globals and locals are both the live worker namespace — which is the
    property `build_upload_script` relies on when it passes ``globals()``.
    """
    script = build_upload_script(name, source)
    exec(script, namespace, namespace)  # noqa: S102


# ---------------------------------------------------------------------------
# Worker half: what lands in the namespace
# ---------------------------------------------------------------------------


def test_upload_script_installs_a_generator_plan_under_its_own_name() -> None:
    import inspect

    namespace = worker_namespace(motor=FakeDevice("motor"))
    run_upload_script(namespace, "sample_plan", PLAN_SOURCE)

    plan = namespace["sample_plan"]
    assert inspect.isgeneratorfunction(plan), "queueserver only exposes generator functions"
    assert plan.__name__ == "sample_plan"
    assert plan.__module__ == SESSION_PLAN_MODULE
    assert "quoted" in plan.__doc__


def test_installed_plan_validates_kwargs_and_resolves_worker_devices() -> None:
    motor = FakeDevice("motor")
    namespace = worker_namespace(motor=motor)
    run_upload_script(namespace, "sample_plan", PLAN_SOURCE)

    assert list(namespace["sample_plan"](steps=3)) == [("move", motor, 3)]


def test_installed_plan_rejects_kwargs_its_schema_refuses() -> None:
    from pydantic import ValidationError

    namespace = worker_namespace(motor=FakeDevice("motor"))
    run_upload_script(namespace, "sample_plan", PLAN_SOURCE)

    # Queueserver surfaces this as a failed item: a **kwargs-only generator
    # validates on first iteration, not at call time.
    with pytest.raises(ValidationError):
        list(namespace["sample_plan"](steps="not-a-number"))


def test_missing_device_names_what_the_worker_actually_has() -> None:
    # A `build_plan` that RETURNS its plan (the shipped exemplars' style)
    # resolves devices at construction, which is where the wrapper can name
    # what the worker actually has. A `build_plan` that yields resolves them
    # mid-iteration and raises the bare `KeyError`, exactly as the catalog
    # wrapper in `qserver_startup` does.
    source = "def build_plan(devices, params):\n    return iter([('move', devices['motor'])])\n"
    namespace = worker_namespace(other_motor=FakeDevice("other_motor"))
    run_upload_script(namespace, "sample_plan", source)

    with pytest.raises(KeyError, match="other_motor"):
        list(namespace["sample_plan"]())


def test_two_session_plans_keep_their_own_schemas_and_bodies() -> None:
    namespace = worker_namespace(motor=FakeDevice("motor"))
    run_upload_script(namespace, "sample_plan", PLAN_SOURCE)
    run_upload_script(namespace, "other_plan", OTHER_PLAN_SOURCE)

    # Both plan files define module-level `PARAMS`/`build_plan`; executing them
    # into the shared namespace would leave each wrapper on the other's schema.
    assert list(namespace["sample_plan"](steps=2))[0][0] == "move"
    assert list(namespace["other_plan"](label="x")) == [("label", "x")]
    assert "PARAMS" not in namespace and "build_plan" not in namespace


def test_plan_file_without_params_takes_no_arguments() -> None:
    source = "def build_plan(devices, params):\n    yield ('bare',)\n"
    namespace = worker_namespace()
    run_upload_script(namespace, "bare_plan", source)

    assert list(namespace["bare_plan"]()) == [("bare",)]


def test_plan_file_missing_build_plan_is_refused() -> None:
    with pytest.raises(ValueError, match="build_plan"):
        install_session_plan(worker_namespace(), "broken", "X = 1\n")


def test_plan_file_with_non_model_params_is_refused() -> None:
    source = "PARAMS = {'steps': int}\n\ndef build_plan(devices, params):\n    yield ()\n"
    with pytest.raises(ValueError, match="BaseModel"):
        install_session_plan(worker_namespace(), "broken", source)


def test_real_bridge_devices_are_collected_like_the_stand_ins() -> None:
    """Keeps `FakeDevice` honest: a real device lands in the same collection.

    The devices a worker actually holds are ophyd-async
    ``StandardReadable``s, which ``is_device`` accepts through its
    ``children`` branch — the same branch the stand-in exercises. If that ever
    changed, every namespace test above would still pass while the real worker
    resolved no devices at all.
    """
    from osprey.services.bluesky_bridge.devices.mock import MockSettable
    from osprey.services.bluesky_bridge.session_upload import collect_devices

    device = MockSettable("m1")
    namespace = worker_namespace(m1=device, stub=FakeDevice("stub"))

    assert collect_devices(namespace) == {"m1": device, "stub": namespace["stub"]}


# ---------------------------------------------------------------------------
# Bridge half: the scripted backend and its fixtures
# ---------------------------------------------------------------------------


class FakeBackend:
    """A scripted stand-in for the three `QueueBackend` methods this module uses.

    ``namespace`` models the worker's plan namespace: an upload adds the plan
    under its name, and :meth:`cycle_environment` empties it exactly as a real
    environment open does when the startup module rebuilds the namespace.
    """

    def __init__(self, *, environment_open: bool = True, manager_state: str = "idle") -> None:
        self.namespace: dict[str, dict[str, Any]] = {}
        self.environment_open = environment_open
        self.manager_state = manager_state
        self.uploads: list[str] = []
        self.task_outcome: dict[str, Any] = {"success": True, "msg": ""}
        self.task_status: str = "completed"
        self.upload_reply: dict[str, Any] | None = None
        self.calls: list[str] = []

    def cycle_environment(self) -> None:
        """Rebuild the namespace from the startup module, as an env open does."""
        self.namespace = {
            name: description
            for name, description in self.namespace.items()
            if description.get("module") != SESSION_PLAN_MODULE
        }

    async def status(self, *, reload: bool = False) -> dict[str, Any]:
        self.calls.append("status")
        return {
            "worker_environment_exists": self.environment_open,
            "manager_state": self.manager_state,
        }

    async def plans_allowed(self) -> dict[str, Any]:
        self.calls.append("plans_allowed")
        return {"success": True, "plans_allowed": dict(self.namespace)}

    async def upload_script(self, script: str, *, update_lists: bool = True) -> dict[str, Any]:
        self.calls.append("upload_script")
        self.uploads.append(script)
        if self.upload_reply is not None:
            return self.upload_reply
        # The plan name is the only thing the fake needs off the script; the
        # real worker gets it by executing it.
        name = script.split("globals(), '")[1].split("'")[0]
        if self.task_outcome.get("success"):
            self.namespace[name] = {"name": name, "module": SESSION_PLAN_MODULE}
        return {"success": True, "task_uid": f"task-{len(self.uploads)}"}

    async def task_result(self, task_uid: str) -> dict[str, Any]:
        self.calls.append("task_result")
        return {"success": True, "status": self.task_status, "result": dict(self.task_outcome)}

    async def close_environment(self) -> dict[str, Any]:
        self.calls.append("close_environment")
        self.environment_open = False
        self.cycle_environment()
        return {"success": True}

    async def ensure_environment(self) -> bool:
        self.calls.append("ensure_environment")
        self.environment_open = True
        return True


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def records() -> ValidationRecordStore:
    return ValidationRecordStore()


@pytest.fixture
def uploader(
    tmp_path: Path, backend: FakeBackend, records: ValidationRecordStore
) -> SessionPlanUploader:
    return SessionPlanUploader(
        backend_getter=lambda: backend,
        records=records,
        session_dir_resolver=lambda: tmp_path,
        poll_interval=0.0,
        poll_attempts=3,
    )


def write_plan(directory: Path, name: str, source: str = PLAN_SOURCE) -> str:
    """Write a session plan file and return its content hash."""
    (directory / f"{name}.py").write_text(source, encoding="utf-8")
    return hash_plan_body(source)


def validate_plan(records: ValidationRecordStore, directory: Path, name: str) -> str:
    """Record a passing validation for a plan file's current content."""
    content = (directory / f"{name}.py").read_text(encoding="utf-8")
    content_hash = hash_plan_body(content)
    records.record(content_hash)
    return content_hash


# ---------------------------------------------------------------------------
# Upload: only on a passing record for the bytes on disk right now
# ---------------------------------------------------------------------------


async def test_upload_validated_pushes_the_current_bytes(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    content_hash = validate_plan(records, tmp_path, "sample_plan")

    assert await uploader.upload_validated("sample_plan") == content_hash
    assert "sample_plan" in backend.namespace
    assert uploader.uploaded_hashes() == {"sample_plan": content_hash}

    # What was uploaded is what was on disk: run the captured script the way
    # the worker would and check the plan it defines still behaves — a
    # round-trip through the script's `repr` literal, quotes and all.
    motor = FakeDevice("motor")
    namespace = worker_namespace(motor=motor)
    exec(backend.uploads[0], namespace, namespace)  # noqa: S102
    assert list(namespace["sample_plan"](steps=4)) == [("move", motor, 4)]


async def test_upload_refused_without_a_passing_record(
    tmp_path: Path, uploader: SessionPlanUploader, backend: FakeBackend
) -> None:
    write_plan(tmp_path, "sample_plan")

    with pytest.raises(SessionPlanNotReadyError) as excinfo:
        await uploader.upload_validated("sample_plan")

    assert excinfo.value.reason == REASON_UNVALIDATED
    assert backend.uploads == []


async def test_upload_refused_for_an_unknown_plan(uploader: SessionPlanUploader) -> None:
    with pytest.raises(SessionPlanNotReadyError, match="No session plan named"):
        await uploader.upload_validated("nope")


async def test_worker_failure_leaves_the_plan_uninstalled(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    backend.task_outcome = {"success": False, "msg": "NameError: bp is not defined"}

    with pytest.raises(SessionPlanUploadError, match="NameError"):
        await uploader.upload_validated("sample_plan")

    assert uploader.uploaded_hashes() == {}


async def test_upload_that_never_completes_fails_closed(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    backend.task_status = "running"

    with pytest.raises(SessionPlanUploadError, match="did not complete"):
        await uploader.upload_validated("sample_plan")

    assert uploader.uploaded_hashes() == {}


async def test_expired_task_result_fails_closed(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    backend.task_status = "not_found"

    with pytest.raises(SessionPlanUploadError, match="lost the result"):
        await uploader.upload_validated("sample_plan")

    assert uploader.uploaded_hashes() == {}


# ---------------------------------------------------------------------------
# The armed-path gate
# ---------------------------------------------------------------------------


async def test_start_gate_refuses_a_queue_holding_one_stale_session_plan(
    tmp_path: Path,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
    backend: FakeBackend,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")
    write_plan(tmp_path, "stale_plan", OTHER_PLAN_SOURCE)

    # One `plans_allowed` fetch covers the whole queue, and a repeated name
    # is not re-checked.
    backend.calls.clear()
    with pytest.raises(SessionPlanNotReadyError) as excinfo:
        await uploader.check_many_ready(["grid_scan", "sample_plan", "sample_plan", "stale_plan"])

    assert excinfo.value.plan == "stale_plan"
    assert backend.calls.count("plans_allowed") == 1


async def test_start_gate_admits_a_queue_of_ready_plans(
    tmp_path: Path, records: ValidationRecordStore, uploader: SessionPlanUploader
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    await uploader.check_many_ready(["grid_scan", "sample_plan"])


async def test_catalog_plan_passes_the_gate_untouched(uploader: SessionPlanUploader) -> None:
    # No session file, nothing session-tagged in the namespace: a shipped plan.
    await uploader.check_ready("grid_scan")


async def test_validated_and_uploaded_plan_passes_the_gate(
    tmp_path: Path, records: ValidationRecordStore, uploader: SessionPlanUploader
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    await uploader.check_ready("sample_plan")


async def test_plan_edited_after_upload_is_refused(
    tmp_path: Path,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
    backend: FakeBackend,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    # Re-authored through `POST /plans/session` with no re-validation: the
    # namespace still holds the version that passed, but the file no longer
    # matches it.
    write_plan(tmp_path, "sample_plan", PLAN_SOURCE.replace("params.steps", "params.steps * 10"))

    with pytest.raises(SessionPlanNotReadyError) as excinfo:
        await uploader.check_ready("sample_plan")

    assert excinfo.value.reason == REASON_UNVALIDATED
    assert "Re-validate" in excinfo.value.detail
    assert excinfo.value.to_dict() == {
        "reason": REASON_UNVALIDATED,
        "detail": excinfo.value.detail,
        "plan": "sample_plan",
    }
    assert "sample_plan" in backend.namespace, "the gate refuses, it does not repair"


async def test_re_validating_an_edit_re_uploads_it(
    tmp_path: Path,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
    backend: FakeBackend,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    edited = PLAN_SOURCE.replace("params.steps", "params.steps * 10")
    write_plan(tmp_path, "sample_plan", edited)
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    await uploader.check_ready("sample_plan")
    assert uploader.uploaded_hashes()["sample_plan"] == hash_plan_body(edited)
    assert repr(edited) in backend.uploads[-1], "the re-upload carries the edited bytes"


async def test_bridge_restart_refuses_a_redis_durable_item(
    tmp_path: Path, backend: FakeBackend, records: ValidationRecordStore
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    before = SessionPlanUploader(
        backend_getter=lambda: backend,
        records=records,
        session_dir_resolver=lambda: tmp_path,
        poll_interval=0.0,
    )
    await before.upload_validated("sample_plan")

    # Restart: the queue server (and its namespace) survives, the bridge's
    # in-memory validation records and upload bookkeeping do not.
    after = SessionPlanUploader(
        backend_getter=lambda: backend,
        records=ValidationRecordStore(),
        session_dir_resolver=lambda: tmp_path,
        poll_interval=0.0,
    )

    with pytest.raises(SessionPlanNotReadyError) as excinfo:
        await after.check_ready("sample_plan")

    assert excinfo.value.reason == REASON_UNVALIDATED
    assert "restarted" in excinfo.value.detail


async def test_restart_that_also_lost_the_file_refuses_the_namespace_leftover(
    tmp_path: Path, backend: FakeBackend, records: ValidationRecordStore
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    before = SessionPlanUploader(
        backend_getter=lambda: backend,
        records=records,
        session_dir_resolver=lambda: tmp_path,
        poll_interval=0.0,
    )
    await before.upload_validated("sample_plan")

    # The session directory is ephemeral by default, so a restarted bridge can
    # find the plan still in the worker with no file behind it at all.
    (tmp_path / "sample_plan.py").unlink()
    after = SessionPlanUploader(
        backend_getter=lambda: backend,
        records=ValidationRecordStore(),
        session_dir_resolver=lambda: tmp_path,
        poll_interval=0.0,
    )

    with pytest.raises(SessionPlanNotReadyError) as excinfo:
        await after.check_ready("sample_plan")

    assert excinfo.value.reason == REASON_NOT_IN_NAMESPACE


async def test_environment_cycle_refuses_until_the_plan_is_back(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    backend.cycle_environment()

    with pytest.raises(SessionPlanNotReadyError) as excinfo:
        await uploader.check_ready("sample_plan")

    assert excinfo.value.reason == REASON_NOT_IN_NAMESPACE


# ---------------------------------------------------------------------------
# Environment cycles: re-uploading the validated set
# ---------------------------------------------------------------------------


async def test_sync_re_uploads_the_whole_validated_set_after_a_cycle(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    write_plan(tmp_path, "other_plan", OTHER_PLAN_SOURCE)
    validate_plan(records, tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "other_plan")
    await uploader.upload_validated("sample_plan")
    await uploader.upload_validated("other_plan")

    backend.cycle_environment()
    assert backend.namespace == {}

    assert sorted(await uploader.sync_namespace()) == ["other_plan", "sample_plan"]
    assert sorted(backend.namespace) == ["other_plan", "sample_plan"]
    await uploader.check_ready("sample_plan")
    await uploader.check_ready("other_plan")


async def test_sync_skips_plans_the_namespace_already_holds(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    assert await uploader.sync_namespace() == []
    assert len(backend.uploads) == 1


async def test_sync_leaves_out_plans_with_no_current_record(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")
    write_plan(tmp_path, "never_validated", OTHER_PLAN_SOURCE)
    write_plan(tmp_path, "sample_plan", PLAN_SOURCE.replace("steps", "count"))

    backend.cycle_environment()

    assert await uploader.sync_namespace() == []
    assert backend.namespace == {}
    assert uploader.uploaded_hashes() == {}


async def test_sync_on_a_closed_environment_uploads_nothing(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    backend.environment_open = False
    backend.cycle_environment()

    assert await uploader.sync_namespace() == []
    assert len(backend.uploads) == 1
    assert uploader.uploaded_hashes() == {}


async def test_reload_restores_the_validated_set(
    tmp_path: Path,
    backend: FakeBackend,
    records: ValidationRecordStore,
    uploader: SessionPlanUploader,
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    await uploader.upload_validated("sample_plan")

    assert await uploader.reload_environment() == ["sample_plan"]
    assert "close_environment" in backend.calls and "ensure_environment" in backend.calls
    assert "sample_plan" in backend.namespace
    await uploader.check_ready("sample_plan")


async def test_reload_is_refused_while_a_run_is_active(
    backend: FakeBackend, uploader: SessionPlanUploader
) -> None:
    backend.manager_state = "executing_queue"

    with pytest.raises(SessionPlanEnvironmentBusyError, match="Stop the queue first"):
        await uploader.reload_environment()

    assert "close_environment" not in backend.calls


# ---------------------------------------------------------------------------
# The validate route's hook
# ---------------------------------------------------------------------------


async def test_validate_hook_reports_a_successful_upload(
    tmp_path: Path, records: ValidationRecordStore, uploader: SessionPlanUploader
) -> None:
    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")
    set_session_uploader(uploader)
    try:
        assert await upload_after_validation("sample_plan") == {
            "uploaded": True,
            "reason": None,
            "detail": None,
        }
    finally:
        set_session_uploader(None)


async def test_validate_hook_never_raises_when_there_is_no_queue_server(
    tmp_path: Path,
    records: ValidationRecordStore,
    backend: FakeBackend,
    uploader: SessionPlanUploader,
) -> None:
    from osprey.services.bluesky_bridge.queue_backend import QueueUnavailableError

    write_plan(tmp_path, "sample_plan")
    validate_plan(records, tmp_path, "sample_plan")

    async def refuse(*_: Any, **__: Any) -> dict[str, Any]:
        raise QueueUnavailableError("no queue server is configured")

    backend.upload_script = refuse  # type: ignore[method-assign]
    set_session_uploader(uploader)
    try:
        outcome = await upload_after_validation("sample_plan")
    finally:
        set_session_uploader(None)

    assert outcome["uploaded"] is False
    assert outcome["reason"] == "manager_unreachable"
    assert "no queue server" in outcome["detail"]


def test_authoring_rejects_a_leading_underscore_plan_name() -> None:
    """A name the worker could never expose is refused at authoring time.

    The manager's permissions forbid `:^_` plans, so a `_foo` session plan
    would author and upload successfully and then never appear in
    `plans_allowed` — leaving `check_ready` refusing it forever with "its
    upload did not land". `_sanitize_plan_name` closes that at the source.
    """
    from fastapi import HTTPException

    from osprey.services.bluesky_bridge.app import _sanitize_plan_name

    assert _sanitize_plan_name("sample_plan") == "sample_plan"

    with pytest.raises(HTTPException) as excinfo:
        _sanitize_plan_name("_hidden_plan")

    assert excinfo.value.status_code == 400
    assert "underscore" in excinfo.value.detail


# ---------------------------------------------------------------------------
# The PEP 563 pin, enumerated across EVERY plan-wrapper factory.
#
# Both modules that hand a plan to the RE manager use
# `from __future__ import annotations`, so every annotation they write is a
# STRING. The manager builds each plan's validation model straight off
# `inspect.signature`, wrapping a VAR_KEYWORD annotation as
# `typing.Dict[str, <annotation>]` for `pydantic.create_model` -- and a string
# there is an unresolvable forward reference, so the manager refuses EVERY
# enqueue of that plan with "`Model` is not fully defined; you should define
# `Any`".
#
# This bit TWICE. It was fixed first in `qserver_startup._make_plan_function`
# (catalog plans) when a real deploy could not enqueue grid_scan, and the
# identical line was missing in `session_upload._make_session_plan` (session
# plans), which the next real deploy found. A one-site fix for a two-site
# pattern is how a defect survives its own repair.
#
# So this test ENUMERATES the factories rather than asserting about one of
# them. The dict below is maintained by hand, so on its own it would go stale
# silently -- a third factory would simply never be listed. That is what
# `test_the_factory_list_covers_every_annotation_rewriting_site` is for: it
# counts the `plan_function.__annotations__ =` sites in the package and fails
# until the list matches, which is what makes "a third factory fails here"
# true rather than merely hoped for. Both real-container failures were found
# by tests/e2e/test_bluesky_queue_e2e.py.
# ---------------------------------------------------------------------------


def _catalog_plan_wrapper():
    """A wrapper as `qserver_startup` hands catalog plans to the manager."""
    from osprey.services.bluesky_bridge import plan_loader, qserver_startup

    plans = plan_loader.get_facility_plans().plans
    assert plans, "no catalog plans to build a wrapper from"
    return qserver_startup._make_plan_function(plans[sorted(plans)[0]], devices={})


def _session_plan_wrapper():
    """A wrapper as `session_upload` hands session plans to the manager."""
    namespace = worker_namespace(motor=FakeDevice("motor"))
    run_upload_script(namespace, "sample_plan", PLAN_SOURCE)
    return namespace["sample_plan"]


_PLAN_WRAPPER_FACTORIES = {
    "qserver_startup._make_plan_function": _catalog_plan_wrapper,
    "session_upload._make_session_plan": _session_plan_wrapper,
}


def test_the_factory_list_covers_every_annotation_rewriting_site() -> None:
    """The hand-maintained factory list above must not silently go stale.

    Every wrapper that reaches the manager fixes its own annotations with a
    ``plan_function.__annotations__ = ...`` assignment, so counting those
    assignments across the bridge package counts the factories that need to be
    listed. A third one added later fails HERE -- naming the file it lives in
    -- which is the prompt to add it to ``_PLAN_WRAPPER_FACTORIES`` and give it
    the same treatment.
    """
    from osprey.services import bluesky_bridge

    package_dir = Path(bluesky_bridge.__file__).resolve().parent
    sites = sorted(
        f"{path.name}:{lineno}"
        for path in package_dir.rglob("*.py")
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "plan_function.__annotations__ =" in line
    )

    assert len(sites) == len(_PLAN_WRAPPER_FACTORIES), (
        f"{len(sites)} plan-wrapper factories rewrite their annotations {sites}, but "
        f"only {len(_PLAN_WRAPPER_FACTORIES)} are listed in _PLAN_WRAPPER_FACTORIES "
        f"({sorted(_PLAN_WRAPPER_FACTORIES)}) -- the PEP 563 pin below does not cover "
        "them all"
    )


@pytest.mark.parametrize("factory_name", sorted(_PLAN_WRAPPER_FACTORIES))
def test_no_plan_wrapper_ships_pep563_string_annotations(factory_name: str) -> None:
    """No plan reaching the manager may carry a string annotation.

    Asserted on the objects rather than on `inspect.signature`'s repr: the repr
    of a resolved annotation is not stable across Python versions, while "is
    not a string" is exactly the property upstream needs.
    """
    plan_function = _PLAN_WRAPPER_FACTORIES[factory_name]()

    annotations = plan_function.__annotations__
    assert annotations, f"{factory_name} produced a wrapper with no annotations at all"
    for parameter, annotation in annotations.items():
        assert not isinstance(annotation, str), (
            f"{factory_name} annotates {parameter!r} with the PEP 563 string "
            f"{annotation!r}; upstream cannot build a pydantic model from it and every "
            "enqueue of this plan would be refused"
        )


@pytest.mark.parametrize("factory_name", sorted(_PLAN_WRAPPER_FACTORIES))
def test_every_plan_wrapper_survives_the_managers_model_construction(factory_name: str) -> None:
    """End to end on the property upstream actually exercises.

    Reproduces what `pydantic_construct_model_class` does to a VAR_KEYWORD
    parameter, so this fails the same way the manager does rather than merely
    noticing a string.
    """
    import inspect

    import pydantic

    plan_function = _PLAN_WRAPPER_FACTORIES[factory_name]()
    kwargs_param = inspect.signature(plan_function).parameters["kwargs"]
    assert kwargs_param.kind is inspect.Parameter.VAR_KEYWORD

    model = pydantic.create_model("Model", kwargs=(dict[str, kwargs_param.annotation], {}))
    assert model(kwargs={"anything": 1}).kwargs == {"anything": 1}
