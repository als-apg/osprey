"""Tests for the unified Jobs API endpoints.

Canonical home for the Jobs API surface: GET /jobs, GET /jobs/{id} (status
and ?detail=summary|full), POST /jobs/{id}/{cancel,start,pause,resume}, PUT
/jobs/{id}/config, GET/PUT /jobs/current, and POST /optimizations (job
creation) — consolidated from the former test_api.py. Also owns the
optimization-summary (``?detail=summary`` result_summary population)
endpoint tests, moved from the former test_api_endpoints.py.
"""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import polars as pl
import pytest
import redis.exceptions
from fastapi.testclient import TestClient
from pydantic import ValidationError

from osprey.services.optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey.services.optimizer.api.job_tracker import ACTIVE_JOB_KEY, JobMeta, JobTracker
from osprey.services.optimizer.api.main import app
from osprey.services.optimizer.models.api import JobResponse
from osprey.services.optimizer.models.config import OptimizationConfig, VariableOverride
from osprey.services.optimizer.models.enums import JobStatus, JobType
from osprey.services.optimizer.models.exceptions import MachineSettingsError
from osprey.services.optimizer.optimization.state import (
    get_optimization_reader,
    get_optimization_writer,
)
from osprey.services.optimizer.utils.job_keys import (
    config_key,
    meta_key,
    state_prefix,
)
from osprey.services.optimizer.utils.redis_client import get_redis_client
from osprey.services.optimizer.utils.serialization import deserialize_obj, serialize_obj

from ._fakes import MockEnvironment

_JOBS = "osprey.services.optimizer.api.routers.jobs"
_JS = "osprey.services.optimizer.api.job_status"
_OS = "osprey.services.optimizer.api.optimization_status"


def make_opt_reader(**overrides: object) -> MagicMock:
    """Return a configured optimization-reader mock; override only what varies.

    Defaults describe a RUNNING optimization with a 3-row frame and a MAXIMIZE
    objective. Keyword overrides map 1:1 to the reader getters (``status`` ->
    ``get_status`` etc.); pass ``None`` explicitly to null a field out.
    """
    defaults: dict[str, object] = {
        "status": JobStatus.RUNNING,
        "error_message": None,
        "last_results_path": None,
        "logs": "",
        "data": pl.DataFrame({"var1": [1.0, 2.0, 3.0], "obj": [0.5, 0.9, 0.7]}),
        "data_version": 0,
        "environment_name": "test_env",
        "objective_name": "obj",
        "variable_names": ["var1"],
        "variable_bounds": None,
        "total_evaluations": 0,
        "available_snapshots": {
            "initial": {"var1": 0.0, "obj": 0.3},
            "target": {"obj": 1.0},
        },
        "objective_direction": "MAXIMIZE",
        "metadata": {
            "variables": {"var1": {"description": "Var 1", "units": "V", "step_size": 0.1}},
            "observables": {
                "obj": {
                    "description": "Observable 1",
                    "units": "mA",
                    "direction": "MAXIMIZE",
                    "target_value": 1.0,
                }
            },
        },
    }
    defaults.update(overrides)
    reader = MagicMock()
    for field, value in defaults.items():
        getattr(reader, f"get_{field}").return_value = value
    return reader


def job_meta(
    job_id: str = "test-job-id",
    job_type: JobType = JobType.OPTIMIZATION,
    status: JobStatus = JobStatus.RUNNING,
) -> JobMeta:
    """A tracker job-meta record, as ``JobTracker.get_job`` returns it."""
    return JobMeta(
        job_id=job_id, pid=1234, status=status, created_at=1712000000.0, job_type=job_type
    )


def corrupt_meta_error() -> ValidationError:
    """The real ValidationError a corrupt ``job:{id}:meta`` record parses into."""
    with pytest.raises(ValidationError) as exc:
        JobMeta.model_validate_json(json.dumps({"job_id": "corrupt-id", "job_type": "bogus"}))
    return exc.value


@contextmanager
def patched_control_seams(
    source: JobStatus, cas_ok: bool = False
) -> Generator[tuple[MagicMock, MagicMock]]:
    """Patch the pause/resume control seams, yielding ``(token_factory, writer)``.

    The writer reports *source* as the job's current status and *cas_ok* as the
    outcome of its compare-and-swap.
    """
    writer = MagicMock()
    writer.get_status.return_value = source
    writer.set_status_cas.return_value = cas_ok
    with (
        patch(f"{_JOBS}.get_cancellation_token") as token_f,
        patch(f"{_JOBS}.get_optimization_writer", return_value=writer),
    ):
        yield token_f, writer


@contextmanager
def patched_job_state(status: JobStatus, logs: str = "") -> Generator[MagicMock]:
    """Patch the job-status seam's ``create_job_state`` with a reader serving
    *status* and *logs* (and no error message), yielding that reader."""
    reader = MagicMock()
    reader.get_status.return_value = status
    reader.get_logs.return_value = logs
    reader.get_error_message.return_value = None
    with patch(f"{_JS}.create_job_state", return_value=reader):
        yield reader


def make_gate_config(**overrides: Any) -> OptimizationConfig:
    """A resolved single-variable config that passes the start gates."""
    return OptimizationConfig(
        environment_name="als_injector_sim",
        control_target="va",
        objective_name="obs1",
        variables=["var1"],
        variable_overrides={"var1": VariableOverride(min=0.0, max=10.0, step_size=0.1)},
        **overrides,
    )


@contextmanager
def created_job_client(
    job_id: str, config: OptimizationConfig, registry: MagicMock
) -> Generator[tuple[TestClient, MagicMock]]:
    """Seed a real-Redis CREATED optimization job (meta + status + stored config)
    and yield ``(client, launch)`` —
    a client wired to that tracker and *registry*, plus the stubbed spawn seam.
    Flushes on exit."""
    tracker = JobTracker()
    redis = tracker._get_redis()
    redis.flushdb()
    tracker.register_created_job(job_id, job_type=JobType.OPTIMIZATION)
    get_optimization_writer(job_id).set_status(JobStatus.CREATED)
    redis.set(config_key(job_id), config.model_dump_json())

    app.dependency_overrides[get_job_tracker] = lambda: tracker
    app.dependency_overrides[get_environment_registry] = lambda: registry
    try:
        with (
            patch("osprey.services.optimizer.api.optimization_lifecycle.launch_job") as launch,
            TestClient(app) as test_client,
        ):
            yield test_client, launch
    finally:
        app.dependency_overrides.clear()
        redis.flushdb()


@pytest.fixture
def client(seam_redis_mock: MagicMock, mock_job_tracker: MagicMock) -> Generator[TestClient]:
    del seam_redis_mock  # wanted for the Redis it patches, not its value
    app.dependency_overrides[get_job_tracker] = lambda: mock_job_tracker
    with TestClient(app) as test_client:
        yield test_client


class TestListJobs:
    def test_no_active_job(self, client: TestClient, mock_job_tracker: MagicMock) -> None:
        mock_job_tracker.list_job_metas.return_value = []
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.parametrize(
        ("job_type", "logs"),
        [
            pytest.param(JobType.OPTIMIZATION, "iteration 5\n", id="optimization"),
        ],
    )
    def test_active_job(
        self, client: TestClient, mock_job_tracker: MagicMock, job_type: JobType, logs: str
    ) -> None:
        meta = job_meta(job_type=job_type)
        mock_job_tracker.list_job_metas.return_value = [meta]
        mock_job_tracker.get_job.return_value = meta

        with patched_job_state(JobStatus.RUNNING, logs):
            resp = client.get("/jobs")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["job_id"] == "test-job-id"
        assert data[0]["job_type"] == job_type
        assert data[0]["status"] == "running"
        assert data[0]["logs"] == logs
        # meta created_at 1712000000.0 rendered as UTC ISO-8601
        assert data[0]["created_at"] == "2024-04-01T19:33:20+00:00"

    def test_meta_with_empty_job_id_is_skipped(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        """A meta record with an empty job_id is silently skipped."""
        mock_job_tracker.list_job_metas.return_value = [job_meta(job_id="")]
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.parametrize(
        "read_side_effect",
        [
            pytest.param(lambda job_id: None, id="missing-metadata"),
            pytest.param(corrupt_meta_error(), id="corrupt-metadata"),
        ],
    )
    def test_corrupt_or_missing_job_is_skipped(
        self, client: TestClient, mock_job_tracker: MagicMock, read_side_effect: object
    ) -> None:
        """A listed job whose per-job read finds the metadata gone (TTL race,
        regression guard for #40) or corrupt is skipped rather than erroring the
        whole listing."""
        mock_job_tracker.list_job_metas.return_value = [job_meta(job_id="job-x")]
        mock_job_tracker.get_job.side_effect = read_side_effect
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_systemic_read_failure_returns_503(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        """A non-HTTPException from list_job_metas (e.g. redis down) maps to 503."""
        mock_job_tracker.list_job_metas.side_effect = redis.exceptions.ConnectionError("redis down")
        resp = client.get("/jobs")
        assert resp.status_code == 503
        assert "Failed to read job state" in resp.json()["detail"]
        assert "redis down" in resp.json()["detail"]


@pytest.mark.parametrize(
    "verb,suffix", [("get", ""), ("post", "/cancel")], ids=["get_status", "cancel"]
)
@pytest.mark.parametrize(
    ("read_side_effect", "expected_status", "detail_token"),
    [
        # An unknown job and one whose metadata expired out from under the read
        # (TTL race) are indistinguishable and share one 404 wording — the
        # dependency's single read is the only existence check.
        # (regression guard for #40)
        pytest.param(lambda job_id: None, 404, "Job test-job-id not found", id="metadata_missing"),
        pytest.param(corrupt_meta_error(), 500, "Corrupt job metadata", id="metadata_corrupt"),
    ],
)
def test_job_lookup_guards(
    client: TestClient,
    mock_job_tracker: MagicMock,
    verb: str,
    suffix: str,
    read_side_effect: object,
    expected_status: int,
    detail_token: str,
) -> None:
    """Both the status and cancel routes guard job lookup identically, through
    the require_job_meta dependency: missing metadata is a 404, corrupt a 500."""
    mock_job_tracker.get_job.side_effect = read_side_effect
    resp = getattr(client, verb)(f"/jobs/test-job-id{suffix}")
    assert resp.status_code == expected_status
    assert detail_token in resp.json()["detail"]


class TestGetJobStatus:
    @pytest.mark.parametrize(
        ("job_id", "job_type", "logs"),
        [
            pytest.param("test-job-id", JobType.OPTIMIZATION, "running...\n", id="optimization"),
        ],
    )
    def test_get_job(
        self,
        client: TestClient,
        mock_job_tracker: MagicMock,
        job_id: str,
        job_type: JobType,
        logs: str,
    ) -> None:
        mock_job_tracker.get_job.return_value = job_meta(job_id, job_type)

        with patched_job_state(JobStatus.RUNNING, logs):
            resp = client.get(f"/jobs/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["job_type"] == job_type
        assert data["status"] == "running"
        assert data["logs"] == logs

    def test_completed_job_marks_tracker(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        with patched_job_state(JobStatus.COMPLETED):
            resp = client.get("/jobs/test-job-id")

        assert resp.status_code == 200
        mock_job_tracker.mark_completed.assert_called_once_with("test-job-id")

    @pytest.mark.usefixtures("mock_job_tracker")
    def test_returns_503_when_real_reader_hits_redis_error(
        self, client: TestClient, seam_redis_mock: MagicMock
    ) -> None:
        """Real OptimizationState propagates redis.exceptions.ConnectionError
        from get() through the jobs router wrapper, yielding 503."""
        seam_redis_mock.get.side_effect = redis.exceptions.ConnectionError("boom")
        resp = client.get("/jobs/test-job-id")

        assert resp.status_code == 503
        assert "Failed to read job state" in resp.json()["detail"]
        assert "boom" in resp.json()["detail"]

    @pytest.mark.usefixtures("mock_job_tracker")
    def test_running_job_returns_200_before_first_log_flush(
        self, client: TestClient, seam_redis_mock: MagicMock
    ) -> None:
        """Polling a freshly-launched job before its first log flush must return
        200 with empty logs — not 503.

        Repro: ``POST /scripts/{id}/run`` (and similarly for optimization) pre-
        initializes state with ``status=RUNNING`` but leaves the ``logs`` Redis
        key unset until the spawned process's first flush. In that window the
        state reader sees ``None`` for the ``logs`` key and raises ``TypeError``,
        which bubbles up as 503. Absent-key is a legitimate early-lifecycle
        state, not a Redis failure.
        """

        def _get_by_key(key: str) -> bytes | None:
            if key.endswith(":status"):
                return serialize_obj(JobStatus.RUNNING)
            # ``:logs`` and ``:error_message`` haven't been written yet
            return None

        seam_redis_mock.get.side_effect = _get_by_key

        resp = client.get("/jobs/test-job-id")

        assert resp.status_code == 200, resp.json()
        data = resp.json()
        assert data["status"] == "running"
        assert data["logs"] == ""
        assert data["error_message"] is None


class TestCancelJob:
    def test_cancel_optimization(
        self, client: TestClient, mock_job_tracker: MagicMock, seam_redis_mock: MagicMock
    ) -> None:
        seam_redis_mock.get.return_value = serialize_obj(JobStatus.RUNNING)
        # CAS (Lua EVAL) wins: the status is still RUNNING when the swap runs.
        seam_redis_mock.register_script.return_value.return_value = 1
        with patch(f"{_JOBS}.read_job_status") as read_f:
            read_f.return_value = JobResponse(
                job_id="test-job-id", job_type=JobType.OPTIMIZATION, status=JobStatus.CANCELLED
            )
            resp = client.post("/jobs/test-job-id/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["job_id"] == "test-job-id"
        mock_job_tracker.mark_completed.assert_called_once_with("test-job-id")

    # Footprint bounding is asserted in test_job_tracker.py; here the contract is only
    # that cancel calls mark_completed (see test_cancel_optimization).

    @pytest.mark.parametrize(
        "terminal", [JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.CANCELLED]
    )
    def test_cancel_rejects_terminal_job_with_409(
        self,
        client: TestClient,
        mock_job_tracker: MagicMock,
        seam_redis_mock: MagicMock,
        terminal: JobStatus,
    ) -> None:
        """Cancelling a finished job must 409, not overwrite its terminal outcome."""
        seam_redis_mock.get.return_value = serialize_obj(terminal)
        resp = client.post("/jobs/test-job-id/cancel")
        assert resp.status_code == 409
        assert terminal.value in resp.json()["detail"]
        mock_job_tracker.mark_completed.assert_not_called()

    @pytest.mark.parametrize("terminal", [JobStatus.COMPLETED, JobStatus.ERROR])
    def test_cancel_rechecks_status_before_write_and_409s_on_concurrent_completion(
        self,
        client: TestClient,
        mock_job_tracker: MagicMock,
        seam_redis_mock: MagicMock,
        terminal: JobStatus,
    ) -> None:
        """A job that goes terminal between the endpoint guard and the status
        write must 409, not have its terminal outcome overwritten with CANCELLED.

        The worker sets its own terminal status when a run ends. The endpoint
        guard read sees RUNNING, but ``cancel_with_reread`` re-reads just before
        the CAS and sees the worker's terminal status, so it aborts before the
        swap ever runs — no SET, no Lua EVAL, no mark_completed.
        """
        # First get_status() (endpoint guard) → RUNNING; second (re-check) → terminal.
        seam_redis_mock.get.side_effect = [
            serialize_obj(JobStatus.RUNNING),
            serialize_obj(terminal),
        ]
        resp = client.post("/jobs/test-job-id/cancel")
        assert resp.status_code == 409
        assert terminal.value in resp.json()["detail"]
        seam_redis_mock.set.assert_not_called()
        seam_redis_mock.register_script.return_value.assert_not_called()
        mock_job_tracker.mark_completed.assert_not_called()

    @pytest.mark.usefixtures("mock_job_tracker")
    def test_cancel_returns_500_when_real_writer_hits_redis_error(
        self, client: TestClient, seam_redis_mock: MagicMock
    ) -> None:
        """Real OptimizationState.set_status_cas propagates
        redis.exceptions.ConnectionError from the CAS Lua EVAL through the cancel
        wrapper, yielding 500."""
        seam_redis_mock.get.return_value = serialize_obj(JobStatus.RUNNING)
        seam_redis_mock.register_script.return_value.side_effect = redis.exceptions.ConnectionError(
            "boom"
        )
        resp = client.post("/jobs/test-job-id/cancel")

        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]

    @pytest.mark.usefixtures("redis_server")
    def test_cancel_cas_loses_to_worker_terminal_write(self) -> None:
        """A5: status goes COMPLETED between the cancel op's initial read and its
        CAS. The CAS fails, the op 409s, and the final status stays COMPLETED."""
        tracker = JobTracker()
        redis = tracker._get_redis()
        redis.flushdb()

        job_id = "race-job"
        redis.set(meta_key(job_id), job_meta(job_id).model_dump_json())
        writer = get_optimization_writer(job_id)
        writer.set_status(JobStatus.RUNNING)

        # Patch the endpoint's *initial* read to return RUNNING once, while the
        # real Redis status is flipped COMPLETED before the CAS — deterministic
        # interleave, no sleeps. The CAS reads the real (COMPLETED) bytes.
        def racing_get_status(_self: object) -> JobStatus:
            # Worker reaches terminal in the gap between this read and the CAS.
            get_optimization_writer(job_id).set_status(JobStatus.COMPLETED)
            return JobStatus.RUNNING

        app.dependency_overrides[get_job_tracker] = lambda: tracker
        try:
            with (
                patch.object(type(writer), "get_status", racing_get_status),
                TestClient(app) as test_client,
            ):
                resp = test_client.post(f"/jobs/{job_id}/cancel")
            assert resp.status_code == 409, resp.json()
            raw_status = get_redis_client(binary=True).get(f"{state_prefix(job_id)}:status")
            assert raw_status is not None
            final = deserialize_obj(raw_status)
            assert final is JobStatus.COMPLETED
        finally:
            app.dependency_overrides.clear()
            redis.flushdb()


class TestStartJob:
    def test_start_dispatches_optimization(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        mock_job_tracker.get_job.return_value = job_meta()
        with (
            # This test targets only the type-based dispatch; start_gates coverage
            # (machine-check + drift) lives in test_start_gates.py and the 412
            # endpoint test below, so the config/gate seams are stubbed out here.
            patch(f"{_JOBS}.read_optimization_config") as read_cfg,
            patch(f"{_JOBS}.run_start_gates") as gates,
            patch(
                f"{_JOBS}.start_optimization_job",
                return_value=JobResponse(
                    job_id="test-job-id", job_type=JobType.OPTIMIZATION, status=JobStatus.RUNNING
                ),
            ) as start,
        ):
            # load_environment (unpatched here) really loads "als_injector_sim" for
            # the gate env; environment_config must be a real (empty) dict for that
            # construction to succeed, not the MagicMock the bare config double
            # would otherwise supply.
            read_cfg.return_value.environment_config = {}
            read_cfg.return_value.environment_name = "als_injector_sim"
            read_cfg.return_value.control_target = "va"
            read_cfg.return_value.execution_mode = None
            resp = client.post("/jobs/test-job-id/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
        assert resp.json()["job_id"] == "test-job-id"
        gates.assert_called_once_with(read_cfg.return_value, ANY, env_name="als_injector_sim")
        start.assert_called_once_with("test-job-id", mock_job_tracker)

    @pytest.mark.usefixtures("mock_job_tracker")
    def test_start_missing_job_404(self, client: TestClient) -> None:
        no_config = MagicMock()
        no_config.get.return_value = None
        with patch(
            "osprey.services.optimizer.api.job_control._redis_client.get_redis_client",
            return_value=no_config,
        ):
            resp = client.post("/jobs/missing/start")
        assert resp.status_code == 404
        assert "No stored config" in resp.json()["detail"]

    @pytest.mark.parametrize("missing", ["control_target", "environment_name"])
    def test_start_without_a_target_or_environment_is_refused(
        self, client: TestClient, mock_job_tracker: MagicMock, missing: str
    ) -> None:
        """A run never lands on a machine nobody named: 409 before any guard runs."""
        mock_job_tracker.get_job.return_value = job_meta()
        config = make_gate_config().model_copy(update={missing: None})
        with (
            patch(f"{_JOBS}.read_optimization_config", return_value=config),
            patch(f"{_JOBS}.start_optimization_job") as start,
        ):
            resp = client.post("/jobs/test-job-id/start")
        assert resp.status_code == 409
        assert "control_target" in resp.json()["detail"]
        start.assert_not_called()

    @pytest.mark.usefixtures("redis_server")
    def test_contended_start_preserves_created_job_then_succeeds(self) -> None:
        """A1/A2: with the slot occupied, starting a CREATED job 409s and leaves
        it intact; once the slot frees, the same start succeeds."""
        real_tracker = JobTracker()
        redis = real_tracker._get_redis()
        redis.flushdb()

        # Occupy the slot with a foreign RUNNING job.
        redis.set(ACTIVE_JOB_KEY, "holder")
        redis.set(meta_key("holder"), job_meta("holder").model_dump_json())

        # Register a real CREATED optimization job: meta + status + stored config.
        created = "created-opt"
        real_tracker.register_created_job(created, job_type=JobType.OPTIMIZATION)
        get_optimization_writer(created).set_status(JobStatus.CREATED)
        config = OptimizationConfig(
            environment_name="als_injector_sim",
            control_target="va",
            objective_name="charge_gtl",
        )
        redis.set(config_key(created), config.model_dump_json())

        app.dependency_overrides[get_job_tracker] = lambda: real_tracker
        try:
            # This test targets slot-contention semantics, not the start-gates
            # precondition, and the config above is deliberately an unresolved
            # draft (no variables/overrides) -- stub the gate out so it doesn't
            # 409 for a reason unrelated to this test's contention scenario.
            with patch(f"{_JOBS}.run_start_gates"), TestClient(app) as test_client:
                resp = test_client.post(f"/jobs/{created}/start")
                assert resp.status_code == 409, resp.json()

                get_resp = test_client.get(f"/jobs/{created}")
                assert get_resp.status_code == 200, get_resp.json()
                assert get_resp.json()["status"] == "created"
                # All three key families still present.
                assert redis.exists(meta_key(created))
                assert redis.exists(config_key(created))
                assert redis.exists(f"job:{created}:state:status")

                redis.delete(ACTIVE_JOB_KEY)
                redis.delete(meta_key("holder"))
                resp2 = test_client.post(f"/jobs/{created}/start")
                assert resp2.status_code == 200, resp2.json()
                assert resp2.json()["status"] == "running"
        finally:
            app.dependency_overrides.clear()
            redis.flushdb()

    @pytest.mark.usefixtures("redis_server")
    def test_start_412_on_machine_check_failure_leaves_job_created_and_no_spawn(self) -> None:
        """A machine-settings failure at start must 412, never mutate job state
        (status stays CREATED), and never reach the spawn seam (launch_job)."""
        job_id = "gate-fail-job"

        def _refuse() -> None:
            raise MachineSettingsError("no", failures=["GATE:A: locked out"])

        env = MockEnvironment()
        env.check_machine_settings = _refuse  # type: ignore[method-assign]
        registry = MagicMock()
        registry.load_environment.return_value = env

        with created_job_client(job_id, make_gate_config(), registry) as (test_client, launch):
            resp = test_client.post(f"/jobs/{job_id}/start")

            assert resp.status_code == 412, resp.json()
            assert "GATE:A" in resp.json()["detail"]
            launch.assert_not_called()
            assert get_optimization_reader(job_id).get_status() is JobStatus.CREATED

    @pytest.mark.usefixtures("redis_server")
    def test_start_builds_gate_env_with_job_environment_config(self) -> None:
        """The gate env is loaded with the job's stored ``environment_config``,
        not an empty dict -- matching what draft resolution and the worker
        itself use it for (regression guard for the start-gate/resolution/worker
        env mismatch)."""
        job_id = "gate-env-config-job"
        env_config = {"sim_seed": 7}
        registry = MagicMock()
        registry.load_environment.return_value = MockEnvironment()

        with created_job_client(
            job_id, make_gate_config(environment_config=env_config), registry
        ) as (
            test_client,
            _,
        ):
            test_client.post(f"/jobs/{job_id}/start")

        # The job's target rides into the gate environment beside its own config.
        registry.load_environment.assert_called_once_with(
            "als_injector_sim", {**env_config, "control_target": "va", "execution_mode": None}
        )


class TestPauseResumeOnJobs:
    def test_pause_optimization_ok(self, client: TestClient, mock_job_tracker: MagicMock) -> None:
        mock_job_tracker.get_job.return_value = job_meta()
        with (
            patched_control_seams(JobStatus.RUNNING, cas_ok=True) as (token_f, writer),
            patch(f"{_JOBS}.read_job_status") as read_f,
        ):
            read_f.return_value = JobResponse(
                job_id="test-job-id", job_type=JobType.OPTIMIZATION, status=JobStatus.PAUSED
            )
            resp = client.post("/jobs/test-job-id/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"
        token_f.return_value.pause.assert_called_once()
        writer.set_status_cas.assert_called_once_with(JobStatus.RUNNING, JobStatus.PAUSED)

    def test_resume_optimization_ok(self, client: TestClient, mock_job_tracker: MagicMock) -> None:
        mock_job_tracker.get_job.return_value = job_meta()
        with (
            patched_control_seams(JobStatus.PAUSED, cas_ok=True) as (token_f, writer),
            patch(f"{_JOBS}.read_job_status") as read_f,
        ):
            read_f.return_value = JobResponse(
                job_id="test-job-id", job_type=JobType.OPTIMIZATION, status=JobStatus.RUNNING
            )
            resp = client.post("/jobs/test-job-id/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
        token_f.return_value.resume.assert_called_once()
        writer.set_status_cas.assert_called_once_with(JobStatus.PAUSED, JobStatus.RUNNING)

    @pytest.mark.parametrize("endpoint", ["pause", "resume"])
    def test_pause_resume_not_found_returns_404(
        self, client: TestClient, mock_job_tracker: MagicMock, endpoint: str
    ) -> None:
        """require_job_meta raises 404 when the job does not exist."""
        mock_job_tracker.get_job.return_value = None
        resp = client.post(f"/jobs/missing-job/{endpoint}")
        assert resp.status_code == 404
        assert "missing-job not found" in resp.json()["detail"]

    def test_internal_failure_maps_to_500(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        """A non-HTTPException inside _control_action (e.g. backend down) maps to 500."""
        mock_job_tracker.get_job.return_value = job_meta()
        with patch(
            f"{_JOBS}.get_cancellation_token",
            side_effect=redis.exceptions.ConnectionError("backend boom"),
        ):
            resp = client.post("/jobs/test-job-id/pause")
        assert resp.status_code == 500
        assert "backend boom" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "source",
        [JobStatus.CREATED, JobStatus.PAUSED, JobStatus.SUBMITTED],
    )
    def test_pause_rejects_non_running_source(
        self, client: TestClient, mock_job_tracker: MagicMock, source: JobStatus
    ) -> None:
        """pause is only legal from RUNNING; any other non-terminal source 409s
        and does not write status."""
        mock_job_tracker.get_job.return_value = job_meta()
        with patched_control_seams(source) as (token_f, writer):
            resp = client.post("/jobs/test-job-id/pause")
        assert resp.status_code == 409
        assert source.value in resp.json()["detail"]
        writer.set_status.assert_not_called()
        token_f.return_value.pause.assert_not_called()

    @pytest.mark.parametrize(
        "source",
        [JobStatus.CREATED, JobStatus.RUNNING, JobStatus.SUBMITTED],
    )
    def test_resume_rejects_non_paused_source(
        self, client: TestClient, mock_job_tracker: MagicMock, source: JobStatus
    ) -> None:
        """resume is only legal from PAUSED; a non-paused running job 409s."""
        mock_job_tracker.get_job.return_value = job_meta()
        with patched_control_seams(source) as (token_f, writer):
            resp = client.post("/jobs/test-job-id/resume")
        assert resp.status_code == 409
        assert source.value in resp.json()["detail"]
        writer.set_status.assert_not_called()
        token_f.return_value.resume.assert_not_called()


class TestGetJobDetail:
    """Tests for the ?detail= query param on GET /jobs/{id}."""

    def test_get_job_detail_summary_optimization_dispatches_builder(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        """Optimization jobs with detail=summary go through build_optimization_response."""
        with patch(f"{_JOBS}.build_optimization_response") as mock_build:
            mock_build.return_value = JobResponse(
                job_id="test-job-id",
                job_type=JobType.OPTIMIZATION,
                status=JobStatus.RUNNING,
            )
            resp = client.get("/jobs/test-job-id?detail=summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["job_type"] == "optimization"
        mock_build.assert_called_once_with("test-job-id", "summary", mock_job_tracker)

    @pytest.mark.usefixtures("mock_job_tracker")
    def test_get_job_detail_full_serializes_nan_objective_as_null(self, client: TestClient) -> None:
        """A NaN objective (failed read) must serialize as JSON null, not 500.

        Starlette renders with ``json.dumps(allow_nan=False)``, so a NaN that
        reaches ``data`` would raise and degrade the response into a 500 (the
        web UI then polls a broken endpoint). NaN must surface as ``null``.
        """
        reader = make_opt_reader(
            data=pl.DataFrame({"charge_ltb": [float("nan")], "LTB:HCM1:Setpoint": [1.0]}),
            objective_name=None,  # keeps result_summary out of this test
            environment_name="corrector_tuning_sim",
            objective_direction=None,
            variable_names=["LTB:HCM1:Setpoint"],
            total_evaluations=1,
            available_snapshots={},
        )

        with patch(f"{_OS}.get_optimization_reader", return_value=reader):
            resp = client.get("/jobs/test-job-id?detail=full")

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == [{"charge_ltb": None, "LTB:HCM1:Setpoint": 1.0}]

    @pytest.mark.usefixtures("mock_job_tracker")
    def test_get_job_detail_full_drops_nan_snapshot_entry(self, client: TestClient) -> None:
        """A NaN inside a captured snapshot (e.g. initial objective unreadable)
        must not 500. Snapshots are sparse reference points, so the non-finite
        entry is dropped (mirroring summary.py), leaving finite entries intact."""
        reader = make_opt_reader(
            data=None,
            objective_name=None,
            environment_name="corrector_tuning_sim",
            objective_direction=None,
            variable_names=["LTB:HCM1:Setpoint"],
            total_evaluations=1,
            available_snapshots={"initial": {"charge_ltb": float("nan"), "LTB:HCM1:Setpoint": 1.0}},
        )

        with patch(f"{_OS}.get_optimization_reader", return_value=reader):
            resp = client.get("/jobs/test-job-id?detail=full")

        assert resp.status_code == 200
        body = resp.json()
        assert body["snapshots"] == {"initial": {"LTB:HCM1:Setpoint": 1.0}}


class TestOptimizationEndpoint:
    """Tests for GET /jobs/{job_id}?detail=... endpoint (optimization reads).

    Consolidated from the former test_api.py.
    """

    @pytest.mark.parametrize(
        (
            "job_id",
            "job_status",
            "err_msg",
            "results_path",
            "expected_status",
            "expected_results_path",
            "expected_message",
            "expects_mark_completed",
        ),
        [
            pytest.param(
                "test-job-id",
                JobStatus.RUNNING,
                None,
                None,
                "running",
                None,
                None,
                False,
                id="returns_reader_status",
            ),
            pytest.param(
                "completed-job",
                JobStatus.COMPLETED,
                None,
                Path("/app/results/test_env/2026-02-05_23-50-01"),
                "completed",
                "test_env/2026-02-05_23-50-01",
                None,
                True,
                id="with_completed_job",
            ),
            pytest.param(
                "error-job",
                JobStatus.ERROR,
                "Something went wrong",
                None,
                "error",
                None,
                "Something went wrong",
                True,  # ERROR is terminal, so the tracker is told to mark completed
                id="with_error_status",
            ),
        ],
    )
    def test_summary_status_variants(
        self,
        client: TestClient,
        mock_job_tracker: MagicMock,
        job_id: str,
        job_status: JobStatus,
        err_msg: str | None,
        results_path: Path | None,
        expected_status: str,
        expected_results_path: str | None,
        expected_message: str | None,
        expects_mark_completed: bool,
    ) -> None:
        """Summary surfaces status, results path, and error message from the reader."""
        mock_job_tracker.get_job.return_value = job_meta(job_id)
        mock_reader = make_opt_reader(
            status=job_status, error_message=err_msg, last_results_path=results_path, data=None
        )

        with patch(f"{_OS}.get_optimization_reader", return_value=mock_reader):
            response = client.get(f"/jobs/{job_id}?detail=summary")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == expected_status
        assert data["results_path"] == expected_results_path
        assert data["error_message"] == expected_message
        if expects_mark_completed:
            mock_job_tracker.mark_completed.assert_called_once_with(job_id)
        else:
            mock_job_tracker.mark_completed.assert_not_called()

    @pytest.mark.parametrize(
        ("get_side_effect", "detail_tokens"),
        [
            pytest.param(
                redis.exceptions.ConnectionError("boom"),
                ("Failed to read job state", "boom"),
                id="redis_connection_error",
            ),
            pytest.param(
                lambda *_a, **_k: "unexpected string",
                ("invalid type",),
                id="corrupt_non_bytes_value",
            ),
        ],
    )
    def test_returns_503_when_real_reader_fails(
        self,
        client: TestClient,
        mock_job_tracker: MagicMock,
        seam_redis_mock: MagicMock,
        get_side_effect: object,
        detail_tokens: tuple[str, ...],
    ) -> None:
        """Real OptimizationState propagates both a redis.exceptions.ConnectionError
        out of get() and the ValueError it raises on a non-bytes Redis value; the
        router wrapper turns each into a 503.

        Distinct from ``TestGetJobStatus``'s connection-error test: this one hits
        ``?detail=summary``, which routes through ``build_optimization_response``
        rather than the plain ``read_job_status`` path."""
        mock_job_tracker.get_job.return_value = job_meta()
        seam_redis_mock.get.side_effect = get_side_effect
        response = client.get("/jobs/test-job-id?detail=summary")

        assert response.status_code == 503
        for token in detail_tokens:
            assert token in response.json()["detail"]

    def test_full_returns_comprehensive_state(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        """detail=full returns comprehensive state including data and snapshots."""
        mock_job_tracker.get_job.return_value = job_meta()
        mock_reader = make_opt_reader(
            last_results_path=Path("/app/results/test_env/2026-02-05_23-50-01"),
            logs="Test logs",
            data_version=5,
            objective_name="test_objective",
            environment_name="test_environment",
            data=pl.DataFrame({"x": [1, 2], "test_objective": [3, 4]}),
            available_snapshots={"initial": {"var1": 1.0}},
            objective_direction="MINIMIZE",
            variable_names=["x"],
            variable_bounds={"x": [0.0, 5.0]},
            total_evaluations=36,
            metadata=None,
        )

        with patch(f"{_OS}.get_optimization_reader", return_value=mock_reader):
            response = client.get("/jobs/test-job-id?detail=full")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "running"
        assert data["logs"] == "Test logs"
        assert data["data_version"] == 5
        assert data["objective_name"] == "test_objective"
        assert data["environment_name"] == "test_environment"
        assert data["results_path"] == "test_env/2026-02-05_23-50-01"
        assert data["data"] == [{"x": 1, "test_objective": 3}, {"x": 2, "test_objective": 4}]
        assert data["snapshots"] == {"initial": {"var1": 1.0}}
        assert data["objective_direction"] == "MINIMIZE"
        assert data["total_evaluations"] == 36
        assert data["variable_bounds"] == {"x": [0.0, 5.0]}


class TestOptimizationSummary:
    """Verify result_summary is populated in the summary status response."""

    def test_summary_contains_best_points(self, client: TestClient) -> None:
        mock_reader = make_opt_reader()
        with patch(
            "osprey.services.optimizer.api.optimization_status.get_optimization_reader",
            return_value=mock_reader,
        ):
            resp = client.get("/jobs/test-job-id?detail=summary")

        assert resp.status_code == 200
        body = resp.json()
        assert body["result_summary"] is not None
        summary = body["result_summary"]
        assert len(summary["best_points"]) == 3
        # Best point first (0.9)
        assert summary["best_points"][0]["values"]["obj"] == 0.9
        assert summary["best_points"][0]["delta_vs_initial"]["obj"] == pytest.approx(0.6)
        assert summary["best_points"][0]["delta_vs_target"]["obj"] == pytest.approx(-0.1)
        assert summary["initial_point"] == {"var1": 0.0, "obj": 0.3}
        assert summary["target_value"] == {"obj": 1.0}

        # Enriched context fields on the same payload.
        assert summary["objective_name"] == "obj"
        assert summary["direction"] == "MAXIMIZE"
        assert summary["total_iterations"] == 3
        assert summary["best_objective_value"] == 0.9
        assert summary["initial_objective_value"] == 0.3
        assert summary["improved_over_initial"] is True
        assert summary["text_summary"] is not None
        assert len(summary["text_summary"]) > 0
        ann = summary["best_points"][0].get("annotated_values")
        assert ann is not None
        assert ann["var1"]["description"] == "Var 1"
        assert ann["obj"]["units"] == "mA"

    @pytest.mark.parametrize(
        "reader_overrides",
        [
            pytest.param({"data": None}, id="no-data"),
            pytest.param({"status": JobStatus.IDLE}, id="idle-status"),
        ],
    )
    def test_result_summary_is_none(
        self, client: TestClient, reader_overrides: dict[str, object]
    ) -> None:
        """result_summary is omitted when there's no data or the job is idle. (The
        unresolvable-direction case is owned by
        ``TestDirectionFallbackToMaximize::test_none_direction_logs_warning_and_omits_summary``,
        which asserts the same omission plus the warning it logs.)"""
        with patch(
            "osprey.services.optimizer.api.optimization_status.get_optimization_reader"
        ) as mk_reader:
            mk_reader.return_value = make_opt_reader(**reader_overrides)
            resp = client.get("/jobs/test-job-id?detail=summary")
        assert resp.status_code == 200
        assert resp.json()["result_summary"] is None

    @pytest.mark.parametrize(
        ("write_model", "expected_model_file"),
        [
            pytest.param(True, "model.joblib", id="with_model_file"),
            pytest.param(False, None, id="without_model_file"),
        ],
    )
    def test_model_file_reported_from_last_results_dir(
        self, client: TestClient, tmp_path: Path, write_model: bool, expected_model_file: str | None
    ) -> None:
        """Summary reports model.joblib when the last_results dir holds one, and
        leaves model_file None when it does not."""
        run_dir = tmp_path / "parent" / "run"
        run_dir.mkdir(parents=True)
        if write_model:
            (run_dir / "model.joblib").write_bytes(b"x")
        data = pl.DataFrame({"var1": [0.0, 1.0], "obj": [0.5, 0.9]})
        with patch(
            "osprey.services.optimizer.api.optimization_status.get_optimization_reader"
        ) as mk_reader:
            mk_reader.return_value = make_opt_reader(
                status=JobStatus.COMPLETED,
                last_results_path=run_dir,
                data=data,
                available_snapshots={},
                metadata=None,
            )
            resp = client.get("/jobs/test-job-id?detail=summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["result_summary"] is not None
        assert body["result_summary"]["model_file"] == expected_model_file

    @pytest.mark.parametrize(
        "bad_section",
        [
            pytest.param({"variables": ["not", "a", "dict"]}, id="variables_not_a_dict"),
            pytest.param({"variables": {"var1": "junk"}}, id="variable_entry_not_a_dict"),
            pytest.param({"observables": None}, id="observables_absent"),
            pytest.param({"observables": {"obj": 42}}, id="observable_entry_not_a_dict"),
        ],
    )
    def test_corrupt_metadata_section_is_dropped_with_warning(
        self, client: TestClient, caplog: pytest.LogCaptureFixture, bad_section: dict[str, object]
    ) -> None:
        """Only the outer metadata container is type-checked on read; a corrupt
        ``variables``/``observables`` section used to reach ``SummaryContext`` and
        500 the status endpoint (#85). It is dropped with a warning naming the key;
        the intact section still annotates the best points. An absent section is
        the normal legacy case and must not warn."""
        metadata: dict[str, object] = {
            "variables": {"var1": {"description": "Var 1", "units": "V", "step_size": 0.1}},
            "observables": {"obj": {"description": "Observable 1", "units": "mA"}},
        }
        metadata.update(bad_section)
        with (
            caplog.at_level(
                logging.WARNING, logger="osprey.services.optimizer.api.optimization_status"
            ),
            patch(
                f"{_OS}.get_optimization_reader", return_value=make_opt_reader(metadata=metadata)
            ),
        ):
            resp = client.get("/jobs/test-job-id?detail=summary")
        assert resp.status_code == 200
        summary = resp.json()["result_summary"]
        assert summary is not None
        assert summary["best_points"][0]["annotated_values"] is not None
        (key,) = bad_section
        if bad_section[key] is None:
            assert not any(f"'{key}'" in msg for msg in caplog.messages)
        else:
            assert any(f"'{key}'" in msg and "test-job-id" in msg for msg in caplog.messages)


class TestDirectionFallbackToMaximize:
    """get_objective_direction() returning None logs a warning and omits
    result_summary from the status response rather than fabricating a
    MAXIMIZE direction. (regression guard for #38, #48)"""

    def test_none_direction_logs_warning_and_omits_summary(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When direction is None, a warning is logged and result_summary is None."""
        mock_reader = make_opt_reader(
            data=pl.DataFrame({"x": [1.0, 2.0, 3.0], "obj": [10.0, 5.0, 1.0]}),
            variable_names=["x"],
            available_snapshots={},
            objective_direction=None,
            metadata=None,
        )

        with (
            caplog.at_level(
                logging.WARNING, logger="osprey.services.optimizer.api.optimization_status"
            ),
            patch(f"{_OS}.get_optimization_reader", return_value=mock_reader),
        ):
            response = client.get("/jobs/test-job-id?detail=summary")

        assert response.status_code == 200
        assert response.json().get("result_summary") is None
        assert any("Objective direction not set" in msg for msg in caplog.messages)

    def test_explicit_direction_is_used(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When direction IS set, it's used correctly and no warning is logged."""
        mock_reader = make_opt_reader(
            data=pl.DataFrame({"x": [1.0, 2.0, 3.0], "obj": [10.0, 5.0, 1.0]}),
            variable_names=["x"],
            available_snapshots={},
            objective_direction="MINIMIZE",
            metadata=None,
        )

        with (
            caplog.at_level(
                logging.WARNING, logger="osprey.services.optimizer.api.optimization_status"
            ),
            patch(f"{_OS}.get_optimization_reader", return_value=mock_reader),
        ):
            response = client.get("/jobs/test-job-id?detail=summary")

        assert response.status_code == 200
        summary = response.json().get("result_summary")
        assert summary is not None
        assert summary["direction"] == "MINIMIZE"
        assert not any("Objective direction not set" in msg for msg in caplog.messages)


class TestListJobsIncludesCreated:
    def test_lists_created_and_running_excludes_terminal(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        mock_job_tracker.list_job_metas.return_value = [
            job_meta("c", status=JobStatus.CREATED),
            job_meta("r", status=JobStatus.RUNNING),
            job_meta("done", status=JobStatus.COMPLETED),
        ]

        def fake_read(_tracker, job_id):
            status_map = {
                "c": JobStatus.CREATED,
                "r": JobStatus.RUNNING,
                "done": JobStatus.COMPLETED,
            }
            return JobResponse(
                job_id=job_id, job_type=JobType.OPTIMIZATION, status=status_map[job_id]
            )

        with patch(f"{_JOBS}.read_job_status", side_effect=fake_read):
            resp = client.get("/jobs")

        assert resp.status_code == 200
        ids = {row["job_id"] for row in resp.json()}
        assert ids == {"c", "r"}  # terminal "done" excluded


class TestUpdateJobConfig:
    def test_update_job_config_returns_status(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        with (
            patch("osprey.services.optimizer.api.routers.jobs._require_optimization"),
            patch("osprey.services.optimizer.api.routers.jobs.update_optimization_config") as upd,
        ):
            upd.return_value = JobResponse(
                job_id="j1", job_type=JobType.OPTIMIZATION, status=JobStatus.CREATED
            )
            resp = client.put(
                "/jobs/j1/config",
                json={"environment_name": "sim", "objective_name": "obj"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "created"
        # The router must forward the parsed body + the exact job_id/tracker/registry;
        # a bare assert_called_once would pass even with a wrong job_id or config.
        upd.assert_called_once_with(
            "j1",
            OptimizationConfig(environment_name="sim", objective_name="obj"),
            mock_job_tracker,
            app.state.environment_registry,
        )

    def test_get_job_config_returns_config(
        self, client: TestClient, seam_redis_mock: MagicMock
    ) -> None:
        # Drive the REAL read_optimization_config against the redis seam so this
        # covers the GET route wiring + parse, not FastAPI echoing a mock's return.
        # (The pure-function read path is also covered by
        # test_read_config_returns_stored_config in test_api_optimization_lifecycle.)
        stored = OptimizationConfig(
            environment_name="sim", control_target="va", objective_name="obj"
        )
        seam_redis_mock.get.return_value = stored.model_dump_json()
        with patch("osprey.services.optimizer.api.routers.jobs._require_optimization"):
            resp = client.get("/jobs/j1/config")

        assert resp.status_code == 200
        assert resp.json()["environment_name"] == "sim"
        assert resp.json()["objective_name"] == "obj"


class TestCurrentJob:
    @pytest.mark.parametrize("pointer", [None, "job-7"], ids=["unset", "set"])
    def test_get_current_job(
        self, client: TestClient, mock_job_tracker: MagicMock, pointer: str | None
    ) -> None:
        # "current" must resolve to the current-job handler, not GET /jobs/{job_id}:
        # this is the CurrentJobResponse shape, not a 404 "Job current not found".
        mock_job_tracker.get_current_job.return_value = pointer
        resp = client.get("/jobs/current")
        assert resp.status_code == 200
        assert resp.json() == {"job_id": pointer}

    def test_put_current_job_sets_pointer(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        mock_job_tracker.job_exists.return_value = True
        resp = client.put("/jobs/current", json={"job_id": "job-7"})
        assert resp.status_code == 200
        assert resp.json() == {"job_id": "job-7"}
        mock_job_tracker.set_current_job.assert_called_once_with("job-7")

    def test_put_current_job_404_for_unknown(
        self, client: TestClient, mock_job_tracker: MagicMock
    ) -> None:
        mock_job_tracker.job_exists.return_value = False
        resp = client.put("/jobs/current", json={"job_id": "ghost"})
        assert resp.status_code == 404
        mock_job_tracker.set_current_job.assert_not_called()


def _stub_optimization_start(
    monkeypatch: pytest.MonkeyPatch, mock_job_tracker: MagicMock, *, target: str = "va"
) -> None:
    """Make an optimization start on ``target`` succeed past the guards under test."""
    mock_job_tracker.get_job.return_value = job_meta("job-1")
    jobs = "osprey.services.optimizer.api.routers.jobs"
    config = make_gate_config().model_copy(update={"control_target": target})
    monkeypatch.setattr(f"{jobs}.read_optimization_config", lambda job_id: config)
    monkeypatch.setattr(f"{jobs}.load_environment", lambda *a, **k: MagicMock())
    monkeypatch.setattr(f"{jobs}.run_start_gates", lambda *a, **k: None)
    monkeypatch.setattr(
        f"{jobs}.start_optimization_job",
        lambda jid, _tracker: JobResponse(
            job_id=jid, job_type=JobType.OPTIMIZATION, status=JobStatus.RUNNING
        ),
    )


def test_start_on_the_simulator_skips_the_health_probe(
    client: TestClient, mock_job_tracker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def probe(*_args: Any) -> bool:
        raise AssertionError("the simulator is never health-probed")

    monkeypatch.setattr(
        "osprey.services.optimizer.api.environment_ops._environment_connected", probe
    )
    _stub_optimization_start(monkeypatch, mock_job_tracker, target="va")
    resp = client.post("/jobs/job-1/start")
    assert resp.status_code == 200


def test_start_override_unhealthy_query_param_bypasses_503(
    client: TestClient, mock_job_tracker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``?override_unhealthy=true`` must reach guard_environment_start: an unhealthy
    real target 503s a plain start, but the query param lets the start proceed."""
    monkeypatch.setattr(
        "osprey.services.optimizer.api.environment_ops._environment_connected",
        lambda n, r, t: False,
    )
    _stub_optimization_start(monkeypatch, mock_job_tracker, target="live")

    # Without the override the unhealthy non-sim env refuses the start.
    resp = client.post("/jobs/job-1/start")
    assert resp.status_code == 503

    # With the override the guard is bypassed and the start dispatches.
    resp = client.post("/jobs/job-1/start?override_unhealthy=true")
    assert resp.status_code == 200


def test_create_optimization_returns_created_and_sets_current_job(
    client: TestClient, mock_job_tracker: MagicMock
) -> None:
    with (
        patch(
            "osprey.services.optimizer.api.routers.optimization.create_optimization_job",
            return_value="new-id",
        ),
        patch("osprey.services.optimizer.api.routers.optimization.read_job_status") as read_f,
    ):
        read_f.return_value = JobResponse(
            job_id="new-id", job_type=JobType.OPTIMIZATION, status=JobStatus.CREATED
        )
        resp = client.post(
            "/optimizations", json={"environment_name": "sim", "objective_name": "obj"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "new-id"
    assert body["status"] == "created"
    assert body["job_type"] == "optimization"
    # New Run must point the server-side current-job at the freshly created job,
    # atomically with creation. Otherwise the web follow-poll can read the stale
    # pointer and revert the selection before Start, starting the wrong job.
    mock_job_tracker.set_current_job.assert_called_once_with("new-id")


def test_create_optimization_rejects_typoed_field(client: TestClient) -> None:
    # 'objectve_name' is a typo of 'objective_name'. Pre-fix this body created
    # a job with no objective; post-fix it must 422 with extra_forbidden.
    resp = client.post(
        "/optimizations",
        json={"environment_name": "sim", "objectve_name": "charge_gtl"},
    )
    assert resp.status_code == 422
    assert any(err["type"] == "extra_forbidden" for err in resp.json()["detail"])


def test_old_start_route_is_gone(client: TestClient) -> None:
    resp = client.post("/optimization/start", json={"environment_name": "sim"})
    assert resp.status_code == 404


def test_start_optimization_missing_environment_name_422(client: TestClient) -> None:
    """optimization_lifecycle.py: missing environment_name -> 422."""
    resp = client.post("/optimizations", json={"objective_name": "obj"})
    assert resp.status_code == 422
    assert "environment_name is required" in resp.json()["detail"]
