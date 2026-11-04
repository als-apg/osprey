"""Tests for api/job_tracker.py — JobTracker, create_job_state, and JobLifecycle.

Canonical home for api/job_tracker.py, covering JobTracker methods,
claim_active_job, create_job_state, and the JobLifecycle context manager
(heartbeat loop, __enter__/__exit__ status transitions, and real-Redis
integration), consolidating what was previously split out into
test_job_lifecycle.py. Also covers the small api/ job-lifecycle ops that sit
next to the tracker: launch (job_control.launch_job) and status read
(job_status), folded in from test_job_ops.py, and the service-shutdown stop of
launched jobs (job_control.stop_launched_jobs) with the worker's signal handler.
"""

from __future__ import annotations

import json
import multiprocessing
import signal
import time
from collections.abc import Generator
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from osprey_optimizer.api import job_control
from osprey_optimizer.api import job_status as js
from osprey_optimizer.api.job_control import (
    KILL_WAIT_SECONDS,
    SHUTDOWN_KILLED_MESSAGE,
    TERMINATE_WAIT_SECONDS,
    launch_job,
    stop_launched_jobs,
)
from osprey_optimizer.api.job_tracker import (
    ACTIVE_JOB_KEY,
    CANCELLED_MESSAGE,
    HEARTBEAT_INTERVAL_SECONDS,
    JobLifecycle,
    JobMeta,
    JobSlotLostError,
    JobTracker,
    create_job_state,
)
from osprey_optimizer.core.state import TERMINAL_STATE_TTL_SECONDS
from osprey_optimizer.models.enums import JobStatus, JobType
from osprey_optimizer.models.exceptions import CancellationError
from osprey_optimizer.optimization.state import OptimizationStateWriter, _OptimizationState
from osprey_optimizer.utils import cancellation
from osprey_optimizer.utils.cancellation import get_cancellation_token
from osprey_optimizer.utils.job_keys import job_scan_pattern, meta_key, state_prefix
from osprey_optimizer.utils.process import (
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    SHUTDOWN_GRACE_ENV,
    install_worker_signal_handlers,
    shutdown_grace_seconds,
)
from osprey_optimizer.utils.redis_client import get_redis_client
from osprey_optimizer.utils.redis_state_store import RUNNING_STATE_TTL_SECONDS
from osprey_optimizer.utils.serialization import deserialize_obj

from . import _shutdown_workers

_JC = "osprey_optimizer.api.job_control"
_JS = "osprey_optimizer.api.job_status"


def meta_json(
    job_id: str = "j1",
    *,
    status: JobStatus = JobStatus.RUNNING,
    pid: int = 1,
    job_type: JobType = JobType.OPTIMIZATION,
) -> str:
    """A serialized ``JobMeta``, exactly as the tracker writes ``job:{id}:meta``."""
    return JobMeta(
        job_id=job_id, pid=pid, status=status, created_at=1712000000.0, job_type=job_type
    ).model_dump_json()


def _corrupt_meta_error() -> ValidationError:
    """The real ValidationError a corrupt ``job:{id}:meta`` record parses into."""
    with pytest.raises(ValidationError) as exc:
        JobMeta.model_validate_json(json.dumps({"job_id": "j1", "job_type": "bogus"}))
    return exc.value


@pytest.mark.parametrize(
    ("second_value", "expected_ids"),
    [
        # A key that expired between the scan and the MGET comes back as None
        # and is silently skipped.
        pytest.param(None, {"a"}, id="skips_key_with_no_value"),
        # An unparsable record is skipped too, so one corrupt job cannot sink
        # the whole GET /jobs listing.
        pytest.param(
            json.dumps({"job_id": "b", "job_type": "bogus"}), {"a"}, id="skips_unparsable_record"
        ),
    ],
)
def test_list_job_metas(
    seam_redis_mock: MagicMock, second_value: str | None, expected_ids: set[str]
) -> None:
    seam_redis_mock.scan_iter.return_value = ["job:a:meta", "job:b:meta"]
    seam_redis_mock.mget.return_value = [meta_json("a", status=JobStatus.CREATED), second_value]
    metas = JobTracker().list_job_metas()

    seam_redis_mock.scan_iter.assert_called_once_with(match="job:*:meta")
    # Scanned keys are fetched in one round trip, not one GET per key.
    seam_redis_mock.mget.assert_called_once_with(["job:a:meta", "job:b:meta"])
    assert {m.job_id for m in metas} == expected_ids


def test_register_created_job_writes_created_meta_without_claiming_slot(
    seam_redis_mock: MagicMock,
) -> None:
    JobTracker().register_created_job("job-1", job_type=JobType.OPTIMIZATION)

    set_calls = {c.args[0]: c for c in seam_redis_mock.set.call_args_list}
    meta = json.loads(set_calls["job:job-1:meta"].args[1])
    assert meta["status"] == JobStatus.CREATED.value
    assert meta["job_type"] == JobType.OPTIMIZATION.value
    # The active slot must NOT be claimed (non-reserving).
    assert ACTIVE_JOB_KEY not in set_calls


@pytest.fixture
def tracker(redis_server: str) -> JobTracker:
    """Real-Redis-backed tracker; flush the DB so each test starts clean."""
    del redis_server  # wanted for the server it starts, not its URL
    t = JobTracker()
    t._get_redis().flushdb()
    return t


class TestJobTrackerMissing:
    """Tests for JobTracker methods not covered in test_jobs_api.py."""

    @pytest.mark.parametrize(
        "stored",
        [
            pytest.param(None, id="missing"),
            pytest.param(meta_json("j1", pid=1234), id="found"),
        ],
    )
    def test_get_job(self, redis_mocks, stored: str | None):
        mock_redis, _ = redis_mocks
        mock_redis.get.return_value = stored
        tracker = JobTracker()
        expected = None if stored is None else JobMeta.model_validate_json(stored)
        assert tracker.get_job("j1") == expected

    def test_get_job_raises_on_corrupt_record(self, redis_mocks):
        """A record that no longer parses raises rather than returning a partial
        job — the job_status read seam maps that to a 500."""
        mock_redis, _ = redis_mocks
        mock_redis.get.return_value = json.dumps({"job_id": "j1", "job_type": "bogus"})
        with pytest.raises(ValidationError):
            JobTracker().get_job("j1")

    def test_meta_serializes_enums_as_canonical_value_strings(self):
        """The stored JSON must keep the exact field names and enum value strings
        the tracker has always written (Redis records survive a redeploy)."""
        stored = json.loads(meta_json("j1", pid=1234))
        assert stored["status"] == JobStatus.RUNNING.value == "running"
        assert stored["job_type"] == JobType.OPTIMIZATION.value == "optimization"
        assert stored["job_id"] == "j1"
        assert stored["pid"] == 1234

    @pytest.mark.parametrize(
        ("method", "expected_ttl"),
        [
            pytest.param(
                "refresh_heartbeat", RUNNING_STATE_TTL_SECONDS, id="heartbeat_running_ttl"
            ),
            pytest.param("mark_completed", TERMINAL_STATE_TTL_SECONDS, id="terminal_sweep_ttl"),
        ],
    )
    def test_sweep_applies_ttl_to_whole_namespace(
        self, redis_mocks, method: str, expected_ttl: int
    ):
        """Both sweeps re-apply their TTL across the whole job:{id}:* namespace
        (meta + state + cancel), not just metadata — so a live job's state/cancel
        keys never lapse between beats, and a terminal job's entire footprint
        self-reclaims. Each sweep runs on the tracker's str client in one pass and
        is value-encoding-agnostic. The EXPIREs go out in one pipelined batch."""
        mock_redis, _ = redis_mocks
        mock_redis.get.side_effect = [
            meta_json("j1"),
            None,  # active job check (mark_completed only)
        ]
        mock_redis.scan_iter.return_value = ["job:j1:meta", "job:j1:state:status", "job:j1:cancel"]
        tracker = JobTracker()
        getattr(tracker, method)("j1")
        mock_redis.scan_iter.assert_called_once_with(match=job_scan_pattern("j1"))
        pipe = mock_redis.pipeline.return_value
        assert pipe.expire.call_args_list == [
            call("job:j1:meta", expected_ttl),
            call("job:j1:state:status", expected_ttl),
            call("job:j1:cancel", expected_ttl),
        ]
        pipe.execute.assert_called_once()

    def test_mark_completed_sweeps_namespace_even_when_meta_lapsed(self, redis_mocks):
        """When ``job:{id}:meta`` has already expired there is nothing to rewrite,
        but the rest of the namespace (state + cancel) can still be alive and must
        still be swept under the terminal TTL, or it leaks (#107)."""
        mock_redis, _ = redis_mocks
        mock_redis.get.side_effect = [None, None]  # meta lapsed; no active job
        mock_redis.scan_iter.return_value = ["job:j1:state:status", "job:j1:cancel"]
        JobTracker().mark_completed("j1")
        mock_redis.set.assert_not_called()
        mock_redis.scan_iter.assert_called_once_with(match=job_scan_pattern("j1"))
        pipe = mock_redis.pipeline.return_value
        assert pipe.expire.call_args_list == [
            call("job:j1:state:status", TERMINAL_STATE_TTL_SECONDS),
            call("job:j1:cancel", TERMINAL_STATE_TTL_SECONDS),
        ]
        pipe.execute.assert_called_once()

    @pytest.mark.parametrize(
        ("status", "active_job", "expect_write", "expect_active_cleared"),
        [
            pytest.param(
                JobStatus.COMPLETED, None, False, False, id="already_completed_skips_rewrite"
            ),
            pytest.param(
                JobStatus.RUNNING, "different-job", True, False, id="other_job_active_not_cleared"
            ),
            pytest.param(JobStatus.RUNNING, "j1", True, True, id="own_job_active_cleared"),
        ],
    )
    def test_mark_completed(
        self,
        redis_mocks,
        status: JobStatus,
        active_job: str | None,
        expect_write: bool,
        expect_active_cleared: bool,
    ):
        """mark_completed rewrites the job's status unless it is already terminal, and
        clears the active-job pointer only when it points at this job."""
        mock_redis, _ = redis_mocks
        mock_redis.get.side_effect = [
            meta_json("j1", status=status),
            active_job,
        ]
        tracker = JobTracker()
        tracker.mark_completed("j1")

        if expect_write:
            mock_redis.set.assert_called_once()
            key, payload = mock_redis.set.call_args.args
            assert key == meta_key("j1")
            written = json.loads(payload)
            assert written["status"] == JobStatus.COMPLETED.value
            assert "completed_at" in written
        else:
            mock_redis.set.assert_not_called()
        if expect_active_cleared:
            mock_redis.delete.assert_called_once_with(ACTIVE_JOB_KEY)
        else:
            mock_redis.delete.assert_not_called()


class TestClaimActiveJob:
    """claim_active_job at both fidelities: the mocked atomic-Lua call shape and
    the real-Redis end state. Both layers are intentional — neither subsumes the
    other (mock proves the call contract, real Redis proves the slot outcome)."""

    @pytest.mark.parametrize(
        ("lua_result", "claimed"),
        [
            pytest.param(1, True, id="claim_won"),  # Lua: NX claim won, meta written
            pytest.param(0, False, id="claim_lost"),  # Lua: slot already held, nothing written
        ],
    )
    def test_claim_active_job_uses_atomic_script(self, redis_mocks, lua_result: int, claimed: bool):
        """The active slot is claimed via the atomic Lua script (KEYS = active +
        meta) so concurrent starts are mutually exclusive; the Lua return decides
        the outcome. Either way no metadata is deleted, so a pre-existing CREATED
        meta at the meta key survives a lost race (H3)."""
        mock_redis, _ = redis_mocks
        script = mock_redis.register_script.return_value
        script.return_value = lua_result
        tracker = JobTracker()
        assert tracker.claim_active_job("j1", 0) is claimed
        call_kwargs = script.call_args.kwargs
        assert call_kwargs["keys"] == [ACTIVE_JOB_KEY, meta_key("j1")]
        assert call_kwargs["args"][0] == "j1"
        # args[1] is the RUNNING meta the winner writes — decode it, don't trust the shape.
        meta = json.loads(call_kwargs["args"][1])
        assert meta["job_id"] == "j1"
        assert meta["status"] == JobStatus.RUNNING.value
        assert meta["pid"] == 0
        assert call_kwargs["args"][2] == RUNNING_STATE_TTL_SECONDS
        mock_redis.delete.assert_not_called()

    def test_claim_failure_preserves_created_meta(self, tracker: JobTracker) -> None:
        """A CREATED job whose start loses the slot race keeps its metadata."""
        # Another job already holds the slot, with RUNNING meta.
        other = "other-running-job"
        redis = tracker._get_redis()
        redis.set(ACTIVE_JOB_KEY, other)
        redis.set(meta_key(other), meta_json(other))

        # Our job is registered CREATED (the launch_job pre-condition).
        created = "created-job"
        tracker.register_created_job(created, job_type=JobType.OPTIMIZATION)
        assert tracker.job_exists(created)

        # Contended claim loses the race.
        claimed = tracker.claim_active_job(created, 0, job_type=JobType.OPTIMIZATION)

        assert claimed is False
        # The CREATED meta must still be present and unchanged.
        assert tracker.job_exists(created) is True
        meta = tracker.get_job(created)
        assert meta is not None
        assert meta.status is JobStatus.CREATED
        # The pre-existing slot holder is untouched.
        assert redis.get(ACTIVE_JOB_KEY) == other

    def test_claim_success_sets_active_and_running_meta(self, tracker: JobTracker) -> None:
        """An uncontended claim takes the slot and writes RUNNING meta atomically."""
        job_id = "fresh-job"
        tracker.register_created_job(job_id, job_type=JobType.OPTIMIZATION)

        claimed = tracker.claim_active_job(job_id, 123, job_type=JobType.OPTIMIZATION)

        assert claimed is True
        redis = tracker._get_redis()
        assert redis.get(ACTIVE_JOB_KEY) == job_id
        meta = tracker.get_job(job_id)
        assert meta is not None
        assert meta.status is JobStatus.RUNNING
        assert meta.pid == 123
        # TTL applied (RUNNING window), not persistent.
        assert 0 < redis.ttl(meta_key(job_id)) <= 60


class TestCreateJobState:
    """Tests for the create_job_state factory function.

    The factory is a thin JobType dispatcher over ``core.state_access``.
    Construction + Redis wiring coverage lives in ``test_state.py``;
    these tests only verify the dispatch contract.
    """

    @pytest.mark.parametrize(
        ("job_type", "expected_cls"),
        [
            pytest.param(JobType.OPTIMIZATION, _OptimizationState, id="optimization"),
        ],
    )
    def test_job_type_returns_matching_state(self, job_type: JobType, expected_cls: type) -> None:
        with patch("osprey_optimizer.utils.redis_client.get_redis_client"):
            writer = create_job_state(job_type, "job-1")
        assert isinstance(writer, expected_cls)


def test_set_and_get_current_job_round_trip(tracker: JobTracker) -> None:
    tracker.register_created_job("job-1", job_type=JobType.OPTIMIZATION)
    tracker.set_current_job("job-1")
    assert tracker.get_current_job() == "job-1"


def test_get_current_job_none_when_unset_or_target_gone(tracker: JobTracker) -> None:
    assert tracker.get_current_job() is None  # pointer never set
    tracker.set_current_job("ghost-job")  # never registered → job_exists False
    assert tracker.get_current_job() is None
    # pointer was cleared, so a second call is still None
    assert tracker.get_current_job() is None


class TestJobTrackerRedisOps:
    """Tests for JobTracker Redis operations (single-mock seam style)."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            pytest.param(None, None, id="no_active_job"),
            pytest.param("running", "job-123", id="running_job_returned"),
            pytest.param("error", None, id="stale_error_cleared"),
            pytest.param("completed", None, id="stale_completed_cleared"),
        ],
    )
    def test_get_active_job(self, seam_redis_mock, status: str | None, expected: str | None):
        """get_active_job returns the running job, and clears the active key when
        the pointed-at job is no longer running."""
        if status is None:
            seam_redis_mock.get.return_value = None
        else:
            seam_redis_mock.get.side_effect = [
                "job-123",
                meta_json("job-123", status=JobStatus(status), pid=1234),
            ]

        tracker = JobTracker()
        result = tracker.get_active_job()

        assert result == expected
        if status in ("error", "completed"):
            seam_redis_mock.delete.assert_called_once_with(ACTIVE_JOB_KEY)

    @pytest.mark.parametrize(
        ("exists_result", "expected"),
        [pytest.param(1, True, id="existing_job"), pytest.param(0, False, id="missing_job")],
    )
    def test_job_exists(self, seam_redis_mock, exists_result: int, expected: bool):
        """job_exists mirrors Redis EXISTS on the job's meta key."""
        seam_redis_mock.exists.return_value = exists_result

        tracker = JobTracker()
        result = tracker.job_exists("job-123")

        assert result is expected
        seam_redis_mock.exists.assert_called_once_with(meta_key("job-123"))


class TestJobLifecycle:
    """Tests for JobLifecycle context manager."""

    @pytest.fixture(autouse=True)
    def _patch_pid(self):
        """Patch os.getpid for all lifecycle tests."""
        with patch("osprey_optimizer.api.job_tracker.os.getpid", return_value=12345):
            yield

    @pytest.fixture(autouse=True)
    def _job_holds_the_slot(self, request: pytest.FixtureRequest) -> None:
        """The slot-checked registration script answers "still holds the slot"."""
        if "redis_mocks" in request.fixturenames:
            str_client, _ = request.getfixturevalue("redis_mocks")
            str_client.register_script.return_value.return_value = 1

    @pytest.mark.parametrize(
        "job_type",
        [pytest.param(JobType.OPTIMIZATION, id="optimization")],
    )
    def test_enter_registers_the_pid_and_leaves_the_status_as_found(
        self, redis_mocks, job_type: JobType
    ):
        mock_redis_str, mock_redis_bin = redis_mocks
        lc = JobLifecycle("job-1", job_type=job_type)
        lc.__enter__()

        # register_job ran the slot-checked script: RUNNING meta under the
        # heartbeat TTL, only if job-1 still holds the slot -- asserted on the
        # decoded payload, not shapes. It never writes the slot itself.
        script = mock_redis_str.register_script.return_value
        script.assert_called_once()
        assert script.call_args.kwargs["keys"] == [ACTIVE_JOB_KEY, meta_key("job-1")]
        job_id_arg, meta_arg, ttl_arg = script.call_args.kwargs["args"]
        assert job_id_arg == "job-1"
        meta = json.loads(meta_arg)
        assert meta["status"] == JobStatus.RUNNING.value
        assert meta["pid"] == 12345
        assert meta["job_type"] == job_type.value
        assert ttl_arg == RUNNING_STATE_TTL_SECONDS
        assert not any(c.args[0] == ACTIVE_JOB_KEY for c in mock_redis_str.set.call_args_list)

        # The launcher set RUNNING before spawning, and a pause or cancel may
        # have landed since: the worker does not write the status on entry.
        status_writes = [
            c for c in mock_redis_bin.set.call_args_list if str(c.args[0]).endswith(":status")
        ]
        assert status_writes == []
        lc._heartbeat.stop(timeout=1)

    def test_enter_refuses_a_job_that_lost_the_slot(self, redis_mocks):
        mock_redis_str, mock_redis_bin = redis_mocks
        mock_redis_str.register_script.return_value.return_value = 0
        lc = JobLifecycle("job-1")

        with pytest.raises(JobSlotLostError, match="lost the active-job slot"):
            lc.__enter__()

        status_writes = [
            c for c in mock_redis_bin.set.call_args_list if str(c.args[0]).endswith(":status")
        ]
        assert [deserialize_obj(c.args[1]) for c in status_writes] == [JobStatus.ERROR]
        assert not lc._heartbeat.is_alive()

    def _enter_with_mock_writer(self, lc: JobLifecycle, mock_redis_str: MagicMock) -> MagicMock:
        """Enter the lifecycle, swap in a mock writer, and prime mark_completed.

        Returns the mock writer so a test can assert the terminal status/message
        __exit__ writes. The RUNNING status set during __enter__ lands on the real
        writer (pre-swap), so the mock sees only __exit__'s terminal write.
        """
        lc.__enter__()
        writer = MagicMock()
        lc.writer = writer
        mock_redis_str.get.side_effect = [
            meta_json(lc.job_id, pid=12345),
            lc.job_id,
        ]
        return writer

    @pytest.mark.parametrize(
        "job_id,exc_type,exc_value,expected_status,expected_message",
        [
            pytest.param(
                "cancel-1",
                CancellationError,
                CancellationError("user cancelled"),
                JobStatus.CANCELLED,
                CANCELLED_MESSAGE,
                id="cancellation_sets_cancelled_with_shared_message",
            ),
            pytest.param(
                "err-1",
                ValueError,
                ValueError("boom detail"),
                JobStatus.ERROR,
                "boom detail",
                id="generic_exception_sets_error_with_str_message",
            ),
            pytest.param(
                "err-2",
                ValueError,
                ValueError(""),
                JobStatus.ERROR,
                "ValueError",
                id="empty_message_falls_back_to_type_name",
            ),
        ],
    )
    def test_exit_sets_terminal_status_and_message(
        self,
        redis_mocks,
        job_id: str,
        exc_type: type[BaseException],
        exc_value: BaseException,
        expected_status: JobStatus,
        expected_message: str,
    ):
        """__exit__'s terminal status/message on an exception: a CancellationError
        → CANCELLED + the shared cancel message (both job types); any other
        exception → ERROR with the cleaned str(exc) message (not a raw
        traceback); an exception with an empty message records its type name,
        never a blank."""
        mock_redis_str, _ = redis_mocks
        lc = JobLifecycle(job_id)
        writer = self._enter_with_mock_writer(lc, mock_redis_str)

        lc.__exit__(exc_type, exc_value, None)

        writer.set_status.assert_called_once_with(expected_status)
        writer.set_error_message.assert_called_once_with(expected_message)

    @pytest.mark.usefixtures("redis_mocks")
    @pytest.mark.parametrize(
        ("refresh_error", "wait_results", "beats", "warnings"),
        [
            pytest.param(None, [False, False, True], 2, 0, id="refreshes_until_stopped"),
            pytest.param(
                RuntimeError("redis down"), [False, True], 1, 1, id="logs_and_continues_on_failure"
            ),
        ],
    )
    def test_heartbeat_loop(
        self,
        refresh_error: Exception | None,
        wait_results: list[bool],
        beats: int,
        warnings: int,
    ):
        """The loop refreshes once per non-stop tick and exits when stopped —
        deterministic via a stubbed stop event (no real timing). A refresh that
        raises is caught and logged so the background thread survives, and the
        loop keeps beating."""
        lc = JobLifecycle("hb-1")
        lc.tracker = MagicMock()
        lc.tracker.refresh_heartbeat.side_effect = refresh_error
        stop = MagicMock()
        stop.wait.side_effect = wait_results
        lc._heartbeat._stop_event = stop

        with patch("osprey_optimizer.utils.interval_thread.logger") as mock_logger:
            lc._heartbeat._loop()

        assert lc.tracker.refresh_heartbeat.call_count == beats
        lc.tracker.refresh_heartbeat.assert_called_with("hb-1")
        stop.wait.assert_called_with(HEARTBEAT_INTERVAL_SECONDS)
        assert mock_logger.warning.call_count == warnings

    def test_enter_starts_heartbeat_and_exit_stops_it(self, redis_mocks):
        """__enter__ launches the daemon heartbeat thread; __exit__ signals stop
        and joins it so no heartbeat survives the lifecycle."""
        mock_redis_str, _ = redis_mocks
        lc = JobLifecycle("hb-2")
        lc.__enter__()
        assert lc._heartbeat.is_alive()

        mock_redis_str.get.side_effect = [
            meta_json("hb-2", pid=12345),
            "hb-2",
        ]
        lc.__exit__(None, None, None)
        assert not lc._heartbeat.is_alive()

    def test_exit_exception_still_marks_completed(self, redis_mocks):
        mock_redis_str, _ = redis_mocks

        mock_redis_str.get.side_effect = [
            meta_json("job-6", pid=12345),
            "job-6",
        ]

        try:
            with JobLifecycle("job-6"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        # mark_completed still ran despite the exception: the job meta was
        # rewritten to COMPLETED — decode the JSON value, not a substring match.
        meta_writes = [
            c for c in mock_redis_str.set.call_args_list if c.args[0] == meta_key("job-6")
        ]
        completed = [json.loads(c.args[1]) for c in meta_writes]
        assert any(
            m["status"] == JobStatus.COMPLETED.value and "completed_at" in m for m in completed
        )

    def test_exit_does_not_suppress_exception(self, redis_mocks):
        mock_redis_str, _ = redis_mocks
        mock_redis_str.get.return_value = None

        with pytest.raises(ValueError, match="propagate me"), JobLifecycle("job-7"):
            raise ValueError("propagate me")

    def test_exit_cleanup_error_logged_not_raised(self, redis_mocks, caplog):
        mock_redis_str, _ = redis_mocks

        # Make mark_completed blow up by having get_job raise
        def get_side_effect(key):
            if ":meta" in key:
                raise Exception("Redis down")

        mock_redis_str.get.side_effect = get_side_effect

        with (
            pytest.raises(ValueError, match="original"),
            caplog.at_level("ERROR"),
            JobLifecycle("job-8"),
        ):
            raise ValueError("original")

        assert "Error during lifecycle cleanup" in caplog.text

    @pytest.mark.usefixtures("redis_mocks")
    @pytest.mark.parametrize(
        ("job_type", "expected_cls"),
        [
            pytest.param(JobType.OPTIMIZATION, OptimizationStateWriter, id="optimization"),
        ],
    )
    def test_job_type_creates_matching_writer(self, job_type: JobType, expected_cls: type):
        lc = JobLifecycle("job-w", job_type=job_type)
        lc.__enter__()
        assert isinstance(lc.writer, expected_cls)


def _launched(job_id: str) -> str:
    """Do what the launcher does before spawning: claim the slot, set RUNNING."""
    assert JobTracker().claim_active_job(job_id, 0, job_type=JobType.OPTIMIZATION)
    create_job_state(JobType.OPTIMIZATION, job_id).set_status(JobStatus.RUNNING)
    return job_id


@pytest.mark.usefixtures("tracker")
class TestJobLifecycleIntegration:
    """Integration tests with real Redis.

    Shares the module-level ``tracker`` fixture (real Redis via ``redis_server``,
    flushed per test) rather than depending on ``redis_server`` directly, so
    there is a single real-Redis setup path for this file.
    """

    def test_full_lifecycle_happy_path_bounds_every_job_key(self):
        """A clean run: RUNNING while the body executes, COMPLETED meta after exit,
        and then EVERY key the job owns — meta, state, and the cancellation token —
        carries a positive TTL. This is the authoritative "nothing leaks" check, and
        it exercises the real path the mocks can't: the tracker's str client
        sweeping the binary-valued state keys."""
        str_client = get_redis_client(binary=False)
        bin_client = get_redis_client(binary=True)

        job_id = _launched("integ-happy-1")
        with JobLifecycle(job_id):
            raw = bin_client.get(f"{state_prefix(job_id)}:status")
            assert raw is not None
            assert deserialize_obj(raw) == JobStatus.RUNNING
            # Touch the cancellation token so its job:{id}:cancel key exists too
            # — it was the key that previously leaked with no TTL.
            get_cancellation_token(job_id)

        job_raw = str_client.get(meta_key(job_id))
        assert job_raw is not None
        job_data = json.loads(job_raw)
        assert job_data["status"] == "completed"

        keys = list(str_client.scan_iter(match=job_scan_pattern(job_id)))
        assert keys, "expected the job to own at least its meta + state keys"
        for k in keys:
            assert str_client.ttl(k) > 0, f"{k} has no TTL after the terminal sweep"

    def test_full_lifecycle_error_path(self):
        bin_client = get_redis_client(binary=True)

        job_id = _launched("integ-error-1")
        with (
            pytest.raises(RuntimeError, match="injected"),
            JobLifecycle(job_id),
        ):
            raise RuntimeError("injected failure")

        error_raw = bin_client.get(f"{state_prefix(job_id)}:error_message")
        assert error_raw is not None
        error_msg = deserialize_obj(error_raw)
        assert "injected failure" in error_msg

    def test_running_keys_are_bounded(self):
        """While a job RUNS, every key it owns carries a positive, bounded TTL —
        so a hard-killed worker (which never reaches the terminal sweep) leaks
        nothing: its whole footprint lapses once the heartbeat stops beating.
        Write-time TTL bounds meta + state inline; one heartbeat beat bounds the
        cancellation token (its TTL comes from the namespace refresh, not inline)."""
        str_client = get_redis_client(binary=False)
        job_id = _launched("integ-running-1")
        with JobLifecycle(job_id) as lc:
            get_cancellation_token(job_id)
            # Drive one beat synchronously (no sleep) so the cancel key is swept too.
            lc.tracker.refresh_heartbeat(job_id)

            keys = list(str_client.scan_iter(match=job_scan_pattern(job_id)))
            assert keys, "expected the running job to own meta + state + cancel keys"
            for k in keys:
                ttl = str_client.ttl(k)
                assert 0 < ttl <= RUNNING_STATE_TTL_SECONDS, (
                    f"{k} unbounded/invalid TTL {ttl} while running"
                )


class TestJobLifecycleBranches:
    """__exit__ branch coverage (hand-mocked collaborators)."""

    @pytest.mark.usefixtures("seam_redis_mock")
    def test_exit_no_exception_sets_completed_without_heartbeat_thread(self) -> None:
        """__exit__ is the single owner of terminal status: no exception → COMPLETED
        and no error message. With no heartbeat thread there is nothing to join, and
        the terminal status is still written."""
        lc = JobLifecycle("no-thread")
        writer = MagicMock()
        lc.writer = writer
        lc.tracker = MagicMock()
        assert not lc._heartbeat.is_alive()  # never started — nothing to join

        lc.__exit__(None, None, None)

        writer.set_status.assert_called_once_with(JobStatus.COMPLETED)
        writer.set_error_message.assert_not_called()


def _call(
    tracker: MagicMock,
    make_writer: MagicMock,
    spawn_target: MagicMock | None = None,
) -> int | None:
    return launch_job(
        tracker=tracker,
        job_id="job-1",
        job_type=JobType.OPTIMIZATION,
        make_writer=make_writer,
        spawn_target=spawn_target or MagicMock(),
        spawn_args=("a",),
    )


def test_claim_failure_raises_409_without_spawning() -> None:
    tracker = MagicMock()
    tracker.claim_active_job.return_value = False
    make_writer = MagicMock()

    with patch(f"{_JC}.multiprocessing") as mp, pytest.raises(HTTPException) as exc:
        _call(tracker, make_writer)

    assert exc.value.status_code == 409
    make_writer.assert_not_called()
    mp.get_context.assert_not_called()


def test_preinit_failure_releases_slot_and_503() -> None:
    tracker = MagicMock()
    tracker.claim_active_job.return_value = True
    make_writer = MagicMock()
    make_writer.return_value.set_status.side_effect = RuntimeError("redis down")

    with pytest.raises(HTTPException) as exc:
        _call(tracker, make_writer)

    assert exc.value.status_code == 503
    tracker.mark_completed.assert_called_once_with("job-1")


def test_spawn_oserror_releases_slot_and_500() -> None:
    tracker = MagicMock()
    tracker.claim_active_job.return_value = True
    make_writer = MagicMock()

    with patch(f"{_JC}.reap_finished_children"), patch(f"{_JC}.multiprocessing") as mp:
        proc = MagicMock()
        proc.start.side_effect = OSError("no fork")
        mp.get_context.return_value.Process.return_value = proc
        with pytest.raises(HTTPException) as exc:
            _call(tracker, make_writer)

    assert exc.value.status_code == 500
    tracker.mark_completed.assert_called_once_with("job-1")


@pytest.mark.parametrize(
    "pid",
    [
        pytest.param(4321, id="pid_registered_and_returned"),
        pytest.param(None, id="no_pid_skips_registration"),
    ],
)
def test_success_returns_pid_and_registers_only_when_present(pid: int | None) -> None:
    tracker = MagicMock()
    tracker.claim_active_job.return_value = True
    make_writer = MagicMock()

    with patch(f"{_JC}.reap_finished_children"), patch(f"{_JC}.multiprocessing") as mp:
        proc = MagicMock()
        proc.pid = pid
        mp.get_context.return_value.Process.return_value = proc
        result = _call(tracker, make_writer)

    assert result == pid
    if pid is None:
        tracker.register_job.assert_not_called()
    else:
        tracker.register_job.assert_called_once_with("job-1", pid, job_type=JobType.OPTIMIZATION)


def test_read_job_status_builds_response_from_meta_and_state() -> None:
    tracker = MagicMock()
    tracker.get_job.return_value = JobMeta(
        job_id="job-1",
        pid=1,
        status=JobStatus.RUNNING,
        created_at=1700000000.0,
        job_type=JobType.OPTIMIZATION,
    )
    reader = MagicMock()
    reader.get_status.return_value = JobStatus.RUNNING
    reader.get_logs.return_value = "log line"
    reader.get_error_message.return_value = None
    with patch(f"{_JS}.create_job_state", return_value=reader) as create_state:
        resp = js.read_job_status(tracker, "job-1")

    # Meta-derived fields exercise the coercion + ISO formatting (the real logic).
    assert resp.job_id == "job-1"
    assert resp.job_type is JobType.OPTIMIZATION
    assert resp.created_at == "2023-11-14T22:13:20+00:00"  # 1700000000.0 as UTC ISO
    # The live reader is built from the meta-derived job_type (arg delegation that
    # fails if the coerced type is not the one used to construct the state reader).
    create_state.assert_called_once_with(JobType.OPTIMIZATION, "job-1")
    # Reader values are surfaced onto the response, using distinguishable values so a
    # hardcoded status or a status/logs field swap fails.
    assert resp.status is JobStatus.RUNNING
    assert resp.logs == "log line"


def test_read_job_status_404_when_meta_missing() -> None:
    tracker = MagicMock()
    tracker.get_job.return_value = None
    with pytest.raises(HTTPException) as exc:
        js.read_job_status(tracker, "missing")
    assert exc.value.status_code == 404
    assert exc.value.detail == "Job missing not found"


def test_load_job_meta_500_on_corrupt_meta() -> None:
    """The tracker's ValidationError is mapped to a 500 at this single read seam."""
    tracker = MagicMock()
    tracker.get_job.side_effect = _corrupt_meta_error()
    with pytest.raises(HTTPException) as exc:
        js.load_job_meta(tracker, "job-1")
    assert exc.value.status_code == 500
    assert "Corrupt job metadata for job job-1" in str(exc.value.detail)


@pytest.mark.usefixtures("tracker")
class TestJobLifecycleSlotIntegration:
    """The single active slot over real Redis: a worker never takes it back."""

    def test_a_worker_that_lost_the_slot_does_not_take_it_from_the_new_holder(self):
        str_client = get_redis_client(binary=False)
        tracker = JobTracker()
        old = _launched("slot-old")
        tracker.mark_completed(old)  # released before the old worker got going
        assert tracker.claim_active_job("slot-new", 0, job_type=JobType.OPTIMIZATION)

        with pytest.raises(JobSlotLostError):
            JobLifecycle(old).__enter__()

        assert str_client.get(ACTIVE_JOB_KEY) == "slot-new"

    def test_a_pause_landed_during_the_spawn_survives_the_worker_starting(self):
        job_id = _launched("slot-paused")
        writer = create_job_state(JobType.OPTIMIZATION, job_id)
        assert writer.set_status_cas(JobStatus.RUNNING, JobStatus.PAUSED)

        lc = JobLifecycle(job_id)
        lc.__enter__()
        try:
            assert writer.get_status() == JobStatus.PAUSED
        finally:
            lc.__exit__(None, None, None)


# --- Service shutdown: job_control.stop_launched_jobs -----------------------------

_SPAWN = multiprocessing.get_context("spawn")


class _FakeProcess:
    """A worker process that exits at a chosen point of the shutdown.

    ``exits_on``: "join" (within the grace), "terminate", "kill", or "never".
    """

    def __init__(self, exits_on: str, exitcode: int = 0) -> None:
        self._exits_on = exits_on
        self._alive = True
        self._exitcode = exitcode
        self.pid = 4321
        self.calls: list[str] = []

    def is_alive(self) -> bool:
        return self._alive

    @property
    def exitcode(self) -> int | None:
        return None if self._alive else self._exitcode

    def _event(self, name: str) -> None:
        self.calls.append(name)
        if name == self._exits_on:
            self._alive = False

    def join(self, timeout: float | None = None) -> None:
        del timeout
        if self.calls and self.calls[-1] in ("terminate", "kill"):
            return
        self._event("join")

    def terminate(self) -> None:
        self._event("terminate")

    def kill(self) -> None:
        self._event("kill")


def _writer(*statuses: JobStatus) -> MagicMock:
    """A state writer whose successive ``get_status`` reads return ``statuses``."""
    writer = MagicMock()
    writer.get_status.side_effect = list(statuses)
    writer.set_status_cas.return_value = True
    return writer


@pytest.fixture
def shutdown_seams() -> Generator[dict[str, MagicMock], None, None]:
    """Patch the cancel path's collaborators where job_control looks them up."""
    with (
        patch(f"{_JC}.cancel_with_reread") as cancel,
        patch(f"{_JC}.create_job_state") as make_writer,
        patch(f"{_JC}.get_cancellation_token") as get_token,
    ):
        yield {"cancel": cancel, "make_writer": make_writer, "get_token": get_token}


def _record(job_id: str, process: object) -> None:
    job_control._launched[job_id] = (JobType.OPTIMIZATION, process)  # type: ignore[assignment]


class TestStopLaunchedJobs:
    """Mocked collaborators: what the shutdown calls, and in what order."""

    def test_running_job_is_cancelled_through_the_cancel_path(self, shutdown_seams):
        """The shutdown cancels like POST /jobs/{id}/cancel: cancel_with_reread with the
        job's own writer and token. A worker that then stops on its own recorded its
        terminal status itself, so it is left as is and never terminated."""
        tracker = MagicMock()
        process = _FakeProcess(exits_on="join")
        _record("j1", process)
        cancel_writer, end_writer = _writer(), _writer(JobStatus.CANCELLED)
        shutdown_seams["make_writer"].side_effect = [cancel_writer, end_writer]

        stop_launched_jobs(tracker, grace_s=5)

        shutdown_seams["cancel"].assert_called_once_with(
            "j1", cancel_writer, shutdown_seams["get_token"].return_value, tracker
        )
        shutdown_seams["get_token"].assert_called_once_with("j1")
        assert process.calls == ["join"]
        end_writer.set_status_cas.assert_not_called()
        end_writer.set_error_message.assert_not_called()
        tracker.mark_completed.assert_called_once_with("j1")
        assert "j1" not in job_control._launched

    @pytest.mark.parametrize("exits_on", ["terminate", "kill"])
    def test_worker_past_the_grace_is_ended_and_the_job_cancelled(
        self, shutdown_seams, exits_on: str
    ):
        """A worker still running after the grace is terminated, then killed if that
        does not end it; the job is driven CANCELLED with the shutdown reason and its
        slot released at once (mark_completed), not when the heartbeat TTL lapses."""
        tracker = MagicMock()
        process = _FakeProcess(exits_on=exits_on)
        _record("j1", process)
        end_writer = _writer(JobStatus.CANCELLING)
        shutdown_seams["make_writer"].side_effect = [_writer(), end_writer]

        stop_launched_jobs(tracker, grace_s=0)

        expected = ["join", "terminate"] + (["kill"] if exits_on == "kill" else [])
        assert process.calls == expected
        end_writer.set_status_cas.assert_called_once_with(JobStatus.CANCELLING, JobStatus.CANCELLED)
        end_writer.set_error_message.assert_called_once_with(SHUTDOWN_KILLED_MESSAGE)
        tracker.mark_completed.assert_called_once_with("j1")

    def test_worker_that_exits_without_an_outcome_ends_the_job_as_error(self, shutdown_seams):
        tracker = MagicMock()
        _record("j1", _FakeProcess(exits_on="join", exitcode=-11))
        end_writer = _writer(JobStatus.CANCELLING)
        shutdown_seams["make_writer"].side_effect = [_writer(), end_writer]

        stop_launched_jobs(tracker, grace_s=5)

        end_writer.set_status_cas.assert_called_once_with(JobStatus.CANCELLING, JobStatus.ERROR)
        assert "code -11" in end_writer.set_error_message.call_args.args[0]
        tracker.mark_completed.assert_called_once_with("j1")

    def test_a_worker_that_survives_the_kill_keeps_its_slot(self, shutdown_seams):
        """Freeing the slot under a live process could let a second job drive the
        machine beside it, so that one is left to the heartbeat TTL."""
        tracker = MagicMock()
        _record("j1", _FakeProcess(exits_on="never"))
        shutdown_seams["make_writer"].side_effect = [_writer()]

        stop_launched_jobs(tracker, grace_s=0)

        tracker.mark_completed.assert_not_called()

    def test_a_failed_cancel_still_stops_the_worker(self, shutdown_seams):
        """Redis down at shutdown: the cancel raises, and the worker is still waited
        for and ended rather than left running past the service."""
        tracker = MagicMock()
        process = _FakeProcess(exits_on="terminate")
        _record("j1", process)
        shutdown_seams["cancel"].side_effect = RuntimeError("redis down")
        shutdown_seams["make_writer"].side_effect = [_writer(), _writer(JobStatus.RUNNING)]

        stop_launched_jobs(tracker, grace_s=0)

        assert process.calls == ["join", "terminate"]
        tracker.mark_completed.assert_called_once_with("j1")

    def test_only_running_workers_are_stopped(self, shutdown_seams):
        tracker = MagicMock()
        finished = _FakeProcess(exits_on="join")
        finished.join()
        _record("done", finished)

        stop_launched_jobs(tracker, grace_s=0)

        shutdown_seams["cancel"].assert_not_called()
        tracker.mark_completed.assert_not_called()

    def test_shutdown_writes_no_setpoint(self, shutdown_seams):
        """No automatic restore: a stopped run holds the machine where it is. The
        shutdown loads no environment, so it cannot write one."""
        tracker = MagicMock()
        _record("j1", _FakeProcess(exits_on="terminate"))
        shutdown_seams["make_writer"].side_effect = [_writer(), _writer(JobStatus.CANCELLING)]

        with (
            patch(
                "osprey_optimizer.services.plugin_registry.EnvironmentRegistry.load_environment"
            ) as load_env,
            patch(
                "osprey_optimizer.services.execution.ExecutionService._restore_on_completion"
            ) as restore,
        ):
            stop_launched_jobs(tracker, grace_s=0)

        load_env.assert_not_called()
        restore.assert_not_called()

    def test_launch_records_the_worker_and_drops_finished_ones(self):
        tracker = MagicMock()
        tracker.claim_active_job.return_value = True
        _record("old", _FakeProcess(exits_on="join"))
        job_control._launched["old"][1].join()

        with patch(f"{_JC}.reap_finished_children"), patch(f"{_JC}.multiprocessing") as mp:
            proc = mp.get_context.return_value.Process.return_value
            proc.pid = 4321
            _call(tracker, MagicMock())

        assert job_control._launched == {"job-1": (JobType.OPTIMIZATION, proc)}

    def test_grace_defaults_to_the_env_setting(self, shutdown_seams, monkeypatch):
        monkeypatch.setenv(SHUTDOWN_GRACE_ENV, "0")
        process = _FakeProcess(exits_on="terminate")
        _record("j1", process)
        shutdown_seams["make_writer"].side_effect = [_writer(), _writer(JobStatus.CANCELLING)]

        stop_launched_jobs(MagicMock())

        assert process.calls == ["join", "terminate"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, DEFAULT_SHUTDOWN_GRACE_SECONDS, id="unset"),
        pytest.param("", DEFAULT_SHUTDOWN_GRACE_SECONDS, id="empty"),
        pytest.param("2.5", 2.5, id="number"),
        pytest.param("0", 0.0, id="zero"),
        pytest.param("soon", DEFAULT_SHUTDOWN_GRACE_SECONDS, id="not_a_number"),
        pytest.param("-1", DEFAULT_SHUTDOWN_GRACE_SECONDS, id="negative"),
        pytest.param("inf", DEFAULT_SHUTDOWN_GRACE_SECONDS, id="unbounded"),
    ],
)
def test_shutdown_grace_from_env(monkeypatch, raw: str | None, expected: float) -> None:
    if raw is None:
        monkeypatch.delenv(SHUTDOWN_GRACE_ENV, raising=False)
    else:
        monkeypatch.setenv(SHUTDOWN_GRACE_ENV, raw)
    assert shutdown_grace_seconds() == expected


def test_default_grace_fits_dockers_stop_grace() -> None:
    """Grace + terminate + kill waits stay under Docker's default 10 s stop grace."""
    assert DEFAULT_SHUTDOWN_GRACE_SECONDS + TERMINATE_WAIT_SECONDS + KILL_WAIT_SECONDS < 10


def _spawn(target, *args, ready_timeout: float = 60.0):  # type: ignore[no-untyped-def]
    """Start a spawn-context daemon worker and wait for it to report ready."""
    ready = _SPAWN.Event()
    process = _SPAWN.Process(target=target, args=(*args, ready), daemon=True)
    process.start()
    if not ready.wait(ready_timeout):
        process.kill()
        pytest.fail("worker never became ready")
    return process


class TestWorkerSignalHandler:
    """The worker's SIGTERM/SIGINT handler, in real spawn children (no Redis)."""

    def test_sigterm_cancels_the_run_through_its_token(self):
        """A worker signalled directly stops through the cancel path: its token
        raises CancellationError, the run returns, and the process exits 0 instead
        of dying of the signal."""
        process = _spawn(_shutdown_workers.signalled_token_loop)
        process.terminate()
        process.join(10)
        assert process.exitcode == 0

    def test_a_worker_that_never_stops_is_ended_by_the_deadline(self, monkeypatch):
        """After the first signal the kernel ends a worker that does not stop within
        the grace (SIGALRM at its default action), so multiprocessing's exit hook,
        which joins terminated daemon children without a timeout, stays bounded."""
        monkeypatch.setenv(SHUTDOWN_GRACE_ENV, "1")
        process = _spawn(_shutdown_workers.signalled_but_stuck)
        started = time.monotonic()
        process.terminate()
        process.join(10)
        assert process.exitcode == -signal.SIGALRM
        assert time.monotonic() - started < 5

    def test_a_second_signal_ends_the_worker_at_once(self, monkeypatch):
        monkeypatch.setenv(SHUTDOWN_GRACE_ENV, "30")
        process = _spawn(_shutdown_workers.signalled_but_stuck)
        process.terminate()
        # The handler resets SIGTERM to its default before it returns; poll briefly
        # until the second one is the one that lands on the default action.
        deadline = time.monotonic() + 5
        while process.is_alive() and time.monotonic() < deadline:
            process.terminate()
            process.join(0.1)
        assert process.exitcode == -signal.SIGTERM

    def test_not_installed_outside_a_worker(self):
        """The API process (and a test running a job in-process) keeps its handlers."""
        before = signal.getsignal(signal.SIGTERM)
        install_worker_signal_handlers()
        assert signal.getsignal(signal.SIGTERM) is before

    def test_a_signalled_cancel_reads_as_cancelled(self, monkeypatch):
        """The handler's flag makes every token in the worker report a cancel,
        whatever Redis holds, so check_cancellation and wait_if_paused both stop."""
        monkeypatch.setattr(cancellation, "_cancel_signalled", False)
        redis = MagicMock()
        redis.hgetall.return_value = {"cancelled": "0", "paused": "1"}
        token = cancellation.CancellationToken(redis_client=redis, redis_key="k")
        assert not token.is_cancelled()

        cancellation.cancel_on_signal()

        assert token.is_cancelled()
        with pytest.raises(CancellationError):
            token.wait_if_paused()


@pytest.mark.usefixtures("tracker")
class TestStopLaunchedJobsIntegration:
    """Real Redis and real spawn workers: the shutdown end to end."""

    def _launch(self, target, job_id: str):  # type: ignore[no-untyped-def]
        tracker = JobTracker()
        ready = _SPAWN.Event()
        launch_job(
            tracker=tracker,
            job_id=job_id,
            job_type=JobType.OPTIMIZATION,
            make_writer=lambda jid: create_job_state(JobType.OPTIMIZATION, jid),
            spawn_target=target,
            spawn_args=(job_id, ready),
        )
        assert ready.wait(60), "worker never became ready"
        return tracker, job_control._launched[job_id][1]

    def test_a_cooperative_worker_records_its_own_cancel(self):
        tracker, process = self._launch(_shutdown_workers.cooperative_job, "shutdown-coop")

        stop_launched_jobs(tracker, grace_s=30)

        assert process.exitcode == 1  # the CancellationError propagates out of the run
        writer = create_job_state(JobType.OPTIMIZATION, "shutdown-coop")
        assert writer.get_status() == JobStatus.CANCELLED
        assert writer.get_error_message() == CANCELLED_MESSAGE
        assert tracker.get_active_job() is None

    def test_a_stuck_worker_is_killed_and_its_slot_freed(self):
        tracker, process = self._launch(_shutdown_workers.stubborn_job, "shutdown-stuck")
        started = time.monotonic()

        stop_launched_jobs(tracker, grace_s=0.5)

        elapsed = time.monotonic() - started
        assert process.exitcode == -signal.SIGKILL
        assert elapsed < 0.5 + TERMINATE_WAIT_SECONDS + KILL_WAIT_SECONDS + 2
        writer = create_job_state(JobType.OPTIMIZATION, "shutdown-stuck")
        assert writer.get_status() == JobStatus.CANCELLED
        assert writer.get_error_message() == SHUTDOWN_KILLED_MESSAGE
        str_client = get_redis_client(binary=False)
        assert str_client.get(ACTIVE_JOB_KEY) is None
        assert tracker.claim_active_job("next-job", 0, job_type=JobType.OPTIMIZATION)
