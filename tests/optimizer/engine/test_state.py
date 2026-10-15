"""Tests for optimization/state.py — _OptimizationState."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import redis.exceptions
from fastapi import HTTPException

from osprey_optimizer.api.job_control import cancel_with_reread
from osprey_optimizer.api.job_tracker import ACTIVE_JOB_KEY, JobTracker
from osprey_optimizer.models.enums import JobStatus, JobType
from osprey_optimizer.models.protocols import JobStateReader, JobStateWriter
from osprey_optimizer.models.types import Direction
from osprey_optimizer.optimization.state import (
    OptimizationStateReader,
    OptimizationStateWriter,
    _OptimizationState,
    get_optimization_reader,
    get_optimization_writer,
)
from osprey_optimizer.utils.cancellation import get_cancellation_token
from osprey_optimizer.utils.job_keys import state_prefix
from osprey_optimizer.utils.redis_client import get_redis_client
from osprey_optimizer.utils.redis_state_store import RUNNING_STATE_TTL_SECONDS
from osprey_optimizer.utils.serialization import (
    deserialize_dataframe,
    deserialize_obj,
    serialize_obj,
)

_TEST_JOB_ID = "test-job"
_EXPECTED_PREFIX = state_prefix(_TEST_JOB_ID)

_INTEG_JOB_ID = "integration-roundtrip"


def _flush_job_state(job_id: str) -> None:
    """Flush a job's state prefix so parallel/sequential real-Redis tests start clean."""
    client = get_redis_client(binary=True)
    for key in client.keys(f"{state_prefix(job_id)}:*"):
        client.delete(key)


@pytest.fixture
def writer(seam_redis_mock: MagicMock) -> _OptimizationState:
    del seam_redis_mock  # wanted for the Redis it patches, not its value
    w = get_optimization_writer(_TEST_JOB_ID)
    assert isinstance(w, _OptimizationState)  # narrow for mypy; no cast
    return w


@pytest.fixture
def reader(seam_redis_mock: MagicMock) -> _OptimizationState:
    del seam_redis_mock  # wanted for the Redis it patches, not its value
    r = get_optimization_reader(_TEST_JOB_ID)
    assert isinstance(r, _OptimizationState)
    return r


class TestOptimizationStateUnit:
    @pytest.mark.parametrize(
        ("write", "suffix", "expected"),
        [
            pytest.param(
                lambda w: w.set_status(JobStatus.RUNNING),
                "status",
                JobStatus.RUNNING,
                id="set_status",
            ),
            pytest.param(
                lambda w: w.set_last_results_path(Path("/var/data/results/foo")),
                "last_results_path",
                Path("/var/data/results/foo"),
                id="set_last_results_path",
            ),
            pytest.param(
                lambda w: w.save_logs("log line 1\nlog line 2"),
                "logs",
                "log line 1\nlog line 2",
                id="save_logs",
            ),
            pytest.param(
                lambda w: w.set_objective_direction("MAXIMIZE"),
                "objective_direction",
                "MAXIMIZE",
                id="set_direction",
            ),
            pytest.param(
                lambda w: w.set_objective_direction(None),
                "objective_direction",
                None,
                id="set_direction_none",
            ),
        ],
    )
    def test_single_key_writer_pickles_with_ttl(
        self,
        writer: _OptimizationState,
        seam_redis_mock: MagicMock,
        write: Any,
        suffix: str,
        expected: Any,
    ) -> None:
        write(writer)
        seam_redis_mock.set.assert_called_once()
        key, value = seam_redis_mock.set.call_args[0]
        assert key == f"{_EXPECTED_PREFIX}:{suffix}"
        assert deserialize_obj(value) == expected
        # Written with a TTL so a hard-killed worker's keys self-reclaim.
        assert seam_redis_mock.set.call_args.kwargs.get("ex") == RUNNING_STATE_TTL_SECONDS

    def test_set_data_writes_ipc_bytes(
        self, writer: _OptimizationState, seam_redis_mock: MagicMock
    ):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        writer.set_data(df)
        # set called once for data; data_version increment is via incr, not set
        data_call = seam_redis_mock.set.call_args_list[0]
        key, value = data_call[0]
        assert key == f"{_EXPECTED_PREFIX}:data"
        restored = deserialize_dataframe(value)
        assert restored.shape == df.shape
        # The hot data key is written with a TTL too, never left unbounded.
        assert data_call.kwargs.get("ex") == RUNNING_STATE_TTL_SECONDS

    def test_set_data_none_deletes_key(
        self, writer: _OptimizationState, seam_redis_mock: MagicMock
    ):
        writer.set_data(None)
        seam_redis_mock.delete.assert_any_call(f"{_EXPECTED_PREFIX}:data")

    @pytest.mark.parametrize(
        "write",
        [
            pytest.param(lambda w: w.set_status(JobStatus.RUNNING), id="set_status"),
            pytest.param(
                lambda w: w._save_dataframe_to_redis("data", pl.DataFrame({"x": [1.0]})),
                id="save_dataframe",
            ),
        ],
    )
    def test_write_raises_on_redis_error(
        self, writer: _OptimizationState, seam_redis_mock: MagicMock, write: Any
    ) -> None:
        seam_redis_mock.set.side_effect = redis.exceptions.ConnectionError("boom")
        with pytest.raises(redis.exceptions.ConnectionError, match="boom"):
            write(writer)

    def test_save_snapshot_preserves_sibling_fields(
        self, writer: _OptimizationState, seam_redis_mock: MagicMock
    ):
        # Per-field HSET is the mechanism that preserves siblings: each snapshot
        # is written as its own field, and our code never reads-modifies-writes
        # the whole hash (which would clobber a concurrent sibling field). This
        # pins that mechanism at the seam; real atomicity via TestStateRoundTrip.
        key = f"{_EXPECTED_PREFIX}:snapshots"

        writer.save_snapshot("initial", {"x": 1.0})
        writer.save_snapshot("final", {"x": 2.0})

        assert seam_redis_mock.hset.call_count == 2
        first, second = seam_redis_mock.hset.call_args_list
        assert first[0][0] == key
        assert first[0][1] == "initial"
        assert deserialize_obj(first[0][2]) == {"x": 1.0}
        assert second[0][0] == key
        assert second[0][1] == "final"
        assert deserialize_obj(second[0][2]) == {"x": 2.0}
        # Never a whole-hash rewrite or delete that would drop a sibling field.
        seam_redis_mock.hgetall.assert_not_called()
        seam_redis_mock.delete.assert_not_called()

    def test_increment_data_version_uses_incr(
        self, writer: _OptimizationState, seam_redis_mock: MagicMock
    ) -> None:
        writer._increment_data_version()
        seam_redis_mock.incr.assert_called_once_with(f"{_EXPECTED_PREFIX}:data_version")

    def test_set_objective_direction_rejects_invalid(
        self, writer: _OptimizationState, seam_redis_mock: MagicMock
    ):
        with pytest.raises(ValueError, match="Invalid objective direction 'SIDEWAYS'"):
            writer.set_objective_direction("SIDEWAYS")
        direction_calls = [
            c
            for c in seam_redis_mock.set.call_args_list
            if c[0][0].endswith(":objective_direction")
        ]
        assert len(direction_calls) == 0


class TestOptimizationStateReaderUnit:
    def test_get_status_raises_when_missing(
        self, reader: _OptimizationState, seam_redis_mock: MagicMock
    ):
        seam_redis_mock.get.return_value = None
        with pytest.raises(TypeError, match="Expected JobStatus"):
            reader.get_status()

    @pytest.mark.parametrize(
        "getter,stored_value",
        [
            ("get_status", JobStatus.RUNNING),
            ("get_logs", "hello"),
            ("get_last_results_path", Path("/var/data/results/foo")),
        ],
    )
    def test_getter_returns_stored_value(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        getter: str,
        stored_value: object,
    ):
        seam_redis_mock.get.return_value = serialize_obj(stored_value)
        assert getattr(reader, getter)() == stored_value

    @pytest.mark.parametrize(
        "getter,expected_default,key_suffix",
        [
            ("get_logs", "", "logs"),
            ("get_last_results_path", None, "last_results_path"),
            ("get_data_version", 0, "data_version"),
            ("get_error_message", None, "error_message"),
            ("get_objective_direction", None, "objective_direction"),
        ],
    )
    def test_getter_returns_default_when_missing(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        getter: str,
        expected_default: object,
        key_suffix: str,
    ):
        seam_redis_mock.get.return_value = None
        assert getattr(reader, getter)() == expected_default
        seam_redis_mock.get.assert_called_once_with(f"{_EXPECTED_PREFIX}:{key_suffix}")

    @pytest.mark.parametrize(
        ("raw", "expected_exc", "match", "expect_truncated"),
        [
            pytest.param(
                b"not a number",
                ValueError,
                r"Invalid data_version at .*",
                False,
                id="non_decimal_bytes",
            ),
            # Megabyte-scale garbage must not dump into the ValueError message —
            # the raise propagates into logs and HTTP 503 bodies.
            pytest.param(
                b"x" * 10_000,
                ValueError,
                r"Invalid data_version at .*",
                True,
                id="large_garbage_truncated",
            ),
            pytest.param(
                42, TypeError, r"Expected bytes at .*:data_version", False, id="non_bytes"
            ),
        ],
    )
    def test_get_data_version_raises_on_invalid_value(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        raw: object,
        expected_exc: type[BaseException],
        match: str,
        expect_truncated: bool,
    ) -> None:
        seam_redis_mock.get.return_value = raw
        with pytest.raises(expected_exc, match=match) as exc_info:
            reader.get_data_version()
        if expect_truncated:
            message = str(exc_info.value)
            assert "..." in message
            assert len(message) < 200

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            pytest.param({}, {}, id="missing_hash_returns_empty"),
            pytest.param(
                {b"initial": serialize_obj({"x": 1.0}), b"final": serialize_obj({"x": 2.0})},
                {"initial": {"x": 1.0}, "final": {"x": 2.0}},
                id="decodes_bytes_fields_and_values",
            ),
        ],
    )
    def test_get_available_snapshots(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        stored: dict[bytes, bytes],
        expected: dict[str, Any],
    ):
        seam_redis_mock.hgetall.return_value = stored
        assert reader.get_available_snapshots() == expected

    @pytest.mark.parametrize(
        ("hash_value", "match"),
        [
            pytest.param(
                {b"initial": serialize_obj(42)},
                r"Expected dict at .*:snapshots\[initial\]",
                id="non_dict_value",
            ),
            pytest.param(
                {123: b"value"},
                r"Expected bytes field at .*:snapshots",
                id="non_bytes_field",
            ),
            pytest.param(
                {b"initial": 42},
                r"Expected bytes at .*:snapshots\[initial\]",
                id="non_bytes_value",
            ),
        ],
    )
    def test_get_available_snapshots_raises_on_malformed_hash(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        hash_value: dict[object, object],
        match: str,
    ):
        seam_redis_mock.hgetall.return_value = hash_value
        with pytest.raises(TypeError, match=match):
            reader.get_available_snapshots()

    @pytest.mark.parametrize(
        "getter_name,redis_value,expected_match",
        [
            ("get_variable_names", "not a list", r"Expected list at .*:variable_names"),
            ("get_last_results_path", "a string", r"Expected Path at .*:last_results_path"),
            ("get_objective_name", 42, r"Expected str at .*:objective_name"),
            ("get_environment_name", 42, r"Expected str at .*:environment_name"),
            ("get_objective_direction", 42, r"Expected str at .*:objective_direction"),
            ("get_metadata", "not a dict", r"Expected dict at .*:metadata"),
            ("get_logs", 12345, r"Expected str at .*:logs"),
        ],
    )
    def test_getter_raises_on_wrong_type(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        getter_name: str,
        redis_value: object,
        expected_match: str,
    ):
        seam_redis_mock.get.return_value = serialize_obj(redis_value)
        with pytest.raises(TypeError, match=expected_match):
            getattr(reader, getter_name)()

    def test_get_objective_direction_raises_on_invalid_domain(
        self, reader: _OptimizationState, seam_redis_mock: MagicMock
    ):
        seam_redis_mock.get.return_value = serialize_obj("SIDEWAYS")
        with pytest.raises(
            ValueError, match=r"Invalid objective_direction at .*:objective_direction"
        ):
            reader.get_objective_direction()

    def test_get_objective_direction_returns_enum_member(
        self, reader: _OptimizationState, seam_redis_mock: MagicMock
    ):
        """The stored string is materialized as a real Direction member, not a
        str hiding behind the annotation (callers may rely on .value/.name)."""
        seam_redis_mock.get.return_value = serialize_obj("MINIMIZE")
        assert reader.get_objective_direction() is Direction.MINIMIZE

    @pytest.mark.parametrize(
        "getter,redis_value_or_exc,expected_exc_type,match",
        [
            (
                "get_status",
                redis.exceptions.ConnectionError("boom"),
                redis.exceptions.ConnectionError,
                "boom",
            ),
            (
                "get_data",
                redis.exceptions.ConnectionError("boom"),
                redis.exceptions.ConnectionError,
                "boom",
            ),
            ("get_status", "unexpected string", ValueError, "invalid type"),
            ("get_data", "unexpected string", ValueError, "invalid type"),
        ],
    )
    def test_load_raises_on_redis_error_or_invalid_value(
        self,
        reader: _OptimizationState,
        seam_redis_mock: MagicMock,
        getter: str,
        redis_value_or_exc: object,
        expected_exc_type: type[BaseException],
        match: str,
    ) -> None:
        if isinstance(redis_value_or_exc, BaseException):
            seam_redis_mock.get.side_effect = redis_value_or_exc
        else:
            seam_redis_mock.get.return_value = redis_value_or_exc
        with pytest.raises(expected_exc_type, match=match):
            getattr(reader, getter)()


@pytest.mark.usefixtures("redis_server")
class TestStateRoundTrip:
    """Integration tests requiring a real Redis server."""

    def _make_clients(self) -> tuple[_OptimizationState, _OptimizationState]:
        _flush_job_state(_INTEG_JOB_ID)
        w = get_optimization_writer(_INTEG_JOB_ID)
        r = get_optimization_reader(_INTEG_JOB_ID)
        assert isinstance(w, _OptimizationState)
        assert isinstance(r, _OptimizationState)
        return w, r

    def test_data_round_trip(self):
        writer, reader = self._make_clients()
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        writer.set_data(df)
        loaded = reader.get_data()
        assert loaded is not None
        assert loaded.equals(df)

    def test_snapshot_round_trip(self):
        writer, reader = self._make_clients()
        writer.save_snapshot("initial", {"x": 1.0})
        writer.save_snapshot("final", {"x": 2.0})
        assert reader.get_available_snapshots() == {"initial": {"x": 1.0}, "final": {"x": 2.0}}

    @pytest.mark.parametrize(
        "setter_name,getter_name,value",
        [
            ("set_status", "get_status", JobStatus.RUNNING),
            ("set_variable_names", "get_variable_names", ["var1", "var2", "var3"]),
            (
                "set_variable_bounds",
                "get_variable_bounds",
                {"var1": [0.0, 10.0], "var2": [-1.5, 1.5]},
            ),
            ("set_error_message", "get_error_message", "boom: something failed"),
            ("set_objective_name", "get_objective_name", "loss"),
            ("set_environment_name", "get_environment_name", "als_injector"),
            ("set_total_evaluations", "get_total_evaluations", 42),
            ("set_objective_direction", "get_objective_direction", "MAXIMIZE"),
            (
                "save_metadata",
                "get_metadata",
                {"variables": {"x": {"units": "mm"}}, "observables": {"y": {"units": "A"}}},
            ),
            ("save_logs", "get_logs", "test log output"),
        ],
    )
    def test_round_trip(self, setter_name: str, getter_name: str, value: object):
        writer, reader = self._make_clients()
        getattr(writer, setter_name)(value)
        assert getattr(reader, getter_name)() == value

    def test_data_version_increments(self):
        writer, reader = self._make_clients()
        df = pl.DataFrame({"x": [1.0]})
        writer.set_data(df)
        writer.set_data(df)
        writer.set_data(df)
        assert reader.get_data_version() == 3


_CAS_JOB_ID = "integration-cas"


@pytest.mark.usefixtures("redis_server")
class TestStatusCas:
    """Real-Redis coverage for the status compare-and-swap primitive.

    ``set_status_cas`` runs a Lua byte-CAS on the pickled status key. The
    ``seam_redis_mock`` MagicMock cannot model ``register_script``/EVAL, so these
    exercise the primitive against the embedded Redis, mirroring
    ``TestStateRoundTrip``'s writer construction.
    """

    @pytest.fixture(autouse=True)
    def _clean_active_slot(self):
        """Clear the global active-job pointer around every test in this class.

        ``cancel_with_reread`` claims the single active slot via the tracker; on
        the 409 (race-loser) path it raises before ``mark_completed`` runs, so the
        ``api:active_job`` pointer (which lives outside the job prefix ``_make_writer``
        flushes) would otherwise leak into the session-scoped Redis and make
        ``get_active_job`` self-heal to a phantom RUNNING job for later real-Redis
        tests. Clearing it before and after keeps this class fully isolated.
        """
        client = get_redis_client(binary=True)
        client.delete(ACTIVE_JOB_KEY)
        yield
        client.delete(ACTIVE_JOB_KEY)

    def _make_writer(self, job_id: str = _CAS_JOB_ID) -> _OptimizationState:
        _flush_job_state(job_id)
        w = get_optimization_writer(job_id)
        assert isinstance(w, _OptimizationState)
        return w

    def test_cas_swaps_when_stored_status_matches_expected(self):
        """CAS succeeds on a match: RUNNING is swapped to CANCELLED and sticks."""
        writer = self._make_writer()
        writer.set_status(JobStatus.RUNNING)

        assert writer.set_status_cas(JobStatus.RUNNING, JobStatus.CANCELLED) is True
        assert writer.get_status() == JobStatus.CANCELLED

    def test_cas_refuses_and_preserves_terminal_when_worker_won_the_race(self):
        """CAS refuses on a mismatch and must NOT clobber the worker's terminal
        status — the core lost-update guard.

        The API writer read RUNNING; the worker (a second, independent writer on
        the same key) drove the job COMPLETED in the gap. The CAS still expects
        RUNNING, so it must lose and leave COMPLETED intact.
        """
        api_writer = self._make_writer()
        api_writer.set_status(JobStatus.RUNNING)

        worker_writer = get_optimization_writer(_CAS_JOB_ID)
        assert isinstance(worker_writer, _OptimizationState)
        worker_writer.set_status(JobStatus.COMPLETED)

        assert api_writer.set_status_cas(JobStatus.RUNNING, JobStatus.CANCELLED) is False
        # Must not be resurrected/clobbered to CANCELLED.
        assert api_writer.get_status() == JobStatus.COMPLETED

    def test_cancel_with_reread_flips_running_job_to_cancelled(self):
        """End-to-end happy path over the REAL Lua CAS: a RUNNING job is
        cancelled (status → CANCELLED, token cancelled). Endpoint-level coverage
        (``test_jobs_api.py::TestJobsCancel::test_cancel_script``) mocks the CAS,
        so this pins the real writer/token/tracker wiring.
        """
        writer = self._make_writer()
        writer.set_status(JobStatus.RUNNING)
        token = get_cancellation_token(_CAS_JOB_ID)
        tracker = JobTracker()
        tracker.claim_active_job(_CAS_JOB_ID, 0, job_type=JobType.OPTIMIZATION)

        cancel_with_reread(_CAS_JOB_ID, writer, token, tracker)

        assert writer.get_status() == JobStatus.CANCELLED
        assert token.is_cancelled() is True

    def test_cancel_with_reread_409s_and_preserves_terminal_when_worker_completed(self):
        """End-to-end race loser over the REAL CAS path: the worker already drove
        the job COMPLETED, so ``cancel_with_reread`` 409s and leaves COMPLETED
        intact (not resurrected to CANCELLED). The 409 detail reports the real
        current status.
        """
        writer = self._make_writer()
        writer.set_status(JobStatus.COMPLETED)
        token = get_cancellation_token(_CAS_JOB_ID)
        tracker = JobTracker()
        tracker.claim_active_job(_CAS_JOB_ID, 0, job_type=JobType.OPTIMIZATION)

        with pytest.raises(HTTPException) as exc_info:
            cancel_with_reread(_CAS_JOB_ID, writer, token, tracker)

        assert exc_info.value.status_code == 409
        assert JobStatus.COMPLETED.value in exc_info.value.detail
        assert writer.get_status() == JobStatus.COMPLETED


# Every accessor — both readers and writers, for both domains — is wired the same way
# and satisfies the (runtime_checkable, superset) Writer protocol. The reader Protocol is
# narrower, so we verify readers against the Writer protocol too.
_ALL_FACTORIES = [
    pytest.param(
        get_optimization_writer,
        OptimizationStateWriter,
        _OptimizationState,
        id="optimization-writer",
    ),
    pytest.param(
        get_optimization_reader,
        OptimizationStateReader,
        _OptimizationState,
        id="optimization-reader",
    ),
]


class TestStateFactories:
    """Factory-wiring tests for the per-domain state factories."""

    @pytest.mark.parametrize("factory,protocol_cls,concrete_cls", _ALL_FACTORIES)
    def test_factory_wires_binary_redis_client_with_prefixed_key(
        self, factory, protocol_cls, concrete_cls
    ) -> None:
        mock_redis_client = MagicMock()
        with patch(
            "osprey_optimizer.utils.redis_client.get_redis_client",
            return_value=mock_redis_client,
        ) as mock_get_redis:
            accessor = factory("job-3")

        mock_get_redis.assert_called_once_with(binary=True)
        assert isinstance(accessor, protocol_cls)
        assert isinstance(accessor, concrete_cls)
        assert accessor._redis_client is mock_redis_client
        assert accessor._redis_key_prefix == state_prefix("job-3")


class TestJobStateProtocols:
    """Concrete state classes satisfy the shared job-state protocols.

    The protocols live in ``osprey_optimizer.models``, but the concrete
    ``_OptimizationState`` that implements them live in the
    app. Asserting conformance from ``models_tests`` would import the app from
    the base package — a reverse-direction (cyclic) test dependency — so the
    check belongs here.
    """

    @pytest.mark.parametrize("protocol", [JobStateReader, JobStateWriter], ids=["reader", "writer"])
    @pytest.mark.parametrize(
        "state_cls",
        [_OptimizationState],
        ids=["optimization_state"],
    )
    def test_state_satisfies_protocol(self, state_cls: type, protocol: type) -> None:
        assert isinstance(state_cls(), protocol)
