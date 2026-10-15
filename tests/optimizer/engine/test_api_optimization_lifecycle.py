"""Tests for api/optimization_lifecycle.py — create/start/update_config/read_config."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from osprey_optimizer.api import optimization_lifecycle as ol
from osprey_optimizer.api.optimization_lifecycle import read_optimization_config
from osprey_optimizer.models.api import JobResponse
from osprey_optimizer.models.config import ConstraintSpec, OptimizationConfig, Preset
from osprey_optimizer.models.enums import AlgorithmType, ConstraintType, JobStatus, JobType
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.models.types import Direction
from osprey_optimizer.utils.redis_state_store import CREATED_STATE_TTL_SECONDS

from ._injector_data import VARIABLES
from .conftest import make_disconnected_env

_OL = "osprey_optimizer.api.optimization_lifecycle"
# The config write + TTL normalization lives in job_control.store_created_config,
# which optimization_lifecycle calls; patch the Redis seam where it is used.
_JC = "osprey_optimizer.api.job_control"


def _config() -> OptimizationConfig:
    return OptimizationConfig(environment_name="sim", control_target="va", objective_name="obj")


def _registry_with_sim() -> MagicMock:
    registry = MagicMock()
    env = MagicMock()
    env.name = "sim"
    registry.list_environments.return_value = [env]
    # lookup_environment -> registry.load_environment(...).observables.get(obj):
    # the 'obj' observable maximizes, so direction materialization records MAXIMIZE.
    observable = MagicMock()
    observable.direction = Direction.MAXIMIZE
    resolved_env = registry.load_environment.return_value
    resolved_env.observables = {"obj": observable}
    # _resolve_draft_or_422 resolves the draft against this same env, so it needs
    # real variable metadata and a readable get_variables (the real injector VARIABLES; a
    # fixed, connected readback per variable).
    resolved_env.variables = VARIABLES
    resolved_env.get_variables.side_effect = lambda names: {n: PVReading(value=0.0) for n in names}
    return registry


def _registry_with_disconnected_env() -> MagicMock:
    registry = MagicMock()
    env = MagicMock()
    env.name = "sim"
    registry.list_environments.return_value = [env]
    registry.load_environment.return_value = make_disconnected_env()
    return registry


def _registry_no_environments() -> MagicMock:
    registry = MagicMock()
    registry.list_environments.return_value = []
    return registry


def test_create_persists_resolved_config_and_registers_created(seam_redis_mock: MagicMock) -> None:
    """Two-phase job contract (issue #2): create resolves the draft, records the
    objective direction from the observable (clients never set it), persists
    outcome constraints through the round-trip, and registers a CREATED job
    without claiming the active slot."""
    cfg = OptimizationConfig(
        environment_name="sim",
        control_target="va",
        objective_name="obj",
        constraints={"loss": ConstraintSpec(type=ConstraintType.LESS_THAN, value=0.25)},
    )
    tracker = MagicMock()
    with patch(f"{_OL}.get_optimization_writer") as writer_f:
        writer = MagicMock()
        writer_f.return_value = writer
        job_id = ol.create_optimization_job(cfg, tracker, _registry_with_sim())

    assert job_id
    writer.set_status.assert_called_once_with(JobStatus.CREATED)
    tracker.register_created_job.assert_called_once()
    tracker.expire_job.assert_called_once()
    tracker.claim_active_job.assert_not_called()

    persisted = OptimizationConfig.model_validate_json(seam_redis_mock.set.call_args[0][1])
    assert persisted.objective_direction is Direction.MAXIMIZE
    assert persisted.constraints["loss"].type is ConstraintType.LESS_THAN
    assert persisted.constraints["loss"].value == 0.25
    persisted.require_resolved()  # explicit variables + full min/max/step


_TWO = list(VARIABLES)[:2]


def _registry_with_preset() -> MagicMock:
    registry = _registry_with_sim()
    registry.get_presets.return_value = {
        "narrow": Preset(
            variables=_TWO,
            variable_overrides={name: {"window_abs": 0.5} for name in _TWO},
            algorithm=AlgorithmType.UPPER_CONFIDENCE_BOUND,
            n_iterations=7,
            objective_name="obj",
        )
    }
    return registry


def _persisted(seam_redis_mock: MagicMock) -> OptimizationConfig:
    return OptimizationConfig.model_validate_json(seam_redis_mock.set.call_args[0][1])


def test_create_fills_unset_fields_from_the_preset(seam_redis_mock: MagicMock) -> None:
    """A draft naming only an environment, target and preset gets the preset's
    variables, windows, algorithm, length and objective, resolved like any other."""
    cfg = OptimizationConfig(environment_name="sim", control_target="va", preset="narrow")
    with patch(f"{_OL}.get_optimization_writer"):
        ol.create_optimization_job(cfg, MagicMock(), _registry_with_preset())

    persisted = _persisted(seam_redis_mock)
    assert persisted.preset == "narrow"
    assert persisted.variables == _TWO
    assert persisted.algorithm is AlgorithmType.UPPER_CONFIDENCE_BOUND
    assert persisted.n_iterations == 7
    assert persisted.objective_name == "obj"
    overrides = persisted.variable_overrides or {}
    for name in _TWO:
        assert overrides[name].max - overrides[name].min <= 1.0 + 1e-9  # current ± 0.5


def test_what_the_draft_sets_beats_the_preset(seam_redis_mock: MagicMock) -> None:
    """Explicit fields win, even at their defaults; overrides merge per variable."""
    first, second = _TWO
    cfg = OptimizationConfig(
        environment_name="sim",
        control_target="va",
        preset="narrow",
        algorithm=AlgorithmType.RANDOM,
        n_iterations=20,
        variable_overrides={first: {"window_abs": 0.1}},
    )
    with patch(f"{_OL}.get_optimization_writer"):
        ol.create_optimization_job(cfg, MagicMock(), _registry_with_preset())

    persisted = _persisted(seam_redis_mock)
    assert persisted.algorithm is AlgorithmType.RANDOM
    assert persisted.n_iterations == 20
    overrides = persisted.variable_overrides or {}
    assert overrides[first].max - overrides[first].min <= 0.2 + 1e-9
    assert overrides[second].max - overrides[second].min > 0.2


def test_create_rejects_an_unknown_preset(seam_redis_mock: MagicMock) -> None:
    cfg = OptimizationConfig(environment_name="sim", control_target="va", preset="wide")
    with pytest.raises(HTTPException) as exc:
        ol.create_optimization_job(cfg, MagicMock(), _registry_with_preset())
    assert exc.value.status_code == 422
    assert "no preset 'wide'" in exc.value.detail
    seam_redis_mock.set.assert_not_called()


def test_create_preserves_explicit_objective_direction(seam_redis_mock: MagicMock) -> None:
    """An explicitly-set direction is never overridden by materialization."""
    cfg = OptimizationConfig(
        environment_name="sim",
        control_target="va",
        objective_name="obj",
        objective_direction=Direction.MINIMIZE,
    )
    with patch(f"{_OL}.get_optimization_writer"):
        ol.create_optimization_job(cfg, MagicMock(), _registry_with_sim())
    persisted = OptimizationConfig.model_validate_json(seam_redis_mock.set.call_args[0][1])
    assert persisted.objective_direction is Direction.MINIMIZE


def test_create_leaves_direction_unset_for_unknown_objective(seam_redis_mock: MagicMock) -> None:
    """Best-effort: an objective absent from the env leaves direction None for start-time validation."""
    registry = _registry_with_sim()
    registry.load_environment.return_value.observables = {}
    with patch(f"{_OL}.get_optimization_writer"):
        ol.create_optimization_job(_config(), MagicMock(), registry)
    persisted = OptimizationConfig.model_validate_json(seam_redis_mock.set.call_args[0][1])
    assert persisted.objective_direction is None


def test_create_survives_direction_lookup_failure(seam_redis_mock: MagicMock) -> None:
    """A failing direction lookup never aborts create; direction is left unset.

    Only the observables lookup fails here (not the environment load itself):
    draft resolution now also depends on a loadable env, so a fully-broken
    ``load_environment`` would legitimately 422 instead of exercising the
    best-effort direction path this test targets.
    """
    registry = _registry_with_sim()
    registry.load_environment.return_value.observables = MagicMock()
    registry.load_environment.return_value.observables.get.side_effect = RuntimeError("boom")
    with patch(f"{_OL}.get_optimization_writer"):
        job_id = ol.create_optimization_job(_config(), MagicMock(), registry)
    assert job_id
    persisted = OptimizationConfig.model_validate_json(seam_redis_mock.set.call_args[0][1])
    assert persisted.objective_direction is None


@pytest.mark.usefixtures("seam_redis_mock")
@pytest.mark.parametrize(
    ("config", "registry", "detail_token"),
    [
        pytest.param(
            _config(), _registry_no_environments(), "Unknown environment", id="unknown_environment"
        ),
        pytest.param(
            _config(),
            _registry_with_disconnected_env(),
            "Cannot resolve draft",
            id="unreadable_readback",
        ),
        pytest.param(
            OptimizationConfig(environment_name=None, objective_name="obj"),
            _registry_with_sim(),
            "environment_name is required",
            id="missing_environment_name",
        ),
        pytest.param(
            OptimizationConfig(environment_name="sim", control_target="va", objective_name=None),
            _registry_with_sim(),
            "objective_name is required",
            id="missing_objective_name",
        ),
    ],
)
def test_create_rejects_invalid_config(
    config: OptimizationConfig, registry: MagicMock, detail_token: str
) -> None:
    """Each row must be refused by ITS gate — the token pins the rejection reason
    so a row can't silently start failing at a different 422 gate."""
    with pytest.raises(HTTPException) as exc:
        ol.create_optimization_job(config, MagicMock(), registry)
    assert exc.value.status_code == 422
    assert detail_token in exc.value.detail


@pytest.mark.parametrize(
    "stored_status",
    [pytest.param(JobStatus.RUNNING, id="non_created"), pytest.param(None, id="unknown_job_id")],
)
def test_start_rejects_job_not_in_created_state(stored_status: JobStatus | None) -> None:
    """Start only accepts a CREATED draft: an already-running job and an unknown
    job id (no status at all) both 409."""
    with patch(f"{_OL}.get_optimization_reader") as reader_f:
        reader_f.return_value.get_status.return_value = stored_status
        with pytest.raises(HTTPException) as exc:
            ol.start_optimization_job("job-1", MagicMock())
    assert exc.value.status_code == 409


def test_start_launches_when_created(seam_redis_mock: MagicMock) -> None:
    seam_redis_mock.get.return_value = _config().model_dump_json()
    tracker = MagicMock()
    with (
        patch(f"{_OL}.get_optimization_reader") as reader_f,
        patch(f"{_OL}.initialize_results_dir", return_value=Path("results/sim/ts")),
        patch(f"{_OL}.launch_job", return_value=4321) as launch,
        patch(f"{_OL}.read_job_status") as read_f,
    ):
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        read_f.return_value = JobResponse(
            job_id="job-1", job_type=JobType.OPTIMIZATION, status=JobStatus.RUNNING
        )
        resp = ol.start_optimization_job("job-1", tracker)

    assert resp.status is JobStatus.RUNNING
    assert resp.job_id == "job-1"
    assert launch.call_args.kwargs["job_type"] is JobType.OPTIMIZATION
    read_f.assert_called_once_with(tracker, "job-1")


def test_start_registers_results_path_after_successful_launch(seam_redis_mock: MagicMock) -> None:
    seam_redis_mock.get.return_value = _config().model_dump_json()
    recorder = MagicMock()
    with (
        patch(f"{_OL}.get_optimization_reader") as reader_f,
        patch(f"{_OL}.get_optimization_writer") as writer_f,
        patch(f"{_OL}.initialize_results_dir", return_value=Path("results/sim/ts")),
        patch(f"{_OL}.launch_job") as launch_f,
        patch(f"{_OL}.read_job_status"),
    ):
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        # attach_mock resets return_value, so set it afterwards.
        recorder.attach_mock(launch_f, "launch")
        recorder.attach_mock(writer_f.return_value.set_last_results_path, "set_path")
        launch_f.return_value = 4321
        ol.start_optimization_job("job-1", MagicMock())

    # The path is registered only once the launch succeeded (never for a
    # 409'd draft), hence launch strictly before set_path.
    assert [name for name, _, _ in recorder.mock_calls] == ["launch", "set_path"]
    recorder.set_path.assert_called_once_with(Path("results/sim/ts").resolve())


def test_start_does_not_register_results_path_when_launch_fails(seam_redis_mock: MagicMock) -> None:
    seam_redis_mock.get.return_value = _config().model_dump_json()
    with (
        patch(f"{_OL}.get_optimization_reader") as reader_f,
        patch(f"{_OL}.get_optimization_writer") as writer_f,
        patch(f"{_OL}.initialize_results_dir", return_value=Path("results/sim/ts")),
        patch(f"{_OL}.launch_job", side_effect=HTTPException(status_code=409, detail="busy")),
    ):
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        with pytest.raises(HTTPException):
            ol.start_optimization_job("job-1", MagicMock())

    writer_f.return_value.set_last_results_path.assert_not_called()


def test_start_config_expired_returns_409(seam_redis_mock: MagicMock) -> None:
    seam_redis_mock.get.return_value = None  # config key expired
    with patch(f"{_OL}.get_optimization_reader") as reader_f:
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        with pytest.raises(HTTPException) as exc:
            ol.start_optimization_job("job-1", MagicMock())
    assert exc.value.status_code == 409
    assert "stored config not found" in exc.value.detail  # this 409, not the not-CREATED one


def test_start_results_dir_failure_returns_500(seam_redis_mock: MagicMock) -> None:
    seam_redis_mock.get.return_value = _config().model_dump_json()
    with (
        patch(f"{_OL}.get_optimization_reader") as reader_f,
        patch(f"{_OL}.initialize_results_dir", side_effect=OSError("disk full")),
    ):
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        with pytest.raises(HTTPException) as exc:
            ol.start_optimization_job("job-1", MagicMock())
    assert exc.value.status_code == 500


def _registry_with_only_sim_name() -> MagicMock:
    registry = MagicMock()
    registry.list_environments.return_value = [MagicMock(name="sim")]
    return registry


@pytest.mark.parametrize(
    ("job_status", "cfg", "registry", "expected_status", "detail_token"),
    [
        pytest.param(
            JobStatus.RUNNING,
            OptimizationConfig(environment_name="sim", control_target="va", objective_name="obj"),
            _registry_with_sim(),
            409,
            "expected status 'created'",
            id="non_created",
        ),
        pytest.param(
            JobStatus.CREATED,
            OptimizationConfig(environment_name=None, objective_name="obj"),
            _registry_with_sim(),
            422,
            "environment_name is required",
            id="missing_environment_name",
        ),
        pytest.param(
            JobStatus.CREATED,
            OptimizationConfig(environment_name="ghost", control_target="va", objective_name="obj"),
            _registry_with_only_sim_name(),
            422,
            "Unknown environment",
            id="unknown_environment",
        ),
    ],
)
def test_update_config_rejects(
    job_status: JobStatus,
    cfg: OptimizationConfig,
    registry: MagicMock,
    expected_status: int,
    detail_token: str,
) -> None:
    with patch(f"{_OL}.get_optimization_reader") as reader_f:
        reader_f.return_value.get_status.return_value = job_status
        with pytest.raises(HTTPException) as exc:
            ol.update_optimization_config("j1", cfg, MagicMock(), registry)
    assert exc.value.status_code == expected_status
    assert detail_token in exc.value.detail


def test_failed_update_keeps_prior_config(seam_redis_mock: MagicMock) -> None:
    """A failed update must leave the previously persisted config untouched (no silent fallback)."""
    with patch(f"{_OL}.get_optimization_writer"):
        job_id = ol.create_optimization_job(_config(), MagicMock(), _registry_with_sim())
    before = seam_redis_mock.set.call_args[0][1]

    with patch(f"{_OL}.get_optimization_reader") as reader_f:
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        with pytest.raises(HTTPException):
            ol.update_optimization_config(
                job_id, _config(), MagicMock(), _registry_with_disconnected_env()
            )

    # The failed resolve raises before client.set is reached, so create's write
    # is still the only one ever made — the prior config was never clobbered.
    seam_redis_mock.set.assert_called_once()
    assert seam_redis_mock.set.call_args[0][1] == before


def test_read_config_returns_stored_config() -> None:
    cfg = OptimizationConfig(environment_name="sim", control_target="va", objective_name="obj")
    with patch(f"{_JC}._redis_client.get_redis_client") as redis_f:
        redis = MagicMock()
        redis.get.return_value = cfg.model_dump_json()
        redis_f.return_value = redis
        result = read_optimization_config("j1")

    assert result.environment_name == "sim"
    assert result.objective_name == "obj"


def test_read_config_404_when_missing() -> None:
    with patch(f"{_JC}._redis_client.get_redis_client") as redis_f:
        redis = MagicMock()
        redis.get.return_value = None
        redis_f.return_value = redis
        with pytest.raises(HTTPException) as exc:
            read_optimization_config("j1")
    assert exc.value.status_code == 404


def test_update_config_writes_refreshes_ttl_and_materializes_direction() -> None:
    """The web pushes config via PUT->update; that path rewrites the config,
    refreshes the CREATED TTL, and records the objective direction too."""
    cfg = OptimizationConfig(environment_name="sim", control_target="va", objective_name="obj")
    tracker = MagicMock()
    with (
        patch(f"{_OL}.get_optimization_reader") as reader_f,
        patch(f"{_JC}._redis_client.get_redis_client") as redis_f,
        patch(f"{_OL}.read_job_status") as status_f,
    ):
        reader_f.return_value.get_status.return_value = JobStatus.CREATED
        redis = MagicMock()
        redis_f.return_value = redis
        status_f.return_value = JobResponse(
            job_id="j1", job_type=JobType.OPTIMIZATION, status=JobStatus.CREATED
        )
        result = ol.update_optimization_config("j1", cfg, tracker, _registry_with_sim())

    redis.set.assert_called_once()
    tracker.expire_job.assert_called_once_with("j1", CREATED_STATE_TTL_SECONDS)
    assert result.status == JobStatus.CREATED
    persisted = OptimizationConfig.model_validate_json(redis.set.call_args[0][1])
    assert persisted.objective_direction is Direction.MAXIMIZE
