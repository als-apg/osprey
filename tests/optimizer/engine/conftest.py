"""Pytest fixtures for test configuration."""

import json
import logging
import re
import socket
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from osprey_optimizer.events import EventBus, EventType, OptimizationEvent
from osprey_optimizer.models import process_scope
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.enums import AlgorithmType
from osprey_optimizer.models.protocols import EnvironmentProtocol, PVReading
from osprey_optimizer.models.types import Direction
from osprey_optimizer.optimization.evaluation import Evaluator as AppEvaluator
from osprey_optimizer.optimization.runner import ControlContext
from osprey_optimizer.optimization.xopt_backend import XoptOptimizer

from ._injector_data import VARIABLES

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from ._fakes import FakeInjectorEnvironment, MockEnvironment


# tuning_scripts ran these tests with its injector plugin installed. Here the same
# names are served by this suite's own fake, registered as entry points, so the
# tests that load "als_injector_sim" through the real registry still do.
_FAKE_PLUGINS = ("als_injector", "als_injector_sim")


@pytest.fixture(autouse=True)
def _fake_environment_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    from importlib.metadata import EntryPoint

    from osprey_optimizer.services import plugin_registry

    eps = [
        EntryPoint(
            name=name,
            value="tests.optimizer.engine._fakes:FakeInjectorEnvironment",
            group=plugin_registry.ENTRY_POINT_GROUP,
        )
        for name in _FAKE_PLUGINS
    ]
    monkeypatch.setattr(plugin_registry, "entry_points", lambda group: eps)
    monkeypatch.setenv("OPTIMIZER_ENVIRONMENTS_DIR", str(tmp_path / "no-environment-files"))
    plugin_registry.EnvironmentRegistry.reset()
    yield
    plugin_registry.EnvironmentRegistry.reset()


@pytest.fixture(autouse=True)
def restore_logging_state() -> Generator[None, None, None]:
    """Contain the global logging mutation the API lifespan performs.

    ``lifespan`` calls ``setup_logging``, i.e. ``basicConfig(force=True)``,
    which replaces the root handlers — pytest's own included — and raises
    levels on named loggers. Any test entering ``TestClient(app)`` as a context
    manager triggers it, so without this the mutation escapes into every test
    that runs afterwards.
    """
    root = logging.getLogger()
    saved_root_level = root.level
    saved_handlers = root.handlers[:]
    saved_levels = {
        name: existing.level
        for name, existing in logging.root.manager.loggerDict.items()
        if isinstance(existing, logging.Logger)
    }
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_root_level)
    # Loggers a test created itself are absent from the snapshot; NOTSET is the
    # level they would have had, so reset rather than skip them.
    for name, existing in logging.root.manager.loggerDict.items():
        if isinstance(existing, logging.Logger):
            existing.setLevel(saved_levels.get(name, logging.NOTSET))


@pytest.fixture(autouse=True)
def restore_process_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contain the multi-target opt-out the API lifespan sets (``allow_multiple_targets``).

    It is process-global, so without this a ``TestClient(app)`` here would make
    a later test on the same xdist worker hand its CA work to child processes.
    """
    monkeypatch.setattr(
        process_scope, "_multiple_targets_allowed", process_scope.multiple_targets_allowed()
    )


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    """Contain FastAPI dependency overrides, which live on the module-level ``app``.

    Fixtures that install overrides used to clear them after their own ``yield``.
    When a test body fails, pytest throws the exception in at that yield, so the
    clear never ran and the overrides leaked into every later test in the worker.
    Autouse teardown runs regardless, so overrides cannot outlive a test.
    """
    yield
    from osprey_optimizer.api.main import app

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_api_redis_client() -> Generator[None, None, None]:
    """Mock the Redis client used by the API lifespan so TestClient(app) does not
    require a running Redis server. Tests that need real Redis should use the
    ``redis_server`` fixture and unpatch as needed.
    """
    with patch("osprey_optimizer.api.main.ping_redis"):
        yield


@pytest.fixture
def seam_redis_mock() -> Generator[MagicMock, None, None]:
    """Patch the Redis client factory; yield the mock client.

    Tests requesting this fixture get a ready-to-configure MagicMock in place
    of the real Redis client that the per-domain state factories would
    otherwise obtain. Call ``seam_redis_mock.get.return_value = ...`` /
    ``seam_redis_mock.set.side_effect = ...`` as needed inside each test.
    """
    mock_client = MagicMock()
    # JobTracker.expire_job() iterates scan_iter() on the terminal sweep; default
    # to an empty sweep so that cleanup is a no-op unless a test populates it.
    mock_client.scan_iter.return_value = []
    with patch(
        "osprey_optimizer.utils.redis_client.get_redis_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch get_redis_client to return a fresh MagicMock per call.

    Unlike ``seam_redis_mock`` (which yields one configurable mock), this
    installs a factory returning a default-empty client (.get->None,
    .hgetall->{}). Use when a test just needs the seam satisfied.
    """

    def _factory(**kw: object) -> MagicMock:
        mock = MagicMock()
        mock.get.return_value = None
        mock.hgetall.return_value = {}
        return mock

    monkeypatch.setattr(
        "osprey_optimizer.utils.redis_client.get_redis_client",
        _factory,
    )


@pytest.fixture
def mock_job_tracker() -> MagicMock:
    from osprey_optimizer.api.job_tracker import JobMeta, JobTracker
    from osprey_optimizer.models.enums import JobStatus, JobType

    tracker = MagicMock(spec=JobTracker)
    tracker.get_active_job.return_value = None
    tracker.job_exists.return_value = True
    tracker.get_job.return_value = JobMeta(
        job_id="test-job-id",
        pid=1234,
        status=JobStatus.RUNNING,
        created_at=1712000000.0,
        job_type=JobType.OPTIMIZATION,
    )
    return tracker


@pytest.fixture
def mock_env_registry() -> MagicMock:
    registry = MagicMock()
    registry.list_environments.return_value = []
    return registry


@pytest.fixture
def redis_mocks() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Patch ``get_redis_client`` so str/binary variants return distinct mocks.

    Yields ``(str_client, bin_client)`` — both installed behind the single
    ``utils.redis_client.get_redis_client`` factory that all consumers now use.
    """
    str_client = MagicMock()
    bin_client = MagicMock()
    # JobTracker.expire_job() iterates scan_iter() on the terminal sweep (str
    # client); default both clients to an empty sweep unless a test populates it.
    str_client.scan_iter.return_value = []
    bin_client.scan_iter.return_value = []

    def _factory(binary: bool = False) -> MagicMock:
        return bin_client if binary else str_client

    with patch(
        "osprey_optimizer.utils.redis_client.get_redis_client",
        side_effect=_factory,
    ):
        yield str_client, bin_client


@pytest.fixture(autouse=True, scope="session")
def _disable_xopt_lhs_random_cd() -> Generator[None, None, None]:
    """Skip xopt's `random-cd` LHS post-optimization in tests (~2.6s/phase).

    xopt's `LatinHypercubeGenerator` overrides scipy's default
    (`optimization=None`) with `"random-cd"`, which runs 10000 perturbation
    iterations of scipy's centered-discrepancy optimizer on the first
    `step()` call. The optimization runner / bringup adapter uses LHS only
    as a BO seed or for exploration with fake envs; the marginal gain in
    space-filling is not worth the constant-time cost in tests. Tests that
    explicitly want `"random-cd"` can still pass it as a kwarg.
    """
    from xopt.generators.scipy.latin_hypercube import LatinHypercubeGenerator

    original_init = LatinHypercubeGenerator.__init__

    def _init(self: LatinHypercubeGenerator, **kwargs: object) -> None:
        kwargs.setdefault("optimization", None)
        original_init(self, **kwargs)

    LatinHypercubeGenerator.__init__ = _init
    try:
        yield
    finally:
        LatinHypercubeGenerator.__init__ = original_init


def _collect_torch_garbage() -> None:
    """Force a GC pass and drop any CUDA cache, ignoring a missing torch."""
    try:
        import gc

        import torch

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# Collect PyTorch garbage periodically rather than after every test. A full
# ``gc.collect()`` with torch resident walks a large object graph (~0.14s) and,
# run after all ~1300 tests, dominated the suite runtime. Bounding peak memory
# every N tests plus a guaranteed final sweep (below) preserves the protection
# against std::terminate() at interpreter shutdown without the per-test cost.
_COLLECT_EVERY = 50


@pytest.fixture(autouse=True)
def cleanup_pytorch(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Periodically reclaim PyTorch state to prevent resource conflicts.

    BO generator tests create PyTorch tensors/models whose finalizers, if left
    until interpreter shutdown, can trigger std::terminate(). We bound their
    accumulation with a periodic collect here and a final collect in the
    session-scoped ``_final_torch_cleanup`` fixture.
    """
    yield

    counter = request.session
    n = getattr(counter, "_torch_gc_counter", 0) + 1
    counter._torch_gc_counter = n  # type: ignore[attr-defined]
    if n % _COLLECT_EVERY == 0:
        _collect_torch_garbage()


@pytest.fixture(autouse=True, scope="session")
def _final_torch_cleanup() -> Generator[None, None, None]:
    """Guarantee one PyTorch GC pass at session end (per xdist worker).

    This is the actual safeguard against torch finalizers running in a bad
    order during interpreter shutdown; the per-test fixture only bounds peak
    memory between sweeps.
    """
    yield
    _collect_torch_garbage()


@pytest.fixture
def test_variable_bounds() -> dict[str, list[float]]:
    """Variable bounds from this package's own constants, as [min, max] lists.

    Feed it to the ``osprey_optimizer.models.testing`` builders
    (``valid_variables`` / ``bounds_variables`` / ``relative_increment``).
    """
    return {name: [meta.min_value, meta.max_value] for name, meta in VARIABLES.items()}


@pytest.fixture
def fake_env() -> "FakeInjectorEnvironment":
    """Shared FakeInjectorEnvironment for tests that just need a fully wired fake
    env (draft resolution, start gates, orchestration) — not asserting on injector
    physics."""
    from ._fakes import FakeInjectorEnvironment

    return FakeInjectorEnvironment()


@pytest.fixture
def mock_plugin_env(monkeypatch: pytest.MonkeyPatch) -> "FakeInjectorEnvironment":
    """Patch EnvironmentRegistry.load_environment to return a fake env.

    Shared across execution/orchestration/e2e tests that use the env only as a
    generic collaborator (bounds/variables/observables), never asserting on injector
    physics, so the fake satisfies them.
    """
    from ._fakes import FakeInjectorEnvironment

    fake_env = FakeInjectorEnvironment()
    monkeypatch.setattr(
        "osprey_optimizer.services.plugin_registry.EnvironmentRegistry.load_environment",
        lambda self, name, config: fake_env,
    )
    return fake_env


@pytest.fixture
def mock_env() -> "MockEnvironment":
    """Lightweight controllable endpoint-test double (var1/obs1 fixed data).

    Distinct from ``fake_env`` (the high-fidelity injector sim). Used by the
    environment/results/run-config API tests to assert exact numbers and drive
    every ``read_pvs`` PVReading shape.
    """
    from ._fakes import MockEnvironment

    return MockEnvironment()


@pytest.fixture
def mock_registry(mock_env: "MockEnvironment") -> MagicMock:
    """EnvironmentRegistry mock whose load_environment returns ``mock_env``."""
    from osprey_optimizer.models.api import EnvironmentInfo
    from osprey_optimizer.services.plugin_registry import EnvironmentRegistry

    registry = MagicMock(spec=EnvironmentRegistry)
    registry.load_environment.side_effect = lambda name, config: mock_env
    registry.list_environments.return_value = [
        EnvironmentInfo(name="sim", display_name="Simulator"),
    ]
    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> "Generator[TestClient, None, None]":
    """TestClient with the environment-registry override installed.

    This ``client`` installs ``get_environment_registry`` (and a permissive job
    tracker) overrides, for the environment/results/run-config API tests. Test
    modules that need a different override (e.g. test_jobs_api, test_script_api,
    test_archiver_endpoint) define their own local ``client`` fixture, which
    wins by pytest's local-over-conftest resolution.
    """
    from fastapi import Query
    from fastapi.testclient import TestClient

    from osprey_optimizer.api.dependencies import (
        Machine,
        get_environment_registry,
        get_job_tracker,
        require_machine,
    )
    from osprey_optimizer.api.job_tracker import JobTracker
    from osprey_optimizer.api.main import app
    from osprey_optimizer.models.config import ControlTarget, ExecutionMode

    tracker = MagicMock(spec=JobTracker)
    tracker.get_active_job.return_value = None

    # Production has no default target (test_machine_target.py pins that); these
    # tests predate targets, so a request that names none reaches the simulator.
    def machine_defaulting_to_va(
        target: ControlTarget = Query(default="va"),
        execution_mode: ExecutionMode | None = Query(default=None),
    ) -> Machine:
        return Machine(target=target, execution_mode=execution_mode)

    app.dependency_overrides[get_environment_registry] = lambda: mock_registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    app.dependency_overrides[require_machine] = machine_defaulting_to_va
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_initial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit ``ExecutionService._measure_initial_state`` to an empty baseline.

    Use in orchestration tests that mock ``OptimizationEngine.run`` and do not
    care about environment loading or baseline capture. Without this, such
    tests would need a real registered environment plugin just to satisfy the
    initial-state call.
    """
    from osprey_optimizer.services.execution import ExecutionService, InitialStateResult

    monkeypatch.setattr(
        ExecutionService,
        "_measure_initial_state",
        lambda self, config: InitialStateResult(initial_point={}, env_metadata={}, start_values={}),
    )


@pytest.fixture
def patched_engine_collaborators() -> Generator[MagicMock]:
    """Patch the three engine collaborators and yield the ``run_optimization_phase`` mock.

    Prevents the silent risk that a test forgets to patch one of the trio.
    """
    with (
        patch("osprey_optimizer.services.orchestration.run_optimization_phase") as mock_run,
        patch("osprey_optimizer.services.orchestration.report_results"),
        patch("osprey_optimizer.services.orchestration.Evaluator"),
    ):
        yield mock_run


@pytest.fixture(scope="session")
def redis_server() -> Generator[str, None, None]:
    """A real Redis for the tests that need its Lua scripts and atomicity.

    ``$OPTIMIZER_TEST_REDIS_URL`` names a server to use (each xdist worker
    takes its own database number on it, flushed by the tests), e.g. ``docker run -d -p 6390:6379 redis:7.4-alpine`` with
    ``redis://localhost:6390/0``; otherwise a ``redis-server`` on ``PATH`` is
    started on a free port. With neither, the test is skipped.
    """
    import os
    import shutil
    import subprocess

    redis_url = os.environ.get("OPTIMIZER_TEST_REDIS_URL")
    process = None
    if redis_url:
        # One shared server: give each xdist worker a database of its own, since
        # the tests flush theirs and contend for the global active-job key.
        worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
        base = re.sub(r"/\d+$", "", redis_url.rstrip("/"))
        redis_url = f"{base}/{int(worker.removeprefix('gw')) % 16}"
    else:
        binary = shutil.which("redis-server")
        if binary is None:
            pytest.skip("no Redis: set OPTIMIZER_TEST_REDIS_URL or put redis-server on PATH")
        # A free port per xdist worker so parallel workers don't collide.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen(
            [binary, "--port", str(port), "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        redis_url = f"redis://localhost:{port}/0"
        import redis as redis_lib

        deadline = time.monotonic() + 10
        while True:
            try:
                redis_lib.Redis.from_url(redis_url).ping()
                break
            except redis_lib.exceptions.ConnectionError:
                if time.monotonic() > deadline:
                    process.kill()
                    raise
                time.sleep(0.05)

    # Session-scoped, so the function-scoped ``monkeypatch`` fixture does not apply.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("REDIS_URL", redis_url)

        # Reset the cached redis clients to ensure they pick up the new URL
        from osprey_optimizer.utils import redis_client

        redis_client.reset_redis_clients()

        yield redis_url

    if process is not None:
        process.terminate()
        process.wait(timeout=10)


# Shared test helpers (non-fixture, importable by test modules)


def wait_until(
    condition: Callable[[], bool], *, timeout: float = 5.0, interval: float = 0.05
) -> bool:
    """Poll ``condition`` until it holds or ``timeout`` elapses.

    ``interval`` is the poll gap, not a fixed synchronization delay.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def make_mock_environment(
    *,
    variables: dict[str, Any] | None = None,
    objectives: dict[str, str] | None = None,
) -> MagicMock:
    """Return a fully-configured mock environment for engine tests.

    Parameters
    ----------
    variables:
        ``{name: {min_value, max_value, units}}`` mapping.  Defaults to a
        single ``"test_var"`` with bounds ``[0, 1]``.
    objectives:
        ``{name: direction}`` mapping.  Defaults to
        ``{"charge_ltb": "MAXIMIZE"}``.
    """
    if variables is None:
        var_meta = MagicMock()
        var_meta.min_value = 0.0
        var_meta.max_value = 1.0
        var_meta.units = "A"
        var_meta.step_size = 0.1
        variables = {"test_var": var_meta}

    if objectives is None:
        objectives = {"charge_ltb": "MAXIMIZE"}

    mock_env = MagicMock()
    mock_env.__enter__.return_value = mock_env
    mock_env.__exit__.return_value = None
    mock_env.variables = variables
    mock_env.get_observables.return_value = {name: PVReading(value=0.5) for name in objectives}
    mock_env.get_variables.return_value = {name: PVReading(value=0.1) for name in variables}
    mock_env.get_setpoints.return_value = {name: PVReading(value=0.1) for name in variables}

    obs_metadata: dict[str, MagicMock] = {}
    for name, direction in objectives.items():
        meta = MagicMock()
        meta.direction = direction
        obs_metadata[name] = meta

    mock_env.observables = obs_metadata
    mock_env.available_objectives = list(objectives.keys())
    return mock_env


def _make_phase(
    evaluate_side_effect: Any,
    *,
    direction: Direction,
    event_type: EventType,
) -> tuple[dict[str, Any], list[OptimizationEvent]]:
    """Build ``run_optimization_phase`` kwargs plus the captured-event list.

    Wires a ``MagicMock(spec=EnvironmentProtocol)`` whose ``evaluate`` is driven
    by *evaluate_side_effect* (an exception instance/class, or a callable taking
    the same args as ``evaluate``, used as ``mock_env.evaluate.side_effect``)
    through the real app ``Evaluator``, a single-variable :class:`XoptOptimizer`
    (``{"x": [0.0, 1.0]}``, objective ``"y"``, ``RANDOM``), a 2-iteration
    ``OptimizationConfig``, and an ``EventBus`` subscribed to *event_type*.
    """
    mock_env = MagicMock(spec=EnvironmentProtocol)
    mock_env.evaluate.side_effect = evaluate_side_effect

    evaluate = AppEvaluator(environment=mock_env, objective_name="y")
    bounds = {"x": [0.0, 1.0]}
    config = OptimizationConfig(
        algorithm=AlgorithmType.RANDOM,
        n_iterations=2,
        n_initial_samples=1,
        objective_name="y",
        objective_direction=direction,
    )
    optimizer = XoptOptimizer(
        variables=bounds,
        objective_name="y",
        direction=direction,
        config=config,
        current_env_values={"x": 0.5},
    )
    event_bus = EventBus()
    emitted: list[OptimizationEvent] = []
    event_bus.subscribe(event_type, emitted.append)

    phase_kwargs: dict[str, Any] = {
        "optimizer": optimizer,
        "evaluate": evaluate,
        "bounds": bounds,
        "objective_name": "y",
        "direction": direction,
        "config": config,
        "context": ControlContext(event_bus=event_bus),
        "current_env_values": {"x": 0.5},
        "n_initial": optimizer.resolve_n_initial(),
    }
    return phase_kwargs, emitted


def make_abort_phase(exc: Any) -> tuple[dict[str, Any], list[OptimizationEvent]]:
    """Runner "hard failure" scaffold for the abort-path tests.

    Returns ``(phase_kwargs, emitted)`` — splat ``phase_kwargs`` into
    ``run_optimization_phase``; ``emitted`` collects ``PHASE_FAILED`` events.
    """
    return _make_phase(exc, direction=Direction.MAXIMIZE, event_type=EventType.PHASE_FAILED)


def make_complete_phase(
    evaluate_side_effect: Any,
    *,
    direction: Direction = Direction.MAXIMIZE,
) -> tuple[dict[str, Any], list[OptimizationEvent]]:
    """Runner "successful completion" scaffold for best-point tests.

    Sibling of :func:`make_abort_phase` that subscribes ``PHASE_COMPLETED`` and
    honours *direction* (so a MINIMIZE run reports the finite minimum).
    Returns ``(phase_kwargs, completed)``.
    """
    return _make_phase(
        evaluate_side_effect, direction=direction, event_type=EventType.PHASE_COMPLETED
    )


def make_disconnected_env() -> "FakeInjectorEnvironment":
    """Return a fake env whose variable channels are always disconnected.

    Exercises the "active variable cannot be read" path (draft resolution and
    the create/update-config lifecycle): ``FakeInjectorEnvironment`` always
    returns a connected value, so this override is the only way to simulate an
    unreadable setpoint or readback.
    """
    from ._fakes import FakeInjectorEnvironment

    class _DisconnectedFakeEnv(FakeInjectorEnvironment):
        def get_variables(self, variable_names: list[str]) -> dict[str, PVReading]:
            return {n: PVReading(value=None, connected=False) for n in variable_names}

        def get_setpoints(self, variable_names: list[str]) -> dict[str, PVReading]:
            return self.get_variables(variable_names)

    return _DisconnectedFakeEnv()


def run_phase_with_optimizer(
    *,
    bounds: dict[str, list[float]],
    objective_name: str,
    config: OptimizationConfig,
    current_env_values: dict[str, float],
    env: Any,
    direction: Direction = Direction.MAXIMIZE,
) -> tuple[XoptOptimizer, Any]:
    """Construct an ``XoptOptimizer`` + run a phase over *env*, as orchestration does.

    ``static_columns`` is always threaded from ``optimizer.fixed_features``,
    which is ``None`` unless a ``model_file`` computed one — and the runner
    normalises that to ``{}`` — so this matches both the fixed-feature callers
    and the plain ones.
    """
    from osprey_optimizer.optimization.runner import run_optimization_phase

    optimizer = XoptOptimizer(
        variables=bounds,
        objective_name=objective_name,
        direction=direction,
        config=config,
        current_env_values=current_env_values,
    )
    result = run_optimization_phase(
        optimizer=optimizer,
        evaluate=AppEvaluator(env, objective_name),
        bounds=bounds,
        objective_name=objective_name,
        direction=direction,
        config=config,
        context=ControlContext(),
        current_env_values=current_env_values,
        n_initial=optimizer.resolve_n_initial(),
        static_columns=optimizer.fixed_features,
    )
    return optimizer, result


def make_blocking_import(module_name: str, message: str = "") -> Callable[..., Any]:
    """Return a ``builtins.__import__`` side-effect that blocks *module_name*.

    Usage::

        with patch("builtins.__import__", side_effect=make_blocking_import("redis")):
            ...
    """
    real_import = __import__

    def _blocking(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise ImportError(message or f"{module_name} is not installed")
        return real_import(name, *args, **kwargs)

    return _blocking


def make_run_dir(
    tmp_path: Path, *, with_initial: bool = True, env_name: str = "test_env", target: str = "va"
) -> str:
    """Write a fake ``results/<env>/<ts>/`` run dir with config + optional snapshot.

    Shared by the results-router restore tests (``test_results_api.py``).
    """
    run_dir = tmp_path / "results" / env_name / "2025-01-15_10-30-00"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment_name": env_name,
                "control_target": target,
                "objective_name": "charge_ltb",
                "variables": ["v1", "v2"],
            }
        )
    )
    if with_initial:
        (run_dir / "initial_point.json").write_text(
            json.dumps({"v1": 1.5, "v2": 3.0, "charge_ltb": 42.0})
        )
    return f"{env_name}/2025-01-15_10-30-00"


def recording_registry() -> tuple[MagicMock, dict[str, float]]:
    """Return an ``EnvironmentRegistry`` mock whose ``set_variables`` records writes.

    Shared by the results-router restore tests (``test_results_api.py``).
    """
    recorded: dict[str, float] = {}
    mock_env = MagicMock()
    mock_env.__enter__.return_value = mock_env
    mock_env.__exit__.return_value = None
    mock_env.set_variables.side_effect = lambda vals: recorded.update(vals)
    registry = MagicMock()
    registry.load_environment.return_value = mock_env
    return registry, recorded
