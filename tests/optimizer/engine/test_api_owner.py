"""Who a run is for: the ``X-Osprey-Owner`` header, carried from ``start`` into the worker.

The connector decides each write by its owner's tier and switch. A run's owner
is read from the start request, handed to the worker it spawns, and stamped as
``OSPREY_CONTROL_OWNER`` there before any connector opens; the write verdict
itself is exercised against a real connector child in
``tests/optimizer/test_owner_write_verdict.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey_connectors.posture_store import CONTROL_OWNER_ENV_VAR
from osprey_optimizer.api import optimization_lifecycle as ol
from osprey_optimizer.api.owner import OWNER_HEADER, owner_from_header, stamp_owner
from osprey_optimizer.api.runner import run_optimization_process
from osprey_optimizer.models.api import JobResponse
from osprey_optimizer.models.enums import JobStatus, JobType
from tests.optimizer.engine.test_jobs_api import job_meta, make_gate_config

_JOBS = "osprey_optimizer.api.routers.jobs"

#: Header values and whether each names an owner. Every shape the framework's
#: reader refuses is here, so the parity test below holds the two to one rule.
_CASES = [
    pytest.param("alice", "alice", id="name"),
    pytest.param("a.b_c-9", "a.b_c-9", id="charset"),
    pytest.param("x" * 64, "x" * 64, id="at_the_bound"),
    pytest.param(None, None, id="absent"),
    pytest.param("", None, id="empty"),
    pytest.param("   ", None, id="blank"),
    pytest.param(" alice", None, id="untrimmed"),
    pytest.param("${OSPREY_TERMINAL_USER}", None, id="placeholder"),
    pytest.param("x" * 65, None, id="too_long"),
    pytest.param("..", None, id="parent_dir"),
    pytest.param(".", None, id="this_dir"),
    pytest.param("alice/bob", None, id="slash"),
    pytest.param("alice\\bob", None, id="backslash"),
    pytest.param("ali\0ce", None, id="nul"),
    pytest.param("alice bob", None, id="space"),
    pytest.param("élise", None, id="non_ascii"),
]


@pytest.mark.parametrize(("value", "owner"), _CASES)
def test_the_header_names_an_owner_only_in_the_roster_shape(
    value: str | None, owner: str | None
) -> None:
    assert owner_from_header(value) == owner


@pytest.mark.parametrize(("value", "_owner"), _CASES)
def test_the_header_is_read_as_the_framework_reads_it(value: str | None, _owner: object) -> None:
    """The optimizer cannot import the framework, so the rule is restated; this holds it equal."""
    from osprey.utils.owner_header import MAX_OWNER_LENGTH
    from osprey.utils.owner_header import OWNER_HEADER as FRAMEWORK_HEADER
    from osprey.utils.owner_header import owner_from_header as framework_reader
    from osprey_optimizer.api import owner as module

    assert (OWNER_HEADER, module.MAX_OWNER_LENGTH) == (FRAMEWORK_HEADER, MAX_OWNER_LENGTH)
    assert owner_from_header(value) == framework_reader(value)


def test_stamping_an_owner_sets_the_variable_the_connectors_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    stamp_owner("alice")
    assert os.environ[CONTROL_OWNER_ENV_VAR] == "alice"


def test_a_run_for_nobody_drops_an_inherited_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONTROL_OWNER_ENV_VAR, "whoever-spawned-the-worker")
    stamp_owner(None)
    assert CONTROL_OWNER_ENV_VAR not in os.environ


@pytest.mark.parametrize(("owner", "seen"), [("alice", "alice"), (None, None)])
def test_the_worker_stamps_its_owner_before_it_builds_the_run(
    monkeypatch: pytest.MonkeyPatch, owner: str | None, seen: str | None
) -> None:
    """The stamp precedes the execution service, which is what opens connectors."""
    monkeypatch.setenv(CONTROL_OWNER_ENV_VAR, "inherited")
    observed: list[str | None] = []

    def service(**_kwargs: object) -> MagicMock:
        observed.append(os.environ.get(CONTROL_OWNER_ENV_VAR))
        built = MagicMock()
        built.start_optimization.return_value = None
        return built

    with (
        patch("osprey_optimizer.api.runner.setup_logging"),
        patch("osprey_optimizer.api.runner.install_worker_signal_handlers"),
        patch("osprey_optimizer.api.runner.JobLifecycle"),
        patch("osprey_optimizer.api.runner.OptimizationConfig"),
        patch("osprey_optimizer.api.runner.ExecutionService", side_effect=service),
    ):
        run_optimization_process("job-1", {}, None, owner)
    assert observed == [seen]


def test_the_start_hands_its_owner_to_the_worker() -> None:
    with (
        patch(f"{ol.__name__}.get_optimization_reader") as reader,
        patch(f"{ol.__name__}.initialize_results_dir", return_value=Path("results/sim/ts")),
        patch(f"{ol.__name__}.launch_job", return_value=1) as launch,
        patch(f"{ol.__name__}.get_optimization_writer"),
        patch(f"{ol.__name__}.read_job_status"),
    ):
        reader.return_value.get_status.return_value = JobStatus.CREATED
        ol.start_optimization_job("job-1", MagicMock(), make_gate_config(), "{}", owner="alice")
    assert launch.call_args.kwargs["spawn_target"] is run_optimization_process
    assert launch.call_args.kwargs["spawn_args"][3] == "alice"


@pytest.mark.parametrize(
    ("headers", "owner"),
    [
        pytest.param({OWNER_HEADER: "alice"}, "alice", id="named"),
        pytest.param({}, None, id="absent"),
        pytest.param({OWNER_HEADER: "../alice"}, None, id="refused_shape"),
    ],
)
def test_the_start_route_reads_the_owner_from_the_header(
    client: TestClient,
    mock_job_tracker: MagicMock,
    headers: dict[str, str],
    owner: str | None,
) -> None:
    mock_job_tracker.get_job.return_value = job_meta()
    config = make_gate_config()
    with (
        patch(f"{_JOBS}.read_optimization_draft", return_value=(config, config.model_dump_json())),
        patch(f"{_JOBS}.load_environment"),
        patch(f"{_JOBS}.run_start_gates"),
        patch(
            f"{_JOBS}.start_optimization_job",
            return_value=JobResponse(
                job_id="test-job-id", job_type=JobType.OPTIMIZATION, status=JobStatus.RUNNING
            ),
        ) as start,
    ):
        resp = client.post(
            "/jobs/test-job-id/start",
            params={"config_revision": ol.draft_revision(config)},
            headers=headers,
        )
    assert resp.status_code == 200, resp.text
    assert start.call_args.kwargs["owner"] == owner
