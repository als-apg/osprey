"""A run writes only under its one bounded write authorization, through a real connector child.

The deployment is the optimizer container's: a provisioned control-context
tree bound read-only, the run's owner stamped, a write ledger the run can
write, and a write tier that allows writes. The machine is the in-process
simulator served by a real connector-host child of the optimizer's own pool,
so every verdict below — the step spent, the range checked, the stop, the
lapsed lease — is the child's, judged against the files on disk. The run is
the engine's, through a file-declared :class:`ConnectorEnvironment`, with the
run's :class:`RunAuthorization` as the hook it calls before its first write.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors import control_context, posture_store
from osprey_connectors import write_authorization as wa
from osprey_connectors_tests._harness import write_payload
from osprey_optimizer.api.owner import stamp_owner
from osprey_optimizer.environments import interface as iface_module
from osprey_optimizer.environments.connector_env import ConnectorEnvironment
from osprey_optimizer.environments.interface import ConnectorInterface
from osprey_optimizer.environments.spec import EnvironmentFile
from osprey_optimizer.environments.write_authority import (
    WriteAuthority,
    install_process_authority,
)
from osprey_optimizer.events import EventBus, EventType
from osprey_optimizer.models.config import OptimizationConfig, VariableOverride
from osprey_optimizer.models.exceptions import (
    AUTHORIZATION_TIMEOUT,
    AUTHORIZATION_TOO_MANY_STEPS,
    AuthorizationEndedError,
    CancellationError,
    WriteRefusedError,
)
from osprey_optimizer.services import orchestration
from osprey_optimizer.services import write_authorization as run_auth
from osprey_optimizer.services.draft_resolution import resolve_draft_config
from osprey_optimizer.services.orchestration import (
    OptimizationEngine,
    RunnerServices,
    apply_best_point,
)
from osprey_optimizer.services.write_authorization import RunAuthorization
from tests.facility.served_tree import in_process_config, served_tree

pytestmark = [pytest.mark.write_authorization_gate, pytest.mark.timeout(180)]

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_PROCESS = "osprey_connectors.control_system._va_in_process_connector.VAInProcessConnector"
#: ``live`` resolves to the section's own type, the in-process simulator.
TARGET = "live"
A, B = "KNOB:A:SP", "KNOB:B:SP"
OWNER = "alice"
HOLDER = "optimizer"
#: Two initial samples and two iterations: four evaluations.
EVALUATIONS = 4

SPEC = EnvironmentFile.model_validate(
    {
        "name": "knobs",
        "variables": {name: {"min": -5, "max": 5, "step_size": 0.5} for name in (A, B)},
        "observables": {"knob_a": {"channel": A, "direction": "MAXIMIZE"}},
        "default_objective": "knob_a",
    }
)


def _record(tree: Path, state: str) -> None:
    """The owner's record with the switch at *state* for :data:`TARGET`."""
    payload = control_context.ControlContext(
        target=TARGET, generation=1, switch_epoch=1
    ).to_payload()
    payload["switch"] = {TARGET: state}
    write_payload(tree / OWNER / control_context.RECORD_FILENAME, payload)
    control_context.invalidate_cache()


class Deployment:
    """The optimizer container's write path, and the run driven over it."""

    def __init__(self, tmp_path: Path) -> None:
        self.tree = tmp_path / posture_store.STATE_DIR_NAME
        self.ledger = tmp_path / "write_ledger"
        self.states: list[dict[str, Any]] = []
        self.authority: WriteAuthority | None = None

    @property
    def grants_dir(self) -> Path:
        return self.tree / OWNER / wa.AUTHORIZATIONS_DIR_NAME

    def switch(self, state: str) -> None:
        _record(self.tree, state)

    def values(self) -> dict[str, float]:
        readings = ConnectorInterface(TARGET).read_many([A, B])
        return {name: float(reading.value) for name, reading in readings.items()}

    def requests(self) -> list[Path]:
        directory = self.ledger / wa.REQUESTS_DIR_NAME
        return sorted(directory.glob("*.json")) if directory.is_dir() else []

    def spent(self, authorization: str) -> int:
        path = self.ledger / f"{authorization}.ledger.json"
        return len(wa._parse_ledger(path.read_text()).steps) if path.exists() else 0

    def approve(self, request_path: Path) -> wa.Grant:
        request = wa.read_request(request_path)
        return wa.approve_request(
            request,
            surface="cli",
            record_file=self.tree / OWNER / control_context.RECORD_FILENAME,
            approver=OWNER,
        )

    def config(self, **fields: Any) -> OptimizationConfig:
        window = VariableOverride(min=-2.0, max=2.0, step_size=0.5)
        values: dict[str, Any] = {
            "environment_name": "knobs",
            "control_target": TARGET,
            "objective_name": "knob_a",
            "algorithm": "random",
            "n_initial_samples": 2,
            "n_iterations": EVALUATIONS - 2,
            "variables": [A, B],
            "variable_overrides": {A: window, B: window},
            **fields,
        }
        config = OptimizationConfig(**values)
        env = self.factory("knobs", {"control_target": TARGET})
        with env:
            resolve_draft_config(config, env)
        return config

    @staticmethod
    def factory(_name: str, env_config: dict[str, Any]) -> ConnectorEnvironment:
        return ConnectorEnvironment({**env_config, "spec": SPEC})

    def run(
        self,
        config: OptimizationConfig,
        *,
        bus: EventBus | None = None,
        check_cancellation: Callable[[], None] | None = None,
        apply_best: bool = False,
    ) -> RunAuthorization:
        """Run *config* as a worker does: authority installed, authorization held, released."""
        self.authority = WriteAuthority(required=True)
        install_process_authority(self.authority)
        held = RunAuthorization(
            self.authority, publish=self.states.append, check_cancellation=check_cancellation
        )
        try:
            result = OptimizationEngine(
                config,
                environment_factory=self.factory,
                services=RunnerServices(event_bus=bus, authorize=held.acquire),
                job_id="job-1",
            ).run()
            if apply_best:
                apply_best_point(self.factory("knobs", {"control_target": TARGET}), result)
        finally:
            held.release()
        return held

    def granted_id(self) -> str:
        granted = [state for state in self.states if state.get("state") == "granted"]
        assert granted, self.states
        return str(granted[-1]["id"])


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Deployment]:
    limits = {name: {"writable": True, "min_value": -5.0, "max_value": 5.0} for name in (A, B)}
    (tmp_path / "channel_limits.json").write_text(json.dumps(limits))
    section = {
        "type": IN_PROCESS,
        "writes": "allowed",
        "login_tiers": {OWNER: "allowed"},
        "limits_checking": {
            "enabled": True,
            "database_path": str(tmp_path / "channel_limits.json"),
        },
        "connector": {
            IN_PROCESS: in_process_config(
                served_tree(tmp_path / "served", (A, B)), response_delay_ms=1
            )
        },
    }
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"control_system": section}))

    built = Deployment(tmp_path)
    built.tree.mkdir()
    (built.tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n")
    built.ledger.mkdir()
    built.switch("off")

    for name in (
        posture_store.DISPATCH_RUN_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
        "OSPREY_SESSION_ID",
        wa.AUTHORIZATION_ENV_VAR,
        wa.REQUESTS_DIR_ENV_VAR,
        wa.LEDGER_VIEW_DIR_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "unset-by-the-fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", HOLDER)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(built.tree))
    monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(built.ledger))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src")]
        ),
    )
    monkeypatch.setattr(run_auth, "POLL_S", 0.05)
    stamp_owner(OWNER)
    control_context.invalidate_cache()
    random.seed(7)
    yield built
    install_process_authority(None)
    iface_module.reset_bridge()
    control_context.invalidate_cache()


def _approve_when_asked(deployment: Deployment) -> threading.Thread:
    """Approve the run's request from another thread, as a person at the chip would."""

    def approve() -> None:
        deadline = time.monotonic() + 60
        while not deployment.requests():
            assert time.monotonic() < deadline, "the run never asked"
            time.sleep(0.05)
        deployment.approve(deployment.requests()[0])

    thread = threading.Thread(target=approve, daemon=True)
    thread.start()
    return thread


# -- self-issue at allowed ------------------------------------------------------


def test_at_allowed_the_run_self_issues_and_spends_one_step_per_evaluation_and_one_to_complete(
    deployment: Deployment,
) -> None:
    deployment.switch("allowed")
    deployment.run(deployment.config(), apply_best=True)

    grant_id = deployment.granted_id()
    granted = next(s for s in deployment.states if s.get("state") == "granted")
    assert granted["approver"] == wa.SELF_ISSUE_SURFACE
    assert granted["steps"] == EVALUATIONS + 1
    assert granted["completion"] == "apply_best"
    # The grant is in the run's own ledger directory, never in the read-only tree.
    assert (deployment.ledger / f"{grant_id}.json").is_file()
    assert not deployment.grants_dir.exists() or not list(deployment.grants_dir.glob("*.json"))
    # Exactly one write call per evaluation, and one for apply_best.
    assert deployment.spent(grant_id) == EVALUATIONS + 1
    assert deployment.requests() == []
    # Done with it, the run ends its own grant rather than leave it standing.
    grant = wa._parse_grant((deployment.ledger / f"{grant_id}.json").read_text())
    assert grant.revoked is not None and grant.revoked.by == HOLDER


def test_the_channels_asked_for_are_the_setpoints_at_the_runs_window(
    deployment: Deployment,
) -> None:
    deployment.switch("allowed")
    deployment.run(deployment.config(on_complete="hold"))
    grant = wa._parse_grant((deployment.ledger / f"{deployment.granted_id()}.json").read_text())
    assert dict(grant.channels) == {
        A: wa.ChannelRange(-2.0, 2.0),
        B: wa.ChannelRange(-2.0, 2.0),
    }
    # hold writes nothing on completion: the budget is the evaluations alone.
    assert grant.steps == EVALUATIONS
    assert deployment.spent(grant.id) == EVALUATIONS


def test_an_expiry_set_past_the_ceiling_is_held_under_it(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held a margin under the ceiling, so an approver's lagging clock still reads it."""
    deployment.switch("allowed")
    monkeypatch.setattr(run_auth, "EXPIRES_S", 7 * 24 * 3600.0)
    deployment.run(deployment.config(on_complete="hold"))
    grant = wa._parse_grant((deployment.ledger / f"{deployment.granted_id()}.json").read_text())
    held = (wa.MAX_EXPIRES_S - run_auth._EXPIRY_MARGIN_S) * 1e9
    assert grant.expires_at_ns - grant.issued_at_ns <= held


def test_a_flaky_objective_read_costs_no_extra_step_and_no_extra_write(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-measurement re-reads the objective; the setpoints are written once per point."""
    from osprey_optimizer.models.protocols import PVReading

    real_evaluate = ConnectorEnvironment.evaluate
    real_set_values = ConnectorInterface.set_values
    writes: list[dict[str, float]] = []

    def flaky_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
        readings = dict(real_evaluate(self, *args, **kwargs))
        readings["knob_a"] = PVReading(value=None, connected=True)
        return readings

    def counted_set_values(self: Any, values: Any, *args: Any, **kwargs: Any) -> Any:
        writes.append(dict(values))
        return real_set_values(self, values, *args, **kwargs)

    monkeypatch.setattr(ConnectorEnvironment, "evaluate", flaky_evaluate)
    monkeypatch.setattr(ConnectorInterface, "set_values", counted_set_values)
    deployment.switch("allowed")
    deployment.run(deployment.config(on_complete="hold"))

    grant_id = deployment.granted_id()
    assert deployment.spent(grant_id) == EVALUATIONS
    assert len(writes) == EVALUATIONS


# -- a person's approval --------------------------------------------------------


def test_below_allowed_the_run_writes_nothing_until_a_person_approves_its_request(
    deployment: Deployment,
) -> None:
    before = deployment.values()
    writes_before_approval: list[dict[str, float]] = []

    def approve() -> None:
        deadline = time.monotonic() + 60
        while not deployment.requests():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        # The run is waiting: it has asked and written nothing.
        writes_before_approval.append(deployment.values())
        request = wa.read_request(deployment.requests()[0])
        assert request.holder == HOLDER
        assert request.owner == OWNER
        assert request.steps == EVALUATIONS + 1
        assert request.completion == "apply_best"
        assert request.label == "knobs optimization (job-1)"
        assert set(request.channels) == {A, B}
        assert deployment.states[-1]["state"] == "awaiting"
        assert deployment.states[-1]["id"] == request.id
        deployment.approve(deployment.requests()[0])

    thread = threading.Thread(target=approve, daemon=True)
    thread.start()
    deployment.run(deployment.config(), apply_best=True)
    thread.join(timeout=5)

    assert writes_before_approval == [before]
    grant_id = deployment.granted_id()
    granted = next(s for s in deployment.states if s.get("state") == "granted")
    assert granted["approver"] == "cli"
    assert deployment.spent(grant_id) == EVALUATIONS + 1
    # The request is withdrawn once granted.
    assert deployment.requests() == []


def test_a_cancel_while_awaiting_withdraws_the_request_and_writes_nothing(
    deployment: Deployment,
) -> None:
    before = deployment.values()
    cancelled = threading.Event()

    def check_cancellation() -> None:
        if deployment.requests():
            cancelled.set()
            raise CancellationError("Optimization was cancelled by user")

    with pytest.raises(CancellationError):
        deployment.run(deployment.config(), check_cancellation=check_cancellation)

    assert cancelled.is_set()
    assert deployment.requests() == []
    assert deployment.values() == before
    assert not list(deployment.ledger.glob("*.ledger.json"))
    assert deployment.authority is not None and deployment.authority.authorization is None
    # The job's status no longer says it awaits an approval nobody can give.
    assert any(state["state"] == "awaiting" for state in deployment.states)
    assert (deployment.states[-1]["state"], deployment.states[-1]["reason"]) == (
        "ended",
        "released",
    )


def test_a_request_nobody_approves_ends_the_run_and_is_withdrawn(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_auth, "WAIT_S", 0.3)
    before = deployment.values()
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config())
    assert caught.value.reason == AUTHORIZATION_TIMEOUT
    assert deployment.states[-1]["state"] == "ended"
    assert deployment.requests() == []
    assert deployment.values() == before


@pytest.mark.parametrize("switch", ["allowed", "asked"])
def test_a_plan_past_the_step_ceiling_ends_the_run_before_it_asks(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    """Refused up front with its own reason, not left to end ``auth_unreadable``."""
    deployment.switch(switch)
    monkeypatch.setattr(wa, "MAX_STEPS", EVALUATIONS - 1)
    before = deployment.values()
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config(on_complete="hold"))
    assert caught.value.reason == AUTHORIZATION_TOO_MANY_STEPS
    assert (deployment.states[-1]["state"], deployment.states[-1]["reason"]) == (
        "ended",
        AUTHORIZATION_TOO_MANY_STEPS,
    )
    assert deployment.requests() == []
    assert not any(state.get("state") == "granted" for state in deployment.states)
    assert deployment.values() == before


def test_a_run_for_nobody_ends_before_it_asks(deployment: Deployment) -> None:
    stamp_owner(None)
    before = deployment.values()
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config())
    assert caught.value.reason == "owner_unknown"
    assert deployment.requests() == []
    assert deployment.values() == before


# -- the end of an authorization mid-run ------------------------------------------


def _with_plan(monkeypatch: pytest.MonkeyPatch, change: Callable[[Any], Any]) -> None:
    """Make the plan the run asks for *change*'s version of the one it resolves."""
    real = orchestration.write_plan

    def changed(*args: Any, **kwargs: Any) -> Any:
        return change(real(*args, **kwargs))

    monkeypatch.setattr(orchestration, "write_plan", changed)


def _assert_no_further_write(deployment: Deployment, grant_id: str, reason: str) -> None:
    """After the end, the worker's writes are refused before any connector is asked."""
    spent = deployment.spent(grant_id)
    values = deployment.values()
    with pytest.raises(AuthorizationEndedError) as caught:
        ConnectorInterface(TARGET).set_values({A: 0.5, B: 0.5})
    assert caught.value.reason == reason
    assert deployment.spent(grant_id) == spent
    assert deployment.values() == values


def test_a_budget_spent_ends_the_run_auth_exhausted_with_no_further_write(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that needs more write calls than it was approved for stops at the bound."""
    deployment.switch("allowed")
    _with_plan(monkeypatch, lambda plan: replace(plan, evaluations=plan.evaluations - 2))
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config(), apply_best=True)
    assert caught.value.reason == "auth_exhausted"
    grant_id = deployment.granted_id()
    # Two evaluations fewer plus apply_best: three steps, all spent on evaluations.
    assert deployment.spent(grant_id) == EVALUATIONS - 2 + 1
    assert (deployment.states[-1]["state"], deployment.states[-1]["reason"]) == (
        "ended",
        "auth_exhausted",
    )
    _assert_no_further_write(deployment, grant_id, "auth_exhausted")


def test_a_value_outside_the_approved_range_ends_the_run_auth_range_writing_nothing(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment.switch("allowed")
    start = deployment.values()
    # Approved only where A stands now: the first evaluation (the current
    # point) is inside it, the random samples after it are not.
    _with_plan(
        monkeypatch,
        lambda plan: replace(plan, channels={**plan.channels, A: (start[A], start[A])}),
    )
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config())
    assert caught.value.reason == "auth_range"
    grant_id = deployment.granted_id()
    # The refused batch spent nothing and wrote nothing, B included.
    assert deployment.spent(grant_id) == 1
    assert deployment.values() == start
    _assert_no_further_write(deployment, grant_id, "auth_range")


def test_a_persons_stop_mid_run_ends_it_auth_revoked_with_no_further_write(
    deployment: Deployment,
) -> None:
    bus = EventBus()
    stopped: list[tuple[str, ...]] = []

    def stop_after_the_second(event: Any) -> None:
        if event.type == EventType.ITERATION_COMPLETED and event.data.iteration == 2:
            stopped.append(wa.revoke_grants(by=OWNER, directory=deployment.grants_dir))

    bus.subscribe_all(stop_after_the_second)
    approver = _approve_when_asked(deployment)
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config(), bus=bus, apply_best=True)
    approver.join(timeout=5)
    assert caught.value.reason == "auth_revoked"
    grant_id = deployment.granted_id()
    assert stopped == [(grant_id,)]
    assert deployment.spent(grant_id) == 2
    _assert_no_further_write(deployment, grant_id, "auth_revoked")


def test_turning_the_switch_down_ends_a_self_issued_run_auth_revoked(
    deployment: Deployment,
) -> None:
    deployment.switch("allowed")
    bus = EventBus()

    def ask_after_the_first(event: Any) -> None:
        if event.type == EventType.ITERATION_COMPLETED and event.data.iteration == 1:
            deployment.switch("asked")

    bus.subscribe_all(ask_after_the_first)
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config(), bus=bus)
    assert caught.value.reason == "auth_revoked"
    assert deployment.spent(deployment.granted_id()) == 1


def test_a_lease_that_cannot_be_renewed_ends_the_run_with_no_further_write(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment.switch("allowed")
    monkeypatch.setattr(run_auth, "LEASE_S", 3.0)
    bus = EventBus()

    def lose_the_lease_after_the_first(event: Any) -> None:
        if event.type == EventType.ITERATION_COMPLETED and event.data.iteration == 1:
            monkeypatch.setattr(wa, "renew_lease", lambda *_args, **_kwargs: False)
            assert deployment.authority is not None
            deadline = time.monotonic() + 10
            while deployment.authority.ended is None:
                assert time.monotonic() < deadline, "the heartbeat never noticed"
                time.sleep(0.05)

    bus.subscribe_all(lose_the_lease_after_the_first)
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config(), bus=bus)
    assert caught.value.reason == "auth_lease_lapsed"
    grant_id = deployment.granted_id()
    assert deployment.spent(grant_id) == 1
    assert deployment.states[-1]["reason"] == "auth_lease_lapsed"
    _assert_no_further_write(deployment, grant_id, "auth_lease_lapsed")


def test_a_lapsed_lease_is_refused_by_the_child_itself(
    deployment: Deployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a worker that has not noticed its lease lapsed cannot write: the child refuses."""
    deployment.switch("allowed")
    monkeypatch.setattr(run_auth, "LEASE_S", 1.5)
    bus = EventBus()

    def outlive_the_lease(event: Any) -> None:
        if event.type == EventType.ITERATION_COMPLETED and event.data.iteration == 1:
            # The worker believes its renewals land; none does.
            monkeypatch.setattr(wa, "renew_lease", lambda *_args, **_kwargs: True)
            time.sleep(2.5)

    bus.subscribe_all(outlive_the_lease)
    with pytest.raises(AuthorizationEndedError) as caught:
        deployment.run(deployment.config(), bus=bus)
    assert caught.value.reason == "auth_lease_lapsed"
    assert deployment.spent(deployment.granted_id()) == 1


def test_a_readonly_run_asks_for_nothing(deployment: Deployment) -> None:
    deployment.switch("allowed")
    asked: list[Any] = []
    with pytest.raises(WriteRefusedError):
        OptimizationEngine(
            deployment.config(execution_mode="readonly"),
            environment_factory=deployment.factory,
            services=RunnerServices(authorize=asked.append),
        ).run()
    assert asked == []
    assert deployment.requests() == []


# -- the API's manual writes: one step each ---------------------------------------


@pytest.fixture
def api(deployment: Deployment) -> Iterator[Any]:
    """The API over the deployment, the process stamped for nobody: only the header names alice."""
    del deployment  # wanted for the deployment it builds first
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from osprey_optimizer.api.dependencies import get_environment_registry, get_job_tracker
    from osprey_optimizer.api.main import app

    stamp_owner(None)
    registry = MagicMock()
    registry.load_environment.side_effect = lambda name, config: Deployment.factory(name, config)
    tracker = MagicMock()
    tracker.get_active_job.return_value = None
    app.dependency_overrides[get_environment_registry] = lambda: registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


ALICE = {"X-Osprey-Owner": OWNER}
SET = f"/environments/knobs/variables?target={TARGET}"
BODY = {"variables": {A: 1.5, B: -1.0}, "override_unhealthy": True}


def _only_grant(directory: Path) -> wa.Grant:
    grants, unreadable = wa.list_grants(directory)
    assert unreadable == ()
    [grant] = grants
    return grant


def test_a_manual_write_at_allowed_is_one_self_issued_step_at_exactly_its_values(
    deployment: Deployment, api: Any
) -> None:
    deployment.switch("allowed")
    response = api.put(SET, json=BODY, headers=ALICE)
    assert response.status_code == 204, response.text
    assert deployment.values() == {A: 1.5, B: -1.0}
    grant = _only_grant(deployment.ledger)
    assert grant.owner == OWNER and grant.holder == HOLDER
    assert grant.approver.surface == wa.SELF_ISSUE_SURFACE
    assert grant.steps == 1
    assert dict(grant.channels) == {A: wa.ChannelRange(1.5, 1.5), B: wa.ChannelRange(-1.0, -1.0)}
    assert deployment.spent(grant.id) == 1
    assert grant.revoked is not None


@pytest.mark.parametrize(
    ("position", "reason"), [("off", "switch_off"), ("asked", "switch_not_allowed")]
)
def test_a_manual_write_below_allowed_is_refused_and_asks_for_allowed(
    deployment: Deployment, api: Any, position: str, reason: str
) -> None:
    """No request is written and no call is repeated: the operator sets the switch.

    At ``asked`` the reason says the switch is not at allowed, never
    ``auth_revoked``: the call held no authorization that could have ended.
    """
    from osprey_optimizer.api.manual_authorization import SET_ALLOWED

    deployment.switch(position)
    before = deployment.values()
    response = api.put(SET, json=BODY, headers=ALICE)
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["reason"] == reason
    assert SET_ALLOWED in detail["message"]
    assert deployment.values() == before
    assert deployment.requests() == []
    assert wa.list_grants(deployment.ledger) == ((), ())


def test_a_manual_write_for_nobody_is_refused_owner_unknown(
    deployment: Deployment, api: Any
) -> None:
    deployment.switch("allowed")
    before = deployment.values()
    response = api.put(SET, json=BODY)
    assert response.status_code == 403
    assert response.json()["detail"]["reason"] == "owner_unknown"
    assert deployment.values() == before
    assert deployment.requests() == []


def test_evaluate_is_one_step_of_its_own_authorization(deployment: Deployment, api: Any) -> None:
    deployment.switch("allowed")
    response = api.post(
        f"/environments/knobs/evaluate?target={TARGET}",
        json={"variables": {A: 0.5}, "observables": ["knob_a"], "override_unhealthy": True},
        headers=ALICE,
    )
    assert response.status_code == 200, response.text
    assert response.json()["observables"]["knob_a"]["value"] == pytest.approx(0.5)
    grant = _only_grant(deployment.ledger)
    assert dict(grant.channels) == {A: wa.ChannelRange(0.5, 0.5)}
    assert deployment.spent(grant.id) == 1


def _run_dir(root: Path, start: dict[str, float]) -> str:
    run = root / "results" / "knobs" / "2026-11-02_10-00-00"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(
            {
                "environment_name": "knobs",
                "control_target": TARGET,
                "objective_name": "knob_a",
                "variables": [A, B],
            }
        )
    )
    (run / "initial_point.json").write_text(json.dumps({**start, "knob_a": 0.0}))
    return "knobs/2026-11-02_10-00-00"


def test_a_manual_restore_is_its_own_one_step_authorization_at_the_start_values(
    deployment: Deployment, api: Any, tmp_path: Path
) -> None:
    """Never part of a run's budget and never automatic: the operator's preview,
    confirm, and one self-issued step at exactly the recorded start values."""
    start = {A: 1.0, B: -0.5}
    run_id = _run_dir(tmp_path, start)
    deployment.switch("allowed")
    api.put(SET, json=BODY, headers=ALICE)  # the machine moved since the run started
    first = _only_grant(deployment.ledger)

    preview = api.get(f"/results/{run_id}/restore-preview")
    assert preview.status_code == 200, preview.text
    token = preview.json()["preview_token"]
    restore = f"/results/{run_id}/restore?confirm_token={token}&override_unhealthy=true"

    deployment.switch("off")
    refused = api.post(restore, headers=ALICE)
    assert refused.status_code == 403, refused.text
    assert deployment.values() == {A: 1.5, B: -1.0}
    assert deployment.requests() == []

    deployment.switch("allowed")
    done = api.post(restore, headers=ALICE)
    assert done.status_code == 200, done.text
    assert done.json()["restored"] == start
    assert deployment.values() == start
    grants, _ = wa.list_grants(deployment.ledger)
    [grant] = [g for g in grants if g.id != first.id]
    assert dict(grant.channels) == {
        name: wa.ChannelRange(value, value) for name, value in start.items()
    }
    assert grant.steps == 1
    assert deployment.spent(grant.id) == 1
    assert grant.revoked is not None
