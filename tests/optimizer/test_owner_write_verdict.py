"""A run's owner decides its writes, in the connector child, as the deployed container has it.

The optimizer's container binds the control-context tree read-only and runs as
an account that names nobody, so the connector governs each write by the switch
of the login the write is for (``current_owner``) and that login's rung in the
staged ``login_tiers`` map. These rows build exactly that: a provisioned tree, a
section whose tier allows writes and whose map names the owner, and a real
connector-host child serving the in-process simulator behind the optimizer's own
:class:`ConnectorInterface`. The owner reaches the child the way a run's worker
hands it on (:func:`osprey_optimizer.api.owner.stamp_owner`), and nothing in the
optimizer decides the verdict.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey_connectors import control_context, posture_store
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import write_payload
from osprey_optimizer.api.owner import stamp_owner
from osprey_optimizer.environments import interface as iface_module
from osprey_optimizer.environments.interface import ConnectorInterface
from osprey_optimizer.models.exceptions import WriteRefusedError
from tests.facility.served_tree import in_process_config, served_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_PROCESS = "osprey_connectors.control_system._va_in_process_connector.VAInProcessConnector"

#: The target the section's own connector type serves: ``live`` resolves to the
#: section's ``type``, here the in-process simulator, so the pool brings up one
#: child serving it without a Channel Access gateway. (``va`` would resolve to
#: the Channel Access simulator type, which the pool refuses without a gateway.)
TARGET = "live"
CHANNEL = "KNOB:POS:SP"
OWNER = "alice"
#: A login the map names ``never``.
DEMOTED = "bob"

pytestmark = pytest.mark.timeout(120)


def _switch(tree: Path, owner: str, state: str) -> None:
    """*owner*'s record with the switch at *state* for :data:`TARGET`, as the chip leaves it."""
    payload = control_context.ControlContext(
        target=TARGET, generation=1, switch_epoch=1
    ).to_payload()
    payload["switch"] = {TARGET: state}
    write_payload(tree / owner / control_context.RECORD_FILENAME, payload)
    control_context.invalidate_cache()


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The optimizer container's write path: tier ``allowed``, a login map, a bound tree.

    Alice's and Bob's switches are ``allowed`` for the simulator; the map gives
    Alice ``allowed`` and Bob ``never``. Every variable that would decide the
    verdict besides the owner is cleared, so the owner is the one thing a row
    changes.
    """
    limits = {CHANNEL: {"writable": True, "min_value": -5.0, "max_value": 5.0}}
    (tmp_path / "channel_limits.json").write_text(json.dumps(limits))
    section = {
        "type": IN_PROCESS,
        "writes": "allowed",
        "login_tiers": {OWNER: "allowed", DEMOTED: "never"},
        "limits_checking": {
            "enabled": True,
            "database_path": str(tmp_path / "channel_limits.json"),
        },
        "connector": {
            IN_PROCESS: in_process_config(
                served_tree(tmp_path / "served", (CHANNEL,)), response_delay_ms=1
            )
        },
    }
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"control_system": section}))

    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir()
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    for owner in (OWNER, DEMOTED):
        _switch(tree, owner, "allowed")

    # Set, then cleared, so the stamp a row makes is undone after it.
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "unset-by-the-fixture")
    for name in (
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.DISPATCH_RUN_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
        "OSPREY_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIG_FILE", str(config))
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src")]
        ),
    )
    control_context.invalidate_cache()
    yield tree
    iface_module.reset_bridge()
    control_context.invalidate_cache()


def _refused(reason: str) -> str:
    """The words of *reason*'s refusal, as the child sends them (kind left out)."""
    return refusal_phrase(reason).rstrip(".")


@pytest.mark.usefixtures("deployment")
def test_a_run_for_an_owner_whose_switch_is_allowed_writes() -> None:
    stamp_owner(OWNER)
    interface = ConnectorInterface(TARGET)
    interface.set_values({CHANNEL: 0.5})
    assert interface.read_many([CHANNEL])[CHANNEL].value == pytest.approx(0.5)


@pytest.mark.usefixtures("deployment")
def test_a_run_for_nobody_is_refused_owner_unknown() -> None:
    stamp_owner(None)
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface(TARGET).set_values({CHANNEL: 0.5})
    assert _refused("owner_unknown") in str(caught.value)


@pytest.mark.usefixtures("deployment")
def test_a_scheduled_run_for_nobody_is_refused_owner_unscheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(posture_store.DISPATCH_RUN_ENV_VAR, "1")
    stamp_owner(None)
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface(TARGET).set_values({CHANNEL: 0.5})
    assert _refused("owner_unscheduled") in str(caught.value)


@pytest.mark.usefixtures("deployment")
def test_a_run_for_an_owner_the_map_does_not_name_is_refused_owner_unknown() -> None:
    stamp_owner("carol")
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface(TARGET).set_values({CHANNEL: 0.5})
    assert _refused("owner_unknown") in str(caught.value)


@pytest.mark.usefixtures("deployment")
def test_a_run_for_an_owner_mapped_never_is_refused_by_the_tier() -> None:
    stamp_owner(DEMOTED)
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface(TARGET).set_values({CHANNEL: 0.5})
    assert _refused("tier_never") in str(caught.value)


def test_a_run_for_an_owner_whose_switch_is_off_is_refused(deployment: Path) -> None:
    _switch(deployment, OWNER, "off")
    stamp_owner(OWNER)
    with pytest.raises(WriteRefusedError) as caught:
        ConnectorInterface(TARGET).set_values({CHANNEL: 0.5})
    assert _refused("switch_off") in str(caught.value)
