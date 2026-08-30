"""One sandbox run carries exactly one limits posture, on both write paths.

A run resolves the posture twice, from different halves of the resolver family.
The HOST resolves it for the session target and embeds it in the generated
script (``LimitsValidator.from_config(target=...)``, serialized into the
wrapper's monkeypatch). A connector built through the runtime's own connector
path resolves its own inside ``connect()``, from the type the stamp resolved to
(``LimitsValidator.from_config(connector_type=...)``). Neither consults the
other.

The two write paths do not map one-to-one onto those halves.
``osprey.runtime.write_channel`` runs the embedded validator and *then*
``write_channel_checked`` on the connector, which runs the connector's own — so
its verdict is the CONJUNCTION of both halves, and a refusal there does not by
itself say which half refused. A direct ``write_channel_checked`` on the
connector exercises the connector's half alone. The ``injected_policy``
assertions are what pin the host's half on its own: they read the policy the
wrapper actually embedded, whatever the writes went on to do with it.

So a per-type block the target adapter and the type adapter read differently
shows up here and nowhere else, in one of two signatures. Host strict over a
permissive connector: ``write_channel`` refused while the direct connector write
is allowed. Host permissive over a strict connector: both refused, since the
conjunction still fails — caught by ``injected_policy`` naming the wrong key
rather than by the write outcomes. Either way the two halves disagree, in a
single run, against a single machine. The deployment under test is the one that
makes their answers differ at all — deployment-wide strict, with
``connector.virtual_accelerator`` relaxed — so an implementation that resolved
either half deployment-wide, or either half per-type, fails one of the two
scenarios rather than passing both.

Both scenarios write the SAME unlisted channel, so the only thing that moves
between them is the session's target record.

What is real here and what is not
---------------------------------
Real: the on-disk ``config.yml``, the limits database, the controls server's
target-state record, ``_execute_via_local`` (which stamps the environment,
resolves the posture and builds the wrapper), the generated script, the
subprocess that runs it, and every ``from_config`` call either half makes.

Substituted: the connector CLASS behind the two types, registered inside the
sandbox script as :class:`MockConnector` so the run needs no soft IOC and no
facility. The substitution is downstream of everything under test — the factory
still stamps ``_connector_type`` from the type the target resolved to, and
``connect()`` still reads its posture from that stamp — so it changes which wire
a write would reach and nothing about which posture answers.
"""

import asyncio
import json
import os
import textwrap
from pathlib import Path

import pytest
import yaml

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor as host_executor

pytestmark = pytest.mark.unit

#: Absent from the limits database on purpose: the posture is the only thing
#: that can decide this write. Free of ``:SP``/``:SET``, which the mock mirrors
#: onto a readback channel.
UNLISTED_CHANNEL = "SANDBOX:POSTURE:PROBE"

#: The deployment-wide key, which is what a strict refusal must name.
DEPLOYMENT_WIDE_KEY = "control_system.limits_checking.allow_unlisted_channels"

#: The per-type key, which is what the relaxed VA posture answers with.
VA_KEY = "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"

#: Marks the one line of sandbox stdout the assertions read.
VERDICT_PREFIX = "SINGLE_POSTURE_VERDICT "

# The script the sandbox runs. Both write paths are attempted whatever the first
# one does, so a scenario always reports two outcomes and a disagreement shows as
# a disagreement rather than as a truncated run.
PROBE_SCRIPT = textwrap.dedent(
    f"""
    import json

    from osprey_connectors.control_system.mock_connector import MockConnector
    from osprey_connectors.factory import ConnectorFactory

    # See the module docstring: the class is substituted, the type keys are not.
    ConnectorFactory.register_control_system("epics", MockConnector)
    ConnectorFactory.register_control_system("virtual_accelerator", MockConnector)

    import osprey.runtime as _rt

    _verdict = {{}}

    # The policy the HOST resolved and embedded, reported so a failure says which
    # posture the sandbox was actually carrying.
    _injected = getattr(_rt, "_limits_validator", None)
    _verdict["injected_policy"] = None if _injected is None else _injected.policy

    try:
        _rt.write_channel({UNLISTED_CHANNEL!r}, 1.0)
        _verdict["write_channel"] = {{"allowed": True, "error": ""}}
    except Exception as exc:
        _verdict["write_channel"] = {{"allowed": False, "error": str(exc)}}


    async def _via_connector():
        connector = await _rt._get_connector()
        # Recorded before the write, so a refused run still reports which type
        # the stamp resolved to.
        _verdict["connector_type"] = connector._connector_type
        await connector.write_channel_checked({UNLISTED_CHANNEL!r}, 2.0)


    _verdict["connector_type"] = None
    try:
        _rt._run_async(_via_connector())
        _verdict["connector"] = {{"allowed": True, "error": ""}}
    except Exception as exc:
        _verdict["connector"] = {{"allowed": False, "error": str(exc)}}

    print({VERDICT_PREFIX!r} + json.dumps(_verdict))
    """
).strip()


def _write_deployment(root: Path) -> Path:
    """Write a VA-permissive / live-strict deployment under *root*.

    ``control_system.type`` is ``epics``, so ``resolve_target`` answers ``live``
    with the baseline type — a type with no ``limits_checking`` block of its own,
    which is what makes it inherit the strict deployment-wide pair. ``va`` always
    resolves to ``virtual_accelerator``, whose block relaxes both leaves.

    Returns:
        The config file's path.
    """
    limits_path = root / "limits.json"
    limits_path.write_text(
        json.dumps({"SANDBOX:LISTED:CHANNEL": {"min_value": 0.0, "max_value": 10.0}}),
        encoding="utf-8",
    )

    config = {
        "project_root": str(root),
        "agent_data": {"base_dir": "var/agent_data"},
        "control_system": {
            "type": "epics",
            # Otherwise the connector's writes_enabled gate refuses first and
            # neither path ever reaches a limits decision.
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": False,
                "database_path": "limits.json",
            },
            "connector": {
                "virtual_accelerator": {
                    "limits_checking": {
                        "enabled": True,
                        "allow_unlisted_channels": True,
                    }
                }
            },
        },
    }
    config_path = root / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """An on-disk deployment this process and its sandbox children both read.

    ``OSPREY_CONFIG`` is what ``load_osprey_config`` resolves (the project root
    and the agent-data root come from it) and ``CONFIG_FILE`` is what the
    connectors' ``get_config_value`` singleton reads. Both are set to the same
    file so the host's posture resolution and the sandbox's cannot be answered by
    two different configs, and both are inherited by the child.
    """
    from osprey_connectors.workspace import reset_config_cache

    root = tmp_path / "deployment"
    root.mkdir()
    config_path = _write_deployment(root)

    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    reset_config_cache()

    (root / "var" / "agent_data" / target_state.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return root


def _publish_target(target: str) -> None:
    """Write the controls-server record that puts this session on *target*.

    ``server_pid`` is this process, which is alive and whose file the sandbox
    reads back for its generation pin; ``owner_ppid`` is this process's parent,
    which is the equality ``_session_target_record`` matches on.
    """
    record = {
        "target": target,
        "generation": 0,
        "server_pid": os.getpid(),
        "owner_ppid": os.getppid(),
        "targets": {
            name: {"label": "", "endpoint": "", "real_machine": False}
            for name in target_state.TARGET_NAMES
        },
        "children": [],
    }
    target_state.state_file_path().write_text(json.dumps(record), encoding="utf-8")


def _run_sandbox(root: Path, target: str) -> dict:
    """Run the probe script in a real sandbox on *target*; return its verdict.

    Drives ``_execute_via_local`` — the real path, including the environment
    stamp, the posture resolution, the generated wrapper and the subprocess.
    """
    _publish_target(target)

    folder = root / "var" / "agent_data" / "python_executions" / target
    (folder / "figures").mkdir(parents=True)

    result = asyncio.run(
        host_executor._execute_via_local(PROBE_SCRIPT, "readwrite", {"timeout": 300}, folder)
    )

    assert result.control_target == target, (
        f"the run was stamped for {result.control_target!r}, not {target!r}; "
        f"stderr:\n{result.stderr}"
    )

    for line in result.stdout.splitlines():
        if line.startswith(VERDICT_PREFIX):
            return json.loads(line[len(VERDICT_PREFIX) :])
    raise AssertionError(
        f"the sandbox printed no verdict.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_permissive_target_allows_both_write_paths(deployment):
    """On ``va``, the per-type block answers both halves of one run.

    One scenario, one subprocess, every assertion about it — a run of the real
    sandbox is expensive enough that splitting these across test functions would
    buy granularity by paying for the same run several times.
    """
    verdict = _run_sandbox(deployment, "va")

    assert verdict["write_channel"] == {"allowed": True, "error": ""}
    assert verdict["connector"] == {"allowed": True, "error": ""}
    # The half of the agreement the embedded policy cannot account for: the
    # connector resolved its own posture from this type, independently.
    assert verdict["connector_type"] == "virtual_accelerator"
    assert verdict["injected_policy"] == {
        "allow_unlisted_channels": True,
        "allow_unlisted_key": VA_KEY,
    }


def test_strict_target_refuses_both_write_paths(deployment):
    """On ``live``, the deployment-wide block answers both halves of one run."""
    verdict = _run_sandbox(deployment, "live")

    assert verdict["write_channel"]["allowed"] is False
    assert DEPLOYMENT_WIDE_KEY in verdict["write_channel"]["error"]
    assert verdict["connector"]["allowed"] is False
    assert DEPLOYMENT_WIDE_KEY in verdict["connector"]["error"]
    # The relaxation belongs to the virtual accelerator. A refusal quoting it on
    # the live machine would send an operator to edit the simulator's line.
    assert VA_KEY not in verdict["write_channel"]["error"]
    assert VA_KEY not in verdict["connector"]["error"]

    assert verdict["connector_type"] == "epics"
    assert verdict["injected_policy"] == {
        "allow_unlisted_channels": False,
        "allow_unlisted_key": DEPLOYMENT_WIDE_KEY,
    }
