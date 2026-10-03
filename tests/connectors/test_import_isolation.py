"""Import isolation for the lean control-system connector chain.

External consumers (a site's tuning-scripts backend, say) import only the
control-system connectors and their support modules. That chain must not
eagerly load the archiver stack (pandas) or any LLM/agent machinery.

The control-context record and the acting-identity ladder are held to a
stricter rule still: they are read inside connector-host children, executor
sandboxes and notebook kernels, so they may not reach ``osprey`` at all.

The connectors distribution as a whole runs with no ``osprey`` on the path.
The mock archiver is the one connector that reads project config beyond its
own block, so it is proven here by an actual ``connect``, not by an import.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

SRC = str(Path(__file__).resolve().parents[2] / "src")

# Sentinels for the two dependency trees that must stay out of the lean chain:
# the archiver's dataframe stack and the LLM/agent platform.
FORBIDDEN = ("pandas", "litellm", "openai", "anthropic", "fastapi", "playwright")


def test_control_system_chain_imports_without_heavy_deps():
    code = (
        "import osprey.connectors.control_system.epics_connector;"
        "import osprey.connectors.control_system.limits_validator;"
        "import osprey.errors, osprey.utils.config, osprey.utils.logger;"
        "import osprey.simulation, sys;"
        f"bad = sorted({{m.split('.')[0] for m in sys.modules}} & set({FORBIDDEN!r}));"
        "assert not bad, f'lean connector chain eagerly imported: {bad}';"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=SRC),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_limits_validator_reaches_for_no_control_system_client():
    """The shared validator holds no client of its own.

    It used to ``import epics`` inside the ``max_step`` check, which measured
    the step over Channel Access whatever control system the write was bound
    for. The read now comes from the connector doing the write, so nothing in
    this module may reach for a client library. Importing the module is not
    enough to prove that -- the old import was inside a function -- so the
    check is run against a validator that actually performs a step check.
    """
    code = (
        "import sys;"
        "from osprey_connectors.control_system.limits_validator import ("
        "    ChannelLimitsConfig, LimitsValidator);"
        "v = LimitsValidator("
        "    {'FOO': ChannelLimitsConfig(channel_address='FOO', max_step=5.0)},"
        "    {'mode': 'exclusive'}, {});"
        "v.validate('FOO', 1.0, read_current=lambda _a: 0.0);"
        "bad = sorted({'epics', 'p4p', 'tango', 'caproto', 'doocs4py'} & set(sys.modules));"
        "assert not bad, f'the limits validator imported a control-system client: {bad}';"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=SRC),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_control_context_imports_no_osprey_module():
    """The record reader must not pull in the framework it is read beneath.

    ``osprey`` is importable in this subprocess — ``src`` is on the path — so
    an accidental import would succeed rather than fail loudly. The assertion
    is therefore on what landed in ``sys.modules``, not on whether the import
    worked. ``osprey_connectors`` is a different distribution and does not
    match: only ``osprey`` itself and its submodules do.
    """
    code = (
        "import osprey_connectors.control_context as cc, sys;"
        "bad = sorted(m for m in sys.modules if m == 'osprey' or m.startswith('osprey.'));"
        "assert not bad, f'the control-context record eagerly imported: {bad}';"
        "assert cc.RECORD_FILENAME == 'control_context.json';"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=SRC),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_identity_imports_no_osprey_module():
    """The ladder must not pull in the framework it resolves an identity for.

    Same shape and same reason as the control-context case: a sandbox child or
    a notebook kernel resolves its own audit and control-state directory
    through this ladder in an interpreter where ``osprey`` is absent. ``src``
    is on the path here, so an accidental import would succeed quietly rather
    than fail -- the assertion is on what landed in ``sys.modules``. The call
    is made, not just the import, because a rung added inside the function
    would escape an import-time check.
    """
    code = (
        "import osprey_connectors.identity as identity, sys;"
        "assert identity.acting_identity(), 'the ladder resolved nothing';"
        "bad = sorted(m for m in sys.modules if m == 'osprey' or m.startswith('osprey.'));"
        "assert not bad, f'the identity ladder eagerly imported: {bad}';"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=SRC),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_mock_archiver_derives_its_simulation_file_without_osprey(tmp_path):
    """The mock archiver serves the control-system machine file with ``osprey`` unimportable.

    A finder at the front of ``sys.meta_path`` refuses ``osprey`` and its
    submodules, because the dev environment has the framework installed and
    no path setting can hide it. The child first proves the finder is live,
    then connects the archiver against a project config whose control-system
    block names a relative machine file, and reads the file's constant back.
    """
    root = tmp_path / "project"
    (root / "data" / "simulation").mkdir(parents=True)
    (root / "data" / "simulation" / "machine.json").write_text(
        json.dumps(
            {
                "name": "Rig",
                "description": "Single-channel machine",
                "channels": {
                    "T:Q1:CUR:SP": {
                        "value": 42.0,
                        "units": "A",
                        "noise": 0.0,
                        "description": "Test quad current setpoint",
                    }
                },
                "scenarios": {"nominal": {"description": "All systems nominal."}},
            }
        )
    )
    config_path = root / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_name": "project",
                "project_root": str(root),
                "control_system": {
                    "type": "mock",
                    "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
                },
                "archiver": {"type": "mock_archiver"},
            }
        )
    )
    code = textwrap.dedent(
        """
        import asyncio
        import sys
        from datetime import datetime


        class _RefuseOsprey:
            def find_spec(self, name, path=None, target=None):
                if name == "osprey" or name.startswith("osprey."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None


        sys.meta_path.insert(0, _RefuseOsprey())
        try:
            import osprey  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            sys.exit("blocker inert")

        from osprey_connectors.archiver.mock_archiver_connector import MockArchiverConnector


        async def main():
            connector = MockArchiverConnector()
            await connector.connect({})
            assert connector._sim_engine is not None, "no engine derived"
            df = await connector.get_data(
                channels=["T:Q1:CUR:SP"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 1),
            )
            values = df.loc[df["channel"] == "T:Q1:CUR:SP", "value"].tolist()
            assert values and all(v == 42.0 for v in values), values
            await connector.disconnect()


        asyncio.run(main())
        bad = sorted(m for m in sys.modules if m == "osprey" or m.startswith("osprey."))
        assert not bad, f"the mock archiver imported: {bad}"
        print("CLEAN")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, CONFIG_FILE=str(config_path)),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_write_door_imports_no_osprey_module():
    """The door is read by the raw-client guard beneath the framework.

    The guard runs in executor sandboxes and notebook kernels, so the module it
    asks must be stdlib-only. Same shape as the ladder case: ``src`` is on the
    path, so the assertion is on ``sys.modules``, and the door is opened and
    read rather than only imported.
    """
    code = (
        "from osprey_connectors.control_system import write_door;"
        "import sys;"
        "assert not write_door.door_is_open();"
        "cm = write_door.open_door(); cm.__enter__();"
        "assert write_door.door_is_open();"
        "cm.__exit__(None, None, None);"
        "bad = sorted(m for m in sys.modules if m == 'osprey' or m.startswith('osprey.'));"
        "assert not bad, f'the write door eagerly imported: {bad}';"
        "print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=SRC),
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
