"""Import isolation for the lean control-system connector chain.

External consumers (e.g. the ALS tuning_scripts backend) import only the
control-system connectors and their support modules. That chain must not
eagerly load the archiver stack (pandas) or any LLM/agent machinery.

The control-context record is held to a stricter rule still: it is read
inside connector-host children, executor sandboxes and notebook kernels, so
it may not reach ``osprey`` at all.
"""

import os
import subprocess
import sys
from pathlib import Path

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
        "    {'allow_unlisted_channels': False}, {});"
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
