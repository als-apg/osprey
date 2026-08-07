"""Import isolation for the lean control-system connector chain.

External consumers (e.g. the ALS tuning_scripts backend) import only the
control-system connectors and their support modules. That chain must not
eagerly load the archiver stack (pandas) or any LLM/agent machinery.
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
