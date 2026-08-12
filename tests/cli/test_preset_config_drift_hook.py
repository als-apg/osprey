"""Every shipped preset deploys the config-drift SessionStart guard.

``config.yml`` is build output, and the ``.claude/`` artifacts beside it are
rendered from it in the same pass — so the two only disagree when something
edits the render in place, which is exactly the situation an operator cannot
see. The ``config-drift`` hook is what makes it visible: it runs at session
start and says the artifacts no longer match the config they were rendered
from.

A preset that omits the hook produces a deployment where that disagreement is
silent, so the guard has to be pinned per preset rather than assumed from one
of them. The hook's own behavior is tested in tests/hooks/test_config_drift_hook.py;
what this file pins is that it actually ships.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PRESETS_DIR = Path(__file__).parents[2] / "src" / "osprey" / "profiles" / "presets"


@pytest.mark.parametrize("preset", ["hello-world", "control-assistant", "ariel-standalone"])
def test_preset_declares_config_drift_hook(preset):
    data = yaml.safe_load((PRESETS_DIR / f"{preset}.yml").read_text())
    assert "config-drift" in data.get("hooks", []), (
        f"{preset} preset must include the config-drift hook"
    )
