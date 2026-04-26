"""Guard that the shipped app templates remain a known, fixed set.

The build pipeline accepts any directory under ``templates/apps/`` as a
``data_bundle``; preset coverage is driven by ``list_presets()`` reflection.
This test pins the on-disk set so that adding/removing an app bundle is an
explicit, reviewed change — and so that preset-parametrized tests stay in
sync with what actually ships.
"""

from pathlib import Path

import osprey


def test_templates_apps_matches_allowlist():
    templates_apps = Path(osprey.__file__).parent / "templates" / "apps"
    on_disk = {
        p.name for p in templates_apps.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    }
    assert on_disk == {"hello_world", "control_assistant"}
