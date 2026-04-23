"""Guard that the shipped app templates match the build_profile allowlist.

Mirrors the filter in ``build_profile.py`` (line ~200) where the hardcoded
``valid_bundles`` set of accepted ``data_bundle`` values lives. If a template
is added on disk without updating the allowlist (or vice versa), profile
validation drifts silently — this test catches the drift at collection time.
"""

from pathlib import Path

import osprey


def test_templates_apps_matches_allowlist():
    templates_apps = Path(osprey.__file__).parent / "templates" / "apps"
    on_disk = {
        p.name for p in templates_apps.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    }
    assert on_disk == {"hello_world", "control_assistant"}
