"""A bundle's ``data/`` tree is content, never templates.

``osprey profile new`` materializes ``apps/<bundle>/data/`` into a facility's
own profile source by literal copy, and ``osprey build`` copies a profile's
``data:`` root back out the same way. Neither side renders Jinja, so a ``.j2``
file staged under ``apps/*/data/`` would reach the built project with its
suffix intact and its placeholders unexpanded -- and the preset path, which
DOES render and strips the suffix, would land it under a different filename
than the profile path. Keeping data trees template-free is what makes the two
paths agree by construction.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APPS_ROOT = REPO_ROOT / "src" / "osprey" / "templates" / "apps"


def test_apps_root_is_discoverable():
    """Guards the other test against passing vacuously on a bad path."""
    bundles = [p for p in APPS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert bundles, f"no app bundles found under {APPS_ROOT}"


def test_no_jinja_templates_under_any_bundle_data_tree():
    offenders = sorted(
        str(p.relative_to(REPO_ROOT)) for p in APPS_ROOT.glob("*/data/**/*.j2") if p.is_file()
    )
    assert offenders == [], (
        "data/ trees are copied verbatim, so these would ship with their .j2 "
        f"suffix and unexpanded placeholders: {offenders}"
    )
