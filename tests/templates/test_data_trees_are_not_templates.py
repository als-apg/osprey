"""A bundle's ``web/`` tree and a bundled facility are content, never templates.

``osprey init`` materializes ``apps/<bundle>/web/`` and the preset's
``facilities/<name>/`` into the deployment repository's ``web/`` and
``facility/`` by literal copy, and ``osprey build`` copies those trees into
the render the same way. Neither side renders Jinja, so a ``.j2`` file staged
under ``apps/*/web/`` would reach the built project with its suffix intact and
its placeholders unexpanded. Keeping content trees template-free is what
makes the copy and the render agree by construction.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APPS_ROOT = REPO_ROOT / "src" / "osprey" / "templates" / "apps"
FACILITIES_ROOT = REPO_ROOT / "src" / "osprey" / "templates" / "facilities"


def test_apps_root_is_discoverable():
    """Guards the other test against passing vacuously on a bad path."""
    bundles = [p for p in APPS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert bundles, f"no app bundles found under {APPS_ROOT}"


def test_no_jinja_templates_under_any_bundle_content_tree():
    offenders = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in [*APPS_ROOT.glob("*/web/**/*.j2"), *FACILITIES_ROOT.glob("**/*.j2")]
        if p.is_file()
    )
    assert offenders == [], (
        "web/ and facility trees are copied verbatim, so these would ship with their .j2 "
        f"suffix and unexpanded placeholders: {offenders}"
    )


def test_no_bundle_ships_a_data_tree():
    """``data`` is a dissolved name: a bundle's content lives under ``web/``."""
    offenders = sorted(
        str(p.relative_to(REPO_ROOT)) for p in APPS_ROOT.glob("*/data") if p.exists()
    )
    assert offenders == [], f"a bundle still ships a data/ tree: {offenders}"
