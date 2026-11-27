"""The canopy package's own spellings of its source paths equal the declaration.

The package imports nothing outside the standard library, so it cannot read
:mod:`osprey.repo_layout`; it keeps private copies of where the deployment repo
holds the screen library and the facility collections, and this test holds each
equal to the source path of the child the deployment-root declaration names.

The landing notices, the other content under ``web``, are config values rather
than a reader's copy: a notices value rewritten to the ``web`` child resolves
against the repo root with no reader change, which the last test pins.
"""

from __future__ import annotations

from pathlib import Path

from osprey.integrations.canopy import FACILITY_COLLECTIONS_SUBPATH, LIBRARY_SUBPATH
from osprey.repo_layout import child_display_path


def test_the_screen_library_copy_equals_the_declared_child() -> None:
    assert Path(child_display_path("web", "canopy-library")) == LIBRARY_SUBPATH


def test_the_facility_collections_copy_equals_the_declared_child() -> None:
    assert Path(child_display_path("web", "canopy-collections")) == FACILITY_COLLECTIONS_SUBPATH


def test_a_rewritten_notices_value_resolves(tmp_path: Path) -> None:
    from osprey.deployment.web_terminals.render import _build_notices

    value = child_display_path("web", "landing", "working-safely.md")
    assert value == "web/landing/working-safely.md"
    doc = tmp_path / value
    doc.parent.mkdir(parents=True)
    doc.write_text("# Working safely\n\nCall the control room first.\n", encoding="utf-8")

    notices = _build_notices({"notices": [value]}, {"project_root": str(tmp_path)})

    assert len(notices) == 1
    assert "Working safely" in str(notices[0])
    assert "Call the control room first." in str(notices[0])
