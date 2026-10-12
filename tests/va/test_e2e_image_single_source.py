"""The VA image under test is read in one module, and nothing else spells it."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.va.e2e import _image

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION = REPO_ROOT / ".github" / "actions" / "build-virtual-accelerator-image" / "action.yml"


@pytest.fixture
def reload_image_after() -> Iterator[None]:
    """Reload the owner once the test's environment patch is undone."""
    yield
    importlib.reload(_image)


@pytest.mark.usefixtures("reload_image_after")
def test_image_follows_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_image.IMAGE_ENV, "side-build:probe")
    assert importlib.reload(_image).IMAGE == "side-build:probe"

    monkeypatch.delenv(_image.IMAGE_ENV)
    assert importlib.reload(_image).IMAGE == _image.DEFAULT_IMAGE


def test_image_defaults_to_the_tag_ci_builds() -> None:
    assert f"--tag {_image.DEFAULT_IMAGE}" in ACTION.read_text(encoding="utf-8")
