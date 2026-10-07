"""Pillow is a core dependency at a version the image render worker supports."""

import PIL
from packaging.version import Version


def test_pillow_meets_core_floor():
    assert Version(PIL.__version__) >= Version("12.3")
