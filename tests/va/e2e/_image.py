"""The virtual accelerator image the live-container suites run, and its override.

This is the one place the image under test and its override are read. Every
module that boots the image takes ``IMAGE`` from here (the ``tests/va/e2e``
modules through their conftest), and every failure that finds the image absent
or stale ends in ``IMAGE_REMEDY``. Importing this module does nothing but read
one environment variable.
"""

import os

IMAGE_ENV = "OSPREY_VA_E2E_IMAGE"
DEFAULT_IMAGE = "osprey-va-full:latest"
BUILD_COMMAND = "scripts/va/build_and_boot_check.sh"

IMAGE = os.environ.get(IMAGE_ENV, DEFAULT_IMAGE)
IMAGE_REMEDY = f"Build it with {BUILD_COMMAND}, or name another with {IMAGE_ENV}."
