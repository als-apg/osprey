"""What OSPREY's images are built for: their Python and their platform.

The profile's ``config.images.platform`` is the one statement of the platform; the
wheel fit check, the ``--platform`` flag of every image build, the compose override
and the rendered ``FROM`` all read it through :func:`stated_image_platform`. Unset,
images are built natively and the fit check judges this host.

Stdlib only, so every deployment module can read it without an import cycle.
"""

from __future__ import annotations

import platform

#: The ``python:<major>.<minor>-slim`` every OSPREY image is built from.
IMAGE_PYTHON = (3, 12)

#: The values ``images.platform`` accepts.
IMAGE_PLATFORMS = ("linux/amd64", "linux/arm64")

#: A host's ``platform.machine()``, lower-cased, to its image platform's arch.
_HOST_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}

#: An image platform's arch to the arch a Linux wheel tag spells.
_WHEEL_ARCH = {"amd64": "x86_64", "arm64": "aarch64"}


def stated_image_platform(value: object) -> str | None:
    """The platform ``images.platform`` states, or None when it states none.

    Raises:
        ValueError: The value is not one of :data:`IMAGE_PLATFORMS`.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if value not in IMAGE_PLATFORMS:
        raise ValueError(
            f"images.platform {value!r} is not an image platform; "
            f"name one of {', '.join(IMAGE_PLATFORMS)}, or leave it unset to build natively"
        )
    return str(value)


def fit_platform(stated: str | None) -> str:
    """The platform an image is built for: *stated*, else this host's."""
    if stated:
        return stated
    machine = platform.machine().lower()
    return f"linux/{_HOST_ARCH.get(machine, machine)}"


def wheel_arch(image_platform: str) -> str:
    """The arch a Linux wheel tag spells for *image_platform*; an unmapped one unchanged."""
    arch = image_platform.rpartition("/")[2]
    return _WHEEL_ARCH.get(arch, arch)
