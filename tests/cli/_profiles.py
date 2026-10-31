"""Read a build profile's validation failures one by one."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import BuildProfile
from osprey_connectors.errors import BuildProfileError


def validation_errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    """Validate ``profile`` and return the individual accumulated failures."""
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")
