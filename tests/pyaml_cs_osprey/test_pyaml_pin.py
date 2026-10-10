"""The framework and the pyAML backend pin the same pyAML version.

``osprey-framework`` declares ``accelerator-middle-layer`` directly, next to
``pyaml-cs-osprey``, because the framework's own code imports pyAML.
``pyaml-cs-osprey`` declares the same package for its control-system backend.
The two specifiers must be string-equal: a drift between them would let one
package be built and tested against a pyAML the other refuses.
"""

from __future__ import annotations

import pathlib
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRAMEWORK_PYPROJECT = REPO_ROOT / "pyproject.toml"
MEMBER_PYPROJECT = REPO_ROOT / "packages" / "pyaml-cs-osprey" / "pyproject.toml"
AML_PACKAGE = "accelerator-middle-layer"


def _aml_requirement(pyproject: pathlib.Path) -> str | None:
    """Return the ``accelerator-middle-layer`` entry of ``project.dependencies``.

    ``None`` when the package is not declared; several entries come back joined,
    so the comparison below fails with every one of them in its message.
    """
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    entries = [
        entry
        for entry in dependencies
        if canonicalize_name(Requirement(entry).name) == canonicalize_name(AML_PACKAGE)
    ]
    return "; ".join(entries) if entries else None


def test_framework_pin_equals_member_specifier() -> None:
    framework = _aml_requirement(FRAMEWORK_PYPROJECT)
    member = _aml_requirement(MEMBER_PYPROJECT)
    assert member is not None, (
        f"packages/pyaml-cs-osprey/pyproject.toml does not declare {AML_PACKAGE}; "
        f"framework pyproject.toml has {framework!r}"
    )
    assert framework == member, (
        f"{AML_PACKAGE} pins differ: framework pyproject.toml has {framework!r}, "
        f"packages/pyaml-cs-osprey/pyproject.toml has {member!r}"
    )
