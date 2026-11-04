"""The ``service`` extra declares what the API needs to import in its own image.

The dev environment the suites run in carries packages the optimizer's image
does not, so an import the service needs can pass every test here and still
stop the deployed app at start. These rows hold the extra to what the app's
source asks for.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "packages" / "osprey-optimizer"
API = PACKAGE / "src" / "osprey_optimizer" / "api"


def _service_requirements() -> set[str]:
    with (PACKAGE / "pyproject.toml").open("rb") as handle:
        extra = tomllib.load(handle)["project"]["optional-dependencies"]["service"]
    return {re.split(r"[\s\[<>=!~;]", entry, maxsplit=1)[0].lower() for entry in extra}


def test_a_form_or_upload_route_has_its_parser_declared() -> None:
    """FastAPI refuses to build a route that takes form data without python-multipart."""
    takes_form = [
        path.relative_to(ROOT)
        for path in API.rglob("*.py")
        if re.search(r"\b(UploadFile|Form|File)\b", path.read_text(encoding="utf-8"))
    ]
    assert takes_form, "no route takes form data: this row guards nothing and goes"
    assert "python-multipart" in _service_requirements(), takes_form
