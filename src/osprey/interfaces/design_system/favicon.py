"""The OSPREY tab icon, for pages that link it and pages that must inline it.

The web terminal links the file at a content-hashed URL. The landing page and
the auth sidecar's sign-in pages have no static-asset mount, so they carry the
same file as a data URI; one file keeps every tab of a deployment alike.
"""

from __future__ import annotations

import base64
import re
from functools import cache
from pathlib import Path

FAVICON_PATH = Path(__file__).parent / "static" / "img" / "osprey-mark.svg"


@cache
def favicon_data_uri() -> str:
    """The tab icon as a ``data:`` URI, base64 so no attribute escaping touches it.

    The file's comment is for its readers and is left out of every page.
    """
    svg = re.sub(r"\s*<!--.*?-->", "", FAVICON_PATH.read_text(encoding="utf-8"), flags=re.DOTALL)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
