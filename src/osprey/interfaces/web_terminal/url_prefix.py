"""Per-container URL prefix contract for multi-user deployments.

Multi-user deployments run one Web Terminal container per user behind a
shared nginx front door, each mounted at ``/u/<user>/``. This module is the
single source of truth for both halves of that contract on the Python side:
what the prefix *is* (:func:`compute_url_prefix`) and how it is *applied* to
a path (:func:`apply_url_prefix`). The browser-side twin is ``withPrefix()``
in ``static/js/api.js``.

Import-cycle note: this module deliberately depends on nothing inside the
package so ``app.py`` and the ``routes/`` modules can all import it at module
load time (``app.py`` imports the routes package, so routes importing back
into ``app`` would be circular).
"""

from __future__ import annotations

import os


def compute_url_prefix() -> str:
    """Compute the per-container URL path prefix for multi-user deployments.

    Shared by ``create_app()`` (the ``window.__OSPREY_PREFIX__``/import-map
    injection into every served HTML document) and downstream server-side
    prefixing (``routes/panels.py``, ``routes/proxy.py``). It is deliberately
    NOT fed to ``FastAPI(root_path=…)`` — nginx strips the prefix before
    proxying, so the app serves bare paths and a non-empty ``root_path``
    would 404 every static Mount (see the note in ``create_app()``).

    Returns:
        ``"/u/<user>"`` when ``OSPREY_TERMINAL_USER`` is set and non-empty;
        otherwise ``""``, preserving single-origin/dev behavior unchanged.
    """
    user = os.environ.get("OSPREY_TERMINAL_USER", "").strip()
    return f"/u/{user}" if user else ""


def apply_url_prefix(prefix: str, path: str) -> str:
    """Prepend ``prefix`` to ``path`` unless it must pass through unchanged.

    An already-absolute URL (``http://``, ``https://``, protocol-relative
    ``//``) is never touched so an external URL is never corrupted, and an
    empty prefix is a byte-identical no-op (single-origin/dev behavior).

    The browser-side ``withPrefix()`` additionally guards non-root-absolute
    and already-prefixed inputs, which server-side callers never produce —
    that asymmetry is deliberate, not drift.
    """
    if not prefix or path.startswith(("http://", "https://", "//")):
        return path
    return f"{prefix}{path}"
