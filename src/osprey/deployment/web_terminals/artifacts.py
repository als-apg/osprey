"""Render-and-write seam for the multi-user web-terminal deployment artifacts.

:func:`osprey.deployment.web_terminals.render.render_web_terminals` produces the
three artifacts in memory (``docker-compose.web.yml``, ``nginx/nginx.conf``,
``nginx/landing.html``) as a ``{relative_path: content}`` mapping. This module is
the single place that decides *where on disk* those relative paths land, so every
consumer agrees on one location:

* ``osprey deploy up`` renders and writes them at bring-up, then includes the web
  compose file in the ``compose up`` invocation.
* the lifecycle verbs (``decommission``/``prune``) re-render and re-write them after
  editing the roster, so the deployed nginx routing and compose services match the
  new roster.

If bring-up and the lifecycle verbs wrote to different directories, a decommission
would update artifacts that ``up`` never reads. Routing every writer through this
one helper makes that class of drift impossible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osprey.deployment.web_terminals.render import render_web_terminals


def write_web_terminal_artifacts(config: Any, dest_dir: str | Path = ".") -> list[Path]:
    """Render the web-terminal artifacts and write them under ``dest_dir``.

    The artifacts' relative paths (``docker-compose.web.yml``,
    ``nginx/nginx.conf``, ``nginx/landing.html``) are preserved beneath
    ``dest_dir``; parent directories (e.g. ``nginx/``) are created as needed. The
    web compose file's own mount paths are relative to itself, so the compose file
    and its ``nginx/`` subtree must stay co-located — writing them together under a
    single ``dest_dir`` guarantees that.

    Args:
        config: The parsed facility config, passed straight through to
            :func:`render_web_terminals` (raises ``ValueError`` on an unrenderable
            config, e.g. a TLS seam enabled without cert/key).
        dest_dir: Directory the relative artifact paths are written beneath.
            Defaults to the current working directory, which is the project root
            that ``osprey deploy`` has already ``chdir``'d into.

    Returns:
        The list of files written, in the render mapping's iteration order.
    """
    artifacts = render_web_terminals(config)
    dest = Path(dest_dir)
    written: list[Path] = []
    for relative_path, content in artifacts.items():
        target = dest / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written
