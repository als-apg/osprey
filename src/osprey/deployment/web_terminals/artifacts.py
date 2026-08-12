"""Render-and-write seam for the multi-user web-terminal deployment artifacts.

:func:`osprey.deployment.web_terminals.render.render_web_terminals` produces the
three artifacts in memory (``docker-compose.web.yml``, ``nginx/nginx.conf``,
``nginx/landing.html``) as a ``{relative_path: content}`` mapping. This module is
the single place that decides *where on disk* those relative paths land, so every
consumer agrees on one location:

* ``osprey up`` renders and writes them at bring-up, then includes the web
  compose file in the ``compose up`` invocation.
* the lifecycle verbs (``decommission``/``prune``) re-render and re-write them after
  editing the roster, so the deployed nginx routing and compose services match the
  new roster.

If bring-up and the lifecycle verbs wrote to different directories, a decommission
would update artifacts that ``up`` never reads. Routing every writer through this
one helper makes that class of drift impossible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.utils.workspace import BUILD_DIR_NAME

#: Filename of the rendered web-stack compose file, as
#: :func:`~osprey.deployment.web_terminals.render.render_web_terminals` keys it.
WEB_COMPOSE_FILENAME = "docker-compose.web.yml"


def web_artifacts_dir(repo_root: Path | str) -> Path:
    """Where a repo's rendered web-terminal artifacts live: ``<repo>/build``.

    They are render output — regenerated from the profile by every build and by
    every roster verb — so they belong in the disposable zone with the rest of
    it, and never at the repo root, whose contents are tracked source.
    """
    return Path(repo_root) / BUILD_DIR_NAME


def web_compose_file(repo_root: Path | str) -> Path:
    """The rendered ``build/docker-compose.web.yml`` of the repo at *repo_root*.

    One spelling for the file every web-stack invocation is pinned to, so the
    writer, the ``-f`` argument, the "is anything deployed here?" probe and the
    image-drift reconcile cannot disagree about which file that is. They did:
    the probe looked in the working directory while the render wrote elsewhere,
    which is how a password rotation could report success having recreated
    nothing.
    """
    return web_artifacts_dir(repo_root) / WEB_COMPOSE_FILENAME


def auth_env_digest(project_root: str | Path) -> str:
    """sha256 hex digest of ``.env.auth``'s current bytes under ``project_root``.

    The digest is rendered as a label on the auth sidecar's compose service
    (see :data:`~osprey.deployment.web_terminals.render.AUTH_ENV_DIGEST_LABEL`),
    so a content change to the file becomes a service-definition change — the
    one recreate trigger every compose implementation honours. An absent (or
    unreadable) file digests as empty content rather than raising: the render
    must never crash over it, and on the deploy path preflight has already
    either created the file or aborted. Erring toward the empty sentinel is
    fail-safe — at worst it flips the label and costs one sidecar recreate,
    never a stale credential.
    """
    try:
        content = (Path(project_root) / AUTH_ENV_FILENAME).read_bytes()
    except OSError:
        content = b""
    return hashlib.sha256(content).hexdigest()


def write_web_terminal_artifacts(config: Any, repo_root: Path | str | None = None) -> list[Path]:
    """Render the web-terminal artifacts into the repo's ``build/`` zone.

    The artifacts' relative paths (``docker-compose.web.yml``,
    ``nginx/nginx.conf``, ``nginx/landing.html``) are preserved beneath the
    destination; parent directories (e.g. ``nginx/``) are created as needed. The
    compose file and its ``nginx/`` subtree must stay co-located, which writing
    them together under one destination guarantees.

    Two directories, deliberately, because the two kinds of file have opposite
    lifetimes. The artifacts are render output and land in ``<repo>/build``;
    ``.env.auth`` is a durable 0600 credential store and stays at the repo root,
    with the deployment's other secrets. Compose sees them as one: every
    relative path in the rendered file — including the sidecar's
    ``env_file: .env.auth`` — resolves against the pinned project directory,
    which IS the repo root (see
    :func:`~osprey.deployment.compose_generator.compose_base_cmd`). So the
    digest below is read from the repo root, not from the destination: it has to
    describe exactly the file the rendered stack will read.

    Every writer (``osprey up``, the roster verbs' re-render, and
    :func:`~osprey.deployment.web_terminals.provision.force_recreate_auth_sidecar`
    just before each forced recreate) goes through here, and each renders AFTER
    its ``.env.auth`` mutation, so the digest is current on every path that
    reaches a compose ``up``.

    Args:
        config: The parsed facility config, passed straight through to
            :func:`render_web_terminals` (raises ``ValueError`` on an unrenderable
            config, e.g. a TLS seam enabled without cert/key).
        repo_root: The deployment repo. Defaults to the one
            :func:`~osprey.deployment.compose_generator.resolve_repo_root`
            derives from the config. There is deliberately no way to write the
            artifacts anywhere but this repo's ``build/``: a second destination
            is how bring-up and the roster verbs came to act on different
            copies. A caller that only wants to *see* the render calls
            :func:`render_web_terminals` and writes it wherever it likes.

    Returns:
        The list of files written, in the render mapping's iteration order.
    """
    from osprey.deployment.compose_generator import resolve_repo_root

    root = Path(repo_root) if repo_root is not None else resolve_repo_root(config)
    artifacts = render_web_terminals(config, auth_env_digest=auth_env_digest(root))
    dest = web_artifacts_dir(root)
    written: list[Path] = []
    for relative_path, content in artifacts.items():
        target = dest / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written
